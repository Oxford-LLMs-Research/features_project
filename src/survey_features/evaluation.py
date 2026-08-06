"""
Matched-k XGBoost CV: oracle vs model vs random vs textbook feature sets.

The target's measurement level selects estimator and score, mirroring the oracle
(oracle.TARGET_TYPE_PROBLEM; docs/pipeline_audit_2026-08.md §A11). Results carry
`primary_score` (higher-is-better, comparable within a cell only).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

REGRESSION_TYPES = {"ordinal", "continuous"}


def _normalize_row_label(x) -> tuple:
    """Canonical form so int / int-like str labels from oracle_meta match the frame index."""
    if isinstance(x, (bool, np.bool_)):
        return ("s", str(x))
    if isinstance(x, (int, np.integer)):
        return ("i", int(x))
    if isinstance(x, float) and np.isfinite(x) and float(x).is_integer():
        return ("i", int(x))
    s = str(x)
    try:
        return ("i", int(s))
    except ValueError:
        return ("s", s)


def _row_index_mask(index: pd.Index, row_index: Sequence) -> np.ndarray:
    wanted = {_normalize_row_label(i) for i in row_index}
    return np.fromiter(
        (_normalize_row_label(i) in wanted for i in index),
        dtype=bool,
        count=len(index),
    )

# Shared tree hyperparameters — identical between the classifier and the regressor so the
# only thing the measurement level changes is the objective and the score.
_XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    verbosity=0,
)


def make_clf(nthread=None):
    """XGBoost classifier with fixed hyperparameters (matching Phase 0a)."""
    kwargs = {"nthread": nthread} if nthread is not None else {}
    return xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="mlogloss", **_XGB_PARAMS, **kwargs
    )


def make_regressor(nthread=None):
    """XGBoost regressor for ordinal/continuous targets (same tree settings)."""
    kwargs = {"nthread": nthread} if nthread is not None else {}
    return xgb.XGBRegressor(objective="reg:squarederror", **_XGB_PARAMS, **kwargs)


def evaluate_feature_set(
    data: pd.DataFrame,
    target_var: str,
    feature_vars: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    nthread: int | None = None,
    target_type: str | None = None,
    target_valid_range: tuple[float, float] | None = None,
    row_index: Sequence | None = None,
) -> dict:
    """
    Train XGBoost on a specific feature set and return CV accuracy.

    Args:
        data: DataFrame with target and feature columns
        target_var: target variable column name
        feature_vars: list of feature column names to use
        n_splits: number of CV folds
        random_state: random seed
        row_index: optional row labels (e.g. oracle_meta train_index) that restrict
            CV to the oracle fit split so selection never saw these eval rows

    Returns:
        dict with accuracy stats
    """
    # Drop rows with missing target
    valid = data[target_var].notna()
    df = data.loc[valid].copy()

    if row_index is not None:
        df = df.loc[_row_index_mask(df.index, row_index)]
        if len(df) < max(n_splits, 2):
            return {
                "accuracy_mean": None, "accuracy_std": None, "n_features": 0,
                "n_samples": len(df), "error": "train_index too thin for CV",
            }

    # Defensive: drop any duplicated columns in the input frame (WVS loader has
    # been known to produce them) so XGBoost's unique-feature-name check holds.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Filter to available features, deduplicated in arrival order. The caller
    # may pass the same code twice when multiple LLM features map to the same
    # WVS variable; XGBoost requires unique feature names.
    seen: set[str] = set()
    available: list[str] = []
    for f in feature_vars:
        if f in df.columns and f not in seen:
            seen.add(f)
            available.append(f)
    if not available:
        return {"accuracy_mean": None, "accuracy_std": None, "n_features": 0,
                "n_samples": len(df), "error": "no available features"}

    X = df[available].copy()
    y_raw = df[target_var]

    # Label-encode any text columns so XGBoost receives only numeric data.
    for col in X.select_dtypes(include="object").columns:
        col_le = LabelEncoder()
        non_null = X[col].notna()
        X.loc[non_null, col] = col_le.fit_transform(X.loc[non_null, col])
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # ── Ordinal / continuous: regress on the codes and score by rank correlation ──
    if target_type in REGRESSION_TYPES:
        y = pd.to_numeric(y_raw, errors="coerce")
        keep = y.notna()
        if target_valid_range is not None:
            lo, hi = target_valid_range
            keep &= y.between(lo, hi)
        X, y = X.loc[keep], y.loc[keep]
        if y.nunique() < 3:
            return {"primary_score": None, "n_features": len(available),
                    "n_samples": len(y), "error": "fewer than 3 distinct values"}
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        rhos, maes = [], []
        for tr, te in kf.split(X):
            reg = make_regressor(nthread=nthread)
            reg.fit(X.iloc[tr], y.iloc[tr])
            pred = reg.predict(X.iloc[te])
            rho = spearmanr(y.iloc[te], pred).correlation
            rhos.append(0.0 if pd.isna(rho) else float(rho))
            maes.append(mean_absolute_error(y.iloc[te], pred))
        rhos = np.asarray(rhos, dtype=float)
        # Baseline for a rank score is 0 (no ordering recovered), not the modal share.
        return {
            "target_type": target_type,
            "spearman_mean": round(float(rhos.mean()), 4),
            "spearman_std": round(float(rhos.std()), 4),
            "mae_mean": round(float(np.mean(maes)), 4),
            "primary_score": round(float(rhos.mean()), 4),
            "primary_metric": "spearmanr",
            "majority_baseline": round(float(y.value_counts().max() / len(y)), 4),
            "n_features": len(available),
            "n_features_requested": len(feature_vars),
            "n_samples": len(y),
            "error": None,
        }

    # Encode target — numeric cast first, fall back to label encoding for text targets.
    le = LabelEncoder()
    try:
        y = pd.Series(le.fit_transform(y_raw.astype(int)), index=y_raw.index)
    except (ValueError, TypeError):
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

    # Drop classes with < 5 observations
    class_counts = y.value_counts()
    rare = class_counts[class_counts < 5].index.tolist()
    if rare:
        keep = ~y.isin(rare)
        X = X.loc[keep]
        y_raw = y_raw.loc[keep]
        le = LabelEncoder()
        try:
            y = pd.Series(le.fit_transform(y_raw.astype(int)), index=y_raw.index)
        except (ValueError, TypeError):
            y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

    if len(le.classes_) < 2:
        return {"accuracy_mean": None, "accuracy_std": None, "n_features": len(available),
                "n_samples": len(y), "error": "fewer than 2 classes"}

    majority_baseline = y.value_counts().max() / len(y)

    # Manual CV avoids sklearn>=1.6 calling is_classifier() on XGBClassifier,
    # which can raise AttributeError with some xgboost / sklearn combinations.
    # Stratified: the target is multiclass and often mode-dominated, so unstratified
    # folds could leave a class out of a training fold entirely.
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_labels = np.unique(y)
    scores, losses = [], []
    for train_idx, test_idx in kf.split(X, y):
        clf = make_clf(nthread=nthread)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_test = y.iloc[test_idx]
        scores.append(accuracy_score(y_test, clf.predict(X.iloc[test_idx])))
        # Log loss is the sensitive metric: on a mode-dominated survey item a genuinely
        # informative feature often shifts probabilities without flipping any argmax,
        # which accuracy cannot see at all.
        proba = clf.predict_proba(X.iloc[test_idx])
        losses.append(log_loss(y_test, proba, labels=all_labels))
    scores = np.asarray(scores, dtype=float)
    losses = np.asarray(losses, dtype=float)

    return {
        "target_type": target_type or "classification",
        "accuracy_mean": round(float(scores.mean()), 4),
        "accuracy_std": round(float(scores.std()), 4),
        "logloss_mean": round(float(losses.mean()), 4),
        "logloss_std": round(float(losses.std()), 4),
        # higher-is-better scalar, comparable within a cell across target types
        "primary_score": round(float(-losses.mean()), 4),
        "primary_metric": "neg_log_loss",
        "majority_baseline": round(float(majority_baseline), 4),
        "n_features": len(available),
        "n_features_requested": len(feature_vars),
        "n_samples": len(y),
        "error": None,
    }


def single_random_draw_result(
    country_data: pd.DataFrame,
    target_var: str,
    all_feature_pool: list[str],
    k: int,
    seed: int,
    nthread: int | None = 1,
    target_type: str | None = None,
    row_index: Sequence | None = None,
) -> dict:
    """One random-k draw evaluated with the same downstream estimator (full result)."""
    rng = np.random.RandomState(seed)
    random_vars = list(rng.choice(all_feature_pool, size=min(k, len(all_feature_pool)), replace=False))
    return evaluate_feature_set(
        country_data, target_var, random_vars,
        nthread=nthread, target_type=target_type, row_index=row_index,
    )


def single_random_draw(
    country_data: pd.DataFrame,
    target_var: str,
    all_feature_pool: list[str],
    k: int,
    seed: int,
    nthread: int | None = 1,
) -> float | None:
    """Accuracy of one random-k draw (thin wrapper kept for existing call sites)."""
    return single_random_draw_result(
        country_data, target_var, all_feature_pool, k, seed, nthread=nthread
    )["accuracy_mean"]


# Backwards-compatible alias (old name in phase0b_evaluation.py).
_single_random_draw = single_random_draw


def run_comparison(
    data: pd.DataFrame,
    target_var: str,
    country_code: int | str,
    country_col: str,
    model_features: list[str],
    oracle_importances: pd.DataFrame,
    n_random_draws: int = 20,
    all_feature_pool: list[str] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    eval_xgb_nthread: int | None = None,
    row_index: Sequence | None = None,
) -> dict:
    """
    Compare oracle, model-selected, and random feature sets.

    Args:
        data: full survey DataFrame
        target_var: target variable code (e.g., "Q199")
        country_code: numeric country code (e.g., 566 for Nigeria)
        country_col: column name for country (e.g., "B_COUNTRY")
        model_features: list of mapped variable codes from disambiguation
            (None entries = unmapped, will be filtered out)
        oracle_importances: DataFrame with columns
            [target_variable, country, feature_variable, importance_mean]
            (prefer importance_select for ranking when present)
        n_random_draws: number of random feature draws for baseline
        all_feature_pool: list of all available feature codes for random draws.
            If None, uses all columns except target and country.
        random_state: random seed
        row_index: optional oracle fit-split labels passed to every arm's CV

    Returns:
        dict with results for oracle, model, and random conditions.
    """
    from .metrics import oracle_topk_codes, rank_score_from_frame

    # Filter to country
    country_data = data[data[country_col] == country_code].copy()

    # Feature pool for random draws (build first so we can filter model_vars against it)
    if all_feature_pool is None:
        exclude = {target_var, country_col}
        all_feature_pool = [c for c in country_data.columns if c not in exclude]

    # Model features: filter out None/unmapped, then restrict to oracle feature pool
    # so the model is evaluated on the same variable universe as the oracle.
    pool_set = set(all_feature_pool)
    k_requested = len(model_features)
    k_mapped = sum(1 for f in model_features if f is not None)
    model_vars = [f for f in model_features if f is not None and f in pool_set]
    k = len(model_vars)

    if k == 0:
        return {
            "target": target_var,
            "country": country_code,
            "error": "no mapped features in pool",
            "k": 0,
            "k_requested": k_requested,
            "k_mapped": k_mapped,
        }

    # Oracle top-k: rank on select split when present, else importance_mean
    oracle_sub = oracle_importances[
        (oracle_importances["target_variable"] == target_var)
        & (oracle_importances["country"] == country_code)
    ]
    rank, _score = rank_score_from_frame(oracle_sub)
    if rank:
        oracle_vars = [c for c in oracle_topk_codes(rank, k) if c in pool_set][:k]
    else:
        oracle_vars = (
            oracle_sub.sort_values("importance_mean", ascending=False)["feature_variable"]
            .head(k).tolist()
        )

    # Evaluate oracle
    oracle_result = evaluate_feature_set(
        country_data, target_var, oracle_vars, nthread=eval_xgb_nthread,
        row_index=row_index,
    )

    # Evaluate model-selected
    model_result = evaluate_feature_set(
        country_data, target_var, model_vars, nthread=eval_xgb_nthread,
        row_index=row_index,
    )

    # Evaluate random-k (averaged over draws, parallelised across cores)
    seeds = [random_state + i for i in range(n_random_draws)]
    raw = Parallel(n_jobs=n_jobs)(
        delayed(single_random_draw_result)(
            country_data, target_var, all_feature_pool, k, s,
            nthread=eval_xgb_nthread, row_index=row_index,
        )
        for s in seeds
    )
    random_scores = [
        d["accuracy_mean"] for d in raw if d.get("accuracy_mean") is not None
    ]

    random_result = {
        "accuracy_mean": round(float(np.mean(random_scores)), 4) if random_scores else None,
        "accuracy_std": round(float(np.std(random_scores)), 4) if random_scores else None,
        "n_draws": len(random_scores),
    }

    return {
        "target": target_var,
        "country": country_code,
        "k": k,
        "k_requested": k_requested,
        "k_mapped": k_mapped,
        "oracle": oracle_result,
        "model": model_result,
        "random": random_result,
    }


def print_comparison(result: dict):
    """Print a single comparison result."""
    if "error" in result and result.get("error"):
        print(
            f"  {result.get('target', '?')} | country={result.get('country', '?')} "
            f"| ERROR: {result['error']}"
        )
        return

    o = result["oracle"]
    m = result["model"]
    r = result["random"]
    bl = o.get("majority_baseline", "?")

    print(f"\n  {result['target']} | country={result['country']} | k={result['k']} (requested={result['k_requested']}, mapped={result['k_mapped']})")
    print(f"    Majority baseline: {bl}")
    print(f"    Oracle top-{result['k']}:    {o['accuracy_mean']} ± {o['accuracy_std']}  (features: {o['n_features']})")
    print(f"    Model-selected:    {m['accuracy_mean']} ± {m['accuracy_std']}  (features: {m['n_features']})")
    print(f"    Random-{result['k']} (n={r['n_draws']}): {r['accuracy_mean']} ± {r['accuracy_std']}")

    if o["accuracy_mean"] and m["accuracy_mean"] and r["accuracy_mean"]:
        cost = o["accuracy_mean"] - m["accuracy_mean"]
        value = m["accuracy_mean"] - r["accuracy_mean"]
        print(f"    Cost of imperfect selection: {cost:+.4f}")
        print(f"    Value of reasoning over random: {value:+.4f}")
