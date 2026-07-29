"""
Downstream prediction evaluation.
Compare XGBoost accuracy using oracle, model-selected, and random feature sets
(matched-k, 5-fold CV). Shared by the current free-text pipeline and the legacy grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


def make_clf(nthread=None):
    """XGBoost classifier with fixed hyperparameters (matching Phase 0a)."""
    kwargs = {"nthread": nthread} if nthread is not None else {}
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
        **kwargs,
    )


def evaluate_feature_set(
    data: pd.DataFrame,
    target_var: str,
    feature_vars: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    nthread: int | None = None,
) -> dict:
    """
    Train XGBoost on a specific feature set and return CV accuracy.

    Args:
        data: DataFrame with target and feature columns
        target_var: target variable column name
        feature_vars: list of feature column names to use
        n_splits: number of CV folds
        random_state: random seed

    Returns:
        dict with accuracy stats
    """
    # Drop rows with missing target
    valid = data[target_var].notna()
    df = data.loc[valid].copy()

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
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in kf.split(X, y):
        clf = make_clf(nthread=nthread)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = clf.predict(X.iloc[test_idx])
        scores.append(accuracy_score(y.iloc[test_idx], pred))
    scores = np.asarray(scores, dtype=float)

    return {
        "accuracy_mean": round(float(scores.mean()), 4),
        "accuracy_std": round(float(scores.std()), 4),
        "majority_baseline": round(float(majority_baseline), 4),
        "n_features": len(available),
        "n_features_requested": len(feature_vars),
        "n_samples": len(y),
        "error": None,
    }


def single_random_draw(
    country_data: pd.DataFrame,
    target_var: str,
    all_feature_pool: list[str],
    k: int,
    seed: int,
    nthread: int | None = 1,
) -> float | None:
    """One random-k draw evaluated with the same downstream classifier."""
    rng = np.random.RandomState(seed)
    random_vars = list(rng.choice(all_feature_pool, size=min(k, len(all_feature_pool)), replace=False))
    r = evaluate_feature_set(country_data, target_var, random_vars, nthread=nthread)
    return r["accuracy_mean"]


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
        n_random_draws: number of random feature draws for baseline
        all_feature_pool: list of all available feature codes for random draws.
            If None, uses all columns except target and country.
        random_state: random seed

    Returns:
        dict with results for oracle, model, and random conditions.
    """
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

    # Oracle top-k (matched to model's k)
    oracle_df = oracle_importances[
        (oracle_importances["target_variable"] == target_var)
        & (oracle_importances["country"] == country_code)
    ].sort_values("importance_mean", ascending=False)
    oracle_vars = oracle_df["feature_variable"].head(k).tolist()

    # Evaluate oracle
    oracle_result = evaluate_feature_set(
        country_data, target_var, oracle_vars, nthread=eval_xgb_nthread
    )

    # Evaluate model-selected
    model_result = evaluate_feature_set(
        country_data, target_var, model_vars, nthread=eval_xgb_nthread
    )

    # Evaluate random-k (averaged over draws, parallelised across cores)
    seeds = [random_state + i for i in range(n_random_draws)]
    raw = Parallel(n_jobs=n_jobs)(
        delayed(single_random_draw)(country_data, target_var, all_feature_pool, k, s)
        for s in seeds
    )
    random_scores = [s for s in raw if s is not None]

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
