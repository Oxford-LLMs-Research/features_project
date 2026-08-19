"""
AutoGluon oracle permutation importances (requires: pip install -e ".[oracle]").

Computes permutation importances for one (target, country) cell and writes
cache/cells/<target>_<country>/{oracle.csv, oracle_meta.json} for the rest of the
pipeline. Contract version: ORACLE_CONTRACT_VERSION (bump if output meaning changes).

Design
------
compute_oracle runs six stages, one helper each (search for "compute_oracle stages"):

  1. rows     _prepare_cell_frame   country slice, metadata-aware cleaning
  2. columns  _build_cell_pool      candidate pool, near-duplicate + skip-pattern excl.
  3. target   _prepare_target       measurement level drives the model:
                                      binary/nominal -> classification, log loss
                                      ordinal/continuous -> regression, Spearman rho
                                    (log loss because accuracy only moves when a
                                    permutation flips an argmax — on mode-dominated
                                    items importance collapsed to exact zeros;
                                    Spearman because an 11-point scale is ordered,
                                    not 11 unrelated classes)
  4. split    _cv_split             THE HONEST CV SPLIT — see below
  5. fit      _fit_and_rank         one AutoGluon fit per fold, two importance passes
  6. output   _assemble_outputs_cv  oracle.csv + oracle_meta.json

The honest CV split (contract v4 — the only contract this module produces):

    one country-cell
    +-- 50%  R   cv       k-fold CV: each fold fits on R\fold, ranks on fold
    +-- 30%  D   eval     downstream evaluator reserve -> train_index
    +-- 20%  V2  score    values the picks; sees no fit and no ranking pass

Captured importance divides the model's mass by "the oracle top-k's mass". If the same
noisy estimates both choose the top-k and value it, the chosen features are exactly the
ones whose noise broke upward — a winner's curse that inflates the denominator, by an
amount that varies per cell. Ranking (CV over R) and valuing (disjoint V2) on separate
rows removes it ("honest" in the Athey–Imbens sample-splitting sense: one subsample
chooses, another estimates). The retired v3 one-shot split measured ranking on ONE fit
and ONE thin 20% holdout: two honest reads of a cell agreed on only ~1/3 of their top
ten. v4 spends those rows on k-fold ranking instead — refit noise and eval-row noise
both average out — and the between-fold agreement lands in meta["reliability"], so the
per-cell noise floor comes free. `oracle_ceiling@k` = V2-mass(top-k chosen on the CV
ranking) / V2-mass(top-k on V2) calibrates what a data-driven oracle achieves when it
cannot cheat — report the LLM against that, not against 1.0. `train_index` in the meta
is the eval reserve D — rows no oracle fit or importance pass ever touched — consumed
by the downstream evaluator (`evaluate_feature_set` / `score_cell`) so no arm is scored
on rows the oracle's ranking already saw.

Output contract
---------------
compute_oracle() returns (oracle_df, feature_pool, meta):
  oracle_df   - target_variable, country, feature_variable,
                importance_select, importance_select_std,
                importance_score,  importance_score_std,
                importance_mean, importance_std   (aliases of the score columns,
                                                   so existing readers are unaffected),
                majority_baseline
  feature_pool - list of feature variable codes included in the model
  meta         - eval_metric, split sizes, reliability, n_positive_score,
                 oracle_ceiling per k, and train_index (row labels of the
                 eval reserve D)

Cache contract
--------------
When run via scripts/compute_oracle.py, results are saved to:
  outputs/cache/cells/<target>_<country>/oracle.csv  (+ oracle_meta.json)
and are picked up by scripts/rerun_oracles.py, which recomputes a cell iff its
cached contract_version differs from ORACLE_CONTRACT_VERSION.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from .config import OUTPUTS_DIR
from .layout import tmp_dir
from .surveys import (
    clean_question_columns,
    detect_target_type,
    _code_variants,
    _substantive_values,
    out_of_scale_codes,
    substantive_numeric_mask,
    to_ordinal_codes,
    flatten_metadata,
)
from .feature_pool import (  # noqa: F401  (re-exported; moved 2026-08)
    build_feature_pool,
    detect_conditional_leakage,
    feature_passes_variation,
    filter_feature_pool_across_countries,
    filter_feature_pool_for_country,
    normalized_entropy,
    standardize_missing,
)


def _prewarm_autogluon_imports() -> None:
    """
    Force-import every AutoGluon tabular model submodule in the main thread.

    AutoGluon lazy-imports model implementations (and their callbacks /
    hyperparameter helpers) the first time each model is fit. When
    archive/run_grid.py uses a ThreadPoolExecutor to fit multiple cells concurrently,
    two worker threads can race the first-ever import of the same submodule;
    one thread then sees a partially-initialised module and fails with
    'cannot import name ... (most likely due to a circular import)', causing
    AutoGluon to silently skip that model. Walking the full submodule tree
    here, in the main thread, fully populates sys.modules before any worker
    spawns. Failures (e.g. optional model deps like mitra/tabpfn/fasttext
    missing) are swallowed — AutoGluon handles missing deps gracefully at
    fit time.
    """
    import importlib
    import pkgutil

    try:
        import autogluon.tabular.models as ag_models
    except Exception:
        return

    for _finder, name, _ispkg in pkgutil.walk_packages(
        ag_models.__path__, prefix="autogluon.tabular.models."
    ):
        try:
            importlib.import_module(name)
        except Exception:
            # Optional model dep missing (e.g. fasttext, mitra) or CUDA-only
            # module on a CPU box. AutoGluon will skip the corresponding model
            # at fit time without affecting other models.
            pass


_prewarm_autogluon_imports()

RANDOM_STATE = 42
SIMILARITY_THRESHOLD = 0.85
TEST_SIZE = 0.2

# Importance metric for classification targets. Proper scoring rule; accuracy collapsed
# most importances to exact zeros (pipeline_audit_2026-08.md #A1).
ORACLE_EVAL_METRIC = "log_loss"

# Bump when cached oracle outputs change MEANING (metric / split / type handling);
# rerun_oracles resumes on it. Version log: docs/onboarding.md #3.
# 4 = k-fold CV ranking + eval reserve + untouched valuation holdout (_cv_split).
# v3 (one-shot honest split) is retired: a dev byproduct, do not build on its caches.
ORACLE_CONTRACT_VERSION = 4

# Default ranking folds for the CV contract.
DEFAULT_CV_FOLDS = 5

# v4 default share of rows reserved for the downstream evaluator (train_index).
# The remaining 1 - test_size - EVAL_RESERVE is the CV ranking region.
EVAL_RESERVE = 0.3

# Measurement level -> (problem_type, importance metric). Ordinal = regression +
# Spearman (rank-based, no spacing assumption, one fit). Units differ per cell by
# design: captured importance is a within-cell ratio. Why: pipeline_audit #A11.
TARGET_TYPE_PROBLEM: dict[str, tuple[str, str]] = {
    "binary":     ("binary",     "log_loss"),
    "nominal":    ("multiclass", "log_loss"),
    "ordinal":    ("regression", "spearmanr"),
    "continuous": ("regression", "spearmanr"),
}

# k values at which the honest-oracle ceiling is recorded (see compute_oracle).
CEILING_KS = (5, 10, 20)

AUTOGLUON_RUNTIME_MODES = {
    "quick": {"preset": "medium_quality", "time_limit": 60},
    "balanced": {"preset": "good_quality", "time_limit": 180},
    "best": {"preset": "best_quality", "time_limit": 600},
}

ENFORCE_IDENTICAL_FEATURE_POOL = False
MAX_MISSINGNESS_THRESHOLD = 0.2
MIN_NORMALIZED_FEATURE_ENTROPY = 0.0
MIN_CLASS_COUNT = 5


def load_similarity_model(similarity_threshold: float) -> object | None:
    """Embedding model for the oracle's semantic near-duplicate exclusion filter."""
    if similarity_threshold <= 0 or similarity_threshold >= 1:
        return None
    from sentence_transformers import SentenceTransformer

    from .config import DEFAULT_EMBEDDING_MODEL

    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


def resolve_runtime_config(runtime_mode: str, autogluon_time_limit: int) -> tuple[str, int]:
    """Resolve AutoGluon preset + time limit from CLI settings."""
    runtime_cfg = AUTOGLUON_RUNTIME_MODES[runtime_mode]
    preset = str(runtime_cfg["preset"])
    time_limit = int(runtime_cfg["time_limit"])
    if autogluon_time_limit > 0:
        time_limit = autogluon_time_limit
    return preset, time_limit


def _resolve_num_gpus(requested: int) -> int:
    if requested and requested > 0:
        return int(requested)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return 0


@contextmanager
def _gpu_scope(requested: int):
    """Yield a resolved GPU count; blank CUDA_VISIBLE_DEVICES only for the fit scope."""
    if requested and requested > 0:
        yield int(requested)
        return
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield 0
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


@contextmanager
def _tmp_scope(tmp_root: Path):
    """Point TMP/TEMP/TMPDIR at tmp_root for the fit, then restore prior values."""
    tmp_root.mkdir(parents=True, exist_ok=True)
    keys = ("TMPDIR", "TEMP", "TMP")
    prev = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = str(tmp_root)
    try:
        yield tmp_root
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _coerce_target(y_raw: pd.Series) -> pd.Series:
    try:
        return y_raw.astype(int)
    except (ValueError, TypeError):
        return y_raw.astype(str)


def _set_local_tmp_dir(tmp_root: Path) -> Path:
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_root)
    os.environ["TEMP"] = str(tmp_root)
    os.environ["TMP"] = str(tmp_root)
    return tmp_root


def oracle_ceiling(select_imp: pd.Series, score_imp: pd.Series, k: int) -> float | None:
    """Fraction of the achievable top-k importance mass an HONEST oracle captures.

    Numerator: score-split mass of the k features chosen on the select split.
    Denominator: score-split mass of the true top k on the score split.

    Below 1 by exactly the amount the ranking is noise. This is the number that turns
    "captured importance is a lower bound of unknown tightness" into a calibrated
    fraction: the model can be reported against what a data-driven method itself
    achieves out-of-sample, not against an in-sample ideal nothing can reach.
    """
    if k <= 0 or select_imp.empty:
        return None
    chosen = select_imp.nlargest(k).index
    scored = score_imp.clip(lower=0)
    denom = float(scored.nlargest(k).sum())
    if denom <= 0:
        return None
    return round(float(scored.reindex(chosen).fillna(0.0).sum()) / denom, 4)



# == compute_oracle stages ====================================================
# One helper per pipeline stage so compute_oracle reads as the design:
#   rows -> columns -> target -> honest split -> fit + rank/value -> assemble.
# Pure re-organisation of the former ~250-line body; no behavioural change.


def _prepare_cell_frame(data, metadata, target_var, country_code, country_col, admin_cols):
    """Rows: country slice, metadata-aware cleaning, column dedupe, drop missing target."""
    country_data = data[data[country_col] == country_code].copy()
    if len(country_data) == 0:
        raise ValueError(
            f"No rows found for {country_col}={country_code!r}. "
            f"Actual values in data: {sorted(data[country_col].dropna().unique().tolist())}"
        )

    # metadata is REQUIRED: without labels the cleaner drops all negative codes,
    # destroying WVS "Don't know"(-1) / "No answer"(-2) (onboarding.md #5.4).
    country_data = clean_question_columns(country_data, country_col, admin_cols, metadata)

    if country_data.columns.duplicated().any():
        country_data = country_data.loc[:, ~country_data.columns.duplicated()]

    valid = country_data[target_var].notna()
    country_data = country_data.loc[valid]
    if len(country_data) == 0:
        raise ValueError(
            f"No valid (non-missing) rows for target '{target_var}' "
            f"in {country_col}={country_code!r}."
        )
    return country_data


def _build_cell_pool(country_data, flat_metadata, target_var, country_col, admin_cols,
                     similarity_model, similarity_threshold,
                     max_missingness_threshold, min_normalized_feature_entropy,
                     feature_pool_override):
    """Columns: candidate pool (built or overridden) -> skip-pattern leakage exclusion
    -> the feature frame X."""
    if feature_pool_override is None:
        base_pool, _ = build_feature_pool(
            country_data,
            flat_metadata,
            target_var,
            admin_cols,
            similarity_model,
            similarity_threshold,
        )
        feature_pool, _ = filter_feature_pool_for_country(
            country_data,
            base_pool,
            max_missingness_threshold,
            min_normalized_feature_entropy,
        )
    else:
        feature_pool = [
            c for c in feature_pool_override
            if c in country_data.columns and c not in admin_cols
            and c != target_var and c != country_col
        ]

    feature_pool = list(dict.fromkeys(feature_pool))

    leakage_set = detect_conditional_leakage(country_data, target_var, feature_pool)
    if leakage_set:
        print(f"  Conditional leakage excluded ({len(leakage_set)}): {sorted(leakage_set)}")
        feature_pool = [f for f in feature_pool if f not in leakage_set]

    if not feature_pool:
        raise ValueError("No usable features after feature-pool filtering.")

    X = country_data[feature_pool].copy()

    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        X = X.drop(columns=all_missing)

    if X.empty:
        raise ValueError("No usable features after dropping all-missing columns.")
    return X


def _prepare_target(country_data, X, y, metadata, target_var, problem_type,
                    min_class_count):
    """Target: type-specific preparation. Returns (X, y, problem_type, majority).

    regression - recover numeric scale positions (label text -> codes), keep only
    substantive scale points, drop out-of-scale sentinel codes.
    classification - restrict to labelled substantive values, drop rare classes, and
    let the DATA pick binary vs multiclass.
    """
    if problem_type == "regression":
        # Numeric scale positions (labels may be stored as text; unlabelled codes drop).
        # Respondent non-response leaves ONLY here — no position on an ordered scale.
        # Rationale: surveys.substantive_numeric_mask / onboarding.md #5.4.
        y = to_ordinal_codes(target_var, metadata, country_data[target_var])
        keep = y.notna() & substantive_numeric_mask(target_var, metadata, y)
        out_of_scale = out_of_scale_codes(_substantive_values(target_var, metadata))
        if out_of_scale:
            keep &= ~y.isin(out_of_scale)
        X, y = X.loc[keep], y.loc[keep]
        if y.nunique() < 3:
            raise ValueError("Ordinal/continuous target has fewer than 3 distinct values.")
    else:
        labelled = _substantive_values(target_var, metadata)
        if labelled:
            allowed = {str(c).strip() for k in labelled for c in _code_variants(k)}
            keep = y.astype(str).str.strip().isin(allowed)
            if keep.any():
                X, y = X.loc[keep], y.loc[keep]
        class_counts = y.value_counts()
        rare = class_counts[class_counts < min_class_count].index.tolist()
        if rare:
            keep = ~y.isin(rare)
            X = X.loc[keep]
            y = y.loc[keep]
        if y.nunique() < 2:
            raise ValueError("Not enough class variation after rare-class filtering.")
        # The DATA decides binary vs multiclass. The label count can disagree with it:
        # afro Q43A is labelled {0: No, 1: Yes} but carries unlabelled 8 and 9.
        problem_type = "binary" if y.nunique() == 2 else "multiclass"

    majority_baseline = float(y.value_counts().max() / len(y))
    return X, y, problem_type, majority_baseline


def _labelled_frames(X, y, train_idx, select_idx, score_idx, is_regression):
    """__label__-bearing frames for fit / select / score.

    Regression labels are floats; classification labels are Categoricals pinned to the
    TRAIN categories, with holdout rows whose class never appeared in training dropped.
    """
    train_data = X.iloc[train_idx].copy()
    if is_regression:
        train_data["__label__"] = y.iloc[train_idx].astype(float)
    else:
        train_data["__label__"] = pd.Categorical(y.iloc[train_idx].astype(str))
    categories = (None if is_regression
                  else train_data["__label__"].cat.categories)

    def _holdout_frame(idx: np.ndarray) -> pd.DataFrame:
        Xh = X.iloc[idx].copy()
        if is_regression:
            Xh["__label__"] = y.iloc[idx].astype(float).to_numpy()
            return Xh
        labels = pd.Categorical(y.iloc[idx].astype(str), categories=categories)
        keep = ~pd.isna(labels)
        Xh = Xh.loc[keep].copy()
        Xh["__label__"] = labels[keep]
        return Xh

    select_data = _holdout_frame(select_idx)
    score_data = _holdout_frame(score_idx)

    if len(select_data) == 0 or len(score_data) == 0:
        raise ValueError("No valid holdout rows after preprocessing.")
    return train_data, select_data, score_data


def _fit_and_rank(train_data, select_data, score_data, run_output_dir, *,
                  problem_type, resolved_metric, preset, time_limit,
                  num_gpus, num_cpus, n_repeats, ag_verbosity):
    """ONE AutoGluon fit on the train split, then TWO permutation-importance passes on
    the disjoint select/score holdouts - the honest-split core: V1 ranks, V2 values.
    Importance is far cheaper than the fit, so the second pass costs ~1.2x, not 2x."""
    predictor = TabularPredictor(
        label="__label__",
        problem_type=problem_type,
        eval_metric=resolved_metric,
        path=str(run_output_dir),
        verbosity=ag_verbosity,
    ).fit(
        train_data=train_data,
        presets=preset,
        time_limit=time_limit,
        num_gpus=num_gpus,
        # The worker's ACTUAL core budget; omitting it over-subscribes concurrent fits
        # and blows the wall-clock time_limit (onboarding.md #5).
        **({"num_cpus": int(num_cpus)} if num_cpus else {}),
        verbosity=ag_verbosity,
        # Both lines: keep Ray out — it deadlocks/crashes on Windows and silently drops
        # models (onboarding.md #5.7). Importance doesn't need stacking anyway.
        dynamic_stacking=False,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )

    def _importance(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        fi = predictor.feature_importance(
            frame, num_shuffle_sets=int(n_repeats), silent=ag_verbosity == 0
        )
        if fi.empty:
            raise ValueError("AutoGluon returned empty feature importance.")
        fi = fi.reset_index().rename(columns={
            "index": "feature_variable",
            "importance": f"importance_{suffix}",
            "stddev": f"importance_{suffix}_std",
        })
        if f"importance_{suffix}_std" not in fi.columns:
            fi[f"importance_{suffix}_std"] = 0.0
        return fi[["feature_variable", f"importance_{suffix}", f"importance_{suffix}_std"]]

    return _importance(select_data, "select"), _importance(score_data, "score")


# == contract v4: CV ranking (design rationale: module docstring) =============
# importance_select = mean over folds of fold-holdout importance; importance_score
# = mean over the fold models' importance on V2. Per-fold vectors are kept in
# oracle.csv; between-fold agreement lands in meta["reliability"]. The evaluator's
# CV shrinks from v3's 60% to D's 30%; paired arm contrasts (VoR/VoT) share eval
# rows, so the extra noise largely cancels (measured in the Phase A pilot).


def _cv_split(y: pd.Series, *, cv_folds: int, score_size: float, eval_reserve: float,
              random_state: int, stratify: bool):
    """v4 split: CV region / eval reserve / score holdout (default 50 / 30 / 20).

    Returns (fold_pairs, eval_idx, score_idx); fold_pairs is a list of
    (fit_idx, holdout_idx) positional-index pairs covering the CV region.
    """
    idx = np.arange(len(y))
    strat = y if stratify else None
    rest_idx, score_idx = train_test_split(
        idx, test_size=score_size, random_state=random_state, stratify=strat,
    )
    y_rest = y.iloc[rest_idx]
    try:
        cv_idx, eval_idx = train_test_split(
            rest_idx, test_size=eval_reserve / (1.0 - score_size),
            random_state=random_state, stratify=y_rest if stratify else None,
        )
    except ValueError:
        # Thin classes: an unstratified carve still gives a clean reserve.
        cv_idx, eval_idx = train_test_split(
            rest_idx, test_size=eval_reserve / (1.0 - score_size),
            random_state=random_state,
        )

    from sklearn.model_selection import KFold, StratifiedKFold

    y_cv = y.iloc[cv_idx]
    fold_pairs = []
    try:
        if not stratify:
            raise ValueError("regression target: plain KFold")
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=random_state)
        for fit_pos, hold_pos in splitter.split(np.zeros(len(y_cv)), y_cv):
            fold_pairs.append((cv_idx[fit_pos], cv_idx[hold_pos]))
    except ValueError:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        fold_pairs = [(cv_idx[f], cv_idx[h])
                      for f, h in splitter.split(np.zeros(len(y_cv)))]
    return fold_pairs, eval_idx, score_idx


def _merge_fold_importances(frames: list[pd.DataFrame], suffix: str) -> pd.DataFrame:
    """Outer-merge per-fold importance frames into fold columns + mean + between-fold std.

    Columns out: feature_variable, importance_<suffix>, importance_<suffix>_std,
    importance_<suffix>_fold<i>. The _std is the BETWEEN-FOLD std (refit + eval-row
    noise), not v3's within-pass shuffle std — meta records which contract wrote it.
    """
    merged: pd.DataFrame | None = None
    fold_cols = []
    for i, fi in enumerate(frames, start=1):
        col = f"importance_{suffix}_fold{i}"
        part = fi[["feature_variable", f"importance_{suffix}"]].rename(
            columns={f"importance_{suffix}": col}
        )
        fold_cols.append(col)
        merged = part if merged is None else merged.merge(
            part, on="feature_variable", how="outer"
        )
    for col in fold_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    merged[f"importance_{suffix}"] = merged[fold_cols].mean(axis=1)
    merged[f"importance_{suffix}_std"] = merged[fold_cols].std(axis=1, ddof=1)
    return merged


def _fold_reliability(merged: pd.DataFrame, suffix: str, k: int = 10) -> dict:
    """Between-fold agreement of one importance side (the per-cell noise floor).

    pairwise_jaccard@k / pairwise_spearman: mean agreement of SINGLE folds — compare
    against the v3 select-vs-score floor (~0.33 Jaccard@10). spearman_of_mean is the
    Spearman–Brown projection m*r / (1 + (m-1)*r): the reliability the AVERAGED
    ranking actually has, which is what v4 ranks on.
    """
    from itertools import combinations

    from scipy.stats import spearmanr

    fold_cols = [c for c in merged.columns if c.startswith(f"importance_{suffix}_fold")]
    m = len(fold_cols)
    if m < 2:
        return {}
    jac, rho = [], []
    for a, b in combinations(fold_cols, 2):
        top_a = set(merged.nlargest(k, a)["feature_variable"])
        top_b = set(merged.nlargest(k, b)["feature_variable"])
        union = top_a | top_b
        if union:
            jac.append(len(top_a & top_b) / len(union))
        r = spearmanr(merged[a], merged[b]).statistic
        if not np.isnan(r):
            rho.append(r)
    r1 = float(np.mean(rho)) if rho else None
    return {
        "n_folds": m,
        f"pairwise_jaccard{k}": round(float(np.mean(jac)), 4) if jac else None,
        "pairwise_spearman": round(r1, 4) if r1 is not None else None,
        "spearman_of_mean": (
            round(m * r1 / (1.0 + (m - 1) * r1), 4)
            if r1 is not None and (1.0 + (m - 1) * r1) > 0 else None
        ),
    }


def _assemble_outputs_cv(select_merged, score_merged, *, target_var, country_code,
                         majority_baseline, resolved_metric, problem_type, ttype,
                         cv_folds, fold_sizes, n_eval, n_score, y, eval_idx):
    """v4 counterpart of _assemble_outputs: fold columns kept, aliases preserved,
    reliability block added, train_index = the untouched eval reserve D."""
    oracle_df = select_merged.merge(score_merged, on="feature_variable", how="outer")
    num_cols = [c for c in oracle_df.columns if c != "feature_variable"]
    for col in num_cols:
        oracle_df[col] = pd.to_numeric(oracle_df[col], errors="coerce").fillna(0.0)

    oracle_df["importance_mean"] = oracle_df["importance_score"]
    oracle_df["importance_std"] = oracle_df["importance_score_std"]
    oracle_df.insert(0, "country", country_code)
    oracle_df.insert(0, "target_variable", target_var)
    oracle_df["majority_baseline"] = round(float(majority_baseline), 4)
    oracle_df = oracle_df.sort_values(
        "importance_select", ascending=False
    ).reset_index(drop=True)

    sel = oracle_df.set_index("feature_variable")["importance_select"]
    sco = oracle_df.set_index("feature_variable")["importance_score"]

    # v3-comparable honest agreement: averaged ranking vs disjoint valuation.
    top_sel = set(sel.nlargest(10).index)
    top_sco = set(sco.nlargest(10).index)
    select_score_j10 = (
        round(len(top_sel & top_sco) / len(top_sel | top_sco), 4)
        if (top_sel | top_sco) else None
    )

    meta_out = {
        "target_variable": target_var,
        "country": country_code,
        "contract_version": ORACLE_CONTRACT_VERSION,
        "eval_metric": resolved_metric,
        "problem_type": problem_type,
        "target_type": ttype,
        "cv_folds": int(cv_folds),
        "fold_fit_sizes": [int(n) for n in fold_sizes],
        "n_cv": int(len(y) - n_eval - n_score),
        "n_eval_reserve": int(n_eval),
        "n_score": int(n_score),
        # Distinct substantive values after _prepare_target; the regression
        # type-1 grid rule (grid_screen) needs it.
        "n_target_unique": int(y.nunique()),
        "n_features": int(len(oracle_df)),
        "n_positive_score": int((oracle_df["importance_score"] > 0).sum()),
        "majority_baseline": round(float(majority_baseline), 4),
        "oracle_ceiling": {str(k): oracle_ceiling(sel, sco, k) for k in CEILING_KS},
        "reliability": {
            "select": _fold_reliability(select_merged, "select"),
            "score": _fold_reliability(score_merged, "score"),
            "select_score_jaccard10": select_score_j10,
        },
        # v4 SEMANTIC CHANGE: these are the eval-RESERVE rows (D) — no oracle fit and
        # no importance pass ever saw them — not the fit rows v3 stored. Downstream
        # consumers (score_cell row_index) use them identically.
        "train_index": [int(i) if isinstance(i, (int, np.integer)) else str(i)
                        for i in y.index[eval_idx]],
    }

    feature_pool = oracle_df["feature_variable"].astype(str).tolist()
    return oracle_df, feature_pool, meta_out


def compute_oracle(
    data: pd.DataFrame,
    metadata: dict,
    target_var: str,
    country_code: int | str,
    country_col: str,
    admin_cols: frozenset[str],
    metadata_flat: dict[str, dict[str, str]] | None = None,
    similarity_model: object | None = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    max_missingness_threshold: float = MAX_MISSINGNESS_THRESHOLD,
    min_normalized_feature_entropy: float = MIN_NORMALIZED_FEATURE_ENTROPY,
    feature_pool_override: list[str] | None = None,
    tmp_root: Path | None = None,
    n_splits: int = 1,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
    runtime_mode: str = "balanced",
    autogluon_time_limit: int = 0,
    test_size: float = TEST_SIZE,
    min_class_count: int = MIN_CLASS_COUNT,
    num_gpus: int = 0,
    num_cpus: int | None = None,
    ag_verbosity: int = 0,
    eval_metric: str | None = None,
    target_type: str | None = None,
    survey_id: str | None = None,
    cv_folds: int = DEFAULT_CV_FOLDS,
    eval_reserve: float = EVAL_RESERVE,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Compute AutoGluon permutation importances for one target-country cell.

    Returns (oracle_df, feature_pool, meta). The frame carries `importance_select`
    (rank on the select holdout) and `importance_score` (value on a disjoint score
    holdout), with `importance_mean`/`importance_std` kept as aliases of the score
    columns so existing readers are unaffected.

    Contract v4: k-fold CV ranking over a 1 - test_size - eval_reserve region,
    valuation on the test_size score holdout, and train_index = the eval_reserve
    rows. cv_folds must be >= 2; the retired v3 one-shot split is gone.

    Note: n_splits is kept for drop-in compatibility with earlier oracle scripts but
    is not used by AutoGluon here (a single fit on the train split).
    """
    if target_var not in data.columns:
        raise ValueError(f"Target {target_var} missing from dataset columns.")
    if n_splits != 1:
        print("  Note: n_splits is ignored in AutoGluon oracle (single holdout).")

    # 1. Rows.
    country_data = _prepare_cell_frame(
        data, metadata, target_var, country_code, country_col, admin_cols
    )

    # 2. Columns.
    flat_metadata = metadata_flat or flatten_metadata(metadata)
    X = _build_cell_pool(
        country_data, flat_metadata, target_var, country_col, admin_cols,
        similarity_model, similarity_threshold,
        max_missingness_threshold, min_normalized_feature_entropy,
        feature_pool_override,
    )

    # 3. Target: measurement level decides how it is modelled and scored.
    #    Detected from value labels (auditable via scripts/audit_target_types.py);
    #    override with target_type=.
    y = _coerce_target(country_data[target_var])
    ttype = target_type or detect_target_type(
        target_var, metadata, y, survey=survey_id
    )[0]
    problem_type, resolved_metric = TARGET_TYPE_PROBLEM.get(
        ttype, ("multiclass", ORACLE_EVAL_METRIC)
    )
    if eval_metric is not None:
        resolved_metric = eval_metric
    X, y, problem_type, majority_baseline = _prepare_target(
        country_data, X, y, metadata, target_var, problem_type, min_class_count
    )
    is_regression = problem_type == "regression"

    preset, time_limit = resolve_runtime_config(runtime_mode, autogluon_time_limit)
    if tmp_root is None:
        tmp_root = tmp_dir(OUTPUTS_DIR)

    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2 (contract v4); the v3 one-shot "
                         "split is retired.")

    # 4-6 (contract v4). CV region ranks, D is reserved, V2 values.
    fold_pairs, eval_idx, score_idx = _cv_split(
        y, cv_folds=cv_folds, score_size=test_size, eval_reserve=eval_reserve,
        random_state=random_state, stratify=not is_regression,
    )
    fi_selects, fi_scores, fold_sizes = [], [], []
    with _gpu_scope(num_gpus) as resolved_gpus, _tmp_scope(tmp_root) as tmp_root:
        for fold_i, (fit_idx, hold_idx) in enumerate(fold_pairs, start=1):
            train_data, select_data, score_data = _labelled_frames(
                X, y, fit_idx, hold_idx, score_idx, is_regression
            )
            fold_sizes.append(len(train_data))
            run_output_dir = Path(tempfile.mkdtemp(
                prefix=f"autogluon_oracle_f{fold_i}_", dir=str(tmp_root)
            ))
            try:
                fi_sel, fi_sco = _fit_and_rank(
                    train_data, select_data, score_data, run_output_dir,
                    problem_type=problem_type, resolved_metric=resolved_metric,
                    preset=preset, time_limit=time_limit,
                    num_gpus=resolved_gpus, num_cpus=num_cpus,
                    n_repeats=n_repeats, ag_verbosity=ag_verbosity,
                )
            finally:
                shutil.rmtree(run_output_dir, ignore_errors=True)
            fi_selects.append(fi_sel)
            fi_scores.append(fi_sco)

    return _assemble_outputs_cv(
        _merge_fold_importances(fi_selects, "select"),
        _merge_fold_importances(fi_scores, "score"),
        target_var=target_var, country_code=country_code,
        majority_baseline=majority_baseline, resolved_metric=resolved_metric,
        problem_type=problem_type, ttype=ttype,
        cv_folds=cv_folds, fold_sizes=fold_sizes,
        n_eval=len(eval_idx), n_score=len(score_idx),
        y=y, eval_idx=eval_idx,
    )


if __name__ == "__main__":
    raise SystemExit("CLI moved to scripts/compute_oracle.py")
