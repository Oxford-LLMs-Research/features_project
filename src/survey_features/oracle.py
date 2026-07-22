"""
AutoGluon oracle permutation importances (requires the [oracle] extra: pip install -e .[oracle]).

Computes AutoGluon-based permutation importances for a single (target, country) cell and
writes outputs in the cache format used by the grid runners so the rest of the pipeline
can reuse them.

Output contract
---------------
compute_oracle() returns:
  oracle_df   - DataFrame with columns:
                target_variable, country, feature_variable,
                importance_mean, importance_std, majority_baseline
  feature_pool - list of feature variable codes included in the model

Cache contract
--------------
When run as a script (python -m survey_features.oracle), results are saved to:
  outputs/<target>_<country>/oracle.csv
and are picked up by scripts/run_grid.py (which skips oracle computation if the
file already exists).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from .config import OUTPUTS_DIR
from .surveys import (
    SURVEY_COUNTRY_COL,
    build_admin_cols,
    build_country_code_map,
    clean_question_columns,
    flatten_metadata,
    load_survey,
)


def _prewarm_autogluon_imports() -> None:
    """
    Force-import every AutoGluon tabular model submodule in the main thread.

    AutoGluon lazy-imports model implementations (and their callbacks /
    hyperparameter helpers) the first time each model is fit. When
    run_grid.py uses a ThreadPoolExecutor to fit multiple cells concurrently,
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

AUTOGLUON_RUNTIME_MODES = {
    "quick": {"preset": "medium_quality", "time_limit": 60},
    "balanced": {"preset": "good_quality", "time_limit": 180},
    "best": {"preset": "best_quality", "time_limit": 600},
}

ENFORCE_IDENTICAL_FEATURE_POOL = False
MAX_MISSINGNESS_THRESHOLD = 0.2
MIN_NORMALIZED_FEATURE_ENTROPY = 0.0

MIN_CLASS_COUNT = 5

DEFAULT_SURVEY = "wvs"
DEFAULT_TARGETS = ["Q47", "Q57", "Q199", "Q235", "Q164"]
DEFAULT_COUNTRIES = ["Germany", "Nigeria", "Japan", "Brazil", "Egypt"]


def load_similarity_model(similarity_threshold: float) -> object | None:
    """Embedding model for the oracle's semantic near-duplicate exclusion filter."""
    if similarity_threshold <= 0 or similarity_threshold >= 1:
        return None
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def build_feature_pool(
    df: pd.DataFrame,
    metadata: dict[str, dict[str, str]],
    target_var: str,
    admin_cols: frozenset[str],
    similarity_model: object | None,
    similarity_threshold: float,
) -> tuple[list[str], dict[str, int]]:
    """Create target-specific feature pool with optional semantic exclusion."""
    if target_var not in metadata:
        raise ValueError(f"Target {target_var} missing from metadata.")

    start_pool = list(dict.fromkeys([c for c in df.columns if c != target_var]))
    in_metadata = [c for c in start_pool if c in metadata]
    not_in_metadata = [c for c in start_pool if c not in metadata]
    admin_excluded = [c for c in in_metadata if c in admin_cols]
    candidates = [c for c in in_metadata if c not in admin_cols]

    if not candidates:
        diagnostics = {
            "start_pool": len(start_pool),
            "missing_metadata_excluded": len(not_in_metadata),
            "admin_excluded": len(admin_excluded),
            "semantic_excluded": 0,
            "final_pool": 0,
        }
        return [], diagnostics

    if similarity_model is None:
        similar_cols: list[str] = []
        feature_pool = candidates
    else:
        from sklearn.metrics.pairwise import cosine_similarity

        target_text = (
            metadata[target_var].get("question")
            or metadata[target_var].get("description")
            or target_var
        )
        candidate_texts = [
            metadata[c].get("question") or metadata[c].get("description") or c
            for c in candidates
        ]

        target_emb = similarity_model.encode([target_text])
        candidate_emb = similarity_model.encode(candidate_texts)
        similarities = cosine_similarity(target_emb, candidate_emb).flatten()

        similar_cols = [c for c, sim in zip(candidates, similarities) if sim > similarity_threshold]
        feature_pool = [c for c, sim in zip(candidates, similarities) if sim <= similarity_threshold]

    diagnostics = {
        "start_pool": len(start_pool),
        "missing_metadata_excluded": len(not_in_metadata),
        "admin_excluded": len(admin_excluded),
        "semantic_excluded": len(similar_cols),
        "final_pool": len(feature_pool),
    }
    return feature_pool, diagnostics


def standardize_missing(series: pd.Series) -> pd.Series:
    """Treat negative values and blanks as missing, preserving type when possible."""
    if pd.api.types.is_numeric_dtype(series):
        series_num = pd.to_numeric(series, errors="coerce")
        return series_num.mask(series_num < 0, np.nan)
    cleaned = series.copy()
    return cleaned.replace("", np.nan)


def normalized_entropy(values: pd.Series) -> float:
    """Compute normalized entropy for a categorical-like series."""
    clean = values.dropna()
    if clean.empty:
        return 0.0
    counts = clean.value_counts()
    n_unique = int(len(counts))
    if n_unique <= 1:
        return 0.0
    probs = counts / counts.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(entropy / np.log(n_unique))


def feature_passes_variation(series: pd.Series, min_normalized_feature_entropy: float) -> bool:
    """Check feature variation using entropy or numeric dispersion rules."""
    standardized = standardize_missing(series)
    non_missing = standardized.dropna()
    n_unique = int(non_missing.nunique())
    if pd.api.types.is_numeric_dtype(standardized) and n_unique >= 10:
        if non_missing.empty:
            return False
        std = float(non_missing.std(ddof=0))
        zero_share = float((non_missing == 0).mean())
        return std > 0 and zero_share < 0.95
    entropy = normalized_entropy(non_missing)
    return entropy >= min_normalized_feature_entropy


def filter_feature_pool_for_country(
    df_country: pd.DataFrame,
    feature_pool: list[str],
    max_missingness_threshold: float,
    min_normalized_feature_entropy: float,
) -> tuple[list[str], dict[str, int]]:
    """Apply missingness and variation filters for one country."""
    kept: list[str] = []
    missing_excluded = 0
    variation_excluded = 0

    for col in feature_pool:
        series = standardize_missing(df_country[col])
        n_total = int(len(series))
        missing_share = float(series.isna().mean()) if n_total else 1.0
        if missing_share >= max_missingness_threshold:
            missing_excluded += 1
            continue
        if not feature_passes_variation(series, min_normalized_feature_entropy):
            variation_excluded += 1
            continue
        kept.append(col)

    diagnostics = {
        "missingness_excluded": int(missing_excluded),
        "variation_excluded": int(variation_excluded),
        "final_pool": int(len(kept)),
    }
    return kept, diagnostics


def filter_feature_pool_across_countries(
    df: pd.DataFrame,
    feature_pool: list[str],
    country_codes: dict[str, int | str],
    country_col: str,
    max_missingness_threshold: float,
    min_normalized_feature_entropy: float,
) -> tuple[list[str], dict[str, int]]:
    """Keep only features that pass filters in every selected country."""
    missing_excluded: set[str] = set()
    variation_excluded: set[str] = set()

    for country_code in country_codes.values():
        df_country = df[df[country_col] == country_code]
        for col in feature_pool:
            if col in missing_excluded:
                continue
            series = standardize_missing(df_country[col])
            n_total = int(len(series))
            missing_share = float(series.isna().mean()) if n_total else 1.0
            if missing_share >= max_missingness_threshold:
                missing_excluded.add(col)
                continue
            if col in variation_excluded:
                continue
            if not feature_passes_variation(series, min_normalized_feature_entropy):
                variation_excluded.add(col)

    final_pool = [c for c in feature_pool if c not in missing_excluded and c not in variation_excluded]
    diagnostics = {
        "missingness_excluded": int(len(missing_excluded)),
        "variation_excluded": int(len(variation_excluded - missing_excluded)),
        "final_pool": int(len(final_pool)),
    }
    return final_pool, diagnostics


def detect_conditional_leakage(
    country_df: pd.DataFrame,
    target_var: str,
    feature_pool: list[str],
    missing_range_threshold: float = 0.7,
    min_n_per_class: int = 20,
) -> set[str]:
    """
    Identify features that are structurally conditional on the target variable.

    A feature is flagged when its missingness rate differs dramatically across
    target classes — the signature of a survey question only administered to
    respondents who gave a specific answer to the target (e.g. follow-up
    sub-modules asked only of respondents who said "Yes" to the target item).

    For example, Arab Barometer Q104A ("What are the main reasons you thought
    about emigrating?") is only asked of respondents who said Yes to Q104
    ("Have you thought about emigrating?").  Its missingness is ~0% for the
    Yes class and ~100% for the No class, yielding a range of ~1.0 — well
    above the default threshold of 0.7.

    Parameters
    ----------
    country_df              : DataFrame for one country; target column still present.
    target_var              : the outcome variable code.
    feature_pool            : candidate feature codes to inspect.
    missing_range_threshold : flag when max_class_missing - min_class_missing
                              >= this value. 0.7 gives a wide margin between clear
                              downstream proxies (~1.0) and legitimate predictors (~0.0).
    min_n_per_class         : minimum observations per class for a reliable rate.

    Returns
    -------
    set of feature variable codes to exclude from the feature pool.
    """
    valid = country_df[target_var].notna()
    df = country_df.loc[valid]
    y = df[target_var]

    class_counts = y.value_counts()
    usable_classes = class_counts[class_counts >= min_n_per_class].index.tolist()
    if len(usable_classes) < 2:
        return set()

    leakage_set: set[str] = set()
    for feat in feature_pool:
        if feat not in df.columns:
            continue
        rates = [df.loc[y == cls, feat].isna().mean() for cls in usable_classes]
        if (max(rates) - min(rates)) >= missing_range_threshold:
            leakage_set.add(feat)

    return leakage_set


def resolve_runtime_config(runtime_mode: str, autogluon_time_limit: int) -> tuple[str, int]:
    """Resolve AutoGluon preset + time limit from CLI settings."""
    runtime_cfg = AUTOGLUON_RUNTIME_MODES[runtime_mode]
    preset = str(runtime_cfg["preset"])
    time_limit = int(runtime_cfg["time_limit"])
    if autogluon_time_limit > 0:
        time_limit = autogluon_time_limit
    return preset, time_limit


def _detect_gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return shutil.which("nvidia-smi") is not None


def _resolve_num_gpus(requested: int) -> int:
    if requested and requested > 0:
        return int(requested)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return 0


def _coerce_target(y_raw: pd.Series) -> pd.Series:
    try:
        return y_raw.astype(int)
    except (ValueError, TypeError):
        return y_raw.astype(str)


def _safe_rmtree(path: Path, retries: int = 3, delay: float = 0.5) -> None:
    for attempt in range(retries):
        try:
            shutil.rmtree(path, ignore_errors=False)
            return
        except (OSError, PermissionError):
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                shutil.rmtree(path, ignore_errors=True)


def _set_local_tmp_dir(tmp_root: Path) -> Path:
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_root)
    os.environ["TEMP"] = str(tmp_root)
    os.environ["TMP"] = str(tmp_root)
    return tmp_root


def _feature_importance(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    n_repeats: int,
    verbosity: int = 0,
) -> pd.DataFrame:
    kwargs = {"silent": verbosity == 0}
    if n_repeats and n_repeats > 0:
        kwargs["num_shuffle_sets"] = int(n_repeats)
    try:
        return predictor.feature_importance(data, **kwargs)
    except TypeError:
        kwargs.pop("num_shuffle_sets", None)
        return predictor.feature_importance(data, **kwargs)


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
    ag_verbosity: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute AutoGluon permutation importances for one target-country cell.

    Note: n_splits is kept for drop-in compatibility with earlier oracle scripts but
    is not used by AutoGluon here (single holdout split).
    """
    if target_var not in data.columns:
        raise ValueError(f"Target {target_var} missing from dataset columns.")

    country_data = data[data[country_col] == country_code].copy()
    if len(country_data) == 0:
        raise ValueError(
            f"No rows found for {country_col}={country_code!r}. "
            f"Actual values in data: {sorted(data[country_col].dropna().unique().tolist())}"
        )

    if n_splits != 1:
        print("  Note: n_splits is ignored in AutoGluon oracle (single holdout).")

    country_data = clean_question_columns(country_data, country_col, admin_cols)

    if country_data.columns.duplicated().any():
        country_data = country_data.loc[:, ~country_data.columns.duplicated()]

    valid = country_data[target_var].notna()
    country_data = country_data.loc[valid]
    if len(country_data) == 0:
        raise ValueError(
            f"No valid (non-missing) rows for target '{target_var}' "
            f"in {country_col}={country_code!r}."
        )

    flat_metadata = metadata_flat or flatten_metadata(metadata)
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
            if c in country_data.columns and c not in admin_cols and c != target_var and c != country_col
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

    y = _coerce_target(country_data[target_var])

    class_counts = y.value_counts()
    rare = class_counts[class_counts < min_class_count].index.tolist()
    if rare:
        keep = ~y.isin(rare)
        X = X.loc[keep]
        y = y.loc[keep]

    if y.nunique() < 2:
        raise ValueError("Not enough class variation after rare-class filtering.")

    majority_baseline = float(y.value_counts().max() / len(y))

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_data = X_train.copy()
    train_data["__label__"] = pd.Categorical(y_train.astype(str))

    test_labels = pd.Categorical(y_test.astype(str), categories=train_data["__label__"].cat.categories)
    valid_test = ~pd.isna(test_labels)
    if not valid_test.all():
        X_test = X_test.loc[valid_test].copy()
        test_labels = test_labels[valid_test]

    test_data = X_test.copy()
    test_data["__label__"] = test_labels

    if len(test_data) == 0:
        raise ValueError("No valid test rows after preprocessing.")

    preset, time_limit = resolve_runtime_config(runtime_mode, autogluon_time_limit)
    num_gpus = _resolve_num_gpus(num_gpus)

    # Use a temp directory to avoid file-lock issues on Dropbox-backed folders.
    if tmp_root is None:
        tmp_root = OUTPUTS_DIR / ".tmp"
    tmp_root = _set_local_tmp_dir(tmp_root)
    temp_dir = tempfile.mkdtemp(prefix="autogluon_oracle_", dir=str(tmp_root))
    run_output_dir = Path(temp_dir)

    try:
        predictor = TabularPredictor(
            label="__label__",
            problem_type="multiclass",
            eval_metric="accuracy",
            path=str(run_output_dir),
            verbosity=ag_verbosity,
        ).fit(
            train_data=train_data,
            presets=preset,
            time_limit=time_limit,
            num_gpus=num_gpus,
            verbosity=ag_verbosity,
            # Ray-backed sub-fit deadlocks on Windows (AutoGluon 1.4 + Ray 2.44);
            # disable dynamic stacking since oracle permutation importance doesn't need it.
            dynamic_stacking=False,
            # Force fold-level fitting to skip Ray. With concurrent grid workers, each
            # cell would otherwise race to start its own Ray cluster + dashboard, which
            # crashes on Windows + Python 3.9 (typing.py:734 "unhashable type: 'list'")
            # and causes every *_BAG_L1/L2 model to be silently skipped.
            ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        )

        fi = _feature_importance(predictor, test_data, n_repeats, verbosity=ag_verbosity)
        if fi.empty:
            raise ValueError("AutoGluon returned empty feature importance.")

        fi = fi.reset_index().rename(
            columns={"index": "feature_variable", "importance": "importance_mean", "stddev": "importance_std"}
        )
        if "importance_std" not in fi.columns:
            fi["importance_std"] = 0.0

    finally:
        _safe_rmtree(run_output_dir)

    oracle_df = fi[["feature_variable", "importance_mean", "importance_std"]].copy()
    oracle_df.insert(0, "country", country_code)
    oracle_df.insert(0, "target_variable", target_var)
    oracle_df["majority_baseline"] = round(float(majority_baseline), 4)

    feature_pool = oracle_df["feature_variable"].astype(str).tolist()
    return oracle_df, feature_pool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoGluon oracle runner.")
    parser.add_argument(
        "--survey",
        default=DEFAULT_SURVEY,
        choices=list(SURVEY_COUNTRY_COL),
        help=f"Survey to run (default: {DEFAULT_SURVEY}).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        metavar="VAR",
        help="Target variable(s). Defaults to WVS targets when --survey=wvs.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        metavar="COUNTRY",
        help="Country name(s). Defaults to WVS countries when --survey=wvs.",
    )
    parser.add_argument(
        "--list-countries",
        action="store_true",
        help="Print available countries for the chosen survey and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root (default: outputs/).",
    )
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="balanced",
        choices=["quick", "balanced", "best"],
        help="AutoGluon runtime profile.",
    )
    parser.add_argument(
        "--autogluon-time-limit",
        type=int,
        default=0,
        help="Override AutoGluon time limit in seconds (0 = default per mode).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Holdout test size (default: {TEST_SIZE}).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed (default: {RANDOM_STATE}).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Semantic exclusion threshold (default: {SIMILARITY_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-missingness-threshold",
        type=float,
        default=MAX_MISSINGNESS_THRESHOLD,
        help=f"Drop features with missingness >= threshold (default: {MAX_MISSINGNESS_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-normalized-feature-entropy",
        type=float,
        default=MIN_NORMALIZED_FEATURE_ENTROPY,
        help=f"Minimum normalized entropy (default: {MIN_NORMALIZED_FEATURE_ENTROPY}).",
    )
    parser.add_argument(
        "--enforce-identical-feature-pool",
        action="store_true",
        default=ENFORCE_IDENTICAL_FEATURE_POOL,
        help="Use a single feature pool per target across selected countries.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=MIN_CLASS_COUNT,
        help=f"Minimum class count (default: {MIN_CLASS_COUNT}).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Permutation repeats for feature importance (AutoGluon shuffle sets).",
    )
    parser.add_argument(
        "--auto-detect-gpu",
        action="store_true",
        default=False,
        help="Use one GPU if available; otherwise force CPU.",
    )
    parser.add_argument(
        "--max-cells-per-run",
        type=int,
        default=0,
        help="If >0, process at most this many target-country cells.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if outputs/<target>_<country>/oracle.csv exists.",
    )
    parser.add_argument(
        "--ag-verbosity",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help=(
            "AutoGluon verbosity (0=silent, 2=info, 3=info+model details, 4=debug). "
            "Use 3+ to surface 'skipping model X' tracebacks while debugging."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    survey_id = args.survey
    country_col = SURVEY_COUNTRY_COL[survey_id]
    targets = args.targets or (DEFAULT_TARGETS if survey_id == DEFAULT_SURVEY else None)
    countries = args.countries or (DEFAULT_COUNTRIES if survey_id == DEFAULT_SURVEY else None)

    config_path = os.environ.get("DATA_CONFIG_PATH")
    if not config_path:
        raise ValueError("DATA_CONFIG_PATH is not set in .env")

    data, metadata = load_survey(survey_id, config_path)
    country_codes = build_country_code_map(metadata, country_col, data)
    admin_cols = build_admin_cols(metadata, country_col)
    metadata_flat = flatten_metadata(metadata)
    similarity_model = load_similarity_model(args.similarity_threshold)

    if args.list_countries:
        print(f"\nAvailable countries for '{survey_id}':")
        for name in sorted(country_codes):
            print(f"  {name} ({country_codes[name]})")
        return

    if not targets:
        raise SystemExit(f"--targets is required for survey '{survey_id}'")
    if not countries:
        raise SystemExit(f"--countries is required for survey '{survey_id}'")

    unknown = [c for c in countries if c not in country_codes]
    if unknown:
        raise ValueError(
            f"Unknown country/ies {unknown} for survey '{survey_id}'. "
            f"Run with --list-countries to see valid names."
        )

    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else OUTPUTS_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root = _set_local_tmp_dir(output_root / ".tmp")

    num_gpus = 1 if (args.auto_detect_gpu and _detect_gpu_available()) else 0
    num_gpus = _resolve_num_gpus(num_gpus)

    processed = 0
    errors: list[dict] = []

    print("\n" + "=" * 72)
    print(f"Survey: {survey_id}  |  country column: {country_col}")
    print(f"Targets:   {targets}")
    print(f"Countries: {countries}")
    print(f"AutoGluon: runtime={args.runtime_mode}, time_limit={args.autogluon_time_limit or 'default'}")
    print(f"GPU: num_gpus={num_gpus}")
    print("=" * 72)

    selected_country_codes = {c: country_codes[c] for c in countries}

    for target_var in targets:
        if target_var not in data.columns:
            raise ValueError(f"Target {target_var} missing from dataset columns.")

        target_feature_pool = None
        if args.enforce_identical_feature_pool:
            df_target = data[data[country_col].isin(selected_country_codes.values())].copy()
            df_target = clean_question_columns(df_target, country_col, admin_cols)
            df_target = df_target[df_target[target_var].notna()].copy()
            base_pool, _ = build_feature_pool(
                df_target,
                metadata_flat,
                target_var,
                admin_cols,
                similarity_model,
                args.similarity_threshold,
            )
            target_feature_pool, _ = filter_feature_pool_across_countries(
                df_target,
                base_pool,
                selected_country_codes,
                country_col,
                args.max_missingness_threshold,
                args.min_normalized_feature_entropy,
            )

        for country_name in countries:
            country_code = country_codes[country_name]
            prefix = f"{target_var}_{country_name}"
            cell_dir = output_root / prefix
            cell_dir.mkdir(parents=True, exist_ok=True)
            oracle_path = cell_dir / "oracle.csv"

            if oracle_path.exists() and not args.force:
                print(f"[skip] {prefix}: oracle.csv already exists")
                continue

            print(f"\n[oracle] {target_var} x {country_name}")
            try:
                oracle_df, feature_pool = compute_oracle(
                    data=data,
                    metadata=metadata,
                    target_var=target_var,
                    country_code=country_code,
                    country_col=country_col,
                    admin_cols=admin_cols,
                    metadata_flat=metadata_flat,
                    similarity_model=similarity_model,
                    similarity_threshold=args.similarity_threshold,
                    max_missingness_threshold=args.max_missingness_threshold,
                    min_normalized_feature_entropy=args.min_normalized_feature_entropy,
                    feature_pool_override=target_feature_pool,
                    tmp_root=tmp_root,
                    n_splits=1,
                    n_repeats=args.n_repeats,
                    random_state=args.random_state,
                    runtime_mode=args.runtime_mode,
                    autogluon_time_limit=args.autogluon_time_limit,
                    test_size=args.test_size,
                    min_class_count=args.min_class_count,
                    num_gpus=num_gpus,
                    ag_verbosity=args.ag_verbosity,
                )

                oracle_df.to_csv(oracle_path, index=False)
                pd.DataFrame({"feature_variable": feature_pool}).to_csv(
                    cell_dir / "feature_pool.csv", index=False
                )
                print(f"  Saved {oracle_path}")
                processed += 1

            except Exception as exc:
                errors.append({"target": target_var, "country": country_name, "error": str(exc)})
                print(f"  [error] {type(exc).__name__}: {exc}")

            if args.max_cells_per_run > 0 and processed >= args.max_cells_per_run:
                print("\nMax cells per run reached; stopping.")
                break

        if args.max_cells_per_run > 0 and processed >= args.max_cells_per_run:
            break

    if errors:
        print("\n" + "=" * 72)
        print(f"Errors in {len(errors)} cell(s):")
        for err in errors:
            print(f"  - {err['target']} x {err['country']}: {err['error']}")


if __name__ == "__main__":
    main()
