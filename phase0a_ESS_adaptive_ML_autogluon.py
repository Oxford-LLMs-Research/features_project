"""Phase 0a feature-importance heterogeneity analysis for ESS.

This script implements the following steps:
- load/filter ESS data and metadata,
- build target-specific feature pools with semantic similarity exclusion,
- fit country-specific XGBoost classifiers,
- compute held-out permutation importances,
- compute cross-country rank correlations,
- produce heatmaps and CSV outputs.

"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularPredictor
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm


RANDOM_STATE = 42
SIMILARITY_THRESHOLD = 0.85
TOP_FEATURES_HEATMAP = 20
TEST_SIZE = 0.2

AUTOGLUON_PRESET = "best_quality"
AUTOGLUON_TIME_LIMIT = 180
AUTOGLUON_RUNTIME_MODES = {
    "quick": {"preset": "medium_quality", "time_limit": 60},
    "balanced": {"preset": "good_quality", "time_limit": 180},
    "best": {"preset": "best_quality", "time_limit": 600},
}

# Set to B_COUNTRY for WVS and cntry for ESS.
COUNTRY_COL = "cntry"

ENFORCE_IDENTICAL_FEATURE_POOL = True
MAX_MISSINGNESS_THRESHOLD = 0.2
MIN_NORMALIZED_FEATURE_ENTROPY = 0.0


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data/ess/ESS10_with_consolidations.csv"
METADATA_PATH = SCRIPT_DIR / "data/ess/ess10_profiles_metadata.json"
OUTPUT_DIR = SCRIPT_DIR / "tests/permutation_importance_ESS_10_outputs"


COUNTRY_CODES = {
    "BE": "Belgium",
    "CH": "Switzerland",
    "GR": "Greece",
    "PT": "Portugal",
    "FR": "France",
}

# TARGET_QUESTIONS = [  # WVS
#     "Q47", # subjective health
#     "Q57", # trust in people 
#     "Q199", # interest in politics
#     "Q235", # perference for strong leader vs democracy
#     "Q164" # importance of god in life
#     ]

TARGET_QUESTIONS = [ # ESS10
    "health", # subjective health
    "ppltrst", # trust in people
    "polintr", # interest in politics
    "accalaw", # acceptability of strong leader above law
    "rlgdgr" # religiosity
    ]


# Define a dataclass to hold results and diagnostics for each target-country cell.
@dataclass
class CellResult:
    target_variable: str
    country: str
    country_name: str
    n_rows: int
    n_classes: int
    min_class_count: int
    cv_folds: int
    cv_accuracy_mean: float
    cv_accuracy_std: float
    majority_baseline: float
    cv_at_or_below_baseline: bool
    class_mapping: str
    skipped: bool
    skip_reason: str
    low_n_flag: bool
    sparse_class_flag: bool


# Helper functions for each step of the process.
def flatten_metadata(raw_metadata: Dict) -> Dict[str, Dict[str, str]]:
    """Flatten nested metadata dict into variable -> metadata fields."""
    flat: Dict[str, Dict[str, str]] = {}
    for section, section_vars in raw_metadata.items():
        if not isinstance(section_vars, dict):
            continue
        for var_name, var_meta in section_vars.items():
            if not isinstance(var_meta, dict):
                continue
            flat[var_name] = {
                "section": section,
                "question": str(var_meta.get("question", "")).strip(),
                "description": str(var_meta.get("description", "")).strip(),
            }
    return flat


def load_inputs() -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """Load data and flattened metadata."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata_raw = json.load(f)
    metadata = flatten_metadata(metadata_raw)
    return df, metadata


def identify_admin_columns(
    columns: List[str], metadata: Dict[str, Dict[str, str]]
) -> List[str]:
    """Heuristically identify admin/ID fields for exclusion."""
    # This is a fairly activist approach to removing potential admin variables - careful when modifying/using for other datasets. Checked for WVS and ESS wave 10
    name_pattern = re.compile(
        r"(^B_COUNTRY$|^CNTRY$|^B_[A-Z0-9_]+$|^S0\d+|^A_YEAR$|^A_MONTH$|^A_DAY$|^W_+|"
        r"^N_REGION_ISO$|^ID$|ID$|CASEID|SERIAL|STRATUM|PSU|WEIGHT)",
        flags=re.IGNORECASE,
    )
    text_keywords = [
        "respondent id",
        "interview date",
        "country code",
        "survey wave",
        "weight",
        "sampling unit",
        "stratum"
    ]

    admin_cols: List[str] = [
        "version",
        "B_COUNTRY",
        "cntry",
        "B_COUNTRY_ALPHA",
        "C_COW_NUM",
        "C_COW_ALPHA",
        "D_INTERVIEW",
        "S007",
        "J_INTDATE",
        "FW_START",
        "FW_END",
        "K_TIME_START",
        "K_TIME_END",
        "K_DURATION",
        "Q_MODE",
        "N_REGION_ISO",
        "N_REGION_WVS",
        "N_REGION_NUTS2",
        "N_REG_NUTS1",
        "N_TOWN",
        "G_TOWNSIZE",
        "G_TOWNSIZE2",
        "H_SETTLEMENT",
        "H_URBRURAL",
        "I_PSU",
        "O1_LONGITUDE",
        "O2_LATITUDE",
        "L_INTERVIEWER_NUMBER",
        "S_INTLANGUAGE",
        "LNGE_ISO",
        "E_RESPINT",
        "F_INTPRIVACY",
        "E1_LITERACY",
        "W_WEIGHT",
        "S018",
        "PWGHT",
        "S025",
    ]
    for col in columns:
        if name_pattern.search(col):
            admin_cols.append(col)
            continue
        meta = metadata.get(col)
        if not meta:
            continue
        text = f"{meta.get('question', '')} {meta.get('description', '')}".lower()
        if any(keyword in text for keyword in text_keywords):
            admin_cols.append(col)

    return sorted(set(admin_cols))


def build_feature_pool(
    df: pd.DataFrame,
    metadata: Dict[str, Dict[str, str]],
    target_var: str,
    model: SentenceTransformer,
) -> Tuple[List[str], Dict[str, int]]:
    """Create target-specific feature pool with semantic exclusion."""
    if target_var not in metadata:
        raise ValueError(f"Target {target_var} missing from metadata.")

    # Deduplicate column names while preserving order.
    start_pool = list(dict.fromkeys([c for c in df.columns if c != target_var]))
    in_metadata = [c for c in start_pool if c in metadata]
    not_in_metadata = [c for c in start_pool if c not in metadata]
    admin_cols = identify_admin_columns(in_metadata, metadata)
    candidates = [c for c in in_metadata if c not in admin_cols]

    # Track variables with missing/empty question wording
    vars_with_missing_text = []
    for c in candidates:
        question = metadata[c].get("question", "").strip()
        description = metadata[c].get("description", "").strip()
        if not question and not description:
            vars_with_missing_text.append(c)

    target_text = metadata[target_var].get("question") or metadata[target_var].get("description") or target_var
    candidate_texts = [metadata[c].get("question") or metadata[c].get("description") or c for c in candidates]

    target_emb = model.encode([target_text])
    candidate_emb = model.encode(candidate_texts)
    similarities = cosine_similarity(target_emb, candidate_emb).flatten()

    similar_cols = [c for c, sim in zip(candidates, similarities) if sim > SIMILARITY_THRESHOLD]
    feature_pool = [c for c, sim in zip(candidates, similarities) if sim <= SIMILARITY_THRESHOLD]

    diagnostics = {
        "start_pool": len(start_pool),
        "missing_metadata_excluded": len(not_in_metadata),
        "admin_excluded": len(admin_cols),
        "semantic_excluded": len(similar_cols),
        "final_pool": len(feature_pool),
        "vars_with_missing_text": len(vars_with_missing_text),
        "vars_missing_text_list": ",".join(vars_with_missing_text[:10]),  # Store first 10 for logging
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


def feature_passes_variation(series: pd.Series) -> bool:
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
    return entropy >= MIN_NORMALIZED_FEATURE_ENTROPY


def filter_feature_pool_for_country(
    df_country: pd.DataFrame,
    feature_pool: List[str],
) -> Tuple[List[str], Dict[str, int]]:
    """Apply missingness and variation filters for one country."""
    kept: List[str] = []
    missing_excluded = 0
    variation_excluded = 0

    for col in feature_pool:
        series = standardize_missing(df_country[col])
        n_total = int(len(series))
        missing_share = float(series.isna().mean()) if n_total else 1.0
        if missing_share >= MAX_MISSINGNESS_THRESHOLD:
            missing_excluded += 1
            continue
        if not feature_passes_variation(series):
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
    feature_pool: List[str],
) -> Tuple[List[str], Dict[str, int]]:
    """Keep only features that pass filters in every country."""
    missing_excluded: set[str] = set()
    variation_excluded: set[str] = set()

    for country_code in COUNTRY_CODES.keys():
        df_country = df[df[COUNTRY_COL] == country_code]
        for col in feature_pool:
            if col in missing_excluded:
                continue
            series = standardize_missing(df_country[col])
            n_total = int(len(series))
            missing_share = float(series.isna().mean()) if n_total else 1.0
            if missing_share >= MAX_MISSINGNESS_THRESHOLD:
                missing_excluded.add(col)
                continue
            if col in variation_excluded:
                continue
            if not feature_passes_variation(series):
                variation_excluded.add(col)

    final_pool = [c for c in feature_pool if c not in missing_excluded and c not in variation_excluded]
    diagnostics = {
        "missingness_excluded": int(len(missing_excluded)),
        "variation_excluded": int(len(variation_excluded - missing_excluded)),
        "final_pool": int(len(final_pool)),
    }
    return final_pool, diagnostics


def prepare_model_inputs(
    df_country_train: pd.DataFrame,
    df_country_test: pd.DataFrame,
    target_var: str,
    feature_pool: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[int, int], Dict[str, object]]:
    """Prepare train/test X and y for one target-country cell."""
    y_train_raw = pd.to_numeric(df_country_train[target_var], errors="coerce")
    y_test_raw = pd.to_numeric(df_country_test[target_var], errors="coerce")

    y_train_mask = y_train_raw >= 0
    y_test_mask = y_test_raw >= 0

    y_train = y_train_raw.loc[y_train_mask].astype(int)
    y_test = y_test_raw.loc[y_test_mask].astype(int)

    X_train = df_country_train.loc[y_train_mask, feature_pool].copy()
    X_test = df_country_test.loc[y_test_mask, feature_pool].copy()

    X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
    X_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()

    X_train = X_train.apply(standardize_missing)
    X_test = X_test.apply(standardize_missing)
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            X_train[col] = X_train[col].mask(X_train[col] < 0, np.nan)
            if col in X_test.columns:
                X_test[col] = X_test[col].mask(X_test[col] < 0, np.nan)

    # Keep features available in train and test to ensure a stable holdout evaluation.
    train_available = set(X_train.columns[X_train.notna().any()].tolist())
    test_available = set(X_test.columns[X_test.notna().any()].tolist())
    available_cols = [c for c in X_train.columns if c in train_available and c in test_available]
    dropped_all_nan = len(X_train.columns) - len(available_cols)
    X_train = X_train[available_cols]
    X_test = X_test[available_cols]

    unique_vals = sorted(y_train.unique().tolist())
    class_mapping = {orig: idx for idx, orig in enumerate(unique_vals)}
    y_train_encoded = y_train.map(class_mapping)
    y_test_encoded = y_test.map(class_mapping)

    valid_test_mask = y_test_encoded.notna()
    X_test = X_test.loc[valid_test_mask].copy()
    y_test_encoded = y_test_encoded.loc[valid_test_mask].astype(int)
    y_train_encoded = y_train_encoded.astype(int)

    counts = y_train_encoded.value_counts().sort_index()
    diagnostics = {
        "n_rows": int(len(y_train_encoded)),
        "n_test_rows": int(len(y_test_encoded)),
        "n_classes": int(y_train_encoded.nunique()),
        "min_class_count": int(counts.min()) if len(counts) else 0,
        "dropped_all_nan_features": int(dropped_all_nan),
        "class_counts": counts.to_dict(),
    }

    return X_train, y_train_encoded, X_test, y_test_encoded, class_mapping, diagnostics


def country_stratified_train_test_split(
    df_target: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a train/test split stratified by country."""
    if df_target.empty:
        raise ValueError("No rows available after target-missing filtering.")

    country_counts = df_target[COUNTRY_COL].value_counts()
    if int(country_counts.min()) < 2:
        raise ValueError("Cannot stratify by country: at least one country has fewer than 2 rows.")

    train_idx, test_idx = train_test_split(
        np.arange(len(df_target)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_target[COUNTRY_COL],
    )
    train_df = df_target.iloc[train_idx].copy()
    test_df = df_target.iloc[test_idx].copy()
    return train_df, test_df


def evaluate_and_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_output_dir: Path,
    autogluon_preset: str,
    autogluon_time_limit: int,
) -> Tuple[pd.DataFrame, float]:
    """Fit AutoGluon and compute held-out feature importance."""
    if len(y_train) < 2 or y_train.nunique() < 2:
        raise ValueError("Not enough train variation for multiclass classification.")
    if len(y_test) < 1:
        raise ValueError("No valid test rows after preprocessing.")

    model_output_dir.mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    test_data = X_test.copy()
    train_data["__label__"] = pd.Categorical(y_train.astype(str))
    test_data["__label__"] = pd.Categorical(y_test.astype(str), categories=train_data["__label__"].cat.categories)

    predictor = TabularPredictor(
        label="__label__",
        problem_type="multiclass",
        eval_metric="accuracy",
        path=str(model_output_dir),
    ).fit(
        train_data=train_data,
        presets=autogluon_preset,
        time_limit=autogluon_time_limit,
        verbosity=0,
    )

    eval_result = predictor.evaluate(test_data, silent=True)
    test_accuracy = float(eval_result.get("accuracy", np.nan))

    fi = predictor.feature_importance(test_data, silent=True)
    if fi.empty:
        raise ValueError("AutoGluon returned empty feature importance.")

    fi = fi.reset_index().rename(columns={"index": "feature_variable"})
    if "importance" not in fi.columns:
        raise ValueError("AutoGluon feature importance output missing 'importance' column.")
    if "stddev" not in fi.columns:
        fi["stddev"] = 0.0

    return fi[["feature_variable", "importance", "stddev"]], test_accuracy


def build_rank_correlations(
    importance_df: pd.DataFrame,
    targets: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Spearman rank correlations per target."""
    records = []
    mean_rho_records = []

    for target in targets:
        subset = importance_df[importance_df["target_variable"] == target]
        pivot = subset.pivot_table(
            index="feature_variable",
            columns="country",
            values="importance_mean",
            aggfunc="mean",
        )

        country_list = list(COUNTRY_CODES.keys())
        for country_a in country_list:
            for country_b in country_list:
                s1 = pivot.get(country_a)
                s2 = pivot.get(country_b)

                if s1 is None or s2 is None:
                    rho = np.nan
                    n_overlap = 0
                else:
                    pair_df = pd.concat([s1, s2], axis=1).dropna()
                    n_overlap = int(len(pair_df))
                    if n_overlap < 2:
                        rho = np.nan
                    else:
                        rho = float(spearmanr(pair_df.iloc[:, 0], pair_df.iloc[:, 1]).statistic) # type: ignore

                records.append(
                    {
                        "target_variable": target,
                        "country_a": country_a,
                        "country_b": country_b,
                        "country_a_name": COUNTRY_CODES[country_a],
                        "country_b_name": COUNTRY_CODES[country_b],
                        "spearman_rho": rho,
                        "n_overlap_features": n_overlap,
                    }
                )

        target_pairs = [
            r
            for r in records
            if r["target_variable"] == target and r["country_a"] < r["country_b"] and not pd.isna(r["spearman_rho"])
        ]
        mean_rho = float(np.mean([r["spearman_rho"] for r in target_pairs])) if target_pairs else np.nan
        mean_rho_records.append({"target_variable": target, "mean_pairwise_rho": mean_rho})

    return pd.DataFrame(records), pd.DataFrame(mean_rho_records)


def make_heatmaps(
    importance_df: pd.DataFrame,
    metadata: Dict[str, Dict[str, str]],
    targets: List[str],
    output_prefix: str,
) -> None:
    """Create per-target heatmaps of top features by average importance."""
    for target in targets:
        subset = importance_df[importance_df["target_variable"] == target]
        if subset.empty:
            continue

        pivot = subset.pivot_table(
            index="feature_variable",
            columns="country",
            values="importance_mean",
            aggfunc="mean",
        )
        if pivot.empty:
            continue

        avg_importance = pivot.mean(axis=1, skipna=True).sort_values(ascending=False)
        top_features = avg_importance.head(TOP_FEATURES_HEATMAP).index.tolist()
        hm = pivot.loc[top_features].copy()

        hm.columns = [COUNTRY_CODES.get(c, str(c)) for c in hm.columns] # type: ignore

        # Ensure heatmap matrix is numeric (protect against object dtype from CSV/concat)
        hm = hm.apply(pd.to_numeric, errors="coerce")
        hm = hm.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if hm.empty:
            continue

        label_map = {
            var: metadata.get(var, {}).get("description") or metadata.get(var, {}).get("question") or var
            for var in hm.index
        }

        # Wrap and slightly truncate long labels so they don't dominate the figure.
        wrapped_labels = []
        for var in hm.index:
            raw = str(label_map[var])
            if len(raw) > 90:
                raw = raw[:87] + "..."
            wrapped_labels.append(textwrap.fill(raw, width=42))
        hm.index = wrapped_labels

        fig, ax = plt.subplots(figsize=(12, max(7, len(hm) * 0.42)))

        sns.heatmap(
            hm,
            cmap="YlOrRd",
            linewidths=0.25,
            ax=ax,
            cbar_kws={"shrink": 0.85},
        )

        ax.set_title(f"Phase 0a Feature Importance Heatmap - {target}", fontsize=13, pad=10)
        ax.set_xlabel("Country", fontsize=11)
        ax.set_ylabel("Feature (metadata description)", fontsize=10)

        # Reduce label size and rotate x labels to avoid overlap.
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=10, rotation=25)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")

        # Give the long y labels enough room.
        fig.subplots_adjust(left=0.50, right=0.93, bottom=0.18, top=0.92)

        fig.savefig(OUTPUT_DIR / f"{output_prefix}_heatmap_{target}.png", dpi=220)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0a ESS feature-importance heterogeneity.")
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(TARGET_QUESTIONS),
        help="Comma-separated target variables to run (default: all 5 targets).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="phase0a",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--max-cells-per-run",
        type=int,
        default=0,
        help="If >0, process at most this many target-country cells in one run.",
    )
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="balanced",
        choices=["quick", "balanced", "best"],
        help="AutoGluon runtime profile: quick, balanced, or best.",
    )
    parser.add_argument(
        "--autogluon-time-limit",
        type=int,
        default=0,
        help="Optional override for AutoGluon time limit (seconds). If 0, uses runtime-mode default.",
    )
    return parser.parse_args()


def resolve_runtime_config(runtime_mode: str, autogluon_time_limit: int) -> Tuple[str, int]:
    """Resolve AutoGluon preset and time limit from CLI settings."""
    runtime_cfg = AUTOGLUON_RUNTIME_MODES[runtime_mode]
    preset = str(runtime_cfg["preset"])
    time_limit = int(runtime_cfg["time_limit"])
    if autogluon_time_limit > 0:
        time_limit = autogluon_time_limit
    return preset, time_limit


def run(
    targets: List[str],
    output_prefix: str,
    max_cells_per_run: int = 0,
    runtime_mode: str = "balanced",
    autogluon_time_limit: int = 0,
) -> None:
    df, metadata = load_inputs()
    df = df[df[COUNTRY_COL].isin(COUNTRY_CODES.keys())].copy()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    autogluon_preset, resolved_time_limit = resolve_runtime_config(runtime_mode, autogluon_time_limit)

    print(f"Loaded filtered sample with {len(df):,} rows.")
    print(f"Countries included: {', '.join(COUNTRY_CODES.values())}")
    print(f"AutoGluon runtime mode: {runtime_mode} (preset={autogluon_preset}, time_limit={resolved_time_limit}s)")

    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    ckpt_importance_path = OUTPUT_DIR / f"{output_prefix}_checkpoint_importance.csv"
    ckpt_cells_path = OUTPUT_DIR / f"{output_prefix}_checkpoint_cells.csv"
    ckpt_pool_path = OUTPUT_DIR / f"{output_prefix}_checkpoint_feature_pool.csv"

    if ckpt_importance_path.exists():
        importance_df_ckpt = pd.read_csv(
            ckpt_importance_path,
            dtype={
                "target_variable": str,
                "country": str,
                "country_name": str,
                "feature_variable": str,
                "importance_mean": float,
                "importance_std": float,
            }
        )
    else:
        importance_df_ckpt = pd.DataFrame(
            {
                "target_variable": pd.Series(dtype="str"),
                "country": pd.Series(dtype="str"),
                "country_name": pd.Series(dtype="str"),
                "feature_variable": pd.Series(dtype="str"),
                "importance_mean": pd.Series(dtype="float64"),
                "importance_std": pd.Series(dtype="float64"),
            }
        )

    if ckpt_cells_path.exists():
        cells_df_ckpt = pd.read_csv(ckpt_cells_path)
        # Fix dtype issues from CSV reading
        numeric_cols = ["n_rows", "n_classes", "min_class_count", "cv_folds"]
        for col in numeric_cols:
            if col in cells_df_ckpt.columns:
                cells_df_ckpt[col] = pd.to_numeric(cells_df_ckpt[col], errors='coerce').fillna(0).astype(int)
        float_cols = ["cv_accuracy_mean", "cv_accuracy_std", "majority_baseline"]
        for col in float_cols:
            if col in cells_df_ckpt.columns:
                cells_df_ckpt[col] = pd.to_numeric(cells_df_ckpt[col], errors='coerce')
        bool_cols = ["cv_at_or_below_baseline", "skipped", "low_n_flag", "sparse_class_flag"]
        for col in bool_cols:
            if col in cells_df_ckpt.columns:
                cells_df_ckpt[col] = cells_df_ckpt[col].astype(bool)
    else:
        cells_df_ckpt = pd.DataFrame()

    if ckpt_pool_path.exists():
        pool_df_ckpt = pd.read_csv(
            ckpt_pool_path,
            dtype={
                "target_variable": str,
                "start_pool": int,
                "missing_metadata_excluded": int,
                "admin_excluded": int,
                "semantic_excluded": int,
                "final_pool": int,
            }
        )
    else:
        pool_df_ckpt = pd.DataFrame()

    completed_cells = set()
    if not cells_df_ckpt.empty:
        completed_cells = {
            (row["target_variable"], str(row["country"]))
            for _, row in cells_df_ckpt.iterrows()
        }

    feature_pool_log = []
    importance_records = []
    cell_results: List[CellResult] = []
    processed_cells_this_run = 0

    expected_cells = len(targets) * len(COUNTRY_CODES)
    completed_so_far = len(completed_cells)
    
    print(f"\n[3/4] Training models and computing importances...")
    print(f"  Total cells to process: {expected_cells} ({len(targets)} targets × {len(COUNTRY_CODES)} countries)")
    print(f"  Already completed: {completed_so_far}")
    print(f"  Remaining: {expected_cells - completed_so_far}\n")

    for target_var in tqdm(targets, desc="Target variables", position=0, leave=True):
        if target_var not in df.columns:
            raise ValueError(f"Target {target_var} missing from dataset columns.")

        target_numeric = pd.to_numeric(df[target_var], errors="coerce")
        target_valid_mask = target_numeric >= 0
        df_target = df.loc[target_valid_mask].copy()
        train_split_df, test_split_df = country_stratified_train_test_split(df_target)

        base_feature_pool, pool_diag = build_feature_pool(df_target, metadata, target_var, similarity_model)
        if ENFORCE_IDENTICAL_FEATURE_POOL:
            feature_pool, country_diag = filter_feature_pool_across_countries(df_target, base_feature_pool)
            pool_diag.update(country_diag)
        else:
            feature_pool = base_feature_pool
        if pool_df_ckpt.empty or target_var not in set(pool_df_ckpt.get("target_variable", pd.Series(dtype=str)).tolist()):
            feature_pool_log.append({"target_variable": target_var, **pool_diag})

        print(f"\n  [{target_var}] Feature pool:")
        print(f"    {pool_diag['start_pool']} start → {pool_diag['final_pool']} final")
        print(f"    Excluded: admin={pool_diag['admin_excluded']}, "
              f"semantic={pool_diag['semantic_excluded']}, missing_metadata={pool_diag['missing_metadata_excluded']}")
        if pool_diag['vars_with_missing_text'] > 0:
            print(f"    ⚠ {pool_diag['vars_with_missing_text']} features use var names only (no question/description): {pool_diag['vars_missing_text_list']}")

        for country_code, country_name in tqdm(
            COUNTRY_CODES.items(),
            desc=f"  {target_var} countries",
            position=1,
            leave=False,
            total=len(COUNTRY_CODES),
        ):
            if (target_var, country_code) in completed_cells:
                continue

            df_country_train = train_split_df[train_split_df[COUNTRY_COL] == country_code].copy()
            df_country_test = test_split_df[test_split_df[COUNTRY_COL] == country_code].copy()
            if ENFORCE_IDENTICAL_FEATURE_POOL:
                feature_pool_country = feature_pool
            else:
                feature_pool_country, country_pool_diag = filter_feature_pool_for_country(
                    pd.concat([df_country_train, df_country_test], axis=0),
                    base_feature_pool,
                )
                feature_pool_log.append(
                    {
                        "target_variable": target_var,
                        "country": country_code,
                        **pool_diag,
                        **country_pool_diag,
                    }
                )

            X_train, y_train, X_test, y_test, class_mapping, prep_diag = prepare_model_inputs(
                df_country_train,
                df_country_test,
                target_var,
                feature_pool_country,
            )

            n_rows = prep_diag["n_rows"]
            n_classes = prep_diag["n_classes"]
            min_class_count = prep_diag["min_class_count"]
            majority_baseline = float(y_train.value_counts(normalize=True).max()) if len(y_train) else np.nan
            sparse_class_flag = min_class_count < 5 if min_class_count else False # type: ignore
            low_n_flag = n_rows < 100 # type: ignore

            if n_classes < 2 or len(y_test) == 0: # type: ignore
                cell_results.append(
                    CellResult(
                        target_variable=target_var,
                        country=country_code,
                        country_name=country_name,
                        n_rows=n_rows, # type: ignore
                        n_classes=n_classes, # type: ignore
                        min_class_count=min_class_count, # type: ignore
                        cv_folds=0,
                        cv_accuracy_mean=np.nan,
                        cv_accuracy_std=np.nan,
                        majority_baseline=majority_baseline,
                        cv_at_or_below_baseline=False,
                        class_mapping=json.dumps(class_mapping),
                        skipped=True,
                        skip_reason="Single class after target-missing drop",
                        low_n_flag=low_n_flag,
                        sparse_class_flag=sparse_class_flag,
                    )
                )
                processed_cells_this_run += 1
                if max_cells_per_run > 0 and processed_cells_this_run >= max_cells_per_run:
                    break
                continue

            try:
                fi_df, test_accuracy = evaluate_and_importance(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    OUTPUT_DIR / "autogluon_models" / output_prefix / target_var / country_code,
                    autogluon_preset,
                    resolved_time_limit,
                )
                cv_mean = test_accuracy
                cv_std = 0.0
                cv_folds = 1
                cv_at_or_below = bool(cv_mean <= majority_baseline)

                for _, fi_row in fi_df.iterrows():
                    importance_records.append(
                        {
                            "target_variable": target_var,
                            "country": country_code,
                            "country_name": country_name,
                            "feature_variable": str(fi_row["feature_variable"]),
                            "importance_mean": float(fi_row["importance"]),
                            "importance_std": float(fi_row["stddev"]),
                        }
                    )

                cell_results.append(
                    CellResult(
                        target_variable=target_var,
                        country=country_code,
                        country_name=country_name,
                        n_rows=n_rows, # type: ignore
                        n_classes=n_classes, # type: ignore
                        min_class_count=min_class_count, # type: ignore
                        cv_folds=cv_folds,
                        cv_accuracy_mean=cv_mean,
                        cv_accuracy_std=cv_std,
                        majority_baseline=majority_baseline,
                        cv_at_or_below_baseline=cv_at_or_below,
                        class_mapping=json.dumps(class_mapping),
                        skipped=False,
                        skip_reason="",
                        low_n_flag=low_n_flag,
                        sparse_class_flag=sparse_class_flag,
                    )
                )
                processed_cells_this_run += 1
                if max_cells_per_run > 0 and processed_cells_this_run >= max_cells_per_run:
                    break
            except ValueError as exc:
                cell_results.append(
                    CellResult(
                        target_variable=target_var,
                        country=country_code,
                        country_name=country_name,
                        n_rows=n_rows, # type: ignore
                        n_classes=n_classes, # type: ignore
                        min_class_count=min_class_count, # type: ignore
                        cv_folds=0,
                        cv_accuracy_mean=np.nan,
                        cv_accuracy_std=np.nan,
                        majority_baseline=majority_baseline,
                        cv_at_or_below_baseline=False,
                        class_mapping=json.dumps(class_mapping),
                        skipped=True,
                        skip_reason=str(exc),
                        low_n_flag=low_n_flag,
                        sparse_class_flag=sparse_class_flag,
                    )
                )
                processed_cells_this_run += 1
                if max_cells_per_run > 0 and processed_cells_this_run >= max_cells_per_run:
                    break

        if max_cells_per_run > 0 and processed_cells_this_run >= max_cells_per_run:
            break

    importance_df_new = pd.DataFrame(importance_records)
    cells_df_new = pd.DataFrame([vars(c) for c in cell_results])
    pool_df_new = pd.DataFrame(feature_pool_log)

    if not importance_df_new.empty:
        importance_df_ckpt = pd.concat([importance_df_ckpt, importance_df_new], ignore_index=True)
        importance_df_ckpt = importance_df_ckpt.drop_duplicates(
            subset=["target_variable", "country", "feature_variable"], keep="last"
        )
        importance_df_ckpt.to_csv(ckpt_importance_path, index=False)

    if not cells_df_new.empty:
        cells_df_ckpt = pd.concat([cells_df_ckpt, cells_df_new], ignore_index=True)
        cells_df_ckpt = cells_df_ckpt.drop_duplicates(
            subset=["target_variable", "country"], keep="last"
        )
        cells_df_ckpt.to_csv(ckpt_cells_path, index=False)

    if not pool_df_new.empty:
        pool_df_ckpt = pd.concat([pool_df_ckpt, pool_df_new], ignore_index=True)
        if ENFORCE_IDENTICAL_FEATURE_POOL:
            pool_df_ckpt = pool_df_ckpt.drop_duplicates(subset=["target_variable"], keep="last")
        else:
            pool_df_ckpt = pool_df_ckpt.drop_duplicates(
                subset=["target_variable", "country"],
                keep="last",
            )
        pool_df_ckpt.to_csv(ckpt_pool_path, index=False)

    importance_df = importance_df_ckpt.copy()
    for col in ["importance_mean", "importance_std"]:
        if col in importance_df.columns:
            importance_df[col] = pd.to_numeric(importance_df[col], errors="coerce")
    
    cells_df = cells_df_ckpt.copy()
    pool_df = pool_df_ckpt.copy()

    expected_cells = len(targets) * len(COUNTRY_CODES)
    n_done = len(cells_df) if not cells_df.empty else 0

    print(f"\n✓ Cell processing complete: {n_done}/{expected_cells} cells finished")

    if not importance_df.empty:
        print(f"\n[4/4] Generating rank correlations and heatmaps...")
        print(f"  Computing Spearman rank correlations across {len(targets)} targets...")
        rank_corr_df, mean_rho_df = build_rank_correlations(importance_df, targets)
        
        print(f"  Generating heatmaps ({len(targets)} total)...")
        for i, target in enumerate(targets, 1):
            make_heatmaps(importance_df, metadata, [target], output_prefix)
            print(f"    ✓ Heatmap {i}/{len(targets)}: {target}")
    else:
        rank_corr_df = pd.DataFrame(
            columns=[
                "target_variable",
                "country_a",
                "country_b",
                "country_a_name",
                "country_b_name",
                "spearman_rho",
                "n_overlap_features",
            ]
        )
        mean_rho_df = pd.DataFrame(columns=["target_variable", "mean_pairwise_rho"])

    if n_done >= expected_cells and not importance_df.empty:
        print(f"\n[FINAL] Saving results...")
        importance_df.to_csv(OUTPUT_DIR / f"{output_prefix}_importance_table.csv", index=False)
        print(f"  ✓ {output_prefix}_importance_table.csv")
        
        rank_corr_df.to_csv(OUTPUT_DIR / f"{output_prefix}_rank_correlations.csv", index=False)
        print(f"  ✓ {output_prefix}_rank_correlations.csv")
        
        mean_rho_df.to_csv(OUTPUT_DIR / f"{output_prefix}_mean_pairwise_rho.csv", index=False)
        print(f"  ✓ {output_prefix}_mean_pairwise_rho.csv")
        
        cells_df.to_csv(OUTPUT_DIR / f"{output_prefix}_cell_diagnostics.csv", index=False)
        print(f"  ✓ {output_prefix}_cell_diagnostics.csv")
        
        pool_df.to_csv(OUTPUT_DIR / f"{output_prefix}_feature_pool_log.csv", index=False)
        print(f"  ✓ {output_prefix}_feature_pool_log.csv")
        
        print(f"  ✓ {len(targets)} heatmap PNG files")
        
        print(f"\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print("="*80)
    else:
        print(f"\n" + "="*80)
        print(f"CHECKPOINT SAVED")
        print(f"  Completed: {n_done}/{expected_cells} cells")
        print(f"  Rerun with: python phase0a_ESS_adaptive_ML_autogluon.py --output-prefix {output_prefix}")
        print("="*80)



if __name__ == "__main__":
    args = parse_args()
    selected_targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = sorted(set(selected_targets) - set(TARGET_QUESTIONS))
    if unknown:
        raise ValueError(f"Unknown targets requested: {unknown}")
    run(
        selected_targets,
        args.output_prefix,
        max_cells_per_run=args.max_cells_per_run,
        runtime_mode=args.runtime_mode,
        autogluon_time_limit=args.autogluon_time_limit,
    )

# Script level run option for testing purposes
# run(targets=TARGET_QUESTIONS, output_prefix="phase0a", max_cells_per_run=0)
