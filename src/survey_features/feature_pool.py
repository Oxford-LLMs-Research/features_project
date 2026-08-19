"""
Candidate feature-pool for one oracle cell: admin exclusions, near-duplicate filter,
missingness/variation filters, and the skip-pattern leakage screen.

Used by oracle fit and (indirectly) by score_cell's random-draw universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
