"""
Survey loading, country maps, and metadata handling — the SINGLE copy.

Previously near-duplicated between run_grid.py and phase0b_oracle_autogluon.py;
everything that touches survey data/metadata structure now lives here.
"""

from __future__ import annotations

import pandas as pd

# ── Survey registry ───────────────────────────────────────────────────────────
# Column in each survey's DataFrame that holds country codes.
SURVEY_COUNTRY_COL: dict[str, str] = {
    "wvs":            "B_COUNTRY",
    "afrobarometer":  "COUNTRY",
    "arabbarometer":  "COUNTRY",
    "asianbarometer": "country",
    "latinobarometer": "IDENPA",
    "ess_wave_10":    "cntry",
    "ess_wave_11":    "cntry",
}

DEFAULT_SURVEY = "wvs"


def load_survey(survey_id: str, config_path: str) -> tuple[pd.DataFrame, dict]:
    """Load survey data + metadata via synthetic_sampling."""
    try:
        from synthetic_sampling.config.base import DataPaths
        from synthetic_sampling.loaders.survey_loader import SurveyLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: synthetic_sampling. Install project dependencies "
            "with `pip install -e .` (or `pip install -r requirements.txt`) to continue."
        ) from exc

    paths = DataPaths.from_yaml(config_path)
    loader = SurveyLoader(paths=paths, verbose=False)
    return loader.load_survey(survey_id)


def build_country_code_map(
    metadata: dict,
    country_col: str,
    data: pd.DataFrame | None = None,
) -> dict[str, int | str]:
    """
    Derive {country_name: code} from the country column's 'values' dict in metadata.
    Codes that parse as integers are returned as int; alpha codes (ESS) stay as str.

    If data is provided, the metadata-derived codes are cross-checked against the
    actual values in the country column. When the data stores country names directly
    (instead of numeric codes), the map is built from the actual data values instead.
    """
    meta_map: dict[str, int | str] = {}
    for section in metadata.values():
        if not isinstance(section, dict):
            continue
        if country_col in section:
            values = section[country_col].get("values", {})
            for code_str, name in values.items():
                try:
                    meta_map[name] = int(code_str)
                except ValueError:
                    meta_map[name] = code_str
            break

    if data is None or not meta_map:
        return meta_map

    actual_values = set(data[country_col].dropna().unique())

    # Normal path: metadata codes exist in data.
    if any(code in actual_values for code in meta_map.values()):
        return {name: code for name, code in meta_map.items() if code in actual_values}

    # Data stores names/strings directly — build map from actual data values.
    # Try to match metadata names to data values case-insensitively, then add
    # any remaining data values as identity entries so --list-countries is complete.
    actual_lower = {str(v).lower(): v for v in actual_values}
    result: dict[str, int | str] = {}
    for meta_name in meta_map:
        if meta_name in actual_values:
            result[meta_name] = meta_name
        elif meta_name.lower() in actual_lower:
            result[meta_name] = actual_lower[meta_name.lower()]
    # Add any data values that didn't match a metadata name.
    matched_data_vals = set(result.values())
    for val in actual_values:
        val_str = str(val)
        if val not in matched_data_vals and val_str not in result:
            result[val_str] = val
    return result


def build_admin_cols(metadata: dict, country_col: str) -> frozenset[str]:
    """
    Derive admin columns from the 'EXCLUDED' section in metadata plus the country column.
    The EXCLUDED section contains all non-substantive variables (IDs, weights, admin codes).
    """
    excluded = metadata.get("EXCLUDED", {})
    return frozenset(excluded.keys()) | {country_col}


def get_question_text(var_code: str, metadata: dict) -> str:
    """Question wording (or description) for one variable code."""
    for section in metadata.values():
        if var_code in section:
            info = section[var_code]
            return (info.get("question") or info.get("description") or var_code).strip()
    raise KeyError(f"{var_code} not found in metadata")


def extract_survey_variables(metadata: dict, exclude_sections: list[str] = None) -> dict[str, str]:
    """
    Extract {var_code: question_text} from ProfileBuilder metadata.

    Args:
        metadata: ProfileBuilder.metadata dict
        exclude_sections: sections to skip (default: ["EXCLUDED"])

    Returns:
        {var_code: question_text}
    """
    exclude = set(exclude_sections or ["EXCLUDED"])
    variables = {}
    for section, vars_dict in metadata.items():
        if section in exclude:
            continue
        for var_code, info in vars_dict.items():
            text = (info.get("question") or info.get("description") or "").strip()
            if text:
                variables[var_code] = text
    return variables


def flatten_metadata(raw_metadata: dict) -> dict[str, dict[str, str]]:
    """Flatten nested metadata dict into variable -> metadata fields."""
    flat: dict[str, dict[str, str]] = {}
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


# Value labels whose presence marks a code as missing/non-substantive.
MISSING_LABEL_PATTERNS: tuple[str, ...] = (
    "missing",
    "no answer",
    "not applicable",
    "not asked",
    "refused",
    "refusal",
    "don't know",
    "do not know",
    "do not understand",
    "can't choose",
    "cannot choose",
    "decline",
    "inap",
    "no contesta",
    "no sabe",
    "no response",
)

# Backwards-compatible alias (old name in phase0b_oracle_autogluon.py).
_MISSING_LABEL_PATTERNS = MISSING_LABEL_PATTERNS


def missing_codes_from_metadata(metadata: dict) -> dict[str, set]:
    """{var_code: set of value codes labelled as missing} from metadata value labels."""
    out: dict[str, set] = {}
    if not metadata:
        return out
    for section in metadata.values():
        if not isinstance(section, dict):
            continue
        for var_code, info in section.items():
            if not isinstance(info, dict):
                continue
            values = info.get("values") or {}
            if not isinstance(values, dict):
                continue
            missing: set = set()
            for code_str, label in values.items():
                label_norm = str(label).strip().lower()
                if any(p in label_norm for p in MISSING_LABEL_PATTERNS):
                    missing.add(str(code_str))
                    missing.add(str(code_str).strip())
                    try:
                        missing.add(int(str(code_str)))
                    except (TypeError, ValueError):
                        pass
                    try:
                        missing.add(float(str(code_str)))
                    except (TypeError, ValueError):
                        pass
            if missing:
                out[var_code] = missing
    return out


def clean_question_columns(
    df: pd.DataFrame,
    country_col: str,
    admin_cols: frozenset[str],
    metadata: dict | None = None,
) -> pd.DataFrame:
    """
    Coerce numeric-coded columns to float and map negative values to NaN.
    When metadata is provided, also strips value codes whose label indicates a
    missing/non-substantive response (e.g. "Don't know", "Refused").
    Text-label columns with no numeric majority are left intact for AutoGluon.
    """
    cleaned = df.copy()
    missing_by_var = missing_codes_from_metadata(metadata) if metadata else {}
    q_cols = [c for c in cleaned.columns if c not in admin_cols and c != country_col]
    for col in q_cols:
        if not pd.api.types.is_object_dtype(cleaned[col]):
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            cleaned[col] = cleaned[col].where(cleaned[col] >= 0)
            mc = missing_by_var.get(col)
            if mc:
                cleaned[col] = cleaned[col].where(~cleaned[col].isin(mc))
        else:
            coerced = pd.to_numeric(cleaned[col], errors="coerce")
            if coerced.notna().mean() > 0.5:
                cleaned[col] = coerced.where(coerced >= 0)
                mc = missing_by_var.get(col)
                if mc:
                    cleaned[col] = cleaned[col].where(~cleaned[col].isin(mc))
            else:
                mc = missing_by_var.get(col)
                if mc:
                    mc_str = {str(v).strip() for v in mc}
                    cleaned[col] = cleaned[col].where(
                        ~cleaned[col].astype(str).str.strip().isin(mc_str)
                    )
    return cleaned
