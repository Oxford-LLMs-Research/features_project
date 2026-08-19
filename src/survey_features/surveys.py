"""
Survey loading, country maps, missing-code taxonomy, and target measurement types.

Owns every behaviour that touches survey data/metadata structure. Downstream:
feature_pool / oracle / score_cell / retrieval (via extract_survey_variables).
"""

from __future__ import annotations

import re

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


# ── Missing-code taxonomy ─────────────────────────────────────────────────────
# Non-substantive value codes fall into two kinds that must be handled differently.
#
# RESPONDENT: the respondent was asked and chose not to give a substantive answer
#   ("Don't know", "Refused", "Can't choose"). These are genuine answers about what
#   the person thinks and are KEPT as valid levels, at their original codes, so that
#   "don't know" stays distinguishable from "refused".
#
# STRUCTURAL: the datapoint is absent for reasons outside the respondent's control —
#   routing/skip patterns, instrument design, fieldwork, coding. These become NaN.
#
# Classification checks RESPONDENT first: "Refused" is a more specific signal than the
# generic word "missing", so a label carrying both resolves to respondent.

RESPONDENT_MISSING_PATTERNS: tuple[str, ...] = (
    "don't know",
    "dont know",
    "do not know",
    "doesn't know",
    "does not know",
    "no sabe",
    "refused",
    "refusal",
    "refuse",
    "decline",
    "can't choose",
    "cant choose",
    "cannot choose",
    "do not understand",
    "don't understand",
    "no opinion",
    "haven't thought",
    "have not thought",
)

STRUCTURAL_MISSING_PATTERNS: tuple[str, ...] = (
    "missing",
    "not applicable",
    "inapplicable",
    "not asked",
    "no preguntado",
    "no aplica",
    "not available",
    "no data",
    "not in survey",
    "not included",
    "item not included",
    "omitted",
    "coding error",
    "other missing",
    # Off-continuum categories found in the 2026-08 metadata review. Each was being
    # treated as a substantive SCALE POINT, so under ordinal->regression a respondent
    # who simply had no contact with an official was placed beyond the top of the scale.
    "no contact",                  # afro Q40B-Q43E: code 7 on a 0-3 scale (16 vars)
    "agree with neither",          # afro Q15-Q71: code 5 as the extreme pole (17 vars)
    "not allowed to vote",         # wvs Q221/Q222, coded above "never"
    "i am not a national",         # wvs Q254
    "never had a job",             # wvs Q281-Q283, asian SE9d/SE10d
    "hard to say",                 # wvs Q119, coded 0 below a 1-4 scale
    "doesn't matter",
    "does not matter",
    "no one in the household",
    "do not have set",             # ess wkhct
    "i don't have a line manager",
    "i don't have colleagues",
    "i don't have a regular workplace",
    "i don't work in an organisation",
)

# Structural patterns that are short enough to produce false positives as prefixes and
# so must match a whole segment. "inap" as a prefix would swallow "inappropriate", which
# is a substantive point on several Arab Barometer scales.
STRUCTURAL_EXACT_PATTERNS: tuple[str, ...] = (
    "inap",
    "na",
    "n/a",
)

# Labels that different survey houses use for both meanings. Defaulted to RESPONDENT
# (in WVS, "No answer" = −2 is the respondent declining) but flagged `ambiguous` in the
# taxonomy audit so they are easy to find and reclassify per survey.
AMBIGUOUS_MISSING_PATTERNS: tuple[str, ...] = (
    "no answer",
    "no response",
    "no contesta",
    "no responde",
    "not stated",
)

RESPONDENT = "respondent"
STRUCTURAL = "structural"
SUBSTANTIVE = "substantive"

_SEGMENT_SEPARATORS = ("/", ";", "|")


def _label_segments(label: str) -> list[str]:
    """Normalize a value label and split it into comparable segments.

    Codebooks routinely compound labels ("Missing; Not specified", "Do not know /
    No answer", "INAP; Father does not live in country"), so each side is matched
    separately.
    """
    norm = str(label).strip().lower().replace("’", "'").replace("´", "'")
    parts = [norm]
    for sep in _SEGMENT_SEPARATORS:
        parts = [chunk for part in parts for chunk in part.split(sep)]
    return [p.strip(" .,:-()[]") for p in parts if p.strip(" .,:-()[]")]


def _segment_matches(segment: str, pattern: str, exact: bool = False) -> bool:
    """Whole-segment match, or a prefix ending on a word boundary.

    Prefix-with-boundary (not substring-anywhere) is what keeps "Agnostic (Do not know
    if there is a God)" and "I don't know how to get the vaccine" — both substantive
    response options — from being read as non-response.
    """
    if segment == pattern:
        return True
    if exact:
        return False
    return segment.startswith(pattern) and not segment[len(pattern):len(pattern) + 1].isalnum()


def classify_missing_label(label: str) -> tuple[str, bool]:
    """Classify one value label -> (RESPONDENT | STRUCTURAL | SUBSTANTIVE, ambiguous).

    Order is respondent -> ambiguous -> structural, so a compound label carrying both
    ("No data / No answer") resolves to the reading that KEEPS the data.
    """
    segments = _label_segments(label)
    if not segments:
        return SUBSTANTIVE, False
    for seg in segments:
        if any(_segment_matches(seg, p) for p in RESPONDENT_MISSING_PATTERNS):
            return RESPONDENT, False
    for seg in segments:
        if any(_segment_matches(seg, p) for p in AMBIGUOUS_MISSING_PATTERNS):
            return RESPONDENT, True
    for seg in segments:
        if any(_segment_matches(seg, p) for p in STRUCTURAL_MISSING_PATTERNS):
            return STRUCTURAL, False
        if any(_segment_matches(seg, p, exact=True) for p in STRUCTURAL_EXACT_PATTERNS):
            return STRUCTURAL, False
    return SUBSTANTIVE, False


def _code_variants(code_str: str) -> set:
    """All comparable spellings of a value code (str / int / float)."""
    out: set = {str(code_str), str(code_str).strip()}
    for cast in (int, float):
        try:
            out.add(cast(str(code_str)))
        except (TypeError, ValueError):
            pass
    return out


def classify_missing_codes(metadata: dict) -> dict[str, dict[str, set]]:
    """{var_code: {RESPONDENT: {codes}, STRUCTURAL: {codes}}} from metadata value labels."""
    out: dict[str, dict[str, set]] = {}
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
            buckets: dict[str, set] = {RESPONDENT: set(), STRUCTURAL: set()}
            for code_str, label in values.items():
                kind, _ambiguous = classify_missing_label(label)
                if kind == SUBSTANTIVE:
                    continue
                buckets[kind] |= _code_variants(code_str)
            if buckets[RESPONDENT] or buckets[STRUCTURAL]:
                out[var_code] = buckets
    return out


# ── Target measurement level ──────────────────────────────────────────────────
# The oracle is the study's gold standard, so how it MODELS a target matters as much as
# which features it ranks. Forcing everything to problem_type="multiclass" treats an
# 11-point left-right scale as 11 unordered categories: ~100 rows per class, and
# predicting 6 when the truth is 5 is penalised exactly as hard as predicting 1.
# Measured on P16ST x Colombia — the signal only survives once ordering is restored:
#   11 unordered classes  log-loss lift -0.134   (looks unpredictable)
#   collapsed to 2 bins   log-loss lift +0.030, accuracy 0.728 vs majority 0.658
#
# Detection runs off value labels, AFTER missing codes are removed (a variable whose only
# labelled values are "Don't know"/"Missing" has a raw numeric range underneath: age).

# Ordinal vocabulary, split by reliability: STRONG fires alone, WEAK needs two distinct
# families (alone it's usually inside a proper noun — "Human rights", "Barbados").
# Validated against all 2,638 variables; evidence: pipeline_audit_2026-08.md §A11.

ORDINAL_STRONG_PATTERNS: tuple[str, ...] = (
    # intensity
    "strongly", "somewhat", "completely", "moderately", "extremely", "slightly",
    "fairly", "a great deal", "a lot", "a little", "not at all", "very much",
    # agreement / evaluation
    "agree", "disagree", "satisfied", "dissatisfied", "acceptable", "unacceptable",
    "unimportant", "essential", "likely", "unlikely", "justifiable",
    "approve", "disapprove", "definitely", "probably", "certainly",
    # frequency
    "never", "rarely", "sometimes", "often", "always", "frequently", "occasionally",
    "usually", "daily", "weekly", "monthly", "once a year", "times a year",
    "once or twice", "a few times", "several times", "many times", "hardly ever",
    "most of the time", "all the time",
    # extent / degree — the Arab Barometer house style (34 variables)
    "to a great extent", "to a medium extent", "to a small extent",
    "to a limited extent", "large degree", "moderate degree", "small degree",
    "no obstacle", "guaranteed",
    # quantifier ladders — afro Q38A-K, wvs Q113-Q117, asian Q160
    "none of them", "some of them", "most of them", "all of them", "few of them",
    "all of the time", "some of the time", "none of the time", "of the time",
    "allow many", "allow none", "allow some",
    # membership ladder — wvs Q94-Q105
    "inactive member", "active member", "not a member",
    # quality ladders
    "excellent", "very bad", "fairly bad", "bad thing", "good thing", "badly",
    "better", "worse", "worsen", "improve", "stronger", "weaker",
    "major problems", "minor problems", "full democracy", "not a democracy",
    "free and fair", "works fine", "needs minor", "needs major",
    "progressing", "standstill", "in decline",
    # ladders with no intensity words at all
    "illiterate", "no formal education", "university education",
    "covers expenses", "does not cover", "not enough",
    "far left", "far right", "left-wing", "right-wing", "leaning",
)

# Fire only in company: each of these has documented false positives when alone. Grouped
# into FAMILIES because near-synonyms co-occur naturally inside nominal option text —
# "A high level of economic growth" + "people have more say" (wvs Q152) trips both `high`
# and `more`, and "less than 30 hours" + "30 hours a week or more" (wvs Q279) trips both
# `less` and `more`. Corroboration therefore requires two distinct FAMILIES, not two words.
ORDINAL_WEAK_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("more", "less"),
    ("high", "higher", "low", "lower"),
    ("good", "bad", "poor"),
    ("increase", "decrease", "increased", "decreased"),
    ("easy", "difficult"),
    ("severe", "serious"),
    ("positive", "negative"),
    ("important",), ("trust",), ("common",), ("proud",), ("close",),
    ("secure",), ("well",),
)
# Phrases where an otherwise-strong scale word is part of a category NAME, not a scale
# anchor. Checked before the strong-pattern test and the match suppressed.
ORDINAL_FALSE_FRIENDS: tuple[str, ...] = (
    "never married", "never in a legally registered", "never had a job",
    "never employed", "no, never", "never been",
    "left school", "left the country", "left full-time education",
)

# Scales have few, PACKED codes; code lists (party/ISCO/region) are large and sparse.
# Loose cardinality cap (ESS education ladders reach 28) + density test.
MAX_ORDINAL_CATEGORIES = 40
MAX_ORDINAL_CODE_DENSITY = 30  # max(code) must be <= this * n_substantive_values

# Sentinel codes far outside the scale core (97="None" on a 0-10 scale, 5555="Other")
# must not enter a rank target. Prevalence + examples: pipeline_audit §A11 / onboarding §5.
OUT_OF_SCALE_MULTIPLE = 3

BINARY = "binary"
NOMINAL = "nominal"
ORDINAL = "ordinal"
CONTINUOUS = "continuous"


_missing_codes_cache: dict[int, dict] = {}


def _cached_missing_codes(metadata: dict) -> dict:
    """classify_missing_codes memoized per metadata object (it walks every variable)."""
    key = id(metadata)
    if key not in _missing_codes_cache:
        _missing_codes_cache[key] = classify_missing_codes(metadata)
    return _missing_codes_cache[key]


def _substantive_values(var_code: str, metadata: dict) -> dict:
    """Value labels for a variable with respondent+structural missing codes removed."""
    buckets = _cached_missing_codes(metadata).get(var_code, {})
    drop = set(buckets.get(RESPONDENT, set())) | set(buckets.get(STRUCTURAL, set()))
    for section in (metadata or {}).values():
        if not isinstance(section, dict) or var_code not in section:
            continue
        info = section[var_code]
        if not isinstance(info, dict):
            return {}
        values = info.get("values") or {}
        if not isinstance(values, dict):  # some codebooks store a free-text note here
            return {}
        return {k: v for k, v in values.items() if not _code_variants(k) & drop}
    return {}



def _segment_matches_anywhere(label: str, pattern: str) -> bool:
    """Word-boundary match of `pattern` anywhere in `label`.

    Substring matching was the single biggest source of false ordinals: "Human rights"
    matched `right`, "Barbados" matched `bad`, "Lower Sorbian" matched `lower`,
    "Landless People's Movement" matched `less`, "Never married" matched `never`.
    """
    return re.search(rf"(?<![a-z]){re.escape(pattern)}(?![a-z])", label) is not None


# Explicit overrides for constructs no lexical rule separates. Keep this SHORT and
# justified — it is a curated exception list, not a substitute for the rules. Keyed by
# (survey, variable); use (None, variable) to apply across surveys.
TARGET_TYPE_OVERRIDES: dict[tuple[str | None, str], tuple[str, str]] = {
    # Inglehart post-materialism batteries: pick ONE goal from four. The options contain
    # "A high level of economic growth" and "people have more say", so two weak families
    # corroborate each other, but the codes are a choice set with no order.
    ("wvs", "Q152"): (NOMINAL, "forced-choice goal battery"),
    ("wvs", "Q153"): (NOMINAL, "forced-choice goal battery"),
    ("wvs", "Q154"): (NOMINAL, "forced-choice goal battery"),
    ("wvs", "Q155"): (NOMINAL, "forced-choice goal battery"),
    ("wvs", "Q156"): (NOMINAL, "forced-choice goal battery"),
    ("wvs", "Q157"): (NOMINAL, "forced-choice goal battery"),
    # Quantifier ladder with bare anchors: All / Most / Some / None. Adding those four
    # words to the lexicon would fire on any list containing "None of these".
    ("asianbarometer", "Q160a"): (ORDINAL, "All/Most/Some/None quantifier ladder"),
    ("asianbarometer", "Q160b"): (ORDINAL, "All/Most/Some/None quantifier ladder"),
    ("asianbarometer", "Q160c"): (ORDINAL, "All/Most/Some/None quantifier ladder"),
    ("asianbarometer", "Q160d"): (ORDINAL, "All/Most/Some/None quantifier ladder"),
    # Monotone restrictiveness / financial-coping ladders with no scale vocabulary at all.
    ("wvs", "Q130"): (ORDINAL, "immigration-restrictiveness ladder"),
    ("wvs", "Q286"): (ORDINAL, "household financial-coping ladder"),
    ("arabbarometer", "Q605"): (ORDINAL, "sharia-to-popular-will ladder"),
    (None, "domicil"): (ORDINAL, "urbanicity ladder (big city -> countryside)"),
}


def _is_range_descriptor(code: str) -> bool:
    """True for pseudo-codes like "0-168" that describe a numeric RANGE, not a value.

    ESS stores these in the `values` dict, which made `wkhct` (contracted hours per week)
    look like a 2-category binary: an 8-hour and a 40-hour week became the same class.
    """
    s = str(code).strip()
    return bool(re.fullmatch(r"-?\d+\s*[-–]\s*-?\d+", s))


def _numeric_codes(vals: dict) -> list[float] | None:
    """Numeric value codes, or None if any code is non-numeric.

    A genuine ordinal scale never has letter codes. ESS `region` (NUTS: AL011, AT130),
    `lnghom1` (ISO-639: AAR, ABK) and `cntbrthd` (ISO country: AD, AE) were all being
    typed ordinal and regressed.
    """
    out = []
    for k in vals:
        try:
            out.append(float(str(k).strip()))
        except (TypeError, ValueError):
            return None
    return out


def _contiguous_duplicate_runs(vals: dict, labels: list[str]) -> bool:
    """True when repeated labels form contiguous runs over a small anchor set.

    Anchored scales repeat their anchors across adjacent codes ("Leaning left" at 2 and 3).
    Big code lists ALSO contain duplicates, but scattered — `region` repeats "Wien" 133
    times out of 2303, `isco08` 9 times out of 587, because the same place or job name
    recurs across countries. Requiring few distinct anchors AND contiguity separates them.
    """
    if len(set(labels)) > 12 or len(labels) - len(set(labels)) < 1:
        return False
    codes = _numeric_codes(vals)
    if codes is None:
        return False
    order = sorted(range(len(codes)), key=lambda i: codes[i])
    ordered_labels = [labels[i] for i in order]
    seen_runs: dict[str, int] = {}
    prev = None
    for lb in ordered_labels:
        if lb != prev:
            seen_runs[lb] = seen_runs.get(lb, 0) + 1
        prev = lb
    # every label occupies exactly one contiguous run
    return all(n == 1 for n in seen_runs.values())


def out_of_scale_codes(vals: dict,
                       multiple: int = OUT_OF_SCALE_MULTIPLE) -> list[float]:
    """Substantive codes lying far outside the scale's own core range.

    These are "Other"/"not applicable" sentinels wearing a substantive label. Under
    ordinal -> regression they enter the target: 5555="Other" on a 1-14 ladder,
    7="No contact" on a 0-3 scale, 97="None" on a 0-10 left-right scale.
    """
    codes = _numeric_codes(vals)
    if not codes or len(codes) < 3:
        return []
    codes = sorted(codes)
    # scale core = the longest run with small consecutive gaps
    core, best = [codes[0]], [codes[0]]
    for a, b in zip(codes, codes[1:]):
        if b - a <= 1.5:
            core.append(b)
        else:
            if len(core) > len(best):
                best = core
            core = [b]
    if len(core) > len(best):
        best = core
    span = max(abs(best[-1]), abs(best[0]), 1.0)
    return [c for c in codes if abs(c) > multiple * span]


def detect_target_type(
    var_code: str,
    metadata: dict,
    series: pd.Series | None = None,
    min_continuous_unique: int = 15,
    survey: str | None = None,
) -> tuple[str, str]:
    """Classify a target's measurement level -> (type, reason).

    Returns BINARY / NOMINAL / ORDINAL / CONTINUOUS with a short justification, so the
    classification can be audited rather than trusted (scripts/audit_target_types.py).

    Rule order is deliberate: the cheap structural disqualifiers for ordinal run BEFORE
    any vocabulary matching, because a single stray keyword inside a proper noun
    ("Human rights", "Barbados", "Lower Sorbian", "Landless People's Movement") was
    otherwise enough to send a 500-category code list into a regression.
    """
    for key in ((survey, var_code), (None, var_code)):
        if key in TARGET_TYPE_OVERRIDES:
            ttype, why = TARGET_TYPE_OVERRIDES[key]
            return ttype, f"override: {why}"

    vals = {k: v for k, v in _substantive_values(var_code, metadata).items()
            if not _is_range_descriptor(k)}
    labels = [str(v).strip().lower() for v in vals.values()]
    n_unique_data = int(series.dropna().nunique()) if series is not None else None
    numeric_series = series is not None and pd.api.types.is_numeric_dtype(
        pd.to_numeric(series, errors="coerce")
    )

    # 1. Nothing labelled, or only range descriptors / a couple of sentinels: trust the data.
    if n_unique_data is not None and n_unique_data >= min_continuous_unique and len(vals) <= 2:
        return CONTINUOUS, f"{n_unique_data} distinct values, {len(vals)} labelled codes"
    if not labels:
        if n_unique_data is None:
            return NOMINAL, "no value labels and no data supplied (UNVERIFIED)"
        if numeric_series:
            return ORDINAL, f"unlabelled numeric, {n_unique_data} distinct values"
        return NOMINAL, "no substantive value labels and few distinct values"

    if len(vals) == 1:
        return NOMINAL, "single substantive category (degenerate)"
    if len(vals) == 2:
        return BINARY, "2 substantive categories"

    # 2. Structural disqualifiers for ordinal.
    codes = _numeric_codes(vals)
    if codes is None:
        return NOMINAL, f"{len(vals)} categories with non-numeric codes"
    if len(vals) > MAX_ORDINAL_CATEGORIES:
        return NOMINAL, f"{len(vals)} categories — too many for a rating scale"
    density = max(abs(c) for c in codes) / max(len(vals), 1)
    if density > MAX_ORDINAL_CODE_DENSITY:
        return NOMINAL, f"sparse codes (max {max(codes):.0f} over {len(vals)} values)"

    # 3. Anchored scale: few distinct anchors repeated over contiguous code runs.
    if len(set(labels)) < len(labels) and _contiguous_duplicate_runs(vals, labels):
        return ORDINAL, "repeated labels across adjacent codes (anchored scale)"

    # 4. Vocabulary. Strong patterns may fire alone; weak ones need corroboration,
    #    because alone they are usually inside a proper noun or a parenthetical.
    def _hits(lb: str, pattern: str) -> bool:
        if any(ff in lb for ff in ORDINAL_FALSE_FRIENDS):
            return False
        return _segment_matches_anywhere(lb, pattern)

    strong = [p for p in ORDINAL_STRONG_PATTERNS if any(_hits(lb, p) for lb in labels)]
    if strong:
        return ORDINAL, f"ordered-scale wording ({', '.join(strong[:3])})"
    fams = [fam for fam in ORDINAL_WEAK_FAMILIES
            if any(_hits(lb, p) for lb in labels for p in fam)]
    if len(fams) >= 2:
        names = [fam[0] for fam in fams[:3]]
        return ORDINAL, f"corroborated weak wording ({', '.join(names)})"

    # 5. Monotone numeric-ish labels: "1 year", "2 years", ... or bare numbers.
    numeric_like = sum(1 for lb in labels if any(ch.isdigit() for ch in lb))
    if numeric_like >= max(3, 0.6 * len(labels)):
        return ORDINAL, "numeric-valued category labels"

    return NOMINAL, f"{len(vals)} unordered categories"


def substantive_numeric_mask(var_code: str, metadata: dict, series: pd.Series) -> pd.Series:
    """Rows whose value is a substantive scale point — for RANK models only.

    Respondent non-response is kept everywhere else in the pipeline, deliberately: "don't
    know" is a real answer and a real predictor. But it has no POSITION on an ordered
    scale, and survey houses code it far outside the range — Latinobarometro puts "don't
    know" at 97 on a 0-10 left-right scale. Regressing on that fits a point 87 units past
    the scale top (observed MAE 9.67 on an 11-point scale before this guard).

    So: keep respondent non-response when the variable is a FEATURE or a nominal target;
    drop it when the variable is the TARGET of an ordinal/continuous rank model. Whether
    someone declines to place themselves is worth studying as its own binary outcome, not
    smuggled in as an extreme scale value.
    """
    num = pd.to_numeric(series, errors="coerce")
    keep = num.notna()
    buckets = _cached_missing_codes(metadata).get(var_code, {})
    drop = set(buckets.get(RESPONDENT, set())) | set(buckets.get(STRUCTURAL, set()))
    numeric_drop = {v for v in drop if isinstance(v, (int, float))}
    if numeric_drop:
        keep &= ~num.isin(numeric_drop)
    return keep


def to_ordinal_codes(var_code: str, metadata: dict, series: pd.Series) -> pd.Series:
    """Numeric scale positions for a target, whatever form the loader stored it in.

    Two shapes occur in these surveys and both broke the rank model:
      * Asian Barometer stores the LABEL text ("Very religious"), not the code, so
        pd.to_numeric wiped the column. Labels are mapped back through the metadata.
      * Afrobarometer Q43A is labelled {0: No, 1: Yes} but the data also carries 8 and 9,
        which no label explains. Unlabelled codes are dropped rather than guessed at.

    Values that are not substantive scale points become NaN.
    """
    vals = _substantive_values(var_code, metadata)
    if not vals:
        return pd.to_numeric(series, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.5:
        allowed = {c for k in vals for c in _code_variants(k)
                   if isinstance(c, (int, float))}
        return numeric.where(numeric.isin(allowed)) if allowed else numeric

    # Text-valued column: invert the metadata to map label -> code.
    label_to_code: dict[str, float] = {}
    for code, label in vals.items():
        try:
            label_to_code[str(label).strip().lower()] = float(str(code).strip())
        except (TypeError, ValueError):
            continue
    return series.astype(str).str.strip().str.lower().map(label_to_code)


def clean_question_columns(
    df: pd.DataFrame,
    country_col: str,
    admin_cols: frozenset[str],
    metadata: dict | None = None,
    keep_respondent_missing: bool = True,
) -> pd.DataFrame:
    """
    Coerce numeric-coded columns to float and drop STRUCTURAL missing codes to NaN.

    Respondent non-response ("Don't know", "Refused", "Can't choose") is a genuine
    answer about what the person thinks and is KEPT, at its original code, so the
    levels stay distinguishable. Only codes attributable to the instrument, routing
    or fieldwork ("Not asked", "Not applicable", "Missing") become NaN.

    Without `metadata` there are no value labels to classify by, so the historical
    conservative rule applies and all negative codes are dropped. This is why the
    oracle must be handed metadata — see `load_survey_clean`.

    Set `keep_respondent_missing=False` for the legacy behaviour (drop both kinds).
    Text-label columns with no numeric majority are left intact for AutoGluon.
    """
    cleaned = df.copy()
    by_var = classify_missing_codes(metadata) if metadata else {}
    q_cols = [c for c in cleaned.columns if c not in admin_cols and c != country_col]

    def _drop_codes(col: str) -> set:
        buckets = by_var.get(col)
        if not buckets:
            return set()
        drop = set(buckets[STRUCTURAL])
        if not keep_respondent_missing:
            drop |= buckets[RESPONDENT]
        return drop

    for col in q_cols:
        buckets = by_var.get(col)
        drop = _drop_codes(col)
        is_object = pd.api.types.is_object_dtype(cleaned[col])

        if not is_object:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        else:
            coerced = pd.to_numeric(cleaned[col], errors="coerce")
            if coerced.notna().mean() > 0.5:
                cleaned[col] = coerced
                is_object = False

        if not is_object:
            if buckets is None:
                # Unlabelled column: cannot tell a respondent code from a sentinel,
                # so fall back to the conservative negatives-are-missing rule.
                cleaned[col] = cleaned[col].where(cleaned[col] >= 0)
            else:
                keep_neg = buckets[RESPONDENT] if keep_respondent_missing else set()
                negative_ok = cleaned[col].isin(keep_neg)
                cleaned[col] = cleaned[col].where((cleaned[col] >= 0) | negative_ok)
            if drop:
                cleaned[col] = cleaned[col].where(~cleaned[col].isin(drop))
        elif drop:
            drop_str = {str(v).strip() for v in drop}
            cleaned[col] = cleaned[col].where(
                ~cleaned[col].astype(str).str.strip().isin(drop_str)
            )
    return cleaned


# ── Cleaned-survey loader (the single preprocessing definition) ───────────────
# The oracle used to call clean_question_columns WITHOUT metadata (so no label-based
# classification ran at all) while score_cell/evaluation applied no cleaning at all —
# meaning the oracle ranked features on one version of the data and the downstream
# evaluator scored them on another. Both now go through this function.

_clean_survey_cache: dict[tuple[str, str, bool], tuple[pd.DataFrame, dict]] = {}


def load_survey_clean(
    survey_id: str,
    config_path: str,
    keep_respondent_missing: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """`load_survey` + metadata-aware `clean_question_columns`, memoized per process."""
    key = (survey_id, str(config_path), bool(keep_respondent_missing))
    if key in _clean_survey_cache:
        return _clean_survey_cache[key]
    data, metadata = load_survey(survey_id, config_path)
    country_col = SURVEY_COUNTRY_COL[survey_id]
    admin_cols = build_admin_cols(metadata, country_col)
    cleaned = clean_question_columns(
        data, country_col, admin_cols, metadata,
        keep_respondent_missing=keep_respondent_missing,
    )
    _clean_survey_cache[key] = (cleaned, metadata)
    return cleaned, metadata
