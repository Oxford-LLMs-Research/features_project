"""
All prompt templates for the confirmatory free-text pipeline.

  SYSTEM_PROMPT + FREETEXT_*   selector essay (gen)
  EXTRACT_PROMPT               essay -> typed feature list (fixed extractor)
  DISAMBIG_PROMPT              per-unit top-N candidates -> one code or "none"
"""

from __future__ import annotations

SYSTEM_PROMPT = "You are a social science researcher."

# Named system-message arms. Value None means omit the system message entirely.
# Confirmatory default = social_scientist. v1 prompt-sensitivity also used none/helpful.
# v2 adds analyst (see PROMPT_PACKS).
PROMPT_ARMS: dict[str, str | None] = {
    "social_scientist": SYSTEM_PROMPT,
    "none": None,
    "helpful": "You are a helpful assistant.",
    "analyst": "You are an analyst.",
}
DEFAULT_PROMPT_ARM = "social_scientist"
DEFAULT_REFERENT = "respondent"

# User-prompt referent: respondent (confirmatory + v1) vs person (v2 alternative).
FREETEXT_BY_REFERENT: dict[str, dict[str, str]] = {
    "respondent": {
        "unprompted": (
            'A survey asks respondents: "{question}"\n\n'
            "You want to predict how a respondent will answer. "
            "What information about the respondent would you need?"
        ),
        "country": (
            'A survey asks respondents in {country}: "{question}"\n\n'
            "You want to predict how a respondent in {country} will answer. "
            "What information about the respondent would you need?"
        ),
    },
    "person": {
        "unprompted": (
            'A survey asks people: "{question}"\n\n'
            "You want to predict how a person will answer. "
            "What information about the person would you need?"
        ),
        "country": (
            'A survey asks people in {country}: "{question}"\n\n'
            "You want to predict how a person in {country} will answer. "
            "What information about the person would you need?"
        ),
    },
}

# Aliases used by the confirmatory pipeline and v1 experiment.
FREETEXT_UNPROMPTED = FREETEXT_BY_REFERENT["respondent"]["unprompted"]
FREETEXT_COUNTRY = FREETEXT_BY_REFERENT["respondent"]["country"]

# Stage-1 packs for prompt-sensitivity v2: (system arm, user referent).
# Replicates of scientist_respondent are two generations of the same pack, not a
# fourth wording.
PROMPT_PACKS: dict[str, dict[str, str]] = {
    "scientist_respondent": {"system": "social_scientist", "referent": "respondent"},
    "analyst_person": {"system": "analyst", "referent": "person"},
    "none_respondent": {"system": "none", "referent": "respondent"},
}
DEFAULT_PROMPT_PACK = "scientist_respondent"

# Stage-1 factorial for prompt-sensitivity v2 (country_provided only).
# Replicate ints are two gens of scientist_respondent, not extra wordings.
PROMPT_SENSITIVITY_V2_SELECTORS = ("deepseek_v4", "kimi", "minimax", "hermes")
PROMPT_SENSITIVITY_V2_CONDITION = "country_provided"
PROMPT_SENSITIVITY_V2_STAGE1_TEMPERATURE = 0.0
# Sidecar sampling draws (t1/t2) of scientist_respondent only. Not in Stage-1
# lock jobs. 1.0 is the model's native softmax (untempered sampling). See
# glossary "temperature run".
PROMPT_SENSITIVITY_V2_TEMPERATURE_RUNS_TEMPERATURE = 1.0
PROMPT_SENSITIVITY_V2_RUNS: tuple[tuple[str, int | None], ...] = (
    ("scientist_respondent", 1),
    ("scientist_respondent", 2),
    ("analyst_person", None),
    ("none_respondent", None),
)


def prompt_sensitivity_v2_runs(
    pack: str | None = None,
    replicate: int | None = None,
) -> list[tuple[str, int | None]]:
    """Stage-1 (pack, replicate) jobs. replicate is only for scientist_respondent."""
    if pack is None:
        if replicate is not None:
            raise ValueError("replicate requires a pack (scientist_respondent)")
        return list(PROMPT_SENSITIVITY_V2_RUNS)
    if pack not in PROMPT_PACKS:
        raise ValueError(f"Unknown prompt pack {pack!r}; choose from {sorted(PROMPT_PACKS)}")
    if pack != "scientist_respondent":
        if replicate is not None:
            raise ValueError(f"replicate is only valid for scientist_respondent, not {pack}")
        return [(pack, None)]
    if replicate is None:
        return [r for r in PROMPT_SENSITIVITY_V2_RUNS if r[0] == pack]
    if replicate not in (1, 2):
        raise ValueError("replicate must be 1 or 2")
    return [(pack, replicate)]


def prompt_sensitivity_v2_temperature_draws(draw: int | None = None) -> list[int]:
    """scientist_respondent t1/t2 draws. Not part of Stage-1 lock jobs."""
    if draw is None:
        return [1, 2]
    if draw not in (1, 2):
        raise ValueError("temperature_draw must be 1 or 2")
    return [draw]


EXTRACT_PROMPT = """A researcher was asked what they would want to know about a respondent in order to predict how that respondent answers a survey question. They replied:
\"\"\"
{response_text}
\"\"\"

List each distinct piece of information the researcher asked for. For each, give:
- "feature": a short label for the information.
- "context": a brief phrase, copied or paraphrased from the reply, capturing how they described it and why (this preserves their reasoning).
- "sub_items": if (and only if) the request explicitly bundles several specific sub-measures, list them; otherwise [].
- "type": classify into exactly one of (use the decision order below):
    - "respondent_attribute": a characteristic OF this individual (demographics, attitudes, behaviours, beliefs, traits, personal experiences — including recent personal events — and their own location). Things you could ask this person about themselves. Includes geo identifiers and locality covariates tied to where THEY live (region, county, zip/census tract, density or housing costs in their area, distance to services) — even if those are looked up from external tables.
    - "temporal_contextual": timing or period context of the interview itself (survey date/wave/year, news cycle, elections, scandals, crises, other ambient events at fieldwork time). NOT the respondent's personal history; NOT static country traits; NOT where the respondent lives.
    - "instrument_methodology": about the survey instrument or method (wording, options, language, mode, reference period, non-response/skip patterns, topic) — not about the respondent or population.
    - "population_statistic": a population- or country-level rate, distribution, average, prevalence, or similar aggregate used as a fallback when individual information is unavailable (e.g. national sex ratio, modal answer in the population, GDP per capita, national internet penetration). Use this only when there is no person-level referent — not for "their region/country/area" or attributes of the place they live.

Decision rule when unsure: prefer respondent_attribute if the ask is about this person or their location/locality; prefer temporal_contextual only for interview-period ambient context; use population_statistic only for aggregate fallbacks with no individual referent; use instrument_methodology only for questionnaire/method commentary.

Map only what the researcher actually requested; do not add information they did not mention. Classify honestly — do not force everything into "respondent_attribute". Output ONLY a JSON list of such objects."""

DISAMBIG_PROMPT = """You are mapping one requested piece of information to the single best-matching survey question, if any.

The researcher wants to know a respondent's:
"{feature_label}"
Context: "{feature_context}"

Candidate survey questions:
{candidates_block}

Pick the ONE candidate that best captures what the researcher is asking for, or answer "none" if none is a genuine match (do not force a weak match).
Respond with ONLY the candidate letter (A, B, C, …; use AA, AB, … if the list goes past Z) or "none". No explanation."""
