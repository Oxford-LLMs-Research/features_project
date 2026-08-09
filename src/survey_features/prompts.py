"""
All prompt templates for the confirmatory free-text pipeline.

  SYSTEM_PROMPT + FREETEXT_*   selector essay (gen)
  EXTRACT_PROMPT               essay -> typed feature list (fixed extractor)
  DISAMBIG_PROMPT              per-unit top-N candidates -> one code or "none"
"""

from __future__ import annotations

SYSTEM_PROMPT = "You are a social science researcher."

# Named arms for prompt-sensitivity (and confirmatory default = social_scientist).
# Value None means omit the system message entirely.
PROMPT_ARMS: dict[str, str | None] = {
    "social_scientist": SYSTEM_PROMPT,
    "none": None,
    "helpful": "You are a helpful assistant.",
}
DEFAULT_PROMPT_ARM = "social_scientist"

FREETEXT_UNPROMPTED = (
    'A survey asks respondents: "{question}"\n\n'
    "You want to predict how a respondent will answer. What information about the respondent would you need?"
)

FREETEXT_COUNTRY = (
    'A survey asks respondents in {country}: "{question}"\n\n'
    "You want to predict how a respondent in {country} will answer. What information about the respondent would you need?"
)

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
