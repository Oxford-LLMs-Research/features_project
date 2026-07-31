"""
ALL prompt templates for the pipeline, in one place.

CURRENT (paper main text; free-text elicitation):
  SYSTEM_PROMPT + FREETEXT_UNPROMPTED / FREETEXT_COUNTRY   selection (selector model)
  EXTRACT_PROMPT                                           essay -> typed feature list (fixed extractor)
  DISAMBIG_PROMPT                                          per-feature top-N -> one code or "none"

LEGACY (pilot-1 JSON grid; kept for appendix reproducibility via scripts/run_grid.py):
  SYSTEM_PROMPT + PROMPT_UNPROMPTED / PROMPT_COUNTRY       strict-JSON selection
  DISAMBIG_PROMPT_LEGACY                                   shortlist (top-5) disambiguation

The free-text prompts are the EXACT JSON prompts with only the JSON formatting block
removed — nothing added, nothing rephrased. This isolates the output-format instruction
as the single variable between the JSON and free-text arms (validated in the format
pilot; see docs/format_findings.md).
"""

from __future__ import annotations

# Shared system prompt (identical across JSON and free-text arms).
SYSTEM_PROMPT = "You are a social science researcher."


# ── CURRENT: free-text selection prompts ─────────────────────────────────────

FREETEXT_UNPROMPTED = (
    'A survey asks respondents: "{question}"\n\n'
    "You want to predict how a respondent will answer. What information about the respondent would you need?"
)

FREETEXT_COUNTRY = (
    'A survey asks respondents in {country}: "{question}"\n\n'
    "You want to predict how a respondent in {country} will answer. What information about the respondent would you need?"
)


# ── CURRENT: extraction (essay -> typed feature list; FIXED extractor model) ──

EXTRACT_PROMPT = """A researcher was asked what they would want to know about a respondent in order to predict how that respondent answers a survey question. They replied:
\"\"\"
{response_text}
\"\"\"

List each distinct piece of information the researcher asked for. For each, give:
- "feature": a short label for the information.
- "context": a brief phrase, copied or paraphrased from the reply, capturing how they described it and why (this preserves their reasoning).
- "sub_items": if (and only if) the request explicitly bundles several specific sub-measures, list them; otherwise [].
- "type": classify the request into exactly one of:
    - "respondent_attribute": a characteristic OF the individual respondent (demographics, attitudes, behaviours, experiences, beliefs, traits) — the things you could ask a person about themselves.
    - "temporal_contextual": timing or external-context factors (when the survey was taken, recent events the respondent was exposed to, the news cycle, period effects).
    - "instrument_methodology": commentary about the survey instrument itself rather than the respondent (question wording/phrasing, response-option design, language of the survey, mode of administration, reference period, non-response patterns, survey topic).
    - "base_rate_prior": appeals to population-level statistics, base rates, modal/average responses, or known distributions rather than information about this individual.

Map only what the researcher actually requested; do not add information they did not mention. Classify honestly — do not force everything into "respondent_attribute". Output ONLY a JSON list of such objects."""


# ── CURRENT: per-feature disambiguation (top-N candidates -> one code / none) ─

DISAMBIG_PROMPT = """You are mapping one requested piece of information to the single best-matching survey question, if any.

The researcher wants to know a respondent's:
"{feature_label}"
Context: "{feature_context}"

Candidate survey questions:
{candidates_block}

Pick the ONE candidate that best captures what the researcher is asking for, or answer "none" if none is a genuine match (do not force a weak match).
Respond with ONLY the candidate letter (A, B, C, …; use AA, AB, … if the list goes past Z) or "none". No explanation."""


# ── LEGACY: strict-JSON selection prompts (pilot-1 grid) ─────────────────────

PROMPT_UNPROMPTED = """A survey asks respondents: "{question}"

You want to predict how a respondent will answer. What information about the respondent would you need?

Output a JSON list where each item describes one piece of information you would want to know. Each item should have:
- "feature": a short label for the information (e.g., "a specific attitude or behaviour")
- "reasoning": one sentence on why this would help predict the answer

Output ONLY the JSON list, no other text."""

PROMPT_COUNTRY = """A survey asks respondents in {country}: "{question}"

You want to predict how a respondent in {country} will answer. What information about the respondent would you need?

Output a JSON list where each item describes one piece of information you would want to know. Each item should have:
- "feature": a short label for the information (e.g., "a specific attitude or behaviour")
- "reasoning": one sentence on why this would help predict the answer

Output ONLY the JSON list, no other text."""


# ── LEGACY: shortlist disambiguation (pilot-1 top-5 mapper) ──────────────────

DISAMBIG_PROMPT_LEGACY = """You are helping map abstract feature descriptions to concrete survey questions.

A researcher said they would want to know a respondent's:
"{feature_label}"
Reasoning: "{feature_reasoning}"

Below are candidate survey questions that might capture this information. Pick the ONE question that best matches what the researcher is asking for, or respond "none" if no question is a good match.

Candidates:
{candidates_block}

Respond with ONLY the letter (A, B, C, ...) of the best match, or "none". No explanation."""
