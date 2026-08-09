"""
Extraction: turn a selector's raw selection response (free-text essay or JSON list)
into a typed, discrete feature list via ONE call to a FIXED capable extractor model.

Why extract instead of trusting a JSON list directly: it makes JSON and free-text arms
go through the IDENTICAL downstream path (extract -> retrieve -> disambiguate), so the
only difference between arms is the selection format. For the JSON arm the extractor is
near-trivial (the items are already discrete); for free-text it does the real work.

The extractor is held FIXED across all arms and disambiguators: a small model cannot
digest a long essay (demo: Nemotron pulled 3 features vs Qwen's 28 from the same
response), and the extracted set defines the model's request, so it must not vary by
disambiguator.
"""

from __future__ import annotations

import json
import re

from .prompts import EXTRACT_PROMPT

FEATURE_TYPES = {"respondent_attribute", "temporal_contextual",
                 "instrument_methodology", "population_statistic"}

# Legacy slug from pre-rename extracts / older prompts → current FEATURE_TYPES.
_LEGACY_TYPE_ALIASES = {"base_rate_prior": "population_statistic"}


def parse_json_list(raw: str) -> list:
    """Parse a (possibly fenced) JSON list from an LLM response; [] on failure."""
    if not raw:
        return []
    s = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    s = re.sub(r"\s*```$", "", s)
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if not m:
            return []
        try:
            v = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    return v if isinstance(v, list) else []


def extract_features(
    response_text: str,
    generate_fn,
    *,
    max_tokens: int = 4096,
) -> tuple[list[dict], str]:
    """Extract typed features from one selection response.

    Returns (features, raw). Each feature: {feature, context, sub_items, type}.
    `type` is a model-assigned class (FEATURE_TYPES); used downstream to pipeline only
    respondent_attribute (and optionally temporal_contextual) into mapping, and to study
    the rest (methodology, population statistics) as behavioral metadata."""
    if not (response_text or "").strip():
        return [], ""
    raw = generate_fn(
        [{"role": "user", "content": EXTRACT_PROMPT.format(response_text=response_text.strip())}],
        max_tokens=max(512, int(max_tokens)), temperature=0.0, usage_phase="extract",
    ) or ""
    out = []
    for it in parse_json_list(raw):
        if not isinstance(it, dict):
            if isinstance(it, str) and it.strip():
                out.append({"feature": it.strip(), "context": "", "sub_items": [],
                            "type": "respondent_attribute"})
            continue
        feat = it.get("feature") or it.get("label") or it.get("name")
        if not feat:
            continue
        si = it.get("sub_items") or []
        si = [str(x).strip() for x in si if str(x).strip()] if isinstance(si, list) else []
        ftype = str(it.get("type", "")).strip().lower()
        ftype = _LEGACY_TYPE_ALIASES.get(ftype, ftype)
        if ftype not in FEATURE_TYPES:
            ftype = "respondent_attribute"  # default if model omits/garbles the label
        out.append({"feature": str(feat).strip(),
                    "context": str(it.get("context", "")).strip(),
                    "sub_items": si, "type": ftype})
    return out, raw
