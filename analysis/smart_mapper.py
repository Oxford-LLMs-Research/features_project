"""
Second pilot — per-feature mapper (format pilot, rebuilt 2026-06-03).

Design (chosen after diagnostics; see memory current_state_2026-05):
  Diagnostics showed (a) holistic top-20 retrieval already contained the right
  variables, so the pilot-1 "trust -> none" failures were MAPPER weakness, not
  retrieval; and (b) per-feature retrieval is far sharper (targeted query sim ~0.7 vs
  ~0.38 in a whole-response pool). So we extract features first, then retrieve and
  disambiguate per feature.

Pipeline for one cell's selection response (JSON list or free-text prose):
  1. EXTRACT discrete feature requests, each WITH its justifying context, via one
     mapper-model call. We also capture `sub_items` (sub-measures a request bundles)
     purely for auditing bundling prevalence; mapping itself is ONE-TO-ONE.
  2. For each feature: retrieve top-N (default 20) survey vars by embedding similarity
     to "label: context" (dual-embed max, like pilot-1), excluding target/leakage codes.
  3. For each feature: one disambiguation call -> exactly one survey code or "none".
  Dedupe codes across features (keep first), preserving arrival order = the feature set.

Why extract instead of trusting a JSON list directly: it makes JSON and free-text arms
go through the IDENTICAL downstream path (extract -> retrieve -> disambiguate), so the
only difference between arms is the selection format. For the JSON arm the extractor is
near-trivial (the items are already discrete); for free-text it does the real work. This
keeps FORMAT the cleanly isolated variable.

Mapper model is held fixed per run (Nemotron small OR Qwen large) and is never the
selector model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np


# ── Step 1: extraction ────────────────────────────────────────────────────────

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


# ── Step 2/3: per-feature disambiguation ──────────────────────────────────────

DISAMBIG_PROMPT = """You are mapping one requested piece of information to the single best-matching survey question, if any.

The researcher wants to know a respondent's:
"{feature_label}"
Context: "{feature_context}"

Candidate survey questions:
{candidates_block}

Pick the ONE candidate that best captures what the researcher is asking for, or answer "none" if none is a genuine match (do not force a weak match).
Respond with ONLY the letter (A, B, C, ...) or "none". No explanation."""

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class FeatureMap:
    feature_label: str
    feature_context: str
    sub_items: list[str]
    ftype: str                      # model-assigned class (FEATURE_TYPES)
    piped: bool                     # did this feature enter retrieve+disambiguate?
    selected_code: str | None
    selected_text: str | None
    candidates: list[dict]          # the top-N pool shown (for audit); [] if not piped
    disambig_raw: str


@dataclass
class CellMap:
    cell: str
    arm: str
    mapper_model: str
    features: list[FeatureMap] = field(default_factory=list)
    mapped_codes: list[str] = field(default_factory=list)   # deduped, arrival order
    extract_raw: str = ""

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_piped(self) -> int:
        return sum(1 for f in self.features if f.piped)

    @property
    def n_mapped(self) -> int:
        return len(self.mapped_codes)

    @property
    def n_none(self) -> int:
        """none-rate among PIPED features only (non-piped never had a chance to map)."""
        return sum(1 for f in self.features if f.piped and f.selected_code is None)

    @property
    def n_bundled(self) -> int:
        return sum(1 for f in self.features if len(f.sub_items) > 1)

    def type_counts(self) -> dict:
        out: dict = {}
        for f in self.features:
            out[f.ftype] = out.get(f.ftype, 0) + 1
        return out


def _parse_json_list(raw: str) -> list:
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


FEATURE_TYPES = {"respondent_attribute", "temporal_contextual",
                 "instrument_methodology", "base_rate_prior"}


def extract_features(response_text: str, generate_fn) -> tuple[list[dict], str]:
    """Step 1. Returns (features, raw). Each feature: {feature, context, sub_items, type}.
    `type` is a model-assigned class (FEATURE_TYPES); used downstream to pipeline only
    respondent_attribute (and optionally temporal_contextual) into mapping, and to study
    the rest (methodology, base-rate-seeking) as behavioral metadata."""
    if not (response_text or "").strip():
        return [], ""
    raw = generate_fn(
        [{"role": "user", "content": EXTRACT_PROMPT.format(response_text=response_text.strip())}],
        max_tokens=4096, temperature=0.0, usage_phase="extract",
    ) or ""
    out = []
    for it in _parse_json_list(raw):
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
        if ftype not in FEATURE_TYPES:
            ftype = "respondent_attribute"  # default if model omits/garbles the label
        out.append({"feature": str(feat).strip(),
                    "context": str(it.get("context", "")).strip(),
                    "sub_items": si, "type": ftype})
    return out, raw


def _parse_letter(raw: str, n: int) -> int | None:
    if not raw:
        return None
    cleaned = str(raw).strip().upper()
    if not cleaned or "NONE" in cleaned:
        return None
    # Prefer a standalone letter token; fall back to first valid letter char.
    for tok in re.findall(r"[A-Z]+", cleaned):
        if len(tok) == 1 and _LETTERS.index(tok) < n:
            return _LETTERS.index(tok)
    for ch in cleaned:
        if ch in _LETTERS and _LETTERS.index(ch) < n:
            return _LETTERS.index(ch)
    return None


def _retrieve(label: str, context: str, embed_fn, survey_embeddings, var_codes,
              survey_variables, excluded: set[str], top_n: int) -> list[dict]:
    """Top-N candidates by max(sim(label), sim(label+context)) — dual embed like pilot-1."""
    qlabel = embed_fn([label])[0]
    combined = f"{label}: {context}" if context else label
    qcomb = embed_fn([combined])[0]
    sims = np.maximum(qlabel @ survey_embeddings.T, qcomb @ survey_embeddings.T)
    order = np.argsort(sims)[::-1]
    pool = []
    for idx in order:
        vc = var_codes[idx]
        if vc in excluded:
            continue
        pool.append({"var_code": vc, "question_text": survey_variables[vc],
                     "similarity": float(sims[idx])})
        if len(pool) >= top_n:
            break
    return pool


def map_features(
    cell: str,
    arm: str,
    features: list[dict],
    survey_embeddings: np.ndarray,
    var_codes: list[str],
    survey_variables: dict[str, str],
    embed_fn,
    disambig_fn,
    mapper_model: str = "",
    excluded_codes: set[str] | None = None,
    top_n: int = 20,
    min_similarity: float = 0.30,
    extract_raw: str = "",
    pipe_types: set[str] | None = None,
) -> CellMap:
    """Retrieve + per-feature disambiguate a PRE-EXTRACTED feature list.

    Separated from extraction so the extractor can be held FIXED (a capable model that
    can digest long free-text) while only the disambiguator (`disambig_fn`) varies as the
    mapper-strength arm. `features` items are {feature, context, sub_items, type}.

    Only features whose `type` is in `pipe_types` enter retrieval+disambiguation (default:
    respondent_attribute only). Other features (methodology, base-rate, temporal) are still
    recorded with piped=False for the behavioral-metadata analysis, but never mapped — so
    they don't inflate k or the none-rate of the capability metric.
    """
    excluded = set(excluded_codes or set())
    pipe = pipe_types or {"respondent_attribute"}
    cm = CellMap(cell=cell, arm=arm, mapper_model=mapper_model, extract_raw=extract_raw)
    seen: set[str] = set()

    for f in features:
        label = f.get("feature", "")
        context = f.get("context", "")
        sub = f.get("sub_items", []) or []
        ftype = f.get("type", "respondent_attribute")
        if not label:
            continue
        if ftype not in pipe:
            cm.features.append(FeatureMap(
                feature_label=label, feature_context=context, sub_items=sub, ftype=ftype,
                piped=False, selected_code=None, selected_text=None, candidates=[], disambig_raw=""))
            continue
        pool = _retrieve(label, context, embed_fn, survey_embeddings, var_codes,
                         survey_variables, excluded, top_n)
        pool = [c for c in pool if c["similarity"] >= min_similarity]
        sel_code = sel_text = None
        raw = ""
        if pool:
            block = "\n".join(f"{_LETTERS[i]}. [{c['var_code']}] {c['question_text']}"
                              for i, c in enumerate(pool))
            raw = disambig_fn(
                [{"role": "user", "content": DISAMBIG_PROMPT.format(
                    feature_label=label, feature_context=context or label, candidates_block=block)}],
                max_tokens=2048, temperature=0.0, usage_phase="disambig",
            ) or ""
            idx = _parse_letter(raw, len(pool))
            if idx is not None:
                sel_code = pool[idx]["var_code"]
                sel_text = pool[idx]["question_text"]
        cm.features.append(FeatureMap(
            feature_label=label, feature_context=context, sub_items=sub, ftype=ftype,
            piped=True, selected_code=sel_code, selected_text=sel_text,
            candidates=pool, disambig_raw=raw,
        ))
        if sel_code and sel_code not in seen:
            seen.add(sel_code)
            cm.mapped_codes.append(sel_code)

    return cm
