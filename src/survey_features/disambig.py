"""
Disambiguation: map each requested feature to exactly one survey code or "none".

CURRENT path (per-feature mapper, pilot-2 design): ``map_features`` takes a
PRE-EXTRACTED typed feature list, retrieves a per-feature top-N pool
(survey_features.retrieval.retrieve_candidates), and asks the disambiguator model one
question per feature. Diagnostics showed per-feature retrieval is far sharper than
whole-response retrieval (targeted query sim ~0.7 vs ~0.38), and pilot-1 "trust -> none"
failures were mapper weakness, not retrieval. Mapping is ONE-TO-ONE; ``sub_items`` are
recorded purely for auditing bundling prevalence.

LEGACY path (shortlist disambiguation, pilot-1): ``disambiguate_mappings`` picks from the
batch top-5 shortlist produced by retrieval.map_features_to_variables.

The disambiguator model is held fixed per run and is never the selector model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .prompts import DISAMBIG_PROMPT, DISAMBIG_PROMPT_LEGACY
from .retrieval import retrieve_candidates

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ── CURRENT: per-feature mapper ───────────────────────────────────────────────

@dataclass
class FeatureMap:
    feature_label: str
    feature_context: str
    sub_items: list[str]
    ftype: str                      # model-assigned class (extraction.FEATURE_TYPES)
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


def parse_letter(raw: str, n: int) -> int | None:
    """Parse a per-feature disambiguation reply into a 0-based index, or None for 'none'.
    Prefers a standalone letter token; falls back to first valid letter char."""
    if not raw:
        return None
    cleaned = str(raw).strip().upper()
    if not cleaned or "NONE" in cleaned:
        return None
    for tok in re.findall(r"[A-Z]+", cleaned):
        if len(tok) == 1 and _LETTERS.index(tok) < n:
            return _LETTERS.index(tok)
    for ch in cleaned:
        if ch in _LETTERS and _LETTERS.index(ch) < n:
            return _LETTERS.index(ch)
    return None


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
        pool = retrieve_candidates(label, context, embed_fn, survey_embeddings, var_codes,
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
            idx = parse_letter(raw, len(pool))
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


# ── LEGACY: shortlist disambiguation (pilot-1) ────────────────────────────────

def format_candidates(candidates: list[dict]) -> str:
    """Format candidates as a lettered list."""
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f"{_LETTERS[i]}. [{c['var_code']}] {c['question_text']}")
    return "\n".join(lines)


def parse_disambig_response(raw: str | None, n_candidates: int) -> int | None:
    """
    Parse the legacy shortlist LLM response into a candidate index or None.
    Returns index (0-based) or None for 'none'.

    NOTE: intentionally kept as the pilot-1 parser (first valid letter anywhere in the
    reply) so the legacy arm's behaviour is reproducible; the current per-feature path
    uses the stricter ``parse_letter``.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raw = str(raw)
    cleaned = raw.strip().upper()
    if not cleaned:
        return None

    if "NONE" in cleaned:
        return None

    for i in range(n_candidates):
        if _LETTERS[i] in cleaned:
            return i

    return None


def disambiguate_single(
    feature_label: str,
    feature_reasoning: str,
    candidates: list[dict],
    generate_fn,
    model: str,
    # Reasoning disambig models (e.g. NVIDIA Nemotron-3-Nano) emit chain-of-thought
    # tokens before the final letter answer. With 256, ~27% of calls hit
    # finish_reason='length' with an empty body, causing silent "no mapping" rows.
    # 2048 leaves ample room for CoT while the actual output is only ~1-5 tokens.
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> dict:
    """
    Ask LLM to pick the best candidate or 'none' (legacy shortlist prompt).

    Returns:
        {
            "selected_code": var_code or None,
            "selected_text": question_text or None,
            "selected_rank": index in candidates or None,
            "raw_response": str,
        }
    """
    prompt = DISAMBIG_PROMPT_LEGACY.format(
        feature_label=feature_label,
        feature_reasoning=feature_reasoning,
        candidates_block=format_candidates(candidates),
    )

    messages = [{"role": "user", "content": prompt}]
    raw = generate_fn(
        messages, max_tokens=max_tokens, temperature=temperature, usage_phase="disambig"
    )
    if raw is None:
        raw_s = ""
    elif isinstance(raw, str):
        raw_s = raw.strip()
    else:
        raw_s = str(raw).strip()

    idx = parse_disambig_response(raw_s, len(candidates))

    if idx is not None and idx < len(candidates):
        return {
            "selected_code": candidates[idx]["var_code"],
            "selected_text": candidates[idx]["question_text"],
            "selected_rank": idx,
            "raw_response": raw_s,
        }
    return {
        "selected_code": None,
        "selected_text": None,
        "selected_rank": None,
        "raw_response": raw_s,
    }


def disambiguate_mappings(
    mappings: list[dict],
    generate_fn,
    model: str = "deepseek-ai/DeepSeek-V3.2",
) -> list[dict]:
    """
    Run legacy disambiguation on all mappings. Adds 'disambig' key to each mapping dict.

    Args:
        mappings: output of retrieval.map_features_to_variables
        generate_fn: callable(messages, max_tokens, temperature) -> str
        model: model identifier

    Returns:
        mappings with 'disambig' field added to each entry.
    """
    for i, m in enumerate(mappings):
        if not m["candidates"]:
            m["disambig"] = {
                "selected_code": None,
                "selected_text": None,
                "selected_rank": None,
                "raw_response": "no candidates",
            }
            print(f"  [{i+1}/{len(mappings)}] {m['target']} | {m['feature_label'][:30]:30s} -> SKIP (no candidates)")
            continue

        result = disambiguate_single(
            feature_label=m["feature_label"],
            feature_reasoning=m["feature_reasoning"],
            candidates=m["candidates"],
            generate_fn=generate_fn,
            model=model,
        )
        m["disambig"] = result

        code = result["selected_code"] or "none"
        print(f"  [{i+1}/{len(mappings)}] {m['target']} | {m['feature_label'][:30]:30s} -> {code:8s} (raw: {result['raw_response']})")

    return mappings


def print_disambig_summary(mappings: list[dict], ground_truth: dict[str, list[str]] = None):
    """Print summary of legacy disambiguation results with optional ground truth comparison."""
    current_key = None
    for m in mappings:
        if "disambig" not in m:
            continue

        key = (m["target"], m["country"], m["condition"])
        if key != current_key:
            current_key = key
            country_str = m["country"] or "unprompted"
            gt_vars = ground_truth.get(m["target"], []) if ground_truth else []
            print(f"\n{'='*70}")
            print(f"{m['target']} | {country_str} | GT: {gt_vars}")
            print(f"{'='*70}")

        code = m["disambig"]["selected_code"]
        rank = m["disambig"]["selected_rank"]

        hit = ""
        if ground_truth and code and code in ground_truth.get(m["target"], []):
            hit = " *** HIT ***"

        if code:
            print(f"  [{m['feature_rank']}] {m['feature_label'][:40]:40s} -> {code} (rank {rank}){hit}")
        else:
            print(f"  [{m['feature_rank']}] {m['feature_label'][:40]:40s} -> none")
