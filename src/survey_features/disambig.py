"""
Disambiguation: map each requested feature to exactly one survey code or "none".

CURRENT path (per-feature mapper, pilot-2 design): ``map_features`` takes a
PRE-EXTRACTED typed feature list, retrieves a per-feature top-N pool
(survey_features.retrieval.retrieve_candidates), and asks the disambiguator model one
question per feature. Diagnostics showed per-feature retrieval is far sharper than
whole-response retrieval (targeted query sim ~0.7 vs ~0.38), and pilot-1 "trust -> none"
failures were mapper weakness, not retrieval.

Main / confirmatory mapping uses dual-layer ``subitem_map.map_features_with_subitems``
(parent + bundled sub_items). ``map_features`` remains the parent-only ablation path.

LEGACY path (shortlist disambiguation, pilot-1): ``disambiguate_mappings`` picks from the
batch top-5 shortlist produced by retrieval.map_features_to_variables.

The disambiguator model is held fixed per run and is never the selector model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .prompts import DISAMBIG_PROMPT, DISAMBIG_PROMPT_LEGACY
from .retrieval import retrieve_candidates_batch, retrieve_ensemble_candidates_batch

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Per-unit mapping outcome (decomposes the old binary "none").
MAP_STATUS_MAPPED = "mapped"
MAP_STATUS_NOT_PIPED = "not_piped"
MAP_STATUS_EMPTY_POOL = "empty_pool"      # retrieve+threshold left nothing; LLM not called
MAP_STATUS_MODEL_NONE = "model_none"      # LLM abstained (explicit none)
MAP_STATUS_MODEL_EMPTY = "model_empty"    # LLM returned empty / blank
MAP_STATUS_UNPARSEABLE = "unparseable"    # non-empty reply, no valid letter, not none


def candidate_label(i: int) -> str:
    """0-based index → A..Z, then AA..AZ, BA.. (supports ensemble max_fused > 26)."""
    if i < 0:
        raise IndexError(i)
    if i < 26:
        return _LETTERS[i]
    i -= 26
    return _LETTERS[i // 26] + _LETTERS[i % 26]


def candidate_labels(n: int) -> list[str]:
    return [candidate_label(i) for i in range(n)]


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
    map_status: str = MAP_STATUS_NOT_PIPED  # see MAP_STATUS_* constants


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

    def status_counts(self) -> dict:
        """Piped/non-piped map_status histogram (diagnostics)."""
        out: dict = {}
        for f in self.features:
            out[f.map_status] = out.get(f.map_status, 0) + 1
        return out


def parse_letter(raw: str, n: int) -> int | None:
    """Parse a per-feature disambiguation reply into a 0-based index, or None for 'none'.

    Accepts A..Z and AA.. labels (for pools larger than 26). Prefers an exact label
    token: longest first so AA wins over A; among equal length, the *last* match
    so chatty replies like "Not A; I'd choose C" resolve to C. For n<=26 falls
    back to the last valid letter character in the reply.
    """
    if not raw:
        return None
    cleaned = str(raw).strip().upper()
    if not cleaned or "NONE" in cleaned:
        return None
    label_to_idx = {lab: i for i, lab in enumerate(candidate_labels(n))}
    tokens = re.findall(r"[A-Z]+", cleaned)
    matches = [
        (len(tok), i, label_to_idx[tok])
        for i, tok in enumerate(tokens)
        if tok in label_to_idx
    ]
    if matches:
        # longest token, then last occurrence
        matches.sort(key=lambda t: (t[0], t[1]))
        return matches[-1][2]
    if n <= 26:
        last = None
        for ch in cleaned:
            if ch in label_to_idx:
                last = label_to_idx[ch]
        return last
    return None


def classify_none_raw(raw: str) -> str:
    """When parse_letter returned None and the pool was non-empty, why?"""
    cleaned = (raw or "").strip()
    if not cleaned:
        return MAP_STATUS_MODEL_EMPTY
    if "NONE" in cleaned.upper():
        return MAP_STATUS_MODEL_NONE
    return MAP_STATUS_UNPARSEABLE


def _disambiguate_pool(
    label: str,
    context: str,
    pool: list[dict],
    disambig_fn,
) -> tuple[str | None, str | None, str, str]:
    """Ask disambiguator for one letter/none; returns (code, text, raw, map_status)."""
    if not pool:
        return None, None, "", MAP_STATUS_EMPTY_POOL
    block = "\n".join(
        f"{candidate_label(i)}. [{c['var_code']}] {c['question_text']}"
        for i, c in enumerate(pool)
    )
    # Reasoning models (e.g. Nemotron) need CoT headroom; do not shrink below 2048.
    raw = disambig_fn(
        [{"role": "user", "content": DISAMBIG_PROMPT.format(
            feature_label=label, feature_context=context or label, candidates_block=block)}],
        max_tokens=2048, temperature=0.0, usage_phase="disambig",
    ) or ""
    idx = parse_letter(raw, len(pool))
    if idx is None:
        return None, None, raw, classify_none_raw(raw)
    return pool[idx]["var_code"], pool[idx]["question_text"], raw, MAP_STATUS_MAPPED


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
    workers: int = 1,
) -> CellMap:
    """Retrieve + per-feature disambiguate a PRE-EXTRACTED feature list.

    Separated from extraction so the extractor can be held FIXED (a capable model that
    can digest long free-text) while only the disambiguator (`disambig_fn`) varies as the
    mapper-strength arm. `features` items are {feature, context, sub_items, type}.

    Only features whose `type` is in `pipe_types` enter retrieval+disambiguation (default:
    respondent_attribute only). Other features (methodology, population statistic, temporal) are still
    recorded with piped=False for the behavioral-metadata analysis, but never mapped — so
    they don't inflate k or the none-rate of the capability metric.

    ``workers`` > 1 parallelizes *disambiguation LLM calls* via ThreadPool after
    batched retrieval (one dual-embed encode for all piped features). Feature order
    and mapped_codes arrival order are unchanged vs workers=1.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    excluded = set(excluded_codes or set())
    pipe = pipe_types or {"respondent_attribute"}
    cm = CellMap(cell=cell, arm=arm, mapper_model=mapper_model, extract_raw=extract_raw)
    n_workers = max(1, int(workers))

    # Slot list preserves input order; piped meta collected then batched retrieve.
    slots: list[FeatureMap | None] = []
    # (slot_idx, label, context, sub, ftype)
    piped_meta: list[tuple[int, str, str, list, str]] = []

    for f in features:
        label = f.get("feature", "")
        context = f.get("context", "")
        sub = f.get("sub_items", []) or []
        ftype = f.get("type", "respondent_attribute")
        if not label:
            continue
        if ftype not in pipe:
            slots.append(FeatureMap(
                feature_label=label, feature_context=context, sub_items=sub, ftype=ftype,
                piped=False, selected_code=None, selected_text=None, candidates=[],
                disambig_raw="", map_status=MAP_STATUS_NOT_PIPED))
            continue
        slot_idx = len(slots)
        slots.append(None)  # filled after disambig
        piped_meta.append((slot_idx, label, context, sub, ftype))

    pools = retrieve_candidates_batch(
        [(label, context) for _, label, context, _, _ in piped_meta],
        embed_fn, survey_embeddings, var_codes, survey_variables, excluded, top_n,
    )
    # (slot_idx, label, context, sub, ftype, pool)
    piped_jobs: list[tuple[int, str, str, list, str, list[dict]]] = []
    for (slot_idx, label, context, sub, ftype), pool in zip(piped_meta, pools):
        pool = [c for c in pool if c["similarity"] >= min_similarity]
        piped_jobs.append((slot_idx, label, context, sub, ftype, pool))

    def _fill(slot_idx: int, label: str, context: str, sub: list, ftype: str, pool: list[dict]) -> FeatureMap:
        sel_code, sel_text, raw, status = _disambiguate_pool(
            label, context or label, pool, disambig_fn,
        )
        return FeatureMap(
            feature_label=label, feature_context=context, sub_items=sub, ftype=ftype,
            piped=True, selected_code=sel_code, selected_text=sel_text,
            candidates=pool, disambig_raw=raw, map_status=status,
        )

    if n_workers <= 1 or len(piped_jobs) <= 1:
        for slot_idx, label, context, sub, ftype, pool in piped_jobs:
            slots[slot_idx] = _fill(slot_idx, label, context, sub, ftype, pool)
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(piped_jobs))) as ex:
            futs = {
                ex.submit(_fill, slot_idx, label, context, sub, ftype, pool): slot_idx
                for slot_idx, label, context, sub, ftype, pool in piped_jobs
            }
            for fut in as_completed(futs):
                slots[futs[fut]] = fut.result()

    seen: set[str] = set()
    for fm in slots:
        assert fm is not None
        cm.features.append(fm)
        if fm.selected_code and fm.selected_code not in seen:
            seen.add(fm.selected_code)
            cm.mapped_codes.append(fm.selected_code)

    return cm


def map_features_ensemble(
    cell: str,
    arm: str,
    features: list[dict],
    model_packs: list[dict],
    survey_variables: dict[str, str],
    disambig_fn,
    mapper_model: str = "",
    excluded_codes: set[str] | None = None,
    top_n: int = 20,
    min_similarity: float = 0.30,
    max_fused: int | None = None,
    extract_raw: str = "",
    pipe_types: set[str] | None = None,
    workers: int = 1,
) -> tuple[CellMap, dict]:
    """Like ``map_features`` but retrieves with multiple embedders, fuses pools, then
    **one** disambiguation call per piped feature (not per embedder).

    ``model_packs``: list of ``{name, embed_fn, survey_embeddings, var_codes}``.
    Fusion: per-model top_n + min_similarity, then union by var_code with
    similarity = max; pool capped at ``max_fused`` (default ``2 * top_n``).

    Returns ``(CellMap, timing_dict)`` where timing includes per-model retrieve
    wall times and disambiguation wall time.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    excluded = set(excluded_codes or set())
    pipe = pipe_types or {"respondent_attribute"}
    cm = CellMap(cell=cell, arm=arm, mapper_model=mapper_model, extract_raw=extract_raw)
    n_workers = max(1, int(workers))

    slots: list[FeatureMap | None] = []
    piped_meta: list[tuple[int, str, str, list, str]] = []

    for f in features:
        label = f.get("feature", "")
        context = f.get("context", "")
        sub = f.get("sub_items", []) or []
        ftype = f.get("type", "respondent_attribute")
        if not label:
            continue
        if ftype not in pipe:
            slots.append(FeatureMap(
                feature_label=label, feature_context=context, sub_items=sub, ftype=ftype,
                piped=False, selected_code=None, selected_text=None, candidates=[],
                disambig_raw="", map_status=MAP_STATUS_NOT_PIPED))
            continue
        slot_idx = len(slots)
        slots.append(None)
        piped_meta.append((slot_idx, label, context, sub, ftype))

    pools, retrieve_times = retrieve_ensemble_candidates_batch(
        [(label, context) for _, label, context, _, _ in piped_meta],
        model_packs,
        survey_variables,
        excluded,
        top_n=top_n,
        min_similarity=min_similarity,
        max_fused=max_fused,
    )
    piped_jobs: list[tuple[int, str, str, list, str, list[dict]]] = [
        (slot_idx, label, context, sub, ftype, pool)
        for (slot_idx, label, context, sub, ftype), pool in zip(piped_meta, pools)
    ]

    def _fill(slot_idx: int, label: str, context: str, sub: list, ftype: str, pool: list[dict]) -> FeatureMap:
        sel_code, sel_text, raw, status = _disambiguate_pool(
            label, context or label, pool, disambig_fn,
        )
        return FeatureMap(
            feature_label=label, feature_context=context, sub_items=sub, ftype=ftype,
            piped=True, selected_code=sel_code, selected_text=sel_text,
            candidates=pool, disambig_raw=raw, map_status=status,
        )

    t_dis_0 = time.perf_counter()
    if n_workers <= 1 or len(piped_jobs) <= 1:
        for slot_idx, label, context, sub, ftype, pool in piped_jobs:
            slots[slot_idx] = _fill(slot_idx, label, context, sub, ftype, pool)
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(piped_jobs))) as ex:
            futs = {
                ex.submit(_fill, slot_idx, label, context, sub, ftype, pool): slot_idx
                for slot_idx, label, context, sub, ftype, pool in piped_jobs
            }
            for fut in as_completed(futs):
                slots[futs[fut]] = fut.result()
    disambig_wall_s = time.perf_counter() - t_dis_0

    seen: set[str] = set()
    for fm in slots:
        assert fm is not None
        cm.features.append(fm)
        if fm.selected_code and fm.selected_code not in seen:
            seen.add(fm.selected_code)
            cm.mapped_codes.append(fm.selected_code)

    timing = {
        "retrieve_wall_s_by_model": {
            (pack.get("name") or f"model_{i}"): float(retrieve_times[i])
            for i, pack in enumerate(model_packs)
            if i < len(retrieve_times)
        },
        "retrieve_wall_s_total": float(sum(retrieve_times)),
        "disambig_wall_s": float(disambig_wall_s),
        "n_piped": len(piped_jobs),
        "n_disambig_calls": len(piped_jobs),
        "max_fused": int(max_fused) if max_fused is not None else int(2 * top_n),
        "top_n": int(top_n),
        "min_similarity": float(min_similarity),
    }
    return cm, timing


# ── LEGACY: shortlist disambiguation (pilot-1) ────────────────────────────────

def format_candidates(candidates: list[dict]) -> str:
    """Format candidates as a lettered list."""
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f"{candidate_label(i)}. [{c['var_code']}] {c['question_text']}")
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
