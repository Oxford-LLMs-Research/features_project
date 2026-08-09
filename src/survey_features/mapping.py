"""
Dual-layer mapping: retrieve + disambiguate parent features and bundled sub_items.

Pipeline order: extraction -> mapping -> score.
Only types in ``pipe_types`` enter retrieve+disambiguate; others are recorded as
not_piped metadata. Headline codes for scoring are ``expanded_codes`` (parents
plus sub_item units when |sub_items| >= 2).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from .prompts import DISAMBIG_PROMPT
from .retrieval import retrieve_candidates_batch

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Only expand when the extractor listed 2+ sub-measures (no singleton inflation).
MIN_SUBITEMS_TO_EXPAND = 2

MAP_STATUS_MAPPED = "mapped"
MAP_STATUS_NOT_PIPED = "not_piped"
MAP_STATUS_EMPTY_POOL = "empty_pool"
MAP_STATUS_MODEL_NONE = "model_none"
MAP_STATUS_MODEL_EMPTY = "model_empty"
MAP_STATUS_UNPARSEABLE = "unparseable"


def candidate_label(i: int) -> str:
    """0-based index → A..Z, then AA..AZ, BA.."""
    if i < 0:
        raise IndexError(i)
    if i < 26:
        return _LETTERS[i]
    i -= 26
    return _LETTERS[i // 26] + _LETTERS[i % 26]


def candidate_labels(n: int) -> list[str]:
    return [candidate_label(i) for i in range(n)]


def parse_letter(raw: str, n: int) -> int | None:
    """Parse a disambiguation reply into a 0-based index, or None for 'none'.

    Prefers an exact label token (longest first so AA wins over A); among equal
    length, the last match so "Not A; I'd choose C" resolves to C.
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
    *,
    max_tokens: int = 2048,
) -> tuple[str | None, str | None, str, str]:
    """Ask disambiguator for one letter/none; returns (code, text, raw, map_status)."""
    if not pool:
        return None, None, "", MAP_STATUS_EMPTY_POOL
    block = "\n".join(
        f"{candidate_label(i)}. [{c['var_code']}] {c['question_text']}"
        for i, c in enumerate(pool)
    )
    # Reasoning models need CoT headroom; Flash/V4 often need >2048.
    raw = disambig_fn(
        [{"role": "user", "content": DISAMBIG_PROMPT.format(
            feature_label=label, feature_context=context or label, candidates_block=block)}],
        max_tokens=max(256, int(max_tokens)), temperature=0.0, usage_phase="disambig",
    ) or ""
    idx = parse_letter(raw, len(pool))
    if idx is None:
        return None, None, raw, classify_none_raw(raw)
    return pool[idx]["var_code"], pool[idx]["question_text"], raw, MAP_STATUS_MAPPED


@dataclass
class FeatureMap:
    feature_label: str
    feature_context: str
    sub_items: list[str]
    ftype: str
    piped: bool
    selected_code: str | None
    selected_text: str | None
    candidates: list[dict]
    disambig_raw: str
    map_status: str = MAP_STATUS_NOT_PIPED


@dataclass
class MapUnit:
    """One retrieval+disambiguation unit (parent feature or a single sub_item)."""

    unit_kind: str  # "parent" | "sub_item"
    parent_feature: str
    unit_label: str
    unit_context: str
    ftype: str
    piped: bool
    selected_code: str | None
    selected_text: str | None
    candidates: list[dict]
    disambig_raw: str
    map_status: str = MAP_STATUS_NOT_PIPED


@dataclass
class ExpandedCellMap:
    """Dual-layer map result: parent units + optional sub_item units."""

    cell: str
    arm: str
    mapper_model: str
    mapping_mode: str = "parent_plus_subitems"
    features: list[FeatureMap] = field(default_factory=list)
    units: list[MapUnit] = field(default_factory=list)
    parent_codes: list[str] = field(default_factory=list)
    subitem_codes: list[str] = field(default_factory=list)
    expanded_codes: list[str] = field(default_factory=list)
    extract_raw: str = ""

    @property
    def mapped_codes(self) -> list[str]:
        """Headline codes for scoring: dual-layer expanded set."""
        return self.expanded_codes

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_piped(self) -> int:
        return sum(1 for f in self.features if f.piped)

    @property
    def n_mapped(self) -> int:
        return len(self.parent_codes)

    @property
    def n_none(self) -> int:
        return sum(1 for f in self.features if f.piped and f.selected_code is None)

    @property
    def n_bundled(self) -> int:
        return sum(1 for f in self.features if len(f.sub_items) >= MIN_SUBITEMS_TO_EXPAND)

    def type_counts(self) -> dict:
        out: dict = {}
        for f in self.features:
            out[f.ftype] = out.get(f.ftype, 0) + 1
        return out

    def status_counts(self, *, units: bool = False) -> dict:
        out: dict = {}
        src = self.units if units else self.features
        for item in src:
            out[item.map_status] = out.get(item.map_status, 0) + 1
        return out


def subitem_context(parent_label: str, parent_context: str) -> str:
    """Retrieval/disambig context for a sub_item (parent-anchored)."""
    base = (parent_context or "").strip()
    if not base:
        return f"sub-measure of {parent_label}"
    if parent_label and parent_label.lower() not in base.lower():
        return f"{base} (sub-measure of {parent_label})"
    if "sub-measure" not in base.lower() and parent_label:
        return f"{base} (sub-measure of {parent_label})"
    return base


def _append_unique(code: str | None, seen: set[str], out: list[str]) -> None:
    if code and code not in seen:
        seen.add(code)
        out.append(code)


def map_features_with_subitems(
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
    expand_subitems: bool = True,
    workers: int = 1,
    disambig_max_tokens: int = 2048,
) -> ExpandedCellMap:
    """Retrieve + disambiguate parents, and optionally each bundled sub_item.

    ``workers`` > 1 parallelizes disambiguation LLM calls after serial retrieval.
    ``disambig_max_tokens`` raises CoT budget for reasoning disambiguators (e.g. Flash).
    """
    excluded = set(excluded_codes or set())
    pipe = pipe_types or {"respondent_attribute"}
    cm = ExpandedCellMap(cell=cell, arm=arm, mapper_model=mapper_model, extract_raw=extract_raw)
    n_workers = max(1, int(workers))
    d_max = max(256, int(disambig_max_tokens))
    pending: list[tuple] = []
    queries: list[tuple[str, str]] = []

    for f in features:
        label = f.get("feature", "")
        context = f.get("context", "")
        sub = list(f.get("sub_items", []) or [])
        ftype = f.get("type", "respondent_attribute")
        if not label:
            continue

        if ftype not in pipe:
            cm.features.append(FeatureMap(
                feature_label=label, feature_context=context, sub_items=sub, ftype=ftype,
                piped=False, selected_code=None, selected_text=None, candidates=[],
                disambig_raw="", map_status=MAP_STATUS_NOT_PIPED,
            ))
            cm.units.append(MapUnit(
                unit_kind="parent", parent_feature=label, unit_label=label,
                unit_context=context, ftype=ftype, piped=False,
                selected_code=None, selected_text=None, candidates=[],
                disambig_raw="", map_status=MAP_STATUS_NOT_PIPED,
            ))
            continue

        pending.append(("parent", label, label, context or label, context, ftype, sub))
        queries.append((label, context))

        if not expand_subitems or len(sub) < MIN_SUBITEMS_TO_EXPAND:
            continue

        ctx_sub = subitem_context(label, context)
        for s in sub:
            s_label = (s or "").strip()
            if not s_label:
                continue
            pending.append(("sub_item", label, s_label, ctx_sub, "", ftype, []))
            queries.append((s_label, ctx_sub))

    pools = retrieve_candidates_batch(
        queries, embed_fn, survey_embeddings, var_codes, survey_variables, excluded, top_n,
    )
    jobs: list[tuple] = []
    for meta, pool in zip(pending, pools):
        pool = [c for c in pool if c["similarity"] >= min_similarity]
        kind, parent_label, unit_label, unit_context, feat_ctx, ftype, sub = meta
        jobs.append((kind, parent_label, unit_label, unit_context, feat_ctx, ftype, sub, pool))

    def _run_job(job: tuple) -> tuple[str | None, str | None, str, str]:
        _kind, _parent, unit_label, unit_context, _feat_ctx, _ftype, _sub, pool = job
        return _disambiguate_pool(
            unit_label, unit_context, pool, disambig_fn, max_tokens=d_max,
        )

    results: list[tuple[str | None, str | None, str, str]] = [
        (None, None, "", MAP_STATUS_NOT_PIPED)
    ] * len(jobs)
    if n_workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs):
            results[i] = _run_job(job)
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(jobs))) as ex:
            futs = {ex.submit(_run_job, job): i for i, job in enumerate(jobs)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()

    seen_parent: set[str] = set()
    seen_sub: set[str] = set()
    seen_exp: set[str] = set()

    for i, job in enumerate(jobs):
        kind, parent_label, unit_label, unit_context, feat_ctx, ftype, sub, pool = job
        sel_code, sel_text, raw, status = results[i]
        if kind == "parent":
            cm.features.append(FeatureMap(
                feature_label=parent_label, feature_context=feat_ctx, sub_items=sub,
                ftype=ftype, piped=True, selected_code=sel_code, selected_text=sel_text,
                candidates=pool, disambig_raw=raw, map_status=status,
            ))
            cm.units.append(MapUnit(
                unit_kind="parent", parent_feature=parent_label, unit_label=unit_label,
                unit_context=feat_ctx, ftype=ftype, piped=True,
                selected_code=sel_code, selected_text=sel_text,
                candidates=pool, disambig_raw=raw, map_status=status,
            ))
            _append_unique(sel_code, seen_parent, cm.parent_codes)
            _append_unique(sel_code, seen_exp, cm.expanded_codes)
        else:
            cm.units.append(MapUnit(
                unit_kind="sub_item", parent_feature=parent_label, unit_label=unit_label,
                unit_context=unit_context, ftype=ftype, piped=True,
                selected_code=sel_code, selected_text=sel_text,
                candidates=pool, disambig_raw=raw, map_status=status,
            ))
            _append_unique(sel_code, seen_sub, cm.subitem_codes)
            _append_unique(sel_code, seen_exp, cm.expanded_codes)

    return cm


def expanded_cell_to_record(
    survey: str,
    target: str,
    country: str,
    cond: str,
    arm: str,
    disambig_key: str,
    cm: ExpandedCellMap,
    embedding_model: str,
) -> dict:
    """JSON-serializable dual-layer cell record (headline = expanded_codes)."""
    return {
        "survey": survey,
        "target": target,
        "country": country,
        "condition": cond,
        "arm": arm,
        "disambiguator": disambig_key,
        "disambig_model": cm.mapper_model,
        "embedding_model": embedding_model,
        "mapping_mode": cm.mapping_mode,
        "n_features": cm.n_features,
        "n_piped": cm.n_piped,
        "n_mapped": cm.n_mapped,
        "n_none": cm.n_none,
        "n_bundled": cm.n_bundled,
        "type_counts": cm.type_counts(),
        "status_counts": cm.status_counts(),
        "unit_status_counts": cm.status_counts(units=True),
        "parent_codes": cm.parent_codes,
        "subitem_codes": cm.subitem_codes,
        "expanded_codes": cm.expanded_codes,
        "mapped_codes": list(cm.expanded_codes),
        "features": [
            {
                "feature": f.feature_label,
                "context": f.feature_context,
                "sub_items": f.sub_items,
                "type": f.ftype,
                "piped": f.piped,
                "selected_code": f.selected_code,
                "selected_text": f.selected_text,
                "map_status": f.map_status,
                "n_candidates": len(f.candidates),
                "disambig_raw": (f.disambig_raw or "")[:80],
            }
            for f in cm.features
        ],
        "units": [
            {
                "unit_kind": u.unit_kind,
                "parent_feature": u.parent_feature,
                "unit_label": u.unit_label,
                "unit_context": u.unit_context,
                "type": u.ftype,
                "piped": u.piped,
                "selected_code": u.selected_code,
                "selected_text": u.selected_text,
                "map_status": u.map_status,
                "n_candidates": len(u.candidates),
                "disambig_raw": (u.disambig_raw or "")[:80],
            }
            for u in cm.units
        ],
    }
