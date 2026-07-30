"""
Sub-item mapping expansion (experiment): map parent features AND each bundled
sub_item as its own retrieve+disambiguate unit.

This module does NOT replace ``disambig.map_features`` (one-to-one parent path used
by the main MiniLM arm-C pipeline). Artifacts must be written under
``outputs/subitem_mapping/`` only — see docs/subitem_mapping.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .disambig import FeatureMap, _disambiguate_pool
from .retrieval import retrieve_candidates_batch

# Aligned with CellMap.n_bundled: only expand when extractor listed 2+ sub-measures.
MIN_SUBITEMS_TO_EXPAND = 2


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


@dataclass
class ExpandedCellMap:
    """Dual-layer map result: parent units + optional sub_item units."""

    cell: str
    arm: str
    mapper_model: str
    mapping_mode: str = "parent_plus_subitems"
    features: list[FeatureMap] = field(default_factory=list)  # parent-level (parity)
    units: list[MapUnit] = field(default_factory=list)
    parent_codes: list[str] = field(default_factory=list)
    subitem_codes: list[str] = field(default_factory=list)
    expanded_codes: list[str] = field(default_factory=list)
    extract_raw: str = ""

    @property
    def mapped_codes(self) -> list[str]:
        """Alias for parent_codes — same contract as CellMap.mapped_codes."""
        return self.parent_codes

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
) -> ExpandedCellMap:
    """Retrieve + disambiguate parents, and optionally each bundled sub_item.

    Parent path mirrors ``disambig.map_features``. Sub_item units use the sub_item
    string as the query label and a parent-anchored context (see ``subitem_context``).

    ``workers`` > 1 parallelizes disambiguation LLM calls after serial retrieval.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    excluded = set(excluded_codes or set())
    pipe = pipe_types or {"respondent_attribute"}
    cm = ExpandedCellMap(cell=cell, arm=arm, mapper_model=mapper_model, extract_raw=extract_raw)
    n_workers = max(1, int(workers))

    # Collect query units, batch-retrieve, then disambig (optionally threaded).
    # Parent job: (kind, parent_label, unit_label, unit_context, feature_context, ftype, sub, pool)
    # Sub job:    (kind, parent_label, unit_label, unit_context, "", ftype, [], pool)
    pending: list[tuple] = []  # job without pool
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
                piped=False, selected_code=None, selected_text=None, candidates=[], disambig_raw="",
            ))
            cm.units.append(MapUnit(
                unit_kind="parent", parent_feature=label, unit_label=label,
                unit_context=context, ftype=ftype, piped=False,
                selected_code=None, selected_text=None, candidates=[], disambig_raw="",
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

    def _run_job(job: tuple) -> tuple[str | None, str | None, str]:
        _kind, _parent, unit_label, unit_context, _feat_ctx, _ftype, _sub, pool = job
        return _disambiguate_pool(unit_label, unit_context, pool, disambig_fn)

    results: list[tuple[str | None, str | None, str]] = [(None, None, "")] * len(jobs)
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
        sel_code, sel_text, raw = results[i]
        if kind == "parent":
            cm.features.append(FeatureMap(
                feature_label=parent_label, feature_context=feat_ctx, sub_items=sub,
                ftype=ftype, piped=True, selected_code=sel_code, selected_text=sel_text,
                candidates=pool, disambig_raw=raw,
            ))
            cm.units.append(MapUnit(
                unit_kind="parent", parent_feature=parent_label, unit_label=unit_label,
                unit_context=feat_ctx, ftype=ftype, piped=True,
                selected_code=sel_code, selected_text=sel_text,
                candidates=pool, disambig_raw=raw,
            ))
            _append_unique(sel_code, seen_parent, cm.parent_codes)
            _append_unique(sel_code, seen_exp, cm.expanded_codes)
        else:
            cm.units.append(MapUnit(
                unit_kind="sub_item", parent_feature=parent_label, unit_label=unit_label,
                unit_context=unit_context, ftype=ftype, piped=True,
                selected_code=sel_code, selected_text=sel_text,
                candidates=pool, disambig_raw=raw,
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
    """JSON-serializable cell record for outputs/subitem_mapping/.../maps/."""
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
        "parent_codes": cm.parent_codes,
        "subitem_codes": cm.subitem_codes,
        "expanded_codes": cm.expanded_codes,
        "mapped_codes": cm.parent_codes,  # baseline-compatible alias
        "features": [
            {
                "feature": f.feature_label,
                "context": f.feature_context,
                "sub_items": f.sub_items,
                "type": f.ftype,
                "piped": f.piped,
                "selected_code": f.selected_code,
                "selected_text": f.selected_text,
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
                "n_candidates": len(u.candidates),
                "disambig_raw": (u.disambig_raw or "")[:80],
            }
            for u in cm.units
        ],
    }
