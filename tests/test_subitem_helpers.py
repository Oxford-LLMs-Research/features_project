"""Unit tests for dual-layer expand helpers (no LLM / retrieval)."""

from survey_features.mapping import (
    MIN_SUBITEMS_TO_EXPAND,
    _append_unique,
    subitem_context,
)


def test_min_subitems_to_expand_is_two():
    """Only |S| >= 2 triggers sub_item units (no singleton inflation)."""
    assert MIN_SUBITEMS_TO_EXPAND == 2


def test_subitem_context_with_parent_context():
    ctx = subitem_context("assets", "wealth")
    assert "wealth" in ctx
    assert "assets" in ctx
    assert "sub-measure" in ctx.lower()


def test_subitem_context_empty_parent_context():
    assert subitem_context("assets", "") == "sub-measure of assets"


def test_expanded_codes_union_parents_first_dedup():
    """Per-layer dedup; expanded is cross-layer union (parents first)."""
    parent_codes: list[str] = []
    subitem_codes: list[str] = []
    expanded_codes: list[str] = []
    seen_parent: set[str] = set()
    seen_sub: set[str] = set()
    seen_exp: set[str] = set()

    _append_unique("Q1", seen_parent, parent_codes)
    _append_unique("Q1", seen_exp, expanded_codes)
    for code in ("Q2", "Q1", None, "Q2"):
        _append_unique(code, seen_sub, subitem_codes)
        _append_unique(code, seen_exp, expanded_codes)

    assert parent_codes == ["Q1"]
    assert subitem_codes == ["Q2", "Q1"]
    assert expanded_codes == ["Q1", "Q2"]
