"""Tests for the scoring helpers that are pure functions (no survey data, no XGBoost)."""

import pytest

from survey_features.score_cell import SCORE_COLS, baseline_fingerprint, score_cols


# ── baseline cache fingerprint ────────────────────────────────────────────────
# Regression: the first version of the per-(cell, k) baseline cache had no identity
# check, so a cache written before the textbook baseline existed kept serving
# `textbook_acc: null` afterwards — silently, and only visible as an empty column.

POOL = [f"V{i}" for i in range(30)]
TEXTBOOK = ["Q1", "Q100", "Q94"]


def test_fingerprint_is_stable_for_identical_inputs():
    assert baseline_fingerprint(POOL, TEXTBOOK, 10) == baseline_fingerprint(POOL, TEXTBOOK, 10)


def test_fingerprint_changes_when_the_textbook_set_appears():
    assert baseline_fingerprint(POOL, [], 10) != baseline_fingerprint(POOL, TEXTBOOK, 10)


def test_fingerprint_changes_when_the_textbook_order_changes():
    """Fixed-k budgets take a prefix, so order is part of the baseline's identity."""
    assert baseline_fingerprint(POOL, TEXTBOOK, 10) != baseline_fingerprint(
        POOL, list(reversed(TEXTBOOK)), 10
    )


def test_fingerprint_changes_when_the_random_pool_changes():
    """The oracle re-run rewrites the feature pool, which redefines the random null."""
    assert baseline_fingerprint(POOL, TEXTBOOK, 10) != baseline_fingerprint(
        POOL[:-1], TEXTBOOK, 10
    )
    assert baseline_fingerprint(POOL, TEXTBOOK, 10) != baseline_fingerprint(
        POOL[:-1] + ["ZZZ"], TEXTBOOK, 10
    )


def test_fingerprint_changes_with_the_draw_count():
    assert baseline_fingerprint(POOL, TEXTBOOK, 10) != baseline_fingerprint(POOL, TEXTBOOK, 5)


def test_fingerprint_ignores_pool_ordering():
    """The pool is a sampling universe, not a ranking."""
    assert baseline_fingerprint(POOL, TEXTBOOK, 10) == baseline_fingerprint(
        list(reversed(POOL)), TEXTBOOK, 10
    )


def test_fingerprint_changes_when_the_target_is_dropped_from_the_textbook_set():
    """The textbook set is filtered per cell (target removed), so the cache key must
    reflect the FILTERED list — otherwise a cell whose target is a textbook variable
    reuses a baseline computed with the target still in it."""
    full = ["SEXO", "EDAD", "S11"]
    filtered = [c for c in full if c != "SEXO"]
    assert baseline_fingerprint(POOL, full, 10) != baseline_fingerprint(POOL, filtered, 10)


# ── score column schema ───────────────────────────────────────────────────────

def test_score_cols_defaults_to_the_canonical_schema():
    assert score_cols() == SCORE_COLS


def test_score_cols_inserts_extras_before_error():
    cols = score_cols("k_mode")
    assert cols[-1] == "error"
    assert "k_mode" in cols
    assert cols.index("k_mode") < cols.index("error")


def test_score_cols_does_not_duplicate_existing_columns():
    assert score_cols("k") == SCORE_COLS


@pytest.mark.parametrize("col", [
    "captured_importance", "textbook_captured",
    "value_over_random", "value_over_textbook", "cost_of_imperfect",
    "value_over_random_ll", "value_over_textbook_ll", "cost_of_imperfect_ll",
])
def test_headline_metrics_are_in_the_schema(col):
    assert col in SCORE_COLS
