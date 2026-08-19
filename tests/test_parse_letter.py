"""Unit tests for survey_features.mapping.parse_letter."""

from survey_features.mapping import (
    MAP_STATUS_MODEL_EMPTY,
    MAP_STATUS_MODEL_NONE,
    MAP_STATUS_UNPARSEABLE,
    candidate_label,
    classify_none_raw,
    parse_letter,
)


def test_bare_letter():
    assert parse_letter("C", 5) == 2


def test_none_variants():
    assert parse_letter("none", 5) is None
    assert parse_letter("None of these", 5) is None
    assert parse_letter("NONE", 5) is None


def test_empty():
    assert parse_letter("", 5) is None
    assert parse_letter("   ", 5) is None


def test_adversarial_not_first_letter():
    """Chatty replies must not return the first letter that appears anywhere."""
    assert parse_letter("Not A; I'd choose C", 5) == 2


def test_aa_longest_token_wins():
    assert candidate_label(26) == "AA"
    assert parse_letter("Answer: AA", 30) == 26


def test_out_of_range_letter():
    assert parse_letter("E", 3) is None


def test_classify_none_raw():
    assert classify_none_raw("") == MAP_STATUS_MODEL_EMPTY
    assert classify_none_raw("   ") == MAP_STATUS_MODEL_EMPTY
    assert classify_none_raw("none") == MAP_STATUS_MODEL_NONE
    assert classify_none_raw("None of these fit") == MAP_STATUS_MODEL_NONE
    assert classify_none_raw("I am unsure which to pick") == MAP_STATUS_UNPARSEABLE
