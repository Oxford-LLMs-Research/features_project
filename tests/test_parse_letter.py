"""Unit tests for survey_features.disambig.parse_letter (current per-feature parser)."""

from survey_features.disambig import candidate_label, parse_letter


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
    # Only A..C are valid when n=3; "E" is not in the pool.
    assert parse_letter("E", 3) is None
