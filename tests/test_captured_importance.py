"""Unit tests for survey_features.metrics.captured_importance."""

from survey_features.metrics import captured_importance, oracle_topk_codes


def test_basic_matched_k():
    imp = {"A": 0.5, "B": 0.3, "C": 0.1}
    # k defaults to 2; denom = top-2 oracle mass = 0.5+0.3
    assert captured_importance(["A", "C"], imp) == (0.5 + 0.1) / (0.5 + 0.3)


def test_duplicates_deduped_arrival_order():
    imp = {"A": 0.5, "B": 0.3, "C": 0.1}
    assert captured_importance(["A", "A", "B"], imp) == captured_importance(["A", "B"], imp)


def test_fixed_k_five_with_two_codes():
    # Five positive features so top-5 denom is well-defined.
    imp = {"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.05, "E": 0.04, "F": 0.01}
    codes = ["A", "C"]
    denom = 0.5 + 0.3 + 0.1 + 0.05 + 0.04
    assert captured_importance(codes, imp, k=5) == (0.5 + 0.1) / denom


def test_empty_imp_or_empty_codes():
    assert captured_importance(["A"], {}) is None
    assert captured_importance([], {"A": 0.5}) is None
    assert captured_importance(["A"], {"A": 0.5}, k=0) is None


def test_negative_importance_clipped():
    # Negatives do not contribute to denom or numerator.
    imp = {"A": 0.5, "B": -0.2, "C": 0.3}
    # top-k=2 positive mass among clipped values: 0.5, 0.3 ( -0.2 → 0)
    got = captured_importance(["A", "B"], imp)
    assert got == (0.5 + 0.0) / (0.5 + 0.3)


def test_honest_select_score_split():
    """Denom top-k chosen on rank (select); mass taken from score."""
    # Select ranks A,B highest; score says C,D are larger — winner's-curse trap.
    rank = {"A": 0.9, "B": 0.8, "C": 0.1, "D": 0.05}
    score = {"A": 0.1, "B": 0.1, "C": 0.9, "D": 0.8}
    # Honest denom = score(A)+score(B) = 0.2; cursed would be score(C)+score(D)=1.7
    assert oracle_topk_codes(rank, 2) == ["A", "B"]
    assert captured_importance(["A", "C"], score, k=2, rank=rank) == (0.1 + 0.9) / (0.1 + 0.1)
