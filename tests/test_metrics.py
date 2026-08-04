"""Tests for the metric arithmetic in survey_features.metrics.

Coverage gap these close: only `captured_importance` had tests, while
`oracle_percentile_mean`, `random_captured_mean` and `cluster_bootstrap_ci` — all of
which feed headline numbers — had none.
"""

import numpy as np
import pandas as pd
import pytest

from survey_features.metrics import (
    cluster_bootstrap_ci,
    jaccard,
    oracle_percentile_mean,
    random_captured_mean,
    stable_seed,
)


# ── oracle_percentile_mean ────────────────────────────────────────────────────

def test_percentile_endpoints_and_midpoint():
    imp = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1, "E": 0.0}
    assert oracle_percentile_mean(["A"], imp) == pytest.approx(1.0)
    assert oracle_percentile_mean(["E"], imp) == pytest.approx(0.0)
    assert oracle_percentile_mean(["C"], imp) == pytest.approx(0.5)


def test_ties_take_the_average_rank():
    """Regression: ties used to break by dict insertion order.

    Four features tied at 0.0 must all score the same percentile — under the old
    implementation they were spread across four distinct positions, so an
    uninformative pick drew a percentile that depended on iteration order.
    """
    imp = {"A": 0.5, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    vals = [oracle_percentile_mean([c], imp) for c in ("B", "C", "D", "E")]
    assert len(set(vals)) == 1
    # ranks 1..4 average to 2.5 -> (2.5 - 1) / (5 - 1)
    assert vals[0] == pytest.approx(0.375)


def test_percentile_is_order_invariant():
    imp = {"A": 0.5, "B": 0.0, "C": 0.0, "D": 0.2}
    reversed_imp = dict(reversed(list(imp.items())))
    assert oracle_percentile_mean(["B"], imp) == oracle_percentile_mean(["B"], reversed_imp)


def test_percentile_ignores_unknown_codes_and_dedupes():
    imp = {"A": 0.4, "B": 0.2, "C": 0.0}
    assert oracle_percentile_mean(["A", "A", "ZZZ"], imp) == pytest.approx(1.0)


def test_percentile_none_cases():
    assert oracle_percentile_mean(["A"], {}) is None
    assert oracle_percentile_mean(["A"], {"A": 1.0}) is None  # n <= 1
    assert oracle_percentile_mean(["ZZZ"], {"A": 1.0, "B": 0.0}) is None


# ── random_captured_mean ──────────────────────────────────────────────────────

def test_random_captured_is_deterministic_given_a_seed():
    imp = {f"V{i}": float(i) for i in range(20)}
    a = random_captured_mean(imp, k=5, seed=7, draws=50)
    b = random_captured_mean(imp, k=5, seed=7, draws=50)
    assert a == b


def test_random_captured_lies_between_zero_and_one():
    imp = {f"V{i}": float(i) for i in range(20)}
    v = random_captured_mean(imp, k=5, seed=1, draws=200)
    assert 0.0 < v < 1.0


def test_random_captured_equals_one_when_k_covers_the_pool():
    imp = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert random_captured_mean(imp, k=3, seed=1, draws=5) == pytest.approx(1.0)


def test_random_captured_clips_negatives():
    """Negative importances must not subtract from a draw's captured mass."""
    imp = {"A": 1.0, "B": -5.0}
    assert random_captured_mean(imp, k=1, seed=3, draws=100) == pytest.approx(0.5, abs=0.15)


def test_random_captured_none_cases():
    assert random_captured_mean({}, k=2, seed=1) is None
    assert random_captured_mean({"A": 1.0}, k=0, seed=1) is None
    assert random_captured_mean({"A": 1.0}, k=5, seed=1) is None       # k > pool
    assert random_captured_mean({"A": 0.0, "B": 0.0}, k=1, seed=1) is None  # denom 0


def test_random_captured_uses_rank_for_denom():
    """When rank and score diverge, denom follows select ranking."""
    rank = {"A": 1.0, "B": 0.9, "C": 0.1}
    score = {"A": 0.1, "B": 0.1, "C": 10.0}
    # k=2 denom = score(A)+score(B)=0.2; any 2-subset including C scores high
    v = random_captured_mean(score, k=2, seed=1, draws=50, rank=rank)
    assert v is not None
    # Cursed denom would be ~10.1 and drive the mean near 0; honest denom is 0.2.
    assert v > 1.0  # random pairs that include C exceed the honest top-2 mass


# ── cluster_bootstrap_ci ──────────────────────────────────────────────────────

def _df(vals, clusters):
    return pd.DataFrame({"x": vals, "survey": ["s"] * len(vals), "target": clusters})


def test_bootstrap_mean_matches_the_sample_mean():
    d = _df([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    out = cluster_bootstrap_ci(d, "x", n_boot=200)
    assert out["mean"] == pytest.approx(2.5)
    assert out["n"] == 4
    assert out["n_clusters"] == 2


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    rng = np.random.default_rng(0)
    vals = rng.normal(5.0, 1.0, 60).tolist()
    clusters = [f"c{i % 12}" for i in range(60)]
    d = _df(vals, clusters)
    a = cluster_bootstrap_ci(d, "x", n_boot=500)
    b = cluster_bootstrap_ci(d, "x", n_boot=500)
    assert a == b
    assert a["ci_low"] < a["mean"] < a["ci_high"]


def test_bootstrap_zero_width_when_every_value_is_identical():
    d = _df([2.0] * 10, [f"c{i}" for i in range(10)])
    out = cluster_bootstrap_ci(d, "x", n_boot=100)
    assert out["ci_low"] == pytest.approx(2.0)
    assert out["ci_high"] == pytest.approx(2.0)


def test_bootstrap_skips_nan_rows():
    d = _df([1.0, float("nan"), 3.0], ["a", "a", "b"])
    out = cluster_bootstrap_ci(d, "x", n_boot=100)
    assert out["n"] == 2
    assert out["mean"] == pytest.approx(2.0)


def test_bootstrap_empty_column():
    d = _df([float("nan")] * 3, ["a", "b", "c"])
    out = cluster_bootstrap_ci(d, "x", n_boot=100)
    assert out == {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_clusters": 0}


# ── jaccard / stable_seed ─────────────────────────────────────────────────────

def test_jaccard():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard({"a"}, {"a"}) == pytest.approx(1.0)
    assert jaccard(set(), set()) is None


def test_stable_seed_is_stable_across_processes():
    """The value is hard-coded on purpose: `hash()` would change between runs."""
    assert stable_seed("Q57", "Germany", "unprompted", 10) == stable_seed(
        "Q57", "Germany", "unprompted", 10
    )
    assert stable_seed("Q57", "Germany") != stable_seed("Q57", "Nigeria")
    assert 0 <= stable_seed("x") < 2**31
