"""Country-blind work is computed once per question, not once per cell.

Regression cover for the duplicate-generation bug: the unprompted prompt never
names the country, so a 3-country question was paying for three identical
generate -> extract -> map chains and — because providers are not deterministic
at temperature 0 — getting three *different* essays, which put an unregistered
generation draw inside the country contrast.
"""

import threading

from survey_features.config import CONDITIONS, COUNTRY_BLIND_CONDITIONS
from survey_features.dedupe import SharedByQuestion, question_siblings


# ── the invariant this rests on ──────────────────────────────────────────────

def test_country_blind_conditions_are_a_subset_of_conditions():
    assert COUNTRY_BLIND_CONDITIONS
    assert COUNTRY_BLIND_CONDITIONS <= set(CONDITIONS)
    # country_provided names the country in the prompt; it must never be shared.
    assert "country_provided" not in COUNTRY_BLIND_CONDITIONS


# ── question_siblings ────────────────────────────────────────────────────────

def test_question_siblings_groups_by_question():
    cells = [
        ("wvs", "Q1", "Japan"),
        ("wvs", "Q1", "Chile"),
        ("wvs", "Q2", "Japan"),
        ("ess_wave_11", "Q1", "Austria"),
    ]
    sib = question_siblings(cells)
    assert sib[("wvs", "Q1")] == ["Japan", "Chile"]
    assert sib[("wvs", "Q2")] == ["Japan"]
    # same target code in another survey is a different question
    assert sib[("ess_wave_11", "Q1")] == ["Austria"]


def test_question_siblings_dedups_repeated_country():
    sib = question_siblings([("wvs", "Q1", "Japan"), ("wvs", "Q1", "Japan")])
    assert sib[("wvs", "Q1")] == ["Japan"]


# ── SharedByQuestion ─────────────────────────────────────────────────────────

def test_computes_once_per_key():
    shared = SharedByQuestion()
    calls = []

    def compute():
        calls.append(1)
        return "essay"

    got = [shared.get(("wvs", "Q1", "unprompted"), compute) for _ in range(3)]
    assert got == ["essay"] * 3
    assert len(calls) == 1
    assert (shared.computed, shared.shared) == (1, 2)


def test_distinct_keys_compute_separately():
    shared = SharedByQuestion()
    shared.get(("wvs", "Q1", "unprompted"), lambda: "a")
    shared.get(("wvs", "Q2", "unprompted"), lambda: "b")
    assert shared.get(("wvs", "Q1", "unprompted"), lambda: "never") == "a"
    assert shared.computed == 2


def test_reuse_wins_over_compute():
    """A previous run's artifact on disk means no API call at all."""
    shared = SharedByQuestion()
    shared.get(
        ("wvs", "Q1", "unprompted"),
        compute=lambda: pytest_fail("compute must not run when reuse returns a value"),
        reuse=lambda: "essay from disk",
    )
    assert shared._values[("wvs", "Q1", "unprompted")] == "essay from disk"
    assert (shared.computed, shared.shared) == (0, 1)


def test_reuse_returning_none_falls_through_to_compute():
    shared = SharedByQuestion()
    got = shared.get(("wvs", "Q1", "unprompted"), lambda: "fresh", reuse=lambda: None)
    assert got == "fresh"
    assert shared.computed == 1


def test_reuse_is_consulted_only_once_per_key():
    """Siblings hit the memo, not the filesystem, for the rest of the run."""
    shared = SharedByQuestion()
    reuse_calls = []

    def reuse():
        reuse_calls.append(1)
        return None

    for _ in range(3):
        shared.get(("wvs", "Q1", "unprompted"), lambda: "x", reuse=reuse)
    assert len(reuse_calls) == 1


def test_concurrent_siblings_pay_once():
    """The pipeline runs several cells of one question at the same time.

    Without the per-key lock the three threads would each miss the memo and
    each fire a generation — exactly the bug, restaged in-process.
    """
    shared = SharedByQuestion()
    started = threading.Event()
    calls = []
    lock = threading.Lock()

    def compute():
        with lock:
            calls.append(1)
        started.set()
        # hold the key long enough that siblings are certain to arrive mid-flight
        threading.Event().wait(0.05)
        return "essay"

    results = {}

    def worker(i):
        results[i] = shared.get(("wvs", "Q1", "unprompted"), compute)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert set(results.values()) == {"essay"}
    assert shared.computed == 1
    assert shared.shared == 3


def test_summary_reports_both_counters():
    shared = SharedByQuestion()
    shared.get("k", lambda: 1)
    shared.get("k", lambda: 1)
    assert shared.summary() == "computed=1 shared=1"


def pytest_fail(msg):  # pragma: no cover - helper for the reuse test
    raise AssertionError(msg)
