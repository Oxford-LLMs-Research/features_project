"""Confirmatory grid keep/drop rules.

A cell stays in the default grid (`layout.genuine_cells()`) unless it is
**unestimable** (type-1) or **leakage**. Accuracy lift vs the majority class is
not a drop rule: ordinals that lose to the mode on accuracy, and binary/nominal
cells with tiny accuracy lift, can still have real log-loss / Spearman PI
(docs/pre_paper_run_decisions.md, 2026-08-16).

`scripts/leakage_audit.py` writes `leakage_class`; this module is the classifier.
"""
from __future__ import annotations

from dataclasses import dataclass

# Classification: minority rows expected on the 20% valuation holdout (V2).
# ≥10 on V2 ⇒ ~50 in the cell (pre_paper_run_decisions.md). Modal share alone
# is not the test — Q43A is 84–91% majority with real log-loss PI on large n.
TYPE1_MIN_V2_MINORITY = 10.0

# oracle_ceiling@5 below this is compromised ranking (Q141 Andorra ≈ 0.24).
TYPE1_MIN_CEILING_AT_5 = 0.30

CLASSIFICATION_PROBLEMS = frozenset({"binary", "multiclass"})
CLASSIFICATION_TYPES = frozenset({"binary", "nominal"})

KEEP_CLASSES = frozenset({"genuine"})


@dataclass(frozen=True)
class ScreenThresholds:
    conc_thresh: float = 0.80
    recover_frac: float = 0.90
    implausible_acc: float = 0.95
    implausible_min_lift: float = 0.20
    suspect_lift: float = 0.10
    min_v2_minority: float = TYPE1_MIN_V2_MINORITY
    min_ceiling_at_5: float = TYPE1_MIN_CEILING_AT_5


def _num(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def is_classification(row: dict) -> bool:
    problem = str(row.get("problem_type") or "")
    if problem in CLASSIFICATION_PROBLEMS:
        return True
    ttype = str(row.get("target_type") or "")
    return ttype in CLASSIFICATION_TYPES


def estimated_minority(n: float | None, majority: float | None) -> float | None:
    """Expected count of non-mode rows in a split of size n (binary: the minority)."""
    if n is None or majority is None:
        return None
    return float(n) * (1.0 - float(majority))


def type1_reason(row: dict, thresholds: ScreenThresholds | None = None) -> str | None:
    """Why the cell is unestimable, or None if type-1 does not fire.

    Classification: minority too thin on V2 (and V1 when n_select is stored).
    All types: oracle_ceiling@5 below the compromised-PI floor.
    """
    t = thresholds or ScreenThresholds()
    ceiling = _num(row.get("ceiling_at_5"))
    if ceiling is not None and ceiling < t.min_ceiling_at_5:
        return "low_ceiling"

    if not is_classification(row):
        return None

    majority = _num(row.get("majority_baseline"))
    v2 = _num(row.get("v2_minority_est"))
    if v2 is None:
        v2 = estimated_minority(_num(row.get("n_score")), majority)
    if v2 is not None and v2 < t.min_v2_minority:
        return "thin_minority_v2"

    v1 = estimated_minority(_num(row.get("n_select")), majority)
    if v1 is not None and v1 < t.min_v2_minority:
        return "thin_minority_v1"
    return None


def classify_cell(row: dict, thresholds: ScreenThresholds | None = None) -> str:
    """Return leakage_class for one audit row.

    unestimable          type-1 (thin minority on the honest split, or low ceiling)
    leakage              concentrated single-feature recovery
    leakage_distributed  implausible accuracy with spread importance (skip-pattern)
    leakage_suspect      offline concentration heuristic (no single-feature test)
    genuine              keep — including type-2/3 accuracy-vs-majority cases
    """
    t = thresholds or ScreenThresholds()
    reason = type1_reason(row, t)
    if reason:
        row["unestimable_reason"] = reason
        return "unestimable"

    majority = _num(row.get("majority_baseline")) or 0.0
    oracle_acc = _num(row.get("oracle_acc"))
    lift = (oracle_acc - majority) if oracle_acc is not None else None
    conc = _num(row.get("top_importance_share")) or 0.0
    sf = _num(row.get("single_feature_acc"))

    if sf is not None:
        if lift is not None and lift > 0:
            recover = (sf - majority) / lift
            row["single_feature_recovery"] = recover
            if recover >= t.recover_frac and conc >= t.conc_thresh:
                return "leakage"
        # Skip-pattern modules: near-perfect accuracy *and* well above the mode
        # (Q67A: acc 0.99 vs majority 0.50–0.70). Acc ≈ majority is just the mode.
        if (
            oracle_acc is not None
            and lift is not None
            and oracle_acc >= t.implausible_acc
            and lift >= t.implausible_min_lift
        ):
            return "leakage_distributed"
        return "genuine"

    # Offline fallback: concentration-only heuristic.
    if lift is not None and conc >= t.conc_thresh and lift >= t.suspect_lift:
        return "leakage_suspect"
    return "genuine"
