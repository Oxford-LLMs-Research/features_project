"""Confirmatory grid keep/drop rules — typed, self-contained.

A cell stays in the default grid (`layout.genuine_cells()`) unless it is
**unestimable** (type-1) or **leakage**. Accuracy lift vs the majority class is
not a drop rule: ordinals that lose to the mode on accuracy, and binary/nominal
cells with tiny accuracy lift, can still have real log-loss / Spearman PI
(docs/pre_paper_run_decisions.md, 2026-08-16).

Type-matched (2026-08-19): every gate is framed in the cell's own primary metric —
accuracy vs majority for binary/nominal, Spearman rho vs 0 for ordinal/continuous.
No ordinal or continuous target is ever judged as multiclass accuracy. Inputs come
only from the cell's own oracle_meta.json / oracle.csv and a fresh typed probe
(`scripts/leakage_audit.py`); no historical score files.

`scripts/leakage_audit.py` writes `leakage_class`; this module is the classifier.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Type-1 thresholds (single home; scripts/target_universe_screen.py imports) ──

# Classification: minority rows expected on the 20% valuation holdout (V2).
# ≥10 on V2 ⇒ ~50 in the cell (pre_paper_run_decisions.md). Modal share alone
# is not the test — Q43A is 84–91% majority with real log-loss PI on large n.
TYPE1_MIN_V2_MINORITY = 10.0

# oracle_ceiling@5 below this is compromised ranking (Q141 Andorra ≈ 0.24).
TYPE1_MIN_CEILING_AT_5 = 0.30

# Regression (ordinal/continuous): the valuation holdout needs enough rows for a
# stable rank correlation, and the target needs an actual scale.
TYPE1_MIN_N = 50
TYPE1_MIN_UNIQUE = 3

CLASSIFICATION_PROBLEMS = frozenset({"binary", "multiclass"})
CLASSIFICATION_TYPES = frozenset({"binary", "nominal"})

KEEP_CLASSES = frozenset({"genuine"})


@dataclass(frozen=True)
class ScreenThresholds:
    conc_thresh: float = 0.80          # top-feature share of positive score mass
    recover_frac: float = 0.90         # single feature recovers this much of oracle PI
    min_v2_minority: float = TYPE1_MIN_V2_MINORITY
    min_ceiling_at_5: float = TYPE1_MIN_CEILING_AT_5
    min_n_regression: float = TYPE1_MIN_N
    min_unique_regression: float = TYPE1_MIN_UNIQUE
    # Distributed (skip-pattern module) leakage.
    implausible_acc: float = 0.95      # classification: oracle acc this high...
    implausible_min_lift: float = 0.20 # ...and well above the mode (not just modal)
    implausible_rho: float = 0.95      # regression: no attitude item ranks this well
    # ABSOLUTE near-duplicate: one column recovers the target near-deterministically,
    # regardless of what the oracle-side numbers say (SE7a <- SE7, oracle lift ~0).
    abs_dup_acc: float = 0.90
    abs_dup_min_lift: float = 0.20
    abs_dup_rho: float = 0.90
    # Regression recovery guard: below this oracle rho the recovery ratio is noise.
    min_oracle_rho: float = 0.05


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


def rank_holdout_size(row: dict) -> float | None:
    """Rows per CV ranking-fold holdout, from v4 meta fields.

    mean(fold_fit_sizes) = n_cv * (k-1)/k, so one fold's holdout is
    mean(fold_fit_sizes) / (k-1).
    """
    folds = _num(row.get("cv_folds"))
    sizes = row.get("fold_fit_sizes")
    if not folds or folds < 2 or not isinstance(sizes, (list, tuple)) or not sizes:
        return None
    vals = [_num(v) for v in sizes]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (sum(vals) / len(vals)) / (folds - 1.0)


def type1_reason(row: dict, thresholds: ScreenThresholds | None = None) -> str | None:
    """Why the cell is unestimable, or None if type-1 does not fire.

    All types: oracle_ceiling@5 below the compromised-PI floor.
    Classification: minority too thin on V2, or on the CV ranking-fold holdouts.
    Regression: valuation holdout too small, or too few distinct scale points.
    """
    t = thresholds or ScreenThresholds()
    ceiling = _num(row.get("ceiling_at_5"))
    if ceiling is not None and ceiling < t.min_ceiling_at_5:
        return "low_ceiling"

    if is_classification(row):
        majority = _num(row.get("majority_baseline"))
        v2 = _num(row.get("v2_minority_est"))
        if v2 is None:
            v2 = estimated_minority(_num(row.get("n_score")), majority)
        if v2 is not None and v2 < t.min_v2_minority:
            return "thin_minority_v2"
        rank_hold = estimated_minority(rank_holdout_size(row), majority)
        if rank_hold is not None and rank_hold < t.min_v2_minority:
            return "thin_minority_rank"
        return None

    n_score = _num(row.get("n_score"))
    if n_score is not None and n_score < t.min_n_regression:
        return "thin_score_regression"
    n_unique = _num(row.get("n_target_unique"))
    if n_unique is not None and n_unique < t.min_unique_regression:
        return "too_few_scale_points"
    return None


def classify_cell(row: dict, thresholds: ScreenThresholds | None = None) -> str:
    """Return leakage_class for one audit row.

    unestimable          type-1 (thin data on the honest split, or low ceiling)
    leakage              near-duplicate: absolute (one column recovers the target
                         near-deterministically) or relative (single feature
                         recovers >= recover_frac of oracle PI), with importance
                         concentrated on that column
    leakage_distributed  implausibly high oracle PI with spread importance
                         (skip-pattern module)
    leakage_suspect      concentration >= conc_thresh but no probe available to
                         confirm or clear it (offline mode / probe failure)
    genuine              keep — including type-2/3 accuracy-vs-majority cases

    Row keys consumed: problem_type/target_type, majority_baseline, n_score,
    cv_folds, fold_fit_sizes, n_target_unique, ceiling_at_5, v2_minority_est,
    top_importance_share, oracle_primary, single_feature_primary. Primary values
    are in the cell's own metric: accuracy for classification, Spearman rho for
    ordinal/continuous (see scripts/leakage_audit.py probe).
    """
    t = thresholds or ScreenThresholds()
    reason = type1_reason(row, t)
    if reason:
        row["unestimable_reason"] = reason
        return "unestimable"

    classification = is_classification(row)
    majority = _num(row.get("majority_baseline")) or 0.0
    oracle = _num(row.get("oracle_primary"))
    sf = _num(row.get("single_feature_primary"))
    conc = _num(row.get("top_importance_share")) or 0.0

    if sf is not None:
        if classification:
            # Absolute near-duplicate (fires even with zero oracle lift).
            if (sf >= t.abs_dup_acc and (sf - majority) >= t.abs_dup_min_lift
                    and conc >= t.conc_thresh):
                return "leakage"
            lift = (oracle - majority) if oracle is not None else None
            if lift is not None and lift > 0:
                recover = (sf - majority) / lift
                row["single_feature_recovery"] = recover
                if recover >= t.recover_frac and conc >= t.conc_thresh:
                    return "leakage"
            # Skip-pattern modules: near-perfect accuracy *and* well above the mode
            # (Q67A: acc 0.99 vs majority 0.50–0.70). Acc ≈ majority is just the mode.
            if (oracle is not None and lift is not None
                    and oracle >= t.implausible_acc
                    and lift >= t.implausible_min_lift):
                return "leakage_distributed"
            return "genuine"

        # Regression: rank-correlation framing; majority share plays no role.
        if sf >= t.abs_dup_rho and conc >= t.conc_thresh:
            return "leakage"
        if oracle is not None and oracle >= t.min_oracle_rho:
            recover = sf / oracle
            row["single_feature_recovery"] = recover
            if recover >= t.recover_frac and conc >= t.conc_thresh:
                return "leakage"
        if oracle is not None and oracle >= t.implausible_rho:
            return "leakage_distributed"
        return "genuine"

    # No probe (offline mode, or the data-backed probe failed): a concentrated
    # cell cannot be cleared, so it degrades to suspect — never to genuine.
    if conc >= t.conc_thresh:
        return "leakage_suspect"
    return "genuine"
