"""Confirmatory grid screen: keep type-2/3 signal; drop type-1 and leakage."""

import pytest

from survey_features.grid_screen import (
    ScreenThresholds,
    classify_cell,
    type1_reason,
)


def test_type2_tiny_accuracy_lift_is_genuine():
    """Q104-style: oracle beats the mode by < 0.03 — still keep."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.60,
        "oracle_acc": 0.629,
        "n_score": 200,
        "ceiling_at_5": 0.70,
        "top_importance_share": 0.20,
    }
    assert type1_reason(row) is None
    assert classify_cell(row) == "genuine"


def test_type3_accuracy_below_mode_ordinal_is_genuine():
    """P16ST-style: Spearman PI, accuracy loses to the mode — still keep."""
    row = {
        "problem_type": "regression",
        "target_type": "ordinal",
        "majority_baseline": 0.22,
        "oracle_acc": 0.18,
        "n_score": 400,
        "ceiling_at_5": 0.85,
        "top_importance_share": 0.15,
    }
    assert type1_reason(row) is None
    assert classify_cell(row) == "genuine"


def test_q43a_high_majority_large_n_is_genuine():
    """Modal share is not the test: 85% majority with ~45 minority on V2 stays."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.85,
        "oracle_acc": 0.853,
        "n_score": 300,
        "ceiling_at_5": 0.69,
        "top_importance_share": 0.12,
    }
    assert row["n_score"] * (1 - row["majority_baseline"]) == pytest.approx(45.0)
    assert type1_reason(row) is None
    assert classify_cell(row) == "genuine"


def test_type1_thin_minority_on_v2_is_unestimable():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.989,
        "oracle_acc": 0.990,
        "n_score": 200,
        "ceiling_at_5": 0.80,
        "top_importance_share": 0.10,
    }
    assert type1_reason(row) == "thin_minority_v2"
    assert classify_cell(row) == "unestimable"


def test_type1_low_ceiling_is_unestimable():
    """Q141 Andorra-style: ceiling@5 ≈ 0.24 is compromised PI."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.50,
        "oracle_acc": 0.55,
        "n_score": 400,
        "ceiling_at_5": 0.24,
        "top_importance_share": 0.10,
    }
    assert type1_reason(row) == "low_ceiling"
    assert classify_cell(row) == "unestimable"


def test_concentrated_leakage_still_drops():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.50,
        "oracle_acc": 0.99,
        "n_score": 400,
        "ceiling_at_5": 0.90,
        "top_importance_share": 0.92,
        "single_feature_acc": 0.97,
    }
    assert classify_cell(row) == "leakage"


def test_distributed_leakage_still_drops():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.55,
        "oracle_acc": 0.99,
        "n_score": 400,
        "ceiling_at_5": 0.90,
        "top_importance_share": 0.12,
        "single_feature_acc": 0.62,
    }
    assert classify_cell(row) == "leakage_distributed"


def test_modal_high_accuracy_is_not_distributed_leakage():
    """Acc ≈ majority (Q141 Germany) is not a skip-pattern module."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.99,
        "oracle_acc": 0.99,
        "n_score": 2000,
        "ceiling_at_5": 0.80,
        "top_importance_share": 0.07,
        "single_feature_acc": 0.99,
    }
    assert classify_cell(row) == "genuine"


def test_zero_lift_does_not_crash_recovery():
    row = {
        "problem_type": "regression",
        "target_type": "ordinal",
        "majority_baseline": 0.40,
        "oracle_acc": 0.40,
        "n_score": 400,
        "ceiling_at_5": 0.80,
        "top_importance_share": 0.20,
        "single_feature_acc": 0.41,
    }
    assert classify_cell(row) == "genuine"


def test_offline_concentration_suspect():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.40,
        "oracle_acc": 0.70,
        "n_score": 400,
        "ceiling_at_5": 0.80,
        "top_importance_share": 0.85,
    }
    assert classify_cell(row) == "leakage_suspect"


def test_type1_thresholds_are_overridable():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.90,
        "n_score": 200,  # V2 minority = 20
        "ceiling_at_5": 0.80,
    }
    assert type1_reason(row) is None
    assert type1_reason(row, ScreenThresholds(min_v2_minority=25)) == "thin_minority_v2"
