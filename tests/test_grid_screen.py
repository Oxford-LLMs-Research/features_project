"""Confirmatory grid screen: keep type-2/3 signal; drop type-1 and leakage.

Rows use the typed schema: oracle_primary / single_feature_primary are in the
cell's own metric — accuracy for binary/nominal, Spearman rho for ordinal/continuous.
"""

import pytest

from survey_features.grid_screen import (
    ScreenThresholds,
    classify_cell,
    rank_holdout_size,
    type1_reason,
)


def test_type2_tiny_accuracy_lift_is_genuine():
    """Q104-style: oracle beats the mode by < 0.03 — still keep."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.60,
        "oracle_primary": 0.629,
        "n_score": 200,
        "ceiling_at_5": 0.70,
        "top_importance_share": 0.20,
        "single_feature_primary": 0.61,
    }
    assert type1_reason(row) is None
    assert classify_cell(row) == "genuine"


def test_type3_low_rho_ordinal_is_genuine():
    """P16ST-style ordinal: modest Spearman PI — keep; majority plays no role."""
    row = {
        "problem_type": "regression",
        "target_type": "ordinal",
        "majority_baseline": 0.22,
        "oracle_primary": 0.30,
        "n_score": 400,
        "n_target_unique": 11,
        "ceiling_at_5": 0.85,
        "top_importance_share": 0.15,
        "single_feature_primary": 0.12,
    }
    assert type1_reason(row) is None
    assert classify_cell(row) == "genuine"


def test_q43a_high_majority_large_n_is_genuine():
    """Modal share is not the test: 85% majority with ~45 minority on V2 stays."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.85,
        "oracle_primary": 0.853,
        "n_score": 300,
        "ceiling_at_5": 0.69,
        "top_importance_share": 0.12,
        "single_feature_primary": 0.85,
    }
    assert row["n_score"] * (1 - row["majority_baseline"]) == pytest.approx(45.0)
    assert type1_reason(row) is None
    assert classify_cell(row) == "genuine"


def test_type1_thin_minority_on_v2_is_unestimable():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.989,
        "oracle_primary": 0.990,
        "n_score": 200,
        "ceiling_at_5": 0.80,
        "top_importance_share": 0.10,
    }
    assert type1_reason(row) == "thin_minority_v2"
    assert classify_cell(row) == "unestimable"


def test_type1_thin_minority_on_rank_holdout():
    """v4 CV folds: fold holdout minority too thin even when V2 passes."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.90,
        "n_score": 300,               # V2 minority = 30, fine
        "cv_folds": 5,
        "fold_fit_sizes": [200, 200, 200, 200, 200],  # holdout = 200/4 = 50 -> minority 5
        "ceiling_at_5": 0.80,
    }
    assert rank_holdout_size(row) == pytest.approx(50.0)
    assert type1_reason(row) == "thin_minority_rank"


def test_type1_low_ceiling_is_unestimable():
    """Q141 Andorra-style: ceiling@5 ≈ 0.24 is compromised PI."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.50,
        "oracle_primary": 0.55,
        "n_score": 400,
        "ceiling_at_5": 0.24,
        "top_importance_share": 0.10,
    }
    assert type1_reason(row) == "low_ceiling"
    assert classify_cell(row) == "unestimable"


def test_type1_regression_thin_score():
    row = {
        "problem_type": "regression",
        "target_type": "ordinal",
        "n_score": 40,
        "n_target_unique": 5,
        "ceiling_at_5": 0.80,
    }
    assert type1_reason(row) == "thin_score_regression"


def test_type1_regression_too_few_scale_points():
    row = {
        "problem_type": "regression",
        "target_type": "ordinal",
        "n_score": 400,
        "n_target_unique": 2,
        "ceiling_at_5": 0.80,
    }
    assert type1_reason(row) == "too_few_scale_points"


def test_se7a_absolute_near_duplicate_drops():
    """SE7a Mongolia: one column is the target in disguise, oracle lift ~0.
    The absolute rule must fire without any oracle-side lift."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.4438,
        "oracle_primary": 0.4419,
        "n_score": 400,
        "ceiling_at_5": 0.90,
        "top_importance_share": 1.0,
        "single_feature_primary": 0.9945,
    }
    assert classify_cell(row) == "leakage"


def test_regression_absolute_near_duplicate_drops():
    row = {
        "problem_type": "regression",
        "target_type": "ordinal",
        "n_score": 400,
        "n_target_unique": 8,
        "ceiling_at_5": 0.90,
        "top_importance_share": 0.90,
        "oracle_primary": 0.30,
        "single_feature_primary": 0.92,
    }
    assert classify_cell(row) == "leakage"


def test_concentrated_leakage_still_drops():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.50,
        "oracle_primary": 0.99,
        "n_score": 400,
        "ceiling_at_5": 0.90,
        "top_importance_share": 0.92,
        "single_feature_primary": 0.97,
    }
    assert classify_cell(row) == "leakage"


def test_regression_relative_recovery_drops():
    row = {
        "problem_type": "regression",
        "target_type": "ordinal",
        "n_score": 400,
        "n_target_unique": 8,
        "ceiling_at_5": 0.90,
        "top_importance_share": 0.85,
        "oracle_primary": 0.60,
        "single_feature_primary": 0.57,
    }
    assert classify_cell(row) == "leakage"
    assert row["single_feature_recovery"] == pytest.approx(0.95)


def test_regression_recovery_guard_on_tiny_oracle_rho():
    """oracle rho below the guard: the recovery ratio is noise, not leakage."""
    row = {
        "problem_type": "regression",
        "target_type": "ordinal",
        "n_score": 400,
        "n_target_unique": 8,
        "ceiling_at_5": 0.90,
        "top_importance_share": 0.85,
        "oracle_primary": 0.03,
        "single_feature_primary": 0.04,
    }
    assert classify_cell(row) == "genuine"


def test_distributed_leakage_still_drops():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.55,
        "oracle_primary": 0.99,
        "n_score": 400,
        "ceiling_at_5": 0.90,
        "top_importance_share": 0.12,
        "single_feature_primary": 0.62,
    }
    assert classify_cell(row) == "leakage_distributed"


def test_regression_distributed_leakage_drops():
    """No attitude item rank-predicts at rho >= 0.95 — skip-pattern module."""
    row = {
        "problem_type": "regression",
        "target_type": "ordinal",
        "n_score": 400,
        "n_target_unique": 8,
        "ceiling_at_5": 0.90,
        "top_importance_share": 0.15,
        "oracle_primary": 0.97,
        "single_feature_primary": 0.40,
    }
    assert classify_cell(row) == "leakage_distributed"


def test_modal_high_accuracy_is_not_distributed_leakage():
    """Acc ≈ majority (Q141 Germany) is not a skip-pattern module."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.99,
        "oracle_primary": 0.99,
        "n_score": 2000,
        "ceiling_at_5": 0.80,
        "top_importance_share": 0.07,
        "single_feature_primary": 0.99,
    }
    assert classify_cell(row) == "genuine"


def test_zero_lift_does_not_crash_recovery():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.40,
        "oracle_primary": 0.40,
        "n_score": 400,
        "ceiling_at_5": 0.80,
        "top_importance_share": 0.20,
        "single_feature_primary": 0.41,
    }
    assert classify_cell(row) == "genuine"


def test_no_probe_concentrated_degrades_to_suspect():
    """Offline mode / failed probe: a concentrated cell is never cleared to genuine."""
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.40,
        "oracle_primary": 0.70,
        "n_score": 400,
        "ceiling_at_5": 0.80,
        "top_importance_share": 0.85,
    }
    assert classify_cell(row) == "leakage_suspect"


def test_no_probe_spread_importance_is_genuine():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.40,
        "oracle_primary": 0.70,
        "n_score": 400,
        "ceiling_at_5": 0.80,
        "top_importance_share": 0.30,
    }
    assert classify_cell(row) == "genuine"


def test_type1_thresholds_are_overridable():
    row = {
        "problem_type": "binary",
        "majority_baseline": 0.90,
        "n_score": 200,  # V2 minority = 20
        "ceiling_at_5": 0.80,
    }
    assert type1_reason(row) is None
    assert type1_reason(row, ScreenThresholds(min_v2_minority=25)) == "thin_minority_v2"
