"""Oracle AutoGluon bag: FASTAI is excluded on every fit."""

from survey_features.oracle import EXCLUDED_MODEL_TYPES, _assemble_outputs_cv

import pandas as pd


def test_fastai_is_excluded():
    assert EXCLUDED_MODEL_TYPES == ("FASTAI",)


def test_oracle_meta_records_excluded_models():
    select = pd.DataFrame({
        "feature_variable": ["a", "b"],
        "importance_select": [0.2, 0.1],
        "importance_select_std": [0.0, 0.0],
        "importance_select_fold1": [0.2, 0.1],
    })
    score = pd.DataFrame({
        "feature_variable": ["a", "b"],
        "importance_score": [0.2, 0.1],
        "importance_score_std": [0.0, 0.0],
        "importance_score_fold1": [0.2, 0.1],
    })
    y = pd.Series([0, 1, 0, 1], index=[10, 11, 12, 13])
    _, _, meta = _assemble_outputs_cv(
        select, score,
        target_var="T", country_code="X",
        majority_baseline=0.5, resolved_metric="log_loss",
        problem_type="binary", ttype="binary",
        cv_folds=2, fold_sizes=[2, 2],
        n_eval=1, n_score=1, y=y, eval_idx=[0],
    )
    assert meta["excluded_model_types"] == ["FASTAI"]
    assert meta["contract_version"] == 4
