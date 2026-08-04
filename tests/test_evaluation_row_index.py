"""Tests for evaluate_feature_set row_index (oracle train_index) filtering."""

import numpy as np
import pandas as pd
import pytest

from survey_features.evaluation import _normalize_row_label, _row_index_mask, evaluate_feature_set


def test_normalize_int_and_str_labels_match():
    assert _normalize_row_label(3) == _normalize_row_label("3")
    assert _normalize_row_label(3.0) == _normalize_row_label(3)


def test_row_index_mask_matches_int_and_str():
    idx = pd.Index([10, 11, 12, 13])
    mask = _row_index_mask(idx, ["10", 12])
    assert mask.tolist() == [True, False, True, False]


def test_evaluate_feature_set_rejects_thin_train_index():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "y": rng.integers(0, 2, size=n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    out = evaluate_feature_set(df, "y", ["f1", "f2"], row_index=[0, 1], n_splits=5)
    assert out["error"] == "train_index too thin for CV"


def test_evaluate_feature_set_restricts_to_row_index():
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame({
        "y": np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)]),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    # Keep half the rows via train_index; n_samples should reflect the filter.
    keep = list(range(0, n, 2))
    out = evaluate_feature_set(df, "y", ["f1", "f2"], row_index=keep, n_splits=5, nthread=1)
    assert out.get("error") is None
    assert out["n_samples"] == len(keep)
