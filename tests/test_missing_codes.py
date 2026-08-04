"""Tests for the respondent-vs-structural missing-code split.

"Don't know" and "Refused" are answers the respondent chose to give and are kept as
levels; "Not asked" / "Not applicable" / "Missing" are artefacts of routing, instrument
design or fieldwork and become NaN. The classification is label-driven, and the labels
are written by seven different survey houses, so the boundary cases are the point.
"""

import numpy as np
import pandas as pd
import pytest

from survey_features.surveys import (
    RESPONDENT,
    STRUCTURAL,
    SUBSTANTIVE,
    classify_missing_codes,
    classify_missing_label,
    clean_question_columns,
)


@pytest.mark.parametrize("label", [
    "Don't know", "Don’t know", "Do not know", "No sabe",
    "Refused", "Refusal", "Refused to answer", "Refuse to answer",
    "Decline to answer", "Can't choose", "Do not understand",
    "Don't know/Haven't heard enough", "Don't know (if used in instrument)",
])
def test_respondent_labels(label):
    assert classify_missing_label(label)[0] == RESPONDENT


@pytest.mark.parametrize("label", [
    "Missing", "Not asked", "Not asked in this country", "Not applicable",
    "Not applicable (Not first choice)", "INAP; Father does not live in country",
    "Missing; Not specified", "Item not included", "Not available", "No data",
])
def test_structural_labels(label):
    assert classify_missing_label(label)[0] == STRUCTURAL


@pytest.mark.parametrize("label", [
    # Real labels that substring matching wrongly captured before segment-boundary
    # matching was introduced. Each is a substantive response option.
    "Neither appropriate nor inappropriate",   # matched "inap"
    "Absolutely inappropriate",                # matched "inap"
    "Somewhat inappropriate",                  # matched "inap"
    "Invalid ballot",                          # matched "invalid" -> would become NaN
    "Invalid vote",
    "Agnostic (Do not know if there is a God)",  # matched "do not know"
    "I don't know how to get the vaccine",
    "I don't know much about their political views",
    "I do not know what to do",
    "Very much",
])
def test_substantive_labels_are_not_swallowed(label):
    assert classify_missing_label(label)[0] == SUBSTANTIVE


def test_ambiguous_defaults_to_respondent_and_is_flagged():
    kind, ambiguous = classify_missing_label("No answer")
    assert (kind, ambiguous) == (RESPONDENT, True)


def test_compound_label_resolves_to_the_reading_that_keeps_data():
    """"No data / No answer": structural on one side, respondent on the other."""
    assert classify_missing_label("No data / No answer")[0] == RESPONDENT


# ── clean_question_columns ────────────────────────────────────────────────────

def _wvs_like_metadata():
    """WVS negative-code convention: -1 DK, -2 No answer, -3/-4/-5 structural."""
    return {
        "SECTION": {
            "Q1": {
                "question": "example",
                "values": {
                    "1": "Agree", "2": "Disagree",
                    "-1": "Don't know", "-2": "No answer",
                    "-3": "Not applicable", "-4": "Not asked", "-5": "Missing",
                },
            }
        }
    }


def _frame():
    return pd.DataFrame({
        "Q1": [1, 2, -1, -2, -3, -4, -5],
        "cntry": ["X"] * 7,
    })


def test_respondent_codes_survive_cleaning_at_their_original_values():
    out = clean_question_columns(_frame(), "cntry", frozenset(), _wvs_like_metadata())
    kept = out["Q1"].dropna().tolist()
    assert sorted(kept) == [-2.0, -1.0, 1.0, 2.0]


def test_structural_codes_become_nan():
    out = clean_question_columns(_frame(), "cntry", frozenset(), _wvs_like_metadata())
    assert out["Q1"].isna().sum() == 3  # -3, -4, -5


def test_legacy_mode_drops_both_kinds():
    out = clean_question_columns(
        _frame(), "cntry", frozenset(), _wvs_like_metadata(),
        keep_respondent_missing=False,
    )
    assert sorted(out["Q1"].dropna().tolist()) == [1.0, 2.0]


def test_without_metadata_all_negatives_are_dropped():
    """The conservative fallback: no labels means no way to tell a DK from a sentinel."""
    out = clean_question_columns(_frame(), "cntry", frozenset())
    assert sorted(out["Q1"].dropna().tolist()) == [1.0, 2.0]


def test_unlabelled_column_keeps_the_negatives_rule():
    df = pd.DataFrame({"Q1": [1, -1], "Q2": [5, -9], "cntry": ["X", "X"]})
    out = clean_question_columns(df, "cntry", frozenset(), _wvs_like_metadata())
    assert out["Q1"].dropna().tolist() == [1.0, -1.0]   # labelled: DK kept
    assert out["Q2"].dropna().tolist() == [5.0]         # unlabelled: negative dropped


def test_admin_and_country_columns_are_untouched():
    df = pd.DataFrame({"Q1": [1, -3], "weight": [-0.5, 2.0], "cntry": ["X", "Y"]})
    out = clean_question_columns(df, "cntry", frozenset({"weight"}), _wvs_like_metadata())
    assert out["weight"].tolist() == [-0.5, 2.0]
    assert out["cntry"].tolist() == ["X", "Y"]


def test_classify_missing_codes_buckets_by_variable():
    buckets = classify_missing_codes(_wvs_like_metadata())["Q1"]
    assert -1 in buckets[RESPONDENT] and -2 in buckets[RESPONDENT]
    assert -3 in buckets[STRUCTURAL] and -5 in buckets[STRUCTURAL]
    assert 1 not in buckets[RESPONDENT] and 1 not in buckets[STRUCTURAL]


def test_text_columns_drop_structural_labels_only():
    df = pd.DataFrame({
        "Q1": ["Agree", "Don't know", "Not asked", "Disagree"],
        "cntry": ["X"] * 4,
    })
    meta = {"S": {"Q1": {"values": {
        "Agree": "Agree", "Disagree": "Disagree",
        "Don't know": "Don't know", "Not asked": "Not asked",
    }}}}
    out = clean_question_columns(df, "cntry", frozenset(), meta)
    assert out["Q1"].tolist()[:2] == ["Agree", "Don't know"]
    assert pd.isna(out["Q1"].iloc[2])
