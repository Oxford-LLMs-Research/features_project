import pandas as pd

import run_grid


def test_run_grid_exports_question_column_cleaner():
    df = pd.DataFrame(
        {
            "COUNTRY": ["A", "A"],
            "RESP_ID": [1, 2],
            "Q1": ["1", "-1"],
            "Q2": ["Agree", "Disagree"],
        }
    )

    cleaned = run_grid.clean_question_columns(
        df,
        country_col="COUNTRY",
        admin_cols=frozenset({"COUNTRY", "RESP_ID"}),
    )

    assert cleaned["Q1"].iloc[0] == 1.0
    assert pd.isna(cleaned["Q1"].iloc[1])
    assert cleaned["Q2"].tolist() == ["Agree", "Disagree"]
    assert cleaned["RESP_ID"].tolist() == [1, 2]
