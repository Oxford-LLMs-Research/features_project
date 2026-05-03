"""
Phase 0b: Oracle permutation importances.

Computes XGBoost-based permutation importance for a single (target, country) cell.
This is the ground-truth feature ranking that LLM selection is evaluated against.

Output contract
---------------
compute_oracle() returns:
  oracle_df  — DataFrame with columns:
                 target_variable, country, feature_variable,
                 importance_mean, importance_std, majority_baseline
  feature_pool — list of all feature variable codes included in the model
                 (used by run_grid.py to build the random-draw baseline)

Cache contract
--------------
run_grid.py writes oracle_df to outputs/<target>_<country>/oracle.csv and
reloads it on subsequent runs. To plug in a pre-computed oracle (e.g. from
a different method or tuned hyperparameters), place a CSV with the columns
above at that path before running the pipeline — compute_oracle() will be
skipped automatically.

Hyperparameters
---------------
n_splits    — CV folds for permutation importance (default 5)
n_repeats   — shuffle repeats per feature per fold (default 10)
random_state — global seed (default 42)

These are passed per cell from run_grid.py. Per-cell tuning is intentional:
optimal values vary by question complexity and sample size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_question_columns(
    df: pd.DataFrame,
    country_col: str,
    admin_cols: frozenset[str],
) -> pd.DataFrame:
    """
    Coerce numeric-coded columns to float and map negative values (missing codes) to NaN.
    Genuine text-label columns are left intact for label encoding downstream.
    """
    cleaned = df.copy()
    q_cols = [c for c in cleaned.columns if c not in admin_cols and c != country_col]
    for col in q_cols:
        if not pd.api.types.is_object_dtype(cleaned[col]):
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            cleaned[col] = cleaned[col].where(cleaned[col] >= 0)
        else:
            coerced = pd.to_numeric(cleaned[col], errors="coerce")
            if coerced.notna().mean() > 0.5:
                cleaned[col] = coerced.where(coerced >= 0)
    return cleaned


_clean_question_columns = clean_question_columns


def compute_oracle(
    data: pd.DataFrame,
    metadata: dict,
    target_var: str,
    country_code: int | str,
    country_col: str,
    admin_cols: frozenset[str],
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute permutation importances for target_var in country_code.

    Parameters
    ----------
    data        : full survey DataFrame (all countries)
    metadata    : survey metadata dict (section → {var_code → info})
    target_var  : variable code of the outcome to predict
    country_code: value of country_col to filter on
    country_col : name of the country column in data
    admin_cols  : columns to exclude from the feature pool
    n_splits    : CV folds
    n_repeats   : permutation repeats per feature per fold
    random_state: random seed

    Returns
    -------
    oracle_df    : DataFrame — one row per feature variable, columns:
                   target_variable, country, feature_variable,
                   importance_mean, importance_std, majority_baseline
    feature_pool : list of feature variable codes in the model
    """
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold

    def make_clf():
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=random_state, eval_metric="mlogloss", verbosity=0,
        )

    country_data = data[data[country_col] == country_code].copy()
    print(f"  n (before dropping missing target): {len(country_data)}")
    if len(country_data) == 0:
        raise ValueError(
            f"No rows found for {country_col}={country_code!r}. "
            f"Actual values in data: {sorted(data[country_col].dropna().unique().tolist())}"
        )

    country_data = clean_question_columns(country_data, country_col, admin_cols)

    if country_data.columns.duplicated().any():
        country_data = country_data.loc[:, ~country_data.columns.duplicated()]

    valid = country_data[target_var].notna()
    country_data = country_data.loc[valid]
    if len(country_data) == 0:
        raise ValueError(
            f"No valid (non-missing) rows for target '{target_var}' "
            f"in {country_col}={country_code!r}."
        )
    y_raw = country_data[target_var]
    print(f"  n (after dropping missing target): {len(country_data)}")

    feat_cols = [
        c for c in country_data.columns
        if c not in admin_cols
        and c != target_var
        and c != country_col
        and (pd.api.types.is_numeric_dtype(country_data[c])
             or pd.api.types.is_object_dtype(country_data[c]))
    ]

    X = country_data[feat_cols].copy()

    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        print(f"  Dropped {len(all_missing)} fully-missing features")
        X = X.drop(columns=all_missing)

    text_cols = [c for c in X.columns if pd.api.types.is_object_dtype(X[c])]
    if text_cols:
        print(f"  Label-encoding {len(text_cols)} text column(s)")
        for col in text_cols:
            col_le = LabelEncoder()
            non_null = X[col].notna()
            X.loc[non_null, col] = col_le.fit_transform(X.loc[non_null, col])
            X[col] = pd.to_numeric(X[col], errors="coerce")

    print(f"  Feature pool: {X.shape[1]} variables")

    le = LabelEncoder()
    try:
        y = pd.Series(le.fit_transform(y_raw.astype(int)), index=y_raw.index)
    except (ValueError, TypeError):
        print(f"  Target '{target_var}' is text-coded; applying label encoding")
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

    class_counts = y.value_counts()
    rare = class_counts[class_counts < 5].index.tolist()
    if rare:
        keep = ~y.isin(rare)
        X, y_raw = X.loc[keep], y_raw.loc[keep]
        le = LabelEncoder()
        try:
            y = pd.Series(le.fit_transform(y_raw.astype(int)), index=y_raw.index)
        except (ValueError, TypeError):
            y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)

    majority_pct = y.value_counts().max() / len(y)
    print(f"  Classes: {len(le.classes_)}, majority baseline: {majority_pct:.3f}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_importances = []
    rng = np.random.default_rng(random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        clf = make_clf()
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        X_test_np = X.iloc[test_idx].to_numpy(copy=True)
        y_test_np = y.iloc[test_idx].to_numpy()
        baseline = accuracy_score(y_test_np, clf.predict(X_test_np))

        importances = np.zeros(X_test_np.shape[1], dtype=float)
        n_test = X_test_np.shape[0]
        for j in range(X_test_np.shape[1]):
            drops = []
            original_col = X_test_np[:, j].copy()
            for _ in range(n_repeats):
                X_test_np[:, j] = original_col[rng.permutation(n_test)]
                perm_acc = accuracy_score(y_test_np, clf.predict(X_test_np))
                drops.append(baseline - perm_acc)
            X_test_np[:, j] = original_col
            importances[j] = float(np.mean(drops))

        fold_importances.append(importances)
        print(f"    fold {fold_idx+1}/{n_splits} done")

    mean_imp = np.mean(fold_importances, axis=0)
    std_imp = np.std(fold_importances, axis=0)

    top5 = np.argsort(mean_imp)[::-1][:5]
    print("  Top 5 features by permutation importance:")
    for i in top5:
        feat = X.columns[i]
        desc = ""
        for section in metadata.values():
            if feat in section:
                desc = (section[feat].get("description") or "")[:50]
                break
        print(f"    {feat:8s}  {mean_imp[i]:.4f} ± {std_imp[i]:.4f}  ({desc})")

    records = [
        {
            "target_variable": target_var,
            "country": country_code,
            "feature_variable": feat,
            "importance_mean": float(mean_imp[i]),
            "importance_std": float(std_imp[i]),
            "majority_baseline": round(float(majority_pct), 4),
        }
        for i, feat in enumerate(X.columns)
    ]
    return pd.DataFrame(records), list(X.columns)
