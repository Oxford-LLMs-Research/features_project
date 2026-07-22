"""
Metric arithmetic shared by the pipeline and all analysis scripts — the SINGLE copy.

Previously triplicated across alignment_analysis.py, freetext_main_results.py and
format_pilot.py (captured importance) and duplicated across uncertainty_analysis.py and
freetext_main_results.py (cluster bootstrap, random captured baseline).

Definitions:
  captured importance   = sum of oracle importance over the model's mapped codes
                          / sum of oracle importance over the oracle top-k (matched k).
                          In [0, 1]; the design's primary selection metric.
  oracle percentile     = mean oracle-rank percentile of the model's codes
                          (0..1, top=1; a random matched-k pick ~0.5).
  adaptation score      = own-country captured importance - mean cross-country captured
                          importance (computed in the analysis scripts from these parts).
  cluster bootstrap CI  = percentile 95% CI for a column mean, resampling clusters
                          (default cluster = survey x target) with replacement, which
                          propagates the correlation of cells sharing a target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUTS_DIR

N_BOOT = 2000
RAND_DRAWS = 200
SEED = 42


def load_oracle_importance(target: str, country: str, outputs_dir: Path = OUTPUTS_DIR) -> dict[str, float]:
    """{feature_variable: importance_mean} from a cell's cached oracle.csv ({} if absent)."""
    p = outputs_dir / f"{target}_{country}" / "oracle.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    imp = pd.to_numeric(df["importance_mean"], errors="coerce").fillna(0.0)
    return dict(zip(df["feature_variable"].astype(str), imp))


def captured_importance(codes: list[str], imp: dict[str, float], k: int | None = None) -> float | None:
    """Sum oracle importance of mapped codes / sum of oracle top-k (matched k). In [0,1].

    ``codes`` are deduped in arrival order and None/empty entries dropped.
    ``k`` defaults to the number of (deduped) codes; when passed explicitly it sets both
    the code budget and the oracle top-k denominator.
    """
    if not imp:
        return None
    codes = [c for c in dict.fromkeys(codes) if c]
    if k is None:
        k = len(codes)
    if k <= 0:
        return None
    codes = codes[:k]
    ordered = sorted((max(0.0, v) for v in imp.values()), reverse=True)
    denom = sum(ordered[:k])
    if denom <= 0:
        return None
    return sum(max(0.0, imp.get(c, 0.0)) for c in codes) / denom


def captured_importance_df(
    mapped_codes: list[str],
    target: str,
    country_code,
    oracle_df: pd.DataFrame,
    k: int | None = None,
) -> float | None:
    """captured_importance against a long-format oracle table
    [target_variable, country, feature_variable, importance_mean]."""
    sub = oracle_df[(oracle_df["target_variable"] == target) & (oracle_df["country"] == country_code)]
    if sub.empty:
        return None
    imp = dict(zip(sub["feature_variable"].astype(str),
                   sub["importance_mean"].clip(lower=0)))
    return captured_importance(mapped_codes, imp, k=k)


def oracle_percentile_mean(codes: list[str], imp: dict[str, float]) -> float | None:
    """Mean oracle-rank percentile of the model's codes (bottom=0, top=1)."""
    if not imp:
        return None
    codes_asc = [c for c, _ in sorted(imp.items(), key=lambda kv: kv[1])]
    n = len(codes_asc)
    if n <= 1:
        return None
    pos = {c: i / (n - 1) for i, c in enumerate(codes_asc)}
    vals = [pos[c] for c in dict.fromkeys(codes) if c in pos]
    return float(np.mean(vals)) if vals else None


def jaccard(a: set, b: set) -> float | None:
    u = a | b
    return (len(a & b) / len(u)) if u else None


def random_captured_mean(imp: dict[str, float], k: int, seed: int,
                         draws: int = RAND_DRAWS) -> float | None:
    """Mean captured importance of `draws` random k-subsets of the cell's oracle pool."""
    pool = list(imp.keys())
    if not imp or k <= 0 or len(pool) < k:
        return None
    ordered = sorted((max(0.0, v) for v in imp.values()), reverse=True)
    denom = sum(ordered[:k])
    if denom <= 0:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        pick = rng.choice(len(pool), size=k, replace=False)
        vals.append(sum(max(0.0, imp[pool[i]]) for i in pick) / denom)
    return float(np.mean(vals))


def cluster_bootstrap_ci(df: pd.DataFrame, col: str, cluster_cols=("survey", "target"),
                         n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Percentile 95% CI for the mean of `col`, resampling clusters with replacement."""
    sub = df[df[col].notna()].copy()
    if sub.empty:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_clusters": 0}
    sub["_cl"] = list(zip(*[sub[c].astype(str) for c in cluster_cols]))
    groups = {kk: g[col].to_numpy() for kk, g in sub.groupby("_cl")}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n_cl = len(keys)
    for b in range(n_boot):
        pick = rng.integers(0, n_cl, size=n_cl)
        means[b] = np.concatenate([groups[keys[i]] for i in pick]).mean()
    return {
        "mean": round(float(sub[col].mean()), 4),
        "ci_low": round(float(np.percentile(means, 2.5)), 4),
        "ci_high": round(float(np.percentile(means, 97.5)), 4),
        "n": int(len(sub)),
        "n_clusters": int(n_cl),
    }
