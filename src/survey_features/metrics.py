"""
Metric arithmetic shared by the pipeline and all analysis scripts — the SINGLE copy.

Previously triplicated across alignment_analysis.py, freetext_main_results.py and
format_pilot.py (captured importance) and duplicated across uncertainty_analysis.py and
freetext_main_results.py (cluster bootstrap, random captured baseline).

Definitions:
  captured importance   = sum of SCORE-split oracle importance over the model's codes
                          / sum of SCORE-split importance over the oracle top-k chosen
                          on the SELECT split (matched k). In [0, 1].
                          When only one importance dict is available (legacy CSV),
                          rank and score are the same column.
  oracle percentile     = mean oracle-rank percentile of the model's codes
                          (0..1, top=1; a random matched-k pick ~0.5). Prefer the
                          select-split ranks when both columns exist.
  adaptation score      = own-country captured importance - mean cross-country captured
                          importance (computed in the analysis scripts from these parts).
  cluster bootstrap CI  = percentile 95% CI for a column mean, resampling clusters
                          (default cluster = survey x target) with replacement, which
                          propagates the correlation of cells sharing a target.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import OUTPUTS_DIR
from .layout import cell_dir, oracle_csv_path

N_BOOT = 2000
RAND_DRAWS = 200
SEED = 42


def oracle_topk_codes(rank: dict[str, float], k: int) -> list[str]:
    """Top-k feature codes by a ranking dict (select split, or legacy single column)."""
    if k <= 0 or not rank:
        return []
    return sorted(rank.keys(), key=lambda c: rank[c], reverse=True)[:k]


def _series_to_imp(df: pd.DataFrame, col: str) -> dict[str, float]:
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return dict(zip(df["feature_variable"].astype(str), vals.astype(float)))


def rank_score_from_frame(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """(rank, score) dicts from an oracle.csv-like frame. Legacy CSVs share one column."""
    if df.empty or "feature_variable" not in df.columns:
        return {}, {}
    if "importance_score" in df.columns:
        score = _series_to_imp(df, "importance_score")
    elif "importance_mean" in df.columns:
        score = _series_to_imp(df, "importance_mean")
    else:
        return {}, {}
    if "importance_select" in df.columns:
        rank = _series_to_imp(df, "importance_select")
    else:
        rank = score
    return rank, score


def load_oracle_splits(
    target: str, country: str, outputs_dir: Path = OUTPUTS_DIR,
) -> tuple[dict[str, float], dict[str, float]]:
    """(rank, score) from a cell's oracle.csv. Empty dicts if the file is absent."""
    p = oracle_csv_path(target, country, outputs_dir)
    if not p.is_file():
        return {}, {}
    return rank_score_from_frame(pd.read_csv(p))


def load_oracle_importance(target: str, country: str, outputs_dir: Path = OUTPUTS_DIR) -> dict[str, float]:
    """{feature_variable: score-split importance} (aliases importance_mean)."""
    _, score = load_oracle_splits(target, country, outputs_dir)
    return score


def load_oracle_train_index(
    target: str, country: str, outputs_dir: Path = OUTPUTS_DIR,
) -> list | None:
    """Fit-split row labels from oracle_meta.json, or None if absent / empty."""
    p = cell_dir(target, country, outputs_dir) / "oracle_meta.json"
    if not p.is_file():
        return None
    meta = json.loads(p.read_text(encoding="utf-8"))
    idx = meta.get("train_index")
    return list(idx) if idx else None


def captured_importance(
    codes: list[str],
    imp: dict[str, float],
    k: int | None = None,
    *,
    rank: dict[str, float] | None = None,
) -> float | None:
    """Sum score importance of mapped codes / score mass of select-ranked top-k. In [0,1].

    ``imp`` is the SCORE split (values). ``rank`` is the SELECT split used only to choose
    the denominator's top-k; defaults to ``imp`` for legacy single-column readers.
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
    rank_map = rank if rank is not None else imp
    denom_keys = oracle_topk_codes(rank_map, k)
    denom = sum(max(0.0, imp.get(c, 0.0)) for c in denom_keys)
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
    """captured_importance against a long-format oracle table (select→score when present)."""
    sub = oracle_df[(oracle_df["target_variable"] == target) & (oracle_df["country"] == country_code)]
    if sub.empty:
        return None
    rank, score = rank_score_from_frame(sub)
    return captured_importance(mapped_codes, score, k=k, rank=rank)


def oracle_percentile_mean(codes: list[str], imp: dict[str, float]) -> float | None:
    """Mean oracle-rank percentile of the model's codes (bottom=0, top=1).

    Pass the SELECT-split ranks when available. Ties take the AVERAGE rank —
    order-dependent tie-breaking once biased this metric (pipeline_audit_2026-08.md #A3).
    """
    if not imp:
        return None
    n = len(imp)
    if n <= 1:
        return None
    keys = list(imp)
    ranks = rankdata([imp[k] for k in keys], method="average")  # 1..n, ties averaged
    pos = {k: (r - 1.0) / (n - 1) for k, r in zip(keys, ranks)}
    vals = [pos[c] for c in dict.fromkeys(codes) if c in pos]
    return float(np.mean(vals)) if vals else None


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from arbitrary key parts.

    `hash()` on strings is randomized per process; this digest is not (audit #A3).
    """
    key = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.blake2b(key, digest_size=4).hexdigest(), 16) % (2**31)


def jaccard(a: set, b: set) -> float | None:
    u = a | b
    return (len(a & b) / len(u)) if u else None


def random_captured_mean(
    imp: dict[str, float],
    k: int,
    seed: int,
    draws: int = RAND_DRAWS,
    *,
    rank: dict[str, float] | None = None,
) -> float | None:
    """Mean captured importance of `draws` random k-subsets of the cell's oracle pool.

    Denominator top-k is chosen on ``rank`` (select) and valued on ``imp`` (score).
    """
    pool = list(imp.keys())
    if not imp or k <= 0 or len(pool) < k:
        return None
    rank_map = rank if rank is not None else imp
    denom = sum(max(0.0, imp.get(c, 0.0)) for c in oracle_topk_codes(rank_map, k))
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
