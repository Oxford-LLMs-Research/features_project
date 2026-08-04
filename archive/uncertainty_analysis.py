"""
Phase B3 — Uncertainty quantification for the headline metrics.

Adds what every prior summary lacked: confidence intervals. The grid has a nested
structure (survey > target > country > condition x model), so independent-row bootstrap
would understate uncertainty. We therefore use a CLUSTER bootstrap with the cluster =
(survey, target): resample clusters with replacement, pool their rows, recompute the
mean, repeat B times, and take percentile CIs. This propagates the fact that cells
sharing a target are correlated.

Metrics covered (each: mean, 95% CI, n):
  - value_over_random, cost_of_imperfect      (from grid_summary__*.csv)
  - captured_importance, oracle_pctile_mean   (from alignment_by_cell.csv)
  - adaptation_score                          (country_provided rows only)

Every metric is reported THREE ways:
  all            : all valid rows
  excl_leakage   : drop the 6 leakage cells (B1 decision: report both ways)
  genuine_only   : keep only leakage_class == 'genuine' (drops degenerate + leakage;
                   the cleanest set for adaptation, which is noisy on degenerate cells)

Plus a matched-k RANDOM captured-importance baseline: for each cell we draw k random
features from that cell's oracle pool and compute captured importance, averaged over
--rand-draws draws. This anchors the observed captured importance (~0.20) against what a
chance pick of the same size would score, with its own cluster-bootstrap CI.

Outputs:
  outputs/analysis/uncertainty_summary.json
  (TeX: python paper/scripts/write_uncertainty_tex.py)

Run:  python archive/uncertainty_analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from survey_features.config import OUTPUTS_DIR  # noqa: E402

OUT = OUTPUTS_DIR

from survey_features.layout import (  # noqa: E402
    alignment_by_cell_path,
    analysis_write_dir,
    leakage_audit_csv_path,
)
from survey_features.metrics import cluster_bootstrap_ci as _cluster_bootstrap_ci  # noqa: E402

N_BOOT = 2000
RAND_DRAWS = 200
SEED = 42


def load_leakage_classes() -> dict[tuple[str, str], str]:
    p = leakage_audit_csv_path(OUT)
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    return {(str(r["target"]), str(r["country"])): str(r["leakage_class"]) for _, r in df.iterrows()}


def load_target_survey() -> dict[str, str]:
    import yaml
    doc = yaml.safe_load((ROOT / "prelim" / "target_selection_detail.yaml").read_text(encoding="utf-8")) or {}
    out = {}
    for blk in doc.get("surveys") or []:
        for s in blk.get("selected") or []:
            out[str(s["variable"])] = blk.get("survey_id")
    return out


def load_grid() -> pd.DataFrame:
    """All grid_summary rows (both models) with leakage class joined."""
    from survey_features.layout import collect_all_grid_summaries
    leak = load_leakage_classes()
    frames = []
    for p, sid, tag in collect_all_grid_summaries(OUT):
        d = pd.read_csv(p)
        d["survey"] = sid
        d["model"] = tag or "untagged"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in ("value_over_random", "cost_of_imperfect", "oracle_acc", "model_acc", "random_acc"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["leakage_class"] = [leak.get((str(t), str(c)), "unknown")
                           for t, c in zip(df["target"], df["country"])]
    df["model_label"] = df["model"].map(lambda t: t.split("_", 1)[1] if "_" in str(t) else str(t))
    return df


def load_alignment() -> pd.DataFrame:
    df = pd.read_csv(alignment_by_cell_path(OUT))
    if "model_label" not in df.columns:
        df["model_label"] = df["model"].map(lambda t: str(t).split("_", 1)[1] if "_" in str(t) else str(t))
    for c in ("captured_importance", "oracle_pctile_mean", "adaptation_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def cluster_bootstrap_ci(df: pd.DataFrame, col: str, cluster_cols=("survey", "target"),
                         n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Percentile 95% CI for the mean of `col` (single copy: survey_features.metrics)."""
    return _cluster_bootstrap_ci(df, col, cluster_cols=cluster_cols, n_boot=n_boot, seed=seed)


def subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all": df,
        "excl_leakage": df[df["leakage_class"] != "leakage"],
        "genuine_only": df[df["leakage_class"] == "genuine"],
    }


# ---- matched-k random captured-importance baseline ---------------------------

def random_captured_importance(align: pd.DataFrame, draws: int = RAND_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """For each cell, expected captured importance if k features were chosen at RANDOM
    from that cell's oracle pool (avg over `draws`). Same matched-k as the model row."""
    from survey_features.metrics import load_oracle_splits, random_captured_mean

    imp_cache: dict[tuple[str, str], tuple[dict[str, float], dict[str, float]]] = {}
    out = []
    for _, r in align.iterrows():
        k = int(r["k_mapped"]) if pd.notna(r["k_mapped"]) else 0
        key = (str(r["target"]), str(r["country"]))
        if key not in imp_cache:
            imp_cache[key] = load_oracle_splits(*key, OUT)
        rank, score = imp_cache[key]
        cell_seed = int(seed + abs(hash(key)) % 10_000)
        rci = random_captured_mean(score, k, cell_seed, draws=draws, rank=rank)
        row = r.to_dict()
        row["random_captured_importance"] = rci if rci is not None else np.nan
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    grid = load_grid()
    align = load_alignment()
    align_cp = align[align["condition"] == "country_provided"]
    align_rand = random_captured_importance(align)

    summary: dict = {"n_boot": args.n_boot, "cluster": "survey x target", "subsets": {}}

    specs = [
        ("value_over_random", grid, "value_over_random"),
        ("cost_of_imperfect", grid, "cost_of_imperfect"),
        ("captured_importance", align, "captured_importance"),
        ("oracle_pctile_mean", align, "oracle_pctile_mean"),
        ("random_captured_importance", align_rand, "random_captured_importance"),
        ("adaptation_score", align_cp, "adaptation_score"),
    ]
    for label, src, col in specs:
        summary["subsets"][label] = {}
        for sname, sdf in subsets(src).items():
            summary["subsets"][label][sname] = cluster_bootstrap_ci(sdf, col, n_boot=args.n_boot)

    # Headline contrast: is captured importance above its matched-k random baseline?
    # paired per row, then cluster-bootstrap the paired difference.
    paired = align_rand[align_rand["captured_importance"].notna()
                        & align_rand["random_captured_importance"].notna()].copy()
    paired["ci_minus_random"] = paired["captured_importance"] - paired["random_captured_importance"]
    summary["captured_vs_random_paired"] = {
        sname: cluster_bootstrap_ci(sdf, "ci_minus_random", n_boot=args.n_boot)
        for sname, sdf in subsets(paired).items()
    }

    # Per-model CIs (report models side-by-side). Same metrics, split by model_label.
    model_specs = specs + [("ci_minus_random", paired, "ci_minus_random")]
    models = sorted(grid["model_label"].dropna().unique())
    summary["models"] = models
    summary["by_model"] = {}
    for label, src, col in model_specs:
        summary["by_model"][label] = {}
        for mdl in models:
            msrc = src[src["model_label"] == mdl]
            summary["by_model"][label][mdl] = {
                sname: cluster_bootstrap_ci(sdf, col, n_boot=args.n_boot)
                for sname, sdf in subsets(msrc).items()
            }

    out_path = analysis_write_dir(OUT) / "uncertainty_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} (n_boot={args.n_boot})")


if __name__ == "__main__":
    main()
