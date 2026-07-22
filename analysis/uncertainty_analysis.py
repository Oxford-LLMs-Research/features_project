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
  outputs/uncertainty_summary.json
  paper/generated_current_state/main_uncertainty.tex            (--write-tex)
  paper/generated_current_state/main_ci_captured_importance.tex (--write-tex)

Run:  python analysis/uncertainty_analysis.py --write-tex
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
OUT = ROOT / "outputs"

from survey_features.metrics import cluster_bootstrap_ci as _cluster_bootstrap_ci  # noqa: E402

N_BOOT = 2000
RAND_DRAWS = 200
SEED = 42


def load_leakage_classes() -> dict[tuple[str, str], str]:
    p = OUT / "leakage_audit.csv"
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
    df = pd.read_csv(OUT / "alignment_by_cell.csv")
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

def load_oracle_importance(target: str, country: str) -> np.ndarray:
    p = OUT / f"{target}_{country}" / "oracle.csv"
    if not p.is_file():
        return np.array([])
    d = pd.read_csv(p)
    return pd.to_numeric(d["importance_mean"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()


def random_captured_importance(align: pd.DataFrame, draws: int = RAND_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """For each cell, expected captured importance if k features were chosen at RANDOM
    from that cell's oracle pool (avg over `draws`). Same matched-k as the model row."""
    rng = np.random.default_rng(seed)
    imp_cache: dict[tuple[str, str], np.ndarray] = {}
    out = []
    for _, r in align.iterrows():
        k = int(r["k_mapped"]) if pd.notna(r["k_mapped"]) else 0
        key = (str(r["target"]), str(r["country"]))
        if key not in imp_cache:
            imp_cache[key] = load_oracle_importance(*key)
        imp = imp_cache[key]
        rci = np.nan
        if k > 0 and imp.size >= k:
            order = np.sort(imp)[::-1]
            denom = order[:k].sum()
            if denom > 0:
                acc = 0.0
                for _ in range(draws):
                    acc += imp[rng.choice(imp.size, size=k, replace=False)].sum() / denom
                rci = acc / draws
        row = r.to_dict()
        row["random_captured_importance"] = rci
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-tex", action="store_true")
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

    (OUT / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs/uncertainty_summary.json (n_boot={args.n_boot})")

    if args.write_tex:
        write_tex(summary)


def write_tex(summary: dict) -> None:
    gen = ROOT / "paper" / "generated_current_state"
    gen.mkdir(parents=True, exist_ok=True)
    EOL = " " + chr(92) + chr(92)

    def z(x: float) -> str:
        """3 dp, leading zero stripped (0.025 -> .025, -0.002 -> -.002)."""
        s = f"{x:.3f}"
        return s.replace("-0.", "-.").replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s

    def fmt(d):
        if not d or d.get("mean") is None:
            return "-"
        return f"{z(d['mean'])} [{z(d['ci_low'])}, {z(d['ci_high'])}]"

    # Models side-by-side (columns). Rows are metrics on all valid rows, with explicit
    # "excl. leakage" rows for the two metrics leakage actually moves (value-over-random,
    # adaptation). Captured importance is decomposed (model / random-k / delta). The
    # genuine-only subset is reported in prose, not as extra columns, to keep width.
    models = summary["models"]
    bm = summary["by_model"]

    def cells(metric_key, subset):
        return " & ".join(fmt(bm[metric_key][m][subset]) for m in models)

    # (metric_key, row label, subset)
    spec = [
        ("value_over_random",          "Value over random",                 "all"),
        ("value_over_random",          "\\quad excl.\\ leakage",            "excl_leakage"),
        ("cost_of_imperfect",          "Cost of imperfect",                 "all"),
        ("captured_importance",        "Captured importance",               "all"),
        ("random_captured_importance", "\\quad random-$k$ baseline",        "all"),
        ("ci_minus_random",            "\\quad $\\Delta$ (model $-$ random)", "all"),
        ("oracle_pctile_mean",         "Oracle percentile",                 "all"),
        ("adaptation_score",           "Adaptation score (own $-$ cross)",  "all"),
        ("adaptation_score",           "\\quad excl.\\ leakage",            "excl_leakage"),
    ]
    rows = [name + " & " + cells(key, sub) for key, name, sub in spec]
    n_all = summary["subsets"]["value_over_random"]["all"]["n_clusters"]
    tab = (
        "{\\setlength{\\tabcolsep}{6pt}\\renewcommand{\\arraystretch}{1.1}\n"
        "\\begin{tabular}{@{}l" + "c" * len(models) + "@{}}\n\\toprule\n"
        "Metric (mean [95\\% CI]) & " + " & ".join(models) + EOL + "\n\\midrule\n"
        + "\n".join(r + EOL for r in rows)
        + "\n\\bottomrule\n\\end{tabular}}\n"
    )
    (gen / "main_uncertainty.tex").write_text(tab, encoding="utf-8")
    print(f"Wrote uncertainty TeX to {gen} (clusters={n_all})")


if __name__ == "__main__":
    main()
