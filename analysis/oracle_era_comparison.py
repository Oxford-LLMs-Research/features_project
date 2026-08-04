"""
Compare the accuracy-era oracle caches against the current log-loss + honest-split ones.

Answers the questions the audit's oracle change was made to answer:
  * how much resolution did the coarse loss cost (share of features with importance > 0,
    share tied at exactly 0.0)?
  * how stable is the ranking really (select vs score correlation, and the honest
    ceiling: what fraction of the achievable top-k mass a data-driven method captures
    when it must rank on one holdout and be valued on a disjoint one)?
  * how far does the ranking move between eras — i.e. how much of the published
    captured-importance denominator was noise?

Reads outputs/cache/cells_accuracy_v1/*/oracle.csv (archived) against
outputs/cache/cells/*/oracle.csv + oracle_meta.json (current). Pure arithmetic.

Usage:
    python analysis/oracle_era_comparison.py
    python analysis/oracle_era_comparison.py --top-k 10 --csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from survey_features.config import OUTPUTS_DIR  # noqa: E402
from survey_features.metrics import jaccard  # noqa: E402


def collect(outputs_dir: Path, top_k: int) -> pd.DataFrame:
    old_root = outputs_dir / "cache" / "cells_accuracy_v1"
    new_root = outputs_dir / "cache" / "cells"
    rows = []
    for meta_p in sorted(new_root.glob("*/oracle_meta.json")):
        cell = meta_p.parent.name
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            new = pd.read_csv(meta_p.parent / "oracle.csv")
        except Exception:
            continue
        if "importance_score" not in new.columns:
            continue  # not yet re-run under the new contract

        r = {
            "cell": cell,
            "eval_metric": meta.get("eval_metric"),
            "n_features_new": len(new),
            "pos_share_new": float((new.importance_score > 0).mean()),
            "zero_share_new": float((new.importance_score == 0).mean()),
            "ceiling_5": meta["oracle_ceiling"].get("5"),
            "ceiling_10": meta["oracle_ceiling"].get("10"),
            "ceiling_20": meta["oracle_ceiling"].get("20"),
            "n_train": meta.get("n_train"),
            "n_score": meta.get("n_score"),
        }
        # How much do the two independent importance estimates of the SAME features
        # under the SAME fitted model agree? Low values mean the ranking is noise.
        if {"importance_select", "importance_score"} <= set(new.columns) and len(new) > 2:
            rho = spearmanr(new.importance_select, new.importance_score).correlation
            r["select_score_spearman"] = float(rho) if pd.notna(rho) else None
            top_sel = set(new.nlargest(top_k, "importance_select").feature_variable)
            top_sco = set(new.nlargest(top_k, "importance_score").feature_variable)
            r[f"top{top_k}_jaccard_select_vs_score"] = jaccard(top_sel, top_sco)

        old_p = old_root / cell / "oracle.csv"
        if old_p.is_file():
            old = pd.read_csv(old_p)
            r["n_features_old"] = len(old)
            r["pos_share_old"] = float((old.importance_mean > 0).mean())
            r["zero_share_old"] = float((old.importance_mean == 0).mean())
            a = set(old.nlargest(top_k, "importance_mean").feature_variable)
            b = set(new.nlargest(top_k, "importance_score").feature_variable)
            r[f"top{top_k}_jaccard_old_vs_new"] = jaccard(a, b)
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--csv", action="store_true", help="Also write the per-cell table.")
    args = ap.parse_args()

    out_root = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR
    d = collect(out_root, args.top_k)
    if d.empty:
        raise SystemExit(
            "No cells carry the new oracle contract yet.\n"
            "Run: python scripts/rerun_oracles.py"
        )

    k = args.top_k
    n_paired = int(d.get(f"top{k}_jaccard_old_vs_new", pd.Series(dtype=float)).notna().sum())
    print(f"cells with the era-3 oracle : {len(d)}   (paired with an archived era-1 cell: {n_paired})")
    if d.eval_metric.nunique() == 1:
        print(f"eval metric                 : {d.eval_metric.iloc[0]}")

    def line(label, s, pct=False):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return
        f = (lambda v: f"{v:6.1%}") if pct else (lambda v: f"{v:6.3f}")
        print(f"  {label:44s} median {f(s.median())}   mean {f(s.mean())}"
              f"   min {f(s.min())}   max {f(s.max())}")

    print("\n=== RESOLUTION: share of the pool with importance > 0 ===")
    line("accuracy era (archived)", d.get("pos_share_old"), pct=True)
    line("log-loss era (current)", d.get("pos_share_new"), pct=True)
    print("\n=== TIES: share sitting at exactly 0.0 ===")
    print("  (these are what made oracle_percentile_mean order-dependent)")
    line("accuracy era (archived)", d.get("zero_share_old"), pct=True)
    line("log-loss era (current)", d.get("zero_share_new"), pct=True)

    print("\n=== HONEST CEILING: fraction of achievable top-k mass a data-driven ===")
    print("=== ranking captures when selection and scoring use disjoint rows      ===")
    for kk in (5, 10, 20):
        line(f"oracle_ceiling@{kk}", d.get(f"ceiling_{kk}"))
    print("\n  This is the number to report the model against. Captured importance of 0.25")
    print("  against a ceiling of 0.90 and against a ceiling of 0.46 are different claims.")

    print("\n=== RANKING STABILITY (within the current era) ===")
    line("select-vs-score Spearman (full pool)", d.get("select_score_spearman"))
    line(f"top-{k} Jaccard, select vs score", d.get(f"top{k}_jaccard_select_vs_score"))

    if n_paired:
        print("\n=== HOW FAR THE RANKING MOVED BETWEEN ERAS ===")
        line(f"top-{k} Jaccard, accuracy vs log-loss", d.get(f"top{k}_jaccard_old_vs_new"))
        print("\n  Every published captured-importance number used the accuracy-era top-k as")
        print("  its denominator. This is how much of that denominator the metric change")
        print("  replaces.")

    if args.csv:
        p = out_root / "analysis" / "oracle_era_comparison.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(p, index=False)
        print(f"\nwrote -> {p}")


if __name__ == "__main__":
    main()
