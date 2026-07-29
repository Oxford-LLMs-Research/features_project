"""
Phase B1 — Leakage audit for the oracle ground truth.

Problem: some targets are demographics (e.g. WVS Q263 "are you an immigrant?",
ESS rtrd "were you retired?") whose oracle top feature is a near-duplicate of the
target itself (Q266 "in which country were you born?"; mnactic "main activity").
On these cells the oracle reaches ~1.0 accuracy by exploiting a single
label-in-disguise column. That is empirical leakage, NOT predictive understanding,
and it inflates cost_of_imperfect. The oracle's own semantic filter (cosine > 0.85
on question text) does not catch it because the wordings are not lexically similar.

This script flags such cells with two complementary, code-grounded signals:

  (A) Importance concentration (offline, always available):
      share of total positive permutation-importance mass held by the top oracle
      feature, plus oracle lift over majority. A cell where one feature owns almost
      all the mass AND oracle is near-ceiling is a leakage suspect.

  (B) Single-feature predictive test (requires DATA_CONFIG_PATH):
      train the SAME downstream XGBoost on ONLY the top oracle feature and measure
      accuracy. If one feature alone recovers ~the full oracle accuracy and sits far
      above majority, the target is (near-)deterministically recoverable from a
      single column -> leakage.

  (C) Question-text proximity (metadata, when resolvable):
      cosine similarity between target and top-feature question text under the same
      embedding model the pipeline uses. High similarity = lexical duplicate; the
      interesting leakage cases are LOW lexical similarity but HIGH predictive
      recoverability (semantic/empirical duplicates the cosine filter missed).

Each cell is classified:
  - degenerate       : oracle lift over majority < --min-signal (nothing to predict)
  - leakage          : single-feature acc recovers >= --recover-frac of oracle lift
                       AND top-importance share >= --conc-thresh
  - leakage_suspect  : (offline only) high concentration + high oracle lift, but the
                       data-backed single-feature test was not run
  - genuine          : real, distributed predictive structure

Outputs:
  outputs/cache/audits/leakage_audit.csv         one row per unique (survey, target, country)
  outputs/cache/audits/leakage_audit_summary.json rollups by survey / bucket / class
  (legacy: outputs/leakage_audit.csv at root — still read via dual-resolve)
  paper/generated_current_state/leakage_audit_longtable.tex (if --write-tex)

Run (offline, concentration only):
  python scripts/leakage_audit.py
Run (full, with single-feature predictive test):
  python scripts/leakage_audit.py --with-data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"

from survey_features.layout import (  # noqa: E402
    cache_cells_dir,
    leakage_audit_write_paths,
)


def load_target_detail() -> dict[str, dict]:
    import yaml

    doc = yaml.safe_load((ROOT / "prelim" / "target_selection_detail.yaml").read_text(encoding="utf-8")) or {}
    out: dict[str, dict] = {}
    for blk in doc.get("surveys") or []:
        sid = blk.get("survey_id")
        for s in blk.get("selected") or []:
            out[str(s["variable"])] = {
                "survey": sid,
                "bucket": s.get("bucket"),
                "section": s.get("section"),
                "topic_key": s.get("topic_key"),
            }
    return out


def survey_of(target: str, detail: dict[str, dict]) -> str | None:
    info = detail.get(target)
    return info["survey"] if info else None


def iter_oracle_cells() -> list[tuple[str, str, Path]]:
    """Return (target, country, oracle_csv_path) for every cell, resolving target
    via known target codes (targets may contain underscores)."""
    detail = load_target_detail()
    known = sorted(detail.keys(), key=len, reverse=True)
    cells = []
    seen: set[tuple[str, str]] = set()
    search_roots = [
        cache_cells_dir(OUT),
        OUT,
        OUT / "grid" / "cells",
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*/oracle.csv")):
            cell = p.parent.name
            target = next((t for t in known if cell.startswith(t + "_")), None)
            if target is None:
                continue
            country = cell[len(target) + 1 :]
            key = (target, country)
            if key in seen:
                continue
            seen.add(key)
            cells.append((target, country, p))
    return cells


def concentration_signal(oracle_csv: Path) -> dict:
    """Top-feature importance share and majority/oracle-style context from oracle.csv."""
    df = pd.read_csv(oracle_csv)
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    majority = float(df["majority_baseline"].iloc[0]) if "majority_baseline" in df and len(df) else np.nan
    pos = df["importance_mean"].clip(lower=0.0)
    total = float(pos.sum())
    top_feat = str(df["feature_variable"].iloc[0]) if len(df) else None
    top_imp = float(df["importance_mean"].iloc[0]) if len(df) else np.nan
    top_share = float(pos.iloc[0] / total) if total > 0 else 0.0
    # Herfindahl of importance mass: 1.0 = one feature owns everything.
    hhi = float(((pos / total) ** 2).sum()) if total > 0 else 0.0
    return {
        "top_feature": top_feat,
        "top_importance": top_imp,
        "top_importance_share": top_share,
        "importance_hhi": hhi,
        "majority_baseline": majority,
        "n_features_scored": int(len(df)),
    }


def add_oracle_acc(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach oracle_acc from the grid summaries (model-independent; take the max
    available across model CSVs per cell to be robust to a missing model)."""
    from survey_features.layout import collect_all_grid_summaries

    frames = []
    for p, sid, _tag in collect_all_grid_summaries(OUT):
        d = pd.read_csv(p, usecols=lambda c: c in {"target", "country", "oracle_acc"})
        frames.append(d)
    if not frames:
        rows["oracle_acc"] = np.nan
        return rows
    alld = pd.concat(frames, ignore_index=True)
    alld["oracle_acc"] = pd.to_numeric(alld["oracle_acc"], errors="coerce")
    best = alld.groupby(["target", "country"], as_index=False)["oracle_acc"].max()
    return rows.merge(best, on=["target", "country"], how="left")


# ---- Optional data-backed single-feature predictive test ---------------------

def single_feature_test(
    survey_id: str,
    target: str,
    country: str,
    top_feature: str,
    cfg_path: str,
    cache: dict,
) -> float | None:
    """Downstream-eval accuracy using ONLY the top oracle feature for this cell.
    Reuses the exact evaluate_feature_set routine so it is comparable to oracle_acc."""
    from survey_features.evaluation import evaluate_feature_set
    from survey_features.surveys import SURVEY_COUNTRY_COL, build_country_code_map, load_survey

    if survey_id not in cache:
        data, meta = load_survey(survey_id, cfg_path)
        ccol = SURVEY_COUNTRY_COL.get(survey_id)
        cmap = build_country_code_map(meta, ccol, data) if ccol else {}
        cache[survey_id] = (data, meta, ccol, cmap)
    data, meta, ccol, cmap = cache[survey_id]
    if top_feature not in data.columns or target not in data.columns:
        return None
    sub = data
    if ccol and ccol in data.columns and country in cmap:
        sub = data[data[ccol] == cmap[country]]
    if len(sub) < 30:
        return None
    try:
        res = evaluate_feature_set(sub, target, [top_feature])
        return float(res.get("accuracy_mean")) if res else None
    except Exception:
        return None


def classify(row: dict, args) -> str:
    lift = (row.get("oracle_acc") or 0) - (row.get("majority_baseline") or 0)
    if pd.isna(row.get("oracle_acc")) or lift < args.min_signal:
        return "degenerate"
    conc = row.get("top_importance_share") or 0
    sf = row.get("single_feature_acc")
    if sf is not None and not pd.isna(sf):
        sf_lift = sf - (row.get("majority_baseline") or 0)
        recover = (sf_lift / lift) if lift > 0 else 0.0
        row["single_feature_recovery"] = recover
        if recover >= args.recover_frac and conc >= args.conc_thresh:
            return "leakage"
        return "genuine"
    # Offline fallback: concentration-only heuristic.
    if conc >= args.conc_thresh and lift >= args.suspect_lift:
        return "leakage_suspect"
    return "genuine"


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage audit for oracle ground truth.")
    ap.add_argument("--with-data", action="store_true",
                    help="Run the single-feature predictive test (needs DATA_CONFIG_PATH).")
    ap.add_argument("--conc-thresh", type=float, default=0.80,
                    help="Top-feature importance share above which a cell is concentration-suspect.")
    ap.add_argument("--recover-frac", type=float, default=0.90,
                    help="Single-feature recovers >= this fraction of oracle lift -> leakage.")
    ap.add_argument("--min-signal", type=float, default=0.03,
                    help="Oracle lift over majority below this -> degenerate (not a leakage case).")
    ap.add_argument("--suspect-lift", type=float, default=0.10,
                    help="Offline-only: oracle lift above this + high concentration -> leakage_suspect.")
    ap.add_argument("--write-tex", action="store_true", help="Also emit a LaTeX longtable.")
    args = ap.parse_args()

    detail = load_target_detail()
    cells = iter_oracle_cells()
    if not cells:
        print("No oracle.csv cells found under outputs/.", file=sys.stderr)
        sys.exit(1)

    records = []
    for target, country, ocsv in cells:
        rec = {"survey": survey_of(target, detail), "target": target, "country": country}
        info = detail.get(target, {})
        rec["bucket"] = info.get("bucket")
        rec["section"] = info.get("section")
        rec.update(concentration_signal(ocsv))
        records.append(rec)
    rows = pd.DataFrame(records)
    rows = add_oracle_acc(rows)

    cfg = None
    if args.with_data:
        from dotenv import dotenv_values
        cfg = dotenv_values(ROOT / ".env").get("DATA_CONFIG_PATH") or os.environ.get("DATA_CONFIG_PATH")
        if not cfg or not os.path.isfile(cfg):
            print(f"--with-data set but DATA_CONFIG_PATH missing/invalid ({cfg!r}); "
                  "falling back to offline concentration heuristic.", file=sys.stderr)
            args.with_data = False

    rows["single_feature_acc"] = np.nan
    rows["single_feature_recovery"] = np.nan
    if args.with_data:
        cache: dict = {}
        for i, r in rows.iterrows():
            if pd.isna(r.get("oracle_acc")):
                continue
            sf = single_feature_test(r["survey"], r["target"], r["country"],
                                     r["top_feature"], cfg, cache)
            rows.at[i, "single_feature_acc"] = sf if sf is not None else np.nan

    rows["leakage_class"] = [classify(r._asdict() if hasattr(r, "_asdict") else dict(r), args)
                             for _, r in rows.iterrows()]
    # classify() may set single_feature_recovery via dict copy; recompute cleanly for the column
    def _recovery(r):
        lift = (r.get("oracle_acc") or 0) - (r.get("majority_baseline") or 0)
        sf = r.get("single_feature_acc")
        if sf is None or pd.isna(sf) or lift <= 0:
            return np.nan
        return (sf - (r.get("majority_baseline") or 0)) / lift
    rows["single_feature_recovery"] = rows.apply(lambda r: _recovery(dict(r)), axis=1)

    rows["oracle_lift"] = rows["oracle_acc"] - rows["majority_baseline"]
    rows = rows.sort_values(["leakage_class", "oracle_lift"], ascending=[True, False])

    out_csv, out_summary = leakage_audit_write_paths(OUT)
    cols = ["survey", "target", "country", "bucket", "section", "majority_baseline",
            "oracle_acc", "oracle_lift", "top_feature", "top_importance_share",
            "importance_hhi", "single_feature_acc", "single_feature_recovery",
            "n_features_scored", "leakage_class"]
    rows[cols].to_csv(out_csv, index=False)

    # Rollups
    def share(df, cls):
        return round(float((df["leakage_class"] == cls).mean()), 3) if len(df) else 0.0
    summary = {
        "mode": "data-backed" if args.with_data else "offline-concentration",
        "thresholds": {"conc_thresh": args.conc_thresh, "recover_frac": args.recover_frac,
                       "min_signal": args.min_signal, "suspect_lift": args.suspect_lift},
        "n_cells": int(len(rows)),
        "class_counts": rows["leakage_class"].value_counts().to_dict(),
        "by_survey": {s: sub["leakage_class"].value_counts().to_dict()
                      for s, sub in rows.groupby("survey")},
        "by_bucket": {str(b): sub["leakage_class"].value_counts().to_dict()
                      for b, sub in rows.groupby("bucket")},
        "leakage_cells": rows.loc[rows["leakage_class"].isin(["leakage", "leakage_suspect"]),
                                  ["survey", "target", "country", "top_feature",
                                   "oracle_lift", "top_importance_share",
                                   "single_feature_recovery"]].round(4).to_dict("records"),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {out_csv} ({len(rows)} cells) and {out_summary.name}")
    print(f"Mode: {summary['mode']}")
    print("Class counts:", summary["class_counts"])
    lk = rows[rows["leakage_class"].isin(["leakage", "leakage_suspect"])]
    if len(lk):
        print("\nFlagged leakage cells:")
        show = ["survey", "target", "country", "top_feature", "oracle_lift",
                "top_importance_share", "single_feature_recovery", "leakage_class"]
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(lk[show].to_string(index=False))

    if args.write_tex:
        write_tex(rows)


def write_tex(rows: pd.DataFrame) -> None:
    gen = ROOT / "paper" / "generated_current_state"
    gen.mkdir(parents=True, exist_ok=True)

    def esc(x):
        s = "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
        for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
            s = s.replace(a, b)
        return s

    def num(x, p=3):
        return "-" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{float(x):.{p}f}"

    lk = rows[rows["leakage_class"].isin(["leakage", "leakage_suspect"])]
    lines = []
    for _, r in lk.iterrows():
        lines.append(
            f"{esc(r['survey'])} & {esc(r['target'])} & {esc(r['country'])} & "
            f"{esc(r['top_feature'])} & {num(r['oracle_lift'])} & "
            f"{num(r['top_importance_share'])} & {num(r['single_feature_recovery'])} & "
            f"{esc(r['leakage_class'])} \\\\"
        )
    header = ("Survey & Target & Country & Top feat & Oracle lift & "
              "Top imp.\\ share & 1-feat recovery & Class \\\\")
    table = (
        "\\begin{longtable}{lll l rrr l}\n\\toprule\n" + header +
        "\n\\midrule\n\\endfirsthead\n\\toprule\n" + header +
        "\n\\midrule\n\\endhead\n" + "\n".join(lines) +
        "\n\\bottomrule\n\\end{longtable}\n"
    )
    (gen / "leakage_audit_longtable.tex").write_text(table, encoding="utf-8")
    print(f"Wrote {gen / 'leakage_audit_longtable.tex'}")


if __name__ == "__main__":
    main()
