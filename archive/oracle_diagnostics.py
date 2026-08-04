"""
Oracle diagnostics: surface leakage and quality signals from oracle.csv outputs.

Reads oracle.csv for every (target, country) cell in a manifest and prints a
per-cell summary table with quality flags:

  MISSING      oracle.csv not found — cell hasn't run yet
  SKEWED       majority_baseline > 0.80 — target is class-imbalanced; even random
               features will score well, making oracle/random comparison uninformative
  CONCENTRATED top feature captures >50% of total positive importance AND its
               absolute importance > 0.03 — one predictor dominates, possible
               surviving leakage (check the feature name)
  SMALL_POOL   fewer than 15 features in pool after all filters — oracle is fragile
  WEAK_SIGNAL  max importance < 0.01 — no feature predicts the target; ranking is noise

Usage:
    python analysis/oracle_diagnostics.py
    python analysis/oracle_diagnostics.py --manifest prelim/trial_manifest.yaml
    python analysis/oracle_diagnostics.py --manifest prelim/trial_manifest.yaml --survey wvs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_MANIFEST = ROOT / "prelim" / "trial_manifest.yaml"

# ── Thresholds ────────────────────────────────────────────────────────────────
SKEWED_THRESHOLD      = 0.80   # majority_baseline above this → SKEWED
CONCENTRATED_SHARE    = 0.50   # top feature > this share of top-5 positive → CONCENTRATED
CONCENTRATED_MIN_ABS  = 0.03   # only flag CONCENTRATED if top importance > this absolute value
SMALL_POOL_THRESHOLD  = 15     # n features below this → SMALL_POOL
WEAK_SIGNAL_THRESHOLD = 0.01   # max importance below this → WEAK_SIGNAL


def diagnose_cell(
    survey: str,
    target: str,
    country: str,
) -> dict:
    oracle_path = OUTPUTS_DIR / f"{target}_{country}" / "oracle.csv"

    row: dict = {
        "survey":           survey,
        "target":           target,
        "country":          country,
        "status":           "MISSING",
        "pool_size":        None,
        "majority_baseline": None,
        "n_positive":       None,
        "top_feature":      None,
        "top_importance":   None,
        "flags":            [],
    }

    if not oracle_path.exists():
        return row

    df = pd.read_csv(oracle_path)
    df = df[df["target_variable"] == target].copy()
    if df.empty:
        row["status"] = "EMPTY"
        return row

    row["status"] = "OK"
    row["pool_size"] = len(df)
    row["majority_baseline"] = round(float(df["majority_baseline"].iloc[0]), 3)

    df_sorted = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    top = df_sorted.iloc[0]
    row["top_feature"]    = str(top["feature_variable"])
    row["top_importance"] = round(float(top["importance_mean"]), 4)

    positive = df_sorted[df_sorted["importance_mean"] > 0]
    row["n_positive"] = len(positive)

    flags: list[str] = []

    if row["majority_baseline"] > SKEWED_THRESHOLD:
        flags.append("SKEWED")

    if row["pool_size"] < SMALL_POOL_THRESHOLD:
        flags.append("SMALL_POOL")

    if row["top_importance"] < WEAK_SIGNAL_THRESHOLD:
        flags.append("WEAK_SIGNAL")
    elif row["top_importance"] >= CONCENTRATED_MIN_ABS:
        top5_pos = positive.head(5)
        total_top5 = float(top5_pos["importance_mean"].sum())
        if total_top5 > 0:
            share = float(top["importance_mean"]) / total_top5
            if share > CONCENTRATED_SHARE:
                flags.append("CONCENTRATED")

    row["flags"] = flags
    return row


def top5_features(target: str, country: str) -> list[tuple[str, float]]:
    """Return top-5 (feature_variable, importance_mean) for a cell."""
    oracle_path = OUTPUTS_DIR / f"{target}_{country}" / "oracle.csv"
    if not oracle_path.exists():
        return []
    df = pd.read_csv(oracle_path)
    df = df[df["target_variable"] == target].copy()
    if df.empty:
        return []
    df_sorted = df.sort_values("importance_mean", ascending=False).head(5)
    return [(str(r["feature_variable"]), round(float(r["importance_mean"]), 4))
            for _, r in df_sorted.iterrows()]


def print_report(rows: list[dict], show_top5: bool = True) -> None:
    header = (
        f"{'Survey':14s} {'Target':8s} {'Country':14s} "
        f"{'Pool':>5s} {'Base':>6s} {'Top feat':12s} {'TopImp':>7s} {'Flags'}"
    )
    print("\n" + "=" * len(header))
    print("Oracle diagnostics")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    flagged_cells: list[dict] = []
    missing_cells: list[dict] = []

    for r in rows:
        pool  = f"{r['pool_size']:5d}" if r["pool_size"] is not None else f"{'?':>5s}"
        base  = f"{r['majority_baseline']:6.3f}" if r["majority_baseline"] is not None else f"{'?':>6s}"
        tfeat = (r["top_feature"] or "?")[:12]
        timp  = f"{r['top_importance']:7.4f}" if r["top_importance"] is not None else f"{'?':>7s}"
        flag_str = " ".join(r["flags"]) if r["flags"] else ""
        status_marker = "?" if r["status"] == "MISSING" else ""

        print(
            f"{r['survey']:14s} {r['target']:8s} {r['country']:14s} "
            f"{pool} {base} {tfeat:12s} {timp}  {flag_str}{status_marker}"
        )
        if r["flags"]:
            flagged_cells.append(r)
        if r["status"] == "MISSING":
            missing_cells.append(r)

    print("-" * len(header))

    from collections import Counter
    all_flags: list[str] = []
    for r in rows:
        all_flags.extend(r["flags"])
    flag_counts = Counter(all_flags)

    n_ok      = sum(1 for r in rows if r["status"] == "OK")
    n_missing = len(missing_cells)
    n_flagged = len(flagged_cells)

    print(f"\nCells: {n_ok} OK, {n_missing} MISSING, {n_flagged} flagged")
    if flag_counts:
        for flag, count in sorted(flag_counts.items()):
            print(f"  {flag}: {count}")

    cells_to_detail = [r for r in flagged_cells if r["status"] == "OK"]
    if show_top5 and cells_to_detail:
        print("\n-- Top-5 features for flagged cells --")
        for r in cells_to_detail:
            feats = top5_features(r["target"], r["country"])
            print(f"\n  {r['survey']} | {r['target']} x {r['country']}  [{' '.join(r['flags'])}]")
            print(f"  majority_baseline={r['majority_baseline']}")
            for rank, (feat, imp) in enumerate(feats, 1):
                print(f"    {rank}. {feat:12s}  {imp:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle diagnostic report.")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST),
        help=f"Manifest YAML (default: {DEFAULT_MANIFEST.name})",
    )
    parser.add_argument(
        "--survey", default=None,
        help="Restrict to one survey id.",
    )
    parser.add_argument(
        "--no-top5", action="store_true",
        help="Skip top-5 feature detail for flagged cells.",
    )
    args = parser.parse_args()

    mf_path = Path(args.manifest)
    if not mf_path.is_file():
        print(f"Manifest not found: {mf_path.resolve()}")
        sys.exit(1)

    with open(mf_path, encoding="utf-8") as f:
        mf = yaml.safe_load(f)

    surveys_block: dict = mf.get("surveys") or {}
    rows: list[dict] = []

    for sid in sorted(surveys_block):
        if args.survey and sid != args.survey:
            continue
        block = surveys_block[sid]
        for target in block.get("targets") or []:
            for country in block.get("countries") or []:
                rows.append(diagnose_cell(sid, target, country))

    if not rows:
        print("No cells found. Check --manifest and --survey.")
        sys.exit(0)

    print_report(rows, show_top5=not args.no_top5)


if __name__ == "__main__":
    main()
