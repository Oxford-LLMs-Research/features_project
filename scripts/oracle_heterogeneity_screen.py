"""
Per-target cross-country heterogeneity screen over cached oracles.

For each target with oracle.csv in two or more countries, compares
  WITHIN-country reliability   agreement of importance_select vs importance_score
                               (the oracle's own noise floor: two honest reads of
                               the same country), against
  BETWEEN-country agreement    importance_score vs importance_score across the
                               target's countries (restricted to shared features).

het = within - between. A target where the between-country disagreement exceeds
the oracle's own noise floor has GENUINE country-specific predictive structure —
those are the cells where a country-adaptation contrast (Test 2) has something to
measure. het ~ 0 means the structure is shared (or the oracle too noisy to tell):
nothing to adapt to at this oracle fidelity.

Leakage guard: leakage MIMICS heterogeneity (a leaky feature dominates in exactly
one country), so every row carries the target's leakage classes from the audit and
`leakage_flag` marks targets with any leakage/leakage_distributed cell. Never
sample a Test-2 stratum from flagged targets.

First validated 2026-08-17 against the 89 era-3 cells (30 targets): within
Jaccard@10 = 0.334, between = 0.183; full-vector Spearman 0.149 vs 0.087;
disattenuated between-country similarity ~ 0.59. Q263 (top raw het) is leakage.

Usage:
    python scripts/oracle_heterogeneity_screen.py                # all cached cells
    python scripts/oracle_heterogeneity_screen.py --targets Q263 level
    python scripts/oracle_heterogeneity_screen.py --candidates cells.csv \
        [--compute-missing]        # cells.csv: survey,target,country
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from survey_features.config import OUTPUTS_DIR  # noqa: E402
from survey_features.layout import (  # noqa: E402
    analysis_dir,
    cache_cells_dir,
    cell_dir,
    leakage_audit_csv_path,
)

TOP_K = 10

# Audit classes that disqualify a target from the Test-2 stratum.
LEAKAGE_CLASSES = {"leakage", "leakage_distributed"}


def _jaccard_topk(a: pd.Series, b: pd.Series, k: int = TOP_K) -> float | None:
    top_a, top_b = set(a.nlargest(k).index), set(b.nlargest(k).index)
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else None


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    r = spearmanr(a, b).statistic
    return None if np.isnan(r) else float(r)


def _load_cell(path: Path) -> pd.DataFrame | None:
    csv = path / "oracle.csv"
    if not csv.is_file():
        return None
    df = pd.read_csv(csv)
    if "feature_variable" not in df.columns:
        return None
    # v3 and v4 both carry importance_select / importance_score; legacy single-column
    # caches fall back to importance_mean for both sides (within-reliability then
    # degenerates to 1.0 and is reported as None instead).
    df = df.groupby("feature_variable").first()
    for col in ("importance_select", "importance_score"):
        if col not in df.columns:
            df[col] = df.get("importance_mean", 0.0)
    return df


def discover_cells(cells_root: Path, targets: list[str] | None):
    """{target: {country: cell_df}} for every cached cell (dir name <target>_<country>)."""
    by_target: dict[str, dict[str, pd.DataFrame]] = {}
    for d in sorted(cells_root.iterdir()) if cells_root.is_dir() else []:
        if not d.is_dir():
            continue
        target, _, country = d.name.rpartition("_")
        if not target or (targets and target not in targets):
            continue
        df = _load_cell(d)
        if df is not None:
            by_target.setdefault(target, {})[country] = df
    return by_target


def screen_target(cells: dict[str, pd.DataFrame]) -> dict:
    """Within/between agreement summary for one target across its countries."""
    within_j, within_r, between_j, between_r = [], [], [], []
    for df in cells.values():
        distinct = not df["importance_select"].equals(df["importance_score"])
        if distinct:
            within_j.append(_jaccard_topk(df["importance_select"], df["importance_score"]))
            within_r.append(_spearman(df["importance_select"], df["importance_score"]))
    for c1, c2 in combinations(sorted(cells), 2):
        common = cells[c1].index.intersection(cells[c2].index)
        if len(common) < 15:
            continue
        s1 = cells[c1].loc[common, "importance_score"]
        s2 = cells[c2].loc[common, "importance_score"]
        between_j.append(_jaccard_topk(s1, s2))
        between_r.append(_spearman(s1, s2))

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    wj, bj = _mean(within_j), _mean(between_j)
    wr, br = _mean(within_r), _mean(between_r)
    return {
        "n_countries": len(cells),
        "n_country_pairs": len(between_j),
        "within_jaccard10": wj,
        "between_jaccard10": bj,
        "within_spearman": wr,
        "between_spearman": br,
        "het_rank": round(wj - bj, 4) if (wj is not None and bj is not None) else None,
        "het_value": round(wr - br, 4) if (wr is not None and br is not None) else None,
    }


def leakage_by_target(outputs_dir: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    """({target: set of leakage_class values}, {target: survey}) from the audit CSV."""
    path = leakage_audit_csv_path(outputs_dir)
    classes: dict[str, set[str]] = {}
    surveys: dict[str, str] = {}
    if path.is_file():
        audit = pd.read_csv(path)
        for t, grp in audit.groupby("target"):
            classes[str(t)] = set(grp["leakage_class"].astype(str))
            surveys[str(t)] = str(grp["survey"].iloc[0])
    return classes, surveys


def missing_candidates(candidates_csv: Path, outputs_dir: Path) -> pd.DataFrame:
    """Candidate rows (survey,target,country) whose cell has no oracle.csv yet."""
    cand = pd.read_csv(candidates_csv)
    required = {"survey", "target", "country"}
    if not required.issubset(cand.columns):
        raise SystemExit(f"--candidates CSV needs columns {sorted(required)}")
    missing = [
        row for _, row in cand.iterrows()
        if not (cell_dir(str(row["target"]), str(row["country"]), outputs_dir) / "oracle.csv").is_file()
    ]
    return pd.DataFrame(missing, columns=cand.columns)


def compute_missing(missing: pd.DataFrame, outputs_dir: Path) -> None:
    """Fill missing cells with cheap v3 oracles via scripts/compute_oracle.py."""
    for survey, grp in missing.groupby("survey"):
        cmd = [
            sys.executable, str(ROOT / "scripts" / "compute_oracle.py"),
            "--survey", str(survey),
            "--targets", *sorted(grp["target"].astype(str).unique()),
            "--countries", *sorted(grp["country"].astype(str).unique()),
            "--output-dir", str(outputs_dir),
        ]
        print(f"[screen] computing {len(grp)} missing {survey} oracles:")
        print("  " + " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-country oracle heterogeneity screen.")
    ap.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    ap.add_argument("--targets", nargs="+", default=None,
                    help="Restrict to these targets (default: every cached cell).")
    ap.add_argument("--candidates", type=Path, default=None,
                    help="CSV of survey,target,country cells the screen should cover.")
    ap.add_argument("--compute-missing", action="store_true",
                    help="Compute v3 oracles for candidate cells with no cache "
                         "(otherwise they are only reported).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV (default: <outputs>/analysis/oracle_heterogeneity.csv).")
    args = ap.parse_args()

    if args.candidates:
        missing = missing_candidates(args.candidates, args.outputs_dir)
        if not missing.empty:
            print(f"[screen] {len(missing)} candidate cells have no oracle.csv")
            if args.compute_missing:
                compute_missing(missing, args.outputs_dir)
            else:
                for _, r in missing.iterrows():
                    print(f"  missing: {r['survey']} {r['target']} {r['country']}")
                print("[screen] re-run with --compute-missing to fill them.")

    by_target = discover_cells(cache_cells_dir(args.outputs_dir), args.targets)
    leak_classes, leak_surveys = leakage_by_target(args.outputs_dir)

    rows = []
    for target, cells in sorted(by_target.items()):
        if len(cells) < 2:
            continue
        rec = screen_target(cells)
        classes = leak_classes.get(target, set())
        rec.update({
            "target": target,
            "survey": leak_surveys.get(target, ""),
            "countries": ";".join(sorted(cells)),
            "leakage_classes": ";".join(sorted(classes)) if classes else "",
            "leakage_flag": bool(classes & LEAKAGE_CLASSES),
        })
        rows.append(rec)

    if not rows:
        raise SystemExit("No target has cached oracles in >= 2 countries.")

    table = pd.DataFrame(rows)[[
        "target", "survey", "n_countries", "n_country_pairs", "countries",
        "within_jaccard10", "between_jaccard10", "within_spearman", "between_spearman",
        "het_rank", "het_value", "leakage_classes", "leakage_flag",
    ]].sort_values("het_value", ascending=False, na_position="last")

    out = args.out or (analysis_dir(args.outputs_dir) / "oracle_heterogeneity.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    wj, bj = table["within_jaccard10"].mean(), table["between_jaccard10"].mean()
    wr, br = table["within_spearman"].mean(), table["between_spearman"].mean()
    print(f"\n[screen] {len(table)} targets "
          f"({int(table['n_countries'].sum())} cells) -> {out}")
    print(f"  within-country (noise floor): Jaccard@{TOP_K} = {wj:.3f}  Spearman = {wr:.3f}")
    print(f"  between-country:              Jaccard@{TOP_K} = {bj:.3f}  Spearman = {br:.3f}")
    if wr:
        print(f"  disattenuated between-country similarity (br/wr): {br / wr:.3f}")
    flagged = table[table["leakage_flag"]]["target"].tolist()
    if flagged:
        print(f"  leakage-flagged (exclude from Test-2 sampling): {', '.join(flagged)}")


if __name__ == "__main__":
    main()
