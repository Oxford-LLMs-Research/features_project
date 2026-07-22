"""
Summarise outputs/grid_summary__*.csv files from multi-survey prelim runs.

Usage (from repo root):

    python analysis/prelim_aggregate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"


def main():
    from survey_features.layout import collect_grid_summary_paths, parse_grid_summary_stem

    csvs = collect_grid_summary_paths(OUT)
    if not csvs:
        print(f"No grid_summary CSVs found under {OUT}")
        sys.exit(0)

    for p in csvs:
        df = pd.read_csv(p)
        n = len(df)
        err_col = df.get("error")
        errs = err_col.notna().sum() if err_col is not None else 0
        sid, tag = parse_grid_summary_stem(p.stem)
        label = f"{sid}" + (f" (tag={tag})" if tag else "")
        print(f"{label}: rows={n} error_rows~={int(errs)}")
        parts = df.get("oracle_acc")
        if parts is not None:
            vv = pd.to_numeric(parts, errors="coerce")
            valid = vv.notna().sum()
            print(f"  oracle_acc populated: {int(valid)} of {n}")


if __name__ == "__main__":
    main()
