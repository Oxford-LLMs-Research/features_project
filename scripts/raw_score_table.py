"""Plain side-by-side scores from score CSVs: no deltas, no composite metrics.

For each score file and answer family, prints the mean raw score of every feature set
the scorer fits on the same held-out rows, so a reader compares like with like:

    binary / nominal targets   -> accuracy  (always-most-common answer, random picks,
                                             10 textbook demographics, model picks,
                                             oracle top picks)
    ordinal / continuous       -> Spearman rank correlation (same sets; the
                                             most-common-answer baseline has none)

plus the share of cells where the model's picks beat random and textbook, and the mean
captured importance of model vs textbook picks. Rows are the model-chosen k
(`k_spec == "model"`) with no error. Families are never pooled (type-matched scoring,
pilot report 2026-08). The pipeline computes accuracy, log loss and Spearman; it does
not compute F1.

Usage (paths relative to OUTPUTS_DIR, globs allowed):
    python scripts/raw_score_table.py selectors/runs/pilot_phase_a/scores_*.csv
    python scripts/raw_score_table.py "experiments/prompt_sensitivity_v2/scores_*_scientist_respondent_r1_*.csv"
    python scripts/raw_score_table.py selectors/scores_*.csv --condition country_provided
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from survey_features.config import OUTPUTS_DIR  # noqa: E402

CATEGORICAL = {"binary", "nominal"}
SCALE = {"ordinal", "continuous"}

# (label, column) per family, in reading order: weakest reference first, ceiling last.
ACC_COLS = [("Most-common answer", "majority"), ("Random picks", "random_acc"),
            ("Textbook demographics", "textbook_acc"), ("Model picks", "model_acc"),
            ("Oracle top picks", "oracle_acc")]
RHO_COLS = [("Most-common answer", None), ("Random picks", "random_rho"),
            ("Textbook demographics", "textbook_rho"), ("Model picks", "model_rho"),
            ("Oracle top picks", "oracle_rho")]


def _source_label(path: Path) -> str:
    stem = path.stem
    return stem[len("scores_"):] if stem.startswith("scores_") else stem


def _usable(df: pd.DataFrame, condition: str | None) -> pd.DataFrame:
    out = df[df["k_spec"].astype(str) == "model"]
    if "error" in out.columns:
        out = out[out["error"].isna() | (out["error"].astype(str).str.strip() == "")]
    if condition:
        out = out[out["condition"] == condition]
    return out


def _fmt(x: float | None) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.3f}"


def _share(a: pd.Series, b: pd.Series) -> str:
    ok = a.notna() & b.notna()
    return "–" if not ok.any() else f"{(a[ok] > b[ok]).mean():.0%}"


def table(paths: list[Path], condition: str | None) -> str:
    head = ["Source", "Answer type", "Score", "Cells"] + [lab for lab, _ in ACC_COLS] + [
        "Model > random", "Model > textbook", "Captured importance: model / textbook"]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for path in paths:
        df = _usable(pd.read_csv(path), condition)
        for family, types, cols, score in (("Yes/no, multiple choice", CATEGORICAL, ACC_COLS, "accuracy"),
                                           ("Rating scale, number", SCALE, RHO_COLS, "rank correlation")):
            sub = df[df["target_type"].isin(types)]
            if sub.empty:
                continue
            vals = [_fmt(sub[c].mean()) if c else "–" for _, c in cols]
            model, rand, tb = cols[3][1], cols[1][1], cols[2][1]
            cap = f"{_fmt(sub['captured_importance'].mean())} / {_fmt(sub['textbook_captured'].mean())}"
            lines.append("| " + " | ".join(
                [_source_label(path), family, score, str(len(sub))] + vals +
                [_share(sub[model], sub[rand]), _share(sub[model], sub[tb]), cap]) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("patterns", nargs="+", help="score CSV paths or globs, relative to OUTPUTS_DIR")
    ap.add_argument("--condition", choices=["country_provided", "unprompted"], default=None,
                    help="keep one prompt condition (default: both)")
    ap.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    args = ap.parse_args()
    paths: list[Path] = []
    for pat in args.patterns:
        hits = sorted(args.outputs_dir.glob(pat))
        if not hits:
            raise SystemExit(f"no score file matches {pat!r} under {args.outputs_dir}")
        paths.extend(hits)
    print(table(paths, args.condition))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
