"""
Dump the detected measurement level for every candidate target, for human review.

The oracle is the study's gold standard, so how it MODELS a target matters as much as
which features it ranks. Measurement level drives problem_type:

    binary      -> binary classification,      log_loss
    nominal     -> multiclass classification,  log_loss
    ordinal     -> REGRESSION on the codes,    spearmanr
    continuous  -> regression,                 spearmanr

Ordinal is the case that was silently broken: forcing an 11-point left-right scale into
11 unordered classes splits ~1,100 respondents across 11 categories and penalises
predicting 6-when-truth-is-5 exactly as hard as predicting 1. Measured on
P16ST x Colombia, log-loss lift went -0.134 as 11 classes vs +0.030 collapsed to 2; under
the regression/spearmanr path the same cell yields 101/193 features with positive
importance and a top-10 led by economic expectations, presidential approval and two
redistribution items.

Detection is label-driven and therefore fallible — review this file before a confirmatory
run, the same way as the missing-code taxonomy.

Writes outputs/cache/audits/target_types.csv

Usage:
    python scripts/audit_target_types.py                 # every audited grid target
    python scripts/audit_target_types.py --all-variables # every variable in every survey
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from survey_features.config import OUTPUTS_DIR  # noqa: E402
from survey_features.layout import cache_audits_dir, leakage_audit_csv_path  # noqa: E402
from survey_features.surveys import (  # noqa: E402
    SURVEY_COUNTRY_COL,
    extract_survey_variables,
    load_survey,
    target_type_rows,
)

COLS = ["survey", "variable", "detected_type", "reason", "n_substantive_values",
        "question", "labels"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", nargs="+", default=None)
    ap.add_argument("--all-variables", action="store_true",
                    help="Classify every variable, not just the grid's targets.")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if not os.environ.get("DATA_CONFIG_PATH"):
        raise SystemExit("DATA_CONFIG_PATH is not set in .env")
    out_root = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR

    targets_by_survey: dict[str, list[str]] = {}
    if not args.all_variables:
        p = leakage_audit_csv_path(out_root)
        if not p.is_file():
            raise SystemExit(f"no leakage audit at {p}; use --all-variables")
        aud = pd.read_csv(p)
        for s, g in aud.groupby("survey"):
            targets_by_survey[str(s)] = sorted(g.target.astype(str).unique())

    surveys = args.survey or list(targets_by_survey or SURVEY_COUNTRY_COL)
    all_rows: list[dict] = []
    for survey in surveys:
        try:
            data, meta = load_survey(survey, os.environ["DATA_CONFIG_PATH"])
        except Exception as exc:
            print(f"[skip] {survey}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        codes = (sorted(extract_survey_variables(meta)) if args.all_variables
                 else targets_by_survey.get(survey, []))
        rows = target_type_rows(meta, codes, data, survey=survey)
        for r in rows:
            r["survey"] = survey
        all_rows.extend(rows)
        counts = pd.Series([r["detected_type"] for r in rows]).value_counts().to_dict()
        print(f"{survey:16s} {len(rows):4d} variables | {counts}")

    out = cache_audits_dir(out_root) / "target_types.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    print(f"\nwrote {len(all_rows)} rows -> {out}")

    totals = pd.Series([r["detected_type"] for r in all_rows]).value_counts()
    print("\noverall:")
    print(totals.to_string())
    print("\nReview `ordinal` vs `nominal` in particular: a nominal target wrongly called "
          "ordinal gets regressed on meaningless codes, and an ordinal target wrongly "
          "called nominal is the failure this detector exists to prevent.")


if __name__ == "__main__":
    main()
