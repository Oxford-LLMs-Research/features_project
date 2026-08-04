"""
Dump every non-substantive value code in every survey with its assigned class.

The pipeline distinguishes RESPONDENT non-response ("Don't know", "Refused" — genuine
answers, kept as levels) from STRUCTURAL missingness ("Not asked", "Not applicable" —
set to NaN). That split is label-driven, and survey houses do not use consistent
wording, so the classification needs a human pass before the oracle re-runs.

Writes outputs/cache/audits/missing_code_taxonomy.csv and prints a per-survey summary.
Rows with ambiguous=1 are the ones worth reading: labels like "No answer" / "No response"
mean a respondent decision in some codebooks and a fieldwork gap in others. They default
to RESPONDENT; flip a pattern in survey_features.surveys to change that.

Usage:
    python scripts/audit_missing_codes.py
    python scripts/audit_missing_codes.py --survey wvs ess_wave_11
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from survey_features.config import OUTPUTS_DIR
from survey_features.layout import cache_audits_dir
from survey_features.surveys import (
    RESPONDENT,
    STRUCTURAL,
    SURVEY_COUNTRY_COL,
    load_survey,
    missing_code_taxonomy_rows,
)

COLS = [
    "survey", "section", "variable", "value_code", "label",
    "classified_as", "ambiguous", "action",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", nargs="+", default=None,
                    help="Surveys to audit (default: all in SURVEY_COUNTRY_COL).")
    ap.add_argument("--output-dir", default=None, help="Output root (default: outputs/).")
    args = ap.parse_args()

    config_path = os.environ.get("DATA_CONFIG_PATH")
    if not config_path:
        raise SystemExit("DATA_CONFIG_PATH is not set in .env")

    surveys = args.survey or list(SURVEY_COUNTRY_COL)
    out_root = args.output_dir or OUTPUTS_DIR
    out_path = cache_audits_dir(out_root) / "missing_code_taxonomy.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for survey in surveys:
        try:
            _data, metadata = load_survey(survey, config_path)
        except Exception as exc:
            print(f"[skip] {survey}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        rows = missing_code_taxonomy_rows(metadata)
        for r in rows:
            r["survey"] = survey
        all_rows.extend(rows)

        n_resp = sum(1 for r in rows if r["classified_as"] == RESPONDENT)
        n_struct = sum(1 for r in rows if r["classified_as"] == STRUCTURAL)
        n_amb = sum(1 for r in rows if r["ambiguous"])
        n_vars = len({r["variable"] for r in rows})
        print(
            f"{survey:16s} {len(rows):5d} codes over {n_vars:4d} vars | "
            f"respondent {n_resp:5d} | structural {n_struct:5d} | ambiguous {n_amb:4d}"
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    print(f"\nwrote {len(all_rows)} rows -> {out_path}")

    amb = sorted({(r["survey"], r["label"]) for r in all_rows if r["ambiguous"]})
    if amb:
        print("\nAmbiguous labels (default -> respondent; review these):")
        for survey, label in amb:
            print(f"  {survey:16s} {label}")


if __name__ == "__main__":
    main()
