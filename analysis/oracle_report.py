"""
Oracle report: flat CSV of oracle results with feature question text.

For each (survey, target, country) cell, shows the top-N features with their
question descriptions from survey metadata alongside importance scores.

Usage:
    python analysis/oracle_report.py
    python analysis/oracle_report.py --top-n 15 --output analysis/oracle_report.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase0b_oracle_autogluon import flatten_metadata

OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_MANIFEST = ROOT / "prelim" / "trial_manifest.yaml"
DEFAULT_TOP_N = 10

SKEWED_THRESHOLD      = 0.80
CONCENTRATED_SHARE    = 0.50
CONCENTRATED_MIN_ABS  = 0.03
WEAK_SIGNAL_THRESHOLD = 0.01


def load_survey_metadata(survey_id: str, config_path: str) -> dict[str, dict]:
    from synthetic_sampling.config.base import DataPaths
    from synthetic_sampling.loaders.survey_loader import SurveyLoader
    paths = DataPaths.from_yaml(config_path)
    loader = SurveyLoader(paths=paths, verbose=False)
    _, raw_meta = loader.load_survey(survey_id)
    return flatten_metadata(raw_meta)


def compute_flags(df: pd.DataFrame) -> str:
    flags = []
    baseline = float(df["majority_baseline"].iloc[0])
    if baseline > SKEWED_THRESHOLD:
        flags.append("SKEWED")
    pos = df[df["importance_mean"] > 0].sort_values("importance_mean", ascending=False)
    if pos.empty or pos["importance_mean"].iloc[0] < WEAK_SIGNAL_THRESHOLD:
        flags.append("WEAK_SIGNAL")
    elif not pos.empty:
        top_imp = float(pos["importance_mean"].iloc[0])
        total_pos = float(pos["importance_mean"].sum())
        if total_pos > 0 and (top_imp / total_pos) > CONCENTRATED_SHARE and top_imp > CONCENTRATED_MIN_ABS:
            flags.append("CONCENTRATED")
    return " ".join(flags)


def build_report(manifest_path: Path, top_n: int, config_path: str) -> pd.DataFrame:
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    meta_cache: dict[str, dict] = {}
    rows: list[dict] = []

    for survey_id, survey_entry in manifest["surveys"].items():

        if survey_id not in meta_cache:
            print(f"Loading metadata: {survey_id}...", flush=True)
            try:
                meta_cache[survey_id] = load_survey_metadata(survey_id, config_path)
            except Exception as exc:
                print(f"  WARNING: metadata load failed for {survey_id}: {exc}")
                meta_cache[survey_id] = {}

        meta = meta_cache[survey_id]

        for target in survey_entry["targets"]:
            countries = survey_entry["countries"]

            tgt = meta.get(target, {})
            tgt_desc = tgt.get("description", "")
            tgt_q    = tgt.get("question", "")
            tgt_sec  = tgt.get("section", "")

            for country in countries:
                oracle_path = OUTPUTS_DIR / f"{target}_{country}" / "oracle.csv"
                base_row = dict(
                    survey=survey_id,
                    target=target,
                    target_description=tgt_desc,
                    target_question=tgt_q,
                    target_section=tgt_sec,
                    country=country,
                )

                if not oracle_path.exists():
                    rows.append({**base_row,
                        "pool_size": None, "majority_baseline": None, "flags": "MISSING",
                        "feat_rank": None, "feat_variable": None,
                        "feat_description": None, "feat_question": None,
                        "feat_section": None,
                        "feat_importance_mean": None, "feat_importance_std": None,
                    })
                    continue

                df = pd.read_csv(oracle_path)
                df = df[df["target_variable"] == target].copy()
                if df.empty:
                    continue

                pool_size = len(df)
                baseline  = round(float(df["majority_baseline"].iloc[0]), 4)
                flags     = compute_flags(df)

                top = df.sort_values("importance_mean", ascending=False).head(top_n)

                for rank, (_, feat_row) in enumerate(top.iterrows(), 1):
                    fvar = feat_row["feature_variable"]
                    fmeta = meta.get(fvar, {})
                    rows.append({**base_row,
                        "pool_size":         pool_size,
                        "majority_baseline": baseline,
                        "flags":             flags,
                        "feat_rank":         rank,
                        "feat_variable":     fvar,
                        "feat_description":  fmeta.get("description", ""),
                        "feat_question":     fmeta.get("question", ""),
                        "feat_section":      fmeta.get("section", ""),
                        "feat_importance_mean": round(float(feat_row["importance_mean"]), 5),
                        "feat_importance_std":  round(float(feat_row["importance_std"]),  5),
                    })

    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate oracle report CSV with feature descriptions")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to trial manifest YAML")
    parser.add_argument("--top-n",   type=int, default=DEFAULT_TOP_N, help="Top N features per cell (default 10)")
    parser.add_argument("--output",  default=str(ROOT / "analysis" / "oracle_report.csv"), help="Output CSV path")
    args = parser.parse_args()

    config_path = os.environ.get("DATA_CONFIG_PATH")
    if not config_path:
        raise ValueError("DATA_CONFIG_PATH not set — add it to your .env file")

    print(f"Building oracle report (top {args.top_n} features per cell)...")
    report = build_report(Path(args.manifest), args.top_n, config_path)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False, encoding="utf-8-sig")

    data_rows = report[report["feat_rank"] == 1]
    missing   = report[report["flags"] == "MISSING"]
    print(f"\nDone. {len(data_rows)} cells with data, {len(missing)} MISSING")
    print(f"Wrote {len(report)} rows to {out}")


if __name__ == "__main__":
    main()
