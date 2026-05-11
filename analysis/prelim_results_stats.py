"""Emit JSON stats for analysis/prelim_findings.md from outputs/grid_summary__*.csv."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"


def main() -> None:
    from output_layout import collect_grid_summary_paths, parse_grid_summary_stem

    paths = collect_grid_summary_paths(OUT)
    if not paths:
        print("{}", file=sys.stderr)
        sys.exit(1)

    dfs: list[pd.DataFrame] = []
    for p in paths:
        sid, tag = parse_grid_summary_stem(p.stem)
        d = pd.read_csv(p)
        d["survey"] = sid
        if tag:
            d["llm_run_tag"] = tag
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    for c in (
        "oracle_acc",
        "model_acc",
        "random_acc",
        "cost_of_imperfect",
        "value_over_random",
        "majority_baseline",
        "k_mapped",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    valid_mask = df["oracle_acc"].notna() & df["model_acc"].notna() & df["random_acc"].notna()
    valid_df = df.loc[valid_mask]

    deg = df.loc[~df["oracle_acc"].notna()]
    k0 = df.loc[df["k_mapped"].fillna(0) == 0]

    survey_stats: dict[str, dict] = {}
    for sid in sorted(df["survey"].unique()):
        vm = valid_mask & (df["survey"] == sid)
        sub = df.loc[vm]
        if sub.empty:
            survey_stats[sid] = {"n_rows": int((df["survey"] == sid).sum()), "valid": 0}
            continue
        survey_stats[sid] = {
            "n_rows": int((df["survey"] == sid).sum()),
            "valid": int(len(sub)),
            "mean_oracle": round(float(sub["oracle_acc"].mean()), 4),
            "mean_model": round(float(sub["model_acc"].mean()), 4),
            "mean_random": round(float(sub["random_acc"].mean()), 4),
            "mean_cost": round(float(sub["cost_of_imperfect"].mean()), 4),
            "mean_value": round(float(sub["value_over_random"].mean()), 4),
            "share_value_positive": round(float((sub["value_over_random"] > 0).mean()), 4),
            "mean_signal": round(
                float((sub["oracle_acc"] - sub["majority_baseline"]).mean()), 4
            ),
        }

    cond_stats: dict[str, dict] = {}
    for cond in ("unprompted", "country_provided"):
        vm = valid_mask & (df["condition"] == cond)
        sub = df.loc[vm]
        cond_stats[cond] = {
            "n": int(len(sub)),
            "mean_cost": round(float(sub["cost_of_imperfect"].mean()), 4) if len(sub) else None,
            "mean_value": round(float(sub["value_over_random"].mean()), 4) if len(sub) else None,
            "share_value_positive": round(float((sub["value_over_random"] > 0).mean()), 4)
            if len(sub)
            else None,
        }

    def rows_to_list(cols: list[str], dff: pd.DataFrame) -> list[dict]:
        out_l: list[dict] = []
        for _, r in dff.iterrows():
            row: dict = {}
            for c in cols:
                v = r.get(c)
                if pd.isna(v):
                    row[c] = None
                elif hasattr(v, "item"):
                    try:
                        row[c] = v.item()
                    except (ValueError, AttributeError):
                        row[c] = str(v)
                else:
                    row[c] = float(v) if isinstance(v, (float, int)) else v
            out_l.append(row)
        return out_l

    top_cost = valid_df.nlargest(12, "cost_of_imperfect")
    worst_value = valid_df.nsmallest(12, "value_over_random")
    best_value = valid_df.nlargest(12, "value_over_random")

    # Join target bucket from manifest detail if present
    bucket_by_survey_target: dict[tuple[str, str], str] = {}
    det_path = ROOT / "prelim" / "target_selection_detail.yaml"
    if det_path.is_file():
        with open(det_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        for blk in doc.get("surveys") or []:
            sid = blk.get("survey_id")
            for s in blk.get("selected") or []:
                vc = s.get("variable")
                b = s.get("bucket")
                if sid and vc and b:
                    bucket_by_survey_target[(sid, str(vc))] = str(b)

    def bucket_row(r) -> str | None:
        return bucket_by_survey_target.get((r["survey"], str(r["target"])))

    valid_df = valid_df.copy()
    valid_df["bucket"] = valid_df.apply(bucket_row, axis=1)

    bucket_agg: dict[str, dict] = {}
    for b in valid_df["bucket"].dropna().unique():
        sub = valid_df.loc[valid_df["bucket"] == b]
        if sub.empty:
            continue
        bucket_agg[str(b)] = {
            "n": int(len(sub)),
            "mean_cost": round(float(sub["cost_of_imperfect"].mean()), 4),
            "mean_value": round(float(sub["value_over_random"].mean()), 4),
            "mean_oracle_minus_majority": round(
                float((sub["oracle_acc"] - sub["majority_baseline"]).mean()), 4
            ),
        }

    payload = {
        "total_rows": int(len(df)),
        "valid_eval_rows": int(len(valid_df)),
        "degenerate_oracle_rows": int(len(deg)),
        "k_zero_rows": int(len(k0)),
        "survey_stats": survey_stats,
        "condition_stats": cond_stats,
        "bucket_stats": bucket_agg,
        "top_cost_of_imperfect": rows_to_list(
            [
                "survey",
                "target",
                "country",
                "condition",
                "oracle_acc",
                "model_acc",
                "random_acc",
                "k_mapped",
                "cost_of_imperfect",
                "value_over_random",
            ],
            top_cost,
        ),
        "worst_value_over_random": rows_to_list(
            [
                "survey",
                "target",
                "country",
                "condition",
                "model_acc",
                "random_acc",
                "value_over_random",
                "cost_of_imperfect",
            ],
            worst_value,
        ),
        "best_value_over_random": rows_to_list(
            ["survey", "target", "country", "condition", "model_acc", "random_acc", "value_over_random"],
            best_value,
        ),
        "degenerate_rows": rows_to_list(
            ["survey", "target", "country", "condition", "k_mapped", "oracle_acc", "error"],
            deg,
        ),
        "k_zero_rows_detail": rows_to_list(
            ["survey", "target", "country", "condition", "k_mapped", "oracle_acc", "model_acc"],
            k0,
        ),
    }

    json.dump(payload, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
