"""
Grid analysis for the 5 targets x 5 countries Phase 0b run.

Reads from ``outputs/``:
  - grid_summary.csv          : one row per (target, country, condition)
  - {TARGET}_{COUNTRY}_disambig.json : per-cell LLM feature selection + mapping
  - {TARGET}_{COUNTRY}_oracle.csv    : per-cell permutation importance table

Produces the tables and numbers used by phase0b_grid_findings.md.

Run:
    python -m analysis.grid_analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GRID_CSV = OUTPUTS_DIR / "grid_summary.csv"

TARGETS = ["Q47", "Q57", "Q199", "Q235", "Q164"]
COUNTRIES = ["Germany", "Nigeria", "Japan", "Brazil", "Egypt"]

TARGET_LABELS = {
    "Q47":  "Self-rated health",
    "Q57":  "Interpersonal trust",
    "Q199": "Political interest",
    "Q235": "Strong leader",
    "Q164": "Importance of God",
}

DEMOGRAPHIC_TERMS = {
    "age", "gender", "sex", "education", "educational",
    "income", "household income", "socioeconomic", "ses",
    "employment", "occupation", "marital", "marriage",
    "race", "ethnic", "ethnicity", "religion", "religious affiliation",
    "urban", "rural", "region", "geographic", "country", "nationality",
    "class", "social class",
}


def load_grid_summary(path: Path = GRID_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cell"] = df["target"] + "_" + df["country"]
    return df


def load_disambig(target: str, country: str) -> list[dict]:
    path = OUTPUTS_DIR / f"{target}_{country}_disambig.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_oracle(target: str, country: str) -> pd.DataFrame | None:
    path = OUTPUTS_DIR / f"{target}_{country}_oracle.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def oracle_top_k(oracle_df: pd.DataFrame, k: int) -> list[str]:
    return (
        oracle_df.sort_values("importance_mean", ascending=False)
        .head(k)["feature_variable"]
        .tolist()
    )


def headline_table(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["unmappable_rate"] = 1.0 - (df["k_mapped"] / df["k_requested"])
    cols = [
        "target", "country", "condition",
        "k_requested", "k_mapped", "unmappable_rate",
        "majority_baseline",
        "oracle_acc", "model_acc", "random_acc",
        "cost_of_imperfect", "value_over_random",
    ]
    return df[cols].round(4)


def headline_aggregate(summary: pd.DataFrame) -> dict:
    df = summary.dropna(subset=["oracle_acc", "model_acc", "random_acc"])
    return {
        "n_cells_rows": int(len(df)),
        "mean_oracle": round(df["oracle_acc"].mean(), 4),
        "mean_model": round(df["model_acc"].mean(), 4),
        "mean_random": round(df["random_acc"].mean(), 4),
        "mean_majority": round(df["majority_baseline"].mean(), 4),
        "mean_cost": round(df["cost_of_imperfect"].mean(), 4),
        "mean_value": round(df["value_over_random"].mean(), 4),
        "share_model_beats_random": round(
            float((df["value_over_random"] > 0).mean()), 3
        ),
        "share_model_beats_majority": round(
            float((df["model_acc"] > df["majority_baseline"]).mean()), 3
        ),
        "share_oracle_beats_majority": round(
            float((df["oracle_acc"] > df["majority_baseline"]).mean()), 3
        ),
    }


def per_target_rollup(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.dropna(subset=["cost_of_imperfect", "value_over_random"])
    g = df.groupby("target").agg(
        n_rows=("cell", "count"),
        mean_cost=("cost_of_imperfect", "mean"),
        std_cost=("cost_of_imperfect", "std"),
        mean_value=("value_over_random", "mean"),
        std_value=("value_over_random", "std"),
        mean_oracle=("oracle_acc", "mean"),
        mean_model=("model_acc", "mean"),
        mean_random=("random_acc", "mean"),
        share_value_positive=("value_over_random", lambda s: float((s > 0).mean())),
    )
    g["target_label"] = g.index.map(TARGET_LABELS)
    g = g.reset_index()
    return g.round(4)


def condition_effect(summary: pd.DataFrame) -> dict:
    pivot = summary.pivot_table(
        index=["target", "country"], columns="condition",
        values=["model_acc", "cost_of_imperfect", "value_over_random", "k_mapped"],
    )
    pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    pivot["delta_model_acc"] = (
        pivot["model_acc__country_provided"] - pivot["model_acc__unprompted"]
    )
    pivot["delta_cost"] = (
        pivot["cost_of_imperfect__country_provided"]
        - pivot["cost_of_imperfect__unprompted"]
    )
    pivot["delta_value"] = (
        pivot["value_over_random__country_provided"]
        - pivot["value_over_random__unprompted"]
    )
    pivot["delta_k_mapped"] = (
        pivot["k_mapped__country_provided"] - pivot["k_mapped__unprompted"]
    )

    clean = pivot.dropna(subset=["delta_model_acc"])
    return {
        "pivot": pivot.round(4),
        "mean_delta_model_acc": round(clean["delta_model_acc"].mean(), 4),
        "median_delta_model_acc": round(float(clean["delta_model_acc"].median()), 4),
        "share_cp_beats_up": round(float((clean["delta_model_acc"] > 0).mean()), 3),
        "mean_delta_k_mapped": round(clean["delta_k_mapped"].mean(), 3),
        "n_cells": int(len(clean)),
    }


def signal_vs_gap(summary: pd.DataFrame) -> dict:
    df = summary.dropna(
        subset=["oracle_acc", "majority_baseline", "cost_of_imperfect"]
    ).copy()
    df["signal"] = df["oracle_acc"] - df["majority_baseline"]
    df["gap"] = df["cost_of_imperfect"]

    corr = df[["signal", "gap"]].corr().iloc[0, 1]
    rank = df[["signal", "gap"]].rank()
    corr_spearman = rank.corr().iloc[0, 1]

    return {
        "pearson": round(float(corr), 3),
        "spearman": round(float(corr_spearman), 3),
        "n": int(len(df)),
        "scatter": df[["target", "country", "condition", "signal", "gap"]].round(4),
    }


def _mapped_codes(cell_maps: Iterable[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in cell_maps:
        code = m.get("disambig", {}).get("selected_code")
        if code is None or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def hit_rate_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        target, country, cond = r["target"], r["country"], r["condition"]
        k = int(r["k_mapped"]) if pd.notna(r["k_mapped"]) else 0
        disambig = load_disambig(target, country)
        oracle_df = load_oracle(target, country)
        if k == 0 or oracle_df is None or not disambig:
            rows.append({
                "target": target, "country": country, "condition": cond,
                "k_mapped": k, "n_hits": 0, "hit_rate": None,
                "hit_codes": [],
            })
            continue
        cell_maps = [m for m in disambig if m["condition"] == cond]
        model_codes = _mapped_codes(cell_maps)
        top_k = set(oracle_top_k(oracle_df, k))
        hits = [c for c in model_codes if c in top_k]
        rows.append({
            "target": target, "country": country, "condition": cond,
            "k_mapped": k, "n_hits": len(hits),
            "hit_rate": round(len(hits) / k, 3) if k > 0 else None,
            "hit_codes": hits,
        })
    return pd.DataFrame(rows)


def hit_rate_by_target(hit_df: pd.DataFrame) -> pd.DataFrame:
    df = hit_df.dropna(subset=["hit_rate"])
    g = df.groupby("target").agg(
        n_rows=("hit_rate", "count"),
        mean_hit_rate=("hit_rate", "mean"),
        total_hits=("n_hits", "sum"),
        total_k=("k_mapped", "sum"),
    )
    g["pooled_hit_rate"] = (g["total_hits"] / g["total_k"]).round(3)
    g["target_label"] = g.index.map(TARGET_LABELS)
    return g.reset_index().round(4)


def unmappable_rate_from_disambig() -> pd.DataFrame:
    """Read mapping rates directly from the full disambig JSONs. The grid
    summary's ``k_requested`` already reflects post-dedup non-None codes, so we
    recompute from scratch: for each cell x condition, how many of the LLM's
    originally named features failed to map (selected_code is None) or were
    dropped as duplicates."""
    rows = []
    for target in TARGETS:
        for country in COUNTRIES:
            disambig = load_disambig(target, country)
            if not disambig:
                continue
            for cond in ("unprompted", "country_provided"):
                cell_maps = [m for m in disambig if m["condition"] == cond]
                if not cell_maps:
                    continue
                n_asked = len(cell_maps)
                codes = [m.get("disambig", {}).get("selected_code")
                         for m in cell_maps]
                n_none = sum(1 for c in codes if c is None)
                unique_non_none = len({c for c in codes if c is not None})
                rows.append({
                    "target": target,
                    "country": country,
                    "condition": cond,
                    "n_asked": n_asked,
                    "n_unmappable": n_none,
                    "n_duplicates_dropped": n_asked - n_none - unique_non_none,
                    "n_usable": unique_non_none,
                    "unmappable_rate": round(n_none / n_asked, 3),
                })
    return pd.DataFrame(rows)


def unmappable_profile(summary: pd.DataFrame | None = None) -> dict:
    df = unmappable_rate_from_disambig()
    by_target = df.groupby("target")["unmappable_rate"].mean().round(3).to_dict()
    by_country = df.groupby("country")["unmappable_rate"].mean().round(3).to_dict()
    by_condition = df.groupby("condition")["unmappable_rate"].mean().round(3).to_dict()

    worst = (
        df.sort_values("unmappable_rate", ascending=False)
          .head(8)[["target", "country", "condition",
                    "n_asked", "n_unmappable", "n_duplicates_dropped",
                    "n_usable", "unmappable_rate"]]
          .reset_index(drop=True)
    )

    return {
        "overall_mean": round(float(df["unmappable_rate"].mean()), 3),
        "total_asked": int(df["n_asked"].sum()),
        "total_unmappable": int(df["n_unmappable"].sum()),
        "total_dup_dropped": int(df["n_duplicates_dropped"].sum()),
        "by_target": by_target,
        "by_country": by_country,
        "by_condition": by_condition,
        "worst_cells": worst,
    }


def _is_demographic(label: str) -> bool:
    if not label:
        return False
    low = label.lower()
    return any(term in low for term in DEMOGRAPHIC_TERMS)


def demographics_lead_table() -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for country in COUNTRIES:
            disambig = load_disambig(target, country)
            if not disambig:
                continue
            for cond in ("unprompted", "country_provided"):
                cond_maps = sorted(
                    [m for m in disambig if m["condition"] == cond],
                    key=lambda m: m.get("feature_rank", 0),
                )
                if not cond_maps:
                    continue
                first = cond_maps[0]
                label = first.get("feature_label", "")
                rows.append({
                    "target": target,
                    "country": country,
                    "condition": cond,
                    "rank0_label": label,
                    "rank0_is_demographic": _is_demographic(label),
                })
    return pd.DataFrame(rows)


def demographics_lead_summary(lead_df: pd.DataFrame) -> dict:
    df = lead_df.copy()
    return {
        "n_cells": int(len(df)),
        "share_rank0_demographic": round(
            float(df["rank0_is_demographic"].mean()), 3
        ),
        "by_condition": df.groupby("condition")["rank0_is_demographic"]
            .mean().round(3).to_dict(),
        "by_target": df.groupby("target")["rank0_is_demographic"]
            .mean().round(3).to_dict(),
    }


@dataclass
class CellReport:
    target: str
    country: str
    condition: str
    oracle_top: list
    model_features: list
    hits: list


def cell_report(target: str, country: str, condition: str, k: int | None = None) -> CellReport:
    disambig = load_disambig(target, country)
    oracle_df = load_oracle(target, country)
    cell_maps = sorted(
        [m for m in disambig if m["condition"] == condition],
        key=lambda m: m.get("feature_rank", 0),
    )

    seen: set[str] = set()
    model_features = []
    for m in cell_maps:
        code = m.get("disambig", {}).get("selected_code")
        if code is None or code in seen:
            continue
        seen.add(code)
        model_features.append((code, m.get("feature_label", "")))

    if k is None:
        k = len(model_features)

    top = oracle_df.sort_values("importance_mean", ascending=False).head(max(k, 10))
    oracle_top = list(zip(top["feature_variable"].tolist(),
                          top["importance_mean"].round(4).tolist()))

    oracle_top_k_codes = set(c for c, _ in oracle_top[:k])
    hits = [c for c, _ in model_features if c in oracle_top_k_codes]

    return CellReport(
        target=target, country=country, condition=condition,
        oracle_top=oracle_top, model_features=model_features, hits=hits,
    )


def pick_extreme_cells(summary: pd.DataFrame) -> tuple[dict, dict]:
    """High = largest cost_of_imperfect; low = smallest non-negative cost among
    cells where oracle has at least 5pp lift over majority (to avoid picking
    flat cells where any feature set looks the same) and k_mapped >= 3."""
    df = summary.dropna(subset=["cost_of_imperfect", "k_mapped"]).copy()
    df = df[df["k_mapped"] >= 3]
    df["signal"] = df["oracle_acc"] - df["majority_baseline"]
    high = df.sort_values("cost_of_imperfect", ascending=False).iloc[0]
    eligible = df[(df["signal"] >= 0.05) & (df["cost_of_imperfect"] >= 0)]
    low = eligible.sort_values("cost_of_imperfect", ascending=True).iloc[0]
    return (
        {"target": high["target"], "country": high["country"],
         "condition": high["condition"], "cost": round(high["cost_of_imperfect"], 4)},
        {"target": low["target"], "country": low["country"],
         "condition": low["condition"], "cost": round(low["cost_of_imperfect"], 4)},
    )


def df_to_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return df.to_string(index=False)


def run_all() -> dict:
    summary = load_grid_summary()

    head_tbl = headline_table(summary)
    head_agg = headline_aggregate(summary)
    per_target = per_target_rollup(summary)
    cond_eff = condition_effect(summary)
    sig_gap = signal_vs_gap(summary)
    hit_df = hit_rate_table(summary)
    hit_by_tgt = hit_rate_by_target(hit_df)
    unmap = unmappable_profile(summary)
    demo_tbl = demographics_lead_table()
    demo_sum = demographics_lead_summary(demo_tbl)
    high, low = pick_extreme_cells(summary)
    high_report = cell_report(high["target"], high["country"], high["condition"])
    low_report = cell_report(low["target"], low["country"], low["condition"])

    return {
        "n_rows_in_summary": len(summary),
        "headline_table": head_tbl,
        "headline_aggregate": head_agg,
        "per_target": per_target,
        "condition_effect": cond_eff,
        "signal_vs_gap": sig_gap,
        "hit_rate_table": hit_df,
        "hit_rate_by_target": hit_by_tgt,
        "unmappable_profile": unmap,
        "demographics_lead_table": demo_tbl,
        "demographics_lead_summary": demo_sum,
        "extreme_cells": {"high_cost": high, "low_cost": low},
        "high_cost_report": high_report,
        "low_cost_report": low_report,
    }


def _print_report(results: dict) -> None:
    print(f"Rows in grid_summary.csv: {results['n_rows_in_summary']}")
    print("\n== Headline aggregate ==")
    for k, v in results["headline_aggregate"].items():
        print(f"  {k:30s} {v}")

    print("\n== Per-target roll-up ==")
    print(df_to_md(results["per_target"]))

    print("\n== Condition effect ==")
    for k, v in results["condition_effect"].items():
        if k == "pivot":
            continue
        print(f"  {k:30s} {v}")

    print("\n== Signal vs gap ==")
    sg = results["signal_vs_gap"]
    print(f"  pearson  = {sg['pearson']}")
    print(f"  spearman = {sg['spearman']}")
    print(f"  n        = {sg['n']}")

    print("\n== Hit-rate by target ==")
    print(df_to_md(results["hit_rate_by_target"]))

    print("\n== Unmappable profile ==")
    um = results["unmappable_profile"]
    print(f"  overall mean: {um['overall_mean']}")
    print(f"  by target:    {um['by_target']}")
    print(f"  by country:   {um['by_country']}")
    print(f"  by condition: {um['by_condition']}")
    print("  worst cells:")
    print(df_to_md(um["worst_cells"]))

    print("\n== Demographics-lead ==")
    ds = results["demographics_lead_summary"]
    print(f"  n_cells: {ds['n_cells']}")
    print(f"  share_rank0_demographic: {ds['share_rank0_demographic']}")
    print(f"  by_condition: {ds['by_condition']}")
    print(f"  by_target:    {ds['by_target']}")

    print("\n== Extreme cells ==")
    print(f"  high cost: {results['extreme_cells']['high_cost']}")
    print(f"  low cost:  {results['extreme_cells']['low_cost']}")

    print("\n  HIGH-COST cell detail:")
    r = results["high_cost_report"]
    print(f"   {r.target} x {r.country} ({r.condition})")
    print(f"   oracle top: {r.oracle_top}")
    print(f"   model features: {r.model_features}")
    print(f"   hits: {r.hits}")

    print("\n  LOW-COST cell detail:")
    r = results["low_cost_report"]
    print(f"   {r.target} x {r.country} ({r.condition})")
    print(f"   oracle top: {r.oracle_top}")
    print(f"   model features: {r.model_features}")
    print(f"   hits: {r.hits}")


if __name__ == "__main__":
    results = run_all()
    _print_report(results)
