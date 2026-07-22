"""
Phase B2 — Selection alignment + cross-national adaptation (Test 1 deepened, Test 2).

Motivation (see framing_and_comparisons.md): the matched-k accuracy horse-race understates
what the LLM is for. The design doc's primary selection metric is *captured importance* —
how much of the oracle's predictive-importance mass the model's chosen features recover —
and the signature test is *cross-national adaptation*: does the model request different
features per country, and do those requests align with each country's own empirical
structure? This module computes both, model-aware and leakage-aware.

DATA SOURCES (verified against on-disk schema):
  - Model feature codes per (target, country, model, condition):
        outputs/<target>_<country>/llm__<tag>/disambig.json
    a LIST of items, each with: condition, feature_rank, disambig.selected_code.
    We take selected_code per condition, drop None, dedupe in arrival order.
    (eval.json holds only accuracy stats, NOT feature codes.)
  - Oracle importances per cell: outputs/<target>_<country>/oracle.csv
    (feature_variable, importance_mean). Same code space as selected_code.
  - Leakage tags: outputs/leakage_audit.csv (leakage_class).

Metrics per (survey, target, country, model, condition):
  captured_importance (CI) = sum I[f] for f in model_features
                           / sum I[f] for top-k oracle features          (design primary)
  oracle_pctile_mean       = mean oracle-rank percentile of model features
                             (0..1, top=1; random matched-k pick ~0.5)   (complementary)

Test 2a (does it adapt at all):
  - unprompted-vs-country Jaccard of mapped feature sets per cell;
  - cross-country Jaccard per (target, model) in country_provided.
Test 2b (does adaptation track reality) — the signature test:
  per country-provided cell, captured importance of that country's model features against
  (own) the country's own oracle vs (cross) the same target's other countries' oracles.
  adaptation_score = own_CI - mean(cross_CI). Positive aggregate = country-specific picks
  fit that country's structure better than other countries'.

Outputs: outputs/alignment_by_cell.csv, outputs/alignment_summary.json,
  paper/generated_current_state/main_alignment_{overall,metrics}.tex + main_test2_adaptation.tex.

Run:  python analysis/alignment_analysis.py --write-tex
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"

from survey_features.metrics import (  # noqa: E402
    captured_importance,
    jaccard,
    load_oracle_importance as _load_oracle_importance,
    oracle_percentile_mean,
)

SURVEY_ORDER = ["wvs", "afrobarometer", "arabbarometer", "asianbarometer",
                "latinobarometer", "ess_wave_11"]
CONDITIONS = ["unprompted", "country_provided"]


def load_target_detail() -> dict[str, dict]:
    import yaml
    doc = yaml.safe_load((ROOT / "prelim" / "target_selection_detail.yaml").read_text(encoding="utf-8")) or {}
    out: dict[str, dict] = {}
    for blk in doc.get("surveys") or []:
        for s in blk.get("selected") or []:
            out[str(s["variable"])] = {"survey": blk.get("survey_id"), "bucket": s.get("bucket")}
    return out


def model_label(tag: str) -> str:
    return tag.split("_", 1)[1] if "_" in tag else tag


def load_oracle_importance(target: str, country: str) -> dict[str, float]:
    return _load_oracle_importance(target, country, OUT)


def mapped_codes(disambig_items: list[dict], condition: str) -> list[str]:
    """Selected survey codes for one condition, deduped in arrival order (None dropped)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in sorted((x for x in disambig_items if x.get("condition") == condition),
                    key=lambda x: x.get("feature_rank", 0)):
        code = (m.get("disambig") or {}).get("selected_code")
        if code and code not in seen:
            seen.add(code)
            out.append(str(code))
    return out


# captured_importance / oracle_percentile_mean / jaccard: survey_features.metrics


def load_leakage_classes() -> dict[tuple[str, str], str]:
    p = OUT / "leakage_audit.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    return {(str(r["target"]), str(r["country"])): str(r["leakage_class"]) for _, r in df.iterrows()}


def iter_cells() -> list[tuple[str, str, str, Path]]:
    """(target, country, model_tag, disambig_path) for every cell×model on disk."""
    detail = load_target_detail()
    known = sorted(detail.keys(), key=len, reverse=True)
    out = []
    for p in sorted(OUT.glob("*/llm__*/disambig.json")):
        cell = p.parent.parent.name
        tag = p.parent.name[len("llm__"):]
        target = next((t for t in known if cell.startswith(t + "_")), None)
        if target is None:
            continue
        country = cell[len(target) + 1:]
        out.append((target, country, tag, p))
    return out


def collect_rows() -> pd.DataFrame:
    detail = load_target_detail()
    leak = load_leakage_classes()
    imp_cache: dict[tuple[str, str], dict[str, float]] = {}
    model_sets: dict[tuple[str, str, str, str], set[str]] = {}
    records = []

    for target, country, tag, dpath in iter_cells():
        try:
            items = json.loads(dpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (target, country) not in imp_cache:
            imp_cache[(target, country)] = load_oracle_importance(target, country)
        imp = imp_cache[(target, country)]
        survey = detail.get(target, {}).get("survey")
        bucket = detail.get(target, {}).get("bucket")

        for cond in CONDITIONS:
            codes = mapped_codes(items, cond)
            model_sets[(target, country, tag, cond)] = set(codes)
            k = len(codes)
            records.append({
                "survey": survey, "target": target, "country": country,
                "model": tag, "model_label": model_label(tag), "condition": cond,
                "bucket": bucket, "k_mapped": k,
                "captured_importance": captured_importance(codes, imp, k),
                "oracle_pctile_mean": oracle_percentile_mean(codes, imp),
                "leakage_class": leak.get((target, country), "unknown"),
                "n_model_features": k,
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Test 2a: unprompted vs country_provided Jaccard
    up_cp = {}
    for (t, c, m, cond), s in model_sets.items():
        if cond == "unprompted":
            cp = model_sets.get((t, c, m, "country_provided"))
            if cp is not None:
                up_cp[(t, c, m)] = jaccard(s, cp)
    df["jaccard_up_vs_cp"] = df.apply(
        lambda r: up_cp.get((r["target"], r["country"], r["model"]))
        if r["condition"] == "country_provided" else np.nan, axis=1)

    # Test 2b: own vs cross-country captured importance (country_provided)
    by_tm: dict[tuple[str, str], dict[str, set]] = {}
    for (t, c, m, cond), s in model_sets.items():
        if cond == "country_provided":
            by_tm.setdefault((t, m), {})[c] = s
    adapt = {}
    for (t, m), per_country in by_tm.items():
        for c, codeset in per_country.items():
            codes = list(codeset)
            k = len(codes)
            own = captured_importance(codes, imp_cache.get((t, c), {}), k)
            cross_vals = []
            for c2, _ in per_country.items():
                if c2 == c:
                    continue
                cv = captured_importance(codes, imp_cache.get((t, c2)) or load_oracle_importance(t, c2), k)
                if cv is not None:
                    cross_vals.append(cv)
            cross = float(np.mean(cross_vals)) if cross_vals else None
            sc = (own - cross) if (own is not None and cross is not None) else None
            adapt[(t, c, m)] = (own, cross, sc)
    df["own_country_ci"] = df.apply(
        lambda r: (adapt.get((r["target"], r["country"], r["model"])) or (np.nan, np.nan, np.nan))[0]
        if r["condition"] == "country_provided" else np.nan, axis=1)
    df["cross_country_ci"] = df.apply(
        lambda r: (adapt.get((r["target"], r["country"], r["model"])) or (np.nan, np.nan, np.nan))[1]
        if r["condition"] == "country_provided" else np.nan, axis=1)
    df["adaptation_score"] = df.apply(
        lambda r: (adapt.get((r["target"], r["country"], r["model"])) or (np.nan, np.nan, np.nan))[2]
        if r["condition"] == "country_provided" else np.nan, axis=1)

    # cross-country Jaccard per (target, model)
    ccj = {}
    for (t, m), per_country in by_tm.items():
        sets = list(per_country.values())
        js = [j for j in (jaccard(a, b) for a, b in combinations(sets, 2)) if j is not None]
        if js:
            ccj[(t, m)] = float(np.mean(js))
    df["xcountry_jaccard_target"] = df.apply(
        lambda r: ccj.get((r["target"], r["model"]))
        if r["condition"] == "country_provided" else np.nan, axis=1)

    return df


def _m(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return round(float(s.mean()), 4) if len(s) else None


def summarize(df: pd.DataFrame) -> dict:
    def block(d):
        return {"n": int(d["captured_importance"].notna().sum()),
                "captured_importance": _m(d["captured_importance"]),
                "oracle_pctile_mean": _m(d["oracle_pctile_mean"])}
    valid = df[df["captured_importance"].notna()]
    noleak = valid[valid["leakage_class"] != "leakage"]
    cp = df[df["condition"] == "country_provided"]
    cp_nl = cp[cp["leakage_class"] != "leakage"]

    def share_pos(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return round(float((s > 0).mean()), 3) if len(s) else None

    return {
        "primary_metric": "captured_importance = sum(model feature importance)/sum(oracle top-k importance)",
        "overall_all": block(valid),
        "overall_excl_leakage": block(noleak),
        "by_model": {m: block(sub) for m, sub in valid.groupby("model_label")},
        "by_survey": {s: block(sub) for s, sub in valid.groupby("survey")},
        "by_condition": {c: block(sub) for c, sub in valid.groupby("condition")},
        "by_bucket": {str(b): block(sub) for b, sub in valid.groupby("bucket")},
        "test2_adaptation": {
            "mean_jaccard_unprompted_vs_country": _m(cp["jaccard_up_vs_cp"]),
            "mean_xcountry_jaccard_per_target": _m(cp["xcountry_jaccard_target"]),
            "mean_own_country_ci": _m(cp["own_country_ci"]),
            "mean_cross_country_ci": _m(cp["cross_country_ci"]),
            "mean_adaptation_score": _m(cp["adaptation_score"]),
            "share_adaptation_positive": share_pos(cp["adaptation_score"]),
            "n_adaptation_cells": int(cp["adaptation_score"].notna().sum()),
            "excl_leakage": {
                "mean_own_country_ci": _m(cp_nl["own_country_ci"]),
                "mean_cross_country_ci": _m(cp_nl["cross_country_ci"]),
                "mean_adaptation_score": _m(cp_nl["adaptation_score"]),
                "share_adaptation_positive": share_pos(cp_nl["adaptation_score"]),
                "n_adaptation_cells": int(cp_nl["adaptation_score"].notna().sum()),
            },
        },
    }


def write_tex(df: pd.DataFrame, summ: dict) -> None:
    """Emit LaTeX tables. All \\begin/\\end/brace scaffolding stays in plain (non-f)
    strings; f-strings carry only numeric content (no literal braces)."""
    gen = ROOT / "paper" / "generated_current_state"
    gen.mkdir(parents=True, exist_ok=True)
    EOL = " " + chr(92) + chr(92)        # ' \\'  (LaTeX row terminator)
    USC = chr(92) + "_"                  # '\_'

    def num(x):
        return "-" if x is None else f"{x:.4f}"

    def tabular(colspec: str, header: str, body_rows: list[str]) -> str:
        return (
            "\\begin{tabular}{" + colspec + "}\n\\toprule\n"
            + header + EOL + "\n\\midrule\n"
            + "\n".join(r + EOL for r in body_rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )

    # Captured importance by survey x model
    valid = df[df["captured_importance"].notna()]
    models = sorted(valid["model_label"].unique())
    rows = []
    for s in SURVEY_ORDER:
        sub = valid[valid["survey"] == s]
        if sub.empty:
            continue
        cells = " & ".join(num(_m(sub[sub["model_label"] == m]["captured_importance"])) for m in models)
        rows.append(s.replace("_", USC) + " & " + cells)
    (gen / "main_alignment_metrics.tex").write_text(
        tabular("l" + "r" * len(models), "Survey & " + " & ".join(models), rows), encoding="utf-8")

    # Overall: model columns, all vs excl-leakage as separate rows.
    def col(sub, fld):
        return num(_m(sub[fld]))
    noleak = valid[valid["leakage_class"] != "leakage"]
    ov_rows = []
    for fld, lbl in [("captured_importance", "Captured importance"),
                     ("oracle_pctile_mean", "Oracle percentile")]:
        ov_rows.append(lbl + " (all) & " + " & ".join(col(valid[valid["model_label"] == m], fld) for m in models))
        ov_rows.append(lbl + " (excl.\\ leakage) & "
                       + " & ".join(col(noleak[noleak["model_label"] == m], fld) for m in models))
    ov_rows.append("N rows (all / excl.\\ leakage) & "
                   + " & ".join(f"{int((valid['model_label'] == m).sum())} / {int((noleak['model_label'] == m).sum())}"
                                for m in models))
    (gen / "main_alignment_overall.tex").write_text(
        tabular("l" + "r" * len(models), "Metric & " + " & ".join(models), ov_rows), encoding="utf-8")

    # Test 2 adaptation: model columns.
    cp = df[df["condition"] == "country_provided"]
    cp_nl = cp[cp["leakage_class"] != "leakage"]

    def cpcol(fld, frame=cp):
        return " & ".join(num(_m(frame[frame["model_label"] == m][fld])) for m in models)

    def cpshare(frame=cp):
        out = []
        for m in models:
            s = pd.to_numeric(frame[frame["model_label"] == m]["adaptation_score"], errors="coerce").dropna()
            out.append(num(round(float((s > 0).mean()), 3)) if len(s) else "-")
        return " & ".join(out)

    t2_rows = [
        "Jaccard, unprompted vs country & " + cpcol("jaccard_up_vs_cp"),
        "Cross-country Jaccard (per target) & " + cpcol("xcountry_jaccard_target"),
        "Own-country captured importance & " + cpcol("own_country_ci"),
        "Cross-country captured importance & " + cpcol("cross_country_ci"),
        "Adaptation score (own $-$ cross) & " + cpcol("adaptation_score"),
        "\\quad excl.\\ leakage & " + cpcol("adaptation_score", cp_nl),
        "Share adaptation score $>0$ & " + cpshare(),
    ]
    (gen / "main_test2_adaptation.tex").write_text(
        tabular("l" + "r" * len(models),
                "Cross-national adaptation (country-provided) & " + " & ".join(models), t2_rows),
        encoding="utf-8")
    print(f"Wrote alignment TeX to {gen}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-tex", action="store_true")
    args = ap.parse_args()

    df = collect_rows()
    if df.empty:
        print("No disambig.json cells found under outputs/.", file=sys.stderr)
        sys.exit(1)
    df.to_csv(OUT / "alignment_by_cell.csv", index=False)
    summ = summarize(df)
    (OUT / "alignment_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(f"Wrote outputs/alignment_by_cell.csv ({len(df)} rows) and alignment_summary.json")
    if args.write_tex:
        write_tex(df, summ)


if __name__ == "__main__":
    main()
