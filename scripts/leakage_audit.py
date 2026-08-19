"""
Leakage audit — the grid screen for the oracle ground truth.

Problem: some targets are demographics (e.g. WVS Q263 "are you an immigrant?",
ESS rtrd "were you retired?") whose oracle top feature is a near-duplicate of the
target itself (Q266 "in which country were you born?"; mnactic "main activity").
On these cells the oracle "wins" by exploiting a single label-in-disguise column.
That is empirical leakage, NOT predictive understanding. The oracle's own semantic
filter (cosine > 0.85 on question text) does not catch it because the wordings are
not lexically similar.

Self-contained and type-matched (2026-08-19): every number the screen uses comes
from the cell's own oracle.csv / oracle_meta.json plus a FRESH typed probe run
here — never from historical score files or a previous audit. Ordinal/continuous
targets are probed with the regression evaluator (Spearman rho); binary/nominal
with the classifier (accuracy + log loss). No ordinal is ever scored as
multiclass accuracy.

Signals per cell:

  (A) Importance concentration (offline, always available): share of positive
      score-split importance mass held by the top select-ranked feature, + HHI.
  (B) Typed probe (--with-data, needs DATA_CONFIG_PATH): the SAME downstream
      evaluator on (i) the oracle top-k and (ii) ONLY the top feature, both in
      the cell's own primary metric. Cached per cell at
      cache/cells/<cell>/audit_probe.json, keyed by
      {contract_version, eval_metric, top_feature, oracle_k, target_type} — the
      cache expires exactly when the oracle it describes changes.

Classes (rules live in survey_features.grid_screen — the classifier module):
  - unestimable         : type-1 — thin data on the honest split (classification:
                          V2 / ranking-fold minority; regression: n_score or scale
                          points) or oracle_ceiling@5 below the floor.
  - leakage             : near-duplicate — absolute (one column recovers the target
                          near-deterministically, regardless of oracle-side numbers)
                          or relative (single feature recovers >= --recover-frac of
                          oracle PI), with concentration >= --conc-thresh.
  - leakage_distributed : implausibly high oracle PI with spread importance
                          (skip-pattern module, e.g. the Q67A climate battery).
  - leakage_suspect     : concentration >= --conc-thresh with no probe available
                          (offline mode / probe failure) — never cleared to genuine.
  - genuine             : keep, including type-2/3 (tiny accuracy lift / acc below
                          the mode — not drop rules; see pre_paper_run_decisions.md).

Default grid = leakage_class == genuine (`layout.genuine_cells()`).

Outputs:
  outputs/cache/audits/leakage_audit.csv          one row per (survey, target, country)
  outputs/cache/audits/leakage_audit_summary.json rollups by survey / bucket / class
  paper/generated_current_state/leakage_audit_longtable.tex (if --write-tex)

Run (offline — type-1 + concentration-suspect only):
  python scripts/leakage_audit.py
Run (full, typed probes — required for real leakage classification):
  python scripts/leakage_audit.py --with-data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from survey_features.config import OUTPUTS_DIR, PAPER_DIR  # noqa: E402
from survey_features.grid_screen import (  # noqa: E402
    ScreenThresholds,
    TYPE1_MIN_CEILING_AT_5,
    TYPE1_MIN_V2_MINORITY,
    classify_cell,
    estimated_minority,
    is_classification,
    rank_holdout_size,
)
from survey_features.layout import (  # noqa: E402
    cache_cells_dir,
    cell_dir,
    leakage_audit_write_paths,
)

OUT = OUTPUTS_DIR

DEFAULT_ORACLE_K = 10


def load_target_detail() -> dict[str, dict]:
    """Target catalog from data/targets.yaml (survey, bucket, section, topic)."""
    import yaml

    path = ROOT / "data" / "targets.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, dict] = {}
    for blk in doc.get("surveys") or []:
        sid = blk.get("survey_id")
        for s in blk.get("selected") or []:
            out[str(s["variable"])] = {
                "survey": sid,
                "bucket": s.get("bucket"),
                "section": s.get("section"),
                "topic_key": s.get("topic_key"),
            }
    return out


def survey_of(target: str, detail: dict[str, dict]) -> str | None:
    info = detail.get(target)
    return info["survey"] if info else None


def iter_oracle_cells() -> list[tuple[str, str, Path]]:
    """Return (target, country, oracle_csv_path) for every cell, resolving target
    via known target codes (targets may contain underscores)."""
    detail = load_target_detail()
    known = sorted(detail.keys(), key=len, reverse=True)
    cells = []
    seen: set[tuple[str, str]] = set()
    search_roots = [cache_cells_dir(OUT)]
    for root in search_roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*/oracle.csv")):
            cell = p.parent.name
            target = next((t for t in known if cell.startswith(t + "_")), None)
            if target is None:
                continue
            country = cell[len(target) + 1 :]
            key = (target, country)
            if key in seen:
                continue
            seen.add(key)
            cells.append((target, country, p))
    return cells


def concentration_signal(oracle_csv: Path) -> dict:
    """Top select-ranked feature + its share of positive score-split mass.

    The top feature is what downstream top-k readers rank on (importance_select);
    the share/HHI stay on the unbiased score-split values (importance_mean alias).
    """
    df = pd.read_csv(oracle_csv)
    rank_col = "importance_select" if "importance_select" in df.columns else "importance_mean"
    df = df.sort_values(rank_col, ascending=False).reset_index(drop=True)
    majority = float(df["majority_baseline"].iloc[0]) if "majority_baseline" in df and len(df) else np.nan
    pos = df["importance_mean"].clip(lower=0.0)
    total = float(pos.sum())
    top_feat = str(df["feature_variable"].iloc[0]) if len(df) else None
    top_share = float(pos.iloc[0] / total) if total > 0 else 0.0
    # Herfindahl of importance mass: 1.0 = one feature owns everything.
    hhi = float(((pos / total) ** 2).sum()) if total > 0 else 0.0
    return {
        "top_feature": top_feat,
        "top_importance_share": top_share,
        "importance_hhi": hhi,
        "majority_baseline": majority,
        "n_features_scored": int(len(df)),
    }


def oracle_topk_codes(oracle_csv: Path, k: int) -> list[str]:
    df = pd.read_csv(oracle_csv)
    rank_col = "importance_select" if "importance_select" in df.columns else "importance_mean"
    df = df.sort_values(rank_col, ascending=False)
    return df["feature_variable"].astype(str).head(k).tolist()


def load_oracle_meta(oracle_csv: Path) -> dict:
    path = oracle_csv.with_name("oracle_meta.json")
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def attach_type1_fields(rec: dict, meta: dict) -> None:
    """Copy split sizes / ceiling / typing from oracle_meta for the type-1 screen."""
    rec["problem_type"] = meta.get("problem_type")
    rec["target_type"] = meta.get("target_type")
    rec["n_score"] = meta.get("n_score")
    rec["n_eval_reserve"] = meta.get("n_eval_reserve")
    rec["cv_folds"] = meta.get("cv_folds")
    rec["fold_fit_sizes"] = meta.get("fold_fit_sizes")  # classify-only; not a CSV column
    rec["n_target_unique"] = meta.get("n_target_unique")
    rec["ceiling_at_5"] = (meta.get("oracle_ceiling") or {}).get("5")
    rec["primary_metric"] = "accuracy" if is_classification(rec) else "spearman"
    rec["v2_minority_est"] = estimated_minority(
        rec.get("n_score"), rec.get("majority_baseline")
    )
    rec["rank_holdout_minority_est"] = estimated_minority(
        rank_holdout_size(rec), rec.get("majority_baseline")
    )


# ---- Typed per-cell probe (data-backed mode) ----------------------------------

def _probe_cache_key(meta: dict, top_feature: str, oracle_k: int) -> dict:
    return {
        "contract_version": meta.get("contract_version"),
        "eval_metric": meta.get("eval_metric"),
        "top_feature": top_feature,
        "oracle_k": oracle_k,
        "target_type": meta.get("target_type"),
    }


def probe_cell(
    rec: dict,
    oracle_csv: Path,
    meta: dict,
    cfg_path: str,
    survey_cache: dict,
    oracle_k: int,
    use_cache: bool = True,
) -> dict | None:
    """Fresh typed downstream probe: oracle top-k and top-feature-only, both in
    the cell's own primary metric. Cached per cell; the key pins the oracle it
    describes, so a recomputed oracle invalidates it automatically."""
    from survey_features.evaluation import REGRESSION_TYPES, evaluate_feature_set
    from survey_features.surveys import SURVEY_COUNTRY_COL, build_country_code_map, load_survey

    survey_id, target, country = rec["survey"], rec["target"], rec["country"]
    top_feature = rec.get("top_feature")
    if not survey_id or not top_feature:
        return None

    cache_path = cell_dir(target, country, OUT) / "audit_probe.json"
    key = _probe_cache_key(meta, top_feature, oracle_k)
    if use_cache and cache_path.is_file():
        try:
            stored = json.loads(cache_path.read_text(encoding="utf-8"))
            if stored.get("key") == key:
                return stored["values"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    if survey_id not in survey_cache:
        data, smeta = load_survey(survey_id, cfg_path)
        ccol = SURVEY_COUNTRY_COL.get(survey_id)
        cmap = build_country_code_map(smeta, ccol, data) if ccol else {}
        survey_cache[survey_id] = (data, smeta, ccol, cmap)
    data, smeta, ccol, cmap = survey_cache[survey_id]
    if top_feature not in data.columns or target not in data.columns:
        return None
    sub = data
    if ccol and ccol in data.columns and country in cmap:
        sub = data[data[ccol] == cmap[country]]
    if len(sub) < 30:
        return None

    ttype = rec.get("target_type")
    typed = ttype in REGRESSION_TYPES
    topk = [c for c in oracle_topk_codes(oracle_csv, oracle_k) if c in sub.columns]
    if not topk:
        return None

    def _primary(res: dict | None) -> float | None:
        if not res or res.get("error"):
            return None
        v = res.get("spearman_mean") if typed else res.get("accuracy_mean")
        return None if v is None else float(v)

    try:
        oracle_res = evaluate_feature_set(
            sub, target, topk, target_type=ttype if typed else None
        )
        sf_res = evaluate_feature_set(
            sub, target, [top_feature], target_type=ttype if typed else None
        )
    except Exception as e:  # per-cell isolation: one broken cell must not kill the sweep
        print(f"  probe failed for {target}/{country}: {e}", file=sys.stderr)
        return None

    values = {
        "oracle_primary": _primary(oracle_res),
        "oracle_logloss": (oracle_res or {}).get("logloss_mean") if not typed else None,
        "single_feature_primary": _primary(sf_res),
    }
    if values["oracle_primary"] is None and values["single_feature_primary"] is None:
        return None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"key": key, "values": values}, indent=1),
                              encoding="utf-8")
    except OSError:
        pass
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description="Typed leakage audit / grid screen.")
    ap.add_argument("--with-data", action="store_true",
                    help="Run the typed probes (needs DATA_CONFIG_PATH). Without it, "
                         "concentrated cells degrade to leakage_suspect.")
    ap.add_argument("--oracle-k", type=int, default=DEFAULT_ORACLE_K,
                    help="Top-k features for the oracle-side probe (default 10).")
    ap.add_argument("--no-probe-cache", action="store_true",
                    help="Ignore cached audit_probe.json values and recompute.")
    ap.add_argument("--conc-thresh", type=float, default=0.80,
                    help="Top-feature importance share above which a cell is concentration-suspect.")
    ap.add_argument("--recover-frac", type=float, default=0.90,
                    help="Single feature recovers >= this fraction of oracle PI -> leakage "
                         "(only in conjunction with --conc-thresh).")
    ap.add_argument("--implausible-acc", type=float, default=0.95,
                    help="Classification: oracle accuracy at or above this, with lift >= 0.20 "
                         "over majority, is distributed (module/skip-pattern) leakage.")
    ap.add_argument("--implausible-rho", type=float, default=0.95,
                    help="Regression: oracle Spearman at or above this is distributed leakage.")
    ap.add_argument("--abs-dup-acc", type=float, default=0.90,
                    help="Classification absolute near-duplicate: single-feature accuracy at or "
                         "above this (and >= 0.20 over majority) with high concentration -> "
                         "leakage regardless of oracle-side numbers.")
    ap.add_argument("--abs-dup-rho", type=float, default=0.90,
                    help="Regression absolute near-duplicate: single-feature Spearman at or "
                         "above this with high concentration -> leakage.")
    ap.add_argument("--min-v2-minority", type=float, default=TYPE1_MIN_V2_MINORITY,
                    help="Type-1 (classification): expected non-mode rows on V2 below this "
                         "-> unestimable. Not an accuracy-vs-majority filter.")
    ap.add_argument("--min-ceiling-at-5", type=float, default=TYPE1_MIN_CEILING_AT_5,
                    help="Type-1: oracle_ceiling@5 below this -> unestimable (compromised ranking).")
    ap.add_argument("--write-tex", action="store_true", help="Also emit a LaTeX longtable.")
    args = ap.parse_args()

    thresholds = ScreenThresholds(
        conc_thresh=args.conc_thresh,
        recover_frac=args.recover_frac,
        implausible_acc=args.implausible_acc,
        implausible_rho=args.implausible_rho,
        abs_dup_acc=args.abs_dup_acc,
        abs_dup_rho=args.abs_dup_rho,
        min_v2_minority=args.min_v2_minority,
        min_ceiling_at_5=args.min_ceiling_at_5,
    )

    detail = load_target_detail()
    cells = iter_oracle_cells()
    if not cells:
        print("No oracle.csv cells found under outputs/.", file=sys.stderr)
        sys.exit(1)

    cfg = None
    if args.with_data:
        from dotenv import dotenv_values
        cfg = dotenv_values(ROOT / ".env").get("DATA_CONFIG_PATH") or os.environ.get("DATA_CONFIG_PATH")
        if not cfg or not os.path.isfile(cfg):
            print(f"--with-data set but DATA_CONFIG_PATH missing/invalid ({cfg!r}); "
                  "running offline (concentrated cells degrade to leakage_suspect).",
                  file=sys.stderr)
            args.with_data = False

    survey_cache: dict = {}
    records = []
    for target, country, ocsv in cells:
        rec = {"survey": survey_of(target, detail), "target": target, "country": country}
        info = detail.get(target, {})
        rec["bucket"] = info.get("bucket")
        rec["section"] = info.get("section")
        rec.update(concentration_signal(ocsv))
        meta = load_oracle_meta(ocsv)
        attach_type1_fields(rec, meta)

        rec["oracle_primary"] = None
        rec["oracle_logloss"] = None
        rec["single_feature_primary"] = None
        rec["audit_mode"] = "offline"
        if args.with_data:
            values = probe_cell(rec, ocsv, meta, cfg, survey_cache,
                                oracle_k=args.oracle_k,
                                use_cache=not args.no_probe_cache)
            if values:
                rec.update(values)
                rec["audit_mode"] = "data"

        rec["leakage_class"] = classify_cell(rec, thresholds)
        # classify_cell writes unestimable_reason / single_feature_recovery in place.
        rec.setdefault("unestimable_reason", None)
        rec.setdefault("single_feature_recovery", None)
        if is_classification(rec):
            op, mb = rec.get("oracle_primary"), rec.get("majority_baseline")
            rec["oracle_lift"] = (op - mb) if (op is not None and mb is not None
                                              and not pd.isna(mb)) else None
        else:
            rec["oracle_lift"] = rec.get("oracle_primary")
        rec.pop("fold_fit_sizes", None)  # classify-only; keep the CSV flat
        records.append(rec)

    rows = pd.DataFrame(records)
    rows = rows.sort_values(["leakage_class", "oracle_lift"], ascending=[True, False])

    out_csv, out_summary = leakage_audit_write_paths(OUT)
    cols = ["survey", "target", "country", "bucket", "section",
            "problem_type", "target_type", "primary_metric",
            "majority_baseline", "n_score", "n_eval_reserve", "cv_folds",
            "ceiling_at_5", "v2_minority_est", "rank_holdout_minority_est",
            "n_target_unique",
            "top_feature", "top_importance_share", "importance_hhi",
            "n_features_scored",
            "oracle_primary", "oracle_logloss", "oracle_lift",
            "single_feature_primary", "single_feature_recovery",
            "audit_mode", "unestimable_reason", "leakage_class"]
    cols = [c for c in cols if c in rows.columns]
    rows[cols].to_csv(out_csv, index=False)

    drop = {"leakage", "leakage_distributed", "leakage_suspect"}
    summary = {
        "mode": "data-backed" if args.with_data else "offline",
        "screen": "type1_and_leakage_typed",
        "thresholds": {
            "conc_thresh": args.conc_thresh, "recover_frac": args.recover_frac,
            "implausible_acc": args.implausible_acc,
            "implausible_rho": args.implausible_rho,
            "abs_dup_acc": args.abs_dup_acc, "abs_dup_rho": args.abs_dup_rho,
            "min_v2_minority": args.min_v2_minority,
            "min_ceiling_at_5": args.min_ceiling_at_5,
            "oracle_k": args.oracle_k,
        },
        "n_cells": int(len(rows)),
        "n_genuine": int((rows["leakage_class"] == "genuine").sum()),
        "class_counts": rows["leakage_class"].value_counts().to_dict(),
        "by_survey": {s: sub["leakage_class"].value_counts().to_dict()
                      for s, sub in rows.groupby("survey")},
        "by_bucket": {str(b): sub["leakage_class"].value_counts().to_dict()
                      for b, sub in rows.groupby("bucket")},
        "leakage_cells": rows.loc[rows["leakage_class"].isin(drop),
                                  ["survey", "target", "country", "top_feature",
                                   "oracle_lift", "top_importance_share",
                                   "single_feature_recovery", "leakage_class"]]
                             .round(4).to_dict("records"),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {out_csv} ({len(rows)} cells) and {out_summary.name}")
    print(f"Mode: {summary['mode']}")
    print("Class counts:", summary["class_counts"])
    print(f"Genuine (default grid): {summary['n_genuine']}")
    unest = rows[rows["leakage_class"] == "unestimable"]
    if len(unest):
        print("\nUnestimable (type-1) cells:")
        show_u = ["survey", "target", "country", "majority_baseline",
                  "v2_minority_est", "ceiling_at_5", "unestimable_reason"]
        show_u = [c for c in show_u if c in unest.columns]
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(unest[show_u].to_string(index=False))
    lk = rows[rows["leakage_class"].isin(drop)]
    if len(lk):
        print("\nFlagged leakage cells:")
        show = ["survey", "target", "country", "top_feature", "oracle_lift",
                "top_importance_share", "single_feature_recovery", "leakage_class"]
        show = [c for c in show if c in lk.columns]
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(lk[show].to_string(index=False))

    if args.write_tex:
        write_tex(rows)


def write_tex(rows: pd.DataFrame) -> None:
    gen = PAPER_DIR / "generated_current_state"
    gen.mkdir(parents=True, exist_ok=True)

    def esc(x):
        s = "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
        for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
            s = s.replace(a, b)
        return s

    def num(x, p=3):
        return "-" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{float(x):.{p}f}"

    lk = rows[rows["leakage_class"].isin(["leakage", "leakage_distributed", "leakage_suspect"])]
    lines = []
    for _, r in lk.iterrows():
        lines.append(
            f"{esc(r['survey'])} & {esc(r['target'])} & {esc(r['country'])} & "
            f"{esc(r['top_feature'])} & {num(r['oracle_lift'])} & "
            f"{num(r['top_importance_share'])} & {num(r['single_feature_recovery'])} & "
            f"{esc(r['leakage_class'])} \\\\"
        )
    header = ("Survey & Target & Country & Top feat & Oracle PI (lift/$\\rho$) & "
              "Top imp.\\ share & 1-feat recovery & Class \\\\")
    table = (
        "\\begin{longtable}{lll l rrr l}\n\\toprule\n" + header +
        "\n\\midrule\n\\endfirsthead\n\\toprule\n" + header +
        "\n\\midrule\n\\endhead\n" + "\n".join(lines) +
        "\n\\bottomrule\n\\end{longtable}\n"
    )
    (gen / "leakage_audit_longtable.tex").write_text(table, encoding="utf-8")
    print(f"Wrote {gen / 'leakage_audit_longtable.tex'}")


if __name__ == "__main__":
    main()
