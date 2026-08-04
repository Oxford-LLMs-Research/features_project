"""
Free-text main results — T1 + T2 + cluster-bootstrap CIs on the arm-C (free-text) data.

Restructure decision (2026-06-10): pilot 2 established that free-text elicitation is the
better instrument (JSON suppresses breadth) and the main experiment will use it, so the
paper's headline results move to the free-text arm. This script produces every number and
tex table the rewritten Results section needs, from:

  - outputs/main/scores_deepseek.csv   (or legacy scores.csv / format_pilot/)
  - outputs/main/scores_kimi.csv
  - outputs/main/<sel>/maps/C__<disambig>__….json
  - outputs/cache/cells/<target>_<country>/oracle.csv
  - outputs/cache/audits/leakage_audit.csv

Primary disambiguator = nemotron (the main-experiment choice; mapper strength was ~null).
qwen235b is computed alongside as robustness and stored in the JSON summary.

Methodology mirrors the pilot-1 analysis scripts exactly (shared arithmetic now lives
in survey_features.metrics):
  - captured importance / oracle percentile / Jaccard / adaptation: metrics helpers
    (same formulas as archive/alignment_analysis.py)
  - cluster bootstrap + matched-k random captured-importance baseline: metrics helpers
    (same formulas as archive/uncertainty_analysis.py)

Outputs:
  outputs/main/freetext_main_summary.json
  (TeX tables: python paper/scripts/write_freetext_tex.py)

Run:  python analysis/freetext_main_results.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from survey_features.config import OUTPUTS_DIR  # noqa: E402

OUT = OUTPUTS_DIR
from survey_features.layout import (  # noqa: E402
    genuine_cells as _genuine_cells,
    main_dir,
    resolve_main_scores_path,
    selector_dirs,
)
PILOT = main_dir(OUT)

from survey_features.metrics import (  # noqa: E402
    captured_importance,
    cluster_bootstrap_ci as _cluster_bootstrap_ci,
    jaccard,
    load_oracle_splits,
    oracle_percentile_mean,
    random_captured_mean as _random_captured_mean,
    stable_seed,
)

N_BOOT = 2000
RAND_DRAWS = 200
SEED = 42
CONDITIONS = ["unprompted", "country_provided"]

SELECTORS = {  # selector key -> display label
    "deepseek": "DeepSeek-V3.2",
    "kimi": "Kimi-K2.5",
}
PRIMARY_DK = "nemotron"
ROBUST_DK = "qwen235b"


# ── loaders ──────────────────────────────────────────────────────────────────

def genuine_cells() -> list[tuple[str, str, str]]:
    return _genuine_cells(OUT)


def load_scores() -> pd.DataFrame:
    frames = []
    for key, label in SELECTORS.items():
        path = resolve_main_scores_path(key, OUT)
        if path is None:
            raise FileNotFoundError(f"No scores CSV for selector {key} under {PILOT}")
        d = pd.read_csv(path)
        d["selector"] = key
        d["model_label"] = label
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["error"].isna() | (df["error"] == "")]
    for c in ("captured_importance", "oracle_acc", "model_acc", "random_acc",
              "majority", "value_over_random", "cost_of_imperfect", "k"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


_imp_cache: dict[tuple[str, str], tuple[dict[str, float], dict[str, float]]] = {}


def oracle_splits(target: str, country: str) -> tuple[dict[str, float], dict[str, float]]:
    key = (target, country)
    if key not in _imp_cache:
        _imp_cache[key] = load_oracle_splits(target, country, OUT)
    return _imp_cache[key]


def map_codes(selector: str, dk: str, survey: str, target: str, country: str,
              cond: str) -> list[str] | None:
    _, _, maps = selector_dirs(selector, OUT)
    from survey_features.layout import cell_tag
    p = maps / f"C__{dk}__{cell_tag(survey, target, country)}__{cond}.json"
    if not p.is_file():
        return None
    codes = json.loads(p.read_text(encoding="utf-8")).get("mapped_codes", [])
    seen, out = set(), []
    for c in codes:
        if c and c not in seen:
            seen.add(c)
            out.append(str(c))
    return out


# ── metric arithmetic (single copy in survey_features.metrics) ───────────────

def random_captured_mean(imp: dict[str, float], k: int, seed: int,
                         draws: int = RAND_DRAWS,
                         rank: dict[str, float] | None = None) -> float | None:
    return _random_captured_mean(imp, k, seed, draws=draws, rank=rank)


def cluster_bootstrap_ci(df: pd.DataFrame, col: str, cluster_cols=("survey", "target"),
                         n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    return _cluster_bootstrap_ci(df, col, cluster_cols=cluster_cols, n_boot=n_boot, seed=seed)


# ── T1: scores-based metrics (arm C) ─────────────────────────────────────────

def t1_frames(scores: pd.DataFrame, dk: str) -> pd.DataFrame:
    c = scores[(scores["arm"] == "C") & (scores["disambiguator"] == dk)].copy()
    c["beat_random"] = (c["value_over_random"] > 0).astype(float)
    return c


def add_alignment_cols(c: pd.DataFrame, dk: str) -> pd.DataFrame:
    """Per-row oracle percentile + matched-k random captured baseline (paired delta)."""
    pct, rnd = [], []
    for _, r in c.iterrows():
        rank, score = oracle_splits(r["target"], r["country"])
        codes = map_codes(r["selector"], dk, r["survey"], r["target"], r["country"],
                          r["condition"]) or []
        kk = int(r["k"]) if pd.notna(r["k"]) else 0
        pct.append(oracle_percentile_mean(codes, rank))
        seed = stable_seed(r["target"], r["country"], r["condition"], kk)
        rnd.append(random_captured_mean(score, kk, seed, rank=rank))
    c = c.copy()
    c["oracle_pctile_mean"] = pct
    c["rand_captured"] = rnd
    c["delta_captured"] = c["captured_importance"] - c["rand_captured"]
    return c


# ── T2: adaptation + movement from arm-C maps ────────────────────────────────

def t2_metrics(dk: str) -> tuple[pd.DataFrame, dict]:
    cells = genuine_cells()
    rows = []
    summ: dict = {}
    for selector, label in SELECTORS.items():
        sets: dict[tuple[str, str, str], set[str]] = {}  # (target,country,cond) -> codes
        survey_of: dict[str, str] = {}
        for survey, target, country in cells:
            survey_of[target] = survey
            for cond in CONDITIONS:
                codes = map_codes(selector, dk, survey, target, country, cond)
                if codes is not None:
                    sets[(target, country, cond)] = set(codes)
        # T2a: unprompted vs country_provided Jaccard, per cell
        # T2b: own vs cross captured importance (country_provided)
        by_t: dict[str, dict[str, set]] = {}
        for (t, ctry, cond), s in sets.items():
            if cond == "country_provided":
                by_t.setdefault(t, {})[ctry] = s
        for (t, ctry, cond), s in sets.items():
            if cond != "country_provided":
                continue
            up = sets.get((t, ctry, "unprompted"))
            j_upcp = jaccard(s, up) if up is not None else None
            codes = list(s)
            k = len(codes)
            rank, score = oracle_splits(t, ctry)
            own = captured_importance(codes, score, k, rank=rank)
            cross_vals = []
            for c2 in by_t.get(t, {}):
                if c2 == ctry:
                    continue
                r2, s2 = oracle_splits(t, c2)
                cv = captured_importance(codes, s2, k, rank=r2)
                if cv is not None:
                    cross_vals.append(cv)
            cross = float(np.mean(cross_vals)) if cross_vals else None
            adapt = (own - cross) if (own is not None and cross is not None) else None
            sib_sets = [v for c2, v in by_t.get(t, {}).items() if c2 != ctry]
            xj = [j for j in (jaccard(s, v) for v in sib_sets) if j is not None]
            rows.append({"selector": selector, "model_label": label,
                         "survey": survey_of.get(t), "target": t, "country": ctry,
                         "k": k, "jaccard_up_cp": j_upcp,
                         "own_ci": own, "cross_ci": cross, "adaptation": adapt,
                         "xcountry_jaccard": float(np.mean(xj)) if xj else None})
        # per-target cross-country jaccard (all pairs), for the summary
        xjs = []
        for t, per_c in by_t.items():
            js = [j for j in (jaccard(a, b) for a, b in combinations(per_c.values(), 2))
                  if j is not None]
            if js:
                xjs.append(float(np.mean(js)))
        summ[selector] = {"xcountry_jaccard_per_target_mean":
                          round(float(np.mean(xjs)), 4) if xjs else None}
    return pd.DataFrame(rows), summ


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    scores = load_scores()
    summary: dict = {"primary_disambiguator": PRIMARY_DK, "n_boot": N_BOOT,
                     "rand_draws": RAND_DRAWS}

    # ---------- T1 primary (nemotron) ----------
    c = t1_frames(scores, PRIMARY_DK)
    ck = {ks: c[c["k_spec"] == ks] for ks in ("model", "k10", "k5")}

    # global table (model-k)
    glob: dict = {}
    for sel, label in SELECTORS.items():
        d = ck["model"][ck["model"]["selector"] == sel]
        glob[sel] = {
            "n_rows": int(len(d)),
            "oracle_acc": round(float(d["oracle_acc"].mean()), 4),
            "model_acc": round(float(d["model_acc"].mean()), 4),
            "random_acc": round(float(d["random_acc"].mean()), 4),
            "majority": round(float(d["majority"].mean()), 4),
            "cost_of_imperfect": round(float(d["cost_of_imperfect"].mean()), 4),
            "value_over_random": round(float(d["value_over_random"].mean()), 4),
            "share_beat_random": round(float(d["beat_random"].mean()), 4),
            "mean_k": round(float(d["k"].mean()), 2),
            "captured_importance": round(float(d["captured_importance"].mean()), 4),
        }
    summary["global_model_k"] = glob

    # fixed-k: captured imp + VoR + beat share at model-k / k10 / k5
    fixedk: dict = {}
    for ks in ("model", "k10", "k5"):
        ent = {}
        for sel in SELECTORS:
            d = ck[ks][ck[ks]["selector"] == sel]
            ent[sel] = {
                "captured_importance": round(float(d["captured_importance"].mean()), 4),
                "value_over_random": round(float(d["value_over_random"].mean()), 4),
                "share_beat_random": round(float(d["beat_random"].mean()), 4),
            }
        fixedk[ks] = ent
    summary["fixed_k"] = fixedk

    # survey (model-k): per survey x model oracle/model/random acc + captured imp
    surv: dict = {}
    for survey in sorted(ck["model"]["survey"].unique()):
        for sel in SELECTORS:
            d = ck["model"][(ck["model"]["selector"] == sel) & (ck["model"]["survey"] == survey)]
            surv.setdefault(survey, {})[sel] = {
                "oracle_acc": round(float(d["oracle_acc"].mean()), 3),
                "model_acc": round(float(d["model_acc"].mean()), 3),
                "random_acc": round(float(d["random_acc"].mean()), 3),
                "captured_importance": round(float(d["captured_importance"].mean()), 3),
            }
    summary["survey_model_k"] = surv

    # ---------- alignment extras + uncertainty (model-k, nemotron) ----------
    cm = add_alignment_cols(ck["model"], PRIMARY_DK)
    unc: dict = {}
    for sel, label in SELECTORS.items():
        d = cm[cm["selector"] == sel]
        unc[sel] = {
            "value_over_random": cluster_bootstrap_ci(d, "value_over_random"),
            "cost_of_imperfect": cluster_bootstrap_ci(d, "cost_of_imperfect"),
            "captured_importance": cluster_bootstrap_ci(d, "captured_importance"),
            "rand_captured": cluster_bootstrap_ci(d, "rand_captured"),
            "delta_captured": cluster_bootstrap_ci(d, "delta_captured"),
            "oracle_pctile_mean": cluster_bootstrap_ci(d, "oracle_pctile_mean"),
        }
    summary["uncertainty_model_k"] = unc

    # ---------- T2 ----------
    t2, t2_summ = t2_metrics(PRIMARY_DK)
    t2_unc: dict = {}
    for sel, label in SELECTORS.items():
        d = t2[t2["selector"] == sel]
        t2_unc[sel] = {
            "jaccard_up_cp": cluster_bootstrap_ci(d, "jaccard_up_cp"),
            "xcountry_jaccard": cluster_bootstrap_ci(d, "xcountry_jaccard"),
            "own_ci": cluster_bootstrap_ci(d, "own_ci"),
            "cross_ci": cluster_bootstrap_ci(d, "cross_ci"),
            "adaptation": cluster_bootstrap_ci(d, "adaptation"),
            "share_adapt_pos": round(float((d.loc[d["adaptation"].notna(), "adaptation"] > 0).mean()), 4)
            if d["adaptation"].notna().any() else None,
            "xcountry_jaccard_per_target": t2_summ[sel]["xcountry_jaccard_per_target_mean"],
        }
    summary["t2_adaptation"] = t2_unc

    # ---------- robustness: qwen235b disambiguator ----------
    cq = t1_frames(scores, ROBUST_DK)
    rob: dict = {}
    for ks in ("model", "k10", "k5"):
        d = cq[cq["k_spec"] == ks]
        ent = {}
        for sel in SELECTORS:
            dd = d[d["selector"] == sel]
            ent[sel] = {
                "captured_importance": round(float(dd["captured_importance"].mean()), 4),
                "value_over_random": round(float(dd["value_over_random"].mean()), 4),
            }
        rob[ks] = ent
    t2q, _ = t2_metrics(ROBUST_DK)
    rob["adaptation"] = {
        sel: cluster_bootstrap_ci(t2q[t2q["selector"] == sel], "adaptation")
        for sel in SELECTORS
    }
    summary["robustness_qwen235b"] = rob

    out = PILOT / "freetext_main_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
