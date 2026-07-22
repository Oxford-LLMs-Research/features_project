"""
Free-text main results — T1 + T2 + cluster-bootstrap CIs on the arm-C (free-text) data.

Restructure decision (2026-06-10): pilot 2 established that free-text elicitation is the
better instrument (JSON suppresses breadth) and the main experiment will use it, so the
paper's headline results move to the free-text arm. This script produces every number and
tex table the rewritten Results section needs, from:

  - outputs/format_pilot/scores.csv        (DeepSeek-V3.2 selector; arm C rows)
  - outputs/format_pilot/scores_kimi.csv   (Kimi-K2.5 selector)
  - outputs/format_pilot/<sel>/maps/C__<disambig>__<survey>__<target>__<country>__<cond>.json
  - outputs/<target>_<country>/oracle.csv  (oracle importances, per cell)
  - outputs/leakage_audit.csv              (the 52 genuine cells = the arm-C grid)

Primary disambiguator = nemotron (the main-experiment choice; mapper strength was ~null).
qwen235b is computed alongside as robustness and stored in the JSON summary.

Methodology mirrors the pilot-1 analysis scripts exactly:
  - captured importance / oracle percentile: alignment_analysis.py arithmetic
  - own-vs-cross adaptation (T2b) + Jaccard movement (T2a): alignment_analysis.py logic,
    applied to arm-C mapped codes (oracle-arithmetic, no model refits)
  - cluster bootstrap (cluster = survey x target, 2000 resamples): uncertainty_analysis.py
  - matched-k random captured-importance baseline: 200 draws from the cell's oracle pool

Outputs:
  outputs/format_pilot/freetext_main_summary.json
  paper/generated_current_state/ft_global_metrics.tex
  paper/generated_current_state/ft_fixedk.tex
  paper/generated_current_state/ft_survey_metrics.tex
  paper/generated_current_state/ft_test2_adaptation.tex
  paper/generated_current_state/ft_uncertainty.tex

Run:  python analysis/freetext_main_results.py
"""
from __future__ import annotations

import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PILOT = OUT / "format_pilot"
GEN = ROOT / "paper" / "generated_current_state"

B = chr(92)  # backslash for tex (avoids \b/\t control-char bugs in python strings)

N_BOOT = 2000
RAND_DRAWS = 200
SEED = 42
CONDITIONS = ["unprompted", "country_provided"]

SELECTORS = {  # selector key -> (scores csv, display label)
    "deepseek": (PILOT / "scores.csv", "DeepSeek-V3.2"),
    "kimi": (PILOT / "scores_kimi.csv", "Kimi-K2.5"),
}
PRIMARY_DK = "nemotron"
ROBUST_DK = "qwen235b"


# ── loaders ──────────────────────────────────────────────────────────────────

def genuine_cells() -> list[tuple[str, str, str]]:
    out = []
    with open(OUT / "leakage_audit.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["leakage_class"] == "genuine":
                out.append((r["survey"], r["target"], r["country"]))
    return out


def load_scores() -> pd.DataFrame:
    frames = []
    for key, (path, label) in SELECTORS.items():
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


_imp_cache: dict[tuple[str, str], dict[str, float]] = {}


def oracle_importance(target: str, country: str) -> dict[str, float]:
    key = (target, country)
    if key not in _imp_cache:
        p = OUT / f"{target}_{country}" / "oracle.csv"
        if not p.is_file():
            _imp_cache[key] = {}
        else:
            d = pd.read_csv(p)
            imp = pd.to_numeric(d["importance_mean"], errors="coerce").fillna(0.0)
            _imp_cache[key] = dict(zip(d["feature_variable"].astype(str), imp))
    return _imp_cache[key]


def map_codes(selector: str, dk: str, survey: str, target: str, country: str,
              cond: str) -> list[str] | None:
    p = PILOT / selector / "maps" / f"C__{dk}__{survey}__{target}__{country}__{cond}.json"
    if not p.is_file():
        return None
    codes = json.loads(p.read_text(encoding="utf-8")).get("mapped_codes", [])
    seen, out = set(), []
    for c in codes:
        if c and c not in seen:
            seen.add(c)
            out.append(str(c))
    return out


# ── metric arithmetic (mirrors alignment_analysis.py) ────────────────────────

def captured_importance(codes: list[str], imp: dict[str, float], k: int) -> float | None:
    if not imp or k <= 0:
        return None
    ordered = sorted((max(0.0, v) for v in imp.values()), reverse=True)
    denom = sum(ordered[:k])
    if denom <= 0:
        return None
    num = sum(max(0.0, imp.get(c, 0.0)) for c in dict.fromkeys(codes))
    return num / denom


def oracle_percentile_mean(codes: list[str], imp: dict[str, float]) -> float | None:
    if not imp:
        return None
    codes_asc = [c for c, _ in sorted(imp.items(), key=lambda kv: kv[1])]
    n = len(codes_asc)
    if n <= 1:
        return None
    pos = {c: i / (n - 1) for i, c in enumerate(codes_asc)}
    vals = [pos[c] for c in dict.fromkeys(codes) if c in pos]
    return float(np.mean(vals)) if vals else None


def jaccard(a: set, b: set) -> float | None:
    u = a | b
    return (len(a & b) / len(u)) if u else None


def random_captured_mean(imp: dict[str, float], k: int, seed: int,
                         draws: int = RAND_DRAWS) -> float | None:
    """Mean captured importance of `draws` random k-subsets of the cell's oracle pool."""
    pool = list(imp.keys())
    if not imp or k <= 0 or len(pool) < k:
        return None
    ordered = sorted((max(0.0, v) for v in imp.values()), reverse=True)
    denom = sum(ordered[:k])
    if denom <= 0:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        pick = rng.choice(len(pool), size=k, replace=False)
        vals.append(sum(max(0.0, imp[pool[i]]) for i in pick) / denom)
    return float(np.mean(vals))


def cluster_bootstrap_ci(df: pd.DataFrame, col: str, cluster_cols=("survey", "target"),
                         n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    sub = df[df[col].notna()].copy()
    if sub.empty:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_clusters": 0}
    sub["_cl"] = list(zip(*[sub[c].astype(str) for c in cluster_cols]))
    groups = {kk: g[col].to_numpy() for kk, g in sub.groupby("_cl")}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n_cl = len(keys)
    for b in range(n_boot):
        pick = rng.integers(0, n_cl, size=n_cl)
        means[b] = np.concatenate([groups[keys[i]] for i in pick]).mean()
    return {"mean": round(float(sub[col].mean()), 4),
            "ci_low": round(float(np.percentile(means, 2.5)), 4),
            "ci_high": round(float(np.percentile(means, 97.5)), 4),
            "n": int(len(sub)), "n_clusters": n_cl}


# ── T1: scores-based metrics (arm C) ─────────────────────────────────────────

def t1_frames(scores: pd.DataFrame, dk: str) -> pd.DataFrame:
    c = scores[(scores["arm"] == "C") & (scores["disambiguator"] == dk)].copy()
    c["beat_random"] = (c["value_over_random"] > 0).astype(float)
    return c


def add_alignment_cols(c: pd.DataFrame, dk: str) -> pd.DataFrame:
    """Per-row oracle percentile + matched-k random captured baseline (paired delta)."""
    pct, rnd = [], []
    for _, r in c.iterrows():
        imp = oracle_importance(r["target"], r["country"])
        codes = map_codes(r["selector"], dk, r["survey"], r["target"], r["country"],
                          r["condition"]) or []
        kk = int(r["k"]) if pd.notna(r["k"]) else 0
        pct.append(oracle_percentile_mean(codes, imp))
        seed = abs(hash((r["target"], r["country"], r["condition"], kk))) % (2**31)
        rnd.append(random_captured_mean(imp, kk, seed))
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
    for selector, (_, label) in SELECTORS.items():
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
            own = captured_importance(codes, oracle_importance(t, ctry), k)
            cross_vals = []
            for c2 in by_t.get(t, {}):
                if c2 == ctry:
                    continue
                cv = captured_importance(codes, oracle_importance(t, c2), k)
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


# ── tex helpers ───────────────────────────────────────────────────────────────

def f3(x, dash="--"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    s = f"{x:.3f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def f4(x, dash="--"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:.4f}"


def ci_str(d: dict) -> str:
    if d.get("mean") is None:
        return "--"
    return f"{f3(d['mean'])} [{f3(d['ci_low'])}, {f3(d['ci_high'])}]"


def write_tex(name: str, body: str) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / name).write_text(body, encoding="utf-8")
    print(f"  wrote {GEN / name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    scores = load_scores()
    summary: dict = {"primary_disambiguator": PRIMARY_DK, "n_boot": N_BOOT,
                     "rand_draws": RAND_DRAWS}

    # ---------- T1 primary (nemotron) ----------
    c = t1_frames(scores, PRIMARY_DK)
    ck = {ks: c[c["k_spec"] == ks] for ks in ("model", "k10", "k5")}
    labels = [SELECTORS[s][1] for s in SELECTORS]

    # global table (model-k)
    g_rows = []
    glob: dict = {}
    for sel, (_, label) in SELECTORS.items():
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
    gd, gk = glob["deepseek"], glob["kimi"]
    g_rows = [
        ("Scored cells (cell $" + B + "times$ condition)", str(gd["n_rows"]), str(gk["n_rows"])),
        ("Mean realised $k$", f"{gd['mean_k']:.1f}", f"{gk['mean_k']:.1f}"),
        ("Mean oracle accuracy", f4(gd["oracle_acc"]), f4(gk["oracle_acc"])),
        ("Mean model accuracy", f4(gd["model_acc"]), f4(gk["model_acc"])),
        ("Mean random-$k$ accuracy", f4(gd["random_acc"]), f4(gk["random_acc"])),
        ("Mean majority baseline", f4(gd["majority"]), f4(gk["majority"])),
        ("Mean cost of imperfect", f4(gd["cost_of_imperfect"]), f4(gk["cost_of_imperfect"])),
        ("Mean value over random", f4(gd["value_over_random"]), f4(gk["value_over_random"])),
        ("Share value " + B + "textgreater{} 0", f3(gd["share_beat_random"]), f3(gk["share_beat_random"])),
        ("Mean captured importance", f4(gd["captured_importance"]), f4(gk["captured_importance"])),
    ]
    tex = [B + "begin{tabular}{lrr}", B + "toprule",
           f"Metric & {labels[0]} & {labels[1]} {B}{B}", B + "midrule"]
    tex += [f"{a} & {b} & {cc} {B}{B}" for a, b, cc in g_rows]
    tex += [B + "bottomrule", B + "end{tabular}"]
    write_tex("ft_global_metrics.tex", "\n".join(tex) + "\n")

    # fixed-k table: captured imp + VoR + beat share at model-k / k10 / k5
    fixedk: dict = {}
    rows_tex = []
    for ks, ks_label in (("model", "model-chosen $k$"), ("k10", "$k=10$"), ("k5", "$k=5$")):
        ent = {}
        cells_out = []
        for sel in SELECTORS:
            d = ck[ks][ck[ks]["selector"] == sel]
            ent[sel] = {"captured_importance": round(float(d["captured_importance"].mean()), 4),
                        "value_over_random": round(float(d["value_over_random"].mean()), 4),
                        "share_beat_random": round(float(d["beat_random"].mean()), 4)}
            cells_out += [f3(ent[sel]["captured_importance"]),
                          f3(ent[sel]["value_over_random"]),
                          f3(ent[sel]["share_beat_random"])]
        fixedk[ks] = ent
        rows_tex.append(f"{ks_label} & " + " & ".join(cells_out) + f" {B}{B}")
    summary["fixed_k"] = fixedk
    tex = ["{" + B + "setlength{" + B + "tabcolsep}{5pt}",
           B + "begin{tabular}{@{}lcccccc@{}}", B + "toprule",
           f"& {B}multicolumn{{3}}{{c}}{{{labels[0]}}} & {B}multicolumn{{3}}{{c}}{{{labels[1]}}} {B}{B}",
           B + "cmidrule(lr){2-4} " + B + "cmidrule(lr){5-7}",
           "Budget & Capt.\\ imp. & VoR & Beat rnd. & Capt.\\ imp. & VoR & Beat rnd. " + B + B,
           B + "midrule"] + rows_tex + [B + "bottomrule", B + "end{tabular}}"]
    write_tex("ft_fixedk.tex", "\n".join(tex) + "\n")

    # survey table (model-k): per survey x model oracle/model/random acc + captured imp
    surv: dict = {}
    body = []
    for survey in sorted(ck["model"]["survey"].unique()):
        row_cells = [survey.replace("_", B + "_")]
        for sel in SELECTORS:
            d = ck["model"][(ck["model"]["selector"] == sel) & (ck["model"]["survey"] == survey)]
            e = {"oracle_acc": round(float(d["oracle_acc"].mean()), 3),
                 "model_acc": round(float(d["model_acc"].mean()), 3),
                 "random_acc": round(float(d["random_acc"].mean()), 3),
                 "captured_importance": round(float(d["captured_importance"].mean()), 3)}
            surv.setdefault(survey, {})[sel] = e
            row_cells += [f3(e["oracle_acc"]), f3(e["model_acc"]), f3(e["random_acc"]),
                          f3(e["captured_importance"])]
        body.append(" & ".join(row_cells) + f" {B}{B}")
    summary["survey_model_k"] = surv
    tex = ["{" + B + "setlength{" + B + "tabcolsep}{4pt}" + B + "small",
           B + "begin{tabular}{@{}lcccccccc@{}}", B + "toprule",
           f"& {B}multicolumn{{4}}{{c}}{{{labels[0]}}} & {B}multicolumn{{4}}{{c}}{{{labels[1]}}} {B}{B}",
           B + "cmidrule(lr){2-5} " + B + "cmidrule(lr){6-9}",
           "Survey & Oracle & Model & Random & Capt. & Oracle & Model & Random & Capt. " + B + B,
           B + "midrule"] + body + [B + "bottomrule", B + "end{tabular}}"]
    write_tex("ft_survey_metrics.tex", "\n".join(tex) + "\n")

    # ---------- alignment extras + uncertainty (model-k, nemotron) ----------
    cm = add_alignment_cols(ck["model"], PRIMARY_DK)
    unc: dict = {}
    for sel, (_, label) in SELECTORS.items():
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
    for sel, (_, label) in SELECTORS.items():
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

    t2_rows = [
        ("Jaccard, unprompted vs.\\ country-provided",
         *(ci_str(t2_unc[s]["jaccard_up_cp"]) for s in SELECTORS)),
        ("Jaccard, across countries (same target)",
         *(ci_str(t2_unc[s]["xcountry_jaccard"]) for s in SELECTORS)),
        ("Own-country captured importance",
         *(ci_str(t2_unc[s]["own_ci"]) for s in SELECTORS)),
        ("Cross-country captured importance",
         *(ci_str(t2_unc[s]["cross_ci"]) for s in SELECTORS)),
        ("Adaptation score (own $-$ cross)",
         *(ci_str(t2_unc[s]["adaptation"]) for s in SELECTORS)),
        ("Share adaptation " + B + "textgreater{} 0",
         *(f3(t2_unc[s]["share_adapt_pos"]) for s in SELECTORS)),
    ]
    tex = ["{" + B + "setlength{" + B + "tabcolsep}{6pt}" + B + "renewcommand{" + B + "arraystretch}{1.1}",
           B + "begin{tabular}{@{}lcc@{}}", B + "toprule",
           f"Metric (mean [95{B}% CI]) & {labels[0]} & {labels[1]} {B}{B}", B + "midrule"]
    tex += [f"{a} & {b} & {cc} {B}{B}" for a, b, cc in t2_rows]
    tex += [B + "bottomrule", B + "end{tabular}}"]
    write_tex("ft_test2_adaptation.tex", "\n".join(tex) + "\n")

    # uncertainty headline table
    u_rows = [
        ("Value over random", *(ci_str(unc[s]["value_over_random"]) for s in SELECTORS)),
        ("Cost of imperfect", *(ci_str(unc[s]["cost_of_imperfect"]) for s in SELECTORS)),
        ("Captured importance", *(ci_str(unc[s]["captured_importance"]) for s in SELECTORS)),
        (B + "quad random-$k$ baseline", *(ci_str(unc[s]["rand_captured"]) for s in SELECTORS)),
        (B + "quad $" + B + "Delta$ (model $-$ random)",
         *(ci_str(unc[s]["delta_captured"]) for s in SELECTORS)),
        ("Oracle percentile", *(ci_str(unc[s]["oracle_pctile_mean"]) for s in SELECTORS)),
        ("Adaptation score (own $-$ cross)",
         *(ci_str(t2_unc[s]["adaptation"]) for s in SELECTORS)),
    ]
    tex = ["{" + B + "setlength{" + B + "tabcolsep}{6pt}" + B + "renewcommand{" + B + "arraystretch}{1.1}",
           B + "begin{tabular}{@{}lcc@{}}", B + "toprule",
           f"Metric (mean [95{B}% CI]) & {labels[0]} & {labels[1]} {B}{B}", B + "midrule"]
    tex += [f"{a} & {b} & {cc} {B}{B}" for a, b, cc in u_rows]
    tex += [B + "bottomrule", B + "end{tabular}}"]
    write_tex("ft_uncertainty.tex", "\n".join(tex) + "\n")

    # ---------- robustness: qwen235b disambiguator ----------
    cq = t1_frames(scores, ROBUST_DK)
    rob: dict = {}
    for ks in ("model", "k10", "k5"):
        d = cq[cq["k_spec"] == ks]
        ent = {}
        for sel in SELECTORS:
            dd = d[d["selector"] == sel]
            ent[sel] = {"captured_importance": round(float(dd["captured_importance"].mean()), 4),
                        "value_over_random": round(float(dd["value_over_random"].mean()), 4)}
        rob[ks] = ent
    t2q, _ = t2_metrics(ROBUST_DK)
    rob["adaptation"] = {sel: cluster_bootstrap_ci(t2q[t2q["selector"] == sel], "adaptation")
                         for sel in SELECTORS}
    summary["robustness_qwen235b"] = rob

    out = PILOT / "freetext_main_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
