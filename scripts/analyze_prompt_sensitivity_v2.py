"""
Analyze prompt-sensitivity-v2: pack contrasts vs the r1/r2 replicate floor.

Implements the registry's planned analysis (docs/experiments_registry.md,
prompt-sensitivity-v2 entry):

Primary (composition, vs the r1-r2 floor, pooled / by theme stratum / by question):
  - Soft Jaccard on all extracted items: MiniLM dual-embed (label and
    "label: context", similarity = elementwise max, matching retrieval.py),
    Hungarian 1-1 matching, tau = 0.75 primary with 0.65 / 0.85 robustness;
    plus a within-type variant (only same-type items may match).
  - Hard Jaccard on mapped expanded_codes.
  - Four-way type shares of extracted items.
  - Textbook share among mapped codes (per-survey textbook set minus the target).

Secondary (scores, k_spec == model, type-matched):
  - captured_importance and type-matched VoR/VoT deltas, pack minus the r1/r2
    mean, with the r2-r1 delta as the floor. Ordinal cells use *_rho columns,
    binary/nominal use *_ll (positive = model better in both).

Frame: the intersection of cells scored in ALL 16 Stage-1 files (4 selectors x
r1/r2/analyst_person/none_respondent). t1/t2 (temperature sidecar) are reported
separately and never enter the lock-rule floor.

Writes per-cell CSVs and a registry-ready digest to outputs/experiments/_analysis/.

  python scripts/analyze_prompt_sensitivity_v2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survey_features.config import DEFAULT_EMBEDDING_MODEL, OUTPUTS_DIR  # noqa: E402
from survey_features.layout import (  # noqa: E402
    experiments_analysis_dir,
    prompt_sensitivity_v2_dirs,
    prompt_sensitivity_v2_scores_path,
)
from survey_features.prompts import PROMPT_SENSITIVITY_V2_SELECTORS  # noqa: E402
from survey_features.retrieval import make_embed_fn  # noqa: E402
from survey_features.score_cell import textbook_codes  # noqa: E402

OUT = OUTPUTS_DIR
ANALYSIS_DIR = experiments_analysis_dir(OUT)
CELLS_YAML = ROOT / "data" / "prompt_sensitivity_v2_cells.yaml"

CONDITION = "country_provided"
SELECTORS = PROMPT_SENSITIVITY_V2_SELECTORS
TAUS = (0.65, 0.75, 0.85)
PRIMARY_TAU = 0.75
TYPES = (
    "respondent_attribute",
    "temporal_contextual",
    "instrument_methodology",
    "population_statistic",
)

# label -> (pack, layout kwargs). Stage 1 is the lock-rule universe; t1/t2 are
# the temperature-1.0 sidecar of the default pack (never in the floor).
RUNS: dict[str, tuple[str, dict]] = {
    "r1": ("scientist_respondent", {"replicate": 1}),
    "r2": ("scientist_respondent", {"replicate": 2}),
    "analyst_person": ("analyst_person", {}),
    "none_respondent": ("none_respondent", {}),
    "t1": ("scientist_respondent", {"temperature_draw": 1}),
    "t2": ("scientist_respondent", {"temperature_draw": 2}),
}
STAGE1 = ("r1", "r2", "analyst_person", "none_respondent")
# (pair_id, run_a, run_b). Pack-vs-default is the mean of the two per-replicate
# pairs at aggregation time; the floor is r1 vs r2.
PAIRS = (
    ("floor", "r1", "r2"),
    ("analyst_person_vs_r1", "analyst_person", "r1"),
    ("analyst_person_vs_r2", "analyst_person", "r2"),
    ("none_respondent_vs_r1", "none_respondent", "r1"),
    ("none_respondent_vs_r2", "none_respondent", "r2"),
    ("sidecar_t1_t2", "t1", "t2"),
    ("sidecar_r1_t1", "r1", "t1"),
)
PACK_CONTRASTS = {
    "analyst_person": ("analyst_person_vs_r1", "analyst_person_vs_r2"),
    "none_respondent": ("none_respondent_vs_r1", "none_respondent_vs_r2"),
}


# ── Grid metadata ──────────────────────────────────────────────────────────────

def load_strata() -> dict[tuple[str, str], str]:
    """(survey, target) -> theme stratum, from the frozen v2 grid YAML."""
    data = yaml.safe_load(CELLS_YAML.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], str] = {}
    for cell in data["cells"]:
        out[(cell["survey"], cell["target"])] = cell["stratum"]
    return out


# ── Score loading and frame ────────────────────────────────────────────────────

def scores_path(selector: str, label: str) -> Path:
    pack, kwargs = RUNS[label]
    return prompt_sensitivity_v2_scores_path(selector, pack, outputs_dir=OUT, **kwargs)


def load_model_k(selector: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(scores_path(selector, label))
    df = df[df["k_spec"].astype(str) == "model"]
    if "error" in df.columns:
        df = df[df["error"].isna()]
    return df.copy()


def compute_frame() -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    """(intersection, union) of scored cells across all 16 Stage-1 files."""
    sets = []
    for sel in SELECTORS:
        for label in STAGE1:
            df = load_model_k(sel, label)
            sets.append(set(map(tuple, df[["survey", "target", "country"]].values)))
    return set.intersection(*sets), set.union(*sets)


# ── Artifact loading ───────────────────────────────────────────────────────────

def run_dirs(selector: str, label: str) -> tuple[Path, Path, Path]:
    pack, kwargs = RUNS[label]
    return prompt_sensitivity_v2_dirs(selector, pack, outputs_dir=OUT, **kwargs)


def load_items(selector: str, label: str, cell: tuple[str, str, str]) -> list[dict]:
    """All extracted items (feature/context/type dicts) for one run-cell."""
    survey, target, country = cell
    _, extract_dir, _ = run_dirs(selector, label)
    rec = json.loads(
        (extract_dir / f"{survey}__{target}__{country}.json").read_text(encoding="utf-8")
    )
    return rec["features"][CONDITION]


def load_codes(selector: str, label: str, cell: tuple[str, str, str]) -> list[str]:
    """Mapped codes for one run-cell (expanded set, matching the scorer)."""
    survey, target, country = cell
    _, _, map_dir = run_dirs(selector, label)
    p = map_dir / f"C__nemotron__{survey}__{target}__{country}__{CONDITION}.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    if "expanded_codes" in rec:
        return list(rec.get("expanded_codes") or [])
    return list(rec.get("mapped_codes") or [])


def item_texts(item: dict) -> tuple[str, str]:
    """(label, dual) texts for one extracted item, mirroring retrieval.py."""
    label = str(item.get("feature") or "").strip()
    ctx = str(item.get("context") or "").strip()
    return label, (f"{label}: {ctx}" if ctx else label)


# ── Soft Jaccard ───────────────────────────────────────────────────────────────

def pair_similarity(
    a_items: list[dict], b_items: list[dict], emb: dict[str, np.ndarray]
) -> np.ndarray:
    """S[i,j] = max(cos(label_i, label_j), cos(dual_i, dual_j)); embeddings are
    L2-normalized so cosine is a dot product."""
    a_lab = np.stack([emb[item_texts(x)[0]] for x in a_items])
    a_dual = np.stack([emb[item_texts(x)[1]] for x in a_items])
    b_lab = np.stack([emb[item_texts(x)[0]] for x in b_items])
    b_dual = np.stack([emb[item_texts(x)[1]] for x in b_items])
    return np.maximum(a_lab @ b_lab.T, a_dual @ b_dual.T)


def _hungarian_matches(sim: np.ndarray, tau: float) -> int:
    ri, ci = linear_sum_assignment(-sim)
    return int((sim[ri, ci] >= tau).sum())


def soft_jaccard(
    a_items: list[dict],
    b_items: list[dict],
    emb: dict[str, np.ndarray],
    tau: float,
    *,
    within_type: bool = False,
) -> float:
    na, nb = len(a_items), len(b_items)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    sim = pair_similarity(a_items, b_items, emb)
    if not within_type:
        m = _hungarian_matches(sim, tau)
    else:
        m = 0
        a_types = [x.get("type") for x in a_items]
        b_types = [x.get("type") for x in b_items]
        for t in set(a_types) & set(b_types):
            ai = [i for i, at in enumerate(a_types) if at == t]
            bi = [j for j, bt in enumerate(b_types) if bt == t]
            m += _hungarian_matches(sim[np.ix_(ai, bi)], tau)
    return m / (na + nb - m)


def hard_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ── Composition per run-cell ───────────────────────────────────────────────────

def composition_row(
    selector: str,
    label: str,
    cell: tuple[str, str, str],
    strata: dict[tuple[str, str], str],
) -> dict:
    survey, target, country = cell
    items = load_items(selector, label, cell)
    codes = load_codes(selector, label, cell)
    tb = {c for c in textbook_codes(survey, OUT) if c != target}
    n = len(items)
    row = {
        "selector": selector,
        "run": label,
        "survey": survey,
        "target": target,
        "country": country,
        "stratum": strata[(survey, target)],
        "n_items": n,
        "n_codes": len(codes),
        "textbook_share": (
            len([c for c in codes if c in tb]) / len(codes) if codes else None
        ),
    }
    for t in TYPES:
        row[f"share_{t}"] = (
            sum(1 for x in items if x.get("type") == t) / n if n else None
        )
    return row


# ── Embedding cache ────────────────────────────────────────────────────────────

def build_embeddings(
    items_by_key: dict[tuple, list[dict]], model_name: str
) -> dict[str, np.ndarray]:
    texts: set[str] = set()
    for items in items_by_key.values():
        for x in items:
            lab, dual = item_texts(x)
            texts.add(lab)
            texts.add(dual)
    ordered = sorted(texts)
    embed = make_embed_fn(model_name)
    vecs = np.asarray(embed(ordered))
    print(f"embedded {len(ordered)} unique item texts ({model_name})")
    return dict(zip(ordered, vecs))


# ── Score metrics (type-matched) ───────────────────────────────────────────────

def matched_scores(selector: str, label: str, frame: set) -> pd.DataFrame:
    """Per-cell PI / type-matched VoR / VoT / k at model k, frame cells only.

    Ordinal/continuous cells use Spearman deltas (*_rho); binary/nominal use
    log-loss deltas (*_ll). Both are signed so positive = model better.
    """
    df = load_model_k(selector, label)
    df = df[
        df[["survey", "target", "country"]].apply(tuple, axis=1).isin(frame)
    ].copy()
    for col in (
        "captured_importance", "k",
        "value_over_random_rho", "value_over_textbook_rho",
        "value_over_random_ll", "value_over_textbook_ll",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    is_reg = df["target_type"].isin(("ordinal", "continuous"))
    df["vor_matched"] = np.where(
        is_reg, df["value_over_random_rho"], df["value_over_random_ll"]
    )
    df["vot_matched"] = np.where(
        is_reg, df["value_over_textbook_rho"], df["value_over_textbook_ll"]
    )
    keep = [
        "survey", "target", "country", "target_type",
        "captured_importance", "vor_matched", "vot_matched", "k",
    ]
    out = df[keep].copy()
    out["selector"] = selector
    out["run"] = label
    return out


SCORE_METRICS = ("captured_importance", "vor_matched", "vot_matched", "k")


def score_deltas(scores: pd.DataFrame, strata: dict) -> pd.DataFrame:
    """Per-cell deltas: pack minus default (r1/r2 mean) and the r2-r1 floor."""
    keys = ["survey", "target", "country"]
    wide = scores.pivot_table(
        index=keys + ["target_type"], columns="run", values=list(SCORE_METRICS),
        aggfunc="first",
    )
    rows = []
    for idx, r in wide.iterrows():
        survey, target, country, ttype = idx
        base = {
            "survey": survey, "target": target, "country": country,
            "target_type": ttype, "stratum": strata[(survey, target)],
        }
        for m in SCORE_METRICS:
            r1, r2 = r.get((m, "r1"), np.nan), r.get((m, "r2"), np.nan)
            default = np.nanmean([r1, r2])
            base[f"floor_d_{m}"] = r2 - r1
            for pack in ("analyst_person", "none_respondent"):
                base[f"{pack}_d_{m}"] = r.get((m, pack), np.nan) - default
        rows.append(base)
    return pd.DataFrame(rows)


# ── Aggregation helpers ────────────────────────────────────────────────────────

def summarize(d: pd.Series) -> dict:
    d = d.dropna()
    if not len(d):
        return {"n": 0}
    return {
        "n": int(len(d)),
        "mean": float(d.mean()),
        "median": float(d.median()),
        "share_pos": float((d > 0).mean()),
        "share_neg": float((d < 0).mean()),
    }


def question_means(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean per question (survey,target) — the cluster unit (3 countries each)."""
    return df.groupby(["survey", "target"])[col].mean()


def fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = ap.parse_args()

    strata = load_strata()
    frame, union = compute_frame()
    dropped = sorted(union - frame)
    print(f"frame: {len(frame)} cells scored in all 16 Stage-1 files "
          f"(union {len(union)}; dropped {len(dropped)}: {dropped})")
    cells = sorted(frame)

    # Load every run-cell's extracted items once; embed all unique texts once.
    items_by_key: dict[tuple, list[dict]] = {}
    for sel in SELECTORS:
        for label in RUNS:
            for cell in cells:
                items_by_key[(sel, label, cell)] = load_items(sel, label, cell)
    emb = build_embeddings(items_by_key, args.embedding_model)

    # Per-cell pair metrics (soft + hard Jaccard).
    pair_rows = []
    for sel in SELECTORS:
        for pair_id, la, lb in PAIRS:
            for cell in cells:
                a, b = items_by_key[(sel, la, cell)], items_by_key[(sel, lb, cell)]
                row = {
                    "selector": sel, "pair": pair_id,
                    "survey": cell[0], "target": cell[1], "country": cell[2],
                    "stratum": strata[(cell[0], cell[1])],
                    "n_items_a": len(a), "n_items_b": len(b),
                    "hard_jaccard": hard_jaccard(
                        load_codes(sel, la, cell), load_codes(sel, lb, cell)
                    ),
                }
                for tau in TAUS:
                    row[f"soft_jaccard_{int(tau * 100)}"] = soft_jaccard(a, b, emb, tau)
                row["soft_jaccard_75_within_type"] = soft_jaccard(
                    a, b, emb, PRIMARY_TAU, within_type=True
                )
                pair_rows.append(row)
        print(f"pairs done: {sel}")
    pairs_df = pd.DataFrame(pair_rows)

    # Composition per run-cell (type shares, textbook share).
    comp_df = pd.DataFrame(
        composition_row(sel, label, cell, strata)
        for sel in SELECTORS for label in RUNS for cell in cells
    )

    # Scores: per-cell type-matched metrics and deltas.
    scores = pd.concat(
        [matched_scores(sel, label, frame) for sel in SELECTORS for label in STAGE1],
        ignore_index=True,
    )
    deltas = pd.concat(
        [
            score_deltas(scores[scores["selector"] == sel], strata).assign(selector=sel)
            for sel in SELECTORS
        ],
        ignore_index=True,
    )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(ANALYSIS_DIR / "ps_v2_pair_metrics.csv", index=False)
    comp_df.to_csv(ANALYSIS_DIR / "ps_v2_composition.csv", index=False)
    scores.to_csv(ANALYSIS_DIR / "ps_v2_score_cells.csv", index=False)
    deltas.to_csv(ANALYSIS_DIR / "ps_v2_score_deltas.csv", index=False)

    write_digest(pairs_df, comp_df, deltas, frame, dropped)
    print(f"wrote digest + 4 CSVs to {ANALYSIS_DIR}")


def write_digest(
    pairs_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    deltas: pd.DataFrame,
    frame: set,
    dropped: list,
) -> None:
    sj = f"soft_jaccard_{int(PRIMARY_TAU * 100)}"
    L: list[str] = [
        "# prompt-sensitivity-v2 — analysis digest",
        "",
        f"Frame: **{len(frame)} cells** scored in all 16 Stage-1 files "
        f"(dropped from the 72-cell grid: {dropped}). "
        "t1/t2 = temperature-1.0 sidecar of the default pack; never in the floor.",
        "",
        "## Primary: composition overlap vs the replicate floor",
        "",
        "Soft Jaccard (tau=0.75, Hungarian 1-1, MiniLM dual-embed) between runs, "
        "mean over frame cells. `pack vs default` = mean of the pack-vs-r1 and "
        "pack-vs-r2 pair values. Lower = composition moved more.",
        "",
        "| Selector | r1-r2 floor | analyst_person vs default | none_respondent vs default | t1-t2 sidecar |",
        "|---|---|---|---|---|",
    ]
    floor_by_sel: dict[str, float] = {}
    pack_means: dict[tuple[str, str], float] = {}
    for sel in SELECTORS:
        g = pairs_df[pairs_df["selector"] == sel]
        floor = g.loc[g["pair"] == "floor", sj].mean()
        floor_by_sel[sel] = floor
        vals = {}
        for pack, pair_ids in PACK_CONTRASTS.items():
            v = g.loc[g["pair"].isin(pair_ids), sj].mean()
            vals[pack] = v
            pack_means[(sel, pack)] = v
        t = g.loc[g["pair"] == "sidecar_t1_t2", sj].mean()
        L.append(
            f"| {sel} | {floor:.3f} | {vals['analyst_person']:.3f} | "
            f"{vals['none_respondent']:.3f} | {t:.3f} |"
        )
    L += [
        "",
        "Movement beyond floor (floor minus pack-vs-default; positive = pack moves "
        "composition more than a replicate does):",
        "",
        "| Selector | analyst_person | none_respondent |",
        "|---|---|---|",
    ]
    for sel in SELECTORS:
        L.append(
            f"| {sel} | {floor_by_sel[sel] - pack_means[(sel, 'analyst_person')]:+.3f} "
            f"| {floor_by_sel[sel] - pack_means[(sel, 'none_respondent')]:+.3f} |"
        )

    # Robustness taus, within-type, hard Jaccard, by stratum, question-level.
    L += ["", "### Robustness and views", ""]
    for sel in SELECTORS:
        g = pairs_df[pairs_df["selector"] == sel]
        parts = [f"**{sel}**:"]
        for col, name in (
            ("soft_jaccard_65", "tau 0.65"),
            ("soft_jaccard_85", "tau 0.85"),
            ("soft_jaccard_75_within_type", "within-type 0.75"),
            ("hard_jaccard", "hard (codes)"),
        ):
            floor = g.loc[g["pair"] == "floor", col].mean()
            ap_ = g.loc[g["pair"].isin(PACK_CONTRASTS["analyst_person"]), col].mean()
            nr_ = g.loc[g["pair"].isin(PACK_CONTRASTS["none_respondent"]), col].mean()
            parts.append(
                f"{name}: floor {floor:.3f}, analyst {ap_:.3f}, none {nr_:.3f};"
            )
        L.append(" ".join(parts))
    L += ["", "### By theme stratum (soft Jaccard 0.75)", ""]
    for sel in SELECTORS:
        g = pairs_df[pairs_df["selector"] == sel]
        for stratum in sorted(g["stratum"].unique()):
            gs = g[g["stratum"] == stratum]
            floor = gs.loc[gs["pair"] == "floor", sj].mean()
            ap_ = gs.loc[gs["pair"].isin(PACK_CONTRASTS["analyst_person"]), sj].mean()
            nr_ = gs.loc[gs["pair"].isin(PACK_CONTRASTS["none_respondent"]), sj].mean()
            L.append(
                f"- {sel} / {stratum}: floor {floor:.3f}, "
                f"analyst {ap_:.3f} ({floor - ap_:+.3f}), "
                f"none {nr_:.3f} ({floor - nr_:+.3f})"
            )
    L += [
        "",
        "### Question-level (cluster) check",
        "",
        "Per-question means (24 questions x 3 countries; the question is the "
        "cluster unit). Count of questions where the pack moved more than the floor:",
        "",
    ]
    for sel in SELECTORS:
        g = pairs_df[pairs_df["selector"] == sel]
        qfloor = question_means(g[g["pair"] == "floor"], sj)
        for pack, pair_ids in PACK_CONTRASTS.items():
            qpack = question_means(g[g["pair"].isin(pair_ids)], sj)
            joined = pd.concat([qfloor.rename("floor"), qpack.rename("pack")], axis=1)
            n_more = int((joined["pack"] < joined["floor"]).sum())
            L.append(f"- {sel} / {pack}: {n_more}/{len(joined)} questions move beyond floor")

    # Type shares and textbook share (pack minus default).
    L += [
        "",
        "## Composition shifts (pack minus r1/r2 mean, mean over cells)",
        "",
        "| Selector | Pack | d instrument_methodology | d population_statistic | "
        "d respondent_attribute | d temporal_contextual | d textbook_share |",
        "|---|---|---|---|---|---|---|",
    ]
    cmean = comp_df.groupby(["selector", "run"]).mean(numeric_only=True)
    for sel in SELECTORS:
        default = (cmean.loc[(sel, "r1")] + cmean.loc[(sel, "r2")]) / 2
        for pack in ("analyst_person", "none_respondent"):
            d = cmean.loc[(sel, pack)] - default
            L.append(
                f"| {sel} | {pack} | {d[f'share_{TYPES[2]}']:+.3f} "
                f"| {d[f'share_{TYPES[3]}']:+.3f} | {d[f'share_{TYPES[0]}']:+.3f} "
                f"| {d[f'share_{TYPES[1]}']:+.3f} | {d['textbook_share']:+.3f} |"
            )

    # Secondary: scores.
    L += [
        "",
        "## Secondary: scores (k_spec=model, type-matched VoR)",
        "",
        "Delta = pack minus r1/r2 mean, per cell. Floor = r2 minus r1. "
        "`vor_matched`: Spearman for ordinal, log-loss for binary/nominal "
        "(positive = model better).",
        "",
    ]
    for sel in SELECTORS:
        g = deltas[deltas["selector"] == sel]
        L.append(f"### {sel}")
        L.append("")
        L.append("| Contrast | Metric | Mean | Median | Share>0 | Share<0 |")
        L.append("|---|---|---|---|---|---|")
        for prefix, label in (
            ("floor_d", "floor r2-r1"),
            ("analyst_person_d", "analyst_person"),
            ("none_respondent_d", "none_respondent"),
        ):
            for m in ("captured_importance", "vor_matched", "vot_matched", "k"):
                s = summarize(g[f"{prefix}_{m}"])
                if s["n"]:
                    L.append(
                        f"| {label} | {m} | {s['mean']:+.3f} | {s['median']:+.3f} "
                        f"| {s['share_pos']:.0%} | {s['share_neg']:.0%} |"
                    )
        # Sign concordance PI & VoR for each pack.
        for pack in ("analyst_person", "none_respondent"):
            dpi = g[f"{pack}_d_captured_importance"]
            dv = g[f"{pack}_d_vor_matched"]
            both_pos = int(((dpi > 0) & (dv > 0)).sum())
            both_neg = int(((dpi < 0) & (dv < 0)).sum())
            conflict = int((((dpi > 0) & (dv < 0)) | ((dpi < 0) & (dv > 0))).sum())
            L.append(
                f"| {pack} | PI&VoR signs | both+={both_pos} | both-={both_neg} "
                f"| conflict={conflict} | |"
            )
        # By stratum.
        for stratum in sorted(g["stratum"].unique()):
            gs = g[g["stratum"] == stratum]
            for pack in ("analyst_person", "none_respondent"):
                s = summarize(gs[f"{pack}_d_vor_matched"])
                if s["n"]:
                    L.append(
                        f"| {pack} / {stratum} | vor_matched | {s['mean']:+.3f} "
                        f"| {s['median']:+.3f} | {s['share_pos']:.0%} "
                        f"| {s['share_neg']:.0%} |"
                    )
        L.append("")
        # By survey (mean delta, counts).
        for pack in ("analyst_person", "none_respondent"):
            rows = []
            for surv, gs in g.groupby("survey"):
                d = gs[f"{pack}_d_vor_matched"].dropna()
                if len(d):
                    rows.append(f"`{surv}` {d.mean():+.3f} ({(d > 0).sum()}+/{(d < 0).sum()}-)")
            L.append(f"- {pack} VoR by survey: " + "; ".join(rows))
        L.append("")

    # Lock rule.
    L += [
        "## Lock-rule check",
        "",
        "Rule: keep the confirmatory default unless the alternative beats the "
        "replicate floor AND has the same sign on >=2 of 4 selectors AND the same "
        "direction in both theme strata AND does not hurt VoR/captured importance. "
        "If composition moves and scores do not: keep the default.",
        "",
    ]
    for pack in ("analyst_person", "none_respondent"):
        comp_beyond = [
            sel for sel in SELECTORS
            if floor_by_sel[sel] - pack_means[(sel, pack)] > 0
        ]
        vor_signs = {}
        strata_dirs: dict[str, set] = {}
        vor_hurt = []
        for sel in SELECTORS:
            g = deltas[deltas["selector"] == sel]
            mv = g[f"{pack}_d_vor_matched"].mean()
            vor_signs[sel] = np.sign(mv) if np.isfinite(mv) else 0.0
            dirs = set()
            for stratum, gs in g.groupby("stratum"):
                dirs.add(float(np.sign(gs[f"{pack}_d_vor_matched"].mean())))
            strata_dirs[sel] = dirs
            mpi = g[f"{pack}_d_captured_importance"].mean()
            if mv < 0 or mpi < 0:
                vor_hurt.append(sel)
        pos = sum(1 for v in vor_signs.values() if v > 0)
        neg = sum(1 for v in vor_signs.values() if v < 0)
        agree = max(pos, neg)
        consistent_strata = [s for s, d in strata_dirs.items() if len(d) == 1]
        L.append(
            f"- **{pack}**: composition moves beyond floor on "
            f"{len(comp_beyond)}/4 selectors ({', '.join(comp_beyond) or 'none'}); "
            f"mean VoR-delta sign agreement {agree}/4 "
            f"({pos} positive, {neg} negative); same direction in both strata for "
            f"{len(consistent_strata)}/4 selectors; VoR or PI hurt on "
            f"{len(vor_hurt)}/4 ({', '.join(vor_hurt) or 'none'})."
        )
    L += [
        "",
        "_Interpretation and the keep/switch verdict belong to the human read of "
        "the numbers above; this digest computes, it does not decide._",
        "",
    ]

    (ANALYSIS_DIR / "ps_v2_digest.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
