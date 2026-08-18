"""
Analyze prompt-sensitivity and pipeline-role-swap score/map artifacts.

Primary filter: k_spec == model. Prompt arm lives in the scores filename, not the
CSV ``arm`` column (that is always pipeline Arm C).

Robust contrast block (registry convention): for each paired contrast report
mean / median / share Δ>0, by condition, survey means, and both+/both-/conflict
on PI & VoR — so cell-level cancelation is visible.

  python scripts/analyze_stack_experiments.py
  python scripts/analyze_stack_experiments.py --skip-maps

Writes digests under outputs/experiments/_analysis/ (incl. registry_contrast_blocks.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survey_features.config import OUTPUTS_DIR  # noqa: E402
from survey_features.layout import (  # noqa: E402
    cell_tag,
    experiments_analysis_dir,
    pipeline_role_swap_dirs,
    pipeline_role_swap_root,
    prompt_sensitivity_dirs,
    prompt_sensitivity_root,
)

OUT = OUTPUTS_DIR
PS_ROOT = prompt_sensitivity_root(OUT)
RS_ROOT = pipeline_role_swap_root(OUT)
ANALYSIS_DIR = experiments_analysis_dir(OUT)

SELECTORS = ("kimi", "deepseek_v4")
ARMS = ("social_scientist", "none", "helpful")
BASE_ARM = "social_scientist"
PRIMARY = ("captured_importance", "value_over_random")
METRICS = (
    "captured_importance",
    "value_over_random",
    "value_over_textbook",
    "value_over_random_ll",
    "value_over_textbook_ll",
    "k",
)
KEYS = ("survey", "target", "country", "condition")


def _load_model_k(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "k_spec" not in df.columns:
        return pd.DataFrame()
    out = df[df["k_spec"].astype(str) == "model"].copy()
    for c in METRICS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _score_path(selector: str, arm: str) -> Path:
    return PS_ROOT / f"scores_{selector}_{arm}.csv"


def _mean_table(df: pd.DataFrame, label: str) -> dict:
    row = {"label": label, "n": int(len(df))}
    for m in METRICS:
        if m in df.columns and len(df):
            row[f"mean_{m}"] = float(df[m].mean(skipna=True))
        else:
            row[f"mean_{m}"] = None
    return row


def _paired_frame(
    base: pd.DataFrame,
    other: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join on cell keys; other - base for each metric."""
    b = base[list(KEYS) + [m for m in METRICS if m in base.columns]].copy()
    o = other[list(KEYS) + [m for m in METRICS if m in other.columns]].copy()
    b = b.rename(columns={m: f"{m}_base" for m in METRICS if m in b.columns})
    o = o.rename(columns={m: f"{m}_other" for m in METRICS if m in o.columns})
    merged = b.merge(o, on=list(KEYS), how="inner")
    for m in METRICS:
        bb, oo = f"{m}_base", f"{m}_other"
        if bb in merged.columns and oo in merged.columns:
            merged[f"d_{m}"] = merged[oo] - merged[bb]
    return merged


def robust_contrast(
    pairs: pd.DataFrame,
    *,
    label: str,
    contrast_id: str,
) -> dict:
    """Honest paired-contrast summary; resists mean-only cancelation."""
    n = int(len(pairs))
    out: dict = {
        "contrast_id": contrast_id,
        "label": label,
        "n_pairs": n,
        "metrics": {},
        "by_condition": {},
        "by_survey": {},
        "sign_pi_vor": {"both_pos": 0, "both_neg": 0, "conflict": 0, "other": 0},
    }
    if n == 0:
        return out

    for m in PRIMARY + ("value_over_textbook",):
        col = f"d_{m}"
        if col not in pairs.columns:
            continue
        d = pairs[col].dropna()
        out["metrics"][m] = {
            "mean": float(d.mean()) if len(d) else None,
            "median": float(d.median()) if len(d) else None,
            "share_pos": float((d > 0).mean()) if len(d) else None,
            "share_neg": float((d < 0).mean()) if len(d) else None,
            "share_zero": float((d.abs() < 1e-9).mean()) if len(d) else None,
        }

    dpi, dvor = pairs["d_captured_importance"], pairs["d_value_over_random"]
    both_pos = int(((dpi > 0) & (dvor > 0)).sum())
    both_neg = int(((dpi < 0) & (dvor < 0)).sum())
    conflict = int(
        (((dpi > 0) & (dvor < 0)) | ((dpi < 0) & (dvor > 0))).sum()
    )
    out["sign_pi_vor"] = {
        "both_pos": both_pos,
        "both_neg": both_neg,
        "conflict": conflict,
        "other": n - both_pos - both_neg - conflict,
    }

    for cond, g in pairs.groupby("condition"):
        row = {"n": int(len(g))}
        for m in PRIMARY:
            col = f"d_{m}"
            d = g[col].dropna()
            row[m] = {
                "mean": float(d.mean()) if len(d) else None,
                "share_pos": float((d > 0).mean()) if len(d) else None,
            }
        out["by_condition"][str(cond)] = row

    for surv, g in pairs.groupby("survey"):
        row = {"n": int(len(g))}
        for m in PRIMARY:
            col = f"d_{m}"
            d = g[col].dropna()
            row[m] = {
                "mean": float(d.mean()) if len(d) else None,
                "n_pos": int((d > 0).sum()),
                "n_neg": int((d < 0).sum()),
            }
        out["by_survey"][str(surv)] = row

    # Survey-level means then count how many surveys have mean Δ>0
    surv_means = pairs.groupby("survey")[list(f"d_{m}" for m in PRIMARY)].mean()
    out["survey_level"] = {
        "n_surveys": int(len(surv_means)),
        "n_pi_pos": int((surv_means["d_captured_importance"] > 0).sum())
        if "d_captured_importance" in surv_means else 0,
        "n_vor_pos": int((surv_means["d_value_over_random"] > 0).sum())
        if "d_value_over_random" in surv_means else 0,
    }
    return out


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.0f}%"


def _fmt_d(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:+.3f}"


def contrast_to_markdown(block: dict) -> str:
    """Registry-ready markdown for one contrast."""
    lines = [
        f"**Contrast.** `{block['label']}` (n_pairs={block['n_pairs']}, k_spec=model).",
        "",
        "| Metric | Mean Δ | Median Δ | Share Δ>0 | Share Δ<0 |",
        "|--------|--------|----------|-----------|-----------|",
    ]
    for m, short in (
        ("captured_importance", "PI (`captured_importance`)"),
        ("value_over_random", "VoR"),
        ("value_over_textbook", "VoT"),
    ):
        met = block.get("metrics", {}).get(m) or {}
        lines.append(
            f"| {short} | {_fmt_d(met.get('mean'))} | {_fmt_d(met.get('median'))} | "
            f"{_fmt_pct(met.get('share_pos'))} | {_fmt_pct(met.get('share_neg'))} |"
        )
    sp = block.get("sign_pi_vor") or {}
    lines.extend([
        "",
        f"**Sign concordance (PI & VoR):** both+={sp.get('both_pos', 0)}, "
        f"both−={sp.get('both_neg', 0)}, conflict={sp.get('conflict', 0)} "
        f"(of {block['n_pairs']}).",
        "",
        "**By condition.**",
    ])
    for cond, row in sorted((block.get("by_condition") or {}).items()):
        pi = row.get("captured_importance") or {}
        vor = row.get("value_over_random") or {}
        lines.append(
            f"- `{cond}` (n={row.get('n')}): PI mean {_fmt_d(pi.get('mean'))} "
            f"({_fmt_pct(pi.get('share_pos'))} >0); VoR mean {_fmt_d(vor.get('mean'))} "
            f"({_fmt_pct(vor.get('share_pos'))} >0)."
        )
    sl = block.get("survey_level") or {}
    lines.append(
        f"- Survey-level: {sl.get('n_pi_pos', 0)}/{sl.get('n_surveys', 0)} surveys "
        f"mean PI>0; {sl.get('n_vor_pos', 0)}/{sl.get('n_surveys', 0)} mean VoR>0."
    )
    lines.append("")
    lines.append("**By survey (mean Δ).**")
    for surv, row in sorted((block.get("by_survey") or {}).items()):
        pi = row.get("captured_importance") or {}
        vor = row.get("value_over_random") or {}
        lines.append(
            f"- `{surv}` (n={row.get('n')}): PI {_fmt_d(pi.get('mean'))} "
            f"({pi.get('n_pos', 0)}+/{pi.get('n_neg', 0)}−); "
            f"VoR {_fmt_d(vor.get('mean'))}."
        )
    return "\n".join(lines)


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _load_expanded(map_path: Path) -> list[str]:
    if not map_path.is_file():
        return []
    rec = json.loads(map_path.read_text(encoding="utf-8"))
    codes = rec.get("expanded_codes")
    if codes is None:
        codes = rec.get("mapped_codes") or []
    return [str(c) for c in codes]


def map_jaccard_prompt(selector: str) -> list[dict]:
    rows = []
    _, _, base_maps = prompt_sensitivity_dirs(selector, BASE_ARM, OUT)
    cells = sorted({p.stem for p in (PS_ROOT / selector / BASE_ARM / "freetext").glob("*.json")})
    for arm in ARMS:
        if arm == BASE_ARM:
            continue
        _, _, arm_maps = prompt_sensitivity_dirs(selector, arm, OUT)
        for stem in cells:
            parts = stem.split("__")
            if len(parts) < 3:
                continue
            survey, target, country = parts[0], parts[1], "__".join(parts[2:])
            ctag = cell_tag(survey, target, country)
            for cond in ("unprompted", "country_provided"):
                bp = list(base_maps.glob(f"C__*__{ctag}__{cond}.json"))
                ap = list(arm_maps.glob(f"C__*__{ctag}__{cond}.json"))
                if not bp or not ap:
                    continue
                rows.append({
                    "selector": selector, "other_arm": arm,
                    "survey": survey, "target": target, "country": country,
                    "condition": cond,
                    "jaccard": _jaccard(_load_expanded(bp[0]), _load_expanded(ap[0])),
                })
    return rows


def map_jaccard_vs_kimi_scientist(run_key: str, disambig_glob: str) -> list[dict]:
    _, _, base_maps = prompt_sensitivity_dirs("kimi", BASE_ARM, OUT)
    _, other_maps = pipeline_role_swap_dirs(run_key, OUT)
    rows = []
    cells = sorted({p.stem for p in (PS_ROOT / "kimi" / BASE_ARM / "freetext").glob("*.json")})
    for stem in cells:
        parts = stem.split("__")
        if len(parts) < 3:
            continue
        survey, target, country = parts[0], parts[1], "__".join(parts[2:])
        ctag = cell_tag(survey, target, country)
        for cond in ("unprompted", "country_provided"):
            bp = list(base_maps.glob(f"C__*__{ctag}__{cond}.json"))
            op = list(other_maps.glob(f"{disambig_glob}__{ctag}__{cond}.json"))
            if not bp or not op:
                continue
            rows.append({
                "survey": survey, "target": target, "country": country,
                "condition": cond,
                "jaccard": _jaccard(_load_expanded(bp[0]), _load_expanded(op[0])),
                "n_base": len(_load_expanded(bp[0])),
                "n_other": len(_load_expanded(op[0])),
            })
    return rows


def empty_gen_rate(selector: str, arm: str) -> dict:
    gen_dir, _, _ = prompt_sensitivity_dirs(selector, arm, OUT)
    n_files = n_empty = 0
    for p in gen_dir.glob("*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        for _cond, resp in (rec.get("responses") or {}).items():
            n_files += 1
            if not (resp or "").strip():
                n_empty += 1
    return {
        "selector": selector,
        "arm": arm,
        "n_conditions": n_files,
        "n_empty": n_empty,
        "empty_rate": (n_empty / n_files) if n_files else None,
    }


def decide_prompt(contrasts: list[dict], empty_rates: list[dict]) -> dict:
    """Keep social_scientist unless an arm wins PI+VoR means on BOTH selectors
    without worse empties, and survey-level PI is positive on >=4 surveys."""
    empty_by = {(e["selector"], e["arm"]): e.get("empty_rate") for e in empty_rates}
    winners = []
    for c in contrasts:
        # contrast_id like kimi__none
        parts = c["contrast_id"].split("__")
        if len(parts) != 2:
            continue
        sel, arm = parts
        pi = (c.get("metrics") or {}).get("captured_importance") or {}
        vor = (c.get("metrics") or {}).get("value_over_random") or {}
        dpi, dvor = pi.get("mean"), vor.get("mean")
        sl = c.get("survey_level") or {}
        base_empty = empty_by.get((sel, BASE_ARM))
        arm_empty = empty_by.get((sel, arm))
        worse_empty = (
            base_empty is not None and arm_empty is not None and arm_empty > base_empty + 1e-9
        )
        surv_ok = sl.get("n_pi_pos", 0) >= 4 and sl.get("n_surveys", 0) >= 4
        wins = (
            dpi is not None and dvor is not None and dpi > 0 and dvor > 0
            and not worse_empty and surv_ok
        )
        winners.append({
            "selector": sel, "arm": arm, "mean_d_pi": dpi, "mean_d_vor": dvor,
            "worse_empty": worse_empty, "survey_pi_pos": sl.get("n_pi_pos"),
            "wins": wins,
        })
    promote_arm = None
    for arm in ("none", "helpful"):
        rows = [w for w in winners if w["arm"] == arm]
        if len(rows) >= 2 and all(w["wins"] for w in rows):
            promote_arm = arm
            break
    return {
        "decision": "keep_social_scientist" if promote_arm is None else f"promote_{promote_arm}",
        "promote_arm": promote_arm,
        "per_contrast": winners,
        "rule": (
            "Promote only if mean PI and VoR both rise on BOTH selectors, "
            "empty-gen not worse, and mean PI>0 on >=4 surveys."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-maps", action="store_true")
    args = ap.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    means_rows = []
    empty_rates = []
    prompt_contrasts: list[dict] = []
    paired_frames = []
    md_sections: list[str] = [
        "# Registry contrast blocks",
        "",
        "Auto-generated by `scripts/analyze_stack_experiments.py`.",
        "Paste under **Result** / **Contrast detail** in `docs/experiments_registry.md`.",
        "",
    ]

    for sel in SELECTORS:
        base = _load_model_k(_score_path(sel, BASE_ARM))
        means_rows.append(_mean_table(base, f"{sel}/{BASE_ARM}"))
        empty_rates.append(empty_gen_rate(sel, BASE_ARM))
        for arm in ARMS:
            if arm == BASE_ARM:
                continue
            other = _load_model_k(_score_path(sel, arm))
            means_rows.append(_mean_table(other, f"{sel}/{arm}"))
            empty_rates.append(empty_gen_rate(sel, arm))
            pairs = _paired_frame(base, other)
            cid = f"{sel}__{arm}"
            block = robust_contrast(
                pairs, label=f"{sel}: {arm} - social_scientist", contrast_id=cid,
            )
            prompt_contrasts.append(block)
            if len(pairs):
                pairs = pairs.copy()
                pairs["selector"] = sel
                pairs["other_arm"] = arm
                paired_frames.append(pairs)
            md_sections.append(f"## prompt-sensitivity / `{sel}` / `{arm}`")
            md_sections.append("")
            md_sections.append(contrast_to_markdown(block))
            md_sections.append("")

    pd.DataFrame(means_rows).to_csv(
        ANALYSIS_DIR / "prompt_sensitivity_means_model_k.csv", index=False,
    )
    pd.DataFrame(empty_rates).to_csv(
        ANALYSIS_DIR / "prompt_sensitivity_empty_gen.csv", index=False,
    )
    if paired_frames:
        pd.concat(paired_frames, ignore_index=True).to_csv(
            ANALYSIS_DIR / "prompt_sensitivity_paired_rows.csv", index=False,
        )
    (ANALYSIS_DIR / "prompt_sensitivity_robust_contrasts.json").write_text(
        json.dumps(prompt_contrasts, indent=2), encoding="utf-8",
    )

    # Role-swap joint + extract-only
    base_kimi = _load_model_k(PS_ROOT / "scores_kimi_social_scientist.csv")
    role_blocks = []
    for run_key, label, score_name, map_glob in (
        (
            "minimax_flash",
            "minimax_flash - qwen_nemotron (joint)",
            "scores_minimax_flash.csv",
            "C__flash",
        ),
        (
            "minimax_nemotron",
            "minimax_nemotron - qwen_nemotron (extract-only)",
            "scores_minimax_nemotron.csv",
            "C__nemotron",
        ),
    ):
        other = _load_model_k(RS_ROOT / score_name)
        pairs = _paired_frame(base_kimi, other)
        block = robust_contrast(pairs, label=label, contrast_id=run_key)
        role_blocks.append(block)
        if len(pairs):
            pairs.to_csv(ANALYSIS_DIR / f"{run_key}_paired_rows.csv", index=False)
        md_sections.append(f"## {run_key}")
        md_sections.append("")
        md_sections.append(contrast_to_markdown(block))
        md_sections.append("")
        if not args.skip_maps:
            mj = map_jaccard_vs_kimi_scientist(run_key, map_glob)
            if mj:
                mjdf = pd.DataFrame(mj)
                mjdf.to_csv(ANALYSIS_DIR / f"{run_key}_map_jaccard.csv", index=False)
                block["map_jaccard_mean"] = float(mjdf["jaccard"].mean())
                md_sections.append(
                    f"Map Jaccard vs kimi/social_scientist Nemotron maps: "
                    f"mean={block['map_jaccard_mean']:.3f} (n={len(mjdf)})."
                )
                md_sections.append("")

    (ANALYSIS_DIR / "role_swap_robust_contrasts.json").write_text(
        json.dumps(role_blocks, indent=2), encoding="utf-8",
    )

    map_prompt_summary = {}
    if not args.skip_maps:
        map_prompt_rows = []
        for sel in SELECTORS:
            map_prompt_rows.extend(map_jaccard_prompt(sel))
        if map_prompt_rows:
            mpdf = pd.DataFrame(map_prompt_rows)
            mpdf.to_csv(ANALYSIS_DIR / "prompt_sensitivity_map_jaccard.csv", index=False)
            map_prompt_summary = {
                "n": int(len(mpdf)),
                "mean_jaccard": float(mpdf["jaccard"].mean()),
            }

    prompt_decision = decide_prompt(prompt_contrasts, empty_rates)
    digest = {
        "prompt_sensitivity": {
            "means": means_rows,
            "empty_gen": empty_rates,
            "robust_contrasts": prompt_contrasts,
            "map_jaccard_summary": map_prompt_summary,
            "decision": prompt_decision,
        },
        "role_swaps": role_blocks,
    }
    (ANALYSIS_DIR / "stack_decisions.json").write_text(
        json.dumps(digest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (ANALYSIS_DIR / "registry_contrast_blocks.md").write_text(
        "\n".join(md_sections) + "\n", encoding="utf-8",
    )

    # Stdout
    print("=== Robust contrasts (mean | median | share>0) ===")
    for c in prompt_contrasts + role_blocks:
        pi = (c.get("metrics") or {}).get("captured_importance") or {}
        vor = (c.get("metrics") or {}).get("value_over_random") or {}
        sp = c.get("sign_pi_vor") or {}
        sl = c.get("survey_level") or {}
        print(
            f"{c['contrast_id']}: n={c['n_pairs']}  "
            f"PI mean={_fmt_d(pi.get('mean'))} med={_fmt_d(pi.get('median'))} "
            f">0={_fmt_pct(pi.get('share_pos'))}  "
            f"VoR mean={_fmt_d(vor.get('mean'))} med={_fmt_d(vor.get('median'))} "
            f">0={_fmt_pct(vor.get('share_pos'))}  "
            f"both+/both-/conflict={sp.get('both_pos')}/{sp.get('both_neg')}/{sp.get('conflict')}  "
            f"surveys PI>0={sl.get('n_pi_pos')}/{sl.get('n_surveys')}"
        )
    print("\n=== DECISIONS ===")
    print("SYSTEM_PROMPT:", prompt_decision["decision"])
    print(f"\nDigests -> {ANALYSIS_DIR.resolve()}")
    print(f"Registry markdown -> {ANALYSIS_DIR / 'registry_contrast_blocks.md'}")


if __name__ == "__main__":
    main()
