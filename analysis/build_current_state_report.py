"""Build TeX-ready artifacts for paper/current_state.tex from outputs/."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
GEN = ROOT / "paper" / "generated_current_state"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from survey_features.prompts import (
    DISAMBIG_PROMPT_LEGACY as DISAMBIG_PROMPT,  # this report documents the legacy JSON pipeline
    PROMPT_COUNTRY,
    PROMPT_UNPROMPTED,
    SYSTEM_PROMPT,
)
from survey_features.surveys import MISSING_LABEL_PATTERNS as _MISSING_LABEL_PATTERNS
from survey_features.layout import (  # noqa: E402
    cell_dir,
    oracle_csv_path,
    prelim_stats_path,
)

SURVEY_ORDER = [
    "wvs",
    "afrobarometer",
    "arabbarometer",
    "asianbarometer",
    "latinobarometer",
    "ess_wave_11",
]


def wrap_verbatim_plain(text: str, width: int = 76) -> str:
    """Hard-wrap wide lines inside verbatim so PDF margins are not overrun."""
    lines: list[str] = []
    for raw in text.splitlines():
        if len(raw) <= width:
            lines.append(raw)
            continue
        lines.extend(
            textwrap.wrap(
                raw,
                width=width,
                break_long_words=True,
                break_on_hyphens=True,
            )
        )
    return "\n".join(lines)


def tex_escape(val: Any) -> str:
    s = "" if val is None else str(val)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def fmt_num(v: Any, places: int = 4) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    try:
        return f"{float(v):.{places}f}"
    except (TypeError, ValueError):
        return tex_escape(v)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_all_summary() -> pd.DataFrame:
    """Load every grid_summary CSV with model as an explicit column.

    Both LLMs (DeepSeek-V3.2, Kimi-K2.5) are loaded side-by-side; ``model`` holds
    the model tag so downstream tables can compare them rather than silently
    collapsing to one model.
    """
    from survey_features.layout import collect_all_grid_summaries

    order = {sid: i for i, sid in enumerate(SURVEY_ORDER)}
    dfs: list[pd.DataFrame] = []
    for p, sid, tag in collect_all_grid_summaries(OUT):
        d = pd.read_csv(p)
        d["survey"] = sid
        d["model"] = tag or "untagged"
        d["llm_run_tag"] = tag or ""
        d["_survey_order"] = order.get(sid, len(order))
        dfs.append(d)
    if not dfs:
        raise FileNotFoundError("No grid_summary__*.csv files found in outputs/")
    df = pd.concat(dfs, ignore_index=True)
    for c in (
        "oracle_acc",
        "model_acc",
        "random_acc",
        "cost_of_imperfect",
        "value_over_random",
        "majority_baseline",
        "k_mapped",
        "k_requested",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_manifest() -> dict:
    with open(ROOT / "prelim" / "prelim_manifest.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_target_detail() -> dict[tuple[str, str], dict]:
    with open(ROOT / "prelim" / "target_selection_detail.yaml", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    out: dict[tuple[str, str], dict] = {}
    for blk in doc.get("surveys") or []:
        sid = blk.get("survey_id")
        for s in blk.get("selected") or []:
            vc = str(s.get("variable"))
            out[(sid, vc)] = s
    return out


def get_question_texts() -> dict[tuple[str, str], str]:
    try:
        from survey_features.surveys import SURVEY_COUNTRY_COL, get_question_text, load_survey
        import os
    except ModuleNotFoundError:
        return {}
    cfg = os.environ.get("DATA_CONFIG_PATH")
    if not cfg:
        return {}
    out: dict[tuple[str, str], str] = {}
    for sid in SURVEY_ORDER:
        try:
            data, metadata = load_survey(sid, cfg)
            _ = data
            for code in metadata.get("EXCLUDED", {}):
                _ = code
            known_targets = [str(t) for t in (load_manifest().get("surveys", {}).get(sid, {}).get("targets", []))]
            for t in known_targets:
                try:
                    out[(sid, t)] = get_question_text(t, metadata)
                except Exception:
                    pass
        except Exception:
            continue
    return out


def category_count_from_metadata(var_info: dict) -> int | None:
    vals = var_info.get("values") or {}
    if not isinstance(vals, dict) or not vals:
        return None
    count = 0
    for label in vals.values():
        ln = str(label).strip().lower()
        if any(p in ln for p in _MISSING_LABEL_PATTERNS):
            continue
        count += 1
    return count if count > 0 else None


def model_label(tag: Any) -> str:
    """Short human label for a model tag (deepseek-ai_DeepSeek-V3.2 -> DeepSeek-V3.2)."""
    s = "" if tag is None else str(tag)
    if "_" in s:
        s = s.split("_", 1)[1]
    return s or "untagged"


def models_in(df: pd.DataFrame) -> list[str]:
    """Distinct model tags present, ordered alphabetically (stable across runs)."""
    return sorted(t for t in df["model"].dropna().unique())


def write_metrics_tables(df: pd.DataFrame) -> None:
    valid = df[df["oracle_acc"].notna() & df["model_acc"].notna() & df["random_acc"].notna()].copy()
    tags = models_in(valid)

    # --- Global metrics: one column per model (side-by-side) -----------------
    metric_rows = [
        ("Total rows", lambda d: str(len(d)), df),
        ("Valid rows", lambda d: str(len(d)), valid),
        ("Mean oracle accuracy", lambda d: fmt_num(d["oracle_acc"].mean()), valid),
        ("Mean model accuracy", lambda d: fmt_num(d["model_acc"].mean()), valid),
        ("Mean random-$k$ accuracy", lambda d: fmt_num(d["random_acc"].mean()), valid),
        ("Mean majority baseline", lambda d: fmt_num(d["majority_baseline"].mean()), valid),
        ("Mean cost of imperfect", lambda d: fmt_num(d["cost_of_imperfect"].mean()), valid),
        ("Mean value over random", lambda d: fmt_num(d["value_over_random"].mean()), valid),
        (r"Share value\textgreater 0", lambda d: fmt_num((d["value_over_random"] > 0).mean(), 3), valid),
    ]
    g_header = "Metric & " + " & ".join(tex_escape(model_label(t)) for t in tags) + r" \\"
    g_lines = []
    for name, fn, src in metric_rows:
        cells = " & ".join(fn(src[src["model"] == t]) for t in tags)
        g_lines.append(f"{name} & {cells} \\\\")
    global_tab = (
        "\\begin{tabular}{l" + "r" * len(tags) + "}\n\\toprule\n"
        + g_header + "\n\\midrule\n"
        + "\n".join(g_lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(GEN / "main_global_metrics.tex", global_tab)

    # --- Survey metrics: row per survey x model -----------------------------
    rows = []
    for sid, ssub in valid.groupby("survey", sort=False):
        for t in tags:
            sub = ssub[ssub["model"] == t]
            if sub.empty:
                continue
            rows.append(
                f"{tex_escape(sid)} & {tex_escape(model_label(t))} & {len(sub)} & "
                f"{fmt_num(sub['oracle_acc'].mean())} & {fmt_num(sub['model_acc'].mean())} & "
                f"{fmt_num(sub['random_acc'].mean())} & {fmt_num(sub['cost_of_imperfect'].mean())} & "
                f"{fmt_num(sub['value_over_random'].mean())} & "
                f"{fmt_num((sub['value_over_random'] > 0).mean(), 3)} \\\\"
            )
    survey_tab = (
        "\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
        "Survey & Model & N & Oracle & Model & Random & Cost & Value & Share value$>0$ \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(GEN / "main_survey_metrics.tex", survey_tab)

    # --- Head-to-head: value & cost by model, per survey --------------------
    cmp_lines = []
    for sid, ssub in valid.groupby("survey", sort=False):
        cells = []
        for t in tags:
            sub = ssub[ssub["model"] == t]
            cells.append(fmt_num(sub["value_over_random"].mean()) if not sub.empty else "-")
        for t in tags:
            sub = ssub[ssub["model"] == t]
            cells.append(fmt_num(sub["cost_of_imperfect"].mean()) if not sub.empty else "-")
        cmp_lines.append(f"{tex_escape(sid)} & " + " & ".join(cells) + r" \\")
    val_hdr = " & ".join(tex_escape(model_label(t)) for t in tags)
    cmp_tab = (
        "\\begin{tabular}{l" + "r" * (2 * len(tags)) + "}\n\\toprule\n"
        + f" & \\multicolumn{{{len(tags)}}}{{c}}{{Value over random}} & "
        + f"\\multicolumn{{{len(tags)}}}{{c}}{{Cost of imperfect}} \\\\\n"
        + f"Survey & {val_hdr} & {val_hdr} \\\\\n\\midrule\n"
        + "\n".join(cmp_lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(GEN / "main_model_comparison.tex", cmp_tab)

    # --- Condition metrics: row per condition x model -----------------------
    cond_rows = []
    for cond in ("unprompted", "country_provided"):
        for t in tags:
            sub = valid[(valid["condition"] == cond) & (valid["model"] == t)]
            if sub.empty:
                continue
            cond_rows.append(
                f"{tex_escape(cond)} & {tex_escape(model_label(t))} & {len(sub)} & "
                f"{fmt_num(sub['cost_of_imperfect'].mean())} & "
                f"{fmt_num(sub['value_over_random'].mean())} & "
                f"{fmt_num((sub['value_over_random'] > 0).mean(), 3)} \\\\"
            )
    cond_tab = (
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Condition & Model & N & Mean cost & Mean value & Share value$>0$ \\\\\n\\midrule\n"
        + "\n".join(cond_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(GEN / "main_condition_metrics.tex", cond_tab)


def write_bucket_and_extremes_tables() -> None:
    stats_path = prelim_stats_path(OUT)
    if not stats_path.is_file():
        return
    doc = read_json(stats_path)

    bucket = doc.get("bucket_stats") or {}
    b_lines = []
    for b in ("binary", "mid", "large"):
        sub = bucket.get(b) or {}
        if not sub:
            continue
        b_lines.append(
            f"{tex_escape(b)} & {tex_escape(sub.get('n'))} & {fmt_num(sub.get('mean_cost'))} & "
            f"{fmt_num(sub.get('mean_value'))} & {fmt_num(sub.get('mean_oracle_minus_majority'))} \\\\"
        )
    if b_lines:
        btab = (
            "\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Bucket & N & Mean cost & Mean value & Mean(oracle-majority) \\\\\n\\midrule\n"
            + "\n".join(b_lines)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )
        write_text(GEN / "main_bucket_metrics.tex", btab)

    def top_rows(key: str, cols: list[str], out_file: str) -> None:
        rows = doc.get(key) or []
        if not rows:
            return
        lines = []
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c)
                if c == "model":
                    vals.append(tex_escape(model_label(v)))
                elif isinstance(v, (int, float)):
                    vals.append(fmt_num(v))
                else:
                    vals.append(tex_escape(v))
            lines.append(" & ".join(vals) + r" \\")
        header = " & ".join(tex_escape(c) for c in cols) + r" \\"
        table = (
            "\\begin{longtable}{" + "l" * len(cols) + "}\n\\toprule\n"
            + header
            + "\n\\midrule\n\\endfirsthead\n\\toprule\n"
            + header
            + "\n\\midrule\n\\endhead\n"
            + "\n".join(lines)
            + "\n\\bottomrule\n\\end{longtable}\n"
        )
        write_text(GEN / out_file, table)

    top_rows(
        "top_cost_of_imperfect",
        ["survey", "target", "country", "model", "condition", "cost_of_imperfect", "value_over_random"],
        "appendix_top_cost_longtable.tex",
    )
    top_rows(
        "worst_value_over_random",
        ["survey", "target", "country", "model", "condition", "value_over_random", "cost_of_imperfect"],
        "appendix_worst_value_longtable.tex",
    )
    top_rows(
        "best_value_over_random",
        ["survey", "target", "country", "model", "condition", "value_over_random"],
        "appendix_best_value_longtable.tex",
    )


def write_target_inventory_tables(df: pd.DataFrame, manifest: dict, detail: dict, qtexts: dict) -> None:
    rows = []
    for sid, blk in (manifest.get("surveys") or {}).items():
        targets = [str(t) for t in blk.get("targets") or []]
        for t in targets:
            info = detail.get((sid, t), {})
            q = qtexts.get((sid, t), "")
            sub = df[(df["survey"] == sid) & (df["target"].astype(str) == t)]
            n_cells = sub.drop_duplicates(subset=["country", "condition"]).shape[0]
            rows.append(
                f"{tex_escape(sid)} & {tex_escape(t)} & {tex_escape(info.get('bucket', '-'))} & "
                f"{tex_escape(info.get('topic_key', '-'))} & {tex_escape(info.get('n_categories_metadata', '-'))} & "
                f"{tex_escape(info.get('n_categories_empirical_sample', '-'))} & {n_cells} & {tex_escape(q)} \\\\"
            )
    body = "\n".join(rows)
    table = (
        "\\begin{longtable}{lllp{2.2cm}rrrp{5.6cm}}\n\\toprule\n"
        "Survey & Target & Bucket & Topic key & Cats(meta) & Cats(emp) & Rows & Question wording \\\\\n\\midrule\n"
        "\\endfirsthead\n\\toprule\n"
        "Survey & Target & Bucket & Topic key & Cats(meta) & Cats(emp) & Rows & Question wording \\\\\n\\midrule\n"
        "\\endhead\n"
        + body
        + "\n\\bottomrule\n\\end{longtable}\n"
    )
    write_text(GEN / "appendix_target_inventory_longtable.tex", table)


def write_full_grid_table(df: pd.DataFrame) -> None:
    d = df.copy()
    d["_model_label"] = d["model"].map(model_label)
    d = d.sort_values(["survey", "target", "country", "condition", "_model_label"], kind="stable")
    lines = []
    for _, r in d.iterrows():
        lines.append(
            f"{tex_escape(r['survey'])} & {tex_escape(r['target'])} & {tex_escape(r['country'])} & "
            f"{tex_escape(r['condition'])} & {tex_escape(r['_model_label'])} & "
            f"{fmt_num(r.get('k_requested'), 0)} & {fmt_num(r.get('k_mapped'), 0)} & "
            f"{fmt_num(r.get('majority_baseline'))} & {fmt_num(r.get('oracle_acc'))} & {fmt_num(r.get('model_acc'))} & "
            f"{fmt_num(r.get('random_acc'))} & {fmt_num(r.get('cost_of_imperfect'))} & {fmt_num(r.get('value_over_random'))} & "
            f"{tex_escape(r.get('error', '') if pd.notna(r.get('error')) else '')} \\\\"
        )
    header = (
        "Survey & Target & Country & Cond. & Model & $k_r$ & $k_m$ & "
        "Maj. & Oracle & Model & Random & Cost & Value & Error \\\\"
    )
    table = (
        "\\begin{longtable}{lll l l rr rrrrrr p{2.4cm}}\n\\toprule\n"
        + header + "\n\\midrule\n\\endfirsthead\n\\toprule\n"
        + header + "\n\\midrule\n\\endhead\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{longtable}\n"
    )
    write_text(GEN / "appendix_full_grid_longtable.tex", table)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def write_mapping_and_oracle_appendix(df: pd.DataFrame) -> None:
    map_rows: list[str] = []
    oracle_rows: list[str] = []

    # Oracle top-5 is model-independent: one block per unique (survey, target, country).
    cells = df.drop_duplicates(subset=["survey", "target", "country"]).sort_values(
        ["survey", "target", "country"], kind="stable"
    )
    for _, r in cells.iterrows():
        ora_path = oracle_csv_path(r["target"], r["country"], OUT)
        if not ora_path.is_file():
            continue
        od = pd.read_csv(ora_path)
        topn = od.sort_values("importance_mean", ascending=False).head(5)
        for _, o in topn.iterrows():
            oracle_rows.append(
                f"{tex_escape(r['survey'])} & {tex_escape(r['target'])} & {tex_escape(r['country'])} & "
                f"{tex_escape(o['feature_variable'])} & {fmt_num(o['importance_mean'])} & {fmt_num(o['importance_std'])} \\\\"
            )

    # Mapping audit IS model-specific: one block per unique (survey, target, country, model).
    cell_models = df.drop_duplicates(subset=["survey", "target", "country", "model"]).sort_values(
        ["survey", "target", "country", "model"], kind="stable"
    )
    for _, r in cell_models.iterrows():
        tag = r["model"]
        dis_path = cell_dir(str(r["target"]), str(r["country"]), OUT) / f"llm__{tag}" / "disambig.json"
        if not dis_path.is_file():
            continue
        try:
            mappings = read_json(dis_path)
        except Exception:
            mappings = []
        for m in mappings:
            sel = (m.get("disambig") or {}).get("selected_code")
            cands = m.get("candidates") or []
            top = cands[0]["var_code"] if cands else None
            map_rows.append(
                f"{tex_escape(r['survey'])} & {tex_escape(m.get('target'))} & {tex_escape(r['country'])} & "
                f"{tex_escape(model_label(tag))} & {tex_escape(m.get('condition'))} & {int(m.get('feature_rank', -1))} & "
                f"{tex_escape(m.get('feature_label'))} & {tex_escape(sel or '-')}"
                f" & {tex_escape(top or '-')} & {len(cands)} \\\\"
            )

    map_header = (
        "Survey & Target & Country & Model & Cond. & Rank & Requested feature & "
        "Selected code & Top retrieval & N cands \\\\"
    )
    map_table = (
        "\\begin{longtable}{lll l llr p{4.8cm} llr}\n\\toprule\n"
        + map_header + "\n\\midrule\n\\endfirsthead\n\\toprule\n"
        + map_header + "\n\\midrule\n\\endhead\n"
        + "\n".join(map_rows)
        + "\n\\bottomrule\n\\end{longtable}\n"
    )
    write_text(GEN / "appendix_mapping_audit_longtable.tex", map_table)

    oracle_table = (
        "\\begin{longtable}{lll lrr}\n\\toprule\n"
        "Survey & Target & Country & Oracle feature & Importance mean & Importance std \\\\\n\\midrule\n"
        "\\endfirsthead\n\\toprule\n"
        "Survey & Target & Country & Oracle feature & Importance mean & Importance std \\\\\n\\midrule\n"
        "\\endhead\n"
        + "\n".join(oracle_rows)
        + "\n\\bottomrule\n\\end{longtable}\n"
    )
    write_text(GEN / "appendix_oracle_top5_longtable.tex", oracle_table)


def write_prompt_fragments() -> None:
    body = wrap_verbatim_plain(
        f"SYSTEM: {SYSTEM_PROMPT}\n\n"
        "UNPROMPTED TEMPLATE:\n"
        f"{PROMPT_UNPROMPTED}\n\n"
        "COUNTRY-PROVIDED TEMPLATE:\n"
        f"{PROMPT_COUNTRY}"
    )
    prompt = "\\begin{verbatim}\n" + body + "\n\\end{verbatim}\n"
    write_text(GEN / "prompt_templates.tex", prompt)
    dis = "\\begin{verbatim}\n" + wrap_verbatim_plain(DISAMBIG_PROMPT) + "\n\\end{verbatim}\n"
    write_text(GEN / "disambig_prompt.tex", dis)


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    df = load_all_summary()
    manifest = load_manifest()
    detail = load_target_detail()
    qtexts = get_question_texts()
    write_prompt_fragments()
    write_metrics_tables(df)
    write_bucket_and_extremes_tables()
    write_target_inventory_tables(df, manifest, detail, qtexts)
    write_full_grid_table(df)
    write_mapping_and_oracle_appendix(df)
    print(f"Wrote report artifacts to {GEN}")


if __name__ == "__main__":
    main()
