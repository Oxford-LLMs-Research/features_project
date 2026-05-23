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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase0b_oracle_autogluon import _MISSING_LABEL_PATTERNS
from phase0b_pipeline import PROMPT_COUNTRY, PROMPT_UNPROMPTED, SYSTEM_PROMPT
from phase0b_disambig import DISAMBIG_PROMPT

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
    from output_layout import parse_grid_summary_stem, resolve_grid_summary_for_survey

    dfs: list[pd.DataFrame] = []
    for sid in SURVEY_ORDER:
        p = resolve_grid_summary_for_survey(OUT, sid)
        if not p or not p.is_file():
            continue
        d = pd.read_csv(p)
        d["survey"] = sid
        _, tag = parse_grid_summary_stem(p.stem)
        if tag:
            d["llm_run_tag"] = tag
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
        from run_grid import SURVEY_COUNTRY_COL, load_survey, get_question_text
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


def write_metrics_tables(df: pd.DataFrame) -> None:
    valid = df[df["oracle_acc"].notna() & df["model_acc"].notna() & df["random_acc"].notna()].copy()

    global_tab = rf"""\begin{{tabular}}{{lr}}
\toprule
Metric & Value \\
\midrule
Total rows & {len(df)} \\
Valid rows & {len(valid)} \\
Mean oracle accuracy & {fmt_num(valid['oracle_acc'].mean())} \\
Mean model accuracy & {fmt_num(valid['model_acc'].mean())} \\
Mean random-$k$ accuracy & {fmt_num(valid['random_acc'].mean())} \\
Mean majority baseline & {fmt_num(valid['majority_baseline'].mean())} \\
Mean cost of imperfect & {fmt_num(valid['cost_of_imperfect'].mean())} \\
Mean value over random & {fmt_num(valid['value_over_random'].mean())} \\
Share value\textgreater 0 & {fmt_num((valid['value_over_random'] > 0).mean(), 3)} \\
\bottomrule
\end{{tabular}}
"""
    write_text(GEN / "main_global_metrics.tex", global_tab)

    rows = []
    for sid, sub in valid.groupby("survey", sort=False):
        rows.append(
            f"{tex_escape(sid)} & {len(sub)} & {fmt_num(sub['oracle_acc'].mean())} & "
            f"{fmt_num(sub['model_acc'].mean())} & {fmt_num(sub['random_acc'].mean())} & "
            f"{fmt_num(sub['cost_of_imperfect'].mean())} & {fmt_num(sub['value_over_random'].mean())} & "
            f"{fmt_num((sub['value_over_random'] > 0).mean(), 3)} \\\\"
        )
    survey_tab = (
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        "Survey & N & Oracle & Model & Random & Cost & Value & Share value$>0$ \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(GEN / "main_survey_metrics.tex", survey_tab)

    cond_rows = []
    for cond in ("unprompted", "country_provided"):
        sub = valid[valid["condition"] == cond]
        cond_rows.append(
            f"{tex_escape(cond)} & {len(sub)} & {fmt_num(sub['cost_of_imperfect'].mean())} & "
            f"{fmt_num(sub['value_over_random'].mean())} & {fmt_num((sub['value_over_random'] > 0).mean(), 3)} \\\\"
        )
    cond_tab = (
        "\\begin{tabular}{lrrrr}\n\\toprule\nCondition & N & Mean cost & Mean value & Share value$>0$ \\\\\n\\midrule\n"
        + "\n".join(cond_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(GEN / "main_condition_metrics.tex", cond_tab)


def write_bucket_and_extremes_tables() -> None:
    stats_path = OUT / "_prelim_stats.json"
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
                if isinstance(v, (int, float)):
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
        ["survey", "target", "country", "condition", "cost_of_imperfect", "value_over_random"],
        "appendix_top_cost_longtable.tex",
    )
    top_rows(
        "worst_value_over_random",
        ["survey", "target", "country", "condition", "value_over_random", "cost_of_imperfect"],
        "appendix_worst_value_longtable.tex",
    )
    top_rows(
        "best_value_over_random",
        ["survey", "target", "country", "condition", "value_over_random"],
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
            rows.append(
                f"{tex_escape(sid)} & {tex_escape(t)} & {tex_escape(info.get('bucket', '-'))} & "
                f"{tex_escape(info.get('topic_key', '-'))} & {tex_escape(info.get('n_categories_metadata', '-'))} & "
                f"{tex_escape(info.get('n_categories_empirical_sample', '-'))} & {len(sub)} & {tex_escape(q)} \\\\"
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
    d = d.sort_values(["survey", "target", "country", "condition"], kind="stable")
    lines = []
    for _, r in d.iterrows():
        lines.append(
            f"{tex_escape(r['survey'])} & {tex_escape(r['target'])} & {tex_escape(r['country'])} & "
            f"{tex_escape(r['condition'])} & {fmt_num(r.get('k_requested'), 0)} & {fmt_num(r.get('k_mapped'), 0)} & "
            f"{fmt_num(r.get('majority_baseline'))} & {fmt_num(r.get('oracle_acc'))} & {fmt_num(r.get('model_acc'))} & "
            f"{fmt_num(r.get('random_acc'))} & {fmt_num(r.get('cost_of_imperfect'))} & {fmt_num(r.get('value_over_random'))} & "
            f"{tex_escape(r.get('error', '') if pd.notna(r.get('error')) else '')} \\\\"
        )
    table = (
        "\\begin{longtable}{lll lrr rrrrrr p{2.8cm}}\n\\toprule\n"
        "Survey & Target & Country & Cond. & $k_r$ & $k_m$ & Maj. & Oracle & Model & Random & Cost & Value & Error \\\\\n\\midrule\n"
        "\\endfirsthead\n\\toprule\n"
        "Survey & Target & Country & Cond. & $k_r$ & $k_m$ & Maj. & Oracle & Model & Random & Cost & Value & Error \\\\\n\\midrule\n"
        "\\endhead\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{longtable}\n"
    )
    write_text(GEN / "appendix_full_grid_longtable.tex", table)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def write_mapping_and_oracle_appendix(df: pd.DataFrame) -> None:
    from output_layout import resolve_llm_artifact

    map_rows: list[str] = []
    oracle_rows: list[str] = []
    for _, r in df.sort_values(["survey", "target", "country"], kind="stable").iterrows():
        prefix = f"{r['target']}_{r['country']}"
        d = OUT / prefix
        dis_path = resolve_llm_artifact(OUT, str(r["target"]), str(r["country"]), "disambig.json")
        eval_path = resolve_llm_artifact(OUT, str(r["target"]), str(r["country"]), "eval.json")
        ora_path = d / "oracle.csv"
        if dis_path and dis_path.is_file():
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
                    f"{tex_escape(m.get('condition'))} & {int(m.get('feature_rank', -1))} & "
                    f"{tex_escape(m.get('feature_label'))} & {tex_escape(sel or '-')}"
                    f" & {tex_escape(top or '-')} & {len(cands)} \\\\"
                )
        if eval_path and eval_path.is_file():
            _ = eval_path
        if ora_path.is_file():
            od = pd.read_csv(ora_path)
            topn = od.sort_values("importance_mean", ascending=False).head(5)
            for _, o in topn.iterrows():
                oracle_rows.append(
                    f"{tex_escape(r['survey'])} & {tex_escape(r['target'])} & {tex_escape(r['country'])} & "
                    f"{tex_escape(o['feature_variable'])} & {fmt_num(o['importance_mean'])} & {fmt_num(o['importance_std'])} \\\\"
                )

    map_table = (
        "\\begin{longtable}{llllr p{5.8cm} llr}\n\\toprule\n"
        "Survey & Target & Country & Cond. & Rank & Requested feature & Selected code & Top retrieval & N cands \\\\\n\\midrule\n"
        "\\endfirsthead\n\\toprule\n"
        "Survey & Target & Country & Cond. & Rank & Requested feature & Selected code & Top retrieval & N cands \\\\\n\\midrule\n"
        "\\endhead\n"
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
