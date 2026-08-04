"""
LEGACY grid runner (pilot-1, strict-JSON elicitation) — kept runnable so the paper's
appendix results stay reproducible. The current pipeline is scripts/run_main.py.

Runs any subset of targets × countries end-to-end:
  1. Oracle permutation importances (AutoGluon; cached, model-independent)
  2. LLM feature selection via the JSON prompts (unprompted + country_provided)
  3. Embedding-retrieval candidate mapping (batch top-5)
  4. LLM shortlist disambiguation (fixed small model)
  5. Downstream XGBoost prediction comparison

Artifact paths (via survey_features.layout; dual-resolves legacy locations):
  outputs/cache/cells/<target>_<country>/oracle.csv   (or legacy outputs/<t>_<c>/)
  outputs/cache/cells/<target>_<country>/llm__<run_tag>/
  outputs/grid/grid_summary__<survey>__<run_tag>.csv  (or legacy outputs/ root)
  outputs/grid/llm_usage__<survey>__<run_tag>.jsonl

Usage:
    python scripts/run_grid.py                                             # full WVS 5×5 grid
    python scripts/run_grid.py --targets Q164 --countries Germany         # single WVS cell
    python scripts/run_grid.py --survey afrobarometer --targets Q1 --countries Nigeria Ghana
    python scripts/run_grid.py --survey afrobarometer --list-countries    # show available countries

Country names and admin columns are derived automatically from the survey
metadata — no hardcoding needed. Use --list-countries to see valid names for
any survey.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

# Force stdout/stderr to UTF-8 so logger / print() calls emitting non-ASCII
# (e.g. AutoGluon's '±' in summaries, the token-usage '~' tally, future
# Unicode in eval output) don't crash on Windows' default cp1252 console.
# Without this, a UnicodeEncodeError at the end of one survey aborts the
# whole multi-survey PS1 pipeline (we lost ~2.5h of runtime to this).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import yaml

from survey_features.config import (
    DEFAULT_EMBEDDING_MODEL,
    DISAMBIG_API_KEY,
    DISAMBIG_BASE_URL,
    DISAMBIG_MODEL,
    OUTPUTS_DIR,
)
from survey_features.layout import (
    cell_dir as layout_cell_dir,
    grid_results_json_path,
    grid_summary_csv_path,
    llm_cache_prefix,
    llm_usage_path,
    manifest_path,
    oracle_csv_path,
    sanitize_model_slug,
)
from survey_features.oracle import (
    SIMILARITY_THRESHOLD,
    compute_oracle,
    load_similarity_model,
)
from survey_features.surveys import (
    SURVEY_COUNTRY_COL,
    build_admin_cols,
    build_country_code_map,
    clean_question_columns,
    flatten_metadata,
    get_question_text,
    load_survey,
)

OUTPUTS_DIR.mkdir(exist_ok=True)

# Default grid (WVS only; always specify --targets and --countries for other surveys).
DEFAULT_SURVEY = "wvs"
DEFAULT_TARGETS = ["Q49", "Q199", "Q71", "Q165", "Q240"]
DEFAULT_COUNTRIES = ["Germany", "Nigeria", "Japan", "Brazil", "Egypt"]

DEFAULT_GRID_WORKERS = 5


def resolve_xgb_nthread(grid_workers: int) -> int | None:
    """
    XGBoost nthread budget per model fit when running concurrent grid cells.
    Set GRID_XGB_NTHREAD in the environment to override (integer >= 1).
    """
    raw = os.environ.get("GRID_XGB_NTHREAD")
    if raw is not None and raw.strip():
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    cpus = os.cpu_count() or 10
    return max(1, cpus // max(1, grid_workers))


def resolve_n_jobs_random(grid_workers: int, xgb_nthread: int) -> int:
    cpus = os.cpu_count() or 10
    budget = cpus // max(1, grid_workers) // max(1, xgb_nthread)
    return max(1, min(budget, 24))


# ── LLM selection + mapping ───────────────────────────────────────────────────

def run_llm_and_map(
    target_var: str,
    question_text: str,
    country_name: str | None,
    condition: str,
    generate_fn,
    disambig_generate_fn,
    model_name: str,
    survey_variables: dict[str, str],
    survey_embeddings: np.ndarray,
    var_codes: list[str],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:
    from survey_features.disambig import disambiguate_mappings
    from survey_features.elicitation import run_single
    from survey_features.retrieval import map_features_to_variables

    conditions_to_run = []
    if condition in ("unprompted", "both"):
        conditions_to_run.append(("unprompted", None))
    if condition in ("country_provided", "both"):
        conditions_to_run.append(("country_provided", country_name))

    all_llm_results = []
    for cond_name, country_arg in conditions_to_run:
        print(f"\n  [{cond_name}] Querying {model_name} ...")
        r = run_single(target_var, question_text, country_arg, model_name, generate_fn)
        status = f"{r['n_features']} features" if r["features"] else f"PARSE ERROR: {r['parse_error']}"
        print(f"  [{cond_name}] {status}")
        all_llm_results.append(r)

    if not any(r["features"] for r in all_llm_results):
        print("  No parseable LLM output. Check raw responses.")
        for r in all_llm_results:
            print(f"  Raw ({r['condition']}):\n{r['raw_response'][:500]}")
        return []

    print("\n  Embedding + retrieving candidates ...")
    mappings = map_features_to_variables(
        all_llm_results, survey_variables, survey_embeddings, var_codes,
        model_name=embedding_model,
        exclude_targets=True,
    )
    print(f"  {len(mappings)} feature->candidate pairs")

    print(f"\n  Disambiguating (fixed model: {DISAMBIG_MODEL}) ...")
    mappings = disambiguate_mappings(mappings, disambig_generate_fn, model=DISAMBIG_MODEL)

    mapped = sum(1 for m in mappings if m["disambig"]["selected_code"])
    print(f"\n  Mapped: {mapped}/{len(mappings)} features -> survey variables")

    return mappings


# ── Per-cell cache helpers ────────────────────────────────────────────────────

def cell_dir(prefix: str) -> Path:
    """Resolve a cell or llm__ subdir. ``prefix`` is ``<t>_<c>`` or ``<t>_<c>/llm__<tag>``."""
    norm = str(prefix).replace("\\", "/")
    if "/llm__" in norm:
        base_key, llm_rest = norm.split("/llm__", 1)
        # Reconstruct target/country from key by preferring existing dirs (same as layout.cell_dir).
        # We only have the joined key here; use cache/legacy dual-resolve on the key.
        from survey_features.layout import cache_cells_dir
        new_base = cache_cells_dir(OUTPUTS_DIR) / base_key
        old_base = OUTPUTS_DIR / base_key
        if new_base.is_dir():
            base = new_base
        elif old_base.is_dir():
            base = old_base
        else:
            base = new_base
        d = base / f"llm__{llm_rest}"
    else:
        from survey_features.layout import cache_cells_dir
        new = cache_cells_dir(OUTPUTS_DIR) / prefix
        old = OUTPUTS_DIR / prefix
        if new.is_dir():
            d = new
        elif old.is_dir():
            d = old
        else:
            d = new
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_or_compute_oracle(
    target: str,
    country_name: str,
    country_code: int | str,
    country_col: str,
    admin_cols: frozenset[str],
    data: pd.DataFrame,
    metadata: dict,
    prefix: str,
    metadata_flat: dict | None = None,
    similarity_model: object | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    existing = oracle_csv_path(target, country_name)
    if existing.is_file():
        print(f"  [oracle] Loading cached {existing}")
        oracle_df = pd.read_csv(existing)
        feature_pool = oracle_df[oracle_df["target_variable"] == target]["feature_variable"].tolist()
        return oracle_df, feature_pool

    print(f"  [oracle] Computing ({target} x {country_name}) ...")
    oracle_df, feature_pool, meta_out = compute_oracle(
        data,
        metadata,
        target,
        country_code,
        country_col,
        admin_cols,
        metadata_flat=metadata_flat,
        similarity_model=similarity_model,
    )
    d = layout_cell_dir(target, country_name)
    d.mkdir(parents=True, exist_ok=True)
    oracle_path = d / "oracle.csv"
    oracle_df.to_csv(oracle_path, index=False)
    (d / "oracle_meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"  [oracle] Saved {oracle_path}")
    return oracle_df, feature_pool


def get_or_run_llm_mapping(
    target: str,
    question_text: str,
    country_name: str,
    generate_fn,
    disambig_generate_fn,
    model_name: str,
    survey_variables: dict[str, str],
    survey_embeddings: np.ndarray,
    var_codes: list[str],
    prefix: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:
    disambig_path = cell_dir(prefix) / "disambig.json"
    if disambig_path.exists():
        print(f"  [llm] Loading cached {prefix}/disambig.json")
        with open(disambig_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"  [llm] Running LLM selection + mapping + disambig ...")
    mappings = run_llm_and_map(
        target_var=target,
        question_text=question_text,
        country_name=country_name,
        condition="both",
        generate_fn=generate_fn,
        disambig_generate_fn=disambig_generate_fn,
        model_name=model_name,
        survey_variables=survey_variables,
        survey_embeddings=survey_embeddings,
        var_codes=var_codes,
        embedding_model=embedding_model,
    )

    with open(disambig_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    print(f"  [llm] Saved {prefix}/disambig.json")
    return mappings


def get_or_run_eval(
    target: str,
    country_name: str,
    country_code: int | str,
    country_col: str,
    mappings: list[dict],
    oracle_df: pd.DataFrame,
    feature_pool: list[str],
    data: pd.DataFrame,
    prefix: str,
    n_jobs_random: int = -1,
    eval_xgb_nthread: int | None = None,
) -> tuple[dict, list[dict]]:
    d = cell_dir(prefix)
    eval_path = d / "eval.json"
    disambig_path = d / "disambig.json"

    if eval_path.exists() and disambig_path.exists():
        if eval_path.stat().st_mtime >= disambig_path.stat().st_mtime:
            print(f"  [eval] Loading cached {prefix}/eval.json")
            with open(eval_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached["results"], cached.get("errors", [])

    results, errors = run_eval_per_condition(
        data=data,
        target_var=target,
        country_code=country_code,
        country_col=country_col,
        mappings=mappings,
        oracle_df=oracle_df,
        feature_pool=feature_pool,
        n_jobs_random=n_jobs_random,
        eval_xgb_nthread=eval_xgb_nthread,
    )

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2,
                  ensure_ascii=False, default=str)
    print(f"  [eval] Saved {prefix}/eval.json")
    return results, errors


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_eval_per_condition(
    data: pd.DataFrame,
    target_var: str,
    country_code: int | str,
    country_col: str,
    mappings: list[dict],
    oracle_df: pd.DataFrame,
    feature_pool: list[str],
    n_jobs_random: int = -1,
    eval_xgb_nthread: int | None = None,
) -> tuple[dict, list[dict]]:
    from survey_features.evaluation import print_comparison, run_comparison

    results: dict[str, dict] = {}
    errors: list[dict] = []

    conditions = sorted({m["condition"] for m in mappings})
    for cond in conditions:
        cond_maps = [m for m in mappings if m["condition"] == cond]

        seen: set[str] = set()
        model_features: list[str] = []
        for m in cond_maps:
            code = m["disambig"]["selected_code"]
            if code is None or code in seen:
                continue
            seen.add(code)
            model_features.append(code)

        print(f"\n  [{cond}]")
        try:
            result = run_comparison(
                data=data,
                target_var=target_var,
                country_code=country_code,
                country_col=country_col,
                model_features=model_features,
                oracle_importances=oracle_df,
                all_feature_pool=feature_pool,
                n_jobs=n_jobs_random,
                eval_xgb_nthread=eval_xgb_nthread,
            )
            print_comparison(result)
            results[cond] = result
        except Exception as e:
            traceback.print_exc()
            print(f"  [error] condition {cond} failed: {e}")
            errors.append({"condition": cond, "error": f"{type(e).__name__}: {e}"})

    return results, errors


# ── Result flattening ─────────────────────────────────────────────────────────

def flatten_eval_result(
    target: str,
    country_name: str,
    country_code: int | str,
    condition: str,
    cell_result: dict,
    *,
    llm_model: str,
    llm_run_tag: str,
) -> dict:
    row = {
        "target": target,
        "country": country_name,
        "country_code": country_code,
        "condition": condition,
        "llm_model": llm_model,
        "llm_run_tag": llm_run_tag,
        "k_requested": cell_result.get("k_requested"),
        "k_mapped": cell_result.get("k_mapped"),
        "majority_baseline": None,
        "oracle_acc": None,
        "oracle_std": None,
        "model_acc": None,
        "model_std": None,
        "random_acc": None,
        "random_std": None,
        "cost_of_imperfect": None,
        "value_over_random": None,
        "error": cell_result.get("error"),
    }

    o = cell_result.get("oracle") or {}
    m = cell_result.get("model") or {}
    r = cell_result.get("random") or {}

    row["majority_baseline"] = o.get("majority_baseline")
    row["oracle_acc"] = o.get("accuracy_mean")
    row["oracle_std"] = o.get("accuracy_std")
    row["model_acc"] = m.get("accuracy_mean")
    row["model_std"] = m.get("accuracy_std")
    row["random_acc"] = r.get("accuracy_mean")
    row["random_std"] = r.get("accuracy_std")

    if row["oracle_acc"] is not None and row["model_acc"] is not None:
        row["cost_of_imperfect"] = round(row["oracle_acc"] - row["model_acc"], 4)
    if row["model_acc"] is not None and row["random_acc"] is not None:
        row["value_over_random"] = round(row["model_acc"] - row["random_acc"], 4)

    return row


def print_summary_table(summary_rows: list[dict]) -> None:
    if not summary_rows:
        print("\n(no results to summarise)")
        return

    header = (
        f"{'target':6s} {'country':10s} {'condition':17s} "
        f"{'k':>3s} {'base':>6s} {'oracle':>7s} {'model':>7s} {'random':>7s} "
        f"{'cost':>7s} {'value':>7s}"
    )
    print("\n" + "=" * len(header))
    print("Grid summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        def fmt(v, width, places=4):
            if v is None:
                return f"{'-':>{width}s}"
            return f"{v:>{width}.{places}f}"

        print(
            f"{row['target']:6s} {row['country']:10s} {row['condition']:17s} "
            f"{(row['k_mapped'] if row['k_mapped'] is not None else 0):>3d} "
            f"{fmt(row['majority_baseline'], 6, 3)} "
            f"{fmt(row['oracle_acc'], 7)} "
            f"{fmt(row['model_acc'], 7)} "
            f"{fmt(row['random_acc'], 7)} "
            f"{fmt(row['cost_of_imperfect'], 7)} "
            f"{fmt(row['value_over_random'], 7)}"
        )


# ── Per-cell worker ───────────────────────────────────────────────────────────

@dataclass
class PipelineContext:
    data: pd.DataFrame
    eval_data: pd.DataFrame
    metadata: dict
    metadata_flat: dict
    similarity_model: object | None
    survey_variables: dict[str, str]
    survey_embeddings: np.ndarray | None
    var_codes: list[str]
    generate_fn: Any
    disambig_generate_fn: Any
    model_name: str
    output_tag: str
    embedding_model: str
    n_jobs_random: int
    xgb_nthread: int | None
    stop_after: str
    country_col: str
    admin_cols: frozenset[str]
    country_codes: dict[str, int | str]


def run_cell(ctx: PipelineContext, target: str, country_name: str) -> dict:
    country_code = ctx.country_codes[country_name]
    prefix = f"{target}_{country_name}"

    try:
        oracle_df, feature_pool = get_or_compute_oracle(
            target=target,
            country_name=country_name,
            country_code=country_code,
            country_col=ctx.country_col,
            admin_cols=ctx.admin_cols,
            data=ctx.data,
            metadata=ctx.metadata,
            prefix=prefix,
            metadata_flat=ctx.metadata_flat,
            similarity_model=ctx.similarity_model,
        )

        if ctx.stop_after == "oracle":
            return {
                "target": target,
                "country_name": country_name,
                "country_code": country_code,
                "prefix": prefix,
                "eval_results": {},
                "cond_errors": [],
                "error": None,
                "stopped_after_oracle": True,
            }

        question_text = get_question_text(target, ctx.metadata)

        assert ctx.survey_embeddings is not None and ctx.generate_fn is not None
        llm_prefix = llm_cache_prefix(prefix, ctx.output_tag)
        mappings = get_or_run_llm_mapping(
            target=target,
            question_text=question_text,
            country_name=country_name,
            generate_fn=ctx.generate_fn,
            disambig_generate_fn=ctx.disambig_generate_fn,
            model_name=ctx.model_name,
            survey_variables=ctx.survey_variables,
            survey_embeddings=ctx.survey_embeddings,
            var_codes=ctx.var_codes,
            prefix=llm_prefix,
            embedding_model=ctx.embedding_model,
        )

        if not mappings:
            return {
                "target": target,
                "country_name": country_name,
                "country_code": country_code,
                "prefix": prefix,
                "eval_results": {},
                "cond_errors": [],
                "error": "no mappings",
                "stopped_after_oracle": False,
            }

        eval_results, cond_errors = get_or_run_eval(
            target=target,
            country_name=country_name,
            country_code=country_code,
            country_col=ctx.country_col,
            mappings=mappings,
            oracle_df=oracle_df,
            feature_pool=feature_pool,
            data=ctx.eval_data,
            prefix=llm_prefix,
            n_jobs_random=ctx.n_jobs_random,
            eval_xgb_nthread=ctx.xgb_nthread,
        )

        return {
            "target": target,
            "country_name": country_name,
            "country_code": country_code,
            "prefix": prefix,
            "eval_results": eval_results,
            "cond_errors": cond_errors,
            "error": None,
            "stopped_after_oracle": False,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "target": target,
            "country_name": country_name,
            "country_code": country_code,
            "prefix": prefix,
            "eval_results": {},
            "cond_errors": [],
            "error": f"{type(e).__name__}: {e}",
            "stopped_after_oracle": False,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature selection grid runner (legacy JSON pipeline).")
    parser.add_argument(
        "--survey", default=DEFAULT_SURVEY,
        choices=list(SURVEY_COUNTRY_COL),
        help=f"Survey to run (default: {DEFAULT_SURVEY})",
    )
    parser.add_argument(
        "--targets", nargs="+", default=None,
        metavar="VAR",
        help=f"Target variable(s). Defaults to {DEFAULT_TARGETS} when --survey=wvs.",
    )
    parser.add_argument(
        "--countries", nargs="+", default=None,
        metavar="COUNTRY",
        help=f"Country name(s). Defaults to {DEFAULT_COUNTRIES} when --survey=wvs.",
    )
    parser.add_argument(
        "--list-countries", action="store_true",
        help="Print all available countries for the chosen survey and exit.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("full", "oracle"),
        default="full",
        help="Stop pipeline after permutation-importance oracle (skip LLM, mapping, evaluation).",
    )
    parser.add_argument(
        "--grid-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max concurrent (target × country) cells "
            f"(default: {DEFAULT_GRID_WORKERS}, capped by CPU cell count)."
        ),
    )
    parser.add_argument(
        "--from-manifest",
        metavar="YAML",
        default=None,
        help="YAML file with surveys → {targets: [...], countries: [...]}; overrides CLI lists.",
    )
    parser.add_argument(
        "--run-tag",
        metavar="TAG",
        default=None,
        help=(
            "Experiment ID (tag) for llm__/ grid_summary / manifest paths "
            "(default: slug from LLM_MODEL). Set explicitly to distinguish runs with "
            "different models, prompts, or embedding models."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        metavar="MODEL",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence-transformer model for survey variable embeddings (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--prompt-variant",
        metavar="VARIANT",
        default="default",
        help="Prompt variant label recorded in the run manifest (default: default).",
    )
    args = parser.parse_args()

    survey_id = args.survey
    country_col = SURVEY_COUNTRY_COL[survey_id]

    targets = args.targets or (DEFAULT_TARGETS if survey_id == DEFAULT_SURVEY else None)
    countries = args.countries or (DEFAULT_COUNTRIES if survey_id == DEFAULT_SURVEY else None)

    if args.from_manifest:
        mf_path = Path(args.from_manifest)
        if not mf_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {mf_path.resolve()}")
        with open(mf_path, encoding="utf-8") as f:
            mf_doc = yaml.safe_load(f) or {}
        block = (mf_doc.get("surveys") or {}).get(survey_id)
        if not block:
            raise KeyError(f"survey '{survey_id}' missing under 'surveys' in {mf_path}")
        targets = block.get("targets") or []
        countries = block.get("countries") or []

    config_path = os.environ.get("DATA_CONFIG_PATH")
    if not config_path:
        raise ValueError("DATA_CONFIG_PATH is not set in .env")

    print(f"\n[setup] Loading {survey_id} data ...")
    data, metadata = load_survey(survey_id, config_path)
    print(f"  Total rows: {len(data)}")

    country_codes = build_country_code_map(metadata, country_col, data)
    admin_cols = build_admin_cols(metadata, country_col)
    print(f"  {len(country_codes)} countries, {len(admin_cols)} admin columns derived from metadata")

    print("\n[setup] Building oracle metadata index and similarity model ...")
    metadata_flat = flatten_metadata(metadata)
    similarity_model = load_similarity_model(SIMILARITY_THRESHOLD)
    print(f"  {len(metadata_flat)} vars in flat metadata index; similarity model loaded")

    if args.list_countries:
        print(f"\nAvailable countries for '{survey_id}':")
        for name in sorted(country_codes):
            print(f"  {name} ({country_codes[name]})")
        return

    if not targets:
        parser.error(f"--targets is required for survey '{survey_id}'")
    if not countries:
        parser.error(f"--countries is required for survey '{survey_id}'")

    unknown = [c for c in countries if c not in country_codes]
    if unknown:
        raise ValueError(
            f"Unknown country/ies {unknown} for survey '{survey_id}'. "
            f"Run with --list-countries to see valid names."
        )

    n_cells = len(targets) * len(countries)
    gw = args.grid_workers if args.grid_workers is not None else DEFAULT_GRID_WORKERS
    gw = max(1, gw)
    n_workers = min(gw, n_cells)
    xgb_nt = resolve_xgb_nthread(n_workers)
    n_jobs_random = resolve_n_jobs_random(n_workers, int(xgb_nt or 1))

    print("\n" + "=" * 72)
    print(f"Survey: {survey_id}  |  country column: {country_col}")
    print(f"Grid runner: {len(targets)} target(s) x {len(countries)} country/ies = {n_cells} cell(s)")
    print(f"Targets:   {targets}")
    print(f"Countries: {countries}")
    print(f"Workers: {n_workers}, XGB nthread/cell: {xgb_nt}, joblib n_jobs (random): {n_jobs_random}")
    print("=" * 72)

    print("\n[setup] Cleaning question columns for evaluation ...")
    eval_data = clean_question_columns(data, country_col, admin_cols, metadata)

    print("\n[setup] Building survey variable index ...")
    from survey_features.surveys import extract_survey_variables
    survey_variables = extract_survey_variables(metadata)
    print(f"  {len(survey_variables)} survey variables")

    stop_after = args.stop_after
    embedding_model: str = args.embedding_model
    output_tag = ""
    llm_usage_log = None
    if stop_after == "oracle":
        print("\n[setup] --stop-after oracle: skipping embeddings and LLM client init")
        survey_embeddings = None
        var_codes: list[str] = []
        generate_fn = None
        disambig_generate_fn = None
        model_name = ""
    else:
        print(f"\n[setup] Loading / building survey embeddings (model: {embedding_model}) ...")
        from survey_features.retrieval import load_or_build_survey_embeddings
        survey_embeddings, var_codes = load_or_build_survey_embeddings(
            survey_variables, survey_id, embedding_model
        )

        print("\n[setup] Initialising LLM client ...")
        from survey_features.llm import TokenUsageLog, make_generate_fn

        usage_log_ref: list[TokenUsageLog | None] = [None]
        generate_fn, model_name = make_generate_fn(usage_log_ref=usage_log_ref)
        print(f"  Model: {model_name}")
        output_tag = sanitize_model_slug(args.run_tag) if args.run_tag else sanitize_model_slug(model_name)
        llm_usage_log = TokenUsageLog(
            llm_usage_path(OUTPUTS_DIR, survey_id, output_tag)
        )
        usage_log_ref[0] = llm_usage_log
        print(f"  Experiment ID / output tag: {output_tag}")
        print(f"  Embedding model: {embedding_model}  |  Prompt variant: {args.prompt_variant}")

        print(f"\n[setup] Initialising disambiguation client ({DISAMBIG_MODEL}) ...")
        disambig_generate_fn, _ = make_generate_fn(
            base_url=DISAMBIG_BASE_URL,
            api_key=DISAMBIG_API_KEY,
            model=DISAMBIG_MODEL,
            usage_log_ref=usage_log_ref,
        )
        disambig_endpoint = DISAMBIG_BASE_URL or "(same as LLM_BASE_URL)"
        print(f"  Disambig model: {DISAMBIG_MODEL}  endpoint: {disambig_endpoint}")

    grid_summary_csv = grid_summary_csv_path(OUTPUTS_DIR, survey_id, output_tag) if output_tag else None
    grid_results_json = grid_results_json_path(OUTPUTS_DIR, survey_id, output_tag) if output_tag else None

    summary_rows: list[dict] = []
    full_results: dict = {}
    errors: list[dict] = []
    started_at = datetime.now(timezone.utc)

    ctx = PipelineContext(
        data=data,
        eval_data=eval_data,
        metadata=metadata,
        metadata_flat=metadata_flat,
        similarity_model=similarity_model,
        survey_variables=survey_variables,
        survey_embeddings=survey_embeddings,
        var_codes=var_codes,
        generate_fn=generate_fn,
        disambig_generate_fn=disambig_generate_fn,
        model_name=model_name,
        output_tag=output_tag,
        embedding_model=embedding_model,
        n_jobs_random=n_jobs_random,
        xgb_nthread=xgb_nt,
        stop_after=stop_after,
        country_col=country_col,
        admin_cols=admin_cols,
        country_codes={c: country_codes[c] for c in countries},
    )

    cells = list(product(targets, countries))
    total = len(cells)
    print(f"\n[grid] Running {total} cell(s) across {n_workers} thread(s) "
          f"(n_jobs_random={n_jobs_random} per cell)")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_cell = {
            executor.submit(run_cell, ctx, t, c): (t, c) for t, c in cells
        }
        completed = 0
        for future in as_completed(future_to_cell):
            completed += 1
            result = future.result()
            target = result["target"]
            country_name = result["country_name"]
            country_code = result["country_code"]
            prefix = result["prefix"]

            print(f"\n[{completed}/{total}] Done: {target} x {country_name}")

            if result.get("stopped_after_oracle"):
                print("  (--stop-after oracle: skipped LLM/eval; oracle.csv cached)")
                continue

            if result["error"]:
                msg = result["error"]
                print(f"  [error] {msg}")
                errors.append({"target": target, "country": country_name, "error": msg})
                continue

            eval_results = result["eval_results"]
            cond_errors = result["cond_errors"]

            full_results[prefix] = eval_results
            for condition, cell_result in eval_results.items():
                summary_rows.append(
                    flatten_eval_result(
                        target,
                        country_name,
                        country_code,
                        condition,
                        cell_result,
                        llm_model=ctx.model_name,
                        llm_run_tag=ctx.output_tag,
                    )
                )
            for ce in cond_errors:
                errors.append({"target": target, "country": country_name,
                               "condition": ce["condition"], "error": ce["error"]})

            assert grid_summary_csv is not None and grid_results_json is not None
            pd.DataFrame(summary_rows).to_csv(grid_summary_csv, index=False)
            with open(grid_results_json, "w", encoding="utf-8") as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

    if summary_rows:
        assert grid_summary_csv is not None and grid_results_json is not None
        pd.DataFrame(summary_rows).to_csv(grid_summary_csv, index=False)
        print(f"\nWrote {grid_summary_csv} ({len(summary_rows)} rows)")
        with open(grid_results_json, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Wrote {grid_results_json} ({len(full_results)} evaluated cells)")
    elif stop_after == "oracle":
        print("\n[oracle-only] No grid_summary / grid_results CSV written (evaluation not run).")

    if stop_after == "oracle":
        print("\nOracle-only run finished — rerun without --stop-after oracle to continue from cached oracle CSVs.")

    print_summary_table(summary_rows)

    if llm_usage_log is not None:
        llm_usage_log.print_summary()

    if errors:
        print("\n" + "=" * 72)
        print(f"Errors in {len(errors)} cell(s):")
        for e in errors:
            print(f"  - {e['target']} x {e['country']}: {e['error']}")
    else:
        print("\nAll cells completed without errors.")

    if output_tag:
        man = {
            "exp_id": output_tag,
            "llm_model": ctx.model_name or None,
            "disambig_model": DISAMBIG_MODEL if stop_after != "oracle" else None,
            "embedding_model": embedding_model,
            "prompt_variant": args.prompt_variant,
            "survey": survey_id,
            "targets": targets,
            "countries": countries,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "n_cells_total": n_cells,
            "n_cells_completed": n_cells - len(errors),
            "n_cells_errored": len(errors),
        }
        mp = manifest_path(OUTPUTS_DIR, survey_id, output_tag)
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2, ensure_ascii=False)
        print(f"\nWrote run manifest: {mp}")


if __name__ == "__main__":
    main()
