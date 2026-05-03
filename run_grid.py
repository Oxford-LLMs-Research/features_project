"""
Grid runner: run any subset of targets × countries end-to-end.

Loads WVS data and survey-variable embeddings ONCE, then iterates through every
(target, country) cell running the full pipeline:
  1. Oracle permutation importances (XGBoost)
  2. LLM feature selection (unprompted + country_provided)
  3. Embedding-retrieval candidate mapping
  4. LLM disambiguation
  5. Downstream XGBoost prediction comparison

Cached intermediates under outputs/<target>_<country>/ are reused across runs,
so a re-run resumes from the last completed cell.

Usage:
    python run_grid.py                                              # full WVS 5×5 grid
    python run_grid.py --targets Q164 --countries Germany           # single WVS cell
    python run_grid.py --survey afrobarometer --targets Q1 --countries Nigeria Ghana
    python run_grid.py --targets Q47 Q164 --countries Germany Nigeria Japan
    python run_grid.py --survey afrobarometer --list-countries      # show available countries

Country names and admin columns are derived automatically from the survey
metadata — no hardcoding needed. Use --list-countries to see valid names for
any survey.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from phase0b_oracle import clean_question_columns, compute_oracle

load_dotenv()

FEATURES_DIR = Path(__file__).parent
OUTPUTS_DIR = FEATURES_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Survey registry ───────────────────────────────────────────────────────────
# Column in each survey's DataFrame that holds country codes.
SURVEY_COUNTRY_COL: dict[str, str] = {
    "wvs":            "B_COUNTRY",
    "afrobarometer":  "COUNTRY",
    "arabbarometer":  "COUNTRY",
    "asianbarometer": "country",
    "latinobarometer": "IDENPA",
    "ess_wave_10":    "cntry",
    "ess_wave_11":    "cntry",
}

# Default grid (WVS only; always specify --targets and --countries for other surveys).
DEFAULT_SURVEY = "wvs"
DEFAULT_TARGETS = ["Q47", "Q57", "Q199", "Q235", "Q164"]
DEFAULT_COUNTRIES = ["Germany", "Nigeria", "Japan", "Brazil", "Egypt"]

SURVEY_EMB_CACHE = OUTPUTS_DIR / "survey_embeddings.npz"
GRID_SUMMARY_CSV = OUTPUTS_DIR / "grid_summary.csv"
GRID_RESULTS_JSON = OUTPUTS_DIR / "grid_results.json"

N_CELL_WORKERS = 5


# ── Data loading ──────────────────────────────────────────────────────────────

def load_survey(survey_id: str, config_path: str) -> tuple[pd.DataFrame, dict]:
    try:
        from synthetic_sampling.config.base import DataPaths
        from synthetic_sampling.loaders.survey_loader import SurveyLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: synthetic_sampling. Install project dependencies "
            "with `pip install -r requirements.txt` to continue."
        ) from exc

    paths = DataPaths.from_yaml(config_path)
    loader = SurveyLoader(paths=paths, verbose=False)
    data, metadata = loader.load_survey(survey_id)
    return data, metadata


def build_country_code_map(
    metadata: dict,
    country_col: str,
    data: pd.DataFrame | None = None,
) -> dict[str, int | str]:
    """
    Derive {country_name: code} from the country column's 'values' dict in metadata.
    Codes that parse as integers are returned as int; alpha codes (ESS) stay as str.

    If data is provided, the metadata-derived codes are cross-checked against the
    actual values in the country column. When the data stores country names directly
    (instead of numeric codes), the map is built from the actual data values instead.
    """
    meta_map: dict[str, int | str] = {}
    for section in metadata.values():
        if not isinstance(section, dict):
            continue
        if country_col in section:
            values = section[country_col].get("values", {})
            for code_str, name in values.items():
                try:
                    meta_map[name] = int(code_str)
                except ValueError:
                    meta_map[name] = code_str
            break

    if data is None or not meta_map:
        return meta_map

    actual_values = set(data[country_col].dropna().unique())

    # Normal path: metadata codes exist in data.
    if any(code in actual_values for code in meta_map.values()):
        return {name: code for name, code in meta_map.items() if code in actual_values}

    # Data stores names/strings directly — build map from actual data values.
    # Try to match metadata names to data values case-insensitively, then add
    # any remaining data values as identity entries so --list-countries is complete.
    actual_lower = {str(v).lower(): v for v in actual_values}
    result: dict[str, int | str] = {}
    for meta_name in meta_map:
        if meta_name in actual_values:
            result[meta_name] = meta_name
        elif meta_name.lower() in actual_lower:
            result[meta_name] = actual_lower[meta_name.lower()]
    # Add any data values that didn't match a metadata name.
    matched_data_vals = set(result.values())
    for val in actual_values:
        val_str = str(val)
        if val not in matched_data_vals and val_str not in result:
            result[val_str] = val
    return result


def build_admin_cols(metadata: dict, country_col: str) -> frozenset[str]:
    """
    Derive admin columns from the 'EXCLUDED' section in metadata plus the country column.
    The EXCLUDED section contains all non-substantive variables (IDs, weights, admin codes).
    """
    excluded = metadata.get("EXCLUDED", {})
    return frozenset(excluded.keys()) | {country_col}


def get_question_text(var_code: str, metadata: dict) -> str:
    for section in metadata.values():
        if var_code in section:
            info = section[var_code]
            return (info.get("question") or info.get("description") or var_code).strip()
    raise KeyError(f"{var_code} not found in metadata")


# ── LLM selection + mapping ───────────────────────────────────────────────────

def run_llm_and_map(
    target_var: str,
    question_text: str,
    country_name: str | None,
    condition: str,
    generate_fn,
    model_name: str,
    survey_variables: dict[str, str],
    survey_embeddings: np.ndarray,
    var_codes: list[str],
) -> list[dict]:
    from phase0b_pipeline import run_single
    from phase0b_mapping import map_features_to_variables
    from phase0b_disambig import disambiguate_mappings

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
        exclude_targets=True,
    )
    print(f"  {len(mappings)} feature->candidate pairs")

    print("\n  Disambiguating ...")
    mappings = disambiguate_mappings(mappings, generate_fn, model=model_name)

    mapped = sum(1 for m in mappings if m["disambig"]["selected_code"])
    print(f"\n  Mapped: {mapped}/{len(mappings)} features -> survey variables")

    return mappings


# ── Embedding cache ───────────────────────────────────────────────────────────

def load_or_build_survey_embeddings(
    survey_variables: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    from phase0b_mapping import build_embeddings

    var_codes = list(survey_variables.keys())
    var_texts = list(survey_variables.values())

    if SURVEY_EMB_CACHE.exists():
        cached = np.load(SURVEY_EMB_CACHE, allow_pickle=True)
        cached_codes = list(cached["var_codes"])
        if cached_codes == var_codes:
            print(f"  Loaded cached embeddings ({len(var_codes)} vars) from {SURVEY_EMB_CACHE}")
            return cached["embeddings"], var_codes
        print("  Cached embeddings are stale (var_codes mismatch); recomputing.")

    embeddings = build_embeddings(var_texts)
    np.savez(SURVEY_EMB_CACHE, embeddings=embeddings, var_codes=np.array(var_codes, dtype=object))
    print(f"  Saved embeddings to {SURVEY_EMB_CACHE}")
    return embeddings, var_codes


# ── Per-cell cache helpers ────────────────────────────────────────────────────

def cell_dir(prefix: str) -> Path:
    d = OUTPUTS_DIR / prefix
    d.mkdir(exist_ok=True)
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
) -> tuple[pd.DataFrame, list[str]]:
    oracle_path = cell_dir(prefix) / "oracle.csv"
    if oracle_path.exists():
        print(f"  [oracle] Loading cached {prefix}/oracle.csv")
        oracle_df = pd.read_csv(oracle_path)
        feature_pool = oracle_df[oracle_df["target_variable"] == target]["feature_variable"].tolist()
        return oracle_df, feature_pool

    print(f"  [oracle] Computing ({target} x {country_name}) ...")
    oracle_df, feature_pool = compute_oracle(
        data, metadata, target, country_code, country_col, admin_cols,
    )
    oracle_df.to_csv(oracle_path, index=False)
    print(f"  [oracle] Saved {prefix}/oracle.csv")
    return oracle_df, feature_pool


def get_or_run_llm_mapping(
    target: str,
    question_text: str,
    country_name: str,
    generate_fn,
    model_name: str,
    survey_variables: dict[str, str],
    survey_embeddings: np.ndarray,
    var_codes: list[str],
    prefix: str,
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
        model_name=model_name,
        survey_variables=survey_variables,
        survey_embeddings=survey_embeddings,
        var_codes=var_codes,
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
) -> tuple[dict, list[dict]]:
    from phase0b_evaluation import run_comparison, print_comparison

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
) -> dict:
    row = {
        "target": target,
        "country": country_name,
        "country_code": country_code,
        "condition": condition,
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
    survey_variables: dict[str, str]
    survey_embeddings: np.ndarray
    var_codes: list[str]
    generate_fn: Any
    model_name: str
    n_jobs_random: int
    country_col: str
    admin_cols: frozenset[str]
    country_codes: dict[str, int | str]


def run_cell(ctx: PipelineContext, target: str, country_name: str) -> dict:
    country_code = ctx.country_codes[country_name]
    prefix = f"{target}_{country_name}"

    try:
        question_text = get_question_text(target, ctx.metadata)

        oracle_df, feature_pool = get_or_compute_oracle(
            target=target,
            country_name=country_name,
            country_code=country_code,
            country_col=ctx.country_col,
            admin_cols=ctx.admin_cols,
            data=ctx.data,
            metadata=ctx.metadata,
            prefix=prefix,
        )

        mappings = get_or_run_llm_mapping(
            target=target,
            question_text=question_text,
            country_name=country_name,
            generate_fn=ctx.generate_fn,
            model_name=ctx.model_name,
            survey_variables=ctx.survey_variables,
            survey_embeddings=ctx.survey_embeddings,
            var_codes=ctx.var_codes,
            prefix=prefix,
        )

        if not mappings:
            return {"target": target, "country_name": country_name,
                    "country_code": country_code, "prefix": prefix,
                    "eval_results": {}, "cond_errors": [], "error": "no mappings"}

        eval_results, cond_errors = get_or_run_eval(
            target=target,
            country_name=country_name,
            country_code=country_code,
            country_col=ctx.country_col,
            mappings=mappings,
            oracle_df=oracle_df,
            feature_pool=feature_pool,
            data=ctx.eval_data,
            prefix=prefix,
            n_jobs_random=ctx.n_jobs_random,
        )

        return {"target": target, "country_name": country_name,
                "country_code": country_code, "prefix": prefix,
                "eval_results": eval_results, "cond_errors": cond_errors, "error": None}

    except Exception as e:
        traceback.print_exc()
        return {"target": target, "country_name": country_name,
                "country_code": country_code, "prefix": prefix,
                "eval_results": {}, "cond_errors": [],
                "error": f"{type(e).__name__}: {e}"}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature selection grid runner.")
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
    args = parser.parse_args()

    survey_id = args.survey
    country_col = SURVEY_COUNTRY_COL[survey_id]

    # Apply WVS defaults only when the default survey is selected and nothing is overridden.
    targets = args.targets or (DEFAULT_TARGETS if survey_id == DEFAULT_SURVEY else None)
    countries = args.countries or (DEFAULT_COUNTRIES if survey_id == DEFAULT_SURVEY else None)

    config_path = os.environ.get("DATA_CONFIG_PATH")
    if not config_path:
        raise ValueError("DATA_CONFIG_PATH is not set in .env")

    print(f"\n[setup] Loading {survey_id} data ...")
    data, metadata = load_survey(survey_id, config_path)
    print(f"  Total rows: {len(data)}")

    country_codes = build_country_code_map(metadata, country_col, data)
    admin_cols = build_admin_cols(metadata, country_col)
    print(f"  {len(country_codes)} countries, {len(admin_cols)} admin columns derived from metadata")

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
    n_workers = min(N_CELL_WORKERS, n_cells)
    n_jobs_random = max(1, (os.cpu_count() or 10) // n_workers)

    print("\n" + "=" * 72)
    print(f"Survey: {survey_id}  |  country column: {country_col}")
    print(f"Grid runner: {len(targets)} target(s) x {len(countries)} country/ies = {n_cells} cell(s)")
    print(f"Targets:   {targets}")
    print(f"Countries: {countries}")
    print("=" * 72)

    print("\n[setup] Cleaning question columns for evaluation ...")
    eval_data = clean_question_columns(data, country_col, admin_cols)

    print("\n[setup] Building survey variable index ...")
    from phase0b_mapping import extract_survey_variables
    survey_variables = extract_survey_variables(metadata)
    print(f"  {len(survey_variables)} survey variables")

    print("\n[setup] Loading / building survey embeddings ...")
    survey_embeddings, var_codes = load_or_build_survey_embeddings(survey_variables)

    print("\n[setup] Initialising LLM client ...")
    from generate import make_generate_fn
    generate_fn, model_name = make_generate_fn()
    print(f"  Model: {model_name}")

    summary_rows: list[dict] = []
    full_results: dict = {}
    errors: list[dict] = []

    ctx = PipelineContext(
        data=data,
        eval_data=eval_data,
        metadata=metadata,
        survey_variables=survey_variables,
        survey_embeddings=survey_embeddings,
        var_codes=var_codes,
        generate_fn=generate_fn,
        model_name=model_name,
        n_jobs_random=n_jobs_random,
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
                    flatten_eval_result(target, country_name, country_code,
                                        condition, cell_result)
                )
            for ce in cond_errors:
                errors.append({"target": target, "country": country_name,
                               "condition": ce["condition"], "error": ce["error"]})

            pd.DataFrame(summary_rows).to_csv(GRID_SUMMARY_CSV, index=False)
            with open(GRID_RESULTS_JSON, "w", encoding="utf-8") as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(GRID_SUMMARY_CSV, index=False)
        print(f"\nWrote {GRID_SUMMARY_CSV} ({len(summary_rows)} rows)")

    with open(GRID_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Wrote {GRID_RESULTS_JSON} ({len(full_results)} cells)")

    print_summary_table(summary_rows)

    if errors:
        print("\n" + "=" * 72)
        print(f"Errors in {len(errors)} cell(s):")
        for e in errors:
            print(f"  - {e['target']} x {e['country']}: {e['error']}")
    else:
        print("\nAll cells completed without errors.")


if __name__ == "__main__":
    main()
