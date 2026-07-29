"""
Cell-level free-text scoring helpers (shared by run_main / run_subitem_mapping).

Oracle top-k and random-k baselines depend only on (cell, k), so score_one_cell
caches them once per k while evaluating each arm/k_mode model feature set.

Parallelism: ProcessPoolExecutor over cells (not joblib over draws). Score-only
workers do not load torch/sentence-transformers, so XGB nthread > 1 is safe.
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from survey_features.config import OUTPUTS_DIR, ROOT
from survey_features.evaluation import evaluate_feature_set, single_random_draw
from survey_features.layout import genuine_cells, oracle_csv_path
from survey_features.metrics import captured_importance_df
from survey_features.surveys import SURVEY_COUNTRY_COL, build_country_code_map, load_survey

FIXED_KS_DEFAULT = (5, 10)

# Process-local caches (one survey load / oracle table per worker process).
_data_cache: dict[str, tuple] = {}
_oracle_cache: dict[str, tuple] = {}


def resolve_score_workers(cli_value: int | None = None) -> int:
    """Number of cell worker processes. CLI > SCORE_WORKERS env > default."""
    if cli_value is not None:
        return max(1, int(cli_value))
    raw = os.environ.get("SCORE_WORKERS")
    if raw is not None and raw.strip():
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    cpus = os.cpu_count() or 4
    return max(1, min(8, cpus - 2))


def resolve_score_xgb_nthread(workers: int, cli_value: int | None = None) -> int:
    """XGB nthread per fit. CLI > SCORE_XGB_NTHREAD env > cpus // workers."""
    if cli_value is not None:
        return max(1, int(cli_value))
    raw = os.environ.get("SCORE_XGB_NTHREAD")
    if raw is not None and raw.strip():
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    cpus = os.cpu_count() or 4
    return max(1, cpus // max(1, workers))


def resolve_score_n_draws() -> int:
    return int(os.environ.get("SCORE_N_DRAWS", "10"))


def _ensure_src_on_pythonpath() -> None:
    """Spawned Windows workers re-import survey_features; ensure src is visible."""
    src = str(ROOT / "src")
    prev = os.environ.get("PYTHONPATH", "")
    parts = [p for p in prev.split(os.pathsep) if p]
    if src not in parts:
        os.environ["PYTHONPATH"] = src + (os.pathsep + prev if prev else "")
    if src not in sys.path:
        sys.path.insert(0, src)


def _survey_data(survey: str):
    if survey not in _data_cache:
        cfg = os.environ["DATA_CONFIG_PATH"]
        _data_cache[survey] = load_survey(survey, cfg)
    return _data_cache[survey]


def _oracle_table(survey: str, outputs_dir: Path):
    """Oracle importances for one survey: (df, country_col, name->code map)."""
    if survey in _oracle_cache:
        return _oracle_cache[survey]
    _, meta = load_survey(survey, os.environ["DATA_CONFIG_PATH"])
    ccol = SURVEY_COUNTRY_COL.get(survey)
    data, _ = _survey_data(survey)
    cmap = build_country_code_map(meta, ccol, data) if ccol else {}
    rows = []
    for s, t, c in genuine_cells(outputs_dir):
        if s != survey:
            continue
        p = oracle_csv_path(t, c, outputs_dir)
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        code = cmap.get(c, c)
        for _, r in df.iterrows():
            rows.append({
                "target_variable": t,
                "country": code,
                "feature_variable": r["feature_variable"],
                "importance_mean": r["importance_mean"],
            })
    result = (pd.DataFrame(rows), ccol, cmap)
    _oracle_cache[survey] = result
    return result


def score_one_cell(spec: dict[str, Any]) -> list[dict]:
    """
    Score one (survey, target, country) cell.

    ``spec`` keys:
      survey, target, country
      evals: list of {condition, arm, disambiguator, embedding_model, codes, k_mode?}
      n_draws, nthread, fixed_ks (optional), outputs_dir (optional str)
    """
    survey = spec["survey"]
    target = spec["target"]
    country = spec["country"]
    evals: list[dict] = spec["evals"]
    n_draws = int(spec.get("n_draws", 10))
    nthread = int(spec.get("nthread", 1))
    fixed_ks = tuple(spec.get("fixed_ks") or FIXED_KS_DEFAULT)
    outputs_dir = Path(spec["outputs_dir"]) if spec.get("outputs_dir") else OUTPUTS_DIR

    if not evals:
        return []

    oracle_df, ccol, cmap = _oracle_table(survey, outputs_dir)
    data, _ = _survey_data(survey)
    code = cmap.get(country, country)
    country_data = data[data[ccol] == code].copy()
    pool = [c for c in country_data.columns if c not in {target, ccol}]

    def oracle_topk(t, country_code, k):
        sub = oracle_df[
            (oracle_df["target_variable"] == t) & (oracle_df["country"] == country_code)
        ]
        return (
            sub.sort_values("importance_mean", ascending=False)["feature_variable"]
            .head(k)
            .tolist()
        )

    oracle_acc_cache: dict[int, Any] = {}
    random_acc_cache: dict[int, Any] = {}
    majority: dict[int, Any] = {}
    rows: list[dict] = []

    for ev in evals:
        codes = ev.get("codes") or []
        cond = ev["condition"]
        arm = ev["arm"]
        dk = ev.get("disambiguator") or ""
        emb_label = ev.get("embedding_model") or ""
        k_mode = ev.get("k_mode")  # None for main experiment rows

        for kspec in ["model"] + [f"k{k}" for k in fixed_ks]:
            kk = None if kspec == "model" else int(kspec[1:])
            use_codes = [c for c in dict.fromkeys(codes) if c][: (kk or len(codes))]
            use_codes = [c for c in use_codes if c in country_data.columns]
            k = len(use_codes)
            if k == 0:
                continue
            ci = captured_importance_df(use_codes, target, code, oracle_df, k=None)
            base = {
                "survey": survey,
                "target": target,
                "country": country,
                "condition": cond,
                "arm": arm,
                "disambiguator": dk,
                "embedding_model": emb_label,
                "k_spec": kspec,
            }
            if k_mode is not None:
                base["k_mode"] = k_mode
            try:
                mres = evaluate_feature_set(
                    country_data, target, use_codes, nthread=nthread,
                )
                m = mres.get("accuracy_mean")
                majority[k] = mres.get("majority_baseline")
                if k not in oracle_acc_cache:
                    ores = evaluate_feature_set(
                        country_data, target, oracle_topk(target, code, k),
                        nthread=nthread,
                    )
                    oracle_acc_cache[k] = ores.get("accuracy_mean")
                    draws = [
                        single_random_draw(
                            country_data, target, pool, k, 42 + i, nthread=nthread,
                        )
                        for i in range(n_draws)
                    ]
                    draws = [d for d in draws if d is not None]
                    random_acc_cache[k] = (
                        round(float(np.mean(draws)), 4) if draws else None
                    )
                o = oracle_acc_cache[k]
                r = random_acc_cache[k]
            except Exception as e:
                err_row = {
                    **base,
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                }
                rows.append(err_row)
                continue
            rows.append({
                **base,
                "k": k,
                "captured_importance": round(ci, 4) if ci is not None else "",
                "oracle_acc": o,
                "model_acc": m,
                "random_acc": r,
                "majority": majority.get(k),
                "value_over_random": (
                    round(m - r, 4) if (m is not None and r is not None) else ""
                ),
                "cost_of_imperfect": (
                    round(o - m, 4) if (o is not None and m is not None) else ""
                ),
                "error": "",
            })
    return rows


def run_score_jobs(
    specs: list[dict[str, Any]],
    out_csv: Path,
    cols: list[str],
    workers: int,
    log_prefix: str = "[score]",
) -> int:
    """
    Score all cell specs and write CSV incrementally (main process owns the file).

    workers==1 runs inline (serial baseline / debug). workers>1 uses ProcessPool.
    Returns number of rows written.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=cols)
    writer.writeheader()
    out_f.flush()
    n_written = 0

    def _flush_rows(cell_rows: list[dict], survey: str, target: str, country: str) -> None:
        nonlocal n_written
        for r in cell_rows:
            writer.writerow({c: r.get(c, "") for c in cols})
        out_f.flush()
        n_written += len(cell_rows)
        print(
            f"  scored {survey} {target} {country} "
            f"(+{len(cell_rows)} rows, {n_written} total)",
            flush=True,
        )

    try:
        if workers <= 1 or len(specs) <= 1:
            for spec in specs:
                rows = score_one_cell(spec)
                _flush_rows(rows, spec["survey"], spec["target"], spec["country"])
        else:
            _ensure_src_on_pythonpath()
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(score_one_cell, spec): spec for spec in specs}
                for fut in as_completed(futures):
                    spec = futures[fut]
                    try:
                        rows = fut.result()
                    except Exception as e:
                        print(
                            f"  ! worker failed {spec['survey']} {spec['target']} "
                            f"{spec['country']}: {type(e).__name__}: {e}",
                            flush=True,
                        )
                        rows = [{
                            "survey": spec["survey"],
                            "target": spec["target"],
                            "country": spec["country"],
                            "error": f"{type(e).__name__}: {str(e)[:80]}",
                        }]
                    _flush_rows(rows, spec["survey"], spec["target"], spec["country"])
    finally:
        out_f.close()

    print(f"{log_prefix} wrote {n_written} rows -> {out_csv}")
    return n_written
