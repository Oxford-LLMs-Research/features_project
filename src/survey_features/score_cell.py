"""
Cell-level scoring helpers for the free-text pipeline (scripts/run_main.py).

Oracle top-k and random-k baselines depend only on (cell, k), so score_one_cell
caches them once per k while evaluating each condition/disambiguator feature set.
Model (and oracle) XGB CV results are also cached by ordered code tuple so
identical feature lists across k_spec share one fit.

Parallelism: ProcessPoolExecutor over cells (not joblib over draws). Score-only
workers do not load torch/sentence-transformers, so XGB nthread > 1 is safe.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from survey_features.config import OUTPUTS_DIR, ROOT
from survey_features.evaluation import (
    REGRESSION_TYPES,
    evaluate_feature_set,
    single_random_draw_result,
)
from survey_features.layout import cell_dir, genuine_cells, oracle_csv_path
from survey_features.metrics import (
    captured_importance_df,
    load_oracle_train_index,
    oracle_topk_codes,
    rank_score_from_frame,
    stable_seed,
)
from survey_features.surveys import (
    SURVEY_COUNTRY_COL,
    build_country_code_map,
    load_survey_clean,
)

FIXED_KS_DEFAULT = (5, 10)

# Canonical scores.csv schema. Runners extend it with their own key (e.g. "k_mode")
# via score_cols(); keeping one definition stops the three runners drifting apart.
SCORE_COLS: list[str] = [
    "survey", "target", "country", "condition", "arm", "disambiguator",
    "embedding_model", "k_spec", "k", "target_type",
    # selection-alignment metrics (pure arithmetic on the cached oracle)
    "captured_importance", "textbook_captured", "textbook_k",
    # accuracy layer
    "oracle_acc", "model_acc", "random_acc", "textbook_acc", "majority",
    "value_over_random", "value_over_textbook", "cost_of_imperfect",
    # log-loss layer (lower is better; the *_ll deltas are signed so + = model better)
    "oracle_ll", "model_ll", "random_ll", "textbook_ll",
    "value_over_random_ll", "value_over_textbook_ll", "cost_of_imperfect_ll",
    # rank layer (Spearman rho of an XGB regressor; higher is better).
    # VALUATION IS TYPE-MATCHED (user decision 2026-08-18): binary/nominal cells
    # fill the accuracy + log-loss columns, ordinal/continuous cells fill ONLY the
    # rho columns — one fit family per cell, two reporting families across a grid,
    # never pooled into a single mean. (Before this, ordinal cells were silently
    # valued as unordered classification — pilot report 2026-08.)
    "oracle_rho", "model_rho", "random_rho", "textbook_rho",
    "value_over_random_rho", "value_over_textbook_rho", "cost_of_imperfect_rho",
    "error",
]


def score_cols(*extra: str) -> list[str]:
    """SCORE_COLS with runner-specific columns inserted before `error`."""
    cols = [c for c in SCORE_COLS if c != "error"]
    cols += [c for c in extra if c not in cols]
    return cols + ["error"]

# Process-local caches (one survey load / oracle table per worker process).
_data_cache: dict[str, tuple] = {}
_oracle_cache: dict[str, tuple] = {}
_textbook_cache: dict[str, list[str]] = {}


# ── Shared feature pool, baselines and textbook set ───────────────────────────

def cell_feature_pool(
    target: str,
    country: str,
    country_data: pd.DataFrame,
    country_col: str,
    outputs_dir: Path = OUTPUTS_DIR,
) -> list[str]:
    """The cell's oracle feature pool — the universe random draws sample from.

    feature_pool.csv, else the oracle table, else a LOUD error. One definition for
    null and ranking (pipeline_audit_2026-08.md #A2).
    """
    pool_csv = cell_dir(target, country, outputs_dir) / "feature_pool.csv"
    if pool_csv.is_file():
        codes = pd.read_csv(pool_csv)["feature_variable"].astype(str).tolist()
        keep = [c for c in codes if c in country_data.columns and c != target]
        if keep:
            return keep
    oracle_csv = oracle_csv_path(target, country, outputs_dir)
    if oracle_csv.is_file():
        codes = pd.read_csv(oracle_csv)["feature_variable"].astype(str).tolist()
        keep = [c for c in codes if c in country_data.columns and c != target]
        if keep:
            return keep
    raise FileNotFoundError(
        f"No oracle feature pool for {target} x {country} ({country_col}) — run the "
        "oracle first. The silent fallback to 'every column' was removed on purpose: "
        "it weakens the random null (pipeline_audit #A2)."
    )


def cell_target_type(target: str, country: str,
                     outputs_dir: Path = OUTPUTS_DIR) -> str | None:
    """The cell's measurement level from oracle_meta.json (None if absent).

    This is what routes ordinal/continuous cells into the type-matched (Spearman)
    valuation alongside the common-currency classification metrics. It comes from
    the oracle meta rather than re-detection so scoring and ranking always agree
    on what the target IS.
    """
    p = cell_dir(target, country, outputs_dir) / "oracle_meta.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("target_type")


def _baselines_path(target: str, country: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return cell_dir(target, country, outputs_dir) / "baselines.json"


def baseline_fingerprint(
    pool: list[str],
    textbook: list[str],
    n_draws: int,
    train_index: list | None = None,
    target_type: str | None = None,
) -> str:
    """Identity of the inputs a cached baseline depends on.

    Anything that changes what a baseline MEANS belongs in here (onboarding.md #4).
    target_type enters the key only for ordinal/continuous cells: their baselines
    gained the rho fields, so pre-existing caches must invalidate; classification
    cells' cached baselines are unchanged and stay valid.
    """
    if train_index:
        train_fp = hashlib.blake2b(
            ",".join(map(str, train_index)).encode("utf-8"), digest_size=4
        ).hexdigest()
        train_key = f"train={len(train_index)}:{train_fp}"
    else:
        train_key = "train=none"
    parts = [
        str(len(pool)),
        ",".join(sorted(pool)[:64]),   # pool identity; prefix is enough with the length
        ",".join(textbook),            # order matters: fixed-k takes a prefix
        f"draws={n_draws}",
        train_key,
    ]
    if target_type in REGRESSION_TYPES:
        parts.append(f"ttype={target_type}")
    key = "|".join(parts)
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


def load_cell_baselines(target: str, country: str,
                        outputs_dir: Path = OUTPUTS_DIR) -> dict[str, dict]:
    """Cached per-(cell, k) oracle/random/textbook results, keyed by str(k).

    These are model-independent, so persisting them means a multi-model zoo pays for
    them once rather than once per model.
    """
    p = _baselines_path(target, country, outputs_dir)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))  # corrupt cache = loud failure


def save_cell_baselines(target: str, country: str, baselines: dict[str, dict],
                        outputs_dir: Path = OUTPUTS_DIR) -> None:
    p = _baselines_path(target, country, outputs_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    tmp.replace(p)  # atomic: concurrent cell workers may share a cell dir


def textbook_codes(survey: str, outputs_dir: Path = OUTPUTS_DIR) -> list[str]:
    """The frozen demographic baseline for a survey, in pre-registered order.

    The "knows something question-specific" contrast (pipeline_audit #A2). Built by
    scripts/build_textbook_baseline.py; absent file = no textbook columns.
    """
    if survey in _textbook_cache:
        return _textbook_cache[survey]
    p = outputs_dir / "cache" / "baselines" / f"textbook__{survey}.json"
    codes: list[str] = []
    if p.is_file():  # absent = baseline not built (designed state); corrupt = loud
        rec = json.loads(p.read_text(encoding="utf-8"))
        codes = [e["var_code"] for e in rec.get("constructs", []) if e.get("var_code")]
        codes = list(dict.fromkeys(codes))
    _textbook_cache[survey] = codes
    return codes


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
    """Cleaned survey frame — the SAME preprocessing the oracle ranked features on.

    Previously this used raw `load_survey`, so negative sentinel codes reached XGBoost
    as ordinary values on the scale while the oracle had already cleaned them: the two
    stages scored features on different data.
    """
    if survey not in _data_cache:
        cfg = os.environ["DATA_CONFIG_PATH"]
        _data_cache[survey] = load_survey_clean(survey, cfg)
    return _data_cache[survey]


def _oracle_table(survey: str, outputs_dir: Path, cells=None):
    """Oracle importances for one survey: (df, country_col, name->code map).

    ``cells`` (list of (survey, target, country)) overrides the leakage-audit
    grid — required for pilot / frame-sampled runs whose cells the audit never saw.
    """
    cache_key = (survey, tuple(map(tuple, cells)) if cells else "audit")
    if cache_key in _oracle_cache:
        return _oracle_cache[cache_key]
    _, meta = _survey_data(survey)
    ccol = SURVEY_COUNTRY_COL.get(survey)
    data, _ = _survey_data(survey)
    cmap = build_country_code_map(meta, ccol, data) if ccol else {}
    rows = []
    for s, t, c in (cells if cells is not None else genuine_cells(outputs_dir)):
        if s != survey:
            continue
        p = oracle_csv_path(t, c, outputs_dir)
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        code = cmap.get(c, c)
        has_select = "importance_select" in df.columns
        has_score = "importance_score" in df.columns
        for _, r in df.iterrows():
            mean_v = r["importance_mean"]
            rows.append({
                "target_variable": t,
                "country": code,
                "feature_variable": r["feature_variable"],
                "importance_mean": mean_v,
                "importance_select": r["importance_select"] if has_select else mean_v,
                "importance_score": r["importance_score"] if has_score else mean_v,
            })
    result = (pd.DataFrame(rows), ccol, cmap)
    _oracle_cache[cache_key] = result
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

    oracle_df, ccol, cmap = _oracle_table(
        survey, outputs_dir, cells=spec.get("grid_cells")
    )
    data, _ = _survey_data(survey)
    code = cmap.get(country, country)
    country_data = data[data[ccol] == code].copy()

    # Random-k null draws from the SAME universe the oracle ranked over (else the null
    # is weaker than matched and inflates value_over_random — pipeline_audit #A2).
    pool = cell_feature_pool(target, country, country_data, ccol, outputs_dir)
    # Drop the target from the textbook set — some grid targets ARE textbook
    # demographics, and the baseline must not predict the target from itself.
    textbook = [c for c in textbook_codes(survey, outputs_dir) if c != target]

    # Fit-split labels: confine every arm's CV to rows the oracle ranking never saw.
    train_index = load_oracle_train_index(target, country, outputs_dir)

    # Measurement level, from the oracle meta. Ordinal/continuous cells get the
    # type-matched (Spearman) valuation IN ADDITION to the common-currency
    # classification metrics — both always computed, never a silent default.
    ttype = cell_target_type(target, country, outputs_dir)
    typed = ttype in REGRESSION_TYPES

    def oracle_topk(t, country_code, k):
        sub = oracle_df[
            (oracle_df["target_variable"] == t) & (oracle_df["country"] == country_code)
        ]
        rank, _score = rank_score_from_frame(sub)
        if rank:
            return oracle_topk_codes(rank, k)
        return (
            sub.sort_values("importance_mean", ascending=False)["feature_variable"]
            .head(k)
            .tolist()
        )

    # Baselines depend only on (cell, k) — never on which model is being scored — so
    # they are cached on disk and shared across every selector in the zoo. They DO
    # depend on the oracle ranking, the random pool, the textbook set, and the eval
    # row regime (train_index), so the cache carries a fingerprint of those.
    fingerprint = baseline_fingerprint(
        pool, textbook, n_draws, train_index=train_index, target_type=ttype
    )
    baselines = load_cell_baselines(target, country, outputs_dir)
    if baselines.get("_fingerprint") != fingerprint:
        baselines = {"_fingerprint": fingerprint}
    baselines_dirty = False
    # Cache XGB CV by ordered code tuple so model-k / k5 / k10 (and duplicate
    # arm code-sets) share one fit when the feature list is identical.
    eval_cache: dict[tuple[str, ...], dict] = {}
    rows: list[dict] = []

    def cached_eval(feature_codes: list[str]) -> dict:
        """Type-matched valuation of one feature set — ONE fit family per cell.

        binary/nominal -> classifier (accuracy + log loss); ordinal/continuous ->
        Spearman regressor only. The two families are reported side by side across
        a grid and never pooled into one mean.
        """
        key = tuple(feature_codes)
        hit = eval_cache.get(key)
        if hit is None:
            hit = evaluate_feature_set(
                country_data, target, feature_codes, nthread=nthread,
                row_index=train_index, target_type=ttype if typed else None,
            )
            eval_cache[key] = hit
        return hit

    def baseline_for_k(k: int) -> dict:
        """Oracle top-k, random-k and textbook-k results for this cell at budget k."""
        nonlocal baselines_dirty
        hit = baselines.get(str(k))
        if hit is not None:
            return hit

        ores = cached_eval(oracle_topk(target, code, k))

        # One fixed draw set per (cell, k), shared across models: cheaper, and it takes
        # the baseline's own Monte-Carlo noise out of every model-vs-model comparison.
        # Draws use the cell's own fit family (same seeds either way).
        draws = [
            single_random_draw_result(
                country_data, target, pool, k, stable_seed(target, country, k, i),
                nthread=nthread, row_index=train_index,
                target_type=ttype if typed else None,
            )
            for i in range(n_draws)
        ]
        r_acc = [d["accuracy_mean"] for d in draws if d.get("accuracy_mean") is not None]
        r_ll = [d["logloss_mean"] for d in draws if d.get("logloss_mean") is not None]
        r_rho = [d["spearman_mean"] for d in draws if d.get("spearman_mean") is not None]

        tb_codes = [c for c in textbook if c in country_data.columns][:k]
        tres = cached_eval(tb_codes) if tb_codes else {}

        hit = {
            "oracle_acc": ores.get("accuracy_mean"),
            "oracle_ll": ores.get("logloss_mean"),
            "random_acc": round(float(np.mean(r_acc)), 4) if r_acc else None,
            "random_ll": round(float(np.mean(r_ll)), 4) if r_ll else None,
            "textbook_acc": tres.get("accuracy_mean"),
            "textbook_ll": tres.get("logloss_mean"),
            "oracle_rho": ores.get("spearman_mean"),
            "random_rho": round(float(np.mean(r_rho)), 4) if r_rho else None,
            "textbook_rho": tres.get("spearman_mean"),
            "textbook_k": len(tb_codes),
            "textbook_captured": captured_importance_df(
                tb_codes, target, code, oracle_df, k=None
            ) if tb_codes else None,
            "majority": ores.get("majority_baseline"),
            "n_draws": len(r_acc),
        }
        baselines[str(k)] = hit
        baselines_dirty = True
        return hit

    def diff(a, b, ndigits=4):
        return round(a - b, ndigits) if (a is not None and b is not None) else ""

    for ev in evals:
        codes = ev.get("codes") or []
        cond = ev["condition"]
        arm = ev["arm"]
        dk = ev.get("disambiguator") or ""
        emb_label = ev.get("embedding_model") or ""
        k_mode = ev.get("k_mode")  # None for main experiment rows

        for kspec in ["model"] + [f"k{k}" for k in fixed_ks]:
            kk = None if kspec == "model" else int(kspec[1:])
            # NOTE: for fixed k this truncates in ARRIVAL order, which is essay order
            # after extraction — the model never ranks its requests, so "k5" is an
            # arbitrary 5 of its picks, not its best 5. See docs/pipeline_audit_2026-08.md
            # A5; model-k is the primary endpoint and fixed k is a diagnostic.
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
                mres = cached_eval(use_codes)
                m = mres.get("accuracy_mean")
                m_ll = mres.get("logloss_mean")
                bl = baseline_for_k(k)
            except Exception as e:
                rows.append({**base, "error": f"{type(e).__name__}: {str(e)[:80]}"})
                continue
            o, r, tb = bl["oracle_acc"], bl["random_acc"], bl["textbook_acc"]
            o_ll, r_ll, tb_ll = bl["oracle_ll"], bl["random_ll"], bl["textbook_ll"]
            m_rho = mres.get("spearman_mean")
            o_rho, r_rho = bl.get("oracle_rho"), bl.get("random_rho")
            tb_rho = bl.get("textbook_rho")
            rows.append({
                **base,
                "k": k,
                "target_type": ttype or "",
                "captured_importance": round(ci, 4) if ci is not None else "",
                "oracle_acc": o,
                "model_acc": m,
                "random_acc": r,
                "textbook_acc": tb,
                "majority": bl["majority"],
                "value_over_random": diff(m, r),
                "value_over_textbook": diff(m, tb),
                "cost_of_imperfect": diff(o, m),
                "model_ll": m_ll,
                "oracle_ll": o_ll,
                "random_ll": r_ll,
                "textbook_ll": tb_ll,
                # Log loss is lower-is-better, so these are signed so that
                # positive = the model is better, matching the accuracy columns.
                "value_over_random_ll": diff(r_ll, m_ll),
                "value_over_textbook_ll": diff(tb_ll, m_ll),
                "cost_of_imperfect_ll": diff(m_ll, o_ll),
                # rho is higher-is-better, so these are model-minus-baseline
                # (positive = model better), matching the sign convention above.
                "model_rho": m_rho if m_rho is not None else "",
                "oracle_rho": o_rho if o_rho is not None else "",
                "random_rho": r_rho if r_rho is not None else "",
                "textbook_rho": tb_rho if tb_rho is not None else "",
                "value_over_random_rho": diff(m_rho, r_rho),
                "value_over_textbook_rho": diff(m_rho, tb_rho),
                "cost_of_imperfect_rho": diff(o_rho, m_rho),
                "textbook_captured": (
                    round(bl["textbook_captured"], 4)
                    if bl["textbook_captured"] is not None else ""
                ),
                "textbook_k": bl["textbook_k"],
                "error": "",
            })

    if baselines_dirty:
        save_cell_baselines(target, country, baselines, outputs_dir)
    return rows


def _existing_header(out_csv: Path) -> list[str] | None:
    if not out_csv.is_file():
        return None
    with open(out_csv, newline="", encoding="utf-8") as f:
        return next(csv.reader(f), None)


def _completed_cells(out_csv: Path) -> set[tuple[str, str, str]]:
    """(survey, target, country) already present in a partial scores CSV."""
    if not out_csv.is_file():
        return set()
    with open(out_csv, newline="", encoding="utf-8") as f:
        return {
            (r.get("survey", ""), r.get("target", ""), r.get("country", ""))
            for r in csv.DictReader(f)
        }


def _archive_stale_csv(out_csv: Path, log_prefix: str) -> None:
    """Move a CSV written under a different schema aside instead of appending to it."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = out_csv.with_name(f"{out_csv.stem}.pre_{stamp}{out_csv.suffix}")
    out_csv.replace(backup)
    print(
        f"{log_prefix} existing CSV has a different column set (schema changed); "
        f"moved it to {backup.name} and starting fresh",
        flush=True,
    )


def run_score_jobs(
    specs: list[dict[str, Any]],
    out_csv: Path,
    cols: list[str],
    workers: int,
    log_prefix: str = "[score]",
    resume: bool = True,
) -> int:
    """
    Score all cell specs and write CSV incrementally (main process owns the file).

    workers==1 runs inline (serial baseline / debug). workers>1 uses ProcessPool.
    Returns number of rows written *this run*.

    ``resume`` appends to an existing CSV and skips cells already in it. Without it a
    crash late in a multi-hour sweep discarded everything, which matters here: the run
    logs already record transient provider errors and a worker collapse mid-sweep.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Resume only makes sense against a CSV written with the same columns. Appending
    # across a schema change would silently misalign every subsequent row.
    if resume and out_csv.is_file() and _existing_header(out_csv) != list(cols):
        _archive_stale_csv(out_csv, log_prefix)

    done = _completed_cells(out_csv) if resume else set()
    if done:
        before = len(specs)
        specs = [s for s in specs
                 if (s["survey"], s["target"], s["country"]) not in done]
        print(f"{log_prefix} resuming: {len(done)} cells already scored, "
              f"{len(specs)}/{before} remain", flush=True)

    # Group by survey so a worker loading a ~75s survey frame reuses it across that
    # survey's cells instead of thrashing between all six.
    specs = sorted(specs, key=lambda s: (s["survey"], s["target"], s["country"]))

    append = resume and out_csv.is_file() and bool(done)
    out_f = open(out_csv, "a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=cols)
    if not append:
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
