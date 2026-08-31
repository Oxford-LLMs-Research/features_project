"""
Process-isolated oracle execution (the safe concurrency for AutoGluon fits).

Processes, not threads: concurrent fits in one process share a native OpenMP runtime
and can wedge the interpreter; a worker process that dies takes one cell only. Mirrors
score_cell.run_score_jobs. Costs one survey load + frame in memory per worker; cells
are dispatched grouped by survey so each worker reuses its loaded frame.
Incident history: docs/onboarding.md §5.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import OUTPUTS_DIR, ROOT

# Process-local, one per worker: the survey frame and its derived maps.
_survey_cache: dict[str, tuple] = {}
# MiniLM for near-duplicate exclusion — one load per worker, not per cell.
_similarity_models: dict[float, object | None] = {}


def _ensure_src_on_pythonpath() -> None:
    """Spawned Windows workers re-import survey_features; ensure src is visible."""
    src = str(ROOT / "src")
    prev = os.environ.get("PYTHONPATH", "")
    parts = [p for p in prev.split(os.pathsep) if p]
    if src not in parts:
        os.environ["PYTHONPATH"] = src + (os.pathsep + prev if prev else "")
    if src not in sys.path:
        sys.path.insert(0, src)


def _survey_assets(survey: str):
    """(data, metadata, country_col, admin_cols, metadata_flat, country_codes), cached."""
    if survey in _survey_cache:
        return _survey_cache[survey]
    from .surveys import (
        SURVEY_COUNTRY_COL,
        build_admin_cols,
        build_country_code_map,
        flatten_metadata,
        load_survey,
    )
    data, metadata = load_survey(survey, os.environ["DATA_CONFIG_PATH"])
    ccol = SURVEY_COUNTRY_COL[survey]
    out = (data, metadata, ccol, build_admin_cols(metadata, ccol),
           flatten_metadata(metadata), build_country_code_map(metadata, ccol, data))
    _survey_cache[survey] = out
    return out


def compute_one_cell(spec: dict[str, Any]) -> dict:
    """Fit one cell and write its artifacts. Runs INSIDE a worker process.

    Returns a small summary dict — never the oracle frame, so nothing large is pickled
    back across the process boundary.
    """
    t0 = time.perf_counter()
    survey, target, country = spec["survey"], spec["target"], spec["country"]
    outputs_dir = Path(spec.get("outputs_dir") or OUTPUTS_DIR)
    try:
        _ensure_src_on_pythonpath()
        from .layout import cell_dir as layout_cell_dir, tmp_dir
        from .oracle import ORACLE_CONTRACT_VERSION, compute_oracle, load_similarity_model

        data, metadata, ccol, admin_cols, meta_flat, ccodes = _survey_assets(survey)
        import pandas as pd

        threshold = float(spec.get("similarity_threshold", 0.85))
        if threshold not in _similarity_models:
            _similarity_models[threshold] = load_similarity_model(threshold)

        oracle_df, feature_pool, meta_out = compute_oracle(
            data=data,
            metadata=metadata,
            target_var=target,
            country_code=ccodes.get(country, country),
            country_col=ccol,
            admin_cols=admin_cols,
            metadata_flat=meta_flat,
            similarity_model=_similarity_models[threshold],
            tmp_root=tmp_dir(outputs_dir),
            n_repeats=spec.get("n_repeats", 5),
            runtime_mode=spec.get("runtime_mode", "balanced"),
            autogluon_time_limit=spec.get("autogluon_time_limit", 0),
            num_cpus=spec.get("num_cpus"),
            eval_metric=spec.get("eval_metric"),
            survey_id=survey,
        )
        d = layout_cell_dir(target, country, outputs_dir)
        d.mkdir(parents=True, exist_ok=True)
        oracle_df.to_csv(d / "oracle.csv", index=False)
        pd.DataFrame({"feature_variable": feature_pool}).to_csv(
            d / "feature_pool.csv", index=False
        )
        (d / "oracle_meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
        # Baselines were computed against the previous ranking.
        bl = d / "baselines.json"
        if bl.is_file():
            bl.unlink()
        return {
            "survey": survey, "target": target, "country": country, "ok": True,
            "n_positive": meta_out["n_positive_score"], "n_features": meta_out["n_features"],
            "ceiling10": meta_out["oracle_ceiling"].get("10"),
            "problem_type": meta_out.get("problem_type"),
            "contract": ORACLE_CONTRACT_VERSION,
            "secs": round(time.perf_counter() - t0, 1),
        }
    except Exception as exc:
        return {
            "survey": survey, "target": target, "country": country, "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            "secs": round(time.perf_counter() - t0, 1),
        }


def run_oracle_pool(specs: list[dict[str, Any]], processes: int) -> tuple[int, int]:
    """Fit every spec across `processes` worker processes. Returns (done, failed).

    Specs are ordered by survey so each worker tends to reuse one loaded frame. A worker
    that dies takes its cell with it and nothing else — the pool reports the failure and
    the remaining cells continue, which is exactly what the thread version could not do.
    """
    specs = sorted(specs, key=lambda s: (s["survey"], s["target"], s["country"]))
    done = failed = 0
    total = len(specs)
    t0 = time.time()
    _ensure_src_on_pythonpath()

    with ProcessPoolExecutor(max_workers=max(1, processes)) as ex:
        futures = {ex.submit(compute_one_cell, s): s for s in specs}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:  # worker died outright (OOM, native crash)
                r = {"survey": s["survey"], "target": s["target"], "country": s["country"],
                     "ok": False, "error": f"worker died: {type(exc).__name__}: {exc}",
                     "secs": 0}
            if r["ok"]:
                done += 1
                print(
                    f"  [{done + failed}/{total}] {r['target']} x {r['country']}  "
                    f"{r['n_positive']}/{r['n_features']} positive, "
                    f"{r['problem_type']}, ceiling@10={r['ceiling10']}, {r['secs']:.0f}s",
                    flush=True,
                )
            else:
                failed += 1
                print(f"  [error] {r['target']} x {r['country']}: {r['error']}", flush=True)

    mins = (time.time() - t0) / 60
    rate = (done + failed) / mins if mins else 0
    print(f"\n[pool] done={done} failed={failed} in {mins:.1f} min "
          f"({rate:.2f} cells/min across {processes} processes)")
    return done, failed
