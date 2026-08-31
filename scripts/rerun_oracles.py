"""
Recompute every cached oracle cell under the current contract (v4: k-fold CV
ranking + eval reserve + untouched valuation holdout).

Drives survey_features.oracle.compute_oracle cell by cell, grouped by survey so each
survey frame is loaded once. Resumable: a cell whose oracle_meta.json already records
ORACLE_CONTRACT_VERSION (and the requested eval_metric, if forced) is skipped
unless --force.

The cell list comes from whatever is already on disk (outputs/cache/cells/<target>_<country>/),
cross-checked against the leakage audit for survey attribution, so this reproduces the
existing grid rather than inventing one.

Before a contract migration, archive the previous caches first, e.g.:
    cp -r outputs/cache/cells outputs/cache/cells_v3

Usage:
    python scripts/rerun_oracles.py --dry-run
    python scripts/rerun_oracles.py                      # all cells, background-friendly
    python scripts/rerun_oracles.py --survey wvs --limit 3

Cluster (SLURM array over the registered grid; see scripts/arc/run_oracles.sbatch):
    python scripts/rerun_oracles.py --cells-csv data/confirmatory_grid_cells.csv \
        --shard 3/8 --processes 12
--shard i/N takes the i-th of N contiguous slices of the CSV (file order is
grouped by survey, so a slice mostly loads one survey frame). Shards are cut
BEFORE the done-check, so a resubmitted task reclaims exactly its own cells.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from survey_features.config import OUTPUTS_DIR  # noqa: E402
from survey_features.layout import (  # noqa: E402
    cache_cells_dir,
    cell_dir as layout_cell_dir,
    leakage_audit_csv_path,
    tmp_dir,
)
from survey_features.oracle import (  # noqa: E402
    AUTOGLUON_RUNTIME_MODES,
    ORACLE_CONTRACT_VERSION,
    ORACLE_EVAL_METRIC,
    _set_local_tmp_dir,
    compute_oracle,
    load_similarity_model,
)
from survey_features.surveys import (  # noqa: E402
    SURVEY_COUNTRY_COL,
    build_admin_cols,
    build_country_code_map,
    flatten_metadata,
    load_survey,
)


def cells_from_yaml(path: Path) -> list[tuple[str, str, str]]:
    """(survey, target, country) from a frozen cells yaml (e.g. prompt-sensitivity v2)."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = doc.get("cells") or []
    out = []
    for r in rows:
        out.append((str(r["survey"]), str(r["target"]), str(r["country"])))
    if not out:
        raise SystemExit(f"No cells in {path}")
    return out


def cells_from_csv(path: Path) -> list[tuple[str, str, str]]:
    """(survey, target, country) from a flat grid CSV (confirmatory_grid_cells.csv)."""
    d = pd.read_csv(path)
    missing = {"survey", "target", "country"} - set(d.columns)
    if missing:
        raise SystemExit(f"{path} lacks columns: {sorted(missing)}")
    if not len(d):
        raise SystemExit(f"No cells in {path}")
    return [(str(r.survey), str(r.target), str(r.country)) for r in d.itertuples()]


def shard_slice(items: list, spec: str) -> list:
    """The i-th of N contiguous slices ("i/N", 0-based)."""
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise SystemExit(f"--shard must look like 3/8, got {spec!r}")
    if not (0 <= i < n):
        raise SystemExit(f"--shard index {i} out of range for {n} shards")
    return items[len(items) * i // n: len(items) * (i + 1) // n]


def cells_from_audit(outputs_dir: Path) -> list[tuple[str, str, str]]:
    """(survey, target, country) for every audited cell — the existing grid."""
    p = leakage_audit_csv_path(outputs_dir)
    if not p.is_file():
        raise SystemExit(f"leakage audit not found at {p}; run scripts/leakage_audit.py first")
    d = pd.read_csv(p)
    return [
        (str(r.survey), str(r.target), str(r.country))
        for r in d.itertuples()
    ]


def already_done(target: str, country: str, metric: str | None, outputs_dir: Path) -> bool:
    """A cell counts as done only if it matches the CURRENT contract, not merely exists.

    The metric is no longer a fixed constant — it now depends on the target's measurement
    level — so the contract VERSION is the reliable signal.
    """
    meta = layout_cell_dir(target, country, outputs_dir) / "oracle_meta.json"
    if not meta.is_file():
        return False
    try:
        rec = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    if rec.get("contract_version") != ORACLE_CONTRACT_VERSION:
        return False
    return metric is None or rec.get("eval_metric") == metric


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--eval-metric", default=None,
        help=("Force one metric for every cell. Default None = let the target's "
              "MEASUREMENT LEVEL choose (log_loss for binary/nominal, spearmanr for "
              "ordinal/continuous); forcing log_loss on a regression target is invalid."),
    )
    ap.add_argument("--runtime-mode", default="balanced", choices=["quick", "balanced", "best"])
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument(
        "--autogluon-time-limit", type=int, default=0,
        help=(
            "Per-fit wall-clock budget in seconds (0 = runtime-mode default). The "
            "preset SPENDS the whole budget, so this — not cell size — sets cell cost. "
            "Do not raise this to unstick a cell that overshoots: lock the tree bag "
            "in oracle.py instead (onboarding.md #5.7)."
        ),
    )
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--cells-yaml",
        type=Path,
        default=None,
        help=(
            "Frozen (survey, target, country) list. Default is the leakage-audit "
            "grid. Use this for experiment grids that are not yet on that audit."
        ),
    )
    ap.add_argument(
        "--cells-csv", type=Path, default=None,
        help="Flat grid CSV (survey,target,country[,...]), e.g. the registered "
             "data/confirmatory_grid_cells.csv. Mutually exclusive with --cells-yaml.",
    )
    ap.add_argument(
        "--shard", default=None,
        help="i/N (0-based): process only the i-th of N contiguous slices of the "
             "cell list — one SLURM array task per slice.",
    )
    ap.add_argument(
        "--processes", type=int, default=0,
        help=(
            "Fit cells in separate PROCESSES (0 = off) — the safe concurrency: each "
            "worker owns its native runtime. Costs one survey load + frame per worker."
        ),
    )
    ap.add_argument(
        "--workers", type=int, default=1,
        help=(
            "THREAD workers — leave at 1 and use --processes. In-process AutoGluon "
            "concurrency starves fits or wedges the interpreter (shared OpenMP "
            "runtime); incident history in docs/onboarding.md #5."
        ),
    )
    args = ap.parse_args()

    if not os.environ.get("DATA_CONFIG_PATH"):
        raise SystemExit("DATA_CONFIG_PATH is not set in .env")

    outputs_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR
    if args.cells_yaml and args.cells_csv:
        raise SystemExit("--cells-yaml and --cells-csv are mutually exclusive")
    if args.cells_yaml:
        cells = cells_from_yaml(args.cells_yaml)
        grid_src = str(args.cells_yaml)
    elif args.cells_csv:
        cells = cells_from_csv(args.cells_csv)
        grid_src = str(args.cells_csv)
    else:
        cells = cells_from_audit(outputs_dir)
        grid_src = "leakage audit"
    if args.shard:
        cells = shard_slice(cells, args.shard)
        grid_src += f" [shard {args.shard}: {len(cells)} cells]"
    if args.survey:
        cells = [c for c in cells if c[0] in set(args.survey)]
    todo = [c for c in cells
            if args.force or not already_done(c[1], c[2], args.eval_metric, outputs_dir)]
    if args.limit:
        todo = todo[: args.limit]

    by_survey: dict[str, list[tuple[str, str]]] = {}
    for survey, target, country in todo:
        by_survey.setdefault(survey, []).append((target, country))

    print(f"[rerun] grid={grid_src}")
    print(f"[rerun] metric={args.eval_metric or 'per target type'} "
          f"runtime={args.runtime_mode} contract=v{ORACLE_CONTRACT_VERSION}")
    # Disk census reads every meta directly (pilot cells outside the audit grid —
    # e.g. targets missing from targets.yaml — still count toward "done").
    census: dict[object, int] = {}
    for meta_path in cache_cells_dir(outputs_dir).glob("*/oracle_meta.json"):
        try:
            v = json.loads(meta_path.read_text(encoding="utf-8")).get("contract_version")
        except Exception:
            v = "unreadable"
        census[v] = census.get(v, 0) + 1
    parts = ", ".join(
        f"{n} at v{v}" + (" (current)" if v == ORACLE_CONTRACT_VERSION else " (stale)")
        if isinstance(v, int) else f"{n} {v}"
        for v, n in sorted(census.items(), key=lambda kv: str(kv[0]))
    )
    print(f"[rerun] disk census: {parts}")
    print(f"[rerun] {len(cells)} cells on the grid, {len(todo)} to (re)compute")
    for s, items in by_survey.items():
        print(f"         {s:16s} {len(items)} cells")
    if args.dry_run:
        return

    tmp_root = _set_local_tmp_dir(tmp_dir(outputs_dir))
    config_path = os.environ["DATA_CONFIG_PATH"]
    done = failed = 0
    counter_lock = threading.Lock()
    cpus = os.cpu_count() or 4
    n_workers = args.workers if args.workers else max(1, min(5, cpus // 4))
    cpus_per_worker = max(1, cpus // n_workers)
    # Wall-clock budget scales with worker count (fits share cores, not time).
    base_limit = AUTOGLUON_RUNTIME_MODES[args.runtime_mode]["time_limit"]
    time_limit = args.autogluon_time_limit or (base_limit * n_workers if n_workers > 1 else 0)
    print(f'[rerun] cell workers: {n_workers} (cpus={cpus}, {cpus_per_worker}/worker, '
          f'time_limit={time_limit or base_limit}s)')
    t_start = time.time()

    if args.processes and args.processes > 1:
        from survey_features.oracle_pool import run_oracle_pool
        cpus_each = max(1, cpus // args.processes)
        specs = [
            {"survey": s, "target": t_, "country": c,
             "outputs_dir": str(outputs_dir),
             "runtime_mode": args.runtime_mode,
             "autogluon_time_limit": args.autogluon_time_limit,
             "n_repeats": args.n_repeats,
             "eval_metric": args.eval_metric,
             "num_cpus": cpus_each}
            for s, t_, c in todo
        ]
        print(f"[rerun] PROCESS pool: {args.processes} workers, {cpus_each} cores each")
        d, f = run_oracle_pool(specs, args.processes)
        print(f"[rerun] done={d} failed={f} in {(time.time()-t_start)/60:.1f} min")
        print("[rerun] next: python scripts/leakage_audit.py --with-data")
        return

    similarity_model = load_similarity_model(0.85)
    for survey, items in by_survey.items():
        print(f"\n=== {survey} ({len(items)} cells) ===", flush=True)
        data, metadata = load_survey(survey, config_path)
        country_col = SURVEY_COUNTRY_COL[survey]
        admin_cols = build_admin_cols(metadata, country_col)
        metadata_flat = flatten_metadata(metadata)
        country_codes = build_country_code_map(metadata, country_col, data)

        def _one(target: str, country: str) -> str:
            nonlocal done, failed
            t0 = time.time()
            try:
                oracle_df, feature_pool, meta_out = compute_oracle(
                    data=data,
                    metadata=metadata,
                    target_var=target,
                    country_code=country_codes.get(country, country),
                    country_col=country_col,
                    admin_cols=admin_cols,
                    metadata_flat=metadata_flat,
                    similarity_model=similarity_model,
                    tmp_root=tmp_root,
                    n_repeats=args.n_repeats,
                    runtime_mode=args.runtime_mode,
                    autogluon_time_limit=time_limit,
                    num_cpus=cpus_per_worker,
                    survey_id=survey,
                    eval_metric=args.eval_metric,
                )
                d = layout_cell_dir(target, country, outputs_dir)
                d.mkdir(parents=True, exist_ok=True)
                oracle_df.to_csv(d / "oracle.csv", index=False)
                pd.DataFrame({"feature_variable": feature_pool}).to_csv(
                    d / "feature_pool.csv", index=False
                )
                (d / "oracle_meta.json").write_text(
                    json.dumps(meta_out, indent=2), encoding="utf-8"
                )
                # Stale baselines were computed against the previous oracle ranking.
                bl = d / "baselines.json"
                if bl.is_file():
                    bl.unlink()
                with counter_lock:
                    done += 1
                    n = done
                print(
                    f"  [{n}/{len(todo)}] {target} x {country}  "
                    f"{meta_out['n_positive_score']}/{meta_out['n_features']} positive, "
                    f"ceiling@10={meta_out['oracle_ceiling'].get('10')}, "
                    f"{time.time()-t0:.0f}s",
                    flush=True,
                )
                return "ok"
            except Exception as exc:
                with counter_lock:
                    failed += 1
                print(f"  [error] {target} x {country}: {type(exc).__name__}: {exc}", flush=True)
                return "error"

        if n_workers <= 1:
            for target, country in items:
                _one(target, country)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_one, t, c) for t, c in items]
                for f in as_completed(futures):
                    f.result()

    mins = (time.time() - t_start) / 60
    print(f"\n[rerun] done={done} failed={failed} in {mins:.1f} min")
    print("[rerun] next: python scripts/leakage_audit.py --with-data  "
          "(the signal/leakage screen is metric-dependent)")


if __name__ == "__main__":
    main()
