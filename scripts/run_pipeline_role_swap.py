"""
Pipeline role-swap experiment: MiniMax extract + Flash disambig on fixed gen essays.

Reuses free-text caches (default: prompt_sensitivity kimi/social_scientist) so only
extract → map → score change. Writes under outputs/experiments/pipeline_role_swap/.

  python scripts/run_pipeline_role_swap.py --smoke
  python scripts/run_pipeline_role_swap.py
  python scripts/run_pipeline_role_swap.py --source-selector deepseek_v4 --source-arm social_scientist

Compare maps/scores to the source tree's Qwen+Nemotron baseline (same gen essays).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_run_main():
    path = ROOT / "scripts" / "run_main.py"
    spec = importlib.util.spec_from_file_location("run_main_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


main_run = _load_run_main()

from survey_features.config import (  # noqa: E402
    CONDITIONS,
    DEFAULT_EMBEDDING_MODEL,
    DISAMBIGUATORS,
    OUTPUTS_DIR,
    PIPE_TYPES,
    ROLE_SWAP_DISAMBIG_KEY,
    ROLE_SWAP_EXTRACTOR,
    ROOT as PKG_ROOT,
)
from survey_features.layout import (  # noqa: E402
    cell_tag,
    pipeline_role_swap_dirs,
    pipeline_role_swap_root,
    pipeline_role_swap_scores_path,
    prompt_sensitivity_dirs,
)

OUT = OUTPUTS_DIR
DEFAULT_CELLS = PKG_ROOT / "data" / "prompt_sensitivity_cells.yaml"
DEFAULT_PIPELINE_WORKERS = 4
DEFAULT_MAP_WORKERS = 8
DEFAULT_SOURCE_SELECTOR = "kimi"
DEFAULT_SOURCE_ARM = "social_scientist"
RUN_KEY = "minimax_flash"


def load_cells(path: Path) -> list[tuple[str, str, str]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = doc.get("cells") or []
    out = [(str(r["survey"]), str(r["target"]), str(r["country"])) for r in rows]
    if not out:
        raise ValueError(f"No cells in {path}")
    return out


def phase_extract_map(
    cells: list[tuple[str, str, str]],
    *,
    gen_dir: Path,
    extract_dir: Path,
    map_dir: Path,
    extractor_model: str,
    disambig_key: str,
    force: bool = False,
    pipeline_workers: int = DEFAULT_PIPELINE_WORKERS,
    map_workers: int = DEFAULT_MAP_WORKERS,
    with_score: bool = True,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
    run_key: str = RUN_KEY,
    disambig_max_tokens: int = 8192,
    extract_max_tokens: int = 8192,
) -> None:
    """extract → map from existing gen essays; optional score."""
    from survey_features.extraction import extract_features
    from survey_features.llm import TokenUsageLog, default_usage_path
    from survey_features.mapping import map_features_with_subitems
    from survey_features.retrieval import make_embed_fn, target_excluded_codes
    from survey_features.timing import TimingLog, default_timing_path

    if disambig_key not in DISAMBIGUATORS:
        raise KeyError(f"Unknown disambiguator {disambig_key!r}")
    dmodel = DISAMBIGUATORS[disambig_key]
    emb_model = DEFAULT_EMBEDDING_MODEL

    extract_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    n_pipe = max(1, int(pipeline_workers))
    n_map = max(1, int(map_workers))
    tag = f"role_swap_{run_key}_{disambig_key}"
    timing = TimingLog(default_timing_path("pipeline", tag))
    usage = TokenUsageLog(default_usage_path("pipeline", tag))

    extract_fn = main_run.mapper_generate_fn(extractor_model, usage_log=usage)
    disambig_fn = main_run.mapper_generate_fn(dmodel, usage_log=usage)
    _raw_embed = make_embed_fn(emb_model)
    _embed_lock = threading.Lock()

    def embed(texts):
        with _embed_lock:
            return _raw_embed(texts)

    print(
        f"[role_swap] extractor={extractor_model} disambig={disambig_key} ({dmodel}) "
        f"cells={len(cells)} pipeline_workers={n_pipe} map_workers={n_map} "
        f"with_score={with_score}"
    )
    print(f"  gen source {gen_dir}")
    print(f"  -> {extract_dir.parent}")

    counts = {
        "ext_wrote": 0, "ext_skip": 0, "map_wrote": 0, "map_skip": 0, "errors": 0,
    }
    counts_lock = threading.Lock()
    print_lock = threading.Lock()

    def _bump(key, n=1):
        with counts_lock:
            counts[key] += n

    def _extract_one(survey, target, country) -> str:
        ctag = cell_tag(survey, target, country)
        op = extract_dir / f"{ctag}.json"
        if op.is_file() and not force:
            _bump("ext_skip")
            return "skipped"
        gp = gen_dir / f"{ctag}.json"
        if not gp.is_file():
            return "missing_gen"
        with timing.span("cell_extract", survey=survey, target=target, country=country):
            gen_rec = json.loads(gp.read_text(encoding="utf-8"))
            responses = gen_rec["responses"]
            rec = {
                "survey": survey, "target": target, "country": country,
                "extractor_model": extractor_model,
                "source_gen": str(gp),
                "source_selector_model": gen_rec.get("selector_model"),
                "source_prompt_arm": gen_rec.get("prompt_arm"),
                "features": {},
            }
            for cond in CONDITIONS:
                feats, _raw = extract_features(
                    responses.get(cond, "") or "",
                    extract_fn,
                    max_tokens=extract_max_tokens,
                )
                rec["features"][cond] = feats
            op.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        _bump("ext_wrote")
        return "wrote"

    def _map_one(survey, target, country) -> str:
        svars, emb, vcodes = main_run.survey_assets(survey, emb_model)
        excluded = target_excluded_codes(target, svars, emb, vcodes, embed)
        ctag = cell_tag(survey, target, country)
        ep = extract_dir / f"{ctag}.json"
        if not ep.is_file():
            return "missing_extract"
        feat_by_cond = json.loads(ep.read_text(encoding="utf-8"))["features"]
        wrote = skipped = 0
        for cond in CONDITIONS:
            op = map_dir / f"C__{disambig_key}__{ctag}__{cond}.json"
            if op.is_file() and not force:
                skipped += 1
                continue
            with timing.span(
                "cell_map", survey=survey, target=target,
                country=country, condition=cond,
            ):
                cm = map_features_with_subitems(
                    f"{target}_{country}", "C_free", feat_by_cond.get(cond, []),
                    emb, vcodes, svars, embed, disambig_fn, mapper_model=dmodel,
                    excluded_codes=excluded, pipe_types=PIPE_TYPES, workers=n_map,
                    disambig_max_tokens=disambig_max_tokens,
                )
                main_run._save_map(op, survey, target, country, cond, disambig_key, cm, emb_model)
            wrote += 1
        if wrote:
            _bump("map_wrote", wrote)
        if skipped:
            _bump("map_skip", skipped)
        return "wrote" if wrote else "skipped"

    def _cell_chain(idx_cell):
        i, (survey, target, country) = idx_cell
        ctag = cell_tag(survey, target, country)
        try:
            with timing.span("cell_pipeline", survey=survey, target=target, country=country):
                e = _extract_one(survey, target, country)
                if e == "missing_gen":
                    with print_lock:
                        print(f"  ! [{i+1}/{len(cells)}] {ctag}: missing gen at {gen_dir}")
                    _bump("errors")
                    return
                m = _map_one(survey, target, country)
                if m == "missing_extract":
                    with print_lock:
                        print(f"  ! [{i+1}/{len(cells)}] {ctag}: missing extract")
                    _bump("errors")
                    return
            with print_lock:
                print(f"  [{i+1}/{len(cells)}] {survey} {target} {country}: extract={e} map={m}")
        except Exception as exc:
            _bump("errors")
            with print_lock:
                print(f"  ! [{i+1}/{len(cells)}] {ctag}: {type(exc).__name__}: {str(exc)[:120]}")

    with timing.span(
        "phase_pipeline",
        run_key=run_key,
        extractor=extractor_model,
        disambiguator=disambig_key,
        n_cells=len(cells),
        pipeline_workers=n_pipe,
        map_workers=n_map,
    ):
        for survey, _, _ in cells:
            main_run.survey_assets(survey, emb_model)
        indexed = list(enumerate(cells))
        if n_pipe <= 1 or len(cells) <= 1:
            for item in indexed:
                _cell_chain(item)
        else:
            with ThreadPoolExecutor(max_workers=min(n_pipe, len(cells))) as ex:
                futs = [ex.submit(_cell_chain, item) for item in indexed]
                for fut in as_completed(futs):
                    fut.result()

    print(
        f"[role_swap] done. extract={counts['ext_wrote']}/{counts['ext_skip']} "
        f"map_files={counts['map_wrote']}/{counts['map_skip']} "
        f"errors={counts['errors']} -> {map_dir}"
    )
    timing.print_summary()
    usage.print_summary()

    if with_score:
        phase_score(
            cells,
            map_dir=map_dir,
            disambig_key=disambig_key,
            run_key=run_key,
            score_workers=score_workers,
            score_xgb_nthread=score_xgb_nthread,
        )


def phase_score(
    cells: list[tuple[str, str, str]],
    *,
    map_dir: Path,
    disambig_key: str,
    run_key: str,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
) -> None:
    from survey_features.score_cell import (
        resolve_score_n_draws,
        resolve_score_workers,
        resolve_score_xgb_nthread,
        run_score_jobs,
        score_cols,
    )
    from survey_features.timing import TimingLog, default_timing_path

    n_draws = resolve_score_n_draws()
    workers = resolve_score_workers(score_workers)
    nthread = resolve_score_xgb_nthread(workers, score_xgb_nthread)
    timing = TimingLog(default_timing_path("score", f"role_swap_{run_key}"))
    out_csv = pipeline_role_swap_scores_path(run_key, OUT)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    emb_label = DEFAULT_EMBEDDING_MODEL
    cols = score_cols()

    specs = []
    for survey, target, country in cells:
        evals = []
        for cond in CONDITIONS:
            codes = main_run._map_codes(map_dir, disambig_key, survey, target, country, cond)
            if codes is None:
                continue
            evals.append({
                "condition": cond, "arm": "C", "disambiguator": disambig_key,
                "embedding_model": emb_label, "codes": codes,
            })
        specs.append({
            "survey": survey, "target": target, "country": country,
            "evals": evals,
            "n_draws": n_draws, "nthread": nthread, "fixed_ks": list(main_run.FIXED_KS),
            "outputs_dir": str(OUT),
        })

    print(
        f"[role_swap score] run={run_key} cells={len(cells)} "
        f"workers={workers} -> {out_csv}"
    )
    with timing.span("phase_score", run_key=run_key, n_cells=len(cells), score_workers=workers):
        run_score_jobs(specs, out_csv, cols, workers=workers, log_prefix="[role_swap score]")
    timing.print_summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-selector", default=DEFAULT_SOURCE_SELECTOR,
                    help="prompt_sensitivity selector key holding gen essays")
    ap.add_argument("--source-arm", default=DEFAULT_SOURCE_ARM,
                    help="prompt_sensitivity prompt arm holding gen essays")
    ap.add_argument("--cells-file", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--smoke", action="store_true", help="first 2 cells only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extractor", default=ROLE_SWAP_EXTRACTOR)
    ap.add_argument("--disambiguator", default=ROLE_SWAP_DISAMBIG_KEY,
                    choices=list(DISAMBIGUATORS))
    ap.add_argument("--run-key", default=RUN_KEY,
                    help="subdir / scores stem under pipeline_role_swap/")
    ap.add_argument("--pipeline-workers", type=int, default=None)
    ap.add_argument("--map-workers", type=int, default=None)
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--score-workers", type=int, default=None)
    ap.add_argument("--score-xgb-nthread", type=int, default=None)
    ap.add_argument(
        "--disambig-max-tokens", type=int, default=8192,
        help="CoT budget for Flash-style disambiguators (default 8192)",
    )
    ap.add_argument(
        "--extract-max-tokens", type=int, default=8192,
        help="token budget for MiniMax-style extractors (default 8192)",
    )
    args = ap.parse_args()

    from survey_features.timing import resolve_workers

    cells = load_cells(args.cells_file)
    if args.smoke:
        cells = cells[:2]

    gen_dir, _, _ = prompt_sensitivity_dirs(args.source_selector, args.source_arm, OUT)
    if not gen_dir.is_dir():
        ap.error(f"gen source missing: {gen_dir}")

    extract_dir, map_dir = pipeline_role_swap_dirs(args.run_key, OUT)
    root = pipeline_role_swap_root(OUT)
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_selector": args.source_selector,
        "source_arm": args.source_arm,
        "gen_dir": str(gen_dir),
        "extractor": args.extractor,
        "disambiguator": args.disambiguator,
        "disambiguator_model": DISAMBIGUATORS[args.disambiguator],
        "run_key": args.run_key,
        "n_cells": len(cells),
    }
    (root / "source_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    pipeline_workers = resolve_workers(
        args.pipeline_workers, "PIPELINE_WORKERS", default=DEFAULT_PIPELINE_WORKERS,
    )
    map_workers = resolve_workers(
        args.map_workers, "MAP_WORKERS", default=DEFAULT_MAP_WORKERS,
    )

    phase_extract_map(
        cells,
        gen_dir=gen_dir,
        extract_dir=extract_dir,
        map_dir=map_dir,
        extractor_model=args.extractor,
        disambig_key=args.disambiguator,
        force=args.force,
        pipeline_workers=pipeline_workers,
        map_workers=map_workers,
        with_score=not args.no_score,
        score_workers=args.score_workers,
        score_xgb_nthread=args.score_xgb_nthread,
        run_key=args.run_key,
        disambig_max_tokens=args.disambig_max_tokens,
        extract_max_tokens=args.extract_max_tokens,
    )


if __name__ == "__main__":
    main()
