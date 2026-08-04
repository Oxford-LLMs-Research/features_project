"""
Ensemble retrieval/mapping experiment (isolated from main/).

Fuses candidate pools from multiple embedding models (default: MiniLM + mpnet),
then runs one Nemotron disambiguation call per piped feature. Gen/extract are
reused from outputs/main/<selector>/; single-model baselines stay in main/ and
embedding_sensitivity/. New artifacts write only under
outputs/experiments/ensemble_mapping/.

v1 protocol is kimi-only map + score (docs/ensemble_mapping.md).

Examples:
  python scripts/run_ensemble_mapping.py --phase map --selector kimi --disambiguator nemotron --limit 2
  python scripts/run_ensemble_mapping.py --phase map --selector kimi --disambiguator nemotron --arms C
  python scripts/run_ensemble_mapping.py --phase score --selector kimi
  python analysis/ensemble_mapping.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survey_features.config import (  # noqa: E402
    CONDITIONS,
    DEFAULT_EMBEDDING_MODEL,
    DISAMBIGUATORS,
    OUTPUTS_DIR,
    PIPE_TYPES,
    SELECTORS,
)
from survey_features.ensemble import (  # noqa: E402
    DEFAULT_ENSEMBLE_MODELS,
    DEFAULT_MAX_FUSED_MULT,
    DEFAULT_MIN_SIM,
    DEFAULT_TOP_N,
    FUSION_RULE,
    ensemble_label,
    fusion_slug,
)
from survey_features.layout import (  # noqa: E402
    cell_tag,
    ensemble_mapping_dir,
    ensemble_run_dirs,
    genuine_cells,
    sanitize_model_slug,
    selector_dirs,
)

OUT = OUTPUTS_DIR
FIXED_KS = [5, 10]
PRIMARY_DISAMBIG = "nemotron"

_survey_cache: dict = {}


def mapper_generate_fn(mapper_model: str):
    from survey_features.llm import make_generate_fn
    fn, _ = make_generate_fn(
        base_url=os.environ.get("DISAMBIG_BASE_URL") or None,
        api_key=os.environ.get("DISAMBIG_API_KEY") or None,
        model=mapper_model,
        on_error="empty",
    )
    return fn


def survey_assets(survey_id, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    key = (survey_id, embedding_model)
    if key not in _survey_cache:
        from survey_features.retrieval import load_or_build_survey_embeddings
        from survey_features.surveys import extract_survey_variables, load_survey
        _, meta = load_survey(survey_id, os.environ["DATA_CONFIG_PATH"])
        svars = extract_survey_variables(meta)
        emb, vcodes = load_or_build_survey_embeddings(svars, survey_id, embedding_model)
        _survey_cache[key] = (svars, emb, vcodes)
    return _survey_cache[key]


def _upsert_manifest(**fields) -> None:
    root = ensemble_mapping_dir(OUT)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "experiment": "ensemble_mapping",
        "baseline": {
            "minilm_scores_root": "outputs/main",
            "alt_scores_root": "outputs/experiments/embedding_sensitivity",
            "note": (
                "Single-model maps/scores are never overwritten; ensemble writes "
                "only under ensemble_mapping/."
            ),
        },
        "fusion_rule": FUSION_RULE,
        "runs": [],
    }
    data.setdefault("runs", []).append({"ts": datetime.now(timezone.utc).isoformat(), **fields})
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _save_map(
    path: Path,
    survey,
    target,
    country,
    cond,
    arm,
    disambig_key,
    cm,
    emb_label: str,
    models: list[str],
    timing: dict,
    top_n: int,
    min_similarity: float,
    max_fused: int,
):
    mean_pool = 0.0
    n_piped = 0
    for f in cm.features:
        if f.piped:
            n_piped += 1
            mean_pool += len(f.candidates)
    if n_piped:
        mean_pool /= n_piped
    rec = {
        "survey": survey, "target": target, "country": country, "condition": cond,
        "arm": arm, "disambiguator": disambig_key, "disambig_model": cm.mapper_model,
        "embedding_model": emb_label,
        "embedding_models": list(models),
        "fusion_rule": FUSION_RULE,
        "top_n": top_n,
        "min_similarity": min_similarity,
        "max_fused": max_fused,
        "n_features": cm.n_features, "n_piped": cm.n_piped, "n_mapped": cm.n_mapped,
        "n_none": cm.n_none, "n_bundled": cm.n_bundled, "type_counts": cm.type_counts(),
        "mapped_codes": cm.mapped_codes,
        "mean_pool_size": round(mean_pool, 3),
        "timing": timing,
        "features": [
            {
                "feature": f.feature_label, "context": f.feature_context,
                "sub_items": f.sub_items, "type": f.ftype, "piped": f.piped,
                "selected_code": f.selected_code, "selected_text": f.selected_text,
                "n_candidates": len(f.candidates),
                "disambig_raw": f.disambig_raw[:80],
            }
            for f in cm.features
        ],
    }
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")


def phase_map(
    selector_key: str,
    disambig_key: str,
    arms=("C",),
    force=False,
    limit=None,
    run_tag: str | None = None,
    map_workers: int = 1,
    embedding_models: list[str] | None = None,
    top_n: int = DEFAULT_TOP_N,
    min_similarity: float = DEFAULT_MIN_SIM,
    max_fused_mult: int = DEFAULT_MAX_FUSED_MULT,
):
    """Ensemble retrieve + single disambig; write under ensemble_mapping/."""
    from survey_features.disambig import map_features_ensemble
    from survey_features.retrieval import make_embed_fn, target_excluded_codes
    from survey_features.timing import TimingLog, default_timing_path

    models = list(embedding_models or DEFAULT_ENSEMBLE_MODELS)
    if len(models) < 2:
        raise SystemExit("ensemble map needs at least two --embedding-models")
    slug = fusion_slug(models)
    emb_label = ensemble_label(models)
    max_fused = int(max_fused_mult) * int(top_n)

    dmodel = DISAMBIGUATORS[disambig_key]
    dgen = mapper_generate_fn(dmodel)
    embed_fns = {m: make_embed_fn(m) for m in models}

    _, extract_dir, _ = selector_dirs(selector_key)
    map_dir, _ = ensemble_run_dirs(slug, selector_key, run_tag=run_tag)
    map_dir.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, int(map_workers))
    timing = TimingLog(default_timing_path("ensemble_map", f"{selector_key}_{disambig_key}"))

    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    print(
        f"[ensemble-map] selector={selector_key} disambiguator={disambig_key} "
        f"arms={arms} cells={len(cells)} models={models} fusion={FUSION_RULE} "
        f"top_n={top_n} min_sim={min_similarity} max_fused={max_fused} "
        f"map_workers={n_workers} -> {map_dir}"
    )

    phase_t0 = time.perf_counter()
    with timing.span(
        "phase_ensemble_map",
        selector=selector_key,
        disambiguator=disambig_key,
        n_cells=len(cells),
        map_workers=n_workers,
        fusion_slug=slug,
    ):
        for i, (survey, target, country) in enumerate(cells):
            # Shared survey_variables from first model; embeddings differ per model.
            svars, _, _ = survey_assets(survey, models[0])
            packs = []
            for m in models:
                _, emb, vcodes = survey_assets(survey, m)
                packs.append({
                    "name": m,
                    "embed_fn": embed_fns[m],
                    "survey_embeddings": emb,
                    "var_codes": vcodes,
                })
            # Candidates come from the union of every model's pool, so the target's
            # near-paraphrases must be excluded in each model's embedding space.
            excluded: set[str] = {target}
            for pack in packs:
                excluded |= target_excluded_codes(
                    target, svars, pack["survey_embeddings"], pack["var_codes"],
                    pack["embed_fn"],
                )
            ctag = cell_tag(survey, target, country)

            if "C" not in arms:
                print("  ! only arm C is supported for ensemble mapping in v1")
                continue

            ep = extract_dir / f"{ctag}.json"
            feat_by_cond = (
                json.loads(ep.read_text(encoding="utf-8"))["features"] if ep.is_file() else None
            )
            for cond in CONDITIONS:
                op = map_dir / f"C__{disambig_key}__{ctag}__{cond}.json"
                if op.is_file() and not force:
                    continue
                if feat_by_cond is None:
                    print(
                        f"  ! missing extraction for {ctag}; "
                        "run run_main --phase extract first"
                    )
                    continue
                with timing.span(
                    "cell_ensemble_map",
                    arm="C",
                    survey=survey,
                    target=target,
                    country=country,
                    condition=cond,
                ):
                    cell_t0 = time.perf_counter()
                    cm, cell_timing = map_features_ensemble(
                        f"{target}_{country}", "C_free", feat_by_cond.get(cond, []),
                        packs, svars, dgen, mapper_model=dmodel,
                        excluded_codes=excluded, pipe_types=PIPE_TYPES,
                        top_n=top_n, min_similarity=min_similarity,
                        max_fused=max_fused, workers=n_workers,
                    )
                    cell_timing["cell_wall_s"] = time.perf_counter() - cell_t0
                    _save_map(
                        op, survey, target, country, cond, "C", disambig_key, cm,
                        emb_label, models, cell_timing, top_n, min_similarity, max_fused,
                    )
                    timing.record(
                        "cell_timing_detail",
                        cell_timing["cell_wall_s"],
                        survey=survey, target=target, country=country, condition=cond,
                        retrieve_s=cell_timing["retrieve_wall_s_total"],
                        disambig_s=cell_timing["disambig_wall_s"],
                        n_disambig_calls=cell_timing["n_disambig_calls"],
                        **{
                            f"retrieve_{sanitize_model_slug(k)}": v
                            for k, v in cell_timing["retrieve_wall_s_by_model"].items()
                        },
                    )
            print(f"  [{i+1}/{len(cells)}] {survey} {target} {country} mapped (ensemble)")

    phase_wall = time.perf_counter() - phase_t0
    print(f"[ensemble-map] done -> {map_dir} (phase wall {phase_wall:.1f}s)")
    timing.print_summary()
    try:
        map_rel = str(map_dir.relative_to(ROOT))
    except ValueError:
        map_rel = str(map_dir)
    _upsert_manifest(
        phase="map",
        selector=selector_key,
        disambiguator=disambig_key,
        embedding_models=models,
        fusion_rule=FUSION_RULE,
        fusion_slug=slug,
        embedding_label=emb_label,
        top_n=top_n,
        min_similarity=min_similarity,
        max_fused=max_fused,
        map_dir=map_rel,
        phase_wall_s=round(phase_wall, 3),
        limit=limit,
        force=force,
        map_workers=n_workers,
    )


def _load_ensemble_map(map_dir: Path, dk: str, survey, target, country, cond):
    p = map_dir / f"C__{dk}__{cell_tag(survey, target, country)}__{cond}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def phase_score(
    selector_key: str,
    force=False,
    limit=None,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
    run_tag: str | None = None,
    embedding_models: list[str] | None = None,
):
    """Score ensemble arm-C maps (natural k + fixed k=5/10).

    Writes only under ensemble_mapping/ — never main/ or embedding_sensitivity.
    """
    from survey_features.score_cell import (
        resolve_score_n_draws,
        resolve_score_workers,
        resolve_score_xgb_nthread,
        run_score_jobs,
        score_cols,
    )

    models = list(embedding_models or DEFAULT_ENSEMBLE_MODELS)
    slug = fusion_slug(models)
    n_draws = resolve_score_n_draws()
    workers = resolve_score_workers(score_workers)
    nthread = resolve_score_xgb_nthread(workers, score_xgb_nthread)

    map_dir, out_csv = ensemble_run_dirs(slug, selector_key, run_tag=run_tag)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _ = force  # accepted for CLI parity; always rewrite like run_main score

    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    cols = score_cols()
    dk = PRIMARY_DISAMBIG
    emb_label = ensemble_label(models)

    specs = []
    for survey, target, country in cells:
        evals = []
        for cond in CONDITIONS:
            rec = _load_ensemble_map(map_dir, dk, survey, target, country, cond)
            if rec is None:
                continue
            label = rec.get("embedding_model") or emb_label
            codes = rec.get("mapped_codes") or []
            evals.append({
                "condition": cond,
                "arm": "C",
                "disambiguator": dk,
                "embedding_model": label,
                "codes": codes,
            })
        specs.append({
            "survey": survey, "target": target, "country": country,
            "evals": evals,
            "n_draws": n_draws, "nthread": nthread, "fixed_ks": list(FIXED_KS),
            "outputs_dir": str(OUT),
        })

    print(
        f"[ensemble-score] selector={selector_key} fusion={slug} "
        f"cells={len(cells)} workers={workers} nthread={nthread} n_draws={n_draws} "
        f"-> {out_csv}"
    )
    n_written = run_score_jobs(
        specs, out_csv, cols, workers=workers, log_prefix="[ensemble-score]",
    )
    try:
        scores_rel = str(out_csv.relative_to(ROOT))
    except ValueError:
        scores_rel = str(out_csv)
    _upsert_manifest(
        phase="score",
        selector=selector_key,
        fusion_slug=slug,
        embedding_models=models,
        n_draws=n_draws,
        n_rows=n_written,
        scores_csv=scores_rel,
        score_workers=workers,
        score_xgb_nthread=nthread,
        limit=limit,
        force=force,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--phase", choices=["map", "score"], required=True)
    ap.add_argument(
        "--selector",
        choices=list(SELECTORS),
        default="kimi",
        help="v1 default: kimi (deepseek is an optional extension)",
    )
    ap.add_argument("--disambiguator", choices=list(DISAMBIGUATORS),
                    help="required for --phase map")
    ap.add_argument("--arms", default="C", help="map arms (default: C only)")
    ap.add_argument(
        "--embedding-models",
        default=",".join(DEFAULT_ENSEMBLE_MODELS),
        help="comma-separated embedders to fuse (default: MiniLM,mpnet)",
    )
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIM)
    ap.add_argument(
        "--max-fused-mult",
        type=int,
        default=DEFAULT_MAX_FUSED_MULT,
        help="fused pool cap = this × top_n (default: 2)",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--run-tag",
        default=None,
        metavar="TAG",
        help="Write under experiments/ensemble_mapping/runs/<TAG>/",
    )
    ap.add_argument(
        "--map-workers",
        type=int,
        default=None,
        help="per-feature disambig ThreadPool for --phase map (default: MAP_WORKERS or 1)",
    )
    ap.add_argument(
        "--score-workers",
        type=int,
        default=None,
        help="cell ProcessPool size for --phase score",
    )
    ap.add_argument(
        "--score-xgb-nthread",
        type=int,
        default=None,
        help="XGBoost nthread per fit",
    )
    args = ap.parse_args()

    from survey_features.timing import resolve_workers

    if args.phase != "score" and (args.score_workers is not None or args.score_xgb_nthread is not None):
        ap.error("--score-workers / --score-xgb-nthread only apply to --phase score")
    if args.phase != "map" and args.map_workers is not None:
        ap.error("--map-workers only applies to --phase map")

    models = [m.strip() for m in args.embedding_models.split(",") if m.strip()]
    map_workers = resolve_workers(args.map_workers, "MAP_WORKERS", default=1)

    if args.phase == "map":
        if not args.disambiguator:
            ap.error("--disambiguator required for --phase map")
        phase_map(
            args.selector,
            args.disambiguator,
            arms=tuple(a.strip() for a in args.arms.split(",") if a.strip()),
            force=args.force,
            limit=args.limit,
            run_tag=args.run_tag,
            map_workers=map_workers,
            embedding_models=models,
            top_n=args.top_n,
            min_similarity=args.min_similarity,
            max_fused_mult=args.max_fused_mult,
        )
    else:
        phase_score(
            args.selector,
            force=args.force,
            limit=args.limit,
            score_workers=args.score_workers,
            score_xgb_nthread=args.score_xgb_nthread,
            run_tag=args.run_tag,
            embedding_models=models,
        )


if __name__ == "__main__":
    main()
