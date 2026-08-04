"""
Sub-item mapping experiment runner (isolated from main/).

Reuses gen/extract from outputs/main/<selector>/ (legacy format_pilot/ dual-resolved).
Writes maps + scores under outputs/experiments/subitem_mapping/<selector>/
(or …/runs/<run_tag>/ with --run-tag). v1 protocol is kimi-only map + score
(docs/subitem_mapping.md): natural-k parent/expanded plus matched k=5/10.

Examples (v1):
  python scripts/run_subitem_mapping.py --phase map --selector kimi --disambiguator nemotron --limit 2
  python scripts/run_subitem_mapping.py --phase map --selector kimi --disambiguator nemotron --arms C
  python scripts/run_subitem_mapping.py --phase score --selector kimi --k-modes parent,expanded
  python scripts/run_subitem_mapping.py --phase map --selector kimi --disambiguator nemotron --run-tag alice_v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from survey_features.layout import (  # noqa: E402
    cell_tag,
    genuine_cells,
    selector_dirs,
    subitem_mapping_dir,
    subitem_run_dirs,
)

OUT = OUTPUTS_DIR
FIXED_KS = [5, 10]
# k_mode -> field on expanded map JSON (docs/subitem_mapping.md scoring table)
K_MODE_CODE_FIELDS = {
    "parent": "parent_codes",
    "expanded": "expanded_codes",
    "subitems_only": "subitem_codes",
}
PRIMARY_DISAMBIG = "nemotron"
_survey_cache: dict = {}


def mapper_generate_fn(mapper_model: str):
    """Extractor/disambiguator client (same contract as scripts/run_main.py)."""
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
    root = subitem_mapping_dir(OUT)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "experiment": "subitem_mapping",
        "baseline": {
            "scores_root": "outputs/main",
            "note": "Parent-only MiniLM arm-C maps/scores; never overwritten by this runner.",
        },
        "runs": [],
    }
    data.setdefault("runs", []).append({"ts": datetime.now(timezone.utc).isoformat(), **fields})
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def phase_map(selector_key: str, disambig_key: str, arms=("C",), force=False, limit=None,
              run_tag: str | None = None, map_workers: int = 1):
    """Expand parent + bundled sub_item mapping units; write under subitem_mapping/."""
    from survey_features.retrieval import make_embed_fn, target_excluded_codes
    from survey_features.subitem_map import expanded_cell_to_record, map_features_with_subitems
    from survey_features.timing import TimingLog, default_timing_path

    emb_model = DEFAULT_EMBEDDING_MODEL
    dmodel = DISAMBIGUATORS[disambig_key]
    dgen = mapper_generate_fn(dmodel)
    embed = make_embed_fn(emb_model)
    _, extract_dir, _ = selector_dirs(selector_key)
    map_dir, _, _ = subitem_run_dirs(selector_key, run_tag=run_tag)
    map_dir.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, int(map_workers))
    timing = TimingLog(default_timing_path("subitem_map", f"{selector_key}_{disambig_key}"))

    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    print(
        f"[subitem-map] selector={selector_key} disambiguator={disambig_key} "
        f"arms={arms} cells={len(cells)} embedding={emb_model} map_workers={n_workers} "
        f"-> {map_dir}"
    )

    with timing.span(
        "phase_subitem_map",
        selector=selector_key,
        disambiguator=disambig_key,
        n_cells=len(cells),
        map_workers=n_workers,
    ):
        for i, (survey, target, country) in enumerate(cells):
            svars, emb, vcodes = survey_assets(survey, emb_model)
            excluded = target_excluded_codes(target, svars, emb, vcodes, embed)
            ctag = cell_tag(survey, target, country)

            if "C" not in arms:
                print("  ! only arm C is supported for subitem mapping in v1")
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
                        "run scripts/run_main.py --phase extract first"
                    )
                    continue
                with timing.span(
                    "cell_map",
                    survey=survey,
                    target=target,
                    country=country,
                    condition=cond,
                ):
                    cm = map_features_with_subitems(
                        f"{target}_{country}",
                        "C_free",
                        feat_by_cond.get(cond, []),
                        emb,
                        vcodes,
                        svars,
                        embed,
                        dgen,
                        mapper_model=dmodel,
                        excluded_codes=excluded,
                        pipe_types=PIPE_TYPES,
                        workers=n_workers,
                    )
                    rec = expanded_cell_to_record(
                        survey, target, country, cond, "C", disambig_key, cm, emb_model,
                    )
                    op.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

            print(f"  [{i+1}/{len(cells)}] {survey} {target} {country} mapped ({disambig_key})")

    print(f"[subitem-map] done -> {map_dir}")
    timing.print_summary()
    _upsert_manifest(
        phase="map",
        selector=selector_key,
        disambiguator=disambig_key,
        arms=list(arms),
        embedding_model=emb_model,
        limit=limit,
    )


def _load_subitem_map(map_dir: Path, disambig_key: str, survey, target, country, cond):
    """Arm-C expanded map JSON, or None if missing."""
    ctag = cell_tag(survey, target, country)
    p = map_dir / f"C__{disambig_key}__{ctag}__{cond}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def phase_score(
    selector_key: str,
    k_modes=("parent", "expanded"),
    force=False,
    limit=None,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
    run_tag: str | None = None,
):
    """Score parent vs expanded code sets (natural k + fixed k=5/10).

    Uses survey_features.score_cell (cell ProcessPool, oracle/random cache per k).
    Writes only under subitem_mapping/ — never main/ or embedding_sensitivity.
    """
    from survey_features.score_cell import (
        resolve_score_n_draws,
        resolve_score_workers,
        resolve_score_xgb_nthread,
        run_score_jobs,
        score_cols,
    )

    unknown = [m for m in k_modes if m not in K_MODE_CODE_FIELDS]
    if unknown:
        raise SystemExit(
            f"unknown k_mode(s) {unknown}; choose from {sorted(K_MODE_CODE_FIELDS)}"
        )

    n_draws = resolve_score_n_draws()
    workers = resolve_score_workers(score_workers)
    nthread = resolve_score_xgb_nthread(workers, score_xgb_nthread)

    map_dir, _, out_csv = subitem_run_dirs(selector_key, run_tag=run_tag)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Always rewrite (same as run_main score); incremental flush keeps partial CSV
    # on interrupt. --force is accepted for CLI parity with map phase.

    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    cols = score_cols("k_mode")
    dk = PRIMARY_DISAMBIG

    specs = []
    for survey, target, country in cells:
        evals = []
        for cond in CONDITIONS:
            rec = _load_subitem_map(map_dir, dk, survey, target, country, cond)
            if rec is None:
                continue
            emb_label = rec.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
            for k_mode in k_modes:
                field = K_MODE_CODE_FIELDS[k_mode]
                codes = rec.get(field) or []
                evals.append({
                    "condition": cond,
                    "arm": "C",
                    "disambiguator": dk,
                    "embedding_model": emb_label,
                    "k_mode": k_mode,
                    "codes": codes,
                })
        specs.append({
            "survey": survey, "target": target, "country": country,
            "evals": evals,
            "n_draws": n_draws, "nthread": nthread, "fixed_ks": list(FIXED_KS),
            "outputs_dir": str(OUT),
        })

    print(
        f"[subitem-score] selector={selector_key} k_modes={k_modes} "
        f"cells={len(cells)} workers={workers} nthread={nthread} n_draws={n_draws} "
        f"-> {out_csv}"
    )
    n_written = run_score_jobs(
        specs, out_csv, cols, workers=workers, log_prefix="[subitem-score]",
    )
    try:
        scores_rel = str(out_csv.relative_to(ROOT))
    except ValueError:
        scores_rel = str(out_csv)
    _upsert_manifest(
        phase="score",
        selector=selector_key,
        k_modes=list(k_modes),
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
        "--k-modes",
        default="parent,expanded",
        help="comma-separated score modes: parent,expanded,subitems_only (score phase)",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--run-tag",
        default=None,
        metavar="TAG",
        help="Write under experiments/subitem_mapping/runs/<TAG>/ (avoids clobbering canonical)",
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
        help="cell ProcessPool size for --phase score (default: SCORE_WORKERS or min(8, cpus-2))",
    )
    ap.add_argument(
        "--score-xgb-nthread",
        type=int,
        default=None,
        help="XGBoost nthread per fit (default: SCORE_XGB_NTHREAD or cpus // workers)",
    )
    args = ap.parse_args()

    from survey_features.timing import resolve_workers

    if args.phase != "score" and (args.score_workers is not None or args.score_xgb_nthread is not None):
        ap.error("--score-workers / --score-xgb-nthread only apply to --phase score")
    if args.phase != "map" and args.map_workers is not None:
        ap.error("--map-workers only applies to --phase map")

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
        )
    else:
        modes = tuple(m.strip() for m in args.k_modes.split(",") if m.strip())
        phase_score(
            args.selector,
            k_modes=modes,
            force=args.force,
            limit=args.limit,
            score_workers=args.score_workers,
            score_xgb_nthread=args.score_xgb_nthread,
            run_tag=args.run_tag,
        )


if __name__ == "__main__":
    main()
