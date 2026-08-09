"""
Prompt-sensitivity experiment: system-message arms × selectors on a frozen cell grid.

Writes under outputs/experiments/prompt_sensitivity/<selector>/<arm>/ so confirmatory
main/<selector>/ trees are never clobbered.

  python scripts/run_prompt_sensitivity.py --selector kimi --arm none --smoke
  python scripts/run_prompt_sensitivity.py --selector deepseek_v4 --arm social_scientist
  python scripts/run_prompt_sensitivity.py --all   # full factorial (costly)

Defaults stress concurrent pipeline workers (pipeline=4, map=8) for iteration speed.
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
    EXPERIMENT_SELECTORS,
    EXTRACTOR_MODEL,
    OUTPUTS_DIR,
    PIPE_TYPES,
    ROOT as PKG_ROOT,
)
from survey_features.layout import (  # noqa: E402
    cell_tag,
    prompt_sensitivity_dirs,
    prompt_sensitivity_root,
    prompt_sensitivity_scores_path,
)
from survey_features.prompts import DEFAULT_PROMPT_ARM, PROMPT_ARMS  # noqa: E402

OUT = OUTPUTS_DIR
DEFAULT_CELLS = PKG_ROOT / "data" / "prompt_sensitivity_cells.yaml"
DEFAULT_DISAMBIG = "nemotron"
# Concurrent defaults for this experiment (override with flags / env).
DEFAULT_PIPELINE_WORKERS = 4
DEFAULT_MAP_WORKERS = 8


def load_cells(path: Path) -> list[tuple[str, str, str]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = doc.get("cells") or []
    out = []
    for r in rows:
        out.append((str(r["survey"]), str(r["target"]), str(r["country"])))
    if not out:
        raise ValueError(f"No cells in {path}")
    return out


def resolve_selector_model(selector_key: str) -> str:
    if selector_key not in EXPERIMENT_SELECTORS:
        raise KeyError(
            f"Unknown experiment selector {selector_key!r}; "
            f"choose from {sorted(EXPERIMENT_SELECTORS)}"
        )
    return EXPERIMENT_SELECTORS[selector_key]["model"]


def phase_pipeline_arm(
    selector_key: str,
    arm: str,
    cells: list[tuple[str, str, str]],
    *,
    disambig_key: str = DEFAULT_DISAMBIG,
    force: bool = False,
    pipeline_workers: int = DEFAULT_PIPELINE_WORKERS,
    map_workers: int = DEFAULT_MAP_WORKERS,
    with_score: bool = False,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
) -> None:
    """gen → extract → map for one (selector, arm); optional score."""
    from survey_features.elicitation import freetext_messages
    from survey_features.extraction import extract_features
    from survey_features.llm import TokenUsageLog, default_usage_path
    from survey_features.mapping import map_features_with_subitems
    from survey_features.retrieval import make_embed_fn, target_excluded_codes
    from survey_features.timing import TimingLog, default_timing_path

    if arm not in PROMPT_ARMS:
        raise ValueError(f"Unknown arm {arm!r}; choose from {sorted(PROMPT_ARMS)}")

    sel_model = resolve_selector_model(selector_key)
    emb_model = DEFAULT_EMBEDDING_MODEL
    dmodel = DISAMBIGUATORS[disambig_key]
    gen_dir, extract_dir, map_dir = prompt_sensitivity_dirs(selector_key, arm, OUT)
    for d in (gen_dir, extract_dir, map_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_pipe = max(1, int(pipeline_workers))
    n_map = max(1, int(map_workers))
    tag = f"prompt_sens_{selector_key}_{arm}_{disambig_key}"
    timing = TimingLog(default_timing_path("pipeline", tag))
    usage = TokenUsageLog(default_usage_path("pipeline", tag))

    gen_fn = main_run.selector_generate_fn(sel_model, usage_log=usage)
    extract_fn = main_run.mapper_generate_fn(EXTRACTOR_MODEL, usage_log=usage)
    disambig_fn = main_run.mapper_generate_fn(dmodel, usage_log=usage)
    _raw_embed = make_embed_fn(emb_model)
    _embed_lock = threading.Lock()

    def embed(texts):
        with _embed_lock:
            return _raw_embed(texts)

    print(
        f"[prompt_sens] selector={selector_key} ({sel_model}) arm={arm} "
        f"disambig={disambig_key} cells={len(cells)} "
        f"pipeline_workers={n_pipe} map_workers={n_map} with_score={with_score}"
    )
    print(f"  -> {prompt_sensitivity_root(OUT) / selector_key / arm}")

    counts = {
        "gen_wrote": 0, "gen_skip": 0, "ext_wrote": 0, "ext_skip": 0,
        "map_wrote": 0, "map_skip": 0, "errors": 0,
    }
    counts_lock = threading.Lock()
    print_lock = threading.Lock()

    def _bump(key, n=1):
        with counts_lock:
            counts[key] += n

    def _gen_one(survey, target, country) -> str:
        ctag = cell_tag(survey, target, country)
        out_path = gen_dir / f"{ctag}.json"
        if out_path.is_file() and not force:
            _bump("gen_skip")
            return "skipped"
        svars, _, _ = main_run.survey_assets(survey)
        qtext = svars.get(target, target)
        with timing.span("cell_gen", survey=survey, target=target, country=country, arm=arm):
            rec = {
                "survey": survey, "target": target, "country": country,
                "question_text": qtext, "selector_model": sel_model,
                "prompt_arm": arm, "system_prompt": PROMPT_ARMS[arm],
                "responses": {},
            }
            for cond in CONDITIONS:
                messages = freetext_messages(
                    qtext,
                    country if cond == "country_provided" else None,
                    prompt_arm=arm,
                )
                # 8192: some Kimi/Pro essays hit 4096 finish_reason=length → empty content.
                resp = gen_fn(messages, max_tokens=8192, usage_phase="feature_list")
                if not resp:
                    rec.setdefault("errors", {})[cond] = "empty response after retries"
                rec["responses"][cond] = resp
            out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        _bump("gen_wrote")
        return "wrote"

    def _extract_one(survey, target, country) -> str:
        ctag = cell_tag(survey, target, country)
        op = extract_dir / f"{ctag}.json"
        if op.is_file() and not force:
            _bump("ext_skip")
            return "skipped"
        gp = gen_dir / f"{ctag}.json"
        if not gp.is_file():
            return "missing_gen"
        with timing.span("cell_extract", survey=survey, target=target, country=country, arm=arm):
            responses = json.loads(gp.read_text(encoding="utf-8"))["responses"]
            rec = {
                "survey": survey, "target": target, "country": country,
                "extractor_model": EXTRACTOR_MODEL, "prompt_arm": arm, "features": {},
            }
            for cond in CONDITIONS:
                feats, _raw = extract_features(responses.get(cond, ""), extract_fn)
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
                "cell_map", arm=arm, survey=survey, target=target,
                country=country, condition=cond,
            ):
                cm = map_features_with_subitems(
                    f"{target}_{country}", "C_free", feat_by_cond.get(cond, []),
                    emb, vcodes, svars, embed, disambig_fn, mapper_model=dmodel,
                    excluded_codes=excluded, pipe_types=PIPE_TYPES, workers=n_map,
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
            with timing.span("cell_pipeline", survey=survey, target=target, country=country, arm=arm):
                g = _gen_one(survey, target, country)
                e = _extract_one(survey, target, country)
                if e == "missing_gen":
                    with print_lock:
                        print(f"  ! [{i+1}/{len(cells)}] {ctag}: missing gen after gen step")
                    _bump("errors")
                    return
                m = _map_one(survey, target, country)
                if m == "missing_extract":
                    with print_lock:
                        print(f"  ! [{i+1}/{len(cells)}] {ctag}: missing extract after extract step")
                    _bump("errors")
                    return
            with print_lock:
                print(
                    f"  [{i+1}/{len(cells)}] {survey} {target} {country}: "
                    f"gen={g} extract={e} map={m}"
                )
        except Exception as exc:
            _bump("errors")
            with print_lock:
                print(
                    f"  ! [{i+1}/{len(cells)}] {ctag}: "
                    f"{type(exc).__name__}: {str(exc)[:120]}"
                )

    with timing.span(
        "phase_pipeline",
        selector=selector_key,
        prompt_arm=arm,
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
        f"[prompt_sens] done. gen wrote/skip={counts['gen_wrote']}/{counts['gen_skip']} "
        f"extract={counts['ext_wrote']}/{counts['ext_skip']} "
        f"map_files={counts['map_wrote']}/{counts['map_skip']} "
        f"errors={counts['errors']} -> {map_dir}"
    )
    timing.print_summary()
    usage.print_summary()

    if with_score:
        phase_score_arm(
            selector_key, arm, cells,
            disambig_key=disambig_key,
            force=force,
            score_workers=score_workers,
            score_xgb_nthread=score_xgb_nthread,
        )


def phase_score_arm(
    selector_key: str,
    arm: str,
    cells: list[tuple[str, str, str]],
    *,
    disambig_key: str = DEFAULT_DISAMBIG,
    force: bool = False,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
) -> None:
    """Score maps for one (selector, arm) subsample only."""
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
    timing = TimingLog(default_timing_path("score", f"prompt_sens_{selector_key}_{arm}"))

    _, _, map_dir = prompt_sensitivity_dirs(selector_key, arm, OUT)
    out_csv = prompt_sensitivity_scores_path(selector_key, arm, OUT)
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
        f"[prompt_sens score] selector={selector_key} arm={arm} cells={len(cells)} "
        f"workers={workers} nthread={nthread} -> {out_csv}"
    )
    with timing.span(
        "phase_score",
        selector=selector_key,
        prompt_arm=arm,
        n_cells=len(cells),
        score_workers=workers,
    ):
        run_score_jobs(specs, out_csv, cols, workers=workers, log_prefix="[prompt_sens score]")
    timing.print_summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--selector",
        choices=sorted(EXPERIMENT_SELECTORS),
        default=None,
        help="experiment selector key (kimi, deepseek_v4, …)",
    )
    ap.add_argument(
        "--arm",
        choices=sorted(PROMPT_ARMS),
        default=None,
        help=f"system-prompt arm (default for single runs: {DEFAULT_PROMPT_ARM})",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="run full factorial: kimi + deepseek_v4 × all three arms",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="first 2 cells only (concurrency stress smoke)",
    )
    ap.add_argument(
        "--cells-file",
        type=Path,
        default=DEFAULT_CELLS,
        help=f"YAML cell list (default: {DEFAULT_CELLS})",
    )
    ap.add_argument("--disambiguator", choices=list(DISAMBIGUATORS), default=DEFAULT_DISAMBIG)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--pipeline-workers", type=int, default=None,
        help=f"default {DEFAULT_PIPELINE_WORKERS} (or PIPELINE_WORKERS env)",
    )
    ap.add_argument(
        "--map-workers", type=int, default=None,
        help=f"default {DEFAULT_MAP_WORKERS} (or MAP_WORKERS env)",
    )
    ap.add_argument("--no-score", action="store_true", help="skip score phase (default: score on)")
    ap.add_argument("--score-workers", type=int, default=None)
    ap.add_argument("--score-xgb-nthread", type=int, default=None)
    args = ap.parse_args()

    from survey_features.timing import resolve_workers

    if args.all and (args.selector or args.arm):
        ap.error("--all cannot be combined with --selector / --arm")
    if not args.all and not args.selector:
        ap.error("pass --selector KEY or --all")
    if not args.all and args.arm is None:
        args.arm = DEFAULT_PROMPT_ARM

    cells = load_cells(args.cells_file)
    if args.smoke:
        cells = cells[:2]

    pipeline_workers = resolve_workers(
        args.pipeline_workers, "PIPELINE_WORKERS", default=DEFAULT_PIPELINE_WORKERS,
    )
    map_workers = resolve_workers(
        args.map_workers, "MAP_WORKERS", default=DEFAULT_MAP_WORKERS,
    )
    with_score = not args.no_score

    jobs: list[tuple[str, str]] = []
    if args.all:
        for sel in ("kimi", "deepseek_v4"):
            for arm in ("social_scientist", "none", "helpful"):
                jobs.append((sel, arm))
    else:
        jobs.append((args.selector, args.arm))

    # Ensure registry root exists before first write.
    prompt_sensitivity_root(OUT).mkdir(parents=True, exist_ok=True)

    for sel, arm in jobs:
        phase_pipeline_arm(
            sel, arm, cells,
            disambig_key=args.disambiguator,
            force=args.force,
            pipeline_workers=pipeline_workers,
            map_workers=map_workers,
            with_score=with_score,
            score_workers=args.score_workers,
            score_xgb_nthread=args.score_xgb_nthread,
        )


if __name__ == "__main__":
    main()
