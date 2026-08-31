"""
Prompt-sensitivity v2: prompt packs × selectors on the frozen 24×3 grid.

Writes under outputs/experiments/prompt_sensitivity_v2/<selector>/<pack>[/rN|/tN]/ so
confirmatory selectors/<selector>/ trees are never clobbered. Stage 1 is
country_provided only (the oracle is per-country; unprompted is a different estimand).
Temperature sidecars t1/t2 are scientist_respondent at 1.0 and are not in --all.

  python scripts/run_prompt_sensitivity_v2.py --selector kimi --pack scientist_respondent --replicate 1 --smoke
  python scripts/run_prompt_sensitivity_v2.py --selector hermes --pack none_respondent --no-score
  python scripts/run_prompt_sensitivity_v2.py --all   # 4 selectors × 4 Stage-1 draws (costly)
  python scripts/run_prompt_sensitivity_v2.py --selector kimi --temperature-runs --no-score

Hermes calls pass extra_body to turn thinking off. Resume skips only complete
(non-empty) gen files unless --force; empty provider failures are retried.
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
    DEFAULT_EMBEDDING_MODEL,
    DISAMBIGUATORS,
    EXPERIMENT_SELECTORS,
    EXTRACTOR_MODEL,
    OUTPUTS_DIR,
    PIPE_TYPES,
    ROOT as PKG_ROOT,
)
from survey_features.layout import (  # noqa: E402
    cell_dir,
    cell_tag,
    prompt_sensitivity_v2_dirs,
    prompt_sensitivity_v2_root,
    prompt_sensitivity_v2_scores_path,
)
from survey_features.oracle import ORACLE_CONTRACT_VERSION  # noqa: E402
from survey_features.prompts import (  # noqa: E402
    PROMPT_ARMS,
    PROMPT_PACKS,
    PROMPT_SENSITIVITY_V2_CONDITION,
    PROMPT_SENSITIVITY_V2_SELECTORS,
    PROMPT_SENSITIVITY_V2_STAGE1_TEMPERATURE,
    PROMPT_SENSITIVITY_V2_TEMPERATURE_RUNS_TEMPERATURE,
    prompt_sensitivity_v2_runs,
    prompt_sensitivity_v2_temperature_draws,
)

OUT = OUTPUTS_DIR
DEFAULT_CELLS = PKG_ROOT / "data" / "prompt_sensitivity_v2_cells.yaml"
DEFAULT_DISAMBIG = "nemotron"
DEFAULT_PIPELINE_WORKERS = 4
DEFAULT_MAP_WORKERS = 8
STAGE1_CONDITION = PROMPT_SENSITIVITY_V2_CONDITION
GEN_MAX_TOKENS = 8192

# Hermes-4: disable the thinking path if the host honours chat_template_kwargs.
HERMES_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


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


def has_v4_oracle(target: str, country: str, outputs_dir: Path = OUT) -> bool:
    meta = cell_dir(target, country, outputs_dir) / "oracle_meta.json"
    if not meta.is_file():
        return False
    try:
        rec = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"unreadable oracle_meta.json for {target} x {country} at {meta}: {exc}"
        ) from exc
    return rec.get("contract_version") == ORACLE_CONTRACT_VERSION


def gen_record_complete(rec: dict, condition: str = STAGE1_CONDITION) -> bool:
    """True if this gen JSON is a usable essay, not an empty/error placeholder."""
    if rec.get("errors", {}).get(condition):
        return False
    resp = (rec.get("responses") or {}).get(condition) or ""
    return bool(str(resp).strip())


def filter_v4_cells(
    cells: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], int]:
    kept = [c for c in cells if has_v4_oracle(c[1], c[2])]
    return kept, len(cells) - len(kept)


def _run_label(
    pack: str,
    replicate: int | None,
    temperature_draw: int | None = None,
) -> str:
    if temperature_draw is not None:
        return f"{pack}/t{temperature_draw}"
    return f"{pack}/r{replicate}" if replicate is not None else pack


def _gen_temperature(temperature_draw: int | None) -> float:
    if temperature_draw is not None:
        return PROMPT_SENSITIVITY_V2_TEMPERATURE_RUNS_TEMPERATURE
    return PROMPT_SENSITIVITY_V2_STAGE1_TEMPERATURE


def _run_kind(replicate: int | None, temperature_draw: int | None) -> str:
    if temperature_draw is not None:
        return "temperature"
    if replicate is not None:
        return "replicate"
    return "pack"


def _selector_extra_body(selector_key: str) -> dict | None:
    if selector_key == "hermes":
        return dict(HERMES_EXTRA_BODY)
    return None


def phase_pipeline(
    selector_key: str,
    pack: str,
    cells: list[tuple[str, str, str]],
    *,
    replicate: int | None = None,
    temperature_draw: int | None = None,
    disambig_key: str = DEFAULT_DISAMBIG,
    force: bool = False,
    pipeline_workers: int = DEFAULT_PIPELINE_WORKERS,
    map_workers: int = DEFAULT_MAP_WORKERS,
    with_score: bool = False,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
) -> None:
    """gen → extract → map for one (selector, pack[, rN|tN]); optional score."""
    from survey_features.elicitation import freetext_messages, resolve_prompt_pack
    from survey_features.extraction import extract_features
    from survey_features.llm import TokenUsageLog, default_usage_path
    from survey_features.mapping import map_features_with_subitems
    from survey_features.retrieval import make_embed_fn, target_excluded_codes
    from survey_features.timing import TimingLog, default_timing_path

    if pack not in PROMPT_PACKS:
        raise ValueError(f"Unknown pack {pack!r}; choose from {sorted(PROMPT_PACKS)}")
    if temperature_draw is not None:
        if replicate is not None:
            raise ValueError("replicate and temperature_draw are mutually exclusive")
        if pack != "scientist_respondent":
            raise ValueError("temperature draws are only valid for scientist_respondent")

    sys_arm, _referent = resolve_prompt_pack(pack)
    sel_model = resolve_selector_model(selector_key)
    emb_model = DEFAULT_EMBEDDING_MODEL
    dmodel = DISAMBIGUATORS[disambig_key]
    gen_temperature = _gen_temperature(temperature_draw)
    gen_dir, extract_dir, map_dir = prompt_sensitivity_v2_dirs(
        selector_key, pack, replicate, OUT, temperature_draw=temperature_draw,
    )
    for d in (gen_dir, extract_dir, map_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_pipe = max(1, int(pipeline_workers))
    n_map = max(1, int(map_workers))
    run_tag = _run_label(pack, replicate, temperature_draw)
    tag = f"prompt_sens_v2_{selector_key}_{run_tag.replace('/', '_')}_{disambig_key}"
    timing = TimingLog(default_timing_path("pipeline", tag))
    usage = TokenUsageLog(default_usage_path("pipeline", tag))

    gen_fn = main_run.selector_generate_fn(
        sel_model, usage_log=usage, extra_body=_selector_extra_body(selector_key)
    )
    extract_fn = main_run.mapper_generate_fn(EXTRACTOR_MODEL, usage_log=usage)
    disambig_fn = main_run.mapper_generate_fn(dmodel, usage_log=usage)
    _raw_embed = make_embed_fn(emb_model)
    _embed_lock = threading.Lock()

    def embed(texts):
        with _embed_lock:
            return _raw_embed(texts)

    print(
        f"[prompt_sens_v2] selector={selector_key} ({sel_model}) pack={run_tag} "
        f"cond={STAGE1_CONDITION} temperature={gen_temperature} "
        f"run_kind={_run_kind(replicate, temperature_draw)} "
        f"disambig={disambig_key} cells={len(cells)} "
        f"pipeline_workers={n_pipe} map_workers={n_map} with_score={with_score}"
    )
    print(f"  -> {gen_dir.parent}")

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
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            if gen_record_complete(existing):
                _bump("gen_skip")
                return "skipped"
        svars, _, _ = main_run.survey_assets(survey)
        qtext = svars.get(target, target)
        extra_body = _selector_extra_body(selector_key)
        with timing.span(
            "cell_gen", survey=survey, target=target, country=country,
            pack=pack, replicate=replicate, temperature_draw=temperature_draw,
            temperature=gen_temperature,
        ):
            rec = {
                "survey": survey, "target": target, "country": country,
                "question_text": qtext, "selector_model": sel_model,
                "prompt_pack": pack, "prompt_arm": sys_arm,
                "system_prompt": PROMPT_ARMS[sys_arm],
                "replicate": replicate,
                "temperature_draw": temperature_draw,
                "temperature": gen_temperature,
                "run_kind": _run_kind(replicate, temperature_draw),
                "condition": STAGE1_CONDITION,
                "max_tokens": GEN_MAX_TOKENS,
                "extra_body": extra_body,
                "responses": {},
            }
            messages = freetext_messages(
                qtext, country, prompt_pack=pack,
            )
            resp = gen_fn(
                messages,
                max_tokens=GEN_MAX_TOKENS,
                temperature=gen_temperature,
                usage_phase="feature_list",
            )
            if not resp:
                rec.setdefault("errors", {})[STAGE1_CONDITION] = (
                    "empty response after retries"
                )
                _bump("errors")
            rec["responses"][STAGE1_CONDITION] = resp
            out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        _bump("gen_wrote")
        return "wrote"

    def _extract_one(survey, target, country, *, redo: bool = False) -> str:
        ctag = cell_tag(survey, target, country)
        op = extract_dir / f"{ctag}.json"
        if op.is_file() and not force and not redo:
            _bump("ext_skip")
            return "skipped"
        gp = gen_dir / f"{ctag}.json"
        if not gp.is_file():
            return "missing_gen"
        with timing.span(
            "cell_extract", survey=survey, target=target, country=country,
            pack=pack, replicate=replicate,
        ):
            responses = json.loads(gp.read_text(encoding="utf-8"))["responses"]
            rec = {
                "survey": survey, "target": target, "country": country,
                "extractor_model": EXTRACTOR_MODEL, "prompt_pack": pack,
                "replicate": replicate, "temperature_draw": temperature_draw,
                "run_kind": _run_kind(replicate, temperature_draw),
                "features": {},
            }
            feats, _raw = extract_features(
                responses.get(STAGE1_CONDITION, ""), extract_fn
            )
            rec["features"][STAGE1_CONDITION] = feats
            op.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        _bump("ext_wrote")
        return "wrote"

    def _map_one(survey, target, country, *, redo: bool = False) -> str:
        ctag = cell_tag(survey, target, country)
        ep = extract_dir / f"{ctag}.json"
        if not ep.is_file():
            return "missing_extract"
        op = map_dir / f"C__{disambig_key}__{ctag}__{STAGE1_CONDITION}.json"
        if op.is_file() and not force and not redo:
            _bump("map_skip")
            return "skipped"
        svars, emb, vcodes = main_run.survey_assets(survey, emb_model)
        excluded = target_excluded_codes(target, svars, emb, vcodes, embed)
        feat_by_cond = json.loads(ep.read_text(encoding="utf-8"))["features"]
        with timing.span(
            "cell_map", pack=pack, replicate=replicate, survey=survey,
            target=target, country=country, condition=STAGE1_CONDITION,
        ):
            cm = map_features_with_subitems(
                f"{target}_{country}", "C_free",
                feat_by_cond.get(STAGE1_CONDITION, []),
                emb, vcodes, svars, embed, disambig_fn, mapper_model=dmodel,
                excluded_codes=excluded, pipe_types=PIPE_TYPES, workers=n_map,
            )
            main_run._save_map(
                op, survey, target, country, STAGE1_CONDITION, disambig_key,
                cm, emb_model,
            )
        _bump("map_wrote")
        return "wrote"

    def _cell_chain(idx_cell):
        i, (survey, target, country) = idx_cell
        ctag = cell_tag(survey, target, country)
        try:
            with timing.span(
                "cell_pipeline", survey=survey, target=target, country=country,
                pack=pack, replicate=replicate,
            ):
                g = _gen_one(survey, target, country)
                e = _extract_one(survey, target, country, redo=(g == "wrote"))
                if e == "missing_gen":
                    with print_lock:
                        print(f"  ! [{i+1}/{len(cells)}] {ctag}: missing gen after gen step")
                    _bump("errors")
                    return
                m = _map_one(survey, target, country, redo=(e == "wrote"))
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
        prompt_pack=pack,
        replicate=replicate,
        temperature_draw=temperature_draw,
        temperature=gen_temperature,
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
        f"[prompt_sens_v2] done. gen wrote/skip={counts['gen_wrote']}/{counts['gen_skip']} "
        f"extract={counts['ext_wrote']}/{counts['ext_skip']} "
        f"map_files={counts['map_wrote']}/{counts['map_skip']} "
        f"errors={counts['errors']} -> {map_dir}"
    )
    timing.print_summary()
    usage.print_summary()

    if with_score:
        phase_score(
            selector_key, pack, cells,
            replicate=replicate,
            temperature_draw=temperature_draw,
            disambig_key=disambig_key,
            force=force,
            score_workers=score_workers,
            score_xgb_nthread=score_xgb_nthread,
        )


def phase_score(
    selector_key: str,
    pack: str,
    cells: list[tuple[str, str, str]],
    *,
    replicate: int | None = None,
    temperature_draw: int | None = None,
    disambig_key: str = DEFAULT_DISAMBIG,
    force: bool = False,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
) -> None:
    """Score maps for one (selector, pack[, rN|tN]). Cells without a v4 oracle are skipped."""
    from survey_features.score_cell import (
        resolve_score_n_draws,
        resolve_score_workers,
        resolve_score_xgb_nthread,
        run_score_jobs,
        score_cols,
    )
    from survey_features.timing import TimingLog, default_timing_path

    scoreable, n_skip = filter_v4_cells(cells)
    if n_skip:
        print(f"[prompt_sens_v2 score] skipping {n_skip} cells without contract-v4 oracles")
    if not scoreable:
        print("[prompt_sens_v2 score] nothing to score (no v4 oracles on this cell list)")
        return

    n_draws = resolve_score_n_draws()
    workers = resolve_score_workers(score_workers)
    nthread = resolve_score_xgb_nthread(workers, score_xgb_nthread)
    run_tag = _run_label(pack, replicate, temperature_draw)
    timing = TimingLog(
        default_timing_path("score", f"prompt_sens_v2_{selector_key}_{run_tag.replace('/', '_')}")
    )

    _, _, map_dir = prompt_sensitivity_v2_dirs(
        selector_key, pack, replicate, OUT, temperature_draw=temperature_draw,
    )
    out_csv = prompt_sensitivity_v2_scores_path(
        selector_key, pack, replicate, OUT, disambiguator=disambig_key,
        temperature_draw=temperature_draw,
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    emb_label = DEFAULT_EMBEDDING_MODEL
    cols = score_cols()
    grid_cells = [list(c) for c in scoreable]

    specs = []
    for survey, target, country in scoreable:
        codes = main_run._map_codes(
            map_dir, disambig_key, survey, target, country, STAGE1_CONDITION
        )
        evals = []
        if codes is not None:
            evals.append({
                "condition": STAGE1_CONDITION, "arm": "C",
                "disambiguator": disambig_key,
                "embedding_model": emb_label, "codes": codes,
            })
        specs.append({
            "survey": survey, "target": target, "country": country,
            "evals": evals,
            "n_draws": n_draws, "nthread": nthread, "fixed_ks": list(main_run.FIXED_KS),
            "outputs_dir": str(OUT),
            "grid_cells": grid_cells,
        })

    print(
        f"[prompt_sens_v2 score] selector={selector_key} pack={run_tag} "
        f"cells={len(scoreable)} workers={workers} nthread={nthread} -> {out_csv}"
    )
    with timing.span(
        "phase_score",
        selector=selector_key,
        prompt_pack=pack,
        replicate=replicate,
        temperature_draw=temperature_draw,
        n_cells=len(scoreable),
        score_workers=workers,
    ):
        run_score_jobs(
            specs, out_csv, cols, workers=workers,
            log_prefix="[prompt_sens_v2 score]",
            resume=not force,
        )
    timing.print_summary()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--selector",
        choices=list(PROMPT_SENSITIVITY_V2_SELECTORS),
        default=None,
        help="deepseek_v4, kimi, minimax, or hermes",
    )
    ap.add_argument(
        "--pack",
        choices=sorted(PROMPT_PACKS),
        default=None,
        help="prompt pack. Omit (with --selector) to run all four Stage-1 draws; "
             f"pass this flag to run one pack (scientist_respondent still needs --replicate or runs r1 and r2).",
    )
    ap.add_argument(
        "--replicate",
        type=int,
        choices=(1, 2),
        default=None,
        help="scientist_respondent greedy draw (1 or 2). Omit to run both.",
    )
    ap.add_argument(
        "--temperature-draw",
        type=int,
        choices=(1, 2),
        default=None,
        help=(
            "scientist_respondent sampling draw t1 or t2 at "
            f"{PROMPT_SENSITIVITY_V2_TEMPERATURE_RUNS_TEMPERATURE}. "
            "Sidecar, not a Stage-1 lock job. Mutually exclusive with --replicate / --all."
        ),
    )
    ap.add_argument(
        "--temperature-runs",
        action="store_true",
        help=(
            "run both t1 and t2 (scientist_respondent at "
            f"{PROMPT_SENSITIVITY_V2_TEMPERATURE_RUNS_TEMPERATURE}). "
            "Not included in --all."
        ),
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="4 selectors × scientist_respondent r1/r2 + analyst_person + none_respondent (not t1/t2)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="first 2 cells only",
    )
    ap.add_argument(
        "--v4-only",
        action="store_true",
        help="drop cells whose oracle_meta.json is not contract v4 (safe while retries run)",
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

    if args.temperature_draw and args.temperature_runs:
        ap.error("use --temperature-draw N or --temperature-runs, not both")
    temp_mode = bool(args.temperature_draw or args.temperature_runs)
    if args.all and (args.selector or args.pack or args.replicate or temp_mode):
        ap.error(
            "--all cannot be combined with --selector / --pack / --replicate / "
            "--temperature-draw / --temperature-runs"
        )
    if not args.all and not args.selector:
        ap.error("pass --selector KEY or --all")
    if temp_mode:
        if args.replicate is not None:
            ap.error("temperature runs cannot be combined with --replicate")
        if args.pack not in (None, "scientist_respondent"):
            ap.error("temperature runs are scientist_respondent only")
        try:
            t_draws = prompt_sensitivity_v2_temperature_draws(args.temperature_draw)
        except ValueError as exc:
            ap.error(str(exc))
        jobs = [
            (args.selector, "scientist_respondent", None, t_draw) for t_draw in t_draws
        ]
    else:
        try:
            runs = prompt_sensitivity_v2_runs(args.pack, args.replicate)
        except ValueError as exc:
            ap.error(str(exc))
        selectors = list(PROMPT_SENSITIVITY_V2_SELECTORS) if args.all else [args.selector]
        jobs = [(sel, pack, repl, None) for sel in selectors for pack, repl in runs]

    cells = load_cells(args.cells_file)
    if args.smoke:
        cells = cells[:2]
    if args.v4_only:
        cells, n_drop = filter_v4_cells(cells)
        print(f"[prompt_sens_v2] --v4-only: dropped {n_drop} cells without current oracles")
        if not cells:
            raise SystemExit("no contract-v4 cells on this grid")

    pipeline_workers = resolve_workers(
        args.pipeline_workers, "PIPELINE_WORKERS", default=DEFAULT_PIPELINE_WORKERS,
    )
    map_workers = resolve_workers(
        args.map_workers, "MAP_WORKERS", default=DEFAULT_MAP_WORKERS,
    )
    with_score = not args.no_score

    prompt_sensitivity_v2_root(OUT).mkdir(parents=True, exist_ok=True)

    for sel, pack, repl, t_draw in jobs:
        phase_pipeline(
            sel, pack, cells,
            replicate=repl,
            temperature_draw=t_draw,
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
