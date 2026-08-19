"""
Main free-text pipeline — confirmatory Arm C loop.

  --phase gen       free-text selection essays (both prompt conditions)
  --phase extract   essay -> typed feature list (FIXED extractor)
  --phase map       dual-layer retrieve + disambiguate -> expanded_codes
  --phase score     XGB vs oracle / random / textbook -> scores_<selector>.csv
  --phase pipeline  gen -> extract -> map per cell (optional --with-score)

Grid = genuine cells from scripts/leakage_audit.py (type-1 + leakage screen;
accuracy-vs-majority is not a drop), unless --cells CSV
(survey,target,country) supplies an explicit grid (pilot / frame-sampled runs).
Use --run-tag to write map/score under main/runs/<tag>/ without clobbering baseline.
Gen/extract always stay under main/<selector>/.

Examples:
  python scripts/run_main.py --phase gen      --selector deepseek
  python scripts/run_main.py --phase extract  --selector deepseek
  python scripts/run_main.py --phase map      --selector deepseek --disambiguator nemotron
  python scripts/run_main.py --phase score    --selector deepseek
  python scripts/run_main.py --phase pipeline --selector deepseek --disambiguator nemotron \\
      --pipeline-workers 4 --map-workers 8 --with-score
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survey_features.config import (  # noqa: E402
    CONDITIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SELECTOR,
    DISAMBIGUATORS,
    EXTRACTOR_MODEL,
    OUTPUTS_DIR,
    PIPE_TYPES,
    SELECTORS,
)
from survey_features.layout import (  # noqa: E402
    cell_tag,
    genuine_cells,
    main_dir,
    main_scores_path,
    selector_dirs,
)

OUT = OUTPUTS_DIR
FIXED_KS = [5, 10]

# --cells override: module-level so every phase resolves the same grid without
# threading one more parameter through five signatures.
_CELLS_CSV: Path | None = None


def grid_cells(outputs_dir: Path = OUT) -> list[tuple[str, str, str]]:
    """The run's (survey, target, country) grid.

    Default = genuine cells from the leakage audit (type-1 + leakage only).
    --cells CSV (columns survey,target,country) overrides it — pilot and
    frame-sampled grids are defined by an explicit cell list, not by the audit.
    """
    if _CELLS_CSV is None:
        return genuine_cells(outputs_dir)
    import csv as _csv

    with open(_CELLS_CSV, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    required = {"survey", "target", "country"}
    if rows and not required.issubset(rows[0]):
        raise SystemExit(f"--cells CSV needs columns {sorted(required)}")
    return [(r["survey"], r["target"], r["country"]) for r in rows]


def selector_generate_fn(selector_model: str, usage_log=None):
    """Selector client; transient errors -> '' (keeps sweeps alive)."""
    from survey_features.llm import make_generate_fn
    fn, _ = make_generate_fn(model=selector_model, usage_log=usage_log, on_error="empty")
    return fn


def mapper_generate_fn(mapper_model: str, usage_log=None):
    """Extractor/disambiguator client (DISAMBIG_* endpoint, else LLM endpoint)."""
    from survey_features.llm import make_generate_fn
    fn, _ = make_generate_fn(
        base_url=os.environ.get("DISAMBIG_BASE_URL") or None,
        api_key=os.environ.get("DISAMBIG_API_KEY") or None,
        model=mapper_model,
        usage_log=usage_log,
        on_error="empty",
    )
    return fn


_survey_cache: dict = {}
_survey_cache_lock = threading.Lock()


def survey_assets(survey_id, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    """(survey_variables, embeddings, var_codes) — cached per (survey, embedding model)."""
    key = (survey_id, embedding_model)
    with _survey_cache_lock:
        if key not in _survey_cache:
            from survey_features.retrieval import load_or_build_survey_embeddings
            from survey_features.surveys import extract_survey_variables, load_survey
            _, meta = load_survey(survey_id, os.environ["DATA_CONFIG_PATH"])
            svars = extract_survey_variables(meta)
            emb, vcodes = load_or_build_survey_embeddings(svars, survey_id, embedding_model)
            _survey_cache[key] = (svars, emb, vcodes)
        return _survey_cache[key]


def phase_gen(selector_key, force=False, limit=None, api_workers: int = 1):
    from survey_features.elicitation import freetext_messages
    from survey_features.llm import TokenUsageLog, default_usage_path
    from survey_features.timing import TimingLog, default_timing_path

    sel_model = SELECTORS[selector_key]["model"]
    gen_dir, _, _ = selector_dirs(selector_key)
    gen_dir.mkdir(parents=True, exist_ok=True)
    usage = TokenUsageLog(default_usage_path("gen", selector_key))
    gen = selector_generate_fn(sel_model, usage_log=usage)
    cells = grid_cells(OUT)
    if limit:
        cells = cells[:limit]
    n_workers = max(1, int(api_workers))
    timing = TimingLog(default_timing_path("gen", selector_key))
    print(
        f"[gen] selector={selector_key} ({sel_model}) {len(cells)} cells x "
        f"{len(CONDITIONS)} conds  api_workers={n_workers}"
    )
    done = skipped = 0
    done_lock = threading.Lock()

    def _one(cell_tuple):
        nonlocal done, skipped
        survey, target, country = cell_tuple
        svars, _, _ = survey_assets(survey)
        qtext = svars.get(target, target)
        out_path = gen_dir / f"{cell_tag(survey, target, country)}.json"
        if out_path.is_file() and not force:
            with done_lock:
                skipped += 1
            return None
        with timing.span("cell_gen", survey=survey, target=target, country=country):
            rec = {
                "survey": survey, "target": target, "country": country,
                "question_text": qtext, "selector_model": sel_model, "responses": {},
            }
            for cond in CONDITIONS:
                messages = freetext_messages(
                    qtext, country if cond == "country_provided" else None,
                )
                resp = gen(messages, max_tokens=4096, usage_phase="feature_list")
                if not resp:
                    rec.setdefault("errors", {})[cond] = "empty response after retries"
                rec["responses"][cond] = resp
            out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        with done_lock:
            done += 1
        return (
            survey, target, country,
            len(rec["responses"].get("unprompted", "")),
            len(rec["responses"].get("country_provided", "")),
        )

    with timing.span("phase_gen", selector=selector_key, n_cells=len(cells), api_workers=n_workers):
        if n_workers <= 1:
            for i, cell in enumerate(cells):
                r = _one(cell)
                if r:
                    survey, target, country, up, cp = r
                    print(f"  [{i+1}/{len(cells)}] {survey} {target} {country}: up={up}c cp={cp}c")
        else:
            for survey, _, _ in cells:
                survey_assets(survey)
            with ThreadPoolExecutor(max_workers=min(n_workers, len(cells) or 1)) as ex:
                futs = {ex.submit(_one, c): i for i, c in enumerate(cells)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    r = fut.result()
                    if r:
                        survey, target, country, up, cp = r
                        print(f"  [{i+1}/{len(cells)}] {survey} {target} {country}: up={up}c cp={cp}c")

    print(f"[gen] done. wrote={done} skipped(existing)={skipped} -> {gen_dir}")
    timing.print_summary()
    usage.print_summary()


def phase_extract(selector_key, force=False, limit=None, api_workers: int = 1):
    """Essay -> typed feature list via the FIXED extractor."""
    from survey_features.extraction import extract_features
    from survey_features.llm import TokenUsageLog, default_usage_path
    from survey_features.timing import TimingLog, default_timing_path

    usage = TokenUsageLog(default_usage_path("extract", selector_key))
    egen = mapper_generate_fn(EXTRACTOR_MODEL, usage_log=usage)
    gen_dir, extract_dir, _ = selector_dirs(selector_key)
    cells = grid_cells(OUT)
    if limit:
        cells = cells[:limit]
    extract_dir.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, int(api_workers))
    timing = TimingLog(default_timing_path("extract", selector_key))
    print(
        f"[extract] selector={selector_key} {len(cells)} cells x {len(CONDITIONS)} conds "
        f"via {EXTRACTOR_MODEL}  api_workers={n_workers}"
    )
    done = skipped = 0
    done_lock = threading.Lock()

    def _one(cell_tuple):
        nonlocal done, skipped
        survey, target, country = cell_tuple
        ctag = cell_tag(survey, target, country)
        op = extract_dir / f"{ctag}.json"
        if op.is_file() and not force:
            with done_lock:
                skipped += 1
            return None
        gp = gen_dir / f"{ctag}.json"
        if not gp.is_file():
            print(f"  ! missing free-text for {ctag}; run --phase gen first")
            return None
        with timing.span("cell_extract", survey=survey, target=target, country=country):
            responses = json.loads(gp.read_text(encoding="utf-8"))["responses"]
            rec = {
                "survey": survey, "target": target, "country": country,
                "extractor_model": EXTRACTOR_MODEL, "features": {},
            }
            for cond in CONDITIONS:
                feats, _raw = extract_features(responses.get(cond, ""), egen)
                rec["features"][cond] = feats
            op.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        with done_lock:
            done += 1
        return (
            survey, target, country,
            len(rec["features"].get("unprompted", [])),
            len(rec["features"].get("country_provided", [])),
        )

    with timing.span("phase_extract", selector=selector_key, n_cells=len(cells), api_workers=n_workers):
        if n_workers <= 1:
            for i, cell in enumerate(cells):
                r = _one(cell)
                if r:
                    survey, target, country, nfu, nfc = r
                    print(f"  [{i+1}/{len(cells)}] {survey} {target} {country}: up={nfu} cp={nfc} features")
        else:
            with ThreadPoolExecutor(max_workers=min(n_workers, len(cells) or 1)) as ex:
                futs = {ex.submit(_one, c): i for i, c in enumerate(cells)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    r = fut.result()
                    if r:
                        survey, target, country, nfu, nfc = r
                        print(f"  [{i+1}/{len(cells)}] {survey} {target} {country}: up={nfu} cp={nfc} features")

    print(f"[extract] done. wrote={done} skipped={skipped} -> {extract_dir}")
    timing.print_summary()
    usage.print_summary()


def _save_map(path, survey, target, country, cond, disambig_key, cm, embedding_model):
    from survey_features.mapping import expanded_cell_to_record

    rec = expanded_cell_to_record(
        survey, target, country, cond, "C", disambig_key, cm, embedding_model,
    )
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")


def phase_map(
    selector_key,
    disambig_key,
    force=False,
    limit=None,
    run_tag: str | None = None,
    map_workers: int = 1,
):
    """Dual-layer map for Arm C; writes C__<disambig>__<cell>__<cond>.json."""
    from survey_features.llm import TokenUsageLog, default_usage_path
    from survey_features.mapping import map_features_with_subitems
    from survey_features.retrieval import make_embed_fn, target_excluded_codes
    from survey_features.timing import TimingLog, default_timing_path

    emb_model = DEFAULT_EMBEDDING_MODEL
    dmodel = DISAMBIGUATORS[disambig_key]
    usage = TokenUsageLog(default_usage_path("map", f"{selector_key}_{disambig_key}"))
    dgen = mapper_generate_fn(dmodel, usage_log=usage)
    embed = make_embed_fn(emb_model)
    _, extract_dir, map_dir = selector_dirs(selector_key, run_tag=run_tag)
    cells = grid_cells(OUT)
    if limit:
        cells = cells[:limit]
    map_dir.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, int(map_workers))
    timing = TimingLog(default_timing_path("map", f"{selector_key}_{disambig_key}"))
    print(
        f"[map] selector={selector_key} disambiguator={disambig_key} "
        f"cells={len(cells)} embedding={emb_model} map_workers={n_workers}"
        + (f" -> {map_dir}" if run_tag else "")
    )

    with timing.span(
        "phase_map",
        selector=selector_key,
        disambiguator=disambig_key,
        n_cells=len(cells),
        map_workers=n_workers,
    ):
        for i, (survey, target, country) in enumerate(cells):
            svars, emb, vcodes = survey_assets(survey, emb_model)
            excluded = target_excluded_codes(target, svars, emb, vcodes, embed)
            ctag = cell_tag(survey, target, country)
            ep = extract_dir / f"{ctag}.json"
            feat_by_cond = (
                json.loads(ep.read_text(encoding="utf-8"))["features"] if ep.is_file() else None
            )
            for cond in CONDITIONS:
                op = map_dir / f"C__{disambig_key}__{ctag}__{cond}.json"
                if op.is_file() and not force:
                    continue
                if feat_by_cond is None:
                    print(f"  ! missing extraction for {ctag}; run --phase extract first")
                    break
                with timing.span(
                    "cell_map", arm="C", survey=survey, target=target,
                    country=country, condition=cond,
                ):
                    cm = map_features_with_subitems(
                        f"{target}_{country}", "C_free", feat_by_cond.get(cond, []),
                        emb, vcodes, svars, embed, dgen, mapper_model=dmodel,
                        excluded_codes=excluded, pipe_types=PIPE_TYPES,
                        workers=n_workers,
                    )
                    _save_map(op, survey, target, country, cond, disambig_key, cm, emb_model)
            print(f"  [{i+1}/{len(cells)}] {survey} {target} {country} mapped ({disambig_key})")
    print(f"[map] done -> {map_dir}")
    timing.print_summary()
    usage.print_summary()


def phase_pipeline(
    selector_key,
    disambig_key,
    force=False,
    limit=None,
    run_tag: str | None = None,
    pipeline_workers: int = 1,
    map_workers: int = 1,
    with_score: bool = False,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
):
    """gen → extract → map per cell; optional score after all maps."""
    from survey_features.elicitation import freetext_messages
    from survey_features.extraction import extract_features
    from survey_features.llm import TokenUsageLog, default_usage_path
    from survey_features.mapping import map_features_with_subitems
    from survey_features.retrieval import make_embed_fn, target_excluded_codes
    from survey_features.timing import TimingLog, default_timing_path

    sel_model = SELECTORS[selector_key]["model"]
    emb_model = DEFAULT_EMBEDDING_MODEL
    dmodel = DISAMBIGUATORS[disambig_key]
    gen_dir, extract_dir, map_dir = selector_dirs(selector_key, run_tag=run_tag)
    # Gen/extract always under shared main/<selector>/; maps honor run_tag.
    gen_dir, extract_dir, _ = selector_dirs(selector_key, run_tag=None)
    _, _, map_dir = selector_dirs(selector_key, run_tag=run_tag)

    gen_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    cells = grid_cells(OUT)
    if limit:
        cells = cells[:limit]
    n_pipe = max(1, int(pipeline_workers))
    n_map = max(1, int(map_workers))
    timing = TimingLog(default_timing_path("pipeline", f"{selector_key}_{disambig_key}"))
    usage = TokenUsageLog(default_usage_path("pipeline", f"{selector_key}_{disambig_key}"))

    gen_fn = selector_generate_fn(sel_model, usage_log=usage)
    extract_fn = mapper_generate_fn(EXTRACTOR_MODEL, usage_log=usage)
    disambig_fn = mapper_generate_fn(dmodel, usage_log=usage)
    _raw_embed = make_embed_fn(emb_model)
    _embed_lock = threading.Lock()

    def embed(texts):
        with _embed_lock:
            return _raw_embed(texts)

    print(
        f"[pipeline] selector={selector_key} disambiguator={disambig_key} "
        f"cells={len(cells)} embedding={emb_model} "
        f"pipeline_workers={n_pipe} map_workers={n_map} with_score={with_score}"
        + (f" -> {map_dir}" if run_tag else "")
    )

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
        svars, _, _ = survey_assets(survey)
        qtext = svars.get(target, target)
        with timing.span("cell_gen", survey=survey, target=target, country=country):
            rec = {
                "survey": survey, "target": target, "country": country,
                "question_text": qtext, "selector_model": sel_model, "responses": {},
            }
            for cond in CONDITIONS:
                messages = freetext_messages(
                    qtext, country if cond == "country_provided" else None,
                )
                resp = gen_fn(messages, max_tokens=4096, usage_phase="feature_list")
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
        with timing.span("cell_extract", survey=survey, target=target, country=country):
            responses = json.loads(gp.read_text(encoding="utf-8"))["responses"]
            rec = {
                "survey": survey, "target": target, "country": country,
                "extractor_model": EXTRACTOR_MODEL, "features": {},
            }
            for cond in CONDITIONS:
                feats, _raw = extract_features(responses.get(cond, ""), extract_fn)
                rec["features"][cond] = feats
            op.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        _bump("ext_wrote")
        return "wrote"

    def _map_one(survey, target, country) -> str:
        svars, emb, vcodes = survey_assets(survey, emb_model)
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
                "cell_map", arm="C", survey=survey, target=target,
                country=country, condition=cond,
            ):
                cm = map_features_with_subitems(
                    f"{target}_{country}", "C_free", feat_by_cond.get(cond, []),
                    emb, vcodes, svars, embed, disambig_fn, mapper_model=dmodel,
                    excluded_codes=excluded, pipe_types=PIPE_TYPES, workers=n_map,
                )
                _save_map(op, survey, target, country, cond, disambig_key, cm, emb_model)
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
        disambiguator=disambig_key,
        n_cells=len(cells),
        pipeline_workers=n_pipe,
        map_workers=n_map,
    ):
        for survey, _, _ in cells:
            survey_assets(survey, emb_model)
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
        f"[pipeline] done. gen wrote/skip={counts['gen_wrote']}/{counts['gen_skip']} "
        f"extract={counts['ext_wrote']}/{counts['ext_skip']} "
        f"map_files={counts['map_wrote']}/{counts['map_skip']} "
        f"errors={counts['errors']} -> {map_dir}"
    )
    timing.print_summary()
    usage.print_summary()

    if with_score:
        phase_score(
            selector_key,
            force=force,
            limit=limit,
            score_workers=score_workers,
            score_xgb_nthread=score_xgb_nthread,
            run_tag=run_tag,
        )


def _map_codes(map_dir, disambig_key, survey, target, country, cond):
    """Codes from a map file: prefer expanded_codes, else mapped_codes."""
    ctag = cell_tag(survey, target, country)
    p = map_dir / f"C__{disambig_key}__{ctag}__{cond}.json"
    if not p.is_file():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    if "expanded_codes" in rec:
        return rec.get("expanded_codes") or []
    return rec.get("mapped_codes", [])


def phase_score(
    selector_key,
    force=False,
    limit=None,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
    run_tag: str | None = None,
):
    """Score Arm C maps via cell-level ProcessPool."""
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
    timing = TimingLog(default_timing_path("score", selector_key))

    _, _, map_dir = selector_dirs(selector_key, run_tag=run_tag)
    out_csv = main_scores_path(selector_key, OUT, run_tag=run_tag)
    if run_tag is None:
        out_csv = main_dir(OUT) / f"scores_{selector_key}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    emb_label = DEFAULT_EMBEDDING_MODEL

    cells = grid_cells(OUT)
    if limit:
        cells = cells[:limit]
    cols = score_cols()

    specs = []
    for survey, target, country in cells:
        evals = []
        for cond in CONDITIONS:
            for dk in DISAMBIGUATORS:
                codes = _map_codes(map_dir, dk, survey, target, country, cond)
                if codes is None:
                    continue
                evals.append({
                    "condition": cond, "arm": "C", "disambiguator": dk,
                    "embedding_model": emb_label, "codes": codes,
                })
        specs.append({
            "survey": survey, "target": target, "country": country,
            "evals": evals,
            "n_draws": n_draws, "nthread": nthread, "fixed_ks": list(FIXED_KS),
            "outputs_dir": str(OUT),
            # Explicit grid so scoring works for cells outside the leakage audit.
            "grid_cells": [list(c) for c in cells],
        })

    print(
        f"[score] selector={selector_key} embedding={emb_label} cells={len(cells)} "
        f"workers={workers} nthread={nthread} n_draws={n_draws} -> {out_csv}"
    )
    with timing.span(
        "phase_score",
        selector=selector_key,
        n_cells=len(cells),
        score_workers=workers,
        n_draws=n_draws,
    ):
        run_score_jobs(specs, out_csv, cols, workers=workers, log_prefix="[score]")
    timing.print_summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--phase",
        choices=["gen", "extract", "map", "score", "pipeline"],
        required=True,
    )
    ap.add_argument(
        "--selector", choices=list(SELECTORS), default=DEFAULT_SELECTOR,
        help=f"test model whose capability is measured (default: {DEFAULT_SELECTOR})",
    )
    ap.add_argument(
        "--disambiguator",
        choices=list(DISAMBIGUATORS),
        help="required for --phase map and --phase pipeline",
    )
    ap.add_argument(
        "--run-tag",
        default=None,
        metavar="TAG",
        help="Write map/score under main/runs/<TAG>/ (gen/extract stay shared)",
    )
    ap.add_argument(
        "--cells",
        type=Path,
        default=None,
        metavar="CSV",
        help="explicit grid CSV (survey,target,country); default = type-1+leakage genuine cells",
    )
    ap.add_argument("--force", action="store_true", help="recompute cells already on disk")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cells (smoke test)")
    ap.add_argument(
        "--api-workers", type=int, default=None,
        help="cell ThreadPool for gen/extract (default: API_WORKERS or 1)",
    )
    ap.add_argument(
        "--map-workers", type=int, default=None,
        help="per-unit disambig ThreadPool for map/pipeline (default: MAP_WORKERS or 1)",
    )
    ap.add_argument(
        "--pipeline-workers", type=int, default=None,
        help="cells in flight for pipeline (default: PIPELINE_WORKERS or 1)",
    )
    ap.add_argument("--with-score", action="store_true", help="after pipeline maps, run score")
    ap.add_argument(
        "--score-workers", type=int, default=None,
        help="cell ProcessPool for score (default: SCORE_WORKERS or min(8, cpus-2))",
    )
    ap.add_argument(
        "--score-xgb-nthread", type=int, default=None,
        help="XGBoost nthread per fit (default: SCORE_XGB_NTHREAD or cpus // workers)",
    )
    args = ap.parse_args()

    if args.cells is not None:
        if not args.cells.is_file():
            ap.error(f"--cells file not found: {args.cells}")
        global _CELLS_CSV
        _CELLS_CSV = args.cells

    from survey_features.timing import resolve_workers

    if args.run_tag and args.phase in ("gen", "extract"):
        ap.error("--run-tag only applies to map, score, and pipeline")
    if args.phase not in ("score", "pipeline") and (
        args.score_workers is not None or args.score_xgb_nthread is not None
    ):
        ap.error("--score-workers / --score-xgb-nthread only apply to score "
                 "(or pipeline with --with-score)")
    if args.phase == "pipeline" and (
        args.score_workers is not None or args.score_xgb_nthread is not None
    ) and not args.with_score:
        ap.error("--score-workers / --score-xgb-nthread require --with-score")
    if args.phase not in ("gen", "extract") and args.api_workers is not None:
        ap.error("--api-workers only applies to gen and extract")
    if args.phase not in ("map", "pipeline") and args.map_workers is not None:
        ap.error("--map-workers only applies to map and pipeline")
    if args.phase != "pipeline" and args.pipeline_workers is not None:
        ap.error("--pipeline-workers only applies to pipeline")
    if args.with_score and args.phase != "pipeline":
        ap.error("--with-score only applies to pipeline")

    api_workers = resolve_workers(args.api_workers, "API_WORKERS", default=1)
    map_workers = resolve_workers(args.map_workers, "MAP_WORKERS", default=1)
    pipeline_workers = resolve_workers(args.pipeline_workers, "PIPELINE_WORKERS", default=1)

    if args.phase == "gen":
        phase_gen(args.selector, force=args.force, limit=args.limit, api_workers=api_workers)
    elif args.phase == "extract":
        phase_extract(args.selector, force=args.force, limit=args.limit, api_workers=api_workers)
    elif args.phase == "score":
        phase_score(
            args.selector,
            force=args.force,
            limit=args.limit,
            score_workers=args.score_workers,
            score_xgb_nthread=args.score_xgb_nthread,
            run_tag=args.run_tag,
        )
    elif args.phase == "map":
        if not args.disambiguator:
            ap.error("--disambiguator required for --phase map")
        phase_map(
            args.selector, args.disambiguator,
            force=args.force, limit=args.limit,
            run_tag=args.run_tag, map_workers=map_workers,
        )
    elif args.phase == "pipeline":
        if not args.disambiguator:
            ap.error("--disambiguator required for --phase pipeline")
        phase_pipeline(
            args.selector,
            args.disambiguator,
            force=args.force,
            limit=args.limit,
            run_tag=args.run_tag,
            pipeline_workers=pipeline_workers,
            map_workers=map_workers,
            with_score=args.with_score,
            score_workers=args.score_workers,
            score_xgb_nthread=args.score_xgb_nthread,
        )


if __name__ == "__main__":
    main()
