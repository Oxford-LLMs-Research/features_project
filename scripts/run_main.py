"""
MAIN experiment orchestrator — free-text elicitation pipeline (the paper's current design).

Generalised from the second-pilot orchestrator (analysis/format_pilot.py); artifacts
live under outputs/main/ (legacy outputs/format_pilot/ still dual-resolved).

Pipeline per selector (the test model whose capability we measure):
  --phase gen     : free-text selection responses, both prompt conditions, cached per cell.
  --phase extract : essay -> typed feature list via the FIXED extractor (Qwen-235B).
  --phase map     : per-feature top-20 retrieval + disambiguation (--disambiguator).
                    Arms: C = free-text (extracted), B = legacy JSON selections re-mapped.
  --phase score   : captured importance + oracle/model/random accuracy at model-chosen k
                    and fixed k=5,10, per arm x disambiguator -> scores_<selector>.csv.

Grid = the GENUINE cells from the leakage audit (scripts/leakage_audit.py),
both prompt conditions.

Use --run-tag to write map/score under main/runs/<tag>/ (or experiments/…/runs/<tag>/
with --embedding-model) so multi-person runs do not clobber the canonical baseline.
Gen/extract always stay under main/<selector>/ (shared cache).

Phased + resumable: per-cell JSON checkpoints make every phase resumable; rerunning
skips cells already on disk unless --force. Selectors are registered in
survey_features.config.SELECTORS; each keeps its own subdir so adding a model never
clobbers another's artifacts.

Examples:
  python scripts/run_main.py --phase gen     --selector deepseek
  python scripts/run_main.py --phase extract --selector deepseek
  python scripts/run_main.py --phase map     --selector deepseek --disambiguator nemotron
  python scripts/run_main.py --phase map     --selector deepseek --disambiguator nemotron --map-workers 8
  python scripts/run_main.py --phase score   --selector deepseek
  python scripts/run_main.py --phase pipeline --selector deepseek --disambiguator nemotron \\
      --pipeline-workers 4 --map-workers 8 --with-score

Concurrency (defaults preserve serial behavior):
  --api-workers / API_WORKERS           cell ThreadPool for gen/extract (default 1)
  --map-workers / MAP_WORKERS           per-feature disambig ThreadPool inside map (default 1)
  --pipeline-workers / PIPELINE_WORKERS cells in flight for --phase pipeline (default 1)
  Each phase writes outputs/logs/timing_<phase>_*.jsonl and prints a span summary.

Embedding-model sensitivity (reuses gen/extract; writes under experiments/embedding_sensitivity/):
  python scripts/run_main.py --phase map   --selector deepseek --disambiguator nemotron \\
      --arms C --embedding-model all-mpnet-base-v2
  python scripts/run_main.py --phase score --selector deepseek --embedding-model all-mpnet-base-v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
    cell_dir,
    cell_tag,
    embedding_run_dirs,
    embedding_sensitivity_dir,
    genuine_cells,
    main_dir,
    main_scores_path,
    selector_dirs,
)

OUT = OUTPUTS_DIR
PILOT = main_dir(OUT)

FIXED_KS = [5, 10]


# ── shared helpers ────────────────────────────────────────────────────────────

def selector_generate_fn(selector_model: str):
    """Selector client on the main LLM endpoint; transient errors -> '' (keeps sweeps alive)."""
    from survey_features.llm import make_generate_fn
    fn, _ = make_generate_fn(model=selector_model, on_error="empty")
    return fn


def mapper_generate_fn(mapper_model: str):
    """Extractor/disambiguator client on the disambig endpoint (falls back to LLM endpoint)."""
    from survey_features.llm import make_generate_fn
    fn, _ = make_generate_fn(
        base_url=os.environ.get("DISAMBIG_BASE_URL") or None,
        api_key=os.environ.get("DISAMBIG_API_KEY") or None,
        model=mapper_model,
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


def _upsert_embedding_manifest(
    *,
    embedding_model: str,
    selector_key: str,
    phase: str,
    disambiguator: str | None = None,
    arms: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> None:
    """Create/update outputs/embedding_sensitivity/manifest.json for provenance."""
    root = embedding_sensitivity_dir(OUT)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    now = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        man = json.loads(path.read_text(encoding="utf-8"))
    else:
        man = {
            "experiment": "embedding_sensitivity",
            "baseline": {
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "scores_root": "outputs/main",
                "note": "Main-experiment MiniLM maps/scores; not re-run unless --embedding-model points at it.",
            },
            "disambiguator": "nemotron",
            "arms": ["C"],
            "models": {},
            "created_at": now,
        }
    models = man.setdefault("models", {})
    entry = models.setdefault(embedding_model, {"runs": []})
    entry["runs"].append({
        "phase": phase,
        "selector": selector_key,
        "disambiguator": disambiguator,
        "arms": list(arms) if arms else None,
        "limit": limit,
        "at": now,
    })
    man["updated_at"] = now
    path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")


_data_cache: dict = {}


def survey_data(survey):
    if survey not in _data_cache:
        from survey_features.surveys import load_survey
        _data_cache[survey] = load_survey(survey, os.environ["DATA_CONFIG_PATH"])
    return _data_cache[survey]


# ── Phase: gen (free-text selection; cheap, irreversible) ─────────────────────

def phase_gen(selector_key, force=False, limit=None, api_workers: int = 1):
    from survey_features.elicitation import freetext_messages
    from survey_features.timing import TimingLog, default_timing_path

    sel_model = SELECTORS[selector_key]["model"]
    gen_dir, _, _ = selector_dirs(selector_key)
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen = selector_generate_fn(sel_model)
    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    n_workers = max(1, int(api_workers))
    timing = TimingLog(default_timing_path("gen", selector_key))
    print(
        f"[gen] selector={selector_key} ({sel_model}) {len(cells)} cells x "
        f"{len(CONDITIONS)} conds -> free-text  api_workers={n_workers}"
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
            rec = {"survey": survey, "target": target, "country": country,
                   "question_text": qtext, "selector_model": sel_model, "responses": {}}
            for cond in CONDITIONS:
                messages = freetext_messages(qtext, country if cond == "country_provided" else None)
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
                    print(f"  [{i+1}/{len(cells)}] {survey} {target} {country}: "
                          f"up={up}c cp={cp}c")
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
                        print(f"  [{i+1}/{len(cells)}] {survey} {target} {country}: "
                              f"up={up}c cp={cp}c")

    print(f"[gen] done. wrote={done} skipped(existing)={skipped} -> {gen_dir}")
    timing.print_summary()


# ── Phase: extract (free-text -> feature list, FIXED extractor) ───────────────

def phase_extract(selector_key, force=False, limit=None, api_workers: int = 1):
    """Arm C only: turn each cached free-text essay into a typed feature list via the FIXED
    extractor (Qwen). Held constant across disambiguators so the request set does not vary
    by mapper. Arm B needs no extraction (JSON is already a feature list)."""
    from survey_features.extraction import extract_features
    from survey_features.timing import TimingLog, default_timing_path

    egen = mapper_generate_fn(EXTRACTOR_MODEL)
    gen_dir, extract_dir, _ = selector_dirs(selector_key)
    cells = genuine_cells(OUT)
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
            rec = {"survey": survey, "target": target, "country": country,
                   "extractor_model": EXTRACTOR_MODEL, "features": {}}
            for cond in CONDITIONS:
                feats, raw = extract_features(responses.get(cond, ""), egen)
                rec["features"][cond] = feats
            op.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        with done_lock:
            done += 1
        nfu = len(rec["features"].get("unprompted", []))
        nfc = len(rec["features"].get("country_provided", []))
        return survey, target, country, nfu, nfc

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


# ── Phase: map (per-feature disambiguation; disambiguator varies) ─────────────

def _json_arm_features(selector_key, target, country, cond):
    """Arm B: the selector's legacy JSON selection as a feature list (parsed directly, no
    extraction). Uses cached disambig.json feature_label/reasoning as feature/context."""
    tag = SELECTORS[selector_key]["pilot1_tag"]
    p = cell_dir(target, country, OUT) / f"llm__{tag}" / "disambig.json"
    if not p.is_file():
        return None
    items = json.loads(p.read_text(encoding="utf-8"))
    out, seen = [], set()
    for m in items:
        if m.get("condition") != cond:
            continue
        lab = (m.get("feature_label") or "").strip()
        if lab and lab.lower() not in seen:
            seen.add(lab.lower())
            out.append({"feature": lab, "context": (m.get("feature_reasoning") or "").strip(), "sub_items": []})
    return out


def phase_map(
    selector_key,
    disambig_key,
    arms=("B", "C"),
    force=False,
    limit=None,
    embedding_model: str | None = None,
    run_tag: str | None = None,
    map_workers: int = 1,
):
    from survey_features.disambig import map_features
    from survey_features.retrieval import make_embed_fn
    from survey_features.timing import TimingLog, default_timing_path

    emb_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    dmodel = DISAMBIGUATORS[disambig_key]
    dgen = mapper_generate_fn(dmodel)
    embed = make_embed_fn(emb_model)
    _, extract_dir, default_map_dir = selector_dirs(selector_key, run_tag=None)
    if embedding_model:
        map_dir, _ = embedding_run_dirs(embedding_model, selector_key, run_tag=run_tag)
    else:
        _, _, map_dir = selector_dirs(selector_key, run_tag=run_tag)
        if map_dir == default_map_dir and run_tag is None:
            map_dir = default_map_dir
    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    map_dir.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, int(map_workers))
    timing = TimingLog(default_timing_path("map", f"{selector_key}_{disambig_key}"))
    print(
        f"[map] selector={selector_key} disambiguator={disambig_key} arms={arms} "
        f"cells={len(cells)} embedding={emb_model} map_workers={n_workers}"
        + (f" -> {map_dir}" if embedding_model else "")
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
            excluded = {target}
            ctag = cell_tag(survey, target, country)

            if "C" in arms:
                ep = extract_dir / f"{ctag}.json"
                feat_by_cond = json.loads(ep.read_text(encoding="utf-8"))["features"] if ep.is_file() else None
                for cond in CONDITIONS:
                    op = map_dir / f"C__{disambig_key}__{ctag}__{cond}.json"
                    if op.is_file() and not force:
                        continue
                    if feat_by_cond is None:
                        print(f"  ! missing extraction for {ctag}; run --phase extract first")
                        continue
                    with timing.span(
                        "cell_map",
                        arm="C",
                        survey=survey,
                        target=target,
                        country=country,
                        condition=cond,
                    ):
                        cm = map_features(
                            f"{target}_{country}", "C_free", feat_by_cond.get(cond, []),
                            emb, vcodes, svars, embed, dgen, mapper_model=dmodel,
                            excluded_codes=excluded, pipe_types=PIPE_TYPES,
                            workers=n_workers,
                        )
                        _save_map(op, survey, target, country, cond, "C", disambig_key, cm, emb_model)

            if "B" in arms:
                for cond in CONDITIONS:
                    op = map_dir / f"B__{disambig_key}__{ctag}__{cond}.json"
                    if op.is_file() and not force:
                        continue
                    feats = _json_arm_features(selector_key, target, country, cond)
                    if feats is None:
                        continue
                    with timing.span(
                        "cell_map",
                        arm="B",
                        survey=survey,
                        target=target,
                        country=country,
                        condition=cond,
                    ):
                        cm = map_features(
                            f"{target}_{country}", "B_json", feats,
                            emb, vcodes, svars, embed, dgen, mapper_model=dmodel,
                            excluded_codes=excluded, pipe_types=PIPE_TYPES,
                            workers=n_workers,
                        )
                        _save_map(op, survey, target, country, cond, "B", disambig_key, cm, emb_model)

            print(f"  [{i+1}/{len(cells)}] {survey} {target} {country} mapped ({disambig_key})")
    print(f"[map] done -> {map_dir}")
    timing.print_summary()
    if embedding_model:
        _upsert_embedding_manifest(
            embedding_model=embedding_model,
            selector_key=selector_key,
            phase="map",
            disambiguator=disambig_key,
            arms=arms,
            limit=limit,
        )


def _save_map(path, survey, target, country, cond, arm, disambig_key, cm, embedding_model=DEFAULT_EMBEDDING_MODEL):
    rec = {
        "survey": survey, "target": target, "country": country, "condition": cond,
        "arm": arm, "disambiguator": disambig_key, "disambig_model": cm.mapper_model,
        "embedding_model": embedding_model,
        "n_features": cm.n_features, "n_piped": cm.n_piped, "n_mapped": cm.n_mapped,
        "n_none": cm.n_none, "n_bundled": cm.n_bundled, "type_counts": cm.type_counts(),
        "mapped_codes": cm.mapped_codes,
        "features": [
            {"feature": f.feature_label, "context": f.feature_context, "sub_items": f.sub_items,
             "type": f.ftype, "piped": f.piped,
             "selected_code": f.selected_code, "selected_text": f.selected_text,
             "n_candidates": len(f.candidates), "disambig_raw": f.disambig_raw[:80]}
            for f in cm.features
        ],
    }
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Phase: pipeline (DAG — overlap gen→extract→map across cells) ──────────────

def phase_pipeline(
    selector_key,
    disambig_key,
    arms=("B", "C"),
    force=False,
    limit=None,
    embedding_model: str | None = None,
    run_tag: str | None = None,
    pipeline_workers: int = 1,
    map_workers: int = 1,
    with_score: bool = False,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
):
    """Run gen → extract → map per cell with multiple cells in flight.

    Within a cell the dependency order is strict. Across cells, up to
    ``pipeline_workers`` chains run concurrently so extract(N) can overlap
    map(N−1). Same checkpoints / --force semantics as the single-phase commands.
    Score (CPU ProcessPool) runs after all maps if ``with_score``.
    """
    from survey_features.disambig import map_features
    from survey_features.elicitation import freetext_messages
    from survey_features.extraction import extract_features
    from survey_features.retrieval import make_embed_fn
    from survey_features.timing import TimingLog, default_timing_path

    sel_model = SELECTORS[selector_key]["model"]
    emb_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    dmodel = DISAMBIGUATORS[disambig_key]
    gen_dir, extract_dir, default_map_dir = selector_dirs(selector_key, run_tag=None)
    if embedding_model:
        map_dir, _ = embedding_run_dirs(embedding_model, selector_key, run_tag=run_tag)
    else:
        _, _, map_dir = selector_dirs(selector_key, run_tag=run_tag)
        if map_dir == default_map_dir and run_tag is None:
            map_dir = default_map_dir

    gen_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    n_pipe = max(1, int(pipeline_workers))
    n_map = max(1, int(map_workers))
    timing = TimingLog(default_timing_path("pipeline", f"{selector_key}_{disambig_key}"))

    gen_fn = selector_generate_fn(sel_model)
    extract_fn = mapper_generate_fn(EXTRACTOR_MODEL)
    disambig_fn = mapper_generate_fn(dmodel)
    # Serialize ST encode across concurrent cell maps (model load is locked; encode
    # is safer single-flight when several cells map at once).
    _raw_embed = make_embed_fn(emb_model)
    _embed_lock = threading.Lock()

    def embed(texts):
        with _embed_lock:
            return _raw_embed(texts)

    print(
        f"[pipeline] selector={selector_key} disambiguator={disambig_key} arms={arms} "
        f"cells={len(cells)} embedding={emb_model} "
        f"pipeline_workers={n_pipe} map_workers={n_map} with_score={with_score}"
        + (f" -> {map_dir}" if embedding_model or run_tag else "")
    )

    counts = {"gen_wrote": 0, "gen_skip": 0, "ext_wrote": 0, "ext_skip": 0,
              "map_wrote": 0, "map_skip": 0, "errors": 0}
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
        excluded = {target}
        ctag = cell_tag(survey, target, country)
        wrote = skipped = 0

        if "C" in arms:
            ep = extract_dir / f"{ctag}.json"
            feat_by_cond = (
                json.loads(ep.read_text(encoding="utf-8"))["features"] if ep.is_file() else None
            )
            for cond in CONDITIONS:
                op = map_dir / f"C__{disambig_key}__{ctag}__{cond}.json"
                if op.is_file() and not force:
                    skipped += 1
                    continue
                if feat_by_cond is None:
                    return "missing_extract"
                with timing.span(
                    "cell_map", arm="C", survey=survey, target=target,
                    country=country, condition=cond,
                ):
                    cm = map_features(
                        f"{target}_{country}", "C_free", feat_by_cond.get(cond, []),
                        emb, vcodes, svars, embed, disambig_fn, mapper_model=dmodel,
                        excluded_codes=excluded, pipe_types=PIPE_TYPES, workers=n_map,
                    )
                    _save_map(op, survey, target, country, cond, "C", disambig_key, cm, emb_model)
                wrote += 1

        if "B" in arms:
            for cond in CONDITIONS:
                op = map_dir / f"B__{disambig_key}__{ctag}__{cond}.json"
                if op.is_file() and not force:
                    skipped += 1
                    continue
                feats = _json_arm_features(selector_key, target, country, cond)
                if feats is None:
                    continue
                with timing.span(
                    "cell_map", arm="B", survey=survey, target=target,
                    country=country, condition=cond,
                ):
                    cm = map_features(
                        f"{target}_{country}", "B_json", feats,
                        emb, vcodes, svars, embed, disambig_fn, mapper_model=dmodel,
                        excluded_codes=excluded, pipe_types=PIPE_TYPES, workers=n_map,
                    )
                    _save_map(op, survey, target, country, cond, "B", disambig_key, cm, emb_model)
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
            with timing.span(
                "cell_pipeline", survey=survey, target=target, country=country,
            ):
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
        # Warm survey / embedding caches before fan-out
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

    if embedding_model:
        _upsert_embedding_manifest(
            embedding_model=embedding_model,
            selector_key=selector_key,
            phase="pipeline",
            disambiguator=disambig_key,
            arms=arms,
            limit=limit,
        )

    if with_score:
        phase_score(
            selector_key,
            force=force,
            limit=limit,
            embedding_model=embedding_model,
            score_workers=score_workers,
            score_xgb_nthread=score_xgb_nthread,
            run_tag=run_tag,
        )


# ── Phase: score (captured importance + value-over-random, all arms) ──────────

def _arm_A_codes(selector_key, target, country, cond):
    """Legacy (pilot-1) mapped codes (deduped, arrival order) for arm A from the selector's
    cached disambig.json."""
    tag = SELECTORS[selector_key]["pilot1_tag"]
    p = cell_dir(target, country, OUT) / f"llm__{tag}" / "disambig.json"
    if not p.is_file():
        return None
    items = json.loads(p.read_text(encoding="utf-8"))
    seen, out = set(), []
    for m in sorted([x for x in items if x.get("condition") == cond],
                    key=lambda x: x.get("feature_rank", 0)):
        c = (m.get("disambig") or {}).get("selected_code")
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _map_codes(map_dir, arm, disambig_key, survey, target, country, cond):
    """Arm B/C mapped codes from the map file."""
    ctag = cell_tag(survey, target, country)
    p = map_dir / f"{arm}__{disambig_key}__{ctag}__{cond}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("mapped_codes", [])


def phase_score(
    selector_key,
    force=False,
    limit=None,
    embedding_model: str | None = None,
    score_workers: int | None = None,
    score_xgb_nthread: int | None = None,
    run_tag: str | None = None,
):
    """Score all arms via cell-level ProcessPool (survey_features.score_cell).

    Oracle/random baselines are cached per (cell, k) inside each worker. Random
    draws stay serial within a cell (Windows joblib lesson); parallelism is across
    cells. Score-only workers do not load torch, so XGB nthread > 1 is safe.

    With --embedding-model, scores only maps under embedding_sensitivity/ (no arm A)
    and never overwrites main/scores_*.csv. With --run-tag, writes under …/runs/<tag>/.
    """
    from survey_features.score_cell import (
        resolve_score_n_draws,
        resolve_score_workers,
        resolve_score_xgb_nthread,
        run_score_jobs,
    )
    from survey_features.timing import TimingLog, default_timing_path

    n_draws = resolve_score_n_draws()
    workers = resolve_score_workers(score_workers)
    nthread = resolve_score_xgb_nthread(workers, score_xgb_nthread)
    timing = TimingLog(default_timing_path("score", selector_key))

    if embedding_model:
        map_dir, out_csv = embedding_run_dirs(
            embedding_model, selector_key, run_tag=run_tag,
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        include_arm_a = False
        emb_label = embedding_model
    else:
        _, _, map_dir = selector_dirs(selector_key, run_tag=run_tag)
        out_csv = main_scores_path(selector_key, OUT, run_tag=run_tag)
        # Prefer canonical scores_<sel>.csv write (not legacy scores.csv)
        if run_tag is None:
            out_csv = PILOT / f"scores_{selector_key}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        include_arm_a = True
        emb_label = DEFAULT_EMBEDDING_MODEL

    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    cols = [
        "survey", "target", "country", "condition", "arm", "disambiguator",
        "embedding_model", "k_spec", "k",
        "captured_importance", "oracle_acc", "model_acc", "random_acc", "majority",
        "value_over_random", "cost_of_imperfect", "error",
    ]

    specs = []
    for survey, target, country in cells:
        evals = []
        for cond in CONDITIONS:
            if include_arm_a:
                codes_a = _arm_A_codes(selector_key, target, country, cond)
                if codes_a is not None:
                    evals.append({
                        "condition": cond, "arm": "A", "disambiguator": "",
                        "embedding_model": emb_label, "codes": codes_a,
                    })
            for dk in DISAMBIGUATORS:
                for arm in ("B", "C"):
                    codes = _map_codes(map_dir, arm, dk, survey, target, country, cond)
                    if codes is None:
                        continue
                    evals.append({
                        "condition": cond, "arm": arm, "disambiguator": dk,
                        "embedding_model": emb_label, "codes": codes,
                    })
        specs.append({
            "survey": survey, "target": target, "country": country,
            "evals": evals,
            "n_draws": n_draws, "nthread": nthread, "fixed_ks": list(FIXED_KS),
            "outputs_dir": str(OUT),
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
    if embedding_model:
        _upsert_embedding_manifest(
            embedding_model=embedding_model,
            selector_key=selector_key,
            phase="score",
            limit=limit,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--phase",
        choices=["gen", "extract", "map", "score", "pipeline"],
        required=True,
    )
    ap.add_argument("--selector", choices=list(SELECTORS), default=DEFAULT_SELECTOR,
                    help=f"test model whose capability is measured (default: {DEFAULT_SELECTOR})")
    ap.add_argument(
        "--disambiguator",
        choices=list(DISAMBIGUATORS),
        help="required for --phase map and --phase pipeline",
    )
    ap.add_argument(
        "--arms",
        default="B,C",
        help="comma-separated arms for --phase map/pipeline (default: B,C)",
    )
    ap.add_argument(
        "--embedding-model",
        default=None,
        metavar="MODEL",
        help=(
            "Sentence-transformer for map/score/pipeline. When set, writes under "
            "outputs/experiments/embedding_sensitivity/<slug>/ "
            f"(default encode model: {DEFAULT_EMBEDDING_MODEL}; "
            "omit this flag to keep writing under outputs/main/)."
        ),
    )
    ap.add_argument(
        "--run-tag",
        default=None,
        metavar="TAG",
        help=(
            "Write map/score under …/runs/<TAG>/ instead of the canonical baseline "
            "(use for multi-person / exploratory runs; gen/extract stay shared)."
        ),
    )
    ap.add_argument("--force", action="store_true", help="recompute cells already on disk")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cells (smoke test)")
    ap.add_argument(
        "--api-workers",
        type=int,
        default=None,
        help="cell ThreadPool size for --phase gen/extract (default: API_WORKERS or 1)",
    )
    ap.add_argument(
        "--map-workers",
        type=int,
        default=None,
        help=(
            "per-feature disambig ThreadPool for --phase map/pipeline "
            "(default: MAP_WORKERS or 1)"
        ),
    )
    ap.add_argument(
        "--pipeline-workers",
        type=int,
        default=None,
        help=(
            "cells in flight for --phase pipeline "
            "(default: PIPELINE_WORKERS or 1)"
        ),
    )
    ap.add_argument(
        "--with-score",
        action="store_true",
        help="after --phase pipeline maps finish, run score",
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

    if args.embedding_model and args.phase in ("gen", "extract"):
        ap.error("--embedding-model only applies to --phase map, score, and pipeline "
                 "(gen/extract are reused from main/)")
    if args.run_tag and args.phase in ("gen", "extract"):
        ap.error("--run-tag only applies to --phase map, score, and pipeline "
                 "(gen/extract are shared under main/<selector>/)")
    if args.phase not in ("score", "pipeline") and (
        args.score_workers is not None or args.score_xgb_nthread is not None
    ):
        ap.error("--score-workers / --score-xgb-nthread only apply to --phase score "
                 "(or pipeline with --with-score)")
    if args.phase == "pipeline" and (
        args.score_workers is not None or args.score_xgb_nthread is not None
    ) and not args.with_score:
        ap.error("--score-workers / --score-xgb-nthread require --with-score")
    if args.phase not in ("gen", "extract") and args.api_workers is not None:
        ap.error("--api-workers only applies to --phase gen and --phase extract")
    if args.phase not in ("map", "pipeline") and args.map_workers is not None:
        ap.error("--map-workers only applies to --phase map and --phase pipeline")
    if args.phase != "pipeline" and args.pipeline_workers is not None:
        ap.error("--pipeline-workers only applies to --phase pipeline")
    if args.with_score and args.phase != "pipeline":
        ap.error("--with-score only applies to --phase pipeline")

    api_workers = resolve_workers(args.api_workers, "API_WORKERS", default=1)
    map_workers = resolve_workers(args.map_workers, "MAP_WORKERS", default=1)
    pipeline_workers = resolve_workers(args.pipeline_workers, "PIPELINE_WORKERS", default=1)
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    if args.phase == "gen":
        phase_gen(args.selector, force=args.force, limit=args.limit, api_workers=api_workers)
    elif args.phase == "extract":
        phase_extract(args.selector, force=args.force, limit=args.limit, api_workers=api_workers)
    elif args.phase == "score":
        phase_score(
            args.selector,
            force=args.force,
            limit=args.limit,
            embedding_model=args.embedding_model,
            score_workers=args.score_workers,
            score_xgb_nthread=args.score_xgb_nthread,
            run_tag=args.run_tag,
        )
    elif args.phase == "map":
        if not args.disambiguator:
            ap.error("--disambiguator required for --phase map")
        phase_map(
            args.selector, args.disambiguator, arms=arms,
            force=args.force, limit=args.limit, embedding_model=args.embedding_model,
            run_tag=args.run_tag, map_workers=map_workers,
        )
    elif args.phase == "pipeline":
        if not args.disambiguator:
            ap.error("--disambiguator required for --phase pipeline")
        phase_pipeline(
            args.selector,
            args.disambiguator,
            arms=arms,
            force=args.force,
            limit=args.limit,
            embedding_model=args.embedding_model,
            run_tag=args.run_tag,
            pipeline_workers=pipeline_workers,
            map_workers=map_workers,
            with_score=args.with_score,
            score_workers=args.score_workers,
            score_xgb_nthread=args.score_xgb_nthread,
        )


if __name__ == "__main__":
    main()
