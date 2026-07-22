"""
MAIN experiment orchestrator — free-text elicitation pipeline (the paper's current design).

Generalised from the second-pilot orchestrator (analysis/format_pilot.py); artifact
paths are unchanged (outputs/format_pilot/) so all existing pilot-2 results stay valid.

Pipeline per selector (the test model whose capability we measure):
  --phase gen     : free-text selection responses, both prompt conditions, cached per cell.
  --phase extract : essay -> typed feature list via the FIXED extractor (Qwen-235B).
  --phase map     : per-feature top-20 retrieval + disambiguation (--disambiguator).
                    Arms: C = free-text (extracted), B = legacy JSON selections re-mapped.
  --phase score   : captured importance + oracle/model/random accuracy at model-chosen k
                    and fixed k=5,10, per arm x disambiguator -> scores_<selector>.csv.

Grid = the GENUINE cells from the leakage audit (scripts/leakage_audit.py ->
outputs/leakage_audit.csv), both prompt conditions.

Phased + resumable: per-cell JSON checkpoints make every phase resumable; rerunning
skips cells already on disk unless --force. Selectors are registered in
survey_features.config.SELECTORS; each keeps its own subdir so adding a model never
clobbers another's artifacts.

Examples:
  python scripts/run_main.py --phase gen     --selector deepseek
  python scripts/run_main.py --phase extract --selector deepseek
  python scripts/run_main.py --phase map     --selector deepseek --disambiguator nemotron
  python scripts/run_main.py --phase score   --selector deepseek
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survey_features.config import (  # noqa: E402
    CONDITIONS,
    DEFAULT_SELECTOR,
    DISAMBIGUATORS,
    EXTRACTOR_MODEL,
    OUTPUTS_DIR,
    PIPE_TYPES,
    SELECTORS,
)
from survey_features.layout import (  # noqa: E402
    cell_tag,
    format_pilot_dir,
    genuine_cells,
    selector_dirs,
)

OUT = OUTPUTS_DIR
PILOT = format_pilot_dir(OUT)

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


def survey_assets(survey_id):
    """(survey_variables, embeddings, var_codes) — cached per survey."""
    if survey_id not in _survey_cache:
        from survey_features.retrieval import load_or_build_survey_embeddings
        from survey_features.surveys import extract_survey_variables, load_survey
        _, meta = load_survey(survey_id, os.environ["DATA_CONFIG_PATH"])
        svars = extract_survey_variables(meta)
        emb, vcodes = load_or_build_survey_embeddings(svars, survey_id)
        _survey_cache[survey_id] = (svars, emb, vcodes)
    return _survey_cache[survey_id]


_data_cache: dict = {}


def survey_data(survey):
    if survey not in _data_cache:
        from survey_features.surveys import load_survey
        _data_cache[survey] = load_survey(survey, os.environ["DATA_CONFIG_PATH"])
    return _data_cache[survey]


# ── Phase: gen (free-text selection; cheap, irreversible) ─────────────────────

def phase_gen(selector_key, force=False, limit=None):
    from survey_features.elicitation import freetext_messages
    sel_model = SELECTORS[selector_key]["model"]
    gen_dir, _, _ = selector_dirs(selector_key)
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen = selector_generate_fn(sel_model)
    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    print(f"[gen] selector={selector_key} ({sel_model}) {len(cells)} cells x {len(CONDITIONS)} conds -> free-text")
    done = skipped = 0
    for i, (survey, target, country) in enumerate(cells):
        svars, _, _ = survey_assets(survey)
        qtext = svars.get(target, target)
        out_path = gen_dir / f"{cell_tag(survey, target, country)}.json"
        if out_path.is_file() and not force:
            skipped += 1
            continue
        rec = {"survey": survey, "target": target, "country": country,
               "question_text": qtext, "selector_model": sel_model, "responses": {}}
        for cond in CONDITIONS:
            messages = freetext_messages(qtext, country if cond == "country_provided" else None)
            resp = gen(messages, max_tokens=4096, usage_phase="feature_list")
            if not resp:
                rec.setdefault("errors", {})[cond] = "empty response after retries"
            rec["responses"][cond] = resp
        out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        done += 1
        print(f"  [{i+1}/{len(cells)}] {survey} {target} {country}: "
              f"up={len(rec['responses'].get('unprompted',''))}c cp={len(rec['responses'].get('country_provided',''))}c")
    print(f"[gen] done. wrote={done} skipped(existing)={skipped} -> {gen_dir}")


# ── Phase: extract (free-text -> feature list, FIXED extractor) ───────────────

def phase_extract(selector_key, force=False, limit=None):
    """Arm C only: turn each cached free-text essay into a typed feature list via the FIXED
    extractor (Qwen). Held constant across disambiguators so the request set does not vary
    by mapper. Arm B needs no extraction (JSON is already a feature list)."""
    from survey_features.extraction import extract_features
    egen = mapper_generate_fn(EXTRACTOR_MODEL)
    gen_dir, extract_dir, _ = selector_dirs(selector_key)
    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] selector={selector_key} {len(cells)} cells x {len(CONDITIONS)} conds via {EXTRACTOR_MODEL}")
    done = skipped = 0
    for i, (survey, target, country) in enumerate(cells):
        ctag = cell_tag(survey, target, country)
        op = extract_dir / f"{ctag}.json"
        if op.is_file() and not force:
            skipped += 1
            continue
        gp = gen_dir / f"{ctag}.json"
        if not gp.is_file():
            print(f"  ! missing free-text for {ctag}; run --phase gen first")
            continue
        responses = json.loads(gp.read_text(encoding="utf-8"))["responses"]
        rec = {"survey": survey, "target": target, "country": country,
               "extractor_model": EXTRACTOR_MODEL, "features": {}}
        for cond in CONDITIONS:
            feats, raw = extract_features(responses.get(cond, ""), egen)
            rec["features"][cond] = feats
        op.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        done += 1
        nfu = len(rec["features"].get("unprompted", []))
        nfc = len(rec["features"].get("country_provided", []))
        print(f"  [{i+1}/{len(cells)}] {survey} {target} {country}: up={nfu} cp={nfc} features")
    print(f"[extract] done. wrote={done} skipped={skipped} -> {extract_dir}")


# ── Phase: map (per-feature disambiguation; disambiguator varies) ─────────────

def _json_arm_features(selector_key, target, country, cond):
    """Arm B: the selector's legacy JSON selection as a feature list (parsed directly, no
    extraction). Uses cached disambig.json feature_label/reasoning as feature/context."""
    tag = SELECTORS[selector_key]["pilot1_tag"]
    p = OUT / f"{target}_{country}" / f"llm__{tag}" / "disambig.json"
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


def phase_map(selector_key, disambig_key, arms=("B", "C"), force=False, limit=None):
    from survey_features.disambig import map_features
    from survey_features.retrieval import make_embed_fn
    dmodel = DISAMBIGUATORS[disambig_key]
    dgen = mapper_generate_fn(dmodel)
    embed = make_embed_fn()
    _, extract_dir, map_dir = selector_dirs(selector_key)
    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    map_dir.mkdir(parents=True, exist_ok=True)
    print(f"[map] selector={selector_key} disambiguator={disambig_key} arms={arms} cells={len(cells)}")

    for i, (survey, target, country) in enumerate(cells):
        svars, emb, vcodes = survey_assets(survey)
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
                cm = map_features(f"{target}_{country}", "C_free", feat_by_cond.get(cond, []),
                                  emb, vcodes, svars, embed, dgen, mapper_model=dmodel,
                                  excluded_codes=excluded, pipe_types=PIPE_TYPES)
                _save_map(op, survey, target, country, cond, "C", disambig_key, cm)

        if "B" in arms:
            for cond in CONDITIONS:
                op = map_dir / f"B__{disambig_key}__{ctag}__{cond}.json"
                if op.is_file() and not force:
                    continue
                feats = _json_arm_features(selector_key, target, country, cond)
                if feats is None:
                    continue
                cm = map_features(f"{target}_{country}", "B_json", feats,
                                  emb, vcodes, svars, embed, dgen, mapper_model=dmodel,
                                  excluded_codes=excluded, pipe_types=PIPE_TYPES)
                _save_map(op, survey, target, country, cond, "B", disambig_key, cm)

        print(f"  [{i+1}/{len(cells)}] {survey} {target} {country} mapped ({disambig_key})")
    print(f"[map] done -> {map_dir}")


def _save_map(path, survey, target, country, cond, arm, disambig_key, cm):
    rec = {
        "survey": survey, "target": target, "country": country, "condition": cond,
        "arm": arm, "disambiguator": disambig_key, "disambig_model": cm.mapper_model,
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


# ── Phase: score (captured importance + value-over-random, all arms) ──────────

def _oracle_table(survey):
    """Load oracle importances for a survey from per-cell oracle.csv into one DataFrame
    with columns [target_variable, country, feature_variable, importance_mean]. country is
    the numeric/code value matching the survey's country column (as run_comparison expects)."""
    import pandas as pd
    from survey_features.surveys import SURVEY_COUNTRY_COL, build_country_code_map, load_survey
    _, meta = load_survey(survey, os.environ["DATA_CONFIG_PATH"])
    ccol = SURVEY_COUNTRY_COL.get(survey)
    data, _ = survey_data(survey)
    cmap = build_country_code_map(meta, ccol, data) if ccol else {}
    rows = []
    for s, t, c in genuine_cells(OUT):
        if s != survey:
            continue
        p = OUT / f"{t}_{c}" / "oracle.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        code = cmap.get(c, c)
        for _, r in df.iterrows():
            rows.append({"target_variable": t, "country": code,
                         "feature_variable": r["feature_variable"],
                         "importance_mean": r["importance_mean"]})
    return pd.DataFrame(rows), ccol, cmap


def _arm_A_codes(selector_key, target, country, cond):
    """Legacy (pilot-1) mapped codes (deduped, arrival order) for arm A from the selector's
    cached disambig.json."""
    tag = SELECTORS[selector_key]["pilot1_tag"]
    p = OUT / f"{target}_{country}" / f"llm__{tag}" / "disambig.json"
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


def phase_score(selector_key, force=False, limit=None):
    """Score all arms. Key efficiency: oracle top-k and the random-k baseline depend only
    on (cell, k) — NOT on the arm or disambiguator — so we compute them ONCE per (cell, k)
    and reuse, evaluating only the cheap per-arm model feature set each time. This avoids
    the ~10x redundant XGBoost fits the naive arm-loop incurred (the random-draw baseline
    on the full column matrix is the dominant cost). All XGBoost is single-threaded
    (nthread=1): torch/sentence-transformers already loaded a conflicting libomp earlier in
    the process (documented Windows issue) and multi-threaded fits hang.

    Random draws are SERIAL: joblib/loky parallelism is unreliable in this environment
    (TerminatedWorkerError / OOM from re-pickling the full survey frame per call on
    Windows). Serial is slow but deterministic; SCORE_N_DRAWS (default 10) keeps a full
    run manageable."""
    import csv as _csv
    import numpy as np
    from survey_features.evaluation import evaluate_feature_set, single_random_draw
    from survey_features.metrics import captured_importance_df

    n_draws = int(os.environ.get("SCORE_N_DRAWS", "10"))
    _, _, map_dir = selector_dirs(selector_key)
    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    PILOT.mkdir(parents=True, exist_ok=True)
    out_csv = PILOT / f"scores_{selector_key}.csv"
    cols = ["survey","target","country","condition","arm","disambiguator","k_spec","k",
            "captured_importance","oracle_acc","model_acc","random_acc","majority",
            "value_over_random","cost_of_imperfect","error"]
    # Incremental write: open now, flush after each cell so progress is visible and a
    # mid-run interruption keeps completed cells.
    out_f = open(out_csv, "w", newline="", encoding="utf-8")
    writer = _csv.DictWriter(out_f, fieldnames=cols)
    writer.writeheader(); out_f.flush()
    n_written = 0

    surveys = sorted({s for s, _, _ in cells})
    for survey in surveys:
        oracle_df, ccol, cmap = _oracle_table(survey)

        def oracle_topk(t, code, k):
            sub = oracle_df[(oracle_df["target_variable"] == t) & (oracle_df["country"] == code)]
            return sub.sort_values("importance_mean", ascending=False)["feature_variable"].head(k).tolist()

        data, _ = survey_data(survey)
        scells = [(s, t, c) for s, t, c in cells if s == survey]
        for s, t, country in scells:
            code = cmap.get(country, country)
            country_data = data[data[ccol] == code].copy()
            pool = [c for c in country_data.columns if c not in {t, ccol}]
            oracle_cache: dict = {}   # k -> oracle_acc ; random_cache: k -> random_acc
            random_cache: dict = {}
            majority = {}
            rows = []  # per-cell, flushed at end of cell
            for cond in CONDITIONS:
                arm_specs = [("A", "", _arm_A_codes(selector_key, t, country, cond))]
                for dk in DISAMBIGUATORS:
                    arm_specs.append(("B", dk, _map_codes(map_dir, "B", dk, survey, t, country, cond)))
                    arm_specs.append(("C", dk, _map_codes(map_dir, "C", dk, survey, t, country, cond)))
                for arm, dk, codes in arm_specs:
                    if codes is None:
                        continue
                    for kspec in ["model"] + [f"k{k}" for k in FIXED_KS]:
                        kk = None if kspec == "model" else int(kspec[1:])
                        use_codes = [c for c in dict.fromkeys(codes) if c][: (kk or len(codes))]
                        use_codes = [c for c in use_codes if c in country_data.columns]
                        k = len(use_codes)
                        if k == 0:
                            continue
                        ci = captured_importance_df(use_codes, t, code, oracle_df, k=None)
                        try:
                            mres = evaluate_feature_set(country_data, t, use_codes, nthread=1)
                            m = mres.get("accuracy_mean"); majority[k] = mres.get("majority_baseline")
                            if k not in oracle_cache:
                                ores = evaluate_feature_set(country_data, t, oracle_topk(t, code, k), nthread=1)
                                oracle_cache[k] = ores.get("accuracy_mean")
                                draws = [single_random_draw(country_data, t, pool, k, 42 + i)
                                         for i in range(n_draws)]
                                draws = [d for d in draws if d is not None]
                                random_cache[k] = round(float(np.mean(draws)), 4) if draws else None
                            o = oracle_cache[k]; r = random_cache[k]
                        except Exception as e:
                            rows.append({"survey": survey, "target": t, "country": country,
                                         "condition": cond, "arm": arm, "disambiguator": dk,
                                         "k_spec": kspec, "error": f"{type(e).__name__}: {str(e)[:80]}"})
                            continue
                        rows.append({
                            "survey": survey, "target": t, "country": country, "condition": cond,
                            "arm": arm, "disambiguator": dk, "k_spec": kspec, "k": k,
                            "captured_importance": round(ci, 4) if ci is not None else "",
                            "oracle_acc": o, "model_acc": m, "random_acc": r,
                            "majority": majority.get(k),
                            "value_over_random": round(m - r, 4) if (m is not None and r is not None) else "",
                            "cost_of_imperfect": round(o - m, 4) if (o is not None and m is not None) else "",
                            "error": "",
                        })
            for r in rows:
                writer.writerow({c: r.get(c, "") for c in cols})
            out_f.flush()
            n_written += len(rows)
            print(f"  scored {s} {t} {country} (+{len(rows)} rows, {n_written} total)", flush=True)
    out_f.close()
    print(f"[score] wrote {n_written} rows -> {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["gen", "extract", "map", "score"], required=True)
    ap.add_argument("--selector", choices=list(SELECTORS), default=DEFAULT_SELECTOR,
                    help=f"test model whose capability is measured (default: {DEFAULT_SELECTOR})")
    ap.add_argument("--disambiguator", choices=list(DISAMBIGUATORS), help="required for --phase map")
    ap.add_argument("--arms", default="B,C", help="comma-separated arms for --phase map (default: B,C)")
    ap.add_argument("--force", action="store_true", help="recompute cells already on disk")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cells (smoke test)")
    args = ap.parse_args()

    if args.phase == "gen":
        phase_gen(args.selector, force=args.force, limit=args.limit)
    elif args.phase == "extract":
        phase_extract(args.selector, force=args.force, limit=args.limit)
    elif args.phase == "score":
        phase_score(args.selector, force=args.force, limit=args.limit)
    elif args.phase == "map":
        if not args.disambiguator:
            ap.error("--disambiguator required for --phase map")
        phase_map(args.selector, args.disambiguator, arms=tuple(args.arms.split(",")),
                  force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
