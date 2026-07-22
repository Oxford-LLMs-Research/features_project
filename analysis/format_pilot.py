"""
Second pilot orchestrator — format-as-condition + fixed-k (2026-06-03).

Tests whether strict-JSON output suppresses measured LLM capability vs free-text, and
reports at fixed k as well as model-chosen k. Selector (test model) = DeepSeek-V3.2.
Grid = the 52 GENUINE cells from the B1 leakage audit, both prompt conditions.

Three arms (decompose format vs mapper):
  A = JSON  + pilot-1 per-feature top-5 disambig   (already on disk; not recomputed here)
  B = JSON  + new per-feature mapper                (re-maps cached pilot-1 JSON outputs)
  C = free-text + new per-feature mapper            (new free-text generation)
Mappers (held fixed per pass, run BOTH): Nemotron-3-Nano-30B (small), Qwen3-235B (large).
Contrasts: C-B = format effect; B-A = mapper effect; C-A = total.

Phased + resumable:
  --phase gen   : generate & cache free-text responses (DeepSeek only). Cheap, irreversible.
  --phase map   : run mapping arms B & C for a given --mapper, per-feature, checkpointed.
  --phase score : (added later) compute captured importance / value-over-random at
                  model-chosen k and fixed k=5,10 per arm x mapper, write tables.

All artifacts under outputs/format_pilot/. Per-cell JSON checkpoints make every phase
resumable; rerunning skips cells already on disk unless --force.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "outputs"
PILOT = OUT / "format_pilot"

# Selector = the test model whose capability we measure. Multiple selectors are kept in
# separate subdirs (selector key) so adding a model never clobbers another's artifacts.
# The pilot-1 cache tag for each selector (its llm__<tag>/ dir) gives arms A and B.
SELECTORS = {
    "deepseek": {"model": "deepseek-ai/DeepSeek-V3.2", "pilot1_tag": "deepseek-ai_DeepSeek-V3.2"},
    "kimi":     {"model": "moonshotai/Kimi-K2.5",      "pilot1_tag": "moonshotai_Kimi-K2.5"},
}
DEFAULT_SELECTOR = "deepseek"


def selector_dirs(selector_key):
    base = PILOT / selector_key
    return base / "freetext", base / "extracted", base / "maps"
# Extraction (essay -> feature list) is a comprehension task held FIXED across all arms
# and disambiguators: a small model cannot digest a long essay (demo: Nemotron pulled 3
# features vs Qwen's 28 from the same response), and the extracted set defines the model's
# request, so it must not vary by disambiguator. Disambiguation (feature -> code/none) is
# the per-feature matching task where small-vs-large is a legitimate comparison.
EXTRACTOR_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DISAMBIGUATORS = {
    "nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "qwen235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
}
CONDITIONS = ["unprompted", "country_provided"]
# Feature types that enter retrieve+disambiguate+score. Decision (2026-06-03): include
# temporal_contextual alongside respondent_attribute — the study is about capability across
# countries AND time (surveys span waves), so be generous to temporally-framed requests.
# instrument_methodology and base_rate_prior stay out (unmappable; studied as metadata).
PIPE_TYPES = {"respondent_attribute", "temporal_contextual"}

# Free-text prompts are the EXACT JSON prompts (phase0b_pipeline.PROMPT_*) with only the
# JSON formatting block removed --- nothing added, nothing rephrased. This isolates the
# output-format instruction as the single variable between the JSON and free-text arms, so
# the C - B contrast is a clean format effect (same system prompt, stem, question wording).
FREETEXT_SYSTEM = "You are a social science researcher."  # == phase0b_pipeline.SYSTEM_PROMPT
FREETEXT_UNPROMPTED = (
    'A survey asks respondents: "{question}"\n\n'
    "You want to predict how a respondent will answer. What information about the respondent would you need?"
)
FREETEXT_COUNTRY = (
    'A survey asks respondents in {country}: "{question}"\n\n'
    "You want to predict how a respondent in {country} will answer. What information about the respondent would you need?"
)


# ── shared loaders ────────────────────────────────────────────────────────────

def genuine_cells() -> list[tuple[str, str, str]]:
    import csv
    out = []
    with open(OUT / "leakage_audit.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["leakage_class"] == "genuine":
                out.append((r["survey"], r["target"], r["country"]))
    return out


def env_values() -> dict:
    from dotenv import dotenv_values
    return dotenv_values(ROOT / ".env")


def selector_generate_fn(env, selector_model):
    from openai import OpenAI
    client = OpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"])

    def g(messages, max_tokens=4096, temperature=0.0, usage_phase=None):
        import time as _t
        last = None
        for attempt in range(5):
            try:
                r = client.chat.completions.create(model=selector_model, messages=messages,
                                                   max_tokens=max_tokens, temperature=temperature)
                return r.choices[0].message.content or ""
            except Exception as e:
                last = e
                _t.sleep(2 * (attempt + 1))
        print(f"[selector] giving up after retries: {type(last).__name__}: {str(last)[:100]}")
        return ""
    return g


def mapper_generate_fn(env, mapper_model):
    from openai import OpenAI
    # Both mappers live on the Studio (disambig) endpoint.
    client = OpenAI(base_url=env["DISAMBIG_BASE_URL"], api_key=env.get("DISAMBIG_API_KEY") or env["LLM_API_KEY"])

    def g(messages, max_tokens=2048, temperature=0.0, usage_phase=None):
        # Retry transient server errors (e.g. Nebius "Already borrowed" 400, 429s, 5xx)
        # so one flaky call cannot abort a long sweep. Last attempt re-raises only if all
        # retries fail; on persistent failure we return "" (treated as no-mapping) rather
        # than crash, keeping the checkpointed run alive.
        import time as _t
        last = None
        for attempt in range(5):
            try:
                r = client.chat.completions.create(model=mapper_model, messages=messages,
                                                   max_tokens=max_tokens, temperature=temperature)
                return r.choices[0].message.content or ""
            except Exception as e:
                last = e
                _t.sleep(2 * (attempt + 1))
        print(f"[mapper] giving up after retries: {type(last).__name__}: {str(last)[:100]}")
        return ""
    return g


_survey_cache: dict = {}


def survey_assets(survey_id, env):
    """(survey_variables, embeddings, var_codes) — cached per survey."""
    if survey_id not in _survey_cache:
        from run_grid import load_survey, load_or_build_survey_embeddings
        from phase0b_mapping import extract_survey_variables
        _, meta = load_survey(survey_id, env["DATA_CONFIG_PATH"])
        svars = extract_survey_variables(meta)
        emb, vcodes = load_or_build_survey_embeddings(svars, survey_id)
        _survey_cache[survey_id] = (svars, emb, vcodes)
    return _survey_cache[survey_id]


def embed_fn_for():
    from phase0b_mapping import _get_sentence_transformer
    st = _get_sentence_transformer("all-MiniLM-L6-v2")
    return lambda texts: st.encode(texts, normalize_embeddings=True)


def cell_tag(survey, target, country):
    safe = f"{survey}__{target}__{country}".replace("/", "_").replace(" ", "_")
    return safe


# ── Phase: gen (free-text, DeepSeek, cheap, irreversible) ─────────────────────

def phase_gen(selector_key, force=False, limit=None):
    env = env_values()
    sel_model = SELECTORS[selector_key]["model"]
    gen_dir, _, _ = selector_dirs(selector_key)
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen = selector_generate_fn(env, sel_model)
    cells = genuine_cells()
    if limit:
        cells = cells[:limit]
    print(f"[gen] selector={selector_key} ({sel_model}) {len(cells)} cells x {len(CONDITIONS)} conds -> free-text")
    done = skipped = 0
    for i, (survey, target, country) in enumerate(cells):
        svars, _, _ = survey_assets(survey, env)
        qtext = svars.get(target, target)
        out_path = gen_dir / f"{cell_tag(survey, target, country)}.json"
        if out_path.is_file() and not force:
            skipped += 1
            continue
        rec = {"survey": survey, "target": target, "country": country,
               "question_text": qtext, "selector_model": sel_model, "responses": {}}
        for cond in CONDITIONS:
            tmpl = FREETEXT_COUNTRY if cond == "country_provided" else FREETEXT_UNPROMPTED
            msg = tmpl.format(question=qtext, country=country)
            for attempt in range(3):
                try:
                    resp = gen([{"role": "system", "content": FREETEXT_SYSTEM},
                                {"role": "user", "content": msg}], max_tokens=4096)
                    break
                except Exception as e:
                    if attempt == 2:
                        resp = ""
                        rec.setdefault("errors", {})[cond] = f"{type(e).__name__}: {str(e)[:120]}"
                    else:
                        time.sleep(2 * (attempt + 1))
            rec["responses"][cond] = resp
        out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        done += 1
        print(f"  [{i+1}/{len(cells)}] {survey} {target} {country}: "
              f"up={len(rec['responses'].get('unprompted',''))}c cp={len(rec['responses'].get('country_provided',''))}c")
    print(f"[gen] done. wrote={done} skipped(existing)={skipped} -> {gen_dir}")



# ── Phase: extract (free-text -> feature list, FIXED Qwen extractor) ──────────

def phase_extract(selector_key, force=False, limit=None):
    """Arm C only: turn each cached free-text essay into a typed feature list via the FIXED
    extractor (Qwen). Held constant across disambiguators so the request set does not vary
    by mapper. Arm B needs no extraction (JSON is already a feature list)."""
    from analysis.smart_mapper import extract_features
    env = env_values()
    egen = mapper_generate_fn(env, EXTRACTOR_MODEL)
    gen_dir, extract_dir, _ = selector_dirs(selector_key)
    cells = genuine_cells()
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
    """Arm B: the selector's pilot-1 JSON selection as a feature list (parsed directly, no
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
    from analysis.smart_mapper import map_features
    env = env_values()
    dmodel = DISAMBIGUATORS[disambig_key]
    dgen = mapper_generate_fn(env, dmodel)
    embed = embed_fn_for()
    _, extract_dir, map_dir = selector_dirs(selector_key)
    cells = genuine_cells()
    if limit:
        cells = cells[:limit]
    map_dir.mkdir(parents=True, exist_ok=True)
    print(f"[map] selector={selector_key} disambiguator={disambig_key} arms={arms} cells={len(cells)}")

    for i, (survey, target, country) in enumerate(cells):
        svars, emb, vcodes = survey_assets(survey, env)
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

FIXED_KS = [5, 10]


def _oracle_table(survey, env):
    """Load oracle importances for a survey from per-cell oracle.csv into one DataFrame
    with columns [target_variable, country, feature_variable, importance_mean]. country is
    the numeric/code value matching the survey's country column (as run_comparison expects)."""
    import pandas as pd
    from run_grid import SURVEY_COUNTRY_COL, load_survey, build_country_code_map
    _, meta = load_survey(survey, env["DATA_CONFIG_PATH"])
    ccol = SURVEY_COUNTRY_COL.get(survey)
    data, _ = survey_data(survey, env)
    cmap = build_country_code_map(meta, ccol, data) if ccol else {}
    rows = []
    for s, t, c in genuine_cells():
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


_data_cache: dict = {}


def survey_data(survey, env):
    if survey not in _data_cache:
        from run_grid import load_survey
        _data_cache[survey] = load_survey(survey, env["DATA_CONFIG_PATH"])
    return _data_cache[survey]


def captured_importance(mapped_codes, target, country_code, oracle_df, k=None):
    """Sum oracle importance of mapped codes / sum of oracle top-k (matched k). In [0,1]."""
    sub = oracle_df[(oracle_df["target_variable"] == target) & (oracle_df["country"] == country_code)]
    if sub.empty:
        return None
    imp = dict(zip(sub["feature_variable"].astype(str),
                   sub["importance_mean"].clip(lower=0)))
    codes = [c for c in dict.fromkeys(mapped_codes) if c]  # dedupe, drop None
    if k is not None:
        codes = codes[:k]
    kk = len(codes)
    if kk == 0:
        return None
    denom = sum(sorted(imp.values(), reverse=True)[:kk])
    if denom <= 0:
        return None
    return sum(imp.get(c, 0.0) for c in codes) / denom


def _arm_A_codes(selector_key, target, country, cond):
    """Pilot-1 mapped codes (deduped, arrival order) for arm A from the selector's cached
    disambig.json."""
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
    """Arm B/C mapped codes from the format-pilot map file."""
    ctag = cell_tag(survey, target, country)
    p = map_dir / f"{arm}__{disambig_key}__{ctag}__{cond}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("mapped_codes", [])


def phase_score(selector_key, force=False, limit=None):
    """Score all arms. Key efficiency: oracle top-k and the random-k baseline depend only
    on (cell, k) — NOT on the arm or disambiguator — so we compute them ONCE per (cell, k)
    and reuse, evaluating only the cheap per-arm model feature set each time. This avoids
    the ~10x redundant XGBoost fits the naive arm-loop incurred (the random-20-draw baseline
    on the full column matrix is the dominant cost). All XGBoost is single-threaded
    (nthread=1): torch/sentence-transformers already loaded a conflicting libomp earlier in
    the process (documented Windows issue) and multi-threaded fits hang."""
    import csv as _csv
    import numpy as np
    from phase0b_evaluation import evaluate_feature_set, _single_random_draw
    env = env_values()
    # SERIAL random draws. joblib/loky parallelism is unreliable in this environment: the
    # worker pool crashes (TerminatedWorkerError, OOM from re-pickling the full survey frame
    # per call) after a few invocations on Windows, which is what stalled earlier runs.
    # Serial is slow but deterministic; we cut n_draws to 10 (from pilot-1's 20) to keep the
    # full run ~2h. Each XGBoost fit is single-threaded (libomp conflict). Oracle+random are
    # computed once per (cell,k) and cached across arms (the real efficiency win).
    n_draws = int(env.get("SCORE_N_DRAWS", "10"))
    _, _, map_dir = selector_dirs(selector_key)
    cells = genuine_cells()
    if limit:
        cells = cells[:limit]
    PILOT.mkdir(parents=True, exist_ok=True)
    out_csv = PILOT / f"scores_{selector_key}.csv"
    cols = ["survey","target","country","condition","arm","disambiguator","k_spec","k",
            "captured_importance","oracle_acc","model_acc","random_acc","majority",
            "value_over_random","cost_of_imperfect","error"]
    # Incremental write: open now, flush after each cell so progress is visible and a
    # mid-run interruption keeps completed cells (no joblib pool to crash; pure serial).
    out_f = open(out_csv, "w", newline="", encoding="utf-8")
    writer = _csv.DictWriter(out_f, fieldnames=cols)
    writer.writeheader(); out_f.flush()
    n_written = 0

    def oracle_topk(t, code, k):
        sub = oracle_df[(oracle_df["target_variable"] == t) & (oracle_df["country"] == code)]
        return sub.sort_values("importance_mean", ascending=False)["feature_variable"].head(k).tolist()

    surveys = sorted({s for s, _, _ in cells})
    for survey in surveys:
        oracle_df, ccol, cmap = _oracle_table(survey, env)
        data, _ = survey_data(survey, env)
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
                        ci = captured_importance(use_codes, t, code, oracle_df, k=None)
                        try:
                            mres = evaluate_feature_set(country_data, t, use_codes, nthread=1)
                            m = mres.get("accuracy_mean"); majority[k] = mres.get("majority_baseline")
                            if k not in oracle_cache:
                                ores = evaluate_feature_set(country_data, t, oracle_topk(t, code, k), nthread=1)
                                oracle_cache[k] = ores.get("accuracy_mean")
                                draws = [_single_random_draw(country_data, t, pool, k, 42 + i)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "extract", "map", "score"], required=True)
    ap.add_argument("--selector", choices=list(SELECTORS), default=DEFAULT_SELECTOR,
                    help="test model whose capability is measured (default: deepseek)")
    ap.add_argument("--disambiguator", choices=list(DISAMBIGUATORS), help="required for --phase map")
    ap.add_argument("--arms", default="B,C")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
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
