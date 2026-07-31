"""
Embedding-based retrieval: sentence-transformer cache, survey-variable embedding cache,
and dual-embed candidate retrieval.

Dual-embed: every feature is embedded twice — label-only and "label: context" — and the
per-variable similarity is the max of the two. This prevents reasoning text from pulling
short labels (e.g. "age") into the target's topic domain instead of matching the
demographic variable.

CURRENT path: ``retrieve_candidates`` — per-feature top-N (default 20) for the
per-feature disambiguator (survey_features.disambig.map_features).

LEGACY path: ``map_features_to_variables`` — batch top-k (default 5) with target/leakage
exclusion, feeding the pilot-1 shortlist disambiguator (scripts/run_grid.py).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .config import DEFAULT_EMBEDDING_MODEL, OUTPUTS_DIR
from .layout import survey_emb_cache_path as _survey_emb_cache_path

# SentenceTransformer is not thread-safe to construct concurrently (multiple grid
# workers can hit meta-tensor / device race). Load once per model name under lock.
_sent_trf_lock = threading.Lock()
_sent_trf_models: dict[str, object] = {}


def get_sentence_transformer(model_name: str = DEFAULT_EMBEDDING_MODEL):
    with _sent_trf_lock:
        if model_name not in _sent_trf_models:
            from sentence_transformers import SentenceTransformer

            _sent_trf_models[model_name] = SentenceTransformer(model_name)
        return _sent_trf_models[model_name]


# Backwards-compatible alias (old name in phase0b_mapping.py).
_get_sentence_transformer = get_sentence_transformer


def build_embeddings(texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Embed a list of texts. Returns (n_texts, dim) array (normalized)."""
    model = get_sentence_transformer(model_name)
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def make_embed_fn(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Small callable: texts -> normalized embeddings (no progress bar)."""
    st = get_sentence_transformer(model_name)
    return lambda texts: st.encode(texts, normalize_embeddings=True)


# ── Survey-variable embedding cache ───────────────────────────────────────────

def survey_emb_cache_path(
    survey_id: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Path:
    """Dual-resolve cache/embeddings/ then legacy outputs/ root (via layout)."""
    return _survey_emb_cache_path(survey_id, embedding_model, outputs_dir)


def load_or_build_survey_embeddings(
    survey_variables: dict[str, str],
    survey_id: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    outputs_dir: Path = OUTPUTS_DIR,
) -> tuple[np.ndarray, list[str]]:
    """Cached embeddings for all survey variables (one .npz per survey × embedding model)."""
    var_codes = list(survey_variables.keys())
    var_texts = list(survey_variables.values())

    cache_path = survey_emb_cache_path(survey_id, embedding_model, outputs_dir)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_codes = list(cached["var_codes"])
        if cached_codes == var_codes:
            print(f"  Loaded cached embeddings ({len(var_codes)} vars) from {cache_path}")
            return cached["embeddings"], var_codes
        print(f"  Cached embeddings at {cache_path} are stale (var_codes mismatch); recomputing.")

    embeddings = build_embeddings(var_texts, model_name=embedding_model)
    np.savez(cache_path, embeddings=embeddings, var_codes=np.array(var_codes, dtype=object))
    print(f"  Saved embeddings to {cache_path}")
    return embeddings, var_codes


# ── CURRENT: per-feature retrieval (top-N pool for per-feature disambiguation) ─

def _pool_from_sims(
    sims: np.ndarray,
    var_codes: list[str],
    survey_variables: dict[str, str],
    excluded: set[str],
    top_n: int,
) -> list[dict]:
    order = np.argsort(sims)[::-1]
    pool: list[dict] = []
    for idx in order:
        vc = var_codes[idx]
        if vc in excluded:
            continue
        pool.append({
            "var_code": vc,
            "question_text": survey_variables[vc],
            "similarity": float(sims[idx]),
        })
        if len(pool) >= top_n:
            break
    return pool


def retrieve_candidates(label: str, context: str, embed_fn, survey_embeddings, var_codes,
                        survey_variables, excluded: set[str], top_n: int) -> list[dict]:
    """Top-N candidates by max(sim(label), sim(label+context)) — dual embed."""
    pools = retrieve_candidates_batch(
        [(label, context)],
        embed_fn,
        survey_embeddings,
        var_codes,
        survey_variables,
        excluded,
        top_n,
    )
    return pools[0] if pools else []


def retrieve_candidates_batch(
    queries: list[tuple[str, str]],
    embed_fn,
    survey_embeddings: np.ndarray,
    var_codes: list[str],
    survey_variables: dict[str, str],
    excluded: set[str],
    top_n: int,
) -> list[list[dict]]:
    """Dual-embed many (label, context) queries in one encode call pair.

    Embeds all labels then all ``label: context`` strings in two batched
    ``embed_fn`` calls (or one concatenated call), then ranks each query.
    """
    if not queries:
        return []
    labels = [lab for lab, _ in queries]
    combined = [f"{lab}: {ctx}" if ctx else lab for lab, ctx in queries]
    # One encode for all query texts (label block + combined block).
    emb = np.asarray(embed_fn(labels + combined))
    n = len(queries)
    qlabels, qcombs = emb[:n], emb[n:]
    out: list[list[dict]] = []
    for i in range(n):
        sims = np.maximum(qlabels[i] @ survey_embeddings.T, qcombs[i] @ survey_embeddings.T)
        out.append(_pool_from_sims(sims, var_codes, survey_variables, excluded, top_n))
    return out


# ── Ensemble retrieval (union of per-model pools → one disambig pool) ─────────

def fuse_candidate_pools(
    pools: list[list[dict]],
    *,
    max_pool: int,
) -> list[dict]:
    """Union candidate pools by ``var_code``; ``similarity`` = max across models.

    Sort by similarity descending, then truncate to ``max_pool``. Sources are
    recorded on each candidate as ``sources`` (list of embedding-model labels)
    when present on inputs as ``source`` / ``sources``.
    """
    by_code: dict[str, dict] = {}
    for pool in pools:
        for c in pool:
            vc = c["var_code"]
            src = c.get("source")
            sources = list(c.get("sources") or ([] if src is None else [src]))
            if vc not in by_code:
                by_code[vc] = {
                    "var_code": vc,
                    "question_text": c["question_text"],
                    "similarity": float(c["similarity"]),
                    "sources": list(sources),
                }
                continue
            cur = by_code[vc]
            cur["similarity"] = max(cur["similarity"], float(c["similarity"]))
            for s in sources:
                if s and s not in cur["sources"]:
                    cur["sources"].append(s)
    fused = sorted(by_code.values(), key=lambda x: x["similarity"], reverse=True)
    if max_pool > 0:
        fused = fused[:max_pool]
    return fused


def retrieve_ensemble_candidates_batch(
    queries: list[tuple[str, str]],
    model_packs: list[dict],
    survey_variables: dict[str, str],
    excluded: set[str],
    top_n: int = 20,
    min_similarity: float = 0.30,
    max_fused: int | None = None,
) -> tuple[list[list[dict]], list[float]]:
    """Per-model retrieve → threshold → union fuse (max sim), capped at ``max_fused``.

    ``model_packs`` items: ``{name, embed_fn, survey_embeddings, var_codes}``.
    Threshold is applied **per model** before fusion. Default ``max_fused = 2 * top_n``
    so unique candidates from a second model are not discarded by re-truncating to
    single-model top_n (see docs/ensemble_mapping.md).

    Returns ``(fused_pools_per_query, retrieve_wall_s_per_model)``.
    """
    import time

    if not queries:
        return [], []
    cap = int(max_fused) if max_fused is not None else int(2 * top_n)
    per_model_pools: list[list[list[dict]]] = []
    retrieve_times: list[float] = []
    for pack in model_packs:
        t0 = time.perf_counter()
        pools = retrieve_candidates_batch(
            queries,
            pack["embed_fn"],
            pack["survey_embeddings"],
            pack["var_codes"],
            survey_variables,
            excluded,
            top_n,
        )
        name = pack.get("name") or ""
        filtered = []
        for pool in pools:
            kept = []
            for c in pool:
                if c["similarity"] < min_similarity:
                    continue
                kept.append({
                    "var_code": c["var_code"],
                    "question_text": c["question_text"],
                    "similarity": float(c["similarity"]),
                    "source": name,
                })
            filtered.append(kept)
        retrieve_times.append(time.perf_counter() - t0)
        per_model_pools.append(filtered)

    fused: list[list[dict]] = []
    n_q = len(queries)
    for i in range(n_q):
        pools_i = [per_model_pools[m][i] for m in range(len(model_packs))]
        fused.append(fuse_candidate_pools(pools_i, max_pool=cap))
    return fused, retrieve_times


# ── LEGACY: batch retrieval (pilot-1 top-5 shortlist) ─────────────────────────

def map_features_to_variables(
    results: list[dict],
    survey_variables: dict[str, str],
    survey_embeddings: np.ndarray,
    var_codes: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
    min_threshold: float = 0.3,
    exclude_targets: bool = True,
    leakage_threshold: float = 0.85,
) -> list[dict]:
    """
    Map JSON-elicited feature descriptions to survey variables via cosine similarity.

    Args:
        results: list of result dicts from elicitation.run_batch / run_single
        survey_variables: {var_code: question_text}
        survey_embeddings: pre-computed embeddings for survey variables
        var_codes: ordered list of var codes matching survey_embeddings rows
        model_name: embedding model (must match survey_embeddings)
        top_k: number of candidate matches to return per feature
        min_threshold: minimum similarity to include a candidate
        exclude_targets: if True, exclude the target variable from candidates
        leakage_threshold: exclude candidates with similarity > this to the
            target question text (prevents near-duplicate matches). Set None to disable.

    Returns:
        List of mapping dicts, one per feature across all results.
    """
    model = get_sentence_transformer(model_name)

    # Pre-compute target question embeddings for leakage filtering
    target_codes = set(r["target"] for r in results if r["features"])
    target_embeddings = {}
    if leakage_threshold is not None:
        for tc in target_codes:
            if tc in survey_variables:
                target_embeddings[tc] = model.encode(
                    [survey_variables[tc]], normalize_embeddings=True
                )[0]

    mappings = []

    for r in results:
        if not r["features"]:
            continue

        target_var = r["target"]

        # Build exclusion set: target itself + semantically leaked variables
        excluded_codes = set()
        if exclude_targets:
            excluded_codes.add(target_var)

        if leakage_threshold is not None and target_var in target_embeddings:
            target_emb = target_embeddings[target_var]
            target_sims = target_emb @ survey_embeddings.T
            for j, vc in enumerate(var_codes):
                if target_sims[j] > leakage_threshold:
                    excluded_codes.add(vc)

        # Embed label-only and label+reasoning separately, take max similarity (dual embed).
        label_texts = []
        combined_texts = []
        for f in r["features"]:
            label = f.get("feature", "")
            reasoning = f.get("reasoning", "")
            label_texts.append(label)
            combined_texts.append(f"{label}: {reasoning}" if reasoning else label)

        label_emb = model.encode(label_texts, normalize_embeddings=True)
        combined_emb = model.encode(combined_texts, normalize_embeddings=True)

        sims_label = label_emb @ survey_embeddings.T
        sims_combined = combined_emb @ survey_embeddings.T
        sims = np.maximum(sims_label, sims_combined)  # (n_features, n_variables)

        for i, f in enumerate(r["features"]):
            # Sort all candidates, then filter
            sorted_indices = np.argsort(sims[i])[::-1]
            candidates = []
            for idx in sorted_indices:
                if len(candidates) >= top_k:
                    break
                vc = var_codes[idx]
                if vc in excluded_codes:
                    continue
                score = float(sims[i, idx])
                if score < min_threshold:
                    break  # sorted descending, so no more above threshold
                candidates.append({
                    "var_code": vc,
                    "question_text": survey_variables[vc],
                    "similarity": round(score, 4),
                })

            mappings.append({
                "target": r["target"],
                "country": r["country"],
                "condition": r["condition"],
                "model": r["model"],
                "feature_label": f.get("feature", ""),
                "feature_reasoning": f.get("reasoning", ""),
                "feature_rank": i,
                "candidates": candidates,
                "top_match_code": candidates[0]["var_code"] if candidates else None,
                "top_match_score": candidates[0]["similarity"] if candidates else None,
            })

    n_empty = sum(1 for m in mappings if not m["candidates"])
    if n_empty and mappings:
        print(
            f"  [map_features_to_variables] {n_empty}/{len(mappings)} rows have zero candidates "
            f"(min_threshold={min_threshold}; target + leakage exclusions apply)"
        )

    return mappings


def print_mapping_summary(mappings: list[dict], ground_truth: dict[str, list[str]] = None):
    """
    Print a readable summary of legacy mappings.

    Args:
        mappings: output of map_features_to_variables
        ground_truth: optional {target: [top_var_codes]} for comparison
    """
    current_key = None
    for m in mappings:
        key = (m["target"], m["country"], m["condition"])
        if key != current_key:
            current_key = key
            country_str = m["country"] or "unprompted"
            print(f"\n{'='*70}")
            print(f"{m['target']} | {country_str} | {m['condition']}")
            if ground_truth and m["target"] in ground_truth:
                print(f"  Ground truth top: {ground_truth[m['target']]}")
            print(f"{'='*70}")

        top = m["candidates"][0] if m["candidates"] else None
        hit_marker = ""
        if ground_truth and m["target"] in ground_truth and top:
            if top["var_code"] in ground_truth[m["target"]]:
                hit_marker = " *** HIT ***"

        if top:
            print(f"  [{m['feature_rank']}] {m['feature_label']}")
            print(f"      -> {top['var_code']} ({top['question_text'][:60]}) sim={top['similarity']:.3f}{hit_marker}")
        else:
            print(f"  [{m['feature_rank']}] {m['feature_label']} -> NO MATCH above threshold")
