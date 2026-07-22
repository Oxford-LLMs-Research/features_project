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
from .layout import sanitize_model_slug

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
    slug = sanitize_model_slug(embedding_model)
    return outputs_dir / f"survey_embeddings__{survey_id}__{slug}.npz"


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

def retrieve_candidates(label: str, context: str, embed_fn, survey_embeddings, var_codes,
                        survey_variables, excluded: set[str], top_n: int) -> list[dict]:
    """Top-N candidates by max(sim(label), sim(label+context)) — dual embed."""
    qlabel = embed_fn([label])[0]
    combined = f"{label}: {context}" if context else label
    qcomb = embed_fn([combined])[0]
    sims = np.maximum(qlabel @ survey_embeddings.T, qcomb @ survey_embeddings.T)
    order = np.argsort(sims)[::-1]
    pool = []
    for idx in order:
        vc = var_codes[idx]
        if vc in excluded:
            continue
        pool.append({"var_code": vc, "question_text": survey_variables[vc],
                     "similarity": float(sims[idx])})
        if len(pool) >= top_n:
            break
    return pool


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
