"""
Embedding-based retrieval: sentence-transformer cache + dual-embed candidate pools.

Every feature is embedded twice — label-only and "label: context" — and per-variable
similarity is the max of the two. Feeds ``mapping.map_features_with_subitems``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .config import DEFAULT_EMBEDDING_MODEL, OUTPUTS_DIR
from .layout import survey_emb_cache_path as _survey_emb_cache_path

_sent_trf_lock = threading.Lock()
_sent_trf_models: dict[str, object] = {}


def get_sentence_transformer(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load SentenceTransformer once per model name (not thread-safe to construct)."""
    with _sent_trf_lock:
        if model_name not in _sent_trf_models:
            from sentence_transformers import SentenceTransformer

            _sent_trf_models[model_name] = SentenceTransformer(model_name)
        return _sent_trf_models[model_name]


def build_embeddings(texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Embed a list of texts. Returns (n_texts, dim) array (normalized)."""
    model = get_sentence_transformer(model_name)
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def make_embed_fn(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Small callable: texts -> normalized embeddings (no progress bar)."""
    st = get_sentence_transformer(model_name)
    return lambda texts: st.encode(texts, normalize_embeddings=True)


def survey_emb_cache_path(
    survey_id: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Path:
    return _survey_emb_cache_path(survey_id, embedding_model, outputs_dir)


def load_or_build_survey_embeddings(
    survey_variables: dict[str, str],
    survey_id: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    outputs_dir: Path = OUTPUTS_DIR,
) -> tuple[np.ndarray, list[str]]:
    """Cached embeddings for all survey variables (one .npz per survey × model)."""
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


def retrieve_candidates(
    label: str,
    context: str,
    embed_fn,
    survey_embeddings,
    var_codes,
    survey_variables,
    excluded: set[str],
    top_n: int,
) -> list[dict]:
    """Top-N candidates by max(sim(label), sim(label+context))."""
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
    """Dual-embed many (label, context) queries in one encode call pair."""
    if not queries:
        return []
    labels = [lab for lab, _ in queries]
    combined = [f"{lab}: {ctx}" if ctx else lab for lab, ctx in queries]
    emb = np.asarray(embed_fn(labels + combined))
    n = len(queries)
    qlabels, qcombs = emb[:n], emb[n:]
    out: list[list[dict]] = []
    for i in range(n):
        sims = np.maximum(qlabels[i] @ survey_embeddings.T, qcombs[i] @ survey_embeddings.T)
        out.append(_pool_from_sims(sims, var_codes, survey_variables, excluded, top_n))
    return out


LEAKAGE_THRESHOLD = 0.85


def target_excluded_codes(
    target_code: str,
    survey_variables: dict[str, str],
    survey_embeddings: np.ndarray,
    var_codes: list[str],
    embed_fn,
    threshold: float = LEAKAGE_THRESHOLD,
) -> set[str]:
    """Target + near-paraphrases (>threshold cosine), matching the oracle's exclusion."""
    excluded = {target_code}
    text = survey_variables.get(target_code)
    if not text or threshold <= 0 or threshold >= 1:
        return excluded
    target_emb = np.asarray(embed_fn([text]))[0]
    sims = target_emb @ survey_embeddings.T
    excluded |= {vc for vc, s in zip(var_codes, sims) if s > threshold}
    return excluded
