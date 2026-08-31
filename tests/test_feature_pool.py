"""Near-duplicate exclusion should reuse the process MiniLM, not reload per cell."""

from __future__ import annotations

import numpy as np
import pandas as pd

from survey_features.config import DEFAULT_EMBEDDING_MODEL
from survey_features.feature_pool import build_feature_pool
from survey_features.oracle import load_similarity_model


class _CountingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts, show_progress_bar=False):
        self.calls.append(list(texts))
        return np.stack([np.full(4, float(len(t)), dtype=np.float32) for t in texts])


def test_load_similarity_model_reuses_retrieval_cache(monkeypatch):
    calls: list[str] = []
    sentinel = object()

    def fake_get(model_name: str):
        calls.append(model_name)
        return sentinel

    monkeypatch.setattr(
        "survey_features.retrieval.get_sentence_transformer", fake_get
    )
    first = load_similarity_model(0.85)
    second = load_similarity_model(0.85)
    assert first is sentinel
    assert second is sentinel
    assert calls == [DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL]


def test_load_similarity_model_disabled_for_extreme_thresholds():
    assert load_similarity_model(0.0) is None
    assert load_similarity_model(1.0) is None


def test_candidate_embeddings_are_cached_across_cells():
    model = _CountingEncoder()
    metadata = {
        "T": {"question": "target question"},
        "A": {"question": "alpha item"},
        "B": {"question": "beta item"},
    }
    df = pd.DataFrame({"T": [1, 0], "A": [1, 1], "B": [0, 1]})

    build_feature_pool(df, metadata, "T", frozenset(), model, 0.85)
    first_encoded = [t for call in model.calls for t in call]
    assert "alpha item" in first_encoded
    assert "beta item" in first_encoded

    model.calls.clear()
    build_feature_pool(df, metadata, "T", frozenset(), model, 0.85)
    assert model.calls == []


def test_new_target_only_encodes_uncached_text():
    model = _CountingEncoder()
    metadata = {
        "T": {"question": "target question"},
        "U": {"question": "other target"},
        "A": {"question": "alpha item"},
    }
    df_t = pd.DataFrame({"T": [1, 0], "A": [1, 1]})
    df_u = pd.DataFrame({"U": [0, 1], "T": [1, 0], "A": [1, 1]})

    build_feature_pool(df_t, metadata, "T", frozenset(), model, 0.85)
    model.calls.clear()
    build_feature_pool(df_u, metadata, "U", frozenset(), model, 0.85)
    second_encoded = [t for call in model.calls for t in call]
    assert second_encoded == ["other target"]
