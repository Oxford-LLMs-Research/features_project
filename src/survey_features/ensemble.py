"""
Ensemble retrieval labels and fusion defaults (shared by runner + analysis).

Fusion rule v1: per-model top_n + min_similarity, then union by var_code with
similarity = max; fused pool capped at ``max_fused_mult * top_n`` (default 2×).
See docs/ensemble_mapping.md.
"""

from __future__ import annotations

from .config import DEFAULT_EMBEDDING_MODEL
from .layout import sanitize_model_slug

FUSION_RULE = "union_max_sim"
DEFAULT_ENSEMBLE_MODELS = (DEFAULT_EMBEDDING_MODEL, "all-mpnet-base-v2")
DEFAULT_TOP_N = 20
DEFAULT_MIN_SIM = 0.30
DEFAULT_MAX_FUSED_MULT = 2


def fusion_slug(models: list[str], rule: str = FUSION_RULE) -> str:
    shorts = []
    for m in models:
        s = sanitize_model_slug(m).lower()
        if "minilm" in s:
            shorts.append("minilm")
        elif "mpnet" in s:
            shorts.append("mpnet")
        elif "roberta" in s:
            shorts.append("roberta")
        else:
            shorts.append(s[:16])
    return sanitize_model_slug(f"{rule}_{'_'.join(shorts)}")


def ensemble_label(models: list[str], rule: str = FUSION_RULE) -> str:
    return f"ensemble_{rule}:{ '+'.join(models)}"
