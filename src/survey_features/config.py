"""
Project paths, environment loading, and the model registry.

ROOT resolution: the package lives at <repo>/src/survey_features, so the repo root is
two levels above this file. Set SURVEY_FEATURES_ROOT to override (e.g. if the package
is installed as a wheel outside a source checkout).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.environ.get("SURVEY_FEATURES_ROOT") or Path(__file__).resolve().parents[2])

# .env at the repo root holds LLM endpoints/keys and DATA_CONFIG_PATH (see .env.example).
load_dotenv(ROOT / ".env")

OUTPUTS_DIR = ROOT / "outputs"


# ── Model registry ────────────────────────────────────────────────────────────
# Selector = the test model whose feature-selection capability we measure.
# Each selector keeps its artifacts in a separate subdir (selector key) so adding a
# model never clobbers another's. pilot1_tag = the llm__<tag>/ cache dir of the
# legacy JSON (pilot-1) run for the same model, used for arms A/B of the format pilot.
SELECTORS: dict[str, dict[str, str]] = {
    "deepseek": {"model": "deepseek-ai/DeepSeek-V3.2", "pilot1_tag": "deepseek-ai_DeepSeek-V3.2"},
    "kimi":     {"model": "moonshotai/Kimi-K2.5",      "pilot1_tag": "moonshotai_Kimi-K2.5"},
}
DEFAULT_SELECTOR = "deepseek"

# Extraction (free-text essay -> feature list) is a comprehension task held FIXED across
# all arms and disambiguators: a small model cannot digest a long essay, and the extracted
# set defines the model's request, so it must not vary by disambiguator.
EXTRACTOR_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# Disambiguation (feature -> code/none) is the per-feature matching task where
# small-vs-large is a legitimate comparison. nemotron is the main-experiment choice.
DISAMBIGUATORS: dict[str, str] = {
    "nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "qwen235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
}

# Fixed small model used for disambiguation in the legacy JSON grid (run_grid.py).
# Keeping this constant ensures differences in final mapped variables are attributable
# only to feature-selection quality, not disambiguation quality.
DISAMBIG_MODEL = os.environ.get("DISAMBIG_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
DISAMBIG_BASE_URL = os.environ.get("DISAMBIG_BASE_URL") or None  # falls back to LLM_BASE_URL
DISAMBIG_API_KEY = os.environ.get("DISAMBIG_API_KEY") or None    # falls back to LLM_API_KEY

# The two prompt conditions run for every cell.
CONDITIONS = ["unprompted", "country_provided"]

# Feature types that enter retrieve+disambiguate+score. Decision (2026-06-03): include
# temporal_contextual alongside respondent_attribute — the study is about capability across
# countries AND time (surveys span waves), so be generous to temporally-framed requests.
# instrument_methodology and base_rate_prior stay out (unmappable; studied as metadata).
PIPE_TYPES = {"respondent_attribute", "temporal_contextual"}
