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

# Live artifact root: shared Dropbox folder `features_project/outputs`, set by
# SURVEY_FEATURES_OUTPUTS in gitignored .env. The <repo>/outputs fallback is only
# for a checkout that has not been pointed at Dropbox.
# Fail loud on the .env.example placeholder: a copied-but-unedited .env would
# otherwise resolve to a garbage drive-relative path and writers would silently
# mkdir + populate it.
_outputs_env = os.environ.get("SURVEY_FEATURES_OUTPUTS")
if _outputs_env and "path/to" in _outputs_env.replace("\\", "/").lower():
    raise RuntimeError(
        "SURVEY_FEATURES_OUTPUTS in .env still holds the .env.example placeholder "
        f"({_outputs_env!r}). Edit .env to point at the shared Dropbox outputs "
        "folder (README § Outputs), or remove the line to fall back to <repo>/outputs."
    )
OUTPUTS_DIR = Path(_outputs_env or (ROOT / "outputs"))

# Paper / writing workspace (gitignored): LaTeX, figures, memos, local builders.
# Mirrors OUTPUTS_DIR — local-only zone under the project folder by default.
PAPER_DIR = Path(os.environ.get("SURVEY_FEATURES_PAPER") or (ROOT / "paper"))


# ── Model registry ────────────────────────────────────────────────────────────
# Selector = the test model whose feature-selection capability we measure.
# Each selector keeps its artifacts in a separate subdir (selector key) so adding a
# model never clobbers another's.
SELECTORS: dict[str, dict[str, str]] = {
    # IDs must match the configured LLM endpoint catalog (Nebius Studio / Token Factory).
    "deepseek": {"model": "deepseek-ai/DeepSeek-V4-Pro"},
    "kimi":     {"model": "moonshotai/Kimi-K2.6"},
    # Phase-A pilot onboarding (2026-08): cheap zoo entries exercising the
    # new-selector path. IDs reused from DISAMBIGUATORS / ROLE_SWAP_EXTRACTOR.
    "flash":    {"model": "deepseek-ai/DeepSeek-V4-Flash"},
    "minimax":  {"model": "MiniMaxAI/MiniMax-M3"},
}
DEFAULT_SELECTOR = "deepseek"

# Selectors used by experiments (may add keys without changing confirmatory zoo semantics).
# deepseek_v4 aliases the live V4-Pro ID for prompt-sensitivity artifacts.
EXPERIMENT_SELECTORS: dict[str, dict[str, str]] = {
    **SELECTORS,
    "deepseek_v4": {"model": "deepseek-ai/DeepSeek-V4-Pro"},
}

# Extraction (free-text essay -> feature list) is held FIXED across selectors.
EXTRACTOR_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# Disambiguation (feature -> code/none). nemotron is the main-experiment choice.
DISAMBIGUATORS: dict[str, str] = {
    "nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "qwen235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    # pipeline-role-swap candidate (not confirmatory default)
    "flash": "deepseek-ai/DeepSeek-V4-Flash",
}

# Extractor candidate for pipeline-role-swap (confirmatory default stays EXTRACTOR_MODEL).
ROLE_SWAP_EXTRACTOR = "MiniMaxAI/MiniMax-M3"
ROLE_SWAP_DISAMBIG_KEY = "flash"

DISAMBIG_MODEL = os.environ.get("DISAMBIG_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
DISAMBIG_BASE_URL = os.environ.get("DISAMBIG_BASE_URL") or None  # falls back to LLM_BASE_URL
DISAMBIG_API_KEY = os.environ.get("DISAMBIG_API_KEY") or None    # falls back to LLM_API_KEY

# Sentence-transformer for survey-variable retrieval and the oracle's near-duplicate filter.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# The two prompt conditions run for every cell.
CONDITIONS = ["unprompted", "country_provided"]

# ── Textbook baseline ─────────────────────────────────────────────────────────
# Generic predictors a researcher would list WITHOUT reading the question.
# Frozen; ORDER matters (fixed-k takes the first k). Resolved per survey by
# scripts/build_textbook_baseline.py.
TEXTBOOK_CONSTRUCTS: list[tuple[str, str]] = [
    ("age", "how old the respondent is, in years"),
    ("gender", "whether the respondent is male or female"),
    ("education", "the highest level of education the respondent completed"),
    ("income", "the respondent's household income level"),
    ("employment status", "whether the respondent works, and in what capacity"),
    ("urban or rural residence", "whether the respondent lives in a town, city or rural area"),
    ("religiosity", "how religious the respondent is, or how often they attend services"),
    ("marital status", "whether the respondent is married, single, divorced or widowed"),
    ("left-right political ideology", "where the respondent places themselves on a left-right political scale"),
    ("ethnicity or language group", "the respondent's ethnic group, language or nationality"),
]

# Feature types that enter retrieve+disambiguate+score.
# instrument_methodology and population_statistic stay out (unmappable; metadata only).
PIPE_TYPES = {"respondent_attribute", "temporal_contextual"}
