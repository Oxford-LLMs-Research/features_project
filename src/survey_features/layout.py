"""
Path contracts for the artifact tree.

Live root is the shared Dropbox folder `features_project/outputs`, via
SURVEY_FEATURES_OUTPUTS in gitignored .env. Fallback <repo>/outputs is not
the current setup. Human map: README § Outputs. Where to write / how to
name experiments: CONTRIBUTING.md.

  outputs/                              # OUTPUTS_DIR (Dropbox)
    cache/                              # confirmatory prerequisites
      cells/<target>_<country>/oracle.csv (+ oracle_meta.json)
      embeddings/survey_embeddings__<survey>__<model>.npz
      audits/leakage_audit.csv          # pipeline leakage — not repo-audit
      baselines/textbook__<survey>.json
    main/                               # confirmatory Arm C
      <selector>/{freetext,extracted,maps}/
      scores_<selector>.csv
      runs/<run_tag>/…                  # --run-tag: map/score only
    experiments/<name>/                 # named studies (register first)
      _analysis/                        # digests from registered experiments
    analysis/                           # one-off screens, not a named experiment
    logs/                               # flat token_usage_*.jsonl, timing_*.jsonl
    audits/                             # housekeeping reports (repo-audit)
    .tmp/                               # AutoGluon scratch; safe to delete
    .trash/                             # repo-audit soft-deletes

Writers always use these helpers. Readers resolve the canonical path (and, for a
few historical filenames, a one-step fallback listed inline). Old experiment /
grid / format_pilot dual-resolve fallbacks are gone; experiments/ itself is live.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .config import OUTPUTS_DIR


def sanitize_model_slug(model_name: str, max_len: int = 96) -> str:
    """Filesystem-safe token from a model name or run tag."""
    s = (model_name or "").strip()
    for ch in r'/\:*?"<>| ':
        s = s.replace(ch, "_")
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("model name / run tag is empty — set LLM_MODEL or pass --run-tag")
    return s[:max_len] if len(s) > max_len else s


def _first_file(*candidates: Path) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


# ── Top-level roots ───────────────────────────────────────────────────────────

def cache_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return outputs_dir / "cache"


def cache_cells_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return cache_dir(outputs_dir) / "cells"


def cache_embeddings_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return cache_dir(outputs_dir) / "embeddings"


def cache_audits_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return cache_dir(outputs_dir) / "audits"


def cache_baselines_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return cache_dir(outputs_dir) / "baselines"


def tmp_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """AutoGluon / scratch root. Safe to delete between runs."""
    return outputs_dir / ".tmp"


def logs_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    d = outputs_dir / "logs"
    d.mkdir(exist_ok=True)
    return d


def main_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return outputs_dir / "main"


def experiments_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return outputs_dir / "experiments"


def analysis_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """One-off diagnostic CSVs that are not a registered experiment."""
    return outputs_dir / "analysis"


def housekeeping_audits_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """outputs/audits/ — repo-audit reports. Leakage stays in cache/audits/."""
    return outputs_dir / "audits"


def trash_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """outputs/.trash/ — repo-audit soft-deletes."""
    return outputs_dir / ".trash"


def prompt_sensitivity_root(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """outputs/experiments/prompt_sensitivity/"""
    return experiments_dir(outputs_dir) / "prompt_sensitivity"


def prompt_sensitivity_dirs(
    selector_key: str,
    arm: str,
    outputs_dir: Path = OUTPUTS_DIR,
) -> tuple[Path, Path, Path]:
    """(freetext, extracted, maps) under prompt_sensitivity/<selector>/<arm>/."""
    base = prompt_sensitivity_root(outputs_dir) / selector_key / arm
    return base / "freetext", base / "extracted", base / "maps"


def prompt_sensitivity_scores_path(
    selector_key: str,
    arm: str,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Path:
    return prompt_sensitivity_root(outputs_dir) / f"scores_{selector_key}_{arm}.csv"


def pipeline_role_swap_root(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """outputs/experiments/pipeline_role_swap/"""
    return experiments_dir(outputs_dir) / "pipeline_role_swap"


def pipeline_role_swap_dirs(
    run_key: str,
    outputs_dir: Path = OUTPUTS_DIR,
) -> tuple[Path, Path]:
    """(extracted, maps) under pipeline_role_swap/<run_key>/."""
    base = pipeline_role_swap_root(outputs_dir) / run_key
    return base / "extracted", base / "maps"


def pipeline_role_swap_scores_path(
    run_key: str,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Path:
    return pipeline_role_swap_root(outputs_dir) / f"scores_{run_key}.csv"


# ── Cell / oracle cache ───────────────────────────────────────────────────────

def cell_dir(target: str, country: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Per-cell directory under cache/cells/<target>_<country>/."""
    return cache_cells_dir(outputs_dir) / f"{target}_{country}"


def oracle_csv_path(target: str, country: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Existing oracle.csv if present, else the canonical write path."""
    preferred = cell_dir(target, country, outputs_dir) / "oracle.csv"
    if preferred.is_file():
        return preferred
    return preferred


def survey_emb_cache_path(
    survey_id: str,
    embedding_model: str,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Path:
    """Embedding cache under cache/embeddings/; creates the directory on write path."""
    slug = sanitize_model_slug(embedding_model)
    name = f"survey_embeddings__{survey_id}__{slug}.npz"
    dest = cache_embeddings_dir(outputs_dir)
    preferred = dest / name
    if preferred.is_file():
        return preferred
    # One-step fallback for older flat caches still on disk.
    legacy = outputs_dir / name
    if legacy.is_file():
        return legacy
    dest.mkdir(parents=True, exist_ok=True)
    return preferred


# ── Audits ────────────────────────────────────────────────────────────────────

def leakage_audit_csv_path(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    preferred = cache_audits_dir(outputs_dir) / "leakage_audit.csv"
    if preferred.is_file():
        return preferred
    legacy = outputs_dir / "leakage_audit.csv"
    if legacy.is_file():
        return legacy
    return preferred


def leakage_audit_summary_path(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    preferred = cache_audits_dir(outputs_dir) / "leakage_audit_summary.json"
    if preferred.is_file():
        return preferred
    legacy = outputs_dir / "leakage_audit_summary.json"
    if legacy.is_file():
        return legacy
    return preferred


def leakage_audit_write_paths(outputs_dir: Path = OUTPUTS_DIR) -> tuple[Path, Path]:
    d = cache_audits_dir(outputs_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / "leakage_audit.csv", d / "leakage_audit_summary.json"


# ── Main free-text pipeline ───────────────────────────────────────────────────

def main_scores_path(
    selector_key: str,
    outputs_dir: Path = OUTPUTS_DIR,
    run_tag: str | None = None,
) -> Path:
    """Scores CSV under main/ (or main/runs/<tag>/)."""
    root = main_dir(outputs_dir)
    if run_tag:
        return root / "runs" / sanitize_model_slug(run_tag) / f"scores_{selector_key}.csv"
    preferred = root / f"scores_{selector_key}.csv"
    if preferred.is_file():
        return preferred
    # Historical filenames still present for deepseek / kimi runs.
    if selector_key == "deepseek":
        legacy = root / "scores.csv"
        if legacy.is_file():
            return legacy
    if selector_key == "kimi":
        legacy_k = root / "scores_kimi.csv"
        if legacy_k.is_file():
            return legacy_k
    return preferred


def resolve_main_scores_path(
    selector_key: str,
    outputs_dir: Path = OUTPUTS_DIR,
    run_tag: str | None = None,
) -> Path | None:
    p = main_scores_path(selector_key, outputs_dir, run_tag=run_tag)
    return p if p.is_file() else None


def selector_dirs(
    selector_key: str,
    outputs_dir: Path = OUTPUTS_DIR,
    run_tag: str | None = None,
) -> tuple[Path, Path, Path]:
    """(freetext, extracted, maps) for one selector.

    Gen/extract live under main/<selector>/. Maps use main/runs/<tag>/<selector>/maps
    when run_tag is set, else main/<selector>/maps.
    """
    root = main_dir(outputs_dir)
    base = root / selector_key
    freetext, extracted = base / "freetext", base / "extracted"
    if run_tag:
        maps = root / "runs" / sanitize_model_slug(run_tag) / selector_key / "maps"
    else:
        maps = base / "maps"
    return freetext, extracted, maps


# ── Cell helpers ──────────────────────────────────────────────────────────────

def cell_tag(survey: str, target: str, country: str) -> str:
    """Filesystem-safe tag for one grid cell."""
    return f"{survey}__{target}__{country}".replace("/", "_").replace(" ", "_")


def genuine_cells(outputs_dir: Path = OUTPUTS_DIR) -> list[tuple[str, str, str]]:
    """(survey, target, country) cells classified 'genuine' by the leakage audit."""
    path = leakage_audit_csv_path(outputs_dir)
    if not path.is_file():
        raise FileNotFoundError(f"leakage audit not found: {path}")
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["leakage_class"] == "genuine":
                out.append((r["survey"], r["target"], r["country"]))
    return out
