"""
Shared path contracts for outputs/.

Target layout (writers prefer new paths; readers dual-resolve new then legacy):

  outputs/
    cache/
      cells/<target>_<country>/oracle.csv
      embeddings/survey_embeddings__<survey>__<model>.npz
      audits/leakage_audit.csv
    main/                          # canonical free-text pipeline (was format_pilot/)
      <selector>/{freetext,extracted,maps}/
      scores_<selector>.csv
      runs/<run_tag>/…             # optional tagged map/score writes
    experiments/<name>/…           # embedding_sensitivity, subitem_mapping, …
      runs/<run_tag>/…             # optional tagged experiment writes
    grid/                          # legacy JSON prelim (tagged summaries + llm caches)
    analysis/                      # derived digests (alignment, uncertainty, …)
    logs/
    .tmp/                          # AutoGluon scratch; safe to delete

Set GRID_SUMMARY_TAG to select tagged grid_summary CSVs when multiple exist.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from .config import OUTPUTS_DIR

_GRID_SUMMARY_PREFIX = "grid_summary__"


def sanitize_model_slug(model_name: str, max_len: int = 96) -> str:
    """Filesystem-safe token from LLM_MODEL (used in folder and summary names)."""
    s = (model_name or "").strip()
    for ch in r'/\:*?"<>| ':
        s = s.replace(ch, "_")
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("model name / run tag is empty — set LLM_MODEL or pass --run-tag")
    return s[:max_len] if len(s) > max_len else s


def _prefer_existing(*candidates: Path, default: Path) -> Path:
    """Return the first existing path among candidates, else default (for writes)."""
    for p in candidates:
        if p.exists():
            return p
    return default


def _first_file(*candidates: Path) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


# ── Top-level named roots ─────────────────────────────────────────────────────

def cache_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return outputs_dir / "cache"


def cache_cells_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return cache_dir(outputs_dir) / "cells"


def cache_embeddings_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return cache_dir(outputs_dir) / "embeddings"


def cache_audits_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return cache_dir(outputs_dir) / "audits"


def analysis_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Derived digests (alignment, uncertainty, prelim stats). Dual-resolve with root."""
    new = outputs_dir / "analysis"
    return _prefer_existing(new, default=new)


def analysis_write_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Directory to write new analysis digests into (always outputs/analysis/)."""
    d = outputs_dir / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def grid_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Legacy JSON-grid artifact root. Prefer grid/ when present; else outputs/ root."""
    new = outputs_dir / "grid"
    if new.is_dir():
        return new
    return outputs_dir


def tmp_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """AutoGluon / scratch root. Safe to delete between runs."""
    return outputs_dir / ".tmp"


def logs_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Subdirectory for run logs."""
    d = outputs_dir / "logs"
    d.mkdir(exist_ok=True)
    return d


# ── Shared cell / oracle cache ────────────────────────────────────────────────

def cell_dir(target: str, country: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Per-cell directory (oracle.csv, optional feature_pool, llm__* caches).

    Prefer cache/cells/<t>_<c>/; fall back to legacy outputs/<t>_<c>/;
    fall back to grid/cells/<t>_<c>/ for migrated llm caches with oracle elsewhere.
    Default write target is cache/cells/.
    """
    key = f"{target}_{country}"
    new = cache_cells_dir(outputs_dir) / key
    legacy = outputs_dir / key
    grid_cells = outputs_dir / "grid" / "cells" / key
    if new.is_dir():
        return new
    if legacy.is_dir():
        return legacy
    if grid_cells.is_dir():
        return grid_cells
    return new


def oracle_csv_path(target: str, country: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Path for oracle.csv — existing file if any, else write path under cell_dir."""
    key = f"{target}_{country}"
    candidates = [
        cache_cells_dir(outputs_dir) / key / "oracle.csv",
        outputs_dir / key / "oracle.csv",
        outputs_dir / "grid" / "cells" / key / "oracle.csv",
    ]
    found = _first_file(*candidates)
    if found is not None:
        return found
    return cell_dir(target, country, outputs_dir) / "oracle.csv"


def llm_cache_prefix(cell_prefix: str, output_tag: str) -> str:
    """Relative path under a cell dir for disambig.json + eval.json (no trailing slash)."""
    return f"{cell_prefix}/llm__{output_tag}"


def resolve_llm_artifact(
    outputs_dir: Path,
    target: str,
    country: str,
    filename: str,
) -> Path | None:
    """
    Find disambig.json or eval.json for a cell.

    Order: GRID_SUMMARY_TAG match under llm__<tag>/, else legacy flat file,
    else newest llm__*/filename by mtime. Searches dual-resolved cell dirs.
    """
    bases: list[Path] = []
    key = f"{target}_{country}"
    for b in (
        cache_cells_dir(outputs_dir) / key,
        outputs_dir / key,
        outputs_dir / "grid" / "cells" / key,
    ):
        if b.is_dir() and b not in bases:
            bases.append(b)
    if not bases:
        return None

    tag = os.environ.get("GRID_SUMMARY_TAG", "").strip()
    for base in bases:
        if tag:
            p = base / f"llm__{tag}" / filename
            if p.is_file():
                return p
        flat = base / filename
        if flat.is_file():
            return flat

    best: Path | None = None
    best_m = 0.0
    for base in bases:
        for sub in base.glob("llm__*"):
            if not sub.is_dir():
                continue
            p = sub / filename
            if p.is_file():
                m = p.stat().st_mtime
                if m > best_m:
                    best_m = m
                    best = p
    return best


# ── Embeddings cache ──────────────────────────────────────────────────────────

def survey_emb_cache_path(
    survey_id: str,
    embedding_model: str,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Path:
    """Resolve existing embedding cache, else return preferred write path under cache/embeddings/."""
    slug = sanitize_model_slug(embedding_model)
    name = f"survey_embeddings__{survey_id}__{slug}.npz"
    candidates = [
        cache_embeddings_dir(outputs_dir) / name,
        outputs_dir / name,
        outputs_dir / "grid" / name,
    ]
    found = _first_file(*candidates)
    if found is not None:
        return found
    dest = cache_embeddings_dir(outputs_dir)
    dest.mkdir(parents=True, exist_ok=True)
    return dest / name


# ── Audits ────────────────────────────────────────────────────────────────────

def leakage_audit_csv_path(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    candidates = [
        cache_audits_dir(outputs_dir) / "leakage_audit.csv",
        outputs_dir / "leakage_audit.csv",
    ]
    found = _first_file(*candidates)
    if found is not None:
        return found
    return cache_audits_dir(outputs_dir) / "leakage_audit.csv"


def leakage_audit_summary_path(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    candidates = [
        cache_audits_dir(outputs_dir) / "leakage_audit_summary.json",
        outputs_dir / "leakage_audit_summary.json",
    ]
    found = _first_file(*candidates)
    if found is not None:
        return found
    return cache_audits_dir(outputs_dir) / "leakage_audit_summary.json"


def leakage_audit_write_paths(outputs_dir: Path = OUTPUTS_DIR) -> tuple[Path, Path]:
    """Canonical write locations for a fresh leakage audit."""
    d = cache_audits_dir(outputs_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / "leakage_audit.csv", d / "leakage_audit_summary.json"


# ── Analysis digests ──────────────────────────────────────────────────────────

def alignment_by_cell_path(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    candidates = [
        outputs_dir / "analysis" / "alignment_by_cell.csv",
        outputs_dir / "alignment_by_cell.csv",
    ]
    found = _first_file(*candidates)
    return found if found is not None else analysis_write_dir(outputs_dir) / "alignment_by_cell.csv"


def alignment_summary_path(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    candidates = [
        outputs_dir / "analysis" / "alignment_summary.json",
        outputs_dir / "alignment_summary.json",
    ]
    found = _first_file(*candidates)
    return found if found is not None else analysis_write_dir(outputs_dir) / "alignment_summary.json"


def uncertainty_summary_path(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    candidates = [
        outputs_dir / "analysis" / "uncertainty_summary.json",
        outputs_dir / "uncertainty_summary.json",
    ]
    found = _first_file(*candidates)
    return found if found is not None else analysis_write_dir(outputs_dir) / "uncertainty_summary.json"


def prelim_stats_path(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    candidates = [
        outputs_dir / "analysis" / "_prelim_stats.json",
        outputs_dir / "_prelim_stats.json",
    ]
    found = _first_file(*candidates)
    return found if found is not None else analysis_write_dir(outputs_dir) / "_prelim_stats.json"


# ── Grid summaries (legacy JSON prelim) ───────────────────────────────────────

def grid_summary_csv_path(outputs_dir: Path, survey_id: str, output_tag: str) -> Path:
    root = grid_dir(outputs_dir)
    # Prefer writing under grid/ when that tree exists or is the default new layout
    if (outputs_dir / "grid").is_dir() or root == outputs_dir / "grid":
        return outputs_dir / "grid" / f"{_GRID_SUMMARY_PREFIX}{survey_id}__{output_tag}.csv"
    return outputs_dir / f"{_GRID_SUMMARY_PREFIX}{survey_id}__{output_tag}.csv"


def grid_results_json_path(outputs_dir: Path, survey_id: str, output_tag: str) -> Path:
    if (outputs_dir / "grid").is_dir():
        return outputs_dir / "grid" / f"grid_results__{survey_id}__{output_tag}.json"
    return outputs_dir / f"grid_results__{survey_id}__{output_tag}.json"


def manifest_path(outputs_dir: Path, survey_id: str, output_tag: str) -> Path:
    """Path for run manifest JSON (one per survey × experiment)."""
    if (outputs_dir / "grid").is_dir():
        return outputs_dir / "grid" / f"run_manifest__{survey_id}__{output_tag}.json"
    return outputs_dir / f"run_manifest__{survey_id}__{output_tag}.json"


def llm_usage_path(outputs_dir: Path, survey_id: str, output_tag: str) -> Path:
    if (outputs_dir / "grid").is_dir():
        return outputs_dir / "grid" / f"llm_usage__{survey_id}__{output_tag}.jsonl"
    return outputs_dir / f"llm_usage__{survey_id}__{output_tag}.jsonl"


def parse_grid_summary_stem(stem: str) -> tuple[str, str | None]:
    """
    grid_summary__wvs -> (wvs, None)
    grid_summary__wvs__moonshotai_Kimi-K2.5 -> (wvs, moonshotai_Kimi-K2.5)
    """
    if not stem.startswith(_GRID_SUMMARY_PREFIX):
        raise ValueError(f"not a grid summary stem: {stem!r}")
    rest = stem[len(_GRID_SUMMARY_PREFIX) :]
    if "__" not in rest:
        return rest, None
    sid, tag = rest.split("__", 1)
    return sid, tag


def _iter_grid_summary_files(outputs_dir: Path) -> list[Path]:
    roots = []
    grid = outputs_dir / "grid"
    if grid.is_dir():
        roots.append(grid)
    roots.append(outputs_dir)
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        for p in root.glob(f"{_GRID_SUMMARY_PREFIX}*.csv"):
            if "all_surveys" in p.stem:
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(p)
    return sorted(out)


def collect_grid_summary_paths(
    outputs_dir: Path,
    *,
    env_tag: str | None = None,
) -> list[Path]:
    """
    Pick which grid_summary CSVs to load (one file per survey_id when possible).

    If GRID_SUMMARY_TAG is set (or env_tag passed), only files
    grid_summary__*__<tag>.csv are returned.
    Otherwise: for each survey_id, prefer the newest tagged file if any exist;
    else the legacy grid_summary__<survey_id>.csv.
    """
    raw = _iter_grid_summary_files(outputs_dir)
    if not raw:
        return []

    tag_filter = (
        env_tag.strip() if env_tag is not None else os.environ.get("GRID_SUMMARY_TAG", "")
    ).strip()

    if tag_filter:
        out: list[Path] = []
        for p in raw:
            try:
                _sid, t = parse_grid_summary_stem(p.stem)
            except ValueError:
                continue
            if t == tag_filter:
                out.append(p)
        return sorted(out)

    by_sid: dict[str, list[tuple[Path, str | None]]] = {}
    for p in raw:
        try:
            sid, t = parse_grid_summary_stem(p.stem)
        except ValueError:
            continue
        by_sid.setdefault(sid, []).append((p, t))

    chosen: list[Path] = []
    for sid in sorted(by_sid):
        entries = by_sid[sid]
        tagged = [p for p, t in entries if t is not None]
        if tagged:
            chosen.append(max(tagged, key=lambda x: x.stat().st_mtime))
        else:
            legacy = next((p for p, t in entries if t is None), None)
            if legacy is not None:
                chosen.append(legacy)
    return sorted(chosen)


def collect_all_grid_summaries(outputs_dir: Path) -> list[tuple[Path, str, str | None]]:
    """All grid_summary CSVs as (path, survey_id, model_tag), one row per file."""
    out: list[tuple[Path, str, str | None]] = []
    for p in _iter_grid_summary_files(outputs_dir):
        try:
            sid, tag = parse_grid_summary_stem(p.stem)
        except ValueError:
            continue
        out.append((p, sid, tag))
    return out


def resolve_grid_summary_for_survey(outputs_dir: Path, survey_id: str) -> Path | None:
    """Single survey CSV resolved with same rules as collect_grid_summary_paths."""
    paths = [
        p for p in collect_grid_summary_paths(outputs_dir)
        if parse_grid_summary_stem(p.stem)[0] == survey_id
    ]
    return paths[0] if paths else None


# ── Main free-text pipeline (was format_pilot/) ───────────────────────────────

def main_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Canonical free-text pipeline root. Prefer main/; fall back to format_pilot/."""
    new = outputs_dir / "main"
    old = outputs_dir / "format_pilot"
    if new.is_dir():
        return new
    if old.is_dir():
        return old
    return new


def format_pilot_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Deprecated alias for main_dir (kept for import compatibility)."""
    return main_dir(outputs_dir)


def main_scores_path(
    selector_key: str,
    outputs_dir: Path = OUTPUTS_DIR,
    run_tag: str | None = None,
) -> Path:
    """Scores CSV under main/ (or main/runs/<tag>/). Prefer scores_<sel>.csv; deepseek also reads legacy scores.csv."""
    root = main_dir(outputs_dir)
    if run_tag:
        return root / "runs" / sanitize_model_slug(run_tag) / f"scores_{selector_key}.csv"
    preferred = root / f"scores_{selector_key}.csv"
    if preferred.is_file():
        return preferred
    if selector_key == "deepseek":
        legacy = root / "scores.csv"
        if legacy.is_file():
            return legacy
    if selector_key == "kimi":
        legacy_k = root / "scores_kimi.csv"
        if legacy_k.is_file() and not preferred.is_file():
            return legacy_k
    return preferred


def resolve_main_scores_path(
    selector_key: str,
    outputs_dir: Path = OUTPUTS_DIR,
    run_tag: str | None = None,
) -> Path | None:
    """Existing scores file for a selector, or None."""
    if run_tag:
        p = main_scores_path(selector_key, outputs_dir, run_tag=run_tag)
        return p if p.is_file() else None
    root = main_dir(outputs_dir)
    candidates = [root / f"scores_{selector_key}.csv"]
    if selector_key == "deepseek":
        candidates.append(root / "scores.csv")
    if selector_key == "kimi":
        candidates.append(root / "scores_kimi.csv")
    return _first_file(*candidates)


def selector_dirs(
    selector_key: str,
    outputs_dir: Path = OUTPUTS_DIR,
    run_tag: str | None = None,
) -> tuple[Path, Path, Path]:
    """(freetext, extracted, maps) for one selector.

    Gen/extract always live under main/<selector>/ (shared cache).
    Maps use main/runs/<tag>/<selector>/maps when run_tag is set, else main/<selector>/maps.
    """
    root = main_dir(outputs_dir)
    base = root / selector_key
    freetext, extracted = base / "freetext", base / "extracted"
    if run_tag:
        maps = root / "runs" / sanitize_model_slug(run_tag) / selector_key / "maps"
    else:
        maps = base / "maps"
    return freetext, extracted, maps


# ── Named experiments ─────────────────────────────────────────────────────────

def experiments_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return outputs_dir / "experiments"


def experiment_dir(name: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Named experiment root. Prefer experiments/<name>/; fall back to outputs/<name>/."""
    new = experiments_dir(outputs_dir) / name
    old = outputs_dir / name
    if new.is_dir():
        return new
    if old.is_dir():
        return old
    return new


def experiment_run_dir(
    name: str,
    run_tag: str,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Path:
    """Tagged run under an experiment: experiments/<name>/runs/<tag>/ (or legacy …/runs/)."""
    return experiment_dir(name, outputs_dir) / "runs" / sanitize_model_slug(run_tag)


def embedding_sensitivity_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Root for embedding-model sensitivity runs (isolated from main/)."""
    return experiment_dir("embedding_sensitivity", outputs_dir)


def embedding_run_dirs(
    embedding_model: str,
    selector_key: str,
    outputs_dir: Path = OUTPUTS_DIR,
    run_tag: str | None = None,
) -> tuple[Path, Path]:
    """(maps_dir, scores_csv) under embedding_sensitivity/[runs/<tag>/]<slug>/<selector>/."""
    slug = sanitize_model_slug(embedding_model)
    root = embedding_sensitivity_dir(outputs_dir)
    if run_tag:
        base = root / "runs" / sanitize_model_slug(run_tag) / slug / selector_key
    else:
        base = root / slug / selector_key
    return base / "maps", base / f"scores_{selector_key}.csv"


def subitem_mapping_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Root for sub-item mapping experiment (isolated from main/)."""
    return experiment_dir("subitem_mapping", outputs_dir)


def subitem_run_dirs(
    selector_key: str,
    outputs_dir: Path = OUTPUTS_DIR,
    run_tag: str | None = None,
) -> tuple[Path, Path, Path]:
    """(maps_dir, diagnostics_csv, scores_csv) under subitem_mapping/[runs/<tag>/]<selector>/."""
    root = subitem_mapping_dir(outputs_dir)
    if run_tag:
        base = root / "runs" / sanitize_model_slug(run_tag) / selector_key
    else:
        base = root / selector_key
    return (
        base / "maps",
        base / "diagnostics.csv",
        base / f"scores_{selector_key}.csv",
    )


def similarity_threshold_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return experiment_dir("similarity_threshold", outputs_dir)


# ── Cell helpers ──────────────────────────────────────────────────────────────

def cell_tag(survey: str, target: str, country: str) -> str:
    """Filesystem-safe tag for one grid cell (used in free-text pipeline filenames)."""
    return f"{survey}__{target}__{country}".replace("/", "_").replace(" ", "_")


def genuine_cells(outputs_dir: Path = OUTPUTS_DIR) -> list[tuple[str, str, str]]:
    """The (survey, target, country) cells classified 'genuine' by the leakage audit."""
    path = leakage_audit_csv_path(outputs_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"leakage audit not found (tried cache/audits/ and outputs/ root): {path}"
        )
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["leakage_class"] == "genuine":
                out.append((r["survey"], r["target"], r["country"]))
    return out
