"""
Shared paths for grid outputs: per-model LLM caches, grid_summary naming, discovery,
and the format-pilot / free-text pipeline directory layout.

Oracle stays at outputs/<target>_<country>/oracle.csv.
LLM + eval caches live under outputs/<target>_<country>/llm__<output_tag>/.
Free-text pipeline artifacts live under outputs/format_pilot/<selector>/.
Set GRID_SUMMARY_TAG to select tagged grid_summary CSVs in analysis scripts when multiple exist.
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


def llm_cache_prefix(cell_prefix: str, output_tag: str) -> str:
    """Relative path under outputs/ for disambig.json + eval.json (no trailing slash)."""
    return f"{cell_prefix}/llm__{output_tag}"


def grid_summary_csv_path(outputs_dir: Path, survey_id: str, output_tag: str) -> Path:
    return outputs_dir / f"{_GRID_SUMMARY_PREFIX}{survey_id}__{output_tag}.csv"


def grid_results_json_path(outputs_dir: Path, survey_id: str, output_tag: str) -> Path:
    return outputs_dir / f"grid_results__{survey_id}__{output_tag}.json"


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
    raw = sorted(
        p
        for p in outputs_dir.glob(f"{_GRID_SUMMARY_PREFIX}*.csv")
        if "all_surveys" not in p.stem
    )
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
    """All grid_summary CSVs as (path, survey_id, model_tag), one row per file.

    Unlike collect_grid_summary_paths (which returns one file per survey for
    single-model analysis), this returns *every* tagged summary so callers can
    treat the model tag as an explicit analysis dimension (side-by-side models).
    """
    out: list[tuple[Path, str, str | None]] = []
    for p in sorted(outputs_dir.glob(f"{_GRID_SUMMARY_PREFIX}*.csv")):
        if "all_surveys" in p.stem:
            continue
        try:
            sid, tag = parse_grid_summary_stem(p.stem)
        except ValueError:
            continue
        out.append((p, sid, tag))
    return out


def resolve_grid_summary_for_survey(outputs_dir: Path, survey_id: str) -> Path | None:
    """Single survey CSV resolved with same rules as collect_grid_summary_paths."""
    paths = [p for p in collect_grid_summary_paths(outputs_dir) if parse_grid_summary_stem(p.stem)[0] == survey_id]
    return paths[0] if paths else None


def manifest_path(outputs_dir: Path, survey_id: str, output_tag: str) -> Path:
    """Path for run manifest JSON (one per survey × experiment)."""
    return outputs_dir / f"run_manifest__{survey_id}__{output_tag}.json"


def logs_dir(outputs_dir: Path) -> Path:
    """Subdirectory for run logs."""
    d = outputs_dir / "logs"
    d.mkdir(exist_ok=True)
    return d


def resolve_llm_artifact(
    outputs_dir: Path,
    target: str,
    country: str,
    filename: str,
) -> Path | None:
    """
    Find disambig.json or eval.json for a cell.

    Order: GRID_SUMMARY_TAG match under llm__<tag>/, else legacy flat file,
    else newest llm__*/filename by mtime.
    """
    base = outputs_dir / f"{target}_{country}"
    if not base.is_dir():
        return None
    tag = os.environ.get("GRID_SUMMARY_TAG", "").strip()
    if tag:
        p = base / f"llm__{tag}" / filename
        if p.is_file():
            return p
    flat = base / filename
    if flat.is_file():
        return flat
    best: Path | None = None
    best_m = 0.0
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


# ── Free-text (format-pilot) pipeline layout ─────────────────────────────────

def format_pilot_dir(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Root for all free-text pipeline artifacts (kept as outputs/format_pilot/ so
    existing pilot-2 artifacts remain valid)."""
    return outputs_dir / "format_pilot"


def selector_dirs(selector_key: str, outputs_dir: Path = OUTPUTS_DIR) -> tuple[Path, Path, Path]:
    """(freetext, extracted, maps) subdirs for one selector."""
    base = format_pilot_dir(outputs_dir) / selector_key
    return base / "freetext", base / "extracted", base / "maps"


def cell_tag(survey: str, target: str, country: str) -> str:
    """Filesystem-safe tag for one grid cell (used in free-text pipeline filenames)."""
    return f"{survey}__{target}__{country}".replace("/", "_").replace(" ", "_")


def genuine_cells(outputs_dir: Path = OUTPUTS_DIR) -> list[tuple[str, str, str]]:
    """The (survey, target, country) cells classified 'genuine' by the leakage audit
    (scripts/leakage_audit.py writes outputs/leakage_audit.csv)."""
    out = []
    with open(outputs_dir / "leakage_audit.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["leakage_class"] == "genuine":
                out.append((r["survey"], r["target"], r["country"]))
    return out
