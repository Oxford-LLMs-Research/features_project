"""
One-shot migration of a local outputs/ tree into the named-cache / named-experiment layout.

Dry-run by default (prints planned moves). Pass --apply to execute.

Moves (skip if destination already exists):
  <target>_<country>/          -> cache/cells/<target>_<country>/
  survey_embeddings__*.npz -> cache/embeddings/
  leakage_audit*.csv/json      -> cache/audits/
  format_pilot/                -> main/
  embedding_sensitivity/       -> experiments/embedding_sensitivity/
  subitem_mapping/             -> experiments/subitem_mapping/
  similarity_threshold/        -> experiments/similarity_threshold/
  grid_summary__*, grid_results__*, llm_usage__*, run_manifest__*
                               -> grid/
  alignment_*, uncertainty_summary.json, _prelim_stats.json
                               -> analysis/

Does not delete outputs/.tmp/ (safe to remove yourself). Does not touch paper/.

Usage:
  python scripts/migrate_outputs_layout.py
  python scripts/migrate_outputs_layout.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Prefixes that are NOT cell dirs
_SKIP_TOP = {
    "cache",
    "main",
    "experiments",
    "grid",
    "analysis",
    "logs",
    ".tmp",
    "format_pilot",
    "embedding_sensitivity",
    "subitem_mapping",
    "similarity_threshold",
}

_GRID_PREFIXES = (
    "grid_summary__",
    "grid_results__",
    "llm_usage__",
    "run_manifest__",
)

_ANALYSIS_NAMES = (
    "alignment_by_cell.csv",
    "alignment_summary.json",
    "uncertainty_summary.json",
    "_prelim_stats.json",
)


def _move(src: Path, dst: Path, *, apply: bool, planned: list[str]) -> None:
    if not src.exists():
        return
    if dst.exists():
        planned.append(f"SKIP (dest exists): {src.relative_to(OUT)} -> {dst.relative_to(OUT)}")
        return
    planned.append(f"MOVE: {src.relative_to(OUT)} -> {dst.relative_to(OUT)}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def plan_moves(apply: bool) -> list[str]:
    planned: list[str] = []
    if not OUT.is_dir():
        print(f"No outputs/ at {OUT}", file=sys.stderr)
        return planned

    # Named experiment dirs
    for name in ("embedding_sensitivity", "subitem_mapping", "similarity_threshold"):
        _move(OUT / name, OUT / "experiments" / name, apply=apply, planned=planned)

    # Main free-text
    _move(OUT / "format_pilot", OUT / "main", apply=apply, planned=planned)

    # Audits
    for name in ("leakage_audit.csv", "leakage_audit_summary.json"):
        _move(OUT / name, OUT / "cache" / "audits" / name, apply=apply, planned=planned)

    # Embeddings
    for p in sorted(OUT.glob("survey_embeddings__*.npz")):
        _move(p, OUT / "cache" / "embeddings" / p.name, apply=apply, planned=planned)

    # Analysis digests
    for name in _ANALYSIS_NAMES:
        _move(OUT / name, OUT / "analysis" / name, apply=apply, planned=planned)

    # Grid summaries / manifests at root
    for p in sorted(OUT.iterdir()):
        if not p.is_file():
            continue
        if any(p.name.startswith(pref) for pref in _GRID_PREFIXES):
            _move(p, OUT / "grid" / p.name, apply=apply, planned=planned)

    # Cell dirs (anything with oracle.csv or llm__* that isn't a reserved name)
    for p in sorted(OUT.iterdir()):
        if not p.is_dir() or p.name in _SKIP_TOP or p.name.startswith("."):
            continue
        if p.name in ("cache", "main", "experiments", "grid", "analysis", "logs"):
            continue
        looks_like_cell = (p / "oracle.csv").is_file() or any(p.glob("llm__*"))
        if looks_like_cell:
            _move(p, OUT / "cache" / "cells" / p.name, apply=apply, planned=planned)

    return planned


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="execute moves (default is dry-run)")
    args = ap.parse_args()

    planned = plan_moves(apply=args.apply)
    if not planned:
        print("Nothing to migrate.")
        return
    print(f"{'Applied' if args.apply else 'Dry-run'} ({len(planned)} actions):")
    for line in planned:
        print(f"  {line}")
    if not args.apply:
        print("\nRe-run with --apply to execute.")


if __name__ == "__main__":
    main()
