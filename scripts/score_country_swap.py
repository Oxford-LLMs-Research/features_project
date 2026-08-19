"""
Country-swap paired contrast — the Test-2 measurement (rehearsed in Phase A).

For every target with >= 2 countries in the grid, score each cell twice per
ordered country pair (A != B):

  own   codes the selector produced FOR B (country_provided)   scored on B
  swap  codes the selector produced FOR A (country_provided)   scored on B

Adaptation value = own - swap on the same evaluation rows: positive means the
selector's country-tailoring tracks real cross-country differences in predictive
structure; ~0 means its selections transport freely (nothing country-specific,
or nothing real to adapt to — disambiguate with the heterogeneity screen).

Rows carry k_mode = own|swap and swap_from = source country. Both arms run
through the same evaluator, baselines and caches as the main scores, so the
contrast is paired cell-by-cell.

Usage:
    python scripts/score_country_swap.py --selector kimi --cells data/pilot_cells.csv \
        --disambiguator nemotron --run-tag pilot_phase_a
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from survey_features.config import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DISAMBIGUATORS,
    OUTPUTS_DIR,
    SELECTORS,
)
from survey_features.layout import cell_tag, selector_dirs  # noqa: E402
from survey_features.score_cell import (  # noqa: E402
    resolve_score_n_draws,
    resolve_score_workers,
    resolve_score_xgb_nthread,
    run_score_jobs,
    score_cols,
)

CONDITION = "country_provided"  # the swap is only meaningful for tailored selections


def _map_codes(map_dir: Path, disambig_key: str, survey: str, target: str,
               country: str) -> list[str] | None:
    p = map_dir / f"C__{disambig_key}__{cell_tag(survey, target, country)}__{CONDITION}.json"
    if not p.is_file():
        return None
    import json

    rec = json.loads(p.read_text(encoding="utf-8"))
    if "expanded_codes" in rec:
        return rec.get("expanded_codes") or []
    return rec.get("mapped_codes", [])


def main() -> None:
    ap = argparse.ArgumentParser(description="Country-swap paired-contrast scorer.")
    ap.add_argument("--selector", choices=list(SELECTORS), required=True)
    ap.add_argument("--disambiguator", choices=list(DISAMBIGUATORS), default="nemotron")
    ap.add_argument("--cells", type=Path, required=True,
                    help="grid CSV (survey,target,country)")
    ap.add_argument("--run-tag", default=None,
                    help="read maps from selectors/runs/<tag>/; scores land beside them")
    ap.add_argument("--score-workers", type=int, default=None)
    ap.add_argument("--score-xgb-nthread", type=int, default=None)
    args = ap.parse_args()

    grid = pd.read_csv(args.cells)
    cells = [(r["survey"], r["target"], r["country"]) for _, r in grid.iterrows()]
    by_target: dict[tuple[str, str], list[str]] = defaultdict(list)
    for survey, target, country in cells:
        by_target[(survey, target)].append(country)

    _, _, map_dir = selector_dirs(args.selector, run_tag=args.run_tag)
    n_draws = resolve_score_n_draws()
    workers = resolve_score_workers(args.score_workers)
    nthread = resolve_score_xgb_nthread(workers, args.score_xgb_nthread)
    emb = DEFAULT_EMBEDDING_MODEL

    specs, skipped = [], []
    for (survey, target), countries in by_target.items():
        if len(countries) < 2:
            continue
        codes_by_country = {
            c: _map_codes(map_dir, args.disambiguator, survey, target, c)
            for c in countries
        }
        for dest in countries:
            evals = []
            for src in countries:
                codes = codes_by_country[src]
                if codes is None:
                    skipped.append((survey, target, src))
                    continue
                evals.append({
                    "condition": CONDITION, "arm": "C",
                    "disambiguator": args.disambiguator,
                    "embedding_model": emb, "codes": codes,
                    "k_mode": "own" if src == dest else "swap",
                    "swap_from": src,
                })
            if evals:
                specs.append({
                    "survey": survey, "target": target, "country": dest,
                    "evals": evals, "n_draws": n_draws, "nthread": nthread,
                    "outputs_dir": str(OUTPUTS_DIR),
                    "grid_cells": [list(c) for c in cells],
                })

    if skipped:
        for survey, target, src in sorted(set(skipped)):
            print(f"  ! no {CONDITION} map for {survey} {target} {src} — run map first")
    if not specs:
        raise SystemExit("no swappable targets (need >= 2 countries per target with maps)")

    tag = args.run_tag or "untagged"
    out_csv = (OUTPUTS_DIR / "main" / "runs" / tag /
               f"swap_scores_{args.selector}.csv")
    cols = score_cols("k_mode", "swap_from")
    n_pairs = sum(len(s["evals"]) - 1 for s in specs)
    print(f"[swap] selector={args.selector} targets="
          f"{len({(s['survey'], s['target']) for s in specs})} "
          f"cells={len(specs)} ordered-swaps={n_pairs} -> {out_csv}")
    run_score_jobs(specs, out_csv, cols, workers=workers, log_prefix="[swap]")

    # Paired summary: own - swap per (target, dest), meaned over swaps.
    df = pd.read_csv(out_csv)
    df = df[df["k_spec"] == "model"]
    own = df[df["k_mode"] == "own"].set_index(["survey", "target", "country"])
    swap = df[df["k_mode"] == "swap"]
    deltas = []
    for key, grp in swap.groupby(["survey", "target", "country"]):
        if key not in own.index:
            continue
        o = own.loc[key]
        # Type-matched: each cell contributes in its own fit family's currency.
        for col in ("value_over_random_ll", "value_over_random_rho",
                    "captured_importance"):
            g = pd.to_numeric(grp[col], errors="coerce").dropna()
            ov = pd.to_numeric(pd.Series([o[col]]), errors="coerce").iloc[0]
            if len(g) and pd.notna(ov):
                deltas.append({"metric": col, "own_minus_swap": float(ov - g.mean())})
    if deltas:
        summary = pd.DataFrame(deltas).groupby("metric")["own_minus_swap"].agg(["mean", "count"])
        print("\n[swap] adaptation value (own - swap), model-k:")
        print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
