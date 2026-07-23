"""
Aggregate sub-item mapping diagnostics (concept vs subconcept failure).

Reads maps from outputs/subitem_mapping/<selector>/maps/ when present.
Can also report parent-only baseline rates from format_pilot maps for comparison.

Writes: outputs/subitem_mapping/<selector>/diagnostics.csv (when expanded maps exist)
        and prints a short summary.

Run:  python analysis/subitem_mapping.py
      python analysis/subitem_mapping.py --selector deepseek

See docs/subitem_mapping.md for metric definitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from survey_features.config import OUTPUTS_DIR, SELECTORS  # noqa: E402
from survey_features.layout import (  # noqa: E402
    format_pilot_dir,
    selector_dirs,
    subitem_mapping_dir,
    subitem_run_dirs,
)
from survey_features.metrics import jaccard  # noqa: E402
from survey_features.subitem_map import MIN_SUBITEMS_TO_EXPAND  # noqa: E402

OUT = OUTPUTS_DIR
PRIMARY_DK = "nemotron"
ARM = "C"


def _iter_maps(map_dir: Path, disambig_key: str = PRIMARY_DK) -> list[dict]:
    if not map_dir.is_dir():
        return []
    rows = []
    for p in sorted(map_dir.glob(f"{ARM}__{disambig_key}__*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def _parent_rates_from_baseline(rec: dict) -> dict:
    """Parent-only rates from a format_pilot-style map record (no units[])."""
    feats = [f for f in rec.get("features", []) if f.get("piped")]
    n = len(feats)
    n_map = sum(1 for f in feats if f.get("selected_code"))
    n_bundled = sum(
        1 for f in feats if len(f.get("sub_items") or []) >= MIN_SUBITEMS_TO_EXPAND
    )
    return {
        "n_piped_parents": n,
        "parent_map_rate": (n_map / n) if n else None,
        "parent_none_rate": (1 - n_map / n) if n else None,
        "bundled_parent_frac": (n_bundled / n) if n else None,
        "n_subitem_units": 0,
        "subitem_map_rate": None,
        "subitem_none_rate": None,
    }


def cell_diagnostics(rec: dict) -> dict:
    """Per-cell metrics; works for expanded maps (units[]) and falls back for baseline."""
    out = {
        "survey": rec.get("survey"),
        "target": rec.get("target"),
        "country": rec.get("country"),
        "condition": rec.get("condition"),
        "arm": rec.get("arm"),
        "disambiguator": rec.get("disambiguator"),
        "mapping_mode": rec.get("mapping_mode", "parent_only"),
        "k_parent": len(rec.get("parent_codes") or rec.get("mapped_codes") or []),
        "k_subitem": len(rec.get("subitem_codes") or []),
        "k_expanded": len(rec.get("expanded_codes") or rec.get("mapped_codes") or []),
    }

    units = rec.get("units")
    if not units:
        out.update(_parent_rates_from_baseline(rec))
        out.update({
            "frac_subitems_map_given_parent_maps": None,
            "frac_subitems_map_given_parent_none": None,
            "parent_maps_all_subitems_miss": None,
            "parent_maps_some_subitems_miss": None,
            "parent_none_some_subitem_maps": None,
            "code_jaccard_parent_vs_subitems": None,
        })
        return out

    parents = [u for u in units if u.get("unit_kind") == "parent" and u.get("piped")]
    subs = [u for u in units if u.get("unit_kind") == "sub_item" and u.get("piped")]
    n_p = len(parents)
    n_s = len(subs)
    n_p_map = sum(1 for u in parents if u.get("selected_code"))
    n_s_map = sum(1 for u in subs if u.get("selected_code"))

    # Group sub_items by parent for conditional metrics
    by_parent: dict[str, dict] = {}
    for u in parents:
        by_parent[u["parent_feature"]] = {
            "parent_code": u.get("selected_code"),
            "subs": [],
        }
    for u in subs:
        slot = by_parent.setdefault(
            u["parent_feature"], {"parent_code": None, "subs": []},
        )
        slot["subs"].append(u.get("selected_code"))

    bundled = {k: v for k, v in by_parent.items() if len(v["subs"]) >= MIN_SUBITEMS_TO_EXPAND}
    # Also treat as bundled if parent feature listed enough sub_items even when
    # units were recorded (len(subs) already implies expansion happened).
    if not bundled and subs:
        # Any parent with >= MIN sub unit rows
        bundled = {
            k: v for k, v in by_parent.items()
            if len(v["subs"]) >= MIN_SUBITEMS_TO_EXPAND
        }

    def _mean_sub_hit(pred) -> float | None:
        vals = []
        for v in bundled.values():
            if not pred(v) or not v["subs"]:
                continue
            vals.append(sum(1 for c in v["subs"] if c) / len(v["subs"]))
        return float(sum(vals) / len(vals)) if vals else None

    n_bundled_parents = len(bundled)
    all_miss = some_miss = none_some = 0
    for v in bundled.values():
        hits = sum(1 for c in v["subs"] if c)
        if v["parent_code"]:
            if hits == 0:
                all_miss += 1
            if hits < len(v["subs"]):
                some_miss += 1
        else:
            if hits > 0:
                none_some += 1

    parent_codes = set(rec.get("parent_codes") or [])
    subitem_codes = set(rec.get("subitem_codes") or [])
    jac = jaccard(parent_codes, subitem_codes) if (parent_codes or subitem_codes) else None

    out.update({
        "n_piped_parents": n_p,
        "parent_map_rate": (n_p_map / n_p) if n_p else None,
        "parent_none_rate": (1 - n_p_map / n_p) if n_p else None,
        "bundled_parent_frac": (n_bundled_parents / n_p) if n_p else None,
        "n_subitem_units": n_s,
        "subitem_map_rate": (n_s_map / n_s) if n_s else None,
        "subitem_none_rate": (1 - n_s_map / n_s) if n_s else None,
        "frac_subitems_map_given_parent_maps": _mean_sub_hit(lambda v: bool(v["parent_code"])),
        "frac_subitems_map_given_parent_none": _mean_sub_hit(lambda v: not v["parent_code"]),
        "parent_maps_all_subitems_miss": (
            all_miss / n_bundled_parents if n_bundled_parents else None
        ),
        "parent_maps_some_subitems_miss": (
            some_miss / n_bundled_parents if n_bundled_parents else None
        ),
        "parent_none_some_subitem_maps": (
            none_some / n_bundled_parents if n_bundled_parents else None
        ),
        "code_jaccard_parent_vs_subitems": jac,
    })
    return out


def summarize(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"[{label}] no rows")
        return
    print(f"\n=== {label} (n_cells={len(df)}) ===")
    for col in [
        "parent_map_rate",
        "parent_none_rate",
        "bundled_parent_frac",
        "subitem_map_rate",
        "subitem_none_rate",
        "frac_subitems_map_given_parent_maps",
        "frac_subitems_map_given_parent_none",
        "parent_maps_all_subitems_miss",
        "parent_none_some_subitem_maps",
        "code_jaccard_parent_vs_subitems",
        "k_parent",
        "k_expanded",
    ]:
        if col not in df.columns or df[col].isna().all():
            continue
        print(f"  mean {col:40s} {df[col].mean():.4f}")


def run_selector(selector_key: str) -> pd.DataFrame | None:
    map_dir, diag_csv, _ = subitem_run_dirs(selector_key)
    records = _iter_maps(map_dir)
    if not records:
        print(f"[{selector_key}] no expanded maps under {map_dir}")
        # Baseline parent rates from format_pilot (informational)
        _, _, base_maps = selector_dirs(selector_key)
        base = _iter_maps(base_maps)
        if base:
            bdf = pd.DataFrame([cell_diagnostics(r) for r in base])
            summarize(bdf, f"{selector_key} format_pilot parent-only (baseline)")
        return None

    df = pd.DataFrame([cell_diagnostics(r) for r in records])
    diag_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(diag_csv, index=False)
    summarize(df, f"{selector_key} subitem_mapping")
    print(f"wrote {diag_csv}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--selector",
        choices=list(SELECTORS),
        default=None,
        help="one selector; default = all with maps or format_pilot baseline",
    )
    args = ap.parse_args()
    selectors = [args.selector] if args.selector else list(SELECTORS)
    root = subitem_mapping_dir(OUT)
    print(f"subitem_mapping root: {root}")
    print(f"format_pilot root:    {format_pilot_dir(OUT)}")
    for sel in selectors:
        run_selector(sel)


if __name__ == "__main__":
    main()
