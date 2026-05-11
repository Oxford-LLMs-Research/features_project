#!/usr/bin/env python3
"""
Build prelim/prelim_manifest.yaml — 5 targets × 5 countries per survey using
algorithmic strata (topics + cardinality) from live metadata/data.

Writes detailed candidate summaries under prelim/target_selection_detail.yaml for audit.
Requires DATA_CONFIG_PATH.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main():
    os.chdir(ROOT)
    cfg = os.environ.get("DATA_CONFIG_PATH")
    if not cfg:
        print("DATA_CONFIG_PATH is not set")
        sys.exit(1)

    from phase0b_mapping import extract_survey_variables
    from run_grid import (
        SURVEY_COUNTRY_COL,
        build_admin_cols,
        build_country_code_map,
        load_survey,
    )

    from prelim.target_selection import (
        DEFAULT_LARGE_CAP,
        build_candidates,
        filter_candidates_for_countries,
        pick_spread_country_names,
        select_five_targets,
    )

    raw_cap = os.environ.get("LARGE_CAP", "").strip()
    try:
        large_cap = int(raw_cap) if raw_cap else DEFAULT_LARGE_CAP
    except ValueError:
        print(
            f"  [warn] LARGE_CAP={raw_cap!r} is not an integer; "
            f"falling back to default={DEFAULT_LARGE_CAP}"
        )
        large_cap = DEFAULT_LARGE_CAP
    print(f"[build_manifest] large_cap={large_cap}  (env LARGE_CAP overrides)")

    surveys = sorted(s for s in SURVEY_COUNTRY_COL if s != "ess_wave_10")

    manifest_surveys: dict = {}
    detail_surveys: list = []

    for sid in surveys:
        print(f"[build_manifest] survey={sid}")
        data, meta = load_survey(sid, cfg)
        country_col = SURVEY_COUNTRY_COL[sid]
        cmap = build_country_code_map(meta, country_col, data)
        sv = extract_survey_variables(meta)

        cntry = pick_spread_country_names(list(cmap.keys()), 5)
        cand = build_candidates(data, meta, sv, large_cap=large_cap)
        bucket_counts = Counter(c.bucket for c in cand)
        print(
            f"  bucket counts (pre-country filter): "
            f"binary={bucket_counts.get('binary', 0)}, "
            f"tertiary={bucket_counts.get('tertiary', 0)}, "
            f"mid={bucket_counts.get('mid', 0)}, "
            f"large={bucket_counts.get('large', 0)}"
        )
        admin_cols = build_admin_cols(meta, country_col)
        cand = filter_candidates_for_countries(
            cand,
            data,
            country_col,
            cntry,
            cmap,
            metadata=meta,
            admin_cols=admin_cols,
        )
        bucket_counts_post = Counter(c.bucket for c in cand)
        print(
            f"  bucket counts (post-country filter): "
            f"binary={bucket_counts_post.get('binary', 0)}, "
            f"tertiary={bucket_counts_post.get('tertiary', 0)}, "
            f"mid={bucket_counts_post.get('mid', 0)}, "
            f"large={bucket_counts_post.get('large', 0)}"
        )
        for bucket, target_n in (("binary", 2), ("mid", 2), ("large", 1)):
            if bucket_counts_post.get(bucket, 0) < target_n:
                print(
                    f"  [warn] {sid}: only {bucket_counts_post.get(bucket, 0)} "
                    f"{bucket} candidate(s) usable across all 5 countries "
                    f"(quota wants {target_n}); fallback will fill the slot"
                )
        if len(cand) < 5:
            print(
                f"  [warn] only {len(cand)} candidates usable across all "
                f"{len(cntry)} countries — relax min_valid_rows or country list if needed"
            )
        codes = select_five_targets(cand)

        picks_det = [{"var": code} for code in codes]
        c_by_code = {c.var_code: c for c in cand}
        enriched = []
        for code in codes:
            cc = c_by_code.get(code)
            if cc:
                enriched.append(
                    {
                        "variable": code,
                        "section": cc.section,
                        "topic_key": cc.topic,
                        "bucket": cc.bucket,
                        "n_categories_metadata": cc.n_cats_meta,
                        "n_categories_empirical_sample": cc.n_cats_empirical,
                    }
                )
            else:
                enriched.append({"variable": code})

        manifest_surveys[sid] = {"targets": codes, "countries": cntry}
        detail_surveys.append(
            {
                "survey_id": sid,
                "country_column": country_col,
                "n_candidate_vars_scored": len(cand),
                "selected": enriched,
                "countries_selected": [{"name": n, "code": cmap[n]} for n in cntry],
            }
        )

    manifest = {
        "description": (
            "Preliminary multi-survey grid: ESS uses ess_wave_11 only "
            "(ess_wave_10 omitted). Targets chosen for topic + cardinality "
            f"spread under quota 2 binary + 2 mid (4-5 cats) + 1 large "
            f"(5-{large_cap} cats); high-cardinality targets above {large_cap} "
            "categories are excluded."
        ),
        "surveys": manifest_surveys,
    }

    prelim_dir = ROOT / "prelim"
    prelim_dir.mkdir(parents=True, exist_ok=True)
    mf = prelim_dir / "prelim_manifest.yaml"
    with open(mf, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {mf}")

    sf = prelim_dir / "target_selection_detail.yaml"
    with open(sf, "w", encoding="utf-8") as f:
        yaml.safe_dump({"surveys": detail_surveys}, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {sf}")


if __name__ == "__main__":
    main()
