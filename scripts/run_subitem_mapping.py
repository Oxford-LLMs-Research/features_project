"""
Sub-item mapping experiment runner (isolated from format_pilot).

Reuses gen/extract from outputs/format_pilot/<selector>/. Writes maps under
outputs/subitem_mapping/<selector>/ only. Score phase is deliberately stubbed
until diagnostics are reviewed (docs/subitem_mapping.md).

Examples:
  python scripts/run_subitem_mapping.py --phase map --selector deepseek --disambiguator nemotron --limit 2
  python scripts/run_subitem_mapping.py --phase map --selector deepseek --disambiguator nemotron --arms C
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survey_features.config import (  # noqa: E402
    CONDITIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SELECTOR,
    DISAMBIGUATORS,
    OUTPUTS_DIR,
    PIPE_TYPES,
    SELECTORS,
)
from survey_features.layout import (  # noqa: E402
    cell_tag,
    genuine_cells,
    selector_dirs,
    subitem_mapping_dir,
    subitem_run_dirs,
)

OUT = OUTPUTS_DIR
_survey_cache: dict = {}


def mapper_generate_fn(mapper_model: str):
    """Extractor/disambiguator client (same contract as scripts/run_main.py)."""
    from survey_features.llm import make_generate_fn
    fn, _ = make_generate_fn(
        base_url=os.environ.get("DISAMBIG_BASE_URL") or None,
        api_key=os.environ.get("DISAMBIG_API_KEY") or None,
        model=mapper_model,
        on_error="empty",
    )
    return fn


def survey_assets(survey_id, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    key = (survey_id, embedding_model)
    if key not in _survey_cache:
        from survey_features.retrieval import load_or_build_survey_embeddings
        from survey_features.surveys import extract_survey_variables, load_survey
        _, meta = load_survey(survey_id, os.environ["DATA_CONFIG_PATH"])
        svars = extract_survey_variables(meta)
        emb, vcodes = load_or_build_survey_embeddings(svars, survey_id, embedding_model)
        _survey_cache[key] = (svars, emb, vcodes)
    return _survey_cache[key]


def _upsert_manifest(**fields) -> None:
    root = subitem_mapping_dir(OUT)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "experiment": "subitem_mapping",
        "baseline": {
            "scores_root": "outputs/format_pilot",
            "note": "Parent-only MiniLM arm-C maps/scores; never overwritten by this runner.",
        },
        "runs": [],
    }
    data.setdefault("runs", []).append({"ts": datetime.now(timezone.utc).isoformat(), **fields})
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def phase_map(selector_key: str, disambig_key: str, arms=("C",), force=False, limit=None):
    """Expand parent + bundled sub_item mapping units; write under subitem_mapping/."""
    from survey_features.retrieval import make_embed_fn
    from survey_features.subitem_map import expanded_cell_to_record, map_features_with_subitems

    emb_model = DEFAULT_EMBEDDING_MODEL
    dmodel = DISAMBIGUATORS[disambig_key]
    dgen = mapper_generate_fn(dmodel)
    embed = make_embed_fn(emb_model)
    _, extract_dir, _ = selector_dirs(selector_key)
    map_dir, _, _ = subitem_run_dirs(selector_key)
    map_dir.mkdir(parents=True, exist_ok=True)

    cells = genuine_cells(OUT)
    if limit:
        cells = cells[:limit]
    print(
        f"[subitem-map] selector={selector_key} disambiguator={disambig_key} "
        f"arms={arms} cells={len(cells)} embedding={emb_model} -> {map_dir}"
    )

    for i, (survey, target, country) in enumerate(cells):
        svars, emb, vcodes = survey_assets(survey, emb_model)
        excluded = {target}
        ctag = cell_tag(survey, target, country)

        if "C" not in arms:
            print("  ! only arm C is supported for subitem mapping in v1")
            continue

        ep = extract_dir / f"{ctag}.json"
        feat_by_cond = (
            json.loads(ep.read_text(encoding="utf-8"))["features"] if ep.is_file() else None
        )
        for cond in CONDITIONS:
            op = map_dir / f"C__{disambig_key}__{ctag}__{cond}.json"
            if op.is_file() and not force:
                continue
            if feat_by_cond is None:
                print(
                    f"  ! missing extraction for {ctag}; "
                    "run scripts/run_main.py --phase extract first"
                )
                continue
            cm = map_features_with_subitems(
                f"{target}_{country}",
                "C_free",
                feat_by_cond.get(cond, []),
                emb,
                vcodes,
                svars,
                embed,
                dgen,
                mapper_model=dmodel,
                excluded_codes=excluded,
                pipe_types=PIPE_TYPES,
            )
            rec = expanded_cell_to_record(
                survey, target, country, cond, "C", disambig_key, cm, emb_model,
            )
            op.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  [{i+1}/{len(cells)}] {survey} {target} {country} mapped ({disambig_key})")

    print(f"[subitem-map] done -> {map_dir}")
    _upsert_manifest(
        phase="map",
        selector=selector_key,
        disambiguator=disambig_key,
        arms=list(arms),
        embedding_model=emb_model,
        limit=limit,
    )


def phase_score(selector_key: str, k_modes=("parent", "expanded"), force=False, limit=None):
    """Score stub — diagnostics-first; full XGB wiring deferred (open decision #3)."""
    raise SystemExit(
        "score phase not wired yet (design: diagnostics-first). "
        "After maps exist, run: python analysis/subitem_mapping.py\n"
        f"Requested selector={selector_key} k_modes={k_modes} force={force} limit={limit}. "
        "See docs/subitem_mapping.md."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--phase", choices=["map", "score"], required=True)
    ap.add_argument("--selector", choices=list(SELECTORS), default=DEFAULT_SELECTOR)
    ap.add_argument("--disambiguator", choices=list(DISAMBIGUATORS),
                    help="required for --phase map")
    ap.add_argument("--arms", default="C", help="map arms (default: C only)")
    ap.add_argument(
        "--k-modes",
        default="parent,expanded",
        help="comma-separated score modes: parent,expanded,subitems_only (score phase)",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.phase == "map":
        if not args.disambiguator:
            ap.error("--disambiguator required for --phase map")
        phase_map(
            args.selector,
            args.disambiguator,
            arms=tuple(a.strip() for a in args.arms.split(",") if a.strip()),
            force=args.force,
            limit=args.limit,
        )
    else:
        modes = tuple(m.strip() for m in args.k_modes.split(",") if m.strip())
        phase_score(args.selector, k_modes=modes, force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
