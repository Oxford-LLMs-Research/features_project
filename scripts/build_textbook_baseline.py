"""
Resolve the frozen "textbook" demographic baseline to concrete variable codes per survey.

Why this baseline exists: the pipeline's only null was a random draw from the ~240
variables the oracle ranked over. Beating that is a low bar — it says the model knows
*something*, not that it knows anything specific to the question it was asked. The
textbook set is what a competent researcher would write down without reading the target
question at all (age, education, income, ...), so `model - textbook` is the contrast
that supports the paper's actual claim.

The constructs live in config.TEXTBOOK_CONSTRUCTS and are pre-registered: they are
identical for every cell, every condition and every model, and they are resolved ONCE
per survey through the same retrieval + disambiguation path the pipeline uses for model
requests, so the baseline is subject to the same mapping attenuation as the treatment.

Output: outputs/cache/baselines/textbook__<survey>.json
Review the resolved codes before scoring — a mis-mapped "education" would quietly
corrupt the headline comparison.

Usage:
    python scripts/build_textbook_baseline.py
    python scripts/build_textbook_baseline.py --survey wvs --force
    python scripts/build_textbook_baseline.py --show          # print what is cached
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
    DEFAULT_EMBEDDING_MODEL,
    DISAMBIGUATORS,
    OUTPUTS_DIR,
    TEXTBOOK_CONSTRUCTS,
)
from survey_features.mapping import _disambiguate_pool  # noqa: E402
from survey_features.retrieval import (  # noqa: E402
    load_or_build_survey_embeddings,
    make_embed_fn,
    retrieve_candidates_batch,
)
from survey_features.surveys import (  # noqa: E402
    SURVEY_COUNTRY_COL,
    extract_survey_variables,
    load_survey,
)

TOP_N = 20
MIN_SIMILARITY = 0.30


def baseline_path(survey: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return Path(outputs_dir) / "cache" / "baselines" / f"textbook__{survey}.json"


def overrides_path(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Optional hand-pinned codes: {survey: {construct_label: "CODE" | null}}.

    Auto-mapping is the DEFAULT and the fairer comparison: the baseline then passes
    through exactly the retrieval + disambiguation chain the model's own requests do,
    so both sides carry the same attenuation. Hand-pinning only the baseline would give
    it an advantage the treatment never gets, biasing against the model.

    Use overrides only for outright construct errors — the disambiguator forcing a match
    the prompt told it to decline (map to null), or picking the wrong variable when the
    right one was in the pool. Record any override in the audit memo.
    """
    return Path(outputs_dir) / "cache" / "baselines" / "textbook_overrides.json"


def load_overrides(survey: str, outputs_dir: Path) -> dict:
    p = overrides_path(outputs_dir)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get(survey, {}) or {}
    except Exception:
        return {}


def build_one(survey: str, disambig_key: str, embedding_model: str,
              outputs_dir: Path) -> dict:
    _, meta = load_survey(survey, os.environ["DATA_CONFIG_PATH"])
    svars = extract_survey_variables(meta)
    emb, vcodes = load_or_build_survey_embeddings(svars, survey, embedding_model)
    embed = make_embed_fn(embedding_model)

    from survey_features.llm import make_generate_fn
    dmodel = DISAMBIGUATORS[disambig_key]
    dgen, _ = make_generate_fn(
        base_url=os.environ.get("DISAMBIG_BASE_URL") or None,
        api_key=os.environ.get("DISAMBIG_API_KEY") or None,
        model=dmodel,
        on_error="empty",
    )

    pools = retrieve_candidates_batch(
        [(label, context) for label, context in TEXTBOOK_CONSTRUCTS],
        embed, emb, vcodes, svars, excluded=set(), top_n=TOP_N,
    )

    overrides = load_overrides(survey, outputs_dir)
    constructs = []
    for (label, context), pool in zip(TEXTBOOK_CONSTRUCTS, pools):
        pool = [c for c in pool if c["similarity"] >= MIN_SIMILARITY]
        code, text, raw, _status = _disambiguate_pool(label, context, pool, dgen)
        entry = {
            "label": label,
            "context": context,
            "var_code": code,
            "question_text": text,
            "raw": raw.strip()[:80],
            "n_candidates": len(pool),
            "overridden": False,
        }
        if label in overrides:
            pinned = overrides[label]
            entry.update({
                "auto_var_code": code,
                "var_code": pinned,
                "question_text": svars.get(pinned) if pinned else None,
                "overridden": True,
            })
        constructs.append(entry)
        status = entry["var_code"] or "none"
        mark = " (override)" if entry["overridden"] else ""
        print(f"    {label:32s} -> {status:14s} {(entry['question_text'] or '')[:52]}{mark}")

    return {
        "survey": survey,
        "embedding_model": embedding_model,
        "disambiguator": dmodel,
        "top_n": TOP_N,
        "min_similarity": MIN_SIMILARITY,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constructs": constructs,
    }


def show(survey: str, outputs_dir: Path) -> None:
    p = baseline_path(survey, outputs_dir)
    if not p.is_file():
        print(f"{survey:16s} (not built)")
        return
    rec = json.loads(p.read_text(encoding="utf-8"))
    mapped = [c for c in rec["constructs"] if c.get("var_code")]
    print(f"\n{survey}  ({len(mapped)}/{len(rec['constructs'])} mapped, "
          f"{rec['disambiguator']}, {rec['built_at']})")
    for c in rec["constructs"]:
        code = c.get("var_code") or "-- none --"
        print(f"  {c['label']:32s} {code:14s} {(c.get('question_text') or '')[:62]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", nargs="+", default=None,
                    help="Surveys to build (default: all).")
    ap.add_argument("--disambiguator", default="nemotron", choices=list(DISAMBIGUATORS))
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--force", action="store_true", help="Rebuild even if cached.")
    ap.add_argument("--show", action="store_true", help="Print cached baselines and exit.")
    args = ap.parse_args()

    outputs_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR
    surveys = args.survey or list(SURVEY_COUNTRY_COL)

    if args.show:
        for s in surveys:
            show(s, outputs_dir)
        return

    if not os.environ.get("DATA_CONFIG_PATH"):
        raise SystemExit("DATA_CONFIG_PATH is not set in .env")

    for survey in surveys:
        out = baseline_path(survey, outputs_dir)
        if out.is_file() and not args.force:
            print(f"[skip] {survey}: {out} exists (use --force)")
            continue
        print(f"\n[textbook] {survey}")
        try:
            rec = build_one(survey, args.disambiguator, args.embedding_model, outputs_dir)
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        n = sum(1 for c in rec["constructs"] if c.get("var_code"))
        print(f"  {n}/{len(rec['constructs'])} mapped -> {out}")

    print("\nReview the resolved codes before scoring:")
    print("  python scripts/build_textbook_baseline.py --show")


if __name__ == "__main__":
    main()
