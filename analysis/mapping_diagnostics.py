"""
Two offline diagnostics of the mapping stage. No LLM calls; embeddings come from the
per-survey caches, so this runs in minutes.

1. RETRIEVAL RECALL. For each cell, how much of the oracle's top-k appears anywhere in
   the candidate pools the mapper actually saw. `top_n=20` and `min_similarity=0.30`
   have never been validated (docs/similarity_threshold.md is an admitted stub), and
   the 28.6% "none" rate is currently uninterpretable: it conflates "the survey has no
   such variable", "retrieval missed it" and "the disambiguator was over-cautious".
   Recall bounds the middle term — it is the ceiling on what any disambiguator, however
   good, could have mapped.

2. DISAMBIGUATOR ABLATION. Replay each pool under a trivial rule — take the top-1
   candidate if its cosine clears a threshold, else "none" — and compare the resulting
   code sets against what the LLM disambiguator chose. Motivation: the format pilot
   already showed mapper *strength* barely matters (qwen235b - nemotron ~ 0), while
   disambiguation is ~94% of mapping wall time and the dominant token cost of a
   multi-model run. If the cheap rule agrees within noise, an entire stage and one
   model dependency can be dropped. If it does not, that is a result worth stating.

Usage:
    python analysis/mapping_diagnostics.py
    python analysis/mapping_diagnostics.py --selector kimi --top-k 10 --taus 0.4 0.5 0.6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from survey_features.config import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    OUTPUTS_DIR,
    PIPE_TYPES,
    SELECTORS,
)
from survey_features.layout import selector_dirs  # noqa: E402
from survey_features.metrics import jaccard, load_oracle_splits, oracle_topk_codes  # noqa: E402
from survey_features.retrieval import (  # noqa: E402
    load_or_build_survey_embeddings,
    make_embed_fn,
    retrieve_candidates_batch,
    target_excluded_codes,
)
from survey_features.surveys import extract_survey_variables, load_survey  # noqa: E402

TOP_N = 20
MIN_SIM = 0.30
DEFAULT_TAUS = (0.40, 0.45, 0.50, 0.55, 0.60)

_assets: dict = {}


def survey_assets(survey: str, embedding_model: str):
    key = (survey, embedding_model)
    if key not in _assets:
        _, meta = load_survey(survey, os.environ["DATA_CONFIG_PATH"])
        svars = extract_survey_variables(meta)
        emb, vcodes = load_or_build_survey_embeddings(svars, survey, embedding_model)
        _assets[key] = (svars, emb, vcodes)
    return _assets[key]


def iter_maps(selector: str, disambig: str, arm: str = "C"):
    _, _, map_dir = selector_dirs(selector)
    for p in sorted(map_dir.glob(f"{arm}__{disambig}__*.json")):
        try:
            yield p, json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue


def analyse(selector: str, disambig: str, embedding_model: str,
            top_k: int, taus: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    embed = make_embed_fn(embedding_model)
    recall_rows: list[dict] = []
    ablation: dict[float, list[dict]] = defaultdict(list)

    for _p, rec in iter_maps(selector, disambig):
        survey, target = rec["survey"], rec["target"]
        country, cond = rec["country"], rec["condition"]
        svars, emb, vcodes = survey_assets(survey, embedding_model)

        piped = [f for f in rec["features"]
                 if f.get("piped") and (f.get("type") in PIPE_TYPES)]
        if not piped:
            continue

        excluded = target_excluded_codes(target, svars, emb, vcodes, embed)
        pools = retrieve_candidates_batch(
            [(f.get("feature", ""), f.get("context", "")) for f in piped],
            embed, emb, vcodes, svars, excluded, TOP_N,
        )
        pools = [[c for c in pool if c["similarity"] >= MIN_SIM] for pool in pools]

        # ── 1. retrieval recall of the oracle's top-k ─────────────────────────
        rank, score = load_oracle_splits(target, country)
        retrieved = {c["var_code"] for pool in pools for c in pool}
        if rank:
            pos_rank = {c: v for c, v in rank.items() if score.get(c, 0.0) > 0}
            oracle_top = oracle_topk_codes(pos_rank, top_k)
            if oracle_top:
                hit = [c for c in oracle_top if c in retrieved]
                mapped = set(rec.get("mapped_codes") or [])
                recall_rows.append({
                    "survey": survey, "target": target, "country": country,
                    "condition": cond,
                    "n_features": len(piped),
                    "pool_union": len(retrieved),
                    "oracle_top_k": len(oracle_top),
                    "recall_at_20": len(hit) / len(oracle_top),
                    "mapped_of_oracle_top": len(
                        [c for c in oracle_top if c in mapped]
                    ) / len(oracle_top),
                })

        # ── 2. top-1-cosine vs the LLM's choice ───────────────────────────────
        llm_codes = [f.get("selected_code") for f in piped]
        for tau in taus:
            cheap = [
                (pool[0]["var_code"] if pool and pool[0]["similarity"] >= tau else None)
                for pool in pools
            ]
            agree = sum(1 for a, b in zip(cheap, llm_codes) if a == b)
            ablation[tau].append({
                "survey": survey, "target": target, "country": country,
                "condition": cond, "n_piped": len(piped),
                "agree": agree,
                "agree_rate": agree / len(piped),
                "llm_mapped": sum(1 for c in llm_codes if c),
                "cheap_mapped": sum(1 for c in cheap if c),
                "jaccard": jaccard({c for c in cheap if c}, {c for c in llm_codes if c}),
            })

    recall = pd.DataFrame(recall_rows)
    abl = pd.concat(
        [pd.DataFrame(rows).assign(tau=tau) for tau, rows in ablation.items()],
        ignore_index=True,
    ) if ablation else pd.DataFrame()
    return recall, abl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selector", default="kimi", choices=list(SELECTORS))
    ap.add_argument("--disambiguator", default="nemotron")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if not os.environ.get("DATA_CONFIG_PATH"):
        raise SystemExit("DATA_CONFIG_PATH is not set in .env")
    out_root = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR
    out_dir = out_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    recall, abl = analyse(
        args.selector, args.disambiguator, args.embedding_model,
        args.top_k, tuple(args.taus),
    )

    print(f"\n=== 1. RETRIEVAL RECALL (oracle top-{args.top_k} within top-{TOP_N} pools) ===")
    if recall.empty:
        print("  no cells with oracle importances on disk")
    else:
        print(f"  cells: {len(recall)}   mean features/cell: {recall.n_features.mean():.1f}"
              f"   mean pool union: {recall.pool_union.mean():.0f} variables")
        print(f"  recall@{TOP_N}                     mean {recall.recall_at_20.mean():.3f}"
              f"   median {recall.recall_at_20.median():.3f}")
        print(f"  of oracle top-k actually MAPPED  mean {recall.mapped_of_oracle_top.mean():.3f}")
        print(f"\n  CEILING: retrieval surfaces {recall.recall_at_20.mean():.1%} of the "
              f"oracle's top-{args.top_k} anywhere in the pools the mapper saw.")
        print(f"  Even a perfect disambiguator could not have mapped more than that.")
        print(f"  Actually mapped: {recall.mapped_of_oracle_top.mean():.1%}.")
        print("\n  The gap between the two is NOT all disambiguator error: an oracle feature")
        print("  can sit in a pool as a near-miss candidate for a request that was about")
        print("  something else entirely, and declining it is then correct. The defensible")
        print("  number is the ceiling.")
        print("\n  by survey:")
        print(recall.groupby("survey")[["recall_at_20", "mapped_of_oracle_top"]]
              .mean().round(3).to_string())
        recall.to_csv(out_dir / "mapping_recall.csv", index=False)

    print("\n=== 2. DISAMBIGUATOR ABLATION (top-1 cosine >= tau vs the LLM) ===")
    if abl.empty:
        print("  no map files found")
    else:
        g = abl.groupby("tau").agg(
            agree_rate=("agree_rate", "mean"),
            jaccard=("jaccard", "mean"),
            llm_mapped=("llm_mapped", "mean"),
            cheap_mapped=("cheap_mapped", "mean"),
        ).round(3)
        print(g.to_string())
        best = g.jaccard.idxmax()
        print(f"\n  best agreement at tau={best}: Jaccard {g.loc[best,'jaccard']:.3f}, "
              f"per-feature agreement {g.loc[best,'agree_rate']:.3f}")
        print("  (for scale: swapping the EMBEDDING model moves map Jaccard to ~0.56-0.60,")
        print("   and that swap left aggregate capability claims unchanged.)")
        abl.to_csv(out_dir / "disambiguator_ablation.csv", index=False)

    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
