# Similarity threshold — separate experiment (stub)

> **Status:** design stub only. Orthogonal to sub-item expansion.
> **Not** part of the subitem-mapping v1 run (`docs/subitem_mapping.md`).
> Suggested artifacts: `outputs/similarity_threshold/`.

## Question / motivation

Retrieval keeps candidates with embedding similarity ≥ `min_similarity` (currently
**0.30**, with `top_n=20`). That floor may be **prohibitive**: informative survey
variables could sit just below 0.3 and never reach the disambiguator, inflating
`none` rates or forcing coarse matches among the survivors.

We want to know where similarity mass sits and how often the threshold empties
the pool — before changing the production default.

## Orthogonality

| Experiment | What it varies | What it holds |
|------------|----------------|---------------|
| Sub-item mapping | Unit of analysis (parent vs sub_item) | `min_similarity=0.30`, MiniLM, kimi v1 |
| **This experiment** | Similarity score distributions / threshold | Parent-only mapping unit (unless nested later) |

Do **not** fold a threshold sweep into `outputs/subitem_mapping/`. A feature that
fails because its best candidate is 0.28 is a retrieval-threshold problem, not a
bundling problem.

## What to measure

1. **Per piped feature** (parent-only baseline maps, reuse extracts):
   - max similarity among retrieved candidates (pre-threshold)
   - top-k similarities (e.g. top-1 … top-20)
2. **Distribution summaries:** histograms / CDFs of max and top-k sims; share of
   mass **below 0.30**, in bands (e.g. `[0.20,0.30)`, `[0.30,0.40)`, …).
3. **Empty-pool none-rate:** fraction of features where the post-threshold pool
   is empty (attributable to the floor, vs disambig choosing `none` from a
   non-empty pool).
4. **Later (optional sweep):** re-map at a few thresholds (e.g. 0.20 / 0.25 / 0.30)
   and compare none-rates and code-set Jaccard — still under
   `outputs/similarity_threshold/`, not subitem_mapping.

## Isolation path

| Artifact | Path |
|----------|------|
| Similarity histograms / tables | `outputs/similarity_threshold/` |
| Optional remaps per threshold | `outputs/similarity_threshold/t_<thr>/…` |
| Manifest | `outputs/similarity_threshold/manifest.json` |

Prefer offline analysis over cached retrieval candidates from existing maps when
possible; full remaps only if candidate lists were not stored.

## Non-goals

- Changing `min_similarity` in the main MiniLM arm-C pipeline without evidence.
- Coupling threshold choice to sub_item expansion in a single confounding run.
