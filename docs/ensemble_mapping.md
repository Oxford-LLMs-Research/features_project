# Ensemble retrieval / mapping — experiment design

> **Status:** v1 protocol locked for **kimi-only** map + score. Artifacts live under
> `outputs/experiments/ensemble_mapping/` (legacy dual-resolve:
> `outputs/ensemble_mapping/`) and never overwrite `outputs/main/` /
> `format_pilot/` or `embedding_sensitivity/`.
> Motivated by [embedding_sensitivity.md](embedding_sensitivity.md): maps diverge
> across embedders (Jaccard ~0.56–0.60) while aggregate scores stay stable.

## Question

Can **fusing candidate pools** from multiple sentence-transformers (then one
Nemotron disambiguation) **stabilise mapped code sets** relative to any single
embedder, without re-running selectors — and at what **latency** cost?

Specifically:

1. Does ensemble map Jaccard vs MiniLM / mpnet / roberta rise above the
   single-model pairwise ~0.56–0.60 band?
2. Do VoR / model_acc / cost_of_imperfect / captured_importance move vs those
   baselines (mean Δ and mean |Δ|)?
3. Latency honesty: ensemble adds multi-embed retrieve cost but must keep
   **1× disambig calls** (one per piped feature, not per embedder).

## Design (held fixed vs varied)

| Held fixed | Varied |
|------------|--------|
| Selector free-text (`gen`) + Qwen extract (reuse `main/`) | Retrieval source: MiniLM alone \| mpnet alone \| roberta alone \| **ensemble** |
| Disambiguator = **nemotron** | |
| Arm **C** only | |
| `top_n=20`, `min_similarity=0.30` (per model, then fuse) | |
| Oracle ceiling (`oracle.csv`; **MiniLM** near-dup filter — see asymmetry) | |
| **v1 selector = `kimi` only** | |

**Default ensemble members:** `all-MiniLM-L6-v2` ∪ `all-mpnet-base-v2`.
Roberta alone is a **baseline** (reuse sensitivity scores), not a default ensemble
member (cost / dim). Optional extension: add roberta to the union, or deepseek.

### Isolation

| Artifact | Path |
|----------|------|
| MiniLM maps / scores (baseline) | `outputs/main/kimi/` (unchanged) |
| mpnet / roberta maps / scores | `outputs/experiments/embedding_sensitivity/<slug>/kimi/` (unchanged) |
| Ensemble maps | `outputs/experiments/ensemble_mapping/union_max_sim_minilm_mpnet/kimi/maps/` |
| Ensemble scores | `…/kimi/scores_kimi.csv` |
| Comparison + latency | `outputs/experiments/ensemble_mapping/comparison.csv`, `latency_*.csv` |
| Provenance | `outputs/experiments/ensemble_mapping/manifest.json` |

Gen/extract are **never** re-run; only ensemble map + score write under
`ensemble_mapping/`.

## Fusion rule (v1 primary)

**Union + max similarity, capped at 2× top_n.**

For each piped feature:

1. For each embedder \(m\): dual-embed retrieve top_n=20; drop candidates with
   similarity &lt; 0.30 (**threshold per model**).
2. **Union** pools by `var_code`; set `similarity = max_m sim_m(code)`.
3. Sort by similarity descending; keep at most **`max_fused = 2 × top_n` (40)**.
4. **One** Nemotron disambiguation call on the fused lettered list.

### Why 2× top_n (not re-truncate to 20)

The point of the ensemble is to keep codes that only one model surfaces. If we
merge then cut back to 20 by max-sim, the higher-scoring model’s ranking
re-dominates and unique second-model candidates are discarded — undoing the
stabilization goal. Allowing up to 40 enlarges the disambig shortlist modestly
while preserving recall from the union. Mean pool size is logged per cell.

### Extension (not v1)

**Rank-fuse** (e.g. RRF over per-model ranks) — note only; do not block on it.
Would need a separate fusion slug and map re-run.

## Soundness notes

- **Disambig multiplicity:** ensemble must not multiply LLM disambig calls vs a
  single-model map of the same features. Timing records `n_disambig_calls == n_piped`.
- **Oracle asymmetry:** near-dup filtering in the oracle still uses MiniLM unless
  explicitly changed. Ensemble retrieval may surface codes that MiniLM would
  have filtered as near-dups of the target in other contexts — report as a
  known asymmetry (same as embedding_sensitivity).
- **Selector reuse:** same kimi extracts as main / sensitivity; only the
  retrieve→disambig stage differs.

## Metrics

### Performance

Paired arm-C / nemotron cells (join on survey, target, country, condition,
arm, disambiguator, k_spec):

- mean Δ / mean |Δ| on `model_acc`, `value_over_random`, `cost_of_imperfect`,
  `captured_importance` (ensemble − baseline)
- mean map Jaccard of `mapped_codes` vs MiniLM, mpnet, roberta
- `conclusions_move` if mean |Δ| &gt; 0.01 on primary metrics or VoR sign flips

Baselines are **reused** from disk (no re-score of single-model arms).

### Latency

From ensemble map JSON `timing` blocks (+ phase TimingLog):

| Span | What |
|------|------|
| `retrieve_wall_s_by_model` | embed+retrieve (cached survey emb load amortized) per model |
| `retrieve_wall_s_total` | sum across models |
| `disambig_wall_s` | all Nemotron calls for the cell |
| `cell_wall_s` | end-to-end map for that cell-condition |
| phase wall | full map phase |

**Expected shape:** retrieve cost ≈ sum of members (or less if emb caches warm);
disambig wall ≈ single-model (same call count); e2e map &gt; MiniLM alone mainly
from extra encode/retrieve. Score phase unchanged in structure vs main.

## Runtime ballpark (kimi-only v1)

Same grid as main arm-C kimi (~52 genuine cells × 2 conditions). Disambig
dominates wall-time (same as single-model map — ~1× call count). Extra cost is
local: load/encode with MiniLM + mpnet (survey emb usually cached from
sensitivity / main).

| Stage | Ballpark |
|-------|----------|
| Ensemble map (kimi extracts, nemotron, MiniLM∪mpnet) | **~1–2 h** serial disambig (similar to a single MiniLM map; +minutes for second embedder) |
| Score (natural k + fixed k=5/10, parallel cells) | **~20–40 min** with default score workers |
| **Total v1** | **~1.5–2.5 h** typical |
| + deepseek (optional) | ~2× map+score |

Smoke: `--limit 2` map before the full kimi sweep.

## How to run (v1 — kimi)

```bash
# Smoke
python scripts/run_ensemble_mapping.py --phase map --selector kimi --disambiguator nemotron --limit 2

# Full kimi ensemble map + score
python scripts/run_ensemble_mapping.py --phase map   --selector kimi --disambiguator nemotron --arms C
python scripts/run_ensemble_mapping.py --phase score --selector kimi

python analysis/ensemble_mapping.py --selector kimi
```

Optional: `--map-workers 8`, `--embedding-models all-MiniLM-L6-v2,all-mpnet-base-v2`,
`--run-tag <tag>` for non-canonical writes.

Prerequisites: `main/kimi/extracted/` present; MiniLM maps/scores in `main/`;
mpnet (and ideally roberta) scores under `embedding_sensitivity/` for full
baseline comparison. Missing single-model baselines skip those rows in
`comparison.csv`.

**Extension (not v1):** `--selector deepseek` after kimi looks sane; or add
`all-roberta-large-v1` to `--embedding-models`.

## Interpretation guide

| Pattern | Read as |
|---------|---------|
| Higher Jaccard vs each single model + stable VoR | Ensemble stabilises codes without changing capability claims |
| Jaccard still ~0.56 vs each, VoR flat | Union does not reconcile embedder disagreement (disambig still picks differently) |
| VoR / cost_of_imperfect move materially | Report as mapping-stage lever; revisit default retrieval |
| Latency: retrieve ↑, disambig ≈ 1× | Expected; document multi-embed overhead |

## Relation to embedding_sensitivity

Sensitivity showed **maps diverge, scores stable**. This experiment is the
recommended next step: fuse MiniLM+mpnet pools → one Nemotron call, measure
whether code sets stabilize and whether scores budge, with explicit latency
accounting. It does **not** replace the MiniLM main-experiment baseline unless
results clearly favor ensemble as the default mapper.

## Open decisions

1. Whether to promote ensemble (or mpnet alone) as the default retrieval for
   future main runs — only after v1 numbers.
2. Rank-fuse vs union — deferred; union is v1.
3. Aligning oracle near-dup embedder with ensemble members — out of scope for v1.
4. Deepseek + roberta-in-union — optional extensions only.
