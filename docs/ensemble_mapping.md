# Ensemble retrieval / mapping — experiment design

> **Status:** v1 **kimi** map + score complete — see Results. Artifacts under
> `outputs/experiments/ensemble_mapping/runs/v1/` (legacy dual-resolve:
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
| Ensemble maps (v1) | `outputs/experiments/ensemble_mapping/runs/v1/union_max_sim_minilm_mpnet/kimi/maps/` |
| Ensemble scores (v1) | `…/kimi/scores_kimi.csv` |
| Comparison + latency (v1) | `…/runs/v1/comparison.csv`, `latency_*.csv` |
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

## Results

v1 kimi · arm C / nemotron · fusion `union_max_sim` MiniLM∪mpnet · max_fused=40.
Source: `outputs/experiments/ensemble_mapping/runs/v1/comparison.csv` +
`latency_comparison.csv` / `latency_cells.csv` (104 map cells; 306–312 paired
score rows vs baselines).

### Performance (ensemble − baseline)

| Baseline | n paired | mean map Jaccard | mean Δ VoR | mean \|Δ\| VoR | mean Δ model_acc | mean Δ CoI | mean Δ captured_imp | conclusions_move\* |
|----------|----------|------------------|------------|----------------|------------------|------------|---------------------|--------------------|
| MiniLM | 306 | **0.651** | +0.0083 | 0.028 | +0.0086 | −0.0092 | +0.032 | true |
| mpnet | 312 | **0.641** | +0.0021 | 0.028 | +0.0031 | −0.0028 | +0.035 | true |
| roberta | 309 | 0.582 | +0.0037 | 0.033 | +0.0044 | −0.0028 | +0.037 | true |

\*Flag is true because **mean \|Δ\|** on VoR / model_acc / cost_of_imperfect
exceeds 0.01 (same semantics as embedding_sensitivity). There is **no**
aggregate VoR sign flip. Mean VoR stays positive under ensemble (~0.050–0.051)
vs baselines (~0.043–0.048).

Absolute means (ensemble side of each join): model_acc ≈ 0.471, VoR ≈ 0.051,
cost_of_imperfect ≈ 0.069, captured_importance ≈ 0.26.

### Latency

| Span | Sum (104 cells) | Mean / cell |
|------|-----------------|-------------|
| retrieve (MiniLM + mpnet) | 90 s | 0.86 s |
| … of which MiniLM | — | 0.14 s |
| … of which mpnet | — | 0.72 s |
| disambig (Nemotron) | 1418 s | 13.6 s |
| cell wall (e2e map) | 1508 s | 14.5 s |

Phase wall (manifest map): **1547 s (~26 min)** with `map_workers=2`.
`n_disambig_calls == n_piped` on every cell (1× vs single-model). Retrieve is
~6% of cell wall; disambig dominates. Mean fused pool size ≈ 16.2 (cap 40).
Full single-model map timing was not auto-joined (only a partial kimi timing
log on disk); head-to-head is therefore structural: same disambig call count,
extra local encode/retrieve for the second embedder.

### Verdict

**Partial map stabilization; modest score lift; latency overhead small.**
Ensemble Jaccard vs MiniLM / mpnet (**0.65 / 0.64**) rises above the
single-model pairwise band from embedding_sensitivity (~0.56–0.60). Vs
roberta (not in the union) Jaccard stays ~0.58. Aggregate VoR / model_acc /
cost_of_imperfect move slightly in the favorable direction (largest vs MiniLM:
Δ VoR +0.008, Δ CoI −0.009); captured_importance rises ~+0.03–0.04. No VoR
sign flip — capability claims (beat matched-k random, gap to oracle) hold.
Disambig remains 1× and ~94% of map wall; multi-embed retrieve adds minutes,
not hours.

**Do not** promote ensemble as the default mapper yet: gains are real but
modest, and MiniLM main scores remain the reported baseline. Treat ensemble
as an optional mapping-stage lever when code-set stability matters more than
a few minutes of local retrieve cost.

### Implications

1. **Keep** MiniLM main-experiment scores as the reported capability baseline.
2. **Report** ensemble Jaccard (0.64–0.65 vs members) as evidence that union
   fusion partially reconciles MiniLM↔mpnet disagreement.
3. **Optional** for future mapping-sensitive work: MiniLM∪mpnet union → one
   Nemotron call (this fusion slug).
4. Rank-fuse / roberta-in-union / deepseek selector remain deferred extensions.

## Relation to embedding_sensitivity

Sensitivity showed **maps diverge, scores stable**. This experiment tested
whether fusing MiniLM+mpnet pools → one Nemotron call stabilises code sets.
v1: Jaccard vs members rises above the pairwise band; scores budge slightly
upward; latency accounting confirms 1× disambig. It does **not** replace the
MiniLM main-experiment baseline.

## Open decisions

1. Whether to promote ensemble (or mpnet alone) as the default retrieval for
   future main runs — **deferred**; v1 gains are modest (see Results).
2. Rank-fuse vs union — deferred; union is v1.
3. Aligning oracle near-dup embedder with ensemble members — out of scope for v1.
4. Deepseek + roberta-in-union — optional extensions only.
