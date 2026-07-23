# Embedding-model sensitivity — protocol & results memo

> **Status:** runs complete; verdict below. Aggregate capability claims are robust to
> embedder size; mapped code sets are not — see Results.

## Question

Do free-text pipeline conclusions (map → nemotron disambiguation → XGB evals) move when
we swap the sentence-transformer used for survey-variable retrieval?

## Design (held fixed vs varied)

| Held fixed | Varied |
|------------|--------|
| Selector free-text (`gen`) + Qwen-235B extract | Embedding model (size ladder) |
| Disambiguator = **nemotron** | |
| Arm **C** only (free-text structured features) | |
| Oracle ceiling (`oracle.csv`; MiniLM near-dup filter) | |
| `top_n=20`, `min_similarity=0.30` | |
| Both selectors: `deepseek`, `kimi` | |

**Model shortlist** (symmetric `all-*` encode path; no E5/BGE query prefixes):

| Role | Model | Approx. size | Dim |
|------|--------|--------------|-----|
| Baseline | `all-MiniLM-L6-v2` | ~22M | 384 |
| Mid | `all-mpnet-base-v2` | ~110M | 768 |
| Large | `all-roberta-large-v1` | ~355M | 1024 |

Baseline maps/scores stay in `outputs/format_pilot/` (main experiment). Mid/large
re-runs write only under `outputs/embedding_sensitivity/<slug>/<selector>/` so prior
artifacts are never overwritten. Provenance: `outputs/embedding_sensitivity/manifest.json`.

Threshold / top-k are **not** swept here (orthogonal knobs; see `pipeline_critique.md` §2b).

## How to run

Serial (one phase at a time):

```bash
python scripts/run_main.py --phase map   --selector deepseek --disambiguator nemotron --arms C --embedding-model all-mpnet-base-v2
python scripts/run_main.py --phase score --selector deepseek --embedding-model all-mpnet-base-v2
# … kimi; then all-roberta-large-v1 × both selectors

python analysis/embedding_sensitivity.py
```

**Parallel resume** (recommended — overlaps API-bound maps with CPU-bound score):

```powershell
powershell -File scripts/run_embedding_sensitivity_parallel.ps1
```

Wave 1 runs `score kimi×mpnet` alongside both `all-roberta-large-v1` maps; wave 2 scores both roberta runs; then analysis. Map skips existing JSON. Optional: `$env:SCORE_N_DRAWS=5` to shorten score.

Smoke: add `--limit 2` on one selector × one model first.

## What to read from `comparison.csv`

Per selector × alt model:

- `mean_delta_*` / `mean_abs_delta_*` on `model_acc`, `value_over_random`,
  `cost_of_imperfect`, `captured_importance`
- `mean_map_jaccard` / `frac_cells_codes_differ` — how much mapped code sets move
- `conclusions_move` — true if mean |Δ| on primary accuracy metrics > 0.01 or mean VoR
  sign flips vs MiniLM

### Interpretation guide

- **Maps and scores close to MiniLM** → main results not sensitive to embedder size in
  this range; keep `all-MiniLM-L6-v2` as the default.
- **Maps diverge, scores stable** → retrieval changes candidates/codes but eval
  conclusions are robust; report as a mapping caveat, not a claim threat.
- **Scores move materially** (esp. VoR / cost_of_imperfect) → report as a limitation;
  consider upgrading the default embedder for the main run.

## Results

| Selector | Alt model | n paired | mean Δ VoR | mean \|Δ\| VoR | mean map Jaccard | frac codes differ | conclusions_move\* |
|----------|-----------|----------|------------|----------------|------------------|-------------------|--------------------|
| deepseek | mpnet | 309 | +0.0006 | 0.029 | 0.584 | 0.961 | true |
| deepseek | roberta | 306 | −0.0069 | 0.030 | 0.560 | 0.971 | true |
| kimi | mpnet | 306 | +0.0063 | 0.022 | 0.605 | 0.941 | true |
| kimi | roberta | 306 | +0.0045 | 0.025 | 0.559 | 0.971 | true |

\*Flag is true because **mean \|Δ\|** on VoR / model_acc / cost_of_imperfect exceeds 0.01.
There is **no** aggregate VoR sign flip. Mean VoR stays positive under all three embedders
(~0.040–0.048); model_acc correlation with MiniLM is ~0.96; VoR correlation ~0.83–0.87;
~83–88% of cells keep the same VoR sign.

**Verdict:** **Maps diverge, aggregate scores stable.** Retrieval+disambiguation code sets
are embedder-sensitive (Jaccard ~0.56–0.60). The main capability claims — beat matched-k
random, leave a gap to oracle — do **not** flip. Treat as a **mapping caveat**, not a
reason to discard MiniLM main-experiment scores. Selector head-to-heads are soft
(deepseek−kimi VoR gap flips from +0.003 under MiniLM to −0.008 under roberta).

### Implications for the main pipeline

1. **Keep** `all-MiniLM-L6-v2` main results as the reported baseline for capability claims.
2. **Report** embedder sensitivity on mapped codes in limitations / methods.
3. **Do not** re-run the full main experiment on roberta solely for this (aggregate VoR
   barely moves; deepseek×roberta even dips slightly).
4. Oracle near-dup still uses MiniLM in this sweep — if the default retrieval embedder
   changes later, align the oracle filter to the same model.

### How to address mapping sensitivity

| Priority | Action | Why |
|----------|--------|-----|
| Now | Document this verdict; separate mean Δ vs mean \|Δ\| in reporting | Avoid over-reading `conclusions_move` |
| Next | Ensemble retrieval (union / rank-fuse MiniLM+mpnet top-N → one Nemotron call) | Stabilise candidate pool without new selector runs |
| Next | Cross-embedder Jaccard as uncertainty on mapped codes / k | Quantify mapping fragility in results |
| Optional | Default future runs to `all-mpnet-base-v2` + matching oracle near-dup | Mid-size; slightly higher VoR here |
| Later | Human mapping validation + recall@20 (already in design) | Ground-truth the unstable stage |
| Later | Threshold / top-k sweep (critique §2b) | Orthogonal knobs; not required to rescue claims |
