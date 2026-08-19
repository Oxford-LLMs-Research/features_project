# Experiment registry

One place to find every experiment: **why it was run, which code produced it, where the
artifacts live, and what the result was**. Paths below are relative to the Dropbox
outputs root (`SURVEY_FEATURES_OUTPUTS`). **When** to add an entry is in
[`CONTRIBUTING.md`](../CONTRIBUTING.md) (named `experiments/` writes, citable
`--run-tag` sweeps, and anything that can change confirmatory numbers). Fill
**Result** when the run finishes (or mark `status: abandoned`).

Past runs on other branches / snapshots still belong here — the registry is the map,
not the code tree.

---

## How to register

1. Copy the **Entry template** below into a new section (newest experiments at the top of Active / Complete). Folder name and slug follow [`CONTRIBUTING.md`](../CONTRIBUTING.md) (Experiment names and storage).
2. Set **Stage** to exactly one pipeline tag from the vocabulary below (primary locus of the manipulation).
3. Fill **Rationale** and **Result** in ≤3 short sentences each (“to test if X improves Y” / “X does not improve Y; caveat …”).
4. Record the **git commit** (or tag) that *produced* the artifacts — not “whatever HEAD was when you wrote the memo”.
5. Link **inputs** and **outputs** as paths under `outputs/` (or snapshot restore commands). Prefer concrete files over directory-only pointers when a headline file exists.
6. Add a one-line row to the **Index** table (keep the index sorted by Stage, then status).
7. When **Result** rests on a paired contrast (treatment vs baseline), add a **Contrast detail** block — do not stop at mean Δ alone.

**Status vocabulary:** `design` → `running` → `complete` | `abandoned` | `superseded`.

**Contrast reporting** (required for complete contrast experiments; use for past/present/future so claims stay comparable):

Means cancel cell-level heterogeneity. Every paired contrast Result must report, at the analysis `k_spec` (usually `model`):

| Piece | Why |
|-------|-----|
| Mean **and** median Δ | median resists cancelation / outliers |
| Share Δ>0 and Δ<0 | directionality without magnitude |
| Sign concordance (PI & VoR): both+ / both− / conflict | joint-endpoint honesty |
| By condition (e.g. `country_provided` vs `unprompted`) | stratum cancelation |
| By survey (mean Δ + +/- counts) | survey cancelation |
| Survey-level share with mean Δ>0 | avoids overweighting multi-k cells |

Paste tables under **Contrast detail** (or regenerate via `scripts/analyze_stack_experiments.py` → `outputs/experiments/_analysis/registry_contrast_blocks.md` for stack experiments). Older entries without reconstructable pairs: keep the ≤3-sentence Result and add `Contrast detail: not reconstructed (pre-convention).`

**Stage vocabulary** (pick one — the stage you *changed*):

| Stage | Means |
|-------|--------|
| `oracle` | Ground-truth PI / contract / fit procedure |
| `grid` | Which cells are in scope (leakage, targets, countries) |
| `elicitation` | Selector prompt / free-text vs JSON / conditions |
| `extraction` | Essay → typed feature list (fixed extractor / taxonomy) |
| `retrieval` | Embeddings, candidate pools, similarity cutoffs, fusion |
| `mapping` | Disambiguation / dual-layer / code assignment |
| `scoring` | Downstream metrics, nulls, baselines, XGB eval |
| `end-to-end` | Full confirmatory loop or multi-selector zoo |

**Compute note:** selectors / extract / disambig run on a remote LLM API. Oracle and
score run locally (CPU). Record both when relevant.

---

## Entry template

```markdown
### `<slug>` — short title

| Field | Value |
|-------|-------|
| **Status** | design \| running \| complete \| abandoned \| superseded |
| **Stage** | oracle \| grid \| elicitation \| extraction \| retrieval \| mapping \| scoring \| end-to-end |
| **Dates** | designed YYYY-MM-DD; ran YYYY-MM-DD → YYYY-MM-DD |
| **Code** | entry script(s); optional design note |
| **Commit** | `<full-or-short-sha>` on branch `<name>` (link if remote) |
| **Compute** | LLM: provider + models; local: CPU/RAM note for oracle/score if used |
| **Inputs** | paths / prerequisites (leakage grid, oracles, gen/extract caches, …) |
| **Outputs** | artifact root + headline files (scores CSV, comparison CSV, …) |

**Rationale.** ≤3 sentences. “To test whether X improves Y under Z.”

**Result.** ≤3 sentences. “X does / does not improve Y. Caveats: …”
Or `—` while `design` / `running`.

**Contrast detail.** (When Result is a paired contrast.) Mean | median | share Δ>0; PI↔VoR concordance; by condition; by survey. Else omit or `not reconstructed`.
```

---

## Index

Sorted by **Stage**, then status.

| Stage | Slug | Status | One-line claim | Outputs |
|-------|------|--------|----------------|---------|
| end-to-end | [`main-freetext`](#main-freetext--confirmatory-free-text-arm-c) | complete | Free-text + dual-layer map is the confirmatory instrument | `outputs/selectors/` |
| end-to-end | [`confirmatory-zoo`](#confirmatory-zoo--multi-selector-lock) | design | Not run | — |
| end-to-end | [`prelim-json-grid`](#prelim-json-grid--strict-json-appendix) | superseded | JSON-era magnitudes are a floor, not the estimate | snapshots / `outputs/grid/` |
| elicitation | [`prompt-sensitivity`](#prompt-sensitivity--selector-system-message) | complete | Keep confirmatory social_scientist system prompt | `outputs/experiments/prompt_sensitivity/` |
| extraction | [`extract-swap-minimax`](#extract-swap-minimax--minimax-extract--nemotron-disambig) | complete | MiniMax extract ≈ Qwen on PI/VoR; dearer — keep Qwen default | `outputs/experiments/pipeline_role_swap/minimax_nemotron/` |
| extraction | [`extract-type-pilot`](#extract-type-pilot--type-taxonomy-wording) | complete (pilot) | Type-prompt wording pilot | `outputs/experiments/extract_type_pilot*` |
| grid | [`leakage-audit`](#leakage-audit--genuine-cell-screen) | complete | Drop type-1 / leakage only; keep type-2/3 signal | `outputs/cache/audits/leakage_audit.csv` |
| mapping | [`pipeline-role-swap`](#pipeline-role-swap--minimax-extract--flash-disambig) | complete | Joint MiniMax+Flash rejected; Flash-as-disambig not followed up | `outputs/experiments/pipeline_role_swap/minimax_flash/` |
| mapping | [`subitem-mapping`](#subitem-mapping--dual-layer-pilot) | complete → promoted | Dual-layer locked into main | `outputs/experiments/subitem_mapping/` |
| oracle | [`oracle-v3`](#oracle-v3--measurement-level-honest-split) | superseded | Era-3 oracle retired 2026-08-19; contract v4 is current (onboarding §3) | `outputs/cache/cells_v3/` (archive) |
| scoring | [`pooled-country-value`](#pooled-country-value--is-country-worth-requesting) | design | Not run — pooled value of the country column vs selector country-requests | — |
| retrieval | [`embedding-sensitivity`](#embedding-sensitivity--sentence-transformer-swap) | complete | Embedder swap moves maps more than scores | `outputs/experiments/embedding_sensitivity/` |
| retrieval | [`ensemble-mapping`](#ensemble-mapping--minilm--mpnet-union) | complete (not promoted) | Union retrieval lifts Jaccard; small VoR gain | `outputs/experiments/ensemble_mapping/` |
| retrieval | [`underscore-label-embed`](#underscore-label-embed--feature-label-tokenization) | complete (ad-hoc) | `_` vs spaces barely moves MiniLM vectors | chat-only |
| retrieval | [`similarity-threshold`](#similarity-threshold--pool-cutoff) | design | Not run | — |
| scoring | [`textbook-baseline`](#textbook-baseline--frozen-demographics-null) | complete | Textbook demographics are the hard null for VoT | `outputs/cache/baselines/` |

---

## Active / complete

### `prompt-sensitivity` — selector system message

| Field | Value |
|-------|-------|
| **Status** | complete |
| **Stage** | elicitation |
| **Dates** | designed 2026-08-09; ran 2026-08-09; analyzed 2026-08-11 |
| **Code** | [`scripts/run_prompt_sensitivity.py`](../scripts/run_prompt_sensitivity.py); analysis [`scripts/analyze_stack_experiments.py`](../scripts/analyze_stack_experiments.py); arms in [`prompts.py`](../src/survey_features/prompts.py) / [`elicitation.py`](../src/survey_features/elicitation.py); cells [`data/prompt_sensitivity_cells.yaml`](../data/prompt_sensitivity_cells.yaml) |
| **Commit** | record SHA when artifacts are written |
| **Compute** | LLM: Nebius — selectors `moonshotai/Kimi-K2.6` + `deepseek-ai/DeepSeek-V4-Pro`; fixed Qwen extract + Nemotron disambig. Concurrent defaults: pipeline_workers=4, map_workers=8 (+ score ProcessPool). Local CPU for score. Token/cost via `TokenUsageLog` + [`data/nebius_pricing.json`](../data/nebius_pricing.json). |
| **Inputs** | [`data/prompt_sensitivity_cells.yaml`](../data/prompt_sensitivity_cells.yaml) (12 genuine cells); era-3 oracles; leakage genuine grid |
| **Outputs** | `outputs/experiments/prompt_sensitivity/<selector>/<arm>/{freetext,extracted,maps}/`; `scores_<selector>_<arm>.csv`; `outputs/logs/token_usage_*.jsonl` |

**Rationale.** To test whether the confirmatory system message (“You are a social science researcher.”) constrains selector feature essays relative to no system message or a neutral “helpful assistant,” holding extractor/disambiguator fixed.

**Result.** Keep confirmatory `social_scientist`. At `k_spec=model`, kimi `helpful` lifts mean PI/VoR vs scientist, but deepseek_v4 `helpful` loses PI (−0.045) — not consistent across selectors. `none` does not win both endpoints. Digests: `scripts/analyze_stack_experiments.py` → `outputs/experiments/_analysis/` (incl. `registry_contrast_blocks.md`).

**Contrast detail** (vs `social_scientist`; `k_spec=model`).

`kimi` / `none` (n=20): PI mean −0.059 | med +0.008 | >0 55%; VoR mean −0.003 | med −0.000 | >0 50%; both+/both−/conflict = 8/7/5. By condition: `country_provided` PI −0.166 (40%>0), `unprompted` PI +0.047 (70%>0). Surveys mean PI>0: 1/5 (WVS alone strongly positive).

| Survey | n | PI mean | VoR mean |
|--------|---|---------|----------|
| afrobarometer | 4 | −0.021 | −0.017 |
| arabbarometer | 4 | −0.076 | +0.025 |
| asianbarometer | 4 | −0.226 | −0.021 |
| ess_wave_11 | 4 | −0.107 | −0.012 |
| wvs | 4 | +0.134 | +0.011 |

`kimi` / `helpful` (n=20): PI mean +0.055 | med +0.039 | >0 80%; VoR mean +0.014 | med +0.011 | >0 65%; both+/both−/conflict = 10/1/9 (many PI↑ with VoR conflict). Surveys mean PI>0: 4/6.

| Survey | n | PI mean | VoR mean |
|--------|---|---------|----------|
| afrobarometer | 4 | +0.087 | −0.018 |
| arabbarometer | 4 | −0.057 | +0.047 |
| asianbarometer | 4 | +0.097 | +0.017 |
| ess_wave_11 | 4 | −0.045 | +0.004 |
| latinobarometer | 1 | +0.027 | +0.087 |
| wvs | 3 | +0.249 | +0.000 |

`deepseek_v4` / `none` (n=23): PI mean −0.010 | med +0.003 | >0 52%; VoR mean +0.011 | med +0.000 | >0 48%; both+/both−/conflict = 8/7/7. By condition: `country_provided` PI +0.055, `unprompted` PI −0.069. ESS alone drives large negative PI (−0.553).

| Survey | n | PI mean | VoR mean |
|--------|---|---------|----------|
| afrobarometer | 4 | +0.163 | +0.000 |
| arabbarometer | 4 | −0.011 | +0.033 |
| asianbarometer | 4 | +0.160 | +0.019 |
| ess_wave_11 | 4 | −0.553 | −0.028 |
| latinobarometer | 3 | −0.006 | +0.044 |
| wvs | 4 | +0.191 | +0.006 |

`deepseek_v4` / `helpful` (n=23): PI mean −0.045 | med −0.004 | >0 43%; VoR mean +0.004 | med −0.008 | >0 39%; both+/both−/conflict = 5/9/9. By condition: `country_provided` PI +0.015, `unprompted` PI −0.110.

| Survey | n | PI mean | VoR mean |
|--------|---|---------|----------|
| afrobarometer | 4 | +0.064 | +0.004 |
| arabbarometer | 4 | −0.201 | −0.063 |
| asianbarometer | 4 | +0.113 | +0.005 |
| ess_wave_11 | 4 | −0.482 | −0.030 |
| latinobarometer | 3 | +0.044 | +0.102 |
| wvs | 4 | +0.215 | +0.028 |

---

### `pipeline-role-swap` — MiniMax extract + Flash disambig

| Field | Value |
|-------|-------|
| **Status** | complete |
| **Stage** | mapping |
| **Dates** | designed 2026-08-09; ran 2026-08-09; analyzed 2026-08-11 |
| **Code** | [`scripts/run_pipeline_role_swap.py`](../scripts/run_pipeline_role_swap.py); models `ROLE_SWAP_EXTRACTOR` / `DISAMBIGUATORS["flash"]` in [`config.py`](../src/survey_features/config.py) |
| **Commit** | record SHA when artifacts land |
| **Compute** | LLM: Nebius [MiniMax-M3](https://tokenfactory.nebius.com/endpoints?modals=endpoint-details&model-id=MiniMaxAI/MiniMax-M3) extract + [DeepSeek-V4-Flash](https://tokenfactory.nebius.com/endpoints?modals=endpoint-details&model-id=deepseek-ai/DeepSeek-V4-Flash) disambig; concurrent pipeline_workers=4 / map_workers=8; local score. Token/cost via `TokenUsageLog`. |
| **Inputs** | gen essays from `outputs/experiments/prompt_sensitivity/kimi/social_scientist/freetext/` (default); [`data/prompt_sensitivity_cells.yaml`](../data/prompt_sensitivity_cells.yaml) |
| **Outputs** | `outputs/experiments/pipeline_role_swap/minimax_flash/{extracted,maps}/`; `scores_minimax_flash.csv`; `source_meta.json` |

**Rationale.** To test whether faster/cheaper extract and disambig models cut wall time without material damage to mapped codes or downstream scores, freeing Qwen-family capacity for selector work.

**Result.** Joint MiniMax+Flash rejected: hurts model-k mean PI (−0.096) and VoR (−0.018); Flash disambig is slow/costly via reasoning tokens. **Do not** follow up with Qwen+Flash. Extract-only follow-up: [`extract-swap-minimax`](#extract-swap-minimax--minimax-extract--nemotron-disambig).

**Contrast detail** (`minimax_flash` − Qwen+Nemotron baseline on kimi/`social_scientist` gens; `k_spec=model`; n=21).

| Metric | Mean Δ | Median Δ | Share Δ>0 | Share Δ<0 |
|--------|--------|----------|-----------|-----------|
| PI | −0.096 | +0.000 | 52% | 48% |
| VoR | −0.018 | +0.003 | 52% | 48% |
| VoT | −0.023 | +0.000 | 48% | 48% |

Sign concordance (PI & VoR): both+=8, both−=7, conflict=6. By condition: `country_provided` PI −0.171 (36%>0); `unprompted` PI −0.015 (70%>0). Surveys mean PI>0: 2/6. Map Jaccard vs baseline Nemotron maps: mean 0.436 (n=24).

| Survey | n | PI mean | VoR mean |
|--------|---|---------|----------|
| afrobarometer | 4 | −0.182 | −0.050 |
| arabbarometer | 4 | −0.221 | −0.053 |
| asianbarometer | 4 | +0.161 | +0.018 |
| ess_wave_11 | 4 | −0.279 | −0.006 |
| latinobarometer | 1 | −0.004 | −0.015 |
| wvs | 4 | +0.015 | −0.003 |

---

### `extract-swap-minimax` — MiniMax extract + Nemotron disambig

| Field | Value |
|-------|-------|
| **Status** | complete |
| **Stage** | extraction |
| **Dates** | designed 2026-08-11; ran 2026-08-11 |
| **Code** | [`scripts/run_pipeline_role_swap.py`](../scripts/run_pipeline_role_swap.py) `--extractor MiniMaxAI/MiniMax-M3 --disambiguator nemotron --run-key minimax_nemotron` |
| **Commit** | record SHA when artifacts land |
| **Compute** | LLM: Nebius MiniMax-M3 extract + Nemotron disambig (Qwen replaced only); concurrent pipeline_workers=4 / map_workers=8; local score. |
| **Inputs** | same gens as joint swap: `outputs/experiments/prompt_sensitivity/kimi/social_scientist/freetext/`; [`data/prompt_sensitivity_cells.yaml`](../data/prompt_sensitivity_cells.yaml) |
| **Outputs** | `outputs/experiments/pipeline_role_swap/minimax_nemotron/{extracted,maps}/`; `scores_minimax_nemotron.csv` |

**Rationale.** To test whether swapping only the extractor to MiniMax (keeping Nemotron disambig) preserves PI/VoR while freeing Qwen — isolating extract quality from Flash’s disambig cost/latency failure in `pipeline-role-swap`.

**Result.** Keep Qwen as confirmatory default. Mean Δ looks ~tied with Qwen+Nemotron (PI +0.013, VoR ≈0, VoT −0.003), but cell/survey cancelation is large (only 42% of pairs PI>0). Nemotron latency restored (~3.5 s mean). Extract $ still higher (~$0.13 vs ~$0.02 for Qwen on this grid). Viable capacity alternate if Qwen is saturated; not a cost win.

**Contrast detail** (`minimax_nemotron` − Qwen+Nemotron extract-only; `k_spec=model`; n=19).

| Metric | Mean Δ | Median Δ | Share Δ>0 | Share Δ<0 |
|--------|--------|----------|-----------|-----------|
| PI | +0.013 | +0.000 | 42% | 47% |
| VoR | −0.000 | +0.004 | 63% | 32% |
| VoT | −0.003 | +0.004 | 58% | 37% |

Sign concordance (PI & VoR): both+=7, both−=5, conflict=5. By condition: `country_provided` PI +0.034 (55%>0); `unprompted` PI −0.017 (25%>0). Surveys mean PI>0: 2/6 (WVS +0.281 pulls the mean). Map Jaccard vs baseline Nemotron maps: mean 0.519 (n=24).

| Survey | n | PI mean | VoR mean |
|--------|---|---------|----------|
| afrobarometer | 3 | −0.244 | −0.074 |
| arabbarometer | 4 | −0.067 | +0.022 |
| asianbarometer | 4 | +0.100 | +0.008 |
| ess_wave_11 | 4 | −0.001 | +0.010 |
| latinobarometer | 1 | +0.000 | +0.000 |
| wvs | 3 | +0.281 | +0.021 |

---

### `main-freetext` — confirmatory free-text Arm C

| Field | Value |
|-------|-------|
| **Status** | complete (baseline; dual-layer map locked) |
| **Stage** | end-to-end |
| **Dates** | free-text main through 2026; dual-layer promoted 2026-08-06 (`949a8cf`) |
| **Code** | [`scripts/run_main.py`](../scripts/run_main.py); library under [`src/survey_features/`](../src/survey_features/) (`mapping.py`, `extraction.py`, …). Design history on `main`: `docs/main_experiment_design.md` |
| **Commit** | dual-layer lock: `949a8cf` on `main` / carried into `rewrite/minimal-core`. Re-record SHA for any new selector sweep |
| **Compute** | LLM API (selector + fixed Qwen extractor + Nemotron disambig); local CPU for score / XGB |
| **Inputs** | `outputs/cache/audits/leakage_audit.csv` (genuine cells); `outputs/cache/cells/*/oracle.csv` (v3); `outputs/cache/baselines/textbook__*.json` |
| **Outputs** | `outputs/selectors/<selector>/{freetext,extracted,maps}/`; `outputs/selectors/scores_<selector>.csv` |

**Rationale.** To test whether an LLM, prompted in free text and mapped dual-layer onto survey variables, captures oracle importance and beats matched-k random and textbook demographic baselines across countries.

**Result.** Free text is the confirmatory instrument (JSON was a suppressed floor). Dual-layer mapping (parent + bundled sub_items → `expanded_codes`) is the headline map path. Quote only era-3-scored numbers; see onboarding §3.

---

### `leakage-audit` — genuine-cell screen

| Field | Value |
|-------|-------|
| **Status** | complete |
| **Stage** | grid |
| **Dates** | 2026 (re-run after any oracle contract change) |
| **Code** | [`scripts/leakage_audit.py`](../scripts/leakage_audit.py); classifier [`src/survey_features/grid_screen.py`](../src/survey_features/grid_screen.py); target catalog [`data/targets.yaml`](../data/targets.yaml) |
| **Commit** | record the SHA used for each audit refresh; logic landed with oracle rebuild era (`21c780d` lineage on `main`) |
| **Compute** | local CPU (+ optional `--with-data` single-feature XGB); no LLM |
| **Inputs** | `outputs/cache/cells/*/oracle.csv`; survey microdata via `DATA_CONFIG_PATH` when `--with-data` |
| **Outputs** | `outputs/cache/audits/leakage_audit.csv`; `leakage_audit_summary.json` |

**Rationale.** To test which (survey, target, country) cells have estimable, non-leaked PI versus leakage (near-deterministic single-column recovery / skip-pattern modules) or type-1 unestimable splits (thin minority on V1/V2, or low `oracle_ceiling@5`).

**Result.** Default confirmatory grid is `leakage_class == genuine` (`layout.genuine_cells()`). That class includes type-2/3 accuracy-vs-majority cells; it excludes `unestimable` and leakage. Re-run after oracle contract bumps or screen-rule changes; do not mix eras in one audit file without noting it. The retired `degenerate` label was accuracy lift < 0.03 and is no longer emitted.

---

### `oracle-v3` — measurement-level honest split

| Field | Value |
|-------|-------|
| **Status** | superseded (2026-08-19: contract v4 is the only contract; v3 was a dev byproduct — do not cite or build on its caches, see onboarding §3) |
| **Stage** | oracle |
| **Dates** | contract v3 rebuild 2026-08 (`21c780d` lineage) |
| **Code** | [`scripts/rerun_oracles.py`](../scripts/rerun_oracles.py), [`scripts/compute_oracle.py`](../scripts/compute_oracle.py); [`src/survey_features/oracle.py`](../src/survey_features/oracle.py), [`oracle_pool.py`](../src/survey_features/oracle_pool.py). Audit: [`pipeline_audit_2026-08.md`](pipeline_audit_2026-08.md) |
| **Commit** | `21c780d` (v3 rebuild) on `main`; `ORACLE_CONTRACT_VERSION = 3` in `oracle.py` |
| **Compute** | local CPU; AutoGluon; multi-cell via `--processes N` (never threaded AG) |
| **Inputs** | survey microdata + metadata; feature-pool filters in `feature_pool.py` |
| **Outputs** | `outputs/cache/cells/<target>_<country>/{oracle.csv,oracle_meta.json,feature_pool.csv}` |

**Rationale.** To test whether an honest fit/select/score split plus measurement-level-aware metrics (log-loss / Spearman) yields trustworthy oracle rankings for LLM evaluation.

**Result.** Era-3 is current; eras 1–2 are archived out of tree (`features_project_snapshots/`). Any change that alters oracle *meaning* must bump `ORACLE_CONTRACT_VERSION` and add a row to onboarding §3.

---

### `textbook-baseline` — frozen demographics null

| Field | Value |
|-------|-------|
| **Status** | complete |
| **Stage** | scoring |
| **Dates** | introduced with shared scoring (`6c111e8` lineage) |
| **Code** | [`scripts/build_textbook_baseline.py`](../scripts/build_textbook_baseline.py); constructs in [`config.TEXTBOOK_CONSTRUCTS`](../src/survey_features/config.py) |
| **Commit** | record SHA when baselines are rebuilt; same mapping stack as model requests |
| **Compute** | LLM API for construct → code disambig; local disk cache thereafter |
| **Inputs** | survey variable lists + embeddings; optional `textbook_overrides.json` |
| **Outputs** | `outputs/cache/baselines/textbook__<survey>.json` |

**Rationale.** To test model picks against a fixed “competent researcher without reading the question” demographic set, so value-over-textbook is harder than value-over-random.

**Result.** Textbook is the headline contrast in scores. Re-resolve only deliberately; overrides are for outright construct errors, not routine tuning.

---

### `embedding-sensitivity` — sentence-transformer swap

| Field | Value |
|-------|-------|
| **Status** | complete |
| **Stage** | retrieval |
| **Dates** | 2026 (see experiment tree / memos on `main`) |
| **Code** | on `main`: `scripts/run_main.py --embedding-model …`, `scripts/run_embedding_sensitivity_parallel.ps1`, `analysis/embedding_sensitivity.py`; design `docs/embedding_sensitivity.md` |
| **Commit** | record the SHA of the sensitivity sweep; organizational commit `7b00be5` (outputs layout) |
| **Compute** | LLM disambig (fixed); local sentence-transformers for embedders under test |
| **Inputs** | reused `outputs/selectors/<selector>/{freetext,extracted}/`; MiniLM main maps as baseline |
| **Outputs** | `outputs/experiments/embedding_sensitivity/<model_slug>/<selector>/`; comparison digests under that tree |

**Rationale.** To test whether swapping the retrieval embedder (holding selector, extractor, and disambiguator fixed) changes mapped codes and downstream scores.

**Result.** Maps move more than scores under the embedders tested; MiniLM remains the main default. Not a reason to change the confirmatory stack without a new registered run.

**Contrast detail.** not reconstructed (pre-convention). Re-run analyzer against embedding_sensitivity digests if this claim is reopened for a stack change.

---

### `underscore-label-embed` — feature-label tokenization

| Field | Value |
|-------|-------|
| **Status** | complete (ad-hoc probe; no durable artifacts) |
| **Stage** | retrieval |
| **Dates** | 2026-08-04 |
| **Code** | one-off local script (not committed); embedder via `sentence_transformers` / same model as [`retrieval.py`](../src/survey_features/retrieval.py) default |
| **Commit** | — (no code or `outputs/` write; logged here for the decision trail) |
| **Compute** | local: `all-MiniLM-L6-v2` only; no LLM |
| **Inputs** | first 10 unprompted features from `outputs/selectors/kimi/extracted/afrobarometer__Q15__Angola.json` (labels + contexts) |
| **Outputs** | none under `outputs/` — cosine table lived in the session only |

**Rationale.** To test whether extractor labels that use underscores (`left_right_economic_ideology`) vs whitespace (`left right economic ideology`) corrupt MiniLM query embeddings enough to threaten dual-embed retrieval.

**Result.** Label-only underscore↔space self-cosine mean ≈ 0.92 (min ≈ 0.86); with dual-embed `label: context` mean ≈ 0.96. Cross-feature underscore baseline ≈ 0.28. Not treated as a first-order failure mode; no pipeline change.

---

### `ensemble-mapping` — MiniLM ∪ mpnet union

| Field | Value |
|-------|-------|
| **Status** | complete (kimi v1); **not promoted** to default |
| **Stage** | retrieval |
| **Dates** | v1 results documented `42f28e5` (2026) |
| **Code** | on `main`: `scripts/run_ensemble_mapping.py`, `analysis/ensemble_mapping.py`; design `docs/ensemble_mapping.md`; PR `#9` / `d18751c` |
| **Commit** | `1ac8ba2` (add), `42f28e5` (results), merge `d18751c` |
| **Compute** | LLM: one Nemotron disambig per feature; local: dual embedders + fuse |
| **Inputs** | kimi gen/extract; single-embedder baselines from `main/` and `embedding_sensitivity/` |
| **Outputs** | `outputs/experiments/ensemble_mapping/` (maps, scores, comparison + latency CSVs) |

**Rationale.** To test whether unioning candidate pools from two embedders before one disambiguation call improves mapping fidelity and value-over-random versus a single embedder.

**Result.** Jaccard lift and a modest VoR gain at small latency cost; **not** promoted to the confirmatory default (see pipeline audit / ensemble memo verdict).

**Contrast detail.** not reconstructed (pre-convention).

---

### `subitem-mapping` — dual-layer pilot

| Field | Value |
|-------|-------|
| **Status** | complete (kimi v1 pilot) → **promoted** into confirmatory Arm C |
| **Stage** | mapping |
| **Dates** | v1 lock `f7a7fb0`; promotion `949a8cf` (2026-08-06) |
| **Code** | pilot on `main`: `scripts/run_subitem_mapping.py`, `analysis/subitem_mapping.py`; production path: [`mapping.map_features_with_subitems`](../src/survey_features/mapping.py) via [`run_main.py`](../scripts/run_main.py) |
| **Commit** | pilot `f7a7fb0`; promotion `949a8cf` |
| **Compute** | LLM disambig per parent and per bundled sub_item (≥2); local score |
| **Inputs** | shared gen/extract under `outputs/selectors/<selector>/` |
| **Outputs** | pilot: `outputs/experiments/subitem_mapping/`; production maps: `outputs/selectors/<selector>/maps/` with `expanded_codes` |

**Rationale.** To test whether mapping each bundled `sub_item` as its own retrieve+disambiguate unit (dual-layer) improves captured importance / predictive score versus parent-only mapping.

**Result.** Dual-layer locked for confirmatory main (`expanded_codes` is the headline). Parent-only remains an ablation on the full research branch, not on `rewrite/minimal-core`.

**Contrast detail.** not reconstructed (pre-convention); promotion decision predates this reporting standard.

---

### `extract-type-pilot` — type-taxonomy wording

| Field | Value |
|-------|-------|
| **Status** | complete (pilot; not a named confirmatory experiment) |
| **Stage** | extraction |
| **Dates** | 2026-08 (WIP around dual-layer promotion) |
| **Code** | was `scripts/pilot_extract_types.py` (untracked / cut on minimal-core); extraction prompt in [`prompts.py`](../src/survey_features/prompts.py) |
| **Commit** | re-extract pilot did not land as a tagged release; taxonomy rename landed with `949a8cf` / prompts edits |
| **Compute** | LLM: fixed extractor only |
| **Inputs** | sample of cached free-text essays under `outputs/selectors/` |
| **Outputs** | `outputs/experiments/extract_type_pilot/`, `extract_type_pilot_v2/` |

**Rationale.** To test whether clearer extract-type taxonomy wording (including `population_statistic` rename) changes typed feature lists without touching the selector.

**Result.** Prompt/taxonomy clarifications kept for main extraction; pilot trees are exploratory only — do not quote as confirmatory evidence without a registered follow-up.

---

## Design only

### `similarity-threshold` — pool cutoff

| Field | Value |
|-------|-------|
| **Status** | design |
| **Stage** | retrieval |
| **Dates** | design on `main` (`docs/similarity_threshold.md`); not run |
| **Code** | design note on `main` only |
| **Commit** | — |
| **Compute** | — |
| **Inputs** | — |
| **Outputs** | intended `outputs/experiments/similarity_threshold/` |

**Rationale.** To test whether raising/lowering the retrieval similarity cutoff changes none-rate and downstream scores independently of the embedder identity.

**Result.** —

---

### `confirmatory-zoo` — multi-selector lock

| Field | Value |
|-------|-------|
| **Status** | design |
| **Stage** | end-to-end |
| **Dates** | design on `main` (`docs/main_experiment_design.md`) |
| **Code** | [`scripts/run_main.py`](../scripts/run_main.py) + `config.SELECTORS` |
| **Commit** | — (record when the zoo is actually swept) |
| **Compute** | LLM API per selector; local score |
| **Inputs** | contract-v4 oracles for the sampled grid + typed leakage screen + textbook |
| **Outputs** | `outputs/selectors/scores_<selector>.csv` (canonical or `--run-tag`) |

**Rationale.** To test the confirmatory free-text + dual-layer stack across a locked set of selector models under identical extractor, disambiguator, and scoring contracts.

**Result.** —

---

### `pooled-country-value` — is country worth requesting?

| Field | Value |
|-------|-------|
| **Status** | design |
| **Stage** | scoring |
| **Dates** | designed 2026-08-19; runs after the confirmatory sweep |
| **Code** | intended `scripts/run_pooled_country_value.py`; reuses [`evaluate_feature_set`](../src/survey_features/evaluation.py) unchanged (pooled rows + a whitelist flag for the country column in the pool builder) |
| **Commit** | — (record at first artifact write) |
| **Compute** | local CPU only (XGB on pooled rows); no LLM calls |
| **Inputs** | the 30 transportability targets (grid memo Q2) with their 6-country microdata pooled **within survey**; balanced v4 oracles for those targets; unprompted-arm mapped picks from the confirmatory sweep; heterogeneity bins + within-country reliability from the het screen |
| **Outputs** | `outputs/experiments/pooled_country_value/pooled_scores.csv` (one row per target × feature-set × ±country), digest in `experiments/_analysis/` |

**Rationale.** Inside a (target, country) cell, country has zero variance — it can
never be a feature — so the pipeline cannot say whether a selector *should* have
asked for the respondent's country. Yet selectors do ask: in the unprompted
condition they request a country/nationality feature in 24–50% of cells (vs 4–12%
when the country is named), and on era-3 data that request rate is uncorrelated
with *structure* heterogeneity (r ≈ −0.02) — which looked like indiscriminate
hedging until we noticed the ground truth was wrong: requesting country is rational
wherever country shifts the *level* of the target, even when the predictive
structure is identical everywhere. This experiment measures the correct ground
truth — the pooled predictive value of the country column, levels and structure
combined — and asks whether selectors request country where it actually pays.

**Design.** For each of the 30 transportability targets, stack its 6 countries'
rows (same questionnaire — within-survey pooling only, per the grid memo's scope
note). Score each feature set twice with the standard typed evaluator — once with
the country column whitelisted in, once out; Δ(country) = the pooled value of
knowing the country for that set. Sets: oracle top-k (Δ_oracle = ground truth
"was country worth requesting"), the selector's unprompted picks (one set per
target: the modal mapped set across its cells — registered collapse rule;
unprompted picks are country-blind by construction, so per-cell copies differ only
by extraction jitter), textbook, and matched-k random.

**Predictions (registered before looking).** (1) Δ_oracle > 0 for most attitude
targets (country level-shifts are near-universal) — if not, country-requesting is
over-hedging even in level terms. (2) A knowledgeable selector's country-request
rate (unprompted arm) rises with Δ_oracle — this replaces the structure-het x-axis
that wrongly suggested hedging. (3) Crossing Δ_oracle with structure-het separates
regimes: high-Δ/low-het = "levels differ, mechanism shared" (the same-pattern-
different-drivers scenario made measurable); high-het targets should additionally
show pick differentiation and swap gains (grid memo, transportability primary).
**Guard:** pooled permutation importance of country lumps level shifts with
structural change — never read Δ(country) alone as "structure varies"; that claim
belongs to the het measure and the swap contrast.

**Result.** —

---

## Superseded

### `prelim-json-grid` — strict-JSON appendix

| Field | Value |
|-------|-------|
| **Status** | superseded |
| **Stage** | end-to-end |
| **Dates** | 2025–early 2026 prelim |
| **Code** | on full tree / archive: `archive/run_grid.py` (not on `rewrite/minimal-core`) |
| **Commit** | various; treat as historical. Some prelim numbers are unreproducible (see old reconciliation notes) |
| **Compute** | LLM API + local oracle/eval of that era |
| **Inputs** | prelim manifests (removed from minimal-core; restore from `main` / snapshots) |
| **Outputs** | `outputs/grid/`; era-1/2 cell zips in `features_project_snapshots/` |

**Rationale.** To test LLM feature selection under a strict-JSON elicitation contract with shortlist mapping (pilot instrument).

**Result.** Free text superseded JSON as the instrument; JSON-era magnitudes are a conservative floor. Do not quote JSON-era Test-2 / VoR figures as current confirmatory results.

---

## Rules (short)

1. **No silent experiments.** Register before the first write when CONTRIBUTING
   requires it (`experiments/<name>/`, a `--run-tag` you will cite, or a run that
   can change confirmatory `cache/` / canonical `main/` numbers).
2. **Commit identity.** The SHA in the entry must match the code that wrote the artifacts.
3. **Result is mandatory at completion.** Status may not sit at `complete` with Result `—`.
4. **Supersession is explicit.** Move claims into **Result** / status `superseded`; do not leave stale “current” language in old entries.
5. **One Stage tag.** Tag the stage you manipulated; if an experiment truly spans two, pick the primary locus and mention the other in Rationale.
