# Experiment registry

One place to find every experiment: **why it was run, which code produced it, where the
artifacts live, and what the result was**. Add an entry **before** the first write to
`outputs/`; fill **Result** when the run finishes (or mark `status: abandoned`).

Past runs on other branches / snapshots still belong here — the registry is the map,
not the code tree.

---

## How to register

1. Copy the **Entry template** below into a new section (newest experiments at the top of Active / Complete).
2. Set **Stage** to exactly one pipeline tag from the vocabulary below (primary locus of the manipulation).
3. Fill **Rationale** and **Result** in ≤3 short sentences each (“to test if X improves Y” / “X does not improve Y; caveat …”).
4. Record the **git commit** (or tag) that *produced* the artifacts — not “whatever HEAD was when you wrote the memo”.
5. Link **inputs** and **outputs** as paths under `outputs/` (or snapshot restore commands). Prefer concrete files over directory-only pointers when a headline file exists.
6. Add a one-line row to the **Index** table (keep the index sorted by Stage, then status).

**Status vocabulary:** `design` → `running` → `complete` | `abandoned` | `superseded`.

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
```

---

## Index

Sorted by **Stage**, then status.

| Stage | Slug | Status | One-line claim | Outputs |
|-------|------|--------|----------------|---------|
| end-to-end | [`main-freetext`](#main-freetext--confirmatory-free-text-arm-c) | complete | Free-text + dual-layer map is the confirmatory instrument | `outputs/main/` |
| end-to-end | [`confirmatory-zoo`](#confirmatory-zoo--multi-selector-lock) | design | Not run | — |
| end-to-end | [`prelim-json-grid`](#prelim-json-grid--strict-json-appendix) | superseded | JSON-era magnitudes are a floor, not the estimate | snapshots / `outputs/grid/` |
| extraction | [`extract-type-pilot`](#extract-type-pilot--type-taxonomy-wording) | complete (pilot) | Type-prompt wording pilot | `outputs/experiments/extract_type_pilot*` |
| grid | [`leakage-audit`](#leakage-audit--genuine-cell-screen) | complete | Drop leakage / degenerate cells from the grid | `outputs/cache/audits/leakage_audit.csv` |
| mapping | [`subitem-mapping`](#subitem-mapping--dual-layer-pilot) | complete → promoted | Dual-layer locked into main | `outputs/experiments/subitem_mapping/` |
| oracle | [`oracle-v3`](#oracle-v3--measurement-level-honest-split) | complete | Era-3 oracle is the current ground truth | `outputs/cache/cells/` |
| retrieval | [`embedding-sensitivity`](#embedding-sensitivity--sentence-transformer-swap) | complete | Embedder swap moves maps more than scores | `outputs/experiments/embedding_sensitivity/` |
| retrieval | [`ensemble-mapping`](#ensemble-mapping--minilm--mpnet-union) | complete (not promoted) | Union retrieval lifts Jaccard; small VoR gain | `outputs/experiments/ensemble_mapping/` |
| retrieval | [`underscore-label-embed`](#underscore-label-embed--feature-label-tokenization) | complete (ad-hoc) | `_` vs spaces barely moves MiniLM vectors | chat-only |
| retrieval | [`similarity-threshold`](#similarity-threshold--pool-cutoff) | design | Not run | — |
| scoring | [`textbook-baseline`](#textbook-baseline--frozen-demographics-null) | complete | Textbook demographics are the hard null for VoT | `outputs/cache/baselines/` |

---

## Active / complete

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
| **Outputs** | `outputs/main/<selector>/{freetext,extracted,maps}/`; `outputs/main/scores_<selector>.csv` |

**Rationale.** To test whether an LLM, prompted in free text and mapped dual-layer onto survey variables, captures oracle importance and beats matched-k random and textbook demographic baselines across countries.

**Result.** Free text is the confirmatory instrument (JSON was a suppressed floor). Dual-layer mapping (parent + bundled sub_items → `expanded_codes`) is the headline map path. Quote only era-3-scored numbers; see onboarding §3.

---

### `leakage-audit` — genuine-cell screen

| Field | Value |
|-------|-------|
| **Status** | complete |
| **Stage** | grid |
| **Dates** | 2026 (re-run after any oracle contract change) |
| **Code** | [`scripts/leakage_audit.py`](../scripts/leakage_audit.py); target catalog [`data/targets.yaml`](../data/targets.yaml) |
| **Commit** | record the SHA used for each audit refresh; logic landed with oracle rebuild era (`21c780d` lineage on `main`) |
| **Compute** | local CPU (+ optional `--with-data` single-feature XGB); no LLM |
| **Inputs** | `outputs/cache/cells/*/oracle.csv`; survey microdata via `DATA_CONFIG_PATH` when `--with-data` |
| **Outputs** | `outputs/cache/audits/leakage_audit.csv`; `leakage_audit_summary.json` |

**Rationale.** To test which (survey, target, country) cells have real distributed predictive structure versus leakage (near-deterministic single-column recovery) or degeneracy (oracle cannot beat the marginal).

**Result.** Only `leakage_class == genuine` cells enter the confirmatory grid (`layout.genuine_cells()`). Re-run after oracle contract bumps; do not mix eras in one audit file without noting it.

---

### `oracle-v3` — measurement-level honest split

| Field | Value |
|-------|-------|
| **Status** | complete (current ground truth) |
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
| **Inputs** | reused `outputs/main/<selector>/{freetext,extracted}/`; MiniLM main maps as baseline |
| **Outputs** | `outputs/experiments/embedding_sensitivity/<model_slug>/<selector>/`; comparison digests under that tree |

**Rationale.** To test whether swapping the retrieval embedder (holding selector, extractor, and disambiguator fixed) changes mapped codes and downstream scores.

**Result.** Maps move more than scores under the embedders tested; MiniLM remains the main default. Not a reason to change the confirmatory stack without a new registered run.

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
| **Inputs** | first 10 unprompted features from `outputs/main/kimi/extracted/afrobarometer__Q15__Angola.json` (labels + contexts) |
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
| **Inputs** | shared gen/extract under `outputs/main/<selector>/` |
| **Outputs** | pilot: `outputs/experiments/subitem_mapping/`; production maps: `outputs/main/<selector>/maps/` with `expanded_codes` |

**Rationale.** To test whether mapping each bundled `sub_item` as its own retrieve+disambiguate unit (dual-layer) improves captured importance / predictive score versus parent-only mapping.

**Result.** Dual-layer locked for confirmatory main (`expanded_codes` is the headline). Parent-only remains an ablation on the full research branch, not on `rewrite/minimal-core`.

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
| **Inputs** | sample of cached free-text essays under `outputs/main/` |
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
| **Inputs** | era-3 oracles + genuine grid + textbook |
| **Outputs** | `outputs/main/scores_<selector>.csv` (canonical or `--run-tag`) |

**Rationale.** To test the confirmatory free-text + dual-layer stack across a locked set of selector models under identical extractor, disambiguator, and scoring contracts.

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

1. **No silent experiments.** If it writes under `outputs/experiments/<name>/` or changes confirmatory numbers, it has a registry entry.
2. **Commit identity.** The SHA in the entry must match the code that wrote the artifacts.
3. **Result is mandatory at completion.** Status may not sit at `complete` with Result `—`.
4. **Supersession is explicit.** Move claims into **Result** / status `superseded`; do not leave stale “current” language in old entries.
5. **One Stage tag.** Tag the stage you manipulated; if an experiment truly spans two, pick the primary locus and mention the other in Rationale.
