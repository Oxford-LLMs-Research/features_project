# Survey Features — LLM feature-selection capability study

**Research question:** Do LLMs understand the conditional structure of human attitudes — can they identify, from prior knowledge alone, which respondent characteristics predict how someone answers a given survey question? Does this reasoning adapt across countries?

The pipeline asks an LLM (in free text) which respondent features would predict a target survey answer, extracts a typed feature list from the essay, maps each feature to a concrete survey variable via embedding retrieval + LLM disambiguation, and evaluates whether the selected variables actually predict the answer better than matched-size random baselines — against an AutoGluon permutation-importance oracle as the ceiling.

---

## Repository layout

```
pyproject.toml                — package metadata; install with `pip install -e .`
src/survey_features/          — the shared library (all pipeline logic lives here)
    config.py                 —   paths, .env loading, model registry (selectors/extractor/disambiguators)
    llm.py                    —   ONE OpenAI-compatible client: retries, backoff, token-usage log
    surveys.py                —   survey loading, country maps, metadata handling (single copy)
    prompts.py                —   ALL prompt templates (current free-text + legacy JSON + extract/disambig)
    elicitation.py            —   selection-prompt calls (free-text current; JSON legacy)
    extraction.py             —   free-text essay -> typed feature list (fixed extractor model)
    retrieval.py              —   sentence-transformer embedding cache + dual-embed candidate retrieval
    disambig.py               —   feature -> survey-code disambiguation (per-feature current; shortlist legacy)
    oracle.py                 —   AutoGluon permutation-importance oracle (needs the [oracle] extra)
    evaluation.py             —   matched-k XGBoost CV: oracle vs model vs random
    metrics.py                —   captured importance, jaccard, oracle percentile, bootstrap CIs
    layout.py                 —   outputs/ path contracts (grid summaries, LLM caches, pilot dirs)
scripts/
    run_main.py               — CANONICAL entry point: phased free-text pipeline (gen/extract/map/score)
    run_grid.py               — legacy JSON-prompt grid (appendix reproducibility; thin wrapper)
    leakage_audit.py          — empirical leakage audit of the oracle ground truth
analysis/                     — one-off analysis + paper-build scripts (import the package)
prelim/                       — manifest build, target selection, metadata introspection
docs/                         — findings and design notes (markdown)
archive/                      — dead scripts kept for reference (do not run)
data/                         — per-survey metadata JSONs
outputs/                      — cached artifacts (gitignored)
```

---

## Setup

### 1. Install the package

```bash
pip install -e .                # core pipeline + analysis
pip install -e ".[oracle]"      # + autogluon, needed only to (re)compute oracles
pip install -e ".[analysis]"    # + tabulate (some report scripts)
```

Requires Python ≥ 3.9. Survey data access goes through [`synthetic_sampling`](https://github.com/Oxford-LLMs-Research/synthetic_sampling) (installed automatically at a pinned commit). Follow that repo's setup instructions to point `configs/local.yaml` at your local data files.

On networks that block PyPI (some institutional proxies intercept `files.pythonhosted.org`), install from a mirror or use a pre-provisioned environment such as a conda base that already carries the scientific stack.

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, DATA_CONFIG_PATH
```

`LLM_BASE_URL` accepts any OpenAI-compatible endpoint: Nebius, Together.ai, OpenRouter, local SGLang, etc.

Two model roles are held **fixed** across all runs so that differences between test models are attributable only to feature-selection quality:

- **Extractor** (`Qwen/Qwen3-235B-A22B-Instruct-2507`) — turns free-text essays into typed feature lists.
- **Disambiguator** (`DISAMBIG_MODEL` in `.env`, default `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) — picks the survey variable matching each feature. Can point at a different endpoint/key via `DISAMBIG_BASE_URL` / `DISAMBIG_API_KEY`.

Test models ("selectors") are registered in `src/survey_features/config.py::SELECTORS` — add an entry there to evaluate a new model.

---

## Running the main (free-text) pipeline

`scripts/run_main.py` is the canonical entry point. It runs over the *genuine* cells from the leakage audit (`outputs/leakage_audit.csv`), both prompt conditions (`unprompted`, `country_provided`), in four resumable phases:

```bash
python scripts/run_main.py --phase gen     --selector deepseek   # free-text selection essays
python scripts/run_main.py --phase extract --selector deepseek   # essay -> typed feature list (fixed extractor)
python scripts/run_main.py --phase map     --selector deepseek --disambiguator nemotron
python scripts/run_main.py --phase score   --selector deepseek   # -> outputs/format_pilot/scores_deepseek.csv
```

- Every phase checkpoints per cell; rerunning skips cells already on disk (`--force` recomputes, `--limit N` smoke-tests on the first N cells).
- `--phase map` arms: `C` = free-text (extracted) features, `B` = the model's legacy JSON selections re-mapped through the same retrieval+disambiguation (for the format comparison). Default `--arms B,C`.
- `--phase score` computes captured importance and oracle/model/random XGBoost accuracy at model-chosen k and fixed k=5,10, for every arm × disambiguator.

Prerequisites on disk: per-cell `outputs/<target>_<country>/oracle.csv` (from the legacy grid or `survey_features.oracle`) and `outputs/leakage_audit.csv` (from `python scripts/leakage_audit.py`).

### Embedding-model sensitivity

To test whether map → disambiguation → score results move with the sentence-transformer, pass `--embedding-model`. Gen/extract stay in `outputs/format_pilot/`; only map and score are re-run into an isolated tree (existing MiniLM `format_pilot` artifacts are the baseline and are never overwritten):

| Role | Model | Approx. size |
|------|--------|--------------|
| Baseline (already in `format_pilot/`) | `all-MiniLM-L6-v2` | ~22M |
| Mid | `all-mpnet-base-v2` | ~110M |
| Large | `all-roberta-large-v1` | ~355M |

```bash
# Mid — both selectors, arm C + nemotron only
python scripts/run_main.py --phase map   --selector deepseek --disambiguator nemotron --arms C --embedding-model all-mpnet-base-v2
python scripts/run_main.py --phase score --selector deepseek --embedding-model all-mpnet-base-v2
python scripts/run_main.py --phase map   --selector kimi     --disambiguator nemotron --arms C --embedding-model all-mpnet-base-v2
python scripts/run_main.py --phase score --selector kimi     --embedding-model all-mpnet-base-v2

# Large
python scripts/run_main.py --phase map   --selector deepseek --disambiguator nemotron --arms C --embedding-model all-roberta-large-v1
python scripts/run_main.py --phase score --selector deepseek --embedding-model all-roberta-large-v1
python scripts/run_main.py --phase map   --selector kimi     --disambiguator nemotron --arms C --embedding-model all-roberta-large-v1
python scripts/run_main.py --phase score --selector kimi     --embedding-model all-roberta-large-v1

python analysis/embedding_sensitivity.py   # -> outputs/embedding_sensitivity/comparison.csv
```

Artifacts: `outputs/embedding_sensitivity/<model_slug>/<selector>/maps/` and `scores_<selector>.csv`, plus `manifest.json`. See `docs/embedding_sensitivity.md`.

### Pipeline steps (concept)

```
[1] Oracle       — AutoGluon permutation importance -> ground-truth feature ranking
[2] Elicit       — free-text essay: which respondent features predict the answer?
[3] Extract      — fixed extractor model -> typed feature list
[4] Map          — embed each feature, retrieve top-20 candidate variables,
                   fixed disambiguator picks the best (or none)
[5] Score        — XGBoost CV: oracle top-k vs model-selected-k vs random-k
```

Key metrics per cell:

- `captured_importance` — share of oracle top-k importance mass the model's variables capture
- `value_over_random` = model_acc − random_acc
- `cost_of_imperfect` = oracle_acc − model_acc

### Analysis / paper artifacts

```bash
python analysis/freetext_main_results.py   # headline T1/T2 numbers + tex tables
python analysis/freetext_figures.py        # free-text figures
python analysis/embedding_sensitivity.py   # MiniLM vs mid/large embedders (after sensitivity runs)
python scripts/leakage_audit.py            # oracle leakage audit -> outputs/leakage_audit.csv
```

Findings and design notes live in `docs/` (`main_experiment_design.md`, `format_findings.md`, `embedding_sensitivity.md`, `leakage_findings.md`, …).

---

## Oracle — extension point

The oracle is decoupled from the LLM pipeline via a cache contract. To plug in a pre-computed or alternative oracle, place a CSV at `outputs/<target>_<country>/oracle.csv` with columns:

| column | description |
|--------|-------------|
| `target_variable` | variable code (e.g. `Q164`) |
| `country` | country code as stored in the survey data |
| `feature_variable` | predictor variable code |
| `importance_mean` | mean permutation importance |
| `importance_std` | standard deviation |
| `majority_baseline` | majority-class accuracy |

The file must contain **all** features considered — evaluation picks its own top-k from the full ranking. If the file exists, the oracle step is skipped entirely.

Standalone oracle run (requires `pip install -e ".[oracle]"`):

```bash
python -m survey_features.oracle \
    --survey wvs --targets Q47 Q57 Q199 Q235 Q164 \
    --countries Germany Nigeria Japan Brazil Egypt \
    --runtime-mode balanced --force
```

Key flags: `--runtime-mode` (`quick`/`balanced`/`best`), `--similarity-threshold` (0.85), `--test-size` (0.2), `--max-missingness-threshold` (0.2), `--force`.

### Supported surveys

| `--survey` value  | Country column | Notes |
|-------------------|----------------|-------|
| `wvs`             | `B_COUNTRY`    | |
| `afrobarometer`   | `COUNTRY`      | |
| `arabbarometer`   | `COUNTRY`      | |
| `asianbarometer`  | `country`      | stores country names directly |
| `latinobarometer` | `IDENPA`       | |
| `ess_wave_10`     | `cntry`        | alpha-2 codes |
| `ess_wave_11`     | `cntry`        | |

---

## Appendix: legacy JSON-prompt grid (`scripts/run_grid.py`)

The first pilot elicited selections as **JSON lists** rather than free text. Pilot 2 established that JSON suppresses selection breadth, so the paper's headline results use the free-text pipeline above — but the JSON grid is kept runnable for appendix reproducibility, and its cached artifacts feed arms A/B of the format comparison.

```bash
# Single cells
python scripts/run_grid.py --targets Q164 --countries Germany
python scripts/run_grid.py --survey afrobarometer --targets Q4A --countries Nigeria Kenya

# Manifest-based prelim grid (5 targets per survey)
python prelim/build_prelim_manifest.py
python scripts/run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml --stop-after oracle
python scripts/run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml

# Discover available countries
python scripts/run_grid.py --survey afrobarometer --list-countries

# Full all-surveys prelim run (PowerShell; logs to outputs/logs/)
.\run_prelim_full_all_surveys.ps1
```

The preliminary results on disk cover **5 targets × 3 countries × 2 prompt conditions** per survey, for **two LLMs** (`deepseek-ai/DeepSeek-V3.2`, `moonshotai/Kimi-K2.5`) run under separate `--run-tag`s. Analysis scripts read every `grid_summary__<survey>__<tag>.csv` and report models side-by-side.

Every invocation writes a run manifest at `outputs/run_manifest__<survey>__<exp_id>.json` recording models, prompt variant, embedding model, grid and completion counts. The experiment ID (`--run-tag`) determines all output paths for a run:

```bash
python scripts/run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml --run-tag deepseek-default
python scripts/run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml --run-tag kimi-default
# Sensitivity runs: --prompt-variant explicit / --embedding-model all-mpnet-base-v2
```

**Resuming:** re-running with the same `--run-tag` skips cells whose `disambig.json` and `eval.json` exist; `oracle.csv` is always reused. Delete a specific `llm__<tag>/` folder to redo only the LLM+eval steps for that model. Steps run per-cell in parallel threads (`--grid-workers`, default 5); XGBoost threads per cell are capped via `cpu_count // grid_workers` or `GRID_XGB_NTHREAD`.

### Output layout

```
outputs/
  <target>_<country>/
    oracle.csv                          # permutation importances — shared across all models
    llm__<exp_id>/
      disambig.json                     # legacy-grid LLM selection + mapping results
      eval.json                         # XGBoost comparison results
  grid_summary__<survey>__<exp_id>.csv  # one row per (target, country, condition)
  grid_results__<survey>__<exp_id>.json # full nested eval payload
  llm_usage__<survey>__<exp_id>.jsonl   # per-request token usage
  run_manifest__<survey>__<exp_id>.json # experiment provenance
  survey_embeddings__<survey>__<embedding_model>.npz  # embedding cache
  leakage_audit.csv                     # genuine/leaky cell classification
  format_pilot/                         # free-text pipeline artifacts (run_main.py)
    <selector>/gen|extract|maps/        # per-cell checkpoints per phase
    scores_<selector>.csv               # per-cell scores, all arms x disambiguators
  embedding_sensitivity/                # --embedding-model map/score runs (isolated)
    manifest.json
    <embed_slug>/<selector>/maps|scores_*.csv
    comparison.csv                      # analysis/embedding_sensitivity.py
  logs/
```

---

## Windows / libomp note

`sentence-transformers` (PyTorch) and `xgboost` both ship `libomp.dll`, which can conflict. If a run crashes or hangs during embedding or evaluation, keep XGBoost single-threaded (the scoring phase already does) or run embedding and evaluation in separate processes.

---

## Contributing / license

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under the [MIT License](LICENSE).
