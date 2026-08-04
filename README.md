# Survey Features — LLM feature-selection capability study

**Research question:** Do LLMs understand the conditional structure of human attitudes — can they identify, from prior knowledge alone, which respondent characteristics predict how someone answers a given survey question? Does this reasoning adapt across countries?

The pipeline asks an LLM (in free text) which respondent features would predict a target survey answer, extracts a typed feature list from the essay, maps each feature to a concrete survey variable via embedding retrieval + LLM disambiguation, and evaluates whether the selected variables actually predict the answer better than matched-size random baselines — against an AutoGluon permutation-importance oracle as the ceiling.

> **New to the repo? Start with [`docs/onboarding.md`](docs/onboarding.md)** — the concepts
> (oracle contracts, the honest split, cache identity), which numbers are current, and the
> sharp edges. This README covers setup and layout; that file covers how the system thinks.

---

## Three zones (read this first)

| Zone | Role | In git? |
|------|------|---------|
| **Pipeline** | `src/`, `scripts/`, `tests/`, `analysis/` digests, slim `docs/`, `prelim/`, `data/` | Yes |
| **`outputs/`** | Run artifacts and digests (`SURVEY_FEATURES_OUTPUTS`) | No |
| **`paper/`** | Writing workspace: LaTeX, figures, memos, builders, talk (`SURVEY_FEATURES_PAPER`) | No |

Clone + install runs the pipeline. Paper PDF rebuild needs a local `paper/` tree (not shipped).
Legacy JSON-grid replication lives under [`archive/`](archive/README.md) — runnable, never imported by live code.

---

## Repository layout

```
pyproject.toml                — package metadata; install with `pip install -e .`
src/survey_features/          — the shared library (all pipeline logic lives here)
    config.py                 —   paths (.env), OUTPUTS_DIR / PAPER_DIR, model registry
    llm.py                    —   ONE OpenAI-compatible client: retries, backoff, token-usage log
    surveys.py                —   survey loading, country maps, metadata handling (single copy)
    prompts.py                —   ALL prompt templates (current free-text + legacy JSON + extract/disambig)
    elicitation.py            —   selection-prompt calls (free-text current; JSON legacy)
    extraction.py             —   free-text essay -> typed feature list (fixed extractor model)
    retrieval.py              —   sentence-transformer embedding cache + dual-embed candidate retrieval
    ensemble.py               —   ensemble fusion labels + defaults
    disambig.py               —   feature -> survey-code disambiguation (per-feature current; shortlist legacy)
    oracle.py                 —   AutoGluon permutation-importance oracle (needs the [oracle] extra)
    evaluation.py             —   matched-k XGBoost CV: oracle vs model vs random
    metrics.py                —   captured importance, jaccard, oracle percentile, bootstrap CIs
    layout.py                 —   outputs/ path contracts (cache/main/experiments; dual-resolve)
    timing.py                 —   wall-clock spans + JSONL timing logs for pipeline phases
    score_cell.py             —   cell-level XGB scoring (ProcessPool; code-set cache)
scripts/
    run_main.py               — CANONICAL free-text pipeline (gen/extract/map/score/pipeline)
    leakage_audit.py          — empirical leakage audit -> cache/audits/leakage_audit.csv
    compute_oracle.py         — standalone oracle fit
    rerun_oracles.py          — recompute oracles over an audit grid
    audit_missing_codes.py    — missing-code taxonomy
    audit_target_types.py     — target-type audit table
    build_textbook_baseline.py
    run_ensemble_mapping.py / run_subitem_mapping.py
    run_embedding_sensitivity_parallel.ps1
analysis/                     — live digests + experiment rollups (write under outputs/ only)
tests/                        — pytest suite (pip install -e ".[dev]")
prelim/                       — manifest build, target selection, metadata introspection
docs/                         — onboarding, experiments_index, design/protocol, audit record
archive/                      — JSON-grid / prelim replication + spent one-shots (see archive/README.md)
data/                         — per-survey metadata JSONs
outputs/                      — cached artifacts (gitignored)
paper/                        — writing workspace (gitignored): LaTeX, figures, memos, builders, talk
```

---

## Setup

### 1. Install the package

```bash
pip install -e .                # core pipeline library
pip install -e ".[oracle]"      # + autogluon, needed only to (re)compute oracles
pip install -e ".[analysis]"    # + tabulate (some analysis/ digest scripts)
pip install -e ".[dev]"         # + pytest
```

Requires Python ≥ 3.9. Survey data access goes through [`synthetic_sampling`](https://github.com/Oxford-LLMs-Research/synthetic_sampling) (installed automatically at a pinned commit). Follow that repo's setup instructions to point `configs/local.yaml` at your local data files.

On networks that block PyPI (some institutional proxies intercept `files.pythonhosted.org`), install from a mirror or use a pre-provisioned environment such as a conda base that already carries the scientific stack.

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, DATA_CONFIG_PATH
# Optional: SURVEY_FEATURES_OUTPUTS, SURVEY_FEATURES_PAPER (defaults: <repo>/outputs, <repo>/paper)
```

`LLM_BASE_URL` accepts any OpenAI-compatible endpoint: Nebius, Together.ai, OpenRouter, local SGLang, etc.

Two model roles are held **fixed** across all runs so that differences between test models are attributable only to feature-selection quality:

- **Extractor** (`Qwen/Qwen3-235B-A22B-Instruct-2507`) — turns free-text essays into typed feature lists.
- **Disambiguator** (`DISAMBIG_MODEL` in `.env`, default `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) — picks the survey variable matching each feature. Can point at a different endpoint/key via `DISAMBIG_BASE_URL` / `DISAMBIG_API_KEY`.

Test models ("selectors") are registered in `src/survey_features/config.py::SELECTORS` — add an entry there to evaluate a new model.

---

## Running the main (free-text) pipeline

`scripts/run_main.py` is the canonical entry point. It runs over the *genuine* cells from the leakage audit (`outputs/cache/audits/leakage_audit.csv` or legacy root path), both prompt conditions (`unprompted`, `country_provided`), in resumable phases:

```bash
python scripts/run_main.py --phase gen     --selector deepseek   # free-text selection essays
python scripts/run_main.py --phase extract --selector deepseek   # essay -> typed feature list (fixed extractor)
python scripts/run_main.py --phase map     --selector deepseek --disambiguator nemotron
python scripts/run_main.py --phase score   --selector deepseek   # -> outputs/main/scores_deepseek.csv

# Or overlap gen→extract→map across cells (same checkpoints; score optional):
python scripts/run_main.py --phase pipeline --selector deepseek --disambiguator nemotron \
    --pipeline-workers 4 --map-workers 8 --with-score
```

- Every phase checkpoints per cell; rerunning skips cells already on disk (`--force` recomputes, `--limit N` smoke-tests on the first N cells).
- Use `--run-tag <who_slug>` so map/score write under `main/runs/<tag>/` and do not clobber the shared baseline.
- `--phase map` arms: `C` = free-text (extracted) features, `B` = the model's legacy JSON selections re-mapped through the same retrieval+disambiguation (for the format comparison). Default `--arms B,C`.
- `--phase score` computes captured importance and oracle/model/random XGBoost accuracy at model-chosen k and fixed k=5,10, for every arm × disambiguator.
- `--phase pipeline` runs gen → extract → map per cell with up to `--pipeline-workers` cells in flight (extract of cell N can overlap map of cell N−1). Add `--with-score` to score after maps finish.

**Concurrency** (defaults are serial / conservative — existing scripts stay unchanged):

| Knob | Env | Applies to | Default |
|------|-----|------------|---------|
| `--api-workers` | `API_WORKERS` | `gen`, `extract` | 1 |
| `--map-workers` | `MAP_WORKERS` | `map`, `pipeline` (per-feature disambig ThreadPool) | 1 |
| `--pipeline-workers` | `PIPELINE_WORKERS` | `pipeline` (cells in flight) | 1 |
| `--score-workers` | `SCORE_WORKERS` | `score` (cell ProcessPool) | `min(8, cpus-2)` |

Each phase writes `outputs/logs/timing_<phase>_*.jsonl` and prints a span summary. LLM token logs also record per-call `latency_ms` when usage logging is enabled.

Prerequisites on disk: per-cell oracle under `outputs/cache/cells/<target>_<country>/oracle.csv` (or legacy `outputs/<t>_<c>/`) and `outputs/cache/audits/leakage_audit.csv` (from `python scripts/leakage_audit.py`).

### Embedding-model sensitivity

To test whether map → disambiguation → score results move with the sentence-transformer, pass `--embedding-model`. Gen/extract stay in `outputs/main/`; only map and score are re-run into an isolated tree (existing MiniLM `main/` artifacts are the baseline and are never overwritten):

| Role | Model | Approx. size |
|------|--------|--------------|
| Baseline (already in `main/`) | `all-MiniLM-L6-v2` | ~22M |
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

python analysis/embedding_sensitivity.py   # -> experiments/embedding_sensitivity/comparison.csv
```

Artifacts: `outputs/experiments/embedding_sensitivity/<model_slug>/<selector>/maps/` and `scores_<selector>.csv`, plus `manifest.json`. See `docs/embedding_sensitivity.md`.

### Ensemble retrieval / mapping

After embedding sensitivity (maps diverge, scores stable), fuse candidate pools from multiple embedders (default MiniLM ∪ mpnet), then **one** Nemotron disambiguation per feature. Writes only under `outputs/experiments/ensemble_mapping/`; reuses single-model baselines from `main/` and `embedding_sensitivity/`. **v1 is kimi-only.** Design: `docs/ensemble_mapping.md`.

```bash
python scripts/run_ensemble_mapping.py --phase map --selector kimi --disambiguator nemotron --limit 2
python scripts/run_ensemble_mapping.py --phase map --selector kimi --disambiguator nemotron --arms C
python scripts/run_ensemble_mapping.py --phase score --selector kimi
python analysis/ensemble_mapping.py --selector kimi   # -> comparison.csv + latency_*.csv
```

### Sub-item (subconcept) mapping

Parent features are still one-to-one in the main pipeline; bundled `sub_items` are audit-only there. A separate experiment maps each sub_item as its own unit under `outputs/experiments/subitem_mapping/` (gen/extract reused; `main/` untouched). **v1 is kimi-only** map + score. Design: `docs/subitem_mapping.md`. Similarity-threshold effects are a **different** experiment (`docs/similarity_threshold.md`).

```bash
python scripts/run_subitem_mapping.py --phase map --selector kimi --disambiguator nemotron --limit 2
# optional: --map-workers 8  (same MAP_WORKERS env as run_main)
python analysis/subitem_mapping.py --selector kimi
# full kimi map, then score (natural-k + matched k=5/10) — see docs/subitem_mapping.md
```

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

### Analysis digests / paper artifacts

```bash
python analysis/freetext_main_results.py          # headline T1/T2 digest → outputs/main/
python paper/scripts/write_freetext_tex.py        # TeX tables from that digest (local)
python paper/scripts/freetext_figures.py          # free-text figures → paper/figures/
python analysis/embedding_sensitivity.py          # MiniLM vs mid/large embedders (after sensitivity runs)
python analysis/ensemble_mapping.py               # ensemble vs single-embedder baselines + latency
python scripts/leakage_audit.py                   # oracle leakage audit -> cache/audits/leakage_audit.csv
```

Design/protocol live in `docs/` — start from [`docs/experiments_index.md`](docs/experiments_index.md).
Findings memos and LaTeX live under local `paper/` (gitignored).

---

## Oracle — extension point

The oracle is decoupled from the LLM pipeline via a cache contract. To plug in a pre-computed or alternative oracle, place a CSV at `outputs/cache/cells/<target>_<country>/oracle.csv` (legacy flat `outputs/<target>_<country>/oracle.csv` is still dual-resolved) with columns:

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
python scripts/compute_oracle.py \
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

## Appendix: legacy JSON-prompt grid (`archive/run_grid.py`)

The first pilot elicited selections as **JSON lists** rather than free text. Pilot 2 established that JSON suppresses selection breadth, so the paper's headline results use the free-text pipeline above — but the JSON grid stays under `archive/` for appendix reproducibility, and its cached artifacts feed arms A/B of the format comparison.

```bash
# Single cells
python archive/run_grid.py --targets Q164 --countries Germany
python archive/run_grid.py --survey afrobarometer --targets Q4A --countries Nigeria Kenya

# Manifest-based prelim grid (5 targets per survey)
python prelim/build_prelim_manifest.py
python archive/run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml --stop-after oracle
python archive/run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml

# Discover available countries
python archive/run_grid.py --survey afrobarometer --list-countries
```

The preliminary results on disk cover **5 targets × 3 countries × 2 prompt conditions** per survey, for **two LLMs** (`deepseek-ai/DeepSeek-V3.2`, `moonshotai/Kimi-K2.5`) run under separate `--run-tag`s. Archived analysis digests (`archive/prelim_*.py`, `archive/alignment_analysis.py`, …) read every `grid_summary__<survey>__<tag>.csv` and report models side-by-side.

Every invocation writes a run manifest at `outputs/run_manifest__<survey>__<exp_id>.json` recording models, prompt variant, embedding model, grid and completion counts. The experiment ID (`--run-tag`) determines all output paths for a run:

```bash
python archive/run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml --run-tag deepseek-default
python archive/run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml --run-tag kimi-default
# Sensitivity runs: --prompt-variant explicit / --embedding-model all-mpnet-base-v2
```

**Resuming:** re-running with the same `--run-tag` skips cells whose `disambig.json` and `eval.json` exist; `oracle.csv` is always reused. Delete a specific `llm__<tag>/` folder to redo only the LLM+eval steps for that model. Steps run per-cell in parallel threads (`--grid-workers`, default 5); XGBoost threads per cell are capped via `cpu_count // grid_workers` or `GRID_XGB_NTHREAD`.

### Output layout

Named shared caches + named experiment dirs. Readers dual-resolve new paths then
legacy flat paths (`format_pilot/`, top-level cells, etc.). See
[`docs/experiments_index.md`](docs/experiments_index.md). (The one-shot migration
script now lives in `archive/`.)

The outputs root defaults to `<repo>/outputs` but is relocatable with
`SURVEY_FEATURES_OUTPUTS=<path>` (`survey_features.config.OUTPUTS_DIR`).
The writing workspace defaults to `<repo>/paper` via `SURVEY_FEATURES_PAPER`
(`PAPER_DIR`). Scripts must never hard-code `ROOT / "outputs"` or `ROOT / "paper"`.

Superseded-era artifacts (era-1 accuracy-oracle cells, era-2 log-loss-v2 cells, raw
`grid_results__*.json`) live as verified zips in
`C:\Users\murrn\cursor\features_project_snapshots\` — see `MANIFEST.md` there for
contents and restore commands.

```
outputs/
  cache/
    cells/<target>_<country>/
      oracle.csv                        # shared across experiments
      llm__<exp_id>/{disambig,eval}.json
    embeddings/survey_embeddings__*.npz
    audits/leakage_audit.csv
  main/                                 # canonical free-text pipeline (was format_pilot/)
    <selector>/{freetext,extracted,maps}/
    scores_<selector>.csv               # canonical per-selector scores
    runs/<run_tag>/…                    # optional tagged map/score writes
  experiments/                          # artifacts + manifest only; logs go to logs/
    embedding_sensitivity/
    ensemble_mapping/                   # MiniLM∪mpnet fuse → one disambig
    subitem_mapping/
    similarity_threshold/               # planned
  grid/                                 # legacy JSON prelim: summaries + run manifests only
    grid_summary__<survey>__<exp_id>.csv   # (raw grid_results/llm_usage → snapshot zips)
  analysis/                             # alignment_*, uncertainty_*, _prelim_stats.json
  logs/                                 # timing JSONL + ALL shell-redirect logs
    experiments/<name>/                 # per-experiment driver logs
  .tmp/                                 # AutoGluon scratch — safe to delete
  .trash/                               # recoverable soft-deletes from /repo-audit
```

---

## Windows / libomp note

`sentence-transformers` (PyTorch) and `xgboost` both ship `libomp.dll`, which can conflict. If a run crashes or hangs during embedding or evaluation, keep XGBoost single-threaded (the scoring phase already does) or run embedding and evaluation in separate processes.

---

## Contributing / license

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under the [MIT License](LICENSE).
