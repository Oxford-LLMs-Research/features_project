# Features Project — Paper 2

**Research question:** Do LLMs understand the conditional structure of human attitudes — can they identify, from prior knowledge alone, which respondent characteristics predict how someone answers a given survey question? Does this reasoning adapt across countries?

The pipeline asks an LLM to list predictive features for a target survey question, maps those free-text descriptions to concrete survey variables, and evaluates whether the selected variables actually predict the answer better than chance.

---

## Repository layout

```
run_grid.py                  — main entry point: runs any subset of targets × countries
generate.py                  — LLM client wrapper (OpenAI-compatible) + token usage log
output_layout.py             — output path helpers: grid summaries, manifests, LLM caches
phase0b_oracle_autogluon.py  — oracle step: AutoGluon permutation importance (ground truth)
phase0b_pipeline.py          — LLM feature-selection prompts and batch runner
phase0b_mapping.py           — embedding-based retrieval (LLM label → survey variable)
phase0b_disambig.py          — LLM disambiguation (picks best candidate from shortlist)
phase0b_evaluation.py        — downstream XGBoost prediction comparison

analysis/                    — analysis and figure-building scripts
prelim/                      — manifest build, target selection, metadata introspection
paper/                       — LaTeX source and generated figures/tables
data/                        — per-survey metadata JSONs
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs `synthetic_sampling` directly from GitHub at a pinned commit so all teammates get the same version. To upgrade it, update the pinned ref and re-run `pip install`.

Follow the `synthetic_sampling` repo setup instructions to point `configs/local.yaml` at your local data files.

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, DATA_CONFIG_PATH
```

`LLM_BASE_URL` accepts any OpenAI-compatible endpoint: Nebius, Together.ai, OpenRouter, local SGLang, etc.

The disambiguation step uses a **separate fixed model** (`DISAMBIG_MODEL` in `.env`, default `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) so that differences between main LLM runs are attributable only to feature-selection quality, not disambiguation quality. It can point to a different endpoint/key via `DISAMBIG_BASE_URL` / `DISAMBIG_API_KEY`.

---

## Running the pipeline

### Single cell (one target × one country)

```bash
python run_grid.py --targets Q164 --countries Germany
python run_grid.py --survey afrobarometer --targets Q4A --countries Nigeria Kenya
```

### Multiple targets and countries

```bash
python run_grid.py --targets Q47 Q164 Q199 --countries Germany Nigeria Japan
```

### Manifest-based run (prelim 5×5 per survey)

```bash
python prelim/build_prelim_manifest.py   # writes prelim/prelim_manifest.yaml

python run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml --stop-after oracle
python run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml
```

### Full all-surveys run (PowerShell)

```powershell
.\run_prelim_full_all_surveys.ps1
# Optional: rebuild manifest first
$env:REBUILD_MANIFEST = "1"; .\run_prelim_full_all_surveys.ps1
# Tune concurrency (default 3)
$env:GRID_WORKERS = "4"; .\run_prelim_full_all_surveys.ps1
```

Logs go to `outputs/logs/run_prelim_full_<timestamp>.log`.

### Discover available countries

```bash
python run_grid.py --survey afrobarometer --list-countries
```

### Supported surveys

| `--survey` value  | Country column | Notes |
|-------------------|----------------|-------|
| `wvs`             | `B_COUNTRY`    | default; 5×5 grid pre-configured |
| `afrobarometer`   | `COUNTRY`      | |
| `arabbarometer`   | `COUNTRY`      | |
| `asianbarometer`  | `country`      | stores country names directly |
| `latinobarometer` | `IDENPA`       | |
| `ess_wave_10`     | `cntry`        | alpha-2 codes |
| `ess_wave_11`     | `cntry`        | |

---

## Pipeline steps

Each (target × country) cell runs five steps in order:

```
[1] Oracle       — AutoGluon permutation importance → ground-truth feature ranking
[2] LLM select   — ask model which respondent features would predict the answer
[3] Embed+map    — embed LLM labels, retrieve top-k matching survey variables
[4] Disambiguate — fixed small LLM picks the best candidate from the shortlist
[5] Evaluate     — compare oracle / model-selected / random feature sets via XGBoost CV
```

Steps run per-cell in parallel threads (`--grid-workers`, default 5). XGBoost thread budget per cell is capped automatically via `cpu_count // grid_workers` or overridden with `GRID_XGB_NTHREAD`.

---

## Experiment tracking

Every `run_grid.py` invocation writes a **run manifest** at `outputs/run_manifest__<survey>__<exp_id>.json`:

```json
{
  "exp_id": "deepseek-v3-default",
  "llm_model": "deepseek-ai/DeepSeek-V3.2",
  "disambig_model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
  "embedding_model": "all-MiniLM-L6-v2",
  "prompt_variant": "default",
  "survey": "wvs",
  "targets": [...],
  "countries": [...],
  "started_at": "...",
  "completed_at": "...",
  "n_cells_total": 25,
  "n_cells_completed": 25,
  "n_cells_errored": 0
}
```

The experiment ID (`--run-tag`) determines all output paths for that run. Set it explicitly to keep runs from different models, prompt variants, or embedding models separate:

```bash
# Main experiment — compare models, same everything else
python run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml \
    --run-tag deepseek-default

python run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml \
    --run-tag kimi-default

# Prompt sensitivity — same model, different prompt variant label
python run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml \
    --run-tag deepseek-prompt-v2 --prompt-variant explicit

# Embedding model sensitivity
python run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml \
    --run-tag deepseek-mpnet --embedding-model all-mpnet-base-v2
```

When multiple `grid_summary__<survey>__*.csv` exist, analysis scripts pick one per survey. Pin a specific run for figures by setting `GRID_SUMMARY_TAG=<exp_id>` in `.env`.

---

## Output layout

```
outputs/
  <target>_<country>/
    oracle.csv                          # permutation importances — shared across all models
    llm__<exp_id>/
      disambig.json                     # LLM selection + mapping results
      eval.json                         # XGBoost comparison results
  grid_summary__<survey>__<exp_id>.csv  # one row per (target, country, condition)
  grid_results__<survey>__<exp_id>.json # full nested eval payload
  llm_usage__<survey>__<exp_id>.jsonl   # per-request token usage
  run_manifest__<survey>__<exp_id>.json # experiment provenance (auto-written)
  survey_embeddings__<survey>__<embedding_model>.npz  # embedding cache, tagged by model
  logs/                                 # run logs
```

**Resuming:** re-running a grid with the same `--run-tag` skips cells whose `disambig.json` and `eval.json` already exist. The `oracle.csv` is always reused if present. Delete a specific `llm__<tag>/` folder to force rerunning only the LLM+eval steps for that model.

---

## Evaluation design

For each (target × country × condition) cell, XGBoost cross-validation accuracy is compared across three feature sets matched to the same k:

- **Oracle top-k** — the k highest-importance features (ceiling)
- **Model-selected** — the k features the LLM nominated, after mapping and disambiguation
- **Random-k** — average over 20 random draws of k features from the full pool (baseline)

Two conditions per cell: **unprompted** (no country context) and **country-provided**.

Key metrics:
- `cost_of_imperfect` = oracle_acc − model_acc
- `value_over_random` = model_acc − random_acc

---

## Oracle — extension point

The oracle is decoupled from the LLM pipeline via the cache contract. To plug in a pre-computed or alternative oracle, place a CSV at `outputs/<target>_<country>/oracle.csv` with these columns:

| column | description |
|--------|-------------|
| `target_variable` | variable code (e.g. `Q164`) |
| `country` | country code as stored in the survey data |
| `feature_variable` | predictor variable code |
| `importance_mean` | mean permutation importance |
| `importance_std` | standard deviation |
| `majority_baseline` | majority-class accuracy |

The file must contain **all features** considered — evaluation picks its own top-k from the full ranking. If the file exists, `run_grid.py` skips `compute_oracle()` entirely.

`phase0b_oracle_autogluon.py` can also run standalone:

```bash
python phase0b_oracle_autogluon.py \
    --survey wvs --targets Q47 Q57 Q199 Q235 Q164 \
    --countries Germany Nigeria Japan Brazil Egypt \
    --runtime-mode balanced --force
```

Key flags: `--runtime-mode` (`quick`/`balanced`/`best`), `--similarity-threshold` (0.85), `--test-size` (0.2), `--max-missingness-threshold` (0.2), `--force` (recompute even if cache exists).

---

## Windows / libomp note

`sentence-transformers` (PyTorch) and `xgboost` both ship `libomp.dll`, which can conflict. If the pipeline crashes during the embedding step, running oracle and LLM steps in one process and evaluation in another is a workaround — or upgrade to a newer xgboost that resolves this.
