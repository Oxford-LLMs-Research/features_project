# Features Project — Paper 2

**Research question:** Do LLMs understand the conditional structure of human attitudes — can they identify, from prior knowledge alone, which respondent characteristics predict how someone answers a given survey question? Does this reasoning adapt across countries?

The pipeline asks an LLM to list predictive features for a target survey question, maps those free-text descriptions to concrete survey variables, and evaluates whether the selected variables actually predict the answer better than chance.

---

## Repository layout

```
run_grid.py            — main entry point: runs any subset of targets × countries
generate.py            — LLM client wrapper (OpenAI-compatible)
phase0b_oracle.py      — oracle step: XGBoost permutation importance (ground truth)
phase0b_pipeline.py    — LLM feature selection prompts and batch runner
phase0b_mapping.py     — embedding-based retrieval (LLM label → survey variable)
phase0b_disambig.py    — LLM disambiguation (picks best candidate from shortlist)
phase0b_evaluation.py  — downstream XGBoost prediction comparison
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs `synthetic_sampling` directly from GitHub at a pinned commit so all teammates get the same version.

To intentionally upgrade it later:
- choose a new `synthetic_sampling` commit/tag
- update the pinned ref in `requirements.txt`
- re-run `pip install -r requirements.txt`

Follow the `synthetic_sampling` repo setup instructions to point `configs/local.yaml` at your local data files.

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, DATA_CONFIG_PATH
```

`LLM_BASE_URL` accepts any OpenAI-compatible endpoint: Nebius, Together.ai, OpenRouter, local SGLang, etc.

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

### Full default grid (WVS, 5 targets × 5 countries)

```bash
python run_grid.py
```

### Discover available countries for any survey

```bash
python run_grid.py --survey afrobarometer --list-countries
python run_grid.py --survey ess_wave_10 --list-countries
```

### Supported surveys

| `--survey` value  | Country column | Notes |
|-------------------|----------------|-------|
| `wvs`             | `B_COUNTRY`    | default; 5×5 grid pre-configured |
| `afrobarometer`   | `COUNTRY`      | |
| `arabbarometer`   | `COUNTRY`      | |
| `asianbarometer`  | `country`      | stores country names, not codes |
| `latinobarometer` | `IDENPA`       | |
| `ess_wave_10`     | `cntry`        | alpha-2 codes; not all countries in every wave's data file |
| `ess_wave_11`     | `cntry`        | |

Country names and admin columns are derived automatically from survey metadata — no hardcoding needed.

---

## Pipeline steps

Each (target × country) cell runs five steps in order:

```
[1] Oracle       — XGBoost permutation importance → ground-truth feature ranking
[2] LLM select   — ask model which respondent features would predict the answer
[3] Embed+map    — embed LLM labels, retrieve top-k matching survey variables
[4] Disambiguate — LLM picks the best candidate from the shortlist (or "none")
[5] Evaluate     — compare oracle / model-selected / random feature sets via XGBoost CV
```

Steps run per-cell in parallel threads (`N_CELL_WORKERS = 5`). The permutation importance inner loop is parallelised across CPU cores via joblib.

---

## Caching and resuming

Every intermediate result is cached in `outputs/<target>_<country>/`:

```
outputs/
  Q164_Germany/
    oracle.csv      ← permutation importances (all features)
    disambig.json   ← LLM selection + mapping results
    eval.json       ← XGBoost comparison results
  survey_embeddings.npz   ← shared across all cells, rebuilt if metadata changes
  grid_summary.csv        ← one row per (target, country, condition)
  grid_results.json       ← full nested results
```

Re-running skips any cell whose outputs already exist. Delete a cell's directory to force a full rerun of that cell, or delete individual files to rerun specific steps.

---

## Oracle step — extension point for teammates

The oracle (`phase0b_oracle.py`) is fully decoupled from the rest of the pipeline via the cache contract:

**To plug in a pre-computed or alternative oracle**, place a CSV at `outputs/<target>_<country>/oracle.csv` with these columns:

| column | description |
|--------|-------------|
| `target_variable` | variable code (e.g. `Q164`) |
| `country` | country code as stored in the survey data |
| `feature_variable` | variable code of the predictor |
| `importance_mean` | mean permutation importance (higher = more important) |
| `importance_std` | standard deviation across folds |
| `majority_baseline` | majority-class accuracy (used for reference) |

The file must contain **all features** considered (not just top-k) — the evaluation step picks its own top-k from the full ranking.

If the file exists, `run_grid.py` loads it and skips `compute_oracle()` entirely.

**To modify the oracle method**, edit `phase0b_oracle.py` directly. The function signature and return types must stay the same. Key hyperparameters (`n_splits`, `n_repeats`, `random_state`) are passed per cell and are intentionally not fixed globally — optimal values vary by question complexity and country sample size.

---

## Evaluation design

For each (target × country × condition) cell, XGBoost cross-validation accuracy is compared across three feature sets matched to the same k:

- **Oracle top-k** — the k highest-importance features by permutation importance (ceiling)
- **Model-selected** — the k features the LLM selected, after mapping and disambiguation
- **Random-k** — average over 20 random draws of k features from the full pool (baseline)

Two conditions per cell: **unprompted** (no country context given to the LLM) and **country-provided**.

Key metrics reported:
- `cost_of_imperfect` = oracle_acc − model_acc (how much the LLM's selection costs)
- `value_over_random` = model_acc − random_acc (whether LLM reasoning beats chance)

---

## Notes on text-coded surveys

Some surveys store response labels as strings ("Agree", "Very bad") rather than numeric codes. The pipeline handles this automatically:
- Genuine text columns are detected and label-encoded before XGBoost
- Numeric-as-string columns (e.g. `"1"`, `"2"`) are coerced to float
- Text-coded target variables fall back to label encoding

---

## Windows / libomp note

`sentence-transformers` (PyTorch) and `xgboost` both ship `libomp.dll`, which can conflict. If the pipeline crashes during the embedding step, run oracle and LLM steps in one process and evaluation in another by temporarily commenting out the eval call — or upgrade to a newer xgboost that resolves this.
