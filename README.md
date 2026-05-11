# Features Project — Paper 2

**Research question:** Do LLMs understand the conditional structure of human attitudes — can they identify, from prior knowledge alone, which respondent characteristics predict how someone answers a given survey question? Does this reasoning adapt across countries?

The pipeline asks an LLM to list predictive features for a target survey question, maps those free-text descriptions to concrete survey variables, and evaluates whether the selected variables actually predict the answer better than chance.

---

## Repository layout

```
run_grid.py            — main entry point: runs any subset of targets × countries
output_layout.py       — output paths: per-model LLM caches, grid_summary naming, discovery helpers
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

`LLM_BASE_URL` accepts any OpenAI-compatible endpoint: Nebius, Together.ai, OpenRouter, Moonshot (if exposed as OpenAI-compatible), local SGLang, etc.

Example second model (after configuring a provider that serves it):

- `LLM_MODEL=moonshotai/Kimi-K2.5`

Each full pipeline run writes LLM-specific files under a **run tag** derived from `LLM_MODEL` (slashes and spaces → underscores). Override with `python run_grid.py --run-tag my_tag ...` if the API model id does not match the folder name you want.

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

### Prelim multi-survey grids (manifest + staging)

```bash
python prelim/introspect_metadata.py        # survey metadata key inventory
python prelim/build_prelim_manifest.py      # writes prelim/prelim_manifest.yaml

python run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml --stop-after oracle
python run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml
```

PowerShell: staged oracle then optional full run with `run_prelim_staged.ps1`; set `RUN_PRELIM_FULL=1` before running to include the LLM+eval pass for all surveys after oracles are validated.

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

Steps run per-cell in parallel threads (default `--grid-workers 5`). XGBoost thread budget per cell is capped via `GRID_XGB_NTHREAD` or `cpu_count // grid_workers` to reduce oversubscription. Permutation importance in the oracle step is sequential; **joblib** parallelises only the random-feature baseline draws during evaluation.

---

## Caching and resuming

Per cell directory `outputs/<target>_<country>/`:

- **`oracle.csv`** — permutation importances (all features). **Shared across LLM models**; if it exists, every model run reuses it.
- **`llm__<run_tag>/disambig.json`** — LLM selection, retrieval, and disambiguation for that model/tag.
- **`llm__<run_tag>/eval.json`** — XGBoost comparison for that model/tag.

Survey-wide files:

- **`survey_embeddings__<survey_id>.npz`** — shared embedding cache for variable texts (not model-specific).
- **`grid_summary__<survey_id>__<run_tag>.csv`** — one row per (target, country, condition), plus `llm_model` / `llm_run_tag` columns.
- **`grid_results__<survey_id>__<run_tag>.json`** — nested eval payload per cell.

Re-running skips steps whose outputs already exist for that **run tag** (LLM paths) and always reuses `oracle.csv` when present. Delete a specific `llm__<tag>` folder to force rerunning the LLM+eval stack for that model only; delete `oracle.csv` to recompute the oracle.

If you still have **`disambig.json` and `eval.json` beside `oracle.csv`** from before this layout, move them into `llm__<tag>/` where `<tag>` matches the slug for the `LLM_MODEL` you used (or use `--run-tag`), so the runner can resume without redoing API calls.

**Analysis:** if you have both legacy `grid_summary__<survey>.csv` (no tag) and new tagged files, tooling under `analysis/` picks **one file per survey** with this priority: env `GRID_SUMMARY_TAG` (exact match on `__<tag>`), else the **newest** tagged `grid_summary__<survey>__*.csv` if any exist, else the legacy file. Set `GRID_SUMMARY_TAG` when you want figures or TeX reports pinned to a specific model run (e.g. the slug for `moonshotai/Kimi-K2.5`).

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
