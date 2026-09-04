# Survey Features — LLM feature-selection capability study

**Research question:** Do LLMs know, from a survey question's wording alone, which respondent characteristics predict the answer — and does that knowledge adapt across countries?

The confirmatory pipeline: free-text essay → typed feature list → dual-layer map to survey codes → score against an AutoGluon permutation-importance oracle.

> **New here?** Start with [`docs/onboarding.md`](docs/onboarding.md).
> **Where should a run write?** [`CONTRIBUTING.md`](CONTRIBUTING.md).
> **Citing an experiment?** [`docs/experiments_registry.md`](docs/experiments_registry.md).

---

## Three zones

| Zone | Role | In git? |
|------|------|---------|
| **Pipeline** | `src/`, `scripts/`, `tests/`, slim `docs/`, `data/` | Yes |
| **`outputs/`** | Run artifacts and caches — **shared Dropbox folder**, not the checkout | No |
| **`paper/`** | Writing workspace (`SURVEY_FEATURES_PAPER`) | No |

The live outputs root is the shared Dropbox folder set by `SURVEY_FEATURES_OUTPUTS`
in `.env` — see [Outputs](#outputs-dropbox-not-git). Path helpers:
[`src/survey_features/layout.py`](src/survey_features/layout.py). Where to write:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

This branch is the **minimal core** (Arm C free-text loop + oracle/leakage/textbook prereqs). Legacy JSON arms, side experiments, and archive live on other branches / snapshots.

---

## Layout

```
src/survey_features/     shared library (see __init__.py module map)
scripts/
  run_main.py            gen → extract → map → score (Arm C)
  leakage_audit.py       default grid (type-1 + leakage; not accuracy-vs-majority)
  compute_oracle.py      fit one cell's oracle
  rerun_oracles.py       recompute oracles (process pool)
  build_textbook_baseline.py
data/                    targets.yaml, pricing, experiment cell lists
tests/                   pytest (no API / data required for unit tests)
docs/                    onboarding, pipeline_audit, experiments_registry
```

---

## Setup

```bash
pip install -e .                # core library
pip install -e ".[oracle]"      # + autogluon (recompute oracles)
pip install -e ".[dev]"         # + pytest
cp .env.example .env            # API keys, DATA_CONFIG_PATH, SURVEY_FEATURES_OUTPUTS
```

Python ≥ 3.9. Survey microdata and codebook metadata come via pinned `synthetic_sampling` (see `pyproject.toml` and `DATA_CONFIG_PATH` in `.env.example`). Prefer the conda interpreter with the scientific stack if `.venv` lacks AutoGluon.

---

## Run the loop

Prerequisites (once, or after contract changes):

```bash
python scripts/rerun_oracles.py --processes 3
python scripts/leakage_audit.py --with-data
python scripts/build_textbook_baseline.py
```

Then per selector:

```bash
python scripts/run_main.py --phase gen      --selector deepseek
python scripts/run_main.py --phase extract  --selector deepseek
python scripts/run_main.py --phase map      --selector deepseek --disambiguator nemotron
python scripts/run_main.py --phase score    --selector deepseek

# or one shot (optional concurrency):
python scripts/run_main.py --phase pipeline --selector deepseek --disambiguator nemotron \
    --pipeline-workers 4 --map-workers 8 --with-score
```

Use `--run-tag <name>` on map/score/pipeline to write under `selectors/runs/<name>/`
without clobbering the canonical baseline. Gen/extract always stay under
`selectors/<selector>/`. Named studies (different prompt, models, or script) go under
`outputs/experiments/<name>/` — see CONTRIBUTING for which bucket to use.

---

## Outputs (Dropbox, not git)

The live artifact root is a **shared Dropbox folder** (`features_project/outputs`
after Dropbox syncs). The local path goes in `.env` as `SURVEY_FEATURES_OUTPUTS`
(gitignored — do not commit a machine path). Confirm:

```bash
python -c "from survey_features.config import OUTPUTS_DIR; print(OUTPUTS_DIR)"
```

Without the env var, code falls back to `<repo>/outputs` — that is **not** the
current setup. Path helpers live in `survey_features.layout`; never hard-code
`ROOT / "outputs"`. One writer machine; collaborators read via Dropbox (the one
registered exception — two laptops on disjoint oracle cells — is in
`docs/oracle_handoff_2026-09.md`). If a
heavy run hits a spurious `PermissionError`, pause Dropbox sync for the run.

The Dropbox folder may also hold a frozen July-2026 pre-rewrite snapshot at the
root (flat files, `format_pilot/`). Do not treat that as live cache.

```
outputs/                              # OUTPUTS_DIR — Dropbox share
  cache/                              # confirmatory prerequisites
    cells/<target>_<country>/oracle.csv
    embeddings/
    audits/leakage_audit.csv          # pipeline leakage (not repo-audit)
    baselines/textbook__<survey>.json
  selectors/                          # canonical confirmatory per-selector outputs (was main/)
    <selector>/{freetext,extracted,maps}/
    scores_<selector>.csv
    runs/<run_tag>/                   # --run-tag map/score only
  experiments/<name>/                 # named studies (register first)
    _analysis/                        # digests from registered experiments
  analysis/                           # one-off screens, not a named experiment
  logs/                               # flat token_usage_*.jsonl, timing_*.jsonl
  audits/                             # housekeeping reports (repo-audit)
  .tmp/                               # AutoGluon scratch; safe to delete
  .trash/                             # repo-audit soft-deletes
```

**Where to write** ( `--run-tag` vs `experiments/`, when to register):
[`CONTRIBUTING.md`](CONTRIBUTING.md). **What caches mean:**
[`docs/onboarding.md`](docs/onboarding.md) §3–4. **What each bucket is and why —
plus how `data/` inputs and `paper/` are organized:**
[`docs/project_layout.md`](docs/project_layout.md). **Project terms in plain
language:** [`docs/glossary.md`](docs/glossary.md).

---

## Tests

```bash
pytest tests/ -q
```
