# Survey Features — LLM feature-selection capability study

**Research question:** Do LLMs know, from a survey question's wording alone, which respondent characteristics predict the answer — and does that knowledge adapt across countries?

The confirmatory pipeline: free-text essay → typed feature list → dual-layer map to survey codes → score against an AutoGluon permutation-importance oracle.

> **New here?** Start with [`docs/onboarding.md`](docs/onboarding.md).

---

## Three zones

| Zone | Role | In git? |
|------|------|---------|
| **Pipeline** | `src/`, `scripts/`, `tests/`, slim `docs/`, `data/` | Yes |
| **`outputs/`** | Run artifacts and caches (`SURVEY_FEATURES_OUTPUTS`) | No |
| **`paper/`** | Writing workspace (`SURVEY_FEATURES_PAPER`) | No |

This branch is the **minimal core** (Arm C free-text loop + oracle/leakage/textbook prereqs). Legacy JSON arms, side experiments, and archive live on other branches / snapshots.

---

## Layout

```
src/survey_features/     shared library (see __init__.py module map)
scripts/
  run_main.py            gen → extract → map → score (Arm C)
  leakage_audit.py       genuine-cell grid
  compute_oracle.py      fit one cell's oracle
  rerun_oracles.py       recompute oracles (process pool)
  build_textbook_baseline.py
data/                    survey metadata JSONs + targets.yaml
tests/                   pytest (no API / data required for unit tests)
docs/                    onboarding + pipeline_audit_2026-08.md
```

---

## Setup

```bash
pip install -e .                # core library
pip install -e ".[oracle]"      # + autogluon (recompute oracles)
pip install -e ".[dev]"         # + pytest
cp .env.example .env            # fill API keys and DATA_CONFIG_PATH
```

Python ≥ 3.9. Survey microdata comes via pinned `synthetic_sampling` (see `pyproject.toml`). Prefer the conda interpreter with the scientific stack if `.venv` lacks AutoGluon.

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

Use `--run-tag <name>` on map/score/pipeline to write under `main/runs/<name>/` without clobbering the canonical baseline. Gen/extract always stay under `main/<selector>/`.

---

## Outputs (canonical)

```
outputs/
  cache/
    cells/<target>_<country>/oracle.csv
    embeddings/
    audits/leakage_audit.csv
    baselines/textbook__<survey>.json
  main/
    <selector>/{freetext,extracted,maps}/
    scores_<selector>.csv
  logs/
  .tmp/                  # AutoGluon scratch; safe to delete
```

Path helpers live in `survey_features.layout` — never hard-code `ROOT / "outputs"` in scripts.

---

## Tests

```bash
pytest tests/ -q
```
