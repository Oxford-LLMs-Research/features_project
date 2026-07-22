# Contributing

Thanks for helping out! This document covers the repo conventions that keep the
pipeline reproducible.

## Getting set up

```bash
git clone https://github.com/Oxford-LLMs-Research/features_project.git
cd features_project
pip install -e .            # or ".[oracle]" if you need to (re)compute oracles
cp .env.example .env        # fill in API keys and DATA_CONFIG_PATH
```

Python ≥ 3.9. Survey data access goes through `synthetic_sampling` (pinned in
`pyproject.toml`); follow that repo's instructions for `configs/local.yaml`.

## Where code goes

- **`src/survey_features/`** — all shared pipeline logic. If two scripts need the
  same function, it belongs here, not copy-pasted. Module responsibilities are
  documented in `src/survey_features/__init__.py`.
- **`scripts/`** — entry points only (`run_main.py` = current free-text pipeline,
  `run_grid.py` = legacy JSON grid, `leakage_audit.py`). Thin orchestration; no
  analytics logic.
- **`analysis/`** — one-off analysis and paper-build scripts. They must import the
  package (add `src/` to `sys.path` at the top if not installed) and must not
  redefine metrics that exist in `survey_features.metrics`.
- **`docs/`** — findings and design notes (markdown).
- **`archive/`** — dead scripts kept for reference. Never import from here.

## Ground rules

1. **Do not break cached-artifact contracts.** Paths and file names under
   `outputs/` are load-bearing (`survey_features/layout.py` is the single source
   of truth). Existing runs must keep resolving after your change — reruns are
   expensive (LLM calls, AutoGluon fits).
2. **No result drift.** If you refactor analysis code, re-run the affected script
   against the existing `outputs/` artifacts and confirm the numbers are
   identical before and after.
3. **Keep model roles fixed.** The extractor and disambiguator models are held
   constant across selectors so comparisons stay clean. To evaluate a new test
   model, register it in `survey_features.config.SELECTORS` — don't change the
   fixed roles.
4. **Python 3.9 compatibility.** Use `from __future__ import annotations` in every
   module; don't use syntax newer than 3.9 outside annotations.
5. **Legacy pipeline stays runnable.** `scripts/run_grid.py` (JSON prompts) backs
   the paper's appendix; changes to shared modules must not alter its behaviour
   or artifact paths.

## Workflow

- Branch from `master`, keep PRs focused and small.
- Smoke-test entry points you touched, e.g.
  `python scripts/run_main.py --phase gen --limit 1` (needs API keys) or at
  minimum `python -c "import survey_features"` plus a compile check of the
  scripts you edited.
- Describe in the PR which cached artifacts (if any) your change reads or writes.

## Reporting issues

Open a GitHub issue with the command you ran, the full traceback, and your
environment (OS, Python version, `pip show survey-features`).
