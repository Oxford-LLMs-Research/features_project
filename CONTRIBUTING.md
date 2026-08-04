# Contributing

Thanks for helping out! This document covers the repo conventions that keep the
pipeline reproducible. Before your first change, read
[`docs/onboarding.md`](docs/onboarding.md) — the concepts, the cache-identity
discipline, and the sharp edges the rules below exist to protect.

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

  **Canonical homes** — every behaviour has exactly one owning module. A new `def` in
  `scripts/` or `analysis/` must either be pure orchestration or call its home;
  reimplementing a behaviour under any name is a violation the daily audit flags:

  | behaviour | home |
  |---|---|
  | metric arithmetic, bootstrap CIs, seeds | `metrics.py` |
  | LLM clients / generate-fn factories | `llm.py` |
  | survey load/clean, metadata, country maps, target types | `surveys.py` |
  | embeddings, retrieval, per-survey (svars, emb, codes) assets | `retrieval.py` |
  | feature-pool construction, filters, skip-pattern screen | `feature_pool.py` |
  | oracle fit, honest split, ceiling, contract | `oracle.py` / `oracle_pool.py` |
  | downstream estimators (XGB CV, per-type scoring) | `evaluation.py` |
  | cell scoring, scores schema, baseline caches | `score_cell.py` |
  | output path contracts | `layout.py` |
  | prompt templates | `prompts.py` |
- **`scripts/`** — entry points only (`run_main.py` = current free-text pipeline,
  `run_grid.py` = legacy JSON grid, `leakage_audit.py`). Thin orchestration; no
  analytics logic.
- **`analysis/`** — one-off analysis and paper-build scripts. They must import the
  package (add `src/` to `sys.path` at the top if not installed) and must not
  redefine metrics that exist in `survey_features.metrics`.
- **`docs/`** — findings, design notes, and [`docs/experiments_index.md`](docs/experiments_index.md).
- **`paper/`** — LaTeX writeup + generated tables/figures (gitignored; not required to run the pipeline).
- **`archive/`** — dead scripts kept for reference. Never import from here.

## Ground rules

1. **Do not break cached-artifact contracts.** Paths under `outputs/` are
   load-bearing (`survey_features/layout.py` is the single source of truth).
   Readers dual-resolve new layout then legacy; writers prefer the new tree.
   Existing runs must keep resolving after your change — reruns are expensive.
2. **Isolate exploratory runs.** Use `--run-tag` for map/score (and experiment
   runners) so you write under `…/runs/<tag>/` instead of clobbering the
   canonical `main/` or another person's experiment output.
3. **Register new experiments.** Add `docs/<name>.md`, a row in
   `docs/experiments_index.md`, and a helper in `layout.py` before first write.
4. **No result drift.** If you refactor analysis code, re-run the affected script
   against the existing `outputs/` artifacts and confirm the numbers are
   identical before and after.
5. **Keep model roles fixed.** The extractor and disambiguator models are held
   constant across selectors so comparisons stay clean. To evaluate a new test
   model, register it in `survey_features.config.SELECTORS` — don't change the
   fixed roles.
6. **Python 3.9 compatibility.** Use `from __future__ import annotations` in every
   module; don't use syntax newer than 3.9 outside annotations.
7. **Legacy pipeline stays runnable.** `scripts/run_grid.py` (JSON prompts) backs
   the paper's appendix; changes to shared modules must not alter its behaviour
   for dual-resolved artifact paths.
8. **`.tmp/` is disposable.** `outputs/.tmp/` holds AutoGluon scratch; delete
   freely between runs.
9. **Fail loud in library code.** No `try/except` around our own files, formats, or
   version assumptions — a corrupt cache or broken invariant should crash with a clear
   message, not fall back silently to a wrong answer. Guards belong only at system
   boundaries: network calls, process-pool workers, per-cell isolation in sweep
   drivers, and genuine statistical fallbacks (e.g. stratification on thin classes).
10. **Comments state the invariant, not the story.** One or two lines saying what must
   hold, plus a pointer (`docs/onboarding.md §5`, `docs/pipeline_audit_2026-08.md §A1`)
   for the rationale. Measured numbers, incident history, and design argument live in
   `docs/`, never inline. One narrative home per module — the top docstring — and
   pointers everywhere else.

## Workflow

- **End every workday with `/repo-audit`** (project skill, `.claude/skills/repo-audit/`):
  it safe-cleans mechanical clutter, checks the rules above on the day's diff, groups
  the day's work into commits, and hands you a ranked walk-through list for your own
  review pass. The agent never pushes.
- Branch from `master`, keep PRs focused and small.
- Smoke-test entry points you touched, e.g.
  `python scripts/run_main.py --phase gen --limit 1` (needs API keys) or at
  minimum `python -c "import survey_features"` plus a compile check of the
  scripts you edited.
- Describe in the PR which cached artifacts (if any) your change reads or writes.

## Reporting issues

Open a GitHub issue with the command you ran, the full traceback, and your
environment (OS, Python version, `pip show survey-features`).
