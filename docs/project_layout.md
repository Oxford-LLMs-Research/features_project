# Project layout — where everything lives and why

The map of the whole project: the repo (code + inputs), the shared Dropbox outputs
tree (artifacts), and the paper folder (writing). Written for a human who was not in
the room; project terms used here are defined in plain language in
[`glossary.md`](glossary.md). Policy for *where a run should write* stays in
[`CONTRIBUTING.md`](../CONTRIBUTING.md); this file explains *what each place is*.

## The three zones

| Zone | What | Where | Tracked in git? |
|---|---|---|---|
| Repo | code, tests, docs, small input configs | this checkout | yes |
| Outputs | every artifact a run produces | shared Dropbox folder, via `SURVEY_FEATURES_OUTPUTS` in gitignored `.env` | no |
| Paper | the manuscript and its figures | `paper/` (env `SURVEY_FEATURES_PAPER` / `PAPER_DIR`) | no |

Rationale: code is reviewed and versioned; artifacts are large, regenerable, and
shared with collaborators who don't use git; the paper is written by humans and
compiled from generated fragments. Keeping the three apart means a `git clean` can
never eat a result and a Dropbox hiccup can never corrupt code.

## Repo (inputs) — what's tracked

- `src/survey_features/` — all shared logic; the module map is in
  `src/survey_features/__init__.py`. Paths are built ONLY by `layout.py` helpers.
- `scripts/` — thin orchestration entry points (`run_main.py` runs the confirmatory
  pipeline; `compute_oracle.py`/`rerun_oracles.py` build oracles;
  `leakage_audit.py` classifies cells; `target_universe_screen.py` screens candidate
  targets).
- `tests/` — pytest suite (pure helpers, no network, no Dropbox writes).
- `docs/` — onboarding, glossary, this file, decision memos, the experiment registry
  (every past and future run), and load-bearing audit records.
- `data/` — the small, tracked **inputs** that define what the study runs on:
  - `targets.yaml` — the candidate target questions per survey (the target universe).
  - `pilot_cells.csv` — the explicit cell list used by the Phase A pilot.
  - `prompt_sensitivity_cells.yaml` — cell sample for the prompt-sensitivity study.
  - `nebius_pricing.json` — per-model API prices for cost tracking.
  - `_target_universe_inventory.json` — *gitignored by design* (830 KB, regenerated
    by `scripts/target_universe_screen.py`); the leading underscore marks it derived.
  - Raw survey data files are NOT here: they live outside the repo and are located
    via `DATA_CONFIG_PATH` in `.env`.
- `.claude/` (skills + settings) and `.cursor/rules/` — agent instructions. The
  no-naked-jargon rule and operational basics live here and in `CLAUDE.md`.
- `paper/` (untracked, in the checkout) — `current_state.tex`/`.pdf`, `figures/`
  (with `FIGURE_MANIFEST.tex`), `generated_current_state/` (machine-written tables).
  The eventual paper is written here; generated fragments are re-derivable from
  outputs, prose is not.

## Outputs (Dropbox) — what each bucket is

Live root: the shared Dropbox folder `features_project/outputs`. One writer machine;
collaborators read. Confirm you're pointed at it:
`python -c "from survey_features.config import OUTPUTS_DIR; print(OUTPUTS_DIR)"`.

| Bucket | What it is | Why it exists | Who writes it |
|---|---|---|---|
| `cache/cells/<target>_<country>/` | one oracle per cell (`oracle.csv` + `oracle_meta.json`) | the ground-truth ranking every selector is judged against; expensive to compute, so cached with a `contract_version` identity | `rerun_oracles.py` / `compute_oracle.py` |
| `cache/embeddings/` | sentence-embedding caches per survey | retrieval reuses them across runs | pipeline (on demand) |
| `cache/audits/` | `leakage_audit.csv` + summary — the grid screen | decides which cells form the default grid (`genuine`) | `leakage_audit.py` |
| `cache/baselines/` | textbook feature sets per survey | the human-expert comparator | `build_textbook_baseline.py` |
| `selectors/` | canonical confirmatory outputs, one folder per selector LLM (`<selector>/{freetext,extracted,maps}/`), plus `scores_<selector>.csv` | these are *the* numbers-in-the-paper artifacts; nothing exploratory may overwrite them | `run_main.py` (untagged) |
| `selectors/runs/<tag>/` | tagged re-runs of map/score (gen/extract stay shared) | probes and pilots that must not clobber the canonical files | `run_main.py --run-tag` |
| `experiments/<name>/` | named exploratory studies (own prompts/models/scripts), registered in `docs/experiments_registry.md` **before** first write | exploration is kept out of the confirmatory tree so paper numbers aren't tinkering survivors | experiment scripts |
| `experiments/_analysis/` | digests derived from registered experiments | keeps derived tables next to, but distinct from, raw study outputs | analysis scripts |
| `analysis/` | one-off diagnostic CSVs that are *not* a named experiment (e.g. oracle heterogeneity screen) | quick diagnostics need a home that is neither canonical nor a registered study | diagnostic scripts |
| `logs/` | flat `token_usage_*.jsonl`, `timing_*.jsonl` (context in the filename) | cost/latency accounting | every run |
| `audits/` | repo-audit housekeeping reports (`repo_audit_YYYY-MM-DD.md`) | the end-of-workday audit trail — NOT the leakage audit | `/repo-audit` |
| `.tmp/` | AutoGluon scratch | safe to delete any time | oracle fits |
| `.trash/` | repo-audit soft-deletes, dated subfolders | recoverable deletion; pruned after ~14 days | `/repo-audit` |

The distinction people ask about most: **`selectors/` vs `experiments/` vs
`analysis/`** — `selectors/` is the locked confirmatory pipeline's canonical output
(one folder per LLM under test); `experiments/` is registered, named exploration;
`analysis/` is unregistered one-off diagnostics. If a run could change a number the
paper will cite, it belongs in `selectors/` (tagged if it's a probe); if it answers a
side question, `experiments/`; if it's a quick look, `analysis/`.

History note (2026-08-19): `selectors/` was called `main/` until the restructure;
legacy era-3 folders `grid/` and `sensitivity/`, and the old accuracy-era
`scores_*.csv`, are deprecated v3 byproducts scheduled for `.trash` after the v4
oracle recompute. The Dropbox root may also hold a frozen July-2026 pre-rewrite
snapshot — history, not live cache.

## Keeping this in sync

The outputs tree is authoritative in two places that must move together:
`README.md` § Outputs and `src/survey_features/layout.py` (docstring + helpers).
This file explains rationale and must be updated third. A copy of the outputs map
lives as `README.md` inside the Dropbox outputs root for collaborators — regenerate
it when the tree changes.
