# Contributing

Conventions that keep the confirmatory pipeline reproducible. Read
[`docs/onboarding.md`](docs/onboarding.md) before your first change.

## Which document

| Question | Read |
|----------|------|
| Where does the Dropbox tree live, and what's in it? | [`README.md`](README.md) § Outputs + [`layout.py`](src/survey_features/layout.py) |
| Where should this run write, and how are experiments named? | this file |
| What do caches mean (oracle era, identity)? | [`docs/onboarding.md`](docs/onboarding.md) §3–4 |
| What was this experiment, and what did it find? | [`docs/experiments_registry.md`](docs/experiments_registry.md) |
| What does this project term mean, in plain words? | [`docs/glossary.md`](docs/glossary.md) |
| How is everything laid out (repo, `data/`, outputs, `paper/`)? | [`docs/project_layout.md`](docs/project_layout.md) |
| End-of-day cleanup | `/repo-audit` |

Do not restate the folder tree elsewhere. If the tree changes, edit README + `layout.py` together.

## Three zones

| Zone | Contents | Tracked? |
|------|----------|----------|
| **Pipeline** | `src/`, `scripts/`, `tests/`, slim `docs/`, `data/` | Yes |
| **`outputs/`** | Artifacts in the **shared Dropbox folder** (`SURVEY_FEATURES_OUTPUTS` → `OUTPUTS_DIR`) | No |
| **`paper/`** | Writing (`PAPER_DIR` / `SURVEY_FEATURES_PAPER`) | No |

The live outputs root is the shared Dropbox folder `features_project/outputs`,
via `SURVEY_FEATURES_OUTPUTS` in **gitignored** `.env`. Confirm with
`python -c "from survey_features.config import OUTPUTS_DIR; print(OUTPUTS_DIR)"`.
The in-repo `<repo>/outputs` fallback is not the live tree. Never commit a
machine-local path.

## Setup

```bash
pip install -e .
pip install -e ".[oracle]"   # if you recompute oracles
pip install -e ".[dev]"      # pytest
cp .env.example .env
```

## Where code goes

- **`src/survey_features/`** — all shared logic. Module map in `__init__.py`.
  Canonical homes: metrics → `metrics.py`; LLM clients → `llm.py`; survey load/clean →
  `surveys.py`; embeddings/retrieval → `retrieval.py`; feature pool → `feature_pool.py`;
  oracle → `oracle.py` / `oracle_pool.py`; XGB eval → `evaluation.py`; cell scoring →
  `score_cell.py`; paths → `layout.py`; prompts → `prompts.py`; dual-layer map →
  `mapping.py`.
- **`scripts/`** — thin orchestration only (`run_main.py`, leakage, oracle, textbook).
- **`tests/`** — pure helpers (`pytest`).
- **`docs/`** — onboarding, load-bearing audit record, and the
  [**experiment registry**](docs/experiments_registry.md) (every past and future run).

## Where artifacts go

All writes resolve under `OUTPUTS_DIR` via `layout.py` helpers. Never hard-code
`ROOT / "outputs"`. Do not invent new top-level folders.

| Bucket | Use for | Do not use for |
|--------|---------|----------------|
| `cache/` | Oracle cells, embeddings, leakage audit, textbook baselines | Exploratory scores, repo-audit reports |
| `selectors/<selector>/` | Canonical confirmatory gen / extract / maps / scores | A probe that might clobber the baseline |
| `selectors/runs/<tag>/` | `--run-tag` on **map / score / pipeline** of the locked stack (`run_main.py`) | Isolating gen/extract (they stay in `selectors/<selector>/`); a different prompt, extractor, or script |
| `experiments/<name>/` | A named study with its own contract, models, or script | Overwriting confirmatory `cache/` or canonical `selectors/` |
| `experiments/_analysis/` | Digests derived from registered experiments | One-off screens that are not an experiment |
| `analysis/` | One-off diagnostic CSVs (e.g. oracle heterogeneity) | Registered experiment artifacts |
| `logs/` | Flat `token_usage_*.jsonl` and `timing_*.jsonl` (context in the filename) | Subfolders per context; score CSVs |
| `cache/audits/` | Pipeline leakage (`leakage_audit.csv`) | Repo-audit markdown |
| `audits/` | Housekeeping reports (`repo_audit_YYYY-MM-DD.md`) | Leakage / oracle audits |
| `.tmp/` | AutoGluon scratch | Anything you might need to cite |
| `.trash/` | Repo-audit soft-deletes | Pipeline writes |

**`--run-tag` vs `experiments/<name>/` are not interchangeable.**

- **`--run-tag`:** same confirmatory pipeline, same gen/extract; only maps and scores
  land under `selectors/runs/<tag>/` so you do not clobber `selectors/<selector>/maps` and
  `scores_<selector>.csv`.
- **`experiments/<name>/`:** different question (prompt, role swap, embedder, …).
  Own tree; experiment scripts must not write canonical `selectors/` or `cache/`.

## Experiment names and storage

Home: `outputs/experiments/<name>/` on the Dropbox tree. Register **before** the
first write (see below). Naming:

| Piece | Rule | Example |
|-------|------|---------|
| **Folder `<name>`** | `snake_case`, short, names the *manipulation* | `prompt_sensitivity`, `pipeline_role_swap` |
| **Registry slug** | `kebab-case` of the same words; more specific if one folder holds several contrasts | folder `pipeline_role_swap/` → slugs `pipeline-role-swap`, `extract-swap-minimax` |
| **`--run-tag`** | `snake_case`, filesystem-safe (`layout.sanitize_model_slug`); not an experiments folder | `pilot_phase_a` → `selectors/runs/pilot_phase_a/` |
| **Arms / variants** | subdirs under the study folder, not sibling top-level names | `pipeline_role_swap/minimax_nemotron/` |
| **Retries** | never overwrite; new attempt is `<name>_v2` or a new arm subdir | `extract_type_pilot_v2/` |
| **Derived digests** | only `experiments/_analysis/` (leading underscore is reserved) | contrast blocks, stack summaries |

Do not put dates, usernames, or selector names in the study folder name (selectors
are subdirs). Do not nest a new study under an unrelated one.

**Inside a study folder**, mirror the confirmatory shape where the pipeline
applies: `<selector>/[<arm>/]{freetext,extracted,maps}/` and a scores CSV at the
study root (`scores_<selector>_<arm>.csv` or `scores_<run_key>.csv`). Add
`source_meta.json` when inputs were reused from another tree. Path helpers for a
study go in `layout.py` next to the existing `prompt_sensitivity_*` /
`pipeline_role_swap_*` functions — do not hard-code `OUTPUTS_DIR / "experiments" / …`
in a new script.

## When to register

Add an entry in [`docs/experiments_registry.md`](docs/experiments_registry.md)
**before the first artifact write** when any of:

1. the run writes under `outputs/experiments/<name>/`, or
2. you will cite a `--run-tag` sweep (including pilots), or
3. the run can change confirmatory numbers under `cache/` or canonical `selectors/`.

Fill **Result** when the run finishes (or mark `status: abandoned`). Do not
register logs, `.tmp/`, or repo-audit reports.

## Ground rules

1. **Do not break cached-artifact contracts.** Paths under `outputs/` go through
   `layout.py` / `OUTPUTS_DIR`. Current-contract oracles and `selectors/` artifacts must keep resolving.
2. **No result drift.** Refactors that touch scoring/metrics must re-check numbers against existing artifacts.
3. **Keep model roles fixed.** Register new selectors in `config.SELECTORS`; don't change extractor/disambiguator for a selector comparison.
4. **Python 3.9 compatibility.** `from __future__ import annotations` in every module.
5. **`.tmp/` is disposable.** Logs are flat files under `outputs/logs/`.
6. **Fail loud in library code.** No silent fallbacks around our own files/formats/versions. Guards only at system boundaries (network, process pools, per-cell sweep isolation).
7. **Comments state the invariant**, with a `docs/` pointer for rationale. One narrative home per module — the top docstring.

## Workflow

End workdays with `/repo-audit` (`.claude/skills/repo-audit/`): audit-only by default;
cleanup/commits only with `apply-clean` / confirmed `apply-all`.
