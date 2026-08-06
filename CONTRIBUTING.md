# Contributing

Conventions that keep the confirmatory pipeline reproducible. Read
[`docs/onboarding.md`](docs/onboarding.md) before your first change.

## Three zones

| Zone | Contents | Tracked? |
|------|----------|----------|
| **Pipeline** | `src/`, `scripts/`, `tests/`, slim `docs/`, `data/` | Yes |
| **`outputs/`** | Artifacts (`OUTPUTS_DIR` / `SURVEY_FEATURES_OUTPUTS`) | No |
| **`paper/`** | Writing (`PAPER_DIR` / `SURVEY_FEATURES_PAPER`) | No |

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
- **`docs/`** — onboarding + load-bearing audit record.

## Ground rules

1. **Do not break cached-artifact contracts.** Paths under `outputs/` go through
   `layout.py` / `OUTPUTS_DIR`. Existing era-3 oracles and `main/` artifacts must keep resolving.
2. **Isolate exploratory runs** with `--run-tag` (map/score under `main/runs/<tag>/`).
3. **No result drift.** Refactors that touch scoring/metrics must re-check numbers against existing artifacts.
4. **Keep model roles fixed.** Register new selectors in `config.SELECTORS`; don't change extractor/disambiguator for a selector comparison.
5. **Python 3.9 compatibility.** `from __future__ import annotations` in every module.
6. **`.tmp/` is disposable.** Logs go under `outputs/logs/<context>/`.
7. **Fail loud in library code.** No silent fallbacks around our own files/formats/versions. Guards only at system boundaries (network, process pools, per-cell sweep isolation).
8. **Comments state the invariant**, with a `docs/` pointer for rationale. One narrative home per module — the top docstring.

## Workflow

End workdays with `/repo-audit` (`.claude/skills/repo-audit/`): audit-only by default;
cleanup/commits only with `apply-clean` / confirmed `apply-all`.
