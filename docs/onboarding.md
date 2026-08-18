# Onboarding — read this first

The short path into this repo. README covers setup and where the Dropbox
outputs tree lives; CONTRIBUTING covers where a run should write; this covers
**how the system thinks**.

---

## 1. What the project measures

```
                    THE MODEL'S SIDE (per selector LLM)
  question text ──▶ free-text essay ──▶ typed feature list ──▶ survey variable codes
                    (gen)               (extract, FIXED model)  (retrieve + dual-layer
                                                                 disambig, FIXED models)
                    THE GROUND-TRUTH SIDE (once per cell)
  survey data  ──▶ AutoGluon permutation importance = "the oracle"

                    THE COMPARISON (per cell × selector)
  captured importance = model's oracle-mass / oracle top-k mass
  + XGBoost: model picks vs random-k vs textbook demographics
```

A **cell** = (survey, target question, country). The grid is cells that survived the
leakage screen (`layout.genuine_cells()`).

Three model roles (never mixed): **selector** (under test), **extractor** (fixed),
**disambiguator** (fixed). Mapping is dual-layer: parents plus bundled `sub_items`
(when ≥2) become `expanded_codes` — the headline set for scoring.

JSON-era arms A/B and side experiments are **removed on this branch**; use the full
research branch / snapshots if you need them.

## 2. Vocabulary

| term | meaning |
|---|---|
| oracle | AutoGluon PI on real survey data — gold standard for selection |
| honest split | fit on 60% (T), rank on 20% (V1), value on 20% (V2). See top of `oracle.py` |
| `oracle_ceiling@k` | V1-chosen, V2-valued top-k mass — report models against this, not 1.0 |
| captured importance | Σ oracle importance of model picks ÷ Σ oracle top-k (matched k) |
| value_over_random / _textbook | predictive score minus matched-k random / textbook demographics |
| k | number of variables the model's requests mapped to (model-chosen k is part of the measurand) |
| genuine / degenerate / leakage | leakage-audit verdict per cell |
| target type | binary / nominal / ordinal / continuous — drives oracle + evaluator |

## 3. Where truth lives — oracle eras

Every published number depends on the oracle. The **contract version** in each cell's
`oracle_meta.json` records which rules built it:

| contract | rules | cache |
|---|---|---|
| v1 | accuracy metric, single 80/20 | archived out of tree (snapshots) |
| v2 | log loss, honest 60/20/20, still multiclass | archived out of tree |
| **v3** | + measurement-level aware; out-of-scale sentinels excluded | `outputs/cache/cells/` (**current**) |

Constant: `ORACLE_CONTRACT_VERSION` in `src/survey_features/oracle.py`.
**Any change that alters what oracle outputs mean must bump the version** and add a
row here. Full audit: `docs/pipeline_audit_2026-08.md`.

**Which numbers are current on this branch:** era-3 oracles under `outputs/cache/cells/`,
and free-text Arm C artifacts under `outputs/main/`. Those paths resolve under the
shared Dropbox root (`SURVEY_FEATURES_OUTPUTS`). Experiment-by-experiment claims
live in [`docs/experiments_registry.md`](experiments_registry.md) — register before
writing when CONTRIBUTING says to.

## 4. Cache identity

A cached artifact must carry the identity of the process that made it:

- **`contract_version`** in `oracle_meta.json`
- **`_fingerprint`** in each cell's `baselines.json` (pool + textbook + draws)
- **schema guard** in `score_cell.run_score_jobs` — mismatched score CSV headers are archived aside

If you add a cache, give it an identity field.

## 5. Sharp edges

1. AutoGluon `time_limit` is wall-clock — don't sleep the laptop mid-fit; thread-level concurrency is useless-to-fatal for AG.
2. Never run concurrent AutoGluon fits in one process — use `rerun_oracles.py --processes N`.
3. `quick` runtime mode degrades rankings; don't quote it.
4. DK/refused are answers; structural missingness → NaN (`surveys.py` taxonomy).
5. Target type detection is fallible — check `TARGET_TYPE_OVERRIDES` when adding targets.
6. The disambiguator is not deterministic at temperature 0; cached maps are the reproducibility unit.
7. Windows: Ray disabled in oracle fit; spawned workers need `_ensure_src_on_pythonpath`.
8. Prefer conda `miniconda3` if `.venv` lacks AutoGluon.
9. Baselines are model-independent and cached per (cell, k).
10. The live `outputs/` tree is the shared Dropbox folder `features_project/outputs`,
    via `SURVEY_FEATURES_OUTPUTS` in gitignored `.env`. Confirm `OUTPUTS_DIR` before
    a run. Dropbox may briefly lock files mid-sync; if a heavy run hits a spurious
    `PermissionError` on write, pause syncing for the run. One writer machine
    only — collaborators read.

## 6. Running things

```bash
python scripts/rerun_oracles.py --processes 3   # after oracle contract changes
python scripts/leakage_audit.py --with-data     # after ANY oracle change
python scripts/build_textbook_baseline.py

python scripts/run_main.py --phase gen|extract|map|score --selector deepseek
# map/pipeline need --disambiguator nemotron
```

Order: oracle → leakage → gen/extract/map → score.

End workdays with `/repo-audit` (see CONTRIBUTING).

## 7. Suggested first week

1. This file → README § Outputs → CONTRIBUTING (where to write) → skim `docs/pipeline_audit_2026-08.md` (§A1 + honest split).
2. Top docstrings of `oracle.py`, `surveys.py`, `mapping.py`, `score_cell.py`, `metrics.py`.
3. Open one cell under `outputs/cache/cells/` and match files to the diagram in §1.
4. `pytest tests/` and optionally `--limit 1` on a phase.
5. Before touching caches, reread §4.
