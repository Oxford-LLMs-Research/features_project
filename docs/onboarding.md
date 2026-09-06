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

A **cell** = (survey, target question, country). The default grid is cells that
survived type-1 (unestimable PI) and leakage (`layout.genuine_cells()`). Tiny
accuracy lift vs majority is **not** a drop — see `grid_screen.py`.

Three model roles (never mixed): **selector** (under test), **extractor** (fixed),
**disambiguator** (fixed). Mapping is dual-layer: parents plus bundled `sub_items`
(when ≥2) become `expanded_codes` — the headline set for scoring.

JSON-era arms A/B and side experiments are **removed on this branch**; use the full
research branch / snapshots if you need them.

## 2. Vocabulary

Metric-level terms below; project-level terms (confirmatory, Arm C, selector, phases,
grid-screen classes) are defined in plain language in [`glossary.md`](glossary.md).

| term | meaning |
|---|---|
| oracle | AutoGluon PI on real survey data — gold standard for selection |
| honest split (v4) | 50% R ranks by k-fold CV, 30% D reserved for the downstream evaluator (= `train_index`), 20% V2 values the picks. See top of `oracle.py` |
| `oracle_ceiling@k` | CV-ranked, V2-valued top-k mass — report models against this, not 1.0 |
| captured importance | Σ oracle importance of model picks ÷ Σ oracle top-k (matched k) |
| value_over_random / _textbook | predictive score minus matched-k random / textbook demographics |
| k | number of variables the model's requests mapped to (model-chosen k is part of the measurand) |
| genuine / unestimable / leakage | grid-screen verdict per cell (type-1 + leakage; not accuracy-vs-majority) |
| target type | binary / nominal / ordinal / continuous — drives oracle + evaluator |

## 3. Where truth lives — oracle eras

Every published number depends on the oracle. The **contract version** in each cell's
`oracle_meta.json` records which rules built it:

| contract | rules | cache |
|---|---|---|
| v1 | accuracy metric, single 80/20 | archived out of tree (snapshots) |
| v2 | log loss, honest 60/20/20, still multiclass | archived out of tree |
| v3 | + measurement-level aware; out-of-scale sentinels excluded; ONE-SHOT split | superseded 2026-08-19 — a dev byproduct; do not cite or build on it |
| **v4** | k-fold CV ranking + 30% eval reserve (= `train_index`) + untouched 20% valuation holdout; reliability block in meta | `outputs/cache/cells/` (**current**) |

Constant: `ORACLE_CONTRACT_VERSION` in `src/survey_features/oracle.py`.
**Any change that alters what oracle outputs mean must bump the version** and add a
row here. Full audit: `docs/pipeline_audit_2026-08.md`.

**Which numbers are current on this branch:** contract-v4 oracles under `outputs/cache/cells/`
(check `contract_version` in each meta — v3 stragglers are stale),
and free-text Arm C artifacts under `outputs/selectors/`. Those paths resolve under the
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
3. Runtime tiers are a registered choice, not a knob: the confirmatory tier is `quick`
   (`medium_quality`) **with `--autogluon-time-limit 600`** so the 11-model bag always
   finishes — at the preset's own 60 s the wall clock cut the bag to 3–8 models that
   varied by fold (smoke 2026-09-04, registry `confirmatory-oracle-map`). `balanced` is
   reserved for the later re-fit of the 30 transportability questions. Never mix tiers
   inside one comparison; check `provenance.folds[*].n_models` before citing a cell.
4. DK/refused are answers; structural missingness → NaN (`surveys.py` taxonomy).
5. Target type detection is fallible — check `TARGET_TYPE_OVERRIDES` when adding targets.
6. The disambiguator is not deterministic at temperature 0; cached maps are the reproducibility unit.
7. Windows: Ray disabled in oracle fit; spawned workers need `_ensure_src_on_pythonpath`. `FASTAI` is excluded (`EXCLUDED_MODEL_TYPES`) so Dropbox locks on `.tmp/*.pth` cannot starve the time_limit.
8. Prefer conda `miniconda3` if `.venv` lacks AutoGluon.
8a. Asian Barometer loads from `data/Asianbarometer/*.dta` (numeric codes), never from the
    combined CSV (label text → ~230 categoricals → 20–50× slower fits and a cut bag;
    registry `confirmatory-oracle-map`). `load_survey` fails loud if the `.dta` files are
    missing: copy them from `data/misc/`. Grid country "Korea" = metadata "South Korea" (code 3).
9. Baselines are model-independent and cached per (cell, k).
10. The live `outputs/` tree is the shared Dropbox folder `features_project/outputs`,
    via `SURVEY_FEATURES_OUTPUTS` in gitignored `.env`. Confirm `OUTPUTS_DIR` before
    a run. Dropbox may briefly lock files mid-sync; if a heavy run hits a spurious
    `PermissionError` on write, pause syncing for the run. One writer machine
    only — collaborators read. **Registered exception:** the September 2026 oracle
    map is split across two laptops writing *disjoint* cell folders under
    `cache/cells/` (partition by survey, never the same cell twice); see
    `docs/oracle_handoff_2026-09.md`. Each machine marks its own `outputs/.tmp`
    as Dropbox-ignored so AutoGluon scratch never syncs.

## 6. Running things

```bash
cp -r outputs/cache/cells outputs/cache/cells_v3   # archive before a contract migration
python scripts/rerun_oracles.py --processes 3   # after oracle contract changes
python scripts/rerun_oracles.py --cells-csv data/confirmatory_grid_cells.csv \
    --role confirmatory --survey wvs --runtime-mode quick --autogluon-time-limit 600 --processes 3
python scripts/oracle_provenance_census.py       # same bag everywhere? (before mixing machines)
python scripts/leakage_audit.py --with-data --cells-csv data/confirmatory_grid_cells.csv
#   ^ after ANY oracle change. --cells-csv is REQUIRED for the confirmatory grid: without it the
#   audit's universe is the old data/targets.yaml catalog and it screens 7 of the 120 grid targets.
# then archive any pre-existing selectors/scores_*.csv before re-scoring — the
# score-phase resume would silently skip cells already present in an old file
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
