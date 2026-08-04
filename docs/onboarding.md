# Onboarding — read this first

The 30-minute path into this repo. README covers setup; CONTRIBUTING covers conventions;
this covers **how the system thinks** — the concepts, the trust rules, and the sharp
edges that otherwise live only in commit history and people's heads.

---

## 1. What the project measures, in one diagram

**Question:** does an LLM know, from a survey question's wording alone, which respondent
characteristics predict the answer — and does that knowledge adapt across countries?

```
                    THE MODEL'S SIDE (per selector LLM)
  question text ──▶ free-text essay ──▶ typed feature list ──▶ survey variable codes
                    (gen)               (extract, FIXED model)  (retrieve + disambig,
                                                                 FIXED models)
                    THE GROUND-TRUTH SIDE (once per cell)
  survey data  ──▶ AutoGluon permutation importance = "the oracle"

                    THE COMPARISON (per cell × selector)
  captured importance = model's oracle-mass / oracle top-k mass
  + XGBoost predictive scoring: model picks vs random-k vs textbook demographics
```

A **cell** = (survey, target question, country), e.g. `stfgov × Austria`. The grid is
the set of cells that survived the leakage/degeneracy screen (`layout.genuine_cells()`).

Three model roles, deliberately separated so no model grades its own homework:
**selector** (the LLM under test), **extractor** (fixed: turns essays into feature
lists), **disambiguator** (fixed: picks the matching survey variable).

## 2. Vocabulary

| term | meaning |
|---|---|
| oracle | AutoGluon permutation importance on the real survey data — the gold standard a selector is compared against |
| honest split | oracle fits on 60% (T), *ranks* features on 20% (V1), *values* them on a disjoint 20% (V2). Removes the winner's curse in "oracle top-k". Explained in full at the top of `oracle.py` |
| `oracle_ceiling@k` | how much of the achievable top-k mass the oracle itself captures when it can't cheat (V1-chosen, V2-valued). Report models against this, not against 1.0 |
| captured importance | Σ oracle importance of the model's picks ÷ Σ oracle top-k (matched k). The primary selection metric |
| value_over_random / _textbook | model's predictive score minus a matched-k random draw / minus ten fixed textbook demographics. Textbook is the harder, headline contrast |
| k | the number of variables the model's requests mapped to. **Model-chosen k is part of the measurand** — never clamp it; fixed-k rows are diagnostics (they truncate in arrival order, since the model doesn't rank its requests) |
| genuine / degenerate / leakage | the screen's verdict per cell: real signal / oracle can't beat the marginal / target predictable via an artifact (label proxy or routed module) |
| arms A/B/C | pilot elicitation formats. C = free text = **the** instrument. A/B are JSON-era, kept for the paper's appendix only |
| target type | binary / nominal / ordinal / continuous, detected from value labels. Drives how the oracle and evaluator model the target (classification+log-loss vs regression+Spearman) |

## 3. Where truth lives — the three eras

Every published number depends on the oracle, and the oracle's rules have changed twice.
The **contract version** in each cell's `oracle_meta.json` records which rules built it:

| contract | rules | cache |
|---|---|---|
| v1 | accuracy metric, single 80/20 split, everything multiclass | `outputs/cache/cells_accuracy_v1/` (archived — provenance of all pre-2026-08 numbers) |
| v2 | log loss, honest 60/20/20 split, still everything multiclass | `outputs/cache/cells_logloss_multiclass_v2/` (archived) |
| **v3** | + measurement-level aware (ordinal→regression+Spearman), out-of-scale sentinels excluded | `outputs/cache/cells/` (**current**) |

The constant lives in `src/survey_features/oracle.py` (`ORACLE_CONTRACT_VERSION`);
**this table is the version log** — the code points here.
**Rule: any change that alters what oracle outputs *mean* must bump the version** (which
auto-invalidates every cached cell) **and add a row to this table**; a pure refactor
must not.

Why this exists: a resume check that only asked "does the file exist?" once reported
0 of 89 stale cells needing recomputation. Existence is not validity.

Before quoting any number, read `docs/experiments_index.md` → *"Which numbers are
current"*. Several memos in `docs/` carry **SUPERSEDED** banners — they are history,
not results. The full audit that produced eras 2–3 is `docs/pipeline_audit_2026-08.md`.

## 4. Cache identity — the general discipline

Three mechanisms, one idea (a cached artifact must carry the identity of the process
that made it):

- **`contract_version`** in `oracle_meta.json` — see above.
- **`_fingerprint`** in each cell's `baselines.json` — hash of the feature pool, the
  textbook set, and the draw count. A stale baseline once silently served
  `textbook_acc: null` for a day after the textbook baseline was added.
- **schema guard** in `score_cell.run_score_jobs` — a scores CSV whose header doesn't
  match the current `SCORE_COLS` is archived aside, never appended to.

If you add a cache, give it an identity field. `CONTRIBUTING.md` rule 1 applies.

## 5. Sharp edges (each learned the expensive way)

1. **AutoGluon's `time_limit` is wall-clock and the preset spends all of it.** Cell cost
   is set by the budget, not cell size (`corr(n_features, seconds) = 0.16`). Corollaries:
   *never sleep the laptop mid-fit* (the clock keeps running), and thread-level
   concurrency is useless-to-fatal.
2. **Never run concurrent AutoGluon fits in one process.** Threads starved fits at one
   setting and wedged the interpreter (LightGBM `bad array new length`) at another. Use
   `scripts/rerun_oracles.py --processes N` (process isolation, `oracle_pool.py`).
3. **`quick` runtime mode degrades the ranking** (ceiling@10 0.33 vs 0.68 on the same
   cell). Don't use it for anything that will be quoted.
4. **DK/refused are answers; "not asked" is not.** Respondent non-response is kept as
   real categories everywhere except as the *target of a rank model* (no position on the
   scale, coded 97 on 0–10). Structural missingness → NaN. `surveys.py` taxonomy;
   review CSVs in `outputs/cache/audits/` when adding a survey.
5. **Target measurement level is detected, and detection is fallible.** Run
   `scripts/audit_target_types.py` and *read* `target_types.csv` before trusting a new
   grid; hand overrides live in `surveys.TARGET_TYPE_OVERRIDES`.
6. **The disambiguator is not deterministic at temperature 0.** Same input has produced
   different mappings across runs. Cached maps are the reproducibility unit, not reruns.
7. **Windows quirks are load-bearing:** Ray disabled (`oracle.py` fit kwargs), spawned
   workers need `_ensure_src_on_pythonpath`, libomp conflict notes in README.
8. **PyPI may be blocked on institutional networks** — the working interpreter is the
   conda base (`miniconda3/python`), not `.venv`.
9. **Baselines are model-independent and cached per (cell, k)** — a 16-model zoo pays
   for them once. Don't restructure scoring in a way that breaks that sharing.

## 6. Running things

```bash
# the pipeline (per selector; each phase resumable, cached per cell)
python scripts/run_main.py --phase gen|extract|map|score --selector kimi

# the oracle (process pool = the safe concurrency; ~5x wall speedup at 3-5 workers)
python scripts/rerun_oracles.py --processes 3

# the grid definition (rerun after ANY oracle change)
python scripts/leakage_audit.py --with-data

# audits worth reading, not just running
python scripts/audit_target_types.py
python scripts/audit_missing_codes.py
python scripts/build_textbook_baseline.py --show
```

End every workday with the `/repo-audit` skill (agent-run audit + cleanup + grouped
commits, then your manual walk-through; see CONTRIBUTING → Workflow).

Order matters: oracle → leakage screen → scoring. Scoring against a half-rebuilt oracle
mixes eras.

## 7. Suggested first week

1. Read this file, then `README.md`, then `docs/pipeline_audit_2026-08.md` (skim the
   section heads; deep-read §A1 and the honest-split part).
2. Read the top-of-file docstrings of `oracle.py`, `surveys.py`, `score_cell.py`,
   `metrics.py` — they carry the design rationale.
3. Open one cell directory under `outputs/cache/cells/` and match every file to the
   diagram in §1.
4. Run `pytest tests/` (fast, no data needed) and one `--limit 1` scoring pass.
5. Before touching anything cached, reread §4.

*A doc can map complexity, not shrink it. The real onboarding win comes after the
confirmatory run locks: deleting the legacy JSON arms, the dual-resolve layout
fallbacks, and the era-1/2 generators. Until the paper's appendix is frozen they must
stay (CONTRIBUTING rule 7).*
