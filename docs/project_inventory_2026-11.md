# Project inventory — where everything stands (6 Nov 2026)

**One document that points to everything:** the question, the pipeline, every run we
have done, where its results live, what robustness checks exist, which decisions are
locked, what is still open, and how the project sits against the published
literature. It does not replace the detailed records it links to; it is the map to
them. Plain-language definitions of project terms are in [`glossary.md`](glossary.md);
each term is also glossed here at first use.

> **Scope of this snapshot.** Compiled from the repo (`main` = `c40b803`,
> 2026-09-06) and its git history. The repo's written record **stops on
> 2026-09-06**, in the middle of the oracle computation. Anything done on the
> Dropbox outputs tree after that date (finished oracles, scores, new runs) is not
> visible from the repo and is marked **"status after 2026-09-06 unknown"** below —
> the owner should fill those cells in. Artifact paths are relative to the shared
> Dropbox outputs root (`SURVEY_FEATURES_OUTPUTS`, resolved by
> [`src/survey_features/layout.py`](../src/survey_features/layout.py)); they were
> taken from the docs, not checked on disk.

---

## 1. The project in one paragraph

**Question:** do large language models know, from the wording of a survey question
alone, which characteristics of a respondent predict how they answer it — and does
that knowledge adapt when told which country the respondents are in? We ask an LLM
(the **selector** — the model under test) to write a free-text essay on what it would
need to know to predict the answer; a fixed second LLM turns the essay into a typed
list of features; a retrieval step plus a fixed third LLM map those features onto the
survey's actual variables. We then score the picks against the **oracle** — a
data-derived ranking of which variables really predict the answer in that survey and
country, built from AutoGluon models and permutation importance — and against two
comparators: a random set of the same size and a fixed **textbook** list of ten
standard demographics. The unit of analysis is a **cell** = one (survey, question,
country).

**Measurand note (for positioning).** We do *not* ask the model to impersonate
respondents or predict answer distributions ("silicon sampling"). We ask what it
knows about the *predictive structure* — which covariates matter — which is a
different and less-studied capability (see §9).

**Registered hypotheses** (grid memo, [`grid_design_memo_2026-08.md`](grid_design_memo_2026-08.md) "To pre-register"):

| | Claim | Endpoint |
|---|---|---|
| Primary 1 (value) | Selector picks predict better than the textbook demographics | mean value-over-textbook at model-chosen k, country-blind prompt, pooled selectors, clustered by question; reported per answer-type family; smallest effect of interest +0.05 log loss |
| Primary 1 heterogeneity | Report the *spread* across questions, not just the mean | share of questions > 0, between-question SD, per-type strata |
| Primary 2 (transportability) | A selection made for country A predicts better in A than a selection made for country B | mean own-vs-swapped gain, high-heterogeneity bin, country-named condition; low bin = negative control |
| Secondary | dose-response of swap gain on heterogeneity; country-named vs country-blind; capability curve across selectors; breadth (model-k) vs precision (k = 5); behavioural adaptation battery; pooled value of the country variable | see memo |

---

## 2. Status at a glance

"Phase A/B/C" = the pre-paper sequence: **A** a small pilot to shake down the
machinery and measure noise, **B** power analysis and grid sizing from that noise,
**C** the **confirmatory run** — the locked, run-once measurement whose numbers go in
the paper (as opposed to exploratory *experiments*).

| Step | State (as of repo, 2026-09-06) | Evidence | Next action |
|---|---|---|---|
| Pipeline code (free-text "Arm C" — the historical name of the free-text pipeline; it *is* the pipeline now) | **done**, minimal core on `main` | [`README.md`](../README.md), [`src/survey_features/__init__.py`](../src/survey_features/__init__.py) | — |
| Oracle contract v4 (the current recipe for building oracles; v3 is retired) | **locked** 2026-08-19 | [`onboarding.md`](onboarding.md) §3 | — |
| Phase A pilot | **done** 2026-08-18 | [`pilot_report_2026-08.md`](pilot_report_2026-08.md) | — |
| Phase B sizing | **done** 2026-08-18/19 | [`grid_design_memo_2026-08.md`](grid_design_memo_2026-08.md) | — |
| Target frame (pool of questions eligible to be drawn) | **frozen** 2026-08-19, re-frozen 2026-08-31 after Asian Barometer fixes | [`frame_freeze_review_2026-08.md`](frame_freeze_review_2026-08.md), [`data/frame_rules.yaml`](../data/frame_rules.yaml) | — |
| Confirmatory grid draw | **registered** 2026-08-31 (seed 20260831): 120 questions, 360 confirmatory cells, 1,103 oracle cells, 132 countries | [`data/confirmatory_grid.yaml`](../data/confirmatory_grid.yaml), [`data/confirmatory_grid_cells.csv`](../data/confirmatory_grid_cells.csv) | — |
| Prompt lock (prompt-sensitivity v2) | **done** 2026-08-28 — keep default wording | [`prompt_sensitivity_v2.html`](prompt_sensitivity_v2.html) (registry entry not yet updated — §8) | update registry |
| Selector zoo (the six LLMs under test) | **locked** 2026-09-03; Kimi-K3 cost probe passed | registry `confirmatory-zoo` | — |
| Gen / extract / map for the six | **done** per the 2026-09-05 runbook ("the LLM side … is already done for the six selectors") | [`oracle_handoff_2026-09.md`](oracle_handoff_2026-09.md) §1 | confirm on disk |
| v4 oracles for the grid | **in flight** 2026-09-05 → 08, two laptops split by survey | registry `confirmatory-oracle-map` | **status after 2026-09-06 unknown** |
| Provenance census → leakage audit → textbook baseline → score | **not started** in the repo record | onboarding §6 order | run, in that order |
| Heterogeneity bins, transportability extension, balanced re-fit, swap scoring | **not started** | grid memo Q2/Q4 | after scoring; gated on Primary 1 ≠ 0 |
| `pooled-country-value` experiment | **design** | registry | after sweep |

---

## 3. Pipeline description

```
 DATA SIDE (once per cell)                         MODEL SIDE (per selector)
 survey microdata ─▶ clean/load ─▶ feature pool    question text (+ country or not)
        │                              │                 │ gen: free-text essay
        ▼                              ▼                 ▼
   target frame ─▶ grid draw ─▶ ORACLE (AutoGluon    extract (fixed Qwen) ─▶ typed features
                                 perm. importance,        │
                                 contract v4)             ▼ retrieve (MiniLM) + disambiguate
                                       │                    (fixed Nemotron), dual-layer
                                       ▼                    ▼
                               grid screen (leakage)   mapped survey codes (`expanded_codes`)
                                       └──────────▶ SCORE ◀─────┘
                       captured importance; XGBoost value over random / textbook
```

| # | Stage | What happens | Code | Key settings | Writes to |
|---|---|---|---|---|---|
| 1 | **Load & clean** | Six cross-national surveys: Afrobarometer, Arab Barometer, Asian Barometer, ESS wave 11, Latinobarómetro, WVS. "Don't know"/refused are kept as answers; structurally missing ("not asked") → NaN. Asian Barometer loaded from per-country Stata files (numeric codes); case-twin columns coalesced; other-specify columns dropped; WVS code 410 labelled South Korea. | [`surveys.py`](../src/survey_features/surveys.py) | microdata via `DATA_CONFIG_PATH` (Dropbox `features_project/data/`) | — |
| 2 | **Target frame** | Which questions may be prediction targets: demographics never (they are predictors + textbook baseline), follow-up/skip items out, ESS `test*` out, gate questions kept, 8 attitude items in demographic blocks rescued; each country must pass type-1 estimability. Frozen: 1,278 questions, 1,179 draw-eligible. | [`target_universe_screen.py`](../scripts/target_universe_screen.py), [`frame_rules.yaml`](../data/frame_rules.yaml) | — | `data/_target_universe_inventory.json` (gitignored), [`target_universe.html`](target_universe.html) |
| 3 | **Grid draw** | 20 questions/survey, type-stratified (floor 2 per answer type), 3 spares/survey; **nested country rule** (one seeded country order per question; first ≤10 get oracles, first 3 are the LLM countries — fixed before any oracle is seen, so nobody can pick "interesting" countries). | [`make_confirmatory_grid.py`](../scripts/make_confirmatory_grid.py) | seed 20260831 | `data/confirmatory_grid*.{yaml,csv}` |
| 4 | **Oracle** | Per cell: AutoGluon on all other questions; permutation importance. **Honest split (v4):** 50% ranks features by 5-fold CV, 30% reserved for the downstream evaluator, 20% untouched holdout values the picks — so no number is measured on the rows that produced it. Type-aware: log loss (binary/nominal) or Spearman (ordinal/continuous). Records a `provenance` block (which models each fold finished, host, versions). | [`oracle.py`](../src/survey_features/oracle.py), [`oracle_pool.py`](../src/survey_features/oracle_pool.py), [`rerun_oracles.py`](../scripts/rerun_oracles.py) | `--runtime-mode quick` (`medium_quality`), `--autogluon-time-limit 600`, 5 folds × 5 shuffles, FASTAI excluded; 11-model bag (classification) / 9 (regression) | `cache/cells/<target>_<country>/{oracle.csv, oracle_meta.json, feature_pool.csv}` |
| 5 | **Grid screen** ("leakage audit") | Classifies each cell: **genuine** (keep), **unestimable** (too little data — thin minority on the valuation holdout, or `oracle_ceiling@5 < 0.30`), **leakage** (a variable is the target in disguise: relative or absolute near-duplicate, or a skip-pattern module). Tiny accuracy lift over the mode is *not* a drop reason. | [`leakage_audit.py`](../scripts/leakage_audit.py), [`grid_screen.py`](../src/survey_features/grid_screen.py) | **always** `--with-data --cells-csv data/confirmatory_grid_cells.csv` | `cache/audits/leakage_audit.csv`, `leakage_audit_summary.json`; per-cell `audit_probe.json` |
| 6 | **Textbook baseline** | Ten pre-registered constructs (age, gender, education, income, employment, urban/rural, religiosity, marital status, left-right, ethnicity/language) resolved per survey through the *same* mapping chain as model picks; target excluded from its own baseline. | [`build_textbook_baseline.py`](../scripts/build_textbook_baseline.py), `config.TEXTBOOK_CONSTRUCTS` | — | `cache/baselines/textbook__<survey>.json` |
| 7 | **Generate** | Selector writes an essay. Two **conditions**: *country_provided* (country named) and *unprompted* (country-blind). Country-blind work is computed once per question and reused across its countries. Default prompt pack: system "You are a social science researcher." + "respondent". Reasoning left at each model's default ("as deployed"). | [`elicitation.py`](../src/survey_features/elicitation.py), [`prompts.py`](../src/survey_features/prompts.py), [`run_main.py`](../scripts/run_main.py) `--phase gen` | temperature 0, `GEN_MAX_TOKENS = 8192` | `selectors/<selector>/freetext/` |
| 8 | **Extract** | Fixed extractor `Qwen/Qwen3-235B-A22B-Instruct-2507` → typed list (`respondent_attribute`, `temporal_contextual` mapped; `instrument_methodology`, `population_statistic` counted only). | [`extraction.py`](../src/survey_features/extraction.py) | — | `selectors/<selector>/extracted/` |
| 9 | **Map** | Dual-embed retrieval (`all-MiniLM-L6-v2`, label and "label: context") → fixed disambiguator `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` picks a code; **dual-layer**: bundled sub-items (≥2) are mapped individually and unioned into `expanded_codes` (the headline set). | [`retrieval.py`](../src/survey_features/retrieval.py), [`mapping.py`](../src/survey_features/mapping.py) | `min_similarity 0.30`, `top_n 20` | `selectors/<selector>/maps/` |
| 10 | **Score** | Per cell × condition × k-spec: **captured importance** (share of oracle top-k mass the picks cover, matched k) vs `oracle_ceiling@k`; XGBoost on the evaluator rows → **value over random (VoR)** and **value over textbook (VoT)**, type-matched (log loss or Spearman, never pooled). Primary k = **model-chosen k** (the number of variables the model's answer mapped to — part of the measurand, never clamped). | [`score_cell.py`](../src/survey_features/score_cell.py), [`evaluation.py`](../src/survey_features/evaluation.py), [`metrics.py`](../src/survey_features/metrics.py) | schema guard archives stale CSVs | `selectors/scores_<selector>.csv` |
| 11 | **Transportability** | Country-swap: score country A's picks on country B's rows vs B's own picks (country-named condition). Heterogeneity screen = within-country reliability − between-country agreement of oracle rankings. | [`score_country_swap.py`](../scripts/score_country_swap.py), [`oracle_heterogeneity_screen.py`](../scripts/oracle_heterogeneity_screen.py) | swap pairs chosen by a registered max-disagreement rule (rule text still to be added to the registry) | swap CSV (see §8 path bug); `analysis/oracle_heterogeneity.csv` |

**Fixed model roles:** selector (varies), extractor (Qwen, fixed), disambiguator
(Nemotron Nano, fixed). Never mixed. **Order of operations:** oracle → provenance
census → leakage audit → textbook → gen/extract/map → score ([`onboarding.md`](onboarding.md) §6).

**The confirmatory six** (locked 2026-09-03, registry `confirmatory-zoo`): Kimi-K3,
Kimi-K2.6, GLM-5.1, Nemotron-3-Ultra-550B, DeepSeek-V4-Pro, Nemotron-3-Super-120B —
two within-family capability ladders (Kimi, Nemotron), four labs, ~$44 total at
2026-09-03 prices. (The grid memo's earlier plan of nine selectors was cut to six;
DeepSeek-V4-Flash dropped because its failures correlate with condition.)

---

## 4. Where results live

All under the Dropbox outputs root. Bucket rules: [`CONTRIBUTING.md`](../CONTRIBUTING.md);
bucket meanings: [`project_layout.md`](project_layout.md).

| What | Path | Produced by | Trust status |
|---|---|---|---|
| Oracles (current) | `cache/cells/<target>_<country>/` | `rerun_oracles.py` | trust only `contract_version: 4` with full bag (`provenance.folds[*].n_models`) |
| Archived oracles | `cache/cells_v3/`, `cache/cells_quick60_archive_2026-09-04/`, `cache/cells_asian_labels_archive_2026-09-05/`; eras 1–2 in `features_project_snapshots/` (out of tree) | — | **do not cite** |
| Grid screen | `cache/audits/leakage_audit.csv`, `leakage_audit_summary.json` | `leakage_audit.py` | the copy written 2026-09-06 before the `--cells-csv` fix screened 149 of 1,103 cells — regenerate |
| Textbook baselines | `cache/baselines/textbook__<survey>.json` | `build_textbook_baseline.py` | rebuild after the grid's embeddings refresh |
| Embedding caches | `cache/embeddings/` | pipeline | Afrobarometer embeddings must be regenerated (pre-fix ones contain removed OTHER variables) |
| Confirmatory gen/extract/maps | `selectors/<selector>/{freetext,extracted,maps}/` | `run_main.py` | six selectors done per runbook; pre-2026-09-03 directories need `--force` to collapse country-blind essays |
| Confirmatory scores | `selectors/scores_<selector>.csv` | `run_main.py --phase score` | existing files are **legacy era-3** — archive before re-scoring (resume would skip cells) |
| Phase A pilot | `selectors/runs/pilot_phase_a/`; cells [`data/pilot_cells.csv`](../data/pilot_cells.csv) | `run_main.py --run-tag pilot_phase_a` | design input only; never quote as a result |
| Kimi-K3 probe | 18 cells in `selectors/kimi_k3/`; cells [`data/probe_cells.csv`](../data/probe_cells.csv) | `run_main.py` | resume skips them |
| Prompt-sensitivity v2 | `experiments/prompt_sensitivity_v2/<selector>/<pack>/…`; digests `experiments/_analysis/ps_v2_digest.md`, `ps_v2_score_cells.csv` | `run_prompt_sensitivity_v2.py`, `analyze_prompt_sensitivity_v2.py` | complete (v4 oracles); report [`prompt_sensitivity_v2.html`](prompt_sensitivity_v2.html) |
| Prompt-sensitivity v1 | `experiments/prompt_sensitivity/` | `run_prompt_sensitivity.py` | superseded; v3 oracles — do not cite |
| Role swap / extract swap | `experiments/pipeline_role_swap/{minimax_flash,minimax_nemotron}/` | `run_pipeline_role_swap.py` | complete; v3 oracles (directional only) |
| Stack-experiment digests | `experiments/_analysis/registry_contrast_blocks.md` | `analyze_stack_experiments.py` | — |
| Embedding / ensemble / sub-item / extract-type pilots | `experiments/{embedding_sensitivity,ensemble_mapping,subitem_mapping,extract_type_pilot,extract_type_pilot_v2}/` | scripts on pre-rewrite history (§10) | pre-v4; decisions only |
| Oracle smoke test | `analysis/oracle_smoke_2026-09-04/DESKTOP-IL78MBC/` | `rerun_oracles.py` | reference for the two-machine comparison |
| Heterogeneity screen | `analysis/oracle_heterogeneity.csv` | `oracle_heterogeneity_screen.py` | era-3 version exists; recompute on v4 |
| Cost / latency logs | `logs/token_usage_*.jsonl`, `logs/timing_*.jsonl`, `logs/oracle_run_<host>_<date>.log` | every run | — |
| JSON-era preliminary grid | `outputs/grid/`, `sensitivity/` | archived `run_grid.py` | superseded; scheduled for `.trash` |
| Paper workspace | `paper/` (untracked): `current_state.tex/.pdf`, `figures/`, `generated_current_state/` | humans + generators | — |

---

## 5. Every run so far

Chronological. "Citable?" = may a number from this run appear in the paper as a
result (as opposed to a design input or a methods paragraph).

| When | Run | Registry slug | Size | Oracle era | Headline outcome | Citable? |
|---|---|---|---|---|---|---|
| 2025 → early 2026 | Strict-JSON preliminary grid (arms A/B) | `prelim-json-grid` | various | v1/v2 | Free text beat JSON; JSON magnitudes are a floor | no (superseded) |
| 2026 (pre-Aug) | Free-text main, DeepSeek + Kimi | `main-freetext` | 89-cell legacy grid | v1→v3 | Established the instrument | no (pre-v4) |
| 2026 | Embedding swap | `embedding-sensitivity` | — | pre-v4 | Maps move more than scores; MiniLM kept | methods only |
| 2026 | MiniLM ∪ mpnet union retrieval | `ensemble-mapping` | — | pre-v4 | Jaccard lift, small VoR gain; not promoted | methods only |
| to 2026-08-06 | Dual-layer sub-item mapping pilot | `subitem-mapping` | Kimi | pre-v4 | Promoted to default (`949a8cf`); Δ captured ≈ 0 at matched k | methods only |
| 2026-08 | Extract-type taxonomy wording | `extract-type-pilot` | sample | — | Wording kept | no |
| 2026-08-03 | Pipeline audit re-run (log-loss oracle, 89 cells; re-score both selectors) | — ([`pipeline_audit_2026-08.md`](pipeline_audit_2026-08.md)) | 89 → 47 genuine | era 3 (v3) | Captured importance ~2.2× textbook; VoT +0.066 acc / +0.14 log loss (pooled, uncorrected for type) | no (v3) |
| 2026-08-04 | Underscore vs space in labels | `underscore-label-embed` | 10 features | — | No effect worth fixing | no |
| 2026-08-09 → 11 | Prompt sensitivity v1 (system line only) | `prompt-sensitivity` | 12 cells × 2 selectors | v3 | Maps rewrite (Jaccard ≈ 0.35), VoR noisy | no (superseded) |
| 2026-08-09 → 11 | MiniMax extract + Flash disambig | `pipeline-role-swap` | 12 cells | v3 | Rejected | methods only |
| 2026-08-11 | MiniMax extract only | `extract-swap-minimax` | 12 cells | v3 | ≈ tie, dearer; keep Qwen | methods only |
| 2026-08 | Leakage audit refreshes | `leakage-audit` | 89 (v3) | v3 | 71 genuine (provisional) | no |
| 2026-08-18 | **Phase A pilot** | (`pilot_phase_a` run-tag) | 26 cells × 2 conditions × 4 selectors; $1.68 | **v4** | Machinery passes; noise inputs for sizing; Flash fails by condition; type-matched VoT +0.029 (log loss, binary/nominal) / +0.203 (Spearman, ordinal) | **design input only** |
| 2026-08-18/19 | **Phase B sizing** | ([`grid_design_memo_2026-08.md`](grid_design_memo_2026-08.md)) | — | v4 + v3 | 72% of variance between questions → wide-and-shallow grid | design |
| 2026-08-19 → 28 | **Prompt sensitivity v2** (role + referent, replicate floor) | `prompt-sensitivity-v2` | 24 q × 3 countries; 68 scored, 62 with VoR; 4 selectors | **v4** | **Keep default prompt**: alternatives rewrite lists for MiniMax/Hermes only; none beats both default draws on VoR | **yes, as a methods robustness result** |
| 2026-09-03 | Kimi-K3 cost/length probe | `confirmatory-zoo` | 18 cells; $0.95 | — | ~$19 for full grid; 72% reasoning tokens; denser features | methods |
| 2026-09-04 | Oracle smoke test (60 s vs 600 s budget) | `confirmatory-oracle-map` | 2 cells | v4 | 60 s cuts the bag (3–8 models, fold-varying) → use 600 s | methods |
| 2026-09-05 → | **Confirmatory oracle map** | `confirmatory-oracle-map` | 1,103 cells, 2 laptops | **v4** | in flight | — |
| 2026-09 | Confirmatory gen/extract/map, six selectors | `confirmatory-zoo` | 360 cells × 2 conditions (480 units/selector after dedupe) | — | done per runbook; scoring blocked on oracles | pending |

---

## 6. Robustness checks — done, registered, and missing

### 6a. Done

| Check | What it guards against | Result | Where |
|---|---|---|---|
| Oracle era comparison (accuracy → log loss → v4) | an oracle too coarse to see the model's picks | log-loss oracle: share of pool at exactly 0 fell 19% → 0%; top-10 between eras overlaps only 0.18 → old numbers retired | audit §A1 |
| Oracle self-agreement (two honest reads of one cell) | noisy ground truth | v3 top-10 Jaccard 0.33; v4 0.47–0.54 | audit §A1, pilot report |
| `oracle_ceiling@k` denominator | confusing "model missed it" with "nothing stable to find" | median ceiling@10 0.89, tail 0.10; cells < 0.30 at k = 5 dropped | audit §A1, grid screen |
| Fit-budget smoke test, provenance block, census script | the oracle depending on which machine fitted it | 60 s cut the bag; 600 s gives the full bag everywhere | registry `confirmatory-oracle-map`; [`oracle_provenance_census.py`](../scripts/oracle_provenance_census.py) |
| Asian Barometer coding fix | label-text categoricals → slow, truncated, order-blind fits | numeric codes, 2,158 pairs cross-checked | registry `confirmatory-oracle-map` |
| FASTAI excluded from the bag | Dropbox file locks starving the time budget | excluded everywhere | [`pre_paper_run_decisions.md`](pre_paper_run_decisions.md) |
| Leakage screen (relative + absolute near-duplicates, skip modules) | "wins" from a duplicated variable | e.g. SE7a Mongolia caught; gates kept but screened | [`grid_screen.py`](../src/survey_features/grid_screen.py) |
| Matched null from the same pool | random baseline padded with junk columns | fixed | audit §A2 |
| Textbook baseline through the same mapping chain | "beats random" being a low bar; unequal mapping attenuation | in place; target excluded from own baseline (7 contaminated cells found) | audit §A2, "First era-3 results" |
| Type-matched valuation | ordinal answers scored as unordered classes | decided 2026-08-18; families never pooled | pilot report addendum |
| Variance decomposition is not a scale artifact | between-question share driven by units | 4 cuts all ≈ 70–90% between-question | grid memo "Robustness" |
| Country-blind arm computed once per question | hidden generation draw confounded with country | fixed in `dedupe.py` | pre-paper decisions 2026-09-03 |
| Prompt wording (v1, v2) with replicate floor and temperature sidecar | results being an artefact of "social science researcher" + "respondent" | wording rewrites lists for two models but never improves VoR → default kept | §5 |
| Extractor swap; disambiguator swap | dependence on the fixed helper models | Qwen and Nemotron kept | registry |
| Embedder swap; union retrieval; label tokenization | mapping artefacts | maps move, scores barely | registry |
| Deterministic tie-breaking and seeding | irreproducible percentiles / baselines | fixed (`rankdata`, `stable_seed`) | audit §A3 |

### 6b. Registered, not yet run

| Check | Why | Source |
|---|---|---|
| Nemotron family affinity: re-map a sample with `--disambiguator qwen235b` | both Nemotron selectors share a family with the disambiguator | registry `confirmatory-zoo` |
| Low-bin reliability gate for the transportability negative control | "no adaptation" manufactured by oracle noise | grid memo Q2 |
| Balanced-tier oracle re-fit on the 30 transportability questions | heterogeneity measured with error shrinks the dose-response slope | grid memo Q4 |
| `pooled-country-value` | is asking for the respondent's country rational? | registry |
| Dual-extractor audit (~10% of cells) + extraction yield per model | capability measured through Qwen's reading | audit §A9 |
| Per-model reasoning-token share reported | "as deployed" comparison includes inference-time compute | registry `prompt-sensitivity-v2` Compute |

### 6c. Argued but not built (decide before the analysis freeze)

| Check | Why it matters | Source |
|---|---|---|
| **Grouped / construct-level importance** (permute clusters of near-synonymous items jointly) | permutation importance splits credit across correlated items, capping correct broad answers; also the most-cited reviewer objection to permutation importance (§9) | audit §A8 |
| Cluster-bootstrap CIs, medians first | means are skewed by a few cells | audit "Outstanding", "First era-3 results" |
| Multiplicity hierarchy (model-k primary) | 6 models × types × conditions × k-specs × metrics | audit §A9 |
| Country-permutation null for transportability (own country vs every other oracle) | own-minus-cross on a ratio is low-powered | audit §A9 |
| `similarity-threshold` / `top_n` sweep | retrieval, not disambiguation, binds mapping recall | registry, audit §B |
| Elicited ranking for fixed-k analyses | k = 5 currently truncates in essay order | audit §A6 |

---

## 7. Locked decisions (with dates)

| Date | Decision | Source |
|---|---|---|
| 2026-08-06 | Dual-layer mapping is the headline map | registry `subitem-mapping` |
| 2026-08-11 | Keep Qwen extractor, Nemotron disambiguator | [`pre_paper_run_decisions.md`](pre_paper_run_decisions.md) |
| 2026-08-16 | Grid screen drops only unestimable + leakage, not low accuracy lift | same |
| 2026-08-18 | Type-matched valuation (log loss / Spearman, never pooled) | [`pilot_report_2026-08.md`](pilot_report_2026-08.md) |
| 2026-08-19 | Oracle contract v4 only; demographics never targets; gates kept; 8 items rescued; ESS `test*` out | onboarding §3; frame review |
| 2026-08-19 | Wide-and-shallow grid; Flash out; quick oracles + balanced pass on 30 transportability questions | grid memo |
| 2026-08-28 | Keep default prompt pack | [`prompt_sensitivity_v2.html`](prompt_sensitivity_v2.html) |
| 2026-08-31 | 20 questions/survey, nested country rule, isolated seeding; Asian case-twin + other-specify cleaning; grid registered | pre-paper decisions |
| 2026-09-03 | Confirmatory six; reasoning on ("as deployed"); `GEN_MAX_TOKENS 8192`; country-blind dedupe | registry, pre-paper decisions |
| 2026-09-04/05 | Oracle budget 600 s; Asian Barometer from Stata files; two-laptop split | registry `confirmatory-oracle-map` |
| 2026-09-06 | Leakage audit takes its universe from the grid CSV | registry |

---

## 8. Inconsistencies and housekeeping found while compiling this

1. **Registry is stale for `prompt-sensitivity-v2`.** Index and entry say `design` and
   Result `—`, but the 2026-08-28 report gives a decision (keep default; 72 → 68 → 62
   cells). The commit SHA that produced the artifacts is also unrecorded. → Fill the
   Result, Contrast detail and Commit fields from `ps_v2_digest.md`.
2. **Registry is stale for `confirmatory-zoo`.** It says "sweep not yet run"; the
   2026-09-05 runbook says gen/extract/map are done for all six. → Record what ran and
   at which SHA.
3. **`oracle-v3` entry contradicts itself:** status says superseded, Result still says
   "Era-3 is current". `main-freetext` Inputs still list v3 oracles. → Edit.
4. **`scripts/score_country_swap.py:130` writes to `OUTPUTS_DIR / "main" / "runs"`** —
   the folder renamed to `selectors/` on 2026-08-19, and a hard-coded path that
   CONTRIBUTING forbids. → Add a `layout.py` helper and point the script at
   `selectors/runs/<tag>/` before swap scoring.
5. **Registry points to design docs "on `main`"** (`main_experiment_design.md`,
   `embedding_sensitivity.md`, `ensemble_mapping.md`, `similarity_threshold.md`,
   `pipeline_critique.md`, `analysis/*.py`) that were deleted in `0c0ef80`
   (2026-08-06). Recover with `git show 0c0ef80^:docs/<file>`.
6. **Glossary "confirmatory" and "Phase A/B/C" entries** still say the grid is
   provisional and Phase C is "pending the v4 re-run" (true on 2026-08-19, now
   outdated in detail: grid is registered at 360/1,103 cells, not 270).
7. **`docs/main_run_tradeoffs.html`** is the 2026-08-19 cost calculator; its zoo and
   grid are superseded by the grid memo and the 2026-09-03 lock. → Mark superseded.
8. **Transportability pair rule** ("registered max-disagreement rule") is referenced
   but its text is not yet in the registry. Register it before oracle heterogeneity is
   inspected on the v4 map.

---

## 9. Related work and novelty check

*Literature search run 2026-11-06 in three parallel sweeps: LLM feature selection;
LLMs in survey research; importance methodology and LLM elicitation.*

**Verification caveat.** The searches could reach search-engine result pages but
not the full texts (arxiv.org, nature.com, tandfonline.com, link.springer.com and
the Semantic Scholar API were blocked by the session's network policy). Every entry
below has a URL that appeared in a search result; summaries come from abstracts and
snippets. **Read the "must read" papers in full before citing their numbers.**
Entries marked *(snippet)* were seen only once and need confirming; *unverified*
marks a venue or author list not confirmed.

### 9.1 Verdict: are we repeating someone else's study?

**No, but two pieces of our framing are no longer new and must not be claimed.**

1. **"An LLM can pick predictive features from variable names alone" is established**
   (LMPriors 2022; LLM-Select, TMLR 2025; a 2025 KDD Explorations survey; a
   systematic Oct-2026 study finding classical selectors still usually win). Do not
   headline "LLMs can select features" or "LLMs match data-driven selection".
2. **"An LLM picks survey predictors from variable descriptions" has one direct
   precedent**: a March-2026 medRxiv preprint on adolescent vaping in the US PATH
   survey (one outcome, one country, downstream AUC; no importance oracle or
   cross-country test seen). Cite it prominently; never claim "first".
3. **"LLM social-science knowledge is worse outside WEIRD countries" is anticipated**
   (WEIRD = Western, educated, industrialised, rich, democratic) — by
   Pataranutaporn et al. (WVS well-being, 64 countries, PNAS 2025), a Nature Human
   Behaviour 2026 audit of economic preferences across 74 countries, and the
   cultural-alignment literature. All of these treat the LLM as a *predictor* or a
   *simulated respondent*, not as a *selector*. Our country result is convergent
   evidence under a new measurand, not a discovery.

**What nobody appears to have done** — the defensible contribution:
- elicit, from question wording, *which respondent characteristics predict the
  answer*, for **many targets (120) across six cross-national survey programmes**
  (no LLM benchmark against Afro-, Arab-, Asian Barometer or Latinobarómetro was found);
- score the picks against a **cross-validated, holdout-honest importance oracle**
  with captured mass and a ceiling, plus matched-k random and textbook baselines,
  a leakage screen and type-matched scoring;
- test **country adaptation** (country-named vs country-blind) and **country-swap
  transfer** with a heterogeneity dose-response and a reliability-gated negative
  control. The closest is synthetic covariate shift on generic tables (LoID 2026).
- capability ladders within model families and a replicate-floor prompt study.

**Suggested positioning sentence.** *Silicon sampling asks a model to be the
respondents and recovers structure afterwards — a route shown to distort
coefficients (Bisbee et al. 2024) and to be dominated by answer-format artefacts
(Dominguez-Olmedo et al. 2024). We ask the model to state the structure directly,
as an expert would, and grade that statement against the data, across 132
countries.* Also say early that our "transportability" is **predictive transfer**,
not causal transportability in the Pearl–Bareinboim sense — or rename the contrast
"cross-country transfer".

### 9.2 Must-read papers (closest to us), with how we differ

| Paper | What it did | How we differ / what to do |
|---|---|---|
| Jeong, Lipton, Ravikumar, **LLM-Select**, TMLR 2025 — [arXiv 2407.02694](https://arxiv.org/abs/2407.02694) | LLM scores/ranks features from names + task; rivals LASSO; Kendall τ with SHAP/permutation importance rises with scale but matches no single importance notion | Method ancestor. We: survey wording, 120 targets, honest oracle, countries. **Report Kendall τ vs the oracle as a bridge metric**; optionally their LLM-Score/Rank/Seq as ablations |
| **AI-driven feature selection using only survey variable descriptions** (adolescent vaping, PATH), medRxiv Mar 2026 — [link](https://www.medrxiv.org/content/10.64898/2026.03.06.26347816v1.full) (authors unverified: the two sweeps reported different author lists) | Several open LLMs pick from 1,396 items; LightGBM on picks ≈ or > full set | **Closest trap.** One outcome, one country, no oracle. Cite and differentiate |
| Pataranutaporn, Powdthavee, Archiwaranguprok, Maes, **LLMs predict human well-being — but not equally everywhere** — [arXiv 2507.06141](https://arxiv.org/abs/2507.06141); PNAS version [PMC12685116](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12685116/) | WVS, 64 countries; models know broad correlates, mis-weight predictors in poorer / under-represented countries; country in prompt does not fix it | Our cross-country hypothesis for one outcome. **Stratify results by a representation proxy** (GDP, web-language share) |
| **Simulated responses from LLMs are weak proxies for human economic preferences across 74 countries**, Nature Human Behaviour 2026 — [link](https://www.nature.com/articles/s41562-026-02611-4) (authors unverified) | Demographic gradients recovered; country-level structure poorly | Predicts that textbook demographics will be hard to beat and country adaptation weak |
| Memar et al., **When do LLMs help with feature selection?**, Oct 2026 — [arXiv 2610.28556](https://arxiv.org/abs/2610.28556v1) *(snippet)* | 18 datasets; classical selectors usually beat small LLM selectors; meaningful names help most | **Borrow the informative vs anonymised vs shuffled names ablation** |
| Ma & Powell, **Can LLMs predict associations among human attitudes?**, 2025 — [arXiv 2503.21011](https://arxiv.org/abs/2503.21011); Hommel & Arslan, AMPPS 2025 — [link](https://journals.sagepub.com/doi/10.1177/25152459251377093) | Item–item correlations inferred from wording (r ≈ .7–.9); overconfident; contamination risk | Supports our "from wording alone" premise. **Borrow the semantic-similarity control** (do hits just share words with the target?) |
| Wang, **Silicon sampling answers with country-level assumptions**, ESS, 2026 — [arXiv 2609.16395](https://arxiv.org/pdf/2609.16395) *(snippet)* | Country label acts as a fixed level shift; a neighbour-country average beats the LLMs | **Borrow the neighbour-country baseline** for adaptation; separates "level" from "structure" — fits our `pooled-country-value` design |
| Wedgwood et al., **Personalization, personas, and forecasting in value alignment**, ICML 2026 — [icml.cc](https://icml.cc/virtual/2026/75669) | WVS; forecasting framing beats persona framing; religiosity easy, institutional trust hard | Precedent for our prompt study; **check the same item-domain gradient** |
| Hewitt, Ashokkumar, Ghezae, Willer, **Predicting results of social science experiments using LLMs**, 2024 — [Stanford GSB](https://www.gsb.stanford.edu/faculty-research/publications/large-language-models-can-predict-results-social-science-experiments) | LLM forecasts of 70 survey experiments correlate ~0.85 with results, overestimate magnitudes; unpublished-study control | Model for a **memorisation control** and a human-expert comparison |

### 9.3 Reviewer objections the literature predicts → robustness checks

Mapped to §6. ✔ = we already have it; ◐ = registered/argued; ✚ = new from this review.

| Objection | Literature | Our status / action |
|---|---|---|
| "Your oracle is one model's opinion" (Rashomon / model multiplicity) | Fisher, Rudin, Dominici 2019; Donnelly et al. NeurIPS 2023 (RID); Breznau et al. PNAS 2022 | ✔ fold-spread reliability, ceiling. ✚ report captured mass with an uncertainty band from fold/bootstrap importance; optionally count a hit if inside the oracle's credible top-k |
| "Permutation importance is biased by correlated batteries" | Strobl et al. 2008; Hooker, Mentch, Zhou 2021; Ewald et al. 2024 | ◐ grouped importance (audit §A8). ✚ build it (Au et al. 2022 grouped importance); conditional/LOCO oracle on a subset (Watson & Wright 2021 `cpi`; Lei et al. 2018; Williamson et al. JASA 2023 `vimp`) |
| "Importance in low-signal cells is noise" | Salganik et al. PNAS 2020; Lundberg et al. PNAS 2024; Molnar et al. 2022 | ✔ type-1 screen, ceiling gate. ✚ report holdout predictability per cell and share unestimable |
| "Fixed k is arbitrary" | Williamson et al. 2023 | ✔ model-chosen k primary. ✚ optional: oracle set = items significantly > 0 |
| "Leakage inflates the oracle" | Kaufman, Rosset, Perlich KDD 2011 / TKDD 2012 | ✔ leakage screen. ✚ report what the screen removed per cell |
| "Prompt/format artefact" | Sclar et al. ICLR 2024; Dominguez-Olmedo et al. 2024; arXiv 2508.19563 | ✔ prompt-sensitivity v2 with replicate floor. ✚ variable/option order and paraphrase perturbation on a subset |
| "Temperature 0 isn't deterministic" | Atıl et al. Eval4NLP 2025 | ✔ replicate floor, country-blind dedupe. ✚ cite; report replicate agreement on mapped sets |
| "It's memorised from published WVS/ESS findings" | Hewitt 2024; Kıcıman et al. TMLR 2024; Hommel & Arslan 2025 | ✚ memorisation probe: obscure national items / waves after model cutoffs; correlate hit rate with how often an item is studied |
| "Does it just use word overlap?" | Ma & Powell 2025; Memar 2026 | ✚ semantic-similarity control; anonymised-name ablation |
| "Country effects are level shifts, not structure" | Wang 2026; NHB 2026 | ◐ `pooled-country-value`. ✚ neighbour-country non-LLM baseline |
| "Transportability is a causal term" | Pearl & Bareinboim 2014 | ✚ scope as predictive transfer; cite leave-one-country-out precedent (JEPP 2025, below) |
| "Does the LLM add anything over data-driven selection?" | LLM-Lasso 2025; AutoElicit ICML 2025; arXiv 2610.36806 | ✚ optional small-sample curve: at what n do LLM picks stop beating a data-driven selector |
| "Results depend on the extractor/mapper" | — | ◐ dual-extractor audit; Nemotron-affinity check |
| "Models differ only in WEIRD-ness" | Atari et al. 2023; Tao et al. 2024 | ✚ regress per-country performance on cultural distance from the US |

### 9.4 Annotated bibliography

**A. LLMs as feature selectors / knowledge-driven tabular ML**

| Citation | Link | Rel. | Takeaway |
|---|---|---|---|
| Choi, Cundy, Srivastava, Ermon (2022) LMPriors, NeurIPS FMDM workshop | [2210.12530](https://arxiv.org/abs/2210.12530) | Med | Origin of names-only feature priors; cite |
| Jeong, Lipton, Ravikumar (2025) LLM-Select, TMLR | [2407.02694](https://arxiv.org/abs/2407.02694) | **High** | See 9.2 |
| Li, Tan, Liu (2025) Exploring LLMs for feature selection: a data-centric perspective, SIGKDD Explorations | [2408.12025](https://arxiv.org/abs/2408.12025) | Med | Text-based selection more robust, scales with size; frames us as text-based |
| Zhang, …, Tibshirani (2025) LLM-Lasso | [2502.10648](https://arxiv.org/abs/2502.10648) | Med | LLM scores as Lasso penalties; knowledge+data combination |
| Memar et al. (2026) When do LLMs help with feature selection? | [2610.28556](https://arxiv.org/abs/2610.28556v1) | Med-High | See 9.2 *(snippet)* |
| Li, Xiu (2025) LLM4FS | [2503.24157](https://arxiv.org/abs/2503.24157v2) | Med | Hybrid LLM+classical; a later version reportedly withdrawn — cite with care |
| Yang et al. (2024) ICE-SEARCH | [2402.18609](https://arxiv.org/abs/2402.18609v1) | Low | LLM inside evolutionary search |
| Lee, Shen et al. (2024) FREEFORM (genotype) | [2410.01795](https://arxiv.org/abs/2410.01795v1) | Low | Knowledge selection helps when data are scarce |
| Skinner, Guo, Li (2026) LLM sparsity prior | [2605.23102](https://arxiv.org/abs/2605.23102) | Low | Naive LLM weights can hurt; LLM priors miscalibrated |
| Bal-Ghaoui, Sabri (2025) LLM-FS-Agent | [2510.05935](https://arxiv.org/html/2510.05935v1) | Low | Role-based deliberation; aside for role framing |
| Hollmann, Müller, Hutter (2023) CAAFE, NeurIPS | [2305.03403](https://arxiv.org/abs/2305.03403) | Low | Feature engineering, not selection |
| Han et al. (2024) FeatLLM, ICML | [PMLR](https://proceedings.mlr.press/v235/han24f.html) | Low | Background |
| Abhyankar, Shojaee, Reddy (2025) LLM-FE | [2503.14434](https://arxiv.org/abs/2503.14434) | Low | Background |
| LLM4CKD (2026) | [2609.04013](https://arxiv.org/pdf/2609.04013) | Med | LLM rankings agree with each other, not with supervised importance — **report between-LLM agreement too** *(snippet)* |
| TabSHAP (2026) | [2604.21120](https://arxiv.org/pdf/2604.21120) | Low-Med | LLMs over-weight semantically rich features *(snippet)* |
| Cruz, Hardt, Mendler-Dünner (2024) Evaluating LMs as risk scores (folktexts), NeurIPS | [2407.14614](https://arxiv.org/abs/2407.14614v3) | Low-Med | Census data; LLM-as-predictor PI vs XGBoost |
| LoID: What LLMs know but don't say (2026) | [2601.17609](https://arxiv.org/html/2601.17609v2) | Med | Closest "LLM knowledge under shift" — synthetic shift only |
| Huntington-Klein et al. (2024) Do LLMs act as repositories of causal knowledge? | [2412.10635](https://arxiv.org/abs/2412.10635) | Low | Over-inclusion; option-order sensitivity |

**B. LLMs and surveys / computational social science**

| Citation | Link | Rel. | Takeaway |
|---|---|---|---|
| Argyle et al. (2023) Out of one, many, Political Analysis | [2209.06899](https://arxiv.org/abs/2209.06899) | Low-Med | Silicon sampling; contrast case |
| Bisbee et al. (2024) Synthetic replacements for human survey data?, Political Analysis | [Cambridge](https://www.cambridge.org/core/journals/political-analysis/article/synthetic-replacements-for-human-survey-data-the-perils-of-large-language-models/B92267DC26195C7F36E63EA04A47D2FE) | Med | 48% of coefficients wrong → motivates direct elicitation |
| Dominguez-Olmedo, Hardt, Mendler-Dünner (2024) Questioning the survey responses of LLMs, NeurIPS | [2306.07951](https://arxiv.org/abs/2306.07951) | Low | Answer-format artefacts; we sidestep them |
| Santurkar et al. (2023) Whose opinions do LMs reflect?, ICML | [PMLR](https://proceedings.mlr.press/v202/santurkar23a.html) | Low | Background |
| Durmus et al. (2023/24) GlobalOpinionQA | [2306.16388](https://arxiv.org/abs/2306.16388v2) | Low-Med | Country prompting shifts answers, can stereotype |
| Suh et al. (2025) SubPOP, ACL | [ACL](https://aclanthology.org/2025.acl-long.1028/) | Low | Fine-tuned distribution prediction |
| Cao et al. (2025) Specializing LLMs to simulate survey distributions for global populations, NAACL | [ACL](https://aclanthology.org/2025.naacl-long.162) | Low-Med | WVS fine-tuning |
| Kim & Lee (2023) AI-augmented surveys | [2305.09620](https://arxiv.org/abs/2305.09620v3) | Low-Med | Unasked-question prediction |
| Miranda & Balbi (2025) Simulating public opinion, Entropy | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12468613/) | Med? | ANES; may include an LLM variable-pick step — **check full text** |
| Wightman, Bied, De Bie (2026) People are not just their countries, AIES | [AAAI](https://ojs.aaai.org/index.php/AIES/article/view/43344) | Med | ESS; determinants of *alignment*, not of attitudes |
| Wang (2026) Silicon sampling answers with country-level assumptions | [2609.16395](https://arxiv.org/pdf/2609.16395) | Med | See 9.2 |
| Ku et al. (2026) Silicon sampling via cross-survey transfer (Taiwan) | [2607.03091](https://arxiv.org/abs/2607.03091) | Low | Predictability varies by construct |
| Wedgwood et al. (2026) Personalization, personas, and forecasting, ICML | [icml.cc](https://icml.cc/virtual/2026/75669) | Med | See 9.2 |
| Luo et al. (2026) ACE-Align | [2601.12962](https://arxiv.org/abs/2601.12962) | Med | Aligns per-country attribute effects to survey estimates (trains, does not elicit) |
| "Validating LLM-generated synthetic survey samples" (2026), Eurobarometer | [2611.00256](https://arxiv.org/html/2611.00256v1) | Low-Med | Compares importance rankings, real vs synthetic *(snippet)* |
| Are LLMs chameleons? (ESS); United in diversity? (EP 2024) | [2405.19323](https://arxiv.org/abs/2405.19323), [2409.09045](https://arxiv.org/pdf/2409.09045) | Low | Cross-country silicon sampling |
| Tao et al. (2024) Cultural bias and alignment of LLMs, PNAS Nexus | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39290441/) | Low-Med | Cultural prompting ≈ our country-named arm at response level |
| Atari et al. (2023) Which humans? | [HKS](https://www.hks.harvard.edu/centers/cid/publications/which-humans) | Low | WEIRD gradient; test cultural distance |
| AlKhamissi et al. (2024) Cultural alignment of LLMs, ACL | [ACL](https://aclanthology.org/2024.acl-long.671) | Low | Anthropological prompting |
| Cao et al. (2023) Cross-cultural alignment ChatGPT | [2303.17466](https://ar5iv.org/abs/2303.17466) | Low | Background |
| Abeliuk, Gaete, Bro (2025/26) Fairness in LLM-simulated public opinion, EPJ Data Science | [2501.15351](https://arxiv.org/abs/2501.15351v1) | Low-Med | US vs Chile |
| Hewitt et al. (2024) | see 9.2 | Med | Forecasting experiments |
| Manning, Zhu, Horton (2024) Automated social science, NBER | [w32381](https://www.nber.org/papers/w32381) | Low-Med | Signs right, magnitudes wrong → favours rank metrics |
| Automating social science: LLMs vs experts in variable relationships (Collabra, unverified) | [Sciety](https://sciety.org/articles/activity/10.31234/osf.io/mkxdt_v2) | Med | LLMs ≈ experts on simple relations; both fail on moderation (country is a moderator) |
| Ludwig & Mullainathan (2024) ML as a tool for hypothesis generation, QJE | [NBER](https://www.nber.org/papers/w31017) | Low | Framing |
| Zhou et al. (2024) HypoGeniC | [ACL](https://aclanthology.org/2024.nlp4science-1.10) | Low | Background |

**C. Predictor discovery in survey data, transfer, and importance methodology**

| Citation | Link | Rel. | Takeaway |
|---|---|---|---|
| Salganik et al. (2020) Fragile Families Challenge, PNAS | [link](https://scholarworks.montana.edu/handle/1/16106) | Method | Simple benchmarks hard to beat; mirrors textbook arm |
| Lundberg et al. (2024) Origins of unpredictability, PNAS | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11181083/) | Bkg | Low-signal cells are legitimate |
| Kern, Klausch, Kreuter (2019) Tree-based ML for survey research, SRM | [link](https://ojs.ub.uni-konstanz.de/srm/article/view/7395) | Bkg | Standard ML-in-surveys cite |
| Breznau et al. (2022) Many analysts, PNAS | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9926231/) | Method | Pre-register oracle settings |
| RF populist-vote predictors per country (EVS), Quality & Quantity 2025 | [Springer](https://link.springer.com/article/10.1007/s11135-025-02064-3) | Bkg | ML predictor rankings vary by country *(snippet)* |
| Europeanisation of policy preferences (leave-one-country-out RF), JEPP 2025 | [T&F](https://www.tandfonline.com/doi/full/10.1080/13501763.2025.2512901) | Method | Precedent for cross-country transfer as similarity *(snippet)* |
| Pearl & Bareinboim (2014) External validity / transportability, Stat Sci | [Project Euclid](https://projecteuclid.org/journals/statistical-science/volume-29/issue-4/External-Validity-From-Do-Calculus-to-Transportability-Across-Populations/10.1214/14-STS486.full) | Bkg | Scope our term |
| Cross-country generalisation of tabular ML on DHS (2026) | [2605.26589](https://arxiv.org/pdf/2605.26589) | Bkg | *(snippet)* |
| Fisher, Rudin, Dominici (2019) Model class reliance, JMLR | [1801.01489](https://arxiv.org/abs/1801.01489) | Method | Rashomon objection |
| Donnelly et al. (2023) Rashomon importance distribution, NeurIPS | [2309.13775](https://arxiv.org/abs/2309.13775) | Method | Recipe for importance distributions |
| Hooker, Mentch, Zhou (2021) Unrestricted permutation forces extrapolation, Stat Comput | [1905.03151](https://arxiv.org/abs/1905.03151) | Method | Correlated-feature bias |
| Strobl et al. (2008) Conditional variable importance, BMC Bioinf | [doi](https://doi.org/10.1186/1471-2105-9-307) | Method | Classic cite |
| Molnar et al. (2022) Pitfalls of model-agnostic interpretation | [2007.04131](https://arxiv.org/abs/2007.04131) | Method | Reviewer checklist |
| Ewald et al. (2024) Feature importance for scientific inference | [2404.12862](https://arxiv.org/abs/2404.12862) | Method | State the oracle's estimand |
| Au et al. (2022) Grouped feature importance, DMKD | [2104.11688](https://arxiv.org/abs/2104.11688) | Method | Implements audit §A8 |
| Williamson et al. (2023) Algorithm-agnostic VIM inference, JASA | [2004.03683](https://arxiv.org/abs/2004.03683) | Method | CIs and null test |
| Lei et al. (2018) LOCO, JASA | [1604.04173](https://arxiv.org/abs/1604.04173) | Method | LOCO oracle variant |
| Watson & Wright (2021) Conditional predictive impact, Mach Learn | [1901.09917](https://arxiv.org/abs/1901.09917) | Method | Conditional oracle variant |
| Kaufman, Rosset, Perlich (2011/12) Leakage in data mining | [dblp](https://dblp.org/rec/conf/kdd/KaufmanRP11.html) | Method | Cite for leakage screen |

**D. LLM elicitation as expert prior; reliability of hosted LLMs**

| Citation | Link | Rel. | Takeaway |
|---|---|---|---|
| Capstick, Krishnan, Barnaghi (2025) AutoElicit, ICML | [PMLR](https://proceedings.mlr.press/v267/capstick25a.html) | Bkg | Knowledge-as-prior framing |
| Selby et al. (2024/25) Had enough of experts? / Quantitative knowledge retrieval | [2402.07770](https://arxiv.org/abs/2402.07770v2) | Bkg | Avoid calling the LLM an "expert" |
| LLM knowledge for OOD tabular prediction (2026) | [2610.36806](https://arxiv.org/abs/2610.36806) | Med | Priors help most under shift + small n *(snippet)* |
| Variable names provide priors (2026) | [2611.02637](https://arxiv.org/abs/2611.02637) | Med | *(snippet; could not be re-found — confirm before citing)* |
| Kıcıman et al. (2023/24) Causal reasoning and LLMs, TMLR | [2305.00050](https://arxiv.org/abs/2305.00050) | Bkg | Names-only knowledge; memorisation concern |
| Long et al. (2023) LLMs as imperfect experts | [2307.02390](https://arxiv.org/abs/2307.02390) | Bkg | Knowledge-vs-data framing |
| Jiralerspong et al. (2024) Efficient causal graph discovery with LLMs | [2402.01207](https://arxiv.org/abs/2402.01207) | Bkg | One-variable-at-a-time elicitation |
| Robustness is important: limitations of LLMs for data fitting (2025) | [2508.19563](https://arxiv.org/html/2508.19563v3) | Method | Name/order perturbations |
| Sclar et al. (2024) Sensitivity to spurious prompt features, ICLR | [2310.11324](https://arxiv.org/abs/2310.11324) | Method | Report spreads across templates |
| Atıl et al. (2025) Non-determinism of "deterministic" hosted LLMs, Eval4NLP | [ACL](https://aclanthology.org/2025.eval4nlp-1.12) | Method | Justifies replicate floor |

**Gaps in this search.** No "cited-by" sweep of LLM-Select (Semantic Scholar
blocked); NeurIPS 2026 accepted papers not visible. Re-run both with
`api.semanticscholar.org` and `arxiv.org` allowed before submission.

---

## 10. Document and code map

| Document | Use it for |
|---|---|
| [`README.md`](../README.md) | setup, run commands, outputs tree |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | where a run writes; when to register |
| [`onboarding.md`](onboarding.md) | how the system thinks; oracle eras; sharp edges |
| [`glossary.md`](glossary.md) | project terms in plain words |
| [`project_layout.md`](project_layout.md) | repo / outputs / paper map |
| [`experiments_registry.md`](experiments_registry.md) | every experiment, with results and contrast tables |
| [`pipeline_audit_2026-08.md`](pipeline_audit_2026-08.md) | the oracle and metric audit; open Tier-1 items |
| [`pilot_report_2026-08.md`](pilot_report_2026-08.md) | Phase A |
| [`grid_design_memo_2026-08.md`](grid_design_memo_2026-08.md) | Phase B sizing; pre-registration text |
| [`frame_freeze_review_2026-08.md`](frame_freeze_review_2026-08.md) | target-frame rulings |
| [`pre_paper_run_decisions.md`](pre_paper_run_decisions.md) | stack locks, grid draw, country-blind dedupe, backlog |
| [`oracle_handoff_2026-09.md`](oracle_handoff_2026-09.md) | two-laptop oracle runbook |
| [`prompt_sensitivity_v2.html`](prompt_sensitivity_v2.html) | prompt-lock report |
| [`target_universe.html`](target_universe.html) | frame visualization |
| [`main_run_tradeoffs.html`](main_run_tradeoffs.html) | superseded cost calculator |
| pre-rewrite history | `git show 0c0ef80^:<path>` for deleted designs and analysis scripts |

| Inputs in `data/` | What |
|---|---|
| `confirmatory_grid.yaml`, `confirmatory_grid_cells.csv` | registered grid |
| `frame_rules.yaml` | frozen frame rules |
| `targets.yaml` | old 30-target catalog — **not** the grid |
| `pilot_cells.csv`, `probe_cells.csv` | Phase A pilot, Kimi-K3 probe |
| `prompt_sensitivity_cells.yaml`, `prompt_sensitivity_v2_cells.yaml` | v1 (superseded), v2 grids |
| `nebius_pricing.json` | API prices |

Tests: `pytest tests/ -q` (14 files, pure helpers, no network).
