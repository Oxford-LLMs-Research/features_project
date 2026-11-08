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

**Verification status (updated 2026-11-08, ARS fact-check pass).** Every citation in
this section was checked by the ARS fact-check pass (the plugin's source-verification
role; rule: if a work's existence cannot be confirmed it is removed): 87 rows, 0
fabricated, 14 corrected in place, 3 whose full text sits on blocked hosts (nature.com)
and are marked *(abstract only)*. arXiv, Semantic Scholar, Springer, T&F and medRxiv
are now reachable from the session; nature.com and pnas.org are not.

### 9.1 Verdict: are we repeating someone else's study?

**No, but two pieces of our framing are no longer new and must not be claimed.**

1. **"An LLM can pick predictive features from variable names alone" is established**
   (LMPriors 2022; LLM-Select, TMLR 2025; a 2025 SIGKDD Explorations empirical study; a
   systematic Oct-2026 study finding classical selectors still usually win). Do not
   headline "LLMs can select features" or "LLMs match data-driven selection".
2. **"An LLM picks survey predictors from variable descriptions" has several direct
   precedents** (scoping, §9.7): PATH vaping (medRxiv 2026), PATH smoking cessation
   (*Nicotine & Tobacco Research* 2025), MIDUS diabetes (IEEE ICHI 2026), and one
   Shanghai survey comparing stated with data importance (Xu & Wang 2025). All are
   single-outcome and single-country, graded mostly by downstream AUC, with no
   reliability-checked importance reference and no cross-country test. Cite them;
   never claim "first".
3. **"LLM social-science knowledge is worse outside WEIRD countries" is anticipated**
   (WEIRD = Western, educated, industrialised, rich, democratic) — by
   Pataranutaporn et al. (WVS well-being, 64 countries, PNAS 2025), a Nature Human
   Behaviour 2026 audit of economic preferences across 74 countries, and the
   cultural-alignment literature. All of these treat the LLM as a *predictor* or a
   *simulated respondent*, not as a *selector*. Our country result is convergent
   evidence under a new measurand, not a discovery.

**Breadth is not novelty (revised 2026-11-06).** More questions, more surveys and
more countries applied to the LLM-Select design is a *replication in new contexts*:
worth doing, but not a contribution on its own, and it should not be argued as one.
Better instrumentation (honest oracle, textbook baseline, leakage screen) is the
same question measured better: also not the claim on its own. The candidate
contributions are the parts that ask a **different question**:

| Candidate | The question it asks | Why prior work cannot answer it | Status in our design | Risk |
|---|---|---|---|---|
| **(1) Knowing where the structure differs** | Not "does the model know what predicts X?" but "does it know *where* that differs?" Do its picks change across countries **in step with (co-varying with) how much the data's predictive structure actually differs** (adaptation), or change everywhere (stereotyping), or nowhere (rigid)? | Cross-country LLM papers measure the *prediction* error by country, not whether the model's stated structure tracks measured heterogeneity. That needs a per-country oracle plus a reliability-gated heterogeneity measure, which none of them have. Closest: a meta-analytic study finding that LLMs and experts both fail on moderation (Collabra) | Registered as **secondary**: dose-response slope; three-layer adaptation battery (grid memo). Primary 2 (swap gain in the high-heterogeneity bin) is the coarse version | Literature predicts weak adaptation, so a null is likely. Still informative, because it is the first test with the right ground truth. Dose-response slope sits at the power margin (needs the balanced-oracle pass) |
| **(2) Level vs structure** | When a model asks for "country", is it right to? Separates country shifting the *average answer* (level) from country changing *which predictors matter* (structure) | Silicon-sampling work finds the country label acts as a fixed level shift (Wang 2026) but cannot test whether that is *correct*; we can, with pooled oracles | `pooled-country-value`, design only | Small; uses only CPU |
| **(3) The ground truth is the bottleneck** | Are existing "LLM vs importance" agreement numbers interpretable at all? Our audit: two honest reads of one cell agree on about ⅓ of their top-10, and two oracle recipes on under ⅕ | LLM-Select, LLM4CKD and TabSHAP compare against one model's importance with no reliability ceiling | Ceiling and noise floor computed per cell; a ceiling-corrected re-analysis "would conclusions flip?" is not built | Becomes a methods paper unless paired with (1) |

**Implication.** If (1) is the headline, the registered order is backwards: Primary 1
(value over textbook) is the replication and should be the *precondition*; the
heterogeneity-calibrated adaptation result should lead. Changing that is a
pre-registration decision, open as of 2026-11-06, and must be made **before** any
v4 scores or heterogeneity bins are inspected. The targeted moderation search (§9.5) found (1) open with three partial precedents;
the draft re-ordering is §9.6.

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
| **AI-driven feature selection using only survey variable descriptions** (adolescent vaping, PATH), medRxiv Mar 2026 — [link](https://www.medrxiv.org/content/10.64898/2026.03.06.26347816v1.full) Zhang, Zhao, Hu, Le | Four LLMs score 214 PATH variables from descriptions; LightGBM AUC 0.791 with 30 picks vs 0.768 with all (full text, §9.5) | **Closest trap.** One outcome, one country, no oracle, no subgroup test. Cite and differentiate |
| Pataranutaporn, Powdthavee, Archiwaranguprok, Maes, **LLMs predict human well-being — but not equally everywhere** — [arXiv 2507.06141](https://arxiv.org/abs/2507.06141); PNAS 122(48) e2519394122 (2025) as *Simulating human well-being with large language models: systematic validation and misestimation across 64,000 individuals from 64 countries*, [PMC12685116](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12685116/) | WVS, 64 countries; models know broad correlates, mis-weight predictors in poorer / under-represented countries; country in prompt does not fix it | Our cross-country hypothesis for one outcome. **Stratify results by a representation proxy** (GDP, web-language share) |
| **Simulated responses from LLMs are weak proxies for human economic preferences across 74 countries**, Chen, Wang, Wang, Bhatia, Nature Human Behaviour 2026 — [link](https://www.nature.com/articles/s41562-026-02611-4) | Demographic gradients recovered; country-level structure poorly | Predicts that textbook demographics will be hard to beat and country adaptation weak |
| Memar et al., **When do LLMs help with feature selection?**, Oct 2026 — [arXiv 2610.28556](https://arxiv.org/abs/2610.28556v1) (Memar, Wang, Agah) | 18 datasets; classical selectors usually beat the tested 7B LLM selectors; meaningful names help mainly for regression (+0.20 R² vs anonymised), less and dataset-dependently for classification | **Borrow the informative vs anonymised vs shuffled names ablation** |
| Ma & Powell, **Can LLMs predict associations among human attitudes?**, CogSci 2025 — [arXiv 2503.21011](https://arxiv.org/abs/2503.21011) | GPT-4o's estimated pairwise correlations among 64 OpinionQA attitudes track human ones (r = .77; .72 for semantically dissimilar pairs) but are pushed toward ±1 | Supports our "from wording alone" premise. **Borrow the semantic-similarity control** |
| Hommel & Arslan, **Language models accurately infer correlations between psychological items and scales from text alone**, AMPPS 2025 — [link](https://journals.sagepub.com/doi/10.1177/25152459251377093) | A *fine-tuned embedding model* (not a prompted LLM) predicts item–item correlations (r = .71 pilot, .59 preregistered validation); scale-level figures are higher | Same premise via a different instrument; do not cite it as evidence about prompted LLMs |
| Wang, **Silicon sampling answers with country-level assumptions**, ESS, 2026 — [arXiv 2609.16395](https://arxiv.org/abs/2609.16395) | Country label acts as a fixed level shift; a neighbour-country average beats the LLMs | **Borrow the neighbour-country baseline** for adaptation; separates "level" from "structure" — fits our `pooled-country-value` design |
| Wedgwood et al., **Personalization, personas, and forecasting in value alignment**, ICML 2026 — [icml.cc](https://icml.cc/virtual/2026/75669) | WVS; forecasting framing beats persona framing; religiosity easy, institutional trust hard | Precedent for our prompt study; **check the same item-domain gradient** |
| Ashokkumar, Hewitt, Ghezae, Willer, **Large language models can predict the results of social science experiments**, *Nature* 656:115–122 (2026), [doi 10.1038/s41586-026-10742-x](https://www.nature.com/articles/s41586-026-10742-x); 2024 preprint at [Stanford GSB](https://www.gsb.stanford.edu/faculty-research/publications/large-language-models-can-predict-results-social-science-experiments) *(abstract only)* | Preprint: LLM forecasts of 70 survey experiments correlate ~0.85 with results, overestimate magnitudes; unpublished-study control | Model for a **memorisation control** and a human-expert comparison. **Read the *Nature* full text**: a snippet suggests it reports accuracy on interaction effects where heterogeneity exists |

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
| "It's memorised from published WVS/ESS findings" | Ashokkumar et al. 2026; Kıcıman et al. TMLR 2024; Hommel & Arslan 2025 | ✚ memorisation probe: obscure national items / waves after model cutoffs; correlate hit rate with how often an item is studied |
| "Does it just use word overlap?" | Ma & Powell 2025; Memar 2026 | ✚ semantic-similarity control; anonymised-name ablation |
| "How does free-text elicitation compare with the menu-based studies?" | Zhang et al. 2026 (PATH); Le et al. 2025; LLM-Select | ✚ **menu ablation**: on a subset of cells give the selector the candidate-variable list and compare its picks and scores with the free-text arm — separates knowing the structure from recognising it when shown, and bridges to the menu literature |
| "Country effects are level shifts, not structure" | Wang 2026; NHB 2026 | ◐ `pooled-country-value`. ✚ neighbour-country non-LLM baseline |
| "Transportability is a causal term" | Pearl & Bareinboim 2014 | ✚ scope as predictive transfer; cite leave-one-country-out precedent (JEPP 2025, below) |
| "Does the LLM add anything over data-driven selection?" | LLM-Lasso 2025; AutoElicit ICML 2025; Langseth et al. 2026 | ✚ optional small-sample curve: at what n do LLM picks stop beating a data-driven selector |
| "Results depend on the extractor/mapper" | — | ◐ dual-extractor audit; Nemotron-affinity check |
| "Models differ only in WEIRD-ness" | Atari et al. 2023; Tao et al. 2024 | ✚ regress per-country performance on cultural distance from the US |

### 9.5 Moderation novelty check (2026-11-06, full texts where reachable)

**Why this was run.** §9.1 lists "knowing where the structure differs" as the
candidate headline. This search asked whether anyone has already tested if an
LLM's view of *which predictors matter* changes across countries in step with how
much the data's structure changes. Full texts were read for the arXiv and medRxiv
papers. The Nature and Research Square papers were abstract-only
(nature.com still serves an identity check the proxy blocks; pnas.org returns 403).

**Verdict: open, with three partial precedents.** No paper elicits *stated*
predictors per country, scores them against a per-country importance ranking,
regresses the model's change in picks on the measured change in structure, or runs
a swap test with a negative control. Three papers each cover one piece:

| Paper | What it shows about country heterogeneity | Access | How we differ |
|---|---|---|---|
| Powdthavee (2026) LLMs substantially compress well-being dispersion but largely preserve its socioeconomic structure — [arXiv 2609.33055](https://arxiv.org/abs/2609.33055) | **Closest.** WVS-7, 66 countries, 6 LLMs: countries with steeper real income gradients also show steeper LLM gradients (Spearman 0.54–0.80) | full text | One predictor (income); outcome is the *spread* of life satisfaction; structure implied by simulated answers, not stated; no importance ranking, swap or reliability gate |
| Luo et al. (2026) ACE-Align — [arXiv 2601.12962](https://arxiv.org/abs/2601.12962) | Per-country effects of 4 attributes, 14 countries; the dominant attribute varies by country; taxonomy **Flipped / Stereotyping / Erasure / Aligned** | full text | A fine-tuning method, not a knowledge probe; 4 attributes. **Their "stereotyping" means overstating an effect within a country, so our term must be defined against theirs** |
| Pataranutaporn, Powdthavee, Archiwaranguprok, Maes (2025), PNAS 122(48) e2519394122 — [arXiv 2507.06141](https://arxiv.org/abs/2507.06141) | Real income gradient steeper in low-HDI countries (HDI = Human Development Index); all four LLMs give "nearly identical" gradients in both groups (rigidity, informally). Country effects on the level of the answer shrunk toward zero | arXiv full text | One predictor, one two-way split, pooled weights. **Pre-empts the "level" half of our level-vs-structure claim** |

Adjacent: Wang 2026 ([2609.16395](https://arxiv.org/abs/2609.16395); the country label carries the country *level*, within-country recovery ≈ 0);
D'Ecclesiis, Castellacci, Ahmadi 2026 ([2611.00256](https://arxiv.org/abs/2611.00256); importance rankings of real vs simulated Eurobarometer outcomes, pooled across countries — closest *method*);
Li & Ji 2026 ([2604.02458](https://arxiv.org/abs/2604.02458); model error clusters by country, ICC 12–16% for effect estimates);
Ashokkumar, Hewitt, Ghezae, Willer, *Nature* 2026 (see §9.2; subgroup and interaction results not yet read — abstract only);
Ji, Xue, Chen, Zhu (2025) *Collabra: Psychology*, [doi 10.1525/collabra.144442](https://online.ucpress.edu/collabra/article/11/1/144442/213650) (complex relationships harder than simple ones, domain expertise helped; moderation-specific results not read — the abstract does not single out moderation);
Eriksson, Vartanova, Strimling 2026 ([2609.30896](https://arxiv.org/abs/2609.30896)) and Tao et al. 2026 ([2605.08486](https://arxiv.org/abs/2605.08486)) on compressed country levels;
Daryani, Bogen, Daepp 2026 ([2609.01902](https://arxiv.org/abs/2609.01902); invented change over time — a temporal analogue of stereotyping);
Cui, Li, Zhou 2025 (venue reported as *Nat Comput Sci* 5(8); the scoping screener could not reconfirm it), preprint [arXiv 2409.00128](https://arxiv.org/abs/2409.00128) (main effects replicated 73–81%, interactions 46–63%);
Powdthavee 2026 ([2605.30934](https://arxiv.org/abs/2605.30934); does cross-language divergence track institutional differences — similar dose-response logic, moral judgements).
No paper on LLM priors for per-environment causal graphs was found. *Note:* Powdthavee co-authors three of these precedents (the PNAS well-being paper, the dispersion paper and the moral-reasoning paper); present them as one research programme, not three independent confirmations.

**Corrections to §9.2 from full texts.** The vaping paper is Zhang, Zhao, Hu, Le
(medRxiv, 9 Mar 2026): GPT-4o, LLaMA-3.1-70B, Qwen-2.5-72B, DeepSeek-V3 score 214
PATH variables (15 runs, temperature 0.7) for 7,943 US adolescents; AUC 0.791 with
30 Qwen picks vs 0.768 for all variables; **no importance-oracle comparison and no
subgroup test**; sample size varies with the feature set. Its predecessor is Le et
al. 2025, *Nicotine & Tobacco Research* (GPT-4.1, smoking cessation). The NHB 2026
paper is Chen, Wang, Wang, Bhatia (Global Preference Survey, 74 countries); whether
its gradients are compared within each country is **unresolved** until the full text
is read. LLM-Select's 60 citing papers (Semantic Scholar) contain nothing on
country- or domain-conditional selection.

**Claim wording.**
- *Safe:* "To our knowledge, the first test of whether LLM-**stated** predictors change
  across countries in proportion to **measured** differences in data-driven importance,
  with a country-swap test and a negative control. Prior work tests pooled predictor
  structure, country levels, or one predictor's cross-country gradient in simulated
  responses."
- *Not safe:* "first to ask whether LLMs know cross-national heterogeneity"; "first to
  show LLMs flatten country differences"; stereotyping/rigidity as a new taxonomy
  without contrasting ACE-Align; level-vs-structure as new *for levels* (only the
  separation is new).

**New controls the closest papers suggest** (added to the §9.6 proposal):
nearest-country and pooled-ranking baselines (Wang); robust slopes and
leave-one-country-out for the dose-response (Powdthavee's Pearson r and slopes move with
Iraq; rank correlations do not); a country random effect or per-country oracle reliability (Li & Ji); a
placebo prompt with a wrong or shuffled country (Pataranutaporn found injected facts
spill over); top-k overlap plus full-ranking Spearman against the oracle (D'Ecclesiis).

### 9.6 Proposed re-ordering of the pre-registration (DRAFT — not registered)

*Status: proposal for the PI to approve, edit or reject. It changes nothing until
it is copied into `experiments_registry.md`. It is valid only if **no v4 score file
and no heterogeneity bin has been inspected yet** — the repo cannot see whether that
is still true after 2026-09-06, so confirm first.*

**Headline question.** Do the models know *where* predictive structure differs
between countries? The **adaptation test** (glossary) asks whether a model's
country-named picks differ between two countries by an amount that tracks how much
those countries' oracle rankings differ.

| Role | Endpoint | Was |
|---|---|---|
| **Primary A — calibration** | Slope of pick differentiation (1 − Jaccard of mapped codes, country-named arm, each within-survey country pair) on reliability-corrected oracle heterogeneity (between-country disagreement of v4 rankings). Read three ways: slope > 0 with intercept near the noise floor = **adapts**; intercept well above the floor with flat slope = **changes picks regardless** (our "stereotyping", distinct from ACE-Align's); flat at the floor = **rigid** | secondary (behavioural battery layer 2) |
| **Primary B — payoff** | Swap gain (own-country picks minus other-country picks, scored on the own country's rows), high-heterogeneity bin; low bin as negative control | Primary 2 |
| **Precondition** | Value over textbook, per answer-type family (the replication of LLM-Select-type findings in our setting). Reported first, but framed as "the knowledge exists", not as the contribution | Primary 1 |
| Secondary | Level vs structure (`pooled-country-value`; claim only the separation); dose-response of swap gain; capability ladders; country-blind vs country-named | mostly unchanged |

**Design additions this requires (decide before launch):**
1. **A noise floor for pick differentiation.** Two greedy draws of the *same* prompt
   overlap only 0.27–0.29 for DeepSeek and Kimi (prompt-sensitivity v2). Without a
   same-country replicate, extraction and generation jitter shows up as
   "stereotyping". Add one replicate generation of the country-named arm per
   confirmatory cell: about 360 extra units per selector, roughly +$33 across the
   six at 2026-09-03 prices, paid out of the existing ~$44 budget line.
2. **Placebo prompt.** A wrong-country or shuffled-country prompt on a subset (for
   example 30 questions × 1 country), so we can tell picks that change *because the
   country is right* from picks that change because *a* country was named.
3. **No-LLM baselines.** Nearest-country oracle ranking and pooled all-country
   ranking as comparators for country-named picks.
4. **Robustness.** Spearman / robust slopes, leave-one-country-out, a country random
   effect, per-country oracle reliability as a covariate; top-k overlap and
   full-ranking Spearman reported against the oracle.
5. **Power.** Primary A uses every within-survey country pair (3 pairs × 120
   questions at 3 countries, plus 15 pairs × 30 questions at 6), so it may be better
   powered than swap gain, but it has never been sized. **Re-run
   `scripts/grid_power_analysis.py` on the Phase A pilot for this endpoint before
   locking**; the balanced-oracle pass on the 30 transportability questions becomes
   load-bearing (heterogeneity measured with noise shrinks the slope).

**What this does not change:** the grid, oracles, selectors, prompts, or any code
already run. It adds two generation jobs (replicate, placebo) and re-labels endpoints.

### 9.7 Scoping map and narrative synthesis (2026-11-08)

**How this was produced.** Scoping searches under the draft protocol
([`scoping_review_protocol_2026-11.md`](scoping_review_protocol_2026-11.md), v0) by
two AI screeners (Q1+Q4, Q2+Q3); the narrative synthesis below follows ARS
`lit-review` mode. The search log, screening decisions (with reason codes) and the
charting table are in [`scoping_review_2026-11/`](scoping_review_2026-11/) —
`chart.csv` is the merged chart. **Limits that matter:** single AI screener; 78 of 108
included records charted from abstracts only; Semantic Scholar keyword search was rate
limited throughout, so non-arXiv venues were reached mainly by citation sweeps and web
search; several citation sweeps are incomplete (Bisbee, Ashokkumar/Hewitt, Ma & Powell,
Powdthavee); arXiv result windows were truncated by relevance for the two broadest
queries. Treat counts as a map of the arXiv-visible literature, not a census.

**Flow (both screeners).** Q1+Q4: 1,996 identified → 1,877 after de-duplication → 46
included. Q2+Q3: 4,034 identified → 3,789 unique → 3,080 screened → 175 assessed in
full or abstract → 66 included. Merged: **108 unique records** (3 found by both).
Ratings: 1 pre-empts *part* of our claim (Pataranutaporn et al., the "level" half),
15 partial, 66 adjacent, 22 method, 4 background.

**The map.** "Heterogeneity level" is a charting field defined for this review:
*none* (no group comparison) → *levels* (averages or accuracy compared across groups)
→ *structure* (a relationship — slope, effect, importance, correlation — estimated
per group and checked against that group's data) → *structure-vs-measured-
heterogeneity* (the paper tests whether the model's between-group *differences* in a
relationship match the measured differences).

| How the model's knowledge is obtained ↓ / heterogeneity level → | none | levels | structure | structure vs measured heterogeneity |
|---|---|---|---|---|
| **Stated** (model names, ranks or weights predictors or associations) | 40 | 5 | **0** | **0** |
| Implied by simulated respondents | 32 | 10 | 7 | 5 |
| Fine-tuned on survey data | 3 | 1 | 3 | 0 |

The two bold zeros are our study. Every paper that checks *structure* against data
per group gets the structure from simulated respondents or a fine-tuned model; none
asks the model directly. Only 10 of 108 records use data-derived *importance* as the
ground truth, and only 3 run any transfer test (none across countries).

**Narrative synthesis.**

1. *Stated predictor knowledge is real, modest, and graded weakly.* Names-only
   selection works and scales with model size (LLM-Select; Li, Tan, Liu), but classical
   selectors usually still win on prediction (Memar et al.). In survey data the
   precedents are single-outcome, single-country, and graded by downstream AUC: PATH
   vaping (Zhang, Zhao, Hu, Le 2026), PATH smoking cessation (Le et al. 2025, *Nicotine
   & Tobacco Research*), MIDUS diabetes (Yang et al. 2026, IEEE ICHI, abstract only).
   Only one survey study compares the model's *stated* importance with data importance
   (Xu & Wang 2025, [arXiv 2505.23262](https://arxiv.org/abs/2505.23262), a Shanghai
   household survey): they diverge zero-shot. Gong et al. (NeurIPS 2025,
   [arXiv 2506.03444](https://arxiv.org/abs/2506.03444)) grade LLM correlation priors on
   2,096 real variable pairs: 78.8% correct sign.
   **Every one of these precedents hands the model the candidate variables** — it
   scores or chooses from a menu (PATH vaping: all 214 variable names and descriptions,
   each scored 0–1; Xu & Wang: a fixed short list, importance assigned within it; PATH
   cessation and MIDUS: chosen from supplied descriptions, per their abstracts; LLM-Select
   and Memar et al. likewise give feature names). Our selector instead writes what it
   would need to know **without seeing the codebook**, and a separate step maps that onto
   variables. That is *recall* rather than *recognition*: no menu cues, at the price of
   mapping noise, which is why our random and textbook baselines pass through the same
   mapping chain.
2. *No one checks how reliable the ground truth is.* Every paper grading LLM
   importance claims uses one model's SHAP or permutation importance with no stability
   estimate or ceiling (LLM-Select, LLM4CKD, TabSHAP, Xu & Wang, Kuo et al. 2026 *JMIR*).
   Our per-cell ceiling and fold reliability (pipeline audit §A1) have no counterpart in
   this literature: candidate contribution (3) is better supported than §9.1 assumed.
3. *Where structure is checked per group, the model is a respondent, not an expert.*
   The structure-level papers simulate respondents and estimate slopes or importance
   from the simulated answers: WVS correlation structure by subgroup (Williams et al.,
   [2601.15755](https://arxiv.org/abs/2601.15755)), the age–loneliness gap in four EU
   countries (Li, Rettinger, Beyer, [2610.25158](https://arxiv.org/abs/2610.25158)),
   five-country group variation (da Silva et al., [2609.36253](https://arxiv.org/abs/2609.36253)),
   Pew intersectional subgroups ([2608.23005](https://arxiv.org/abs/2608.23005)),
   factorial-survey coefficients ([2610.22409](https://arxiv.org/abs/2610.22409)),
   pooled Eurobarometer importance rankings (D'Ecclesiis et al.), plus ACE-Align and Li &
   Ji. This is the "silicon sampling" route, which is known to distort coefficients
   (Bisbee et al.) — the reason a direct, stated measurand is worth having.
4. *The few tests of "does the model know where structure differs" are single-predictor
   or US-only.* At the top heterogeneity level: income gradients across countries
   (Powdthavee 2026; Pataranutaporn et al. — rigid across low/high-HDI), and three US
   subgroup-moderation tests (Cai et al. [2607.27100](https://arxiv.org/abs/2607.27100);
   Hoq & Weninger [2604.15329](https://arxiv.org/abs/2604.15329); Ashokkumar/Hewitt et
   al., whose interaction-effect correlations a secondary review reports as low). The
   general finding across the field: models recover demographic gradients better than
   cultural ones (NHB 2026: direction right for 77.8% of gender/age/ability dimensions,
   48.1% of cultural ones), compress country differences, and treat the country label as
   a level shift (Wang 2026).
5. *Implication for our claim.* The literature predicts a mostly *rigid* answer to our
   headline question. Our contribution is therefore the **measurement** — the first
   stated-knowledge, multi-predictor, multi-country, reliability-aware test, with a
   transfer (swap) test — and a null is informative only if it is equivalence-tested
   against an attenuation-corrected slope (§9.8, items 1 and 4).

**Gaps (empty or near-empty cells), in order of relevance to us:**
(i) stated predictors per country graded against per-country data importance — empty;
(ii) multi-predictor importance per country — only pooled (D'Ecclesiis);
(iii) cross-country transfer of LLM structural knowledge — empty;
(iv) reliability of the ground truth or of measured heterogeneity — empty;
(v) level vs structure separated — empty;
(vi) memorisation probes and expert/textbook baselines in survey selection — empty;
(vii) cross-national survey selection beyond the US/one city — empty.

**Follow-ups.** Complete the rate-limited citation sweeps (resumable); read the full
texts held back by blocked hosts (*Nature* Ashokkumar et al., NHB 2026, PNAS version of
Pataranutaporn, Collabra tables); retrieve 9 records that exist in Semantic Scholar but
had no reachable abstract (code E6\*), including a PLS-SEM validity audit of synthetic
survey data (doi 10.1016/j.esd.2026.102075) and a *Communications Psychology* 2025 paper
on predicting correlations between personality items; PI spot-check of a sample of
exclusions (protocol §4).

### 9.8 ARS review of the §9.6 proposal — challenges and what they change

The ARS review pass (editor-in-chief, devil's advocate, ethics) returned **MAJOR
REVISION / REVISE / CONDITIONAL**. Its bottom line: *the question is new enough to lead
the paper; the endpoint is not yet good enough to lead it.* Its counter-evidence search
(22 arXiv queries, 9 web searches) found no pre-empting paper; the claim wording should
say "co-varies with", not "in proportion to" (changed in §9.1). Recommended order of
revisions, with what each would change in §9.6:

| # | Issue (severity) | Proposed fix | Decision needed? |
|---|---|---|---|
| 1 | **Three-way reading biased by noise and unfalsifiable** (CRITICAL). Oracle reliability ≈ 0.5 halves the slope and inflates the intercept, pushing an adaptive model into "changes regardless"; no thresholds; categories not exhaustive; pair-level heterogeneity undefined | Define pair-level heterogeneity; errors-in-variables slope (split-half instrument, SIMEX or latent model); test "changes regardless" in the reliability-gated low bin, not via the intercept; pre-register a full slope × excess decision table with equivalence bounds | no — required |
| 2 | **Confounding can fake "adapts"** (CRITICAL). ~72% of variance is between questions; pool size, answer type, k, survey move both axes; WVS pairs are region-stratified, so more different by name *and* more heterogeneous | Question fixed effects (within-question slope primary); survey FE; WVS reported separately; within-survey label-permutation null; **common-country panel** on the 30 transportability questions so country-pair FE are possible | **yes — the panel costs extra generations (est. tens of dollars)** |
| 3 | Validity of re-ordering not verifiable (MAJOR) | Dated PI attestation that no v4 scores, no oracle heterogeneity and no cross-country pick comparison have been inspected; file timestamps recorded | **yes — attestation scope (below)** |
| 4 | Primary A never powered (MAJOR) | Power with dyadic-robust SEs, attenuation at reliability 0.5 and 0.7, against a stated smallest slope of interest | no — run before lock |
| 5 | Measure is unsigned (MAJOR): picks can change "the right amount" toward the wrong variables | Add a signed captured-importance cross-delta (audit §A9) | no |
| 6 | 1 − Jaccard depends on k; oracle side at top-10, picks at model-chosen k (MAJOR) | Chance-corrected, k-adjusted overlap; heterogeneity at matched depth | no |
| 7 | Noise floor misquoted (MAJOR): 0.27–0.29 is soft Jaccard on extracted features for the prompt-study models, not mapped codes for the confirmatory six | Re-measure on mapped codes for the six; per-cell floor; headroom rule | no (the replicate generation in §9.6 already buys it) |
| 8 | Range restriction and leverage (MAJOR): ~12 questions carry the slope; within-region contrasts | Robust slope; leave-one-question-out and leave-one-country-out as registered outputs; within-region scope in the endpoint text | no |
| 9 | Fixed mapper may create differentiation (MAJOR): country words in the essay reach the extractor and disambiguator | Re-map a subset with country tokens masked; report differentiation at the extracted-feature level too | no |
| 10 | No multiple-testing plan (MAJOR) | Gatekeeping: precondition → Primary A → Primary B; per-selector results descriptive (Holm) | no |
| 11 | "So what" argued only as a gap; a null may read as trivially implied (MAJOR) | Equivalence-tested null; frame as a user decision ("can an analyst trust LLM covariate advice in a new country?"); stated-vs-implied comparison on a subset | partly — subset size |
| 12 | Ethics conditions | AI-assistance disclosure in the paper; neutral label instead of "stereotyping"; avoid deficit framing of Global South countries; source hygiene (done in §9) | no |

**Fallback the reviewers recommend:** if the common-country panel (item 2) is not
affordable or the power analysis (item 4) fails, keep the registered Primary 2 (swap
gain) as the primary and register Primary A — made directional (item 5) — as the
leading secondary.

**Status of item 3.** On 2026-11-08 the PI stated that nobody has looked at v4 scores
yet. The reviewers ask for a broader attestation: also no inspection of v4 oracle
heterogeneity and no comparison of the six selectors' picks across countries (those
picks have existed since September). To be recorded in the registry entry with date
and file timestamps before the re-ordering is registered.

### 9.4 Annotated bibliography

**A. LLMs as feature selectors / knowledge-driven tabular ML**

| Citation | Link | Rel. | Takeaway |
|---|---|---|---|
| Choi, Cundy, Srivastava, Ermon (2022) LMPriors, NeurIPS FMDM workshop | [2210.12530](https://arxiv.org/abs/2210.12530) | Med | Origin of names-only feature priors; cite |
| Jeong, Lipton, Ravikumar (2025) LLM-Select, TMLR | [2407.02694](https://arxiv.org/abs/2407.02694) | **High** | See 9.2 |
| Li, Tan, Liu (2025) Exploring LLMs for feature selection: a data-centric perspective, SIGKDD Explorations | [2408.12025](https://arxiv.org/abs/2408.12025) | Med | Text-based selection more robust, scales with size; frames us as text-based |
| Zhang, …, Tibshirani (2025) LLM-Lasso | [2502.10648](https://arxiv.org/abs/2502.10648) | Med | LLM scores as Lasso penalties; knowledge+data combination |
| Memar, Wang, Agah (2026) When do LLMs help with feature selection? | [2610.28556](https://arxiv.org/abs/2610.28556v1) | Med-High | See 9.2 |
| Li, Xiu (2025) LLM4FS — **withdrawn by the author (v4, 10 Dec 2025)** | [2503.24157](https://arxiv.org/abs/2503.24157) | — | Do not cite as evidence |
| Yang et al. (2024) ICE-SEARCH | [2402.18609](https://arxiv.org/abs/2402.18609v1) | Low | LLM inside evolutionary search |
| Lee, …, Shen (2025) FREEFORM (genotype), AMIA Informatics Summit | [2410.01795](https://arxiv.org/abs/2410.01795v1) | Low | Knowledge selection helps when data are scarce |
| Skinner, Guo, Li (2026) LLM sparsity prior | [2605.23102](https://arxiv.org/abs/2605.23102) | Low | Naive LLM weights can hurt; LLM priors miscalibrated |
| Bal-Ghaoui, Sabri (2025) LLM-FS-Agent | [2510.05935](https://arxiv.org/html/2510.05935v1) | Low | Role-based deliberation; aside for role framing |
| Hollmann, Müller, Hutter (2023) CAAFE, NeurIPS | [2305.03403](https://arxiv.org/abs/2305.03403) | Low | Feature engineering, not selection |
| Han et al. (2024) FeatLLM, ICML | [PMLR](https://proceedings.mlr.press/v235/han24f.html) | Low | Background |
| Abhyankar, Shojaee, Reddy (2025) LLM-FE | [2503.14434](https://arxiv.org/abs/2503.14434) | Low | Background |
| Kabir, Munira (2026) LLM4CKD, ICDM 2026 | [2609.04013](https://arxiv.org/abs/2609.04013) | Med | LLM rankings agree with each other, weakly with supervised importance — **report between-LLM agreement too** |
| Chaudhary, Agarwal, Alladi (2026) TabSHAP | [2604.21120](https://arxiv.org/abs/2604.21120) | Low-Med | Fine-tuned LLM *classifiers* (not selectors) over-weight text-rich features |
| Cruz, Hardt, Mendler-Dünner (2024) Evaluating LMs as risk scores (folktexts), NeurIPS | [2407.14614](https://arxiv.org/abs/2407.14614v3) | Low-Med | Census data; LLM-as-predictor PI vs XGBoost |
| LoID: What LLMs know but don't say (2026) | [2601.17609](https://arxiv.org/html/2601.17609v2) | Med | Closest "LLM knowledge under shift" — synthetic shift only |
| Huntington-Klein & Murray, Do LLMs act as repositories of causal knowledge?, *J. Causal Inference* ([doi 10.1515/jci-2024-0072](https://doi.org/10.1515/jci-2024-0072)) | [2412.10635](https://arxiv.org/abs/2412.10635) | Low | Over-inclusion; option-order sensitivity |

**B. LLMs and surveys / computational social science**

| Citation | Link | Rel. | Takeaway |
|---|---|---|---|
| Argyle et al. (2023) Out of one, many, Political Analysis | [2209.06899](https://arxiv.org/abs/2209.06899) | Low-Med | Silicon sampling; contrast case |
| Bisbee et al. (2024) Synthetic replacements for human survey data?, Political Analysis | [Cambridge](https://www.cambridge.org/core/journals/political-analysis/article/synthetic-replacements-for-human-survey-data-the-perils-of-large-language-models/B92267DC26195C7F36E63EA04A47D2FE) | Med | 48% of regression coefficients differ significantly from ANES (about a third of those with flipped sign) → motivates direct elicitation |
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
| D'Ecclesiis, Castellacci, Ahmadi (2026) Validating LLM-generated synthetic survey samples, Eurobarometer | [2611.00256](https://arxiv.org/abs/2611.00256) | Low-Med | Importance rankings, real vs synthetic outcomes (Spearman, top-5/10 overlap) |
| Are LLMs chameleons? (ESS); United in diversity? (EP 2024) | [2405.19323](https://arxiv.org/abs/2405.19323), [2409.09045](https://arxiv.org/pdf/2409.09045) | Low | Cross-country silicon sampling |
| Tao et al. (2024) Cultural bias and alignment of LLMs, PNAS Nexus | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39290441/) | Low-Med | Cultural prompting ≈ our country-named arm at response level |
| Atari et al. (2023) Which humans? | [HKS](https://www.hks.harvard.edu/centers/cid/publications/which-humans) | Low | WEIRD gradient; test cultural distance |
| AlKhamissi et al. (2024) Cultural alignment of LLMs, ACL | [ACL](https://aclanthology.org/2024.acl-long.671) | Low | Anthropological prompting |
| Cao et al. (2023) Cross-cultural alignment ChatGPT | [2303.17466](https://ar5iv.org/abs/2303.17466) | Low | Background |
| Abeliuk, Gaete, Bro (2026) Fairness in LLM-generated surveys, EPJ Data Science | [2501.15351](https://arxiv.org/abs/2501.15351v1) | Low-Med | US vs Chile |
| Manning, Zhu, Horton (2024) Automated social science, NBER | [w32381](https://www.nber.org/papers/w32381) | Low-Med | Signs right, magnitudes wrong → favours rank metrics |
| Ji, Xue, Chen, Zhu (2025) Automating social science: LLMs vs human experts in variable relationship identification, *Collabra: Psychology* | [doi](https://online.ucpress.edu/collabra/article/11/1/144442/213650) | Med | LLMs ≈ experts on simple relations; both fail on moderation (country is a moderator) |
| Ludwig & Mullainathan (2024) ML as a tool for hypothesis generation, QJE | [NBER](https://www.nber.org/papers/w31017) | Low | Framing |
| Zhou et al. (2024) HypoGeniC | [ACL](https://aclanthology.org/2024.nlp4science-1.10) | Low | Background |

**C. Predictor discovery in survey data, transfer, and importance methodology**

| Citation | Link | Rel. | Takeaway |
|---|---|---|---|
| Salganik et al. (2020) Fragile Families Challenge, PNAS | [link](https://scholarworks.montana.edu/handle/1/16106) | Method | Simple benchmarks hard to beat; mirrors textbook arm |
| Lundberg et al. (2024) Origins of unpredictability, PNAS | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11181083/) | Bkg | Low-signal cells are legitimate |
| Kern, Klausch, Kreuter (2019) Tree-based ML for survey research, SRM | [link](https://ojs.ub.uni-konstanz.de/srm/article/view/7395) | Bkg | Standard ML-in-surveys cite |
| Breznau et al. (2022) Many analysts, PNAS | [doi 10.1073/pnas.2203150119](https://doi.org/10.1073/pnas.2203150119) | Method | Pre-register oracle settings |
| Benati, Bon, Nardi (2025) RF populist-vote predictors per country (EVS), Quality & Quantity 59 | [Springer](https://link.springer.com/article/10.1007/s11135-025-02064-3) | Bkg | Top predictors differ by country (Poland vs France) |
| Sorace (2025) Europeanisation of policy preferences (leave-one-country-out RF), JEPP | [T&F](https://www.tandfonline.com/doi/full/10.1080/13501763.2025.2512901) | Method | Precedent for cross-country transfer as similarity |
| Pearl & Bareinboim (2014) External validity / transportability, Stat Sci | [Project Euclid](https://projecteuclid.org/journals/statistical-science/volume-29/issue-4/External-Validity-From-Do-Calculus-to-Transportability-Across-Populations/10.1214/14-STS486.full) | Bkg | Scope our term |
| Brima et al. (2026) Cross-country generalisation of tabular ML on DHS (childhood anaemia, 16 countries, leave-one-country-out) | [2605.26589](https://arxiv.org/abs/2605.26589) | Bkg | Precedent for country-held-out evaluation |
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
| Langseth, Hellton, Hammer (2026) LLM knowledge for improved OOD tabular prediction | [2610.36806](https://arxiv.org/abs/2610.36806) | Med | On three datasets the LLM prior helps in-domain as much as out-of-domain (regularisation, not shift correction); it improves transfer only with large training sets — **a caution for Primary B** |
| Ma, Chen, Jiang, Feng, Beer (2026) Variable names provide priors | [2611.02637](https://arxiv.org/abs/2611.02637) | Med | LLMs use variable semantics as a prior for numerical prediction |
| Kıcıman et al. (2023/24) Causal reasoning and LLMs, TMLR | [2305.00050](https://arxiv.org/abs/2305.00050) | Bkg | Names-only knowledge; memorisation concern |
| Long et al. (2023) LLMs as imperfect experts | [2307.02390](https://arxiv.org/abs/2307.02390) | Bkg | Knowledge-vs-data framing |
| Jiralerspong et al. (2024) Efficient causal graph discovery with LLMs | [2402.01207](https://arxiv.org/abs/2402.01207) | Bkg | One-variable-at-a-time elicitation |
| Robustness is important: limitations of LLMs for data fitting (2025) | [2508.19563](https://arxiv.org/html/2508.19563v3) | Method | Name/order perturbations |
| Sclar et al. (2024) Sensitivity to spurious prompt features, ICLR | [2310.11324](https://arxiv.org/abs/2310.11324) | Method | Report spreads across templates |
| Atıl et al. (2025) Non-determinism of "deterministic" LLM system settings in hosted environments, Eval4NLP pp. 135–148 | [ACL](https://aclanthology.org/2025.eval4nlp-1.12) | Method | Justifies replicate floor |

**Gaps in this search.** LLM-Select's citation sweep was done in §9.5 (Semantic Scholar coverage looked incomplete); Hewitt/Ashokkumar citations not swept (rate limit). Still unreadable: nature.com full texts (NHB 2026, Ashokkumar *Nature* 2026), Collabra moderation tables, Research Square, OSF; pnas.org returns 403. NeurIPS 2026 accepted papers not visible.

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
