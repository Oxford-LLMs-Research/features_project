# Pre–paper-run decisions

Status after analyzing the 12-cell stack pilots (2026-08-11). Digests:
`python scripts/analyze_stack_experiments.py` → `outputs/experiments/_analysis/`.

Pilot n is small (≈20–24 model-k rows per arm); treat deltas as go/no-go signals,
not confirmatory effect sizes. **Do not decide from mean Δ alone** — registry
**Contrast detail** + `outputs/experiments/_analysis/registry_contrast_blocks.md`
report median, share Δ>0, PI↔VoR concordance, and by-condition / by-survey strata.

---

## Stack locks (from experiments)

| Knob | Decision | Why |
|------|----------|-----|
| **System prompt** | Provisional: keep `"You are a social science researcher."` until [`prompt-sensitivity-v2`](experiments_registry.md#prompt-sensitivity-v2--role-and-referent-framing) | v1 (`helpful` / `none`) only changed the system line, used v3 oracles, and disagreed across two selectors. v2 tests role+referent against a replicate floor. |
| **Extractor** | Keep Qwen default; MiniMax OK as capacity alternate | [`extract-swap-minimax`](experiments_registry.md#extract-swap-minimax--minimax-extract--nemotron-disambig): mean Δ PI +0.013 / VoR ≈0 but only 42% pairs PI>0 (WVS pulls the mean); MiniMax extract still ~$0.13 vs ~$0.02. |
| **Disambiguator** | Keep Nemotron | Flash-as-disambig slow/costly; mean PI −0.096 masks near-50/50 share>0; no Qwen+Flash follow-up. |
| **Instrument / oracle / embedder** | Unchanged except AutoGluon bag | Free-text Arm C + dual-layer maps; contract-v4 oracles only (v3 retired 2026-08-19); MiniLM default. **Exclude `FASTAI`** (`NeuralNetFastAI`) from every oracle fit — it burns the wall-clock budget on Dropbox-locked `.pth` files and does not carry the ranking. |
| **Targets: no demographics** (2026-08-19) | Demographic variables are **features and the textbook baseline, never prediction targets**. Frame: 1,525 → 1,233 questions (−292 demographic targets; 1,133 with ≥3 draw-eligible countries) — still ~14× the 90 the draw needs. Operational rule: exclude `section == demographics` (universe inventory / `targets.yaml` tagging); borderline items (e.g. self-rated religiosity sitting in an SE/demographic block) are resolved at the frame-freeze flag review with the default *facts about the respondent's situation are out; attitudes, evaluations, and behaviors stay*, wherever the section label disagrees with content. | Three reasons. (1) The research question is whether models understand the predictive structure of attitudes and behaviors — demographics are the *predictors* in that story; predicting them answers nothing the paper asks. (2) They are the main structural leakage source: 6 of 13 era-3 flagged leakage cells were demographic targets (Q263 immigrant ← country of birth; rtrd ← main activity; SE7a ← SE7 worship attendance — all near-duplicates inside demographic modules). (3) The value metric degenerates: value-over-textbook scores picks against a demographic feature list, so for a demographic target the null baseline *is* the target's own module. |

No confirmatory `config.py` / `prompts.py` defaults need changing for these knobs.

Full stratum tables live in the registry entries for `prompt-sensitivity`, `pipeline-role-swap`, and `extract-swap-minimax`.

### Prompt-sensitivity snapshot (`k_spec=model`)

Paired Δ vs `social_scientist` (mean | median | share>0):

| Selector | Arm | PI mean/med/>0 | VoR mean/med/>0 | both+/both−/conflict |
|----------|-----|----------------|-----------------|----------------------|
| kimi | none | −0.059 / +0.008 / 55% | −0.003 / −0.000 / 50% | 8/7/5 |
| kimi | helpful | +0.055 / +0.039 / 80% | +0.014 / +0.011 / 65% | 10/1/9 |
| deepseek_v4 | none | −0.010 / +0.003 / 52% | +0.011 / +0.000 / 48% | 8/7/7 |
| deepseek_v4 | helpful | −0.045 / −0.004 / 43% | +0.004 / −0.008 / 39% | 5/9/9 |

Map Jaccard vs scientist maps ≈0.35 (arms rewrite feature sets substantially).

### Role-swap / extract-swap snapshot

| Contrast | PI mean/med/>0 | VoR mean/med/>0 | both+/both−/conflict | Map Jaccard |
|----------|----------------|-----------------|----------------------|-------------|
| MiniMax+Flash − Qwen+Nemotron | −0.096 / 0 / 52% | −0.018 / +0.003 / 52% | 8/7/6 | 0.44 |
| MiniMax+Nemotron − Qwen+Nemotron | +0.013 / 0 / 42% | −0.000 / +0.004 / 63% | 7/5/5 | 0.52 |

---

## Grid screen: which (target × country) cells to keep

Decided 2026-08-16; **typed, self-contained machinery as of 2026-08-19** in
[`src/survey_features/grid_screen.py`](../src/survey_features/grid_screen.py)
(written by [`scripts/leakage_audit.py`](../scripts/leakage_audit.py);
consumed by [`layout.genuine_cells()`](../src/survey_features/layout.py)).
Oracles (contract v4) rank on **log-loss** (binary/nominal) or **Spearman**
(ordinal/continuous). Accuracy lift vs majority is **not** a drop rule, and every
screen gate is framed in the cell's own metric — no ordinal is ever judged as
multiclass accuracy. The audit takes NO input from historical score files or a
previous audit: leakage gates use a fresh typed probe (`--with-data`, cached per
cell in `audit_probe.json`, keyed to the oracle it describes); offline runs can
only mark type-1 and degrade concentrated cells to `leakage_suspect` — never
clear them to genuine.

| Keep / drop | Name | What it is | Decision |
|-------------|------|------------|----------|
| **Keep as a drop rule** | Type 1 — unestimable PI | Classification: minority too thin on V2 (`n_score × (1−majority) < 10`) or on the CV ranking-fold holdouts (v4 `fold_fit_sizes`). Regression: `n_score < 50` or fewer than 3 distinct scale points (`n_target_unique`). Modal share alone is not the test (`majority ≥ 0.80` was retracted in [`pipeline_audit_2026-08.md`](pipeline_audit_2026-08.md) §A10 — Q43A is 84–91% majority with the largest log-loss signal in the discard pile). After the oracle, `oracle_ceiling@5 < 0.30` is compromised ranking (Q141 Andorra `ceiling@5 ≈ 0.24`). Oracle fit still uses `MIN_CLASS_COUNT = 5` on the full cell. | **Drop** → `unestimable`. |
| **Do not use** | Type 2 — tiny accuracy lift | Oracle top-k XGB accuracy beats the mode by less than 0.03, e.g. Q104 Mauritania +0.029, Q43A Angola +0.003. | **Not a reason to drop.** Class is `genuine` if type-1 and leakage pass. |
| **Do not use** | Type 3 — accuracy below the mode | Downstream XGB accuracy of the oracle top-k *loses* to majority, typical of high-cardinality ordinals (P16ST, P61ST). | **Not a reason to drop.** Spearman PI can still rank. |

Still drop **leakage**: relative near-duplicate (single feature recovers ≥ 0.90 of
oracle PI with concentration ≥ 0.80), **absolute** near-duplicate (single-feature
accuracy ≥ 0.90 and ≥ +0.20 over the mode, or single-feature Spearman ≥ 0.90, with
concentration ≥ 0.80 — fires with ZERO oracle-side lift; added 2026-08-19 after
`SE7a Mongolia`: `SE7` alone hit acc 0.9945 vs mode 0.4438 with importance share
1.0 yet oracle lift ≈ 0 hid it), and distributed skip-pattern modules
(implausible oracle PI with spread importance / `Q67A`). Do not drop on
accuracy-vs-majority.

History: the 2026-08-19 refresh on the era-3 (v3) oracles gave 89 cells →
**71 genuine** (was 42), 5 unestimable, 10 concentrated leakage, 3 distributed
(Q67A). That grid is **provisional** — v3 inputs are retired. The binding grid
comes from: `rerun_oracles.py` (all 89 → v4) → `leakage_audit.py --with-data`.
Do not mix any grid with the old 42/47 accuracy-era score files, and archive
`selectors/scores_*.csv` before re-scoring (the score-phase resume would silently
skip cells already present in an old file).

---

## Confirmatory grid draw — mechanism decided 2026-08-31

Implemented in [`scripts/make_confirmatory_grid.py`](../scripts/make_confirmatory_grid.py)
(draws from the frozen frame; refuses to run if the frame rules stop reproducing the
frozen counts). Decisions, superseding the memo's 15×3 where they differ:

- **20 questions per survey** (was 15). Tightens the smallest detectable pooled
  effect from 0.040 to 0.034 and buys insurance against the budgeted 15–20% cell
  attrition; the marginal cost (~90 more LLM cells) is small because countries, not
  questions, were the expensive axis being cut.
- **Type-stratified allocation**: proportional to the survey's draw-eligible pool
  with a floor of 2 per answer type present (memo rule); sections ("themes") are
  spread inside each type by round-robin rather than hard quotas. 3 spares per
  survey; replacement = same survey, same type, first unused spare (else next
  unused spare of any type; log the substitution).
- **Nested country rule** (see glossary): one seeded permutation of each question's
  estimable countries (WVS ordered region-round-robin so every prefix spans
  regions); first min(10, roster) = **oracle set** (computed on the ARC cluster,
  selector-independent), first 3 = **LLM confirmatory countries**, fixed *before*
  any oracle heterogeneity exists. Heterogeneity later selects transportability
  swap *pairs* from the oracled set (registered max-disagreement rule, to be added
  to the registry with the het screen), never the headline countries.
- **Isolated seeding**: each survey's question draw and each question's country
  permutation come from a child RNG keyed by (seed, survey[, target]) — a
  re-freeze that changes one survey's pool does not reshuffle the others.

**Status: the 2026-08-31 draw output is provisional and unregistered.** Two data
rulings must land first, then the inventory refresh → frame re-freeze → re-draw:

1. **Asian Barometer case-twin coalesce (data bug, discovered 2026-08-31).** The
   merged file holds each question twice — uppercase columns for one release batch
   (Indonesia, Taiwan, Philippines; 4,272 respondents), lowercase for the other
   (Mongolia, Cambodia, Vietnam, Australia, Korea, Thailand; 7,380) — with zero
   row overlap and identical value labels where scales are shared (Q53/q53).
   221 twin pairs. Consequence: the inventory's estimability counts are wrong for
   ~185 of 248 Asian items (`n_countries=9` but `n_type1_pass=3`), the "Asian is
   a 3-country survey" ceiling is an artifact, and every Asian cell's oracle
   feature pool was effectively half dead columns. Fix: case-insensitive coalesce
   in the cleaning path before anything else consumes the frame.
2. **Other-specify columns out of the question frame.** Verbatim/post-code
   appendages of a base question (Afrobarometer `*OTHER`, 13 sibling pairs; Asian
   `*other`/`*other_clarify` variants) are response-coding artifacts, not
   questions: as targets they are junk, as features for their own base question
   they are skip-pattern leakage (three drawn targets — Afro Q96, Asian Q34/Q53 —
   currently have their sibling sitting in the feature pool). Drop them from the
   cleaned question columns per survey-specific pattern. ESS `*oth` items
   (dscroth, dngoth, medtroth) are genuine checkbox variables and stay.

Both are cleaning-contract changes: they must land **before** the ARC oracle map so
the whole map is computed on one feature universe.

## Ranked backlog before the confirmatory paper run

0. **`prompt-sensitivity-v2` (stack lock).** Registered 2026-08-19. Freeze is
   [`data/prompt_sensitivity_v2_cells.yaml`](../data/prompt_sensitivity_v2_cells.yaml)
   (24 questions × 3 countries). Next: v4 oracles for those 72 cells → leakage
   screen on that 72 → gen/extract/map/score. Do not start Phase C until Stage 1
   either clears the default prompt or forces a change.

1. **Build the confirmatory grid fresh; compute v4 oracles only for it** — code
   done 2026-08-19 (v4 is the only contract; typed self-contained audit; absolute
   near-duplicate rule; regression type-1). Do **not** blanket-recompute the 89
   era-3 cells: they are the legacy grid, and the design memo
   (`grid_design_memo_2026-08.md`) replaces it with a sampled 90×3 + 30×6 grid.
   Runbook: (a) lock the target universe (`target_universe_screen.py`, no oracles
   needed; decide flagged cases — e.g. whether ESS test-battery items like
   `testji4` are targets at all); (b) sample the grid per the memo; (c)
   `cp -r outputs/cache/cells outputs/cache/cells_v3`, then compute v4 oracles for
   exactly the sampled cells (the 22 pilot cells already have v4); (d)
   `python scripts/leakage_audit.py --with-data` → the binding genuine set
   (expect SE7a-style near-duplicates to drop); (e) only then archive the legacy
   `grid/`, `sensitivity/`, v3 cells, and old `selectors/scores_*.csv` to
   `.trash/`. Legacy 89-cell oracles get recomputed only if drawn into the sample.

2. **Lock the selector zoo**  
   `config.SELECTORS` today is only `deepseek` (V4-Pro) + `kimi` (K2.6). Registry `confirmatory-zoo` is still design-only. Decide the locked list (IDs on Nebius Studio / Token Factory), register keys in `config.SELECTORS`, then sweep with fixed Qwen+Nemotron + `social_scientist`.

3. **Optional: similarity / `top_n` sweep**  
   `similarity-threshold` remains design-only. Map Jaccard across prompt arms is already low (~0.35), so retrieval caps still matter for capability ceilings — but this is **not** required to start a first zoo pass if you accept MiniLM + current `min_similarity=0.30` / `top_n=20` as a known floor.

4. **Confirmatory zoo mechanics**  
   Per selector: `scripts/run_main.py --phase pipeline --with-score` (or phased gen→extract→map→score) → `outputs/selectors/scores_<selector>.csv`. Concurrent workers proven on the pilots. Wire usage/cost logs (already in `run_main`).

5. **After first zoo sweep (paper, not stack)**  
   Multiplicity hierarchy (model-k primary), dual-extractor audit (~10%), Test-2 / construct-level importance — keep post first full scores.

---

## Explicit non-goals right now

- Promoting MiniMax or Flash into confirmatory defaults
- Switching the default system prompt to `helpful` on kimi-only evidence
- ~~Expanding the 12-cell prompt-sensitivity factorial~~ (retracted 2026-08-19: replaced by registered `prompt-sensitivity-v2`, not an expansion of the v1 12-cell file)
- ~~Rebuilding era-3 oracles~~ (retracted 2026-08-19: the v3→v4 recompute of all 89 grid cells is now REQUIRED — v3 is a dev byproduct and nothing may cite it)
