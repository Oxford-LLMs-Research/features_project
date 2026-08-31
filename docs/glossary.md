# Glossary — project terms in plain language

Every entry answers four things: **what it means**, **why it exists**, **where it
fits**, and **what it implies right now**. Agents (Claude and Cursor) are required —
see `CLAUDE.md` and `.cursor/rules/terminology.mdc` — to give a one-to-two-sentence
reminder of this kind the first time they use any of these terms in text addressed
to a human. If you coin a new name, add it here in the same format, in plain words
(no circular jargon). Metric-level definitions stay in
[`docs/onboarding.md`](onboarding.md) §2; this file owns the project-level terms.

---

## cell

**What:** one (survey, target question, country) combination — e.g. "predict WVS
Q40 in Germany". **Why:** it is the unit at which everything is computed: one oracle,
one selector prompt, one score row per cell. **Fits:** the study grid is a list of
cells; all headline numbers aggregate over cells. **Implies:** when a doc says "89
cells" or "71 genuine", it is counting these combinations, not variables or surveys.

## oracle (and oracle contract v3 / v4, "era")

**What:** for a given cell, a gradient-boosting model trained on all *other* questions
in that survey, used to rank which questions actually predict the target — a
data-derived benchmark ranking. **Why:** nobody knows the "true best" predictors; the
oracle approximates the ceiling achievable from the data itself, so LLM selections can
be judged against something real. **Fits:** every selector comparison (captured
importance, cost of imperfection) is relative to the oracle; oracles live in
`outputs/cache/cells/`. **Implies:** the oracle recipe is versioned as a *contract*.
v3 (one train/holdout split) proved noisy — two honest reads of the same cell agree on
only about a third of their top-10 — and is deprecated; v4 (5-fold cross-validated
ranking, a 30% reserve for the downstream evaluator, an untouched 20% valuation
holdout) is the current contract. Any number computed from a v3 oracle is provisional.
"Era-3" in older docs means "produced under contract v3".

## selector

**What:** the LLM under test (e.g. DeepSeek, Kimi) that reads a target question and
names the survey questions it believes predict it. **Why:** the research question is
whether LLMs can do social-science feature selection; the selector is the treatment.
**Fits:** each selector gets its own folder of generations, extractions, maps, and a
`scores_<selector>.csv`. **Implies:** "the zoo" = the set of selectors swept in the
confirmatory run.

## confirmatory (run / grid / stack) — a.k.a. "main run"

**What:** the locked final measurement: a fixed grid of cells, fixed prompts, fixed
models, fixed pipeline, run once to produce the numbers that go in the paper. "Main
run" is an informal synonym — prefer "confirmatory run". **Why:** exploration and
measurement are kept separate so the paper's numbers are not cherry-picked survivors
of tinkering. **Fits:** contrasts with *experiments* (named exploratory studies,
registered first) and *analysis* (one-off diagnostics); confirmatory artifacts are the
canonical outputs. **Implies:** anything feeding the confirmatory run (oracle
contract, grid screen, prompts) must be settled *before* it starts; as of 2026-08-19
it has not started — the grid is still provisional pending the v4 oracle re-run.

## Arm C

**What:** the historical name of the free-text pipeline variant — the selector answers
in natural language, and a separate extract/map step turns that into variable codes.
**Why:** early designs had arms A/B (JSON-format answers); free text won and A/B were
removed. **Fits:** Arm C *is* the pipeline on this branch; the letter survives only in
docs and comments. **Implies:** if you read "Arm C", just read "the pipeline".

## genuine / unestimable / leakage (grid screen; types 1/2/3)

**What:** the leakage audit's verdict for each cell, which defines the default grid:
*genuine* = keep; *unestimable* (type-1) = too little data to measure anything
(minority class too thin on the holdout, or a compromised oracle ceiling); *leakage* =
some feature is the target in disguise (one near-duplicate variable, or a whole
skip-pattern module). **Why:** cells that cannot be measured, or that any method
"wins" by finding a duplicated variable, would corrupt the comparison. **Fits:**
`layout.genuine_cells()` reads this classification and every run uses it as the
default grid. **Implies:** types 2 and 3 (tiny accuracy lift over the mode; accuracy
below the mode) are *not* reasons to drop a cell — log-loss or rank signal can be real
anyway. Decided 2026-08-16.

## honest split / V1 / V2 / eval reserve

**What:** the way each cell's rows are partitioned so no number is measured on rows
that produced it. Under v4: 50% ranks features by cross-validation, 30% is reserved
for the downstream evaluator, 20% (V2) values the final picks and is never touched by
ranking. (v3's V1/V2 were a single 20/20 split — deprecated.) **Why:** re-using rows
inflates results (winner's curse). **Fits:** the type-1 grid screen counts minority
rows *on V2*, because that is where value is measured. **Implies:** sample-size
arguments must reference the split a number is computed on, not the full cell.

## value_over_random (VoR) / cost_of_imperfect (CoI)

**What:** VoR = how much better the downstream model predicts with the selector's k
features than with random-k features; CoI = how far the selector falls short of the
oracle's top-k. **Why:** VoR answers "is the LLM better than nothing?"; CoI answers
"how much is left on the table?". **Fits:** the two headline effects reported per cell
in `scores_<selector>.csv`. **Implies:** both are type-matched — accuracy/log-loss for
binary/nominal targets, Spearman rank correlation for ordinal/continuous. Any
accuracy-only number for an ordinal cell predates the 2026-08-18 fix.

## textbook baseline

**What:** the feature set a social scientist would pick from the literature (standard
demographics etc.), stored once per survey. **Why:** a human-expert comparator —
beating random is weak; beating textbook practice is the interesting claim. **Fits:**
scored the same way as selector/oracle/random per cell. **Implies:** `value_over_
textbook` in scores files.

## captured importance

**What:** the share of the oracle's top-k importance mass that the selector's picks
cover (matched k). **Why:** a scoring-free way to see how close the selection is to
the oracle ranking. **Fits:** complements VoR/CoI in scores files. **Implies:** it is
a within-cell ratio; do not compare its raw units across cells.

## model-chosen k

**What:** the number of variables the selector's answer actually mapped to — the
selector decides how many to name. **Why:** "how many features does the model think it
needs" is part of what we measure, so k must not be clamped to a fixed number.
**Fits:** primary analyses are at model-chosen k; fixed-k views are secondary.
**Implies:** never force k in a confirmatory run.

## Phase A / Phase B / Phase C

**What:** the pre-paper sequence: Phase A = a small pilot (the 22 v4-oracle cells)
measuring noise and design trade-offs; Phase B = power analysis and grid sizing from
the pilot's variance estimates; Phase C = the confirmatory run itself. **Why:** each
phase gates the next on evidence instead of committing everything upfront. **Fits:**
A and B are done (2026-08); C is pending the v4 oracle re-run and grid refresh.
**Implies:** "pilot" numbers are for design decisions only — never quote them as
results.

## prompt pack (prompt-sensitivity v2)

**What:** a locked pair of selector wordings — the system line (role) plus the user
template's referent (`respondent` vs `person`). Stage 1 packs are
`scientist_respondent` (current default), `analyst_person` (the bundled
alternative), and `none_respondent` (no system line, current user prompt).
Two generations of `scientist_respondent` are replicates of one pack, not a
fourth wording. **Why:** the first prompt study only changed the system line and
could not test whether "respondent" boxed models into survey-research space.
**Fits:** elicitation-stage experiment `prompt-sensitivity-v2`; confirmatory
default stays `scientist_respondent` until that study reports. **Implies:** do
not describe Arm 2 as a system-only swap; `person` replaces `respondent` in the
user prompt as well.

## temperature run (t1 / t2)

**What:** two extra generations of the default prompt pack
(`scientist_respondent`) at temperature 1.0, stored under `t1/` and `t2/` —
not `r3`/`r4`. Each gen JSON records `run_kind: temperature` and the
temperature used. **Why:** r1/r2 are greedy (temperature 0), so they can be
almost identical and understate sampling noise; t1/t2 measure how much the
same wording wanders when the model samples. **Fits:** a sidecar on
`prompt-sensitivity-v2`, not a Stage-1 lock job and not a prompt-wording
arm. **Implies:** do not put t1–t2 in the lock-rule denominator for the
temp-0 pack contrasts; r1 vs t1 is a temperature contrast, not a replicate.
`--all` does not launch them.

## theme stratum (prompt-sensitivity v2)

**What:** the two question-content bins the v2 grid forces to be equal: 
political–institutional (political attitudes, institutional trust, political
participation) vs everyday/person (social attitudes, wellbeing, values/identity).
**Why:** a grid that is almost all political items would make a social-scientist
role look native even if it boxed everyday questions. **Fits:** 24 questions =
6 surveys × 2 strata × 2 items, scored pooled *and* by stratum. **Implies:** a
prompt change for the confirmatory run requires the same direction in both
strata, not a grand mean that the two themes disagree on.

## --run-tag

**What:** a `run_main.py` flag that writes map/score outputs under
`runs/<tag>/` instead of the canonical per-selector folders. **Why:** so probes and
re-runs never clobber baseline artifacts. **Fits:** anything citable needs either a
run-tag or an experiment registration (see `CONTRIBUTING.md`). **Implies:** an
untagged run writes canonical files — only do that for the confirmatory pipeline
itself.

## nested country rule

**What:** the confirmatory grid draws one seeded random ordering of each drawn
question's estimable countries; the first up-to-10 (roster permitting) get oracles
computed, and the first 3 of those are the countries the LLM pipeline actually runs
on. **Why:** oracles are cheap on a cluster and selector-independent, so measuring
many countries costs little — but the LLM subsample must be committed *before* any
oracle heterogeneity is seen, or the headline claim is open to a
picked-the-interesting-countries objection. **Fits:** the wide oracle layer feeds
the heterogeneity screen and the transportability pair selection; the narrow first-3
layer is the confirmatory LLM grid (`role == "confirmatory"` in
`data/confirmatory_grid_cells.csv`). **Implies:** heterogeneity may inform which
*pairs* get country-swap scoring, never which countries enter the headline estimate.
