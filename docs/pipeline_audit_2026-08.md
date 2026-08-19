# Pipeline audit — 2026-08

**Status (2026-08-03): complete.** All findings verified against artifacts on disk; Tier-0
fixes implemented; the log-loss oracle re-run finished (89/89 cells, 0 failures), the
leakage screen was re-run against it, and both selectors were re-scored. Superseded caches
(cells + `leakage_audit_accuracy_v1.csv`) are archived out of tree at
`C:\Users\murrn\cursor\features_project_snapshots\era1_cells_accuracy_v1_*.zip`
(see MANIFEST.md there to restore).

Outstanding: cluster-bootstrap CIs (`analysis/freetext_main_results.py` predates the
textbook columns — read it before running) and era-3 mapping recall.

`docs/pipeline_critique.md` (2026-06) audited the **model → score** path and correctly
framed the results as a lower bound. It explicitly treated the oracle as sound and did
not re-litigate it. This audit goes at that part — the oracle and the metric arithmetic —
plus step-level aptness and the cost structure of the confirmatory run.

**Headline:** the binding constraint on every number in the paper was plausibly the
oracle's *resolution*, not the model's knowledge. Permutation importance was measured in
accuracy, which is not a proper scoring rule, and the ranking was both selected and
scored on the same rows. Fixing the first recovers roughly an order of magnitude of
resolution; fixing the second makes the captured-importance denominator unbiased and
yields a calibrated ceiling.

---

## A. Theoretical soundness

### A1. The oracle had almost no resolution, and the ranking was largely noise

Measured across all 89 cached `oracle.csv` files (accuracy era):

| | median |
|---|---|
| candidate features per cell | 239 |
| features with importance > 0 | 67 — so **~72% were ≤ 0** (46 exactly 0, 55 negative) |
| top-1 importance (accuracy units) | **0.0367** |
| top-1 mean / shuffle-SD | 3.9 |
| rank-5 mean / SD | 1.86 |
| rank-10 mean / SD | 1.41 |
| rank-20 mean / SD | 1.23 |
| cells with < 10 positively-important features | 15.7% |

Two things follow.

**The loss was too coarse.** Under `eval_metric="accuracy"`, permuting an informative
feature on a mode-dominated survey item usually flips no argmax at all, so importance
collapses to exactly zero. That single setting produced the 46-exact-zero mass, the ties
that corrupted `oracle_percentile_mean` (A3), and plausibly a large share of the 31/89
cells the leakage audit discarded as "degenerate" — its `--min-signal 0.03` threshold is
itself an accuracy lift.

Switching to log loss on one cell (`Q67A × Angola`), holding everything else fixed:

| | accuracy era | log-loss era |
|---|---|---|
| features with positive importance | **13 / 239** | **121 / 223** |
| features at exactly 0.0 | 225 | 1 |

Sign verified directly rather than assumed — AutoGluon negates lower-is-better metrics
internally, so this needed checking. Training the downstream classifier on the top / middle /
bottom 10 features by the new `importance_score`:

| feature set | accuracy | log loss |
|---|---|---|
| top-10 | 0.6553 | 0.8984 |
| mid-10 (ranks 101–110) | 0.5868 | 0.9804 |
| bottom-10 | 0.5818 | 0.9862 |

Higher importance = more important. Confirmed.

**The ranking was selected and scored on the same rows.** `captured_importance`'s
denominator is "the oracle's top-k". Choosing that top-k with the same noisy estimates
that supply the denominator gives the denominator a winner's curse — the selected
features' estimates are biased upward — while the model's numerator gets no such
selection. The bias grows with pool size and shrinks with cell n, so it varied
systematically across surveys and countries: precisely the cross-cell comparisons Test 2
and the theme/region breakdowns rest on.

**Fix (implemented).** `compute_oracle` now fits on a stratified 60% and computes
permutation importance twice, on disjoint 20% holdouts: `importance_select` (ranks) and
`importance_score` (values). One fit, two importance passes — importance is much cheaper
than the fit, so this costs ~1.2×, not 2×. `importance_mean`/`importance_std` remain as
aliases of the score columns, so every existing reader picks up the unbiased values with
no change. The fit-split row labels are written to `oracle_meta.json` as `train_index`,
so the downstream evaluator can confine its CV to rows the oracle's ranking never saw —
which also closes `pipeline_critique.md` §4d.

**What this buys: a calibrated ceiling.** `oracle_ceiling@k` = score-split mass of the
k features chosen on the select split ÷ score-split mass of the true top-k. It is below
1 by exactly the amount the ranking is noise (full-grid distribution below).

This is the number the paper has been missing. "Captured importance is a lower bound of
unknown tightness" becomes "the model captures X% where a data-driven method, held to
the same honesty, captures Y%." It is also a per-cell quality screen.

#### Re-run results (all 89 cells; regenerate with `analysis/oracle_era_comparison.py --csv`)

| | accuracy era | log-loss era |
|---|---|---|
| share of pool with importance > 0 (median) | 35.8% | **54.9%** |
| share at exactly 0.0 (median) | 19.2% | **0.0%** |
| share at exactly 0.0 (mean) | 32.6% | **4.6%** |

| honest ceiling | median | min | max |
|---|---|---|---|
| `oracle_ceiling@5` | 0.944 | 0.088 | 1.000 |
| `oracle_ceiling@10` | 0.888 | 0.095 | 1.000 |
| `oracle_ceiling@20` | 0.837 | 0.132 | 1.000 |

The median cell's honest oracle recovers ~89% of the achievable top-10 mass, but the **tail
reaches 0.095** — cells where a data-driven ranking captures almost nothing out of sample.
Reporting a model's captured importance without this denominator conflates "the model missed
it" with "there was nothing stably there to find".

Ranking stability *within* the current era — two independent importance estimates of the
same features under the same fitted model, differing only in which 20% of rows measured them:

| | median | min | max |
|---|---|---|---|
| select-vs-score Spearman (full pool) | **0.167** | −0.078 | 1.000 |
| top-10 Jaccard, select vs score | **0.333** | 0.000 | 1.000 |

Two honest estimates of the same cell agree on only a third of their top ten. This is the
audit's central claim, measured.

**And the sharpest result:**

| | median | mean | max |
|---|---|---|---|
| top-10 Jaccard, accuracy era vs log-loss era | **0.176** | 0.178 | 0.538 |

The two eras' oracle top-10 share fewer than two features in ten. That is not a shift in the
ground truth; it is a different ground truth. The mechanism is in the resolution table —
where a third of the pool sat at exactly 0.0, the "top 10" was a couple of real features plus
arbitrary tie-breaks.

Every published captured-importance number used the accuracy-era top-k as its denominator.
Expect all of them to move substantially, and not by a constant, so cross-cell comparisons
(Test 2, theme and region breakdowns) are affected unevenly.

> **Correction.** An earlier draft of this memo reported this figure as 0.053 from the first
> 14 cells. The full-grid value is **0.176** — same conclusion, but the interim subset (all
> Afrobarometer, which had unusually sparse accuracy-era oracles) overstated it by ~3×. The
> interim also understated the accuracy era's positive-importance share (22.8% vs 35.8%).

### A2. The random null was drawn from the wrong universe, and the right null is not random

Two separate problems.

**Mismatched pools (fixed).** The captured-importance null drew from the cell's oracle
pool — correct. The *accuracy* null drew from `[c for c in country_data.columns if c not
in {target, ccol}]`: every column, including admin columns, high-missingness columns and
the structural-leakage columns the oracle had explicitly removed. On `Q67A × Angola` that
is 379 columns against the oracle's 239 — **140 columns of ballast that no informed
selector would ever pick**, making the null weaker than matched and inflating
`value_over_random`. Both nulls now draw from one `cell_feature_pool()`.

**The deeper problem: beating a random draw is a low bar.** The claim the paper wants is
"the model knows what predicts *this* attitude". Beating a uniform draw from ~240 survey
variables only shows it knows *something*: any competent researcher writes down age,
education, income, gender, religiosity, ideology without reading the question and beats
that null comfortably. `paper/memos/framing_and_comparisons.md` anticipates the reviewer
objection ("why not just use the oracle?") and, as the pipeline stood, could not answer it.

**Fix (implemented): a frozen textbook baseline.** `config.TEXTBOOK_CONSTRUCTS` is a
pre-registered, order-fixed list of ten generic predictors, resolved once per survey
through the *same* retrieval + disambiguation chain the model's requests go through — so
both sides carry the same mapping attenuation. `value_over_textbook` is then the
"knows something question-specific" contrast, and `model − random` stays as the floor test.

This also sharpens the format-pilot result. If free text wins by naming ten predictors
instead of five, and the first five are demographics either way, the textbook contrast at
matched budget is exactly the test that separates "knows about this attitude" from "knows
to say age, income, education".

Coverage varies a lot by survey and must be reported with k, not assumed constant:

| survey | constructs resolved |
|---|---|
| ess_wave_10 / ess_wave_11 | 10 / 10 |
| wvs, asianbarometer | 9 / 10 |
| arabbarometer, latinobarometer | 8 / 10 |
| afrobarometer | 5 / 10 |

### A3. `oracle_percentile_mean` broke ties by dict insertion order

`metrics.py` sorted ascending and assigned `i/(n-1)`. With a median of 46 features at
exactly 0.0 out of 239, an uninformative pick drew a percentile anywhere in a wide band
depending on iteration order. Reported values (0.559 / 0.564) sit just above the 0.50
chance line, so this was not negligible relative to the effect. Now uses
`scipy.stats.rankdata(method="average")`; there is an explicit order-invariance test.

Also fixed: `analysis/freetext_main_results.py` seeded the random captured-importance
baseline with `abs(hash((...)))`. Python randomizes string hashing per process unless
`PYTHONHASHSEED` is set, so **the headline random baseline was not reproducible between
runs**. Replaced with a stable digest (`metrics.stable_seed`).

### A4. "Don't know" was being discarded; "not asked" was being kept

The taxonomy was inverted relative to what the study is about.

`clean_question_columns` applied a blanket `.where(col >= 0)`. In WVS the negative codes
are −1 Don't know, −2 No answer, −3 Not applicable, −4 Not asked, −5 Missing. So the rule
**destroyed the respondent's own answers** (−1, −2) while `MISSING_LABEL_PATTERNS` — a
single flat tuple mixing "refused" and "don't know" with "not asked" and "inap" — never
even ran in the oracle, which called the function without `metadata`. Meanwhile
`score_cell`/`evaluation` applied no cleaning at all, so −3 "Not applicable" reached
XGBoost as a value on the scale. The deprecated `archive/run_grid.py` was the only path
passing metadata: the legacy code was cleaner than the current one.

Don't-know and refusal are answers the respondent chose to give and are as interesting as
substantive ones. What must go is missingness caused by the instrument, routing or
fieldwork. The patterns are now split into `RESPONDENT_MISSING_PATTERNS` (kept, at their
original codes, so DK stays distinguishable from Refused) and `STRUCTURAL_MISSING_PATTERNS`
(→ NaN), with negatives dropped only when the *label* says structural. `load_survey_clean`
is the single definition, used by both the oracle and the scorer.

Verified on WVS: DK/refusal now survive (Q288 household income keeps 2,631 respondent
non-answers), **895,344 observations recovered across 400 question columns (+2.7%)**, and
zero structural codes survive cleaning.

Matching on label substrings turned out to over-capture, and the first pass silently
destroyed substantive categories. Real examples caught and fixed by moving to
segment-boundary matching: `inap` matched **"Neither appropriate nor inappropriate"** and
**"Absolutely inappropriate"** (Arab Barometer scale points); `invalid` matched
**"Invalid ballot"** and **"Invalid vote"** — those would have become NaN. `refuse`
matched **"Refuse workers"** (an ISCO occupation), which is harmless only because
respondent-classified codes are kept. `scripts/audit_missing_codes.py` writes the full
classification (7,503 codes across 7 surveys) to
`outputs/cache/audits/missing_code_taxonomy.csv` for review; the only genuinely ambiguous
label family is "No answer" / "No response" / "No data / No answer", which defaults to
respondent and is flagged `ambiguous=1`.

### A5. A skip-pattern leak was in the grid, and the audit's rule let it through

Fixing A4 immediately caught a leak the pipeline had been scoring as a genuine result.

`Q67A` = "Have you heard about climate change?" · `Q69` = "Who should have primary
responsibility for limiting climate change?" — a follow-up asked **only of respondents
who said yes**. `detect_conditional_leakage` exists precisely for this, and missed it:
the "not asked" routing code was positive, the old `>= 0` rule kept it as a value, so Q69
showed *zero* missingness variation across target classes. Under the metadata-driven
rule the range is 1.000 and it is flagged.

In the accuracy era Q69 carried importance 0.497 of the cell's ~0.53 total mass — and the
cell's oracle accuracy was 0.9868 against a 0.5042 majority baseline.

The leakage audit classed all three `Q67A` cells **genuine**, because its rule is
`recovery ≥ 0.90 AND top_importance_share ≥ 0.80`:

| cell | single-feature recovery | top importance share | class |
|---|---|---|---|
| Q67A × Angola | 0.8976 | 0.9298 | genuine (recovery missed by 0.0024) |
| Q67A × Mali | **0.9877** | 0.2509 | genuine (share below threshold) |
| Q67A × Gabon | **1.0000** | 0.2276 | genuine (share below threshold) |

**Removing Q69 was not enough, and that is the real finding.** The structural fix works —
Q69 is gone from all three Q67A feature pools, and the top feature is now Q94 (education)
with recovery 0.26. But `oracle_acc` **stayed at 0.9868 / 0.9992 / 0.9992**. The whole
Afrobarometer climate battery is asked only of respondents who had heard of climate change,
so the routing signal is spread across dozens of items, each carrying a little of it.

That is a **second kind of leakage**, with the inverse signature of the one the screen was
built for:

| | oracle_acc | top-importance share | single-feature recovery | example |
|---|---|---|---|---|
| **concentrated** | 0.98–1.00 | ~1.00 | ~1.00 | `Q263 ← Q266`, `rtrd ← mnactic` |
| **degenerate** | ≈ majority | low | — | `Q141` (majority 0.989) |
| **distributed** | 0.99 *with real lift* | **0.07–0.17** | **0.10–0.26** | all three `Q67A` |

No single-feature test can see the distributed case, because no single feature is doing the
work. The detectable signature is simply that the cell predicts *implausibly well*: no
attitude item should be recoverable at 99% accuracy from other survey items.

**A rule change I tried and reverted, because the correction matters.** I first replaced the
conjunction with "recovery ≥ 0.95 is leakage on its own". Under the log-loss oracle that
fires on **34 of 68 cells** — the median cell has recovery 0.951 — because one strong
correlate normally does about as much *accuracy* work as a whole top-k. (Median recovery
rose 0.895 → 0.951 between eras not because leakage increased, but because the log-loss
oracle actually identifies the strongest single predictor, where the accuracy oracle's "top
feature" was often noise. `top_feature` changed in 45 of 89 cells.) A rule that classifies
half the grid as leakage is not a detector. It also would not have caught Q67A, whose
recovery is 0.10–0.26.

The screen now uses: the original conjunction for concentrated leakage, plus
`oracle_acc ≥ 0.95` with real lift for distributed leakage.

**Result of the re-screen** (`leakage_audit_accuracy_v1.csv` holds the previous run):

| class | accuracy era | log-loss era + fixed rules |
|---|---|---|
| genuine | 52 | **47** |
| degenerate | 31 | 29 |
| leakage (concentrated) | 6 | 10 |
| leakage (distributed) | — | 3 |

Transitions: 2 degenerate → genuine (recovered by the finer metric), 4 genuine → leakage
(newly caught concentrated proxies), 3 genuine → distributed (the Q67A family), and the 6
original leakage cells all stayed. The grid the paper is built on shrinks from 52 to 47
cells, and all 7 cells that left "genuine" did so because they were carrying leaked signal
(2 arrived from "degenerate" in the other direction).

### A6. k is part of the measurand — and the fixed-k numbers do not mean what they appear to

There is no principled uniform prior on how many predictors a target has; each target
genuinely has its own k, and a model that consistently asks for many is telling us
something we are trying to measure. Captured importance is already budget-matched *within*
a cell — oracle top-k, random-k and textbook-k all use the model's own k — so the model-k
comparison is fair. k is only a confound in *between-arm* contrasts, and the fix there is
to report k as an outcome, not to clamp it.

**The fixed-k rows are weaker evidence than they look.** The model never ranks its
requests. `metrics.captured_importance` and `score_cell` both truncate with `codes[:k]` in
**arrival order** — essay order, after a third model's extraction pass. So "k=5" is an
arbitrary five of the model's picks, not its best five. Fixed-k is therefore a *noisier*
test, not a cleaner one, and it is biased downward relative to any ranked truncation.

This bears directly on the format pilot's headline reading — "the effect is breadth, not
per-item quality", resting on Δ ≈ 0 at k=5. A random five drawn from ten good picks looks
much like five picks. The breadth conclusion may well survive, but it is not established
by the k=5 row as currently computed. Recommend: model-k as the primary endpoint, fixed-k
demoted to a diagnostic carrying this caveat, k reported as a first-class outcome (which
is the design's unbuilt Test 3), and — if a ranked truncation is ever wanted — eliciting
the ranking rather than inferring it from essay order.

### A7. Captured importance is additive and cannot see feature interplay

It sums marginal permutation importances, so a set chosen because its members work
*together* scores no better than the sum of its parts — and a model that reasons about
interaction gets no credit for it. The downstream classifier fits the set jointly and is
the only thing in the pipeline that can reward interaction.

That is the argument for keeping the expensive predictive layer at full density rather
than treating it as a redundant check on captured importance. The two metrics are not
measuring the same thing, and the gap between them is itself reportable. (I had proposed
subsampling that layer as the biggest latency saving; withdrawn — see §C.)

### A8. Credit assignment under construct redundancy (open, Tier 1)

Permutation importance splits credit across correlated indicators, so a *correct broad*
answer ("trust in political institutions") is mechanically capped when the survey carries
three near-synonymous trust items. This is the most likely reason the dual-layer sub-item
experiment came back null at matched k (Δcaptured +0.0004): the problem is credit
*sharing*, not bundling, and expansion does not address it.

The principled fix is grouped importance — cluster the survey variables (embeddings, the
existing 8 harmonized theme sections, or per-cell correlation), permute each cluster
jointly, and score whether the model named the right *constructs*. That is closer to
"does the model understand what predicts X" than variable-level matching, and it is
robust to arbitrary survey redundancy. Clusters are computed once per survey and
AutoGluon's `feature_importance` accepts grouped features, so this is cheap. Not built.

### A9. Other open items (Tier 1, argued not built)

- **Test 2 construct.** Own−cross on a ratio with a cell-varying ceiling is low-powered.
  A country-permutation null (score country A's picks against every other country's
  oracle; ask where own-country falls) gives a per-cell exact p. A rank-correlation
  between the model's cross-country delta and the oracle's is a directional test on the
  same artifacts. Also: the design's MDE uses the JSON-era SD of 0.167 and needs
  re-deriving on free-text variance.
- **Multiplicity.** None anywhere, and the confirmatory design multiplies 16 models ×
  themes × regions × 3 k-specs × ~6 metrics. A stated primary endpoint and testing
  hierarchy are needed at the pre-registration freeze — otherwise a CI that barely
  excludes zero is uninterpretable.
- **Extractor confound.** Measured capability is `f(essay, Qwen's comprehension of it)`.
  The planned 10% dual-extractor audit can detect the bias but not correct it. Reporting
  per-model extraction yield (features per 1k response chars) alongside every capability
  number costs nothing and lets readers see it.

---

## B. Step aptness

**Keep as-is.** Free-text elicitation (the prompt-delta discipline — the free-text prompt
is the exact JSON prompt minus the formatting block — is exemplary and worth stating in
the paper); dual-embed retrieval; the per-feature disambiguation architecture; the 4-way
feature typing (it produced a real behavioural finding); the leakage screen as the grid
definition, with the rule fixed per A5.

**Do not promote ensemble retrieval.** Its own memo's verdict is right. Jaccard 0.65,
ΔVoR +0.008, Δcaptured +0.032, at 2× embedding cost and a 40-candidate pool that forced
the AA-label change. It is a sensitivity result, not a default; the added machinery
(fusion, source tracking, cap tuning, longer prompts) is not repaid.

**`PIPE_TYPES` includes `temporal_contextual`.** Timing and period effects have ~zero
within-cell variance in a single country-wave, so they are structurally unmappable to
individual-level variables regardless of real-world importance. Including them costs LLM
calls and inflates the none-rate without being able to earn credit. Low stakes (7.4% of
features), and it was a deliberate decision — flagged, not pressed.

**The disambiguator is NOT replaceable — proposal withdrawn.** Format-pilot evidence shows
mapper *strength* barely matters (qwen235b − nemotron ≈ 0), and disambiguation is ~94% of
mapping wall time, so it was the obvious deletion candidate. `analysis/mapping_diagnostics.py`
replays every cached pool offline under a trivial `top-1 cosine ≥ τ else none` rule
(kimi, arm C, nemotron, 104 cells):

| τ | agreement with LLM | Jaccard of code sets | codes mapped (LLM 10.6) |
|---|---|---|---|
| 0.40 | 0.459 | 0.330 | 12.6 |
| **0.45** | **0.514** | **0.334** | 9.5 |
| 0.50 | 0.532 | 0.298 | 6.5 |
| 0.60 | 0.495 | 0.199 | 2.6 |

Best case is Jaccard 0.334 — **worse agreement than swapping the entire embedding model
produces (0.56–0.60)**, and that swap left aggregate capability claims intact. The LLM
disambiguator is doing work cosine ranking does not reproduce. Keep it, and drop this from
the efficiency list.

**Retrieval caps measured capability at about half — this is the bigger finding.** The same
diagnostic asks how much of the oracle's top-10 appears *anywhere* in the pools the mapper
saw:

| | mean |
|---|---|
| features requested per cell | 17.1 |
| union of retrieved candidates per cell | 98 variables |
| **recall@20 of the oracle's top-10** | **0.489** (median 0.500) |
| oracle top-10 actually mapped | 0.128 |

Even a perfect disambiguator could not have mapped more than ~49% of the oracle's top-10,
given the model's requests and `top_n=20`. (The gap down to 12.8% is *not* all
disambiguator error — an oracle feature can sit in a pool as a near-miss candidate for a
request about something else, where declining it is correct. The ceiling is the defensible
number.)

And the ceiling varies by more than a factor of two across surveys:

| survey | recall@20 | mapped |
|---|---|---|
| afrobarometer | 0.641 | 0.173 |
| asianbarometer | 0.605 | 0.185 |
| ess_wave_11 | 0.469 | 0.131 |
| arabbarometer | 0.443 | 0.086 |
| wvs | 0.421 | 0.086 |
| latinobarometer | 0.283 | 0.072 |

That directly contaminates cross-survey and cross-region comparisons: a model measured on
Latinobarometer is read through an instrument with less than half the headroom of one
measured on Afrobarometer. `top_n=20` and `min_similarity=0.30` have never been validated
(`docs/similarity_threshold.md` is an admitted stub), and this is the first evidence that
they bind. Sweeping `top_n` is now a priority, and per-survey headroom should be reported
alongside any per-survey capability number.

*Recall recomputes against the era-3 oracle once the re-run lands; the numbers above are
on the accuracy-era ranking.*

**The disambiguator is not deterministic.** Building the textbook baseline twice at
temperature 0 with identical inputs mapped WVS "urban or rural residence" to `Q255` on
one run and `none` on the next. Map reproducibility is therefore not guaranteed even with
fixed seeds and prompts — worth a sentence in the reproducibility section.

**First ground-truth check of mapping quality.** The project has never had a
human-validated gold set (`pipeline_critique.md` §3c: the 3-annotator/300-pair study was
never run). The ten WVS textbook constructs are a miniature one, since their correct
targets are knowable: **8/10 correct**; "urban or rural residence" was forced onto Q255
("How close do you feel to your village, town or city?") when the construct **does not
exist in the WVS candidate universe at all** (zero hits for urban/rural/town-size/
settlement) and the correct answer was `none` — a forced weak match the prompt explicitly
tells the model not to make; and "employment status" went to Q284 (employment *sector*)
when Q279 ("Are you employed now or not?") was available. n=10 is not a validation study,
but it is the first measured mapping error rate in the project and both failure modes are
the ones the design worries about. An override file
(`outputs/cache/baselines/textbook_overrides.json`) exists but is deliberately unused:
hand-pinning only the baseline would give it an advantage the treatment never gets.

---

## C. Latency and the confirmatory run

Measured unit costs: oracle ~207 s/cell at `balanced` (this re-run); mapping ~14.5 s/cell
of which ~94% is disambiguation; one XGBoost 5-fold fit ≈ 7 s regardless of `nthread`;
survey load ≈ 75 s per worker process.

**The oracle cannot be parallelised across cells, and this is worth knowing before the
600-cell run.** The obvious optimisation — fit several cells at once on a 20-core box — was
tried and reverted. AutoGluon's `time_limit` is **wall clock**, and the preset *spends the
whole budget* rather than stopping when converged. Evidence: `corr(n_features, seconds) =
0.16`, and cells of 29 vs 282 features both take ~210s against a 180s limit. So concurrency
does not buy throughput; it starves each fit of cores inside an unchanged wall-clock budget.

Measured at `--workers 5`: **11 of 30 Latinobarometer cells failed** with *"Not enough time
left to train models. Time remaining: −55.21s"* and *"No models were trained successfully
during fit()"*. Worse than the failures, the cells that *did* finish trained fewer models
than their serial counterparts — an oracle whose quality varies with machine load, which is
a confound across the grid rather than merely slow. All 19 contended cells were invalidated
and recomputed serially.

The only genuine lever is `--autogluon-time-limit` (equivalently `--runtime-mode quick`,
60s). Whether 60s costs anything is an empirical question worth answering before the
confirmatory run, and the audit's own findings suggest it may not: the ranking is
noise-dominated regardless (select-vs-score Spearman 0.160, top-10 Jaccard 0.429), so a
better-calibrated predictor may not yield a better-ordered feature list. Test on ~5 cells by
comparing `top-10 Jaccard` and `oracle_ceiling@10` between `quick` and `balanced`. At ~600
cells this is the difference between ~35 h and ~10 h.

Ranked actions:

1. **Persist the per-(cell, k) baseline cache — done.** Oracle top-k, random-k, textbook
   and majority depend only on the cell and the budget, never on which model is being
   scored. The cache was process-local and died with the worker, so a 16-model zoo would
   have paid for them 16 times. Now written to `cache/cells/<cell>/baselines.json`.
   Single largest available saving.
2. **Drop legacy arms A/B from the confirmatory grid.** The format question is answered;
   carrying the arms multiplies the two most expensive stages.
4. **Fixed shared random draw set — done.** Draws are now seeded per (cell, k) via
   `stable_seed` and cached, so every model is compared against the *same* baseline
   realisation. This is cheaper, and it removes the baseline's own Monte-Carlo noise from
   every model-vs-model gap — with draws at 5–10 and effects around 0.02–0.06, that noise
   was a real part of those gaps.
5. **Resume — done.** `run_score_jobs` opened the CSV with `"w"`: a crash late in a
   multi-hour sweep discarded everything, and the run logs already record transient
   provider errors and a worker collapse mid-sweep. It now appends and skips completed
   cells, and archives the file if the column set has changed rather than appending across
   a schema change.
6. **Survey-ordered dispatch — done.** Specs were submitted unordered, so a worker could
   touch all six surveys and pay the ~75 s load repeatedly.
7. `MIN_NORMALIZED_FEATURE_ENTROPY = 0.0` makes the variation filter a no-op for
   categorical features, so near-constant columns sit in the pool, weaken the null and
   enlarge the AutoGluon fit.

**Two proposals withdrawn after testing them.** Subsampling the XGBoost layer (per A7 it is
the only metric that can see feature interplay, so it earns its cost) and deleting the LLM
disambiguator (§B: a cosine rule agrees with it less than a different embedding model does).
The remaining savings above are real, but the mapping stage's cost is not recoverable this
way — the only honest lever left there is dropping arms A/B.

---

## D. Documentation contradictions

1. `docs/experiments_index.md` lists ensemble mapping as "run pending" while
   `docs/ensemble_mapping.md` carries full v1 results and commit `42f28e5` documents them.
2. **Test 2 sign contradiction — the most consequential ambiguity in the corpus.**
   `paper/memos/alignment_findings.md` / `paper/memos/uncertainty_findings.md` report adaptation as a clean null
   (−0.002, "every interval straddles zero"); the free-text results report Kimi +0.023
   [0.001, 0.045]. Both are true of their own runs, neither carries a supersession banner,
   and the index lists both as "Complete" with no ordering. T2 is the designated signature
   result.
3. `paper/memos/framing_and_comparisons.md` argues "the LLM is weak as an importance estimator" from
   the superseded JSON numbers (VoR 0.025, ~half beat random) without saying so; free text
   gives 0.054–0.063 and 76–78%.
4. `paper/memos/prelim_findings.md` describes the 5×5 run `RECONCILIATION_PLAN.md` records as
   unrecoverable; the index still points to it as the results memo.
5. `subitem_mapping.md` still carries the pre-lock hard rule ("never append expanded-k
   rows / main paper numbers stay parent-only"). Dual-layer stays per the 2026-07-31
   decision — so the construct-coverage rationale should be stated as a pre-registered
   choice and the null matched-k result reported **in the same table as the headline**,
   not in a separate memo. The adoption rule changed after the evidence; that is
   defensible on construct-coverage grounds but must be argued, not silently substituted.
6. Path drift after the `7b00be5` layout reorg; stale root `RECONCILIATION_PLAN.md`.

---

## What changed in code

| Area | Change |
|---|---|
| `surveys.py` | respondent/structural missing split; segment-boundary label matching; `load_survey_clean` |
| `oracle.py` | `log_loss`; 60/20/20 fit/select/score; `importance_select`+`importance_score`; `oracle_ceiling`; `oracle_meta.json`; metadata passed to cleaning |
| `evaluation.py` | `StratifiedKFold` (was plain `KFold` on a multiclass target); log loss alongside accuracy |
| `score_cell.py` | one `cell_feature_pool` for the null; persistent `baselines.json`; textbook arm; resume + schema guard; survey-ordered dispatch; canonical `SCORE_COLS` |
| `metrics.py` | tie-averaged percentile; `stable_seed` |
| `retrieval.py` | `target_excluded_codes` — near-paraphrase exclusion now shared by oracle and mapper (the per-feature mapper excluded only the target, so a request could map onto a restatement of the outcome: absent from the oracle table, so scoring 0 captured importance, yet still handed to the classifier, inflating model accuracy — one error pushing the two headline metrics in opposite directions) |
| new scripts | `audit_missing_codes.py`, `build_textbook_baseline.py`, `rerun_oracles.py`, `analysis/mapping_diagnostics.py` |
| tests | 14 → 75; new coverage for the metric functions that feed headline numbers and for the missing-code classifier |

**Re-runs this invalidates.** Every cached `oracle.csv` (archived to
`features_project_snapshots/era1_cells_accuracy_v1_*.zip`), the leakage/degeneracy screen, all `scores_*.csv`
(auto-archived on schema change), and every downstream table.

## A10. Where the grid goes: 89 → 47, and why a pre-screen belongs before the oracle

The grid loses nearly half its cells. The loss is **not** mostly leakage:

| class | n | median majority | median oracle acc | median lift |
|---|---|---|---|---|
| genuine | **47** | 0.374 | 0.468 | +0.102 |
| **degenerate** | **29** | 0.659 | 0.650 | **−0.001** |
| leakage (concentrated) | 10 | 0.646 | 0.902 | +0.242 |
| leakage (distributed) | 3 | 0.592 | 0.999 | +0.407 |

**Degeneracy, not leakage, is the main loss (29 vs 13).** And degenerate cells have a median
lift of −0.001: the full AutoGluon oracle cannot beat predicting the modal answer. That is
the target being unpredictable, not the pipeline damaging it. Only 5 of the 29 are the
trivial "no headroom" case (majority ≥ 0.90); **16 of 29 have majority < 0.70** — real
headroom, and still nothing to find.

This traces straight back to target selection: `prelim/target_selection.py` picks by a
**cardinality quota** (2 binary + 2 mid + 1 large) with topic spread. Nothing in that rule
asks whether the target is predictable, so roughly a third of the grid was chosen without
checking there was anything to find. The confirmatory design already fixes this (screen on
oracle lift ≥ 0.05), but it proposes doing the screening *with the oracle*, at ~210 s/cell.

**A cheap screen reproduces the expensive one.** One XGBoost fit on the whole feature pool —
no permutation importance, no AutoGluon stack — against the AutoGluon oracle, on all 15
ESS cells:

| | |
|---|---|
| corr(cheap lift, oracle lift) | **0.98** |
| cost | 21 s vs ~210 s → **10× cheaper** |
| reproduces the degenerate label | 14/15 (1 borderline false drop at lift 0.024 vs 0.032) |
| also flags leakage | yes — `rtrd` lift 0.407/0.286/0.223 matches the oracle exactly |

And the free part is free: **majority ≥ 0.80 flags 11 cells, 91% of them degenerate, with
zero genuine cells wrongly dropped** — computable from a cross-tab before any model is fit.

### A11. The oracle modelled every target as unordered multiclass — 17 of 30 are ordinal

The single most consequential defect found, because it is in the gold standard itself.
`compute_oracle` hardcoded `problem_type="multiclass"`, so an 11-point left–right scale was
treated as 11 *unordered* categories: ~1,100 respondents split across 11 classes (~100
each), with predicting 6-when-the-truth-is-5 penalised exactly as hard as predicting 1.

`P16ST × Colombia`, same features, same data, only the target's treatment changed:

| treatment | classes | log-loss lift | accuracy | majority |
|---|---|---|---|---|
| as-is, DK kept | 12 | −0.123 | 0.323 | 0.328 |
| substantive only | 11 | −0.134 | 0.351 | 0.343 |
| 3 bins | 3 | −0.055 | 0.549 | 0.484 |
| **2 bins** | 2 | **+0.030** | **0.728** | 0.658 |

The signal is real and was being destroyed by the encoding, not absent from the data. This
was **not** a near-duplicate-filter problem — the 0.85 filter excluded 0 candidates for
P16ST, and the pool already contained party ID, party closeness and social class.

**Fix.** Measurement level is now detected from value labels and drives the model:

| level | problem_type | eval_metric | n of 30 grid targets |
|---|---|---|---|
| binary | `binary` | log_loss | 12 |
| nominal | `multiclass` | log_loss | 1 |
| **ordinal** | **`regression`** | **`spearmanr`** | **17** |
| continuous | `regression` | spearmanr | 0 |

Ordinal uses regression scored by Spearman ρ rather than a cumulative-link decomposition:
rank-based, so it assumes nothing about spacing between scale points; permutation importance
against it answers exactly *"how much does this feature help order respondents on the
scale"*; and it is one fit, versus K−1 for Frank–Hall. Importance units then differ per
cell, which is fine and deliberate — captured importance is a within-cell ratio, so units
cancel, and raw importances were never comparable across cells anyway.

Detection is label-driven (`surveys.detect_target_type`) and therefore fallible, so
`scripts/audit_target_types.py` writes `cache/audits/target_types.csv` for review, the same
pattern as the missing-code taxonomy. Signals used: ordered-scale wording, labels repeated
across adjacent codes (the signature of an anchored numeric scale — how `P16ST` and `Q242`
were caught), numeric-valued labels, and "only missing codes are labelled" ⇒ continuous.

**Validation, substantive rather than numerical.** Re-running `P16ST × Colombia` through the
ordinal path gives 101/193 features with positive importance (versus negative lift under
multiclass), ceiling@10 = 0.680, and a top-10 of: expectations for the economy, whether
parties work well, rating of Gustavo Petro, trust in the President, family party support,
**acceptability of current inequality**, **fairness of income distribution**, urbanicity.
Two redistribution items in the top seven and economic evaluation first — which is what
ought to predict left–right placement, and is not what the multiclass oracle was reporting.

**Consequence: 14 of the 29 degenerate cells sit on ordinal targets.** The grid re-run under
the corrected oracle is expected to recover a substantial share of them.

### The degeneracy screen is itself accuracy-based, and that discards real signal

Degeneracy is **target-level, not cell-level** — the same targets fail across all their
countries (`P16ST` 3/3, `Q141` 3/3, `Q43A` 3/3), which is the signature of a selection rule
that never asked about predictability. But the screen's criterion (`--min-signal 0.03`) is an
*accuracy* lift, so §A1's argument applies to it directly.

Testing every degenerate cell with one XGBoost on the full pool, comparing model log loss
against the marginal-distribution baseline (its entropy):

| target × country | majority | classes | accuracy lift | **log-loss lift (nats)** |
|---|---|---|---|---|
| Q43A × Angola | 0.836 | 3 | +0.008 | **+0.433** |
| Q43A × Mali | 0.882 | 2 | 0.000 | **+0.349** |
| Q43A × Gabon | 0.914 | 2 | 0.000 | **+0.279** |
| Q242 × Germany | 0.766 | 13 | +0.028 | +0.172 |
| Q242 × Andorra | 0.496 | 11 | −0.017 | +0.138 |
| Q15 × Gabon | 0.590 | 6 | +0.024 | +0.105 |
| … | | | | |
| P16ST × Colombia | 0.328 | 12 | −0.054 | −0.123 |
| Q501E_2 × Iraq | 0.457 | 15 | −0.073 | −0.241 |

**10 of 25 assessable degenerate cells have no accuracy lift but real probabilistic signal**
(> 0.02 nats). `Q43A` is the extreme case: the *largest* log-loss signal anywhere in the
degenerate set (0.28–0.43 nats across all three countries), discarded entirely because its
modal answer takes 84–91% of responses so no permutation ever flips an argmax.

The direction is diagnostic: `corr(majority, log-loss lift) = +0.32`. **The higher the modal
share, the more signal accuracy hides** — exactly the mechanism from §A1, now shown to be
deleting a third of the grid rather than merely blurring a ranking.

Two counter-notes, so this is not over-read. The high-cardinality ordinal targets I expected
to be rescued are *not*: `P16ST` (left–right ideology, 12 categories) and `P61ST` and
`Q501E_2` all have **negative** log-loss lift — the model is worse than the marginal. Those
are genuinely unpredictable, and `corr(n_classes, log-loss lift) = 0.20` is weak, so
cardinality is not the driver. And four Asian Barometer cells (`SE7a`, `level`, `SE14a`)
could not be assessed at all: those targets are stored as text, and the numeric marginal
baseline in this test collapsed to zero. They need a categorical-aware version before any
claim is made about them.

**Recommendation:** re-screen degeneracy on log-loss lift against the marginal, not accuracy
lift against the majority class. On this grid that recovers ~10 cells — a larger sample-size
gain than several extra countries, and free.

**Recommended screening order for the confirmatory run**, cheapest first:

1. **Marginal only (free):** drop only on **support** — require enough respondents in the
   *minority* class for it to be estimable. Pure cross-tab.

   > **Retracted.** An earlier version of this list recommended dropping `majority ≥ 0.80`.
   > That is wrong on the study's own terms: a question where 85% agree and 15% do not is
   > exactly where it matters *who* the 15% are, and edge-case behaviour is a large part of
   > why one would simulate respondents with an LLM at all. The data already refuted the
   > rule — `Q43A` has majority 0.836–0.914 and the **largest** log-loss signal in the whole
   > degenerate set (+0.28 to +0.43 nats). A modal-share filter would have deleted the most
   > informative target in the discard pile. Skew is a reason to measure carefully, not to
   > exclude.
2. **One XGBoost on the full pool (~20 s):** judge signal on **log-loss lift vs the
   marginal**, not accuracy lift vs the majority class (see above — the accuracy criterion
   discards ~40% of the cells it rejects). Flag `accuracy ≥ 0.95` or an implausibly large
   lift for leakage review. This is where degeneracy should die.
3. **AutoGluon oracle (~210 s):** only on survivors, since only survivors need a *ranking*.

At ~650 candidate cells that is roughly 3.5 h of screening plus the oracle on ~400 survivors
(~23 h) instead of ~38 h of oracle on everything — and, more importantly, it moves target
selection onto a signal criterion instead of a cardinality quota.

**Is any of the loss self-inflicted?** The era-1 → era-3 transitions say no: 2 degenerate →
genuine (recovered by the finer metric), 4 genuine → concentrated leakage, 3 genuine →
distributed, 29 degenerate stayed, 6 leakage stayed. Every reclassification is individually
evidenced — `Q263 ← Q266` ("in which country were you born?" predicting "are you an
immigrant?"), `rtrd ← mnactic`, `Q67A ←` the routed climate module. Nothing moved because a
processing step damaged it.

## First era-3 results (arm C, nemotron, 47-cell grid)

Read back from `outputs/selectors/scores_{kimi,deepseek}.csv` (1323 / 1347 rows, 0 errors;
n = 90 arm-C rows each = cells × conditions). These are means over cells, **not**
cluster-bootstrap estimates — no CIs yet, so treat as provisional.

**Both selectors at model-k** (after the self-prediction fix below):

| | DeepSeek | Kimi |
|---|---|---|
| mean k | 10.2 | 9.8 |
| captured importance | 0.340 | 0.335 |
| textbook captured | 0.149 | 0.155 |
| value over random (acc) | +0.053 | +0.059 |
| **value over textbook (acc)** | **+0.066** | **+0.066** |
| value over textbook (log loss) | +0.144 | +0.147 |
| share beating random | 0.756 | 0.818 |
| **share beating textbook** | **0.878** | 0.795 |

The model captures roughly **2.2x** the oracle importance mass of ten standard demographics
and beats them by +0.066 accuracy in ~80-88% of cells. That is the claim SS A2 argued the
pipeline could not previously make, and it is a far harder test than beating a random draw
from ~240 variables.

> **Bug found and fixed while writing this up.** Several grid targets ARE textbook
> demographics -- `asianbarometer` SE2 (gender) and `level` (urban/rural),
> `latinobarometer` SEXO (sex). In 7 of 47 genuine cells the baseline therefore contained
> the target and predicted it from itself: textbook accuracy 0.81-0.97 on those cells
> versus 0.38 elsewhere, dragging mean value-over-textbook from +0.079 to +0.015. Fixed by
> excluding the target from the textbook set (`score_cell`), with a regression test on the
> cache fingerprint. After the fix those cells score 0.577/0.589 -- in line with the rest.
>
> This also retracts an interim claim: the pre-fix numbers appeared to show the textbook
> contrast *discriminating between models* (DeepSeek +0.030 vs Kimi +0.015). That gap was
> contamination -- the two selectors had different numbers of affected rows. Corrected,
> both sit at **+0.066**, indistinguishable, as they are on every other metric.



| | model-k (mean k 9.8) | k=5 | k=10 |
|---|---|---|---|
| captured importance | 0.328 (median 0.189) | 0.249 | 0.318 |
| **textbook captured** | **0.150** (median 0.059) | 0.140 | 0.150 |
| value over random (acc) | +0.053 | +0.033 | +0.048 |
| **value over textbook (acc)** | **+0.015** (median +0.058) | −0.002 | +0.005 |
| value over random (log loss) | +0.114 | +0.048 | +0.096 |
| **value over textbook (log loss)** | **+0.058** | +0.047 | +0.036 |
| share beating random | 0.800 | 0.622 | 0.722 |
| **share beating textbook** | **0.750** | 0.682 | 0.750 |

Three things worth noting, all provisional.

**The textbook contrast is the harder test, and the model passes it — at its own k.** It
captures roughly 2.2× the oracle importance mass of ten standard demographics and beats them
in 75% of cells. That is the claim §A2 argued the pipeline could not previously make.

**At a five-feature budget the advantage against demographics essentially vanishes** on the
mean (−0.002 accuracy), though the median stays positive (+0.027) and the log-loss delta
does not collapse (+0.047). Mean and median disagree throughout, so the distribution is
skewed by a few cells where demographics win decisively — a reason to lead with
cluster-bootstrap medians rather than means when these are written up.

**Log loss consistently shows a larger effect than accuracy** (+0.114 vs +0.053 over random
at model-k). That is the predicted consequence of §A1: accuracy cannot see probability shifts
that do not flip an argmax, so it understated every effect in the study.

Note the k=5 caveat from §A6 applies to the fixed-k rows: they truncate in arrival order, so
they are an arbitrary five of the model's picks, not its best five.

## One cell, end to end

`Q67A × Angola`, kimi arm C, unprompted — the same mapped codes scored before and after:

| | accuracy-era oracle | log-loss oracle |
|---|---|---|
| captured importance (model-k = 10) | **0.0000** | **0.4219** |

Zero, because under the accuracy oracle only 13 of 239 features carried any importance at
all and none of them was anything the model asked for. The model's picks were not worthless;
the instrument could not see them.

The textbook baseline lands on the same cell:

| k_spec | model captured | textbook captured | model acc | textbook acc | Δ acc | Δ log loss |
|---|---|---|---|---|---|---|
| model (k=10) | 0.4219 | **0.4253** | 0.6185 | 0.5827 | +0.036 | −0.045 |
| k5 | 0.5493 | 0.4253 | 0.6419 | 0.5827 | +0.059 | +0.086 |

At model-k the model does **not** beat five demographic variables on captured importance,
and the accuracy and log-loss deltas disagree in sign. One cell proves nothing — but this
is exactly the comparison the pipeline could not previously make, and it is visibly a
harder test than the random null.

## Next

1. Finish the oracle re-run (`scripts/rerun_oracles.py`), then
   `python analysis/oracle_era_comparison.py --csv` to refresh §A1's interim table.
2. Re-run the leakage screen with the A5 rule fix — expect the grid to change in both
   directions (recovered "degenerate" cells, newly caught skip-pattern leaks).
3. Re-score all selectors. **Do not score against a partially re-run oracle**: the cache
   would mix eras. Baseline caches carry a fingerprint and invalidate themselves, but the
   grid definition does not.
4. Re-run `analysis/mapping_diagnostics.py` for era-3 recall, then sweep `top_n` — §B
   shows retrieval, not disambiguation, is the binding constraint on mapping.
5. Decide the Tier-1 items in §A8/§A9 before the confirmatory freeze.
