# Grid design memo — sizing the main run (18 Aug 2026)

**Bottom line: replace the provisional "5 targets × 5 countries" grid with a wider,
shallower one — 90 targets (15 per survey) × 3 countries = 270 cells — plus a 30-target
transportability stratum extended to 6 countries (+90 cells). Run 9 selector models
(drop DeepSeek-V4-Flash), quick oracle fits everywhere, and one balanced re-fit pass on
the 30 transportability targets. Cost ≈ $180 in API spend and ≈ 4–5 elapsed days, mostly
unattended CPU. This grid detects a main effect of +0.04 (the pilot estimate is
+0.09–0.13) and a country-adaptation effect of +0.02 — the provisional 150-cell grid
could detect neither reliably.**

All numbers below come from the pilot cells and the last full run's free-text arm
(scores in `outputs/selectors/runs/pilot_phase_a/` and `outputs/selectors/`), not from
assumptions. Analysis scripts: `scripts/grid_power_analysis.py` (variance components,
swap-pair construction) and `scripts/grid_sizing_tables.py` (nested decomposition,
sizing tables, cost model).

---

## The one fact that changes the design

Scores from the same question in different countries move together — strongly. Splitting
the cell-to-cell spread (SD ≈ 0.19–0.20 in value-over-textbook, log loss) into layers:

| Layer | SD | Share of variance |
|---|---|---|
| Between questions (within survey) | 0.126 | ~72% |
| Between countries, same question | 0.078 | ~28% |
| Between surveys | ~0.03 | ~0 (surveys are design strata) |

Measured on the last full run's 47 clean cells (23 questions, both selectors pooled);
the pilot's 26 cells give the same picture.

**Robustness — is the between-question share just a scale artifact?** (checked
2026-08-19 after the objection that questions sit on different scales — log-loss
differences grow with a question's class count/entropy, so raw-unit spread between
questions could reflect units, not substance.) Four cuts on the same data say no:

1. *Rescale each question by its own scale* (`value_over_textbook_ll / |textbook_ll|`,
   i.e. proportional improvement): between-question share 70.5% — unchanged from the
   raw 70.6%.
2. *Hold scale constant*: binary-only questions (identical 2-class family) show the
   **highest** between-question share of all, 89.7% (7 questions, 4 multi-country).
3. *Typed, contract-v4 evidence*: the Phase A pilot's ordinal cells scored the
   type-matched way (`value_over_textbook_rho`, a bounded Spearman difference —
   comparable across questions by construction) give 84% between-question (6
   questions × 2 countries).
4. Scale correlates with effect *magnitudes* (corr ≈ 0.56 with |question mean|) but
   explains almost none of the signed question means (R² ≈ 0.02) — questions differ
   in direction and substance, not just units.

A sign-only version (`VoT > 0`) drops to ~45% between-question, but binarizing
discards magnitude and mechanically pumps Bernoulli noise into the within term — it
does not support the scale story. Caveats that stand: cluster counts are small
(4–16 multi-country questions per cut), so the exact 72/28 is imprecise even though
every cut points the same way; and the headline numbers are era-3-derived — restate
this decomposition from typed v4 scores once the sampled grid's first cells exist,
and define the paper's estimand per target type (log-loss VoT for binary/nominal,
Spearman VoT for ordinal) rather than pooling raw log-loss units across class counts.

Consequence: **adding countries to a question barely adds information about the average
effect; adding questions does.** Two cells of the same question are ~72% redundant. So
for the same budget, a wide-and-shallow grid beats the square 5×5 preset:

| Grid | Cells | Questions | Smallest detectable main effect* |
|---|---|---|---|
| 5 per survey × 5 countries (old preset) | 150 | 30 | **0.067** |
| 10 per survey × 3 countries | 180 | 60 | 0.049 |
| **15 per survey × 3 countries (recommended)** | **270** | **90** | **0.040** |
| 20 per survey × 3 countries | 360 | 120 | 0.034 |

\* value-over-textbook in log loss, 80% power, two-sided α = 0.05, standard errors
clustered by question. Computing the same numbers without the survey stratification
credit changes them by less than 0.001, so the choice is robust to that modeling detail.

The old preset could only detect an effect of 0.067 — three-quarters of the pilot point
estimate. If the confirmatory estimate shrinks moderately (confirmatory estimates
usually do), the 5×5 grid reads "null" without being evidence of one. That is the
worst outcome available and the reason not to run it.

A second, independent reason for 3 countries instead of 5: the sampling frame. At C=3
every survey still offers a wide pool (Asian Barometer: 229 eligible questions; Arab:
151). At C=5, Asian collapses to 29 and Arab to 148 — the draw stops being a sample and
starts being "most of what's left" for one survey.

## Q1 — Confirmatory stratum: how many questions × countries, and what effect is worth detecting?

**The smallest effect worth caring about: +0.05 in log loss over the textbook baseline.**
Three anchors, all from data:

1. **The textbook baseline itself is worth ≈ 0.** Against randomly drawn features, the
   textbook list gains −0.04 (last run) to −0.01 (pilot) in log loss — i.e., nothing.
   So "beats textbook" is a low bar; the claim has to be about magnitude.
2. **The ceiling is +0.18 to +0.30.** That is what the oracle's own top features gain
   over textbook — the most *any* selector could get. +0.05 is roughly a quarter of the
   realistic ceiling; below that, the honest headline would be "little practical value
   over a textbook list," and no survey designer changes behavior.
3. **Shrinkage insurance.** Observed estimates are +0.09 (pilot, 4 selectors) and +0.12
   (last run, 2 selectors). Powering at +0.05 means the finding survives even if the
   confirmatory estimate halves.

**Decision: 15 questions per survey × 6 surveys = 90 questions, × 3 countries each =
270 cells.** Detects +0.040 pooled across selectors; +0.045 even for a single selector
(so per-model claims work too). With ~15–20% attrition (dead cells, leakage discovered
late) the detectable effect drifts to ~0.044 — still under the +0.05 bar. Draw is
type-stratified within survey (proportional to the frame with a floor of 2 per answer
type, so every type is reportable and the log-loss/rank metric families stay balanced),
**demographics excluded as targets** (decision 2026-08-19, `pre_paper_run_decisions.md`
— they stay as features and as the textbook baseline), skip items off,
leakage-flagged targets excluded (the retired accuracy min-signal rule is NOT a frame
filter), with 3 pre-listed spare questions per survey and a registered replacement
rule (same survey, same type).

## Q2 — Transportability stratum: how many screened questions × countries?

The country-swap contrast (score a question's own-country selection vs. the selection
made for another country, same everything else) is the transportability measurement.
Pilot numbers: 122 swaps, per-pair SD 0.081, mean ≈ 0 on unscreened questions — as
expected, since ~2/3 of questions have no cross-country structure to adapt to. Power
therefore depends almost entirely on **how many genuinely heterogeneous questions** are
in the stratum, not on raw swap count.

Pairs from the same question showed essentially zero clustering in the pilot, but that
was measured on unscreened questions with no adaptation signal; sizing assumes modest
clustering (ρ = 0.05–0.10) to be safe. With 6 countries a question yields 30 directed
swap pairs per selector; pooling 9 selectors:

| High-bin questions | Countries | Detectable adaptation effect (ρ=0.05 / ρ=0.10) |
|---|---|---|
| 8 | 6 | 0.019 / 0.026 |
| **12** | **6** | **0.015 / 0.021** |
| 12 | 8 | 0.015 / 0.021 (no gain — countries are nearly free of information here too) |

**Decision: 30 transportability questions, selected from the 90 confirmatory questions
by their measured heterogeneity (which comes free from the confirmatory oracles): 12
high / 8 mid / 10 low, extended from 3 to 6 countries (5 where Asian Barometer caps
out) = +90 cells.** **Low-bin gate (added 2026-08-19):** het ≈ 0 conflates "structure
genuinely shared" with "oracle too noisy to tell" (the screen's own definition:
het = within-country reliability − between-country agreement, so both terms being low
also gives het ≈ 0). A question is eligible for the low bin — the negative control —
only if its within-country reliability (the het screen's `within` term, on the
balanced v4 oracles) is at or above the median across the 90 confirmatory questions;
the frozen numeric value is recorded in the registry entry when the bins are computed.
Questions failing the floor are ineligible for the low bin (replace with the
next-lowest-het question that passes). Without this gate the negative control is
biased toward null mechanically — unmeasurable questions would pass it for free. The target effect of +0.02 is detectable in the high bin at 80%
power under the conservative clustering assumption. The legacy screen suggests the top
tercile starts around het ≈ +0.08, and ~30 of 90 questions should land there — so
choosing 12 leaves margin for screening noise; if fewer than 12 usable high-het
questions emerge, top up from the frame (registered rule).

Two honesty notes, both consequential for registration:

- **The binned high-vs-low difference is *not* powered at +0.02** (detectable ≈ 0.03
  under ρ=0.10). So the pre-registered primary is *high-bin mean gain > 0*, with the
  low bin as a negative control (its confidence interval, ± ≈ 0.016, must cover zero).
  The dose-response claim is registered as a **continuous slope** of swap gain on
  measured heterogeneity across all 30 questions — it uses the full variation and is
  strictly more efficient than the binned difference, which becomes descriptive.
- Swaps exist only in the country-named condition, so this stratum's claims are
  condition-specific by construction.

### Scope: what "cross-country" means in this design (made explicit 2026-08-19)

Every heterogeneity and adaptation measurement here holds the **instrument
constant**: oracle importance structures are only comparable across countries that
answered the *same questionnaire*, so heterogeneity, the low/high bins, and the swap
contrast are all defined **within a survey**. Because the surveys are regional
(Latinobarometro = Latin America, Afrobarometer = Africa, Arab Barometer = MENA,
Asian Barometer = Asia, ESS = Europe), within-survey country variation is mostly
**within-region** variation. Three consequences to state and act on:

1. **Claims wording.** The adaptation result is about within-region country
   variation with the instrument fixed. "Universal" (low bin) means *no measurable
   variation across that survey's country span* — a target can be universal within
   Latin America and still vary globally. Do not write "models adapt across the
   world's diversity" from this design; within-region spans compress true
   heterogeneity, which also makes the test conservative.
2. **WVS is the one cross-region instrument.** Only WVS asks the same battery
   globally, so WVS questions are the only cells where wide (cross-region)
   heterogeneity is directly testable with the instrument held constant.
   **Rule added to the draw:** for WVS questions in the transportability stratum,
   the 6 countries are drawn region-stratified (no two from the same region until
   all represented regions are covered), so the one instrument that can span
   regions actually does. Report het distributions per survey, with WVS flagged as
   the wide-span anchor.
3. **Cross-survey generalization is a different test, deliberately deferred.**
   Comparing the *same concept* asked *differently* across surveys (trust,
   religiosity, …) requires construct-level mapping, not variable-level importance
   — that is the registered post-first-run item (Test-2 / construct-level
   importance in `pre_paper_run_decisions.md`), not something this grid claims.

## Q3 — Selector zoo: how many models, and what about DeepSeek-V4-Flash?

**Keep 9 of the 10; drop Flash from the confirmatory zoo.**

The zoo is priced as capability scaling, and the pilot settles the marginal-cost
question: each non-thinking selector adds only $4–13 of API spend across the whole
grid; Kimi-K3 adds ~$120 if its thinking tokens bill at the feared 5× (keep the
2-cell probe before its sweep). The binding cost — oracle CPU — is selector-independent.
Meanwhile model-vs-model differences are paired within cell (paired SD ≈ 0.10–0.12),
so on this grid adjacent models are distinguishable down to ~0.02–0.03. A flat
capability curve will be flat *with tight error bars* — a publishable result in either
direction. Nine models is cheap resolution; there is no power reason to trim.

Flash is different in kind, not degree: 5 of its 52 pilot cells came back missing or
degenerate (0–1 usable features from token-cap rambling; one 2-feature curt answer),
and the failures concentrate in the country-named condition — that is, **its missing
data are correlated with the experimental manipulation**, and the transportability
stratum lives entirely in that condition. A model that fails differentially by arm
poisons the two contrasts it touches. Its price tier ($0.14/M) is already covered by
gpt-oss-120b and Nemotron-Super-120B, so the capability floor survives without it.
Keep its failure mode in the paper as a documented finding about cheap selectors
(decided now, pre-registered, not after seeing results).

## Q4 — Oracle budget: quick or balanced fits?

**Quick everywhere; one balanced pass on the 30 transportability questions only.**

The headline metrics (value over textbook / random) are computed by refitting an
evaluator on held-out data — the oracle's *ranking* reliability does not enter them. It
enters exactly two places: the heterogeneity screen (needs only a rough ranking, which
quick provides) and the dose-response measurement, where noise in the heterogeneity
score directly shrinks the estimated slope in proportion to its reliability.

Quick fits measure their own reliability at 0.47–0.54, so the slope arrives roughly
halved. Tripling compute buys at most 0.75 reliability (the standard test-lengthening
bound; realistically 0.6–0.7) — i.e., ~25–40% less shrinkage. For the confirmatory
stratum that buys nothing anyone uses. For the dose-response slope, whose target effect
sits right at the detection margin, it is plausibly the difference between a visible
dose-response and a smeared one — and it doubles as a check that the bins assigned from
quick fits hold up under a better oracle. Cost: ~180 balanced cells ≈ 1.5 days of CPU,
confined to the cells that carry the headline transportability claim. (Either way, the
5-fold contract measures per-cell noise from fold spread, so the published noise floor
is free at both tiers.)

## Q5 — Cost and wall-clock

Using the tradeoffs-canvas cost model with measured pilot rates (quick fit ≈ 12.5 min
per cell on a capped CPU budget, balanced ≈ 3×; one oracle queue, 3 workers ×
`--num-cpus 6` — never uncapped, per the pilot's oversubscription lesson):

| | Old 5×5 preset | Recommended grid |
|---|---|---|
| LLM cells (× 2 conditions × 9 selectors) | 150 | 360 |
| API spend | ~$75 | **~$180** (K3 at 5× thinking tax = $120 of it) |
| Oracle CPU wall (one queue) | ~10 h | **~63 h** (360 quick + 180 balanced) |
| LLM pipeline wall | ~11 h | ~25 h (overlaps with oracles) |
| Scoring, all selectors | ~9 h | ~21 h + ~1 day swap scoring (overlappable) |
| Smallest detectable main effect | 0.067 | **0.040** |
| Transportability test | not powered | powered at +0.02 |

Elapsed: ≈ 4–5 working days, staged so nothing waits on anything it doesn't need:
confirmatory oracles run nights 1–2 while generation/extraction/mapping runs by day;
heterogeneity bins are computed the moment confirmatory oracles land (free — the script
reads cached oracle files); country extension + balanced pass nights 3–4; scoring
overlapped throughout. The power gate stays: **if the confirmatory stratum shows ≈ 0
value over textbook, skip the extension and balanced pass entirely** (saves ~2 days and
leaves the short boundary-paper exit open).

---

## The grid to approve

1. **Frame:** type-1-passing targets, **demographics excluded as targets** (features
   and textbook baseline only — see `pre_paper_run_decisions.md` stack locks), skip
   items off, leakage-flagged excluded: **1,233 questions** (1,133 with ≥3
   draw-eligible countries; was ~1,525 with demographics in).
2. **Confirmatory stratum:** 15 questions per survey × 6 surveys, type-stratified
   within survey (proportional with floor 2 per answer type **where the survey
   fields ≥ 2 eligible questions of that type; else take all it has** — on the
   demographics-free frame only ESS fields continuous at all, exactly 2, and
   binary pools are thin in Afrobarometer (6) and Arab Barometer (4), so those
   floor strata are near-censuses, not samples — checked 2026-08-19), × 3 countries
   drawn uniformly from each question's eligible countries, registered seed =
   **270 cells**; 3 spares per survey with a same-survey-same-type replacement rule
   (spares must exist for a type to use its floor; continuous has no spares).
3. **Transportability stratum:** 30 of those 90 questions by measured heterogeneity —
   12 high / 8 mid / 10 low (negative controls; low-bin membership additionally
   requires within-country oracle reliability at or above the 90-question median,
   see Q2 gate) — extended to 6 countries (5 for Asian Barometer) = **+90 cells**;
   top-up from frame only if the high bin comes up short.
4. **Selectors (9):** Kimi-K3 (2-cell probe first), Kimi-K2.6, DeepSeek-V4-Pro,
   GLM-5.1, Qwen3.5-397B, Nemotron-Ultra-550B, MiniMax-M3, Nemotron-Super-120B,
   gpt-oss-120b. **DeepSeek-V4-Flash dropped** (failure mode correlated with condition).
5. **Both prompt conditions; primary analyses at model-chosen k** (the measurand).
6. **Oracles:** v4 5-fold quick on all 360 cells; balanced re-fit on the 30
   transportability questions (~180 cells).
7. **Budget:** ≈ $180 API; ≈ 63 h oracle CPU on one capped queue; ≈ 4–5 elapsed days.

## To pre-register in `docs/experiments_registry.md` before launch

One entry, stage `end-to-end`, status `design`, with:

- **Primary 1 (value):** mean value-over-textbook in log loss at model-chosen k,
  unprompted condition, pooled selectors; standard errors clustered by question;
  surveys reported as strata. Smallest effect of interest +0.05 (this memo's anchors);
  grid powered to 0.040.
- **Primary 1 heterogeneity (registered reported result, not a test):** questions ask
  genuinely different things, so ~72% of cell-to-cell variance is between questions —
  that spread is substance, not nuisance. Report the between-question distribution of
  the effect alongside its mean: share of questions with value-over-textbook > 0,
  between-question SD, and the per-target-type strata (log-loss effects for
  binary/nominal, Spearman effects for ordinal — never pooled across metric families).
  A grand mean alone would misrepresent a heterogeneous answer to "do models capture
  predictive structure."
- **Primary 2 (transportability):** mean own-vs-swapped gain in log loss, high-het bin,
  country condition, pooled selectors, clustered by question; low bin registered as
  negative control (CI must cover 0), with low-bin membership reliability-gated
  (Q2 gate) so "no adaptation signal" cannot be manufactured by oracle noise.
- **Secondary:** continuous dose-response slope of swap gain on balanced-oracle
  heterogeneity (30 questions); condition contrast (country-named vs unprompted);
  capability curve via within-cell paired selector contrasts ordered by model tier;
  breadth-vs-precision (model-k vs fixed k=5). Binned high-vs-low reported as
  descriptive only.
- **Secondary (behavioral adaptation battery, 30 transportability questions):**
  three layers, one x-axis each, separating *adaptation* from *stereotyping/hedging*
  and *rigidity*. (1) Country-request rate in the unprompted arm (share of cells
  whose extracted features name country/nationality — text level, pre-mapping, since
  country is unmappable within a cell) regressed on Δ_oracle from the registered
  `pooled-country-value` experiment — the pooled value of the country column, the
  correct ground truth for "should the model ask" (level + structure; structure-het
  is the wrong x-axis here). (2) Pick differentiation across countries in the
  country-named arm (1 − Jaccard of mapped codes) regressed on structure
  heterogeneity — slope > 0 with low intercept = adaptation; high flat intercept =
  stereotyping; flat zero = rigid. (3) Swap gain (Primary 2) = differentiation
  paying off in value. Cross-arm consistency check: for an adaptive selector the
  same questions drive all three layers. Read slopes/intercepts, not cells — pick
  lists carry extraction jitter (map Jaccard across arms ≈ 0.35). Pilot-grade fact
  motivating layer 1: era-3 request rates 24–50% unprompted vs 4–12% country-named,
  uncorrelated with structure-het (r ≈ −0.02).
- **Frozen before launch:** variance inputs and detectable-effect table (this memo);
  exclusion rules (leakage flag, minimum-signal, provider-caused missingness only —
  don't-know/refused are genuine answers); Flash exclusion; sampling seed and
  replacement rules; full contrast-block reporting per registry convention.

**If this grid is approved, the next actions are: write the registry entry, generate
`data/main_cells.csv` with the registered seed, and start the confirmatory oracle
queue.**
