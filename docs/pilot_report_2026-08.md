# Pilot report — machinery shakedown before the main run (18 Aug 2026)

**Bottom line: every piece of machinery the main run needs now works end-to-end, the whole
pilot cost $1.68 in API spend plus one overnight of CPU, and the noise measurements say a
main run of roughly the size we sketched is well powered. One decision is now needed (end
of this page).**

## What we ran

A deliberately small, deliberately unrepresentative grid: 26 cells (12 targets × 2–3
countries, all six surveys, all four answer types), two prompt conditions, four selector
models — the two established ones (DeepSeek-V4-Pro, Kimi-K2.6) plus two cheap new ones
(DeepSeek-V4-Flash, MiniMax-M3) to test what onboarding a new model takes. Nothing here is
a result; the grid exists to exercise code paths and measure costs and noise.

## Machinery checklist — all pass

- Essay generation, feature extraction, and codebook mapping: 100% coverage, all four
  models, zero missing files.
- The upgraded oracle (5-fold cross-validated ranking + untouched valuation holdout):
  ran on 25 fresh cells; the five legacy cells keep their old oracles untouched.
- Scoring against oracle / random / textbook baselines: 4 × 26 cells, zero errors.
- **Country-swap contrast (the transportability measurement): ran for the first time**,
  on all 12 targets × 4 selectors (122 paired swaps). Values near zero, as expected on a
  grid not screened for cross-country heterogeneity — the point was that it computes.
- Heterogeneity screen: reproduces the earlier diagnostic on the 89 legacy cells and
  correctly flags 7 targets whose apparent heterogeneity is data leakage.

## What broke, and what it taught us

- **The Latinobarómetro mystery is solved.** Three things stacked up. This week: running
  several oracle processes at once let each one claim all 20 CPU cores, so every fit ran
  ~3× over its time budget — fixed with a new `--num-cpus` flag (the retried cells then
  passed 4/4). Historically: before the log-loss rework, every long ordinal scale ran as
  an 11–15-class classification, the slowest problem type, and Latinobarómetro is full of
  them — that is why the rework "made things slightly better." What remains: nominal
  targets still cost ~2× a binary fit, and every cell pays ~70s of data-cleaning overhead
  before fitting starts. The data itself is clean — 191 numeric columns, nothing weird.
- **DeepSeek-V4-Flash needs special handling or exclusion.** When told the country, it
  writes to the 4,096-token ceiling (~18k characters of rambling) and extraction then
  finds 0–1 usable features (3 cells); on the age target it answers in ~450 characters
  with 2–4 features (2 cells). Cheap models must be validated per-model before joining
  the zoo — which is exactly what this pilot was for. MiniMax-M3 behaved normally.
- **Japan is not fielded in this Asian Barometer wave** — grid regenerated with Taiwan.
- One pilot cell (SE9 × Philippines) is effectively unpredictable (1 of 245 features
  carries any signal): the minimum-signal screen stays necessary.
- **Ordinal answers are valued as if unordered (user-spotted).** The oracle ranks
  features type-aware (rating scales scored by rank correlation), and the evaluator has
  a matching rank-correlation path — but the scoring layer never passes the answer type,
  so that path is dead code: every cell, rating scales and even the continuous age
  target, is *valued* as unordered classification log loss. Being off by one scale point
  costs the same as being off by nine, and on rating-scale cells the ranking currency
  (rank correlation) doesn't match the valuation currency (log loss). The numbers are
  still proper scores — nothing is arithmetically wrong, and the last full run's
  headline values were computed the same way — but the choice between "one common
  currency for all cells" and "type-matched valuation, reported per type" was never
  actually made; it fell out of an unpassed parameter.

## What it cost

| Item | Measured |
|---|---|
| All LLM calls, whole pilot, 4 selectors | **$1.68** (disambiguation = $0.78 of it, 7,332 calls) |
| Oracle compute, 5-fold, "quick" fits | ~10–15 min per cell on a capped CPU budget |
| End-to-end wall clock | one working day, mostly unattended background runs |

Scaling by cells × selectors, a 150-cell main run with 10 selectors lands near **$25–40
of API spend** plus **1–2 days of oracle CPU** ("balanced" fits triple the oracle time —
budget accordingly). Cost is not a constraint; wall-clock discipline (CPU caps, one
oracle queue) is.

## The oracle upgrade did what it was meant to

On the same cell, agreement between the oracle's ranking and an untouched second read
rose from 0.33–0.35 (old contract) to 0.47–0.54 (new, averaged over folds) — despite the
pilot using the weakest fit setting. More importantly, each cell now measures its own
noise for free, and on the three-country test target the cross-country differences stayed
put while the noise shrank: with the better oracle, that target now shows *detectable*
country-specific structure where the old contract showed none. This directly improves the
odds that the transportability question is answerable.

## The numbers the sizing decision needs

*Reading guide.* Every cell produces one number: how much the predictor improves when it
swaps the 10 generic demographics for the model's chosen features, measured — for every
cell, rating scales included; see the valuation-currency finding above — in log loss
(the average of −ln(probability assigned to the answer each respondent actually gave);
lower is better). A typical pilot cell sits at log loss ≈ 1.09 with demographics — i.e.
the true answer gets ~34% probability. An improvement of **+0.05** means every
respondent's actual answer gets ~5% *relatively* higher probability (34% → ~35.7%); the
pilot's observed average was +0.09, the last full run's ~+0.14. That per-cell improvement
scatters widely across cells — pilot range −0.46 to +0.68, positive in 68% of cells —
and **SD ≈ 0.19 is the width of that scatter**, mostly genuine question-to-question
differences (some questions need question-specific features; others, demographics
already saturate). Grid size follows from the standard rule: to detect a true average
effect E when cells scatter with SD s at 80% power, you need N ≈ (2.8·s/E)² cells.

- Value over demographics: SD ≈ 0.19 → **~115 cells** detect a +0.05 average effect
  (a deliberately conservative bar: half the last full run's level); **~30 cells** would
  suffice at +0.10. The sketched 150-cell grid is comfortable.
- Country-swap contrast: each paired swap (own-country selection vs another country's,
  scored on the same rows) yields one difference; these scatter with **SD ≈ 0.078** →
  **~125 pairs** detect a +0.02 average adaptation effect. A transportability stratum of
  ~10 heterogeneity-screened targets × 5 countries yields ~200 pairs per selector —
  powered with room to spare, before pooling selectors.
- Caveat on both: cells cluster within targets and pairs cluster within targets, so these
  counts are optimistic lower bounds — the sizing exercise must redo them with
  cluster-robust math using the intraclass correlation estimated from the pilot data.

## Addendum (18 Aug, after the type-matched re-score) — the two families look different

Re-scored under type-matched valuation, the pilot's value-over-demographics splits into:

| Family | n (rows) | Mean | SD | Share > 0 |
|---|---|---|---|---|
| Binary/nominal, log loss | 112 | **+0.029** | 0.103 | 60% |
| Rating-scale/continuous, Spearman rho | 91 | **+0.203** | 0.203 | 80% |

Two things follow. First, the earlier pooled numbers (+0.09 mean, SD 0.19) were a blend:
much of the apparent log-loss improvement lived in rating-scale cells being scored as
unordered classes. Under honest per-family valuation, the rating-scale story is strong
(+0.20 rank-correlation gain over demographics, 80% of cells positive) while the
binary/nominal story is modest (+0.03 in log loss). Treat both with caution — 26
deliberately unrepresentative cells — but as variance inputs they are the real thing.
Second, the sizing memo's power analysis must be redone per family: the binary/nominal
stratum is now the power-critical one (small mean, needs ~90+ cells of its own family to
detect +0.03 at 80% power), while the rating-scale stratum is comfortably powered
(~30 cells detect half its observed effect). The type-proportional draw must therefore
guarantee enough binary/nominal questions specifically — a floor per type is no longer a
formality but the binding constraint.

The country-swap pairs split the same way: binary/nominal pairs scatter with SD 0.046
(n=80 — detecting a +0.02 adaptation effect needs only ~40 such pairs, half the pooled
estimate suggested), rating-scale pairs with SD 0.112 (n=42 — noisier per pair, but that
family's effects are larger in its own units, so its sensible target effect is larger
too). The transportability stratum's power calculation should be redone per family with
these inputs.

## Decision needed now

The pilot clears the runway. The next step we agreed on is the sizing exercise: turn the
noise numbers above into a justified choice of targets-per-survey, countries-per-target,
and the size of the heterogeneity-screened transportability stratum — replacing the
provisional "5 × 5 = 150 cells" preset. **Say the word and that sizing exercise starts;
its output is the final grid design for the main run, brought back to you before anything
is launched.** Two open sub-questions it should settle: (a) keep or drop
DeepSeek-V4-Flash given its failure modes; (b) "quick" vs "balanced" oracle fits for the
main run (3× compute for sturdier rankings).

**(c) — decided (user, 18 Aug): valuation is type-matched.** Binary/nominal answers are
valued by classification log loss, rating scales and continuous answers ONLY by rank
correlation (Spearman) — one fit per cell, two reporting families across the grid, never
pooled into a single average. The scorer now routes each cell by the measurement level
recorded in its oracle metadata, so the two layers can never disagree again. Consequences:
headline claims are stated per answer-type family (the type-stratified draw guarantees
both families are populated); the power analysis must be redone per family (the log-loss
numbers above now cover only the binary/nominal stratum; the rank-family spread comes
from the re-scored pilot); and comparisons with the previous full run's pooled number are
retired rather than maintained.
