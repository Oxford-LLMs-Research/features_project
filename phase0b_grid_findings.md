# Phase 0b Grid Findings — April 2026

Supervisor-facing summary of the full 5 targets × 5 countries × 2 conditions
Phase 0b pipeline run. All numbers in this doc come from
`outputs/grid_summary.csv` (50 rows) and the per-cell `*_disambig.json` /
`*_oracle.csv` artefacts. Reproduce with `python -m analysis.grid_analysis`.

## Setup

- **Targets:** Q47 (self-rated health), Q57 (interpersonal trust), Q199
  (political interest), Q235 (strong leader), Q164 (importance of God).
- **Countries:** Germany, Nigeria, Japan, Brazil, Egypt.
- **Conditions:** `unprompted` (no country information) and `country_provided`.
- **Model:** `deepseek-ai/DeepSeek-V3.2`, temperature 0, structured output.
- **Oracle:** permutation importance from XGBoost (n_splits=5, n_repeats=10,
  fixed hyperparameters, same as Phase 0a).
- **Downstream eval:** XGBoost 5-fold CV accuracy on (i) oracle top-k, (ii)
  model-selected k, (iii) 50 random draws of size k, where k = number of
  model features that mapped to a unique WVS variable.

**Fix applied before this run.** The first grid pass hit
`ValueError: feature_names must be unique. Duplicates found` on five cells
because disambiguation sometimes maps two LLM feature labels to the same
WVS code (e.g. both "age" and "socioeconomic status" → `Q262`). Added
defensive dedup inside `evaluate_feature_set`
([phase0b_evaluation.py](phase0b_evaluation.py)) and `compute_oracle`
([run_e2e.py](run_e2e.py)); rerunning filled all 25 cells cleanly. The v1
summary is kept at `outputs/grid_summary_v1.csv` for diffing.

## 1. Headline numbers

Across the 50 target × country × condition rows:

| Metric                                     | Value  |
|--------------------------------------------|-------:|
| Mean majority baseline                     | 0.531  |
| Mean oracle accuracy                       | 0.611  |
| Mean model accuracy                        | 0.538  |
| Mean random-k accuracy                     | 0.529  |
| Mean `cost_of_imperfect` (oracle − model)  | +0.073 |
| Mean `value_over_random` (model − random)  | +0.010 |
| Share of rows where model > random         | 0.58   |
| Share of rows where model > majority       | 0.40   |
| Share of rows where oracle > majority      | 0.80   |

The model's selections **lose about 7 accuracy points to oracle on
average** and **gain about 1 point over a random feature draw**. In more
than half of the rows the model fails to beat the majority baseline, even
though the oracle does so in 80% of cells. The experiment therefore
has a clearly measurable gap between what models ask for and what the
data actually rewards.

## 2. By target

| Target | Label               | Mean oracle | Mean model | Mean random | Cost   | Value  | Share value > 0 |
|--------|---------------------|------------:|-----------:|------------:|-------:|-------:|----------------:|
| Q164   | Importance of God   | 0.672       | 0.644      | 0.616       | 0.028  | +0.028 | 0.70            |
| Q57    | Interpersonal trust | 0.829       | 0.786      | 0.785       | 0.043  | +0.001 | 0.50            |
| Q47    | Self-rated health   | 0.483       | 0.432      | 0.436       | 0.051  | −0.004 | 0.40            |
| Q199   | Political interest  | 0.573       | 0.453      | 0.411       | 0.120  | +0.042 | 0.90            |
| Q235   | Strong leader       | 0.499       | 0.377      | 0.397       | 0.122  | −0.020 | 0.40            |

Three patterns stand out.

**Same-construct adjacency wins.** Q164 has the lowest cost (2.8pp) and
the only consistently positive value. When asked about importance of God,
the model names religion-adjacent constructs (religious affiliation,
prayer frequency) that are exactly what the oracle uses. Similarly for
Q57, where in near-ceiling countries there is very little to predict and
everything converges.

**Cross-construct empirical correlations lose.** Q199 (political
interest) and Q235 (strong leader) have the largest cost. Q199's oracle
leans on Q200 (discusses politics) and Q4 (importance of politics); the
model almost never names these and instead proposes age, education, party
affiliation, SES. Q235's oracle is dominated by Q236 (expert governance
attitudes), which appears in zero model outputs across the 10 Q235 cells.

**Q199's value is still positive while Q235's is negative.** For Q199 the
model's politically adjacent picks (party ID, civic participation) carry
enough signal to beat random. For Q235 the model defaults to political
ideology + demographics, which predict authoritarianism worse than a
random feature pull — the "value over random" is −0.02.

## 3. Condition effect (country_provided vs unprompted)

Paired across all 25 cells:

| Statistic                                        | Value    |
|--------------------------------------------------|---------:|
| Mean delta (`cp − up`) in `model_acc`            | −0.006   |
| Median delta in `model_acc`                      | +0.001   |
| Share of cells where `cp > up`                   | 0.52     |
| Mean delta in `k_mapped`                         | +0.4     |

Country conditioning makes **no meaningful difference** to downstream
accuracy. It produces very slightly longer feature lists and splits the
cells 13–12 in favour of `country_provided`. Two cells where `cp` does
noticeably worse than `up` are `Q164 × Japan` (delta −0.07, model drops
belief-in-higher-power in favour of country-specific Shinto/Buddhism
framing that has low WVS coverage) and `Q235 × Egypt` (delta −0.10,
model adds Egypt-specific "role of military/religion" features that are
dominated by the cross-national oracle's Q236).

This confirms the qualitative Phase-0 observation: country conditioning
adds **surface-level cultural features** without tracking the predictive
structure. On the current grid it is a wash-to-slight-loss.

## 4. Signal strength vs. model gap (the main finding)

Plot `signal = oracle − majority` against `gap = cost_of_imperfect` for
each of the 50 rows.

- **Pearson correlation = 0.84**
- **Spearman correlation = 0.82**
- **n = 50**

The model underperforms the oracle **specifically where there is signal
to catch**. In flat cells (Q57 Brazil/Egypt, Q164 Egypt — all majority
baselines ≥ 0.93) everything converges and cost is zero. In cells with
10–28pp of predictable variation (Q199 all countries, Q235 Egypt, Q57
Germany) the model recovers only a fraction of the oracle's lift. This
is the bullseye observation for the paper: LLM feature selection is not
random noise on top of the oracle, it is **a systematic shortfall that
scales with how much structured predictability the survey contains**.

Top 8 highest-gap rows:

| Target | Country | Condition         | Signal | Gap    |
|--------|---------|-------------------|-------:|-------:|
| Q235   | Egypt   | country_provided  | 0.218  | 0.242  |
| Q199   | Egypt   | country_provided  | 0.280  | 0.193  |
| Q199   | Egypt   | unprompted        | 0.267  | 0.192  |
| Q235   | Egypt   | unprompted        | 0.246  | 0.165  |
| Q199   | Japan   | unprompted        | 0.128  | 0.162  |
| Q199   | Nigeria | unprompted        | 0.195  | 0.161  |
| Q57    | Germany | unprompted        | 0.191  | 0.155  |
| Q57    | Germany | country_provided  | 0.198  | 0.149  |

## 5. Mapping pipeline: hit rate and unmappables

Hit rate = fraction of the k model-selected features that fall inside the
oracle's top-k for the same cell.

| Target | Label               | Pooled hit rate | Mean per cell | Total hits / total k |
|--------|---------------------|----------------:|--------------:|---------------------:|
| Q164   | Importance of God   | 0.129           | 0.133         | 8 / 62               |
| Q47    | Self-rated health   | 0.100           | 0.075         | 3 / 30               |
| Q199   | Political interest  | 0.091           | 0.083         | 6 / 66               |
| Q235   | Strong leader       | 0.033           | 0.022         | 2 / 60               |
| Q57    | Interpersonal trust | 0.020           | 0.014         | 1 / 50               |

Hit rates are **low across the board**. Q164's religion-heavy oracle is
the only target where the pipeline catches more than one in ten picks.
Q57 (trust) is a single hit in 50 attempts — the model names "prior
survey trust" rarely, and when it does the disambiguator often routes
elsewhere. Q235 catches two hits in 60 attempts, both Q239 (religiosity)
rather than the actual top-importance Q236.

Unmappable rate — fraction of LLM-requested features that return
"none" at the disambiguation step:

| Scope            | Unmappable rate |
|------------------|----------------:|
| Overall (386 requests) | 0.30       |
| Q47              | 0.58            |
| Q57              | 0.36            |
| Q235             | 0.22            |
| Q199             | 0.19            |
| Q164             | 0.15            |
| unprompted       | 0.34            |
| country_provided | 0.26            |

Q47 is the extreme case: **58% of what the model requests for health is
not measured by WVS** (chronic conditions, BMI, lifestyle, access to
care). This is a content-coverage finding more than a pipeline finding
— when the model reasons from "ideal predictors" for health, most of
those live outside the survey instrument. Q164 is the opposite: WVS has
dense religion coverage and the unmappable rate sits at 15%.

Across all 25 cells the model issued 386 feature requests. The mapper
found valid WVS codes for 271 of them, left 115 unmappable, and dropped
a further 3 as same-code duplicates. The mapping pipeline itself is not
the bottleneck — correctness of what the LLM chooses to ask for is.

## 6. What leads the model's list?

For each of the 50 cells we inspected the `feature_rank = 0` label and
checked whether it is a demographic term (age, gender, education, race,
religion, region, income, class, etc.).

- 78% of cells lead with a demographic term.
- By target: Q47 90%, Q57 100%, Q164 100%, Q199 100%, Q235 0%.

The breakdown per target:

| Target | Rank-0 feature (both conditions, across 5 countries)          |
|--------|---------------------------------------------------------------|
| Q47    | "age" × 9, "self-reported health status" × 1                  |
| Q57    | "age" × 10                                                    |
| Q199   | "age" × 10                                                    |
| Q164   | "religious affiliation" × 10                                  |
| Q235   | "political ideology / orientation" × 10                       |

It is worth stating this precisely: the model is not uniformly
demographics-anchored. It picks **a single construct-salient variable**
for its rank-0 slot. For three of the five targets that variable is age
(and age is never in the oracle top-5 for any of them). For Q164 it is
religious affiliation (which maps to Q289CS9 — category, not intensity —
and underperforms Q165 belief). For Q235 it is political ideology (Q240),
which is not in the oracle top-10 for any of the five countries.

In other words, the model's rank-0 pick is predictable, but it is
predictable from the target wording rather than from the oracle. The
rank-0 slot is a wording-anchor, not a signal-anchor.

## 7. Qualitative examples

### High cost: Q235 × Egypt, country_provided (cost = +0.242, 0 hits)

Oracle top-5: Q236 (expert governance, 0.042), Q239 (religiosity in
public life, 0.015), Q234A (0.012), Q253 (0.007), Q228 (0.006).

Model-selected (4 usable after dedup + mapping):

1. Q262 (age)
2. Q275 (education level)
3. Q71 (trust in current government institutions)
4. Q15 (religiosity and social values)

Two of the four are generic demographics. The other two are adjacent
political/religious attitudes but not the ones the oracle rewards.
Q236 — the clear top predictor — does not appear in any of the
model's seven requested features (three were filtered as unmappable).

### Low cost: Q47 × Japan, country_provided (cost = +0.019, 3 usable features)

Oracle top-3: Q46 (happiness, 0.034), Q195 (security, 0.006), Q157 (0.005).

Model-selected:

1. Q262 (age)
2. Q54 ("subjective economic hardship")
3. Q49 ("subjective well-being or life satisfaction")

Here the model's Q49 is a well-being/life-satisfaction item that is
strongly correlated with Q46 (happiness) — not a hit by our strict
top-k definition, but a near-miss that captures similar variance.
This illustrates a lurking measurement choice: our hit-rate metric
demands exact code match, but the model often picks a proximate item
from the same construct family. The raw accuracy metric does not
penalise that, which is why Q47 Japan has a very small cost despite
zero strict hits.

## 8. Open questions and decision points

The grid is complete enough to support a real conversation about the
next iteration. Ranked roughly by how much they would change the study:

1. **Is this enough evidence to move to Phase 1, or do we want one more
   Phase-0 pass?** The effect is clean and the instruments work.
   Arguments for moving: signal-vs-gap correlation of 0.84 is
   publication-grade, and the per-target story is coherent. Arguments
   for another pass: only one model family, k is small in many cells,
   hit rate is a blunt metric.
2. **Add a second model family for external validity.** DeepSeek is the
   only LLM in the grid. GPT-4o / Claude / a smaller open model would
   test whether "systematic shortfall that scales with signal" is
   model-specific or general. This is the single change that most
   strengthens the paper.
3. **Change the hit-rate metric.** Strict oracle-top-k match is harsh
   (Q47 Japan has zero hits but a 2pp cost). Consider (a) "near-miss"
   hits via oracle similarity clusters or construct families, (b)
   fraction-of-oracle-importance-captured — weight oracle top-k by
   importance and report how much of that mass the model's picks
   capture.
4. **Prompt changes.** The rank-0 slot is locked to one construct per
   target regardless of country. Two experiments worth running: (a)
   remove the "e.g. education level" anchor from the prompt — the
   current demographics-first pattern may be partly our fault; (b)
   ask the model to explicitly rank by "predictive power for population
   variation" rather than for an individual respondent — pushing it
   from clinical framing toward epidemiological framing.
5. **Scale the grid.** 5 × 5 is enough to see the effect; 10 targets
   × 10 countries would nail the signal-vs-gap correlation and let us
   test whether it holds across WVS-style vs. non-WVS-style constructs
   (e.g. wellbeing, civic participation, gender values). Medium cost.
6. **Downstream metric beyond accuracy.** For skewed targets (Q57 in
   Egypt / Brazil / Nigeria) accuracy saturates at the majority
   baseline and buries the signal. Switching to macro-F1 or AUC would
   expose variation that accuracy smooths away.
7. **Country-conditioning doesn't help on accuracy, but may still be
   informative as a qualitative instrument.** The feature lists differ
   in surface-level culture even when downstream accuracy doesn't
   move. That's worth one section in the paper but not a blocker.
8. **Q164 Egypt is a dead cell.** Majority baseline 0.977, oracle =
   model = random = 0.977. The country is homogeneously religious and
   carries no variation to predict. Consider swapping it out of the
   Phase-1 grid or flagging it as a known-null cell.

If the decision is to keep moving, the highest-leverage single next step
is **adding a second model family** — it converts every table in this
document into a within-study comparison rather than a single-point
estimate.
