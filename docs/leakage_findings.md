# Phase B1 — Leakage Audit Findings

> **PARTLY SUPERSEDED.** The 52 genuine / 31 degenerate / 6 leakage split was computed on
> the accuracy-era oracle, and its `--min-signal` threshold is an *accuracy* lift — under
> log loss some "degenerate" cells are likely recoverable. Two known defects:
> (1) the rule `recovery >= 0.90 AND top_importance_share >= 0.80` is a conjunction, so
> `Q67A x Gabon` was classed **genuine** despite a single feature reproducing the oracle
> perfectly (recovery 1.0000, share 0.2276);
> (2) all three `Q67A` cells are contaminated by a skip-pattern follow-up (`Q69`, asked
> only of respondents who had heard of climate change) that `detect_conditional_leakage`
> missed because the routing code was positive and survived the old cleaning rule.
> Re-run this screen after the oracle re-run. See [pipeline_audit_2026-08.md](pipeline_audit_2026-08.md) §A5.

*Generated 2026-05-31 from `analysis/leakage_audit.py`, data-backed run:*
`python analysis/leakage_audit.py --with-data --write-tex`
*Per-cell results: `outputs/leakage_audit.csv`. Rollups: `outputs/leakage_audit_summary.json`.*
*LaTeX table: `paper/generated_current_state/leakage_audit_longtable.tex`.*
*Every number below was read back from those generated files.*

## Why this audit

Several targets in the frozen grid are demographics. The concern (flagged when we got
back up to speed) was that the oracle reaches ~1.0 accuracy on these by exploiting a
single near-duplicate column — the label in disguise — which is **empirical leakage,
not predictive understanding**. The oracle already filters *lexically* similar features
(question-text cosine > 0.85), but that cannot catch features that are **empirically**
near-duplicates while being worded differently.

## Method

Two signals per cell (unique target × country; oracle is model-independent):

1. **Importance concentration** (offline): share of total positive permutation-importance
   mass held by the single top oracle feature.
2. **Single-feature predictive test** (data-backed): train the *same* downstream XGBoost
   on **only** the top oracle feature and measure how much of the oracle's lift over
   majority one feature alone recovers (`single_feature_recovery`).

Classification: `degenerate` (oracle lift < 0.03 — nothing to predict);
`leakage` (single feature recovers ≥ 0.90 of oracle lift **and** top-importance
share ≥ 0.80); `genuine` otherwise. The data-backed test is authoritative; the offline
concentration heuristic alone over-flags and is only a fallback when survey data is not
reachable.

## Headline result (data-backed)

Of 89 unique cells: **52 genuine, 31 degenerate, 6 leakage.**

The 6 leakage cells span **3 targets, all demographic**, all with single-feature
recovery ≥ 0.98:

| Survey | Target (gloss) | Top feature (gloss) | Leaky countries | recovery | imp. share |
|---|---|---|---|---|---|
| ess_wave_11 | **rtrd** (retired last 7 days) | **mnactic** (main activity last 7 days) | Austria, France, Italy | 1.00–1.02 | 0.91–0.99 |
| wvs | **Q263** (born here / immigrant?) | **Q266** (in which country born?) | Andorra, Germany | 1.00 | 1.00 |
| asianbarometer | **level** (binary demographic) | **region** | Mongolia | 1.00 | 0.90 |

The two textbook cases are confirmed in plain question text:
- **Q263** "Were you born in this country or are you an immigrant?" ← **Q266** "In which
  country were you born?" (one feature recovers 100% of the oracle's lift).
- **rtrd** "Were you retired in the past seven days?" ← **mnactic** "What was your main
  activity in the last 7 days?" (recovery 1.00).

(Recovery slightly above 1.0 for rtrd × France just means the single feature, scored by
the fixed downstream XGBoost, edges the AutoGluon oracle's top-k on that split — i.e. the
one column is the entire story.)

All 6 leakage cells are **binary** targets. Of 35 binary cells, only 16 are genuine
(13 degenerate, 6 leakage) — binary demographic targets are where both leakage and
degeneracy concentrate.

## What is NOT leakage (correcting earlier suspicions)

- **Arab Barometer has zero leakage cells** (7 genuine / 7 degenerate). In particular
  **Q1005B is not flagged** — its high-oracle cells are degenerate (no learnable signal),
  not label-in-disguise.
- **Asian Barometer SE7a × Mongolia is not leakage** — it stays genuine. (The leaky
  Asian Barometer cell is a *different* target, `level`.)

## Consequence for the headline numbers — leakage was FLATTERING the LLM

Removing the 24 leakage rows (6 cells × 2 conditions × 2 models) from the 351 valid rows:

| Metric | All valid rows (351) | Leakage removed (327) |
|---|---|---|
| mean value_over_random | 0.0252 | **0.0153** |
| mean cost_of_imperfect | 0.0645 | 0.0650 |
| share beating random | 0.547 | **0.517** |

**Interpretation (this corrects an earlier draft of this file).** Leakage cells are ones
where the LLM, too, names the obvious near-duplicate — asked what predicts "are you an
immigrant," it requests country/region of birth, which maps to the leaky feature and
scores near-ceiling. So these cells had **high** value-over-random and were **inflating**
the value-of-reasoning metric, not the cost. Stripping them:

- mean value-over-random falls ~40% (0.025 → 0.015),
- the share of cells beating random drops from 55% to ~52% (i.e. barely better than a
  coin flip),
- cost-of-imperfect is essentially unchanged.

So the corrected, leakage-free result is **weaker, not stronger**: on genuinely
predictable cells, LLM-selected features beat random by only ~1.5 accuracy points and
only about half the time. The qualitative conclusion (priors carry a little signal but
fall well short of the oracle) holds; the *magnitude* of the "value of reasoning" was
partly a demographic-duplicate artifact.

## Recommendation (decision taken: exclude + report both ways)

- **Adopted:** exclude the 6 leakage cells from pooled selection-quality summaries and
  report headline numbers with and without them (table above), with this audit as a
  standing validity check. `outputs/leakage_audit.csv` (`leakage_class == "leakage"`) is
  the canonical exclusion list for downstream analyses.
- **Still worth considering** (open, for co-authors): re-selecting targets toward
  signal-bearing **attitude/opinion** items. Note degeneracy (31/89) is a larger numeric
  problem than leakage (6/89) and also argues for signal-driven target selection rather
  than the current cardinality quota.
- **Pipeline fix (future runs):** the oracle's lexical cosine filter misses empirical
  duplicates; add an empirical guard that drops any candidate feature whose single-feature
  recovery of the target exceeds a threshold, so the oracle self-protects.

## Reproduce

```
python analysis/leakage_audit.py --with-data --write-tex
```
Needs the miniconda interpreter (pandas + autogluon + survey loader) and a valid
`DATA_CONFIG_PATH` in `.env`. Without `--with-data` it falls back to the offline
concentration heuristic, which over-flags — do not report those as confirmed leakage.
