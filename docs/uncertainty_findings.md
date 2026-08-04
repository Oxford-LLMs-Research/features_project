# Phase B3 — Uncertainty Quantification

> **SUPERSEDED — read before quoting.** **JSON-grid** numbers (era 1), including the
> adaptation null. See the note in `alignment_findings.md`: the free-text result differs
> in sign for Kimi. The **accuracy-era oracle** these numbers were computed against is archived at `outputs/cache/cells_accuracy_v1/`; the current oracle uses log loss on an honest fit/select/score split. Every CI here is computed on that oracle, and
> `oracle_percentile_mean` carried an order-dependent tie-break at the time (~46 of 239
> features tied at exactly 0.0 in the median cell), since fixed.
> See [pipeline_audit_2026-08.md](pipeline_audit_2026-08.md) §A1, §A3.

*Generated 2026-06-01 from `analysis/uncertainty_analysis.py --write-tex`.*
*Rollups: `outputs/uncertainty_summary.json`. TeX: `paper/generated_current_state/main_uncertainty.tex`,
`main_ci_captured_importance.tex`. All numbers read back from the generated JSON.*

## Method

Cluster bootstrap, 2000 resamples, cluster = **(survey, target)** (30 clusters). The grid
is nested (survey > target > country > condition × model); resampling whole target
clusters with replacement propagates the correlation among cells that share a target,
which an independent-row bootstrap would ignore. Reported as mean [2.5th, 97.5th
percentile]. Every metric is given three ways: **all** valid rows, **excl. leakage** (drop
the 6 B1 leakage cells), and **genuine only** (drop the 31 degenerate + 6 leakage cells).

## Headline table (mean [95% CI])

| Metric | All | Excl. leakage | Genuine only |
|---|---|---|---|
| Value over random | 0.025 [0.008, 0.046] | 0.015 [0.003, 0.029] | 0.026 [0.009, 0.046] |
| Cost of imperfect | 0.065 [0.043, 0.092] | 0.065 [0.042, 0.092] | 0.097 [0.066, 0.132] |
| Captured importance | 0.196 [0.142, 0.259] | 0.186 [0.140, 0.232] | 0.195 [0.141, 0.257] |
| Oracle percentile (mean) | 0.562 [0.527, 0.603] | 0.563 [0.527, 0.605] | 0.579 [0.530, 0.637] |
| Captured imp. — random-k baseline | 0.102 [0.079, 0.126] | 0.108 [0.085, 0.131] | 0.104 [0.081, 0.130] |
| Adaptation score (own − cross) | −0.002 [−0.032, +0.032] | −0.014 [−0.042, +0.013] | +0.003 [−0.033, +0.034] |

(Oracle percentile here is 0.562 over all 351 percentile-valued rows; B2 quoted 0.573 on
the n=332 captured-importance-valid subset. Same quantity, slightly different row set.)

## What the intervals say

**1. The LLM beats random feature selection — significantly, but by a sliver.**
Value-over-random is 0.025 [0.008, 0.046]: the CI excludes zero, so the effect is real,
but it is ~2.5 accuracy points and the lower bound is near zero. Excluding leakage it
shrinks to 0.015 [0.003, 0.029] — still excludes zero, barely. So "priors carry signal" is
statistically supported; "priors are useful" is not.

**2. The cleanest positive result: captured importance roughly doubles the matched-k
random baseline.** Observed captured importance 0.196 vs a random-k pick's 0.102, a paired
difference of **0.095 [0.041, 0.159]** (excludes zero in all three subsets). This is the
B2 caveat resolved: the model's features capture about twice the oracle importance mass
that an equal-size random draw would. It genuinely leans toward important features — it
just leans only partway, and the features are redundant enough that the edge mostly does
not survive into downstream accuracy.

**3. Cross-national adaptation is statistically indistinguishable from zero.** Adaptation
score −0.002 [−0.032, +0.032] (all), −0.014 [−0.042, +0.013] (excl. leakage), +0.003
[−0.033, +0.034] (genuine). Every interval straddles zero. The B2 "movement without fit"
reading is now on firm footing: the model changes its requests across countries, but
whether those changes fit the country's own structure better than another country's is a
coin flip. This is the strongest, most defensible negative result in the study, and it is
the one question no data-driven importance baseline could even be posed against.

**4. Leakage robustness.** Captured importance, oracle percentile, and cost-of-imperfect
are stable across all/excl-leakage/genuine subsets; only value-over-random moves
materially when leakage cells are dropped (0.025 → 0.015), consistent with B1's finding
that leakage cells flattered that specific metric. Cost-of-imperfect rises on the
genuine-only subset (0.065 → 0.097) because dropping degenerate cells removes the
zero-cost flat cells that were diluting the average.

## Recommendation for the paper

- Report the headline metrics with these CIs (not bare means).
- Lead the positive claim with **captured importance vs the random-k baseline** (0.095
  [0.041, 0.159]) — it is the clearest evidence the model's priors are non-trivially
  informative, and it pre-empts the "why not just use ML?" framing.
- State cross-national adaptation as a **null with a tight-ish interval around zero**, not
  as a weak positive. With only 3 countries/target this is low-powered; a wider country
  set is the obvious way to give Test 2 a fair chance.

## Reproduce

```
python analysis/uncertainty_analysis.py --write-tex          # 2000 boot, ~seconds
python analysis/uncertainty_analysis.py --n-boot 10000 --write-tex   # tighter CIs
```
Offline. Reads grid_summary CSVs + alignment_by_cell.csv + per-cell oracle.csv +
leakage_audit.csv.
