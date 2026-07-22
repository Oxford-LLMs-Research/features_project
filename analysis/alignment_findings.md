# Phase B2 — Selection Alignment & Cross-National Adaptation

*Generated 2026-06-01 from `analysis/alignment_analysis.py --write-tex`.*
*Per-cell: `outputs/alignment_by_cell.csv` (356 rows). Rollups: `outputs/alignment_summary.json`.*
*TeX: `paper/generated_current_state/main_alignment_overall.tex`, `main_alignment_metrics.tex`,
`main_test2_adaptation.tex`. Every number below was read back from the generated JSON.*

## Why this analysis

The matched-k accuracy horse-race (value-over-random ≈ +0.02) frames the LLM as a weak
*importance estimator*. The design doc's primary selection metric is instead **captured
importance** — the share of the oracle's permutation-importance mass that the model's
mapped features recover — and the signature test is **cross-national adaptation**. Both
are reported model-aware and leakage-aware (decision: report with and without the 6
leakage cells from B1).

## Data sources (verified schema)

- Model feature codes: `outputs/<cell>/llm__<tag>/disambig.json` — a list; per item
  `condition`, `feature_rank`, `disambig.selected_code`. Codes deduped per condition,
  `None` dropped. (`eval.json` holds only accuracy stats, **not** codes — an earlier
  draft of this script wrongly read `eval.json` and is fixed.)
- Oracle importances: `outputs/<cell>/oracle.csv` (`feature_variable`, `importance_mean`),
  same code space as `selected_code` (verified: Q263×Germany selected Q266/Q268/Q272, all
  present in that cell's oracle).
- Leakage tags: `outputs/leakage_audit.csv`.

## Metrics

- **Captured importance (CI)** = Σ importance(model features) / Σ importance(oracle top-k),
  matched k. In [0,1]; 1.0 = model picked exactly the oracle's most-important set.
- **Oracle percentile (mean)** = mean rank-percentile of the model's features within the
  oracle ordering (top = 1.0). A *random* matched-k pick sits at ≈0.50 — the chance line.

## Result 1 — selection alignment is modest, and a touch above chance

| Metric | All valid (n=332) | Excl. leakage (n=308) |
|---|---|---|
| Captured importance | **0.196** | 0.186 |
| Oracle percentile (mean) | **0.573** | 0.574 |

- DeepSeek-V3.2 CI = 0.199, Kimi-K2.5 CI = 0.193 — **near-identical across models** (as everywhere else).
- Country-provided CI = 0.210 vs unprompted CI = 0.183 — naming the country *does* nudge
  alignment up slightly (more below).
- Removing leakage cells lowers CI only a little (0.196 → 0.186).

**Reading.** The model's chosen features recover only ~20% of the predictive-importance
mass the oracle's top-k captures, and sit at the ~57th percentile of the oracle ranking
versus 50th for random. So the model's priors are **weakly informative** — it leans
slightly toward higher-importance features, but well short of the oracle and not far above
chance. This is consistent with, and a bit more sober than, the matched-k accuracy story:
on the importance-mass metric the LLM recovers about a fifth of what is there, and the
percentile is only ~7 points above a coin-flip pick.

*(Caveat: a matched-k random captured-importance baseline would let us state CI-above-random
directly; the percentile (0.573 vs 0.50) is the cleaner above-chance anchor for now.
Bootstrap CIs come in B3.)*

## Result 2 — Cross-national adaptation (Test 2): present as movement, NOT as fit

All on the country-provided condition (n = 165 adaptation cells).

| Quantity | All (cp) | Excl. leakage |
|---|---|---|
| Mean Jaccard, unprompted vs country-provided | 0.304 | — |
| Mean cross-country Jaccard (same target) | 0.363 | — |
| Mean own-country captured importance | 0.210 | — |
| Mean cross-country captured importance | 0.206 | — |
| **Adaptation score (own − cross)** | **−0.002** | −0.014 |
| Share of cells with positive adaptation | 0.406 | 0.405 |

Two sub-findings:

1. **The model changes a lot of its picks by country.** Naming the country changes most
   of the requested set (unprompted-vs-country Jaccard only 0.30 → ~70% of features
   differ), and feature sets overlap just 0.36 across different countries for the same
   target. So this is *not* a rigid universal template — there is substantial
   country-driven movement in what it asks for.

2. **But that movement does not track reality.** The model's country-specific picks align
   essentially **no better** with that country's own oracle than with other countries'
   oracles for the same target (adaptation score −0.002; only 40.6% of cells positive —
   *below* the 50% coin flip). Excluding leakage makes it slightly *worse* (−0.014, 40.5%).

**Reading.** This is the signature test — and the answer is the interesting negative: the
model adapts its feature requests across countries (lots of movement), but the adaptation
is **not** aimed in the empirically correct direction — country-specific requests fit the
target country's predictive structure no better than a different country's. This sharpens
the Paper-1 "conditional stereotyping" intuition: the model reacts to country cues, but
the reaction is essentially uncorrelated with how predictive structure actually shifts.
"Movement without fit" is a cleaner, more defensible claim than "weak adaptation."

**Power caveat.** Only 3 countries per target ⇒ each cell's cross average is over just 2
other countries, and many cells are low-signal. The −0.002 is descriptive; B3 will attach
clustered/bootstrap CIs, and the adaptation test is worth recomputing on genuine
high-signal cells only (degenerate cells inject noise into both own and cross CI).

## How this reframes the paper

- **Test 1:** report captured importance (≈0.20) and oracle percentile (≈0.57) alongside
  the accuracy horse-race. The honest composite: the model's priors carry a little signal
  (above chance on percentile) but recover only ~a fifth of the oracle's importance mass
  and barely move downstream accuracy.
- **Test 2 becomes a genuine finding, not a null:** the model *does* vary its requests by
  country (Jaccard 0.30/0.36), but the variation does **not** align with real
  cross-national structure (adaptation −0.002, 41% positive). Movement without fit.
- This is the LLM-specific contribution the horse-race hides — and Test 2 in particular is
  a question no data-driven importance baseline can even be posed against.

## Reproduce

```
python analysis/alignment_analysis.py --write-tex
```
Offline (no survey-data load). Reads `outputs/*/llm__*/disambig.json` + per-cell
`oracle.csv` + `outputs/leakage_audit.csv`.
