# Sub-item mapping — results (kimi v1)

> **Status:** results appendix for the kimi-only v1 run described in `docs/subitem_mapping.md`. Numbers computed from `outputs/subitem_mapping/kimi/diagnostics.csv` (104 cells) and `outputs/subitem_mapping/kimi/scores_kimi.csv` (618 rows; 102 paired parent↔expanded cells at each `k_spec`). Do **not** treat expanded-k VoR as a replacement for the main MiniLM arm-C headline.

*Generated from on-disk artifacts; aggregates are unit-/code-weighted as noted.*

## A. Setup reminder

This experiment asks whether treating extractor `sub_items` as separate mapping units changes map success and downstream capability, relative to parent-label mapping alone. Full protocol: `docs/subitem_mapping.md`.

| Held fixed | Value |
|------------|-------|
| Selector extracts | kimi free-text (arm **C**), reused from `outputs/format_pilot/kimi/` |
| Embedder | `all-MiniLM-L6-v2` |
| Disambiguator | nemotron |
| Retrieval | `top_n=20`, `min_similarity=0.30` (not swept; see `docs/similarity_threshold.md`) |
| Grid | 52 genuine cells × 2 conditions = **104** map cells |
| Mapping mode | dual-layer: parent codes **copied** from format_pilot; sub_items API-mapped when `|S|≥2` |
| Scoring | parent vs expanded code sets; natural `k` (`k_spec=model`) and matched `k=5` / `k=10`; `SCORE_N_DRAWS=10` |

Primary eval metrics: `value_over_random` (VoR), `model_acc`, `cost_of_imperfect`, `captured_importance`.

### Artifact paths

| Artifact | Path |
|----------|------|
| Diagnostics (per cell) | `outputs/subitem_mapping/kimi/diagnostics.csv` |
| Scores | `outputs/subitem_mapping/kimi/scores_kimi.csv` |
| Expanded maps | `outputs/subitem_mapping/kimi/maps/` |
| Manifest | `outputs/subitem_mapping/manifest.json` |
| Parent baseline (unchanged) | `outputs/format_pilot/kimi/` |
| Design | `docs/subitem_mapping.md` |

## B. Mapping quality

### B.1 Overall counts and rates

Denominators are **piped** units. Parent rates are unit-weighted (Σ mapped / Σ units). Across 104 cells: **1,782** piped parent units and **1,497** sub_item units (99 cells have ≥1 sub_item unit; ~28.8% of parents are bundled, mean `|S|` among bundled ≈ 2.91).

| Quantity | Parent | Sub_item | Expanded (union of unique codes) |
|----------|-------:|---------:|---------------------------------:|
| Requested units | 1,782 | 1,497 | 3,279 (call volume) |
| Units mapped (code ≠ none) | 1120 | 747 | 1867 (unit-level, pre-dedup) |
| Map rate (unit-weighted) | 62.9% | 49.9% | — |
| None rate | 37.1% | 50.1% | — |
| Unique codes (Σ over cells) | 1,047 | 526 | 1,357 |

| Count / inflation (per cell) | Mean | Median |
|------------------------------|-----:|-------:|
| `k_parent` | 10.07 | 10.0 |
| `k_expanded` | 13.05 | 13.5 |
| Δk (`k_expanded − k_parent`) | 2.98 | — |
| `k_inflation` (`k_expanded / k_parent`) | 1.309 | 1.240 |
| Bundling expansion factor (units) | 1.870 | — |
| Code inflation (totals) | 1.296× (1047 → 1357) | — |

**Takeaway:** sub_items map less often than parents (~50% vs ~63%). Expansion still adds codes: mean Δk ≈ +3.0 unique codes/cell (~1.30× k). Call volume rises ~1.87× if parents were remapped; v1 only API-called sub_items.

### B.2 Conditional / joint patterns (bundled parents)

Among cells with ≥1 sub_item unit (n=99), cell-mean conditional diagnostics:

| Metric | Mean |
|--------|-----:|
| Frac. sub_items map \| parent maps | 68.8% |
| Frac. sub_items map \| parent none | 21.7% |
| Parent maps, **all** sub_items miss | 5.2% |
| Parent maps, **some** sub_items miss | 28.7% |
| Parent none, **some** sub_item maps (recovery) | 17.5% |
| Jaccard(`parent_codes`, `subitem_codes`) | 0.161 (median 0.143) |

Recovery (`parent_none_some_subitem_maps` > 0) occurs in **53** / 99 cells; parent-maps-all-subitems-miss > 0 in **22** cells. Low parent↔subitem Jaccard (~0.16) means sub_item hits are largely **non-overlapping** with parent codes — not mere duplicates — even though many sub_items still fail.

### B.3 By prompt condition

| Condition | Cells | Piped | Sub units | Parent map | Sub map | k inflation | Jaccard | Recovery |
|-----------|------:|------:|----------:|-----------:|--------:|------------:|--------:|---------:|
| country_provided | 52 | 858 | 858 | 64.7% | 52.9% | 1.323× | 0.181 | 14.6% |
| unprompted | 52 | 924 | 639 | 61.1% | 45.9% | 1.269× | 0.125 | 20.4% |

Country-provided cells bundle more (more sub units) and show slightly higher parent/sub map rates; unprompted shows a bit more recovery when the parent is none.

### B.4 By survey

| Survey | Cells | Piped | Sub | Parent map | Sub map | k inflation | Δk mean | Jaccard | Recovery |
|--------|------:|------:|----:|-----------:|--------:|------------:|--------:|--------:|---------:|
| afrobarometer | 22 | 409 | 299 | 62.6% | 66.9% | 1.335× | 3.73 | 0.147 | 14.7% |
| arabbarometer | 14 | 252 | 270 | 68.3% | 43.7% | 1.383× | 4.43 | 0.134 | 16.4% |
| asianbarometer | 20 | 327 | 272 | 57.2% | 36.4% | 1.244× | 2.10 | 0.138 | 15.7% |
| ess_wave_11 | 16 | 299 | 151 | 69.2% | 51.0% | 1.187× | 2.31 | 0.100 | 20.0% |
| latinobarometer | 18 | 289 | 315 | 50.5% | 40.0% | 1.369× | 2.67 | 0.171 | 30.0% |
| wvs | 14 | 206 | 190 | 73.8% | 66.8% | 1.279× | 2.79 | 0.241 | 7.2% |

Sub_item map rates are highest on afrobarometer and wvs; lowest on asianbarometer and latinobarometer. ESS has the mildest k inflation.

### B.5 By country

| Country | Cells | Piped | Sub | Parent map | Sub map | k inflation | Jaccard | Recovery |
|---------|------:|------:|----:|-----------:|--------:|------------:|--------:|---------:|
| Andorra | 4 | 60 | 43 | 75.0% | 60.5% | 1.262× | 0.140 | 0.0% |
| Angola | 8 | 151 | 109 | 64.2% | 67.0% | 1.304× | 0.146 | 7.6% |
| Argentina | 4 | 67 | 104 | 44.8% | 35.6% | 1.536× | 0.145 | 47.0% |
| Australia | 6 | 94 | 46 | 57.4% | 34.8% | 1.118× | 0.079 | 11.1% |
| Austria | 6 | 109 | 31 | 68.8% | 54.8% | 1.127× | 0.057 | 33.3% |
| Colombia | 6 | 92 | 98 | 44.6% | 38.8% | 1.421× | 0.187 | 29.2% |
| France | 6 | 117 | 75 | 69.2% | 53.3% | 1.221× | 0.156 | 5.6% |
| Gabon | 6 | 112 | 77 | 61.6% | 59.7% | 1.294× | 0.144 | 12.5% |
| Germany | 2 | 32 | 13 | 68.8% | 38.5% | 1.053× | 0.146 | 0.0% |
| Guatemala | 8 | 130 | 113 | 57.7% | 45.1% | 1.266× | 0.172 | 21.0% |
| Indonesia | 8 | 139 | 133 | 55.4% | 33.1% | 1.314× | 0.121 | 21.3% |
| Iraq | 2 | 40 | 45 | 65.0% | 40.0% | 1.320× | 0.136 | 21.7% |
| Italy | 4 | 73 | 45 | 69.9% | 44.4% | 1.220× | 0.079 | 25.0% |
| Kuwait | 4 | 66 | 50 | 72.7% | 56.0% | 1.409× | 0.135 | 0.0% |
| Macao SAR | 8 | 114 | 134 | 74.6% | 71.6% | 1.342× | 0.314 | 12.6% |
| Mali | 8 | 146 | 113 | 61.6% | 71.7% | 1.400× | 0.151 | 21.5% |
| Mauritania | 8 | 146 | 175 | 67.1% | 41.1% | 1.387× | 0.132 | 23.4% |
| Mongolia | 6 | 94 | 93 | 59.6% | 41.9% | 1.275× | 0.221 | 12.6% |

### B.6 By question (target)

| Survey | Target | Cells | Piped | Sub | Parent map | Sub map | k inflation | Jaccard | Recovery | All-sub miss |
|--------|--------|------:|------:|----:|-----------:|--------:|------------:|--------:|---------:|-------------:|
| afrobarometer | Q15 | 4 | 79 | 54 | 73.4% | 70.4% | 1.327× | 0.112 | 38.3% | 5.0% |
| afrobarometer | Q18 | 6 | 109 | 99 | 84.4% | 80.8% | 1.411× | 0.203 | 2.8% | 4.8% |
| afrobarometer | Q67A | 6 | 97 | 76 | 63.9% | 71.1% | 1.323× | 0.126 | 6.7% | 0.0% |
| afrobarometer | Q93B | 6 | 124 | 70 | 35.5% | 40.0% | 1.195× | 0.136 | 16.7% | 19.2% |
| arabbarometer | Q104 | 2 | 35 | 51 | 62.9% | 54.9% | 1.762× | 0.167 | 32.5% | 0.0% |
| arabbarometer | Q501E_2 | 2 | 41 | 27 | 63.4% | 33.3% | 1.192× | 0.101 | 14.3% | 0.0% |
| arabbarometer | Q534_2 | 6 | 105 | 141 | 70.5% | 37.6% | 1.388× | 0.151 | 12.9% | 11.4% |
| arabbarometer | Q630 | 4 | 71 | 51 | 70.4% | 54.9% | 1.312× | 0.108 | 14.8% | 0.0% |
| asianbarometer | SE13a | 6 | 131 | 86 | 59.5% | 38.4% | 1.264× | 0.130 | 35.6% | 17.2% |
| asianbarometer | SE14a | 4 | 85 | 93 | 54.1% | 30.1% | 1.333× | 0.194 | 14.1% | 11.6% |
| asianbarometer | SE2 | 6 | 52 | 34 | 59.6% | 41.2% | 1.103× | 0.100 | 0.0% | 0.0% |
| asianbarometer | SE7a | 2 | 34 | 18 | 61.8% | 38.9% | 1.095× | 0.050 | 0.0% | 0.0% |
| asianbarometer | level | 2 | 25 | 41 | 44.0% | 41.5% | 1.455× | 0.254 | 21.7% | 0.0% |
| ess_wave_11 | bctprd | 4 | 84 | 44 | 72.6% | 59.1% | 1.169× | 0.128 | 16.7% | 0.0% |
| ess_wave_11 | eqparlv | 6 | 116 | 60 | 62.9% | 43.3% | 1.157× | 0.085 | 5.6% | 1.9% |
| ess_wave_11 | stfgov | 6 | 99 | 47 | 73.7% | 53.2% | 1.232× | 0.095 | 40.0% | 0.0% |
| latinobarometer | P61ST | 2 | 44 | 38 | 79.5% | 71.1% | 1.207× | 0.283 | 4.2% | 8.3% |
| latinobarometer | P8STGBS | 6 | 134 | 203 | 54.5% | 36.0% | 1.471× | 0.220 | 22.0% | 17.9% |
| latinobarometer | S20.A | 4 | 67 | 44 | 43.3% | 40.9% | 1.080× | 0.298 | 17.5% | 5.0% |
| latinobarometer | SEXO | 6 | 44 | 30 | 20.5% | 26.7% | 2.000× | 0.000 | 60.0% | 0.0% |
| wvs | Q205 | 6 | 97 | 60 | 60.8% | 41.7% | 1.096× | 0.207 | 3.3% | 0.0% |
| wvs | Q242 | 2 | 33 | 36 | 87.9% | 91.7% | 1.560× | 0.156 | 5.0% | 0.0% |
| wvs | Q263 | 2 | 16 | 30 | 62.5% | 73.3% | 1.300× | 0.597 | 29.2% | 0.0% |
| wvs | Q40 | 4 | 60 | 64 | 90.0% | 73.4% | 1.321× | 0.155 | 3.1% | 0.0% |

Notable extremes: **SEXO** (latinobarometer) has very low parent map (~21%) and the highest recovery rate among questions (~60% of bundled parents none with ≥1 sub_item hit) plus high k inflation when any codes appear; **Q93B** has low parent map (~36%) and elevated all-sub-miss; **Q18** / **Q242** / **Q40** show high parent and sub map rates.

### B.7 Per-cell mapping summary

Compact per-cell table (104 rows). `sub_map` is blank when `n_sub=0`. `k_inf` is blank when `k_parent=0` (2 SEXO country_provided cells).

| Survey | Target | Country | Cond. | Piped | Sub | Parent map | Sub map | k_p | k_e | k_inf | Jaccard | Recover | All-miss |
|--------|--------|---------|-------|------:|----:|-----------:|--------:|---:|---:|------:|--------:|--------:|---------:|
| afrobarometer | Q15 | Angola | cp | 19 | 28 | 84.2% | 67.9% | 16 | 21 | 1.312 | 0.286 | 0.0% | 0.0% |
| afrobarometer | Q15 | Angola | up | 24 | 9 | 66.7% | 55.6% | 12 | 15 | 1.250 | 0.067 | 33.3% | 0.0% |
| afrobarometer | Q15 | Mali | cp | 20 | 14 | 80.0% | 78.6% | 15 | 21 | 1.400 | 0.095 | 20.0% | 20.0% |
| afrobarometer | Q15 | Mali | up | 16 | 3 | 62.5% | 100.0% | 9 | 12 | 1.333 | 0.000 | 100.0% | 0.0% |
| afrobarometer | Q18 | Angola | cp | 16 | 22 | 93.8% | 81.8% | 14 | 24 | 1.714 | 0.167 | 0.0% | 0.0% |
| afrobarometer | Q18 | Angola | up | 18 | 17 | 88.9% | 70.6% | 16 | 22 | 1.375 | 0.182 | 0.0% | 14.3% |
| afrobarometer | Q18 | Gabon | cp | 25 | 14 | 80.0% | 78.6% | 19 | 24 | 1.263 | 0.167 | 16.7% | 0.0% |
| afrobarometer | Q18 | Gabon | up | 17 | 10 | 82.4% | 70.0% | 14 | 17 | 1.214 | 0.235 | 0.0% | 0.0% |
| afrobarometer | Q18 | Mali | cp | 15 | 19 | 73.3% | 100.0% | 11 | 17 | 1.545 | 0.294 | 0.0% | 0.0% |
| afrobarometer | Q18 | Mali | up | 18 | 17 | 88.9% | 76.5% | 16 | 23 | 1.438 | 0.174 | 0.0% | 14.3% |
| afrobarometer | Q67A | Angola | cp | 13 | 10 | 76.9% | 80.0% | 10 | 12 | 1.200 | 0.167 | 0.0% | 0.0% |
| afrobarometer | Q67A | Angola | up | 17 | 0 | 64.7% | — | 11 | 11 | 1.000 | 0.000 | — | — |
| afrobarometer | Q67A | Gabon | cp | 13 | 22 | 84.6% | 72.7% | 11 | 19 | 1.727 | 0.211 | 0.0% | 0.0% |
| afrobarometer | Q67A | Gabon | up | 17 | 0 | 52.9% | — | 9 | 9 | 1.000 | 0.000 | — | — |
| afrobarometer | Q67A | Mali | cp | 20 | 40 | 45.0% | 65.0% | 9 | 18 | 2.000 | 0.222 | 26.7% | 0.0% |
| afrobarometer | Q67A | Mali | up | 17 | 4 | 70.6% | 100.0% | 12 | 13 | 1.083 | 0.154 | 0.0% | 0.0% |
| afrobarometer | Q93B | Angola | cp | 19 | 0 | 26.3% | — | 5 | 5 | 1.000 | 0.000 | — | — |
| afrobarometer | Q93B | Angola | up | 25 | 23 | 32.0% | 47.8% | 8 | 10 | 1.250 | 0.300 | 12.5% | 0.0% |
| afrobarometer | Q93B | Gabon | cp | 15 | 8 | 40.0% | 12.5% | 6 | 7 | 1.167 | 0.000 | 33.3% | 33.3% |
| afrobarometer | Q93B | Gabon | up | 25 | 23 | 36.0% | 47.8% | 9 | 12 | 1.333 | 0.250 | 12.5% | 12.5% |
| afrobarometer | Q93B | Mali | cp | 17 | 7 | 47.1% | 42.9% | 7 | 8 | 1.143 | 0.125 | 0.0% | 50.0% |
| afrobarometer | Q93B | Mali | up | 23 | 9 | 34.8% | 22.2% | 6 | 7 | 1.167 | 0.143 | 25.0% | 0.0% |
| arabbarometer | Q104 | Mauritania | cp | 16 | 36 | 75.0% | 52.8% | 11 | 18 | 1.636 | 0.333 | 25.0% | 0.0% |
| arabbarometer | Q104 | Mauritania | up | 19 | 15 | 52.6% | 60.0% | 10 | 19 | 1.900 | 0.000 | 40.0% | 0.0% |
| arabbarometer | Q501E_2 | Mauritania | cp | 22 | 20 | 54.5% | 35.0% | 12 | 17 | 1.417 | 0.059 | 28.6% | 0.0% |
| arabbarometer | Q501E_2 | Mauritania | up | 19 | 7 | 73.7% | 28.6% | 14 | 14 | 1.000 | 0.143 | 0.0% | 0.0% |
| arabbarometer | Q534_2 | Iraq | cp | 23 | 37 | 65.2% | 45.9% | 15 | 22 | 1.467 | 0.273 | 10.0% | 0.0% |
| arabbarometer | Q534_2 | Iraq | up | 17 | 8 | 64.7% | 12.5% | 10 | 11 | 1.100 | 0.000 | 33.3% | 0.0% |
| arabbarometer | Q534_2 | Kuwait | cp | 21 | 7 | 66.7% | 14.3% | 11 | 11 | 1.000 | 0.091 | 0.0% | 0.0% |
| arabbarometer | Q534_2 | Kuwait | up | 8 | 23 | 100.0% | 69.6% | 8 | 19 | 2.375 | 0.263 | 0.0% | 0.0% |
| arabbarometer | Q534_2 | Mauritania | cp | 20 | 16 | 75.0% | 18.8% | 15 | 16 | 1.067 | 0.062 | 20.0% | 40.0% |
| arabbarometer | Q534_2 | Mauritania | up | 16 | 50 | 68.8% | 30.0% | 8 | 14 | 1.750 | 0.214 | 14.3% | 28.6% |
| arabbarometer | Q630 | Kuwait | cp | 20 | 16 | 70.0% | 43.8% | 13 | 17 | 1.308 | 0.118 | 0.0% | 0.0% |
| arabbarometer | Q630 | Kuwait | up | 17 | 4 | 70.6% | 100.0% | 12 | 15 | 1.250 | 0.067 | 0.0% | 0.0% |
| arabbarometer | Q630 | Mauritania | cp | 17 | 27 | 70.6% | 48.1% | 12 | 17 | 1.417 | 0.176 | 9.1% | 0.0% |
| arabbarometer | Q630 | Mauritania | up | 17 | 4 | 70.6% | 100.0% | 11 | 14 | 1.273 | 0.071 | 50.0% | 0.0% |
| asianbarometer | SE13a | Australia | cp | 19 | 13 | 57.9% | 30.8% | 10 | 11 | 1.100 | 0.273 | 0.0% | 20.0% |
| asianbarometer | SE13a | Australia | up | 24 | 8 | 37.5% | 25.0% | 9 | 10 | 1.111 | 0.100 | 66.7% | 0.0% |
| asianbarometer | SE13a | Indonesia | cp | 22 | 17 | 77.3% | 47.1% | 15 | 22 | 1.467 | 0.045 | 16.7% | 16.7% |
| asianbarometer | SE13a | Indonesia | up | 24 | 9 | 58.3% | 33.3% | 14 | 16 | 1.143 | 0.000 | 66.7% | 33.3% |
| asianbarometer | SE13a | Mongolia | cp | 18 | 31 | 66.7% | 45.2% | 10 | 17 | 1.700 | 0.294 | 30.0% | 0.0% |
| asianbarometer | SE13a | Mongolia | up | 24 | 8 | 62.5% | 25.0% | 14 | 15 | 1.071 | 0.067 | 33.3% | 33.3% |
| asianbarometer | SE14a | Indonesia | cp | 27 | 32 | 44.4% | 18.8% | 10 | 14 | 1.400 | 0.143 | 27.3% | 9.1% |
| asianbarometer | SE14a | Indonesia | up | 21 | 20 | 61.9% | 25.0% | 10 | 14 | 1.400 | 0.071 | 16.7% | 0.0% |
| asianbarometer | SE14a | Mongolia | cp | 16 | 17 | 37.5% | 52.9% | 6 | 8 | 1.333 | 0.375 | 0.0% | 0.0% |
| asianbarometer | SE14a | Mongolia | up | 21 | 24 | 71.4% | 33.3% | 13 | 16 | 1.231 | 0.188 | 12.5% | 37.5% |
| asianbarometer | SE2 | Australia | cp | 10 | 4 | 90.0% | 75.0% | 7 | 9 | 1.286 | 0.000 | 0.0% | 0.0% |
| asianbarometer | SE2 | Australia | up | 7 | 3 | 57.1% | 0.0% | 4 | 4 | 1.000 | 0.000 | 0.0% | 0.0% |
| asianbarometer | SE2 | Indonesia | cp | 9 | 7 | 55.6% | 71.4% | 5 | 5 | 1.000 | 0.200 | 0.0% | 0.0% |
| asianbarometer | SE2 | Indonesia | up | 11 | 7 | 45.5% | 0.0% | 5 | 5 | 1.000 | 0.000 | 0.0% | 0.0% |
| asianbarometer | SE2 | Mongolia | cp | 8 | 10 | 50.0% | 60.0% | 4 | 5 | 1.250 | 0.400 | 0.0% | 0.0% |
| asianbarometer | SE2 | Mongolia | up | 7 | 3 | 57.1% | 0.0% | 4 | 4 | 1.000 | 0.000 | 0.0% | 0.0% |
| asianbarometer | SE7a | Australia | cp | 15 | 5 | 66.7% | 100.0% | 10 | 10 | 1.000 | 0.100 | 0.0% | 0.0% |
| asianbarometer | SE7a | Australia | up | 19 | 13 | 57.9% | 15.4% | 11 | 13 | 1.182 | 0.000 | 0.0% | 0.0% |
| asianbarometer | level | Indonesia | cp | 19 | 31 | 31.6% | 41.9% | 6 | 9 | 1.500 | 0.222 | 10.0% | 0.0% |
| asianbarometer | level | Indonesia | up | 6 | 10 | 83.3% | 40.0% | 5 | 7 | 1.400 | 0.286 | 33.3% | 0.0% |
| ess_wave_11 | bctprd | Austria | cp | 20 | 8 | 70.0% | 50.0% | 14 | 15 | 1.071 | 0.067 | 33.3% | 0.0% |
| ess_wave_11 | bctprd | Austria | up | 22 | 8 | 68.2% | 100.0% | 15 | 20 | 1.333 | 0.100 | 33.3% | 0.0% |
| ess_wave_11 | bctprd | France | cp | 25 | 23 | 76.0% | 39.1% | 17 | 19 | 1.118 | 0.211 | 0.0% | 0.0% |
| ess_wave_11 | bctprd | France | up | 17 | 5 | 76.5% | 100.0% | 13 | 15 | 1.154 | 0.133 | 0.0% | 0.0% |
| ess_wave_11 | eqparlv | Austria | cp | 15 | 7 | 73.3% | 14.3% | 11 | 12 | 1.091 | 0.000 | 0.0% | 0.0% |
| ess_wave_11 | eqparlv | Austria | up | 19 | 3 | 63.2% | 33.3% | 10 | 10 | 1.000 | 0.100 | 0.0% | 0.0% |
| ess_wave_11 | eqparlv | France | cp | 21 | 2 | 57.1% | 50.0% | 12 | 13 | 1.083 | 0.000 | 0.0% | 0.0% |
| ess_wave_11 | eqparlv | France | up | 20 | 24 | 50.0% | 54.2% | 9 | 14 | 1.556 | 0.286 | 33.3% | 11.1% |
| ess_wave_11 | eqparlv | Italy | cp | 20 | 5 | 80.0% | 60.0% | 16 | 18 | 1.125 | 0.056 | 0.0% | 0.0% |
| ess_wave_11 | eqparlv | Italy | up | 21 | 19 | 57.1% | 36.8% | 12 | 14 | 1.167 | 0.071 | 0.0% | 0.0% |
| ess_wave_11 | stfgov | Austria | cp | 15 | 0 | 73.3% | — | 10 | 10 | 1.000 | 0.000 | — | — |
| ess_wave_11 | stfgov | Austria | up | 18 | 5 | 66.7% | 60.0% | 11 | 13 | 1.182 | 0.077 | 100.0% | 0.0% |
| ess_wave_11 | stfgov | France | cp | 16 | 11 | 87.5% | 45.5% | 14 | 17 | 1.214 | 0.118 | 0.0% | 0.0% |
| ess_wave_11 | stfgov | France | up | 18 | 10 | 72.2% | 70.0% | 12 | 16 | 1.333 | 0.188 | 0.0% | 0.0% |
| ess_wave_11 | stfgov | Italy | cp | 14 | 11 | 64.3% | 36.4% | 9 | 13 | 1.444 | 0.000 | 100.0% | 0.0% |
| ess_wave_11 | stfgov | Italy | up | 18 | 10 | 77.8% | 60.0% | 13 | 16 | 1.231 | 0.188 | 0.0% | 0.0% |
| latinobarometer | P61ST | Guatemala | cp | 21 | 34 | 71.4% | 67.6% | 13 | 19 | 1.462 | 0.316 | 8.3% | 16.7% |
| latinobarometer | P61ST | Guatemala | up | 23 | 4 | 87.0% | 100.0% | 16 | 16 | 1.000 | 0.250 | 0.0% | 0.0% |
| latinobarometer | P8STGBS | Argentina | cp | 24 | 52 | 54.2% | 34.6% | 13 | 19 | 1.462 | 0.263 | 16.7% | 22.2% |
| latinobarometer | P8STGBS | Argentina | up | 29 | 42 | 51.7% | 38.1% | 13 | 19 | 1.462 | 0.316 | 21.4% | 21.4% |
| latinobarometer | P8STGBS | Colombia | cp | 17 | 22 | 64.7% | 45.5% | 10 | 15 | 1.500 | 0.267 | 28.6% | 0.0% |
| latinobarometer | P8STGBS | Colombia | up | 29 | 42 | 48.3% | 38.1% | 14 | 21 | 1.500 | 0.238 | 21.4% | 21.4% |
| latinobarometer | P8STGBS | Guatemala | cp | 20 | 14 | 70.0% | 21.4% | 13 | 14 | 1.077 | 0.071 | 16.7% | 33.3% |
| latinobarometer | P8STGBS | Guatemala | up | 15 | 31 | 40.0% | 32.3% | 5 | 12 | 2.400 | 0.167 | 27.3% | 9.1% |
| latinobarometer | S20.A | Colombia | cp | 13 | 11 | 46.2% | 54.5% | 5 | 6 | 1.200 | 0.333 | 0.0% | 0.0% |
| latinobarometer | S20.A | Colombia | up | 20 | 10 | 40.0% | 30.0% | 7 | 7 | 1.000 | 0.286 | 25.0% | 0.0% |
| latinobarometer | S20.A | Guatemala | cp | 15 | 13 | 46.7% | 46.2% | 6 | 7 | 1.167 | 0.286 | 20.0% | 20.0% |
| latinobarometer | S20.A | Guatemala | up | 19 | 10 | 42.1% | 30.0% | 7 | 7 | 1.000 | 0.286 | 25.0% | 0.0% |
| latinobarometer | SEXO | Argentina | cp | 4 | 3 | 0.0% | 33.3% | 0 | 1 | — | 0.000 | 100.0% | 0.0% |
| latinobarometer | SEXO | Argentina | up | 10 | 7 | 20.0% | 28.6% | 2 | 4 | 2.000 | 0.000 | 50.0% | 0.0% |
| latinobarometer | SEXO | Colombia | cp | 3 | 6 | 0.0% | 16.7% | 0 | 1 | — | 0.000 | 50.0% | 0.0% |
| latinobarometer | SEXO | Colombia | up | 10 | 7 | 20.0% | 28.6% | 2 | 4 | 2.000 | 0.000 | 50.0% | 0.0% |
| latinobarometer | SEXO | Guatemala | cp | 7 | 0 | 42.9% | — | 2 | 2 | 1.000 | 0.000 | — | — |
| latinobarometer | SEXO | Guatemala | up | 10 | 7 | 20.0% | 28.6% | 2 | 4 | 2.000 | 0.000 | 50.0% | 0.0% |
| wvs | Q205 | Andorra | cp | 16 | 9 | 56.2% | 44.4% | 8 | 8 | 1.000 | 0.250 | 0.0% | 0.0% |
| wvs | Q205 | Andorra | up | 17 | 6 | 64.7% | 33.3% | 9 | 10 | 1.111 | 0.100 | 0.0% | 0.0% |
| wvs | Q205 | Germany | cp | 15 | 7 | 80.0% | 28.6% | 11 | 11 | 1.000 | 0.182 | 0.0% | 0.0% |
| wvs | Q205 | Germany | up | 17 | 6 | 58.8% | 50.0% | 8 | 9 | 1.125 | 0.111 | 0.0% | 0.0% |
| wvs | Q205 | Macao SAR | cp | 15 | 26 | 53.3% | 50.0% | 8 | 10 | 1.250 | 0.600 | 20.0% | 0.0% |
| wvs | Q205 | Macao SAR | up | 17 | 6 | 52.9% | 16.7% | 8 | 9 | 1.125 | 0.000 | 0.0% | 0.0% |
| wvs | Q242 | Macao SAR | cp | 16 | 28 | 81.2% | 100.0% | 11 | 25 | 2.273 | 0.240 | 10.0% | 0.0% |
| wvs | Q242 | Macao SAR | up | 17 | 8 | 94.1% | 62.5% | 14 | 14 | 1.000 | 0.071 | 0.0% | 0.0% |
| wvs | Q263 | Macao SAR | cp | 6 | 16 | 66.7% | 81.2% | 4 | 4 | 1.000 | 0.750 | 25.0% | 0.0% |
| wvs | Q263 | Macao SAR | up | 10 | 14 | 60.0% | 64.3% | 6 | 9 | 1.500 | 0.444 | 33.3% | 0.0% |
| wvs | Q40 | Andorra | cp | 10 | 15 | 80.0% | 60.0% | 8 | 12 | 1.500 | 0.167 | 0.0% | 0.0% |
| wvs | Q40 | Andorra | up | 17 | 13 | 100.0% | 84.6% | 17 | 23 | 1.353 | 0.043 | 0.0% | 0.0% |
| wvs | Q40 | Macao SAR | cp | 16 | 25 | 81.2% | 72.0% | 13 | 17 | 1.308 | 0.353 | 12.5% | 0.0% |
| wvs | Q40 | Macao SAR | up | 17 | 11 | 94.1% | 81.8% | 15 | 18 | 1.200 | 0.056 | 0.0% | 0.0% |

## C. Capability / score aggregates

Scores compare `k_mode=parent` vs `k_mode=expanded` within the same cell. Paired n=**102** at each `k_spec` (2 cells have `k_parent=0` — latinobarometer `SEXO` Argentina/Colombia `country_provided` — and lack parent score rows; expanded-only rows for those cells are excluded from paired Δ). Embedding / arm / disambiguator are fixed as in §A.

### C.1 Overall means and Δ (expanded − parent)

#### Natural k (`k_spec=model`)

| Metric | Parent mean | Expanded mean | Δ mean | Median Δ | #Δ>0 / #Δ<0 / #Δ=0 |
|--------|------------:|--------------:|-------:|---------:|-------------------:|
| VoR | 0.0561 | 0.0613 | +0.0052 | +0.0000 | 44 / 37 / 21 |
| Model acc | 0.4807 | 0.4915 | +0.0107 | +0.0000 | 50 / 28 / 24 |
| Cost of imperfect | 0.0654 | 0.0567 | -0.0087 | +0.0000 | 31 / 47 / 24 |
| Captured importance | 0.2539 | 0.2828 | +0.0289 | +0.0000 | 32 / 44 / 26 |
| k (unique codes used) | 10.1275 | 13.0588 | +2.9314 | +2.0000 | 81 / 0 / 21 |

#### Matched k = 5 (`k_spec=k5`)

| Metric | Parent mean | Expanded mean | Δ mean | Median Δ | #Δ>0 / #Δ<0 / #Δ=0 |
|--------|------------:|--------------:|-------:|---------:|-------------------:|
| VoR | 0.0303 | 0.0341 | +0.0038 | +0.0000 | 29 / 30 / 43 |
| Model acc | 0.4432 | 0.4472 | +0.0040 | +0.0000 | 29 / 30 / 43 |
| Cost of imperfect | 0.0925 | 0.0886 | -0.0039 | +0.0000 | 30 / 29 / 43 |
| Captured importance | 0.2125 | 0.2128 | +0.0004 | +0.0000 | 18 / 26 / 58 |
| k (after truncate/pad) | 4.7843 | 4.8627 | +0.0784 | +0.0000 | 7 / 2 / 93 |

#### Matched k = 10 (`k_spec=k10`)

| Metric | Parent mean | Expanded mean | Δ mean | Median Δ | #Δ>0 / #Δ<0 / #Δ=0 |
|--------|------------:|--------------:|-------:|---------:|-------------------:|
| VoR | 0.0482 | 0.0469 | -0.0013 | +0.0000 | 41 / 37 / 24 |
| Model acc | 0.4694 | 0.4687 | -0.0007 | +0.0000 | 39 / 38 / 25 |
| Cost of imperfect | 0.0721 | 0.0719 | -0.0002 | +0.0000 | 35 / 43 / 24 |
| Captured importance | 0.2415 | 0.2512 | +0.0097 | +0.0000 | 28 / 34 / 40 |
| k (after truncate/pad) | 8.4608 | 8.9118 | +0.4510 | +0.0000 | 31 / 3 / 68 |

**Budget vs quality:** at natural k, expanded sets are larger (mean Δk = +2.93) and captured importance rises by +0.0289, but VoR only moves +0.0052. At matched k=5, captured-importance Δ collapses to +0.0004; VoR Δ stays small (+0.0038). At k=10, VoR Δ is near zero / slightly negative (-0.0013). Expansion does not deliver a clear equal-budget capability gain.

### C.2 Δ VoR by survey

| Survey | n | Natural Δ VoR | k=5 Δ VoR | k=10 Δ VoR |
|--------|--:|--------------:|----------:|-----------:|
| afrobarometer | 22 | -0.0015 | -0.0019 | +0.0043 |
| arabbarometer | 14 | +0.0067 | +0.0297 | -0.0135 |
| asianbarometer | 20 | +0.0047 | +0.0095 | +0.0001 |
| ess_wave_11 | 16 | +0.0067 | -0.0003 | +0.0028 |
| latinobarometer | 16 | +0.0091 | -0.0130 | -0.0114 |
| wvs | 14 | +0.0088 | +0.0023 | +0.0072 |

### C.3 Δ VoR by country (natural k)

| Country | n | Parent VoR | Expanded VoR | Δ VoR | #+/−/0 | Mean Δk |
|---------|--:|-----------:|-------------:|------:|-------:|--------:|
| Andorra | 4 | 0.0860 | 0.0935 | +0.0074 | 2/1/1 | 2.75 |
| Angola | 8 | 0.0091 | 0.0126 | +0.0036 | 4/2/2 | 3.50 |
| Argentina | 3 | 0.0568 | 0.0581 | +0.0013 | 1/2/0 | 4.67 |
| Australia | 6 | 0.0183 | 0.0320 | +0.0137 | 4/0/2 | 1.00 |
| Austria | 6 | 0.0485 | 0.0513 | +0.0028 | 2/2/2 | 1.50 |
| Colombia | 5 | 0.0385 | 0.0495 | +0.0111 | 2/2/1 | 3.00 |
| France | 6 | 0.0753 | 0.0813 | +0.0061 | 3/3/0 | 2.83 |
| Gabon | 6 | 0.0350 | 0.0368 | +0.0019 | 3/2/1 | 3.33 |
| Germany | 2 | 0.1459 | 0.1778 | +0.0319 | 1/0/1 | 0.50 |
| Guatemala | 8 | 0.0413 | 0.0521 | +0.0108 | 4/1/3 | 1.88 |
| Indonesia | 8 | 0.0577 | 0.0511 | -0.0066 | 1/5/2 | 2.75 |
| Iraq | 2 | 0.2241 | 0.2616 | +0.0375 | 1/1/0 | 4.00 |
| Italy | 4 | 0.1143 | 0.1277 | +0.0134 | 4/0/0 | 2.75 |
| Kuwait | 4 | 0.1600 | 0.1559 | -0.0041 | 1/2/1 | 4.25 |
| Macao SAR | 8 | 0.0971 | 0.1008 | +0.0037 | 2/4/2 | 3.38 |
| Mali | 8 | -0.0007 | -0.0098 | -0.0092 | 2/6/0 | 4.25 |
| Mauritania | 8 | 0.0394 | 0.0439 | +0.0045 | 2/4/2 | 3.75 |
| Mongolia | 6 | 0.0266 | 0.0375 | +0.0109 | 5/0/1 | 2.33 |

### C.4 Δ VoR by question (natural k)

| Target | n | Parent VoR | Expanded VoR | Δ VoR | #+/−/0 |
|--------|--:|-----------:|-------------:|------:|-------:|
| P61ST | 2 | -0.0099 | 0.0006 | +0.0105 | 1/0/1 |
| P8STGBS | 6 | 0.0717 | 0.0869 | +0.0152 | 3/3/0 |
| Q104 | 2 | 0.0151 | 0.0616 | +0.0464 | 1/1/0 |
| Q15 | 4 | 0.0306 | 0.0165 | -0.0140 | 1/3/0 |
| Q18 | 6 | 0.0176 | 0.0145 | -0.0032 | 3/3/0 |
| Q205 | 6 | 0.1401 | 0.1574 | +0.0173 | 3/1/2 |
| Q242 | 2 | 0.0312 | 0.0544 | +0.0232 | 1/0/1 |
| Q263 | 2 | 0.1979 | 0.1941 | -0.0038 | 0/1/1 |
| Q40 | 4 | 0.0283 | 0.0234 | -0.0049 | 1/3/0 |
| Q501E_2 | 2 | 0.0052 | -0.0198 | -0.0250 | 0/1/1 |
| Q534_2 | 6 | 0.2140 | 0.2227 | +0.0087 | 1/3/2 |
| Q630 | 4 | 0.0199 | 0.0197 | -0.0002 | 2/2/0 |
| Q67A | 6 | -0.1283 | -0.1308 | -0.0025 | 2/2/2 |
| Q93B | 6 | 0.1365 | 0.1459 | +0.0094 | 3/2/1 |
| S20.A | 4 | 0.0618 | 0.0599 | -0.0019 | 1/1/2 |
| SE13a | 6 | 0.0390 | 0.0455 | +0.0065 | 4/2/0 |
| SE14a | 4 | 0.0619 | 0.0665 | +0.0046 | 3/1/0 |
| SE2 | 6 | -0.0078 | -0.0003 | +0.0076 | 2/0/4 |
| SE7a | 2 | 0.0101 | 0.0206 | +0.0105 | 1/0/1 |
| SEXO | 4 | 0.0088 | 0.0191 | +0.0103 | 2/1/1 |
| bctprd | 4 | 0.0540 | 0.0659 | +0.0119 | 3/1/0 |
| eqparlv | 6 | 0.0398 | 0.0475 | +0.0077 | 3/2/1 |
| level | 2 | 0.1381 | 0.1237 | -0.0144 | 0/2/0 |
| stfgov | 6 | 0.1241 | 0.1263 | +0.0021 | 3/2/1 |

### C.5 Multi-metric by survey (natural k)

| Survey | n | Δ VoR | Δ model_acc | Δ cost_imp | Δ capt.imp | Δk |
|--------|--:|------:|------------:|-----------:|-----------:|---:|
| afrobarometer | 22 | -0.0015 | +0.0075 | -0.0045 | +0.0321 | +3.73 |
| arabbarometer | 14 | +0.0067 | +0.0229 | -0.0130 | -0.0050 | +3.93 |
| asianbarometer | 20 | +0.0047 | +0.0074 | -0.0079 | -0.0110 | +2.10 |
| ess_wave_11 | 16 | +0.0067 | +0.0058 | -0.0044 | +0.0404 | +2.31 |
| latinobarometer | 16 | +0.0091 | +0.0128 | -0.0177 | +0.0335 | +2.75 |
| wvs | 14 | +0.0088 | +0.0118 | -0.0068 | +0.0964 | +2.79 |

### C.6 Extreme cells (natural-k Δ VoR)

Largest **negative** Δ VoR (expanded − parent):

| Survey | Target | Country | Cond. | Parent VoR | Expanded VoR | Δ VoR | Δk | Δ capt.imp |
|--------|--------|---------|-------|-----------:|-------------:|------:|---:|-----------:|
| arabbarometer | Q501E_2 | Mauritania | cp | 0.0083 | -0.0417 | -0.0500 | +5 | -0.0054 |
| arabbarometer | Q104 | Mauritania | cp | 0.1235 | 0.0762 | -0.0473 | +6 | -0.0404 |
| afrobarometer | Q15 | Mali | cp | 0.0307 | -0.0005 | -0.0312 | +6 | -0.0084 |
| latinobarometer | SEXO | Colombia | up | -0.0153 | -0.0439 | -0.0286 | +2 | +0.0861 |
| afrobarometer | Q18 | Gabon | up | 0.0534 | 0.0261 | -0.0273 | +3 | -0.0513 |
| latinobarometer | P8STGBS | Argentina | up | 0.0989 | 0.0748 | -0.0241 | +6 | -0.0592 |
| asianbarometer | level | Indonesia | up | 0.1354 | 0.1123 | -0.0231 | +2 | -0.0697 |
| afrobarometer | Q67A | Mali | up | -0.1435 | -0.1648 | -0.0213 | +1 | +0.0000 |
| afrobarometer | Q18 | Mali | cp | 0.0160 | -0.0048 | -0.0208 | +6 | +0.0335 |
| arabbarometer | Q534_2 | Kuwait | up | 0.3161 | 0.2956 | -0.0205 | +10 | -0.0204 |

Largest **positive** Δ VoR:

| Survey | Target | Country | Cond. | Parent VoR | Expanded VoR | Δ VoR | Δk | Δ capt.imp |
|--------|--------|---------|-------|-----------:|-------------:|------:|---:|-----------:|
| arabbarometer | Q104 | Mauritania | up | -0.0932 | 0.0469 | +0.1401 | +7 | +0.0197 |
| latinobarometer | P8STGBS | Colombia | up | -0.0008 | 0.0812 | +0.0820 | +7 | +0.3479 |
| arabbarometer | Q534_2 | Iraq | cp | 0.1802 | 0.2581 | +0.0779 | +7 | -0.0416 |
| wvs | Q205 | Germany | up | 0.1508 | 0.2146 | +0.0638 | +1 | +0.6276 |
| latinobarometer | SEXO | Argentina | up | -0.0235 | 0.0246 | +0.0481 | +2 | +0.3103 |
| wvs | Q242 | Macao SAR | cp | -0.0124 | 0.0339 | +0.0463 | +14 | +0.0911 |
| afrobarometer | Q93B | Gabon | up | 0.1545 | 0.1986 | +0.0441 | +3 | +0.4601 |
| latinobarometer | P8STGBS | Guatemala | up | -0.0016 | 0.0393 | +0.0409 | +7 | +0.0308 |
| asianbarometer | SE2 | Australia | cp | 0.0218 | 0.0605 | +0.0387 | +2 | -0.0468 |
| afrobarometer | Q93B | Angola | up | 0.0961 | 0.1280 | +0.0319 | +2 | +0.0333 |

Cell-level VoR is identical for parent and expanded in **21** / 102 paired cells; mean absolute Δ VoR ≈ 0.0156.

### C.7 Full cell-level Δ VoR (natural k)

| Survey | Target | Country | Cond. | VoR_p | VoR_e | Δ VoR | k_p | k_e | Δk | CI_p | CI_e | Δ CI |
|--------|--------|---------|-------|------:|------:|------:|----:|----:|----:|-----:|-----:|-----:|
| afrobarometer | Q15 | Angola | cp | 0.0218 | 0.0057 | -0.0161 | 16 | 21 | +5 | 0.1280 | 0.1597 | +0.0317 |
| afrobarometer | Q15 | Angola | up | 0.0434 | 0.0480 | +0.0046 | 12 | 15 | +3 | 0.1779 | 0.1767 | -0.0012 |
| afrobarometer | Q15 | Mali | cp | 0.0307 | -0.0005 | -0.0312 | 15 | 21 | +6 | 0.0481 | 0.0397 | -0.0084 |
| afrobarometer | Q15 | Mali | up | 0.0264 | 0.0130 | -0.0134 | 9 | 12 | +3 | 0.1316 | 0.1099 | -0.0217 |
| afrobarometer | Q18 | Angola | cp | 0.0009 | 0.0118 | +0.0109 | 14 | 24 | +10 | 0.0000 | 0.0763 | +0.0763 |
| afrobarometer | Q18 | Angola | up | 0.0170 | 0.0058 | -0.0112 | 16 | 22 | +6 | 0.0374 | 0.1200 | +0.0826 |
| afrobarometer | Q18 | Gabon | cp | 0.0013 | 0.0114 | +0.0101 | 19 | 24 | +5 | 0.2491 | 0.2938 | +0.0447 |
| afrobarometer | Q18 | Gabon | up | 0.0534 | 0.0261 | -0.0273 | 13 | 16 | +3 | 0.3860 | 0.3347 | -0.0513 |
| afrobarometer | Q18 | Mali | cp | 0.0160 | -0.0048 | -0.0208 | 11 | 17 | +6 | 0.2677 | 0.3012 | +0.0335 |
| afrobarometer | Q18 | Mali | up | 0.0171 | 0.0365 | +0.0194 | 16 | 23 | +7 | 0.2312 | 0.2900 | +0.0588 |
| afrobarometer | Q67A | Angola | cp | -0.1344 | -0.1259 | +0.0085 | 10 | 12 | +2 | 0.0000 | 0.0000 | +0.0000 |
| afrobarometer | Q67A | Angola | up | -0.1051 | -0.1051 | +0.0000 | 11 | 11 | +0 | 0.0000 | 0.0000 | +0.0000 |
| afrobarometer | Q67A | Gabon | cp | -0.0800 | -0.0959 | -0.0159 | 11 | 19 | +8 | 0.0000 | 0.0000 | +0.0000 |
| afrobarometer | Q67A | Gabon | up | -0.1146 | -0.1146 | +0.0000 | 9 | 9 | +0 | 0.0000 | 0.0000 | +0.0000 |
| afrobarometer | Q67A | Mali | cp | -0.1922 | -0.1784 | +0.0138 | 9 | 18 | +9 | 0.0000 | 0.0000 | +0.0000 |
| afrobarometer | Q67A | Mali | up | -0.1435 | -0.1648 | -0.0213 | 12 | 13 | +1 | 0.0000 | 0.0000 | +0.0000 |
| afrobarometer | Q93B | Angola | cp | 0.1328 | 0.1328 | +0.0000 | 5 | 5 | +0 | 0.4722 | 0.4722 | +0.0000 |
| afrobarometer | Q93B | Angola | up | 0.0961 | 0.1280 | +0.0319 | 8 | 10 | +2 | 0.3446 | 0.3779 | +0.0333 |
| afrobarometer | Q93B | Gabon | cp | 0.1951 | 0.1953 | +0.0002 | 6 | 7 | +1 | 0.8767 | 0.8397 | -0.0370 |
| afrobarometer | Q93B | Gabon | up | 0.1545 | 0.1986 | +0.0441 | 9 | 12 | +3 | 0.2578 | 0.7179 | +0.4601 |
| afrobarometer | Q93B | Mali | cp | 0.1362 | 0.1245 | -0.0117 | 7 | 8 | +1 | 0.6206 | 0.6403 | +0.0197 |
| afrobarometer | Q93B | Mali | up | 0.1041 | 0.0960 | -0.0081 | 6 | 7 | +1 | 0.6137 | 0.5995 | -0.0142 |
| arabbarometer | Q104 | Mauritania | cp | 0.1235 | 0.0762 | -0.0473 | 10 | 16 | +6 | 0.3158 | 0.2754 | -0.0404 |
| arabbarometer | Q104 | Mauritania | up | -0.0932 | 0.0469 | +0.1401 | 10 | 17 | +7 | 0.3684 | 0.3881 | +0.0197 |
| arabbarometer | Q501E_2 | Mauritania | cp | 0.0083 | -0.0417 | -0.0500 | 11 | 16 | +5 | 0.1304 | 0.1250 | -0.0054 |
| arabbarometer | Q501E_2 | Mauritania | up | 0.0020 | 0.0020 | +0.0000 | 14 | 14 | +0 | 0.1667 | 0.1667 | +0.0000 |
| arabbarometer | Q534_2 | Iraq | cp | 0.1802 | 0.2581 | +0.0779 | 14 | 21 | +7 | 0.1991 | 0.1575 | -0.0416 |
| arabbarometer | Q534_2 | Iraq | up | 0.2680 | 0.2651 | -0.0029 | 8 | 9 | +1 | 0.0612 | 0.0562 | -0.0050 |
| arabbarometer | Q534_2 | Kuwait | cp | 0.2827 | 0.2827 | +0.0000 | 10 | 10 | +0 | 0.0496 | 0.0496 | +0.0000 |
| arabbarometer | Q534_2 | Kuwait | up | 0.3161 | 0.2956 | -0.0205 | 7 | 17 | +10 | 0.0435 | 0.0231 | -0.0204 |
| arabbarometer | Q534_2 | Mauritania | cp | 0.1024 | 0.1024 | +0.0000 | 15 | 15 | +0 | 0.2308 | 0.2308 | +0.0000 |
| arabbarometer | Q534_2 | Mauritania | up | 0.1344 | 0.1323 | -0.0021 | 8 | 13 | +5 | 0.0000 | 0.0000 | +0.0000 |
| arabbarometer | Q630 | Kuwait | cp | 0.0119 | -0.0007 | -0.0126 | 12 | 16 | +4 | 0.1489 | 0.1400 | -0.0089 |
| arabbarometer | Q630 | Kuwait | up | 0.0294 | 0.0460 | +0.0166 | 12 | 15 | +3 | 0.3404 | 0.3200 | -0.0204 |
| arabbarometer | Q630 | Mauritania | cp | 0.0318 | 0.0212 | -0.0106 | 12 | 16 | +4 | 0.0751 | 0.1348 | +0.0597 |
| arabbarometer | Q630 | Mauritania | up | 0.0064 | 0.0122 | +0.0058 | 11 | 14 | +3 | 0.0402 | 0.0332 | -0.0070 |
| asianbarometer | SE13a | Australia | cp | 0.0629 | 0.0824 | +0.0195 | 10 | 11 | +1 | 0.3131 | 0.3113 | -0.0018 |
| asianbarometer | SE13a | Australia | up | 0.0127 | 0.0159 | +0.0032 | 9 | 10 | +1 | 0.3034 | 0.2963 | -0.0071 |
| asianbarometer | SE13a | Indonesia | cp | 0.0831 | 0.0752 | -0.0079 | 15 | 22 | +7 | 0.0649 | 0.1026 | +0.0377 |
| asianbarometer | SE13a | Indonesia | up | 0.0993 | 0.0829 | -0.0164 | 14 | 16 | +2 | 0.0432 | 0.0413 | -0.0019 |
| asianbarometer | SE13a | Mongolia | cp | -0.0136 | 0.0143 | +0.0279 | 10 | 17 | +7 | 0.1614 | 0.1808 | +0.0194 |
| asianbarometer | SE13a | Mongolia | up | -0.0103 | 0.0024 | +0.0127 | 14 | 15 | +1 | 0.0560 | 0.1000 | +0.0440 |
| asianbarometer | SE14a | Indonesia | cp | 0.0344 | 0.0417 | +0.0073 | 10 | 14 | +4 | 0.3882 | 0.3383 | -0.0499 |
| asianbarometer | SE14a | Indonesia | up | 0.0513 | 0.0443 | -0.0070 | 10 | 14 | +4 | 0.4101 | 0.3567 | -0.0534 |
| asianbarometer | SE14a | Mongolia | cp | 0.0595 | 0.0720 | +0.0125 | 6 | 8 | +2 | 0.5208 | 0.5248 | +0.0040 |
| asianbarometer | SE14a | Mongolia | up | 0.1023 | 0.1078 | +0.0055 | 13 | 16 | +3 | 0.4369 | 0.4923 | +0.0554 |
| asianbarometer | SE2 | Australia | cp | 0.0218 | 0.0605 | +0.0387 | 7 | 9 | +2 | 0.5205 | 0.4737 | -0.0468 |
| asianbarometer | SE2 | Australia | up | -0.0078 | -0.0078 | +0.0000 | 4 | 4 | +0 | 0.3000 | 0.3000 | +0.0000 |
| asianbarometer | SE2 | Indonesia | cp | -0.0483 | -0.0483 | +0.0000 | 5 | 5 | +0 | 0.0198 | 0.0198 | +0.0000 |
| asianbarometer | SE2 | Indonesia | up | -0.0346 | -0.0346 | +0.0000 | 5 | 5 | +0 | 0.0218 | 0.0218 | +0.0000 |
| asianbarometer | SE2 | Mongolia | cp | 0.0370 | 0.0436 | +0.0066 | 4 | 5 | +1 | 0.4046 | 0.3312 | -0.0734 |
| asianbarometer | SE2 | Mongolia | up | -0.0150 | -0.0150 | +0.0000 | 4 | 4 | +0 | 0.0840 | 0.0840 | +0.0000 |
| asianbarometer | SE7a | Australia | cp | -0.0471 | -0.0471 | +0.0000 | 10 | 10 | +0 | 0.1429 | 0.1429 | +0.0000 |
| asianbarometer | SE7a | Australia | up | 0.0673 | 0.0883 | +0.0210 | 11 | 13 | +2 | 0.8142 | 0.7824 | -0.0318 |
| asianbarometer | level | Indonesia | cp | 0.1408 | 0.1350 | -0.0058 | 6 | 9 | +3 | 0.4706 | 0.4251 | -0.0455 |
| asianbarometer | level | Indonesia | up | 0.1354 | 0.1123 | -0.0231 | 5 | 7 | +2 | 0.5323 | 0.4626 | -0.0697 |
| ess_wave_11 | bctprd | Austria | cp | 0.0305 | 0.0298 | -0.0007 | 14 | 15 | +1 | 0.2069 | 0.2000 | -0.0069 |
| ess_wave_11 | bctprd | Austria | up | 0.0298 | 0.0517 | +0.0219 | 15 | 20 | +5 | 0.0667 | 0.4199 | +0.3532 |
| ess_wave_11 | bctprd | France | cp | 0.0751 | 0.0939 | +0.0188 | 17 | 19 | +2 | 0.1397 | 0.3915 | +0.2518 |
| ess_wave_11 | bctprd | France | up | 0.0807 | 0.0882 | +0.0075 | 13 | 15 | +2 | 0.4359 | 0.4583 | +0.0224 |
| ess_wave_11 | eqparlv | Austria | cp | -0.0137 | -0.0215 | -0.0078 | 11 | 12 | +1 | 0.2005 | 0.1915 | -0.0090 |
| ess_wave_11 | eqparlv | Austria | up | 0.0515 | 0.0515 | +0.0000 | 10 | 10 | +0 | 0.3854 | 0.3854 | +0.0000 |
| ess_wave_11 | eqparlv | France | cp | 0.0134 | 0.0125 | -0.0009 | 12 | 13 | +1 | 0.1393 | 0.1340 | -0.0053 |
| ess_wave_11 | eqparlv | France | up | 0.0204 | 0.0415 | +0.0211 | 9 | 14 | +5 | 0.0920 | 0.1429 | +0.0509 |
| ess_wave_11 | eqparlv | Italy | cp | 0.0685 | 0.0805 | +0.0120 | 16 | 18 | +2 | 0.1096 | 0.1048 | -0.0048 |
| ess_wave_11 | eqparlv | Italy | up | 0.0987 | 0.1206 | +0.0219 | 12 | 14 | +2 | 0.4173 | 0.4894 | +0.0721 |
| ess_wave_11 | stfgov | Austria | cp | 0.0898 | 0.0898 | +0.0000 | 10 | 10 | +0 | 0.3961 | 0.3961 | +0.0000 |
| ess_wave_11 | stfgov | Austria | up | 0.1030 | 0.1064 | +0.0034 | 11 | 13 | +2 | 0.3119 | 0.2979 | -0.0140 |
| ess_wave_11 | stfgov | France | cp | 0.1222 | 0.1172 | -0.0050 | 14 | 17 | +3 | 0.2976 | 0.2834 | -0.0142 |
| ess_wave_11 | stfgov | France | up | 0.1398 | 0.1347 | -0.0051 | 12 | 16 | +4 | 0.3497 | 0.3160 | -0.0337 |
| ess_wave_11 | stfgov | Italy | cp | 0.1604 | 0.1687 | +0.0083 | 9 | 13 | +4 | 0.4552 | 0.4545 | -0.0007 |
| ess_wave_11 | stfgov | Italy | up | 0.1296 | 0.1409 | +0.0113 | 13 | 16 | +3 | 0.3570 | 0.3412 | -0.0158 |
| latinobarometer | P61ST | Guatemala | cp | -0.0297 | -0.0088 | +0.0209 | 12 | 16 | +4 | 0.1587 | 0.1361 | -0.0226 |
| latinobarometer | P61ST | Guatemala | up | 0.0100 | 0.0100 | +0.0000 | 16 | 16 | +0 | 0.2789 | 0.2789 | +0.0000 |
| latinobarometer | P8STGBS | Argentina | cp | 0.0950 | 0.0750 | -0.0200 | 13 | 19 | +6 | 0.6467 | 0.5782 | -0.0685 |
| latinobarometer | P8STGBS | Argentina | up | 0.0989 | 0.0748 | -0.0241 | 12 | 18 | +6 | 0.6872 | 0.6280 | -0.0592 |
| latinobarometer | P8STGBS | Colombia | cp | 0.0502 | 0.0635 | +0.0133 | 10 | 15 | +5 | 0.4523 | 0.4394 | -0.0129 |
| latinobarometer | P8STGBS | Colombia | up | -0.0008 | 0.0812 | +0.0820 | 14 | 21 | +7 | 0.0286 | 0.3765 | +0.3479 |
| latinobarometer | P8STGBS | Guatemala | cp | 0.1884 | 0.1877 | -0.0007 | 12 | 13 | +1 | 0.7506 | 0.7373 | -0.0133 |
| latinobarometer | P8STGBS | Guatemala | up | -0.0016 | 0.0393 | +0.0409 | 4 | 11 | +7 | 0.0220 | 0.0528 | +0.0308 |
| latinobarometer | S20.A | Colombia | cp | 0.0874 | 0.0760 | -0.0114 | 5 | 6 | +1 | 0.6350 | 0.5899 | -0.0451 |
| latinobarometer | S20.A | Colombia | up | 0.0709 | 0.0709 | +0.0000 | 6 | 6 | +0 | 0.6313 | 0.6313 | +0.0000 |
| latinobarometer | S20.A | Guatemala | cp | 0.0351 | 0.0388 | +0.0037 | 6 | 7 | +1 | 0.2957 | 0.2713 | -0.0244 |
| latinobarometer | S20.A | Guatemala | up | 0.0539 | 0.0539 | +0.0000 | 7 | 7 | +0 | 0.4341 | 0.4341 | +0.0000 |
| latinobarometer | SEXO | Argentina | up | -0.0235 | 0.0246 | +0.0481 | 2 | 4 | +2 | 0.0714 | 0.3817 | +0.3103 |
| latinobarometer | SEXO | Colombia | up | -0.0153 | -0.0439 | -0.0286 | 2 | 4 | +2 | 0.0972 | 0.1833 | +0.0861 |
| latinobarometer | SEXO | Guatemala | cp | 0.0871 | 0.0871 | +0.0000 | 2 | 2 | +0 | 0.0000 | 0.0000 | +0.0000 |
| latinobarometer | SEXO | Guatemala | up | -0.0129 | 0.0087 | +0.0216 | 2 | 4 | +2 | 0.0000 | 0.0076 | +0.0076 |
| wvs | Q205 | Andorra | cp | 0.0398 | 0.0398 | +0.0000 | 8 | 8 | +0 | 0.0809 | 0.0809 | +0.0000 |
| wvs | Q205 | Andorra | up | 0.2343 | 0.2621 | +0.0278 | 9 | 10 | +1 | 0.1429 | 0.7228 | +0.5799 |
| wvs | Q205 | Germany | cp | 0.1410 | 0.1410 | +0.0000 | 11 | 11 | +0 | 0.0750 | 0.0750 | +0.0000 |
| wvs | Q205 | Germany | up | 0.1508 | 0.2146 | +0.0638 | 8 | 9 | +1 | 0.1149 | 0.7425 | +0.6276 |
| wvs | Q205 | Macao SAR | cp | 0.1287 | 0.1255 | -0.0032 | 8 | 10 | +2 | 0.5876 | 0.5803 | -0.0073 |
| wvs | Q205 | Macao SAR | up | 0.1462 | 0.1615 | +0.0153 | 8 | 9 | +1 | 0.5085 | 0.5676 | +0.0591 |
| wvs | Q242 | Macao SAR | cp | -0.0124 | 0.0339 | +0.0463 | 11 | 25 | +14 | 0.0751 | 0.1662 | +0.0911 |
| wvs | Q242 | Macao SAR | up | 0.0749 | 0.0749 | +0.0000 | 14 | 14 | +0 | 0.1608 | 0.1608 | +0.0000 |
| wvs | Q263 | Macao SAR | cp | 0.1562 | 0.1562 | +0.0000 | 4 | 4 | +0 | 0.4731 | 0.4731 | +0.0000 |
| wvs | Q263 | Macao SAR | up | 0.2396 | 0.2321 | -0.0075 | 6 | 9 | +3 | 0.5848 | 0.5615 | -0.0233 |
| wvs | Q40 | Andorra | cp | -0.0165 | -0.0047 | +0.0118 | 8 | 12 | +4 | 0.1815 | 0.1705 | -0.0110 |
| wvs | Q40 | Andorra | up | 0.0864 | 0.0766 | -0.0098 | 17 | 23 | +6 | 0.2547 | 0.2734 | +0.0187 |
| wvs | Q40 | Macao SAR | cp | 0.0081 | 0.0056 | -0.0025 | 13 | 17 | +4 | 0.0791 | 0.0950 | +0.0159 |
| wvs | Q40 | Macao SAR | up | 0.0353 | 0.0163 | -0.0190 | 15 | 18 | +3 | 0.0106 | 0.0098 | -0.0008 |

## D. Interpretation

1. **Mapping loss is real at the subconcept layer.** Parent units map ~62.9% of the time; sub_item units ~49.9%. Collapse under parent-only mapping is therefore not just bookkeeping: many fine measures never get a code.
2. **Expansion does change the code set.** Mean k rises by ~3.0 unique codes (~1.31×). Parent↔subitem Jaccard is low (~0.16), and recovery when the parent is none appears in 53 cells — so sub_items sometimes surface variables the parent query missed.
3. **Capability barely moves.** Natural-k mean Δ VoR is only +0.0052 (sign-mixed across cells). Captured importance rises more (+0.0289) alongside Δk ≈ +2.9, which is the expected **budget** channel.
4. **Matched-k checks cut against a quality story.** At k=5, Δ captured importance ≈ +0.0004; at k=10, Δ VoR ≈ -0.0013. Equal-budget rows do not show a robust gain from the extra / different codes.
5. **Paper implication.** Treat this as a **mapping-granularity caveat**: parent-only MiniLM arm-C understates requested specificity and misses some recoverable subconcepts, but expanding for scoring does not justify replacing the headline capability metric. Main results should remain parent-only unless a future protocol change is explicit.

## E. Reproducibility

```bash
# diagnostics table (if regenerating)
python analysis/subitem_mapping.py --selector kimi

# map + score (already run for these artifacts)
python scripts/run_subitem_mapping.py --phase map   --selector kimi --disambiguator nemotron --arms C
python scripts/run_subitem_mapping.py --phase score --selector kimi --k-modes parent,expanded
```

Paired score contrasts in this appendix use cells present under both `k_mode=parent` and `k_mode=expanded` for the stated `k_spec`.
