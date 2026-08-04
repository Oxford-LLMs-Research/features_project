# Preliminary multi-survey grid: analysis and summary

> **SUPERSEDED — numbers are not reproducible from disk.** This describes a 5x5
> single-model run that `RECONCILIATION_PLAN.md` records as overwritten and unrecoverable.
> Kept for narrative history only. Do not cite these figures.
> See `experiments_index.md` -> "Which numbers are current".

This document summarises the **first full preliminary run** after the big pipeline update (4 May 2026): six surveys, **five targets × five countries** each, full stack (oracle permutation importance, LLM feature listing, embedding retrieval, disambiguation, XGBoost comparison). Statistics below are computed from `c:\Users\murrn\cursor\features_project\outputs\grid_summary__*.csv` unless noted. A machine-readable digest of the same aggregates is in `c:\Users\murrn\cursor\features_project\outputs\_prelim_stats.json` (regenerate with `python analysis/prelim_results_stats.py > outputs/_prelim_stats.json`).

---

## 1. What we ran

| Setting | Value |
|--------|--------|
| Surveys | `wvs`, `afrobarometer`, `arabbarometer`, `asianbarometer`, `latinobarometer`, `ess_wave_11` (`ess_wave_10` omitted) |
| Grid | 5 targets × 5 countries per survey → **25 cells** × 6 surveys = **150 unique cells** |
| Conditions per cell | `unprompted` and `country_provided` (two LLM passes, same mapping/disambig pipeline) |
| Summary rows | **300** (150 cells × 2 conditions) |
| Concurrency | `grid_workers=3`, XGBoost `nthread≈6` per fit on 20 logical CPUs, random baseline draws `n_jobs=1` per cell |

**Target sampling (manifest).** Targets were chosen by `c:\Users\murrn\cursor\features_project\prelim\build_prelim_manifest.py` and `c:\Users\murrn\cursor\features_project\prelim\target_selection.py` with:

- **Quota:** 2 binary + 2 mid (4–5 substantive categories) + 1 “large” (**5–15** categories; `DEFAULT_LARGE_CAP=15`, overridable via `LARGE_CAP`).
- **Topic spread:** `topic_tag` from metadata is preferred over raw section names, so diversity is driven by harmonised tags (e.g. `democratic_values`, `media_information`) where available.
- **Country spread:** evenly spaced names from the sorted country list per survey.
- **Feasibility:** targets must have at least 30 non-missing responses after the same cleaning as the oracle in **all** five countries.

Details per survey (variable codes, section, `topic_key`, bucket, category counts) are in `c:\Users\murrn\cursor\features_project\prelim\target_selection_detail.yaml`. The frozen grid for this run is `c:\Users\murrn\cursor\features_project\prelim\prelim_manifest.yaml`.

---

## 2. Outcome definitions (how to read the tables)

For each (target × country × condition) row in `outputs\grid_summary__*.csv`:

| Metric | Meaning |
|--------|--------|
| `oracle_acc` | 5-fold CV accuracy using the **top-k** features from **permutation importance** (oracle), with *k* matched to the number of **distinct** survey variables the LLM pipeline mapped for that condition. |
| `model_acc` | Same CV setup using the LLM’s mapped features after disambiguation. |
| `random_acc` | Mean CV accuracy over **20** random draws of *k* features from the same feature pool used for the oracle in that cell. |
| `majority_baseline` | Majority-class share in the cleaned target (reference “no skill”). |
| **cost_of_imperfect** | `oracle_acc − model_acc` — how much worse the LLM’s picked variables are than the oracle’s top-k *for the same k*. |
| **value_over_random** | `model_acc − random_acc` — whether the LLM beats *chance* feature choice at matched *k*. |

**Interpretation guide.**

- Large **positive cost** with **positive value**: the model recovers predictive signal but not as much as the oracle ceiling; still beats random.
- Large **positive cost** with **negative value**: the model’s variables underperform both oracle and random (wrong features or *k* too small/large for what it named).
- Near-zero cost with flat oracle vs majority: often **degenerate** (see §4) — little learnable signal in *Y* after cleaning.

---

## 3. Coverage and validity

| Category | Count (of 300 rows) |
|----------|---------------------|
| Rows with full numeric triple (`oracle_acc`, `model_acc`, `random_acc`) | **295** |
| Rows with missing `oracle_acc` (evaluation not comparable or failed upstream) | **5** |
| Rows with **no mapped features** (`k_mapped=0`) | **1** |

**Pooled valid *n* for means below:** **295** rows.

**Global** means on those 295 valid rows (exact, from a straight average over rows):

| Metric | Mean |
|--------|------|
| Oracle accuracy | **0.6425** |
| Model accuracy | **0.5303** |
| Random-*k* accuracy | **0.5083** |
| Majority baseline | **0.5143** |
| Cost of imperfect | **0.1122** |
| Value over random | **0.0220** |
| Share of rows with value over random > 0 | **47.5%** |
| Share of rows with oracle − majority > 0.05 (“>5 pp signal”) | **66.8%** |

So on average the LLM sits between oracle and random: it **narrows** the gap toward oracle but does not match the ceiling; it **slightly** beats random on average, with roughly **half** of cells beating chance at matched *k*.

---

## 4. Special rows (degeneracy, mapping failure)

### 4.1 Missing oracle (5 rows)

These appear as empty `oracle_acc` / `model_acc` / `random_acc` in the CSV.

| Survey | Target | Country | Condition | Note |
|--------|--------|---------|-----------|------|
| `arabbarometer` | Q1005B | Kuwait | both | Oracle/eval did not produce usable numeric row (likely **<2 classes** or pipeline skip after cleaning in that slice; see `phase0b_evaluation` / oracle logs for the cell). |
| `wvs` | Q263 | Romania | both | After cleaning, **single class** on immigrant status in Romania (effectively no variation); oracle majority 100%. |
| `latinobarometer` | SEXO | Paraguay | unprompted | **k_mapped = 0** (see below). |

### 4.2 Zero mapped features (1 row)

**SEXO × Paraguay, `unprompted`:** the LLM’s feature list did not survive retrieval + disambiguation to any distinct variable codes, so *k*=0 and no accuracy comparison is defined. The **country_provided** condition for the same cell **did** map features and completed normally. This is a **content** result (prompting / model behaviour), not a crash.

A mid-run bug (`KeyError: 'target'` when printing zero-*k* results) was fixed in `c:\Users\murrn\cursor\features_project\phase0b_evaluation.py`; the cell was re-evaluated so the summary row is stable.

---

## 5. Results by survey

Valid row counts and **survey-level** means (valid rows only):

| Survey | Valid / 50 | Mean oracle | Mean model | Mean random | Mean cost | Mean value | Share value > 0 | Mean “signal” (oracle − majority) |
|--------|------------|-------------|------------|-------------|-----------|------------|-----------------|-------------------------------------|
| afrobarometer | 50 | 0.670 | 0.525 | 0.520 | **0.145** | 0.005 | 0.38 | 0.151 |
| arabbarometer | 48 | 0.649 | 0.535 | 0.555 | 0.114 | **−0.019** | 0.23 | 0.108 |
| asianbarometer | 50 | 0.683 | 0.555 | 0.493 | 0.128 | **0.062** | **0.70** | 0.181 |
| ess_wave_11 | 50 | 0.621 | 0.520 | 0.456 | 0.101 | **0.064** | **0.68** | 0.154 |
| latinobarometer | 49 | 0.509 | 0.424 | 0.421 | **0.085** | 0.003 | 0.45 | 0.065 |
| wvs | 48 | 0.724 | 0.625 | 0.609 | 0.099 | 0.016 | 0.40 | 0.107 |

**Reading the table.**

- **Asianbarometer** and **ESS** show the strongest **average value over random** and the highest **share** of cells where the model beats random. Part of that is sample/country mix, part is that several targets in those surveys gave **large** lifts for well-named predictors (see §7).
- **Arab Barometer** is the **only** survey with **negative mean value over random**: on aggregate, matched-*k* random draws do slightly *better* than the LLM’s mapped set. That is driven heavily by **Q104** (migration attitudes) across multiple countries — oracle remains strong but **model-selected** features underperform **random** in several cells (§7). This deserves case-by-case qualitative review (disambiguation choosing weak correlates, or *k* mismatch effects).
- **Afrobarometer** has the **highest mean cost** (oracle ceiling high vs what LLM picks) but near-zero mean value: model ≈ random on average.
- **Latinobarometer** has the **lowest mean signal** (oracle barely above majority on average) and lowest accuracies — harder prediction problem for this target mix in these countries.

---

## 6. Condition effect: unprompted vs country-provided

On the **295 valid** rows:

| Condition | *n* | Mean cost | Mean value | Share value > 0 |
|-----------|-----|-----------|------------|-----------------|
| unprompted | 147 | 0.111 | 0.022 | 0.47 |
| country_provided | 148 | 0.114 | 0.022 | 0.48 |

**Takeaway:** at this scale, **providing the country name in the LLM prompt** barely moves **average** cost or value. That does **not** rule out **local** effects (specific surveys/targets may still show larger deltas); it suggests the headline comparison is **subtle** and may need finer stratification (e.g. by survey, by topic_tag, or by whether the target is “country-specific” in wording).

---

## 7. By target **bucket** (binary / mid / large)

Aggregated over valid rows with a joined bucket from the manifest (115 binary + 120 mid + 60 large = **295**):

| Bucket | *n* | Mean cost | Mean value | Mean oracle − majority |
|--------|-----|-----------|------------|------------------------|
| binary | 115 | **0.125** | 0.025 | **0.165** |
| mid | 120 | 0.117 | 0.019 | 0.120 |
| large (5–15 cats) | 60 | **0.077** | 0.023 | 0.074 |

**Interpretation.** Binary targets carry the **largest oracle lift** over majority (high signal) and the **highest** average cost of imperfect selection — the ceiling is high, so there is more room to “miss” with imperfect features. **Large** (but capped) multiclass targets show **lower** average cost: oracle and model are closer, partly because ceilings are lower and labels are harder. This pattern is useful for the paper: **cardinality and baseline** should be reported alongside raw accuracy.

---

## 8. Extremes: where the model shines or struggles

### 8.1 Largest **cost_of_imperfect** (oracle ≫ model)

Examples from the top of the distribution:

1. **Latinobarometer, SEXO × Colombia** — oracle accuracy ~0.99 vs model ~0.49–0.50 vs random ~0.54–0.55. The oracle finds a near-deterministic predictor of **sex** in the feature pool; the LLM names reasonable demographics but the **mapped** variables do not reproduce that shortcut. **Value over random** is **negative** here: random *k* beats the LLM’s picks.
2. **Asianbarometer, SE2 × Vietnam / Mongolia** — similar pattern: oracle ≈0.99, model ~0.56–0.59, **value** mixed (sometimes slightly positive, sometimes slightly negative vs random).
3. **Afrobarometer, Q67A** (climate policy, binary) in **Zimbabwe, Mali** — oracle ≈1.0, model ~0.60–0.62; random is close to model, so value is small. Strong ceiling, weak alignment between LLM concepts and the specific variables the oracle exploits.
4. **Arab Barometer, Q534_2 × Iraq, unprompted** — multiclass media item; high oracle, much lower model.

**Substantive note:** When oracle accuracy is **near 1.0** for a **demographic** target (sex, immigration-related columns), the “oracle” may be exploiting **survey-specific** combinations that are almost labels in disguise. That is **not** proof of “superhuman” prediction — it is a warning to **audit** top oracle features for **leakage** or **near-duplicate** items before interpreting cost as “reasoning failure.”

### 8.2 Most **negative value_over_random** (model worse than chance)

Dominated by **Arab Barometer Q104** (migration attitudes) across **Kuwait, Mauritania, Morocco, Iraq** under one or both conditions: model accuracy stays in the 0.67–0.86 range but **random *k*** is **higher** (up to ~0.93). That implies the **shortlist of variables the LLM locked onto** is systematically **weaker** than randomly sampling from the full pool at the same *k* — a strong caution for claims about “semantic understanding” without checking **baseline strength** and **pool size**.

Other contributors: **Afrobarometer Q15/Q18 × Seychelles** (country-provided), **Q501E_2 × Morocco** (unprompted).

### 8.3 Largest **positive value_over_random** (model ≫ random)

Standouts:

1. **Asianbarometer, SE7a × Mongolia** — model ≈0.9945 vs random ≈0.39 for **both** conditions (value **~+0.60**). Large lift: LLM-named features map to **highly predictive** variables for that target/country.
2. **ESS, rtrd** (immigrant background) — repeated large values in **Austria, France, Serbia, UK** (value commonly **+0.22 to +0.34**). Demographic/background predictors align well with free-text **priors**.
3. **WVS, Q263 × Andorra** — model and oracle both **1.0** vs random ~0.72; immigrant/native structure is stark in that N.

These cells are the best evidence for the paper’s **positive** claim: in some countries, **prior-only** feature naming **beats chance** by a wide margin when mapped into the survey’s coding.

---

## 9. Pipeline performance and design (what we learned operationally)

1. **Oracle dominates wall time.** Multiclass targets with **hundreds** of predictors and **5×10** permutation repeats per feature are expensive. Capping “large” targets at **15** categories removed earlier pathologies (e.g. year-of-birth ~80 classes) and stabilised run time; **WVS oracle-only smoke** for 25 cells was **~25 minutes** on this machine; the **full six-survey** run including LLM + eval was **~3.5 hours** with `grid_workers=3`.

2. **Parallelism is healthy** with the current budget: three overlapping cells × ~6 XGB threads ≈ full CPU; embedding model loading is **singleton + lock** to avoid thread races (`phase0b_mapping.py`).

3. **Caching** under `outputs/<target>_<country>/` makes iterative analysis cheap; `grid_summary__<survey>.csv` is the first stop for quantitative summaries.

4. **Windows stdout:** avoid non-ASCII in print paths for small utilities (fixed in `analysis/prelim_aggregate.py`).

---

## 10. Limitations and next steps

1. **Single LLM / single run** — results are for one model id (see run logs for provider). No variance across prompts, seeds, or models yet.
2. ***k* is not fixed** — it follows mapped count per condition, so oracle-top-*k*, model, and random are tied to **realised** mapping richness. Comparing cells with very different *k* requires care.
3. **Leakage audit** — for cells with oracle ≈1.0 on demographic-like targets, manually inspect `oracle.csv` top rows for **naming overlap** with the target or administrative fields.
4. **Arab Barometer Q104** — priority for **qualitative** review of `disambig.json` (which variables were chosen vs oracle top features).
5. **Statistical inference** — 300 rows are enough for exploratory **patterns**, not for precise hierarchical inference; cluster standard errors by survey × target if moving to regression.
6. **Interaction with country prompt** — pool condition-level effects within survey × target_tag to see if “country_provided” helps where targets reference **national** institutions.

---

## 11. How to reproduce or extend

```text
# Regenerate manifest (optional)
python prelim/introspect_metadata.py
python prelim/build_prelim_manifest.py

# One survey
python run_grid.py --survey wvs --from-manifest prelim/prelim_manifest.yaml --grid-workers 3

# Summaries
python analysis/prelim_aggregate.py
python analysis/prelim_results_stats.py > outputs/_prelim_stats.json
```

Orchestrated end-to-end (wipe, rebuild, smoke, gated full run): `c:\Users\murrn\cursor\features_project\run_big_update.ps1`.

---

## 12. One-sentence conclusion

Across six heterogeneous surveys, **LLM-proposed predictors, after mapping, on average land between oracle and random**, with **clear winners** (e.g. ESS `rtrd`, Asianbarometer `SE7a` in Mongolia, WVS `Q263` in Andorra) and **clear failures** (notably **Arab Barometer `Q104`**, and **sex** targets where the oracle approaches deterministic classification); **degenerate or missing cells are rare** and mostly interpretable (no variation in *Y*, or zero mappings).
