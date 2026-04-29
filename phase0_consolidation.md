# Phase 0 Results — March 2026

## 0a: Ground Truth Variation

**Data.** 5 targets × 5 countries (WVS). XGBoost with permutation importance on held-out folds, 10 repeats, fixed hyperparameters (max_depth=6, min_child_weight=5, n_estimators=300).

**Result.** Mean pairwise Spearman ρ = 0.038 across all targets. Importance rankings are essentially uncorrelated across countries. Sufficient cross-national variation to justify the main experiment.

**Robustness check.** Reran Q199 Nigeria with max_depth=4. Top-3 features identical in same order (Q200, Q4, Q98). Pipeline is stable for high-signal cells.

**Note.** Design doc says "hyperparameters tuned per cell via grid search." Code uses fixed parameters. Resolve before Phase 1.

---

## 0b: LLM Feature Selection

**Setup.** 5 targets × 2–3 countries × 2 models (DeepSeek-V3.2, DeepSeek-R1) × 2 output formats (structured JSON, free-form). All at temperature=0.

### Results by target

| Target | Ground truth top predictor(s) | Model names it? (structured) | Model names it? (free-form) |
|--------|------------------------------|-----------------------------|-----------------------------|
| Q164 (importance of God) | Q165 (believe in God), Q172 (prayer frequency), Q6 (importance of religion) | Yes — names religious affiliation, prayer, service attendance in correct priority area | Yes |
| Q57 (interpersonal trust) | Q61 (trust in strangers) | Partially — "prior survey trust" listed last | V3.2 Germany: "best single predictor is a related trust question" |
| Q47 (self-rated health) | Q262 (age), Q46 (happiness) | Age correct at rank 1. Happiness not named. Rest is public health textbook (chronic conditions, lifestyle, SES) | Not tested |
| Q199 (political interest) | Q200 (discusses politics), Q4 (importance of politics) | Not named | R1 unprompted names "political discussion frequency"; absent in country-conditioned outputs |
| Q235 (strong leader) | Q236 (expert governance attitudes) | Not named | Adjacent constructs mentioned (RWA, SDO) but not the specific item |

### Pattern

Models succeed when the ground truth is "X predicts X" — religiosity predicts religiosity, trust predicts trust. Models fail when the ground truth requires cross-domain knowledge — discussing politics predicts political interest, expert governance attitudes predict strongman preferences. The boundary is between same-construct adjacency (which models capture) and cross-construct empirical correlation (which they miss).

### Other observations

**1. All outputs lead with demographics.** Age, education, SES appear first in every structured output regardless of target or country. These features rarely appear in the ground truth top-10.

**2. Structured format degrades output.** Free-form outputs are more specific and closer to ground truth across all targets tested in both formats. In structured format, diagnostic features either disappear or appear last.

**3. Country conditioning adds surface-level features.** Nigeria gets ethnicity/religion. Germany gets East/West, AfD. Japan gets uchi/soto. These do not track ground truth differences in predictive structure.

**4. R1 traces contain knowledge not in final output.** R1 for Q199 names "political discussion frequency" in its thinking trace and unprompted output, but drops it from the Nigeria-conditioned output during deliberation about what's "practical to collect."

**5. Outputs are near-identical across models.** V3.2 and R1 structured outputs are interchangeable for all targets tested.

---

## Design Implications

1. **The study has signal.** Models produce measurably different outputs from ground truth. The gap is quantifiable and varies systematically by target type.
2. **Structured prompt as primary instrument.** Standardised and parseable. Degradation relative to free-form is a finding, not a reason to change protocol.
3. **Free-form as ablation.** Documents the operationalisation gap.
4. **Save reasoning traces.** Trace-to-output comparison is informative.
5. **Change the prompt example.** "Education level" may anchor toward demographics. Replace or drop.
6. **Feature order may carry information.** Demographics first, diagnostic features last. Consider analysing rank position.
7. **Mapping pipeline is functional.** See results below.

---

## Mapping Pipeline Results

**Architecture.** Two-stage: (1) embedding retrieval (all-MiniLM-L6-v2, top-5 candidates) → (2) LLM disambiguator picks best match or "none."

**Key fix during development.** Embedding label+reasoning as a single string caused short demographic labels ("age") to drift into the target's topic domain. Computing similarity twice (label-only and label+reasoning) and taking the max per candidate resolved this. Age now maps to Q262 across all targets.

**Results (5 targets × 5 countries, unprompted condition).**
- Pipeline maps 81% of features to a survey variable; 19% return "none."
- Hit rate against ground truth top predictors: 5.7% (unprompted), 7.2% (all conditions).
- The low hit rate is not a pipeline failure. The mappings are correct — the pipeline accurately maps what the model requests. The model requests the wrong features.

**Three-category outcome:**
- **Hit:** mapped to a ground truth variable (rare — model rarely names the right construct)
- **Miss:** mapped to a valid survey variable that isn't a ground truth predictor (most common)
- **Unmappable:** "none" — model requests something the survey doesn't measure (19%)

The unmappable rate is itself informative. Models request constructs like chronic conditions, lifestyle behaviours, personality traits, and betrayal experiences — things not measured in WVS. This reflects reasoning from ideal predictors rather than available data.

---

## Not Yet Tested

- Low-signal cells (Q57 Brazil/Egypt). Does the model calibrate list length to signal strength?
- Human validation of mapping accuracy (annotation study).
- Downstream prediction with model-selected vs oracle-selected features (decoupled evaluation).
