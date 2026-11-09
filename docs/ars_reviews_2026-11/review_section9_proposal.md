# ARS review mode: project inventory §1 and §9 (novelty claim and draft re-ordered pre-registration)

**Reviewed text:** `docs/project_inventory_2026-11.md` (snapshot dated 6 Nov 2026). Focus: §1 (registered hypotheses), §9.1, §9.2, §9.3, §9.5, §9.6.
**Context read:** `docs/grid_design_memo_2026-08.md` (opening variance section, Q2, Q4, "To pre-register"), `docs/pipeline_audit_2026-08.md` §A1 and §A6–A9, `docs/glossary.md` ("adaptation test", "nested country rule"), `docs/prompt_sensitivity_v2.html` (source of the 0.27–0.29 figure), `scripts/oracle_heterogeneity_screen.py` (docstring), `scripts/run_main.py` (country-blind mapping logic).
**Review date:** 2026-11-08. **Method:** ARS deep-research review mode (editor_in_chief_agent, devils_advocate_agent at Checkpoint 3, ethics_review_agent), using `logical_fallacies.md`, `argumentation_reasoning_framework.md` and `cross_agent_quality_definitions.md`. The Devil's Advocate ran a new counter-evidence search (Anti-Pattern #1); its log is in §5.
**Severity scale (ARS):** CRITICAL = would invalidate a core conclusion if not fixed. MAJOR = a serious weakness that can be fixed. MINOR = does not threaten the core claim.

### Terms used below (short reminders, as CLAUDE.md asks)

- **Selector**: the LLM under test. It writes an essay on what predicts a survey answer. A fixed extractor (Qwen) and a fixed mapper (Nemotron) turn that essay into survey variable codes, the "picks".
- **Oracle**: a ranking, built from the data, of which variables actually predict the answer in one survey × question × country (a **cell**). It uses AutoGluon permutation importance under **contract v4**, the current recipe. Every claim in the project is graded against it.
- **Oracle heterogeneity**: how much two countries' oracle rankings disagree beyond what the oracle's own noise would produce (within-country reliability minus between-country agreement).
- **Primary A / adaptation test**: the draft's proposed new headline. It regresses how different a selector's picks are between two countries (1 − Jaccard of mapped codes) on that pair's oracle heterogeneity. The result is read three ways: adapts, changes picks regardless ("stereotyping"), or rigid.
- **Primary B / swap gain**: the registered Primary 2, relabelled. It scores country A's own picks against picks made for country B, both on A's data. It asks whether differentiation pays off in prediction.
- **Noise floor**: how much picks differ between two runs of the same prompt for the same country. Any "differentiation" must be measured above it.
- **Low bin**: questions with near-zero heterogeneity whose oracles are reliable enough to show it. They serve as the negative control.
- **Confirmatory run**: the locked, run-once measurement whose numbers go in the paper.

---

## 1. editor_in_chief_agent: Editorial Review

### Overall assessment
**Verdict: MAJOR REVISION** (as a research plan and novelty case)
**Weighted score: 2.9 / 5.0**

The inventory's literature section is unusually honest. It concedes that breadth is not novelty, flags snippet-only sources, and gives "safe" and "not safe" claim wordings. The proposed new headline asks the right different question. But the endpoint chosen to answer it (Primary A) is not identified as specified. Noise in the x-axis biases the three-way reading in a known direction. The slope is mostly identified between questions, where confounders live. The endpoint has never been sized. The plan is also only valid under a condition the authors cannot currently verify. None of this is fatal to the research programme, but all of it must be fixed before registration.

### Dimension scores
| Dimension | Weight | Score | Notes |
|---|---|---|---|
| Originality & Contribution | 20% | 3/5 | The exact operationalisation looks unclaimed: stated predictor sets, a per-country importance oracle, a dose-response on measured heterogeneity, and a swap test with a negative control. But the underlying question ("do LLM-implied predictor effects track real cross-country differences?") has been asked for one predictor (Pataranutaporn et al. 2025; Powdthavee 2026, arXiv 2609.33055). By the document's own "breadth is not novelty" rule, going from one predictor to a set is partly breadth. The gain from "stated" over "implied" structure is asserted, not argued. |
| Methodological Rigor | 25% | 2/5 | Primary A has two problems. Attenuation biases the three-way classification. Generic country distance and question-level features are uncontrolled. Also: the measure is unsigned; 1 − Jaccard depends on k; the noise-floor figure comes from a different metric and different selectors; power is unsized; there is no testing hierarchy across two primaries × six selectors. |
| Evidence Sufficiency | 25% | 3/5 | 11 of 11 spot-checked arXiv IDs resolve with matching titles and authors (§3). The Collabra paper is confirmed as published. Several load-bearing comparisons (NHB 2026, Ashokkumar et al. *Nature* 2026) are still snippet-only. The novelty verdict rests on absence of evidence from searches with known access gaps. |
| Argument Coherence | 15% | 3/5 | §9.1's "breadth is not novelty" is coherent and well argued. Two things undercut it. (a) §9.1 predicts a null is likely, then calls it "still informative" without an equivalence design that would make a null informative. (b) The three-way reading is presented as exhaustive and falsifiable, but it is neither (DA C1). |
| Writing Quality | 15% | 4/5 | Clear, decision-framed, jargon glossed. Small inconsistencies: §9 says the Semantic Scholar API was blocked, but §9.5 reports a check of LLM-Select's 60 citing papers on Semantic Scholar. Hewitt et al. 2024 (§9.2) and Ashokkumar et al. 2026 (§9.5) are the same work under two names. §9.4 is placed after §9.6. |

Weighted: 0.20·3 + 0.25·2 + 0.25·3 + 0.15·3 + 0.15·4 = **2.9**. That is Major Revision, and it is consistent with the open CRITICAL items from the DA.

### Strengths
1. **§9.1 "Breadth is not novelty"** is the right standard. It explicitly gives up the easy headline ("120 questions × 132 countries"), and it gives up instrumentation as a contribution on its own.
2. **§9.5 claim wording** separates safe from unsafe claims and pins the term "stereotyping" against ACE-Align's prior use. This is rare and good scholarly hygiene.
3. **§9.6 recognises its own validity condition** (no v4 scores or heterogeneity bins inspected) and the need for a replicate noise floor and a placebo prompt, before anyone asks.
4. **The ground-truth design** (an honest split oracle with a measured ceiling, plus a reliability-gated low bin from grid memo Q2) is a real methodological advance over LLM-Select-style "agreement with one model's importance". It is what makes a heterogeneity-calibrated test possible at all.
5. **§9.3** maps predictable reviewer objections onto concrete checks with citations.

### Required revisions

#### Critical
- [ ] **Primary A is not identified as specified** (§9.6 table). Noise in the x-axis halves the slope and inflates the intercept, so the "stereotyping" reading is mechanically favoured. Country-pair distance and question features can produce the "adapts" pattern without any knowledge of structure. *Fix:* see DA C1 and C2. Specify an errors-in-variables estimator. Read "changes regardless" from the reliability-gated low bin, not from the regression intercept. Add question and country-pair fixed effects or distance controls. Pre-specify equivalence bounds for each reading.

#### Major
- [ ] **The validity condition is mis-specified and unverifiable** (§9.6 status line). Gen/extract/map for all six selectors finished in Sept 2026 (§2, §5), so the y-axis (pick differentiation) can already be computed from files on disk. The x-axis is needed to pick the 30 transportability questions. *Fix:* extend the condition to "no cross-country comparison of mapped picks and no v4 heterogeneity values inspected". Have the PI sign a dated statement listing the file timestamps of `oracle_heterogeneity.csv` and any analysis notebooks. If anything was seen, register Primary A as "post-hoc-informed" and keep Primary 2 primary.
- [ ] **Primary A is unpowered** (§9.6 item 5). *Fix:* run the power analysis with dyadic-robust SEs (country pairs share countries) and errors-in-variables shrinkage at reliability 0.5 and 0.7, before locking. If the minimum detectable slope exceeds a pre-stated smallest slope of interest, Primary A cannot lead.
- [ ] **"So what?" is gap-based only** (§9.1 table, row 1). *Fix:* name the decision this informs, e.g. "can an analyst in a data-poor country trust an LLM's covariate advice for weighting, imputation or questionnaire design?". State what each of the three outcomes would change for that user.
- [ ] **Multiplicity** (§9.6, plus audit §A9). *Fix:* use a fixed-sequence or gatekeeping hierarchy. Example: Precondition (VoT, value over textbook) → Primary A pooled → Primary B pooled → per-selector readings under Holm. State that per-selector three-way labels are descriptive unless they pass the hierarchy.

#### Minor
- [ ] Reconcile the §9 access statement with the §9.5 Semantic Scholar citation check.
- [ ] Merge Hewitt et al. 2024 and Ashokkumar et al. 2026 into one entry. Upgrade Memar et al. (2610.28556, now verified) from "(snippet)". Record the published Collabra URL (§5).
- [ ] Move §9.4 before §9.5 or renumber.

### Line-level feedback
| Section | Issue | Recommendation |
|---|---|---|
| §9.1 row (1), Risk column | "Still informative, because it is the first test with the right ground truth" | A null is informative only with an equivalence test against a stated smallest slope of interest. Novelty of the instrument does not make a null informative. |
| §9.5 "Safe" wording | "in proportion to measured differences" | "in proportion" implies a calibrated slope (≈1 on a common scale). The two axes have different units, so say "co-vary with". |
| §9.6 Primary A | "intercept near the noise floor" / "well above" | Both thresholds are undefined. Pre-register numbers (e.g. equivalence ±δ in 1 − Jaccard units). |
| §9.6 item 1 | "two greedy draws overlap only 0.27–0.29 for DeepSeek and Kimi" | That figure is the *soft* Jaccard of *extracted feature lists*, for prompt-sensitivity-v2 selectors (DeepSeek, Kimi, MiniMax, Hermes). It is not mapped-code Jaccard for the confirmatory six. The prompt-sensitivity report says mapped codes behave differently (DeepSeek's codes overlap more). Re-quote the figure or drop it. |
| §9.6 item 5 | "3 pairs × 120 questions … plus 15 pairs × 30 questions at 6" | Double counts: the 3 original pairs of the 30 questions are inside the 15. Asian Barometer caps at 5 countries (10 pairs). The 6-country generation for the extension is "not started" (§2). |
| §9.1 positioning sentence | "across 132 countries" | Every heterogeneity contrast is within-survey, so mostly within-region (grid memo Q2 "Scope"). Don't let the headline sentence imply global cross-country calibration. |

### Summary
This is a strong self-critical inventory that has found the one question in the project prior work has not already answered. As a research plan it is not ready. The headline endpoint is a slope between two noisy, unsigned, set-overlap quantities. It has no identification strategy against the obvious third variables. The three-way reading is not falsifiable as written and is biased by known attenuation. Fixing this is mostly analysis design, not new data. The exception is one cheap generation job (a common-country panel, DA C2), which would make the slope interpretable. With the fixes below, Primary A could plausibly lead. Without them, keep the registered order and report Primary A as secondary.

---

## 2. devils_advocate_agent: Checkpoint 3 report

### Verdict: **REVISE**
Two CRITICAL issues block registering §9.6 as written.

### Steelman first
The strongest version of the proposal runs as follows. Every prior cross-country LLM paper either (i) scores simulated *answers*, from which predictor structure is recovered afterwards and distorted by answer-format artefacts, or (ii) studies one predictor. This project alone has a per-country, honestly split, reliability-measured importance oracle for 1,103 cells. It also has stated predictor sets from six models. So it can ask whether a model's change in stated structure tracks measured change in structure, and it can check that against a reliability-gated negative control. Prior work cannot. If the answer is "rigid", that is the first direct evidence that LLM covariate advice does not localise. If the answer is "adapts", it is a positive capability claim no one has made.

That steelman is sound about *the question*. The attacks below are on *the measurement*, and on whether the novelty is more than operationalisation.

### Critical issues (block progression)

**C1. The three-way reading (adapts / changes regardless / rigid) is biased by attenuation and not falsifiable as written**
- **Type:** Method / logical fallacy (false trichotomy; Texas sharpshooter risk)
- **Location:** §9.6 Primary A; glossary "adaptation test"
- **Problem:**
  - (a) Oracle reliability in quick fits is 0.47–0.54 (grid memo Q4), so an OLS slope arrives roughly halved. Because the intercept is ȳ − b̂·x̄, an attenuated slope also *inflates the intercept* by about b(1−λ)x̄, where λ is reliability. A truly adaptive selector is therefore pushed toward the "intercept well above floor, flat slope" cell, which the draft labels "stereotyping". The bias runs toward the most quotable negative finding.
  - (b) "Near the floor", "well above" and "flat" have no thresholds. Without them every result lands in some category, so the reading cannot fail.
  - (c) The categories are not exhaustive. A positive slope with a high intercept (adapts plus indiscriminate change), and a negative slope, are both possible. With six selectors, mixed labels are likely and will invite post-hoc stories.
  - (d) "Reliability-corrected heterogeneity" is not defined at the *pair* level. The screen defines het per *target* (within − between), not per country pair. Dividing by per-country reliabilities of about 0.5 gives heavy-tailed, unstable x values.
- **Impact:** The headline label attached to each selector could be an artefact of oracle noise.
- **Recommendation:**
  1. Define pair-level het explicitly. For example: het_AB = r̄_within(A,B) − agreement(A,B), where r̄ is the mean of A's and B's select-vs-score reliabilities and agreement uses the same metric.
  2. Estimate the slope with an errors-in-variables method fixed in advance. Options: an instrument from a split-half het (split the 50% ranking partition into two halves and use one half's het to instrument the other's); SIMEX using the known fold-level variance; or a latent-variable (Deming-type) model with the reliability fixed from the data.
  3. Test "changes regardless" *without the intercept*: mean (1 − J) − floor in the reliability-gated low bin (grid memo Q2 gate) is a direct, unbiased test, with an equivalence bound.
  4. Pre-register a full decision table over slope ∈ {>+δ_s, equivalent to 0, <−δ_s} × excess-at-low-bin ∈ {>+δ_i, equivalent to 0}, using two one-sided tests, with every cell labelled in advance.

**C2. Generic country distance and question-level features can produce the "adapts" pattern without knowledge of predictive structure**
- **Type:** Method / confounding (cum hoc; Goodhart)
- **Location:** §9.6 Primary A; grid memo Q2 "Scope"
- **Problem:**
  - (a) *Country-name heuristic.* A model that changes its picks more when two countries sound more different (different language, religion, income or region) produces a positive slope whenever true structural heterogeneity also grows with country distance, which is plausible. That is reacting to country names, not knowing where structure differs. The planned placebo (wrong or shuffled country on 30 questions × 1 country) shows only that *a* country name moves picks. It cannot separate distance-driven differentiation from structure-tracking.
  - (b) *Question-level co-variation.* About 72% of variance is between questions (grid memo). With 3 countries per question there are only 3 dependent pairs per question, so the pooled slope is identified mostly *between questions*. Here both axes can co-move through question features. Larger candidate pools and more ambiguous constructs mean noisier mapping (higher 1 − J) and noisier or more diffuse oracle top-10s (higher apparent het if the reliability correction is imperfect). Answer type matters (ordinal oracles are Spearman-scored, binary/nominal log-loss-scored), and so do model-chosen k (see M2), survey, and number of countries fielded.
  - (c) *Survey leverage.* WVS transportability countries are drawn region-stratified by rule, so WVS pairs are both more heterogeneous *and* more different by name. The pooled slope may just be "WVS vs regional barometers".
  - (d) *Sample-design differences.* Countries differ in n, fieldwork mode and translation. That raises measured het without the model's knowledge being involved.
- **Impact:** A positive Primary A would not license "the model knows where predictive structure differs". That is the core claim.
- **Recommendation:**
  1. **Identify within question and within pair.** Fit question fixed effects (or a Mundlak within/between split) and report the within-question slope as primary. The between-question slope is secondary.
  2. Add a **common-country panel**. For the 30 transportability questions, run the same 6 countries per survey (or a fixed 4 for every question in a survey), so each country pair recurs across about 5–10 questions and pair fixed effects can be fitted. This removes every pair-level confound (distance, n, mode). Identification then comes from "does the model differentiate this pair more on the questions where this pair's structure differs more?". Cost: country-named generations only, about tens of dollars at 2026-09-03 prices. It does not touch the nested-country rule for the confirmatory estimate, so register it as an added stratum.
  3. As a fallback, partial out country-pair distance covariates: a cultural distance index, |ΔGDP per capita|, geographic distance, a shared-language indicator, |Δn|. Report the slope before and after.
  4. Use survey fixed effects, report the slope per survey, and give WVS separately.
  5. Add a **label-permutation null**. Recompute the slope after permuting which country's picks are paired with which oracle pair, within survey and within distance strata. This gives a design-based p-value that respects the dependence structure.

### Major issues

**M1. Unsigned measure: "how much" is not "where" or "how"**
- **Location:** §9.6 Primary A.
- **Problem:** 1 − Jaccard rewards changing picks by the right *amount*, even if every change is toward variables that are less important in the target country. Audit §A9 already proposed a *directional* test: rank-correlate the model's cross-country delta with the oracle's. The draft drops it.
- **Recommendation:** Add a signed companion inside Primary A, for example directional captured importance: D_AB = CI_A(picks_A) − CI_A(picks_B) + CI_B(picks_B) − CI_B(picks_A), where CI is captured importance and both terms use honest score-split importances. Or use the share of codes in picks_A \ picks_B whose oracle-A rank beats their oracle-B rank. Report the magnitude slope only alongside a positive directional estimate.

**M2. 1 − Jaccard is mechanically tied to k and to k differences**
- **Location:** §9.6, plus audit §A6 (k is part of the measurand).
- **Problem:**
  - Jaccard is capped at min(k_A, k_B)/max(k_A, k_B). If naming a country changes how many variables a model asks for, differentiation rises without any change in content.
  - Chance overlap rises with k relative to pool size.
  - Oracle het is measured at a fixed top-10, while picks use model-chosen k (mean 10–19 in prompt-sensitivity v2). The two axes are measured at different depths.
- **Recommendation:**
  - Use a chance-corrected overlap, (J − E[J | k_A, k_B, N]) / (1 − E[J | ·]), or rank-biased overlap if a ranking is ever elicited.
  - Add |k_A − k_B| and mean k as covariates.
  - Measure oracle het at matched depth (top-k̄ of the pair) as a sensitivity check.

**M3. Noise floor: wrong source number, and a one-replicate floor has its own noise**
- **Location:** §9.6 item 1.
- **Problem:** The 0.27–0.29 figure is soft Jaccard on *extracted features* for DeepSeek and Kimi (prompt-sensitivity v2, four selectors that are not the confirmatory six). It is not mapped-code Jaccard. So the size of the floor for Primary A is unknown, and with it the headroom.
- **Recommendation:**
  - Keep the replicate (it is needed).
  - Compute a *per-cell* floor and use y* = (1 − J_AB) − ½(floor_A + floor_B).
  - Report floor reliability.
  - Check the headroom: if the mapped-code floor is around 0.65 dissimilarity, y has a range of about 0.35, which compresses the slope (ceiling effect).
  - Pre-specify what happens if the floor is ≥ 0.8.

**M4. Range restriction and leverage on the x-axis**
- **Location:** §9.6; grid memo Q2 ("~2/3 of questions have no cross-country structure").
- **Problem:** Within-region spans compress heterogeneity, most pairs pile up near het = 0, and only about 12 high-het questions carry the slope. Powdthavee's (2026) slope moves with one country, which §9.5 notes.
- **Recommendation:**
  - Make leave-one-question-out and leave-one-country-out influence diagnostics part of the registered output.
  - Use a robust (Theil–Sen or Huber) slope as co-primary or as a registered sensitivity check.
  - State the within-region scope in the endpoint text, not only in a design memo.

**M5. Mapper and extractor can create differentiation the selector did not intend**
- **Location:** §3 stages 8–9; §9.6.
- **Problem:** The country-named essay's context strings reach the fixed extractor and the mapper (Nemotron), which have their own country priors. The same construct can then map to different codes depending on surrounding country text.
- **Recommendation:** On a subset, re-map country-named features with country tokens masked from labels and context. Report the share of differentiation that disappears. Also report differentiation at the extracted-feature level (soft Jaccard), so that mapper-induced change shows up as a gap between the two levels.

**M6. Novelty is real but narrow; the "trivially implied null" risk is not handled**
- **Location:** §9.1 row (1); §9.5 Verdict and "Safe" wording.
- **Problem:** see §2b below. Prior work (Pataranutaporn 2025; Powdthavee 2026; Eriksson et al. 2026; Wang 2026; NHB 2026; Shi & Haupt 2026; Collabra 2025) strongly predicts rigidity or compression. A null then confirms a predicted result with a noisier instrument. A positive result needs C1 and C2 to be believed.
- **Recommendation:**
  - Pre-register the *smallest slope of interest* and an equivalence test, so a null is a bounded, informative "no".
  - Add a stated-vs-implied bridge on a subset: elicit simulated answers from the same model for the same cells, derive implied importance, and test whether stated and implied structure differ in how well they localise. This is what turns "stated" from a measurand label into a finding.

**M7. Multiplicity**
- **Location:** §9.6, plus audit §A9.
- **Recommendation:** Use a gatekeeping order as in the EIC section. Pool selectors for the primary tests. Treat per-selector three-way labels as descriptive unless they pass Holm within the hierarchy.

**M8. The validity precondition can be broken without anyone noticing**
- **Location:** §9.6 status line.
- **Problem:** The y data already exist. The 30-question selection needs het. Re-ordering after seeing the het distribution, for example because swap gain now looks underpowered, is a forking path even if no "score" was opened.
- **Recommendation:** Same as the EIC Major item: a dated attestation plus file timestamps. If anything was seen, Primary A is secondary.

### Minor issues
- **Dyadic dependence:** the 3 pairs within a question share countries. Use dyadic-robust or multiway (question × country) SEs.
- **Selection of the 30 questions on quick-oracle het:** selecting on x does not bias E[y | x], but het re-measured on balanced oracles will regress toward the mean. Report bin stability (quick vs balanced) as the grid memo already intends.
- **Codes unavailable in one country:** the mapper uses a survey-wide pool, so a pick may be all-missing in one country. The oracle het is restricted to shared features. Restrict 1 − J to codes present in both countries' pools.
- **Term collision:** "stereotyping" for a measurement category invites confusion with national stereotypes and with ACE-Align. Use a neutral label, e.g. "indiscriminate change".
- **Wording:** "in proportion to" in the safe claim should be "co-varies with" (different units on the two axes).

### Observations
- The registered Primary 2 (swap gain in the high bin) is less exposed to C1 and C2 than Primary A. It is directional by construction, it does not need an intercept reading, and the low bin gives a reliability-gated negative control. A defensible alternative to re-ordering: keep Primary 2 as the confirmatory test, make directional Primary A (M1) co-primary only if the power analysis supports it, and report the magnitude slope as the descriptive "battery".
- The Langseth, Hellton & Hammer (2026) result is relevant to Primary B. In that paper, LLM priors helped in-domain as much as out-of-domain, so "transfer" gains can be ordinary regularisation. The swap test is immune to this only because both arms use LLM picks. Say so.

### Strongest counter-argument
"The study regresses how much an LLM pipeline's mapped-code lists change between two countries on a within-region oracle disagreement score. Both quantities are dominated by noise: same-prompt replicates already disagree on most items, and oracle reliability is about 0.5. Both vary with question-level features, and the measure ignores whether the changes point toward the variables that matter. A model that simply changes its answer more for countries that sound more different would earn the 'adapts' label. A truly adaptive model would be pushed toward the 'stereotyping' label by attenuation. Prior work already shows LLMs compress country differences and give nearly identical predictor gradients across country groups, so the likely null is predicted, and as designed it cannot be told apart from noise."

### What's missing
- An identification strategy: within-question and within-pair variation, or distance controls.
- A directional measure.
- An errors-in-variables estimator.
- Equivalence bounds and a power analysis for Primary A.
- A stated-vs-implied bridge.
- An explicit use-case for the significance argument.
- Verification of the NHB 2026 per-country gradient question, and of the Ashokkumar et al. heterogeneity result. An earlier version reportedly found LLMs "reasonably accurate for interaction effects when heterogeneity is present" (search snippet only). That is the closest evidence on LLMs predicting where effects differ, in US subgroups.

### Stress test results
| Test | Result |
|---|---|
| Remove the strongest source: does the novelty argument hold? | **Yes, mostly.** Novelty rests on the absence of a matching paper. The new search (§5) found no pre-emption, but the argument stays an absence-of-evidence claim with access gaps (nature.com, pnas.org, mdpi.com, preprints.org, Semantic Scholar rate-limited). |
| Flip the research question: is the opposing view credible? | **Yes.** "LLMs are rigid about predictor structure across countries" is the literature's default expectation (§9.1 says so). |
| Apply to a different context: does the finding generalise? | **No, as designed.** Contrasts are within-survey, so mostly within-region. Only WVS spans regions. |
| "So what?": is the significance justified? | **Not yet.** It is argued only as a gap. A user-decision framing and an informative-null design are needed (EIC Major; M6). |

### 2b. Novelty attack (§9.5 and §9.6 claim: "first test of whether LLM-stated predictors change across countries in proportion to measured differences in data-driven importance")

**Steelman of pre-emption.**
1. *The logic already exists.* Pataranutaporn et al. (PNAS 2025) and Powdthavee (arXiv 2609.33055) test whether LLM-implied predictor effects track real per-country (or per-HDI-group) effects. Powdthavee reports a cross-country Spearman of 0.54–0.80 between real and LLM income gradients. Going from one predictor to a set is, by §9.1's own rule, breadth.
2. *Stated vs implied may not be a different capability.* LLM-Select finds stated importance agrees only loosely with any data importance. Ma & Powell show associations are inferred from wording. A reviewer can say "stated" is a measurement channel, not a new construct, unless the paper shows the two channels diverge.
3. *The likely result is trivially implied.* Country-level compression (Wang 2026; Eriksson, Vartanova & Strimling 2026; Chen et al., NHB 2026), "nearly identical" gradients across HDI groups (Pataranutaporn 2025), heterogeneity collapse (Shi & Haupt 2026, arXiv 2604.23575), and moderation being hard even for experts (Collabra 2025) together predict "rigid". Confirming a predicted null is a weak contribution.
4. *An adjacent pre-emption for the payoff half.* LLM priors for out-of-distribution transfer across real covariate splits are tested by Langseth, Hellton & Hammer (arXiv 2610.36806) and LoID (arXiv 2601.17609). Their result (priors mostly help as regularisation, not as shift correction) speaks to Primary B's interpretation.

**Assessment.** After the new search (§5), no paper was found that elicits stated predictor sets per country, scores them against per-country importance, and regresses differentiation on measured heterogeneity. The §9.5 "safe" wording survives *for that operationalisation*. The contribution is defensible only if the paper (i) argues why stated-structure localisation matters to users (M6 / "so what"), (ii) shows its result is not implied by the single-predictor work, ideally with the stated-vs-implied bridge, and (iii) makes a null informative. **Severity: MAJOR, not CRITICAL.** The claim is not pre-empted, but it is narrow and at risk of being called trivially implied.

---

## 3. ethics_review_agent: Ethics Review Report

### AI-assisted research-integrity verdict: **CONDITIONAL**
Scope: this verdict covers AI-assisted research integrity only. It is not human-subjects clearance or authorisation.

### Dimension assessment
| Dimension | Status | Notes |
|---|---|---|
| AI Disclosure | warn | The inventory does not say who or what compiled it. Its literature sweeps ("three parallel sweeps", snippet/full-text annotations) read as AI-agent output. The eventual paper needs a disclosure that separates three LLM roles: object of study (six selectors), instrument (Qwen extractor, Nemotron mapper, MiniLM retrieval), and research assistance (literature search, drafting, this review). |
| Attribution Integrity | pass (with fixes) | 11/11 spot-checked arXiv IDs resolve with matching titles and first authors: 2407.02694, 2507.06141, 2503.21011, 2601.12962, 2604.02458, 2605.08486, 2605.30934, 2609.01902, 2609.16395, 2609.30896, 2610.28556. Also 2609.33055, 2611.00256 and 2610.36806 were checked against abstracts. The 2611.00256 summary ("importance rankings … pooled across countries") is consistent with its full text: its cross-country analysis concerns levels, and predictor structure is compared for the pooled sample. The Collabra paper exists in Collabra 11(1) (URL in §5). Open items: the §9 vs §9.5 Semantic Scholar inconsistency; NHB 2026 and Ashokkumar 2026 still snippet-only; the Hewitt/Ashokkumar duplicate. |
| Dual-Use Screening | pass | Risk level: None–Low. |
| Fair Representation | warn | See advisory items below. |
| Data Ethics | pass (confirm) | Secondary analysis of published survey microdata. Only question wording, not respondent records, is sent to third-party LLM APIs, per the pipeline description. Confirm that each survey's licence (Afrobarometer, Arab Barometer, Asian Barometer, Latinobarómetro, ESS, WVS) permits the derived per-cell artefacts kept on a shared Dropbox, and any planned public release of oracles. |
| Conflict of Interest | pass | No funding or commercial interest stated. Note any provider credits used for the selector zoo. |
| Human Subjects Ethics | n/a | Secondary-data study of published, de-identified survey microdata. No new human subjects. ARS human-subjects authority and packet machinery is not applicable and was not run. |

### Issues found
#### Critical
No critical integrity issues. No fabricated references were found among those checked.

#### Conditional (must fix)
- Add an AI-involvement statement to the inventory and plan one for the paper, covering the three roles above.
- Resolve or annotate the Semantic Scholar access contradiction (§9 caveat vs §9.5).
- Before citing numbers from the NHB 2026 and Ashokkumar et al. 2026 papers, read their full texts. Gray zone = do not cite numbers.

#### Advisory
- **Representation framing.**
  - §9.2 proposes stratifying by a "representation proxy (GDP, web-language share)". Report this as a property of *model training coverage*, not of the countries. Avoid "LLMs fail on poorer countries" phrasing that reads as a deficit of the populations.
  - The within-region design means African, Arab and Asian countries are compared only with their regional neighbours. Do not let pooled results be read as statements about the "Global South" as a block.
  - The text does not currently use "Global South". Keep it that way unless defined.
- **"Stereotyping" label.** Using a socially loaded word for a statistical category about countries risks misreading, as if the models hold national stereotypes. Prefer "indiscriminate change".
- **Country naming in outputs.** Per-country "model misreads this country" tables can be quoted out of context. Pair every such table with oracle reliability and sample size.
- **Attribution in this review.** All papers cited here were seen on arxiv.org or the abstract API, except two hits that could not be opened (mdpi.com and preprints.org are outside the session's network allowlist). Those two are listed as unverified and are not relied on.

### Responsible use statement
Not required (dual-use risk below Moderate).

### Ethics decision log
No user decisions recorded yet; the CONDITIONAL items are pending.

> **Human-subjects boundary:** This output does not authorise recruitment, consent, access to identifiable data, intervention, or data collection.

---

## 4. Revision recommendation summary

### Issues raised by more than one agent
1. **Primary A is not identified as written:** attenuation biasing the three-way reading, uncontrolled confounding, no thresholds (EIC Critical; DA C1, C2).
2. **The pre-registration validity condition cannot be verified and is too narrow** (EIC Major; DA M8).
3. **Significance is argued only as a gap; a null is not yet informative** (EIC Major; DA M6; stress test "so what").
4. **Source hygiene:** snippet-only load-bearing sources, the duplicate entry, the access-statement contradiction (EIC Minor; Ethics Conditional).

### Priority revision items
| Priority | Revision | Source |
|---|---|---|
| 1 | Define pair-level het; use an errors-in-variables slope (split-half IV, SIMEX or a latent model); test "changes regardless" in the reliability-gated low bin, not via the intercept; pre-register a full slope × excess decision table with equivalence bounds | DA C1, EIC |
| 2 | Identify within question (question FE / Mundlak) and within pair (common-country panel on the 30 transportability questions; else distance covariates); survey FE; label-permutation null | DA C2 |
| 3 | Dated attestation that neither cross-country pick comparisons nor v4 het were inspected; otherwise keep the registered order | EIC, DA M8 |
| 4 | Power analysis for Primary A with dyadic SEs and attenuation at reliability 0.5 and 0.7; set the smallest slope of interest | EIC, DA |
| 5 | Add a directional measure (signed captured-importance cross-delta, audit §A9) | DA M1 |
| 6 | Chance-corrected, k-adjusted overlap; matched-depth het | DA M2 |
| 7 | Re-measure the noise floor on mapped codes for the confirmatory six; per-cell floor; headroom rule | DA M3 |
| 8 | Robust slope, leave-one-question-out and leave-one-country-out as registered outputs; state within-region scope in the endpoint text | DA M4 |
| 9 | Country-masked re-mapping check; extracted-level differentiation | DA M5 |
| 10 | Gatekeeping hierarchy across the precondition, Primary A and Primary B; per-selector labels descriptive | EIC, DA M7 |
| 11 | User-decision "so what"; stated-vs-implied bridge on a subset | EIC, DA M6 |
| 12 | AI disclosure; source-hygiene fixes; neutral label in place of "stereotyping"; representation framing | Ethics, EIC Minor |

### Bottom line for the PI
The question is new enough to lead the paper. The endpoint is not yet good enough to lead the paper. Approve the re-ordering only after items 1–4 are written into the registry draft. If the common-country panel (item 2) is not affordable or the power analysis fails, keep Primary 2 (swap gain) as the primary and register Primary A, made directional, as the leading secondary.

---

## 5. Counter-evidence search log (Devil's Advocate, Anti-Pattern #1)

**Date:** 2026-11-08. **Tools:** arXiv export API (HTTPS; plain HTTP failed DNS), 3 s sleeps between calls; arxiv.org HTML full text for two papers; WebSearch (extended mode); WebFetch; Semantic Scholar API. Downloads are in `.../scratchpad/review_dl/` and were treated as untrusted data; parsing used `python3 -I`.
**Access failures:** Semantic Scholar API returned HTTP 429 on all 4 attempts, so no citation-graph search was possible. mdpi.com and preprints.org do not resolve in this session (outside the network allowlist). nature.com and pnas.org full texts were not attempted (blocked earlier per §9.5). openreview.net is reachable (HTTP 200), but no OpenReview-hosted hit surfaced beyond the ICLR 2026 workshop paper LLM-FE.

### Queries
arXiv API (`abs:` field search; 22 queries):
1. "large language model" AND "feature importance" AND countr* → 0 hits
2. LLM AND "feature selection" AND (context OR domain OR subgroup) AND (shift OR heterogeneity) → 3 (all clinical, irrelevant)
3. "language model" AND "heterogeneous treatment effects" AND (predict OR forecast) → 1 (Twin-2K-500, irrelevant)
4. LLM AND moderat* AND (survey OR "social science") AND (effect OR association) → 0
5. "language model" AND "interaction effects" AND knowledge → 8 (SleuthBench relevant; rest irrelevant)
6. LLM AND prior AND (environment OR domain) AND causal AND "distribution shift" → 1 (irrelevant)
7. LLM AND "cross-national" AND (predictors OR correlates OR determinants) → 3 (Li & Ji 2604.02458 and Wang 2609.16395, both already cited)
8. "language model" AND "World Values Survey" AND (correlat* OR predictor*) AND countr* → 0
9. LLM AND "domain-adaptive" AND "feature selection" → 1 (irrelevant)
10. LLM AND "variable selection" AND (transfer OR transportab*) → 0
11. "language model" AND "feature selection" AND survey → 4 (2611.00661 adjacent)
12. LLM AND "predictor structure" → 3 (2611.00256, already cited)
13. LLM AND "which variables" AND (predict OR matter) → 2 (irrelevant)
14. "silicon sampling" AND (heterogeneity OR moderation OR interaction) → 4 relevant (2604.23575, 2604.06663, 2608.14079, 2609.16395)
15. LLM AND "context-dependent" AND "feature importance" → 0
16. LLM AND stated AND revealed AND (importance OR predictors) → 15 (none relevant)
17. "language model" AND "cross-country" AND (importance OR determinants OR predictors) → 1 (2611.00256)
18. LLM AND subgroup AND "effect heterogeneity" AND (forecast OR predict OR elicit) → 0
19. "language model" AND Afrobarometer → 0
20. "language model" AND (Latinobarometro OR "Arab Barometer" OR "Asian Barometer") → 0
21. LLM AND "correlation structure" AND survey AND countr* → 0
22. LLM AND "variable importance" AND (survey OR "social science") → 1 (irrelevant)

WebSearch (8 queries):
- "LLM stated predictors differ across countries feature importance heterogeneity survey data 2026"
- "large language models knowledge of moderation effects cross-cultural correlates country heterogeneity"
- "LLM expert elicitation which variables predict outcome subgroup-specific predictors compared with data-driven importance"
- "heterogeneity forecasting LLM predict which countries effect differs survey experiments many-analysts 2026"
- "IC2S2 2026 LLM survey variable selection cross-national predictors"
- survey-methods venues (POQ / SMR / JSSAM) + LLM predictors vary by country
- "Collabra 2025 LLMs vs human experts variable relationship identification moderation"
- "openreview 2026 workshop LLM feature selection country-specific / site-specific / population-specific priors tabular"
- "do LLMs know how associations differ across contexts … interaction effects … 2026"

Semantic Scholar (3 queries, all HTTP 429): "language model predict cross-country differences in determinants survey"; "LLM effect modification across populations elicitation"; "large language model heterogeneity of associations across countries".

### Candidates found, with closeness to the §9.6 claim
**Scale:** Pre-empts = the same test. High = the same logic, different measurand. Moderate = shares a key component. Low = background.

| # | Paper | URL | Verified | Closeness | Why |
|---|---|---|---|---|---|
| 1 | Abels, Fernandez Domingos, Shah, Lenaerts (2026). Improving Cross-Cultural Survey Simulation with Calibrated Value Personas | https://arxiv.org/abs/2605.16193 | abstract + full-text scan | **Moderate–Low (new)** | Finds that the contribution of persona value items to simulation accuracy "varies substantially across countries". These are inputs to a simulator, not LLM-stated predictors scored against an importance oracle. Cite as adjacent evidence that the relevant predictors differ by country. |
| 2 | Wightman, Bied, De Bie (2026). People Are Not Just Their Countries: Social Determinants of LLM Value Alignment Across Europe (ESS; AIES 2026 per search snippet) | https://arxiv.org/abs/2608.07367 | abstract | **Low (new)** | Country vs socio-demographics as predictors of *alignment error*. Not the model's view of predictors. |
| 3 | Langseth, Hellton, Hammer (2026). LLM knowledge for improved OOD tabular prediction | https://arxiv.org/abs/2610.36806 | abstract | **Moderate for Primary B** (already in §9.3) | LLM coefficient priors across real covariate splits help in-domain as much as OOD, so "transfer" gains can be regularisation. Use this in interpreting the swap test. |
| 4 | Rezaeimanesh et al. (2026). LoID: non-generative prior extraction for generalization | https://arxiv.org/abs/2601.17609 | abstract | **Low–Moderate (new)** | LLM priors under synthetic covariate shift. A different shift notion and a pooled feature-effect prior. |
| 5 | Shi & Haupt (2026). The Collapse of Heterogeneity in Silicon Philosophers | https://arxiv.org/abs/2604.23575 | abstract | **Low (new)**; supports the "trivially implied null" attack | LLMs over-correlate judgements and collapse group heterogeneity. |
| 6 | Na, Zhu, Cross, Bakker (2026). Designing Behavioral Profiles for LM Simulation of OOD Human Behavior | https://arxiv.org/abs/2611.00661 | abstract | **Low–Moderate (new)** | Feature selection for LLM simulation, evaluated under design shift. Feature choice transfers across designs, but this is not per-population importance. |
| 7 | Jia, …, Caruana, Lengerich (2026). SleuthBench | https://arxiv.org/abs/2609.34228 | abstract | **Low (new)** | LLM agents recover feature contributions (including injected interaction effects) poorly, at 41.9%. Data-analysis agents, not priors. |
| 8 | Brima et al. (2026). Few-shot cross-country generalization … childhood anemia (DHS, 16 countries) | https://arxiv.org/abs/2605.26589 | abstract | **Low (new)**; method analogue | LOCO and reverse-LOCO with asymmetric transfer, plus SHAP per country. No LLM. A precedent for leave-one-country-out reporting. |
| 9 | Vivalt (2026). Can AI Forecast Treatment Effects in Economics? | https://arxiv.org/abs/2610.38883 | abstract | **Low (new)** | Forecasts effect *magnitudes* with a post-cutoff design. A model for the memorisation control. |
| 10 | Czolgowski & Iyasele (2026). Cultural Misalignment in LLMs | https://arxiv.org/abs/2609.04485 | abstract | **Low (new)** | Persona-level distributional misalignment across 3 countries. |
| 11 | "Automating Social Science: LLMs vs. Human Experts in Variable Relationship Identification", *Collabra* 11(1) 144442 | https://online.ucpress.edu/collabra/article/11/1/144442/213650/Automating-Social-Science-LLMs-vs-Human-Experts-in (preprint: https://doi.org/10.31234/osf.io/mkxdt) | search result pages only | **Moderate** (already in §9.5) | Confirms publication. Moderation items are hard for LLMs and for experts; 247 relationships from 56 social-psychology meta-analyses. Supports the predicted null. |
| 12 | Ashokkumar, Hewitt, Ghezae, Willer (*Nature* 2026) | https://www.nature.com/articles/s41586-026-10742-x | snippet only | **Moderate–High for the "knows where effects differ" logic** (already cited) | A search snippet attributes to an earlier version the finding that predictions were "reasonably accurate for interaction effects when heterogeneity … is present". This is the closest evidence on LLMs predicting where effects differ (US demographic subgroups, treatment effects). Read the full text before positioning. |
| 13 | D'Ecclesiis, Castellacci, Ahmadi (2026) | https://arxiv.org/abs/2611.00256 | full-text scan | Moderate (already cited) | Checked: cross-country analysis concerns levels; predictor-importance comparison is for the pooled sample. §9.5's description stands. |
| 14 | Powdthavee (2026) | https://arxiv.org/abs/2609.33055 | abstract | High (already cited; closest) | The abstract emphasises the income gradient in *dispersion* across groups, robust to country fixed effects. Confirms §9.5's description. |
| — | Preprint on ANES backstories and LLM variable selection (preprints.org 202507.0531) | https://www.preprints.org/manuscript/202507.0531 | **NOT verified** (host blocked) | Unknown; US-only per snippet | Excluded under the ARS rule that unverified items are treated as failing. Re-check if the host is allowed. |
| — | German GSS persona attribute selection by variable importance (MDPI *Entropy* 27(9):923) | https://www.mdpi.com/1099-4300/27/9/923 | **NOT verified** (host blocked) | Unknown; one country per snippet | Excluded; re-check. |

**Result:** no pre-empting paper was found. Two new adjacent papers are worth citing (#1 Abels et al.; #6 Na et al. for feature selection under design shift). Several new background papers strengthen the "null is predicted" attack (#5, #7, #11). The closest unresolved risk is #12: verify whether Ashokkumar et al. test subgroup heterogeneity prediction in a way that speaks to "knowing where".
