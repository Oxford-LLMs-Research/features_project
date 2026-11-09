# Fact-check report: `docs/project_inventory_2026-11.md` §9 (Related work and novelty check)

*Method: ARS deep-research fact-check mode, run in the source_verification_agent role, on 2026-11-08.
AI-assisted: this report was produced by an AI agent (Claude) using the tools listed below.*

## How it was checked

- **Existence (Tier 0/1):** all 55 arXiv IDs in §9 were batch-queried against the arXiv API (all 55 resolve).
  DOIs, PMCIDs and PMIDs went through the Semantic Scholar batch API, and titles without an ID went through
  S2 `search/match`.
- **Claim alignment:** full texts were read for 2407.02694, 2503.21011, 2601.12962, 2604.02458, 2604.21120,
  2607.24782, 2608.07367, 2609.04013, 2609.16395, 2609.33055, 2611.00256, the vaping medRxiv paper and the
  arXiv version of Pataranutaporn et al. The rest were checked against arXiv/S2 abstracts. Where only a
  search-engine snippet was reachable, that is marked "(snippet-level)" in the grade.
- **Blocked hosts:** nature.com and pnas.org. cambridge.org and gsb.stanford.edu also failed to resolve
  through WebFetch in this session; Bisbee and Hewitt were checked through search snippets instead.
- **Evidence grade** = study-design level (I–VII) / overall grade (A–F), per `source_quality_hierarchy.md`.
  The field is Technology/Social Science, where the gold standard is Level III. Peer-reviewed controlled
  computational or observational studies get III/B. Preprints get III/C, because the Peer Review criterion
  is graded D. Method and theory papers get "n/a (method)/B".
- **Verdicts:** PASS · CORRECT-METADATA (exists, but a detail is wrong) · CLAIM-MISMATCH (our summary
  misstates the source) · FAIL (cannot confirm it exists, so remove it) · UNVERIFIABLE-HERE (exists, but
  its content is behind a blocked host).

---

## A. §9.2 Must-read papers

| # | Citation (as in doc) | Verdict | Grade | Evidence / note | Replacement text (if needed) |
|---|---|---|---|---|---|
| 1 | Jeong, Lipton, Ravikumar, LLM-Select, TMLR 2025, arXiv 2407.02694 | PASS | III/B | Full text, Result 4 / Fig. 5: "LLM-Score generally exhibits higher rank correlation with standard feature importance metrics as model scale increases, but does not uniquely align to a specific notion of feature importance." Matches our summary, and TMLR 04/2025 is confirmed. | — |
| 2 | Zhang, Zhao, Hu, Le, AI-driven feature selection… vaping, medRxiv Mar 2026 | PASS | III/C (preprint) | Full text: 7,943 youth; 214 PATH variables; 15 runs; temperature 0.7; Qwen-2.5-72B reaches AUC 0.791 with 30 predictors vs a baseline of 0.768; posted 9 Mar 2026; DOI 10.64898/2026.03.06.26347816. | — |
| 3 | Pataranutaporn et al., "LLMs predict human well-being — but not equally everywhere", arXiv 2507.06141; PNAS 122(48) e2519394122 (2025), PMC12685116 | CORRECT-METADATA | IV/B | The DOI, PMCID and authors all resolve. **However, the PNAS version has a different title**: "Simulating human well-being with large language models: Systematic validation and misestimation across 64,000 individuals from 64 countries". The claims are confirmed in the arXiv full text ("despite having access to country information"). They could not be re-checked in the PNAS text (pnas.org is blocked). | "Pataranutaporn, Powdthavee, Archiwaranguprok, Maes, **Simulating human well-being with large language models: Systematic validation and misestimation across 64,000 individuals from 64 countries**, PNAS 122(48) e2519394122 (2025) [preprint: *LLMs predict human well-being — but not equally everywhere*, arXiv 2507.06141]" |
| 4 | Chen, Wang, Wang, Bhatia, Simulated responses… 74 countries, Nature Human Behaviour 2026 | UNVERIFIABLE-HERE | IV/B (abstract-level) | S2 confirms the title, the four authors (Fa-Dong Chen, Qi-Jun Wang, Lei Wang, Sudeep Bhatia), the venue and DOI 10.1038/s41562-026-02611-4. The abstract is consistent with our summary: association direction is recovered for 77.8% of gender/age/cognitive-ability dimensions but only 48.1% of cultural dimensions. The abstract also says alignment "is weaker when removing demographic information than when removing country labels". Full text is on nature.com (blocked). | Optional sharpening: "Directions of demographic associations recovered in 77.8% of cases, cultural dimensions in 48.1%; weaker in less-developed countries" |
| 5 | Memar et al., When do LLMs help with feature selection?, arXiv 2610.28556 *(snippet)* | PASS | III/C (preprint) | Abstract confirms: 18 datasets; "classical selectors generally achieve higher predictive performance than the tested 7B LLM/RL selectors". The name effect is large for regression (+0.196 R² vs anonymised names) but "smaller… and dataset-dependent" for classification. | Drop "*(snippet)*". Authors: Memar, Wang, Agah. Optional: "meaningful names help mainly for regression targets" |
| 6 | Ma & Powell 2025, arXiv 2503.21011; Hommel & Arslan, AMPPS 2025 — "Item–item correlations inferred from wording (r ≈ .7–.9); overconfident; contamination risk" | **CLAIM-MISMATCH** | Ma & Powell: III/B (CogSci 2025); Hommel & Arslan: III/B | **This row merges two different papers into one summary, which is a vibe-citing risk.** Ma & Powell (GPT-4o, attitudes): estimated and human correlations r = .77, and still r = .724 for semantically dissimilar pairs. GPT-4o estimates are "more extreme, following a bimodal distribution". Neither "overconfident" nor "contamination" appears in the paper. Hommel & Arslan instead use a *fine-tuned embedding model* (SurveyBot3000), not a prompted LLM: ".71 for item correlations, .89 for reliabilities, and .89 for scale correlations" in the pilot, and ".59 … .84 … .84" in the preregistered validation. The ".9" figure is for reliabilities and scale correlations, not item–item correlations. | "Ma & Powell (CogSci 2025): GPT-4o-estimated correlations among attitudes track human ones (r = .77; r = .72 for semantically dissimilar pairs), but its estimates are more extreme than the human ones. Hommel & Arslan (AMPPS 2025): a fine-tuned sentence-embedding model predicts item–item correlations out of sample at .71 (pilot) and .59 (preregistered validation), and scale reliabilities and inter-scale correlations at .84–.89." Drop "overconfident; contamination risk" unless another source is cited for it |
| 7 | Wang, Silicon sampling answers with country-level assumptions, ESS, arXiv 2609.16395 *(snippet)* | PASS | IV/C (preprint) | Full text and abstract: "The respondent's country label acts as a country-level assumption"; "An average of neighboring countries, using no LLM, recovers country levels more accurately than every model condition". | Drop "*(snippet)*" |
| 8 | Wedgwood et al., Personalization, personas, and forecasting in value alignment, ICML 2026 | PASS | III/B | arXiv 2607.24782 (Wedgwood, Thaker, Kale, Smith). Full text: forecasting framing is strongest for 3 of 4 hosted models; "religiosity… while institutional trust and democracy-related questions remain harder". | Optional: add "arXiv 2607.24782" |
| 9 | Hewitt, Ashokkumar, Ghezae, Willer, Predicting results of social science experiments using LLMs, 2024 (Stanford GSB) | CORRECT-METADATA | III/A (published version) | **This is the preprint of the same work that §9.5 cites as "Ashokkumar, Hewitt, Ghezae, Willer, *Nature* 2026"**, so one study is being cited twice under two author orders. Snippet-level checks of the preprint confirm 70 experiments, r = 0.85, overestimated effect sizes, and r = 0.90 on unpublished studies. The Nature version (656, 115–122) uses 469 effects and 119,330 participants, against 476 and 105,165 in the preprint. Its r value was not verified (nature.com is blocked). | "Ashokkumar, Hewitt, Ghezae, Willer, **Large language models can predict the results of social science experiments**, *Nature* 656, 115–122 (2026), doi 10.1038/s41586-026-10742-x (preprint 2024: r = 0.85 across 70 survey experiments; effect sizes overestimated; accuracy held for studies unpublished at the training cutoff)". Use the same citation in §9.3 ("Hewitt 2024") and §9.4B |

## B. §9.5 Moderation novelty check

| # | Citation | Verdict | Grade | Evidence / note | Replacement text |
|---|---|---|---|---|---|
| 10 | Powdthavee 2026, arXiv 2609.33055 — "Spearman 0.54–0.80" | PASS | IV/C (preprint) | Full text: "Rank correspondence was relatively stable across models, with Spearman correlations ranging from 0.54 to 0.80". The correlation uses **65** countries (Iraq excluded) and is labelled "Exploratory". WVS-7 and six LLMs are confirmed. | Optional: "(Spearman 0.54–0.80 across 65 countries, exploratory)" |
| 11 | Luo et al. 2026, ACE-Align, arXiv 2601.12962 | PASS | III/B | Full text, Table 2: Flipped / Stereotyping (\|Δm\| > \|Δd\| + ε, "Overestimated effect") / Erasure / Aligned. "the dominant attribute varies substantially across topics and across countries"; 14 countries; 4 attributes. Accepted at EMNLP 2026. | Optional: add "EMNLP 2026" |
| 12 | Pataranutaporn et al. 2025 (9.5 row claims: HDI gradient, "nearly identical", country effects shrunk) | PASS (metadata fix as in #3) | IV/B | Full text: "all four LLMs predict nearly identical income–satisfaction gradients across both groups"; "muted country fixed effects"; the spillover from injected facts is confirmed. | Apply the #3 title fix |
| 13 | D'Ecclesiis, Castellacci, Ahmadi 2026, arXiv 2611.00256 | PASS | IV/C | Full text: predictor-importance rankings of observed vs LLM outcomes, compared by Spearman and Top-5/Top-10 overlap. | §9.4B: drop "*(snippet)*" and add the authors |
| 14 | Li & Ji 2026, arXiv 2604.02458 — "ICC 12–16% for effect estimates" | PASS | IV/C | Full text: country-intercept ICCs "increased from 0.8%-1.6% for MAE to 11.8%-16.1% for absolute ATE error". | Optional: "ICC 12–16% for treatment-effect error (vs ≈1% for response error)" |
| 15 | Ashokkumar, Hewitt, Ghezae, Willer, *Nature* 2026 — "subgroup accuracy holds but little heterogeneity existed" | UNVERIFIABLE-HERE | III/A (metadata) | The work exists (S2: *Nature* 656:115–122). The heterogeneity claim is not in the abstract, and the full text is on nature.com (blocked). | Keep "snippet" until the full text is read. See #9 for the duplicate citation |
| 16 | Collabra 2025 "LLMs vs human experts in variable relationship identification" (unverified) | CORRECT-METADATA | III/B | **The venue is confirmed**: S2 lists Ji, Xue, Chen, Zhu, *Collabra: Psychology* 2025, doi 10.1525/collabra.144442. The preprint abstract says LLMs ≈ domain experts on simple relations, and that "moderating effects presented a challenge for both". The relations came from 56 meta-analyses (247 relations). | "Ji, Xue, Chen, Zhu (2025) Automating social science: LLMs vs. human experts in variable relationship identification, *Collabra: Psychology*, doi 10.1525/collabra.144442". Remove "unverified" |
| 17 | Eriksson, Vartanova, Strimling 2026, arXiv 2609.30896 | PASS | IV/C | "estimating differences between societies to be… less than half their measured size". | — |
| 18 | Tao et al. 2026, arXiv 2605.08486 | PASS | IV/B (L@S'26) | TALIS, 55 countries: "Models compress country differences". | Optional: title "Teachers' Perceived Benefits and Risks of AI Across Fifty-Five Countries" |
| 19 | Daryani, Bogen, Daepp 2026, arXiv 2609.01902 | PASS | IV/C | "introduce movement where little occurred". | — |
| 20 | Cui et al. 2025, *Nat Comput Sci* | UNVERIFIABLE-HERE | III/B (snippet-level) | Existence is confirmed through the arXiv preprint 2409.00128 (Cui, Li, Zhou) and a Tsinghua feature: *Nat Comput Sci* 5(8). The preprint reports main effects replicated at 73–81% and interactions at 46–63%, which supports the direction of our claim. The journal text is on nature.com (blocked). | "Cui, Li, Zhou (2025) Replication of scenario-based experiments in psychology and management using LLMs, *Nat Comput Sci* 5(8) [arXiv 2409.00128]" |
| 21 | Powdthavee 2026, arXiv 2605.30934 | PASS | III/C | Abstract: cross-linguistic moral divergence "was associated with real-world institutional differences" (Study 2). | — |
| 22 | Le et al. 2025, *Nicotine & Tobacco Research* (GPT-4.1, smoking cessation) | PASS | III/B (via reference list) | Vaping paper ref [16]: Le, Yang, Zhao, Zhang, Li, Hu, "Identifying key predictors of smoking cessation success: text-based feature selection using a large language model", NTR ntaf257 (2025); the text states that it used GPT-4.1. | — |
| 23 | §9.5 controls: "Powdthavee's result moves with one country" | **CLAIM-MISMATCH** (minor) | — | Source: Spearman correlations are "relatively insensitive to retaining Iraq, whereas Pearson correlations are considerably more sensitive". Only the Pearson correlations and the slopes move with Iraq. | "(Powdthavee's Pearson correlations and slopes move with one country, Iraq; the rank correlations do not)" |

## C. §9.4 Annotated bibliography sweep (rows not already covered above)

**A. LLMs as feature selectors**

| Citation | Verdict | Grade | Note / fix |
|---|---|---|---|
| Choi et al. 2022 LMPriors, arXiv 2210.12530 | PASS | III/C (workshop) | — |
| Li, Tan, Liu 2025, SIGKDD Explorations, 2408.12025 | PASS | III/B | Published in SIGKDD Explorations 26(2):44–53. §9.1 calls it a "survey". It is an empirical comparison of data-driven vs text-based selection, so "study" is more accurate |
| Zhang, …, Tibshirani 2025 LLM-Lasso, 2502.10648 | PASS | III/C | — |
| Li, Xiu 2025 LLM4FS, 2503.24157 | CORRECT-METADATA | III/F (withdrawn) | The arXiv abs page says: "This paper has been withdrawn by Xianchao Xiu" (v4, 10 Dec 2025). Fix: "Li, Xiu (2025) LLM4FS, arXiv 2503.24157 (**withdrawn Dec 2025**; do not cite as evidence)" |
| Yang et al. 2024 ICE-SEARCH, 2402.18609 | PASS | III/C | — |
| "Lee, Shen et al." 2024 FREEFORM, 2410.01795 | CORRECT-METADATA | III/B | The authors are Joseph Lee … Li Shen (11 authors), and Shen is the senior/last author. Fix: "Lee, …, Shen (2025) FREEFORM, AMIA Informatics Summit" |
| Skinner, Guo, Li 2026, 2605.23102 | PASS | III/C | — |
| Bal-Ghaoui, Sabri 2025, 2510.05935 | PASS | III/C | — |
| Hollmann et al. 2023 CAAFE, 2305.03403 | PASS | III/B | — |
| Han et al. 2024 FeatLLM, ICML | PASS | III/B | S2: Han, Yoon, Arik, Pfister, ICML 2024 |
| Abhyankar et al. 2025 LLM-FE, 2503.14434 | PASS | III/B | Add "TMLR" |
| LLM4CKD (2026), 2609.04013 *(snippet)* | PASS | III/B (ICDM 2026) | Full text: "LLM rankings show moderate to strong agreement with each other… correlations between the ML reference ranking and LLM rankings are generally low". Fix: "Kabir, Munira (2026) LLM4CKD, ICDM 2026"; drop "*(snippet)*" |
| TabSHAP (2026), 2604.21120 *(snippet)* | PASS | III/C | §1: "LLMs exhibiting semantic biases toward text-rich features". The paper is about *fine-tuned LLM classifiers*, not selectors. Fix: "Chaudhary, Agarwal, Alladi (2026)"; drop "*(snippet)*" |
| Cruz, Hardt, Mendler-Dünner 2024, NeurIPS, 2407.14614 | PASS | III/B | — |
| LoID (2026), 2601.17609 | PASS | III/C | Abstract: "synthetic out-of-distribution (OOD) settings". Add "Rezaeimanesh et al. (2026)" |
| "Huntington-Klein et al. (2024)", 2412.10635 | CORRECT-METADATA | IV/B | There are two authors, and the paper is published. Fix: "Huntington-Klein & Murray (2024/25) Do LLMs act as repositories of causal knowledge?, *Journal of Causal Inference*, doi 10.1515/jci-2024-0072". The abstract supports the option-ordering point |

**B. LLMs and surveys**

| Citation | Verdict | Grade | Note / fix |
|---|---|---|---|
| Argyle et al. 2023, Political Analysis | PASS | IV/B | PA 31:337–351 |
| Bisbee et al. 2024 — "48% of coefficients wrong" | **CLAIM-MISMATCH** (minor) | IV/B (snippet-level) | The source (snippet-level) says about half (48%) of regression coefficients **differ significantly** from the ANES estimates, and 32% of those have flipped signs. "Wrong" overstates it. Fix: "48% of regression coefficients differ significantly from ANES estimates (sign flipped in about a third of those) → motivates direct elicitation". PA 32:401–416, doi 10.1017/pan.2024.5 |
| Dominguez-Olmedo et al. 2024, NeurIPS | PASS | III/B | — |
| Santurkar et al. 2023, ICML | PASS | IV/B | — |
| Durmus et al. GlobalOpinionQA | PASS | IV/C | Abstract: country prompting shifts answers "but can reflect harmful cultural stereotypes" |
| Suh et al. 2025 SubPOP, ACL | PASS | III/B | — |
| Cao et al. 2025, NAACL | PASS | III/B | — |
| Kim & Lee 2023 AI-augmented surveys | PASS | III/C | — |
| Miranda & Balbi 2025, Entropy | PASS | IV/B | Exists (doi 10.3390/e27090923). The abstract describes no variable-selection step: it compares LLMs with an RF on prediction and distributions. This lowers the "Med?" relevance |
| Wightman, Bied, De Bie 2026, AIES | PASS | IV/B | Full title: "People Are Not Just Their Countries. Disentangling Social Determinants of LLM Value Alignment Across Europe"; uses the ESS |
| Ku et al. 2026, 2607.03091 | PASS | IV/C | — |
| Geng et al. 2405.19323; von der Heyde et al. 2409.09045 | PASS | IV/C | — |
| Tao et al. 2024, PNAS Nexus | PASS | IV/B | doi 10.1093/pnasnexus/pgae346 |
| Atari et al. 2023 Which humans? | PASS | IV/C (preprint) | PsyArXiv 10.31234/osf.io/5b26t |
| AlKhamissi et al. 2024, ACL | PASS | IV/B | — |
| Cao et al. 2023, 2303.17466 | PASS | IV/C (C3NLP workshop) | — |
| Abeliuk, Gaete, Bro — "Fairness in LLM-simulated public opinion", EPJ DS | CORRECT-METADATA | IV/B | The real title is **"Fairness in LLM-Generated Surveys"** (arXiv journal_ref: EPJ Data Science 2026). The US vs Chile content is confirmed |
| Manning, Zhu, Horton 2024 Automated social science, NBER | PASS | III/C | Exists (arXiv 2404.11794; SSRN). The NBER number w32381 was not independently resolved |
| Ludwig & Mullainathan 2024, QJE | PASS | VII/B | NBER w31017 |
| Zhou et al. 2024 HypoGeniC | PASS | III/C | ACL 2024.nlp4science-1.10 |

**C. Importance methodology and transfer**

| Citation | Verdict | Grade | Note / fix |
|---|---|---|---|
| Salganik et al. 2020, PNAS | PASS | III/A | Exists (doi 10.1073/pnas.1915006117; PNAS 117:8398–8403). The scholarworks.montana.edu link was not checked; prefer the DOI |
| Lundberg et al. 2024, PNAS, PMC11181083 | PASS | IV/B | — |
| Kern, Klausch, Kreuter 2019, SRM | PASS | VII/B | — |
| Breznau et al. 2022 Many analysts, PNAS — link PMC9926231 | CORRECT-METADATA | III/A | **The link points to a different paper**: PMC9926231 is Engzell (2023) "A universe of uncertainty hiding in plain sight", a PNAS commentary. Fix: "Breznau, Rinke, Wuttke et al. (2022) Observing many researchers using the same data and hypothesis reveals a hidden universe of uncertainty, PNAS 119(44), doi 10.1073/pnas.2203150119 (PMID 36306328)" |
| RF populist-vote predictors (EVS), Q&Q 2025 *(snippet)* | PASS | IV/B | Abstract: the top predictors differ between Poland and France and are reported at national level. Fix: "Benati, Bon, Nardi (2025), Quality & Quantity 59:1393–1426"; drop "*(snippet)*" |
| Europeanisation of policy preferences (LOCO RF), JEPP 2025 *(snippet)* | PASS | IV/B | Abstract: "'leave-one-country-out' random forest models". Fix: "Sorace (2025), JEPP"; drop "*(snippet)*" |
| Pearl & Bareinboim 2014, Stat Sci | PASS | VII/A (foundational) | doi 10.1214/14-STS486 |
| Cross-country tabular ML on DHS (2026), 2605.26589 *(snippet)* | PASS | IV/C | Childhood anemia, 16 DHS countries, LOCO. Fix: "Brima et al. (2026)"; drop "*(snippet)*" |
| Fisher, Rudin, Dominici 2019 JMLR · Donnelly et al. 2023 NeurIPS · Hooker et al. 2021 · Strobl et al. 2008 BMC Bioinf · Molnar et al. 2022 · Ewald et al. 2024 · Au et al. 2022 DMKD · Williamson et al. 2023 JASA · Lei et al. 2018 JASA · Watson & Wright 2021 · Kaufman et al. 2011/12 | PASS (11 rows) | n/a (method)/A–B | All IDs resolve to the stated titles and authors. Strobl 2008 was confirmed through an S2 title match (the DOI lookup was not indexed). Lei et al. 1604.04173 = "Distribution-Free Predictive Inference for Regression" (the LOCO source), which is correct |

**D. Elicitation and reliability**

| Citation | Verdict | Grade | Note / fix |
|---|---|---|---|
| Capstick et al. 2025 AutoElicit, ICML | PASS | III/B | arXiv 2411.17284 |
| Selby et al. 2402.07770 | PASS | III/B | *Stat* 14:e70054 (2025) |
| "LLM knowledge for OOD tabular prediction (2026)", 2610.36806 — "Priors help most under shift + small n" | **CLAIM-MISMATCH** | III/C | The abstract says close to the opposite for small n: "On three datasets the prior improves in-domain accuracy as much as out-of-domain, making it better regularisation than shift correction. **Only where the training data are large enough that the in-domain gain disappears does elicited knowledge demonstrably improve transfer.**" Fix: "Langseth, Hellton, Hammer (2026): LLM coefficient priors cut OOD error on 4/5 datasets (up to 7.9% MSE), but mostly act as regularisation; a shift-specific benefit appears only with large training sets, and variation across priors exceeds the mean gain" |
| "Variable names provide priors (2026)", 2611.02637 *(could not be re-found)* | CORRECT-METADATA | III/C | **It exists** (arXiv 2611.02637v1, 2 Nov 2026). Fix: "Ma, Chen, Jiang, Feng, Beer (2026) Variable names provide priors: semantic grounding enables data-efficient numerical prediction by LLMs. Five LLMs match or beat seven supervised models across 20 datasets, especially when examples are scarce". Remove "could not be re-found" |
| Kıcıman et al. TMLR · Long et al. 2023 · Jiralerspong et al. 2024 · Liu, Yang, Adomavicius 2508.19563 · Sclar et al. ICLR 2024 | PASS (5 rows) | III/B–C | 2508.19563: "changing variable names can sway the size of prediction error by as much as 82%" |
| Atıl et al. 2025, Eval4NLP | CORRECT-METADATA | III/B | The published title is "Non-Determinism of 'Deterministic' LLM **System Settings in Hosted Environments**", Eval4NLP 2025, pp. 135–148 (arXiv 2408.04667). Fix the title |

---

## Other observations (not counted as verdicts)

- **§9 verification caveat is out of date.** It says arxiv.org, tandfonline, springer and Semantic Scholar
  were blocked. All of them were reachable today, and every *(snippet)* entry on arXiv has now been
  confirmed. Suggested replacement: "Re-verified 2026-11-08 (ARS fact-check): all arXiv IDs resolve;
  nature.com/pnas.org full texts still unread."
- **No FAIL and no fabricated reference was found.** The one vibe-citing pattern is #6, where two real papers
  are blended under one numeric summary. #9/#15 is a duplicate citation of a single work, not a fabrication.
- **Predatory venues / COI:** none of the venues raise a predatory-publisher flag. No undisclosed conflicts of
  interest were detected at abstract level. Powdthavee co-authors both #3 and #10 (and #21), which is an
  intellectual-COI note only, worth knowing because three "independent" precedents share an author.
- **Limitations:** Bisbee, Hewitt (preprint), Cui, Atari and Atıl were checked through search snippets only.
  Nature, NHB and PNAS full texts were not read. The 11 classic methods papers were verified for existence and
  metadata, not for content.

## Verdict counts (87 citation rows)

| Verdict | Count |
|---|---|
| PASS | 70 |
| CORRECT-METADATA | 10 |
| CLAIM-MISMATCH | 4 |
| UNVERIFIABLE-HERE | 3 |
| FAIL | 0 |

CORRECT-METADATA: Pataranutaporn PNAS title · Hewitt 2024 = Ashokkumar *Nature* 2026 · Collabra (Ji et al.) ·
LLM4FS withdrawn · FREEFORM authors · Huntington-Klein & Murray · Abeliuk title · Breznau wrong PMC link ·
2611.02637 exists · Atıl title.
CLAIM-MISMATCH: Ma & Powell / Hommel & Arslan blend · "Powdthavee moves with one country" · Bisbee "wrong" ·
2610.36806 "small n".
UNVERIFIABLE-HERE: NHB Chen et al. · Ashokkumar *Nature* heterogeneity claim · Cui *Nat Comput Sci*.
