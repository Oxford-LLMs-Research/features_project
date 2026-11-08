# Search log — Q1/Q4 scoping sub-search (protocol v0 draft, 2026-11-08)

Screener: one AI agent (single-screener deviation, protocol §4). All dates 2026-11-08. Raw responses: `raw/*.jsonl`; machine log: `log.jsonl`.

## A. Programmatic searches (arXiv API, Semantic Scholar Graph API)

| # | Source | Tag | Query string | Total hits | Retrieved | Role |
|---|---|---|---|---|---|---|
| 1 | Semantic Scholar forward citations | cites_llmselect | `citations of arXiv:2407.02694` | 60 | 60 | PROTOCOL: forward citations of LLM-Select, all pages |
| 2 | Semantic Scholar forward citations | cites_lmpriors | `citations of arXiv:2210.12530` | 77 | 77 | PROTOCOL: forward citations of LMPriors, all pages |
| 3 | arXiv API | ax_A_llm_q1_data | `(abs:"large language model" OR abs:"large language models" OR abs:LLM OR abs:LLMs OR abs:GPT OR abs:"language model") AND (abs:"feature selection" OR abs:"variable selection" OR abs:"predictor selection" OR abs:prior OR abs:priors OR abs:"feature importance" OR abs:covariate OR abs:covariates) AND (abs:survey OR abs:"World Values Survey" OR abs:"European Social Survey" OR abs:barometer OR abs:panel OR abs:"social science" OR abs:tabular)` | 445 | 445 | PROTOCOL: LLM ∧ Q1 ∧ Data (abs: fields) |
| 4 | arXiv API | ax_B_llm_q4_q1q2 | `(abs:"large language model" OR abs:"large language models" OR abs:LLM OR abs:LLMs OR abs:GPT OR abs:"language model") AND (abs:"permutation importance" OR abs:SHAP OR abs:"variable importance" OR abs:"ground truth" OR abs:"regression coefficients") AND ((abs:"feature selection" OR abs:"variable selection" OR abs:"predictor selection" OR abs:prior OR abs:priors OR abs:"feature importance" OR abs:covariate OR abs:covariates) OR (abs:correlation OR abs:association OR abs:causal OR abs:coefficient OR abs:"predictive structure" OR abs:determinants OR abs:moderation OR abs:moderator OR abs:interaction))` | 1071 | 500 | PROTOCOL: LLM ∧ Q4 ∧ (Q1 ∨ Q2); top 500 by relevance of 1071 |
| 5 | arXiv API | ax_C_llm_featsel | `(abs:"large language model" OR abs:"large language models" OR abs:LLM OR abs:LLMs OR abs:GPT OR abs:"language model") AND (abs:"feature selection" OR abs:"variable selection" OR abs:"predictor selection")` | 123 | 123 | supplementary recall |
| 6 | arXiv API | ax_D_llm_priorelic | `(abs:"large language model" OR abs:"large language models" OR abs:LLM OR abs:LLMs OR abs:GPT OR abs:"language model") AND (abs:"prior elicitation" OR abs:"elicit priors" OR abs:"elicited priors" OR abs:"expert prior" OR abs:"informative prior" OR abs:"informative priors" OR abs:"elicitation")` | 2315 | 400 | supplementary recall; 'elicitation' too broad, top 400 of 2315 by relevance |
| 7 | arXiv API | ax_E_llm_imp_agree | `(abs:"large language model" OR abs:"large language models" OR abs:LLM OR abs:LLMs OR abs:GPT OR abs:"language model") AND (abs:"feature importance" OR abs:SHAP OR abs:"variable importance" OR abs:"permutation importance") AND (abs:"domain knowledge" OR abs:agreement OR abs:align OR abs:alignment OR abs:"rank correlation" OR abs:Kendall OR abs:Spearman)` | 49 | 49 | supplementary recall |
| 8 | arXiv API | ax_F_llm_varnames | `(abs:"large language model" OR abs:"large language models" OR abs:LLM OR abs:LLMs OR abs:GPT OR abs:"language model") AND (abs:"variable names" OR abs:"feature names" OR abs:"variable descriptions" OR abs:"feature descriptions" OR abs:"column names")` | 136 | 136 | supplementary recall |
| 9 | arXiv API | ax_G_llm_confounder | `(abs:"large language model" OR abs:"large language models" OR abs:LLM OR abs:LLMs OR abs:GPT OR abs:"language model") AND (abs:confounder OR abs:confounders OR abs:"covariate selection" OR abs:"adjustment set") AND (abs:observational OR abs:epidemiolog* OR abs:clinical OR abs:survey OR abs:"real-world")` | 180 | 180 | supplementary recall (covariate/confounder) |
| 10 | arXiv API | ax_H_llm_risk_factors | `(abs:"large language model" OR abs:"large language models" OR abs:LLM OR abs:LLMs OR abs:GPT OR abs:"language model") AND (abs:"risk factors" OR abs:predictors) AND (abs:"feature importance" OR abs:SHAP OR abs:"logistic regression" OR abs:"variable importance") AND (abs:survey OR abs:cohort OR abs:questionnaire OR abs:EHR OR abs:"electronic health")` | 5 | 5 | supplementary recall |
| 11 | arXiv API id_list (records from inventory §9 and WebSearch) | other_inventory_websearch | `id_list=2609.04013,2604.21120,2611.00256,2407.14614,2502.20268,2510.15096,2604.01896,2510.21884,2608.08126,2601.12343,2601.08703,2512.22240,2602.11760,2605.26589` | 14 | 14 | metadata fetch for records found via WebSearch / inventory §9 not hit above |

Notes: arXiv queries were sent to `https://export.arxiv.org/api/query` (the `http://` host did not resolve; https worked), sorted by relevance, 100 per page. 
Concept blocks exactly as protocol §3, mapped to `abs:` fields; `moderat*` expanded to `moderation OR moderator` (arXiv API has no reliable truncation). 
Q1 term `prior` also matched `priors`; Data term `survey` also matches 'a survey of' (literature reviews) — a known precision loss, handled in screening.

### Semantic Scholar keyword search — FAILED (rate limit)

| Query | Attempts | Result |
|---|---|---|
| `large language model feature selection survey data` | /graph/v1/paper/search, 2 runs × 6–8 retries (30–60 s back-off) | HTTP 429 every attempt; 0 records |
| `LLM predictor selection variable descriptions` | /graph/v1/paper/search, 2 runs × 6–8 retries (30–60 s back-off) | HTTP 429 every attempt; 0 records |
| `LLM feature importance SHAP agreement domain knowledge` | /graph/v1/paper/search, 2 runs × 6–8 retries (30–60 s back-off) | HTTP 429 every attempt; 0 records |
| `large language model prior elicitation` | /graph/v1/paper/search, 2 runs × 6–8 retries (30–60 s back-off) | HTTP 429 every attempt; 0 records |
| `language model priors variable names tabular` | /graph/v1/paper/search, 2 runs × 6–8 retries (30–60 s back-off) | HTTP 429 every attempt; 0 records |
| `LLM selected covariates regression` | /graph/v1/paper/search, 2 runs × 6–8 retries (30–60 s back-off) | HTTP 429 every attempt; 0 records |
| `LLM knowledge variable importance ground truth comparison` | /graph/v1/paper/search, 2 runs × 6–8 retries (30–60 s back-off) | HTTP 429 every attempt; 0 records |
| `LLM feature selection health survey questionnaire` | /graph/v1/paper/search, 2 runs × 6–8 retries (30–60 s back-off) | HTTP 429 every attempt; 0 records |
| `"feature selection" + "large language model" + survey` | /paper/search/bulk, 1 attempt | HTTP 429 |

The citation endpoint (`/paper/arXiv:ID/citations`) worked earlier in the session (60 and 77 citing records, single page each, no `next` token). Keyword search on S2 is therefore **not done**; recall relied on arXiv + WebSearch. This is a limitation to report.

## B. WebSearch (Google-Scholar-style recall), all 2026-11-08

| # | Query | Records newly surfaced (then verified or excluded) |
|---|---|---|
| W1 | LLM feature selection survey variables questionnaire predictors compared with SHAP importance | arXiv 2510.25701 (credit risk, LLM vs SHAP) |
| W2 | large language model identify predictors from variable descriptions survey data medRxiv 2026 | medRxiv vaping (Zhang et al. 2026) confirmed; 2601.12343 |
| W3 | LLM-elicited feature importance agreement with data-driven importance reliability ground truth Rashomon | JMIR Med Inform e81644 (excluded) |
| W4 | medinform.jmir.org 2026 e81644 large language model feature importance | metadata of e81644 |
| W5 | LLM priors Bayesian regression social science survey elicitation coefficients compared with estimated coefficients | OpenEstimate (OpenReview); 2506.21964 |
| W6 | ChatGPT select covariates confounders epidemiology study compared with data-driven selection | 2603.14273 (excluded) |
| W7 | "language model" elicited priors accuracy calibration real data social science health ... openreview | OpenEstimate 2510.15096; 2604.01896 |
| W8 | LLM ranks risk factors compared with logistic regression coefficients cohort data clinical prediction ... | JMIR 2026 e29701 (Kuo et al.); OHDSI abstract (E3) |
| W9 | large language models select predictors social survey data sociology political science "feature selection" LLM 2026 arXiv | ACL Findings 2025.findings-acl.27 (E2s); Warner et al. (E1) |
| W10 | reliability of SHAP permutation importance as ground truth for evaluating LLM feature importance explanations disagreement problem | 2608.08126 (E2) |
| W11 | site:medrxiv.org large language model feature selection predictors variable descriptions | nothing new (vaping, smoking cessation) |
| W12 | GPT-4 selected predictors survey cohort "knowledge-driven" variable selection compared LASSO random forest importance public health 2025 2026 | nothing new |
| W13 | LLM ranks determinants of life satisfaction compared with SHAP survey data cross-country | PNAS title of Pataranutaporn et al. (routed to Q3) |
| W14 | LLM feature importance scores rank correlation with permutation importance student dropout education survey dataset | nothing eligible |
| W15 | large language model propensity score covariate selection observational study evaluation ground truth confounders 2025 | Liu et al. MLHC 2026 = arXiv 2604.16763 (E2) |
| W16 | "feature importance" LLM elicited "unstable" OR "Rashomon" OR "model multiplicity" ground truth evaluation LLM-Select critique | 2601.08703, 2512.22240, 2602.11760 (no LLM, E1 method cites) |
| W17 | site:aclanthology.org LLM feature selection tabular OR survey predictors importance 2025 | EMNLP/NAACL Findings feature-engineering papers (E2) |
| W18 | site:proceedings.mlr.press large language model priors feature importance tabular elicitation | LAAT (2502.20268) |
| W19 | jmir.org 2026 e29701 intracerebral hemorrhage mortality large language model feature prioritization | Kuo et al. metadata |
| W20 | LAAT large language model attribution alignment tabular feature importance smaller models springer 2025 | arXiv 2502.20268 |

## C. Direct page / full-text access attempts

| Host | Result |
|---|---|
| arxiv.org/pdf | 200 — 17 PDFs retrieved; with the medRxiv HTML, 18 full texts retrieved, 17 consulted (15 included `access=full`, 2 excluded after reading) |
| www.medrxiv.org (vaping full text) | 200 |
| openreview.net | 403 (curl) / not resolvable (web-fetch) — OpenEstimate read via arXiv instead |
| api2.openreview.net | proxy connect rejected |
| link.springer.com (LAAT chapter) | 200 but bot 'Client Challenge' page, no content — read arXiv 2502.20268 abstract instead |
| doi.org → IEEE (ICHI 2026 MIDUS paper) | 403 — abstract from Semantic Scholar record only |
| www.jmir.org (Kuo et al. 2026) | 403 — snippet/metadata only |
| api.semanticscholar.org search | 429 throughout (see A) |
| nature.com, pnas.org, osf.io, researchsquare, crossref, openalex | blocked by policy; not attempted |
