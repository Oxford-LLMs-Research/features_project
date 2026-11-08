# PRISMA-ScR flow counts — Q1/Q4 sub-search (2026-11-08)

Scope: concept blocks **Q1** (LLM-elicited predictor/feature selection or priors) and **Q4**
(data-derived ground truth used to grade LLM claims, and its reliability). Single AI screener
(protocol §4 deviation). Files: `screening.csv` (every record), `chart.csv` (included), `search_log.md`.

## Identified (records retrieved, before de-duplication)

| Source | Search | Hits | Retrieved |
|---|---|---|---|
| arXiv API | PROTOCOL LLM ∧ Q1 ∧ Data | 445 | 445 |
| arXiv API | PROTOCOL LLM ∧ Q4 ∧ (Q1 ∨ Q2) | 1,071 | 500 (top by relevance) |
| arXiv API | supplementary: LLM ∧ feature/variable/predictor selection | 123 | 123 |
| arXiv API | supplementary: LLM ∧ prior elicitation | 2,315 | 400 (top by relevance) |
| arXiv API | supplementary: LLM ∧ importance ∧ agreement | 49 | 49 |
| arXiv API | supplementary: LLM ∧ variable/feature names | 136 | 136 |
| arXiv API | supplementary: LLM ∧ confounder/covariate selection | 180 | 180 |
| arXiv API | supplementary: LLM ∧ risk factors ∧ importance ∧ cohort/survey | 5 | 5 |
| Semantic Scholar | PROTOCOL forward citations of LLM-Select (2407.02694), all pages | 60 | 60 |
| Semantic Scholar | PROTOCOL forward citations of LMPriors (2210.12530), all pages | 77 | 77 |
| Semantic Scholar | keyword search (8 queries + 1 bulk) | — | **0 (HTTP 429 throughout)** |
| arXiv id_list | records surfaced by WebSearch / inventory §9 not hit above | 14 | 14 |
| WebSearch (20 queries) | records with no arXiv/S2 record | 7 | 7 |
| **Total identified** | | | **1,996** |

## Flow

| Stage | n |
|---|---|
| Identified | 1,996 |
| Duplicates removed automatically (id / normalised title) | 119 |
| Records screened (title/abstract) | 1,877 |
| — screened by automatic keyword rule only (no LLM × selection/prior/importance × human-data signal; titles spot-checked by keyword grep) | 1,057 |
| — screened by title by the agent (keyword-triage pool plus records added from WebSearch/inventory), abstract read for ~95 | 820 |
| Full text sought | 22 |
| Full text retrieved | 18 (17 arXiv PDFs + 1 medRxiv HTML) |
| Full text read (assessed) | 17 (one retrieved PDF, 2510.25701, charted from abstract only) |
| Full text not retrievable (openreview 403, Springer bot page, IEEE 403, JMIR 403; nature/pnas/osf blocked by policy) | 4 sought + blocked hosts |
| **Included (charted)** | **46** (15 full text, 31 abstract-level) |
| Excluded | 1,831 |

## Exclusions by reason code

| Code | Meaning | n |
|---|---|---|
| E1 | population: not an LLM (e.g. BERT STS encoders, non-LLM importance-methods papers kept as method cites) or no human-data analogue (IoT, malware, GPQA) | 9 |
| E2 | concept: no LLM naming/scoring/ranking of predictors, no priors over structure, no grading against data-derived importance | 1,790 |
| — of which automatic rule | | 1,057 |
| — of which agent title screen (literature "surveys", LLM-as-predictor, XAI of LLMs, feature engineering, persona simulation) | | 689 |
| — of which agent abstract/full-text decision with note | | 44 |
| E2s | eligible for the review but Q2/Q3-only: routed to the Q2/Q3 screener, not ineligible | 6 |
| E3 | type (symposium abstract) | 1 |
| E4 | date: published before 2022 | 20 |
| E5 | duplicate found at screening (DOI/arXiv twin, medRxiv vs journal, OpenReview vs arXiv) | 5 |
| E6 | unverifiable | 0 |

## Included by relevance rating

| pre-empts | partial | adjacent | method | background |
|---|---|---|---|---|
| 0 | 10 | 28 | 5 | 3 |

## Limitations specific to this run

1. Semantic Scholar keyword search failed (rate limit), so non-arXiv venues (IEEE, Springer, Elsevier, JMIR, medRxiv) were reached only through the two citation sweeps and WebSearch. Expect missed clinical/applied papers.
2. Two of the protocol searches return more hits than were retrieved (Q4 search 1,071 → top 500; prior elicitation 2,315 → top 400, both relevance-sorted).
3. The automatic E2 rule cannot tell "off-concept" from "on-concept but outside Q1/Q4" (silicon-sampling Q2/Q3 papers in the pool are marked E2, not E2s). The Q2/Q3 screeners must not rely on this file for their blocks.
4. 31 of 46 included records are charted from abstracts only; their numbers come from abstracts or search metadata, not tables.
5. Single AI screener; the PI should spot-check a sample of the 689 agent-title E2s.
