# Scoping review protocol — LLM knowledge of predictive structure (DRAFT v0, 2026-11-08; run once)

**Status: draft for the PI to edit and approve.** The first search was run under
this v0 so the map exists; if the protocol changes, re-run the affected searches and
bump the version. A scoping review maps *what has been studied, with which concepts,
and where the gaps are*, using a documented search; it does not grade effect sizes.
ARS (the research-skills plugin, see glossary) has no scoping mode, so this protocol
and the PRISMA-ScR checklist are ours. The narrative synthesis built on the map runs
in ARS `lit-review` mode and lands in `project_inventory_2026-11.md` §9.

## 1. Review question

What has been studied about **whether large language models know which variables
predict human attitudes, behaviours or outcomes** — from variable names, question
wording or task descriptions — and about **how that knowledge varies across
countries, cultures or subpopulations**, and how have such claims been evaluated
against data?

Sub-questions (each maps to a concept block in §3):
- **Q1** LLM-elicited predictor / feature selection or priors (stated knowledge).
- **Q2** LLM knowledge of associational or causal structure among human-data variables
  (stated, or implied through simulated respondents).
- **Q3** Cross-national / cultural / subgroup heterogeneity of that knowledge,
  including whether it tracks real heterogeneity.
- **Q4** Evaluation ground truth: data-derived importance, coefficients or effects
  used to grade LLM claims, and their reliability.

## 2. Eligibility (PCC: population, concept, context)

| | Include | Exclude |
|---|---|---|
| Population | LLMs (≥ 2022 instruction-tuned or later); human-subject data (surveys, panels, experiments, registries) or tabular social/health data | non-LLM NLP only; synthetic/toy tables with no human-data analogue |
| Concept | the model names, scores, ranks or weights predictors; or its implied structure (coefficients, correlations, gradients) is compared with data; or heterogeneity of either across groups | pure response-distribution matching with no structural comparison; pure fairness/bias auditing of outputs |
| Context | any discipline; any country | — |
| Types | peer-reviewed papers, preprints (arXiv, medRxiv, SSRN, PsyArXiv/SocArXiv), conference & workshop papers | blogs, news, product docs (may be used to locate a paper, never charted) |
| Dates | 2022-01-01 → 2026-11-08 (+ pre-2022 methods classics for Q4 only, cited not charted) | — |
| Language | English | — (noted as a limitation) |

## 3. Sources and search strings (v0)

Sources: arXiv API (export.arxiv.org), Semantic Scholar Graph API (search +
forward citations), OpenReview, ACL Anthology, PMLR, medRxiv, publisher sites
reachable from the session; WebSearch for Google-Scholar-style recall. Blocked in
this environment (record, don't skip silently): nature.com full text, pnas.org,
osf.io (PsyArXiv/SocArXiv), researchsquare.com, crossref, openalex.

Concept blocks (combine within block with OR, across blocks with AND as listed):
- **LLM:** "large language model" OR LLM OR GPT OR "language model"
- **Q1:** "feature selection" OR "variable selection" OR "predictor selection" OR prior OR "feature importance" OR "covariate"
- **Q2:** correlation OR association OR "causal" OR coefficient OR "predictive structure" OR "determinants" OR moderat* OR interaction
- **Q3:** cross-national OR cross-cultural OR country OR culture OR subgroup OR heterogeneity OR "distribution shift" OR transfer
- **Q4:** "permutation importance" OR SHAP OR "variable importance" OR "ground truth" OR "regression coefficients"
- **Data:** survey OR "World Values Survey" OR "European Social Survey" OR barometer OR panel OR "social science" OR tabular

Searches: LLM ∧ Q1 ∧ Data; LLM ∧ Q2 ∧ Data; LLM ∧ Q3 ∧ (Q1 ∨ Q2); LLM ∧ Q4 ∧ (Q1 ∨ Q2).
Forward-citation sweeps of seed papers: LLM-Select (arXiv 2407.02694), LMPriors
(2210.12530), Argyle et al. 2023 (2209.06899), Pataranutaporn et al. (2507.06141),
Ma & Powell (2503.21011), Hewitt et al. 2024, Bisbee et al. 2024, ACE-Align
(2601.12962), Powdthavee (2609.33055).

## 4. Selection

One screener (an AI agent) on title/abstract against §2, then full text where
reachable. Every exclusion gets a reason code (E1 population, E2 concept, E3 type,
E4 date, E5 duplicate, E6 unverifiable — cannot confirm the record exists). This is
**single-screener with AI**: a known deviation from PRISMA-ScR's dual-screening
recommendation, reported as a limitation; the PI spot-checks a sample of exclusions.

## 5. Charting form (one row per included record)

`id, citation, url, year, venue/type, access (full|abstract), Q-blocks, elicitation
(stated | implied-by-simulation | fine-tuned | none), ground truth (importance |
coefficients | correlations | effects | none), data source, n countries/groups,
heterogeneity tested? (none | levels | structure | structure-vs-measured-heterogeneity),
transfer test? (y/n), models, headline finding (≤ 25 words, quoted numbers only from
text read), relevance to us (pre-empts | partial | adjacent | method | background)`.

## 6. Reporting

PRISMA-ScR flow counts (identified / screened / full-text / included / excluded with
reasons), the charting table (CSV in the scratch run, summary table in §9), a concept
map (which cells of Q1–Q4 × heterogeneity level are empty), and the narrative
synthesis. Every charted citation passes the ARS fact-check rule: if existence cannot
be confirmed, it is excluded (E6), not marked "uncertain".

## 7. As-run record for v0 (2026-11-08) — deviations to review

Results: [`scoping_review_2026-11/`](scoping_review_2026-11/) (merged `chart.csv`,
per-screener PRISMA counts, search logs, gzipped screening decisions); synthesis in
`project_inventory_2026-11.md` §9.7. Deviations from §3–§5, each a PI decision to accept
or fix in v1:

1. **Screeners added two reason codes**: `E2s` (eligible but outside that screener's
   blocks, handed to the other) and `E6*` (record exists in Semantic Scholar but no
   abstract or text was reachable; 9 records to retrieve).
2. **Automated keyword pre-screens** removed 1,057 (Q1+Q4) and 661 (Q2+Q3) records
   before title screening; the rules are in the search logs. Spot-check them.
3. **Relevance windows**: for the two broadest arXiv queries only the top 400–500 hits by
   relevance were screened (up to ~6,600 further hits identified, not screened).
4. **Semantic Scholar** keyword search returned HTTP 429 throughout; citation sweeps
   completed for LLM-Select, LMPriors, Argyle, Pataranutaporn (arXiv + PNAS), ACE-Align;
   **incomplete** for Ma & Powell, Powdthavee, Bisbee, Ashokkumar/Hewitt.
5. **Venue site searches not run**: OpenReview, ACL Anthology, PMLR, medRxiv, Springer,
   T&F, SAGE (reached only via citations and web search).
6. **Charting**: 78 of 108 included records charted from abstracts; the
   "heterogeneity tested" field was given four strict levels (glossary: *heterogeneity
   level*).
7. **Blocked hosts** (full texts not read): nature.com, pnas.org, pmc.ncbi.nlm.nih.gov,
   online.ucpress.edu, cambridge.org, researchsquare.com, osf.io.
