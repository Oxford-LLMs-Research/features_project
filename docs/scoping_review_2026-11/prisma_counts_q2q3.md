# PRISMA-ScR flow counts — Q2/Q3 screener (protocol v0, run 2026-11-08)

Counts are generated from `screening.csv` by `scripts/prisma.py`; one row per record per source.

## Identification
- arXiv API: 2883 records retrieved across 21 queries (A, B, F01–F19); a further ≤6582 hits lay beyond the retrieval windows (A, B: top 400; F04, F12, F13, F18: top 300) and were **identified but not screened** (overlapping upper bound).
- Semantic Scholar forward citations: 1594 citing records (argyle: 1572; patara_arxiv: 2; patara_pnas: 20; ma_powell: 0 (INCOMPLETE); aceal: 0; powdthavee: 0 (INCOMPLETE); bisbee: 0 (INCOMPLETE); hewitt: 0 (INCOMPLETE); ashokkumar: 0 (INCOMPLETE)).
- WebSearch-only records: 7.
- Total record rows: 4034; duplicates removed (E5): 245; **unique records: 3789**.

## Screening
- Removed before screening: 48 dated before 2022 (E4) and 661 excluded by the documented keyword pre-screen of S2 citers (E2, automated).
- **Title/abstract screened by the agent: 3080**; excluded at title level: 2905 (E1 15, E2 2879, E6 2, E6* 9).
- Assessed on abstract or full text: 175 (of which full text read or skimmed: 16).
- Excluded after abstract/full-text assessment: 109 (E1 4, E2 105).

## Included
- **Included: 66 records → 65 chart rows** (one companion record, arXiv:2605.15473, charted with arXiv:2602.00685).
- Access: 16 full text, 49 abstract only.
- Heterogeneity tested: none 36, levels 15, structure 9, structure-vs-measured-heterogeneity 5.
- Relevance to us: adjacent 39, method 18, partial 6, pre-empts 1, background 1.

## Exclusion codes (protocol §4)
E1 population (not an LLM ≥2022 / not human data) · E2 concept (no structural or heterogeneity comparison with human data; includes pure response-distribution matching and bias audits) · E3 type · E4 date · E5 duplicate · E6 unverifiable (existence not confirmed) · **E6*** (added here, for the PI to confirm): record exists in Semantic Scholar but no abstract or full text was reachable, so relevance could not be judged.

## Known incompleteness
- Forward-citation sweeps **not completed** because Semantic Scholar returned HTTP 429 on every attempt: ma_powell, powdthavee, bisbee, hewitt, ashokkumar. Re-run `scripts/s2_sweep.py raw/s2` (it resumes from `raw/s2/s2_state.json`).
- OpenReview, ACL Anthology, PMLR, medRxiv, Springer, T&F and SAGE site searches were not run.
