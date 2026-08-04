"""
survey_features — shared library for the LLM feature-selection capability study.

Module map (pipeline order):
    config      paths, .env loading, OUTPUTS_DIR / PAPER_DIR, model registry
    surveys     survey loading, country maps, metadata handling (single copy)
    llm         OpenAI-compatible client with retries + token-usage logging
    timing      wall-clock spans + JSONL timing logs for pipeline phases
    prompts     ALL prompt templates (current free-text + legacy JSON + extract/disambig)
    elicitation selection-prompt calls (free-text current; JSON legacy)
    extraction  free-text response -> typed feature list (fixed extractor model)
    retrieval   sentence-transformer embeddings + dual-embed / ensemble candidate retrieval
    ensemble    ensemble fusion labels + defaults (union max-sim)
    disambig    feature -> survey-code disambiguation (per-feature current; shortlist legacy)
    oracle      AutoGluon permutation-importance oracle (requires the [oracle] extra)
    evaluation  matched-k XGBoost CV: oracle vs model vs random
    metrics     captured importance, jaccard, oracle percentile, bootstrap CIs
    layout      outputs/ path contracts (cache/main/experiments; dual-resolve)

Live entry points: scripts/run_main.py (free-text pipeline), scripts/leakage_audit.py,
and the other scripts/ runners. JSON-grid appendix replication: archive/run_grid.py
(never import archive/ into this package). Paper builders live under local paper/scripts/
(gitignored; use PAPER_DIR).
"""

__version__ = "0.1.0"
