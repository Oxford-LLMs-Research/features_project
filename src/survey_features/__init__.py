"""
survey_features — shared library for the LLM feature-selection capability study.

Module map (pipeline order):
    config      paths, .env loading, model registry
    surveys     survey loading, country maps, metadata handling (single copy)
    llm         OpenAI-compatible client with retries + token-usage logging
    timing      wall-clock spans + JSONL timing logs for pipeline phases
    prompts     ALL prompt templates (current free-text + legacy JSON + extract/disambig)
    elicitation selection-prompt calls (free-text current; JSON legacy)
    extraction  free-text response -> typed feature list (fixed extractor model)
    retrieval   sentence-transformer embeddings + dual-embed candidate retrieval
    disambig    feature -> survey-code disambiguation (per-feature current; shortlist legacy)
    oracle      AutoGluon permutation-importance oracle (requires the [oracle] extra)
    evaluation  matched-k XGBoost CV: oracle vs model vs random
    metrics     captured importance, jaccard, oracle percentile, bootstrap CIs
    layout      outputs/ path contracts (grid summaries, LLM caches, format-pilot dirs)

Entry points live in scripts/ (run_main.py = current free-text pipeline;
run_grid.py = legacy JSON grid kept for appendix reproducibility).
"""

__version__ = "0.1.0"
