"""
survey_features — shared library for the LLM feature-selection capability study.

Module map (pipeline order):
    config      paths, .env, OUTPUTS_DIR, model registry
    surveys     survey loading, country maps, missing taxonomy, target types
    feature_pool  oracle feature-pool construction + skip-pattern screen
    llm         OpenAI-compatible client (retries + token-usage log)
    timing      wall-clock spans + JSONL timing logs
    prompts     free-text / extract / disambig prompt templates
    elicitation free-text selection messages
    extraction  essay -> typed feature list (fixed extractor)
    retrieval   sentence-transformer embeddings + dual-embed retrieval
    mapping     dual-layer feature -> survey-code map (parent + sub_items)
    oracle      AutoGluon permutation-importance oracle ([oracle] extra)
    oracle_pool process isolation for concurrent oracle fits
    evaluation  matched-k XGBoost CV scoring
    metrics     captured importance, jaccard, bootstrap CIs
    score_cell  cell-level scoring + scores schema + baseline caches
    grid_screen confirmatory keep/drop (type-1 + leakage; not accuracy-vs-majority)
    layout      Dropbox outputs/ path contracts (see README § Outputs)

Entry points: scripts/run_main.py, leakage_audit.py, compute_oracle.py,
rerun_oracles.py, build_textbook_baseline.py.
"""

__version__ = "0.1.0"
