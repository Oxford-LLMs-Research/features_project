# Experiment index

Map from named experiments to design notes, results memos, and on-disk roots.
Path contracts live in `src/survey_features/layout.py` (dual-resolves legacy locations).

| Experiment | Status | Design | Results | Outputs root |
|------------|--------|--------|---------|--------------|
| Prelim JSON grid | Complete | — | [prelim_findings.md](prelim_findings.md) | `outputs/grid/` (legacy: top-level `grid_summary__*`) + `cache/cells/*/llm__*` |
| Main free-text pipeline | Complete (baseline) | [format_findings.md](format_findings.md), [main_experiment_design.md](main_experiment_design.md) | [format_findings.md](format_findings.md) | `outputs/main/` (legacy: `format_pilot/`) |
| Leakage audit (B1) | Complete | — | [leakage_findings.md](leakage_findings.md) | `outputs/cache/audits/leakage_audit.csv` |
| Alignment / adaptation (B2) | Complete | — | [alignment_findings.md](alignment_findings.md) | `outputs/analysis/alignment_*.csv/json` |
| Uncertainty (B3) | Complete | — | [uncertainty_findings.md](uncertainty_findings.md) | `outputs/analysis/uncertainty_summary.json` |
| Embedding sensitivity | Complete | [embedding_sensitivity.md](embedding_sensitivity.md) | same | `outputs/experiments/embedding_sensitivity/` |
| Ensemble mapping | Design + runner (kimi v1) | [ensemble_mapping.md](ensemble_mapping.md) | — (run pending) | `outputs/experiments/ensemble_mapping/` |
| Sub-item mapping | Complete (kimi v1) | [subitem_mapping.md](subitem_mapping.md) | [subitem_mapping_results.md](subitem_mapping_results.md) | `outputs/experiments/subitem_mapping/` |
| Similarity threshold | Design only | [similarity_threshold.md](similarity_threshold.md) | — | `outputs/experiments/similarity_threshold/` (not run) |
| Confirmatory main zoo | Design only | [main_experiment_design.md](main_experiment_design.md) | — | not run |

## Docs vs paper

| Location | Role | In git? |
|----------|------|---------|
| **`docs/`** | Design, protocol, findings memos; link exact `outputs/…` paths and reproduce commands | Yes |
| **`paper/`** | Submission-shaped LaTeX, figures, generated tables | No (gitignored) — clone alone does not rebuild the PDF |
| **This index** | One place to find experiment → disk → memo | Yes |

Conceptual framing: [framing_and_comparisons.md](framing_and_comparisons.md).

## Overwrite rules

1. **`outputs/cache/`** — shared; skip-if-exists unless `--force`.
2. **`outputs/main/`** — team-canonical baseline. Use `--run-tag` so map/score land under `main/runs/<tag>/` instead of clobbering canonical scores/maps. Gen/extract stay shared under `main/<selector>/`.
3. **`outputs/experiments/<name>/`** — exploratory. Prefer `--run-tag` → `…/runs/<tag>/`.
4. Logs go under `outputs/logs/`. **`outputs/.tmp/`** is AutoGluon scratch — safe to delete anytime.

## Adding a new experiment

1. Write `docs/<name>.md` (design) before the first write; add results in the same file or `docs/<name>_results.md`.
2. Register a row in this index.
3. Add a helper in `layout.py` (named dir under `experiments/`) and wire the runner to use it.
4. Do not put findings-only prose in `paper/` until numbers are locked in `docs/`.
