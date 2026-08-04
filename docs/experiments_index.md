# Experiment index

Map from named experiments to design notes, results memos, and on-disk roots.
Path contracts live in `src/survey_features/layout.py` (dual-resolves legacy locations).

| Experiment | Status | Design | Results | Outputs root |
|------------|--------|--------|---------|--------------|
| Pipeline audit (2026-08) | Complete; Tier-0 fixes applied, oracle re-run in flight | — | [pipeline_audit_2026-08.md](pipeline_audit_2026-08.md) | `outputs/cache/cells/*/oracle_meta.json`, `cache/audits/missing_code_taxonomy.csv` |
| Prelim JSON grid | Superseded (see note below) | — | [prelim_findings.md](prelim_findings.md) | `outputs/grid/` (legacy: top-level `grid_summary__*`) + `cache/cells/*/llm__*` |
| Main free-text pipeline | Complete (baseline) | [format_findings.md](format_findings.md), [main_experiment_design.md](main_experiment_design.md) | [format_findings.md](format_findings.md) | `outputs/main/` (legacy: `format_pilot/`) |
| Leakage audit (B1) | Complete | — | [leakage_findings.md](leakage_findings.md) | `outputs/cache/audits/leakage_audit.csv` |
| Alignment / adaptation (B2) | Complete | — | [alignment_findings.md](alignment_findings.md) | `outputs/analysis/alignment_*.csv/json` |
| Uncertainty (B3) | Complete | — | [uncertainty_findings.md](uncertainty_findings.md) | `outputs/analysis/uncertainty_summary.json` |
| Embedding sensitivity | Complete | [embedding_sensitivity.md](embedding_sensitivity.md) | same | `outputs/experiments/embedding_sensitivity/` |
| Ensemble mapping | Complete (kimi v1); **not promoted to default** | [ensemble_mapping.md](ensemble_mapping.md) | same | `outputs/experiments/ensemble_mapping/` |
| Sub-item mapping | Complete (kimi v1); **dual-layer locked for confirmatory main** ([main_experiment_design.md](main_experiment_design.md) §4) | [subitem_mapping.md](subitem_mapping.md) | [subitem_mapping_results.md](subitem_mapping_results.md) | `outputs/experiments/subitem_mapping/` (pilot); production path → `run_main` |
| Similarity threshold | Design only | [similarity_threshold.md](similarity_threshold.md) | — | `outputs/experiments/similarity_threshold/` (not run) |
| Confirmatory main zoo | Design only | [main_experiment_design.md](main_experiment_design.md) | — | not run |

## Which numbers are current

"Complete" marks that a run finished, not that its numbers are the ones to quote. The
memos were written in sequence and several report the same quantity from different runs.
Read this before citing any figure.

**Three eras, in order.** (1) *JSON grid* — strict-JSON elicitation, top-5 shortlist
mapper. (2) *Free text* — the current instrument; `docs/format_findings.md` established
that the JSON contract suppressed measured capability, so JSON-era magnitudes are a
conservative floor, not the estimate. (3) *Post-audit* — from
[pipeline_audit_2026-08.md](pipeline_audit_2026-08.md): log-loss oracle on an honest
fit/select/score split, respondent-vs-structural missing-code split, oracle-matched random
null, textbook baseline. **Era 3 invalidates every cached number from eras 1 and 2**; the
accuracy-era oracles are archived at `outputs/cache/cells_accuracy_v1/`.

**Test 2 (cross-national adaptation) — the one to be careful with.** The corpus contains
two opposite-looking results and neither memo says so:

- `alignment_findings.md` (B2) and `uncertainty_findings.md` (B3): adaptation ≈ **−0.002,
  every interval straddles zero** — "movement without fit". This is the **JSON grid**.
- Free-text results: **Kimi +0.023 [0.001, 0.045]**, DeepSeek still null. This is **arm C**,
  a different instrument, different n, different clusters.

Both are correct about their own run. The JSON-era null is **not** the current statement
of Test 2 and should not be quoted as one. Since T2 is the designated signature result,
and it is measured against oracle rankings the audit shows were largely noise-selected,
treat *both* as pending re-computation on the era-3 oracle.

**Also superseded:** `prelim_findings.md` describes a 5×5 single-model run that
`RECONCILIATION_PLAN.md` records as unrecoverable — its numbers are not reproducible from
disk. `framing_and_comparisons.md` argues "the LLM is weak as an importance estimator"
from JSON-era figures (VoR 0.025, ~half beat random) without flagging them as such; free
text gives 0.054–0.063 and 76–78% at model-k. The conceptual argument may still hold, but
check which era a number comes from before repeating it.

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
