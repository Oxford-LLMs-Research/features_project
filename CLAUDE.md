# Agent instructions — features_project

Read before your first change: [`CONTRIBUTING.md`](CONTRIBUTING.md) (where to write,
when to register), [`docs/onboarding.md`](docs/onboarding.md) (caches, oracle
contract), [`docs/glossary.md`](docs/glossary.md) (project terms),
[`docs/project_layout.md`](docs/project_layout.md) (repo + outputs map).

## No naked jargon (required)

When you use a project shorthand in text addressed to a human — "confirmatory run",
"Arm C", "selector", "oracle contract/era", "genuine/unestimable cell", "Phase A/B/C",
"VoR/CoI", or any name you coin — accompany its **first use** in that message or
document with a one-to-two-sentence plain reminder covering: what it means, why it
exists, how it fits the project, and the immediate implication. Definitions live in
[`docs/glossary.md`](docs/glossary.md); extend that file (same four-part format)
whenever you introduce a new name. Never assume the reader holds the insider context.

## Operational basics

- **Interpreter:** `C:\Users\murrn\miniconda3\python.exe` (PowerShell) /
  `/c/Users/murrn/miniconda3/python` (Git Bash). Never `.venv` — it lacks the
  scientific stack.
- **Outputs live on Dropbox**, not in the repo: all artifact paths resolve through
  `src/survey_features/layout.py` helpers under `SURVEY_FEATURES_OUTPUTS` (gitignored
  `.env`). **Never hard-code an absolute or machine-local path** in a tracked file;
  never write artifacts into the repo.
- **Oracle contract v4 is current; v3 is a deprecated dev byproduct.** Do not build
  on v3-era artifacts (old `grid/` summaries, accuracy-era `scores_*.csv`, v3
  `oracle_meta.json`) — check `contract_version` before trusting a cached number.
- **No untyped scoring anywhere:** ordinal/continuous targets are scored by Spearman
  (regression), never as multiclass accuracy. Pass `target_type` to
  `evaluate_feature_set`; get it from the cell's `oracle_meta.json`.
- Registered experiments and citable runs: rules in `CONTRIBUTING.md`. Human-facing
  reports: plain language, decision-framed.
- End of workday: run `/repo-audit` (Claude) — proposes cleanup + commit plan, never
  pushes.
