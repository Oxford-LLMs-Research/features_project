---
name: repo-audit
description: End-of-workday repo audit — inventory the day's delta, safe-clean mechanical clutter, check written policy, group the day's work into commits (never push), and hand the user a ranked walk-through list for their manual pass.
---

# repo-audit — end-of-workday audit & cleanup

You are auditing this repository at the end of a workday of agent-assisted development.
The failure mode you exist to prevent: layered edits, new files, logs and caches pile up
faster than a human can track, and the sediment then distorts what both the user and
future agents attend to.

**Division of labour:** you enforce the repo's *written* policy — `CONTRIBUTING.md`
rules 1–10 and `docs/onboarding.md` §3–4 (contracts, cache identity) — you do not invent
taste. Mechanical cleanup you do yourself; anything requiring judgment you PROPOSE in
the report for the user's walk-through. The user reviews after you finish.

**Interpreter note:** use the conda python (`/c/Users/murrn/miniconda3/python`) for
pytest and analysis; `.venv` lacks the scientific stack.

## Phase 0 — Preconditions

- If a background sweep is running (`tasklist //FI "IMAGENAME eq python.exe"` shows
  workers, or recent-modified logs in `outputs/logs/`), STOP and tell the user — never
  audit a repo mid-run.
- Note the current branch. If on `main`, create a branch before any commit.

## Phase 1 — Delta inventory

- `git status --short`, `git diff --stat`, and `git log --oneline -3` for message style.
- Classify EVERY untracked file: {track in git | add to .gitignore | scratch → delete |
  move to archive/}. When unsure, classify as "propose".
- Summarize the day's work as 3–7 logical change groups (these become the commits).

## Phase 2 — Safe auto-clean (act without asking; log every action in the report)

- All `__pycache__/` dirs and stray `.pyc` (including repo root), `.pytest_cache/`.
- `outputs/.tmp/*` (AutoGluon scratch — CONTRIBUTING r8 says disposable).
- `outputs/logs/`: files older than 7 days; 0-byte or obvious failed-start stubs of any
  age; keep everything referenced by a doc.
- Prune `outputs/audits/` to the newest 14 reports.

## Phase 3 — Policy checks (report violations; fix only with approval)

Run on **today's diff**, not the whole repo (`git diff` + untracked files):

1. **Registration (r1, r3):** new experiment artifacts without a `docs/<name>.md`, an
   `experiments_index.md` row, and a `layout.py` helper. New caches without an identity
   field (onboarding §4).
2. **Fail-loud (r9):** new `try/except` in `src/` that swallows errors around our own
   files/formats. Boundary guards (network, process pools, per-cell sweep isolation)
   are allowed.
3. **Comment discipline (r10):** new comment blocks >3 lines without a `docs/` pointer.
4. **Reuse & duplication review** — three layers, because a rename defeats any grep:
   - *(a) Name tripwire (cheap, whole repo):*
     `grep -rn "^def \|^    def " --include=*.py src/ scripts/ analysis/` → list any
     def name defined in >1 non-test file; for each known duplicate family, diff the
     copies for DIVERGENCE (the killer case: `survey_assets` existed in 4 files and
     only one was thread-safe).
   - *(b) Semantic review of TODAY'S delta (the real check):* for every new or
     substantially-changed function in the diff, search the package by the function's
     **primitives, not its name** — the library calls it makes, distinctive constants
     (thresholds, seeds, n_boot), signature shape, docstring concepts — then READ the
     top candidates and classify: {pure orchestration | duplicate of X → propose
     consolidation | belongs in its canonical home → propose move | genuinely new}.
   - *(c) Housing rule:* every non-orchestration behaviour has ONE owning module —
     the canonical-homes table in CONTRIBUTING → "Where code goes". A new def that
     neither calls its home nor is orchestration is flagged **regardless of naming**.
   - Known-frozen exceptions: era-1 analysis scripts carrying SUPERSEDED banners
     (`alignment_analysis.py`, `uncertainty_analysis.py`) keep their local wrappers —
     do not propose consolidating dead code.
   - If the day's diff exceeds ~10 new functions, fan the semantic review out to a
     subagent per directory rather than skimming.
5. **Contract hygiene:** if `src/survey_features/oracle.py` changed, check whether the
   change alters output MEANING; verify `ORACLE_CONTRACT_VERSION` was bumped or
   correctly not bumped, and that the `docs/onboarding.md` §3 table matches.
6. **Health gates (must pass before commits):**
   - `py_compile` every touched `.py`
   - `pytest tests/ -q`
   - `PYTHONPATH=src python scripts/rerun_oracles.py --dry-run` → expect 0 cells to
     recompute unless a contract migration is knowingly in flight.

## Phase 4 — Staleness & duplication sweep (propose only)

- Docs whose claims today's changes contradict → propose a SUPERSEDED banner or refresh
  (pattern: see existing banners in `docs/alignment_findings.md`).
- Overlapping-purpose files: for each suspect, count references
  (`grep -rn <name> --include=*.py --include=*.md`) and propose keep/merge/archive with
  the counts as evidence.
- Data files inside code dirs; one-shot scripts that have already served their purpose
  → propose `archive/`.
- Large stray files at root.

## Phase 5 — Report

Write `outputs/audits/repo_audit_<YYYY-MM-DD>.md`:

1. Day's delta summary (the change groups).
2. Actions TAKEN (each: what + why, one line).
3. Actions PROPOSED (each: rationale + the exact command to execute it).
4. Policy violations found.
5. Health-gate results.
6. **Ranked walk-through list for the user — max 10 items**, highest-stakes first.

Print to the terminal: a compact summary table + the walk-through list + the report path.

## Phase 6 — Grouped commits

- Stage and commit the day's work as the Phase-1 groups. Real messages in the style of
  recent `git log`; end each with the Co-Authored-By line per harness rules.
- **Never push. Never commit** `outputs/`, `.env`, scratch, or anything classified
  "propose". The user's review is `git log -p` per commit; wrong groupings are one
  revert away.

## Phase 7 — Self-amendment

If today produced a NEW recurring failure mode this checklist misses, propose (never
apply) a one-line addition to this file, in the report's proposed-actions section.

## Guardrails

- NEVER delete anything under `outputs/cache/cells*` — the era archives are the
  provenance of published numbers.
- NEVER touch `.env` or credentials.
- `data.zip` / `outputs.zip` / cache eras / anything referenced by a doc: propose-only.
- Ambiguity resolves to "propose", never to action.
