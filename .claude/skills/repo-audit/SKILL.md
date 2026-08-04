---
name: repo-audit
description: >-
  End-of-workday repo audit — inventory WIP, check written policy, soft-clean
  mechanical clutter (or propose-only), and hand a ranked walk-through plus a
  commit plan. Use when the user runs /repo-audit, asks for an end-of-day audit,
  or explicitly requests repo cleanup/maintenance. Never auto-invoke from ambient
  "clean up" chatter alone. Never pushes; commits only after the user confirms
  the commit plan.
---

# repo-audit — end-of-workday audit & cleanup

You are auditing this repository at the end of a workday of agent-assisted development.
The failure mode you exist to prevent: layered edits, new files, logs and caches pile up
faster than a human can track, and the sediment then distorts what both the user and
future agents attend to.

**Division of labour:** you enforce the repo's *written* policy — `CONTRIBUTING.md`
rules 1–10 and `docs/onboarding.md` §3–4 (contracts, cache identity) — you do not invent
taste. Default is **propose**; irreversible actions need an explicit apply mode (below).
The user reviews the report before any commits.

**Modes** (read from the user's message; if unspecified, use `audit`):

| mode | cleans | commits |
|---|---|---|
| `audit` (default) | propose only — list exact soft-delete commands in the report | propose commit plan only |
| `apply-clean` | soft-delete mechanical clutter (Phase 2) | still propose-only |
| `apply-all` | soft-delete + commit **only after** user confirms the plan in-chat | yes, after confirm |

Never upgrade the mode yourself. If the user says only `/repo-audit`, stay on `audit`.

**Interpreter:** prefer `C:\Users\murrn\miniconda3\python.exe` (PowerShell) or
`/c/Users/murrn/miniconda3/python` (Git Bash). Do not use `.venv` — it lacks the
scientific stack. Verify with `python -c "import autogluon"` / the conda path before
pytest if unsure.

## Phase 0 — Preconditions (fail closed)

**Busy check — prefer tree mtimes over process names.** STOP and tell the user if any
file anywhere under `outputs/` has mtime within the last **15 minutes** — excluding
`outputs/audits/` and `outputs/.trash/`, which are this skill's own artifacts (a
same-evening second audit must not be blocked by the first one's report). The whole
tree matters: a score sweep's freshest writes land in `outputs/main/`, not in logs.

`tasklist` / `Get-Process python` is **advisory only** (idle orphans false-positive;
missed children false-negative). Never treat "no python.exe" as proof the repo is idle.
If the busy check is inconclusive, STOP — do not clean or commit.

Also note:

- Current branch. If on `main`/`master`, **do not create a branch yet** — propose a
  branch name in the report; create it only under `apply-all` after the user confirms.
- Ask once if unclear: commit **all current WIP**, or only paths touched since a
  ref/timestamp the user names? Default assumption for the *plan*: all current WIP,
  labelled clearly as "full WIP, not necessarily one calendar day."

## Phase 1 — Delta inventory

- `git status --short`, `git diff --stat`, and `git log --oneline -3` for message style.
- Classify EVERY untracked file: {track in git | add to .gitignore | scratch →
  soft-delete | move to archive/ | propose}. When unsure → **propose**.
- Summarize WIP as 3–7 logical change groups (**commit plan**, not commits yet).
- For each group: list the exact paths to stage. Exclude anything classified propose,
  scratch, or under `outputs/`.

## Phase 2 — Mechanical clean

Targets (same set in every mode):

- All `__pycache__/` dirs and stray `.pyc` (including repo root), `.pytest_cache/`.
- `outputs/.tmp/*` (AutoGluon scratch — CONTRIBUTING r8).
- `outputs/logs/`: files older than 7 days; 0-byte or obvious failed-start stubs.
  **Never hard-delete.** Soft-delete only if the basename is **not** mentioned in any
  `docs/**/*.md` or `*.md` at repo root (grep the basename). If grep is ambiguous → propose.
- `outputs/audits/`: reports beyond the newest 14 → soft-delete (keep today's draft).

**Soft-delete procedure** (when mode is `apply-clean` or `apply-all`):

1. Move to `outputs/.trash/repo-audit_<YYYY-MM-DD>/` preserving relative paths.
2. Log every move in the report (src → trash path).
3. Do **not** empty trash in this skill. Propose pruning trash older than 14 days as a
   walk-through item only.

In `audit` mode: write the exact move commands into Actions PROPOSED; take no deletes.

## Phase 3 — Policy checks (report violations; fix only with approval)

Run on the **WIP under review** (`git diff` + untracked files in scope), not the whole repo:

1. **Registration (r1, r3):** new experiment artifacts without a `docs/<name>.md`, an
   `experiments_index.md` row, and a `layout.py` helper. New caches without an identity
   field (onboarding §4). Analysis utilities ≠ experiments — if unsure, propose, do not
   invent docs.
2. **Fail-loud (r9):** new `try/except` in `src/` that swallows errors around our own
   files/formats. Boundary guards (network, process pools, per-cell sweep isolation)
   are allowed.
3. **Comment discipline (r10):** new comment blocks >3 lines without a `docs/` pointer
   (skip module/file top docstrings).
4. **Reuse & duplication review:**
   - *(a) Name tripwire (daily = touched only):* from the WIP diff, collect new or
     substantially changed `def` names in `src/`, `scripts/`, `analysis/`. Grep those
     names across those trees; flag any name also defined in another non-test file.
     Diff copies for DIVERGENCE (killer case: `survey_assets` in 4 files, only one
     thread-safe). Run the full-repo census (`grep` all defs for multi-file families)
     when the newest PRIOR report in `outputs/audits/` is older than 5 days, or on
     request — not every audit. (Mechanical trigger; needs no memory of past runs.)
   - *(b) Semantic review of WIP delta:* for every new or substantially-changed
     function, search by **primitives, not name** — library calls, distinctive
     constants (thresholds, seeds, n_boot), signature shape, docstring concepts —
     then READ top candidates and classify: {pure orchestration | duplicate of X →
     propose consolidation | belongs in canonical home → propose move | genuinely new}.
     Mark consolidation proposals **do not execute without reading both sides**.
   - *(c) Housing rule:* every non-orchestration behaviour has ONE owning module —
     CONTRIBUTING → "Where code goes". Flag a new def that neither calls its home nor
     is orchestration, **regardless of naming**.
   - Known-frozen: era-1 scripts with SUPERSEDED banners (`alignment_analysis.py`,
     `uncertainty_analysis.py`) — do not propose consolidating their local wrappers.
   - If WIP exceeds ~10 new functions, fan semantic review to a subagent per directory;
     merge verdicts conservatively (any "duplicate" wins over "genuinely new").
5. **Contract hygiene:** if `src/survey_features/oracle.py` changed, check whether the
   change alters output MEANING; verify `ORACLE_CONTRACT_VERSION` was bumped or
   correctly not bumped, and that `docs/onboarding.md` §3 matches. If a migration is
   in flight, say so explicitly — do not treat "0 stale cells" as success when a bump
   is expected.
6. **Health gates:**
   - `py_compile` every touched `.py`
   - conda python: `pytest tests/ -q`
   - `PYTHONPATH=src` + conda python: `scripts/rerun_oracles.py --dry-run` → expect 0
     cells to recompute unless a contract migration is knowingly in flight.

**Abort rule:** if any health gate fails, mark the report **GATES FAILED — no commits**.
Skip Phase 6 entirely (even under `apply-all`) until the user overrides in a follow-up
message. Still write the report and walk-through.

## Phase 4 — Staleness & duplication sweep (propose only)

- Docs whose claims WIP contradicts → propose SUPERSEDED banner or refresh.
- Overlapping-purpose files: reference counts via grep; propose keep/merge/archive.
- Data files in code dirs; spent one-shot scripts → propose `archive/`.
- Large stray files at root.

## Phase 5 — Report

Write `outputs/audits/repo_audit_<YYYY-MM-DD>.md` (if a same-day report exists, append
`_HHMM` rather than overwrite).

**Write for a human who was NOT in the session.** Plain sentences; every item framed
as a decision with "say X and I'll do Y"; no internal phase/rule numbers or repo
jargon (era, contract, WIP, gate, mode) without a one-line plain explanation;
summarize any referenced document inline instead of pointing at it. If the report
needs this skill file to be understood, it has failed.

Structure:

1. Opening paragraph in plain words: what this is, what was checked and its outcome
   (branch, scope, whether tests and caches are healthy), what the reader needs to do.
2. Delta summary (the change groups).
3. Actions TAKEN (each: what + why). Empty in `audit` mode except read-only checks.
4. Actions PROPOSED (rationale + exact command) — include soft-delete commands when
   mode was `audit`.
5. **Commit plan** (even when not committing): one subsection per group with message
   draft + path list. State clearly: "Not committed — confirm to apply."
6. Policy violations.
7. **Ranked walk-through — max 10**, highest-stakes first. Put gate failures and
   contract issues above clutter. Tag consolidation items `READ-BOTH-SIDES`.

Print: compact summary + walk-through + report path + whether commits are blocked.

## Phase 6 — Commits (only `apply-all` + user confirmation + gates green)

Do **not** commit in the same turn you first present the plan.

1. Present the commit plan in the report / chat.
2. Confirmation must arrive in a LATER turn than the plan's first presentation —
   never present and commit in the same turn, whatever the wording (e.g. "commit the
   audit plan", "apply-all confirmed"). Edits to the plan from the user win.
3. Then: create branch if needed, stage only listed paths, commit as the Phase-1 groups.
   Messages match recent `git log` style; Co-Authored-By per harness rules if required.
4. **Never push. Never commit** `outputs/`, `.env`, scratch, trash, or "propose" items.
5. `git status` after; if anything unexpected staged, abort that commit and report.

## Phase 7 — Self-amendment

If today produced a NEW recurring failure mode this checklist misses, propose (never
apply) a one-line addition to this file in proposed-actions. Do not let this file grow
without bound — if proposing an add, also propose one obsolete line to remove.

## Guardrails

- NEVER delete (hard or soft) anything under `outputs/cache/cells*` — era archives are
  provenance of published numbers.
- NEVER touch `.env` or credentials.
- NEVER hard-delete logs, audits, or anything under `outputs/` except emptying paths
  already inside `outputs/.trash/` when the user explicitly asks to purge trash.
- `data.zip` / `outputs.zip` / cache eras / doc-referenced artifacts: propose-only.
- Ambiguity → **propose**, never action.
- Never upgrade `audit` → `apply-clean` / `apply-all` without the user's words.
