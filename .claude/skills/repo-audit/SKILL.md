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
and `docs/onboarding.md` §3–4 (contracts, cache identity) — you do not invent taste.
Default is **propose**; irreversible actions need an explicit apply mode (below).
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
`outputs/audits/` and `outputs/.trash/`. The whole tree matters: a score sweep's
freshest writes land in `outputs/main/`, not in logs.

`tasklist` / `Get-Process python` is **advisory only**. Never treat "no python.exe"
as proof the repo is idle. If the busy check is inconclusive, STOP.

Also note:

- Current branch. If on `main`/`master`, **do not create a branch yet** — propose a
  branch name; create it only under `apply-all` after the user confirms.
- Ask once if unclear: commit **all current WIP**, or only paths touched since a
  ref/timestamp the user names? Default: all current WIP.

## Phase 1 — Delta inventory

- `git status --short`, `git diff --stat`, and `git log --oneline -3`.
- Classify EVERY untracked file: {track in git | add to .gitignore | scratch →
  soft-delete | propose}. When unsure → **propose**.
- Summarize WIP as 3–7 logical change groups (**commit plan**, not commits yet).
- For each group: list the exact paths to stage. Exclude propose, scratch, or
  anything under `outputs/`.

## Phase 2 — Mechanical clean

Targets (same set in every mode):

- All `__pycache__/` dirs and stray `.pyc`, `.pytest_cache/`.
- `outputs/.tmp/*` (AutoGluon scratch).
- `outputs/logs/`: files older than 7 days; 0-byte or failed-start stubs.
  **Never hard-delete.** Soft-delete only if the basename is **not** mentioned in any
  `docs/**/*.md` or root `*.md`. If ambiguous → propose.
- Logs sitting outside `outputs/logs/` (e.g. next to `main/` data): propose
  relocation to `outputs/logs/<context>/`.
- `outputs/audits/`: reports beyond the newest 14 → soft-delete (keep today's draft).

**Soft-delete procedure** (`apply-clean` / `apply-all`):

1. Move to `outputs/.trash/repo-audit_<YYYY-MM-DD>/` preserving relative paths.
2. Log every move in the report.
3. Do **not** empty trash here. Propose pruning trash older than 14 days only.

In `audit` mode: write the exact move commands into Actions PROPOSED; take no deletes.

## Phase 3 — Policy checks (report violations; fix only with approval)

Run on the **WIP under review** (`git diff` + untracked files in scope):

1. **Cache identity (onboarding §4):** new caches without an identity field.
2. **Fail-loud:** new `try/except` in `src/` that swallows errors around our own
   files/formats. Boundary guards (network, process pools, per-cell isolation) are allowed.
3. **Comment discipline:** new comment blocks >3 lines without a `docs/` pointer
   (skip module/file top docstrings).
4. **Reuse & duplication:** for new/changed `def` names in `src/` and `scripts/`,
   grep for duplicates; semantic review by primitives. Housing rule: every non-orchestration
   behaviour has ONE owning module (CONTRIBUTING → Where code goes).
5. **Oracle contract:** if `oracle.py` changed meaning, verify `ORACLE_CONTRACT_VERSION`
   was bumped and `docs/onboarding.md` §3 matches.
6. **Health gates:**
   - `py_compile` every touched `.py`
   - conda python: `pytest tests/ -q`
   - `PYTHONPATH=src` + conda: `scripts/rerun_oracles.py --dry-run` → expect 0 cells
     to recompute unless a contract migration is in flight.

**Abort rule:** if any health gate fails, mark the report **GATES FAILED — no commits**.
Skip Phase 6 until the user overrides. Still write the report and walk-through.

## Phase 4 — Staleness (propose only)

- Docs whose claims WIP contradicts → propose refresh.
- Data files in code dirs; spent one-shot scripts → propose soft-delete or out-of-tree zip.
- Never propose deleting `outputs/cache/cells/` without a verified external snapshot.

## Phase 5 — Report

Write `outputs/audits/repo_audit_<YYYY-MM-DD>.md` (append `_HHMM` if same-day exists).

Write for a human who was NOT in the session. Structure:

1. Opening paragraph: what was checked, outcome, what the reader needs to do.
2. Delta summary.
3. Actions TAKEN (empty in `audit` mode except read-only checks).
4. Actions PROPOSED (rationale + exact command).
5. **Commit plan** — "Not committed — confirm to apply."
6. Policy violations.
7. **Ranked walk-through — max 10**, highest-stakes first.

## Phase 6 — Commits (only `apply-all` + user confirmation + gates green)

Do **not** commit in the same turn you first present the plan.

1. Present the commit plan.
2. Confirmation must arrive in a LATER turn.
3. Then: create branch if needed, stage only listed paths, commit.
4. **Never push. Never commit** `outputs/`, `.env`, scratch, trash, or "propose" items.
5. `git status` after; if unexpected staged, abort that commit and report.

## Guardrails

- NEVER delete (hard or soft) `outputs/cache/cells/` — current-era provenance.
- NEVER touch `.env` or credentials.
- NEVER hard-delete logs, audits, or anything under `outputs/` except emptying paths
  already inside `outputs/.trash/` when the user explicitly asks to purge trash.
- Ambiguity → **propose**, never action.
- Never upgrade `audit` → `apply-clean` / `apply-all` without the user's words.
