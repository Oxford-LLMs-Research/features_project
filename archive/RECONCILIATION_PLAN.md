# Reconciliation Plan — make the writeups match the data

*Drafted 2026-05-29. Status: awaiting approval. No code/doc edits have been made yet
(other than this plan, the conceptual note `framing_and_comparisons.md`, and Claude's
private memory). Goal chosen this session: **reconcile docs to data**, building toward
**submission**, presenting **both models side-by-side**, with **`current_state.tex` as the
single canonical writeup**.*

---

## 0. The core problem in one paragraph

The experiment on disk is **6 surveys × 5 targets × 3 countries × 2 conditions × 2 models
(DeepSeek-V3.2 + Kimi-K2.5)** — ~178 rows per model, run 2026-05-14/15. But every writeup
describes an **older, different run**: single-model DeepSeek, **5×5** grid, 300 rows /
n=295. So the prose, tables, and figures do **not** match the CSVs, and Kimi is absent
everywhere. Worse, the analysis scripts silently load **only the newest CSV per survey**
(= Kimi, which ran a day later), so re-running them today would produce Kimi-only numbers
that match neither the papers nor a clean two-model comparison. The old 5×5 data is **not
recoverable** (both git-tracked zips contain the current 5×3 run, not the old one).

**Verified current numbers (computed from the CSVs this session):**

| Model | rows | valid | oracle | model | random | value/random | cost | share value>0 |
|---|---|---|---|---|---|---|---|---|
| Kimi-K2.5 | 178 | 174 | 0.592 | 0.528 | 0.503 | **+0.025** | 0.064 | 0.55 |
| DeepSeek-V3.2 | 178 | 177 | 0.592 | 0.527 | 0.502 | **+0.026** | 0.064 | 0.55 |

Per-survey value-over-random (both models track closely):

| survey | Kimi | DeepSeek |
|---|---|---|
| ess_wave_11 | +0.066 | +0.063 |
| wvs | +0.036 | +0.050 |
| afrobarometer | +0.020 | +0.023 |
| asianbarometer | +0.018 | +0.016 |
| latinobarometer | +0.008 | +0.007 |
| arabbarometer | −0.005 | −0.008 |

---

## 1. Principles for the reconciliation

1. **CSVs are the single source of truth.** Every number in prose/tables/figures must be
   regenerable from `outputs/grid_summary__*.csv` by a script. No hand-typed stats.
2. **Model is a first-class dimension.** Both models appear side-by-side in every table;
   figures encode model (grouped bars / hue / facet). Never silently collapse to one.
3. **Nothing destructive without a git-safe trail.** Archive (move), don't delete.
4. **Honesty about strength.** The real result (value ≈ +0.025, ~55% beat random) is
   weaker than the old pilot's story. The writeup states this plainly; we do not inherit
   the older draft's more optimistic phrasing.

---

## 2. Phase A — file-by-file changes

### 2a. Analysis layer (the silent-single-model bug)

| File | Problem | Planned change |
|---|---|---|
| `output_layout.py` | `collect_grid_summary_paths()` returns **one CSV per survey** (newest mtime = Kimi). | Add `collect_all_grid_summaries()` (or a `per_model=True` mode) returning **every** CSV with `(survey_id, model_tag)` parsed. Leave the old fn for back-compat but stop using it for multi-model loads. |
| `build_current_state_report.py` | `load_all_summary()` loads one model; tables have no model column. | Load all models; add `llm_model` column; rewrite `write_metrics_tables()` so global/survey/condition/bucket tables carry **DeepSeek + Kimi columns side-by-side**. |
| `prelim_results_stats.py` | Same single-model load; **its output `_prelim_stats.json` is currently MISSING**, so bucket/extremes tables silently skip. | Make model-aware; regenerate `_prelim_stats.json` (per-model bucket + worst/best/top-cost rows). |
| `build_paper_figures.py` | `load_grid_concat()` loads one model; figures have no model encoding. | Load both; add model as hue/facet on the survey-bars, histograms, scatter, box. |
| `prelim_aggregate.py` | Prints per-survey, one model. | Iterate all CSVs incl. model tag in the label. (Low priority.) |
| `grid_analysis.py` | **Hardcoded to deleted pilot** (`TARGETS=["Q47"…]`, `COUNTRIES=[…]`, `GRID_CSV=grid_summary.csv` which doesn't exist). Dead against current data. | Phase A: add a deprecation header pointing to the manifest-driven path. **Do not delete** — salvage its `hit_rate_*`, `demographics_lead_*`, `unmappable_*` logic in Phase B for the alignment metric + leakage audit. |

### 2b. Regenerate artifacts (run, in order, with miniconda `python`)

```
python analysis/prelim_results_stats.py > outputs/_prelim_stats.json   # was missing
python analysis/build_current_state_report.py                          # tables → paper/generated_current_state/
python analysis/build_flagship_appendix.py                             # flagship cells still exist on disk
python analysis/build_paper_figures.py                                 # figures → paper/figures/
```

### 2c. `current_state.tex` prose fixes (canonical doc)

- Grid: "five targets and five countries" → **5 targets × 3 countries**; "300 rows / 295
  valid" → **178/model (~351 valid across both)**.
- "The LLM model recorded … is deepseek-ai/DeepSeek-V3.2" → **both models**, side-by-side.
- Replace the 0.6425/0.5303/0.5083 global means with the regenerated two-model `\input{}`
  tables (already auto-generated, so this is mostly deleting stale inline numbers).
- Fix code references: `phase0b_oracle.py` → **`phase0b_oracle_autogluon.py`**.
- **Verify** the oracle hyperparameters quoted in prose (max_depth=4, n_estimators=300,…)
  actually match `phase0b_oracle_autogluon.py` — the old `phase0_consolidation.md` said
  max_depth=6, so this needs checking against code, not docs.
- Keep/strengthen the honest "Test 1 partial / Test 2 preliminary / Test 3 not done" and
  "no uncertainty yet" framing — it's accurate.

### 2d. Merge + archive the second paper draft

- Fold any worthwhile **narrative/figure framing** from `prelim_paper.tex` into
  `current_state.tex` (intro motivation, figure captions), then **archive
  `prelim_paper.tex`** → `archive/`. (Confirm the specific bits to keep before archiving.)

### 2e. README touch-ups

- Note the actual run is **5×3, two models**; the manifest example currently implies 5×5
  single-model.
- Add an **interpreter note**: analysis scripts need pandas/yaml/matplotlib; the repo
  `.venv` lacks them — document which interpreter is canonical (see §4).

### 2f. Archive the orphaned pilot docs (git-safe move)

- `phase0_consolidation.md`, `phase0b_grid_findings.md` → `archive/`. Both describe the
  deleted WVS-only 5×5 DeepSeek pilot and reference `run_e2e.py` (gone). *(Both are already
  in `.gitignore`, so this is local housekeeping only.)*

### 2g. Relocate the conceptual note

- `framing_and_comparisons.md` → **`docs/`** (proposed). Rationale: you chose "keep but
  move"; `paper/` is git-ignored, so burying a submission-framing note there would drop it
  from version control. A new top-level `docs/` keeps it tracked. **Confirm destination.**

---

## 3. Verification (definition of done for Phase A)

1. `build_current_state_report.py` + figures run clean under the canonical interpreter.
2. Every table/figure shows **both models**.
3. Spot-check: the global means in the regenerated tables equal the §0 numbers
   (oracle 0.592 / model 0.527–0.528 / value +0.025).
4. `current_state.tex` compiles (pdflatex) with no stale inline numbers and correct file
   references.
5. `git status` shows only intended changes; archived files moved, not deleted.

---

## 4. One decision needed before execution

**Canonical Python interpreter.** The documented `.venv` has no pandas; miniconda
`python` (3.9.1) has the full analysis stack. Options: (a) `pip install` the analysis deps
into `.venv` and standardize on it; (b) document miniconda as the analysis environment.
This matters for reproducibility in a submission. *(Recommend (a) so one env runs both
pipeline and analysis — but confirm.)*

---

## 5. Phase B / C preview (NOT part of this reconciliation — for later)

Once the docs are honest, the submission-strengthening work (your stated goal) lines up as:

- **B1 — Leakage audit.** Quantify demographic-target leakage (oracle≈1.0 via near-label
  proxies). Decides whether to re-select attitude targets or stratify. Salvage
  `grid_analysis.demographics_lead_table` + `_is_demographic`.
- **B2 — Reframe around *alignment*, not the accuracy horse-race.** Lead with selection
  alignment (captured-importance, rank correlation) + make cross-national **Test 2** the
  signature result — the one thing no ML-importance baseline can do. (See
  `framing_and_comparisons.md`.)
- **B3 — Uncertainty.** Cluster/bootstrap CIs on value-over-random and cost (currently none).
- **C — Restore breadth if wanted.** Reconcile the 5-country manifest vs the 3-country run;
  decide final grid size; consider the design's ablations (reasoning, recognition).

---

## 6. Proposed execution order (after approval)

1. Confirm: interpreter (§4), framing-doc destination (§2g), prelim_paper bits to keep (§2d).
2. Code: make `output_layout` + 3 analysis scripts model-aware (§2a).
3. Regenerate `_prelim_stats.json`, tables, flagship, figures (§2b).
4. Rewrite `current_state.tex` prose; merge from prelim_paper (§2c–2d).
5. Archive orphaned docs + prelim_paper; relocate framing note; README touch-ups (§2e–2g).
6. Compile, verify (§3), report diff.
