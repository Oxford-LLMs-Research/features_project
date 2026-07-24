# Sub-item (subconcept) mapping — experiment design

> **Status:** v1 protocol locked for **kimi-only** map + score. Do **not** fold
> similarity-threshold sweeps into this run (see `docs/similarity_threshold.md`).
> Artifacts live under `outputs/subitem_mapping/` and never overwrite
> `outputs/format_pilot/`.
> **Results appendix:** `docs/subitem_mapping_results.md`.

## Question

When the extractor bundles several specific measures under one parent feature
(e.g. `assets_owned` with `sub_items = [land, vehicle, electronics, housing_quality]`),
the current mapper is **one-to-one on the parent label only** — `sub_items` are
audit-only (`n_bundled`). That collapses several potentially mappable survey
concepts into a single code (or a single `none`).

This experiment asks:

1. **Concept failure:** how often do **parent features** fail to map (`none`)?
2. **Subconcept failure:** how often do **sub_items** fail to map when each is
   treated as its own retrieval + disambiguation unit?
3. Conditionally: when the parent maps, what fraction of its sub_items also map?
   When the parent is `none`, do any sub_items still find codes?
4. **Downstream performance:** does expanding to sub_item codes **raise** eval
   metrics (VoR, model_acc, cost_of_imperfect, captured_importance) vs parent-only?
5. **Count inflation:** how do requested / mapped feature counts and `k` change
   under expansion?

The goal is fine-grained measurement of mapping loss **and** whether expansion
moves capability numbers — without silently changing the main MiniLM arm-C
headline results.

## Motivation (current behaviour)

| Stage | Behaviour today |
|-------|-----------------|
| Extract (Qwen, fixed) | Stores bundled measures in `sub_items` |
| Map (`map_features`) | ONE-TO-ONE on parent `feature`; retrieval query = label+context |
| Disambig prompt | Does not iterate `sub_items` |
| Score (XGB) | Uses deduped `mapped_codes`; matched-k random uses that `k` |

Example collapse: parent `assets_owned` → one code (or none), even if land /
vehicle / housing each have close survey variables.

## Design (held fixed vs varied)

| Held fixed | Varied |
|------------|--------|
| Selector free-text (`gen`) + Qwen extract (reuse `format_pilot/`) | Mapping **unit of analysis**: parent-only vs parent **+** sub_item units |
| Disambiguator = **nemotron** | Score `k_mode` / `k_spec` (see Scoring) |
| Embedder = **`all-MiniLM-L6-v2`** (main baseline) | |
| Arm **C** only | |
| `top_n=20`, `min_similarity=0.30` (held; **not** swept here) | |
| `pipe_types = {respondent_attribute}` | |
| **v1 selector = `kimi` only** | |

**Not** swept here: embedder size (`docs/embedding_sensitivity.md`), similarity
threshold (`docs/similarity_threshold.md`), extractor model, selector essays,
deepseek (optional extension after kimi v1).

### Isolation (mirror embedding_sensitivity)

| Artifact | Path |
|----------|------|
| Baseline parent maps / scores | `outputs/format_pilot/kimi/` (unchanged) |
| Expanded maps | `outputs/subitem_mapping/kimi/maps/` |
| Diagnostics | `outputs/subitem_mapping/kimi/diagnostics.csv` |
| Expanded scores | `outputs/subitem_mapping/kimi/scores_kimi.csv` |
| Provenance | `outputs/subitem_mapping/manifest.json` |

Gen/extract are **never** re-run; only mapping expansion + score write under
`subitem_mapping/`.

### Orthogonal experiment: similarity threshold

`min_similarity=0.30` may exclude informative candidates. Studying score
distributions and threshold effects is a **separate** experiment — do **not**
change the threshold or fold a sweep into this v1 run. See
`docs/similarity_threshold.md` (`outputs/similarity_threshold/`).

## Mapping protocol

### Units

For each piped parent feature `F` with label `L`, context `C`, sub_items `S`:

1. **Parent unit** (always): outcome taken from **`format_pilot` parent maps**
   (copy; no re-call) so parent none-rates stay bit-identical to baseline.
2. **Sub_item units** (only if `|S| ≥ 2` — same bundling threshold as `n_bundled`):
   for each `s ∈ S`, retrieve+disambig with:
   - `feature_label = s`
   - `feature_context =` parent-anchored string
     `"{C} (sub-measure of {L})"` (fall back to `"sub-measure of {L}"` if `C` empty).

Features with `|S| ≤ 1` contribute **parent units only** (no inflation from
degenerate singleton lists).

Non-piped types remain recorded, not mapped (unchanged).

### Recommended default: dual-layer (parent AND sub_items)

Keep parent outcomes **and** add sub_item calls. Rationale:

- Parent map rate stays directly comparable to `format_pilot` maps.
- Subconcept rates are additive diagnostics, not a replacement definition of “a feature.”
- Enables conditional metrics (parent hit × sub_item hits).

**Rejected for v1:** “expand-only” (skip parent when bundled) — breaks head-to-head
parent none-rate vs baseline and confuses concept vs subconcept failure.

**v1 parent handling:** **copy** parent selected codes / candidates metadata from
`format_pilot` maps; API-call **sub_items only**. Rationale: saves ~parent-share of
disambig wall-time (~half of expanded units are parents; see Runtime) and guarantees
zero parent drift vs the MiniLM arm-C baseline. Remap-parents remains available as
a checksum if prompt/embed paths change later.

### Dedup and double-counting

| Layer | Rule |
|-------|------|
| Unit-level success | Count each parent / each sub_item independently (before dedup). Two sub_items mapping to the same code = **two successes**, one unique code. |
| `parent_codes` | Deduped codes from parent units only (arrival order) — **identical contract** to today’s `mapped_codes`. |
| `subitem_codes` | Deduped codes from sub_item units only. |
| `expanded_codes` | Deduped union of parent + sub_item codes (arrival: parents first, then sub_items). |

Never write expanded code lists into `format_pilot` map JSON.

### Artifact shape (per cell map JSON)

Extend (under `subitem_mapping/` only) beyond the parent feature list:

```json
{
  "mapping_mode": "parent_plus_subitems",
  "parent_codes": ["Q12"],
  "subitem_codes": ["Q12", "Q15"],
  "expanded_codes": ["Q12", "Q15"],
  "units": [
    {
      "unit_kind": "parent",
      "parent_feature": "assets_owned",
      "unit_label": "assets_owned",
      "piped": true,
      "selected_code": "Q12",
      "n_candidates": 20
    },
    {
      "unit_kind": "sub_item",
      "parent_feature": "assets_owned",
      "unit_label": "land",
      "piped": true,
      "selected_code": null,
      "n_candidates": 14
    }
  ]
}
```

Parent-level `features[]` (current schema) may still be included for parity with
baseline audits.

## Metrics taxonomy

v1 reports **three blocks**: map diagnostics, count / bundling changes, and final
eval scores. Diagnostics alone are not enough.

### 1. Map diagnostics (concept vs subconcept)

Denominators are always **piped** units of the stated kind.

| Metric | Definition |
|--------|------------|
| `parent_map_rate` | `# parent units with code` / `# piped parents` |
| `parent_none_rate` | `1 − parent_map_rate` |
| `subitem_map_rate` | `# sub_item units with code` / `# sub_item units` (bundled parents only) |
| `subitem_none_rate` | `1 − subitem_map_rate` |
| `bundled_parent_frac` | `# parents with \|S\|≥2` / `# piped parents` |
| `mean_subitems_per_bundled` | mean `|S|` among bundled parents |

### 2. Count / bundling changes (requested vs mapped)

Per cell, compare parent-only vs expanded:

| Metric | Definition |
|--------|------------|
| `n_features` / `n_piped` | Extracted / piped parent counts (unchanged by expansion) |
| `n_subitem_units` | Sub_item mapping units (`|S|` sum over bundled) |
| `n_mapped_parent` / `k_parent` | `|parent_codes|` |
| `n_mapped_expanded` / `k_expanded` | `|expanded_codes|` |
| `bundling_expansion_factor` | `(n_piped + n_subitem_units) / n_piped` (call volume) |
| `k_inflation` | `k_expanded / k_parent` (unique-code inflation) |

Aggregate: mean/median expansion and `k` inflation across kimi cells.

### 3. Conditional / joint (concept × subconcept)

Among **bundled** piped parents only:

| Metric | Definition |
|--------|------------|
| `frac_subitems_map_given_parent_maps` | mean over bundled parents with parent code: (sub_item hits / `|S|`) |
| `frac_subitems_map_given_parent_none` | same, among parent=`none` |
| `parent_maps_all_subitems_miss` | fraction of bundled parents where parent has a code and **all** sub_items are `none` |
| `parent_maps_some_subitems_miss` | parent has code and **≥1** sub_item is `none` |
| `parent_none_some_subitem_maps` | parent `none` but **≥1** sub_item has a code |
| `code_jaccard_parent_vs_subitems` | per cell, Jaccard(`parent_codes`, `subitem_codes`); then mean |

Interpretation sketch:

- High `parent_maps_all_subitems_miss` → parent label finds a coarse/proxy variable;
  fine measures do not (or retrieval for short sub-labels is weak).
- High `parent_none_some_subitem_maps` → bundling hurt the parent query; expanding
  recovers concepts the one-to-one mapper dropped.
- Similar parent and sub_item map rates with high code Jaccard → expansion mostly
  duplicates; little new predictive content.

### 4. Final eval metrics (required in v1)

Hold cell, oracle, CV, and `SCORE_N_DRAWS` fixed. Emit rows tagged by explicit
`k_mode` / `k_spec` so expanded `k` does **not** silently confound MiniLM arm-C.

Primary score columns (same as main pipeline):

- `value_over_random` (VoR)
- `model_acc`
- `cost_of_imperfect`
- `captured_importance`

Compare **expanded** vs **parent** within this experiment (same kimi extracts).

| Row family | Feature set | `k` | Purpose |
|------------|-------------|-----|---------|
| `k_mode=parent`, natural | `parent_codes` | `|parent_codes|` | Parent-only capability (checksum vs `format_pilot` scores) |
| `k_mode=expanded`, natural | `expanded_codes` | `|expanded_codes|` | Shows count inflation + raw performance at model `k` |
| `k_mode=parent`, `k_spec=5` and `10` | `parent_codes` truncated/pad policy as `run_main` | fixed 5 / 10 | Equal-budget baseline |
| `k_mode=expanded`, `k_spec=5` and `10` | `expanded_codes` at same fixed k | fixed 5 / 10 | Equal-budget test: does expansion help at matched budget? |

Optional later: `k_mode=subitems_only` (cells with ≥1 sub_item unit).

Matched-k random baseline **must** use the same `k` as that row’s feature set
(existing `evaluate_feature_set` / `single_random_draw` contract).

**Hard rule:** never append expanded-k rows to `outputs/format_pilot/scores_*.csv`.
Main paper numbers stay parent-only MiniLM arm C.

### 5. Other downstream diagnostics worth reporting

| Check | Why |
|-------|-----|
| None-rate concept vs subconcept | Core map-loss story |
| Code-set Jaccard: expanded vs `format_pilot` parent maps | How much the scored set actually changes |
| Miss patterns (`parent_none_some_subitem_maps`, etc.) | Where expansion recovers or fails |
| Optional type-of-miss audit | Retrieval empty pool vs disambig `none` vs duplicate codes |

## Scoring rules (summary)

1. **Unit of analysis for diagnostics** = mapping unit (`parent` or `sub_item`), not
   unique codes.
2. **Unit of analysis for XGB** = deduped code list under an explicit `k_mode` /
   `k_spec`.
3. **Do not** silently inflate `k` in the main experiment path.
4. **v1 = map + score** (kimi-only). Prefer a short `--limit` map smoke before the
   full kimi sweep; do not wait for a deepseek pass.

## Runtime estimate (kimi-only v1)

Evidence from embedding_sensitivity / format_pilot (MiniLM or mpnet-ish parent
maps, nemotron disambig, serial calls):

| Phase | Observed / assumed | Note |
|-------|-------------------|------|
| Parent map | ~50s / map file; ~80 min / 104 files (52 cells × 2 cond) | ~1 LLM call per piped parent |
| Parent score | **~15–25 min** / selector×embedder at `SCORE_N_DRAWS=10` with cell ProcessPool (default `min(8, cpus-2)` workers); was ~1.5–2 h serial | Cell-level XGB workers; see Parallelism |

Kimi extract sample (`outputs/format_pilot/kimi/extracted/`, all 52 cells × 2 cond):

| Quantity | Value |
|----------|-------|
| Piped parents | 1699 |
| Bundled parents (`|S|≥2`) | 487 (28.7%) |
| Mean `|S|` among bundled | 2.92 |
| Sub_item units | 1421 |
| Expansion factor (calls if remap parents) | **1.84×** (3120 / 1699) |
| Expansion if **copy parents** (v1 default) | **~0.84×** parent-only wall-time for **new** API calls (1421 / 1699) |

### Wall-time ballpark

| Scope | Map | Score | Total (order-of-magnitude) |
|-------|-----|-------|----------------------------|
| **v1: kimi only**, copy parents + map sub_items, then score parent+expanded (+ fixed k=5/10) | ~1–1.5 h map (sub_items only; ~0.8× parent map) | **~20–30 min** with `--score-workers` / default cell pool (was ~2–3 h serial) | **~1.5–2.5 h** typical after parallel score |
| Both selectors (kimi + deepseek), same protocol | ~2× map | ~2× score | **~3–5 h** — defer; deepseek is an extension |

Assumes serial disambig + **parallel cell-level XGB** (`survey_features.score_cell`). Kimi parent
maps/scores in `format_pilot` already exist — reuse them for the parent baseline row.

### Cost / integrity knobs (recommended)

| Knob | Recommendation |
|------|----------------|
| `SCORE_N_DRAWS` | Keep **10** for v1 integrity (same as main / embedding_sensitivity). Use `5` only for smoke timing, not headline numbers. |
| Parallelism | **Score:** cell-level `ProcessPool` via `--score-workers` / `SCORE_WORKERS` (default `min(8, cpus-2)`); XGB threads per fit via `--score-xgb-nthread` / `SCORE_XGB_NTHREAD` (`cpus // workers`). Random draws stay serial *within* a cell. **Map:** process-level overlap of map (API) with unrelated work; optional cell-level map workers if rate limits allow. Do **not** change CV folds or draw seeds for speed. |
| Smoke | `--limit 2` map, then `--limit 4 --score-workers 4` score, before full kimi sweep. |

## How to run (v1 — kimi)

Smoke:

```bash
python scripts/run_subitem_mapping.py --phase map --selector kimi --disambiguator nemotron --limit 2
python analysis/subitem_mapping.py --selector kimi
```

Full kimi map + score:

```bash
python scripts/run_subitem_mapping.py --phase map   --selector kimi --disambiguator nemotron --arms C
python analysis/subitem_mapping.py --selector kimi

python scripts/run_subitem_mapping.py --phase score --selector kimi \
  --k-modes parent,expanded
# optional: --score-workers 8  (default already parallelizes cells)
# score runner should also emit k_spec=5,10 equal-budget rows (see Scoring table)
```

**Extension (not v1):** repeat with `--selector deepseek` after kimi results look sane.

Prerequisites: existing `format_pilot/kimi/extracted/` and parent maps (for copy);
parent scores preferred for checksum. Same genuine cells as `run_main.py`.

Scaffolding:

- Paths: `survey_features.layout.subitem_mapping_dir` / `subitem_run_dirs`
- Mapper helper: `survey_features.subitem_map.map_features_with_subitems`
- Runner: `scripts/run_subitem_mapping.py`
- Analysis: `analysis/subitem_mapping.py`

## Interpretation guide

| Pattern | Likely reading |
|---------|----------------|
| Parent map rate ≈ baseline; sub_item none rate high | Collapse is real: fine measures often unmapped; main k understates requested specificity |
| Parent none high but `parent_none_some_subitem_maps` high | One-to-one parent query is the bottleneck; expansion recovers |
| Expanded VoR ≫ parent VoR at natural (larger) k only | Gains may be **budget (k)** — check matched k=5/10 rows before claiming better mapping |
| Expanded VoR ≫ parent VoR at **matched** k | Extra / better codes carry signal at equal budget |
| Expanded VoR ≈ parent VoR; Jaccard parent↔subitem high | Sub_items mostly re-hit the same variables; little eval upside |
| Sub_item map rate high, codes rarely in oracle top-k | Mapping “succeeds” semantically but not predictively — report separately from capability claims |

Relate findings back to main results as a **mapping granularity caveat**, analogous
to embedder sensitivity on code sets: capability claims stay on parent-only MiniLM
arm C unless this experiment shows material, well-controlled score movement.

## Resolved decisions (v1)

1. **Selector scope:** **kimi only** (wall-clock control). Deepseek optional extension.
2. **Score in v1:** **yes** — map + score; include natural-k and matched-k (`k_spec=5,10`).
3. **Sub_item context string:** parent-anchored `"{C} (sub-measure of {L})"`.
4. **Parent outcomes:** **copy** from `format_pilot` maps; API-call sub_items only.
5. **`k_mode` discipline:** always tag rows; never write expanded scores into
   `format_pilot`.

## Open decisions (still unresolved)

1. **Bundling threshold** — keep `|S| ≥ 2` (aligned with `n_bundled`) vs also map
   `|S| == 1`.
2. **Human audit sample** — annotate a small set of bundled features for
   “should this sub_item have a survey home?” to separate retrieval failure from
   true absence.
3. **Dedicated `DISAMBIG_PROMPT_SUBITEM`** — stick with shared prompt + parent
   anchor for v1; revisit only if sub_item none-rates look spuriously high.
4. **Interaction with embedding_sensitivity** — out of scope for v1; if both
   matter later, nest as `subitem_mapping/<embed_slug>/…` rather than mixing trees.
5. **Score wiring details** — exact CLI for emitting all `k_spec` rows in one pass
   vs multiple invocations (implementation when score phase is unstubbed).

## Non-goals

- Changing default `map_features` behaviour or `format_pilot` scores.
- Re-extracting with a different bundling policy.
- Sweeping `min_similarity` / `top_n` inside this experiment.
- Claiming that expanded-k VoR is the new headline capability metric without an
  explicit protocol change in the main design memo.
- Running deepseek as part of v1.
