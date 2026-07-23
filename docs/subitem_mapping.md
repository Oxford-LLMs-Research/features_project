# Sub-item (subconcept) mapping — experiment design

> **Status:** design only. Do **not** run the full map/score sweep until the open
> decisions below are settled. Artifacts (when run) live under
> `outputs/subitem_mapping/` and never overwrite `outputs/format_pilot/`.

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

The goal is **fine-grained measurement** of mapping loss at two granularities,
without silently changing the main MiniLM arm-C capability results.

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
| Disambiguator = **nemotron** | Optional score `k_mode` (see Scoring rules) |
| Embedder = **`all-MiniLM-L6-v2`** (main baseline) | |
| Arm **C** only | |
| `top_n=20`, `min_similarity=0.30` | |
| `pipe_types = {respondent_attribute}` | |
| Both selectors: `deepseek`, `kimi` | |

**Not** swept here: embedder size (see `docs/embedding_sensitivity.md`),
threshold / top-k, extractor model, selector essays.

### Isolation (mirror embedding_sensitivity)

| Artifact | Path |
|----------|------|
| Baseline parent maps / scores | `outputs/format_pilot/<selector>/` (unchanged) |
| Expanded maps | `outputs/subitem_mapping/<selector>/maps/` |
| Diagnostics | `outputs/subitem_mapping/<selector>/diagnostics.csv` |
| Optional expanded scores | `outputs/subitem_mapping/<selector>/scores_<selector>.csv` |
| Provenance | `outputs/subitem_mapping/manifest.json` |

Gen/extract are **never** re-run; only mapping expansion (+ optional score) writes
under `subitem_mapping/`.

## Mapping protocol

### Units

For each piped parent feature `F` with label `L`, context `C`, sub_items `S`:

1. **Parent unit** (always, same as today): retrieve+disambig with query `(L, C)`.
2. **Sub_item units** (only if `|S| ≥ 2` — same bundling threshold as `n_bundled`):
   for each `s ∈ S`, retrieve+disambig with:
   - `feature_label = s`
   - `feature_context =` parent context plus an explicit parent anchor, e.g.
     `"{C} (sub-measure of {L})"` (fall back to `"sub-measure of {L}"` if `C` empty).

Features with `|S| ≤ 1` contribute **parent units only** (no inflation from
degenerate singleton lists).

Non-piped types remain recorded, not mapped (unchanged).

### Recommended default: dual-layer (parent AND sub_items)

Keep the parent call **and** add sub_item calls. Rationale:

- Parent map rate stays directly comparable to `format_pilot` maps.
- Subconcept rates are additive diagnostics, not a replacement definition of “a feature.”
- Enables conditional metrics (parent hit × sub_item hits).

**Rejected for v1 (open decision if revisited):** “expand-only” (skip parent when
bundled) — breaks head-to-head parent none-rate vs baseline and confuses concept
vs subconcept failure.

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

### Primary (map diagnostics — run these first)

Denominators are always **piped** units of the stated kind.

| Metric | Definition |
|--------|------------|
| `parent_map_rate` | `# parent units with code` / `# piped parents` |
| `parent_none_rate` | `1 − parent_map_rate` |
| `subitem_map_rate` | `# sub_item units with code` / `# sub_item units` (bundled parents only) |
| `subitem_none_rate` | `1 − subitem_map_rate` |
| `bundled_parent_frac` | `# parents with \|S\|≥2` / `# piped parents` |
| `mean_subitems_per_bundled` | mean `|S|` among bundled parents |

### Conditional / joint (concept × subconcept)

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

### Optional secondary (XGB score — only under `subitem_mapping/`)

Hold cell, oracle, CV, `SCORE_N_DRAWS` fixed. Emit rows tagged by `k_mode`:

| `k_mode` | Feature set | `k` |
|----------|-------------|-----|
| `parent` | `parent_codes` | `|parent_codes|` |
| `expanded` | `expanded_codes` | `|expanded_codes|` |
| `subitems_only` | `subitem_codes` (cells with ≥1 sub_item unit; else skip or empty) | `|subitem_codes|` |

Matched-k random baseline **must** use the same `k` as that row’s feature set
(existing `evaluate_feature_set` / `single_random_draw` contract). Expanding codes
**raises k** and is therefore **not** comparable as a drop-in replacement for main
arm-C VoR without an explicit `k_mode` split.

**Hard rule:** never append expanded-k rows to `outputs/format_pilot/scores_*.csv`.
Main paper numbers stay parent-only MiniLM arm C.

Optional fairness checks (if scoring):

- Report Δ VoR / Δ captured_importance for `expanded` vs `parent` **within** this
  experiment (same extracts).
- Optionally also score `parent` codes re-read from `format_pilot` maps as a
  checksum that isolation did not drift.

## Scoring rules (summary)

1. **Unit of analysis for diagnostics** = mapping unit (`parent` or `sub_item`), not
   unique codes.
2. **Unit of analysis for XGB** = deduped code list under an explicit `k_mode`.
3. **Do not** silently inflate `k` in the main experiment path.
4. Prefer shipping **map diagnostics** before any full XGB sweep (cheaper; answers
   the concept vs subconcept question directly).

## How to run (when approved)

Smoke (limit cells; confirm artifact layout):

```bash
python scripts/run_subitem_mapping.py --phase map --selector deepseek --disambiguator nemotron --limit 2
python analysis/subitem_mapping.py --selector deepseek
```

Full map diagnostics (both selectors), then optional score:

```bash
python scripts/run_subitem_mapping.py --phase map   --selector deepseek --disambiguator nemotron --arms C
python scripts/run_subitem_mapping.py --phase map   --selector kimi     --disambiguator nemotron --arms C
python analysis/subitem_mapping.py

# optional — expensive; only after diagnostics look sane
python scripts/run_subitem_mapping.py --phase score --selector deepseek --k-modes parent,expanded
python scripts/run_subitem_mapping.py --phase score --selector kimi     --k-modes parent,expanded
```

Prerequisites: existing `format_pilot/<selector>/extracted/` (and baseline maps for
comparison). Same genuine cells as `run_main.py`.

Scaffolding:

- Paths: `survey_features.layout.subitem_mapping_dir` / `subitem_run_dirs`
- Mapper helper: `survey_features.subitem_map.map_features_with_subitems`
- Runner stub: `scripts/run_subitem_mapping.py`
- Analysis stub: `analysis/subitem_mapping.py`

## Interpretation guide

| Pattern | Likely reading |
|---------|----------------|
| Parent map rate ≈ baseline; sub_item none rate high | Collapse is real: fine measures often unmapped; main k understates requested specificity |
| Parent none high but `parent_none_some_subitem_maps` high | One-to-one parent query is the bottleneck; expansion recovers |
| Expanded VoR ≫ parent VoR at larger k | Extra codes carry signal — but attribute gains to **budget (k)** vs **better mapping** carefully (compare matched-k and/or fixed-k rows) |
| Expanded VoR ≈ parent VoR; Jaccard parent↔subitem high | Sub_items mostly re-hit the same variables; little eval upside |
| Sub_item map rate high, codes rarely in oracle top-k | Mapping “succeeds” semantically but not predictively — report separately from capability claims |

Relate findings back to main results as a **mapping granularity caveat**, analogous
to embedder sensitivity on code sets: capability claims stay on parent-only MiniLM
arm C unless this experiment shows material, well-controlled score movement.

## Cost note

Extra disambiguator calls ≈ sum of `|S|` over bundled piped parents per cell.
Bundling is common in current extracts (often several bundled features per cell).
Budget API cost before a full sweep; use `--limit` smoke runs first. Map skips
existing JSON under `subitem_mapping/` (resume-friendly).

## Open decisions

1. **Context string for sub_items** — parent-anchored paraphrase (proposed) vs raw
   sub_item label only vs dedicated `DISAMBIG_PROMPT_SUBITEM`.
2. **Bundling threshold** — keep `|S| ≥ 2` (aligned with `n_bundled`) vs also map
   `|S| == 1`.
3. **Score in v1?** — diagnostics-only first (recommended) vs immediate
   `parent`+`expanded` XGB.
4. **Fixed-k rows** — whether optional score should always include `k_spec=5,10`
   (like `run_main`) so expanded vs parent can be compared at equal budget.
5. **Re-map parents under this runner** vs copy parent outcomes from
   `format_pilot` maps and only call the API for sub_items (saves cost; risks
   tiny drift if prompt/embed path changes).
6. **Human audit sample** — annotate a small set of bundled features for
   “should this sub_item have a survey home?” to separate retrieval failure from
   true absence.
7. **Interaction with embedding_sensitivity** — out of scope for v1; if both
   matter, nest later as `subitem_mapping/<embed_slug>/…` rather than mixing trees.

## Non-goals

- Changing default `map_features` behaviour or `format_pilot` scores.
- Re-extracting with a different bundling policy.
- Claiming that expanded-k VoR is the new headline capability metric without an
  explicit protocol change in the main design memo.
