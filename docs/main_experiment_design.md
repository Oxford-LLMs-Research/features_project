# Main Experiment Design — confirmatory study

*Drafted 2026-06-11 from the two pilots + grounding computations (power analysis, country
availability, cross-survey overlap, theme taxonomy, Nebius model inventory). Decisions
marked **[USER]** were set by Maksim on 2026-06-10/11; items marked **[OPEN]** still need
a call before launch.*

---

## 1. Purpose and the claims this design buys

The pilots established the instrument; the main experiment buys **power and scope** for
four claims the pilot can only gesture at:

1. **Theme-level selection claims** — "models know what predicts institutional trust but
   not social attitudes" (or wherever the pattern falls). Requires targets stratified by
   theme, not by response cardinality. **[USER: solid theme representation]**
2. **A powered adaptation test (T2)** — the pilot's free-text hint (Kimi
   +0.023 [0.001, 0.045], DeepSeek null) decided at MDE ≈ 0.02, not 0.07.
3. **A scaling/zoo claim** — does *selection* capability scale with model
   size/family/reasoning-training, where Paper-1 found *prediction* didn't (model choice
   explained ~1.3% of variance)?
4. **A powered behavioural layer** — base-rate-seeking and methodology-request rates by
   region × theme × model (the marginal-not-conditional fingerprint).

Plus one credibility deliverable: the **mapping-validation study** (human-coded), which
converts "conditional on mapping quality" into a measured number.

## 2. What the pilots locked (not revisited here)

- **Free-text elicitation** (JSON suppresses breadth: −0.05..−0.07 captured importance,
  both models). Fixed-k reporting (k = 5, 10 + model-k) always.
- **Extraction fixed and capable** (Qwen3-235B-Instruct-2507); extraction, not
  disambiguation, is the mapper bottleneck.
- **Cheap disambiguator** (Nemotron-3-Nano; large swap moves ≤0.02).
- **4-way feature typing**; respondent + temporal piped to scoring; methodology +
  base-rate kept as behavioural metadata.
- **Dual-layer mapping** (parent + `sub_items` when `|S| ≥ 2`) is the locked map
  protocol for confirmatory runs — see §4. Pilot parent-only MiniLM arm-C remains the
  continuity baseline / ablation; kimi v1 showed collapse is real (~29% parents bundled,
  parent↔subitem Jaccard ~0.16, recovery when parent is `none` in ~half of bundled cells)
  while matched-k VoR barely moves — so expansion is about **faithful construct coverage**,
  not a secret quality lift (`paper/memos/subitem_mapping_results.md`).
- **Oracle** = AutoGluon permutation importance (single split, test 0.2, 5 shuffle sets);
  downstream evaluator = fixed XGBoost, 5-fold CV, matched-k.
- **Leakage + degeneracy screen** runs as a *design-stage* gate (pilot: 6 leakage + 31
  degenerate of 89 — the quota-based target selection was the cause of the degenerate
  mass).
- Both prompt conditions (`unprompted`, `country_provided`).

## 3. Grid construction (the "careful and with purpose" part)

### 3.1 Targets: theme-stratified, signal-screened

The metadata already harmonises all six surveys into the same 8 sections:
`institutional_trust, political_attitudes, social_attitudes, political_participation,
values_identity, contemporary_issues, wellbeing, demographics`.

- **1 target per substantive theme per survey = 7 × 6 = 42 targets.** Demographics are
  excluded as *targets* (they were the pilot's leakage/degeneracy offenders and clash
  with the attitudes framing) but remain as predictors.
- **Selection within a theme slot is empirical, not quota-based.** For each survey ×
  theme, score every candidate variable (the inventory already has ~238–600 candidates
  per survey) on a pre-screen country sample with a cheap-preset oracle, and pick by:
  1. **Signal**: oracle accuracy lift over majority baseline ≥ 0.05 (kills degeneracy);
  2. **Leakage**: single-feature recovery test passes (the B1 audit, now upstream);
  3. **Cross-national variation**: low mean pairwise rank-correlation / top-k Jaccard of
     the oracle ranking across pre-screen countries (so T2 has something to detect —
     adaptation is zero *by construction* on targets whose structure is identical
     everywhere);
  4. **Support**: ≥ 300 post-cleaning respondents in every selected country;
  5. Tie-break toward attitude/opinion items over behaviours.
- The chosen 42 targets + their oracle screening stats are frozen in a **pre-registered
  manifest** (`prelim/main_manifest.yaml`) *before any selector model runs*.
- If a theme slot has no candidate passing (1)–(4) in some survey, the slot stays empty
  and the manifest records why (do not back-fill with a weak target).

### 3.2 Countries: bridge-first, excessive on purpose **[USER]**

Measured availability (n ≥ 300/cell) and overlap (WVS is the only bridge; regional
surveys never overlap each other):

| Survey | available | WVS overlap | plan |
|---|---|---|---|
| WVS | 65 | — | all 32 bridge countries + ~6 global-spread (USA, India, Japan, Egypt, Brazil, Canada…) ≈ **38** |
| Afrobarometer | 39 | 6 (Ethiopia, Kenya, Morocco, Nigeria, Tunisia, Zimbabwe) | 6 bridges + ~7 spread ≈ **13** |
| Arab Barometer | 8 | 5 (Iraq, Jordan, Lebanon, Morocco, Tunisia) | **all 8** |
| Asian Barometer | 9 | 6 (Australia, Indonesia, Mongolia, Philippines, Thailand, Vietnam) | **all 9** |
| Latinobarómetro | 17 | 10 | **all 17** |
| ESS W11 | 30 | 7 (Cyprus, Germany, Greece, Netherlands, Serbia, Slovakia, Ukraine) | 7 bridges + ~6 spread ≈ **13** |

- Every regional survey **includes all of its WVS-bridge countries**; WVS includes the
  union of all bridges. → ~35 country-survey pairs where the same country appears in two
  surveys, **Morocco and Tunisia in three**. This breaks the perfect country→survey
  correlation and enables the **within-country, cross-instrument replication check**
  (§6.7): same country, same theme, different questionnaire.
- Estimated grid: 7 targets × per-survey C ≈ **650 candidate country-cells → ~550–600
  genuine after screening attrition**.

### 3.3 Power (from pilot variance, conservative)

Adaptation variance is ~entirely within-target (SD 0.167; between-target SD ≈ 0), so
power scales with total country-cells N: MDE ≈ 2.8 × 0.167/√N (80% power, α=.05).

| N country-cells | MDE |
|---|---|
| 42 (pilot) | 0.073 |
| 240 | 0.030 |
| 400 | 0.023 |
| **600 (this design)** | **0.019** |

Two conservatisms: with ~10+ sibling countries the cross-country mean is far less noisy
than at the pilot's 2.6 (within-SD shrinks), and variation-screened targets should carry
a larger true effect. Both push the realised MDE below 0.019.

## 4. Elicitation and mapping protocol

- Free-text prompts (the pilot-validated pair: JSON prompt minus formatting block), both
  conditions, temperature 0, one generation per cell-condition.
- **Stability sub-study**: 5% stratified subsample re-generated 3× at t = 0.7 →
  within-model selection variance (cheap; closes a critique).
- Extraction: Qwen3-235B-Instruct-2507, typed 4-way (including `sub_items` under each
  parent). **Extractor audit [USER]**: ~10% stratified subsample re-extracted by a
  second extractor (Kimi-K2.6); report feature count/type/mapped-code agreement *and*
  sub_item-list agreement. This bounds extractor bias for *all* selectors and
  specifically clears the Qwen-family selectors (same-family extraction concern).
- **Mapping unit (locked) — dual-layer:** free-text often names a broad *construct*
  measurable by several survey variables, not one. For each piped parent feature with
  label `L`, context `C`, sub_items `S`:
  1. **Parent unit** (always): retrieve + disambiguate on `(L, C)`.
  2. **Sub_item units** (only if `|S| ≥ 2`): for each `s ∈ S`, retrieve + disambiguate
     with `feature_label = s` and parent-anchored context
     `"{C} (sub-measure of {L})"` (fallback `"sub-measure of {L}"` if `C` empty).
  3. **Score codes** = `expanded_codes` = deduped union of parent + sub_item codes
     (parents first, then sub_items). Store `parent_codes` / `subitem_codes` for
     diagnostics and the parent-only ablation.
  4. Features with `|S| ≤ 1` contribute parent units only (no singleton inflation).
  Protocol detail and conditional diagnostics:
  [subitem_mapping.md](subitem_mapping.md). Implementation today lives in
  `survey_features.subitem_map`; confirmatory `run_main.py` must use this path as the
  default map (parent-only remains an ablation write, not the headline).
- Retrieval: per **mapping unit** top-20, dual-embed (all-MiniLM-L6-v2, cached per
  survey). Call volume ≈ 1.3–1.9× parent-only (kimi v1: ~1.87× if parents remapped;
  ~0.8× *new* API if parents were copied — production remaps both for a clean
  checkpoint).
- Disambiguation: Nemotron-3-Nano (primary), one call per mapping unit. Qwen-235B
  re-disambiguation on a 10% subsample of units as a continuing robustness check.
- Scoring: matched-k XGBoost (5-fold) on **`expanded_codes`**, captured importance vs
  oracle, random baselines. Always report model-k **and** fixed k = 5, 10 (expansion
  inflates natural k; fixed-k keeps the capability estimate comparable). Parent-only
  scores are a secondary ablation, not the lead row.
  **Random accuracy draws cut to 5** (pilot used 20/10; the baseline mean is stable and
  this is the scoring-cost driver); random *captured-importance* baseline stays at 200
  draws (oracle-arithmetic, free).

## 5. Model zoo **[USER: open + closed frontier]**

Selector ≠ extractor ≠ disambiguator throughout (Qwen-family exception audited, §4).

**Open (Nebius, keys already working) — scale × family curve:**

| Tier | Models |
|---|---|
| Frontier-class | DeepSeek-V4-Pro, Kimi-K2.6, Qwen3.5-397B, GLM-5, Nemotron-3-Ultra-550B |
| Pilot anchors (continuity) | DeepSeek-V3.2, Kimi-K2.5 |
| Mid | MiniMax-M2.5, gpt-oss-120b, Llama-3.3-70B |
| Small | Qwen3-30B-A3B, gemma-3-27b |
| Reasoning-trained (open) | Qwen3-Next-80B-Thinking |

**Closed frontier [OPEN: needs API keys + budget sign-off]:** GPT-5.x, Claude (Opus 4.8
or Sonnet 4.6), Gemini 3. All three are callable through the already-installed `openai`
client via their OpenAI-compatible endpoints — no PyPI access needed (Oxford network
blocks it). Keys go in `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).

≈ 13–16 selectors. Within-family scale pairs (DeepSeek V3.2→V4, Kimi 2.5→2.6, Qwen
30B→397B) + a reasoning-vs-instruct contrast give the scaling analysis real structure.
**Run in waves**: anchors + one frontier + one small first (validates infra at scale),
then the rest. Final zoo pinned in the manifest before launch.

## 6. Tests and analysis plan (pre-registered before the zoo runs)

1. **T1 selection quality**: captured importance + Δ vs matched-k random (the lead
   metric), value-over-random, beat-random share; at model-k and fixed k = 5, 10;
   cluster-bootstrap CIs (survey × target clusters). Reported per model × theme ×
   region.
2. **T2 adaptation**: own − cross captured importance (country_provided), per model,
   with the §3.3 power. Heterogeneity by theme and by the manifest's measured
   cross-national-variation score (prediction: adaptation concentrates where variation
   exists). Movement (Jaccard) reported alongside so "movement without fit" stays
   testable.
3. **T3 complexity calibration** (first run ever): does model-chosen k (on
   `expanded_codes`) track target complexity (outcome entropy, oracle ceiling,
   #features-to-90%-mass)? Free text + dual-layer give a natural k; this completes the
   original design. Report bundling rate and k-inflation alongside.
4. **Scaling analysis**: T1/T2 metrics vs parameter count / family / reasoning-training /
   open-vs-closed. The pointed question: does selection scale where prediction didn't.
5. **Behavioural layer**: typed-request rates (base-rate, methodology, temporal) by
   region × theme × model; powered test of the pilot's ESS-vs-ArabBarometer skew.
6. **Mapping-validation study**: ~300 stratified **mapping-unit**→variable pairs
   (parents *and* sub_items; across surveys, selectors, mapped/none), 2–3 human coders,
   agreement (κ), retrieval recall@20, and the "none" decomposition (coverage gap /
   retrieval miss / over-caution). Stratify so bundled constructs are not under-sampled.
   Produces the attenuation estimate that turns "lower bound" into "bounded".
7. **Cross-instrument replication**: for bridge countries, compare a model's per-theme
   captured importance and request overlap across the two surveys covering the same
   country (Morocco/Tunisia: three). Instrument-robustness of the capability estimate —
   no other design gets this for free.
8. **Parent-only ablation**: same cells scored on `parent_codes` only — reports how much
   one-to-one collapse attenuates the dual-layer headline (continuity with pilot arm-C).

Exclusion rules, metric definitions, and CI procedure frozen in the manifest +
`analysis/main_analysis_plan.md` before wave 1.

## 7. Infrastructure deltas (build list)

1. **Dual-layer default in `run_main.py`**: map phase calls
   `map_features_with_subitems` (or equivalent); score on `expanded_codes`; write
   parent ablation scores under the same run tag. Do not leave dual-layer only in
   `scripts/run_subitem_mapping.py`.
2. **Screening pipeline** (`analysis/target_screening.py`): candidate scoring per theme
   slot — cheap-preset oracle on pre-screen countries, signal/leakage/variation stats →
   `prelim/main_manifest.yaml`. (Empirical leakage guard lives here now, upstream.)
3. **Orchestrator**: manifest-driven `run_main.py` — free-text only, dual-layer map,
   N selectors, closed-model client wrappers, per-cell checkpoints.
4. **Scoring at scale**: multi-selector single pass with shared per-(cell,k)
   oracle/random caches (cache hit across selectors); random accuracy draws = 5;
   incremental flush; resumable. Natural k is larger under expansion — shared caches
   keyed by sorted code-set hash still apply.
5. **Compute budget (this machine, background) [USER]**: screening oracles ~14–18h;
   final oracles ~600 × 3min ≈ 30h; API phases: days, rate-limit bound — dual-layer
   adds ~0.3–0.9× disambig volume vs parent-only (still dominated by cheap Nemotron);
   scoring ≈ 600 cells × shared-k baseline (~4 days) + ~5–7h per selector →
   **~1.5–2 weeks wall-clock background for the full zoo**, in waves.
   Mitigations if painful: drop to draws=3, score wave-1 selectors first, or move
   scoring to a Linux box later (everything is CSV-resumable).

## 8. Decision log

- **[USER 2026-06-10]** Free-text is the instrument; JSON = appendix lower bound.
- **[USER 2026-06-11]** Scale: maximal-leaning, built purposefully → theme-stratified 42
  targets, bridge-first ~600 cells.
- **[USER 2026-06-11]** Zoo: open + closed frontier.
- **[USER 2026-06-11]** Extractor: keep Qwen fixed, include Qwen selectors, 10%
  dual-extractor audit.
- **[USER 2026-06-11]** Compute: this machine, background, waves.
- **[USER 2026-07-31]** Dual-layer mapping (parent + `sub_items` when `|S| ≥ 2`) is the
  locked map protocol; score on `expanded_codes`; parent-only is ablation. Rationale:
  elicited features are often multi-indicator constructs, not single survey variables.
- **[OPEN]** Closed-model keys + budget ceiling.
- **[OPEN]** Final zoo list + wave assignment (pin in manifest).
- **[OPEN]** Mapping-validation coders (who annotates besides Maksim?) and timing
  (recommend: run on wave-1 outputs while later waves generate; include sub_item units).
- **[OPEN]** Exact WVS global-spread picks and Afro/ESS non-bridge picks (propose at
  screening time, with support stats in hand).

## 9. Sequencing

Ordered cheapest / fastest → hardest (detail in the planning note below the freeze).
Summary critical path:

1. **Docs + cheap gates** — dual-layer locked (this doc); unit tests; offline
   similarity CDF; sync experiments_index / cost estimate for dual-layer call volume.
2. **Wire dual-layer into `run_main.py`** — smoke on 2 genuine cells × 1 selector;
   parent ablation scores written alongside.
3. Build + run **target screening** → draft manifest (targets, countries, stats).
4. Review manifest together; freeze it + `analysis/main_analysis_plan.md`
   (pre-registration moment; dual-layer + expanded-k metrics named explicitly).
5. Final oracles for the frozen grid (background, ~30h).
6. Orchestrator hardening + closed-model wrappers; end-to-end smoke 2 cells × 3
   selectors (dual-layer map + score).
7. **Wave 1**: anchors (DeepSeek-V3.2, Kimi-K2.5) + DeepSeek-V4-Pro + gemma-3-27b →
   full T1/T2 on 4 models; mapping-validation annotation starts on these outputs
   (parents + sub_items).
8. Waves 2–3: remaining open + closed models; stability + extractor-audit substudies.
9. Analysis per the frozen plan; paper v2.
