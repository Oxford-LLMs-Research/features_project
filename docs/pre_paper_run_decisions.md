# Pre–paper-run decisions

Status after analyzing the 12-cell stack pilots (2026-08-11). Digests:
`python scripts/analyze_stack_experiments.py` → `outputs/experiments/_analysis/`.

Pilot n is small (≈20–24 model-k rows per arm); treat deltas as go/no-go signals,
not confirmatory effect sizes. **Do not decide from mean Δ alone** — registry
**Contrast detail** + `outputs/experiments/_analysis/registry_contrast_blocks.md`
report median, share Δ>0, PI↔VoR concordance, and by-condition / by-survey strata.

---

## Stack locks (from experiments)

| Knob | Decision | Why |
|------|----------|-----|
| **System prompt** | Keep `"You are a social science researcher."` (`social_scientist`) | `helpful` wins mean PI+VoR on **kimi** only (80% PI>0 but 9/20 PI↔VoR conflicts); on **deepseek_v4** mean PI −0.045 with only 43% PI>0. `none` does not win both endpoints on both selectors. |
| **Extractor** | Keep Qwen default; MiniMax OK as capacity alternate | [`extract-swap-minimax`](experiments_registry.md#extract-swap-minimax--minimax-extract--nemotron-disambig): mean Δ PI +0.013 / VoR ≈0 but only 42% pairs PI>0 (WVS pulls the mean); MiniMax extract still ~$0.13 vs ~$0.02. |
| **Disambiguator** | Keep Nemotron | Flash-as-disambig slow/costly; mean PI −0.096 masks near-50/50 share>0; no Qwen+Flash follow-up. |
| **Instrument / oracle / embedder** | Unchanged | Free-text Arm C + dual-layer maps; era-3 oracles only; MiniLM default. |

No confirmatory `config.py` / `prompts.py` defaults need changing for these knobs.

Full stratum tables live in the registry entries for `prompt-sensitivity`, `pipeline-role-swap`, and `extract-swap-minimax`.

### Prompt-sensitivity snapshot (`k_spec=model`)

Paired Δ vs `social_scientist` (mean | median | share>0):

| Selector | Arm | PI mean/med/>0 | VoR mean/med/>0 | both+/both−/conflict |
|----------|-----|----------------|-----------------|----------------------|
| kimi | none | −0.059 / +0.008 / 55% | −0.003 / −0.000 / 50% | 8/7/5 |
| kimi | helpful | +0.055 / +0.039 / 80% | +0.014 / +0.011 / 65% | 10/1/9 |
| deepseek_v4 | none | −0.010 / +0.003 / 52% | +0.011 / +0.000 / 48% | 8/7/7 |
| deepseek_v4 | helpful | −0.045 / −0.004 / 43% | +0.004 / −0.008 / 39% | 5/9/9 |

Map Jaccard vs scientist maps ≈0.35 (arms rewrite feature sets substantially).

### Role-swap / extract-swap snapshot

| Contrast | PI mean/med/>0 | VoR mean/med/>0 | both+/both−/conflict | Map Jaccard |
|----------|----------------|-----------------|----------------------|-------------|
| MiniMax+Flash − Qwen+Nemotron | −0.096 / 0 / 52% | −0.018 / +0.003 / 52% | 8/7/6 | 0.44 |
| MiniMax+Nemotron − Qwen+Nemotron | +0.013 / 0 / 42% | −0.000 / +0.004 / 63% | 7/5/5 | 0.52 |

---

## Ranked backlog before the confirmatory paper run

1. **Freeze genuine grid (42 vs ~47)**  
   Live `outputs/cache/audits/leakage_audit.csv`: genuine=42, degenerate=34, leakage=10, leakage_distributed=3. Older docs cite ~47. Either re-run `scripts/leakage_audit.py` after confirming era-3 oracle coverage, or explicitly accept **42** as the confirmatory grid and update onboarding/audit notes so zoo sweeps do not mix stories.

2. **Lock the selector zoo**  
   `config.SELECTORS` today is only `deepseek` (V4-Pro) + `kimi` (K2.6). Registry `confirmatory-zoo` is still design-only. Decide the locked list (IDs on Nebius Studio / Token Factory), register keys in `config.SELECTORS`, then sweep with fixed Qwen+Nemotron + `social_scientist`.

3. **Optional: similarity / `top_n` sweep**  
   `similarity-threshold` remains design-only. Map Jaccard across prompt arms is already low (~0.35), so retrieval caps still matter for capability ceilings — but this is **not** required to start a first zoo pass if you accept MiniLM + current `min_similarity=0.30` / `top_n=20` as a known floor.

4. **Confirmatory zoo mechanics**  
   Per selector: `scripts/run_main.py --phase pipeline --with-score` (or phased gen→extract→map→score) → `outputs/main/scores_<selector>.csv`. Concurrent workers proven on the pilots. Wire usage/cost logs (already in `run_main`).

5. **After first zoo sweep (paper, not stack)**  
   Multiplicity hierarchy (model-k primary), dual-extractor audit (~10%), Test-2 / construct-level importance — keep post first full scores.

---

## Explicit non-goals right now

- Expanding the 12-cell prompt-sensitivity factorial
- Promoting MiniMax or Flash into confirmatory defaults
- Switching the default system prompt to `helpful` on kimi-only evidence
- Rebuilding era-3 oracles
