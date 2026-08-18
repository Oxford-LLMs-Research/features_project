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

## Grid screen: which (target × country) cells to keep

Decided 2026-08-16. Era-3 oracle ranks on **log-loss** (binary/nominal) or **Spearman** (ordinal). The live leakage audit’s `degenerate` label is **accuracy lift vs majority** (`oracle_acc − majority_baseline < 0.03` in [`scripts/leakage_audit.py`](../scripts/leakage_audit.py)). Those are different claims. For confirmatory cell selection:

| Keep / drop | Name | What it is | Decision |
|-------------|------|------------|----------|
| **Keep as a drop rule** | Type 1 — unestimable PI | Minority class too thin for the honest 60/20/20 split (V1 ranks, V2 values). Modal share alone is not the test (`majority ≥ 0.80` was tried and retracted in [`pipeline_audit_2026-08.md`](pipeline_audit_2026-08.md) §A10 — Q43A is 84–91% majority with the largest log-loss signal in the discard pile). | **Drop** the cell. Floor in code today is only `MIN_CLASS_COUNT = 5` on the *full* cell (~1 minority row on V2). Tighten to minority support **on V1 and V2** (rule of thumb: ≥10 on V2 ⇒ ~50 in the cell). After the oracle, `oracle_ceiling@k` is the check that the ranking replicates; low ceiling (Q141 Andorra `ceiling@5 ≈ 0.24`) means compromised PI. |
| **Do not use** | Type 2 — tiny accuracy lift | Oracle top-k XGB accuracy beats the mode by less than `--min-signal` (0.03), e.g. Q104 Mauritania +0.029, Q43A Angola +0.003. | **Not a reason to drop.** Argmax can sit on the mode while log-loss PI is real (Q43A Angola: 124/206 positive features, `ceiling@10 = 0.69`). |
| **Do not use** | Type 3 — accuracy below the mode | Downstream XGB accuracy of the oracle top-k *loses* to majority, typical of high-cardinality ordinals (P16ST, P61ST). | **Not a reason to drop.** Spearman PI can still rank (P16ST Colombia: 86 positive features, `ceiling@10 = 0.85`). Accuracy-vs-mode is the wrong metric for an ordinal oracle. |

Still drop **leakage** (concentrated near-duplicate; distributed skip-pattern / `Q67A`). Do not drop on accuracy-vs-majority.

**Consequence.** Live `leakage_audit.csv` genuine=42 is an *accuracy-screen* remainder, not the confirmatory lock. Re-grid before the zoo: keep type-1 + leakage drops; restore type-2/3 cells that pass support / ceiling. Do not mix a new grid with the old 42/47 score files.

---

## Ranked backlog before the confirmatory paper run

1. **Re-grid on type-1 + leakage, not accuracy-vs-majority**  
   Live audit genuine=42 used type 2/3 as “degenerate.” Implement the screen above (minority support on V1/V2, then leakage, then `oracle_ceiling` as a PI-quality check). Older docs/scores still say 47. Do not freeze 42.

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
