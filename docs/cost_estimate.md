# Token & Cost Estimate — Nebius Token Factory grant

> **Superseded (2026-08): unit costs and call volumes are stale.** This predates the
> dual-layer decision, the measured oracle cost (~300 s/cell serial, process pool ~3-4x),
> and the measured scoring structure (~90k XGBoost fits). Use the budget arithmetic in
> the 2026-08 session notes / `pipeline_audit_2026-08.md` §C instead.

*Drafted 2026-06-19 for the inference-provider grant application. Token volumes are
measured from the second-pilot (arm-C, free-text) artifacts; prices are Nebius Token
Factory list prices supplied 2026-06-19. Update the **Parameters** block and re-derive if
the grid, model count, or prices change.*

---

## 1. How this was measured

All per-cell token figures come from the real pilot artifacts under
`outputs/format_pilot/{deepseek,kimi}/` (208 free-text responses, 208 mapped cells),
not from guesses:

- **Selector output** (the essay): avg 2,673 chars (DeepSeek) / 3,435 (Kimi) per response.
- **Extraction output** (typed feature JSON): avg ~950 tokens per condition.
- **Features per cell-condition**: ~18 extracted, ~17 piped into mapping.
- **Disambiguation**: ~17 calls/condition, each re-sending ~12 candidate survey questions
  (avg survey question text = 81 chars, n=13,226 sampled).

Conversion: **~4 characters per token** (English; essays, prompts, and survey texts are
English). Local steps — embeddings (all-MiniLM, cached), the AutoGluon oracle, and the
XGBoost scorer — cost **$0** in API tokens.

The main study runs a **single** free-text elicitation path (no JSON arm), so the pilot's
arm-C numbers *are* the main-study per-cell unit.

---

## 2. Parameters (edit these)

| Parameter | Value | Note |
|---|---|---|
| Cells (country × target) | 600 | **Not yet settled**; grows linearly with targets/countries |
| Prompt conditions | 2 | unprompted, country-provided (folded into per-cell numbers) |
| Selector models | 15 | range 10–20; Nebius-hosted (open) only |
| Paper-2 rounds | 4 | iterative revision rounds |
| Paper-2 subset | full grid | iterative arm may run on all or a subset of cells |
| chars/token | 4 | English approximation |

**Prices used ($/1M tokens, Nebius list, 2026-06-19):**

| Role | Model | In | Out |
|---|---|---|---|
| Extractor (fixed) | Qwen3-235B-A22B-Instruct | 0.20 | 0.60 |
| Disambiguator (fixed) | Nemotron-3-Nano-30B-A3B | 0.06 | 0.24 |
| Selectors (blended mean of 15) | mixed zoo* | 0.58 | 1.95 |
| Trace judge (Paper 2) | mid-tier (e.g. Nemotron-Super-120B / MiniMax-M2.5) | 0.30 | 1.00 |

\*Blended selector zoo = DeepSeek-V4-Pro, Kimi-K2.6, GLM-5.2, Qwen3.5-397B,
Nemotron-Ultra-550B, Hermes-4-405B, MiniMax-M2.5, Nemotron-Super-120B,
Llama-Nemotron-253B, gpt-oss-120b, Qwen3-Next-80B-Thinking, INTELLECT-3, Qwen3-30B,
Gemma-3-27B, Llama-3.3-70B. Closed frontier models (GPT/Claude/Gemini) are **not** on
Nebius and are billed separately — excluded from this ask.

---

## 3. Tokens per cell (one country × target, one model, both conditions)

| Stage | Model | Input tok | Output tok |
|---|---|---|---|
| Elicitation | selector | ~160 | ~1,500 |
| Extraction | Qwen3-235B | ~2,400 | ~1,900 |
| Disambiguation (~35 calls) | Nemotron-30B | ~15,000 | ~70 |
| **Per cell, per model** | | **~17,600** | **~3,470** |

≈ **21k tokens per cell per model.** Disambiguation input dominates the *volume* (re-sends
~12 candidates per feature) but runs on the cheapest model.

---

## 4. Paper 1 — main study (clean single pass)

Grid: 600 cells × 15 models = **9,000 cell-model units.**

| Stage | Model | Tokens (in / out) | Cost |
|---|---|---|---|
| Elicitation | selectors (blended) | 1.4M / 13.5M | $27 |
| Extraction | Qwen3-235B | 21.6M / 17.1M | $15 |
| Disambiguation | Nemotron-30B | 135M / 0.6M | $8 |
| **Total** | | **158M / 31M** | **≈ $50** |

So low because the two high-volume stages run on the cheapest models. Largest single line
is selector *output* ($26), since frontier models charge $3–4.40/M out.

---

## 5. Paper 2 — iterative loop (clean single run, with trace evaluation)

The model is shown how its prior features performed and reasons about why it fell short,
then revises — repeated over rounds. **This arm has had no pilot.**

Per cell-model-round (central estimates):

| Stage | Model | Input tok | Output tok | Note |
|---|---|---|---|---|
| Reasoning elicitation | selector (thinking) | ~2,500 | ~5,000 | feedback context in; reasoning trace + revision out |
| Extraction | Qwen3-235B | ~3,000 | ~1,000 | |
| Disambiguation | Nemotron-30B | ~15,000 | ~70 | |
| Trace evaluation | LLM judge | ~6,000 | ~500 | re-reads the trace at scale (a second pass) |
| **Per cell-model-round** | | **~26,500** | **~6,570** | ~33k tokens |

Full grid × 15 models × 4 rounds = 36,000 cell-model-rounds (≈ 1.2B tokens). Costed at the
prices above ≈ **$650–700** for one clean run. The reasoning-output line and the judge
layer are what make this ~6–7× Paper 1 per pass.

---

## 6. Why the clean numbers are the wrong number to request

A grant must cover the *realistic* cost, not the clean floor — especially for an unpiloted
second arm. The multipliers we expect:

- **Reasoning output is structural, not optional** (Paper 2 requires it): 5–10× the output
  tokens of a normal essay. Already baked into §5; the upper end pushes it higher.
- **No pilot → heavy iteration**: defensive/degenerate model behaviors (e.g. "you can't
  really predict this anyway"), feedback-format engineering, redesigns, reruns. Realistic
  3–5× on Arm 2.
- **Trace-evaluation layer may need multiple passes** (different rubrics, a second judge,
  or human+LLM hybrid) — potentially doubling the judge line.
- **Grid not settled**: more targets/countries scale Paper 1 (and the Paper-2 base)
  linearly.
- **General reruns/exploration** across both arms.

---

## 7. Tiered totals

| Tier | Tokens | Cost | Basis |
|---|---|---|---|
| **Floor — clean single pass** | ~2B | ~$750 | Paper 1 $50 + Paper 2 one clean 4-round run with judge (~$700) |
| **Realistic** | ~3–4B | **~$1,500–1,800** | P1 ×3 (reruns, reasoning selectors, modest grid growth) ≈ $200–300; P2 ×4 (unpiloted iteration, defensive-model reruns, trace-judge layer) ≈ $1,300 |
| **Buffered (recommended ask)** | **~6–8B** | **~$3,000–5,000** | Realistic + headroom for the unforeseen (grid expansion, extra rounds, second judge pass, exploration) |

Paper 2 dominates and carries essentially all the uncertainty; Paper 1 is a rounding error
against it.

---

## 8. Recommended grant ask

> **~6–8 billion tokens (≈ $3,000–5,000).**

Justification (paste-ready): *Arm 1 is pilot-grounded and inexpensive (~$200–300). Arm 2 is
exploratory and unpiloted: it requires reasoning-model generation (5–10× the output
tokens), multiple revision rounds, an LLM-based layer to evaluate reasoning traces at
scale, and substantial iteration to handle unanticipated model behaviors. The request
reflects realistic costs for the second arm plus headroom, since under-provisioning would
stall an unpiloted study mid-experiment.*

Optional split for the form:
- **Core ask (~$2,000):** both papers, minimal iteration, current grid.
- **Extended ask (~$5,000):** full iteration, trace-evaluation layer, grid expansion.

---

## 9. Cost levers (if the budget needs trimming)

1. **Cap reasoning length / use fewer thinking models** in Paper 2 — the dominant cost.
2. **Run Paper 2 on a subset of cells** rather than the full grid (halving cells ≈ halves
   Arm-2 cost).
3. **Cheaper trace judge** (e.g. Qwen3-30B at $0.10/$0.30 instead of a mid-tier model).
4. **Trim disambiguation candidates** (fewer than ~12) or cache the prompt prefix — cuts
   the largest *token* line, though it is already on the cheapest model.
5. **Fewer selector models** — Paper 1 and the Paper-2 base scale linearly with model count.
