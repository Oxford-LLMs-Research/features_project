# Second Pilot — Output Format & Fixed-k Findings

> **Numbers superseded (2026-08): computed on the era-1 accuracy oracle.** The
> qualitative finding stands — the JSON format suppressed measured capability, mostly
> via breadth — but every number here predates the log-loss, measurement-level oracle
> (contract v3) and the arrival-order caveat on fixed-k rows. Do not quote figures from
> this memo; see `pipeline_audit_2026-08.md` §A6 and `experiments_index.md` →
> "Which numbers are current".

> **Status (2026-06-10):** the paper now *leads* with the free-text (arm C) results —
> `analysis/freetext_main_results.py` + `analysis/freetext_figures.py` generate the
> main-text tables/figures (ft_*.tex, fig_ft_*) from this pilot's arm-C artifacts; the
> JSON-constrained results and this format experiment are appendix material in
> `paper/prelim_paper.tex`. Notable: on free text Kimi's adaptation score turns positive
> with a CI excluding zero (+0.023 [0.001, 0.045], robust to disambiguator); DeepSeek
> stays null. See `outputs/format_pilot/freetext_main_summary.json`.

*Generated 2026-06-04 from `analysis/format_pilot.py` (phases gen/extract/map/score).
Per-cell artifacts under `outputs/format_pilot/`; scored rows in
`outputs/format_pilot/scores.csv` (1{,}545 rows, 6 errors). All numbers read back from
that CSV; paired contrasts use cells present in both arms.*

## Why this pilot

The pipeline audit flagged the strict-JSON output contract as the top threat to the
central claim: if models know more than the list format lets them say, our "shallow
knowledge" result would be measuring the instrument, not the model. This pilot tests that
directly, holding everything else fixed and varying only the output format, with fixed-k
reporting so a format effect cannot hide inside a larger list.

## Design

Selector (test model) = DeepSeek-V3.2. Grid = the 52 genuine cells (B1 leakage audit),
both prompt conditions. Three arms decompose two pipeline choices:

- **A** = JSON selection + pilot-1 mapper (per-feature top-5 disambiguation).
- **B** = JSON selection + new mapper (per-feature top-20, fixed Qwen-235B extractor).
- **C** = free-text selection + the same new mapper.

The free-text prompt is the JSON prompt with **only** the JSON-formatting block removed
(verified character-identical stem), so C vs B isolates output format. A two-step mapper
(fixed Qwen extractor → per-feature retrieval → disambiguation) is held constant across
arms; the disambiguator is run as both **Nemotron (small)** and **Qwen-235B (large)** to
test mapper strength. Each arm is scored at model-chosen k and at fixed k = 5, 10.
Contrasts: **C−B = format**, **B−A = mapper architecture**, **Qwen−Nemotron = mapper
strength**.

Free-text features are typed (respondent_attribute / temporal_contextual /
instrument_methodology / base_rate_prior); only respondent + temporal enter scoring
(decision: the study spans countries and time). Methodology and base-rate requests are
held out and analysed as behaviour (below).

## Two-model update (2026-06-08): the format effect replicates in Kimi-K2.5

The pilot was rerun with Kimi-K2.5 as the selector (same grid, pipeline, fixed Qwen
extractor; scoring used 10 random draws vs DeepSeek's 20 — captured importance is
draw-independent so the headline is directly comparable). Format effect (free text − JSON,
paired captured importance):

| Budget | DeepSeek | Kimi |
|---|---|---|
| model-chosen $k$ | +0.074 | +0.054 |
| $k=10$ | +0.049 | +0.039 |
| $k=5$ | −0.008 | +0.025 |

**The effect is model-general** (same direction/rough magnitude in both). The breadth story
also replicates — it shrinks as $k$ is capped — but with a nuance: **for Kimi it does not
fully collapse at $k=5$** (+0.025), so Kimi has a small genuine per-item-quality component on
top of breadth, whereas DeepSeek's is pure breadth. Defensible general claim: predominantly
breadth, with a modest model-dependent quality component. Mapper strength and mapper
architecture remained negligible for both models. Means at model-$k$: DeepSeek A/B/C captured
importance 0.193/0.182/0.253 (k 5.3/5.5/10.2); Kimi 0.199/0.201/0.251 (k 5.7/6.2/9.9).

## Headline: format matters, mainly through breadth (DeepSeek detail below)

Means at model-chosen k (captured importance = share of oracle top-k importance mass the
mapped features recover):

| Arm | Disambiguator | Captured imp. | Value/random | Model acc | Mean k |
|---|---|---|---|---|---|
| A (JSON, old mapper) | — | 0.193 | 0.038 | 0.451 | 5.3 |
| B (JSON, new mapper) | Nemotron | 0.182 | 0.042 | 0.456 | 5.5 |
| B (JSON, new mapper) | Qwen-235B | 0.171 | 0.033 | 0.444 | 5.6 |
| **C (free-text)** | Nemotron | **0.253** | 0.063 | 0.484 | 10.2 |
| **C (free-text)** | Qwen-235B | **0.245** | 0.062 | 0.479 | 10.1 |

**Format effect (paired C−B), captured importance:**

| k | Nemotron | Qwen-235B |
|---|---|---|
| model-chosen | **+0.074** | **+0.070** |
| k=10 | +0.049 | +0.062 |
| k=5 | −0.008 | +0.011 |

At the models' natural output length, free-text recovers ~0.07 more oracle-importance mass
than JSON — about 38% more (0.25 vs 0.18) — and this replicates across both disambiguators
and **all six surveys** (range +0.029 Afrobarometer to +0.146 Latinobarómetro; positive
everywhere) and both prompt conditions (+0.069 country-provided, +0.078 unprompted). The
same pattern holds for value-over-random (+0.022 / +0.029 at model-k). **So the strict JSON
format was suppressing measured capability — the audit hypothesis is confirmed.**

## The honest decomposition: it is breadth, not per-item quality

The fixed-k contrasts show *why* free-text wins, and it is not that its top features are
better aligned one-for-one:

- At **k=5** the format effect is essentially zero (−0.008 / +0.011).
- At **k=10** it is partial (+0.049 / +0.062).
- Free-text volunteers **~10 features vs JSON's ~5.5** — nearly double.

In other words, when both formats are capped to their first five requests, JSON's tightest
five are about as oracle-aligned as free-text's first five; free-text's advantage comes
from **volunteering roughly twice as many relevant predictors**. The capability JSON hides
is *breadth* — how much of what it knows the model offers — not per-feature correctness.
This is a real measurement-format effect with direct consequences (any study that elicits
a short structured list will under-measure model knowledge), but it should be stated as a
breadth effect, not as "JSON makes the model wrong." Reporting only model-chosen k would
have over-claimed; the fixed-k view is what disciplines the interpretation.

## Two secondary contrasts: both ~null (and both reassuring)

- **Mapper strength (Qwen − Nemotron disambiguator): ≈0** across all arms/k (−0.016 to
  +0.011). Once extraction is fixed and retrieval supplies a good top-20, the
  disambiguation step is insensitive to disambiguator size. The mapper-ceiling problem
  seen earlier lived in *extraction* (which we fixed by holding Qwen constant), not
  disambiguation. Practical upshot: the cheap disambiguator suffices; the main experiment
  need not pay for a large one.
- **Mapper architecture (B − A, new top-20 vs pilot-1 top-5, JSON both): ≈0** (−0.011 to
  −0.016). The richer per-feature retrieval did not beat pilot-1's simpler mapper on the
  same JSON inputs — so pilot-1's mapping was not the bottleneck, which retroactively
  supports the validity of pilot-1's numbers.

Net: of the pipeline choices tested, **only output format moved the result.**

## Behavioural finding: base-rate-seeking (the mode-collapse fingerprint)

The free-text extraction typed each requested feature. Beyond respondent attributes
(87.6% of 1{,}927 features), the model sometimes asked for things that are not respondent
attributes at all: instrument/methodology commentary (3.7%), temporal/contextual factors
(7.4%), and — most tellingly — **base-rate or modal-response information (1.2%, appearing
in 19% of cell-conditions)**: e.g. "national statistics", "actual national attitudes",
"baseline prediction by age group". Asked what it needs to predict an *individual*, the
model sometimes reaches for the *population marginal* — the exact conditional-vs-marginal
failure documented in Paper 1, now visible in the model's own stated reasoning (cf.
`paper/IIA assumption.tex`). It is **skewed**: present in Arab Barometer and Afrobarometer,
**absent in ESS (0%)** — directionally consistent with weaker individual-level priors in
non-Western contexts. With only 24 such features this is descriptive and underpowered; we
report it as a behaviour and flag the cross-cultural pattern as a hypothesis for the main
experiment.

## Implications for the main experiment

1. **Elicit in free text, not constrained JSON** (then extract to a list with a fixed
   capable model). Constrained list formats under-measure capability via breadth.
2. **Always report fixed-k alongside model-chosen k** — the headline capability number is
   sensitive to elicited list length, which is a format artifact.
3. **A small disambiguator is adequate**; spend compute on the selector models and the
   extractor, not the disambiguator.
4. **Type free-text requests** and report the base-rate-seeking rate as a behavioural
   measure; power the cross-cultural comparison with more countries/models.

## Reproduce

```
python analysis/format_pilot.py --phase gen        # free-text (DeepSeek), cached
python analysis/format_pilot.py --phase extract    # fixed Qwen extractor -> typed features
python analysis/format_pilot.py --phase map --disambiguator nemotron
python analysis/format_pilot.py --phase map --disambiguator qwen235b
python analysis/format_pilot.py --phase score      # -> outputs/format_pilot/scores.csv
```
