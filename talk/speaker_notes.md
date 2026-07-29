# Speaker notes — 15-minute project talk

**Deck:** `slides.tex` → `slides.pdf` (metropolis, 16:9).
**Balance:** equal weight Paper 1 / Paper 2; mixed social-science + ML audience.
**Total budget ≈ 15 min.** Times below are *cumulative targets* — if you're past
the marker, cut the sub-bullets, not the headline number.

Numbers are read from the generated tables (`paper/generated_current_state/ft_*.tex`)
and the EMNLP source (`paper.tex`); don't quote anything not on a slide without
checking those.

---

## Motivation (~3:00)

**S1 — Where we are going** *(→0:30)*
- One sentence: "I'll show two studies on using LLMs as synthetic survey
  respondents — one asks *can they*, one asks *why not*."
- Don't read the box; it's a signpost.

**S2 — The promise** *(→1:15)*
- Sell the appeal honestly: this is a real and growing literature, and the upside
  (cost/access/ethics) is genuine.
- Name the startups to raise the stakes — this isn't just academic: **Aaru**
  (population simulation, forecasting elections/policy; ~$1B valuation) and
  **Simile** ("agentic twins" of real people for polling; CVS/Gallup; $100M Series A).
  Nice hook if asked: Simile's founders (Park, Bernstein, Liang) are the Stanford
  *generative-agents* team — the same lineage as the literature this work tests.
  Links are clickable (orange) if you want to open one live.
- Land the alertblock: "real money is riding on this — but almost all the evidence is
  *aggregate*, and a sample is made of people."

**S3 — Why individual-level is the right bar** *(→2:15)*
- This slide answers the obvious pushback (printed on it): *"isn't the individual just
  noise — aren't the aggregates what matter?"* Take it head-on.
- The one conceptual move of the whole talk. Say it slowly:
  *get every individual right and you're automatically right about any subgroup —
  even ones you never thought to test. The reverse isn't true.*
- The killer line (orange, bottom): a model that matches the marginal but misses the
  individual has **only echoed a base rate you already had** — so "aggregate is enough"
  defeats the point of building a silicon sample at all (you query it for the un-pre-
  specified cross-tabs). That's the conditional structure, and it's what both papers test.

**S4 — Two questions** *(→3:00)*
- Set up the spine: Paper 1 = capability; Paper 2 = diagnosis (selection vs use,
  and adaptation). "Same six surveys, two lenses."

## Paper 1 — capability (~4:30, →7:30)

**S5 — Design** *(→4:30)*
- Emphasise *realistic conditions* = the contribution. Three fixes; mention the
  XGBoost baseline — it's the punchline setup.
- **Spend your time on the "Why random?" block** — it's the most-challenged choice.
  The argument: *which features matter is the unknown.* At design time nobody knows it
  (it's literally what Paper 2 then studies). Curating "relevant" demographics does the
  model's feature-selection job for it → you're testing easy **cue recognition** in a
  best case, not the deployment task. Every prior eval curates, so the whole literature
  runs optimistic. Random also keeps LLM vs. XGBoost on *identical* inputs (fair) and
  keeps the test independent of our own beliefs about relevance (no circularity).
- If pushed "isn't random unfair to the model?": that's the point — it's the honest
  *expectation over feature subsets*, and the result is therefore a floor, not a worst case.

**S6 — Headline** *(→6:00)* — **the money slide, give it time**
- Walk the three numbers: best LLM 0.161; a plain tree on the *same* features
  0.345; majority 0.368.
- The point is the *comparison*, not the absolute: ">2× the signal from identical
  information → the bottleneck is conditional interaction, not missing info or
  scale." Point at the dash-dot (XGBoost) vs the cloud of models below it.
- **Define VR here** (it's the x-axis): variance ratio = the model's response
  diversity ÷ humans'. <1 = the model flattens / under-disperses; =1 = human-level.
  You'll lean on it again on the next slide.

**S7 — Two blades** *(→6:50)*
- Quick. Mode collapse (every model less diverse than humans; beats the
  *within-country* mode only 3–8% of the time) and no scaling (170× range, 2.8%
  of variance). "Two independent ways the capability is shallow, not just low."

**S7b — More information helps; diversity doesn't** *(→7:20)* — the profile-richness figure
- Quick visual. Left→right = sparse/medium/rich (6/12/24 features).
- Panel (a): accuracy climbs ~linearly (0.061→0.102) — they *do* use extra info.
- Panel (b): VR stays <1 at every level — **mode collapse is not an information
  problem**; more data doesn't restore diversity. That's the line to land.

**S8 — Geography** *(→7:30)*
- Hierarchy is mostly *compositional* (resist "the model is biased" oversimpls).
- Conditional stereotyping is the bridge: naming the country helps sparse regions,
  hurts strong-prior ones → *does it actually adapt, or just apply a prior?* —
  hand straight to Paper 2.

**S8b — Paper 1: takeaways & implications** *(→8:30)* — the Paper 1 closer
- Land three things: (1) signal but shallow, 2× gap = conditional interaction, won't
  scale away; (2) implications for silicon samples — they flatten people, worst in the
  Global South, country-conditioning is double-edged → "topline yes, who-believes-what
  no"; (3) the bridge: "predict" conflates *selection* vs *use* → Paper 2.
- This is where the room should feel the stakes for the Aaru/Simile pitch from S2.

## Paper 2 — diagnosis (~4:30, →12:00)

**S9 — The pivot** *(→8:20)*
- "Paper 1 gives the model the features and grades the answer — so it can't tell
  *not knowing which features matter* from *misusing them*." New task: ask the
  model what it would *want to know*, with no data and no options list. It's the
  design-stage question.

**S10 — The instrument** *(→9:40)*
- Use the analogy, it lands: oracle = answer key (has the data), LLM = student
  with no notes, random = didn't study. "We're not asking if the student beats the
  answer key — we're asking how much of it they already knew."
- Define captured importance in one breath; don't belabour the pipeline boxes.

**S11 — Result 1** *(→11:10)* — **second money slide**
- Table left: recovers ~1.8× a random pick's predictive mass; beats random ~76%;
  Δ-CI excludes zero. "The prior carries real, measurable signal."
- Right: the format finding. JSON output *hid* capability — free text gave ~2×
  more usable features. But honesty: it's mostly *breadth* (effect → 0 at k=5).
  "How you ask is part of what you measure."

**S12 — Result 2** *(→12:00)*
- Test 2: the model *moves* its picks across countries (~70% change) but doesn't
  *fit* the country better — a clean null, one hint of real adaptation (Kimi).
  Stress: *no data-driven importance method can even ask this question.*
- The base-rate tell ties back: the same "give me the average, not this person"
  reflex from Paper 1's mode collapse, now caught in the reasoning.

## Discussion (~3:00, →15:00)

**S13 — Limitations** *(→13:00)*
- Own the pilot's size. Key framing: the mapping pipeline can only *lose* signal,
  so these are a **lower bound** — and the oracle is independent, so every bias is
  conservative. That's the reviewer-proofing line.

**S14 — Synthesis + next** *(→14:30)*
- Put the two papers in one sentence (the block). Practice implications in one
  breath. Then the main experiment: pre-registered, ~600 cells, more models
  (open + closed), wide country set so Test 2 finally has power.

**S15 — Standout** *(→14:50)*
- Say it, pause, stop. "They know a little about *who* matters — much less about
  *how*, or *where*." Then take questions.

---

## Backup slides — which question jumps where
After "Thank you" there are 12 appendix slides with a hyperlinked index (click to jump;
"back to index" button on each). Map the likely question to the slide:

| If they ask about… | Jump to |
|---|---|
| "Isn't the LLM just a worse importance method?" | **LLM vs. ML importance** |
| normalized accuracy / VR / how baselines work (P1) | **Metrics & baselines** |
| the surveys, harmonisation, profile richness (P1) | **Data, harmonisation, profiles** |
| variance decomposition, scaling, perplexity validity | **Decomposition / scaling / validation** |
| which topics fail, the "Don't know" hedging | **Topics & hedging** |
| "walk me through the Paper-2 pipeline" | **Full pipeline (6 steps)** |
| captured importance / adaptation score / matched-k | **Metrics defined** |
| "how do you rule out leakage?" | **Leakage screen** |
| the JSON-vs-free-text experiment design | **Format experiment** |
| "show me a concrete example" | **Concrete example** (stfgov-Austria) |
| the JSON numbers / base-rate behaviour | **JSON lower bound + behaviour** |

## Anticipated questions (prep)
- **"Why not just use the oracle / XGBoost if you have the data?"** — Exactly the
  point: the oracle is the *answer key*. The LLM's value is at the *design stage*,
  before data exists, and for constructs the survey never measured / cross-national
  transfer the oracle can't attempt. (See `docs/framing_and_comparisons.md`.)
- **"Is the weak result just your mapping pipeline?"** — Possibly attenuated, hence
  *lower bound*; the format experiment + leakage screen are exactly the audit that
  bounds this. Free-text arm doubles recovered features.
- **"Closed/frontier models?"** — Paper 1 needs log-probs (open-weight only); the
  170× no-scaling range is the indirect evidence. Closed models are in the main
  experiment via a generation-based variant.
- **"Contamination?"** — Microdata is numeric CSV; aggregate knowledge ≠ knowing
  *this* respondent; contamination would show high cross-model variance — it's 2.8%.
- **"3 countries for Test 2?"** — Acknowledged; underpowered by design; the main
  experiment's bridged country set fixes it.
