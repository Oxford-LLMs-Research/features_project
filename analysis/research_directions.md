# Research Directions — main experiment + future arcs

*Drafted 2026-06-04, after the two pilots. Working notes for discussion, not commitments.*

---

## Part 1 — The main/confirmatory experiment (this paper)

What the pilots settled, and what the confirmatory run should therefore be.

### Locked-in design decisions (evidence-based)
- **Elicit in free text, extract to a typed feature list with a fixed capable model.**
  Free text recovers ~38% more oracle importance mass than JSON; the gain is *breadth*.
- **Report fixed-k alongside model-chosen k.** The headline capability number is sensitive
  to elicited list length, which is a format artefact; fixed-k is the honest control.
- **Cheap disambiguator is fine** (mapper strength ~null once extraction is fixed). Spend
  compute on selector models, not the mapper.
- **Type every request** (respondent / temporal / methodology / base-rate). Pipeline
  respondent+temporal; report methodology + base-rate as behaviour.
- **Oracle + matched-k logic + leakage screen are sound** — keep as is.

### What changes from pilot to main
1. **Targets: signal-driven, not cardinality quota.** Select for high non-demographic
   predictable signal + cross-national variation (the original design intent). This removes
   the degenerate dilution (31/89 in pilot) and makes Test 2 meaningful. Pre-screen with
   the oracle; keep an attitude/opinion emphasis to match the "what shapes attitudes" frame.
2. **Countries: widen substantially.** Test 2 (adaptation) was the cleanest null but the
   most under-powered (3 countries/target). Need enough countries per target that the
   own-vs-cross adaptation contrast has power, and enough cross-national *variation* in the
   oracle to have something to adapt to.
3. **Models: a real model zoo.** Pilot had 2; they were near-identical. Span scale, family,
   reasoning-trained vs not, open vs frontier-closed. This is where "does capability scale"
   gets answered. (Paper-1 found model choice explained ~1.3% of prediction variance — a
   pointed question is whether *selection* capability is similarly flat or actually scales.)
4. **Mapping-validation study (the credibility gap).** Annotated request->variable pairs,
   multiple coders, kappa; gives retrieval recall and decomposes the "none" bucket
   (coverage gap vs retrieval miss vs over-caution). Converts "conditional on mapping
   quality" into a measured number. This is the single biggest believability upgrade.
5. **Uncertainty as standard** (cluster bootstrap, already built in B3).
6. **Empirical leakage guard in the oracle** (drop features whose single-feature recovery
   of the target exceeds threshold) so future runs self-protect.

### Tests, restated for the main run
- **T1 Selection quality**: captured importance + oracle percentile vs random baseline,
  fixed-k. Lead metric.
- **T2 Cross-national adaptation**: own-vs-cross captured importance, now powered. The
  signature claim; the one no ML-importance baseline can be posed against.
- **T3 Complexity calibration** (never run): does requested-k track target complexity
  (entropy, oracle ceiling, n-features-to-90%)? Now tractable because free-text gives a
  natural k. Cheap to add, completes the original design.
- **Behavioural layer** (new, from pilot-2): rate of base-rate-seeking and methodology
  requests by region/topic/model — the marginal-not-conditional fingerprint.

### Two framing assets the pilots produced
- The **lower-bound / pipeline-as-lossy-instrument** framing (audit section).
- The **"movement without fit"** characterisation of adaptation (B2).

---

## Part 2 — Future directions (next papers)

### 2A. Iterative self-correction toward scientific discovery  (user's idea)
**The arc:** model proposes predictors -> we show it the realised predictive performance
(and/or which of its features mattered) -> it reasons, reflects, revises -> repeat. Question:
can a model, in the loop with real data feedback, *recover genuine social-science findings*
about a population it was not told about? This reframes the whole programme from "audit what
models statically know" to "use models as discovery engines that learn the conditional
structure from feedback." It is a different and bigger paper.

**Why it's exciting:** it turns the oracle from a static answer key into a teacher, and
tests something no one has cleanly shown — whether LLMs can do *iterative empirical
reasoning* about human populations, not just one-shot recall. If it works, the deliverable
is not "models know X%" but "models + data feedback can rediscover that, e.g., corruption
experience drives institutional trust in West Africa" — a constructive, not just
diagnostic, result.

**The trap I'd flag hardest (leakage of the answer, dressed as discovery):** the moment we
show the model *which features mattered*, "discovery" collapses into "hill-climbing on our
feedback signal." A model that just adds the top oracle features we revealed isn't doing
social science — it's reading the answer key we handed it. The scientific claim requires the
feedback to be *outcome-level, not mechanism-level*: show prediction accuracy (or error on
held-out respondents), NOT the importance ranking. Then improvement must come from the
model's own hypotheses about *why* it underperformed, which is genuine abduction. Design the
feedback to be as information-poor as a real scientist's ("your model got 54%, here are some
cases you got wrong") and see if it can still climb. The gap between mechanism-feedback and
outcome-feedback performance is itself the headline finding.

**Second trap (what counts as "genuine finding"):** we need a held-out notion of truth the
model can't have memorised. Risk: the model "recovers" that education predicts X because
that's in its training corpus, not because it reasoned from feedback. Mitigations: use
cross-national *variation* as the discovery target (the fact that a predictor flips sign or
importance between countries is far less likely to be a memorised fact than a main effect);
and test on recent survey waves / less-documented populations where the finding is unlikely
to be in pretraining. The cross-national angle we already built is the natural escape from
the memorisation critique — lean on it.

**Measurement:** convergence curve (captured importance / accuracy vs iteration), ceiling
reached, iterations-to-ceiling, and crucially *does it discover the country-specific
structure* (own-vs-cross adaptation improving over iterations). Ablate feedback richness
(none / outcome-only / error-cases / mechanism) as the key independent variable.

### 2B. The recognition vs generation gap (from the original design, never run)
Show the model the candidate variable list and let it select (recognition), vs the current
generate-from-nothing. The gap is the cost of reasoning without a menu. Cheap, sharp, and
it bounds how much of the "shallow knowledge" is retrieval-of-own-knowledge vs genuine
absence. A natural robustness companion to the main paper.

### 2C. Prediction-side decoupling (the design's decoupled matrix)
We have only tested *selection*. Cross selection {model, oracle} x prediction {LLM, XGBoost}
to localise the bottleneck: is it knowing-what-matters or using-it? Connects directly back
to Paper-1's prediction failures and would unify the two papers.

### 2D. Behaviour-as-signal: the base-rate / mode-collapse probe, powered
Pilot-2 found models asking for population base rates instead of individual attributes,
skewed by region. Turn this into a designed study: across many countries/topics/models,
when does a model reach for the marginal vs the conditional, and does that predict its
downstream flattening (the Paper-1 failure)? This is a behavioural-mechanism paper that
links "what models ask for" to "how they fail" — potentially the most novel contribution
because it's about model *reasoning behaviour*, not just accuracy.

### 2E. Open-world / unmappable features as a coverage instrument
Models request constructs surveys don't measure (chronic conditions, betrayal experience).
The unmappable set, properly typed (insightful vs vacuous), is a map of *where survey
instruments lag what's predictively relevant* — a contribution to survey methodology, not
just LLM evaluation. The IIA-assumption note already sets this up.

---

## My read on sequencing
- **Now:** finish the main paper on the static audit (T1-T3 + behavioural layer), powered,
  with free-text + mapping validation. It is publishable and it de-risks the infrastructure.
- **Next:** 2A (iterative discovery) is the highest-ceiling follow-up and the most distinctive
  — but only attempt it once the static pipeline is bulletproof, because it stacks more
  inference on top of the same mapping/oracle machinery. Its outcome-vs-mechanism feedback
  ablation is the crux; design it in from day one.
- **Cheap high-value add-ons** that could even ride in the main paper: 2B (recognition gap)
  and 2D (base-rate behaviour) are low-cost given existing infra and would both strengthen it.
