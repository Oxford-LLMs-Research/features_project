# Pipeline Critique — preparing the main experiment

> **Largely superseded (2026-08).** This was the 2026-06 audit of the model→score path;
> it deliberately treated the oracle as sound. The 2026-08 audit
> (`pipeline_audit_2026-08.md`) re-examined the oracle itself and fixed most of what
> this memo flagged (format condition, parser, mapping validation via recall
> diagnostics). Kept for the record of what was known when.

*2026-06-02. Honest stage-by-stage evaluation of the measurement pipeline, grounded in
the pilot code and outputs. Goal: capture LLM capability as faithfully as possible before
scaling to more targets, countries, and models. The oracle (ground truth) is treated as
sound and is not re-litigated here; the focus is everything that sits between the model and
the score.*

## Framing: the pipeline is a lossy measuring instrument with a conservative bias

Every stage between "model's prior knowledge" and "captured importance" can only
*lose* signal: a format constraint, a retrieval miss, a disambiguation abstention, or a
redundancy collapse each subtracts. None can add. So the headline numbers (captured
importance ~0.20, value-over-random ~0.02) are a **lower bound on capability**, and the
gap between that lower bound and true capability is exactly the pipeline's attenuation.
For a paper whose claim is "models have only partial/ shallow knowledge," this is the
dangerous direction of bias: we could be measuring our instrument, not the model. The whole
critique below is about sizing and shrinking that attenuation so the capability claim is
defensible.

Quantified attrition in the pilot (2,796 requests across all cells/models):
- retrieval returned zero candidates: 0.6%
- disambiguation returned "none": **28.6%**
- so ~29% of what the models asked for never reached the evaluation, and the headline
  `k_requested` column silently equals `k_mapped` (the loss is upstream of the metric).

---

## Stage 1 — Selection prompt (HIGHEST PRIORITY)

This is where the user's instinct is correct and where the biggest threat lives.

**1a. JSON vs. free-text — format may be suppressing capability.** Both prompts
(`PROMPT_UNPROMPTED`, `PROMPT_COUNTRY`) demand strict JSON ("Output ONLY the JSON list").
The archived Phase-0 notes observed that free-text answers were *more specific and closer
to ground truth* than JSON for the same targets — diagnostic features that appeared in
prose disappeared or moved to the end under JSON. If that holds, the pilot's weak result is
partly an artifact of the output contract, not the model's knowledge. This directly
threatens the paper's central claim.
- *Why it happens:* JSON-mode / "output only the list" pushes models toward terse,
  enumerable, demographic-flavoured items and away from nuanced multi-clause reasoning.
- *Severity:* critical. It is the difference between "models don't know" and "we didn't let
  them say what they know."
- *Fix for main experiment:* run format as an explicit **condition**, not a fixed choice.
  At minimum: free-text elicitation → LLM-extracts-features-into-list as a second step, vs.
  direct JSON. Compare captured importance across formats. If free-text wins, it becomes the
  primary instrument and JSON becomes the ablation (the reverse of the current setup). This
  single experiment de-risks the whole paper.

**1b. The model chooses its own k, and k drives the metric.** The model decides how many
features to emit (pilot mean ~5.4); matched-k then uses that k for both the model row and
the oracle/random rows. If JSON shrinks k, captured importance is computed over fewer
features (smaller numerator) against a smaller oracle top-k denominator — format effects
propagate straight into the headline. We never analysed the k distribution by format or
told the model how many to give.
- *Fix:* report results at fixed k (e.g. 5, 10) as well as model-chosen k, so the capability
  estimate is not hostage to elicited list length. Also log k by format/model.

**1c. Prompt anchoring.** The feature example given is "a specific attitude or behaviour"
(good — neutral), but the system prompt "You are a social science researcher" and the
single-example format may still bias toward survey-jargon answers. Worth a small ablation
(persona vs none; example vs none) rather than assuming neutrality.

**1d. Single sample at temperature 0.** One deterministic draw per cell. We have no
estimate of within-model elicitation variance, and temp-0 can mode-collapse to generic
demographics. A few samples at t>0 (or a self-consistency union of features) would both
measure stability and likely raise the capability ceiling.

---

## Stage 2 — Embedding retrieval (mapping)

`map_features_to_variables`: embed each request (label-only and label+reasoning, take max
cosine), return top-5 survey variables above similarity 0.3.

**2a. Retrieval recall is an unmeasured ceiling.** Disambiguation can only pick from the
top-5 the embedder surfaced. If the survey variable that truly matches a request is not in
those 5, the request is unmappable *by construction* — counted as model failure when it is
really a retrieval failure. We have never measured retrieval recall (does the correct
variable appear in the shortlist?), because that needs human request→variable labels.
- *Severity:* high, and it inflates the "models ask for unmeasured things" story with what
  may be embedder misses.
- *Fix:* the design's annotation study (300+ request–variable pairs) is the right tool and
  was never run. Even a small annotated set lets us estimate recall@5 and separate
  retrieval misses from genuine coverage gaps. Also test a stronger embedder (BGE/E5) and
  larger top-k as sensitivity.

**2b. Single embedding model, single threshold.** MiniLM at 0.3 is one arbitrary point.
The dual-embedding max-trick (label vs label+reasoning) is a sensible hack but undocumented
in effect size. These are knobs we set once and never varied.
- *Fix:* embedding-model and threshold sensitivity sweep; report whether conclusions move.

**2c. Asymmetric leakage filter.** Retrieval excludes candidates with cosine > 0.85 to the
*target* (good — the B1 leakage motivation). But this is lexical; B1 showed empirical
leakage (Q263←Q266) slips through at 0.73 cosine. The retrieval filter and the B1 audit use
different mechanisms; they should be reconciled so the same leakage definition governs both
exclusion and reporting.

---

## Stage 3 — Disambiguation

`disambiguate_single` + `parse_disambig_response`, fixed model
(Nemotron-3-Nano-30B-A3B), top-5 → one letter or "none".

**3a. LATENT PARSER BUG — will bite with new models.** `parse_disambig_response` does:
`for i in range(n_candidates): if letters[i] in cleaned: return i`. This returns the
**lowest-lettered candidate that appears anywhere in the string**, not the one the model
chose by position. In the pilot this was harmless because 71.4% of responses are bare
letters and the other 28.6% are literally "none" (verified). But a chattier disambig model
("Not A; I'd choose C") would be mis-parsed as A. Since the main experiment adds models,
this is a real hazard.
- *Fix:* parse the *last* standalone letter token, or require an exact-match format
  ("Answer: C"), or constrain decoding. Add a unit test over adversarial response strings.

**3b. Fixed disambig model: right for relative comparison, wrong for absolute capability.**
Holding disambig constant so model comparisons reflect only selection quality is a sound
design choice. But the user's goal is also to *capture capability as best we can* — and a
single small disambiguator imposes a systematic ceiling on captured importance for
everyone. If Nemotron routes "economic insecurity" to a mediocre item, every model's score
is depressed equally. Fine for "model A vs B," corrosive for "how capable are LLMs."
- *Fix:* keep a fixed disambiguator for cross-model comparability, but additionally run a
  stronger disambiguator (or human adjudication on a sample) to estimate the *absolute*
  ceiling the small model is costing us. Report both.

**3c. No human validation of mapping.** The design promised a 3-annotator, 300-pair study
with Cohen's/Fleiss' κ; it was never run. Every result is currently "conditional on mapping
quality" (current_state.tex says so explicitly). This is the single biggest credibility gap
for the mapping half of the pipeline.

**3d. "none" is overloaded.** A "none" can mean (i) the survey genuinely lacks the
construct (informative — a real coverage gap / the model reasoning beyond the instrument),
or (ii) retrieval didn't surface the right candidate (instrument failure), or (iii) the
disambiguator was over-conservative. The pilot collapses all three into one 28.6% bucket.
The design's "insightful vs vacuous unmappable" taxonomy is exactly the missing
disambiguation and should be coded on a sample.

---

## Stage 4 — Evaluation / metric coupling

**4a. Matched-k couples three things through one k.** Oracle top-k, model-k, random-k all
use the model's k. So elicited list length (a prompt artifact, Stage 1b) sets the budget for
the ceiling and the baseline simultaneously. Fixed-k reporting (4 → also 1b) decouples this.

**4b. Redundancy is real and under-explained.** B2 showed captured importance ~2× random
yet near-zero downstream value — the model picks above-median-but-correlated features. This
is a genuine finding, but it also means downstream accuracy is a blunt instrument for
capability; captured importance and the percentile metric are better, and the paper should
lead with them (it now does). Worth adding a redundancy diagnostic (e.g. incremental
importance / variance inflation among the model's picks).

**4c. Oracle and evaluator are both tree ensembles.** Oracle = AutoGluon permutation
importance; evaluator = XGBoost. Shared inductive bias could make "oracle-important"
features mechanically easier for the evaluator to exploit, slightly understating
non-tree-friendly model picks. Minor, but worth one robustness check with a different
evaluator family (e.g. logistic / MI-based) on a subset.

**4d. The oracle's selection saw the evaluation rows; the model's did not.** This is not a
claim about the oracle's internal validity (Stage 0 framing stands) — it is about the
*coupling* between the two stages. `compute_oracle` derives importances by permuting on a
20% holdout of the country data (`oracle.py:521-526`, `:581`), but `evaluate_feature_set`
then runs 5-fold CV over **all** of that country's rows. So oracle top-k was chosen with
access to rows that later appear in the evaluation folds, while the LLM arm's selection has
zero data exposure and the random arm's is exposure-free by construction. The three arms are
therefore not selected under the same information regime.
- *Consequence:* "cost of imperfect selection" (oracle − model) is inflated by an
  unquantified selection-on-eval-data margin. Under matched-k this propagates to every
  headline that uses the oracle as the ceiling.
- *Why it is defensible anyway:* the oracle is *intended* as a full-information upper bound,
  not a fair competitor — a selector that had to generalise out-of-sample would be a
  different (weaker) ceiling. The problem is that the writeups do not say which of the two
  the number is.
- *Fix (cheap, reporting-only):* name it explicitly as an in-sample ceiling wherever the
  oracle row appears. *Fix (if a defensible gap estimate is wanted):* refit the oracle on
  each CV training fold and take top-k per fold, so selection and evaluation share a split
  discipline; report the fold-honest oracle alongside the full-information one and treat the
  difference as the size of this bias. Costly (one AutoGluon fit per fold per cell), so a
  subset is enough.
- *Note:* this is orthogonal to the learner choice. Scoring with the AutoGluon ensemble
  instead of XGBoost would not fix it, and would add a circularity problem on top (oracle
  top-k is by construction the set that ensemble most relies on, so the ensemble would be
  grading features chosen to maximise its own permutation sensitivity). Fixed-hyperparameter
  XGBoost as a neutral third learner is the right call and should stay.

---

## Stage 0 — Target selection (feeds everything)

Not a "pipeline step" but it gates the whole thing. The pilot used a cardinality quota
(2 binary / 2 mid / 1 large) + topic spread, which produced 31/89 degenerate cells and the
6 leakage cells. The design intended *signal-driven* selection (high non-demographic
signal, high cross-national variation). Re-aligning to that is the highest-value change for
the main run: it removes the degenerate dilution, reduces leakage exposure, and ensures
Test 2 (adaptation) is run only where structure actually varies across countries.

---

## Priority ranking for the main experiment

1. **Format experiment (1a) + fixed-k reporting (1b/4a).** Settles whether JSON suppresses
   capability. Without this the headline capability claim is not defensible.
2. **Mapping validation study (3c) + retrieval recall (2a) + "none" taxonomy (3d).** Turns
   "conditional on mapping quality" into a measured quantity; separates model failure from
   instrument failure.
3. **Signal-driven target re-selection (Stage 0).** Removes degeneracy/leakage dilution;
   makes Test 2 meaningful.
4. **Disambig parser fix + unit tests (3a); stronger-disambiguator ceiling (3b).** Robustness
   for scaling to more models, and an absolute-capability estimate.
5. **Sensitivity sweeps (2b embedding/threshold; 4c evaluator family; 1d sampling).** Show
   conclusions are not knife-edge on arbitrary knobs.
6. **Label the oracle as an in-sample ceiling (4d).** Reporting-only, near-zero cost, and it
   stops the oracle−model gap being read as a like-for-like comparison. Fold-honest oracle
   on a subset if a size estimate for the bias is wanted.
7. **Reconcile leakage definitions (2c).**

## One-line bottom line

The oracle is sound, but the model→score path is a conservative, lossy instrument whose
attenuation we have not yet measured; before scaling, we should (a) prove the JSON format
isn't hiding capability, and (b) replace "conditional on mapping quality" with an actual
mapping-validation number. Those two convert the pilot's "models know little" into a claim
about models rather than about our pipeline.
