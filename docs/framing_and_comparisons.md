# What we are comparing, and why — a conceptual guide

> **Numbers here are JSON-era; the argument may outlive them.** The claim that the LLM is
> "weak as an importance estimator" is built on VoR ~0.025 and ~half of cells beating
> random. Those are **era-1** figures; free text gives 0.054-0.063 and 76-78% at model-k.
> Two later developments bear on the argument directly: the random null was drawn from a
> wider pool than the oracle's (so VoR was inflated), and a **textbook demographic
> baseline** now exists — which is the answer to the "why not just use the oracle?"
> objection this document raises but could not answer. See [pipeline_audit_2026-08.md](pipeline_audit_2026-08.md) §A2.

*Living document. Plain-language companion to `paper/research_design.tex` (the vision),
`paper/current_state.tex` (the canonical results doc), and `paper/IIA assumption.tex`
(the open-world caveat). Written 2026-05-29.*

This note exists to answer two recurring sources of confusion:

1. **What exactly are the four quantities we compute per cell, and what does each
   comparison *mean*?**
2. **Is there a meaningful difference between asking an LLM what predicts an attitude
   and just running any ML feature-importance algorithm?** (Short answer: yes, a
   fundamental one — but our current *accuracy horse-race* framing hides it. This doc
   explains the distinction and argues for a framing that makes it visible.)

---

## 1. The cast of characters

Every (target × country × condition) cell produces predictions from a **single, fixed
downstream model** (XGBoost, 5-fold CV). The *only* thing that changes between the
numbers below is **which features that fixed model is allowed to use**. So this is a
**feature-selection experiment**, not a prediction-method experiment. Hold that thought —
it is the key to the whole thing.

| Quantity | Who picked the features? | What it tells us |
|---|---|---|
| `majority_baseline` | nobody — predict the modal answer | "no-skill" floor; how (im)balanced the target is |
| `random_acc` | random draw of *k* features (avg of 20) | what an *arbitrary* equal-size feature set achieves |
| `model_acc` | **the LLM**, from question wording alone, then mapped to survey variables | how good the LLM's *prior-knowledge* feature guesses are |
| `oracle_acc` | **permutation importance on the real data** (top-*k*) | the empirical ceiling: the best the data itself can identify |

All four are scored by the *same* XGBoost on the *same* respondents. The feature **count
*k* is matched** across oracle / model / random within a cell ("matched-*k* fairness"),
where *k* = the number of distinct survey variables the LLM's requests mapped to.

The two headline contrasts:

- **`value_over_random` = model_acc − random_acc.** Did the LLM's picks beat *throwing
  darts*? This is the floor test: does the LLM know **anything**?
- **`cost_of_imperfect` = oracle_acc − model_acc.** How far below the empirical ceiling
  did the LLM land? This is the gap to the *best possible* in-survey selection.

The two prompt **conditions** (`unprompted` vs `country_provided`) ask a third thing:
does telling the model the respondent's country change *which* features it asks for —
and does that change track how the predictive structure *actually* differs by country?

---

## 2. What each comparison answers (and what it does NOT)

| Contrast | Question it answers | Trap to avoid |
|---|---|---|
| model vs **random** | Does the LLM's prior beat chance feature choice? | A *positive* value can still be tiny; "beats random" ≠ "useful." |
| model vs **oracle** | How close is the LLM's prior to empirical truth? | A *large* gap on near-deterministic demographic targets is leakage, not "bad reasoning" (see §4). |
| model vs **majority** | Does using the LLM's features beat predicting the modal class? | On skewed targets, *nothing* beats majority — accuracy saturates. |
| **unprompted vs country** | Does the model adapt its reasoning to the population? | Pooled means wash this out; must be sliced by survey × topic. |

---

## 3. The big question: LLM feature selection vs. any ML feature-importance method

This is the question that keeps feeling unclear, so let us be precise.

**The oracle *is* "any predictive ML algorithm for assessing feature importance."**
Permutation importance on gradient-boosted trees is a textbook ML importance method. So
inside our experiment, the `oracle` column and the `model` column literally *are* "ML
feature importance vs. LLM feature selection." They are picking features for the same
downstream predictor.

So why is the LLM interesting at all, if a standard ML method does the same job better
(and by construction it *will* do it better — it has seen the data)? Because **they are
not actually doing the same job.** They operate in different information regimes:

> **The oracle is the answer key. The LLM is a student sitting the exam with no notes.**
> Random guessing is the student who didn't study. We are not asking "is the student
> better than the answer key" (impossible by definition). We are asking **"how much of
> the answer key did the student already know — from prior knowledge alone — and does
> that knowledge adapt across countries?"**

### The six distinctions that make the LLM a different instrument

1. **What they consume.** The ML method consumes the *realized dataset* (the full
   feature matrix X and the outcome y). The LLM consumes *only a sentence* (the question
   wording) plus everything it learned in pretraining. The ML method has the sample and
   no outside knowledge; the LLM has outside knowledge and *none of the sample*.

2. **Prior vs. posterior (timing).** Permutation importance can only run *after* you have
   collected data. The LLM runs *before any data exists*. This is the actual research
   workflow it mirrors: **"what should I measure?"** (design stage, only priors available)
   vs. **"what mattered?"** (analysis stage, data in hand). Data-driven importance is
   useless at the design stage — there is nothing to fit.

3. **Open world vs. closed world.** Permutation importance can only rank the columns you
   already collected. The LLM can propose constructs that **aren't in the survey at all** —
   e.g. for self-rated health it asks for "chronic conditions" and "BMI," which WVS never
   measured. That is a *real* prediction about the world that our within-survey oracle
   *structurally cannot make or score*. (This is exactly the open-world / closed-world
   mismatch handled in `paper/IIA assumption.tex`, and our "unmappable rate" is the
   measurement of it.)

4. **Mechanism / explanation.** The ML method returns a number with no story. The LLM
   returns a *reason* per feature — a falsifiable theoretical claim. Its output is a set
   of hypotheses, not a ranking.

5. **What "correct" even means.** For the ML method, the importance *is* the ground truth
   by definition — it cannot be "wrong." The LLM *can* be wrong, and its wrongness is the
   scientific signal: the gap between its prior and the empirical posterior is precisely
   what tells us whether models "understand" the structure of human attitudes.

6. **Transportability.** A permutation-importance model fit on Germany tells you nothing
   about Nigeria — you simply refit on Nigerian data. It has no concept of "reasoning
   about a different country." The LLM is being asked whether it can *anticipate* how the
   predictive structure shifts across populations **without** refitting. **No ML
   importance method can even attempt our Test 2.** This is the cleanest place where the
   LLM is doing something categorically beyond data-driven importance.

### The catch (and why the current framing undersells the project)

If we lead with the **downstream-accuracy horse-race** (oracle vs model vs random
accuracy), we are implicitly framing the LLM as *a cheap estimator of feature
importance*. In that frame the obvious reviewer question is fatal:

> "If you already have the data to compute these accuracies, why not just use the oracle?
> The LLM only barely beats random — it's a worse version of a tool you already have."

And on our current data that critique mostly lands: pooled `value_over_random ≈ +0.025`,
only ~half of cells beat random. As an *importance estimator*, the LLM is weak.

But that is the **least interesting** thing to ask, and it measures the LLM on exactly
the axis where data-driven methods are guaranteed to win. The distinctions that make the
LLM valuable — works with no data, proposes unmeasured constructs, gives reasons,
anticipates cross-national variation — are **invisible** in an accuracy horse-race.

### Recommendation for the canonical write-up

Reframe so the contribution is about **knowledge**, not **accuracy substitution**:

- **Lead with selection *alignment*** (does the LLM's prior recover the *shape* of the
  empirical importance structure — captured-importance, rank correlation), with downstream
  accuracy as a *validity check* that the alignment is real, not the headline.
- **Make Test 2 (cross-national adaptation) the signature result**, because it is the one
  question no ML importance baseline can answer — it is uniquely an LLM-knowledge question.
- **Treat the open-world / unmappable features as a finding, not noise** (it is the other
  thing the oracle structurally cannot do).
- Keep the accuracy horse-race as a *floor* ("the LLM's prior carries real signal: it
  beats random") and a *humility* result ("but it falls short of the empirical ceiling,
  especially where..."), not as the thesis.

In one line: **we are not proposing the LLM as a replacement for permutation importance;
we are using permutation importance as the answer key to test what the LLM already knows
about the conditional structure of human attitudes — including the parts a data-driven
method can never see (unmeasured features) or attempt (cross-national priors).**

---

## 4. Standing caveat that colours every comparison: demographic leakage

Several current targets are demographics (sex, immigrant status, education band). On
these the oracle can hit ~1.0 by finding a near-duplicate proxy column — that is the data
leaking the label, not "deep predictive structure." Both the biggest model *wins* and
biggest model *losses* in the current run sit on demographic targets. Until a leakage
audit separates these out, read demographic-target cells as a **ceiling/leakage
demonstration**, and attitude/opinion targets as the **real test** of the research
question. (Open decision — see memory `current_state_2026-05`.)

---

## 5. One-paragraph summary to give a co-author

We ask a language model, given only a survey question's wording, to name what it would
want to know about a respondent to predict their answer — with no access to the data and
no list of options. We map those guesses to real survey variables and check, against
permutation-importance ground truth, how much of the true predictive structure the
model's *prior knowledge* recovered, whether that beats random feature choice, and
whether it *adapts* when we name the respondent's country. The permutation-importance
oracle is not a competitor — it is the answer key. The point is not that the model could
replace it (if you have the data, you would just use the oracle); the point is what the
model knows *without* the data, including features the survey never measured and
cross-country differences a data-driven method could never anticipate from priors alone.
