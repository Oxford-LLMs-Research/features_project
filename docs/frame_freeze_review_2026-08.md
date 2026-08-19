# Frame freeze review — 19 Aug 2026

**What this is.** The target frame is the pool of survey questions the confirmatory
grid (the locked 270-cell measurement whose numbers go in the paper) will be sampled
from. Freezing it is the first registered step of the grid design
([`grid_design_memo_2026-08.md`](grid_design_memo_2026-08.md)). The mechanical
screens have all run; this document lists the **four decisions that need a human**,
each with a recommendation. Once they're decided, the frame is frozen and the next
step is the seeded draw.

**How it was produced.** `scripts/target_universe_screen.py` re-run 2026-08-19 on
all six surveys' microdata (skip/follow-up flags refreshed from data; type-1
estimability counts per country retained). Rules already decided and applied:
demographics are never targets (features + textbook baseline only); follow-up items
(368) are out — their missingness is routing, not response; type-1 requires ≥ 50
respondents with ≥ 2 classes (classification) or ≥ 3 scale points (regression) per
country.

## The frame as it stands

**1,233 questions** (demographics-free, test-free, skip-free, ≥ 1 estimable
country); **1,133** have the ≥ 3 estimable countries the draw needs. The 90-question
draw uses < 8% of the pool. Per-survey, draw-eligible:

| survey | binary | nominal | ordinal | continuous | total |
|---|---|---|---|---|---|
| afrobarometer | 6 | 8 | 165 | 0 | 179 |
| arabbarometer | 4 | 24 | 106 | 0 | 134 |
| asianbarometer | 13 | 28 | 147 | 0 | 188 |
| ess_wave_11 | 49 | 6 | 99 | 2 | 156 |
| latinobarometer | 23 | 14 | 137 | 0 | 174 |
| wvs | 43 | 17 | 242 | 0 | 302 |

(The thin binary pools in Afrobarometer/Arab Barometer and the ESS-only continuous
pair are already handled by the amended floor rule — memo, "The grid to approve".)

---

## Decision 1 — the 42 gate questions (the substantive call)

A **gate** is a question whose answer controls whether follow-ups are asked at all
("Have you heard of climate change?" → the whole climate battery). Follow-ups are
already out; the question is whether the gates themselves can be *targets*. The
risk is empirical and we have seen it: the follow-up battery collectively encodes
the gate's answer, so an oracle "predicts" the gate almost perfectly from routing,
not understanding — Afrobarometer `Q67A` reached accuracy 0.99 exactly this way, and
removing single offender features did not help. The typed leakage screen now
catches this pattern per cell (distributed-leakage rule), but only *after* paying
for that cell's oracle.

Recommendation was drop-all (cheap insurance against the Q67A pattern), noting the
list contains prominent constructs — ESS `vote` (turnout), `polintr`, `netusoft`,
`clsprty`, the `dscr*` discrimination battery — and three provisional-grid cells
(`Q43A`, `Q104`, `Q630`).

**DECIDED 2026-08-19: keep all 42 gates as targets** (user ruling — the constructs
outweigh the insurance). Consequences accepted and recorded: (a) the typed leakage
screen is now load-bearing for gate cells — its distributed-leakage rule is what
stands between a Q67A-style routing artifact and the confirmatory grid; (b) oracle
compute will be spent on gate cells that may drop at screening, so expect attrition
above the memo's 15–20% planning figure on this subset; (c) the drawn grid carries
each question's `gate` flag so post-oracle drops are attributable, and the spare
pool should anticipate gate-cell replacements.

<details>
<summary>Full gate list (42)</summary>

| survey | variable | section |
|---|---|---|
| afrobarometer | Q11E | political_participation |
| afrobarometer | Q40A, Q41A, Q42A, Q43A, Q43D, Q46L | institutional_trust |
| afrobarometer | Q67A, Q74B, Q74D | contemporary_issues |
| afrobarometer | Q78C, Q78D | political_attitudes |
| afrobarometer | Q91B, Q91C, Q92A | wellbeing |
| arabbarometer | Q104, Q630, Q882, QMOR4_2 | social_attitudes |
| arabbarometer | Q210 | institutional_trust |
| arabbarometer | Q274A | wellbeing |
| arabbarometer | Q409, Q424, Q534_4, Q534_5 | contemporary_issues |
| arabbarometer | Q501D | political_participation |
| arabbarometer | Q610_8 | values_identity |
| arabbarometer | Q734C, QPAL18 | political_attitudes |
| ess_wave_11 | clsprty, polintr, vote | political_participation |
| ess_wave_11 | dscretn, dscrntn, dscrrce, dscrrlg, hlpfmly, mbtru | social_attitudes |
| ess_wave_11 | fltlnl | wellbeing |
| ess_wave_11 | netusoft | contemporary_issues |

(Latinobarometer and WVS contribute no would-be-eligible gates.)
</details>

## Decision 2 — rescue 8 attitude/behavior items filed in demographic blocks

The no-demographic-targets rule excludes by codebook **section**; these eight sit in
demographic blocks but their content is self-assessment or behavior, which the
registered default ("facts about the respondent's situation are out; attitudes,
evaluations, behaviors stay") would keep. All eight are otherwise fully eligible
(no skip flags, ample estimable countries):

| survey | variable | what it is | kind |
|---|---|---|---|
| wvs | Q287 | subjective social class | self-assessment |
| latinobarometer | S2 | subjective social class | self-assessment |
| ess_wave_11 | rlgdgr | self-rated religiosity (0–10) | self-assessment |
| ess_wave_11 | impbemw | importance-of item | attitude |
| afrobarometer | Q90H, Q90I | practice frequency items | behavior |
| ess_wave_11 | alcfreq | alcohol frequency | behavior |
| ess_wave_11 | fnsdfml | eats family meals frequency | behavior |

**DECIDED 2026-08-19: rescue all 8** — ruled substantively important (user).
The registered whitelist of (survey, variable) pairs above rides alongside the
section rule and goes into `frame_rules` when the frame is frozen. The leakage
screen stays their guard — self-rated religiosity will correlate hard with
worship-attendance *features*, which is fine and substantive unless it crosses the
near-duplicate threshold, which the screen now tests per cell.

## Decision 3 — ESS test-battery exclusion (mechanical, confirm)

18 ESS variables named `test*` (`testjc34`–`42`, `testji1`–`9`) are questionnaire
experiments, not substantive items; they sit in `contemporary_issues`, so no
section rule catches them. **DECIDED 2026-08-19: excluded via the name rule** — ESS variables matching `^test`
are never targets. (This also removes the `testji4`/`testji6` sibling-item
near-duplicate found in the pipeline evaluation.)

## Decision 4 — nothing else needs judgment (for transparency)

- **368 follow-ups**: auto-excluded, no walk needed.
- **12 fielded-but-thin items** (0 estimable countries despite being fielded):
  asianbarometer Q68/Q69/Q70/Q71/Q30/Q158; ESS cmsrv, dscrsex, dscrdsb, medtroc,
  trhltacp, trhlthy. Auto-excluded by type-1; listed so their absence is not a
  surprise later.
- **177 questions with zero estimable countries** overall (86 of them demographic —
  moot). Auto-excluded.

---

## The frame is FROZEN (2026-08-19)

All rulings are in: gates kept, 8 demographic-block items rescued, ESS `test*`
excluded. The machine-readable rules live in
[`data/frame_rules.yaml`](../data/frame_rules.yaml) (the *rules* are frozen, not
the regenerated inventory file, per registry convention).

**Frozen frame: 1,283 questions (42 gate-flagged); 1,178 draw-eligible with ≥ 3
estimable countries (37 gate-flagged).** Per-survey draw-eligible pools after the
rulings (gates back in improved the thin binary pools: Afrobarometer 6 → 13,
Arab Barometer 4 → 6 — the near-census caveat on those strata stands):

| survey | binary | nominal | ordinal | continuous | total |
|---|---|---|---|---|---|
| afrobarometer | 13 | 10 | 173 | 0 | 196 |
| arabbarometer | 6 | 24 | 116 | 0 | 146 |
| asianbarometer | 13 | 28 | 147 | 0 | 188 |
| ess_wave_11 | 52 | 8 | 106 | 2 | 168 |
| latinobarometer | 23 | 15 | 139 | 0 | 177 |
| wvs | 43 | 18 | 242 | 0 | 303 |

Next step: `make_confirmatory_grid.py` implementing the draw from
`frame_rules.yaml` (type-stratified 15/survey with the amended floor rule,
3 countries uniform from eligible, WVS region-stratified, registered seed, 3
spares per survey, `gate` flag carried into the grid CSV).
