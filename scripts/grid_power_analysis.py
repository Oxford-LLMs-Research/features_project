# Phase B power analysis: empirical variance components + cluster-robust sizing
import pandas as pd, numpy as np, json, glob, os, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from survey_features.config import OUTPUTS_DIR
from survey_features.layout import analysis_dir, selectors_dir

ROOT = str(OUTPUTS_DIR)
PILOT = str(selectors_dir(OUTPUTS_DIR) / "runs" / "pilot_phase_a")

pd.set_option("display.width", 200)

# ---------- Load ----------
def load_scores(pattern, tagcol=None):
    frames = []
    for f in glob.glob(pattern):
        sel = os.path.basename(f).replace("scores_", "").replace("swap_scores_", "").replace(".csv", "")
        df = pd.read_csv(f)
        df["selector"] = sel
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

pilot = load_scores(os.path.join(PILOT, "scores_*.csv"))
swap = load_scores(os.path.join(PILOT, "swap_scores_*.csv"))
era3 = load_scores(os.path.join(ROOT, "main", "scores_*.csv"))
het = pd.read_csv(analysis_dir(OUTPUTS_DIR) / "oracle_heterogeneity.csv")

print("pilot rows", len(pilot), "swap rows", len(swap), "era3 rows", len(era3))
print("pilot selectors", pilot.selector.unique())
print("era3 selectors", era3.selector.unique())
print("pilot conditions", pilot.condition.unique(), "arms", pilot.arm.unique(), "k_specs", pilot.k_spec.unique())
print("era3 arms", era3.arm.unique(), "k_specs", era3.k_spec.unique(), "conditions", era3.condition.unique())
print("swap k_mode", swap.k_mode.unique() if "k_mode" in swap else None)
print("errors pilot:", pilot.error.notna().sum(), "era3:", era3.error.notna().sum())

# ---------- Primary outcome: value_over_textbook_ll at model-chosen k ----------
def prep(df):
    d = df[(df.k_spec == "model") & df.value_over_textbook_ll.notna()].copy()
    d["cell"] = d.survey + "|" + d.target + "|" + d.country
    d["tgt"] = d.survey + "|" + d.target
    return d

p = prep(pilot)
e = prep(era3)

print("\n=== era-3 (free-text arms) VoT_ll by arm/condition ===")
print(e.groupby(["arm", "condition"]).value_over_textbook_ll.agg(["mean", "std", "count"]))
print("\n=== pilot VoT_ll by condition/selector ===")
print(p.groupby(["condition", "selector"]).value_over_textbook_ll.agg(["mean", "std", "count"]))

# ---------- Variance components (method of moments, crossed: target & survey) ----------
def varcomp(d, y="value_over_textbook_ll"):
    # average over selectors & conditions first? No - primary contrast is per selector-condition cell mean.
    # Use cell-level obs averaged over selectors to represent a "grid cell" measurement per selector.
    out = {}
    v_tot = d[y].var(ddof=1)
    # between-target variance via random-effects ANOVA (unbalanced, MoM)
    g = d.groupby("tgt")[y]
    means, ns = g.mean(), g.size()
    grand = d[y].mean()
    # one-way ANOVA MoM
    k = len(means)
    N = ns.sum()
    ssb = (ns * (means - grand) ** 2).sum()
    ssw = ((d[y] - d.tgt.map(means)) ** 2).sum()
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    s2_t = max(0.0, (msb - msw) / n0)
    icc_t = s2_t / (s2_t + msw) if (s2_t + msw) > 0 else 0
    # survey ICC
    gs = d.groupby("survey")[y]
    means_s, ns_s = gs.mean(), gs.size()
    ks = len(means_s)
    ssb_s = (ns_s * (means_s - grand) ** 2).sum()
    ssw_s = ((d[y] - d.survey.map(means_s)) ** 2).sum()
    msb_s = ssb_s / (ks - 1)
    msw_s = ssw_s / (N - ks)
    n0_s = (N - (ns_s ** 2).sum() / N) / (ks - 1)
    s2_s = max(0.0, (msb_s - msw_s) / n0_s)
    icc_s = s2_s / (s2_s + msw_s) if (s2_s + msw_s) > 0 else 0
    return dict(sd_total=np.sqrt(v_tot), icc_target=icc_t, icc_survey=icc_s,
                sd_within_target=np.sqrt(msw), n=N, n_targets=k, n_surveys=ks)

# per selector (cells are the unit; selector-level analyses run separately then pooled)
print("\n=== variance components, pilot (per selector, unprompted) ===")
for sel in sorted(p.selector.unique()):
    d = p[(p.selector == sel) & (p.condition == "unprompted")]
    if len(d) > 5:
        print(sel, varcomp(d))

print("\n=== variance components, era-3 (per selector, per condition) ===")
for sel in sorted(e.selector.unique()):
    for cond in sorted(e.condition.unique()):
        d = e[(e.selector == sel) & (e.condition == cond)]
        if len(d) > 5:
            print(sel, cond, varcomp(d))

# pooled across selectors, averaging within cell first (cell = target x country), per condition
print("\n=== era-3 pooled (cell means over selectors), by condition ===")
for cond in sorted(e.condition.unique()):
    d = e[e.condition == cond].groupby(["survey", "tgt", "cell"], as_index=False).value_over_textbook_ll.mean()
    print(cond, varcomp(d))

# ---------- Swap contrast ----------
sw = swap[(swap.k_spec == "model") & swap.value_over_textbook_ll.notna()].copy()
sw["tgt"] = sw.survey + "|" + sw.target
own = sw[sw.k_mode == "own"].set_index(["selector", "survey", "target", "country", "condition"])
swp = sw[sw.k_mode == "swap"]
rows = []
for _, r in swp.iterrows():
    key = (r.selector, r.survey, r.target, r.country, r.condition)
    if key in own.index:
        o = own.loc[key]
        if isinstance(o, pd.DataFrame):
            o = o.iloc[0]
        # adaptation gain: own selection beats swapped-in selection on same destination cell
        rows.append(dict(selector=r.selector, survey=r.survey, target=r.target, tgt=r.tgt,
                         country=r.country, condition=r.condition, swap_from=r.swap_from,
                         delta_ll=r.model_ll - o.model_ll))  # positive = own better (lower ll)
pairs = pd.DataFrame(rows)
print("\n=== swap pairs ===")
print("n pairs:", len(pairs))
print(pairs.groupby(["condition"]).delta_ll.agg(["mean", "std", "count"]))
print(pairs.groupby(["selector", "condition"]).delta_ll.agg(["mean", "std", "count"]))

def varcomp_pairs(d, y="delta_ll"):
    return varcomp(d.rename(columns={y: "value_over_textbook_ll"}).assign(survey=d.survey))

print("\n=== swap pair variance components (per condition, pooled selectors, cluster=target) ===")
for cond in sorted(pairs.condition.unique()):
    d = pairs[pairs.condition == cond]
    print(cond, varcomp_pairs(d))
# also destination-cell clustering
pairs["cell"] = pairs.tgt + "|" + pairs.country
print("\npairs per target (per selector-condition):")
print(pairs.groupby(["selector", "condition", "tgt"]).size().groupby(level=[0, 1]).mean())

# ---------- Heterogeneity distribution (for binning) ----------
h = het[~het.leakage_flag.astype(bool)]
print("\n=== het distribution (leakage-clean, n=%d) ===" % len(h))
print(h.het_value.describe())
print("terciles:", h.het_value.quantile([0.33, 0.67]).values)

# ---------- Power functions ----------
from scipy.stats import norm

def n_required(sd, effect, alpha=0.05, power=0.80, two_sided=True):
    za = norm.ppf(1 - alpha / (2 if two_sided else 1))
    zb = norm.ppf(power)
    return ((za + zb) * sd / effect) ** 2

def deff(m, icc):
    return 1 + (m - 1) * icc

print("\n=== power table: confirmatory VoT_ll ===")
# scenario grid
for sd in [0.15, 0.19, 0.22]:
    for eff in [0.03, 0.05, 0.08, 0.10]:
        n_iid = n_required(sd, eff)
        print(f"sd={sd} eff={eff}: n_iid={n_iid:.0f}")

print("\n=== power: swap contrast ===")
for sd in [0.06, 0.078, 0.09]:
    for eff in [0.01, 0.015, 0.02, 0.03]:
        print(f"sd={sd} eff={eff}: n_iid={n_required(sd, eff):.0f}")
