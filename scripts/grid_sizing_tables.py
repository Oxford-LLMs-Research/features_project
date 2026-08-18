# Phase B part 2: nested variance, sizing tables, flash diagnosis, cost model
import pandas as pd, numpy as np, glob, os
from scipy.stats import norm

ROOT = r"C:\Users\murrn\Dropbox\features_project\outputs"
PILOT = os.path.join(ROOT, "main", "runs", "pilot_phase_a")
pd.set_option("display.width", 220)

def load_scores(pattern):
    frames = []
    for f in glob.glob(pattern):
        sel = os.path.basename(f).replace("scores_", "").replace("swap_scores_", "").replace(".csv", "")
        df = pd.read_csv(f); df["selector"] = sel; frames.append(df)
    return pd.concat(frames, ignore_index=True)

pilot = load_scores(os.path.join(PILOT, "scores_*.csv"))
swap = load_scores(os.path.join(PILOT, "swap_scores_*.csv"))
era3 = load_scores(os.path.join(ROOT, "main", "scores_*.csv"))

def prep(df):
    d = df[(df.k_spec == "model") & df.value_over_textbook_ll.notna()].copy()
    d["tgt"] = d.survey + "|" + d.target
    d["cell"] = d.tgt + "|" + d.country
    return d

p, e = prep(pilot), prep(era3)
e_main = e[e.selector.isin(["deepseek", "kimi"])]  # exclude __hgb evaluator-ablation arms

# ---------- 1. Yardstick: textbook's own edge over random ----------
for name, d in [("era3", e_main), ("pilot", p)]:
    tb_edge = (d.random_ll - d.textbook_ll)
    orc_edge = (d.textbook_ll - d.oracle_ll)
    print(f"{name}: textbook-over-random ll mean={tb_edge.mean():.4f}, oracle-over-textbook (ceiling) mean={orc_edge.mean():.4f}")
    print(f"   VoT_ll mean={d.value_over_textbook_ll.mean():.4f}  (unprompted only: "
          f"{d[d.condition=='unprompted'].value_over_textbook_ll.mean():.4f})")

# ---------- 2. Nested variance decomposition (survey -> target -> cell) ----------
def nested(d, y="value_over_textbook_ll", label=""):
    # cell-level obs (already averaged over selectors if desired)
    # level 1: residual within target (across countries)
    tgt_stats = d.groupby(["survey", "tgt"])[y].agg(["mean", "var", "count"])
    s2_e = np.average(tgt_stats["var"].dropna(), weights=(tgt_stats["count"] - 1)[tgt_stats["var"].notna()])
    mbar = tgt_stats["count"].mean()
    # variance of target means = s2_t + s2_e/m
    s2_tmeans = tgt_stats["mean"].var(ddof=1)
    s2_t_marginal = max(0.0, s2_tmeans - s2_e / mbar)
    # split target-mean variance into between-survey and within-survey
    sv = tgt_stats.reset_index().groupby("survey")["mean"].agg(["mean", "var", "count"])
    s2_t_within = max(0.0, np.average(sv["var"].dropna(), weights=(sv["count"] - 1)[sv["var"].notna()]) - s2_e / mbar)
    s2_s = max(0.0, s2_tmeans - (s2_t_within + s2_e / mbar))
    print(f"{label}: sd_e(within-target)={np.sqrt(s2_e):.4f}  sd_t(marginal)={np.sqrt(s2_t_marginal):.4f}  "
          f"sd_t(within-survey)={np.sqrt(s2_t_within):.4f}  sd_survey~{np.sqrt(s2_s):.4f}  "
          f"[T={len(tgt_stats)}, mean C/target={mbar:.1f}]")
    return s2_e, s2_t_marginal, s2_t_within

print("\n=== nested decomposition ===")
e_cellmean = e_main.groupby(["survey", "tgt", "cell", "condition"], as_index=False).value_over_textbook_ll.mean()
s2_e_A, s2_tm_A, s2_tw_A = nested(e_cellmean[e_cellmean.condition == "unprompted"], label="era3 pooled-selector unprompted")
nested(e_cellmean[e_cellmean.condition == "country_provided"], label="era3 pooled-selector country")
p_cellmean = p.groupby(["survey", "tgt", "cell", "condition"], as_index=False).value_over_textbook_ll.mean()
s2_e_P, s2_tm_P, s2_tw_P = nested(p_cellmean[p_cellmean.condition == "unprompted"], label="pilot pooled-selector unprompted")
# per single selector (what one zoo member's estimate looks like)
s2_e_1, s2_tm_1, s2_tw_1 = nested(e_main[(e_main.selector == "kimi") & (e_main.condition == "unprompted")],
                                  label="era3 kimi-only unprompted")

# ---------- 3. Confirmatory sizing table ----------
za, zb = norm.ppf(0.975), norm.ppf(0.80)
K = za + zb  # 2.80

def mde_conf(T, C, s2_t, s2_e):
    se = np.sqrt(s2_t / T + s2_e / (T * C))
    return se, K * se

print("\n=== confirmatory MDE (80% power, alpha .05 two-sided) ===")
print("variance inputs: era-3 pooled-selector, target var within-survey (surveys as fixed strata)")
print(f"  s2_t={s2_tw_A:.5f} (sd {np.sqrt(s2_tw_A):.3f}), s2_e={s2_e_A:.5f} (sd {np.sqrt(s2_e_A):.3f})")
for tps in [5, 10, 15, 20, 25]:
    for C in [3, 5]:
        T = tps * 6
        se, mde = mde_conf(T, C, s2_tw_A, s2_e_A)
        se2, mde2 = mde_conf(T, C, s2_tm_A, s2_e_A)  # marginal (conservative, no survey stratification credit)
        print(f"T/survey={tps:>2} C={C}: cells={T*C:>4}  SE={se:.4f} MDE={mde:.3f}  (marginal-var MDE={mde2:.3f})")

# single-selector version (secondary per-model claims)
print("\nper-single-selector MDE (kimi-only variance):")
for tps in [10, 15, 20]:
    T = tps * 6
    se, mde = mde_conf(T, 3, s2_tw_1, s2_e_1)
    print(f"T/survey={tps} C=3: MDE={mde:.3f}")

# ---------- 4. Test-2 sizing ----------
# pair-level sd from pilot; ICC within target measured ~0 but small sample -> sensitivity
sd_pair = 0.081
print("\n=== Test-2 swap sizing: MDE for high-bin mean (80% power) ===")
print("B = targets in bin, C = countries, pairs/target = C*(C-1); S = selectors pooled (indep errors assumed for residual, not target)")
for rho in [0.0, 0.05, 0.10, 0.20]:
    s2_t = rho * sd_pair**2
    s2_r = (1 - rho) * sd_pair**2
    for B in [5, 8, 10, 12, 15]:
        for C in [5, 6, 8]:
            m = C * (C - 1)
            for S in [1, 9]:
                se = np.sqrt(s2_t / B + s2_r / (B * m * S))
                print(f"rho={rho:.2f} B={B:>2} C={C} S={S}: MDE={K*se:.4f}")
        # only print subset
    print("---")

# high-vs-low bin difference
print("\nhigh-vs-low bin contrast MDE (equal bins B, C=6, S=9 pooled):")
for rho in [0.05, 0.10, 0.20]:
    s2_t = rho * sd_pair**2; s2_r = (1 - rho) * sd_pair**2
    for B in [8, 10, 12]:
        m = 30
        se_bin2 = s2_t / B + s2_r / (B * m * 9)
        print(f"rho={rho} B={B}: MDE_diff={K*np.sqrt(2*se_bin2):.4f}")

# ---------- 5. Flash diagnosis ----------
print("\n=== flash k (n features) by condition, model-k spec ===")
print(p[p.selector == "flash"].groupby("condition").k.describe()[["count", "mean", "min", "25%", "50%", "max"]])
print(p[p.selector != "flash"].groupby("condition").k.describe()[["count", "mean", "min", "50%", "max"]])

# ---------- 6. Cost / wall-clock model (canvas constants) ----------
GEN_IN, GEN_OUT, EXTRACT_IN, EXTRACT_OUT = 65, 2233, 1333, 1128
DISAMBIG_IN, DISAMBIG_OUT, DISAMBIG_PER_COND, CONDITIONS = 474, 324, 30, 2
EXTRACT_LAT_S, DISAMBIG_LAT_S = 12.85, 2.47
QWEN_IN, QWEN_OUT, NEMO_IN, NEMO_OUT = 0.2, 0.6, 0.06, 0.24
PIPE_WORKERS, MAP_WORKERS = 4, 8
SCORE_S_PER_42, LEGACY_42 = 16 * 60, 42
ZOO = [
    ("Kimi-K3", 3, 15, 120, dict(thinking=True)),
    ("Kimi-K2.6", 0.95, 4, 68, dict(genIn=65, genOut=2233, genLatS=32.87)),
    ("DeepSeek-V4-Pro", 1.75, 3.5, 67, dict(genIn=61, genOut=1149, genLatS=17.07)),
    ("GLM-5.1", 1.4, 4.4, 70, {}),
    ("Qwen3.5-397B", 0.6, 3.6, 80, {}),
    ("Nemotron-Ultra-550B", 1, 3, 80, {}),
    ("MiniMax-M3", 0.3, 1.2, 80, {}),
    ("Nemotron-Super-120B", 0.3, 0.9, 90, {}),
    ("gpt-oss-120b", 0.15, 0.6, 90, {}),
]  # flash dropped

def usd(pin, pout, i, o): return pin / 1e6 * i + pout / 1e6 * o
EXTRACT_PC = usd(EXTRACT_IN, EXTRACT_OUT, QWEN_IN, QWEN_OUT)
MAP_PC = usd(DISAMBIG_IN, DISAMBIG_OUT, NEMO_IN, NEMO_OUT) * DISAMBIG_PER_COND

def run_cost(llm_cells, oracle_quick, oracle_balanced, zoo, think_mult_k3=5,
             oracle_quick_min=12.5, procs=3):
    nCond = llm_cells * CONDITIONS
    gen = genWall = 0.0
    for (name, i, o, tps, ex) in zoo:
        gout = ex.get("genOut", GEN_OUT) * (think_mult_k3 if ex.get("thinking") else 1)
        gin = ex.get("genIn", GEN_IN)
        gen += nCond * usd(gin, gout, i, o)
        lat = ex.get("genLatS") if (ex.get("genLatS") and not ex.get("thinking")) else gout / tps
        genWall += nCond * lat / PIPE_WORKERS
    nM = len(zoo)
    extract = nCond * nM * EXTRACT_PC
    mapc = nCond * nM * MAP_PC
    extractWall = nCond * EXTRACT_LAT_S * nM / PIPE_WORKERS
    mapWall = nCond * DISAMBIG_PER_COND * DISAMBIG_LAT_S * nM / (PIPE_WORKERS * MAP_WORKERS)
    oracleWall_h = (oracle_quick * oracle_quick_min + oracle_balanced * oracle_quick_min * 3) / 60 / procs
    scoreWall_h = SCORE_S_PER_42 * (llm_cells / LEGACY_42) * nM / 3600
    llm_h = (genWall + extractWall + mapWall) / 3600
    print(f"  LLM $: gen={gen:.0f} extract={extract:.0f} map={mapc:.0f} total=${gen+extract+mapc:.0f}")
    print(f"  wall: LLM={llm_h:.1f}h  oracle={oracleWall_h:.1f}h  score(seq, all models)={scoreWall_h:.1f}h")

print("\n=== cost scenarios (9-model zoo incl. K3 at 5x thinking) ===")
print("A) old preset 5x5=150 cells:")
run_cost(150, 150, 0, ZOO)
print("B) recommended: confirmatory 90 targets x3 =270 + Test-2 ext 30x3=90 -> 360 cells; balanced re-fit 120 Test-2 high/low cells:")
run_cost(360, 360, 120, ZOO)
print("C) bigger: 120 targets x3 + ext = 450 cells:")
run_cost(450, 450, 120, ZOO)
