"""Screen the target universe for skip/follow-up items and type-1 cells.

Type-1 is knowable from a country × target cross-tab: classification needs two
classes with n >= 50 (so V2 has ~10 minority rows on a 60/20/20 split);
ordinal/continuous need n >= 50 and >= 3 distinct scale points after the same
prep the oracle uses. No AutoGluon.

Skip/follow-up:
  follow-up — median country structural-missing share >= 0.20 (routing / not asked)
  gate      — some follow-up's missingness splits on this item's classes
              (range >= 0.7) in at least one country (immigrant -> country of birth)

Writes flags back onto data/_target_universe_inventory.json and prints canvas
aggregates.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from survey_features.config import ROOT as _PKG  # noqa: F401  loads .env
from survey_features.grid_screen import TYPE1_MIN_N, TYPE1_MIN_UNIQUE
from survey_features.oracle import TARGET_TYPE_PROBLEM
from survey_features.surveys import (
    SURVEY_COUNTRY_COL,
    _code_variants,
    _substantive_values,
    build_admin_cols,
    clean_question_columns,
    coalesce_case_twin_columns,
    drop_other_specify,
    out_of_scale_codes,
    substantive_numeric_mask,
    to_ordinal_codes,
)
from synthetic_sampling.config.base import DataPaths
from synthetic_sampling.config.surveys import get_survey_config
from synthetic_sampling.loaders.survey_loader import SurveyLoader

META_DIR = Path(
    r"C:/Users/murrn/cursor/synthetic_sampling/synthetic_sampling/src/synthetic_sampling/surveys/metadata"
)
SURVEY_META = {
    "afrobarometer": "pulled_metadata_afrobarometer.json",
    "arabbarometer": "pulled_metadata_arabbarometer.json",
    "asianbarometer": "pulled_metadata_asianbarometer.json",
    "ess_wave_11": "pulled_metadata_ess11.json",
    "latinobarometer": "pulled_metadata_latinobarometer.json",
    "wvs": "pulled_metadata_wvs.json",
}

SURVEYS = [
    "afrobarometer",
    "arabbarometer",
    "asianbarometer",
    "ess_wave_11",
    "latinobarometer",
    "wvs",
]

# Among countries where the item is fielded (structural NA < this), the median
# skip rate must clear SKIP_NA. Median-over-all-countries was tagging
# country-not-fielded modules (Asian/ESS median NA = 1.0) as follow-ups.
FIELDED_NA = 0.85
SKIP_NA = 0.25
GATE_RANGE = 0.7
GATE_MIN_CLASS = 20
GATE_MAX_CARD = 8
# TYPE1_MIN_N / TYPE1_MIN_UNIQUE live in survey_features.grid_screen (one home
# for grid keep/drop thresholds; imported above).
CHECK_VARS = {
    "afrobarometer": ["Q67A", "Q67B", "Q67"],
    "wvs": ["Q263", "Q266", "Q254"],
    "ess_wave_11": ["rtrd", "mnactic"],
}

INV = ROOT / "data" / "_target_universe_inventory.json"


def _class_y(series: pd.Series, var: str, metadata: dict) -> pd.Series:
    y = series.dropna()
    labelled = _substantive_values(var, metadata)
    if labelled:
        allowed = {str(c).strip() for k in labelled for c in _code_variants(k)}
        keep = y.astype(str).str.strip().isin(allowed)
        if keep.any():
            y = y.loc[keep]
    return y


def _reg_y(series: pd.Series, var: str, metadata: dict) -> pd.Series:
    y = to_ordinal_codes(var, metadata, series)
    keep = y.notna() & substantive_numeric_mask(var, metadata, y)
    oos = out_of_scale_codes(_substantive_values(var, metadata))
    if oos:
        keep &= ~y.isin(oos)
    return y.loc[keep]


def type1_pass(series: pd.Series, var: str, metadata: dict, ttype: str) -> bool:
    problem, _ = TARGET_TYPE_PROBLEM.get(ttype, ("multiclass", "log_loss"))
    if problem == "regression":
        y = _reg_y(series, var, metadata)
        return len(y) >= TYPE1_MIN_N and int(y.nunique()) >= TYPE1_MIN_UNIQUE
    y = _class_y(series, var, metadata)
    if y.empty:
        return False
    counts = y.value_counts()
    return int((counts >= TYPE1_MIN_N).sum()) >= 2


def screen_survey(sid: str, variables: list[str], types: dict[str, str]) -> dict[str, dict]:
    print(f"loading {sid}...", flush=True)
    cfg = os.environ["DATA_CONFIG_PATH"]
    paths = DataPaths.from_yaml(cfg)
    loader = SurveyLoader(paths=paths, verbose=False)
    config = get_survey_config(sid)
    data = loader._load_data(config)
    data = loader._preprocess(data, config)
    metadata = json.loads((META_DIR / SURVEY_META[sid]).read_text(encoding="utf-8"))
    # Registered cleaning amendments (2026-08-31): this script drives the loader's
    # internals directly, so it must apply what load_survey applies.
    data = coalesce_case_twin_columns(data, metadata, sid)
    data, metadata = drop_other_specify(data, metadata, sid)
    ccol = SURVEY_COUNTRY_COL[sid]
    admin_cols = build_admin_cols(metadata, ccol)
    cleaned = clean_question_columns(data, ccol, admin_cols, metadata)
    del data
    countries = [c for c in cleaned[ccol].dropna().unique()]
    n_cty = len(countries)
    print(f"  {sid}: {len(cleaned):,} rows, {n_cty} countries, {len(variables)} vars", flush=True)

    present = [v for v in variables if v in cleaned.columns]
    na_by_cty = {}
    for v in present:
        # structural NA share per country (post-clean isna == structural)
        shares = cleaned.groupby(ccol, observed=True)[v].apply(lambda s: float(s.isna().mean()))
        na_by_cty[v] = shares

    followup = set()
    fielded_median = {}
    n_fielded = {}
    for v, shares in na_by_cty.items():
        fielded = shares[shares < FIELDED_NA]
        n_fielded[v] = int(len(fielded))
        if len(fielded) == 0:
            fielded_median[v] = None
            continue
        med = float(fielded.median())
        fielded_median[v] = med
        if med >= SKIP_NA:
            followup.add(v)

    # Gates: low-cardinality items that split missingness of a follow-up in-country.
    gate = set()
    candidates = []
    for v in present:
        nun = int(cleaned[v].nunique(dropna=True))
        if 2 <= nun <= GATE_MAX_CARD:
            candidates.append(v)
    followup_list = [v for v in present if v in followup]
    print(
        f"  follow-ups {len(followup)} / gate-candidates {len(candidates)} / "
        f"follow-up cols {len(followup_list)}",
        flush=True,
    )

    if followup_list and candidates:
        for country, gdf in cleaned.groupby(ccol, observed=True):
            if len(gdf) < TYPE1_MIN_N:
                continue
            na_share = gdf[followup_list].isna().mean()
            active = [c for c in followup_list if float(na_share[c]) >= SKIP_NA]
            if not active:
                continue
            na = gdf[active].isna().to_numpy()
            for a in candidates:
                if a in gate:
                    continue
                y = gdf[a]
                classes = []
                for cls, cnt in y.value_counts(dropna=True).items():
                    if cnt < GATE_MIN_CLASS:
                        continue
                    mask = (y == cls).to_numpy()
                    classes.append(na[mask].mean(axis=0))
                if len(classes) < 2:
                    continue
                stacked = np.vstack(classes)
                rng = stacked.max(axis=0) - stacked.min(axis=0)
                if float(rng.max()) >= GATE_RANGE:
                    gate.add(a)

    # Type-1 estimability per country, recomputed fresh (the coalesce fix makes
    # the previously stored counts wrong for Asian Barometer).
    n_pass = {v: 0 for v in present}
    print(f"  type-1 recompute over {n_cty} countries x {len(present)} vars", flush=True)
    for _, gdf in cleaned.groupby(ccol, observed=True):
        for v in present:
            if type1_pass(gdf[v], v, metadata, types[v]):
                n_pass[v] += 1

    out = {}
    for v in variables:
        rec = {
            "in_data": v in cleaned.columns,
            "skip_followup": v in followup,
            "skip_gate": v in gate,
            "skip": v in followup or v in gate,
            "n_countries": n_cty,
            "n_fielded": n_fielded.get(v, 0),
            "n_type1_pass": n_pass.get(v, 0),
            "n_type1_fail": n_cty - n_pass.get(v, 0),
            "median_struct_na": None,
            "median_struct_na_fielded": None,
        }
        if v not in cleaned.columns:
            out[v] = rec
            continue
        rec["median_struct_na"] = round(float(na_by_cty[v].median()), 4)
        med_f = fielded_median.get(v)
        rec["median_struct_na_fielded"] = None if med_f is None else round(med_f, 4)
        out[v] = rec

    for v in CHECK_VARS.get(sid, []):
        print(f"  CHECK {sid} {v}: {out.get(v)}", flush=True)

    del cleaned
    return out


def main() -> None:
    inv = json.loads(INV.read_text(encoding="utf-8"))
    rows = inv["rows"]
    by_survey: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_survey[r["survey"]].append(r)

    screens: dict[str, dict[str, dict]] = {}
    for sid in SURVEYS:
        vars_ = [r["variable"] for r in by_survey[sid]]
        types = {r["variable"]: r["type"] for r in by_survey[sid]}
        screens[sid] = screen_survey(sid, vars_, types)

    n_follow = n_gate = n_skip = n_type1_ok = n_type1_zero = 0
    cells_pass = cells_fail = 0
    for r in rows:
        sc = screens[r["survey"]][r["variable"]]
        # Type-1 counts are recomputed by screen_survey (2026-08-31: the Asian
        # case-twin coalesce invalidates the stored counts).
        r.update(sc)
        if sc["skip_followup"]:
            n_follow += 1
        if sc["skip_gate"]:
            n_gate += 1
        if sc["skip"]:
            n_skip += 1
        cells_pass += r.get("n_type1_pass") or 0
        cells_fail += r.get("n_type1_fail") or 0
        if (r.get("n_type1_pass") or 0) >= 1:
            n_type1_ok += 1
        else:
            n_type1_zero += 1

    inv["screen"] = {
        "skip_na_threshold": SKIP_NA,
        "fielded_na_threshold": FIELDED_NA,
        "gate_range": GATE_RANGE,
        "type1_min_n": TYPE1_MIN_N,
        "n_followup": n_follow,
        "n_gate": n_gate,
        "n_skip": n_skip,
        "n_type1_targets_ok": n_type1_ok,
        "n_type1_targets_zero": n_type1_zero,
        "n_type1_cells_pass": cells_pass,
        "n_type1_cells_fail": cells_fail,
    }
    INV.write_text(json.dumps(inv, ensure_ascii=False), encoding="utf-8")
    print("wrote", INV, flush=True)
    print("skip followup", n_follow, "gate", n_gate, "union", n_skip, flush=True)
    print("type1 targets ok", n_type1_ok, "zero-country", n_type1_zero, flush=True)
    print("type1 cells pass", cells_pass, "fail", cells_fail, flush=True)

    SURVEY_IDS = SURVEYS
    TYPES = ["binary", "ordinal", "nominal", "continuous"]
    SECS = [
        "political_attitudes",
        "institutional_trust",
        "social_attitudes",
        "political_participation",
        "wellbeing",
        "contemporary_issues",
        "values_identity",
        "demographics",
    ]

    def xtab(pred):
        st = defaultdict(Counter)
        ss = defaultdict(Counter)
        tags = Counter()
        n = 0
        for r in rows:
            if not pred(r):
                continue
            n += 1
            st[r["survey"]][r["type"]] += 1
            ss[r["survey"]][r["section"] or ""] += 1
            if r["section"] != "demographics" and r.get("topic_tag"):
                tags[r["topic_tag"]] += 1
        return n, st, ss, tags

    combos = {
        "all": lambda r: True,
        "type1": lambda r: r["n_type1_pass"] >= 1,
        "type1_noskip": lambda r: r["n_type1_pass"] >= 1 and not r["skip"],
        "type1_noskip_nodemo": lambda r: (
            r["n_type1_pass"] >= 1 and not r["skip"] and r["section"] != "demographics"
        ),
        "type1_nodemo": lambda r: r["n_type1_pass"] >= 1 and r["section"] != "demographics",
    }
    for name, pred in combos.items():
        n, st, ss, tags = xtab(pred)
        print(f"\n=== {name} n={n} ===")
        print("survey x type")
        for s in SURVEY_IDS:
            print(s, {t: st[s][t] for t in TYPES}, "tot", sum(st[s].values()))
        print("survey x section")
        for s in SURVEY_IDS:
            print(s, {k: ss[s][k] for k in SECS}, "tot", sum(ss[s].values()))
        print("top tags", tags.most_common(20))

    print("\nskip by section", Counter(r["section"] for r in rows if r["skip"]))
    print("type1-zero by section", Counter(r["section"] for r in rows if r["n_type1_pass"] < 1))
    print(
        "type1-zero examples",
        [
            (r["survey"], r["variable"], r["type"], r["section"], r["n_type1_pass"], r["n_type1_fail"])
            for r in rows
            if r["n_type1_pass"] < 1
        ][:25],
    )


if __name__ == "__main__":
    main()
