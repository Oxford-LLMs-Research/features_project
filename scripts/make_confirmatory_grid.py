"""Seeded confirmatory-grid draw: 20 questions/survey, nested country permutation.

Implements the registered draw over the frozen frame (data/frame_rules.yaml,
docs/frame_freeze_review_2026-08.md) with the 2026-08-31 amendments recorded in
docs/pre_paper_run_decisions.md:

  * 20 core questions per survey (was 15), type-stratified proportional to the
    draw-eligible pool with a floor of 2 per answer type present; sections are
    balanced inside each type by round-robin (soft theme spread, no hard quota).
  * Nested country rule: one seeded permutation of each question's type-1
    countries (WVS ordered region-round-robin for spread). The first
    min(10, roster) countries are the ORACLE set (computed on the cluster);
    the first 3 of those are the LLM CONFIRMATORY countries. The confirmatory
    subdraw is therefore fixed before any oracle heterogeneity exists.
  * 3 spare questions per survey; replacement rule: same survey, same type
    (first unused spare of that type; if that pool is exhausted, the next
    unused spare of any type — record the substitution in the registry).

Writes data/confirmatory_grid.yaml (full record) and
data/confirmatory_grid_cells.csv (flat oracle grid for the SLURM array; the
LLM grid is the role == "confirmatory" subset).

  python scripts/make_confirmatory_grid.py
  python scripts/make_confirmatory_grid.py --seed 20260831

Deterministic for a seed, with isolated sub-streams: each survey's question
draw and each question's country permutation use a child RNG keyed by
(seed, survey[, target]), so a pool change in one survey (e.g. a re-freeze
after a data fix) never reshuffles the draws of the others.
Do not hand-edit outputs; replacements go through the spare rule.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survey_features.config import ROOT as PKG_ROOT  # noqa: E402  loads .env
from survey_features.surveys import (  # noqa: E402
    SURVEY_COUNTRY_COL,
    build_admin_cols,
    build_country_code_map,
    clean_question_columns,
    load_survey,
)

INVENTORY = PKG_ROOT / "data" / "_target_universe_inventory.json"
FRAME_RULES = PKG_ROOT / "data" / "frame_rules.yaml"
OUT_YAML = PKG_ROOT / "data" / "confirmatory_grid.yaml"
OUT_CSV = PKG_ROOT / "data" / "confirmatory_grid_cells.csv"

SEED = 20260831
N_CORE = 20
N_SPARES = 3
TYPE_FLOOR = 2
N_ORACLE_COUNTRIES = 10
N_CONFIRMATORY_COUNTRIES = 3

# Registered counts from the 2026-08-31 re-freeze (post Asian case-twin coalesce
# and other-specify removal); the draw refuses to run if the rules reproduce
# anything else (frame drift = re-freeze first).
EXPECT_FRAME = 1278
EXPECT_DRAW_ELIGIBLE = 1179

SURVEYS = [
    "afrobarometer",
    "arabbarometer",
    "asianbarometer",
    "ess_wave_11",
    "latinobarometer",
    "wvs",
]

# Same wave-7-ish region map as make_prompt_sensitivity_v2_grid.py; unlisted -> "other".
WVS_REGION = {
    "Andorra": "western_europe", "Argentina": "latin_america",
    "Australia": "settler_oceania", "Bangladesh": "south_asia",
    "Bolivia": "latin_america", "Brazil": "latin_america",
    "Canada": "north_america", "Chile": "latin_america",
    "China": "east_asia", "Colombia": "latin_america",
    "Cyprus": "western_europe", "Czechia": "eastern_europe",
    "Czech Republic": "eastern_europe", "Ecuador": "latin_america",
    "Egypt": "mena", "Ethiopia": "africa", "Germany": "western_europe",
    "Great Britain": "western_europe", "Greece": "western_europe",
    "Guatemala": "latin_america", "Hong Kong SAR": "east_asia",
    "Indonesia": "se_asia", "Iran": "mena", "Iraq": "mena",
    "Japan": "east_asia", "Jordan": "mena", "Kazakhstan": "central_asia",
    "Kenya": "africa", "Kyrgyzstan": "central_asia", "Lebanon": "mena",
    "Libya": "mena", "Macao SAR": "east_asia", "Malaysia": "se_asia",
    "Maldives": "south_asia", "Mexico": "latin_america",
    "Mongolia": "east_asia", "Morocco": "mena", "Myanmar": "se_asia",
    "Netherlands": "western_europe", "New Zealand": "settler_oceania",
    "Nicaragua": "latin_america", "Nigeria": "africa",
    "Pakistan": "south_asia", "Peru": "latin_america",
    "Philippines": "se_asia", "Puerto Rico": "latin_america",
    "Romania": "eastern_europe", "Russia": "eastern_europe",
    "Serbia": "eastern_europe", "Singapore": "se_asia",
    "Slovakia": "eastern_europe", "South Korea": "east_asia",
    "Taiwan ROC": "east_asia", "Taiwan": "east_asia",
    "Tajikistan": "central_asia", "Thailand": "se_asia",
    "Tunisia": "mena", "Turkey": "mena", "Ukraine": "eastern_europe",
    "United States": "north_america", "Uruguay": "latin_america",
    "Uzbekistan": "central_asia", "Venezuela": "latin_america",
    "Vietnam": "se_asia", "Zimbabwe": "africa",
}


def _child_rng(seed: int, *parts: str) -> np.random.Generator:
    """Deterministic child RNG keyed by (seed, *parts) — isolates sub-draws."""
    key = [int(seed)] + [ord(ch) for p in parts for ch in p]
    return np.random.default_rng(key)


def _load_type1_pass():
    path = ROOT / "scripts" / "target_universe_screen.py"
    spec = importlib.util.spec_from_file_location("tus_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.type1_pass


def _load_frame_rules() -> dict:
    return yaml.safe_load(FRAME_RULES.read_text(encoding="utf-8"))


def frame_rows(inv_rows: list[dict], rules: dict) -> list[dict]:
    """Apply the frozen eligibility rules; returns the frame (>=1 estimable country)."""
    elig = rules["eligibility"]
    rescue = {(r["survey"], r["variable"]) for r in rules.get("rescue_whitelist", [])}
    pats = [
        (p["survey"], re.compile(p["pattern"], re.I))
        for p in elig.get("exclude_name_patterns", [])
    ]
    out = []
    for r in inv_rows:
        key = (r["survey"], r["variable"])
        if not r.get("in_data"):
            continue
        if elig.get("exclude_followups") and r.get("skip_followup"):
            continue
        # skip items off (memo Q1) — but keep_gates: a row whose only skip flag
        # is the gate flag stays in the frame, carrying gate=true into the grid.
        if r.get("skip") and not r.get("skip_gate"):
            continue
        if r.get("section") in elig.get("exclude_sections", []) and key not in rescue:
            continue
        if any(s == r["survey"] and p.match(str(r.get("variable") or "")) for s, p in pats):
            continue
        if int(r.get("n_type1_pass") or 0) < int(elig["min_estimable_countries_frame"]):
            continue
        out.append(r)
    return out


def allocate_by_type(pools: dict[str, int], n: int, floor: int) -> dict[str, int]:
    """Largest-remainder proportional allocation with a per-type floor, capped by pool."""
    types = [t for t, c in pools.items() if c > 0]
    total = sum(pools[t] for t in types)
    alloc = {t: min(floor, pools[t]) for t in types}
    if sum(alloc.values()) > n:
        raise RuntimeError(f"Type floors alone exceed n={n}: {alloc}")
    remaining = n - sum(alloc.values())
    quota = {t: remaining * pools[t] / total for t in types}
    add = {t: min(int(quota[t]), pools[t] - alloc[t]) for t in types}
    remaining -= sum(add.values())
    frac = sorted(
        types, key=lambda t: (quota[t] - int(quota[t]), pools[t]), reverse=True
    )
    while remaining > 0:
        progressed = False
        for t in frac:
            if remaining <= 0:
                break
            if alloc[t] + add[t] < pools[t]:
                add[t] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError(f"Pools exhausted before n={n}: {pools}")
    return {t: alloc[t] + add[t] for t in types}


def section_round_robin(pool: list[dict], k: int, rng: np.random.Generator) -> list[dict]:
    """Draw k questions spreading sections: shuffle within section, rotate sections."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: str(r["variable"])):
        by[str(r.get("section"))].append(r)
    for sec in by:
        idx = rng.permutation(len(by[sec]))
        by[sec] = [by[sec][int(i)] for i in idx]
    secs = sorted(by.keys())
    order = rng.permutation(len(secs))
    secs = [secs[int(i)] for i in order]
    picked: list[dict] = []
    while len(picked) < k:
        progressed = False
        for sec in secs:
            if by[sec] and len(picked) < k:
                picked.append(by[sec].pop())
                progressed = True
        if not progressed:
            raise RuntimeError(f"Pool exhausted at {len(picked)}/{k}")
    return picked


def order_countries(names: list[str], rng: np.random.Generator, *, wvs: bool) -> list[str]:
    """Full seeded permutation; WVS rotates regions so every prefix is region-spread."""
    names = sorted(set(names))
    if not wvs:
        idx = rng.permutation(len(names))
        return [names[int(i)] for i in idx]
    by: dict[str, list[str]] = defaultdict(list)
    for n in names:
        by[WVS_REGION.get(n, "other")].append(n)
    for reg in by:
        idx = rng.permutation(len(by[reg]))
        by[reg] = [by[reg][int(i)] for i in idx]
    regions = sorted(by.keys())
    order = rng.permutation(len(regions))
    regions = [regions[int(i)] for i in order]
    out: list[str] = []
    while any(by[r] for r in regions):
        for reg in regions:
            if by[reg]:
                out.append(by[reg].pop())
    return out


def type1_countries_for_vars(
    survey_id: str, variables: list[str], types: dict[str, str], type1_pass,
) -> dict[str, list[str]]:
    data, metadata = load_survey(survey_id, os.environ["DATA_CONFIG_PATH"])
    ccol = SURVEY_COUNTRY_COL[survey_id]
    admin_cols = build_admin_cols(metadata, ccol)
    cleaned = clean_question_columns(data, ccol, admin_cols, metadata)
    del data
    country_codes = build_country_code_map(metadata, ccol, cleaned)
    code_to_name = {code: name for name, code in country_codes.items()}
    out: dict[str, list[str]] = {}
    for var in variables:
        if var not in cleaned.columns:
            out[var] = []
            continue
        names = []
        for code, g in cleaned.groupby(ccol, observed=True):
            if type1_pass(g[var], var, metadata, types[var]):
                names.append(code_to_name.get(code, str(code)))
        out[var] = sorted(set(names))
    del cleaned
    return out


def qrec(r: dict, role: str) -> dict:
    return {
        "target": r["variable"],
        "type": r["type"],
        "section": r["section"],
        "gate": bool(r.get("skip_gate")),
        "role": role,
        "n_type1_pass": int(r.get("n_type1_pass") or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not INVENTORY.is_file():
        raise SystemExit(f"Missing {INVENTORY}; run scripts/target_universe_screen.py")
    if "DATA_CONFIG_PATH" not in os.environ:
        raise SystemExit("DATA_CONFIG_PATH is not set")

    rules = _load_frame_rules()
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    frame = frame_rows(inv["rows"], rules)
    min_draw = int(rules["eligibility"]["min_estimable_countries_draw"])
    elig = [r for r in frame if int(r.get("n_type1_pass") or 0) >= min_draw]

    n_gate_frame = sum(1 for r in frame if r.get("skip_gate"))
    n_gate_elig = sum(1 for r in elig if r.get("skip_gate"))
    print(f"[frame] {len(frame)} questions ({n_gate_frame} gate-flagged); "
          f"draw-eligible >= {min_draw}: {len(elig)} ({n_gate_elig} gate-flagged)")
    if len(frame) != EXPECT_FRAME or len(elig) != EXPECT_DRAW_ELIGIBLE:
        raise SystemExit(
            f"Frame drift: got {len(frame)}/{len(elig)}, frozen counts are "
            f"{EXPECT_FRAME}/{EXPECT_DRAW_ELIGIBLE}. Re-freeze before drawing."
        )

    type1_pass = _load_type1_pass()
    by_survey: dict[str, dict] = {}

    for survey in SURVEYS:
        rng = _child_rng(args.seed, survey)
        srows = [r for r in elig if r["survey"] == survey]
        pools_by_type: dict[str, list[dict]] = defaultdict(list)
        for r in srows:
            pools_by_type[str(r["type"])].append(r)
        pool_sizes = {t: len(v) for t, v in pools_by_type.items()}
        alloc = allocate_by_type(pool_sizes, N_CORE, TYPE_FLOOR)
        print(f"[alloc] {survey}: pools={pool_sizes} -> {alloc}")

        cores: list[dict] = []
        remaining: dict[str, list[dict]] = {}
        for t in sorted(alloc):
            picked = section_round_robin(pools_by_type[t], alloc[t], rng)
            cores.extend(qrec(r, "core") for r in picked)
            taken = {r["variable"] for r in picked}
            remaining[t] = [r for r in pools_by_type[t] if r["variable"] not in taken]

        # Spares: one from each of the largest allocations, next in section order.
        spares: list[dict] = []
        for t in sorted(alloc, key=lambda t: alloc[t], reverse=True):
            if len(spares) >= N_SPARES:
                break
            if remaining.get(t):
                sp = section_round_robin(remaining[t], 1, rng)[0]
                spares.append(qrec(sp, "spare"))
                remaining[t] = [
                    r for r in remaining[t] if r["variable"] != sp["variable"]
                ]

        need = [q["target"] for q in cores + spares]
        types = {q["target"]: q["type"] for q in cores + spares}
        print(f"[type1] {survey}: {len(need)} questions", flush=True)
        cmap = type1_countries_for_vars(survey, need, types, type1_pass)

        for q in cores + spares:
            names = cmap.get(q["target"], [])
            if len(names) < N_CONFIRMATORY_COUNTRIES:
                raise RuntimeError(
                    f"{survey} {q['target']}: {len(names)} type-1 countries on "
                    f"recompute, inventory said {q['n_type1_pass']}"
                )
            rng_q = _child_rng(args.seed, survey, "countries", str(q["target"]))
            perm = order_countries(names, rng_q, wvs=(survey == "wvs"))
            q["countries_oracle"] = perm[:min(N_ORACLE_COUNTRIES, len(perm))]
            q["countries_confirmatory"] = perm[:N_CONFIRMATORY_COUNTRIES]
            q["n_eligible_countries"] = len(names)

        by_survey[survey] = {"questions": cores, "spares": spares}

    cells: list[dict] = []
    for survey, spec in by_survey.items():
        for q in spec["questions"]:
            for rank, country in enumerate(q["countries_oracle"], start=1):
                cells.append({
                    "survey": survey,
                    "target": q["target"],
                    "country": country,
                    "type": q["type"],
                    "section": q["section"],
                    "gate": q["gate"],
                    "country_rank": rank,
                    "role": (
                        "confirmatory"
                        if rank <= N_CONFIRMATORY_COUNTRIES else "oracle_only"
                    ),
                })

    n_q = sum(len(s["questions"]) for s in by_survey.values())
    n_conf = sum(1 for c in cells if c["role"] == "confirmatory")
    n_cty = len({c["country"] for c in cells})
    doc = {
        "grid": "confirmatory_2026-08",
        "seed": int(args.seed),
        "frame_frozen": rules["frozen"],
        "n_questions_per_survey": N_CORE,
        "n_spares_per_survey": N_SPARES,
        "type_floor": TYPE_FLOOR,
        "n_oracle_countries": N_ORACLE_COUNTRIES,
        "n_confirmatory_countries": N_CONFIRMATORY_COUNTRIES,
        "n_questions": n_q,
        "n_oracle_cells": len(cells),
        "n_confirmatory_cells": n_conf,
        "n_unique_countries": n_cty,
        "notes": (
            "Nested country rule: seeded permutation per question (WVS "
            "region-round-robin); first min(10, roster) = oracle set, first 3 = "
            "LLM confirmatory. Confirmatory countries fixed before oracles run. "
            "Transportability pairs are chosen later from oracle_only+confirmatory "
            "cells by the registered max-disagreement rule. Replacement rule: "
            "same-survey same-type spare, else next unused spare; log in registry."
        ),
        "surveys": by_survey,
    }
    OUT_YAML.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    cols = ["survey", "target", "country", "type", "section", "gate",
            "country_rank", "role"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(cells)

    print(f"[ok] {n_q} questions, {len(cells)} oracle cells "
          f"({n_conf} confirmatory), {n_cty} unique countries")
    print(f"[ok] -> {OUT_YAML}")
    print(f"[ok] -> {OUT_CSV}")
    for survey, spec in by_survey.items():
        n_oc = sum(len(q["countries_oracle"]) for q in spec["questions"])
        types_drawn = defaultdict(int)
        for q in spec["questions"]:
            types_drawn[q["type"]] += 1
        print(f"  {survey}: {dict(types_drawn)}; oracle cells {n_oc}; "
              f"spares {[s['target'] for s in spec['spares']]}")


if __name__ == "__main__":
    main()
