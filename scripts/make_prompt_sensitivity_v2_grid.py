"""Seeded 24-question x 3-country grid for prompt-sensitivity v2.

Writes data/prompt_sensitivity_v2_cells.yaml. Re-run is deterministic for a seed;
do not hand-edit the yaml to swap items (leakage replacements go through --replace).

  python scripts/make_prompt_sensitivity_v2_grid.py
  python scripts/make_prompt_sensitivity_v2_grid.py --seed 20260819
"""
from __future__ import annotations

import argparse
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
OUT_YAML = PKG_ROOT / "data" / "prompt_sensitivity_v2_cells.yaml"

SEED = 20260819
N_CORE_PER_STRATUM = 2
N_COUNTRIES = 3
MIN_CLASSIFICATION = 3
MAX_RESEEDS = 20

SURVEYS = [
    "afrobarometer",
    "arabbarometer",
    "asianbarometer",
    "ess_wave_11",
    "latinobarometer",
    "wvs",
]
STRATUM_SECTIONS = {
    "political_institutional": (
        "political_attitudes",
        "institutional_trust",
        "political_participation",
    ),
    "everyday_person": (
        "social_attitudes",
        "wellbeing",
        "values_identity",
    ),
}
CLASSIFICATION_TYPES = frozenset({"binary", "nominal"})
TEST_VAR = re.compile(r"^test", re.I)

# WVS wave-7-ish regions for the 3-country panel. Unlisted names -> "other".
WVS_REGION = {
    "Andorra": "western_europe",
    "Argentina": "latin_america",
    "Australia": "settler_oceania",
    "Bangladesh": "south_asia",
    "Bolivia": "latin_america",
    "Brazil": "latin_america",
    "Canada": "north_america",
    "Chile": "latin_america",
    "China": "east_asia",
    "Colombia": "latin_america",
    "Cyprus": "western_europe",
    "Czechia": "eastern_europe",
    "Czech Republic": "eastern_europe",
    "Ecuador": "latin_america",
    "Egypt": "mena",
    "Ethiopia": "africa",
    "Germany": "western_europe",
    "Great Britain": "western_europe",
    "Greece": "western_europe",
    "Guatemala": "latin_america",
    "Hong Kong SAR": "east_asia",
    "Indonesia": "se_asia",
    "Iran": "mena",
    "Iraq": "mena",
    "Japan": "east_asia",
    "Jordan": "mena",
    "Kazakhstan": "central_asia",
    "Kenya": "africa",
    "Kyrgyzstan": "central_asia",
    "Lebanon": "mena",
    "Libya": "mena",
    "Macao SAR": "east_asia",
    "Malaysia": "se_asia",
    "Maldives": "south_asia",
    "Mexico": "latin_america",
    "Mongolia": "east_asia",
    "Morocco": "mena",
    "Myanmar": "se_asia",
    "Netherlands": "western_europe",
    "New Zealand": "settler_oceania",
    "Nicaragua": "latin_america",
    "Nigeria": "africa",
    "Pakistan": "south_asia",
    "Peru": "latin_america",
    "Philippines": "se_asia",
    "Puerto Rico": "latin_america",
    "Romania": "eastern_europe",
    "Russia": "eastern_europe",
    "Serbia": "eastern_europe",
    "Singapore": "se_asia",
    "Slovakia": "eastern_europe",
    "South Korea": "east_asia",
    "Taiwan ROC": "east_asia",
    "Taiwan": "east_asia",
    "Tajikistan": "central_asia",
    "Thailand": "se_asia",
    "Tunisia": "mena",
    "Turkey": "mena",
    "Ukraine": "eastern_europe",
    "United States": "north_america",
    "Uruguay": "latin_america",
    "Uzbekistan": "central_asia",
    "Venezuela": "latin_america",
    "Vietnam": "se_asia",
    "Zimbabwe": "africa",
}


def _load_type1_pass():
    path = ROOT / "scripts" / "target_universe_screen.py"
    spec = importlib.util.spec_from_file_location("tus_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.type1_pass


def eligible_rows(inv_rows: list[dict]) -> list[dict]:
    out = []
    for r in inv_rows:
        if not r.get("in_data"):
            continue
        if r.get("skip") or r.get("skip_gate") or r.get("skip_followup"):
            continue
        if r.get("section") == "demographics":
            continue
        if TEST_VAR.match(str(r.get("variable") or "")):
            continue
        if int(r.get("n_type1_pass") or 0) < N_COUNTRIES:
            continue
        out.append(r)
    return out


def stratum_of(section: str) -> str | None:
    for name, secs in STRATUM_SECTIONS.items():
        if section in secs:
            return name
    return None


def draw_from_pool(pool: list[dict], n_core: int, rng: np.random.Generator) -> tuple[list[dict], dict | None]:
    if len(pool) < n_core:
        raise RuntimeError(f"Need {n_core} candidates, have {len(pool)}")
    order = rng.permutation(len(pool))
    core = [pool[int(i)] for i in order[:n_core]]
    spare = pool[int(order[n_core])] if len(pool) > n_core else None
    return core, spare


def pick_k(names: list[str], k: int, rng: np.random.Generator, *, wvs: bool) -> list[str]:
    names = sorted(set(names))
    if len(names) < k:
        raise RuntimeError(f"Need {k} countries, have {len(names)}: {names}")
    if not wvs:
        idx = rng.choice(len(names), size=k, replace=False)
        return sorted(names[int(i)] for i in idx)
    by: dict[str, list[str]] = defaultdict(list)
    for n in names:
        by[WVS_REGION.get(n, "other")].append(n)
    for reg in by:
        rng.shuffle(by[reg])
    regions = list(by.keys())
    rng.shuffle(regions)
    picked: list[str] = []
    while len(picked) < k and any(by[r] for r in regions):
        for reg in regions:
            if by[reg] and len(picked) < k:
                picked.append(by[reg].pop())
    if len(picked) < k:
        raise RuntimeError(f"WVS region draw short: {picked}")
    return sorted(picked)


def _type1_countries_for_vars(
    survey_id: str,
    variables: list[str],
    types: dict[str, str],
    type1_pass,
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
            name = code_to_name.get(code, str(code))
            if type1_pass(g[var], var, metadata, types[var]):
                names.append(name)
        out[var] = sorted(set(names))
    del cleaned
    return out


def _intersection(maps: dict[str, list[str]], variables: list[str]) -> set[str]:
    sets = [set(maps[v]) for v in variables]
    if not sets:
        return set()
    inter = sets[0]
    for s in sets[1:]:
        inter &= s
    return inter


def _qrec(r: dict, stratum: str, role: str) -> dict:
    return {
        "target": r["variable"],
        "type": r["type"],
        "section": r["section"],
        "stratum": stratum,
        "role": role,
        "n_type1_pass": int(r.get("n_type1_pass") or 0),
    }


def draw_questions(elig: list[dict], rng: np.random.Generator) -> dict[str, dict]:
    by_survey: dict[str, dict] = {}
    for survey in SURVEYS:
        srows = [r for r in elig if r["survey"] == survey]
        cores: list[dict] = []
        spares: list[dict] = []
        for stratum, secs in STRATUM_SECTIONS.items():
            pool = [r for r in srows if r.get("section") in secs]
            pool = sorted(pool, key=lambda r: str(r["variable"]))
            if len(pool) < N_CORE_PER_STRATUM:
                raise RuntimeError(
                    f"{survey} / {stratum}: {len(pool)} eligible questions, "
                    f"need {N_CORE_PER_STRATUM}"
                )
            core, spare = draw_from_pool(pool, N_CORE_PER_STRATUM, rng)
            cores.extend(_qrec(r, stratum, "core") for r in core)
            if spare is not None:
                spares.append(_qrec(spare, stratum, "spare"))
        by_survey[survey] = {"questions": cores, "spares": spares}
    return by_survey


def n_classification(by_survey: dict[str, dict]) -> int:
    n = 0
    for spec in by_survey.values():
        for q in spec["questions"]:
            if q["type"] in CLASSIFICATION_TYPES:
                n += 1
    return n


def assign_country_panels(
    by_survey: dict[str, dict],
    rng: np.random.Generator,
    type1_pass,
) -> None:
    for survey, spec in by_survey.items():
        cores = spec["questions"]
        spares = spec["spares"]
        need_vars = [q["target"] for q in cores] + [s["target"] for s in spares]
        types = {q["target"]: q["type"] for q in cores + spares}
        print(f"[type1] {survey}: {need_vars}", flush=True)
        cmap = _type1_countries_for_vars(survey, need_vars, types, type1_pass)
        for q in cores + spares:
            elig_cty = cmap.get(q["target"], [])
            q["n_eligible_countries"] = len(elig_cty)

        core_vars = [q["target"] for q in cores]
        inter = _intersection(cmap, core_vars)
        used_spare = None
        if len(inter) < N_COUNTRIES:
            best = (len(inter), None, inter)
            spare_by_stratum = {s["stratum"]: s for s in spares}
            for i, q in enumerate(cores):
                sp = spare_by_stratum.get(q["stratum"])
                if sp is None:
                    continue
                trial = core_vars[:]
                trial[i] = sp["target"]
                tinter = _intersection(cmap, trial)
                if len(tinter) > best[0]:
                    best = (len(tinter), i, tinter)
            if best[1] is not None and best[0] >= N_COUNTRIES:
                i = best[1]
                old = cores[i]
                sp = spare_by_stratum[old["stratum"]]
                cores[i] = {**sp, "role": "core"}
                spares[:] = [
                    s for s in spares if s["target"] != sp["target"]
                ] + [{**old, "role": "spare"}]
                inter = best[2]
                used_spare = (old["target"], cores[i]["target"])
                core_vars = [q["target"] for q in cores]
            else:
                inter = best[2]

        if len(inter) < N_COUNTRIES:
            raise RuntimeError(
                f"{survey}: shared type-1 intersection is {sorted(inter)} "
                f"(need {N_COUNTRIES}) for {core_vars}"
            )
        countries = pick_k(
            sorted(inter), N_COUNTRIES, rng, wvs=(survey == "wvs"),
        )
        spec["countries"] = countries
        spec["shared_eligible_n"] = len(inter)
        spec["used_spare_swap"] = (
            {"replaced": used_spare[0], "with": used_spare[1]}
            if used_spare else None
        )


def cells_from(by_survey: dict[str, dict]) -> list[dict]:
    rows = []
    for survey, spec in by_survey.items():
        for q in spec["questions"]:
            for country in spec["countries"]:
                rows.append({
                    "survey": survey,
                    "target": q["target"],
                    "country": country,
                    "type": q["type"],
                    "section": q["section"],
                    "stratum": q["stratum"],
                })
    return rows


def dump_yaml(doc: dict, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT_YAML)
    args = parser.parse_args()
    if not INVENTORY.is_file():
        raise SystemExit(f"Missing {INVENTORY}; run scripts/target_universe_screen.py")
    if "DATA_CONFIG_PATH" not in os.environ:
        raise SystemExit("DATA_CONFIG_PATH is not set")

    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    elig = eligible_rows(inv["rows"])
    type1_pass = _load_type1_pass()

    seed_used = int(args.seed)
    by_survey = None
    for attempt in range(MAX_RESEEDS):
        rng = np.random.default_rng(seed_used)
        by_survey = draw_questions(elig, rng)
        n_cls = n_classification(by_survey)
        if n_cls >= MIN_CLASSIFICATION:
            print(f"[draw] seed={seed_used} classification questions={n_cls}")
            break
        print(
            f"[draw] seed={seed_used} classification={n_cls} < {MIN_CLASSIFICATION}; reseed"
        )
        seed_used += 1
    else:
        raise SystemExit("Could not meet classification floor")

    rng_c = np.random.default_rng(seed_used + 10_000)
    assign_country_panels(by_survey, rng_c, type1_pass)
    cells = cells_from(by_survey)
    n_q = sum(len(s["questions"]) for s in by_survey.values())
    n_cty = len({c["country"] for c in cells})
    doc = {
        "experiment": "prompt_sensitivity_v2",
        "seed_requested": int(args.seed),
        "seed_used": seed_used,
        "n_questions": n_q,
        "n_countries_per_survey": N_COUNTRIES,
        "n_cells": len(cells),
        "n_unique_countries": n_cty,
        "condition": "country_provided",
        "min_classification_questions": MIN_CLASSIFICATION,
        "n_classification_questions": n_classification(by_survey),
        "notes": (
            "24 questions (2 per survey x theme stratum) on a shared 3-country "
            "panel per survey. Stage 1 is country-named only. Do not hand-swap "
            "items; leakage/unestimable cells use the same-survey-same-stratum spare."
        ),
        "surveys": by_survey,
        "cells": cells,
    }
    dump_yaml(doc, args.out)
    print(f"[ok] {len(cells)} cells, {n_q} questions, {n_cty} unique countries -> {args.out}")
    for survey, spec in by_survey.items():
        qs = ", ".join(q["target"] for q in spec["questions"])
        print(f"  {survey}: {spec['countries']} | {qs}")


if __name__ == "__main__":
    main()
