"""
Assemble the Phase-A pilot grid (plan: i-am-having-doubts-clever-spring).

The pilot is a MACHINERY shakedown, deliberately non-representative: nothing it
produces is a result. Composition rules:

  - 2 targets x 2 countries per survey (all six surveys exercised);
  - two targets get a third country (the country-swap paired-contrast rehearsal):
    ESS bctprd on the legacy Austria/France/Italy trio (direct v4-vs-v3 oracle
    comparison on cached cells) and one Afrobarometer target;
  - all four measurement types (binary / ordinal / nominal / continuous) appear
    somewhere in the grid;
  - candidates come from data/_target_universe_inventory.json: in_data, not
    skip/follow-up, type-1 pass in >= 80% of fielded countries (so mainstream
    countries almost surely pass — the oracle still fails loudly if one doesn't);
  - leakage-audited targets are excluded except the two deliberate ESS overlaps
    (bctprd, stfgov — both 'genuine' in the audit).

Writes data/pilot_cells.csv (survey,target,country,type,section,role).
Deterministic: no randomness, ties broken by variable name.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from survey_features.config import OUTPUTS_DIR  # noqa: E402
from survey_features.layout import leakage_audit_csv_path  # noqa: E402

INVENTORY = ROOT / "data" / "_target_universe_inventory.json"
OUT_CSV = ROOT / "data" / "pilot_cells.csv"

# Mainstream fielded countries per survey; third entry = swap-rehearsal extra.
COUNTRIES = {
    "afrobarometer":  ["Kenya", "Ghana", "Nigeria"],
    "arabbarometer":  ["Jordan", "Morocco"],
    # NB: Japan is not fielded in this ABS wave (valid: Australia, Cambodia,
    # Indonesia, Korea, Mongolia, Philippines, Taiwan, Thailand, Vietnam).
    "asianbarometer": ["Taiwan", "Philippines"],
    "ess_wave_11":    ["Austria", "France", "Italy"],
    "latinobarometer": ["Chile", "Colombia"],
    "wvs":            ["Germany", "Brazil"],
}

# Deliberate legacy overlaps (cached v3 oracles exist for Austria/France/Italy).
ESS_OVERLAP = ["bctprd", "stfgov"]

# Which survey covers which "rare" type so all four types appear in the grid.
# Everything else defaults to one binary + one ordinal.
TYPE_ASSIGNMENTS = {
    "afrobarometer":  ["binary", "continuous"],
    "arabbarometer":  ["binary", "ordinal"],
    "asianbarometer": ["ordinal", "binary"],
    "latinobarometer": ["nominal", "ordinal"],
    "wvs":            ["binary", "ordinal"],
}

SWAP_TARGET_SURVEYS = {"ess_wave_11", "afrobarometer"}  # 3-country targets

MIN_PASS_SHARE = 0.8
MAX_NOMINAL_CLASSES = 8
PREFERRED_SECTIONS = (
    "political_attitudes", "institutional_trust", "social_attitudes",
    "political_participation", "wellbeing", "contemporary_issues",
)


def _nominal_class_count(row: dict) -> int | None:
    m = re.search(r"(\d+)\s+(?:unordered\s+)?categor", str(row.get("type_why", "")))
    return int(m.group(1)) if m else None


def load_candidates() -> pd.DataFrame:
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    df = pd.DataFrame(inv["rows"])
    df = df[df["in_data"] & ~df["skip"]]
    df = df[df["n_type1_pass"] >= MIN_PASS_SHARE * df["n_countries"]]
    return df


def audited_targets() -> set[str]:
    path = leakage_audit_csv_path(OUTPUTS_DIR)
    if not path.is_file():
        return set()
    return set(pd.read_csv(path)["target"].astype(str))


def pick(df: pd.DataFrame, survey: str, ttype: str, taken: set[str]) -> dict:
    sub = df[(df["survey"] == survey) & (df["type"] == ttype)
             & ~df["variable"].isin(taken)].copy()
    if ttype == "nominal":
        sub["n_classes"] = sub.apply(_nominal_class_count, axis=1)
        sub = sub[sub["n_classes"].notna() & (sub["n_classes"] <= MAX_NOMINAL_CLASSES)]
    # Prefer substantive sections, then breadth of type-1 support, then name.
    sub["section_rank"] = sub["section"].map(
        {s: i for i, s in enumerate(PREFERRED_SECTIONS)}
    ).fillna(len(PREFERRED_SECTIONS))
    sub = sub.sort_values(["section_rank", "n_type1_pass", "variable"],
                          ascending=[True, False, True])
    if sub.empty:
        raise SystemExit(f"No pilot candidate for {survey} / {ttype}")
    return sub.iloc[0].to_dict()


def main() -> None:
    df = load_candidates()
    legacy = audited_targets() - set(ESS_OVERLAP)
    df = df[~df["variable"].isin(legacy)]

    rows, taken = [], set()

    def add(survey: str, rec: dict, extra_country: bool) -> None:
        taken.add(rec["variable"])
        countries = COUNTRIES[survey][: (3 if extra_country else 2)]
        for i, country in enumerate(countries):
            rows.append({
                "survey": survey,
                "target": rec["variable"],
                "country": country,
                "type": rec["type"],
                "section": rec.get("section", ""),
                "role": "swap_extra" if (extra_country and i == 2) else "core",
            })

    # ESS: the two legacy-overlap targets; bctprd carries the swap rehearsal.
    ess = df[df["survey"] == "ess_wave_11"].set_index("variable")
    for var in ESS_OVERLAP:
        rec = (ess.loc[var].to_dict() | {"variable": var}) if var in ess.index else {
            "variable": var,
            "type": {"bctprd": "binary", "stfgov": "ordinal"}[var],
            "section": "legacy_overlap",
        }
        add("ess_wave_11", rec, extra_country=(var == "bctprd"))

    for survey, types in TYPE_ASSIGNMENTS.items():
        for j, ttype in enumerate(types):
            rec = pick(df, survey, ttype, taken)
            add(survey, rec, extra_country=(survey in SWAP_TARGET_SURVEYS and j == 0))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"[pilot-grid] {len(out)} cells, {out['target'].nunique()} targets -> {OUT_CSV}")
    print(out.groupby(["survey", "target", "type"])["country"]
             .apply(lambda s: ", ".join(s)).to_string())
    missing_types = {"binary", "ordinal", "nominal", "continuous"} - set(out["type"])
    if missing_types:
        raise SystemExit(f"type coverage gap: {missing_types}")


if __name__ == "__main__":
    main()
