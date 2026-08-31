"""Frozen prompt-sensitivity v2 grid: 24 questions x 3 shared countries."""

from pathlib import Path

import yaml

from survey_features.config import ROOT
from survey_features.layout import (
    prompt_sensitivity_v2_dirs,
    prompt_sensitivity_v2_root,
    prompt_sensitivity_v2_scores_path,
)

YAML = ROOT / "data" / "prompt_sensitivity_v2_cells.yaml"


def test_v2_grid_is_24x3_with_shared_panels():
    doc = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    assert doc["seed_used"] == 20260819
    assert doc["n_questions"] == 24
    assert doc["n_cells"] == 72
    assert doc["n_unique_countries"] == 18
    assert doc["condition"] == "country_provided"
    assert doc["n_classification_questions"] >= 3
    assert len(doc["cells"]) == 72
    surveys = doc["surveys"]
    assert set(surveys) == {
        "afrobarometer",
        "arabbarometer",
        "asianbarometer",
        "ess_wave_11",
        "latinobarometer",
        "wvs",
    }
    for spec in surveys.values():
        assert len(spec["countries"]) == 3
        cores = spec["questions"]
        assert len(cores) == 4
        strata = {q["stratum"] for q in cores}
        assert strata == {"political_institutional", "everyday_person"}
        assert sum(1 for q in cores if q["stratum"] == "political_institutional") == 2
        assert sum(1 for q in cores if q["stratum"] == "everyday_person") == 2
        for q in cores:
            for country in spec["countries"]:
                row = {
                    "survey": None,
                    "target": q["target"],
                    "country": country,
                    "stratum": q["stratum"],
                }
                # filled below
                del row
    cell_keys = {(r["survey"], r["target"], r["country"]) for r in doc["cells"]}
    expected = set()
    for sid, spec in surveys.items():
        for q in spec["questions"]:
            for c in spec["countries"]:
                expected.add((sid, q["target"], c))
    assert cell_keys == expected


def test_v2_layout_keeps_replicates_out_of_v1_tree():
    root = prompt_sensitivity_v2_root(Path("/tmp/out"))
    assert root.as_posix().endswith("experiments/prompt_sensitivity_v2")
    gen, ext, maps = prompt_sensitivity_v2_dirs("kimi", "scientist_respondent", 2, Path("/tmp/out"))
    assert "r2" in gen.as_posix()
    assert "scientist_respondent" in gen.as_posix()
    p = prompt_sensitivity_v2_scores_path("hermes", "analyst_person", outputs_dir=Path("/tmp/out"))
    assert p.name == "scores_hermes_analyst_person_nemotron.csv"
    p2 = prompt_sensitivity_v2_scores_path(
        "kimi", "scientist_respondent", 1, Path("/tmp/out"), disambiguator="nemotron"
    )
    assert p2.name == "scores_kimi_scientist_respondent_r1_nemotron.csv"
    gen_t, _, _ = prompt_sensitivity_v2_dirs(
        "kimi", "scientist_respondent", outputs_dir=Path("/tmp/out"), temperature_draw=1,
    )
    assert "/t1/" in gen_t.as_posix() or gen_t.as_posix().endswith("/t1/freetext")
    assert "r1" not in gen_t.as_posix()
    pt = prompt_sensitivity_v2_scores_path(
        "kimi", "scientist_respondent", outputs_dir=Path("/tmp/out"),
        temperature_draw=2, disambiguator="nemotron",
    )
    assert pt.name == "scores_kimi_scientist_respondent_t2_nemotron.csv"
    try:
        prompt_sensitivity_v2_dirs(
            "kimi", "scientist_respondent", 1, Path("/tmp/out"), temperature_draw=1,
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
