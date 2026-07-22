"""
Stage 0: introspect synthetic_sampling ProfileBuilder-style metadata per survey.

Prints sections, aggregates keys observed on variable info dicts, and sample variables
that carry topic-like fields — writes YAML for reproducibility.

Usage:
  set DATA_CONFIG_PATH and run:
    python prelim/introspect_metadata.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

# project root + package src on path
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def iter_var_entries(metadata: dict):
    """Yield (section_name, var_code, info_dict) for substantive sections."""
    for section, block in metadata.items():
        if section == "EXCLUDED" or not isinstance(block, dict):
            continue
        for var_code, info in block.items():
            if not isinstance(info, dict):
                continue
            yield section, var_code, info


def survey_introspection(survey_id: str, loader) -> dict:
    """Build a structured summary dict for one survey."""
    data, metadata = loader(survey_id)
    key_counts: Counter = Counter()
    topic_like_samples: list[dict] = []
    sections_out: dict[str, int] = {}

    topic_keys_hint = frozenset(
        {"topic", "topics", "tags", "tag", "section_tag", "theme", "themes", "domains"}
    )

    for section, var_code, info in iter_var_entries(metadata):
        sections_out[section] = sections_out.get(section, 0) + 1
        key_counts.update(info.keys())
        for k in topic_keys_hint:
            if k in info:
                topic_like_samples.append(
                    {"section": section, "variable": var_code, k: info[k]}
                )
                break

    # dedupe samples by variable
    seen: set[tuple[str, str]] = set()
    uniq_samples: list[dict] = []
    for row in topic_like_samples:
        t = (row["section"], row["variable"])
        if t not in seen:
            seen.add(t)
            uniq_samples.append(row)

    return {
        "survey_id": survey_id,
        "n_rows": int(len(data)),
        "n_columns": int(len(data.columns)),
        "sections_substantive": dict(sorted(sections_out.items())),
        "variable_info_key_counts": dict(key_counts.most_common()),
        "topic_like_field_samples": uniq_samples[:24],
        "survey_vars_also_data_columns": int(
            sum(1 for _s, vc, __ in iter_var_entries(metadata) if vc in data.columns)
        ),
        "metadata_substantive_var_count": int(sum(sections_out.values())),
    }


def main():
    os.chdir(ROOT)
    config_path = os.environ.get("DATA_CONFIG_PATH")
    if not config_path:
        print("DATA_CONFIG_PATH is not set; cannot load surveys.")
        sys.exit(1)

    from survey_features.surveys import SURVEY_COUNTRY_COL, load_survey

    def loader(sid: str):
        return load_survey(sid, config_path)

    # Prelim excludes ess_wave_10 (use ess_wave_11 only per plan).
    surveys = sorted(s for s in SURVEY_COUNTRY_COL if s != "ess_wave_10")

    summaries = []
    for sid in surveys:
        print(f"\n=== Introspecting {sid} ===")
        try:
            summaries.append(survey_introspection(sid, loader))
        except Exception as e:
            print(f"  [skip] failed: {e}")
            summaries.append({"survey_id": sid, "error": str(e)})

    out_path = ROOT / "prelim" / "metadata_introspection.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"surveys": summaries}, f, sort_keys=False, allow_unicode=True)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
