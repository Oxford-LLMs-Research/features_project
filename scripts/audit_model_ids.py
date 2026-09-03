"""Check every model ID in config.py against the live endpoint catalog.

A retired ID does not fail at import — it fails mid-sweep, after you have paid
for the cells that already ran. `deepseek-ai/DeepSeek-V4-Flash` sat in three
config slots for weeks after the catalog replaced it with a dated build.

  python scripts/audit_model_ids.py            # exits 1 if anything is missing
  python scripts/audit_model_ids.py --quiet    # only report problems

Reads LLM_BASE_URL / LLM_API_KEY from .env, same as the pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from survey_features import config as C  # noqa: E402  loads .env


def live_model_ids(timeout: int = 60) -> set[str]:
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    req = urllib.request.Request(
        base + "/models", headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"]}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {m["id"] for m in json.load(resp).get("data", [])}


def configured_ids() -> list[tuple[str, str]]:
    """(config slot, model id) for everything the pipeline can actually call."""
    used: list[tuple[str, str]] = []
    for key, spec in C.SELECTORS.items():
        used.append((f"SELECTORS[{key}]", spec["model"]))
    for key, spec in C.EXPERIMENT_SELECTORS.items():
        if (f"SELECTORS[{key}]", spec["model"]) not in used:
            used.append((f"EXPERIMENT_SELECTORS[{key}]", spec["model"]))
    used.append(("EXTRACTOR_MODEL", C.EXTRACTOR_MODEL))
    for key, mid in C.DISAMBIGUATORS.items():
        used.append((f"DISAMBIGUATORS[{key}]", mid))
    used.append(("ROLE_SWAP_EXTRACTOR", C.ROLE_SWAP_EXTRACTOR))
    used.append(("DISAMBIG_MODEL", C.DISAMBIG_MODEL))
    return used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="only print problems")
    args = ap.parse_args()

    live = live_model_ids()
    rows = configured_ids()
    missing = [(slot, mid) for slot, mid in rows if mid not in live]

    if not args.quiet:
        print(f"{len(live)} models on the endpoint; {len(rows)} config slots\n")
        for slot, mid in rows:
            print(f"  {'ok ' if mid in live else 'GONE'} {slot:34s} {mid}")

    # Historical IDs are expected to be absent — artifacts were generated under
    # them and their provenance must not be rewritten. Report, never fail.
    historical = {C.FLASH_MODEL_HISTORICAL}
    for mid in sorted(historical):
        state = "still live" if mid in live else "retired (expected)"
        print(f"\n  historical ID {mid}: {state}")

    if missing:
        print("\nNOT ON THE ENDPOINT — repoint these before any sweep:")
        for slot, mid in missing:
            stem = mid.split("/")[-1].split("-")[0].lower()
            near = sorted(x for x in live if stem in x.lower())
            print(f"  {slot:34s} {mid}\n      candidates: {near or 'none'}")
        return 1

    print("\nAll configured model IDs are on the endpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
