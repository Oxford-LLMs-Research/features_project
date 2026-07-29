"""
Compare free-text (arm C, nemotron) scores and mapped codes across embedding models.

Baseline: outputs/main/scores_{selector}.csv (or legacy format_pilot/scores.csv / scores_kimi.csv)
          + main/<selector>/maps/C__nemotron__*.json
          (assumed DEFAULT_EMBEDDING_MODEL = all-MiniLM-L6-v2)

Alternatives: outputs/experiments/embedding_sensitivity/<slug>/<selector>/scores_*.csv + maps/

Writes: outputs/experiments/embedding_sensitivity/comparison.csv
        and prints a short verdict to stdout.

Run:  python analysis/embedding_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from survey_features.config import DEFAULT_EMBEDDING_MODEL, OUTPUTS_DIR, SELECTORS  # noqa: E402
from survey_features.layout import (  # noqa: E402
    embedding_run_dirs,
    embedding_sensitivity_dir,
    main_dir,
    resolve_main_scores_path,
    sanitize_model_slug,
    selector_dirs,
)
from survey_features.metrics import jaccard  # noqa: E402

OUT = OUTPUTS_DIR
PILOT = main_dir(OUT)
SENS = embedding_sensitivity_dir(OUT)

PRIMARY_DK = "nemotron"
ARM = "C"
JOIN_KEYS = ["survey", "target", "country", "condition", "arm", "disambiguator", "k_spec"]
METRIC_COLS = [
    "model_acc",
    "value_over_random",
    "cost_of_imperfect",
    "captured_importance",
]
# Mean |Δ| above this (or VoR sign flip) => "conclusions move"
TOL = 0.01


def _baseline_scores_path(selector_key: str) -> Path | None:
    """Resolve main/ scores CSV; tolerate legacy names from the pilot."""
    return resolve_main_scores_path(selector_key, OUT)


def _load_arm_c_scores(path: Path, embedding_model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["arm"] == ARM) & (df["disambiguator"] == PRIMARY_DK)].copy()
    if "embedding_model" not in df.columns or df["embedding_model"].isna().all():
        df["embedding_model"] = embedding_model
    else:
        df["embedding_model"] = df["embedding_model"].fillna(embedding_model)
    for c in METRIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _map_path(map_dir: Path, survey: str, target: str, country: str, cond: str) -> Path:
    from survey_features.layout import cell_tag
    return map_dir / f"{ARM}__{PRIMARY_DK}__{cell_tag(survey, target, country)}__{cond}.json"


def _mapped_codes(path: Path) -> set[str] | None:
    if not path.is_file():
        return None
    codes = json.loads(path.read_text(encoding="utf-8")).get("mapped_codes") or []
    return {c for c in codes if c}


def _pair_map_jaccard(
    base_map_dir: Path,
    alt_map_dir: Path,
    score_pairs: pd.DataFrame,
) -> float | None:
    """Mean Jaccard of mapped_codes over unique cells present in the paired score rows."""
    cells = score_pairs[["survey", "target", "country", "condition"]].drop_duplicates()
    vals = []
    for r in cells.itertuples(index=False):
        a = _mapped_codes(_map_path(base_map_dir, r.survey, r.target, r.country, r.condition))
        b = _mapped_codes(_map_path(alt_map_dir, r.survey, r.target, r.country, r.condition))
        if a is None or b is None:
            continue
        j = jaccard(a, b)
        if j is not None:
            vals.append(j)
    return float(np.mean(vals)) if vals else None


def _frac_codes_differ(
    base_map_dir: Path,
    alt_map_dir: Path,
    score_pairs: pd.DataFrame,
) -> float | None:
    cells = score_pairs[["survey", "target", "country", "condition"]].drop_duplicates()
    n = n_diff = 0
    for r in cells.itertuples(index=False):
        a = _mapped_codes(_map_path(base_map_dir, r.survey, r.target, r.country, r.condition))
        b = _mapped_codes(_map_path(alt_map_dir, r.survey, r.target, r.country, r.condition))
        if a is None or b is None:
            continue
        n += 1
        if a != b:
            n_diff += 1
    return (n_diff / n) if n else None


def discover_alt_models() -> list[str]:
    """Embedding model names that have at least one scores CSV under embedding_sensitivity/."""
    if not SENS.is_dir():
        return []
    found: list[str] = []
    for slug_dir in sorted(SENS.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name == "__pycache__":
            continue
        if slug_dir.name == "manifest.json":
            continue
        has_scores = any(slug_dir.glob("*/scores_*.csv"))
        if not has_scores:
            continue
        # Prefer model name recorded in a scores file; else unslug as best-effort
        model_name = None
        for csv_path in slug_dir.glob("*/scores_*.csv"):
            try:
                sample = pd.read_csv(csv_path, nrows=5)
                if "embedding_model" in sample.columns and sample["embedding_model"].notna().any():
                    model_name = str(sample["embedding_model"].dropna().iloc[0])
                    break
            except Exception:
                continue
        found.append(model_name or slug_dir.name)
    return found


def compare_one(selector_key: str, alt_model: str) -> dict | None:
    base_path = _baseline_scores_path(selector_key)
    if base_path is None:
        print(f"  ! skip {selector_key}: no baseline scores under {PILOT}")
        return None
    _, alt_scores_path = embedding_run_dirs(alt_model, selector_key)
    if not alt_scores_path.is_file():
        print(f"  ! skip {selector_key} × {alt_model}: missing {alt_scores_path}")
        return None

    base = _load_arm_c_scores(base_path, DEFAULT_EMBEDDING_MODEL)
    alt = _load_arm_c_scores(alt_scores_path, alt_model)
    merged = base.merge(alt, on=JOIN_KEYS, suffixes=("_base", "_alt"), how="inner")
    if merged.empty:
        print(f"  ! skip {selector_key} × {alt_model}: no overlapping arm-C/nemotron rows")
        return None

    _, _, base_maps = selector_dirs(selector_key)
    alt_maps, _ = embedding_run_dirs(alt_model, selector_key)

    row: dict = {
        "selector": selector_key,
        "baseline_model": DEFAULT_EMBEDDING_MODEL,
        "alt_model": alt_model,
        "n_paired_rows": len(merged),
        "mean_map_jaccard": _pair_map_jaccard(base_maps, alt_maps, merged),
        "frac_cells_codes_differ": _frac_codes_differ(base_maps, alt_maps, merged),
    }

    conclusions_move = False
    for m in METRIC_COLS:
        bcol, acol = f"{m}_base", f"{m}_alt"
        if bcol not in merged.columns or acol not in merged.columns:
            continue
        delta = merged[acol] - merged[bcol]
        mean_d = float(delta.mean()) if delta.notna().any() else float("nan")
        mean_abs = float(delta.abs().mean()) if delta.notna().any() else float("nan")
        row[f"mean_delta_{m}"] = mean_d
        row[f"mean_abs_delta_{m}"] = mean_abs
        if m == "value_over_random" and delta.notna().any():
            base_mean = float(merged[bcol].mean())
            alt_mean = float(merged[acol].mean())
            row["vor_sign_flip"] = (base_mean >= 0) != (alt_mean >= 0)
            if row["vor_sign_flip"]:
                conclusions_move = True
        if mean_abs == mean_abs and mean_abs > TOL:  # not NaN
            if m in ("model_acc", "value_over_random", "cost_of_imperfect"):
                conclusions_move = True

    row["conclusions_move"] = conclusions_move
    return row


def main() -> None:
    SENS.mkdir(parents=True, exist_ok=True)
    alts = discover_alt_models()
    if not alts:
        print(f"No sensitivity score CSVs under {SENS}. Run map+score with --embedding-model first.")
        out = SENS / "comparison.csv"
        pd.DataFrame().to_csv(out, index=False)
        print(f"Wrote empty {out}")
        return

    print(f"Baseline embedder: {DEFAULT_EMBEDDING_MODEL}")
    print(f"Alt models found: {alts}")
    rows = []
    for sel in SELECTORS:
        for alt in alts:
            # Skip comparing a model to itself if someone re-ran MiniLM into sensitivity/
            if sanitize_model_slug(alt) == sanitize_model_slug(DEFAULT_EMBEDDING_MODEL):
                continue
            print(f"Comparing {sel}: {DEFAULT_EMBEDDING_MODEL} vs {alt}")
            r = compare_one(sel, alt)
            if r:
                rows.append(r)
                verdict = "MOVE" if r["conclusions_move"] else "stable"
                j = r.get("mean_map_jaccard")
                j_s = f"{j:.3f}" if j is not None else "n/a"
                print(
                    f"  -> {verdict}; n={r['n_paired_rows']}; "
                    f"mean delta VoR={r.get('mean_delta_value_over_random', float('nan')):.4f}; "
                    f"map Jaccard={j_s}"
                )

    out = SENS / "comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {len(rows)} comparison rows -> {out}")
    if rows:
        n_move = sum(1 for r in rows if r["conclusions_move"])
        print(f"Verdict: {n_move}/{len(rows)} selector×model pairs flagged as conclusions_move "
              f"(mean |Δ| > {TOL} on primary metrics or VoR sign flip).")


if __name__ == "__main__":
    main()
