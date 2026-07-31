"""
Compare ensemble retrieval maps/scores vs single-embedder baselines.

Baselines (reused, never overwritten):
  - MiniLM: outputs/main/scores_{selector}.csv + main/<selector>/maps/
  - mpnet / roberta: outputs/experiments/embedding_sensitivity/<slug>/<selector>/

Ensemble (new):
  - outputs/experiments/ensemble_mapping/<fusion_slug>/<selector>/

Writes under ensemble_mapping/:
  - comparison.csv          performance + map Jaccard
  - latency_comparison.csv  retrieve / disambig / cell wall from ensemble map JSONs
                            (+ optional single-model timing logs if present)

Run:  python analysis/ensemble_mapping.py
      python analysis/ensemble_mapping.py --selector kimi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from survey_features.config import DEFAULT_EMBEDDING_MODEL, OUTPUTS_DIR, SELECTORS  # noqa: E402
from survey_features.ensemble import (  # noqa: E402
    DEFAULT_ENSEMBLE_MODELS,
    FUSION_RULE,
    fusion_slug,
)
from survey_features.layout import (  # noqa: E402
    embedding_run_dirs,
    ensemble_mapping_dir,
    ensemble_run_dirs,
    main_dir,
    resolve_main_scores_path,
    sanitize_model_slug,
    selector_dirs,
)
from survey_features.metrics import jaccard  # noqa: E402

OUT = OUTPUTS_DIR
PILOT = main_dir(OUT)
ENS = ensemble_mapping_dir(OUT)

PRIMARY_DK = "nemotron"
ARM = "C"
JOIN_KEYS = ["survey", "target", "country", "condition", "arm", "disambiguator", "k_spec"]
METRIC_COLS = [
    "model_acc",
    "value_over_random",
    "cost_of_imperfect",
    "captured_importance",
]
TOL = 0.01
SINGLE_MODELS = [
    DEFAULT_EMBEDDING_MODEL,
    "all-mpnet-base-v2",
    "all-roberta-large-v1",
]


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


def _scores_and_maps_for_single(selector_key: str, model: str) -> tuple[Path | None, Path | None]:
    if sanitize_model_slug(model) == sanitize_model_slug(DEFAULT_EMBEDDING_MODEL):
        scores = resolve_main_scores_path(selector_key, OUT)
        _, _, maps = selector_dirs(selector_key)
        return scores, maps
    maps, scores = embedding_run_dirs(model, selector_key)
    return (scores if scores.is_file() else None), (maps if maps.is_dir() else None)


def _map_path(map_dir: Path, survey: str, target: str, country: str, cond: str) -> Path:
    from survey_features.layout import cell_tag
    return map_dir / f"{ARM}__{PRIMARY_DK}__{cell_tag(survey, target, country)}__{cond}.json"


def _mapped_codes(path: Path) -> set[str] | None:
    if not path.is_file():
        return None
    codes = json.loads(path.read_text(encoding="utf-8")).get("mapped_codes") or []
    return {c for c in codes if c}


def _pair_map_jaccard(base_map_dir: Path, alt_map_dir: Path, score_pairs: pd.DataFrame) -> float | None:
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


def compare_ensemble_vs(
    selector_key: str,
    baseline_model: str,
    ens_scores: pd.DataFrame,
    ens_maps: Path,
) -> dict | None:
    base_scores_path, base_maps = _scores_and_maps_for_single(selector_key, baseline_model)
    if base_scores_path is None or base_maps is None:
        print(f"  ! skip baseline {baseline_model}: missing scores/maps")
        return None
    base = _load_arm_c_scores(base_scores_path, baseline_model)
    merged = base.merge(ens_scores, on=JOIN_KEYS, suffixes=("_base", "_ens"), how="inner")
    if merged.empty:
        print(f"  ! skip {selector_key} vs {baseline_model}: no overlapping rows")
        return None

    row: dict = {
        "selector": selector_key,
        "baseline_model": baseline_model,
        "retrieval": "ensemble",
        "n_paired_rows": len(merged),
        "mean_map_jaccard": _pair_map_jaccard(base_maps, ens_maps, merged),
    }
    conclusions_move = False
    for m in METRIC_COLS:
        bcol, ecol = f"{m}_base", f"{m}_ens"
        if bcol not in merged.columns or ecol not in merged.columns:
            continue
        delta = merged[ecol] - merged[bcol]
        mean_d = float(delta.mean()) if delta.notna().any() else float("nan")
        mean_abs = float(delta.abs().mean()) if delta.notna().any() else float("nan")
        row[f"mean_delta_{m}"] = mean_d
        row[f"mean_abs_delta_{m}"] = mean_abs
        if m == "value_over_random" and delta.notna().any():
            base_mean = float(merged[bcol].mean())
            ens_mean = float(merged[ecol].mean())
            row["vor_sign_flip"] = (base_mean >= 0) != (ens_mean >= 0)
            if row["vor_sign_flip"]:
                conclusions_move = True
        if mean_abs == mean_abs and mean_abs > TOL:
            if m in ("model_acc", "value_over_random", "cost_of_imperfect"):
                conclusions_move = True
    row["conclusions_move"] = conclusions_move

    # Absolute means for ensemble side (handy for tables)
    for m in METRIC_COLS:
        ecol = f"{m}_ens"
        if ecol in merged.columns:
            row[f"mean_ens_{m}"] = float(merged[ecol].mean())
            row[f"mean_base_{m}"] = float(merged[f"{m}_base"].mean())
    return row


def collect_ensemble_latency(ens_maps: Path) -> pd.DataFrame:
    rows = []
    if not ens_maps.is_dir():
        return pd.DataFrame()
    for p in sorted(ens_maps.glob(f"{ARM}__{PRIMARY_DK}__*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        t = rec.get("timing") or {}
        by_m = t.get("retrieve_wall_s_by_model") or {}
        rows.append({
            "survey": rec.get("survey"),
            "target": rec.get("target"),
            "country": rec.get("country"),
            "condition": rec.get("condition"),
            "n_piped": t.get("n_piped", rec.get("n_piped")),
            "n_disambig_calls": t.get("n_disambig_calls"),
            "retrieve_wall_s_total": t.get("retrieve_wall_s_total"),
            "disambig_wall_s": t.get("disambig_wall_s"),
            "cell_wall_s": t.get("cell_wall_s"),
            "mean_pool_size": rec.get("mean_pool_size"),
            "max_fused": rec.get("max_fused"),
            **{f"retrieve_{sanitize_model_slug(k)}": v for k, v in by_m.items()},
        })
    return pd.DataFrame(rows)


def latency_summary(lat: pd.DataFrame, selector_key: str, fusion: str) -> dict | None:
    if lat.empty:
        return None
    out = {
        "selector": selector_key,
        "retrieval": "ensemble",
        "fusion_slug": fusion,
        "n_map_cells": len(lat),
        "sum_retrieve_s": float(lat["retrieve_wall_s_total"].sum())
        if "retrieve_wall_s_total" in lat else float("nan"),
        "sum_disambig_s": float(lat["disambig_wall_s"].sum())
        if "disambig_wall_s" in lat else float("nan"),
        "sum_cell_wall_s": float(lat["cell_wall_s"].sum())
        if "cell_wall_s" in lat else float("nan"),
        "mean_retrieve_s": float(lat["retrieve_wall_s_total"].mean())
        if "retrieve_wall_s_total" in lat else float("nan"),
        "mean_disambig_s": float(lat["disambig_wall_s"].mean())
        if "disambig_wall_s" in lat else float("nan"),
        "mean_cell_wall_s": float(lat["cell_wall_s"].mean())
        if "cell_wall_s" in lat else float("nan"),
        "mean_n_disambig_calls": float(lat["n_disambig_calls"].mean())
        if "n_disambig_calls" in lat and lat["n_disambig_calls"].notna().any()
        else float("nan"),
        "mean_pool_size": float(lat["mean_pool_size"].mean())
        if "mean_pool_size" in lat and lat["mean_pool_size"].notna().any()
        else float("nan"),
        "note": (
            "Ensemble adds multi-embed retrieve cost; disambig call count equals "
            "n_piped (1× vs single-model). Single-model map latency for head-to-head "
            "comes from outputs/logs/timing_map_*.jsonl when available — not "
            "auto-joined here."
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--selector",
        choices=list(SELECTORS),
        default="kimi",
        help="v1 default: kimi",
    )
    ap.add_argument(
        "--embedding-models",
        default=",".join(DEFAULT_ENSEMBLE_MODELS),
        help="ensemble members (must match the map run)",
    )
    args = ap.parse_args()
    models = [m.strip() for m in args.embedding_models.split(",") if m.strip()]
    slug = fusion_slug(models)
    ENS.mkdir(parents=True, exist_ok=True)

    ens_maps, ens_scores_path = ensemble_run_dirs(slug, args.selector)
    if not ens_scores_path.is_file():
        print(
            f"No ensemble scores at {ens_scores_path}. "
            "Run scripts/run_ensemble_mapping.py --phase map then --phase score."
        )
        # Still emit latency if maps exist
        lat = collect_ensemble_latency(ens_maps)
        if not lat.empty:
            lat_path = ENS / "latency_cells.csv"
            lat.to_csv(lat_path, index=False)
            summary = latency_summary(lat, args.selector, slug)
            pd.DataFrame([summary] if summary else []).to_csv(
                ENS / "latency_comparison.csv", index=False,
            )
            print(f"Wrote latency tables (no performance comparison yet) under {ENS}")
        else:
            pd.DataFrame().to_csv(ENS / "comparison.csv", index=False)
            pd.DataFrame().to_csv(ENS / "latency_comparison.csv", index=False)
        return

    ens = _load_arm_c_scores(ens_scores_path, f"ensemble_{FUSION_RULE}")
    print(f"Ensemble: {slug} / {args.selector}  n_score_rows={len(ens)}")
    print(f"Comparing vs single-model baselines: {SINGLE_MODELS}")

    rows = []
    for base_model in SINGLE_MODELS:
        print(f"Comparing ensemble vs {base_model}")
        r = compare_ensemble_vs(args.selector, base_model, ens, ens_maps)
        if r:
            rows.append(r)
            j = r.get("mean_map_jaccard")
            j_s = f"{j:.3f}" if j is not None else "n/a"
            print(
                f"  -> n={r['n_paired_rows']}; "
                f"mean Δ VoR={r.get('mean_delta_value_over_random', float('nan')):.4f}; "
                f"map Jaccard={j_s}; "
                f"{'MOVE' if r['conclusions_move'] else 'stable'}"
            )

    out = ENS / "comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {len(rows)} comparison rows -> {out}")

    lat = collect_ensemble_latency(ens_maps)
    if not lat.empty:
        lat.to_csv(ENS / "latency_cells.csv", index=False)
        summary = latency_summary(lat, args.selector, slug)
        pd.DataFrame([summary] if summary else []).to_csv(
            ENS / "latency_comparison.csv", index=False,
        )
        print(f"Wrote latency tables -> {ENS / 'latency_comparison.csv'}")
        if summary:
            print(
                f"  sum retrieve={summary['sum_retrieve_s']:.1f}s  "
                f"sum disambig={summary['sum_disambig_s']:.1f}s  "
                f"mean pool={summary['mean_pool_size']:.1f}  "
                f"mean disambig calls/cell={summary['mean_n_disambig_calls']:.1f}"
            )
    else:
        pd.DataFrame().to_csv(ENS / "latency_comparison.csv", index=False)
        print("No ensemble map timing blocks found (re-run map to populate timing).")


if __name__ == "__main__":
    main()
