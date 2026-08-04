"""
Standalone AutoGluon oracle CLI for arbitrary target x country cells (the audited
grid is better served by scripts/rerun_oracles.py, which adds contract-version
resume and the process pool). Moved out of survey_features/oracle.py 2026-08.

Usage:
    python scripts/compute_oracle.py --survey wvs --targets Q57 --countries Germany
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / 'src'), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from survey_features.config import OUTPUTS_DIR  # noqa: E402
from survey_features.layout import (  # noqa: E402
    cell_dir as layout_cell_dir,
    oracle_csv_path,
    tmp_dir,
)
from survey_features.oracle import (  # noqa: E402
    ENFORCE_IDENTICAL_FEATURE_POOL,
    MAX_MISSINGNESS_THRESHOLD,
    MIN_CLASS_COUNT,
    MIN_NORMALIZED_FEATURE_ENTROPY,
    ORACLE_EVAL_METRIC,
    RANDOM_STATE,
    SIMILARITY_THRESHOLD,
    TEST_SIZE,
    _resolve_num_gpus,
    _set_local_tmp_dir,
    compute_oracle,
    load_similarity_model,
)
from survey_features.feature_pool import (  # noqa: E402
    build_feature_pool,
    filter_feature_pool_across_countries,
)
from survey_features.surveys import (  # noqa: E402
    SURVEY_COUNTRY_COL,
    build_admin_cols,
    build_country_code_map,
    clean_question_columns,
    flatten_metadata,
    load_survey,
)

DEFAULT_SURVEY = "wvs"
DEFAULT_TARGETS = ["Q47", "Q57", "Q199", "Q235", "Q164"]
DEFAULT_COUNTRIES = ["Germany", "Nigeria", "Japan", "Brazil", "Egypt"]


def _detect_gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return shutil.which("nvidia-smi") is not None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoGluon oracle runner.")
    parser.add_argument(
        "--survey",
        default=DEFAULT_SURVEY,
        choices=list(SURVEY_COUNTRY_COL),
        help=f"Survey to run (default: {DEFAULT_SURVEY}).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        metavar="VAR",
        help="Target variable(s). Defaults to WVS targets when --survey=wvs.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        metavar="COUNTRY",
        help="Country name(s). Defaults to WVS countries when --survey=wvs.",
    )
    parser.add_argument(
        "--list-countries",
        action="store_true",
        help="Print available countries for the chosen survey and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root (default: outputs/).",
    )
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="balanced",
        choices=["quick", "balanced", "best"],
        help="AutoGluon runtime profile.",
    )
    parser.add_argument(
        "--autogluon-time-limit",
        type=int,
        default=0,
        help="Override AutoGluon time limit in seconds (0 = default per mode).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Holdout test size (default: {TEST_SIZE}).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed (default: {RANDOM_STATE}).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Semantic exclusion threshold (default: {SIMILARITY_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-missingness-threshold",
        type=float,
        default=MAX_MISSINGNESS_THRESHOLD,
        help=f"Drop features with missingness >= threshold (default: {MAX_MISSINGNESS_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-normalized-feature-entropy",
        type=float,
        default=MIN_NORMALIZED_FEATURE_ENTROPY,
        help=f"Minimum normalized entropy (default: {MIN_NORMALIZED_FEATURE_ENTROPY}).",
    )
    parser.add_argument(
        "--enforce-identical-feature-pool",
        action="store_true",
        default=ENFORCE_IDENTICAL_FEATURE_POOL,
        help="Use a single feature pool per target across selected countries.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=MIN_CLASS_COUNT,
        help=f"Minimum class count (default: {MIN_CLASS_COUNT}).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Permutation repeats for feature importance (AutoGluon shuffle sets).",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default=None,
        help=(
            "Force one metric for every cell (default None = the target's measurement "
            "level chooses). Use 'accuracy' to reproduce the pre-2026-08 caches."
        ),
    )
    parser.add_argument(
        "--auto-detect-gpu",
        action="store_true",
        default=False,
        help="Use one GPU if available; otherwise force CPU.",
    )
    parser.add_argument(
        "--max-cells-per-run",
        type=int,
        default=0,
        help="If >0, process at most this many target-country cells.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cache/cells/<target>_<country>/oracle.csv exists.",
    )
    parser.add_argument(
        "--ag-verbosity",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help=(
            "AutoGluon verbosity (0=silent, 2=info, 3=info+model details, 4=debug). "
            "Use 3+ to surface 'skipping model X' tracebacks while debugging."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    survey_id = args.survey
    country_col = SURVEY_COUNTRY_COL[survey_id]
    targets = args.targets or (DEFAULT_TARGETS if survey_id == DEFAULT_SURVEY else None)
    countries = args.countries or (DEFAULT_COUNTRIES if survey_id == DEFAULT_SURVEY else None)

    config_path = os.environ.get("DATA_CONFIG_PATH")
    if not config_path:
        raise ValueError("DATA_CONFIG_PATH is not set in .env")

    data, metadata = load_survey(survey_id, config_path)
    country_codes = build_country_code_map(metadata, country_col, data)
    admin_cols = build_admin_cols(metadata, country_col)
    metadata_flat = flatten_metadata(metadata)
    similarity_model = load_similarity_model(args.similarity_threshold)

    if args.list_countries:
        print(f"\nAvailable countries for '{survey_id}':")
        for name in sorted(country_codes):
            print(f"  {name} ({country_codes[name]})")
        return

    if not targets:
        raise SystemExit(f"--targets is required for survey '{survey_id}'")
    if not countries:
        raise SystemExit(f"--countries is required for survey '{survey_id}'")

    unknown = [c for c in countries if c not in country_codes]
    if unknown:
        raise ValueError(
            f"Unknown country/ies {unknown} for survey '{survey_id}'. "
            f"Run with --list-countries to see valid names."
        )

    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else OUTPUTS_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root = _set_local_tmp_dir(tmp_dir(output_root))

    num_gpus = 1 if (args.auto_detect_gpu and _detect_gpu_available()) else 0
    num_gpus = _resolve_num_gpus(num_gpus)

    processed = 0
    errors: list[dict] = []

    print("\n" + "=" * 72)
    print(f"Survey: {survey_id}  |  country column: {country_col}")
    print(f"Targets:   {targets}")
    print(f"Countries: {countries}")
    print(f"AutoGluon: runtime={args.runtime_mode}, time_limit={args.autogluon_time_limit or 'default'}")
    print(f"GPU: num_gpus={num_gpus}")
    print("=" * 72)

    selected_country_codes = {c: country_codes[c] for c in countries}

    for target_var in targets:
        if target_var not in data.columns:
            raise ValueError(f"Target {target_var} missing from dataset columns.")

        target_feature_pool = None
        if args.enforce_identical_feature_pool:
            df_target = data[data[country_col].isin(selected_country_codes.values())].copy()
            df_target = clean_question_columns(df_target, country_col, admin_cols, metadata)
            df_target = df_target[df_target[target_var].notna()].copy()
            base_pool, _ = build_feature_pool(
                df_target,
                metadata_flat,
                target_var,
                admin_cols,
                similarity_model,
                args.similarity_threshold,
            )
            target_feature_pool, _ = filter_feature_pool_across_countries(
                df_target,
                base_pool,
                selected_country_codes,
                country_col,
                args.max_missingness_threshold,
                args.min_normalized_feature_entropy,
            )

        for country_name in countries:
            country_code = country_codes[country_name]
            prefix = f"{target_var}_{country_name}"
            # Prefer dual-resolved existing oracle; write under cache/cells/ for new runs.
            existing = oracle_csv_path(target_var, country_name, output_root)
            if existing.is_file() and not args.force:
                print(f"[skip] {prefix}: oracle.csv already exists ({existing})")
                continue

            cell_path = layout_cell_dir(target_var, country_name, output_root)
            cell_path.mkdir(parents=True, exist_ok=True)
            oracle_path = cell_path / "oracle.csv"

            print(f"\n[oracle] {target_var} x {country_name}")
            try:
                oracle_df, feature_pool, meta_out = compute_oracle(
                    data=data,
                    metadata=metadata,
                    target_var=target_var,
                    country_code=country_code,
                    country_col=country_col,
                    admin_cols=admin_cols,
                    metadata_flat=metadata_flat,
                    similarity_model=similarity_model,
                    similarity_threshold=args.similarity_threshold,
                    max_missingness_threshold=args.max_missingness_threshold,
                    min_normalized_feature_entropy=args.min_normalized_feature_entropy,
                    feature_pool_override=target_feature_pool,
                    tmp_root=tmp_root,
                    n_splits=1,
                    n_repeats=args.n_repeats,
                    random_state=args.random_state,
                    runtime_mode=args.runtime_mode,
                    autogluon_time_limit=args.autogluon_time_limit,
                    test_size=args.test_size,
                    min_class_count=args.min_class_count,
                    num_gpus=num_gpus,
                    ag_verbosity=args.ag_verbosity,
                    eval_metric=args.eval_metric,
                    survey_id=survey_id,
                )

                oracle_df.to_csv(oracle_path, index=False)
                pd.DataFrame({"feature_variable": feature_pool}).to_csv(
                    cell_path / "feature_pool.csv", index=False
                )
                (cell_path / "oracle_meta.json").write_text(
                    json.dumps(meta_out, indent=2), encoding="utf-8"
                )
                ceil = meta_out["oracle_ceiling"]
                print(
                    f"  Saved {oracle_path} "
                    f"({meta_out['n_positive_score']}/{meta_out['n_features']} features "
                    f"with positive importance; honest ceiling@10 = {ceil.get('10')})"
                )
                processed += 1

            except Exception as exc:
                errors.append({"target": target_var, "country": country_name, "error": str(exc)})
                print(f"  [error] {type(exc).__name__}: {exc}")

            if args.max_cells_per_run > 0 and processed >= args.max_cells_per_run:
                print("\nMax cells per run reached; stopping.")
                break

        if args.max_cells_per_run > 0 and processed >= args.max_cells_per_run:
            break

    if errors:
        print("\n" + "=" * 72)
        print(f"Errors in {len(errors)} cell(s):")
        for err in errors:
            print(f"  - {err['target']} x {err['country']}: {err['error']}")


if __name__ == "__main__":
    main()
