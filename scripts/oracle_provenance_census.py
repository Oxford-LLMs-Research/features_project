"""
Census of HOW the cached oracles were produced — the check to run before mixing cells
fitted on different machines (or under different runtime modes).

Reads every outputs/cache/cells/<cell>/oracle_meta.json and tabulates the `provenance`
block written by survey_features.oracle since 2026-09-04: preset, per-fold time limit,
host, and the size of the model bag each fold actually finished. AutoGluon's time
limit is a wall-clock budget that the preset spends rather than converges on, so a
slower or busier machine silently trains a smaller bag under identical settings — a
cell whose folds disagree, or whose bag is shorter than its neighbours', is a cell to
recompute, not to cite (docs/pipeline_audit_2026-08.md section C).

Usage:
    python scripts/oracle_provenance_census.py                 # the live cache
    python scripts/oracle_provenance_census.py --cells-dir X   # any cells folder
    python scripts/oracle_provenance_census.py --compare A B   # same cells, two folders

--compare prints, per cell present in both folders, the top-10 overlap (Jaccard) of the
select-side ranking and the two oracle ceilings — the cross-machine agreement check for
a smoke batch, to be read against the between-fold reliability each meta already
carries (meta["reliability"]["select"]["pairwise_jaccard10"]).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from survey_features.layout import cache_cells_dir  # noqa: E402
from survey_features.oracle import ORACLE_CONTRACT_VERSION  # noqa: E402


def _load_metas(cells_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for meta_path in sorted(cells_dir.glob("*/oracle_meta.json")):
        try:
            out[meta_path.parent.name] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # unreadable meta is itself a finding
            out[meta_path.parent.name] = {"_unreadable": f"{type(exc).__name__}: {exc}"}
    return out


def census(cells_dir: Path) -> None:
    metas = _load_metas(cells_dir)
    if not metas:
        raise SystemExit(f"no oracle_meta.json under {cells_dir}")
    rows = []
    for cell, m in metas.items():
        prov = m.get("provenance") or {}
        folds = prov.get("folds") or []
        n_models = sorted({int(f.get("n_models", 0)) for f in folds})
        rows.append({
            "cell": cell,
            "contract": m.get("contract_version"),
            "preset": prov.get("preset", "(no provenance)"),
            "time_limit_s": prov.get("time_limit_per_fold_s"),
            "host": prov.get("hostname", "?"),
            "n_models": "/".join(str(n) for n in n_models) if n_models else "?",
            "bag_same_all_folds": prov.get("bag_identical_across_folds"),
            "fit_secs_mean": (
                round(sum(f.get("fit_secs", 0) for f in folds) / len(folds), 1)
                if folds else None
            ),
            "autogluon": (prov.get("lib_versions") or {}).get("autogluon.tabular", "?"),
        })
    d = pd.DataFrame(rows)
    current = d[d["contract"] == ORACLE_CONTRACT_VERSION]
    print(f"[census] {len(d)} cells under {cells_dir}; "
          f"{len(current)} at contract v{ORACLE_CONTRACT_VERSION}, "
          f"{int((d['preset'] == '(no provenance)').sum())} without a provenance block")
    print("\n[census] cells by (host, preset, time limit, models per fold):")
    grp = (current.groupby(["host", "preset", "time_limit_s", "n_models", "autogluon"],
                           dropna=False).size().rename("cells").reset_index())
    print(grp.to_string(index=False))

    short = current[current["bag_same_all_folds"] == False]  # noqa: E712
    if len(short):
        print(f"\n[census] {len(short)} cell(s) whose folds finished DIFFERENT bags "
              f"(wall clock cut a fold short) — recompute, do not cite:")
        print(short[["cell", "host", "n_models", "fit_secs_mean"]].to_string(index=False))
    else:
        print("\n[census] every provenance-bearing cell fitted the same bag on all folds.")

    # Cross-host disagreement in bag size under the same settings is the machine confound.
    key = ["preset", "time_limit_s"]
    for k, sub in current[current["preset"] != "(no provenance)"].groupby(key):
        bags = sub.groupby("host")["n_models"].agg(lambda s: Counter(s).most_common(1)[0][0])
        if bags.nunique() > 1:
            print(f"\n[census] WARNING: under preset={k[0]} time_limit={k[1]} the modal bag "
                  f"size differs by host: {bags.to_dict()} — cells from the smaller-bag "
                  f"host are not comparable; raise its time limit and recompute.")


def compare(dir_a: Path, dir_b: Path, k: int = 10) -> None:
    metas_a, metas_b = _load_metas(dir_a), _load_metas(dir_b)
    common = sorted(set(metas_a) & set(metas_b))
    if not common:
        raise SystemExit("no cell present in both folders")
    rows = []
    for cell in common:
        oa = pd.read_csv(dir_a / cell / "oracle.csv")
        ob = pd.read_csv(dir_b / cell / "oracle.csv")
        top_a = set(oa.nlargest(k, "importance_select")["feature_variable"])
        top_b = set(ob.nlargest(k, "importance_select")["feature_variable"])
        jac = len(top_a & top_b) / len(top_a | top_b) if (top_a | top_b) else None
        ma, mb = metas_a[cell], metas_b[cell]
        rel_a = ((ma.get("reliability") or {}).get("select") or {}).get(f"pairwise_jaccard{k}")
        rel_b = ((mb.get("reliability") or {}).get("select") or {}).get(f"pairwise_jaccard{k}")
        pa, pb = ma.get("provenance") or {}, mb.get("provenance") or {}
        rows.append({
            "cell": cell,
            f"jaccard{k}_A_vs_B": round(jac, 3) if jac is not None else None,
            f"fold_jaccard{k}_A": rel_a, f"fold_jaccard{k}_B": rel_b,
            "ceiling10_A": (ma.get("oracle_ceiling") or {}).get("10"),
            "ceiling10_B": (mb.get("oracle_ceiling") or {}).get("10"),
            "host_A": pa.get("hostname", "?"), "host_B": pb.get("hostname", "?"),
            "bag_A": "/".join(str(f.get("n_models")) for f in pa.get("folds", [])) or "?",
            "bag_B": "/".join(str(f.get("n_models")) for f in pb.get("folds", [])) or "?",
        })
    print(f"[compare] {len(common)} cell(s) in both {dir_a} and {dir_b}")
    print("[compare] read A-vs-B top-10 overlap against the within-machine fold overlap: "
          "two machines agree 'as well as two folds do' when the first is not clearly "
          "below the second.")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", type=Path, default=None,
                    help="cells folder to census (default: the live outputs/cache/cells)")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("DIR_A", "DIR_B"),
                    help="two cells folders holding the same cells (e.g. two smoke runs)")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    census(args.cells_dir or cache_cells_dir())


if __name__ == "__main__":
    main()
