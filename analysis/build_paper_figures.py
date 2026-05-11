"""
Build publication-style figures for paper/prelim_paper.tex from grid_summary CSVs.
Run from repo root: python analysis/build_paper_figures.py
Outputs: paper/figures/*.pdf and *.png
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIG_DIR = REPO_ROOT / "paper" / "figures"
MANIFEST_TARGETS = REPO_ROOT / "prelim" / "target_selection_detail.yaml"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)

# Okabe–Ito–inspired, print-friendly (bars + consistent survey hues)
C_ORACLE = "#0072B2"
C_MODEL = "#E69F00"
C_RANDOM = "#009E73"
C_VALUE_METRIC = "#56B4E9"
C_COST_METRIC = "#D55E00"
C_HIST_FILL = "#8DA0CB"
C_MEAN_LINE = "#CC3311"
# Six distinct survey colors (ColorBrewer Set2–like, fixed order by sort)
SURVEY_HEX = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F"]
BAR_EDGE = "white"
BAR_EDGELW = 0.6


def survey_colors(surveys_sorted: list[str]) -> dict[str, str]:
    """Map survey id to hex fill for scatter/box consistency."""
    return {s: SURVEY_HEX[i % len(SURVEY_HEX)] for i, s in enumerate(surveys_sorted)}


def survey_label(s: str) -> str:
    return s.replace("ess_wave_11", "ESS W11")


def load_target_buckets() -> dict[tuple[str, str], str]:
    """Map (survey_id, target) -> bucket."""
    data = yaml.safe_load(MANIFEST_TARGETS.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], str] = {}
    for block in data.get("surveys", []):
        sid = block["survey_id"]
        for sel in block.get("selected", []):
            out[(sid, sel["variable"])] = sel.get("bucket", "unknown")
    return out


def load_grid_concat() -> pd.DataFrame:
    from output_layout import collect_grid_summary_paths, parse_grid_summary_stem

    frames = []
    for p in collect_grid_summary_paths(OUTPUTS_DIR):
        df = pd.read_csv(p)
        sid, tag = parse_grid_summary_stem(p.stem)
        df["survey"] = sid
        if tag:
            df["llm_run_tag"] = tag
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["oracle_acc", "model_acc", "random_acc"]
    m = df[cols].notna().all(axis=1)
    return df.loc[m].copy()


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)


def plot_survey_bars(summary: pd.DataFrame) -> None:
    surveys = summary["survey"].tolist()
    x = range(len(surveys))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    kw = dict(edgecolor=BAR_EDGE, linewidth=BAR_EDGELW)
    ax.bar([i - w for i in x], summary["oracle_acc"], width=w, label="Oracle", color=C_ORACLE, **kw)
    ax.bar([i for i in x], summary["model_acc"], width=w, label="Model", color=C_MODEL, **kw)
    ax.bar([i + w for i in x], summary["random_acc"], width=w, label="Random $k$", color=C_RANDOM, **kw)
    ax.set_xticks(list(x))
    ax.set_xticklabels([survey_label(s) for s in surveys], rotation=25, ha="right")
    ax.set_ylabel("Mean CV accuracy")
    ax.set_title("Matched-$k$: mean oracle, model, and random baseline by survey ($n_{\\mathrm{rows}}$ varies)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    save(fig, "fig_survey_accuracy_bars")


def plot_value_histogram(v: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(v["value_over_random"], bins=35, color=C_HIST_FILL, edgecolor="white", linewidth=0.5, alpha=0.92)
    ax.axvline(
        v["value_over_random"].mean(),
        color=C_MEAN_LINE,
        linestyle="--",
        label=f"Mean = {v['value_over_random'].mean():.3f}",
    )
    ax.set_xlabel("Value over random (model acc $-$ random acc)")
    ax.set_ylabel("Count of rows")
    ax.set_title("Distribution across valid condition rows ($n=${})".format(len(v)))
    ax.legend()
    save(fig, "fig_value_over_random_hist")


def plot_cost_vs_value(v: pd.DataFrame) -> None:
    surveys = sorted(v["survey"].unique())
    survey_to_color = survey_colors(surveys)
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in surveys:
        sub = v[v["survey"] == s]
        ax.scatter(
            sub["value_over_random"],
            sub["cost_of_imperfect"],
            label=survey_label(s),
            color=survey_to_color[s],
            alpha=0.65,
            s=22,
        )
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("Value over random")
    ax.set_ylabel("Cost of imperfect (oracle $-$ model)")
    ax.set_title("Per-row tradeoff (each point = target $\\times$ country $\\times$ condition)")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    save(fig, "fig_scatter_cost_vs_value")


def plot_bucket(v: pd.DataFrame, buckets: dict[tuple[str, str], str]) -> None:
    v = v.copy()
    v["bucket"] = [
        buckets.get((r.survey, r.target), "unknown") for _, r in v.iterrows()
    ]
    order = ["binary", "mid", "large", "unknown"]
    rows = []
    for b in order:
        sub = v[v["bucket"] == b]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "bucket": b,
                "n": len(sub),
                "mean_cost": sub["cost_of_imperfect"].mean(),
                "mean_value": sub["value_over_random"].mean(),
                "oracle_minus_majority": (sub["oracle_acc"] - sub["majority_baseline"]).mean(),
            }
        )
    g = pd.DataFrame(rows).set_index("bucket")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = range(len(g))
    w = 0.35
    kw = dict(edgecolor=BAR_EDGE, linewidth=BAR_EDGELW)
    ax.bar([i - w / 2 for i in x], g["mean_value"], width=w, label="Mean value over random", color=C_VALUE_METRIC, **kw)
    ax.bar([i + w / 2 for i in x], g["mean_cost"], width=w, label="Mean cost of imperfect", color=C_COST_METRIC, **kw)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{lbl}\n(n={int(g.loc[lbl]['n'])})" for lbl in g.index])
    ax.set_title("Metrics by manifest target bucket")
    ax.legend()
    save(fig, "fig_bucket_tradeoff")


def plot_value_by_survey_box(v: pd.DataFrame) -> None:
    surveys = sorted(v["survey"].unique())
    scm = survey_colors(surveys)
    data = [v.loc[v["survey"] == s, "value_over_random"].values for s in surveys]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bp = ax.boxplot(data, tick_labels=[survey_label(s) for s in surveys], showfliers=False, patch_artist=True)
    for patch, s in zip(bp["boxes"], surveys):
        patch.set_facecolor(scm[s])
        patch.set_alpha(0.75)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Value over random")
    ax.set_title("Spread of matched-$k$ value-over-random by survey (box: quartiles; whiskers: 1.5 IQR)")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    save(fig, "fig_value_over_random_by_survey_box")


def plot_signal_vs_value(v: pd.DataFrame) -> None:
    v = v.copy()
    v["signal"] = v["oracle_acc"] - v["majority_baseline"]
    surveys = sorted(v["survey"].unique())
    survey_to_color = survey_colors(surveys)
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in surveys:
        sub = v[v["survey"] == s]
        ax.scatter(sub["signal"], sub["value_over_random"], label=survey_label(s), color=survey_to_color[s], alpha=0.65, s=24)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("Oracle lift over majority ($\\mathrm{oracle}_{\\mathrm{acc}} - \\mathrm{majority}$)")
    ax.set_ylabel("Value over random")
    ax.set_title("Empirical ``room to learn'' vs.\ LLM surplus over random draws")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    plt.tight_layout()
    save(fig, "fig_signal_vs_value_over_random")


def plot_cost_vs_value_sized_k(v: pd.DataFrame) -> None:
    v = v.copy()
    surveys = sorted(v["survey"].unique())
    survey_to_color = survey_colors(surveys)
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for s in surveys:
        sub = v[v["survey"] == s]
        kk = sub["k_mapped"].fillna(1).astype(int).clip(lower=1, upper=15)
        sizes = 25 + kk * 16
        ax.scatter(
            sub["value_over_random"],
            sub["cost_of_imperfect"],
            s=sizes,
            label=survey_label(s),
            color=survey_to_color[s],
            alpha=0.55,
            edgecolors="white",
            linewidths=0.3,
        )
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("Value over random")
    ax.set_ylabel("Cost of imperfect")
    ax.set_title("Trade space with marker area $\\propto k_{\\mathrm{mapped}}$ (capped visualization scale)")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    plt.tight_layout()
    save(fig, "fig_scatter_cost_vs_value_sized_k")


def plot_hist_facets_by_survey(v: pd.DataFrame) -> None:
    surveys = sorted(v["survey"].unique())
    n = len(surveys)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2 * nrows), sharey=True)
    axes = np.atleast_2d(np.array(axes)).ravel()
    bins = np.linspace(v["value_over_random"].min(), v["value_over_random"].max(), 22)
    for i, s in enumerate(surveys):
        ax = axes[i]
        sub = v[v["survey"] == s]["value_over_random"]
        fill = SURVEY_HEX[i % len(SURVEY_HEX)]
        ax.hist(sub, bins=bins, color=fill, edgecolor="white", linewidth=0.4, alpha=0.88)
        ax.axvline(sub.mean(), color=C_MEAN_LINE, linestyle="--", linewidth=1.05)
        ax.set_title(survey_label(s) + rf" ($n={len(sub)}$)")
        ax.set_xlabel("Value over random")
    for j in range(len(surveys), len(axes)):
        axes[j].set_visible(False)
    axes[0].set_ylabel("Count")
    fig.suptitle("Value-over-random distributions by survey", y=1.02)
    plt.tight_layout()
    save(fig, "fig_value_hist_by_survey_grid")


def write_figure_manifest(summary: pd.DataFrame, valid_n: int) -> None:
    lines = [
        "% Auto-generated by analysis/build_paper_figures.py — do not edit by hand",
        "\\newcommand{\\PaperFigDir}{figures}",
        f"% Valid evaluation rows loaded: {valid_n}",
        "% Include in LaTeX: \\includegraphics[width=…]{\\PaperFigDir/fig_…}",
    ]
    (FIG_DIR / "FIGURE_MANIFEST.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    buckets = load_target_buckets()
    raw = load_grid_concat()
    if raw.empty:
        raise FileNotFoundError(
            "No grid_summary CSVs selected under outputs/. "
            "Run the grid or set GRID_SUMMARY_TAG to match an existing __<tag> suffix."
        )
    v = valid_rows(raw)
    summary = (
        v.groupby("survey")[["oracle_acc", "model_acc", "random_acc", "cost_of_imperfect", "value_over_random"]]
        .mean()
        .reset_index()
    )

    plot_survey_bars(summary)
    plot_value_histogram(v)
    plot_cost_vs_value(v)
    plot_bucket(v, buckets)
    plot_value_by_survey_box(v)
    plot_signal_vs_value(v)
    plot_cost_vs_value_sized_k(v)
    plot_hist_facets_by_survey(v)
    write_figure_manifest(summary, len(v))
    print(f"Wrote figures under {FIG_DIR} ({len(list(FIG_DIR.glob('*.pdf')))} PDF files)")


if __name__ == "__main__":
    main()
