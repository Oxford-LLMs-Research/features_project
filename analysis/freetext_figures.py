"""
Free-text (arm C) versions of the two main-text figures, for the restructured paper.

Same visual conventions as build_paper_figures.py (whose helpers are imported), but the
data source is the format-pilot arm-C scores (free-text elicitation, nemotron mapper,
model-chosen k, the 52 genuine cells), both selectors side-by-side:

  fig_ft_survey_accuracy_bars   : oracle / model / random matched-k accuracy by survey,
                                  one panel per selector
  fig_ft_scatter_cost_vs_value_sized_k : per-row trade space, marker area ~ realised k

Run:  python analysis/freetext_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_paper_figures import (  # noqa: E402
    BAR_EDGE, BAR_EDGELW, C_MODEL, C_ORACLE, C_RANDOM,
    save, survey_colors, survey_label,
)

from survey_features.layout import main_dir, resolve_main_scores_path  # noqa: E402

PILOT = main_dir(ROOT / "outputs")
SELECTORS = {"deepseek": "DeepSeek-V3.2", "kimi": "Kimi-K2.5"}
PRIMARY_DK = "nemotron"


def load_arm_c() -> pd.DataFrame:
    frames = []
    for key, label in SELECTORS.items():
        path = resolve_main_scores_path(key, ROOT / "outputs")
        if path is None:
            continue
        d = pd.read_csv(path)
        d["model_label"] = label
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["error"].isna() | (df["error"] == "")]
    df = df[(df["arm"] == "C") & (df["disambiguator"] == PRIMARY_DK)
            & (df["k_spec"] == "model")]
    for c in ("oracle_acc", "model_acc", "random_acc", "value_over_random",
              "cost_of_imperfect", "k"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["oracle_acc", "model_acc", "random_acc"])


def plot_survey_bars(v: pd.DataFrame) -> None:
    models = sorted(v["model_label"].unique())
    surveys = sorted(v["survey"].unique())
    x = range(len(surveys))
    w = 0.25
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4), sharey=True)
    axes = np.atleast_1d(axes)
    kw = dict(edgecolor=BAR_EDGE, linewidth=BAR_EDGELW)
    for ax, m in zip(axes, models):
        g = (v[v["model_label"] == m]
             .groupby("survey")[["oracle_acc", "model_acc", "random_acc"]]
             .mean().reindex(surveys))
        ax.bar([i - w for i in x], g["oracle_acc"], width=w, label="Oracle", color=C_ORACLE, **kw)
        ax.bar(list(x), g["model_acc"], width=w, label="Model", color=C_MODEL, **kw)
        ax.bar([i + w for i in x], g["random_acc"], width=w, label="Random $k$", color=C_RANDOM, **kw)
        ax.set_xticks(list(x))
        ax.set_xticklabels([survey_label(s) for s in surveys], rotation=25, ha="right")
        ax.set_title(m)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Mean CV accuracy")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Free-text elicitation, matched-$k$: oracle, model, random by survey", y=1.02)
    plt.tight_layout()
    save(fig, "fig_ft_survey_accuracy_bars")


def plot_cost_vs_value_sized_k(v: pd.DataFrame) -> None:
    surveys = sorted(v["survey"].unique())
    survey_to_color = survey_colors(surveys)
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for s in surveys:
        sub = v[v["survey"] == s]
        kk = sub["k"].fillna(1).astype(int).clip(lower=1, upper=15)
        sizes = 25 + kk * 16
        ax.scatter(sub["value_over_random"], sub["cost_of_imperfect"], s=sizes,
                   label=survey_label(s), color=survey_to_color[s], alpha=0.55,
                   edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("Value over random")
    ax.set_ylabel("Cost of imperfect")
    ax.set_title("Free-text trade space, marker area $\\propto$ realised $k$")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    plt.tight_layout()
    save(fig, "fig_ft_scatter_cost_vs_value_sized_k")


def main() -> None:
    v = load_arm_c()
    print(f"arm-C rows: {len(v)} "
          f"({', '.join(f'{m}: {n}' for m, n in v['model_label'].value_counts().items())})")
    plot_survey_bars(v)
    plot_cost_vs_value_sized_k(v)
    print("wrote fig_ft_survey_accuracy_bars + fig_ft_scatter_cost_vs_value_sized_k (pdf/png)")


if __name__ == "__main__":
    main()
