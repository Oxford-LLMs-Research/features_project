"""
Schematic of the full feature-selection pipeline, for page 1 of the paper.

Two converging tracks --- the MODEL (prior knowledge, no data) and the ORACLE (the data,
no outside knowledge) --- meeting at a fixed downstream scorer. Rendered with matplotlib to
a vector PDF (+ PNG) and included via \includegraphics, consistent with the other figures
(this TinyTeX has no tikz/standalone).

Run:  python analysis/pipeline_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"

# palette (matches build_paper_figures Okabe-Ito-ish accents)
BLUE = "#2C6FB5"
TEAL = "#1B998B"
ORANGE = "#D9772B"
INK = "#222222"
SUB = "#5a5a5a"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.bbox": "tight",
                     "savefig.dpi": 200})


def box(ax, cx, cy, w, h, title, sub, edge, fill, title_size=10.5, sub_size=8.0):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.15,rounding_size=0.6",
                       linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=INK, zorder=4)
        ax.text(cx, cy - h * 0.27, sub, ha="center", va="center", fontsize=sub_size,
                style="italic", color=SUB, zorder=4, wrap=True)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=INK, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=INK, rad=0.0, lw=1.3):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=13, linewidth=lw,
                        color=color, zorder=2,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def lane(ax, x0, y0, x1, y1, edge, fill):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle="round,pad=0.2,rounding_size=0.8",
                       linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=1)
    ax.add_patch(p)


def build():
    fig, ax = plt.subplots(figsize=(9.8, 4.7))
    ax.set_xlim(0, 114)
    ax.set_ylim(0, 50)
    ax.axis("off")

    BW, BH = 19, 10
    y_m, y_o = 40, 17            # model / oracle lane y-centres
    xm = [12, 35, 58, 81]        # model boxes
    xo = [12, 46.5, 81]          # oracle boxes (rank sits under map)

    # lane backgrounds
    lane(ax, 2, y_m - BH / 2 - 3.4, 92, y_m + BH / 2 + 1.3, BLUE, "#2C6FB508")
    lane(ax, 2, y_o - BH / 2 - 1.3, 92, y_o + BH / 2 + 1.3, TEAL, "#1B998B0d")
    ax.text(4, y_m + BH / 2 + 2.7, "MODEL", fontsize=12, fontweight="bold", color=BLUE)
    ax.text(23, y_m + BH / 2 + 2.7, "— prior knowledge, no data", fontsize=10,
            style="italic", color=BLUE)
    ax.text(4, y_o + BH / 2 + 2.7, "ORACLE", fontsize=12, fontweight="bold", color=TEAL)
    ax.text(25, y_o + BH / 2 + 2.7, "— the data, no outside knowledge", fontsize=10,
            style="italic", color=TEAL)

    TS, SS = 12.0, 9.0
    # model lane boxes
    box(ax, xm[0], y_m, BW, BH, "Survey question", "shown alone;\nno candidate list",
        "#888", "#f3f3f3", TS, SS)
    box(ax, xm[1], y_m, BW, BH, "Free-text\nelicitation",
        "selector LLM;\nunprompted / country-named", BLUE, "#eaf1f9", TS, SS)
    box(ax, xm[2], y_m, BW, BH, "Extract &\ntype features",
        "fixed extractor\n(Qwen3-235B)", BLUE, "#eaf1f9", TS, SS)
    box(ax, xm[3], y_m, BW, BH, "Map to survey\nvariables",
        "embed → top-20\n→ disambiguate", BLUE, "#eaf1f9", TS, SS)

    # oracle lane boxes
    box(ax, xo[0], y_o, BW, BH, "Survey\nmicrodata", "respondents\n× variables",
        "#888", "#f3f3f3", TS, SS)
    box(ax, xo[1], y_o, BW, BH, "Permutation\nimportance",
        "AutoGluon:\nthe “answer key”", TEAL, "#e8f5f2", TS, SS)
    box(ax, xo[2], y_o, BW, BH, "Oracle ranking",
        "top-$k$ predictive\nfeatures", TEAL, "#e8f5f2", TS, SS)

    # scoring (converge) — three-way comparison folded in here
    xs, ys = 103, 28.5
    box(ax, xs, ys, 19, 15, "Scoring",
        "fixed XGBoost, 5-fold CV\nmodel vs. oracle-$k$\nvs. random-$k$\nat matched budget $k$",
        ORANGE, "#fbeede", 12.5, 8.6)

    # arrows: lanes
    for a, b in zip(xm, xm[1:]):
        arrow(ax, a + BW / 2, y_m, b - BW / 2, y_m, BLUE)
    for a, b in zip(xo, xo[1:]):
        arrow(ax, a + BW / 2, y_o, b - BW / 2, y_o, TEAL)

    # converge into scoring
    arrow(ax, xm[3] + BW / 2, y_m, xs - 9.5, ys + 4.2, BLUE, rad=-0.12, lw=1.6)
    ax.text(93.5, 37.5, r"$S_{\mathrm{model}}$", fontsize=10, color=BLUE, ha="center")
    arrow(ax, xo[2] + BW / 2, y_o, xs - 9.5, ys - 4.2, TEAL, rad=0.12, lw=1.6)
    ax.text(94, 19.5, "oracle top-$k$", fontsize=9.5, color=TEAL, ha="center")

    # metrics bar (two centred lines)
    mb = FancyBboxPatch((6, 1.0), 104, 9.0,
                        boxstyle="round,pad=0.2,rounding_size=0.8",
                        linewidth=1.2, edgecolor=ORANGE, facecolor="#fdf6ee", zorder=3)
    ax.add_patch(mb)
    ax.text(58, 7.0,
            "Outputs:  captured importance  ·  value over random  ·  oracle-rank percentile",
            fontsize=10, color=INK, va="center", ha="center", fontweight="bold")
    ax.text(58, 3.4,
            "Tests:  T1 selection quality  ·  T2 cross-national adaptation",
            fontsize=10, color=INK, va="center", ha="center", fontweight="bold")
    arrow(ax, xs, ys - 7.5, xs, 10.3, ORANGE, rad=0.0, lw=1.6)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_pipeline.{ext}")
    plt.close(fig)
    print(f"wrote {FIG_DIR/'fig_pipeline.pdf'} (+ .png)")


if __name__ == "__main__":
    build()
