# talk/ — 15-minute project presentation

Beamer (metropolis) deck covering both studies:
- **Paper 1** (EMNLP resubmission, the *updated* version from
  `synthetic_sampling_archive.tar.gz`): *Can open-weight LLMs predict individual
  survey responses?*
- **Paper 2** (this repo's pilot): *Do LLMs understand the conditional structure
  of attitudes — selection vs. use, and cross-national adaptation?*

## Files
| File | What |
|---|---|
| `slides.tex` | The deck. Equal weight P1/P2, mixed soc-sci+ML audience. |
| `slides.pdf` | Compiled output (33 pages: ~21 main incl. section dividers + 12 navigable backup slides; footer counts 14 content frames). |
| `speaker_notes.md` | Per-slide talking points + cumulative timing budget + Q&A prep. |
| `figures/p1/` | Paper 1 figures, extracted from the archive's EMNLP `latex/figures/`. |
| `figures/p2/` | Paper 2 figures, copied from `../paper/figures/`. |

## Backup / appendix slides
After the "Thank you" slide, `\appendix` holds 12 **backup slides** (not part of the
linear talk, not counted in the frame numbering). The first is a **hyperlinked index** —
click a button to jump to a topic; each backup has a "back to index" button. They cover:

- *Concept:* LLM selection vs. ML feature importance (the "answer key vs. student" defense).
- *Paper 1:* metrics & baselines · data/harmonisation/profiles · variance decomposition,
  scaling & perplexity validation · topics & the "Don't know" hedging failure.
- *Paper 2:* the full 6-step pipeline · metric definitions · the leakage screen · the
  output-format experiment · the concrete `stfgov`-Austria example · the JSON lower bound
  + base-rate behavioural trace.

In a PDF viewer the index buttons are clickable; if your presenter tool doesn't follow
links, the backups run in the order listed above (pp. 23–33).

## Build
TinyTeX (`pdflatex`) with `beamer` + `metropolis` (installed from the TeX Live
2024 historic repo). Run twice for the progress bar / frame fractions:

```sh
cd talk
pdflatex slides.tex
pdflatex slides.tex
```

Metropolis falls back to Computer Modern Sans under pdflatex (no Fira/XeLaTeX
needed) — compiles clean, 0 errors, 0 overfull boxes.

## Source of truth for numbers
- **Paper 1** figures/numbers: the EMNLP `paper.tex` in the archive.
- **Paper 2** numbers: `../paper/generated_current_state/ft_*.tex`
  (`ft_global_metrics`, `ft_uncertainty`, `ft_fixedk`, `ft_test2_adaptation`,
  `format_pilot_effect`). Every figure on a slide was read back from these.
- Conceptual framing (oracle = answer key, etc.): `../framing_and_comparisons.md`.

## To regenerate / swap a figure
Re-run the Paper 2 figure scripts (`analysis/build_paper_figures.py`,
`analysis/freetext_figures.py`) with miniconda `python`, then re-copy the PDF into
`figures/p2/`. Paper 1 figures are static (from the archive).
