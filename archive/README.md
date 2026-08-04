# archive/

Legacy and one-shot material kept next to the checkout so appendix / JSON-grid /
prelim results stay reproducible. **Never import from here** into `src/` or live
`scripts/` / `analysis/` — copy shared logic into `survey_features` instead.

| Kind | Examples | Still runnable? |
|------|----------|-----------------|
| **Replication runners** | `run_grid.py`, `alignment_analysis.py`, `uncertainty_analysis.py`, `prelim_*.py` | Yes — `python archive/<name>.py …` |
| **Spent one-shots** | `migrate_outputs_layout.py`, `stop_after_kimi_map.ps1`, `run_ensemble_mapping_v1.ps1` | Reference only |
| **Superseded reporters** | `oracle_report.py`, `oracle_diagnostics.py`, `grid_analysis.py` | Reference only |

The live free-text pipeline is `scripts/run_main.py`. Digests for current
experiments stay under `analysis/`.
