# Oracle weekend run — hand-off runbook (5–8 September 2026)

Written for the collaborator running half of the oracle computation on her own
laptop. Everything here is self-contained; the only prerequisites are access to the
GitHub repo (`Oxford-LLMs-Research/features_project`, branch `main`) and the shared
Dropbox folder `features_project`.

## 1. What we are computing and why it is split

An **oracle** is, for one survey question in one country (a **cell**), a data-driven
ranking of every other question in that survey by how much it helps predict the
target — AutoGluon models fitted on the real microdata, then permutation importance.
It is the benchmark the LLM feature selections are judged against. The **confirmatory
grid** (the fixed list of cells whose numbers go in the paper) has 1,103 cells, listed
in `data/confirmatory_grid_cells.csv`. None has a usable oracle yet (eight had one
fitted under a 60-second budget that cut the model bag short; they are archived and
recomputed with the rest). The LLM side
(generation, extraction, mapping) is already done for the six selectors; scoring is
blocked on the oracles. Each cell takes roughly a quarter of an hour of one CPU worker,
so two laptops over a weekend is the difference between scoring on Monday and scoring
on Thursday.

Two partitions, split by survey so no two machines ever touch the same cell folder:

| Machine | Surveys | `role == confirmatory` | `role == oracle_only` |
|---|---|---|---|
| **A** (Maksim, `DESKTOP-IL78MBC`) | afrobarometer, arabbarometer, asianbarometer | 180 | 333 |
| **B** (you) | ess_wave_11, latinobarometer, wvs | 180 | 410 |

**Cell role** — every grid row is either *confirmatory* (one of the 360 cells the LLM
selectors were actually run on; scoring needs these) or *oracle-only* (extra countries
per question that get an oracle for the transportability analysis but no LLM run).
Confirmatory cells come first on both machines; oracle-only cells are best effort.

Both machines write straight into the shared Dropbox tree
`features_project/outputs/cache/cells/<target>_<country>/`. Because the partitions
are disjoint, Dropbox simply merges them. **Never run a survey that is not in your
row of the table.**

## 2. What you need before starting

- A laptop that will stay **awake, plugged in and online** all weekend (Settings →
  System → Power: sleep *Never* when plugged in). AutoGluon's time budget is wall
  clock — a sleeping laptop produces a cut-short fit, not a paused one.
- Dropbox fully synced, including `features_project/data/` (~550 MB of survey files
  plus `data/metadata/`) and `features_project/outputs/`.
- Conda (Miniconda is fine). Python 3.9 matches machine A; 3.10 or 3.11 also work.
- At least 8 GB of RAM free; 16 GB is comfortable for two workers.

## 3. One-time setup (PowerShell, ~20 minutes)

```powershell
git clone https://github.com/Oxford-LLMs-Research/features_project.git
cd features_project
conda create -n features python=3.9 -y
conda activate features
pip install -e ".[oracle]"
```

The `[oracle]` extra pins AutoGluon and the tree libraries to machine A's versions
(AutoGluon 1.4.0, LightGBM 4.6.0, XGBoost 2.1.4, CatBoost 1.2.10, scikit-learn 1.6.1,
torch 2.6.0 CPU). Check the install:

```powershell
python -c "import autogluon.tabular as a, lightgbm, xgboost, catboost, sklearn; print(a.__version__, lightgbm.__version__, xgboost.__version__, catboost.__version__, sklearn.__version__)"
# expected: 1.4.0 4.6.0 2.1.4 1.2.10 1.6.1
```

**Data config.** Copy `data_config.example.yaml` to `data_config.yaml` (gitignored)
and replace `<DROPBOX>` with your Dropbox root, e.g. `C:/Users/<you>/Dropbox`. The
three paths point at the shared `features_project/data` folder; nothing is copied.

**Environment file.** Create `.env` in the repo root (gitignored) with exactly:

```
DATA_CONFIG_PATH=C:/Users/<you>/features_project/data_config.yaml
SURVEY_FEATURES_OUTPUTS=C:\Users\<you>\Dropbox\features_project\outputs
LLM_BASE_URL=https://api.studio.nebius.com/v1/
LLM_API_KEY=not-needed-for-oracles
LLM_MODEL=moonshotai/Kimi-K2.6
```

Oracles never call an LLM, so the key is a placeholder. Verify both paths resolve:

```powershell
python -c "from survey_features.config import OUTPUTS_DIR; print(OUTPUTS_DIR)"
python -c "import os, survey_features.config; from survey_features.surveys import load_survey; d, m = load_survey('latinobarometer', os.environ['DATA_CONFIG_PATH']); print(d.shape)"
# expected: your Dropbox outputs path, then (19205, 275)
```

**Keep AutoGluon scratch out of Dropbox.** Fits write temporary model files under
`outputs/.tmp`. On your machine that folder must exist and be marked ignored, or
Dropbox will try to sync gigabytes of scratch and lock files mid-fit:

```powershell
$t = "C:\Users\<you>\Dropbox\features_project\outputs\.tmp"
New-Item -ItemType Directory -Force $t | Out-Null
Set-Content -Path $t -Stream com.dropbox.ignored -Value 1
Get-Content -Path $t -Stream com.dropbox.ignored    # prints 1
```

**Run the unit tests once** (no data needed, ~10 s): `python -m pytest tests -q`.

## 4. Smoke test (~20 minutes) — do this before the real run

The same two cells machine A already fitted, into a *local* scratch folder so nothing
touches the shared cache:

```powershell
python scripts/rerun_oracles.py --cells-csv data/confirmatory_grid_cells.csv `
    --role confirmatory --survey latinobarometer --limit 2 `
    --runtime-mode quick --autogluon-time-limit 600 --processes 2 --output-dir C:\temp\oracle_smoke
```

Then compare against machine A's copy of the same two cells, which sits on Dropbox
under `outputs/analysis/oracle_smoke_2026-09-04/DESKTOP-IL78MBC/cache/cells`:

```powershell
python scripts/oracle_provenance_census.py --cells-dir C:\temp\oracle_smoke\cache\cells
python scripts/oracle_provenance_census.py --compare `
    C:\Users\<you>\Dropbox\features_project\outputs\analysis\oracle_smoke_2026-09-04\DESKTOP-IL78MBC\cache\cells `
    C:\temp\oracle_smoke\cache\cells
```

What "comparable" means, and what to do if it is not:

- **Same model bag.** The census prints `n_models` per fold; it must be the same on
  every fold and match machine A: **11 for binary/nominal targets, 9 for
  ordinal/continuous** (regression drops two of the four forest variants — that is
  the preset, not a cut). The two smoke cells are binary, so expect 11; in the real
  run most cells are ordinal and show 9. `bag_same_all_folds` must be `True`. AutoGluon
  works inside a wall-clock budget per fold; a laptop that runs out of it fits
  *fewer* models under the same settings, and that — not the laptop itself — is
  what would make the two halves non-comparable. The 600 s budget exists so it
  never binds: machine A needs 70–106 s per fold for the full bag, so even a
  laptop four times slower finishes. If a fold still shows fewer than 11 models,
  stop and message me — do not start the real run.
- **Rankings agree as well as two folds do.** The compare table shows the top-10
  overlap between your cell and A's, next to each cell's own between-fold overlap.
  The first should not be clearly below the second. It will not be identical — the
  fit is not bit-reproducible across machines — and that is fine.

Send me the two printouts before starting the real run.

## 5. The real run

Open one PowerShell window in the repo, activate the env, then:

```powershell
# Pin native threads so N workers do not each grab every core (AutoGluon's budget
# is wall clock, so oversubscription is paid in fit quality, not just speed).
$procs = 2                                   # see rule below
$threads = [math]::Max(1, [math]::Floor([Environment]::ProcessorCount / $procs))
$env:OMP_NUM_THREADS = $threads; $env:OPENBLAS_NUM_THREADS = $threads
$env:MKL_NUM_THREADS = $threads;  $env:NUMEXPR_NUM_THREADS = $threads
$log = "C:\Users\<you>\Dropbox\features_project\outputs\logs\oracle_run_$(hostname)_$(Get-Date -Format yyyyMMdd).log"

# Phase 1 — the 180 confirmatory cells of your surveys (must finish; ~25–45 h at 2 workers)
python -u scripts/rerun_oracles.py --cells-csv data/confirmatory_grid_cells.csv `
    --role confirmatory --survey ess_wave_11 latinobarometer wvs `
    --runtime-mode quick --autogluon-time-limit 600 --processes $procs | Tee-Object -FilePath $log -Append

# Phase 2 — the 410 oracle-only cells of your surveys (best effort; stop Monday 08:00)
python -u scripts/rerun_oracles.py --cells-csv data/confirmatory_grid_cells.csv `
    --role oracle_only --survey ess_wave_11 latinobarometer wvs `
    --runtime-mode quick --autogluon-time-limit 600 --processes $procs | Tee-Object -FilePath $log -Append
```

**Worker rule.** `--processes` = logical cores ÷ 4, rounded down, at least 1, at most
3. (8 logical cores → 2; 12–16 → 3.) Each worker gets its own share of cores and its
own copy of the survey in memory; more workers than that starve the fits.

**Settings that must not change.** `--runtime-mode quick --autogluon-time-limit 600`
on every cell: AutoGluon's `medium_quality` preset (the registered tier), 5 folds,
5 shuffle repeats, no neural FastAI model, and a per-fold budget large enough that
the full 11-model bag always finishes — the fit stops on its own when the bag is
done, so the 600 is a ceiling, not a cost. Machine A runs the identical flags.

**What you will see.** One line per finished cell:
`[17/180] B18ST x Chile  143/197 positive, regression, ceiling@10=0.61, 960s`.
Expect roughly 15–20 minutes per cell per worker on a fast desktop and 25–45 on a
laptop with four cores per worker; two workers therefore land about 3–5 cells an
hour.
A `[error]` line is one failed cell; the run continues. A `[pool] executor broke`
message means a worker died (usually memory); the script rebuilds the pool with fewer
workers by itself. Nothing prints while a cell is fitting — that is normal.

**Interrupting and resuming.** The run is resumable: rerun the identical command and
it skips every cell whose `oracle_meta.json` already carries contract version 4. To
stop, press Ctrl+C once and wait; if python processes linger, close the window. Cells
that were mid-fit are simply recomputed next time.

**Dropbox.** Leave it running. If the log fills with repeated `PermissionError`
lines on `oracle.csv` or `oracle_meta.json`, pause Dropbox syncing for the rest of
the run and resume it when the run ends — the files sync then.

## 6. Rules of the road

1. Run only your surveys, only through the two commands above, only from `origin/main`
   at the commit you cloned. Do not edit code; if something needs changing, message
   me and I push a fix that we both pull.
2. Do not run anything else that writes under `outputs/` (no `run_main.py`, no
   scoring, no leakage audit). Reading is fine.
3. Do not delete or rename anything under `outputs/cache/cells/`, including cells
   that look stale (the 88 folders whose metadata says contract version 3 are known
   and handled on Monday).
4. Do not use `--force`, `--balanced`, `--best`, or a different `--n-repeats`.

## 7. Monday morning

Stop your run by 08:00 (Ctrl+C), let Dropbox finish syncing, and send me the output of:

```powershell
python scripts/oracle_provenance_census.py
tail -n 5 $log
```

I then run the provenance census across both halves, the leakage audit, the textbook
baseline, and scoring; nothing on your side is needed after that. Cells that did not
finish are picked up by a plain rerun of the same command on either machine — just
say which survey you stopped in.

## Appendix — machine A reference

- Host `DESKTOP-IL78MBC`: Intel i9-10900K, 10 cores / 20 threads, 34 GB RAM,
  Windows 11, Python 3.9.1, AutoGluon 1.4.0, LightGBM 4.6.0, XGBoost 2.1.4,
  CatBoost 1.2.10, scikit-learn 1.6.1, torch 2.6.0+cpu, numpy 2.0.2, pandas 2.3.3.
- Machine A runs `--processes 3` with the same flags.
- Smoke test on machine A, 2026-09-04, two Latinobarómetro confirmatory cells
  (P19N El Salvador, P19N Peru), two workers with 10 cores each:
  - at the registered 60 s budget: 3–8 models per fold, bags differ between folds,
    12.6 min for the pair — the budget was binding, so the bag depended on the
    machine (this is why the weekend uses 600);
  - at 600 s (300 s tested, same outcome): 11 models on every fold, 70–106 s of fit
    per fold, 16.0 min for the pair. In the real run with threads pinned as in §5,
    machine A's folds fit in 10–20 s and whole cells take 1.5–4 min at 3 workers,
    so the partitions should finish well inside the weekend;
  - top-10 overlap between the two budgets: 0.33 and 0.43 — larger than the
    between-fold overlap (0.10–0.21), i.e. the budget changes the ranking more
    than fold noise does. Both copies are on Dropbox under
    `outputs/analysis/oracle_smoke_2026-09-04/DESKTOP-IL78MBC/` (`cache/cells` = the
    600 s reference you compare against; `quick60/` = the cut-short version).
