#!/bin/bash
# Oracle map for the registered confirmatory grid, as a SLURM array on ARC.
#
# One array task = one contiguous shard of data/confirmatory_grid_cells.csv
# (file order is grouped by survey, so each task mostly loads one survey frame).
# Resumable: a resubmitted task skips cells whose oracle_meta.json already
# records contract v4 — resubmit the same array after any failure.
#
# BEFORE FIRST SUBMIT — edit for your setup:
#   * the module load line (whatever Python module built ~/envs/features_oracle)
#   * partition/time/mem if your allocation differs
#   * N_SHARDS must equal the --array range size (default 8: --array=0-7)
#
# Submit from the repo root (so .env is found):
#   mkdir -p logs && sbatch scripts/arc/run_oracles.sbatch
# Smoke batch first (about 5 cells, small pool):
#   python scripts/rerun_oracles.py --cells-csv data/confirmatory_grid_cells.csv \
#       --shard 0/220 --processes 2
#
#SBATCH --job-name=oracle-grid
#SBATCH --partition=medium
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/oracle_%A_%a.out

set -euo pipefail

module purge
module load Python/3.11        # EDIT: the module features_oracle was built with
source ~/envs/features_oracle/bin/activate

cd "${SLURM_SUBMIT_DIR}"

# Compute nodes may lack internet: everything (MiniLM included) must come from
# the caches named in .env. If the smoke batch fails here, warm the HF cache
# from the login node first.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

N_SHARDS="${N_SHARDS:-8}"      # must match the --array range above
PROCS="${PROCS:-12}"           # process workers per node (~4 cores each at 48)

python scripts/rerun_oracles.py \
    --cells-csv data/confirmatory_grid_cells.csv \
    --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}" \
    --processes "${PROCS}"
