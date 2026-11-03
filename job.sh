#!/bin/bash
# Run one command on a compute node with the project's environment.
# src/launch.py submits this with sbatch and passes the command as arguments,
# for example: job.sh python -m src.evaluate --approach direct ...
set -euo pipefail

module --force purge
module load python/3.11.5
source "$SCRATCH/venvs/m2-align/bin/activate"

# Compute nodes have no internet. Every model must already be in the cache.
export HF_HOME="$SCRATCH/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job ${SLURM_JOB_ID:-local} on $(hostname): $*"
"$@"
