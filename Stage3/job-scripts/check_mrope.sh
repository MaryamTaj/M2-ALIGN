#!/bin/bash
#SBATCH --job-name=check_mrope
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage3/logs/check_mrope_%x_%j.log

# M-RoPE position-id diagnostic (Stage3/check_mrope_positions.py). No GPU
# needed -- it only loads the Qwen3-VL config/processor (local cache) and
# calls Qwen3VLModel.get_rope_index directly, no model weights.
#
# Usage:
#   sbatch Stage3/job-scripts/check_mrope.sh

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo

echo "=== Load modules ==="
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load cuda/13.2
module load arrow/18.1.0
echo

echo "=== Activate virtual environment ==="
source "$SCRATCH/venvs/m2-align/bin/activate"
python -V
echo

echo "=== Hugging Face cache (offline) ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Resolve to the cached snapshot dir directly (same convention as
# evaluate.sh/train.sh) instead of the bare repo id: the repo id path
# requires huggingface_hub to resolve "main" via refs/main, which is
# missing from this cache, so `from_pretrained(MODEL_ID, local_files_only=True)`
# fails with LocalEntryNotFoundError even though the snapshot is fully cached.
export LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
echo "LLM_PATH=$LLM_PATH"
echo

if [ ! -d "$LLM_PATH" ]; then
  echo "ERROR: LLM snapshot not found: $LLM_PATH"
  exit 1
fi

echo "=== Start M-RoPE diagnostic ==="
cd "$PROJECT_ROOT"
python -u Stage3/check_mrope_positions.py

echo "=== Done ==="
date
