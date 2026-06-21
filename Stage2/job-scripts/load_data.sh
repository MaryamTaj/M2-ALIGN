#!/bin/bash
#SBATCH --job-name=stage2_load_data
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage2/logs/stage2_load_data_%j.log

# Usage: TASK=mgsm sbatch load_data.sh
#   Supported tasks: mgsm, msvamp, xnli, xcsqa
#   Default: runs all four tasks sequentially.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE2="$PROJECT_ROOT/Stage2"
MINDMERGER_DATA="$PROJECT_ROOT/mindmerger_data"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo

echo "=== Load modules ==="
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load arrow/18.1.0
echo

echo "=== Activate virtual environment ==="
source "$SCRATCH/venvs/m2-align/bin/activate"
python -V
echo

echo "=== Hugging Face cache config ==="
export HF_HOME="$SCRATCH/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo

if [ ! -d "$MINDMERGER_DATA" ]; then
  echo "ERROR: mindmerger_data directory not found: $MINDMERGER_DATA"
  exit 1
fi

cd "$PROJECT_ROOT"

run_task() {
  local task="$1"
  local out_dir="$STAGE2/data/stage2/$task"
  echo "--- Loading task: $task → $out_dir ---"
  python -u Stage2/load_data.py \
    --task       "$task" \
    --data_dir   "$MINDMERGER_DATA" \
    --output_dir "$out_dir" \
    --seed       42
  echo "Done: $task"
  echo
}

if [ -n "${TASK:-}" ]; then
  run_task "$TASK"
else
  echo "=== No TASK set — running all four tasks ==="
  run_task mgsm
  run_task msvamp
  run_task xnli
  run_task xcsqa
fi

echo "=== Done ==="
date
