#!/bin/bash
#SBATCH --job-name=baseline_evaluate_text_EN
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Baseline/logs/baseline_evaluate_text_EN_%x_%j.log

# Usage:
# sbatch --export=TASK=mgsm   evaluate_text.sh
# sbatch --export=TASK=msvamp evaluate_text.sh
# sbatch --export=TASK=xnli   evaluate_text.sh
# sbatch --export=TASK=x-csqa evaluate_text.sh
#
# Optional overrides:
#   sbatch --export=TASK=mgsm,LANGS=sw             evaluate_text.sh
#   sbatch --export=TASK=xnli,LANGS="sw en zh"     evaluate_text.sh
#   sbatch --export=TASK=mgsm,MAX_EXAMPLES=50       evaluate_text.sh

set -euo pipefail

TASK="${TASK:-mgsm}"
LANGS="${LANGS:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "TASK=$TASK"
nvidia-smi || true
echo

echo "=== Load modules ==="
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.2.2
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
echo "LLM_PATH=$LLM_PATH"
echo

if [ ! -d "$LLM_PATH" ]; then
  echo "ERROR: LLM snapshot not found: $LLM_PATH"
  exit 1
fi

# Build optional args
EXTRA_ARGS="--local-files-only"
if [ -n "$LANGS" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --langs $LANGS"
fi
if [ -n "$MAX_EXAMPLES" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --max-examples $MAX_EXAMPLES"
fi

echo "=== Start baseline evaluation: $TASK ==="
echo "LANGS=${LANGS:-<task default>}"
cd "$PROJECT_ROOT"
python -u Baseline/evaluate_text.py \
  --task "$TASK" \
  --model-id "$LLM_PATH" \
  $EXTRA_ARGS

echo "=== Done ==="
date
