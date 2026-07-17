#!/bin/bash
#SBATCH --job-name=stage3a_train_text
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage3/logs/stage3a_train_text_%j.log

# Usage: TASK=mgsm sbatch train_text.sh
#   Supported tasks: mgsm, msvamp (xnli/xcsqa dropped for now — see
#   Stage3/load_text_data.py)
#   Reads data from Stage3/data/stage3a/<task>/<task>.jsonl
#   Warm-starts from Stage 2's vision-mapping checkpoint (chained lineage:
#   Stage 1 -> Stage 2 -> Stage 3a -> Stage 3b)
#   Saves checkpoint to Stage3/outputs/stage3a/<task>/mapping/pytorch_model.bin

set -euo pipefail

TASK="${TASK:-mgsm}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE2="$PROJECT_ROOT/Stage2"
STAGE3="$PROJECT_ROOT/Stage3"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
STAGE2_MAPPING_CKPT="$STAGE2/outputs/wit/mapping/pytorch_model.bin"

DATA_DIR="$STAGE3/data/stage3a/$TASK"
OUTPUT_DIR="$STAGE3/outputs/stage3a/$TASK"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then
      MT_PATH="$d"
      break
    fi
  done
fi

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
module load gcc/12.3
module load cuda/13.2
module load arrow/18.1.0
echo

echo "=== Activate virtual environment ==="
source "$SCRATCH/venvs/m2-align/bin/activate"
python -V
python -m pip -V
echo

echo "=== Hugging Face cache/offline config ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HF_HOME=$HF_HOME"
echo "LLM_PATH=$LLM_PATH"
echo "MT_PATH=$MT_PATH"
echo

echo "=== Load secrets (.tokens) ==="
if [ -f "$PROJECT_ROOT/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.tokens"
  echo "Loaded .tokens file"
else
  echo "WARNING: .tokens file not found; W&B may run in offline mode without API key"
fi
echo

if [ ! -d "$LLM_PATH" ]; then
  echo "ERROR: LLM snapshot path not found: $LLM_PATH"
  exit 1
fi
if [ ! -d "$MT_PATH" ]; then
  echo "ERROR: MT snapshot path not found: $MT_PATH"
  exit 1
fi
if [ ! -f "$STAGE2_MAPPING_CKPT" ]; then
  echo "ERROR: Stage 2 vision-mapping checkpoint not found: $STAGE2_MAPPING_CKPT"
  echo "       Run Stage2/job-scripts/train.sh first."
  exit 1
fi
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls "$DATA_DIR"/*.jsonl 2>/dev/null)" ]; then
  echo "ERROR: No .jsonl files found in DATA_DIR: $DATA_DIR"
  echo "       Run: TASK=$TASK sbatch Stage3/job-scripts/load_text.sh"
  exit 1
fi

echo "=== Start Stage 3a training: $TASK ==="
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "STAGE2_MAPPING_CKPT=$STAGE2_MAPPING_CKPT"
echo
cd "$PROJECT_ROOT"
python -u Stage3/train_text.py \
  --task       "$TASK" \
  --init-mapping-ckpt "$STAGE2_MAPPING_CKPT" \
  --data-dir   "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mt-path    "$MT_PATH" \
  --llm-path   "$LLM_PATH" \
  --epochs     3 \
  --lr         2e-5 \
  --train-batch-size 2 \
  --eval-batch-size  2 \
  --grad-accum 8 \
  --max-seq-len 512 \
  --max-gen-len 512 \
  --use-wandb \
  --wandb-mode    offline \
  --wandb-project m2-align \
  --wandb-run-name "stage3a-$TASK" \
  --local-files-only

echo "=== Done ==="
date
