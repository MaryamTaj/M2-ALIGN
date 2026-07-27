#!/bin/bash
#SBATCH --job-name=stage2_train_wit
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage2/logs/stage2_train_wit_%j.log

# Usage: sbatch train.sh
#   Reads data from Stage2/data/wit_pairs.jsonl (see Stage2/load_image.py)
#   Warm-starts the Mapping layer from Stage 1's checkpoint (chained
#   lineage: Stage 1 -> Stage 2 -> Stage 3a -> Stage 3b)
#   Saves checkpoint to Stage2/outputs/wit/pytorch_model.bin

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE1="$PROJECT_ROOT/Stage1"
STAGE2="$PROJECT_ROOT/Stage2"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
STAGE1_MAPPING_CKPT="$STAGE1/outputs/M2-ALIGN/nllb_corpus/mapping/pytorch_model.bin"

DATA_PATH="$STAGE2/data/wit_pairs.jsonl"
OUTPUT_DIR="$STAGE2/outputs/wit"

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
if [ ! -f "$STAGE1_MAPPING_CKPT" ]; then
  echo "ERROR: Stage 1 mapping checkpoint not found: $STAGE1_MAPPING_CKPT"
  echo "       Run Stage1/job-scripts/train.sh first."
  exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: Data file not found: $DATA_PATH"
  echo "       Run: python Stage2/load_image.py --languages bn --n-per-language 100000 --output-dir Stage2/data"
  exit 1
fi

echo "=== Start Stage 2 training ==="
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "STAGE1_MAPPING_CKPT=$STAGE1_MAPPING_CKPT"
echo

RESUME_ARGS=()
TRAINING_STATE="$OUTPUT_DIR/training_state.pt"
if [ -f "$TRAINING_STATE" ]; then
  echo "Found existing training_state.pt — resuming: $TRAINING_STATE"
  RESUME_ARGS=(--resume-from-checkpoint "$TRAINING_STATE")
fi
echo

cd "$PROJECT_ROOT"
python -u Stage2/train.py \
  --data-path  "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --image-cache-dir "$STAGE2/data/image_cache" \
  --stage1-mapping-ckpt "$STAGE1_MAPPING_CKPT" \
  --mt-path    "$MT_PATH" \
  --llm-path   "$LLM_PATH" \
  --epochs     3 \
  --lr         2e-5 \
  --train-batch-size 4 \
  --eval-batch-size  4 \
  --grad-accum 8 \
  --max-mt-seq-len 512 \
  --max-gen-len 512 \
  --save-steps 200 \
  --use-wandb \
  --wandb-mode    offline \
  --wandb-project m2-align \
  --wandb-run-name "stage2-wit" \
  --local-files-only \
  "${RESUME_ARGS[@]}"

echo "=== Done ==="
date
