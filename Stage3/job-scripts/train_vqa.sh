#!/bin/bash
#SBATCH --job-name=stage3b_train_vqa
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage3/logs/stage3b_train_vqa_%j.log

# Stage 3b: VQA augmentation training (AugmentedVisualMindMerger).
#   Reads GQA-translated data from Stage3/data/stage3b/*.jsonl
#   Warm-starts directly from Stage 2's vision-mapping checkpoint (chained
#   lineage: Stage 1 -> Stage 2 -> Stage 3 (VQA)). Stage 3a (text-only
#   augmentation) is no longer part of the pipeline -- see train_text.py /
#   train_text.sh if you want to run it separately.
#   Saves checkpoint to Stage3/outputs/stage3b/mapping/pytorch_model.bin
#
# BEFORE SUBMITTING: download and extract GQA's official image archive
# (https://nlp.stanford.edu/data/gqa/images.zip) to GQA_IMAGES_DIR below.
#
# Usage: sbatch Stage3/job-scripts/train_vqa.sh

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE2="$PROJECT_ROOT/Stage2"
STAGE3="$PROJECT_ROOT/Stage3"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
GQA_IMAGES_DIR="$STAGE3/data/gqa/images"
STAGE2_MAPPING_CKPT="$STAGE2/outputs/wit/mapping/pytorch_model.bin"

DATA_DIR="$STAGE3/data/stage3b"
OUTPUT_DIR="$STAGE3/outputs/stage3b"

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
module load cudacore/.12.2.2
module load arrow/18.1.0
echo

echo "=== Activate virtual environment ==="
source "$SCRATCH/venvs/m2-align/bin/activate"
python -V
echo

echo "=== Hugging Face cache/offline config ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "LLM_PATH=$LLM_PATH"
echo "MT_PATH=$MT_PATH"
echo "GQA_IMAGES_DIR=$GQA_IMAGES_DIR"
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
if [ ! -d "$GQA_IMAGES_DIR" ]; then
  echo "ERROR: GQA images directory not found: $GQA_IMAGES_DIR"
  echo "       Download+extract https://nlp.stanford.edu/data/gqa/images.zip there."
  exit 1
fi
if [ ! -f "$STAGE2_MAPPING_CKPT" ]; then
  echo "ERROR: Stage 2 vision-mapping checkpoint not found: $STAGE2_MAPPING_CKPT"
  echo "       Run Stage2/job-scripts/train.sh first."
  exit 1
fi
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls "$DATA_DIR"/*.jsonl 2>/dev/null)" ]; then
  echo "ERROR: No .jsonl files found in DATA_DIR: $DATA_DIR"
  echo "       Run: sbatch Stage3/job-scripts/load_vqa_data.sh"
  exit 1
fi

echo "=== Start Stage 3b VQA training ==="
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "STAGE2_MAPPING_CKPT=$STAGE2_MAPPING_CKPT"
echo
cd "$PROJECT_ROOT"
python -u Stage3/train_vqa.py \
  --data-dir    "$DATA_DIR" \
  --images-dir  "$GQA_IMAGES_DIR" \
  --output-dir  "$OUTPUT_DIR" \
  --init-mapping-ckpt "$STAGE2_MAPPING_CKPT" \
  --mt-path     "$MT_PATH" \
  --llm-path    "$LLM_PATH" \
  --epochs      3 \
  --lr          2e-5 \
  --train-batch-size 4 \
  --eval-batch-size  4 \
  --grad-accum  8 \
  --max-mt-seq-len 256 \
  --max-seq-len 256 \
  --max-gen-len 16 \
  --use-wandb \
  --wandb-mode    offline \
  --wandb-project m2-align \
  --wandb-run-name "stage3b-vqa" \
  --local-files-only

echo "=== Done ==="
date
