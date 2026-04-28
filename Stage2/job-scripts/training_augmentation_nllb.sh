#!/bin/bash
#SBATCH --job-name=stage2_aug
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=training_stage2_aug_%j.log

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE2="$PROJECT_ROOT/Stage2"
STAGE1="$PROJECT_ROOT/Stage1"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
STAGE1_MAPPING_CKPT="$STAGE1/outputs/MindMerger/nllb_corpus/mapping/pytorch_model.bin"

EN_DATA="$STAGE2/data/task_specialization_en.jsonl"
TRANSLATED_DATA="$STAGE2/data/task_specialization_translated.jsonl"
OUTPUT_DIR="$STAGE2/outputs/augmentation"

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
  echo "ERROR: Stage1 mapping checkpoint not found: $STAGE1_MAPPING_CKPT"
  exit 1
fi
if [ ! -f "$EN_DATA" ]; then
  echo "ERROR: English task-specialization data not found: $EN_DATA"
  exit 1
fi
if [ ! -f "$TRANSLATED_DATA" ]; then
  echo "ERROR: Translated task-specialization data not found: $TRANSLATED_DATA"
  exit 1
fi

echo "=== Start Stage2 augmentation training ==="
cd "$PROJECT_ROOT"
python -u Stage2/run_augmentation.py \
  --stage1-mapping-ckpt "$STAGE1_MAPPING_CKPT" \
  --english-data "$EN_DATA" \
  --translated-data "$TRANSLATED_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --mt-path "$MT_PATH" \
  --llm-path "$LLM_PATH" \
  --epochs 3 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --grad-accum 8 \
  --max-seq-len 256 \
  --max-gen-len 64 \
  --use-wandb \
  --wandb-mode offline \
  --wandb-project m2-align \
  --wandb-run-name stage2-augmentation \
  --local-files-only

