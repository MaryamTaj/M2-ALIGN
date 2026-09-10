#!/bin/bash
#SBATCH --job-name=stage0_sft
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage0/logs/stage0_sft_%x_%j.log

# Stage 0 -- English-GQA SFT of Qwen3-VL-8B-Instruct. Produces the "MonoVQA"
# checkpoint used as BOTH a baseline row AND the frozen --llm-path for the
# pooled Stage 1 / 2 / 3b runs. Mirrors MindMerger's Stage 0 (English-task
# SFT -> MonoReason). Trains on the SAME Stage3/data/english.jsonl (30k,
# seed 42) whose translations feed Stage 3b.
#
# HARDWARE NOTE: Narval is A100-SXM4-40GB. A full-weight 8B SFT does NOT fit
# one 40 GB card (bf16 weights 16 GB + grads 16 GB alone) and full-FT for a
# 30k-example job would need 2-4 of the node's GPUs + ZeRO-3 + CPU offload
# for ~15 h -- disproportionate. So:
#
#   default (MODE=lora)  LoRA on the LM projections, merged before saving.
#                        Fits one A100-40 with margin (~24 GB), ~3-5 h for
#                        3 epochs. Needs `pip install peft` on the
#                        workstation once (not in the m2-align venv).
#   MODE=full            Full-weight LM SFT, DeepSpeed ZeRO-3 + CPU offload
#                        across the node's 4 A100s. Closest MetaMath-style
#                        mirror; ~15 h, whole node. Also pass --gres below.
#
# MindMerger's Stage 0 backbone (MetaMath) was 3 epochs -> EPOCHS=3 default.
#
# Usage:
#   sbatch Stage0/job-scripts/train.sh
#   MODE=full sbatch --gres=gpu:4 --cpus-per-task=48 --mem=490G --time=20:00:00 Stage0/job-scripts/train.sh

set -euo pipefail

MODE="${MODE:-lora}"
EPOCHS="${EPOCHS:-3}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
DATA_ROOT="$SCRATCH/M2-ALIGN"
STAGE3="$DATA_ROOT/Stage3"

BASE_LLM="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DATA_PATH="$STAGE3/data/english.jsonl"
GQA_IMAGES_DIR="$STAGE3/data/gqa/images"
OUTPUT_DIR="$DATA_ROOT/Stage0/outputs/qwen3vl-8b-gqa"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  MODE=$MODE  EPOCHS=$EPOCHS"
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
echo

echo "=== Hugging Face cache/offline config ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Stage0/train.py imports Stage1/tools/deepspeed_config.py for the full path.
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
echo "BASE_LLM=$BASE_LLM"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo

echo "=== Load secrets (.tokens) ==="
if [ -f "$PROJECT_ROOT/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.tokens"
  echo "Loaded .tokens file"
else
  echo "WARNING: .tokens file not found; W&B will run offline without an API key"
fi
echo

if [ ! -d "$BASE_LLM" ]; then
  echo "ERROR: base Qwen3-VL snapshot not found: $BASE_LLM"
  exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: English GQA data not found: $DATA_PATH"
  echo "       Run (workstation): python Stage3/load_base_data.py --n_samples 30000 --seed 42 --output Stage3/data/english.jsonl"
  exit 1
fi
if [ ! -d "$GQA_IMAGES_DIR" ]; then
  echo "ERROR: GQA images dir not found: $GQA_IMAGES_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

COMMON_ARGS=(
  --data-path   "$DATA_PATH"
  --images-dir  "$GQA_IMAGES_DIR"
  --output-dir  "$OUTPUT_DIR"
  --llm-path    "$BASE_LLM"
  --local-files-only
  --epochs      "$EPOCHS"
  --max-seq-len 256
  --visual-pixels 65536
  --lr 1e-5
  --use-wandb
  --wandb-mode    offline
  --wandb-project m2-align
  --wandb-run-name "stage0-english-gqa-sft-$MODE"
)

echo "=== Start Stage 0 SFT ($MODE) ==="
if [ "$MODE" = "full" ]; then
  deepspeed --master_port "${MASTER_PORT:-50040}" Stage0/train.py --deepspeed --full-finetune \
    "${COMMON_ARGS[@]}" \
    --train-batch-size 1 \
    --global-batch-size 64
else
  python -u Stage0/train.py \
    "${COMMON_ARGS[@]}" \
    --train-batch-size 4 \
    --grad-accum 8 \
    --lora-r 32 --lora-alpha 64
fi

echo "=== Done ==="
date
