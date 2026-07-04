#!/bin/bash
#SBATCH --job-name=stage3a_evaluate_text
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage3/logs/stage3a_evaluate_text_%x_%j.log

# Usage — run this TWICE per the Stage 3 plan: once right after Stage 3a
# training (establishes the reference point) and again after Stage 3b (VQA
# augmentation) to check for regression. Point CHECKPOINT_STAGE at whichever
# checkpoint you're evaluating.
#   TASK=mgsm   sbatch evaluate_text.sh
#   TASK=msvamp sbatch evaluate_text.sh
# (xnli/xcsqa dropped for now — see Stage3/load_text.py)
#
# Optional overrides:
#   TASK=mgsm LANGS="sw bn en zh" sbatch evaluate_text.sh
#   TASK=mgsm MAX_EXAMPLES=50     sbatch evaluate_text.sh
#   CHECKPOINT_STAGE=stage3b TASK=mgsm sbatch evaluate_text.sh   # post-VQA regression check

set -euo pipefail

TASK="${TASK:-mgsm}"
LANGS="${LANGS:-sw}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
CHECKPOINT_STAGE="${CHECKPOINT_STAGE:-stage3a}"   # stage3a (reference) or stage3b (post-VQA)

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE3="$PROJECT_ROOT/Stage3"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"

# Stage3a checkpoints are saved by Stage3/job-scripts/train_text.sh; Stage3b
# checkpoints (post-VQA) are saved by Stage3/job-scripts/train_vqa.sh.
if [ "$CHECKPOINT_STAGE" = "stage3b" ]; then
  MAPPING_CKPT="$STAGE3/outputs/stage3b/mapping/pytorch_model.bin"
else
  MAPPING_CKPT="$STAGE3/outputs/stage3a/$TASK/mapping/pytorch_model.bin"
fi

EVAL_TASK="$TASK"

# Resolve nested snapshot directory for MT model.
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
echo "TASK=$TASK  EVAL_TASK=$EVAL_TASK  LANGS=$LANGS"
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
echo "MT_PATH=$MT_PATH"
echo "MAPPING_CKPT=$MAPPING_CKPT"
echo

if [ -f "$PROJECT_ROOT/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.tokens"
  echo "Loaded .tokens file"
else
  echo "WARNING: .tokens file not found"
fi
echo

if [ ! -d "$LLM_PATH" ]; then
  echo "ERROR: LLM snapshot not found: $LLM_PATH"
  exit 1
fi
if [ ! -d "$MT_PATH" ]; then
  echo "ERROR: NLLB snapshot not found: $MT_PATH"
  exit 1
fi
if [ ! -f "$MAPPING_CKPT" ]; then
  echo "ERROR: $CHECKPOINT_STAGE mapping checkpoint not found: $MAPPING_CKPT"
  if [ "$CHECKPOINT_STAGE" = "stage3b" ]; then
    echo "       Run: sbatch Stage3/job-scripts/train_vqa.sh"
  else
    echo "       Run: TASK=$TASK sbatch Stage3/job-scripts/train_text.sh"
  fi
  exit 1
fi

# Build optional args.
EXTRA_ARGS="--local-files-only"
# shellcheck disable=SC2086
EXTRA_ARGS="$EXTRA_ARGS --langs $LANGS"
if [ -n "$MAX_EXAMPLES" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --max-examples $MAX_EXAMPLES"
fi

echo "=== Start $CHECKPOINT_STAGE text evaluation: $EVAL_TASK ==="
cd "$PROJECT_ROOT"
# shellcheck disable=SC2086
python -u Stage3/evaluate_text.py \
  --task         "$EVAL_TASK" \
  --llm-path     "$LLM_PATH" \
  --mt-path      "$MT_PATH" \
  --mapping-ckpt "$MAPPING_CKPT" \
  --max-mt-seq-len  256 \
  --max-llm-seq-len 2048 \
  --max-gen-len     512 \
  $EXTRA_ARGS

echo "=== Done ==="
date
