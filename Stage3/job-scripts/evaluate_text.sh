#!/bin/bash
#SBATCH --job-name=stage3_evaluate_text
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage3/logs/stage3_evaluate_text_%x_%j.log

# Usage — evaluate the Stage 3 VQA checkpoint (stage3b, warm-started
# directly from Stage 2) on the multilingual text benchmarks MGSM/MSVAMP,
# alongside xGQA (see evaluate_vqa.sh), to get the full picture of what
# VQA training did to the model:
#   TASK=mgsm   sbatch evaluate_text.sh
#   TASK=msvamp sbatch evaluate_text.sh
# (xnli/xcsqa dropped for now — see Stage3/load_text_data.py)
#
# Optional overrides:
#   TASK=mgsm LANGS="sw bn en zh" sbatch evaluate_text.sh
#   TASK=mgsm MAX_EXAMPLES=50     sbatch evaluate_text.sh
#   CHECKPOINT_STAGE=stage3a TASK=mgsm sbatch evaluate_text.sh   # legacy: a standalone
#                                                                 # Stage 3a checkpoint,
#                                                                 # if trained separately
#                                                                 # via train_text.sh

set -euo pipefail

TASK="${TASK:-mgsm}"
LANGS="${LANGS:-sw}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
CHECKPOINT_STAGE="${CHECKPOINT_STAGE:-stage3b}"   # stage3b (default, VQA-trained) or stage3a (legacy, text-only)

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE3="$PROJECT_ROOT/Stage3"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
EVAL_DATA_DIR="$STAGE3/data/stage3a_eval"

# Stage3b checkpoints (the default -- VQA-trained, warm-started directly
# from Stage 2) are saved by Stage3/job-scripts/train_vqa.sh. Stage3a
# checkpoints (legacy, text-only) are saved by Stage3/job-scripts/train_text.sh.
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
echo "LLM_PATH=$LLM_PATH"
echo "MT_PATH=$MT_PATH"
echo "MAPPING_CKPT=$MAPPING_CKPT"
echo "EVAL_DATA_DIR=$EVAL_DATA_DIR"
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
for LANG in $LANGS; do
  if [ ! -f "$EVAL_DATA_DIR/$EVAL_TASK/$LANG.jsonl" ]; then
    echo "ERROR: Eval data not found: $EVAL_DATA_DIR/$EVAL_TASK/$LANG.jsonl"
    echo "       Run: python Stage3/load_text_evaluation.py"
    exit 1
  fi
done

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
  --task          "$EVAL_TASK" \
  --llm-path      "$LLM_PATH" \
  --mt-path       "$MT_PATH" \
  --mapping-ckpt  "$MAPPING_CKPT" \
  --eval-data-dir "$EVAL_DATA_DIR" \
  --max-mt-seq-len  256 \
  --max-llm-seq-len 2048 \
  --max-gen-len     512 \
  $EXTRA_ARGS

echo "=== Done ==="
date
