#!/bin/bash
#SBATCH --job-name=stage3_evaluate_text_lang
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

# Per-language MGSM/MSVAMP evaluation -- LANG selects both the eval-data
# language AND the checkpoint to load (each language has its own Stage 3b
# checkpoint, unlike the shared Bengali evaluate_text.sh which always reads
# Stage3/outputs/stage3b and takes a separate LANGS list of eval languages
# to run through that one checkpoint).
#
# Usage:
#   TASK=mgsm   LANG=ru sbatch evaluate_text_lang.sh
#   TASK=msvamp LANG=ru sbatch evaluate_text_lang.sh
#   (repeat with LANG=de, LANG=zh)

set -euo pipefail

TASK="${TASK:-mgsm}"
LANG="${LANG:-ru}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE3="$PROJECT_ROOT/Stage3"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
EVAL_DATA_DIR="$STAGE3/data/stage3a_eval"
MAPPING_CKPT="$STAGE3/outputs/stage3b_$LANG/mapping/pytorch_model.bin"

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
echo "TASK=$TASK  LANG=$LANG"
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
  echo "ERROR: stage3b mapping checkpoint not found: $MAPPING_CKPT"
  echo "       Run: LANG=$LANG sbatch Stage3/job-scripts/train_vqa_lang.sh"
  exit 1
fi
if [ ! -f "$EVAL_DATA_DIR/$TASK/$LANG.jsonl" ]; then
  echo "ERROR: Eval data not found: $EVAL_DATA_DIR/$TASK/$LANG.jsonl"
  echo "       Run: python Stage3/load_text_evaluation.py --languages $LANG"
  exit 1
fi

EXTRA_ARGS="--local-files-only --langs $LANG"
if [ -n "$MAX_EXAMPLES" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --max-examples $MAX_EXAMPLES"
fi

echo "=== Start stage3b text evaluation: $TASK / $LANG ==="
cd "$PROJECT_ROOT"
# shellcheck disable=SC2086
python -u Stage3/evaluate_text.py \
  --task          "$TASK" \
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
