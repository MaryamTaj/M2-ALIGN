#!/bin/bash
#SBATCH --job-name=stage1_evaluate_text
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage1/logs/stage1_evaluate_text_%x_%j.log

# Usage:
# sbatch --export=TASK=mgsm   evaluate_text.sh
# sbatch --export=TASK=msvamp evaluate_text.sh
# sbatch --export=TASK=xnli   evaluate_text.sh
# sbatch --export=TASK=x-csqa evaluate_text.sh
#
# Optional overrides:
#   sbatch --export=TASK=mgsm,LANGS="sw en zh"       evaluate_text.sh
#   sbatch --export=TASK=xnli,LANGS="sw en"          evaluate_text.sh
#   sbatch --export=TASK=mgsm,MAX_EXAMPLES=50         evaluate_text.sh

set -euo pipefail

TASK="${TASK:-mgsm}"
LANGS="${LANGS:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE1="$PROJECT_ROOT/Stage1"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
MAPPING_CKPT="$STAGE1/outputs/MindMerger/nllb_corpus/mapping/pytorch_model.bin"

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
echo "MT_PATH=$MT_PATH"
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
  ALT_MAPPING_CKPT="$STAGE1/outputs/MindMerger/translation/mapping/pytorch_model.bin"
  if [ -f "$ALT_MAPPING_CKPT" ]; then
    MAPPING_CKPT="$ALT_MAPPING_CKPT"
    echo "Using fallback mapping checkpoint: $MAPPING_CKPT"
  else
    echo "ERROR: Mapping checkpoint not found: $MAPPING_CKPT"
    echo "Also checked fallback: $ALT_MAPPING_CKPT"
    exit 1
  fi
fi

# Build optional args
EXTRA_ARGS="--local-files-only"
if [ -n "$LANGS" ]; then
  # shellcheck disable=SC2086
  EXTRA_ARGS="$EXTRA_ARGS --langs $LANGS"
fi
if [ -n "$MAX_EXAMPLES" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --max-examples $MAX_EXAMPLES"
fi

echo "=== Start Stage 1 text evaluation: $TASK ==="
echo "LANGS=${LANGS:-<task default>}"
cd "$PROJECT_ROOT"
# shellcheck disable=SC2086
python -u Stage1/evaluate_text.py \
  --task "$TASK" \
  --llm-path "$LLM_PATH" \
  --mt-path "$MT_PATH" \
  --mapping-ckpt "$MAPPING_CKPT" \
  --max-mt-seq-len 256 \
  --max-gen-len 512 \
  $EXTRA_ARGS

echo "=== Done ==="
date
