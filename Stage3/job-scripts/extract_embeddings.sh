#!/bin/bash
#SBATCH --job-name=stage3_extract_embeddings
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage3/logs/stage3_extract_embeddings_%x_%j.log

# Extracts cross-lingual embeddings from one mapping checkpoint for
# Stage3/analysis/tsne_plot.py and layerwise_retrieval.py -- see
# Stage3/analysis/extract_embeddings.py's module docstring for what each
# --pooling mode computes and why. CHECKPOINT_STAGE picks the checkpoint
# (stage1/stage2/stage3b); run this 3x (once per stage) to compare them on
# the same t-SNE/retrieval plot.
#
# Usage:
#   BENCHMARK=xgqa CHECKPOINT_STAGE=stage1  POOLING=xm         sbatch extract_embeddings.sh
#   BENCHMARK=xgqa CHECKPOINT_STAGE=stage2  POOLING=xm         sbatch extract_embeddings.sh
#   BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b POOLING=xm         sbatch extract_embeddings.sh
#   BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b POOLING=llm-layers sbatch extract_embeddings.sh
#
# LANGUAGES defaults to xGQA's 7 non-English active languages. CKPT_LANG
# picks *whose* per-language checkpoint to probe (the mapping is trained
# per-language, see README) -- defaults to the first of LANGUAGES.
#
# Optional: MAX_EXAMPLES_PER_LANG=150 sbatch extract_embeddings.sh

set -euo pipefail

BENCHMARK="${BENCHMARK:-xgqa}"
CHECKPOINT_STAGE="${CHECKPOINT_STAGE:-stage3b}"   # stage1, stage2, or stage3b
POOLING="${POOLING:-xm}"                          # xm or llm-layers
LANGUAGES="${LANGUAGES:-bn,de,ru,zh,pt,id,ko}"
CKPT_LANG="${CKPT_LANG:-${LANGUAGES%%,*}}"
MAX_EXAMPLES_PER_LANG="${MAX_EXAMPLES_PER_LANG:-300}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"

DATA_ROOT="$SCRATCH/M2-ALIGN"
STAGE1="$DATA_ROOT/Stage1"
STAGE2="$DATA_ROOT/Stage2"
STAGE3="$DATA_ROOT/Stage3"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
GQA_IMAGES_DIR="$STAGE3/data/gqa/images"
WORLDCUISINES_IMAGES_DIR="$STAGE3/data/worldcuisines/images"

case "$BENCHMARK" in
  worldcuisines_task1) EVAL_DATA_DIR="$STAGE3/data/worldcuisines/task1" ;;
  worldcuisines_task2) EVAL_DATA_DIR="$STAGE3/data/worldcuisines/task2" ;;
  *)                   EVAL_DATA_DIR="$STAGE3/data/$BENCHMARK" ;;
esac

case "$CHECKPOINT_STAGE" in
  stage1) MAPPING_CKPT="$STAGE1/outputs/$CKPT_LANG/pytorch_model.bin" ;;
  stage2) MAPPING_CKPT="$STAGE2/outputs/$CKPT_LANG/pytorch_model.bin" ;;
  *)      MAPPING_CKPT="$STAGE3/outputs/$CKPT_LANG/pytorch_model.bin" ;;
esac

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
echo "BENCHMARK=$BENCHMARK  CHECKPOINT_STAGE=$CHECKPOINT_STAGE  POOLING=$POOLING  LANGUAGES=$LANGUAGES  CKPT_LANG=$CKPT_LANG"
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
  exit 1
fi
if [ ! -d "$EVAL_DATA_DIR" ]; then
  echo "ERROR: Eval data dir not found: $EVAL_DATA_DIR"
  echo "       Run: python Stage3/load_evaluation_data.py --benchmark $BENCHMARK --languages $LANGUAGES"
  exit 1
fi

EXTRA_ARGS="--local-files-only"
if [ "$POOLING" = "llm-layers" ]; then
  if [ "$BENCHMARK" = "xgqa" ]; then
    if [ ! -d "$GQA_IMAGES_DIR" ]; then
      echo "ERROR: GQA images directory not found: $GQA_IMAGES_DIR"
      exit 1
    fi
    EXTRA_ARGS="$EXTRA_ARGS --images-dir $GQA_IMAGES_DIR"
  else
    if [ ! -d "$WORLDCUISINES_IMAGES_DIR" ]; then
      echo "ERROR: WorldCuisines image cache not found: $WORLDCUISINES_IMAGES_DIR"
      echo "       This node has no internet access -- pre-fetch on the workstation node first:"
      echo "       python Stage3/load_evaluation_data.py --benchmark $BENCHMARK --languages $LANGUAGES"
      exit 1
    fi
    EXTRA_ARGS="$EXTRA_ARGS --image-cache-dir $WORLDCUISINES_IMAGES_DIR"
  fi
fi

OUT_DIR="$STAGE3/analysis_out"
mkdir -p "$OUT_DIR"
OUT_PATH="$OUT_DIR/${POOLING}_${CHECKPOINT_STAGE}_${BENCHMARK}.pt"

echo "=== Start embedding extraction: $BENCHMARK / $CHECKPOINT_STAGE / $POOLING ==="
cd "$PROJECT_ROOT"
# shellcheck disable=SC2086
python -u Stage3/analysis/extract_embeddings.py \
  --benchmark             "$BENCHMARK" \
  --languages             "$LANGUAGES" \
  --eval-data-dir         "$EVAL_DATA_DIR" \
  --mapping-ckpt          "$MAPPING_CKPT" \
  --llm-path              "$LLM_PATH" \
  --mt-path               "$MT_PATH" \
  --pooling               "$POOLING" \
  --max-examples-per-lang "$MAX_EXAMPLES_PER_LANG" \
  --out                   "$OUT_PATH" \
  $EXTRA_ARGS

echo "Wrote $OUT_PATH"
echo "=== Done ==="
date
