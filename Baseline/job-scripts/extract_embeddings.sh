#!/bin/bash
#SBATCH --job-name=baseline_extract_embeddings
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Baseline/logs/baseline_extract_embeddings_%x_%j.log

# Extracts raw-Qwen3-VL layer-wise embeddings (no mapping layer) for
# Stage3/analysis/tsne_plot.py and layerwise_retrieval.py -- the "before"
# reference curve, parallel to Baseline/evaluate.py's role for accuracy.
# See Stage3/analysis/extract_embeddings_baseline.py's module docstring.
#
# Usage:
#   BENCHMARK=xgqa sbatch extract_embeddings.sh
#
# LANGUAGES defaults to xGQA's 7 non-English active languages.
# Optional: MAX_EXAMPLES_PER_LANG=150 sbatch extract_embeddings.sh

set -euo pipefail

BENCHMARK="${BENCHMARK:-xgqa}"
LANGUAGES="${LANGUAGES:-bn,de,ru,zh,pt,id,ko}"
MAX_EXAMPLES_PER_LANG="${MAX_EXAMPLES_PER_LANG:-300}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"

DATA_ROOT="$SCRATCH/M2-ALIGN"
STAGE3="$DATA_ROOT/Stage3"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
GQA_IMAGES_DIR="$STAGE3/data/gqa/images"
WORLDCUISINES_IMAGES_DIR="$STAGE3/data/worldcuisines/images"

case "$BENCHMARK" in
  worldcuisines_task1) EVAL_DATA_DIR="$STAGE3/data/worldcuisines/task1" ;;
  worldcuisines_task2) EVAL_DATA_DIR="$STAGE3/data/worldcuisines/task2" ;;
  *)                   EVAL_DATA_DIR="$STAGE3/data/$BENCHMARK" ;;
esac

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "BENCHMARK=$BENCHMARK  LANGUAGES=$LANGUAGES"
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
echo "EVAL_DATA_DIR=$EVAL_DATA_DIR"
echo

if [ ! -d "$LLM_PATH" ]; then
  echo "ERROR: LLM snapshot not found: $LLM_PATH"
  exit 1
fi
if [ ! -d "$EVAL_DATA_DIR" ]; then
  echo "ERROR: Eval data dir not found: $EVAL_DATA_DIR"
  echo "       Run: python Stage3/load_evaluation_data.py --benchmark $BENCHMARK --languages $LANGUAGES"
  exit 1
fi

EXTRA_ARGS="--local-files-only"
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

OUT_DIR="$STAGE3/analysis_out"
mkdir -p "$OUT_DIR"
OUT_PATH="$OUT_DIR/layers_baseline_${BENCHMARK}.pt"

echo "=== Start baseline embedding extraction: $BENCHMARK ==="
cd "$PROJECT_ROOT"
# shellcheck disable=SC2086
python -u Stage3/analysis/extract_embeddings_baseline.py \
  --benchmark             "$BENCHMARK" \
  --languages             "$LANGUAGES" \
  --eval-data-dir         "$EVAL_DATA_DIR" \
  --model-id              "$LLM_PATH" \
  --max-examples-per-lang "$MAX_EXAMPLES_PER_LANG" \
  --out                   "$OUT_PATH" \
  $EXTRA_ARGS

echo "Wrote $OUT_PATH"
echo "=== Done ==="
date
