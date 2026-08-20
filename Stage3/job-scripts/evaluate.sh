#!/bin/bash
#SBATCH --job-name=stage3b_evaluate
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage3/logs/stage3b_evaluate_%x_%j.log

# Per-language evaluation -- LANG selects both the eval-data language AND
# the checkpoint (every language, including Bengali, has its own
# Stage2/Stage3b checkpoint at outputs/$LANG). BENCHMARK selects which
# eval set: xgqa, cvqa, worldcuisines_task1, or worldcuisines_task2
# (see load_evaluation_data.py's module docstring for per-language
# coverage of each).
#
# Usage:
#   BENCHMARK=xgqa                LANG=ru CHECKPOINT_STAGE=stage2  sbatch evaluate.sh
#   BENCHMARK=xgqa                LANG=ru CHECKPOINT_STAGE=stage3b sbatch evaluate.sh
#   BENCHMARK=worldcuisines_task1 LANG=jv CHECKPOINT_STAGE=stage3b sbatch evaluate.sh
#   BENCHMARK=worldcuisines_task2 LANG=jv CHECKPOINT_STAGE=stage3b sbatch evaluate.sh
#   (repeat with LANG=de, LANG=zh, LANG=bn, ...)
#
# Optional: MAX_EXAMPLES=50 sbatch evaluate.sh

set -euo pipefail

BENCHMARK="${BENCHMARK:-xgqa}"
LANG="${LANG:-ru}"
CHECKPOINT_STAGE="${CHECKPOINT_STAGE:-stage3b}"   # stage2 (before) or stage3b (after)
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"

# Data/outputs/logs live on $SCRATCH; the git checkout under $HOME only holds code.
DATA_ROOT="$SCRATCH/M2-ALIGN"
STAGE2="$DATA_ROOT/Stage2"
STAGE3="$DATA_ROOT/Stage3"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
GQA_IMAGES_DIR="$STAGE3/data/gqa/images"
WORLDCUISINES_IMAGES_DIR="$STAGE3/data/worldcuisines/images"
case "$BENCHMARK" in
  worldcuisines_task1) EVAL_DATA="$STAGE3/data/worldcuisines/task1/$LANG.jsonl" ;;
  worldcuisines_task2) EVAL_DATA="$STAGE3/data/worldcuisines/task2/$LANG.jsonl" ;;
  *)                   EVAL_DATA="$STAGE3/data/$BENCHMARK/$LANG.jsonl" ;;
esac

if [ "$CHECKPOINT_STAGE" = "stage2" ]; then
  MAPPING_CKPT="$STAGE2/outputs/$LANG/pytorch_model.bin"
else
  MAPPING_CKPT="$STAGE3/outputs/$LANG/pytorch_model.bin"
fi

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
echo "BENCHMARK=$BENCHMARK  LANG=$LANG  CHECKPOINT_STAGE=$CHECKPOINT_STAGE"
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
echo "EVAL_DATA=$EVAL_DATA"
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
if [ ! -f "$EVAL_DATA" ]; then
  echo "ERROR: Eval data not found: $EVAL_DATA"
  echo "       Run: python Stage3/load_evaluation_data.py --benchmark $BENCHMARK --languages $LANG"
  exit 1
fi
if [ "$BENCHMARK" = "xgqa" ] && [ ! -d "$GQA_IMAGES_DIR" ]; then
  echo "ERROR: GQA images directory not found: $GQA_IMAGES_DIR"
  exit 1
fi
case "$BENCHMARK" in
  worldcuisines_task*)
    if [ ! -d "$WORLDCUISINES_IMAGES_DIR" ]; then
      echo "ERROR: WorldCuisines image cache not found: $WORLDCUISINES_IMAGES_DIR"
      echo "       This node has no internet access -- pre-fetch on the workstation node first:"
      echo "       python Stage3/load_evaluation_data.py --benchmark $BENCHMARK --languages $LANG"
      exit 1
    fi
    ;;
esac

EXTRA_ARGS="--local-files-only"
if [ -n "$MAX_EXAMPLES" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --max-examples $MAX_EXAMPLES"
fi
if [ "$BENCHMARK" = "xgqa" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --images-dir $GQA_IMAGES_DIR"
fi

echo "=== Start $CHECKPOINT_STAGE VQA evaluation: $BENCHMARK/$LANG ==="
cd "$PROJECT_ROOT"
# shellcheck disable=SC2086
python -u Stage3/evaluate.py \
  --benchmark    "$BENCHMARK" \
  --lang         "$LANG" \
  --eval-data    "$EVAL_DATA" \
  --llm-path     "$LLM_PATH" \
  --mt-path      "$MT_PATH" \
  --mapping-ckpt "$MAPPING_CKPT" \
  $EXTRA_ARGS

echo "=== Done ==="
date
