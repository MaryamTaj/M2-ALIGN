#!/bin/bash
#SBATCH --job-name=stage0_evaluate
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage0/logs/stage0_evaluate_%x_%j.log

# CONDITION 2 -- MonoVQA. Evaluates the Stage 0 English-GQA SFT checkpoint
# directly (no NLLB, no Mapping) on xGQA / CVQA: raw fine-tuned Qwen3-VL,
# native-language question fed straight in -> zero-shot cross-lingual. This
# is the MindMerger "MonoReason" analog and the frozen phi for condition 3.
#
# Identical to Baseline/job-scripts/evaluate.sh except --model-id points at
# Stage0/outputs/qwen3vl-8b-gqa instead of vanilla Qwen3-VL, and results go
# under Stage0/results/.
#
# Usage:
#   sbatch --export=BENCHMARK=xgqa,LANG=bn  Stage0/job-scripts/evaluate.sh
#   sbatch --export=BENCHMARK=cvqa,LANG=jv  Stage0/job-scripts/evaluate.sh
#   for L in bn de ru zh pt id ko;          do sbatch --export=BENCHMARK=xgqa,LANG=$L Stage0/job-scripts/evaluate.sh; done
#   for L in bn ru zh pt id ko jv mn si ga; do sbatch --export=BENCHMARK=cvqa,LANG=$L Stage0/job-scripts/evaluate.sh; done
#
# Optional: MAX_EXAMPLES=50 , RESULTS_JSONL=1

set -euo pipefail

BENCHMARK="${BENCHMARK:-xgqa}"
LANG="${LANG:-bn}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
RESULTS_JSONL="${RESULTS_JSONL:-}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
DATA_ROOT="$SCRATCH/M2-ALIGN"

# The Stage 0 SFT output -- NOT vanilla Qwen3-VL.
LLM_PATH="$DATA_ROOT/Stage0/outputs/qwen3vl-8b-gqa"
GQA_IMAGES_DIR="$DATA_ROOT/Stage3/data/gqa/images"
WORLDCUISINES_IMAGES_DIR="$DATA_ROOT/Stage3/data/worldcuisines/images"
case "$BENCHMARK" in
  worldcuisines_task1) EVAL_DATA="$DATA_ROOT/Stage3/data/worldcuisines/task1/$LANG.jsonl" ;;
  worldcuisines_task2) EVAL_DATA="$DATA_ROOT/Stage3/data/worldcuisines/task2/$LANG.jsonl" ;;
  cvqa_generation)     EVAL_DATA="$DATA_ROOT/Stage3/data/cvqa/$LANG.jsonl" ;;
  *)                   EVAL_DATA="$DATA_ROOT/Stage3/data/$BENCHMARK/$LANG.jsonl" ;;
esac

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "BENCHMARK=$BENCHMARK  LANG=$LANG  (MonoVQA / Stage 0 checkpoint)"
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
echo "EVAL_DATA=$EVAL_DATA"
echo

if [ ! -d "$LLM_PATH" ]; then
  echo "ERROR: Stage 0 MonoVQA checkpoint not found: $LLM_PATH"
  echo "       Run: sbatch Stage0/job-scripts/train.sh"
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
      echo "       Pre-fetch on the workstation node first:"
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
if [ -n "$RESULTS_JSONL" ]; then
  RESULTS_PATH="$DATA_ROOT/Stage0/results/$BENCHMARK/$LANG.jsonl"
  EXTRA_ARGS="$EXTRA_ARGS --results-jsonl $RESULTS_PATH"
  echo "RESULTS_JSONL=$RESULTS_PATH"
fi

echo "=== Start MonoVQA evaluation: $BENCHMARK/$LANG ==="
cd "$PROJECT_ROOT"
# shellcheck disable=SC2086
python -u Baseline/evaluate.py \
  --benchmark "$BENCHMARK" \
  --lang      "$LANG" \
  --eval-data "$EVAL_DATA" \
  --model-id  "$LLM_PATH" \
  $EXTRA_ARGS

echo "=== Done ==="
date
