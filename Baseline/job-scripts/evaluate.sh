#!/bin/bash
#SBATCH --job-name=baseline_evaluate
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Baseline/logs/baseline_evaluate_%x_%j.log

# Raw Qwen3-VL VQA baseline -- no NLLB encoder, no Mapping layer. This is
# the "does the base model already do this" reference point, parallel to
# Baseline/evaluate_text.py for text reasoning. BENCHMARK selects xgqa,
# cvqa, worldcuisines_task1, or worldcuisines_task2 -- see
# Baseline/evaluate.py's module docstring for per-language coverage of
# each (same as Stage3/load_evaluation_data.py's).
#
# Usage:
#   sbatch --export=BENCHMARK=xgqa,LANG=bn                       evaluate.sh
#   sbatch --export=BENCHMARK=worldcuisines_task1,LANG=jv        evaluate.sh
#   sbatch --export=BENCHMARK=worldcuisines_task2,LANG=jv        evaluate.sh
#
# Optional: sbatch --export=BENCHMARK=xgqa,LANG=bn,MAX_EXAMPLES=50 evaluate.sh
#
# Optional: sbatch --export=BENCHMARK=xgqa,LANG=bn,RESULTS_JSONL=1 evaluate.sh
#   writes one JSON line per example (row + pred/correct) to
#   $DATA_ROOT/Baseline/results/$BENCHMARK/$LANG.jsonl -- pair with a
#   Stage3 run's --results-jsonl for Stage3/analysis/qualitative_cases.py.

set -euo pipefail

BENCHMARK="${BENCHMARK:-xgqa}"
LANG="${LANG:-bn}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
RESULTS_JSONL="${RESULTS_JSONL:-}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"

# Data/outputs/logs live on $SCRATCH; the git checkout under $HOME only holds code.
DATA_ROOT="$SCRATCH/M2-ALIGN"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
GQA_IMAGES_DIR="$DATA_ROOT/Stage3/data/gqa/images"
WORLDCUISINES_IMAGES_DIR="$DATA_ROOT/Stage3/data/worldcuisines/images"
case "$BENCHMARK" in
  worldcuisines_task1) EVAL_DATA="$DATA_ROOT/Stage3/data/worldcuisines/task1/$LANG.jsonl" ;;
  worldcuisines_task2) EVAL_DATA="$DATA_ROOT/Stage3/data/worldcuisines/task2/$LANG.jsonl" ;;
  *)                   EVAL_DATA="$DATA_ROOT/Stage3/data/$BENCHMARK/$LANG.jsonl" ;;
esac

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "BENCHMARK=$BENCHMARK  LANG=$LANG"
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
  echo "ERROR: LLM snapshot not found: $LLM_PATH"
  exit 1
fi
if [ ! -f "$EVAL_DATA" ]; then
  echo "ERROR: Eval data not found: $EVAL_DATA"
  echo "       Run: python Stage3/load_evaluation_data.py --benchmark $BENCHMARK --languages $LANG"
  exit 1
fi
if [ "$BENCHMARK" = "xgqa" ] && [ ! -d "$GQA_IMAGES_DIR" ]; then
  echo "ERROR: GQA images directory not found: $GQA_IMAGES_DIR"
  echo "       Download+extract https://nlp.stanford.edu/data/gqa/images.zip there."
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

# Build optional args
EXTRA_ARGS="--local-files-only"
if [ -n "$MAX_EXAMPLES" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --max-examples $MAX_EXAMPLES"
fi
if [ "$BENCHMARK" = "xgqa" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --images-dir $GQA_IMAGES_DIR"
fi
if [ -n "$RESULTS_JSONL" ]; then
  RESULTS_PATH="$DATA_ROOT/Baseline/results/$BENCHMARK/$LANG.jsonl"
  EXTRA_ARGS="$EXTRA_ARGS --results-jsonl $RESULTS_PATH"
  echo "RESULTS_JSONL=$RESULTS_PATH"
fi

echo "=== Start baseline VQA evaluation: $BENCHMARK/$LANG ==="
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
