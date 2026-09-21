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

# Raw Qwen3-VL baseline -- no NLLB encoder, no Mapping layer. This is the
# "does the base model already do this" reference point, for BOTH visual QA
# and text reasoning -- one job script, one Python entry point
# (Baseline/evaluate.py), one LANG per job:
#   xgqa, cvqa, cvqa_generation, cvqa_mcq, worldcuisines_task1/2  (image + question)
#   mgsm, msvamp  (question only, no image; data from
#       $DATA_ROOT/Stage3/data/{mgsm,msvamp}/$LANG.jsonl; per-item outputs
#       always written to Baseline/results/<task>/<lang>.jsonl)
# (Replaces the old evaluate_text.sh / evaluate_text.py.) For the VQA benchmarks, see
# Baseline/evaluate.py's module docstring for per-language coverage of
# each (same as Stage3/load_evaluation_data.py's).
#
# Usage:
#   sbatch --export=BENCHMARK=xgqa,LANG=bn                       evaluate.sh
#   sbatch --export=BENCHMARK=worldcuisines_task1,LANG=jv        evaluate.sh
#   sbatch --export=BENCHMARK=worldcuisines_task2,LANG=jv        evaluate.sh
#   sbatch --export=BENCHMARK=mgsm,LANG=bn                       evaluate.sh
#   sbatch --export=BENCHMARK=msvamp,LANG=de                     evaluate.sh
#     (mgsm langs: bn de ru zh en es fr ja sw th te; msvamp: same minus te)
#
# Optional: sbatch --export=BENCHMARK=xgqa,LANG=bn,MAX_EXAMPLES=50 evaluate.sh
#
# Optional: sbatch --export=BENCHMARK=cvqa_mcq,LANG=bn,VISUAL_PIXELS=65536 evaluate.sh
#   pins min_pixels = max_pixels, matching Stage3/evaluate.py's --visual-pixels.
#
# Optional: sbatch --export=BENCHMARK=xgqa,LANG=bn,TRANSLATE_TEST=1 evaluate.sh
#   translate-test: reads the NLLB-3.3B English translations from
#   Baseline/load_translated_data.py ($DATA_ROOT/Baseline/data/translate_test/)
#   instead of the native-language file (xgqa and cvqa* only). Results go
#   under a "_tt" benchmark suffix so they never overwrite native runs.
#
# Optional: sbatch --export=BENCHMARK=cvqa_mcq,LANG=bn,UPPER_BOUND=1 evaluate.sh
#   English upper bound (cvqa* only): asks CVQA's human English question
#   (`translated_query`, added by Baseline/load_english_upper_bound.py)
#   instead of the native one. Results go under a "_ub" benchmark suffix.
#   xGQA needs no flag -- its upper bound is BENCHMARK=xgqa,LANG=en.
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
case "${BENCHMARK:-}" in mgsm|msvamp) RESULTS_JSONL=1 ;; esac  # text: always keep per-item outputs
VISUAL_PIXELS="${VISUAL_PIXELS:-}"   # empty = processor default (native res); 65536 = Stage3's 256*256 budget
TRANSLATE_TEST="${TRANSLATE_TEST:-}" # 1 = English (NLLB-translated) questions instead of native
UPPER_BOUND="${UPPER_BOUND:-}"       # 1 = CVQA's human English questions (translated_query) instead of native

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"

# Data/outputs/logs live on $SCRATCH; the git checkout under $HOME only holds code.
DATA_ROOT="$SCRATCH/M2-ALIGN"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
GQA_IMAGES_DIR="$DATA_ROOT/Stage3/data/gqa/images"
WORLDCUISINES_IMAGES_DIR="$DATA_ROOT/Stage3/data/worldcuisines/images"
case "$BENCHMARK" in
  worldcuisines_task1) EVAL_DATA="$DATA_ROOT/Stage3/data/worldcuisines/task1/$LANG.jsonl" ;;
  worldcuisines_task2) EVAL_DATA="$DATA_ROOT/Stage3/data/worldcuisines/task2/$LANG.jsonl" ;;
  cvqa_generation|cvqa_mcq) EVAL_DATA="$DATA_ROOT/Stage3/data/cvqa/$LANG.jsonl" ;;
  *)                   EVAL_DATA="$DATA_ROOT/Stage3/data/$BENCHMARK/$LANG.jsonl" ;;
esac
TT_TAG=""
if [ -n "$TRANSLATE_TEST" ]; then
  case "$BENCHMARK" in
    xgqa)   EVAL_DATA="$DATA_ROOT/Baseline/data/translate_test/xgqa/$LANG.jsonl" ;;
    cvqa*)  EVAL_DATA="$DATA_ROOT/Baseline/data/translate_test/cvqa/$LANG.jsonl" ;;
    *) echo "ERROR: TRANSLATE_TEST not supported for BENCHMARK=$BENCHMARK"; exit 1 ;;
  esac
  TT_TAG="_tt"
fi
if [ -n "$UPPER_BOUND" ]; then
  if [ -n "$TRANSLATE_TEST" ]; then
    echo "ERROR: TRANSLATE_TEST and UPPER_BOUND are mutually exclusive"; exit 1
  fi
  case "$BENCHMARK" in
    cvqa*) TT_TAG="_ub" ;;
    xgqa)  echo "ERROR: xGQA upper bound is BENCHMARK=xgqa,LANG=en (no UPPER_BOUND flag)"; exit 1 ;;
    *)     echo "ERROR: UPPER_BOUND not supported for BENCHMARK=$BENCHMARK"; exit 1 ;;
  esac
fi

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "BENCHMARK=$BENCHMARK  LANG=$LANG  VISUAL_PIXELS=${VISUAL_PIXELS:-native}  TRANSLATE_TEST=${TRANSLATE_TEST:-0}  UPPER_BOUND=${UPPER_BOUND:-0}"
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
  if [ -n "$TRANSLATE_TEST" ]; then
    echo "       Run: sbatch --export=LANG=$LANG Baseline/job-scripts/load_translated_data.sh"
  else
    case "$BENCHMARK" in
      mgsm|msvamp) echo "       Run: python Stage3/load_text_evaluation.py --output_dir $DATA_ROOT/Stage3/data" ;;
      *)           echo "       Run: python Stage3/load_evaluation_data.py --benchmark $BENCHMARK --languages $LANG" ;;
    esac
  fi
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
if [ -n "$VISUAL_PIXELS" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --visual-pixels $VISUAL_PIXELS"
fi
if [ -n "$UPPER_BOUND" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --query-field translated_query"
fi
if [ -n "$RESULTS_JSONL" ]; then
  RESULTS_PATH="$DATA_ROOT/Baseline/results/$BENCHMARK$TT_TAG${VISUAL_PIXELS:+_px$VISUAL_PIXELS}/$LANG.jsonl"
  EXTRA_ARGS="$EXTRA_ARGS --results-jsonl $RESULTS_PATH"
  echo "RESULTS_JSONL=$RESULTS_PATH"
fi

cd "$PROJECT_ROOT"
echo "=== Start baseline evaluation: $BENCHMARK/$LANG ==="
# shellcheck disable=SC2086
python -u Baseline/evaluate.py \
  --benchmark "$BENCHMARK" \
  --lang      "$LANG" \
  --eval-data "$EVAL_DATA" \
  --model-id  "$LLM_PATH" \
  $EXTRA_ARGS

echo "=== Done ==="
date
