#!/bin/bash
#SBATCH --job-name=stage3b_evaluate_pooled_replay
# REPLAY ARM copy of evaluate_pooled.sh: identical except the stage3b checkpoint is
# Stage3/outputs/pooled_replay (train_pooled_replay.sh) and results go to
# Stage3/results/pooled_replay_<stage>/. CHECKPOINT_STAGE=stage2 still reads the
# shared pooled Stage 2 checkpoint (replay does not change Stage 2).
# Also covers the text benchmarks (BENCHMARK=mgsm|msvamp, no image), exactly as
# evaluate_pooled.sh does for the no-replay checkpoint.
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage3/logs/stage3b_evaluate_pooled_replay_%x_%j.log

# Evaluation for the POOLED pipeline. Unlike evaluate.sh, LANG selects ONLY
# the eval-data language -- the mapping checkpoint is the single shared
# outputs/pooled one for every language, and the LLM is vanilla (frozen)
# Qwen3-VL-8B-Instruct, matching what the pooled Stage 1/2/3b runs used.
#
# CHECKPOINT_STAGE = stage2 (pooled, before VQA aug) or stage3b (pooled, after).
#
# Usage:
#   BENCHMARK=xgqa LANG=bn CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_pooled_replay.sh
#   BENCHMARK=cvqa LANG=jv CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_pooled_replay.sh
#   BENCHMARK=mgsm LANG=bn sbatch Stage3/job-scripts/evaluate_pooled_replay.sh
#   (mgsm/msvamp langs: bn de ru zh en es fr ja sw th, + te for mgsm)
#   (repeat over LANG in bn de ru zh pt id ko  for xgqa;
#              in bn ru zh pt id ko jv mn si ga for cvqa)
#
# Optional: MAX_EXAMPLES=50 ... , RESULTS_JSONL=1 ...

set -euo pipefail

BENCHMARK="${BENCHMARK:-xgqa}"
LANG="${LANG:-bn}"
CHECKPOINT_STAGE="${CHECKPOINT_STAGE:-stage3b}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
RESULTS_JSONL="${RESULTS_JSONL:-}"
case "${BENCHMARK:-}" in mgsm|msvamp) RESULTS_JSONL=1 ;; esac  # text: always keep per-item outputs

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
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
  cvqa_generation|cvqa_mcq) EVAL_DATA="$STAGE3/data/cvqa/$LANG.jsonl" ;;
  *)                   EVAL_DATA="$STAGE3/data/$BENCHMARK/$LANG.jsonl" ;;
esac

if [ "$CHECKPOINT_STAGE" = "stage2" ]; then
  MAPPING_CKPT="$STAGE2/outputs/pooled/pytorch_model.bin"
else
  MAPPING_CKPT="$STAGE3/outputs/pooled_replay/pytorch_model.bin"
fi

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "BENCHMARK=$BENCHMARK  LANG=$LANG  CHECKPOINT_STAGE=$CHECKPOINT_STAGE (pooled)"
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
  echo "ERROR: LLM snapshot path not found: $LLM_PATH"
  exit 1
fi
if [ ! -d "$MT_PATH" ]; then
  echo "ERROR: NLLB snapshot not found: $MT_PATH"
  exit 1
fi
if [ ! -f "$MAPPING_CKPT" ]; then
  echo "ERROR: pooled $CHECKPOINT_STAGE checkpoint not found: $MAPPING_CKPT"
  exit 1
fi
if [ ! -f "$EVAL_DATA" ]; then
  echo "ERROR: Eval data not found: $EVAL_DATA"
  case "$BENCHMARK" in
    mgsm|msvamp) echo "       Run (login node): python Stage3/load_text_evaluation.py --output_dir $STAGE3/data" ;;
    *)           echo "       Run: python Stage3/load_evaluation_data.py --benchmark $BENCHMARK --languages $LANG" ;;
  esac
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
  RESULTS_PATH="$STAGE3/results/pooled_replay_$CHECKPOINT_STAGE/$BENCHMARK${MAX_EXAMPLES:+_max$MAX_EXAMPLES}/$LANG.jsonl"
  EXTRA_ARGS="$EXTRA_ARGS --results-jsonl $RESULTS_PATH"
  echo "RESULTS_JSONL=$RESULTS_PATH"
fi

cd "$PROJECT_ROOT"
echo "=== Start pooled $CHECKPOINT_STAGE evaluation: $BENCHMARK/$LANG ==="
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
