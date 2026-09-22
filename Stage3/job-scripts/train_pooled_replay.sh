#!/bin/bash
#SBATCH --job-name=stage3b_train_pooled_replay
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage3/logs/stage3b_train_pooled_replay_%x_%j.log

# REPLAY ARM -- copy of train_pooled.sh that runs Stage3/train_replay.py:
# the same pooled VQA data, warm start, hyperparameters and epochs, plus
# Santiago's text replay (Approach2 DESIGN.md D6/D13): one text-only batch
# every REPLAY_EVERY VQA batches, drawn from Stage3/data/replay/
# (built by Stage3/job-scripts/load_replay_data.sh):
#   translation_<L>.jsonl  5k Stage 1 rows x 11 langs          =  55k
#   math_<L>.jsonl         5k MetaMathQA GSM_* x 11 langs      =  55k
#   math_en.jsonl          5k MetaMathQA untranslated (INCLUDE_ENGLISH=1)
# Writes to its OWN dirs, so it never touches the no-replay run:
#   data  Stage3/data/pooled_replay/   ckpt  Stage3/outputs/pooled_replay/
# Evaluate with evaluate_pooled_replay.sh (xgqa/cvqa*/mgsm/msvamp).
#
# Runtime: the no-replay run took 14.5 h (4.8 h/epoch, 85.5k batches).
# Replay adds 85.5k/3 = 28.5k longer text batches per epoch (~114k rows,
# i.e. about one pass over the ~115k pool per epoch) -> budget 48 h.
#
# Smoke test (a few steps, separate output dir):
#   SMOKE=1 sbatch --time=1:00:00 Stage3/job-scripts/train_pooled_replay.sh
#
# ---- original train_pooled.sh header ----
# Pooled Stage 3b VQA-augmentation training -- one shared checkpoint over
# all 11 languages, replacing the per-language train.sh runs. Stage3/train.py
# already globs *.jsonl in --data-dir and shuffles the union, so this
# launcher just assembles a pooled/ directory and points --data-dir at it.
#
#   Warm mapping = Stage2/outputs/pooled/pytorch_model.bin
#   LLM          = vanilla Qwen3-VL-8B-Instruct (frozen; only the mapping layer trains)
#   Reads  Stage3/data/{bn,ru,de,zh,pt,id,ko,jv,mn,si,ga}.jsonl (30k each)
#   Saves  $SCRATCH/M2-ALIGN/Stage3/outputs/pooled/pytorch_model.bin
#
# ENGLISH: ON by default. Verified against CONE-MT/MindMerger's
# mindmerger_tools/read_datasets.py -- the augmentation stage (read_math_train
# / read_x_csqa_train / read_xnli_train) DOES include English (x-csqa's list
# ends with 'English'; math.json is 10 langs incl. English -> 30k x 10 = the
# paper's 300k). The mapping stage does NOT (Stage 1 pooled stays
# English-free). English rows come from Stage3/data/en.jsonl -- the
# untranslated GQA sample (also the source load_translated_data.py
# translates), in the per-language schema (nllb_lang_tag=eng_Latn);
# pool = 30k x 12.
# Set INCLUDE_ENGLISH=0 for the no-English ablation.
#
# 1 GPU is enough: measured per-language run = 30k x 3 epochs = ~80 min on
# one A100-40. Pool ~12x -> ~16 h for 3 epochs. MindMerger augmentation
# stage = 3 epochs -- matched. Override with EPOCHS=.
#
# Usage:
#   sbatch Stage3/job-scripts/train_pooled.sh
#   EPOCHS=5 sbatch Stage3/job-scripts/train_pooled.sh
#   INCLUDE_ENGLISH=0 sbatch Stage3/job-scripts/train_pooled.sh

set -euo pipefail

LANG_CODES="bn ru de zh pt id ko jv mn si ga"
EPOCHS="${EPOCHS:-3}"
INCLUDE_ENGLISH="${INCLUDE_ENGLISH:-1}"
REPLAY_EVERY="${REPLAY_EVERY:-3}"
# Replay only in languages MGSM/MSVAMP can measure: the overlap of their
# languages with our Stage 1-3 languages. The other 7 have no text-reasoning
# eval, so replay there would cost VQA steps without a readout.
REPLAY_LANGS="${REPLAY_LANGS:-bn de ru zh}"
SMOKE="${SMOKE:-0}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
DATA_ROOT="$SCRATCH/M2-ALIGN"
STAGE2="$DATA_ROOT/Stage2"
STAGE3="$DATA_ROOT/Stage3"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
GQA_IMAGES_DIR="$STAGE3/data/gqa/images"
STAGE2_MAPPING_CKPT="$STAGE2/outputs/pooled/pytorch_model.bin"
POOLED_DATA_DIR="$STAGE3/data/pooled_replay"
OUTPUT_DIR="$STAGE3/outputs/pooled_replay"
REPLAY_DIR="$STAGE3/data/replay"
MAX_TRAIN_STEPS=0
if [ "$SMOKE" = 1 ]; then
  OUTPUT_DIR="$STAGE3/outputs/pooled_replay_smoke"
  POOLED_DATA_DIR="$STAGE3/data/pooled_replay_smoke"
  MAX_TRAIN_STEPS=24
  EPOCHS=1
fi

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  LANGS=$LANG_CODES  EPOCHS=$EPOCHS"
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

echo "=== Hugging Face cache/offline config ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "LLM_PATH=$LLM_PATH"
echo "MT_PATH=$MT_PATH"
echo "GQA_IMAGES_DIR=$GQA_IMAGES_DIR"
echo

echo "=== Load secrets (.tokens) ==="
if [ -f "$PROJECT_ROOT/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.tokens"
  echo "Loaded .tokens file"
else
  echo "WARNING: .tokens file not found; W&B may run in offline mode without API key"
fi
echo

if [ ! -d "$LLM_PATH" ]; then
  echo "ERROR: LLM snapshot path not found: $LLM_PATH"
  exit 1
fi
if [ ! -d "$MT_PATH" ]; then
  echo "ERROR: MT snapshot path not found: $MT_PATH"
  exit 1
fi
if [ ! -d "$GQA_IMAGES_DIR" ]; then
  echo "ERROR: GQA images directory not found: $GQA_IMAGES_DIR"
  exit 1
fi
if [ ! -f "$STAGE2_MAPPING_CKPT" ]; then
  echo "ERROR: pooled Stage 2 checkpoint not found: $STAGE2_MAPPING_CKPT"
  echo "       Run: sbatch Stage2/job-scripts/train_pooled.sh"
  exit 1
fi

# Assemble the pooled data dir as symlinks so train.py's *.jsonl glob picks
# up exactly the per-language files and nothing else.
rm -rf "$POOLED_DATA_DIR"
mkdir -p "$POOLED_DATA_DIR"
for code in $LANG_CODES; do
  src="$STAGE3/data/$code.jsonl"
  if [ ! -f "$src" ]; then
    echo "ERROR: missing $src -- run: LANG=$code sbatch Stage3/job-scripts/load_translated_data.sh"
    exit 1
  fi
  ln -sf "$src" "$POOLED_DATA_DIR/$code.jsonl"
done

if [ "$INCLUDE_ENGLISH" = 1 ]; then
  # Untranslated English GQA rows, already in the per-language schema
  # (query/answer/vg_image_id/nllb_lang_tag=eng_Latn).
  ENGLISH_SRC="$STAGE3/data/en.jsonl"
  if [ ! -f "$ENGLISH_SRC" ]; then
    echo "ERROR: INCLUDE_ENGLISH=1 but missing $ENGLISH_SRC."
    exit 1
  fi
  ln -sf "$ENGLISH_SRC" "$POOLED_DATA_DIR/en.jsonl"
  echo "INCLUDE_ENGLISH=1 -- English rows in the pool (MindMerger-faithful)."
else
  echo "INCLUDE_ENGLISH=0 -- no-English ablation (NOT MindMerger-faithful)."
fi

# Replay files: translation + math per language, plus English math if on.
REPLAY_FILES=()
for code in $REPLAY_LANGS; do
  REPLAY_FILES+=("$REPLAY_DIR/translation_$code.jsonl" "$REPLAY_DIR/math_$code.jsonl")
done
[ "$INCLUDE_ENGLISH" = 1 ] && REPLAY_FILES+=("$REPLAY_DIR/math_en.jsonl")
for f in "${REPLAY_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing replay file $f -- run: sbatch Stage3/job-scripts/load_replay_data.sh"
    exit 1
  fi
done
REPLAY_DATA="$(IFS=,; echo "${REPLAY_FILES[*]}")"
echo "Replay files (${#REPLAY_FILES[@]}), every $REPLAY_EVERY VQA batches:"
wc -l "${REPLAY_FILES[@]}"

echo "Pooled data dir: $POOLED_DATA_DIR"
ls -l "$POOLED_DATA_DIR"
echo

echo "=== Start pooled Stage 3b VQA training + text replay (SMOKE=$SMOKE) ==="
cd "$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u Stage3/train_replay.py \
  --data-dir    "$POOLED_DATA_DIR" \
  --images-dir  "$GQA_IMAGES_DIR" \
  --output-dir  "$OUTPUT_DIR" \
  --init-mapping-ckpt "$STAGE2_MAPPING_CKPT" \
  --mt-path     "$MT_PATH" \
  --llm-path    "$LLM_PATH" \
  --epochs      "$EPOCHS" \
  --lr          2e-5 \
  --train-batch-size 4 \
  --eval-batch-size  4 \
  --grad-accum  8 \
  --max-mt-seq-len 256 \
  --max-seq-len 256 \
  --max-gen-len 16 \
  --replay-data "$REPLAY_DATA" \
  --replay-every "$REPLAY_EVERY" \
  --replay-max-gen-len 512 \
  --replay-max-query-len 1024 \
  --gradient-checkpointing \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --use-wandb \
  --wandb-mode    offline \
  --wandb-project m2-align \
  --wandb-run-name "stage3b-vqa-pooled-replay" \
  --local-files-only

echo "=== Done ==="
date
