#!/bin/bash
#SBATCH --job-name=stage3b_train_pooled
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage3/logs/stage3b_train_pooled_%x_%j.log

# Pooled Stage 3b VQA-augmentation training -- one shared checkpoint over
# all 11 languages, replacing the per-language train.sh runs. Stage3/train.py
# already globs *.jsonl in --data-dir and shuffles the union, so this
# launcher just assembles a pooled/ directory and points --data-dir at it.
#
#   Warm mapping = Stage2/outputs/pooled/pytorch_model.bin
#   Warm LLM     = the Stage 0 MonoVQA checkpoint (frozen phi)
#   Reads  Stage3/data/{bn,ru,de,zh,pt,id,ko,jv,mn,si,ga}.jsonl (30k each)
#   Saves  $SCRATCH/M2-ALIGN/Stage3/outputs/pooled/pytorch_model.bin
#
# ENGLISH: ON by default. Verified against CONE-MT/MindMerger's
# mindmerger_tools/read_datasets.py -- the augmentation stage (read_math_train
# / read_x_csqa_train / read_xnli_train) DOES include English (x-csqa's list
# ends with 'English'; math.json is 10 langs incl. English -> 30k x 10 = the
# paper's 300k). The mapping stage does NOT (Stage 1 pooled stays
# English-free). make_english_aug.py remaps english.jsonl into the
# query/answer/nllb_lang_tag=eng_Latn schema; pool = 30k x 12.
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

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
DATA_ROOT="$SCRATCH/M2-ALIGN"
STAGE2="$DATA_ROOT/Stage2"
STAGE3="$DATA_ROOT/Stage3"

LLM_PATH="$SCRATCH/M2-ALIGN/Stage0/outputs/qwen3vl-8b-gqa"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
GQA_IMAGES_DIR="$STAGE3/data/gqa/images"
STAGE2_MAPPING_CKPT="$STAGE2/outputs/pooled/pytorch_model.bin"
POOLED_DATA_DIR="$STAGE3/data/pooled"
OUTPUT_DIR="$STAGE3/outputs/pooled"

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
  echo "ERROR: Stage 0 MonoVQA checkpoint not found: $LLM_PATH"
  echo "       Run: sbatch Stage0/job-scripts/train.sh"
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
  ENGLISH_SRC="$STAGE3/data/english.jsonl"
  ENGLISH_AUG="$STAGE3/data/english_aug.jsonl"
  if [ ! -f "$ENGLISH_SRC" ]; then
    echo "ERROR: INCLUDE_ENGLISH=1 but missing $ENGLISH_SRC."
    exit 1
  fi
  python "$PROJECT_ROOT/Stage3/make_english_aug.py" "$ENGLISH_SRC" "$ENGLISH_AUG"
  ln -sf "$ENGLISH_AUG" "$POOLED_DATA_DIR/english.jsonl"
  echo "INCLUDE_ENGLISH=1 -- English rows in the pool (MindMerger-faithful)."
else
  echo "INCLUDE_ENGLISH=0 -- no-English ablation (NOT MindMerger-faithful)."
fi

echo "Pooled data dir: $POOLED_DATA_DIR"
ls -l "$POOLED_DATA_DIR"
echo

echo "=== Start pooled Stage 3b VQA training ==="
cd "$PROJECT_ROOT"
python -u Stage3/train.py \
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
  --use-wandb \
  --wandb-mode    offline \
  --wandb-project m2-align \
  --wandb-run-name "stage3b-vqa-pooled" \
  --local-files-only

echo "=== Done ==="
date
