#!/bin/bash
#SBATCH --job-name=stage3b_load_vqa_lang
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage3/logs/stage3b_load_vqa_%x_%j.log

# Per-language Stage 3b VQA training data: GQA (English) -> NLLB-3.3B
# translation into one target language at a time. Writes a single flat file
# per language (Stage3/data/$LANG.jsonl) -- does not touch Bengali's
# Stage3/data/stage3b.
#
# Reads the pre-sampled GQA rows from Stage3/data/english.jsonl (produced
# offline by Stage3/dump_gqa_sample.py on a workstation and Globus-
# transferred here -- see the README) instead of calling load_dataset()
# itself, so this job makes NO network calls at all, not even to the
# pre-warmed HF cache. That JSONL is sampled once with a fixed seed and is
# independent of target language, so the SAME file is reused for ru/de/zh
# (and was for Bengali too) -- meaning the vg_image_ids referenced by every
# per-language output file are identical across languages. The vg_image_ids
# here should therefore already be covered by whatever's in
# Stage3/data/gqa/images/ from the Bengali run -- verify with the id-diff
# check in the README before assuming no new images.zip extraction is needed.
#
# The NLLB-200-3.3B model weights are assumed already downloaded at NLLB_MODEL.
#
# Usage:
#   LANG=ru sbatch --job-name=stage3b_load_vqa_ru load_vqa_data_lang.sh
#   LANG=de sbatch --job-name=stage3b_load_vqa_de load_vqa_data_lang.sh
#   LANG=zh sbatch --job-name=stage3b_load_vqa_zh load_vqa_data_lang.sh

set -euo pipefail

declare -A LANG_NAMES=( [ru]="Russian" [de]="German" [zh]="Chinese" )
LANG="${LANG:-ru}"
LANG_NAME="${LANG_NAMES[$LANG]:-}"
if [ -z "$LANG_NAME" ]; then
  echo "ERROR: Unknown LANG=$LANG. Known codes: ${!LANG_NAMES[*]}"
  exit 1
fi

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
NLLB_MODEL="$HOME/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/1a07f7d195896b2114afcb79b7b57ab512e7b43e"
DATA_DIR="$PROJECT_ROOT/Stage3/data"
OUT_FILE="$DATA_DIR/$LANG.jsonl"
GQA_JSONL="$DATA_DIR/english.jsonl"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  LANG=$LANG  LANG_NAME=$LANG_NAME"
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

echo "=== HuggingFace cache config (offline mode) ==="
export HF_HOME="$SCRATCH/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
echo "HF_HOME=$HF_HOME"
echo "NLLB_MODEL=$NLLB_MODEL"
echo "DATA_DIR=$DATA_DIR"
echo

if [ ! -d "$NLLB_MODEL" ]; then
  echo "ERROR: NLLB snapshot not found: $NLLB_MODEL"
  exit 1
fi
if [ ! -f "$GQA_JSONL" ]; then
  echo "ERROR: Pre-sampled GQA file not found: $GQA_JSONL"
  echo "       Run Stage3/dump_gqa_sample.py on a workstation and Globus-transfer it here."
  exit 1
fi

cd "$PROJECT_ROOT"

echo "=== GQA -> NLLB translation ($LANG_NAME) ==="
python -u Stage3/load_vqa_data.py \
  --languages   "$LANG_NAME" \
  --gqa_jsonl   "$GQA_JSONL" \
  --output_dir  "$DATA_DIR" \
  --nllb_model  "$NLLB_MODEL" \
  --batch_size  32 \
  --num_beams   4

# load_vqa_data.py writes <output_dir>/<lang name, lowercased>.jsonl (e.g.
# Stage3/data/russian.jsonl) -- rename to the flat $LANG.jsonl convention.
WRITTEN_FILE="$DATA_DIR/$(echo "$LANG_NAME" | tr '[:upper:]' '[:lower:]').jsonl"
if [ -e "$WRITTEN_FILE" ]; then
  mv "$WRITTEN_FILE" "$OUT_FILE"
fi

echo "=== Done ==="
echo "Output:"
if [ -e "$OUT_FILE" ]; then wc -l "$OUT_FILE"; else echo "MISSING: $OUT_FILE"; fi
date
