#!/bin/bash
#SBATCH --job-name=stage2_load_translated_data
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage2/logs/stage2_load_translated_data_%x_%j.log

# Per-language Stage 2 CC3M training data: English CC3M captions -> NLLB-3.3B
# translation into one target language at a time. Writes
# Stage2/data/$LANG/cc3m_pairs.jsonl -- does not touch that language's
# wit_pairs.jsonl.
#
# Reads the pre-sampled + pre-downloaded CC3M rows from
# Stage2/data/cc3m/english.jsonl (produced by Stage2/load_base_data.py,
# which also cached every image into Stage2/data/cc3m/image_cache/ -- shared
# across all languages since the images themselves don't change per
# language, only the translated caption does). This job makes NO image
# downloads and no dataset network calls -- just NLLB translation.
#
# The NLLB-200-3.3B model weights are assumed already downloaded at NLLB_MODEL.
#
# Usage:
#   LANG=bn sbatch --job-name=stage2_load_translated_data_bn load_translated_data.sh
#   LANG=ru sbatch --job-name=stage2_load_translated_data_ru load_translated_data.sh
#   LANG=de sbatch --job-name=stage2_load_translated_data_de load_translated_data.sh
#   LANG=zh sbatch --job-name=stage2_load_translated_data_zh load_translated_data.sh
#   LANG=pt sbatch --job-name=stage2_load_translated_data_pt load_translated_data.sh
#   LANG=id sbatch --job-name=stage2_load_translated_data_id load_translated_data.sh
#   LANG=ko sbatch --job-name=stage2_load_translated_data_ko load_translated_data.sh
#   LANG=jv sbatch --job-name=stage2_load_translated_data_jv load_translated_data.sh
#   LANG=mn sbatch --job-name=stage2_load_translated_data_mn load_translated_data.sh
#   LANG=si sbatch --job-name=stage2_load_translated_data_si load_translated_data.sh
#   LANG=ga sbatch --job-name=stage2_load_translated_data_ga load_translated_data.sh

set -euo pipefail

LANG_CODES="bn ru de zh pt id ko jv mn si ga"
LANG="${LANG:-ru}"
if [[ ! " $LANG_CODES " =~ " $LANG " ]]; then
  echo "ERROR: Unknown LANG=$LANG. Known codes: $LANG_CODES"
  exit 1
fi

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"

# Data/outputs/logs live on $SCRATCH; the git checkout under $HOME only holds code.
DATA_ROOT="$SCRATCH/M2-ALIGN"
NLLB_MODEL="$HOME/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/1a07f7d195896b2114afcb79b7b57ab512e7b43e"
DATA_DIR="$DATA_ROOT/Stage2/data"
OUT_FILE="$DATA_DIR/$LANG/cc3m_pairs.jsonl"
CC3M_JSONL="$DATA_DIR/cc3m/english.jsonl"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  LANG=$LANG"
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
if [ ! -f "$CC3M_JSONL" ]; then
  echo "ERROR: Pre-sampled CC3M file not found: $CC3M_JSONL"
  echo "       Run Stage2/load_base_data.py on a workstation and Globus-transfer it here."
  exit 1
fi

cd "$PROJECT_ROOT"

echo "=== CC3M -> NLLB translation ($LANG) ==="
python -u Stage2/load_translated_data.py \
  --languages   "$LANG" \
  --cc3m_jsonl  "$CC3M_JSONL" \
  --output_dir  "$DATA_DIR" \
  --nllb_model  "$NLLB_MODEL" \
  --batch_size  32 \
  --num_beams   4

echo "=== Done ==="
echo "Output:"
if [ -e "$OUT_FILE" ]; then wc -l "$OUT_FILE"; else echo "MISSING: $OUT_FILE"; fi
date
