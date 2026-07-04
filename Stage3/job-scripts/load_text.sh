#!/bin/bash
#SBATCH --job-name=stage3a_load_text
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage3/logs/stage3a_load_text_%j.log

# Build Stage 3a training data (math only — mgsm/msvamp) for BOTH Swahili
# and Bengali on a compute node. Writes one file per (task, language) —
# e.g. Stage3/data/stage3a/mgsm/mgsm_swahili.jsonl and .../mgsm_bengali.jsonl
# — so Stage3/train_text.py (which globs every *.jsonl in --data-dir)
# trains jointly on both languages, not Swahili alone.
#
# xnli/xcsqa are not built here: standard XNLI's language list is
# ar/bg/de/el/en/es/fr/hi/ru/sw/th/tr/ur/vi/zh -- "bg" is Bulgarian, not
# Bengali -- and INK-USC/xcsr (X-CSQA) has the same gap. Both tasks are
# dropped from Stage 3a entirely for now (see Stage3/load_text.py).
#
# BEFORE SUBMITTING: run the following on a login node (internet access) to
# populate the HuggingFace cache that this job reads offline:
#
#   source "$SCRATCH/venvs/m2-align/bin/activate"
#   export HF_HOME="$SCRATCH/huggingface"
#   python -c "from datasets import load_dataset; \
#     load_dataset('meta-math/MetaMathQA', split='train')"
#
# The NLLB-200-3.3B model weights are assumed already downloaded at NLLB_MODEL.
#
# Usage: sbatch Stage3/job-scripts/load_text.sh

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
NLLB_MODEL="$HOME/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/1a07f7d195896b2114afcb79b7b57ab512e7b43e"
DATA_DIR="$PROJECT_ROOT/Stage3/data/stage3a"
LANGUAGES=(Swahili Bengali)

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
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
echo "LANGUAGES=${LANGUAGES[*]}"
echo

if [ ! -d "$NLLB_MODEL" ]; then
  echo "ERROR: NLLB snapshot not found: $NLLB_MODEL"
  echo "Download it on a login node with: python -c \"from transformers import AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-3.3B')\""
  exit 1
fi

cd "$PROJECT_ROOT"

for LANG in "${LANGUAGES[@]}"; do
  LANG_LOWER=$(echo "$LANG" | tr '[:upper:]' '[:lower:]')

  # ---------------------------------------------------------------------
  # Math — MetaMathQA × 30K, translated EN->$LANG with NLLB.
  # mgsm and msvamp use identical training data; translate once, symlink second.
  # ---------------------------------------------------------------------
  echo "=== [$LANG] Math (MetaMathQA -> $LANG) ==="
  python -u Stage3/load_text.py \
    --task        mgsm \
    --language    "$LANG" \
    --output_dir  "$DATA_DIR/mgsm" \
    --nllb_model  "$NLLB_MODEL" \
    --batch_size  32 \
    --num_beams   4 \
    --seed        42

  mkdir -p "$DATA_DIR/msvamp"
  ln -sf "$DATA_DIR/mgsm/mgsm_${LANG_LOWER}.jsonl" "$DATA_DIR/msvamp/msvamp_${LANG_LOWER}.jsonl"
  echo "Symlinked msvamp_${LANG_LOWER} -> mgsm_${LANG_LOWER} (identical source and seed)"
  echo
done

echo "=== All tasks complete ==="
echo "Outputs:"
for f in "$DATA_DIR"/mgsm/*.jsonl \
          "$DATA_DIR"/msvamp/*.jsonl; do
  if [ -e "$f" ]; then
    wc -l "$f"
  else
    echo "MISSING: $f"
  fi
done
date
