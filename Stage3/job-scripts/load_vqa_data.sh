#!/bin/bash
#SBATCH --job-name=stage3b_load_vqa
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage3/logs/stage3b_load_vqa_%j.log

# Build Stage 3b VQA training data: GQA (English) -> NLLB-3.3B translation
# into Bengali (currently the sole active Stage 3 VQA language --
# Indonesian/Javanese are deferred, see Stage3/load_vqa_data.py).
#
# BEFORE SUBMITTING: run the following on a login node (internet access) to
# populate the HuggingFace cache that this job reads offline:
#
#   source "$SCRATCH/venvs/m2-align/bin/activate"
#   export HF_HOME="$SCRATCH/huggingface"
#   python -c "from datasets import load_dataset; \
#     load_dataset('lmms-lab/GQA', 'train_balanced_instructions', split='train')"
#
# The NLLB-200-3.3B model weights are assumed already downloaded at NLLB_MODEL.
#
# Usage: sbatch Stage3/job-scripts/load_vqa_data.sh

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
NLLB_MODEL="$HOME/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/1a07f7d195896b2114afcb79b7b57ab512e7b43e"
DATA_DIR="$PROJECT_ROOT/Stage3/data/stage3b"

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

cd "$PROJECT_ROOT"

echo "=== GQA -> NLLB translation (Bengali) ==="
python -u Stage3/load_vqa_data.py \
  --languages   Bengali \
  --n_samples   30000 \
  --output_dir  "$DATA_DIR" \
  --nllb_model  "$NLLB_MODEL" \
  --batch_size  32 \
  --num_beams   4 \
  --seed        42

echo "=== Done ==="
echo "Outputs:"
for f in "$DATA_DIR"/bengali.jsonl; do
  if [ -e "$f" ]; then
    wc -l "$f"
  else
    echo "MISSING: $f"
  fi
done
date
