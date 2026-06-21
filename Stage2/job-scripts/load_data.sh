#!/bin/bash
#SBATCH --job-name=stage2_load_data
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage2/logs/stage2_load_data_%j.log

# Build all Stage 2 training data (math, XNLI, X-CSQA) on a compute node.
#
# BEFORE SUBMITTING: run the following on a login node (internet access) to
# populate the HuggingFace cache that this job reads offline:
#
#   source "$SCRATCH/venvs/m2-align/bin/activate"
#   export HF_HOME="$SCRATCH/huggingface"
#   python -c "from datasets import load_dataset; \
#     load_dataset('meta-math/MetaMathQA', split='train'); \
#     load_dataset('xnli', 'sw', split='validation'); \
#     load_dataset('commonsense_qa', split='train')"
#
# The NLLB-200-3.3B model weights are assumed already downloaded at NLLB_MODEL.
#
# Usage: sbatch Stage2/job-scripts/load_data.sh

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
NLLB_MODEL="$HOME/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/1a07f7d195896b2114afcb79b7b57ab512e7b43e"
DATA_DIR="$PROJECT_ROOT/Stage2/data/stage2"

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
echo

if [ ! -d "$NLLB_MODEL" ]; then
  echo "ERROR: NLLB snapshot not found: $NLLB_MODEL"
  echo "Download it on a login node with: python -c \"from transformers import AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-3.3B')\""
  exit 1
fi

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 1. XNLI — Swahili validation split (no translation needed, very fast)
# ---------------------------------------------------------------------------
echo "=== [1/3] XNLI ==="
python -u Stage2/load_data.py \
  --task        xnli \
  --output_dir  "$DATA_DIR/xnli" \
  --seed        42
echo

# ---------------------------------------------------------------------------
# 2. Math — MetaMathQA × 30K, translated EN→SW with NLLB
#    mgsm and msvamp use identical training data; translate once, symlink second.
# ---------------------------------------------------------------------------
echo "=== [2/3] Math (MetaMathQA → Swahili) ==="
python -u Stage2/load_data.py \
  --task        mgsm \
  --output_dir  "$DATA_DIR/mgsm" \
  --nllb_model  "$NLLB_MODEL" \
  --batch_size  32 \
  --num_beams   4 \
  --seed        42

mkdir -p "$DATA_DIR/msvamp"
ln -sf "$DATA_DIR/mgsm/mgsm.jsonl" "$DATA_DIR/msvamp/msvamp.jsonl"
echo "Symlinked msvamp → mgsm (identical source and seed)"
echo

# ---------------------------------------------------------------------------
# 3. X-CSQA — CommonsenseQA train × 8,888, translated EN→SW with NLLB
# ---------------------------------------------------------------------------
echo "=== [3/3] X-CSQA (CommonsenseQA → Swahili) ==="
python -u Stage2/load_data.py \
  --task        xcsqa \
  --output_dir  "$DATA_DIR/xcsqa" \
  --nllb_model  "$NLLB_MODEL" \
  --batch_size  32 \
  --num_beams   4 \
  --seed        42
echo

echo "=== All tasks complete ==="
echo "Outputs:"
for f in "$DATA_DIR"/mgsm/mgsm.jsonl \
          "$DATA_DIR"/msvamp/msvamp.jsonl \
          "$DATA_DIR"/xnli/xnli.jsonl \
          "$DATA_DIR"/xcsqa/xcsqa.jsonl; do
  if [ -e "$f" ]; then
    wc -l "$f"
  else
    echo "MISSING: $f"
  fi
done
date
