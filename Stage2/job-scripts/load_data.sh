#!/bin/bash
#SBATCH --job-name=stage2_load_data
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage2/logs/stage2_load_data_%j.log

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE2="$PROJECT_ROOT/Stage2"

NLLB_MODEL="${NLLB_MODEL:-$SCRATCH/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/manual}"
OUTPUT_DIR="$STAGE2/data"
TARGET_LANGS="${TARGET_LANGS:-sw,yo,wo}"
N_SAMPLES="${N_SAMPLES:-3000}"

# Resolve NLLB snapshot directory if needed.
if [ -d "$NLLB_MODEL/snapshots" ]; then
  SNAPSHOT_DIR="$(ls -d "$NLLB_MODEL"/snapshots/* 2>/dev/null | head -n 1 || true)"
  if [ -n "$SNAPSHOT_DIR" ]; then
    NLLB_MODEL="$SNAPSHOT_DIR"
  fi
fi

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

echo "=== Hugging Face cache config ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
# NOTE: datasets (MMLU, MetaMathQA, MultiNLI) must be pre-downloaded.
# Run once without HF_HUB_OFFLINE=1 to cache them, then re-enable for
# subsequent runs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HF_HOME=$HF_HOME"
echo "NLLB_MODEL=$NLLB_MODEL"
echo

echo "=== Optional token loading (.tokens) ==="
if [ -f "$PROJECT_ROOT/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.tokens"
  echo "Loaded .tokens"
else
  echo "WARNING: .tokens not found"
fi
echo

if [ ! -d "$NLLB_MODEL" ]; then
  echo "ERROR: NLLB model not found: $NLLB_MODEL"
  exit 1
fi
if [ ! -f "$NLLB_MODEL/tokenizer_config.json" ]; then
  echo "ERROR: tokenizer_config.json missing in $NLLB_MODEL"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Start Stage 2 translated data build ==="
echo "Target languages : $TARGET_LANGS"
echo "Samples per dataset: $N_SAMPLES"
echo "Output dir: $OUTPUT_DIR"
cd "$PROJECT_ROOT"
python -u Stage2/load_data.py \
  --output-dir "$OUTPUT_DIR" \
  --nllb-model "$NLLB_MODEL" \
  --target-languages "$TARGET_LANGS" \
  --n-samples "$N_SAMPLES" \
  --batch-size 16 \
  --num-beams 4 \
  --max-source-length 256 \
  --max-source-length-long 512 \
  --max-target-length-short 256 \
  --max-target-length-long 512

echo "=== Done ==="
date
