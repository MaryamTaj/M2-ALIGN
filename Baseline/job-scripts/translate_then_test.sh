#!/bin/bash
#SBATCH --job-name=baseline_load_translated_data
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Baseline/logs/baseline_load_translated_data_%x_%j.log

# Translate-test eval data: native-language xGQA/CVQA questions -> English
# with NLLB-200-3.3B (the translator Stage 3b's training data came from).
# Writes $DATA_ROOT/Baseline/data/translate_test/{xgqa,cvqa}/$LANG.jsonl.
# A benchmark with no file for $LANG is skipped (xGQA: no jv/mn/si/ga;
# CVQA: no de). No network calls -- NLLB weights are read from NLLB_MODEL.
#
# Usage:
#   sbatch --export=LANG=bn --job-name=baseline_load_translated_data_bn load_translated_data.sh
#   sbatch --export=LANG=de,BENCHMARKS=xgqa                             load_translated_data.sh
#
# Languages: bn de id ko pt ru zh (xGQA), bn ga id jv ko mn pt ru si zh (CVQA)

set -euo pipefail

LANG_CODES="bn ru de zh pt id ko jv mn si ga"
LANG="${LANG:-bn}"
BENCHMARKS="${BENCHMARKS:-xgqa,cvqa}"
if [[ ! " $LANG_CODES " =~ " $LANG " ]]; then
  echo "ERROR: Unknown LANG=$LANG. Known codes: $LANG_CODES"
  exit 1
fi

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"

# Data/outputs/logs live on $SCRATCH; the git checkout under $HOME only holds code.
DATA_ROOT="$SCRATCH/M2-ALIGN"
NLLB_MODEL="$HOME/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/1a07f7d195896b2114afcb79b7b57ab512e7b43e"
EVAL_DATA_DIR="$DATA_ROOT/Stage3/data"
OUTPUT_DIR="$DATA_ROOT/Baseline/data/translate_test"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  LANG=$LANG  BENCHMARKS=$BENCHMARKS"
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
echo "EVAL_DATA_DIR=$EVAL_DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo

if [ ! -d "$NLLB_MODEL" ]; then
  echo "ERROR: NLLB snapshot not found: $NLLB_MODEL"
  exit 1
fi

cd "$PROJECT_ROOT"

echo "=== Native -> English NLLB translation ($LANG: $BENCHMARKS) ==="
python -u Baseline/load_translated_data.py \
  --lang        "$LANG" \
  --benchmarks  "$BENCHMARKS" \
  --data-dir    "$EVAL_DATA_DIR" \
  --output-dir  "$OUTPUT_DIR" \
  --nllb-model  "$NLLB_MODEL" \
  --batch-size  32 \
  --num-beams   4

echo "=== Done ==="
echo "Output:"
for BENCH in ${BENCHMARKS//,/ }; do
  OUT_FILE="$OUTPUT_DIR/$BENCH/$LANG.jsonl"
  if [ -e "$OUT_FILE" ]; then wc -l "$OUT_FILE"; else echo "SKIPPED/MISSING: $OUT_FILE"; fi
done
date
