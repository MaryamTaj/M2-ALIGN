#!/bin/bash
#SBATCH --job-name=stage2_translate
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=build_query_translation_%j.log

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE2="$PROJECT_ROOT/Stage2"

INPUT_PATH="$STAGE2/data/task_specialization_en.jsonl"
OUTPUT_PATH="$STAGE2/data/task_specialization_translated.jsonl"
NLLB_MODEL="$SCRATCH/huggingface/nllb-200-3.3B-full"
TARGET_LANGS="sw,yo,wo"

if [ -d "$NLLB_MODEL" ]; then
  for d in "$NLLB_MODEL"/*; do
    if [ -d "$d" ]; then
      NLLB_MODEL="$d"
      break
    fi
  done
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
python -m pip -V
echo

echo "=== Hugging Face cache config ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
echo "HF_HOME=$HF_HOME"
echo "NLLB_MODEL=$NLLB_MODEL"
echo

echo "=== Optional token loading (.tokens) ==="
if [ -f "$PROJECT_ROOT/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.tokens"
  echo "Loaded .tokens file"
else
  echo "WARNING: .tokens file not found; HF downloads may be unauthenticated"
fi
echo

if [ ! -f "$INPUT_PATH" ]; then
  echo "ERROR: Input file not found: $INPUT_PATH"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "=== Start Stage2 query translation build ==="
cd "$PROJECT_ROOT"
python -u Stage2/build_query_translation_data.py \
  --input-path "$INPUT_PATH" \
  --output-path "$OUTPUT_PATH" \
  --target-languages "$TARGET_LANGS" \
  --nllb-model "$NLLB_MODEL"

echo "=== Done ==="
date

