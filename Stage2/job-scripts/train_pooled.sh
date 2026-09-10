#!/bin/bash
#SBATCH --job-name=stage2_train_pooled
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage2/logs/stage2_train_pooled_%x_%j.log

# Pooled Stage 2 visual-grounding training -- one shared checkpoint over all
# 11 languages, replacing the per-language train.sh runs. Stage2/train.py
# already accepts many --data-path / --image-cache-dir values (paired
# positionally), concatenates and shuffles them, so this is a launcher-only
# change.
#
#   Warm mapping = Stage1/outputs/pooled/pytorch_model.bin
#   Warm LLM     = the Stage 0 MonoVQA checkpoint (frozen phi)
#   Reads  wit_pairs.jsonl + cc3m_pairs.jsonl for each of the 11 languages
#   Saves  $SCRATCH/M2-ALIGN/Stage2/outputs/pooled/pytorch_model.bin
#
# 1 GPU is enough: measured per-language data is ~1.2k WIT + ~10k CC3M
# (~11k/lang, NOT 51k -- the CC3M runs landed at ~10k), so the pool is
# ~120k. At Stage-2 throughput that is ~10 h for 3 epochs on one A100-40.
# Stage 2 has no MindMerger equivalent; --epochs 3 keeps the repo default.
# Override with EPOCHS=.
#
# Usage:
#   sbatch Stage2/job-scripts/train_pooled.sh
#   EPOCHS=5 sbatch Stage2/job-scripts/train_pooled.sh

set -euo pipefail

LANG_CODES="bn ru de zh pt id ko jv mn si ga"
EPOCHS="${EPOCHS:-3}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
DATA_ROOT="$SCRATCH/M2-ALIGN"
STAGE1="$DATA_ROOT/Stage1"
STAGE2="$DATA_ROOT/Stage2"

LLM_PATH="$SCRATCH/M2-ALIGN/Stage0/outputs/qwen3vl-8b-gqa"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
STAGE1_MAPPING_CKPT="$STAGE1/outputs/pooled/pytorch_model.bin"
OUTPUT_DIR="$STAGE2/outputs/pooled"

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
if [ ! -f "$STAGE1_MAPPING_CKPT" ]; then
  echo "ERROR: pooled Stage 1 mapping checkpoint not found: $STAGE1_MAPPING_CKPT"
  echo "       Run: sbatch Stage1/job-scripts/train_pooled.sh"
  exit 1
fi

# Build the paired --data-path / --image-cache-dir lists over all languages.
DATA_PATHS=()
CACHE_DIRS=()
CC3M_CACHE="$STAGE2/data/cc3m/image_cache"
for code in $LANG_CODES; do
  wit="$STAGE2/data/$code/wit_pairs.jsonl"
  cc3m="$STAGE2/data/$code/cc3m_pairs.jsonl"
  if [ ! -f "$wit" ]; then
    echo "ERROR: missing $wit -- run load_base_data.py for $code and Globus it over."
    exit 1
  fi
  DATA_PATHS+=("$wit")
  CACHE_DIRS+=("$STAGE2/data/$code/image_cache")
  if [ -f "$cc3m" ]; then
    DATA_PATHS+=("$cc3m")
    CACHE_DIRS+=("$CC3M_CACHE")
  else
    echo "NOTE: $cc3m not found -- $code contributes WIT only."
  fi
done

echo "=== Start pooled Stage 2 training ==="
echo "DATA_PATHS (${#DATA_PATHS[@]}): ${DATA_PATHS[*]}"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo
cd "$PROJECT_ROOT"
python -u Stage2/train.py \
  --data-path  "${DATA_PATHS[@]}" \
  --image-cache-dir "${CACHE_DIRS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --stage1-mapping-ckpt "$STAGE1_MAPPING_CKPT" \
  --mt-path    "$MT_PATH" \
  --llm-path   "$LLM_PATH" \
  --epochs     "$EPOCHS" \
  --lr         2e-5 \
  --train-batch-size 4 \
  --eval-batch-size  4 \
  --grad-accum 8 \
  --max-mt-seq-len 512 \
  --max-gen-len 512 \
  --save-steps 200 \
  --use-wandb \
  --wandb-mode    offline \
  --wandb-project m2-align \
  --wandb-run-name "stage2-wit-pooled" \
  --local-files-only

echo "=== Done ==="
date
