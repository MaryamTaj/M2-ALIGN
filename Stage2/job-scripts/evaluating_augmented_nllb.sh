#!/bin/bash
#SBATCH --job-name=stage2_eval
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=evaluating_stage2_aug_%j.log

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE1="$PROJECT_ROOT/Stage1"
STAGE2="$PROJECT_ROOT/Stage2"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
MAPPING_CKPT="$STAGE2/outputs/augmentation/mapping/pytorch_model.bin"

# Resolve HuggingFace cache repo directory to a concrete snapshot directory.
if [ -d "$MT_PATH/snapshots" ]; then
  SNAPSHOT_DIR="$(ls -d "$MT_PATH"/snapshots/* 2>/dev/null | head -n 1 || true)"
  if [ -n "$SNAPSHOT_DIR" ]; then
    MT_PATH="$SNAPSHOT_DIR"
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

echo "=== Hugging Face cache (offline) ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -f "$PROJECT_ROOT/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.tokens"
  echo "Loaded .tokens file"
else
  echo "WARNING: .tokens file not found"
fi
echo

if [ ! -d "$LLM_PATH" ]; then
  echo "ERROR: LLM snapshot not found: $LLM_PATH"
  exit 1
fi
if [ ! -d "$MT_PATH" ]; then
  echo "ERROR: NLLB snapshot not found: $MT_PATH"
  exit 1
fi
if [ ! -f "$MAPPING_CKPT" ]; then
  echo "ERROR: Stage2 mapping checkpoint not found: $MAPPING_CKPT"
  exit 1
fi

echo "=== Cached MMLU-ProX check (sw, wo, yo) ==="
python - <<'PY'
from datasets import load_dataset
for lang in ["sw", "wo", "yo"]:
    load_dataset("li-lab/MMLU-ProX", lang, split="validation", download_mode="reuse_dataset_if_exists")
    load_dataset("li-lab/MMLU-ProX", lang, split="test", download_mode="reuse_dataset_if_exists")
print("MMLU-ProX cache OK.")
PY

echo "=== Run post-augmentation MMLU-ProX eval ==="
cd "$PROJECT_ROOT"
python -u Stage1/run_evaluating.py \
  --llm-path "$LLM_PATH" \
  --mt-path "$MT_PATH" \
  --mapping-ckpt "$MAPPING_CKPT" \
  --langs sw wo yo \
  --max-seq-len 256 \
  --max-gen-len 256 \
  --local-files-only

