#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=baseline_eval_%j.log

set -euo pipefail

MODEL_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
nvidia-smi || true
echo

# ---------------------------
# Load modules
# ---------------------------
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.2.2
module load arrow/18.1.0

# ---------------------------
# Activate SCRATCH virtualenv
# ---------------------------
source "$SCRATCH/venvs/m2-align/bin/activate"


echo "=== Python env ==="
which python
python -V
python -m pip -V
echo

echo "=== Import sanity checks ==="
python -c "import datasets, transformers, huggingface_hub, fsspec, dill, httpx; print('imports ok')"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo

# ---------------------------
# Hugging Face cache on SCRATCH
# ---------------------------
export HF_HOME=$SCRATCH/huggingface
export HF_DATASETS_CACHE=$SCRATCH/huggingface/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# ---------------------------
# Hugging Face token
# ---------------------------
if [ -f "$PROJECT_ROOT/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.tokens"
  echo "Loaded .tokens file"
else
  echo "WARNING: .tokens file not found"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------------------------
# Offline preflight checks
# ---------------------------
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: Model snapshot not found: $MODEL_PATH"
  echo "Please cache Qwen3-VL under \$SCRATCH/huggingface first."
  exit 1
fi

echo "=== Check cached MMLU-ProX for sw/wo/yo ==="
python - <<'PY'
from datasets import load_dataset
langs = ["sw", "wo", "yo"]
for lang in langs:
    _ = load_dataset("li-lab/MMLU-ProX", lang, split="validation", download_mode="reuse_dataset_if_exists")
    _ = load_dataset("li-lab/MMLU-ProX", lang, split="test", download_mode="reuse_dataset_if_exists")
print("Cached dataset check passed for sw/wo/yo.")
PY

# ---------------------------
# Run script
# ---------------------------
python -u "$PROJECT_ROOT/Baseline/mmlu_prox.py" \
  --model-id "$MODEL_PATH" \
  --langs sw wo yo \
  --local-files-only

