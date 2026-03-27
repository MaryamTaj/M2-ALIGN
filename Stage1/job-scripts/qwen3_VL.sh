#!/bin/bash
#SBATCH --job-name=qwen3_VL
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=qwen3_VL_%j.log

set -euo pipefail

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
module load python/3.10
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
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# ---------------------------
# Hugging Face token
# ---------------------------
source ~/projects/def-annielee/tajm/M2-ALIGN/.tokens

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------------------------
# Run Stage1 smoke test
# ---------------------------
python -u ~/projects/def-annielee/tajm/M2-ALIGN/Stage1/smoke_test_qwen3vl.py \
  --llm_path "Qwen/Qwen3-VL-8B-Instruct" \
  --prompt "Hello from Stage1 smoke test." \
  --dtype auto \
  --device_map auto

