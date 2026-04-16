#!/bin/bash
#SBATCH --job-name=qwen3_VL
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=qwen3_VL_%j.log

set -euo pipefail

MODEL_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
nvidia-smi || true
echo

echo "=== Load modules ==="
module --force purge
module load StdEnv/2023
module load python/3.10
module load cudacore/.12.2.2
echo

echo "=== Activate virtual environment ==="
source "$SCRATCH/venvs/m2-align/bin/activate"

echo "=== Python environment ==="
which python
python -V
python -m pip -V
echo

echo "=== Installed packages (key ones) ==="
python -m pip list | grep -E 'torch|transformers|accelerate|huggingface-hub|tokenizers|safetensors|numpy' || true
echo

echo "=== Hugging Face cache ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "MODEL_PATH=$MODEL_PATH"
echo

echo "=== Hugging Face offline mode ==="
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo

echo "=== Optional token load ==="
if [ -f "$HOME/projects/def-annielee/tajm/M2-ALIGN/.tokens" ]; then
    source "$HOME/projects/def-annielee/tajm/M2-ALIGN/.tokens"
    echo "Loaded .tokens file"
else
    echo "No .tokens file found; continuing because model is cached locally"
fi
echo

echo "=== Run Stage1 smoke test ==="
python -u "$HOME/projects/def-annielee/tajm/M2-ALIGN/Stage1/smoke_test_qwen3vl.py" \
  --llm_path "$MODEL_PATH" \
  --prompt "Hello from Stage1 smoke test." \
  --dtype float16 \
  --device_map auto