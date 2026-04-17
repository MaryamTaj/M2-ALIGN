#!/bin/bash
#SBATCH --job-name=training_nllb
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=training_nllb_%j.log

set -euo pipefail

# Local model snapshots on SCRATCH (offline compute nodes).
LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then
      MT_PATH="$d"
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

echo "=== Hugging Face cache/offline config ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HF_HOME=$HF_HOME"
echo "LLM_PATH=$LLM_PATH"
echo "MT_PATH=$MT_PATH"
echo

if [ ! -d "$LLM_PATH" ]; then
  echo "ERROR: LLM snapshot path not found: $LLM_PATH"
  exit 1
fi
if [ ! -d "$MT_PATH" ]; then
  echo "ERROR: MT snapshot path not found: $MT_PATH"
  exit 1
fi

echo "=== Load secrets (.tokens) ==="
if [ -f "$HOME/projects/def-annielee/tajm/M2-ALIGN/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$HOME/projects/def-annielee/tajm/M2-ALIGN/.tokens"
  echo "Loaded .tokens file"
else
  echo "WARNING: .tokens file not found; W&B may fail if WANDB_API_KEY is missing"
fi
echo

echo "=== Start Stage1 NLLB mapping training ==="
cd "$HOME/projects/def-annielee/tajm/M2-ALIGN/Stage1"
deepspeed --master_port 50002 run_training.py --deepspeed \
  --llm_path "$LLM_PATH" \
  --mt_path "$MT_PATH" \
  --stage_name mapping \
  --task nllb_corpus \
  --augmentation False \
  --nllb_data_dir ./data/nllb \
  --train_num 3000 \
  --val_size 900 \
  --train_batch_size 24 \
  --train_micro_batch_size_per_gpu 1 \
  --epoch_num 3 \
  --max_seq_len 256 \
  --max_gen_len 256 \
  --eval_batch_size 2 \
  --use_wandb True \
  --wandb_mode offline \
  --wandb_project m2-align \
  --wandb_run_name stage1
