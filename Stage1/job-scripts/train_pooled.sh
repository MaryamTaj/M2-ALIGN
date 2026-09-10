#!/bin/bash
#SBATCH --job-name=stage1_train_pooled
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=490G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage1/logs/stage1_train_pooled_%x_%j.log

# Pooled Stage 1 mapping training -- MindMerger-style single shared checkpoint
# over all 11 languages, replacing the per-language train.sh runs. read_nllb()
# already loops a comma-separated language list, caps each at --train_num,
# concatenates and shuffles, so no train.py change is needed.
#
#   Reads   $SCRATCH/M2-ALIGN/Stage1/data/{Language}_to_English.jsonl for all 11
#   Warm LLM = the Stage 0 MonoVQA checkpoint (frozen phi), NOT vanilla Qwen3-VL
#   Saves   $SCRATCH/M2-ALIGN/Stage1/outputs/pooled/pytorch_model.bin
#
# WHY 4 GPUs / whole node:
#   Measured per-language run (bn): 97k examples x 3 epochs = 13.5 h on 1x
#   A100-40. Pooled is ~11x the data (~1.07M x 3 epochs) -> ~150 h on 1 GPU,
#   over the 7-day limit with zero margin and no preemption tolerance.
#   DeepSpeed data-parallel across the node's 4 A100s brings this to ~37 h.
#   Total GPU-hours (~148) are the SAME as the 11 per-language jobs this
#   replaces -- just concentrated into one run that yields the one pooled
#   checkpoint the design needs. Narval requires whole-node for 4-GPU jobs.
#
# MindMerger mapping stage = 3 epochs, English NOT in the pool (paper: math
# mapping = 9 languages, no English) -- matched here.
#
# Usage:  sbatch Stage1/job-scripts/train_pooled.sh

set -euo pipefail

# 11 pooled languages (Amharic/Igbo/Oromo excluded -- no data yet).
LANG_NAMES="Bengali,Russian,German,Chinese,Portuguese,Indonesian,Korean,Javanese,Mongolian,Sinhalese,Irish"

LLM_PATH="$SCRATCH/M2-ALIGN/Stage0/outputs/qwen3vl-8b-gqa"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
DATA_ROOT="$SCRATCH/M2-ALIGN/Stage1"
OUTPUT_DIR="$DATA_ROOT/outputs/pooled"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  LANGS=$LANG_NAMES"
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
python -m pip -V
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
echo "VISIBLE GPUS=${CUDA_VISIBLE_DEVICES:-unset}"
echo

echo "=== Load secrets (.tokens) ==="
if [ -f "$HOME/projects/def-annielee/tajm/M2-ALIGN/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$HOME/projects/def-annielee/tajm/M2-ALIGN/.tokens"
  echo "Loaded .tokens file"
else
  echo "WARNING: .tokens file not found; W&B may fail if WANDB_API_KEY is missing"
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
IFS=',' read -ra _NAMES <<< "$LANG_NAMES"
for name in "${_NAMES[@]}"; do
  f="$DATA_ROOT/data/${name}_to_English.jsonl"
  if [ ! -f "$f" ]; then
    echo "ERROR: Stage 1 data not found: $f"
    echo "       Run: python Stage1/load_text.py --languages $name --output_dir $DATA_ROOT/data --n_samples 100000"
    exit 1
  fi
done

cd "$HOME/projects/def-annielee/tajm/M2-ALIGN/Stage1"
# No --num_gpus: the deepspeed launcher auto-detects all 4 visible A100s.
# train.py derives gradient_accumulation = 24 / (1 * 4) = 6.
deepspeed --master_port "${MASTER_PORT:-50030}" train.py --deepspeed \
  --llm_path "$LLM_PATH" \
  --mt_path "$MT_PATH" \
  --save_name "M2-ALIGN-pooled" \
  --output_dir "$OUTPUT_DIR" \
  --stage_name mapping \
  --task nllb_corpus \
  --augmentation False \
  --nllb_data_dir "$DATA_ROOT/data" \
  --nllb_languages "$LANG_NAMES" \
  --train_num 100000 \
  --val_size 3000 \
  --train_batch_size 24 \
  --train_micro_batch_size_per_gpu 1 \
  --epoch_num 3 \
  --max_seq_len 256 \
  --max_gen_len 256 \
  --eval_batch_size 2 \
  --use_wandb True \
  --wandb_mode offline \
  --wandb_project m2-align \
  --wandb_run_name "stage1-nllb-pooled"

echo "=== Done ==="
date
