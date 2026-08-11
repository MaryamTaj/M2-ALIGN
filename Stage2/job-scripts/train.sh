#!/bin/bash
#SBATCH --job-name=stage2_train
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage2/logs/stage2_train_%x_%j.log

# Per-language Stage 2 training -- warm-starts from that language's own
# Stage 1 checkpoint ($SCRATCH/M2-ALIGN/Stage1/outputs/$LANG) and saves its own
# checkpoint, independent of the other languages (run with LANG=bn for
# Bengali -- same outputs/$LANG convention as everyone else, no exception).
#
# Trains on wit_pairs.jsonl (own per-language image cache) plus, if present,
# cc3m_pairs.jsonl (shared Stage2/data/cc3m/image_cache -- same images
# reused across every language, only the caption differs). cc3m_pairs.jsonl
# is optional: languages whose Stage2/job-scripts/load_translated_data.sh
# job hasn't run yet fall back to WIT-only training instead of failing.
#
# Usage:
#   LANG=ru sbatch --job-name=stage2_train_ru train.sh
#   LANG=de sbatch --job-name=stage2_train_de train.sh
#   LANG=zh sbatch --job-name=stage2_train_zh train.sh
#
# Reads $SCRATCH/M2-ALIGN/Stage2/data/$LANG/wit_pairs.jsonl (required),
#       $SCRATCH/M2-ALIGN/Stage2/data/$LANG/cc3m_pairs.jsonl (optional), and
#       $SCRATCH/M2-ALIGN/Stage1/outputs/$LANG/pytorch_model.bin
# Saves  $SCRATCH/M2-ALIGN/Stage2/outputs/$LANG/pytorch_model.bin

set -euo pipefail

LANG="${LANG:-ru}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"

# Data/outputs/logs live on $SCRATCH; the git checkout under $HOME only holds code.
DATA_ROOT="$SCRATCH/M2-ALIGN"
STAGE1="$DATA_ROOT/Stage1"
STAGE2="$DATA_ROOT/Stage2"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
STAGE1_MAPPING_CKPT="$STAGE1/outputs/$LANG/pytorch_model.bin"

WIT_DATA_PATH="$STAGE2/data/$LANG/wit_pairs.jsonl"
CC3M_DATA_PATH="$STAGE2/data/$LANG/cc3m_pairs.jsonl"
OUTPUT_DIR="$STAGE2/outputs/$LANG"

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
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  LANG=$LANG"
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
echo "HF_HOME=$HF_HOME"
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
  echo "ERROR: LLM snapshot path not found: $LLM_PATH"
  exit 1
fi
if [ ! -d "$MT_PATH" ]; then
  echo "ERROR: MT snapshot path not found: $MT_PATH"
  exit 1
fi
if [ ! -f "$STAGE1_MAPPING_CKPT" ]; then
  echo "ERROR: Stage 1 mapping checkpoint not found: $STAGE1_MAPPING_CKPT"
  echo "       Run: LANG=$LANG sbatch Stage1/job-scripts/train.sh"
  exit 1
fi
if [ ! -f "$WIT_DATA_PATH" ]; then
  echo "ERROR: Data file not found: $WIT_DATA_PATH"
  echo "       Run load_base_data.py for --languages $LANG and transfer wit_pairs.jsonl + image_cache here (see the workstation+Globus step)."
  exit 1
fi

DATA_PATHS=("$WIT_DATA_PATH")
CACHE_DIRS=("$STAGE2/data/$LANG/image_cache")
if [ -f "$CC3M_DATA_PATH" ]; then
  DATA_PATHS+=("$CC3M_DATA_PATH")
  CACHE_DIRS+=("$STAGE2/data/cc3m/image_cache")
else
  echo "NOTE: $CC3M_DATA_PATH not found -- training on WIT only."
  echo "      Run LANG=$LANG sbatch Stage2/job-scripts/load_translated_data.sh to add CC3M."
fi

echo "=== Start Stage 2 training ($LANG) ==="
echo "DATA_PATHS=${DATA_PATHS[*]}"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "STAGE1_MAPPING_CKPT=$STAGE1_MAPPING_CKPT"
echo

RESUME_ARGS=()
TRAINING_STATE="$OUTPUT_DIR/training_state.pt"
if [ -f "$TRAINING_STATE" ]; then
  echo "Found existing training_state.pt — resuming: $TRAINING_STATE"
  RESUME_ARGS=(--resume-from-checkpoint "$TRAINING_STATE")
fi
echo

cd "$PROJECT_ROOT"
python -u Stage2/train.py \
  --data-path  "${DATA_PATHS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --image-cache-dir "${CACHE_DIRS[@]}" \
  --stage1-mapping-ckpt "$STAGE1_MAPPING_CKPT" \
  --mt-path    "$MT_PATH" \
  --llm-path   "$LLM_PATH" \
  --epochs     3 \
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
  --wandb-run-name "stage2-wit-$LANG" \
  --local-files-only \
  "${RESUME_ARGS[@]}"

echo "=== Done ==="
date
