#!/bin/bash
#SBATCH --job-name=stage1_train
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage1/logs/stage1_train_%x_%j.log

# Per-language Stage 1 training -- one independent checkpoint per language,
# so ru/de/zh (or any future addition) can be submitted and run in parallel
# instead of serially through one shared checkpoint. Bengali has been
# migrated onto this same per-language layout ($SCRATCH/M2-ALIGN/Stage1/outputs/bn/);
# the old outputs/M2-ALIGN/ checkpoint is no longer read by anything.
#
# Usage:
#   LANG=bn sbatch --job-name=stage1_train_bn train.sh
#   LANG=ru sbatch --job-name=stage1_train_ru train.sh
#   LANG=de sbatch --job-name=stage1_train_de train.sh
#   LANG=zh sbatch --job-name=stage1_train_zh train.sh
#   LANG=pt sbatch --job-name=stage1_train_pt train.sh
#   LANG=id sbatch --job-name=stage1_train_id train.sh
#   LANG=ko sbatch --job-name=stage1_train_ko train.sh
#   LANG=jv sbatch --job-name=stage1_train_jv train.sh
#   LANG=mn sbatch --job-name=stage1_train_mn train.sh
#   LANG=si sbatch --job-name=stage1_train_si train.sh
#   LANG=ga sbatch --job-name=stage1_train_ga train.sh
#
# Saves to $SCRATCH/M2-ALIGN/Stage1/outputs/$LANG/pytorch_model.bin

set -euo pipefail

declare -A LANG_NAMES=(
  [bn]="Bengali"
  [ru]="Russian" [de]="German" [zh]="Chinese"
  [pt]="Portuguese" [id]="Indonesian" [ko]="Korean" [jv]="Javanese"
  [mn]="Mongolian" [si]="Sinhalese" [ga]="Irish"
)
LANG="${LANG:-ru}"
LANG_NAME="${LANG_NAMES[$LANG]:-}"
if [ -z "$LANG_NAME" ]; then
  echo "ERROR: Unknown LANG=$LANG. Known codes: ${!LANG_NAMES[*]}"
  echo "       Add a new entry to LANG_NAMES here (and to Stage1/load_text.py's LANG_TO_NLLB) to support another language."
  exit 1
fi

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"

# Data/outputs/logs live on $SCRATCH; the git checkout under $HOME only holds code.
DATA_ROOT="$SCRATCH/M2-ALIGN/Stage1"

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
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  LANG=$LANG  LANG_NAME=$LANG_NAME"
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

if [ ! -d "$LLM_PATH" ]; then
  echo "ERROR: LLM snapshot path not found: $LLM_PATH"
  exit 1
fi
if [ ! -d "$MT_PATH" ]; then
  echo "ERROR: MT snapshot path not found: $MT_PATH"
  exit 1
fi
if [ ! -f "$DATA_ROOT/data/${LANG_NAME}_to_English.jsonl" ]; then
  echo "ERROR: Stage 1 data not found: $DATA_ROOT/data/${LANG_NAME}_to_English.jsonl"
  echo "       Run: python Stage1/load_text.py --languages $LANG_NAME --output_dir $DATA_ROOT/data --n_samples 100000"
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

cd "$HOME/projects/def-annielee/tajm/M2-ALIGN/Stage1"
# --master_port varies by LANG so parallel submissions for different
# languages don't collide on the same deepspeed rendezvous port.
declare -A PORTS=(
  [bn]=50010
  [ru]=50011 [de]=50012 [zh]=50013
  [pt]=50017 [id]=50018 [ko]=50019 [jv]=50020 [mn]=50021 [si]=50022 [ga]=50023
)
deepspeed --master_port "${PORTS[$LANG]}" train.py --deepspeed \
  --llm_path "$LLM_PATH" \
  --mt_path "$MT_PATH" \
  --save_name "M2-ALIGN-$LANG" \
  --output_dir "$DATA_ROOT/outputs/$LANG" \
  --stage_name mapping \
  --task nllb_corpus \
  --augmentation False \
  --nllb_data_dir "$DATA_ROOT/data" \
  --nllb_languages "$LANG_NAME" \
  --train_num 100000 \
  --val_size 3000 \
  --train_batch_size 24 \
  --train_micro_batch_size_per_gpu 1 \
  --epoch_num "${EPOCH_NUM:-3}" \
  --max_seq_len 256 \
  --max_gen_len 256 \
  --eval_batch_size 2 \
  --use_wandb True \
  --wandb_mode offline \
  --wandb_project m2-align \
  --wandb_run_name "stage1-nllb-$LANG"
