#!/bin/bash
#SBATCH --job-name=stage3b_load_replay_data
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage3/logs/stage3b_load_replay_data_%x_%j.log

# Text replay pool for Stage3/job-scripts/train_pooled_replay.sh (Santiago's
# D6 replay, ported). Per pooled language (bn ru de zh pt id ko jv mn si ga):
#   Stage3/data/replay/translation_<L>.jsonl  N_TRANSLATION Stage 1 rows
#   Stage3/data/replay/math_<L>.jsonl         N_MATH MetaMathQA GSM_* problems,
#                                             question NLLB-translated into <L>
# plus math_en.jsonl (untranslated) when INCLUDE_ENGLISH=1, mirroring the
# English rows in the Stage 3b VQA pool.
#
# Math questions are translated with NLLB-3.3B, the model that translated
# the Stage 3 VQA data (load_translated_data.sh). Idempotent: existing files
# are skipped, so a timed-out job can simply be resubmitted (OVERWRITE=1 to
# rebuild).
#
# Usage:
#   sbatch Stage3/job-scripts/load_replay_data.sh
#   N_MATH=10000 OVERWRITE=1 sbatch Stage3/job-scripts/load_replay_data.sh

set -euo pipefail

LANG_CODES="bn ru de zh pt id ko jv mn si ga"
N_TRANSLATION="${N_TRANSLATION:-5000}"
N_MATH="${N_MATH:-5000}"
INCLUDE_ENGLISH="${INCLUDE_ENGLISH:-1}"
OVERWRITE="${OVERWRITE:-0}"

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
DATA_ROOT="$SCRATCH/M2-ALIGN"
NLLB_MODEL="$HOME/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/1a07f7d195896b2114afcb79b7b57ab512e7b43e"
STAGE1_DATA="$DATA_ROOT/Stage1/data"
OUT_DIR="$DATA_ROOT/Stage3/data/replay"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  LANGS=$LANG_CODES  N_TRANSLATION=$N_TRANSLATION  N_MATH=$N_MATH  INCLUDE_ENGLISH=$INCLUDE_ENGLISH"
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
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "NLLB_MODEL=$NLLB_MODEL"
echo "OUT_DIR=$OUT_DIR"
echo

if [ ! -d "$NLLB_MODEL" ]; then
  echo "ERROR: NLLB snapshot not found: $NLLB_MODEL"
  exit 1
fi

EXTRA_ARGS=(--local-files-only)
[ "$INCLUDE_ENGLISH" = 1 ] && EXTRA_ARGS+=(--include-english-math)
[ "$OVERWRITE" = 1 ] && EXTRA_ARGS+=(--overwrite)

echo "=== Build replay pool ==="
cd "$PROJECT_ROOT"
python -u Stage3/load_replay_data.py \
  --langs "${LANG_CODES// /,}" \
  --stage1-dir "$STAGE1_DATA" \
  --output-dir "$OUT_DIR" \
  --mt-path "$NLLB_MODEL" \
  --n-translation "$N_TRANSLATION" \
  --n-math "$N_MATH" \
  "${EXTRA_ARGS[@]}"

echo
wc -l "$OUT_DIR"/*.jsonl
echo "=== Done ==="
date
