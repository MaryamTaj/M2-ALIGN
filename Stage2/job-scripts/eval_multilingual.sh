#!/bin/bash
#SBATCH --job-name=stage2_eval_multilingual
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/home/tajm/projects/def-annielee/tajm/M2-ALIGN/Stage2/logs/stage2_eval_multilingual_%j.log

# Evaluates the Stage 2 AugmentedMindMerger checkpoint on six multilingual
# benchmarks sequentially in a single GPU job.
#
# Language scope (matching MERLIN evaluation conventions):
#   MGSM, MSVAMP, X-CSQA, XNLI  →  Swahili only
#   AfriMGSM, AfriXNLI           →  Swahili, Wolof, Yoruba
#
# Tasks run in order; a failure in any task aborts the job. To re-run a
# single task, pass --task and --langs directly to eval_multilingual.py.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
STAGE2="$PROJECT_ROOT/Stage2"

LLM_PATH="$SCRATCH/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MT_PATH="$SCRATCH/huggingface/nllb-200-distilled-600M-full"
MAPPING_CKPT="$STAGE2/outputs/augmentation/mapping/pytorch_model.bin"

# Resolve nested snapshot directory for MT model (mirrors evaluate.sh).
if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then
      MT_PATH="$d"
      break
    fi
  done
fi

# ── Job header ──────────────────────────────────────────────────────────────
echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
nvidia-smi || true
echo

# ── Modules ─────────────────────────────────────────────────────────────────
echo "=== Load modules ==="
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.2.2
module load arrow/18.1.0
echo

# ── Virtual environment ──────────────────────────────────────────────────────
echo "=== Activate virtual environment ==="
source "$SCRATCH/venvs/m2-align/bin/activate"
python -V
echo

# ── Hugging Face offline mode ────────────────────────────────────────────────
echo "=== Hugging Face cache (offline) ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HF_HOME=$HF_HOME"
echo

if [ -f "$PROJECT_ROOT/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.tokens"
  echo "Loaded .tokens file"
else
  echo "WARNING: .tokens file not found"
fi
echo

# ── Pre-flight checks ────────────────────────────────────────────────────────
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

cd "$STAGE2"

# ── Shared flags ─────────────────────────────────────────────────────────────
COMMON_ARGS=(
  --llm-path      "$LLM_PATH"
  --mt-path       "$MT_PATH"
  --mapping-ckpt  "$MAPPING_CKPT"
  --max-mt-seq-len  256
  --max-llm-seq-len 2048
  --max-gen-len     512
  --local-files-only
)

# ── 1. MGSM (Swahili) ───────────────────────────────────────────────────────
echo "=== [1/6] MGSM — Swahili ==="
date
python -u eval_multilingual.py \
  --task  mgsm \
  --langs sw \
  "${COMMON_ARGS[@]}"
echo

# ── 2. MSVAMP (Swahili) ──────────────────────────────────────────────────────
echo "=== [2/6] MSVAMP — Swahili ==="
date
python -u eval_multilingual.py \
  --task  msvamp \
  --langs sw \
  "${COMMON_ARGS[@]}"
echo

# ── 3. X-CSQA (Swahili) ──────────────────────────────────────────────────────
echo "=== [3/6] X-CSQA — Swahili ==="
date
python -u eval_multilingual.py \
  --task  x-csqa \
  --langs sw \
  "${COMMON_ARGS[@]}"
echo

# ── 4. XNLI (Swahili) ────────────────────────────────────────────────────────
echo "=== [4/6] XNLI — Swahili ==="
date
python -u eval_multilingual.py \
  --task  xnli \
  --langs sw \
  "${COMMON_ARGS[@]}"
echo

# ── 5. AfriMGSM (Swahili, Wolof, Yoruba) ────────────────────────────────────
echo "=== [5/6] AfriMGSM — Swahili, Wolof, Yoruba ==="
date
python -u eval_multilingual.py \
  --task  afrimgsm \
  --langs sw,wo,yo \
  "${COMMON_ARGS[@]}"
echo

# ── 6. AfriXNLI (Swahili, Wolof, Yoruba) ────────────────────────────────────
echo "=== [6/6] AfriXNLI — Swahili, Wolof, Yoruba ==="
date
python -u eval_multilingual.py \
  --task  afrixnli \
  --langs sw,wo,yo \
  "${COMMON_ARGS[@]}"
echo

echo "=== All benchmarks complete ==="
date
