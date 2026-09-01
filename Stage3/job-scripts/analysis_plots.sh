#!/bin/bash
#SBATCH --job-name=stage3_analysis_plots
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryam.taj@mail.utoronto.ca
#SBATCH --output=/scratch/tajm/M2-ALIGN/Stage3/logs/stage3_analysis_plots_%x_%j.log

# Phase 4D (t-SNE by stage) + Phase 4E (layer-wise cross-lingual retrieval)
# for xGQA, WorldCuisines task1, and WorldCuisines task2. CPU-only -- these
# only load already-extracted embedding .pt files (from
# Stage3/analysis/extract_embeddings.py) and run sklearn/numpy over them, no
# model inference. Do NOT run this on a login node: t-SNE + the layer-wise
# retrieval matmuls are exactly the kind of shared-resource-hogging compute
# login nodes are meant to keep off of.
#
# All six input .pt files (xm_* for 4D, llm-layers_* for 4E, x3 stages x3
# benchmarks) must already exist under Stage3/analysis_out/ -- this job does
# not run extract_embeddings.py itself. If any are missing, see
# Stage3/job-scripts/extract_embeddings.sh.
#
# Usage:
#   sbatch analysis_plots.sh

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
DATA_ROOT="$SCRATCH/M2-ALIGN"
OUT_DIR="$DATA_ROOT/Stage3/analysis_out"

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo

echo "=== Load modules ==="
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load arrow/18.1.0
echo

echo "=== Activate virtual environment ==="
source "$SCRATCH/venvs/m2-align/bin/activate"
python -V
echo

# Matplotlib's first-import font-cache scan is painfully slow over the
# network-backed $HOME/.cache on this cluster; point it at node-local
# storage instead.
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

echo "=== Check inputs ==="
MISSING=0
for f in \
  xm_stage1_xgqa.pt xm_stage2_xgqa.pt xm_stage3b_xgqa.pt \
  xm_stage1_worldcuisines_task1.pt xm_stage2_worldcuisines_task1.pt xm_stage3b_worldcuisines_task1.pt \
  xm_stage1_worldcuisines_task2.pt xm_stage2_worldcuisines_task2.pt xm_stage3b_worldcuisines_task2.pt \
  llm-layers_stage1_xgqa.pt llm-layers_stage2_xgqa.pt llm-layers_stage3b_xgqa.pt \
  llm-layers_stage1_worldcuisines_task1.pt llm-layers_stage2_worldcuisines_task1.pt llm-layers_stage3b_worldcuisines_task1.pt \
  llm-layers_stage1_worldcuisines_task2.pt llm-layers_stage2_worldcuisines_task2.pt llm-layers_stage3b_worldcuisines_task2.pt
do
  if [ ! -f "$OUT_DIR/$f" ]; then
    echo "ERROR: missing $OUT_DIR/$f"
    MISSING=1
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "One or more input embedding files are missing -- see Stage3/job-scripts/extract_embeddings.sh"
  exit 1
fi
echo "All inputs present."
echo

cd "$PROJECT_ROOT"

echo "=== Phase 4D: t-SNE by stage ==="

echo "--- xGQA ---"
python -u Stage3/analysis/tsne_plot.py \
  --embeddings stage1="$OUT_DIR/xm_stage1_xgqa.pt" \
  --embeddings stage2="$OUT_DIR/xm_stage2_xgqa.pt" \
  --embeddings stage3b="$OUT_DIR/xm_stage3b_xgqa.pt" \
  --out "$OUT_DIR/tsne_xgqa_by_stage.png"

echo "--- WorldCuisines task1 ---"
python -u Stage3/analysis/tsne_plot.py \
  --embeddings stage1="$OUT_DIR/xm_stage1_worldcuisines_task1.pt" \
  --embeddings stage2="$OUT_DIR/xm_stage2_worldcuisines_task1.pt" \
  --embeddings stage3b="$OUT_DIR/xm_stage3b_worldcuisines_task1.pt" \
  --out "$OUT_DIR/tsne_worldcuisines_task1_by_stage.png"

echo "--- WorldCuisines task2 ---"
python -u Stage3/analysis/tsne_plot.py \
  --embeddings stage1="$OUT_DIR/xm_stage1_worldcuisines_task2.pt" \
  --embeddings stage2="$OUT_DIR/xm_stage2_worldcuisines_task2.pt" \
  --embeddings stage3b="$OUT_DIR/xm_stage3b_worldcuisines_task2.pt" \
  --out "$OUT_DIR/tsne_worldcuisines_task2_by_stage.png"

echo
echo "=== Phase 4E: layer-wise cross-lingual retrieval ==="

echo "--- xGQA ---"
python -u Stage3/analysis/layerwise_retrieval.py \
  --embeddings stage1="$OUT_DIR/llm-layers_stage1_xgqa.pt" \
  --embeddings stage2="$OUT_DIR/llm-layers_stage2_xgqa.pt" \
  --embeddings stage3b="$OUT_DIR/llm-layers_stage3b_xgqa.pt" \
  --k 1 --out "$OUT_DIR/xgqa_layerwise_retrieval.csv" \
  --plot "$OUT_DIR/xgqa_layerwise_retrieval.png"

echo "--- WorldCuisines task1 ---"
python -u Stage3/analysis/layerwise_retrieval.py \
  --embeddings stage1="$OUT_DIR/llm-layers_stage1_worldcuisines_task1.pt" \
  --embeddings stage2="$OUT_DIR/llm-layers_stage2_worldcuisines_task1.pt" \
  --embeddings stage3b="$OUT_DIR/llm-layers_stage3b_worldcuisines_task1.pt" \
  --k 1 --out "$OUT_DIR/worldcuisines_task1_layerwise_retrieval.csv" \
  --plot "$OUT_DIR/worldcuisines_task1_layerwise_retrieval.png"

echo "--- WorldCuisines task2 ---"
python -u Stage3/analysis/layerwise_retrieval.py \
  --embeddings stage1="$OUT_DIR/llm-layers_stage1_worldcuisines_task2.pt" \
  --embeddings stage2="$OUT_DIR/llm-layers_stage2_worldcuisines_task2.pt" \
  --embeddings stage3b="$OUT_DIR/llm-layers_stage3b_worldcuisines_task2.pt" \
  --k 1 --out "$OUT_DIR/worldcuisines_task2_layerwise_retrieval.csv" \
  --plot "$OUT_DIR/worldcuisines_task2_layerwise_retrieval.png"

echo
echo "=== Done ==="
date
