#!/bin/bash
# REPLAY ARM of the pooled M2-ALIGN pipeline (Santiago's D6 text replay,
# ported). Separate from launch.sh, which is untouched. Stage 1 and Stage 2
# are NOT retrained: replay only changes Stage 3b, which warm-starts from the
# existing pooled Stage 2 checkpoint (Stage2/outputs/pooled).
#
#   load_replay_data ........ 5k Stage-1 rows + 5k MetaMathQA (NLLB-translated) per lang
#      `-- smoke ............. (SMOKE_FIRST=1) 24 VQA steps + replay, catches OOM / shape bugs
#             `-- stage3b_pooled_replay ... 30k VQA x 12 + replay, 3 epochs
#                    |-- eval_<BM>_<L> ....... xGQA / CVQA on the replay checkpoint
#                    `-- text_<TASK>_<L> ..... MGSM / MSVAMP on the replay checkpoint
#
# All arrows are afterok. Outputs never collide with the no-replay arm:
# checkpoint Stage3/outputs/pooled_replay, results Stage3/results/pooled_replay_stage3b.
# Compare against the no-replay pooled Stage 3b and Baseline/evaluate.py (mgsm/msvamp).
#
# Usage:
#   DRY_RUN=1 ./launch_replay.sh
#   ./launch_replay.sh
#   SKIP_LOAD=1 SMOKE_FIRST=0 ./launch_replay.sh    # replay files already loaded
#   TEXT_LANGS="bn de ru zh en es fr ja sw th" EVAL_BENCHMARKS="xgqa cvqa_mcq" ./launch_replay.sh

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
cd "$PROJECT_ROOT"
DATA_ROOT="$SCRATCH/M2-ALIGN"

DRY_RUN="${DRY_RUN:-0}"
ACCOUNT="${ACCOUNT:-def-annielee}"
ARM="${ARM:-replay}"
SKIP_LOAD="${SKIP_LOAD:-0}"
SMOKE_FIRST="${SMOKE_FIRST:-1}"
INCLUDE_ENGLISH="${INCLUDE_ENGLISH:-1}"
REPLAY_EVERY="${REPLAY_EVERY:-3}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-xgqa cvqa_mcq}"
XGQA_LANGS="${XGQA_LANGS:-bn de id ko pt ru zh}"
CVQA_LANGS="${CVQA_LANGS:-bn ga id jv ko mn pt ru si zh}"
TEXT_TASKS="${TEXT_TASKS:-mgsm msvamp}"
TEXT_LANGS="${TEXT_LANGS:-bn de ru zh en}"

LOAD="Stage3/job-scripts/load_replay_data.sh"
TRAIN="Stage3/job-scripts/train_pooled_replay.sh"
EV="Stage3/job-scripts/evaluate_pooled_replay.sh"

langs_for() {
  case "$1" in
    xgqa) echo "$XGQA_LANGS" ;;
    cvqa|cvqa_generation|cvqa_mcq) echo "$CVQA_LANGS" ;;
    *) echo "ERROR: unsupported benchmark: $1" >&2; exit 1 ;;
  esac
}
for BM in $EVAL_BENCHMARKS; do langs_for "$BM" >/dev/null || exit 1; done

command -v sbatch >/dev/null || { echo "ERROR: sbatch not on PATH"; exit 1; }
for f in "$LOAD" "$TRAIN" "$EV"; do
  [ -f "$f" ] || { echo "ERROR: missing job script: $f"; exit 1; }
done
if [ "$DRY_RUN" != 1 ]; then
  [ -f "$DATA_ROOT/Stage2/outputs/pooled/pytorch_model.bin" ] \
    || { echo "ERROR: pooled Stage 2 checkpoint missing (warm start)"; exit 1; }
fi

RUN_DIR="$DATA_ROOT/pipeline_runs/pooled_${ARM}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
JOBS_TSV="$RUN_DIR/jobs.tsv"
EVAL_TSV="$RUN_DIR/eval_jobs.tsv"     # phase<TAB>benchmark<TAB>lang<TAB>jobid<TAB>logdir
printf 'phase\tjob\tjobid\n' > "$JOBS_TSV"
: > "$EVAL_TSV"

# submit <jobname> <afterok-dep|""> <script> <export-kv-csv|""> [extra-sbatch-opts...]
submit() {
  local jobname="$1" dep="$2" script="$3" exports="$4"; shift 4
  local args=(--parsable --account="$ACCOUNT" --job-name="$jobname" --kill-on-invalid-dep=yes)
  if [ -n "$exports" ]; then args+=(--export="ALL,$exports"); else args+=(--export=ALL); fi
  [ -n "$dep" ] && args+=(--dependency="afterok:$dep")
  args+=("$@")
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry-run] sbatch ${args[*]} $script" >&2
    echo "DRYRUN-$jobname"
    return
  fi
  local out; out="$(sbatch "${args[@]}" "$script")"
  echo "${out%%;*}"
}

echo "run dir : $RUN_DIR"
DEP=""
if [ "$SKIP_LOAD" != 1 ]; then
  DEP="$(submit "${ARM}_load_replay_data" "" "$LOAD" "INCLUDE_ENGLISH=$INCLUDE_ENGLISH")"
  printf 'load\tload_replay_data\t%s\n' "$DEP" >> "$JOBS_TSV"
  echo "load    -> $DEP"
fi
if [ "$SMOKE_FIRST" = 1 ]; then
  DEP="$(submit "${ARM}_stage3b_smoke" "$DEP" "$TRAIN" \
         "SMOKE=1,INCLUDE_ENGLISH=$INCLUDE_ENGLISH,REPLAY_EVERY=$REPLAY_EVERY" --time=1:30:00)"
  printf 'smoke\tstage3b_smoke\t%s\n' "$DEP" >> "$JOBS_TSV"
  echo "smoke   -> $DEP"
fi
S3_ID="$(submit "${ARM}_stage3b_pooled" "$DEP" "$TRAIN" \
         "INCLUDE_ENGLISH=$INCLUDE_ENGLISH,REPLAY_EVERY=$REPLAY_EVERY")"
printf 'stage3b\tstage3b_pooled_replay\t%s\n' "$S3_ID" >> "$JOBS_TSV"
echo "stage3b -> $S3_ID"

ALL_EVAL=()
for BM in $EVAL_BENCHMARKS; do
  for L in $(langs_for "$BM"); do
    id="$(submit "${ARM}_eval_${BM}_$L" "$S3_ID" "$EV" "BENCHMARK=$BM,LANG=$L,CHECKPOINT_STAGE=stage3b")"
    ALL_EVAL+=("$id")
    printf 'pooled_replay\t%s\t%s\t%s\t%s\n' "$BM" "$L" "$id" "$DATA_ROOT/Stage3/logs" >> "$EVAL_TSV"
  done
done
for T in $TEXT_TASKS; do
  for L in $TEXT_LANGS; do
    id="$(submit "${ARM}_text_${T}_$L" "$S3_ID" "$EV" "BENCHMARK=$T,LANG=$L,CHECKPOINT_STAGE=stage3b")"
    ALL_EVAL+=("$id")
    printf 'pooled_replay\t%s\t%s\t%s\t%s\n' "$T" "$L" "$id" "$DATA_ROOT/Stage3/logs" >> "$EVAL_TSV"
  done
done
echo "evals   -> ${#ALL_EVAL[@]} jobs (after $S3_ID), map in $EVAL_TSV"

# Scrape every eval accuracy into one table once all evals finish.
COLLECT_SH="$RUN_DIR/collect.sh"
cat > "$COLLECT_SH" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=${ARM}_collect
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=$RUN_DIR/collect_%j.log
set -euo pipefail
OUT="$RUN_DIR/results.tsv"
printf 'phase\tbenchmark\tlang\taccuracy\tn\tjobid\n' > "\$OUT"
while IFS=\$'\t' read -r phase bm lang jid logdir; do
  [ -n "\$jid" ] || continue
  f="\$(ls -t "\$logdir"/*_"\$jid".log 2>/dev/null | head -1 || true)"
  if [ -z "\$f" ]; then printf '%s\t%s\t%s\t%s\t\t%s\n' "\$phase" "\$bm" "\$lang" NO_LOG "\$jid" >> "\$OUT"; continue; fi
  line="\$(grep -oE 'accuracy=[0-9.]+% \(n=[0-9]+\)' "\$f" | tail -1 || true)"
  acc="\$(printf '%s' "\$line" | grep -oE '[0-9.]+' | head -1 || true)"
  n="\$(printf '%s' "\$line" | grep -oE 'n=[0-9]+' | grep -oE '[0-9]+' || true)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "\$phase" "\$bm" "\$lang" "\${acc:-MISSING}" "\${n:-}" "\$jid" >> "\$OUT"
done < "$EVAL_TSV"
sort -k2,2 -k3,3 -o "\$OUT" "\$OUT"
column -t -s\$'\t' "\$OUT"
EOF
chmod +x "$COLLECT_SH"
if [ "$DRY_RUN" = 1 ]; then
  echo "  [dry-run] sbatch --dependency=afterany:<${#ALL_EVAL[@]} evals> $COLLECT_SH"
  echo "DRY RUN -- nothing was submitted."
else
  cid="$(IFS=:; sbatch --parsable --dependency="afterany:${ALL_EVAL[*]}" --kill-on-invalid-dep=yes "$COLLECT_SH")"
  echo "collect -> ${cid%%;*}   (writes $RUN_DIR/results.tsv)"
fi
