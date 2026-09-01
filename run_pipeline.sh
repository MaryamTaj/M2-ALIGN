#!/bin/bash
# Full M2-ALIGN pipeline as chained SLURM jobs, in order:
#   Stage 1 train (EPOCH_NUM epochs) -> Stage 2 train -> Stage 3b train -> eval (xGQA + CVQA)
#
# Each language advances through the four phases in order via `--dependency=afterok`
# (a phase only starts once the previous phase for that language finished with
# exit 0). Languages run in parallel with each other -- that is the per-language
# layout the individual job scripts were built for.
#
# Set BARRIER=1 for global stage barriers instead: every language's Stage N waits
# for ALL languages' Stage N-1 (a stricter, slower reading of "sequential").
#
# Usage:
#   ./run_pipeline.sh                 # submit everything (14 langs, 3 Stage-1 epochs)
#   DRY_RUN=1 ./run_pipeline.sh       # print the sbatch commands, submit nothing
#   LANGS="id mn zh" ./run_pipeline.sh
#   EPOCH_NUM=3 BARRIER=1 ./run_pipeline.sh
#
# Nothing here needs internet; it only calls sbatch. Data/checkpoints are assumed
# to already be on $SCRATCH (see README for the download/transfer steps).

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
cd "$PROJECT_ROOT"

DATA_ROOT="$SCRATCH/M2-ALIGN"

# ---------------------------------------------------------------------------
# Config (all env-overridable)
# ---------------------------------------------------------------------------
KNOWN_LANGS="bn ru de zh pt id ko jv mn si ga am ig om"
LANGS="${LANGS:-$KNOWN_LANGS}"
EPOCH_NUM="${EPOCH_NUM:-3}"          # Stage 1 epochs
BARRIER="${BARRIER:-0}"             # 1 = global stage barriers, 0 = per-language chains
DRY_RUN="${DRY_RUN:-0}"
COLLECT="${COLLECT:-1}"             # 1 = append a final job that scrapes eval accuracies
ACCOUNT="${ACCOUNT:-def-annielee}"

# Languages that have eval data for each benchmark (see $DATA_ROOT/Stage3/data/).
XGQA_LANGS="${XGQA_LANGS:-bn de id ko pt ru zh}"
CVQA_LANGS="${CVQA_LANGS:-bn ga id jv ko mn pt ru si zh am ig om}"

S1="Stage1/job-scripts/train.sh"
S2="Stage2/job-scripts/train.sh"
S3="Stage3/job-scripts/train.sh"
EV="Stage3/job-scripts/evaluate.sh"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$DATA_ROOT/pipeline_runs/$RUN_TAG"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v sbatch >/dev/null || { echo "ERROR: sbatch not on PATH"; exit 1; }
for f in "$S1" "$S2" "$S3" "$EV"; do
  [ -f "$f" ] || { echo "ERROR: missing job script: $f"; exit 1; }
done
for L in $LANGS; do
  case " $KNOWN_LANGS " in
    *" $L "*) ;;
    *) echo "ERROR: unknown language '$L' (known: $KNOWN_LANGS)"; exit 1 ;;
  esac
done

mkdir -p "$RUN_DIR"
JOBS_TSV="$RUN_DIR/jobs.tsv"
EVAL_TSV="$RUN_DIR/eval_jobs.tsv"
printf 'lang\tstage1\tstage2\tstage3\txgqa\tcvqa\n' > "$JOBS_TSV"
: > "$EVAL_TSV"

echo "run tag        : $RUN_TAG"
echo "run dir        : $RUN_DIR"
echo "languages      : $LANGS"
echo "Stage-1 epochs : $EPOCH_NUM"
echo "mode           : $([ "$BARRIER" = 1 ] && echo 'global stage barriers' || echo 'per-language chains')"
echo "dry run        : $DRY_RUN"
echo

join_colon() { local IFS=:; echo "$*"; }

# submit <jobname> <dep-spec|""> <script> <export-kv-csv>  -> echoes job id
submit() {
  local jobname="$1" dep="$2" script="$3" exports="$4"
  local args=(--parsable --account="$ACCOUNT" --job-name="$jobname"
              --export="ALL,$exports" --kill-on-invalid-dep=yes)
  [ -n "$dep" ] && args+=(--dependency="afterok:$dep")
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry-run] sbatch ${args[*]} $script" >&2
    echo "DRYRUN-$jobname"
    return
  fi
  local out; out="$(sbatch "${args[@]}" "$script")"
  echo "${out%%;*}"          # strip ";cluster" suffix if the site adds one
}

has_data() { [ "$DRY_RUN" = 1 ] || [ -f "$1" ]; }

declare -A S1 S2ID S3ID
ALL_S1=(); ALL_S2=(); ALL_S3=(); ALL_EVAL=()

# ---------------------------------------------------------------------------
# Phase 1: Stage 1 (no dependencies)
# ---------------------------------------------------------------------------
echo "== Stage 1 =="
for L in $LANGS; do
  id="$(submit "stage1_train_$L" "" "$S1" "LANG=$L,EPOCH_NUM=$EPOCH_NUM")"
  S1[$L]="$id"; ALL_S1+=("$id")
  echo "  $L -> $id"
done
S1_BARRIER="$(join_colon "${ALL_S1[@]}")"
echo

# ---------------------------------------------------------------------------
# Phase 2: Stage 2  (dep: this lang's Stage 1, or all of Stage 1 in BARRIER mode)
# ---------------------------------------------------------------------------
echo "== Stage 2 =="
for L in $LANGS; do
  if ! has_data "$DATA_ROOT/Stage2/data/$L/wit_pairs.jsonl"; then
    echo "  $L -> SKIP (no Stage2/data/$L/wit_pairs.jsonl); chain stops here for $L"
    S2ID[$L]=""; continue
  fi
  dep="$([ "$BARRIER" = 1 ] && echo "$S1_BARRIER" || echo "${S1[$L]}")"
  id="$(submit "stage2_train_$L" "$dep" "$S2" "LANG=$L")"
  S2ID[$L]="$id"; ALL_S2+=("$id")
  echo "  $L -> $id   (after $dep)"
done
S2_BARRIER="$(join_colon "${ALL_S2[@]}")"
echo

# ---------------------------------------------------------------------------
# Phase 3: Stage 3b  (dep: this lang's Stage 2)
# ---------------------------------------------------------------------------
echo "== Stage 3b =="
for L in $LANGS; do
  [ -n "${S2ID[$L]:-}" ] || { echo "  $L -> SKIP (no Stage 2 job)"; S3ID[$L]=""; continue; }
  if ! has_data "$DATA_ROOT/Stage3/data/$L.jsonl"; then
    echo "  $L -> SKIP (no Stage3/data/$L.jsonl)"; S3ID[$L]=""; continue
  fi
  dep="$([ "$BARRIER" = 1 ] && echo "$S2_BARRIER" || echo "${S2ID[$L]}")"
  id="$(submit "stage3b_train_$L" "$dep" "$S3" "LANG=$L")"
  S3ID[$L]="$id"; ALL_S3+=("$id")
  echo "  $L -> $id   (after $dep)"
done
S3_BARRIER="$(join_colon "${ALL_S3[@]}")"
echo

# ---------------------------------------------------------------------------
# Phase 4: evaluation  (dep: this lang's Stage 3b)
# ---------------------------------------------------------------------------
echo "== Evaluation (xGQA + CVQA) =="
declare -A XID CID
for L in $LANGS; do
  [ -n "${S3ID[$L]:-}" ] || { echo "  $L -> SKIP eval (no Stage 3 job)"; continue; }
  dep="$([ "$BARRIER" = 1 ] && echo "$S3_BARRIER" || echo "${S3ID[$L]}")"

  for BM in xgqa cvqa; do
    eval "cov=\$${BM^^}_LANGS"
    case " $cov " in *" $L "*) ;; *) continue ;; esac
    if ! has_data "$DATA_ROOT/Stage3/data/$BM/$L.jsonl"; then
      echo "  $BM/$L -> SKIP (no eval data)"; continue
    fi
    id="$(submit "stage3b_eval_${BM}_$L" "$dep" "$EV" "BENCHMARK=$BM,LANG=$L,CHECKPOINT_STAGE=stage3b")"
    ALL_EVAL+=("$id")
    printf '%s\t%s\t%s\n' "$BM" "$L" "$id" >> "$EVAL_TSV"
    [ "$BM" = xgqa ] && XID[$L]="$id" || CID[$L]="$id"
    echo "  $BM/$L -> $id   (after $dep)"
  done
done
echo

# ---------------------------------------------------------------------------
# jobs.tsv summary row per language
# ---------------------------------------------------------------------------
for L in $LANGS; do
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$L" \
    "${S1[$L]:-}" "${S2ID[$L]:-}" "${S3ID[$L]:-}" "${XID[$L]:-}" "${CID[$L]:-}" >> "$JOBS_TSV"
done

# ---------------------------------------------------------------------------
# Phase 5 (optional): scrape eval accuracies into one table once evals finish
# ---------------------------------------------------------------------------
if [ "$COLLECT" = 1 ] && [ "${#ALL_EVAL[@]}" -gt 0 ]; then
  COLLECT_SH="$RUN_DIR/collect.sh"
  cat > "$COLLECT_SH" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=pipeline_collect_$RUN_TAG
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=$RUN_DIR/collect_%j.log
set -euo pipefail
LOGDIR="$DATA_ROOT/Stage3/logs"
OUT="$RUN_DIR/results.tsv"
printf 'benchmark\tlang\taccuracy\tn\tjobid\n' > "\$OUT"
while IFS=\$'\t' read -r bm lang jid; do
  [ -n "\$jid" ] || continue
  f="\$(ls -t "\$LOGDIR"/*_"\$jid".log 2>/dev/null | head -1 || true)"
  if [ -z "\$f" ]; then printf '%s\t%s\t%s\t\t%s\n' "\$bm" "\$lang" NO_LOG "\$jid" >> "\$OUT"; continue; fi
  line="\$(grep -oE 'accuracy=[0-9.]+% \(n=[0-9]+\)' "\$f" | tail -1 || true)"
  acc="\$(printf '%s' "\$line" | grep -oE '[0-9.]+' | head -1 || true)"
  n="\$(printf '%s' "\$line" | grep -oE 'n=[0-9]+' | grep -oE '[0-9]+' || true)"
  printf '%s\t%s\t%s\t%s\t%s\n' "\$bm" "\$lang" "\${acc:-MISSING}" "\${n:-}" "\$jid" >> "\$OUT"
done < "$EVAL_TSV"
sort -k1,1 -k2,2 -o "\$OUT" "\$OUT"
echo "=== $RUN_TAG eval results ==="
column -t -s\$'\t' "\$OUT"
EOF
  chmod +x "$COLLECT_SH"
  if [ "$DRY_RUN" = 1 ]; then
    echo "== Collect =="
    echo "  [dry-run] sbatch --dependency=afterany:$(join_colon "${ALL_EVAL[@]}") $COLLECT_SH"
  else
    cid="$(sbatch --parsable --dependency="afterany:$(join_colon "${ALL_EVAL[@]}")" \
                  --kill-on-invalid-dep=yes "$COLLECT_SH")"
    echo "== Collect =="
    echo "  results job -> ${cid%%;*}   (writes $RUN_DIR/results.tsv)"
  fi
  echo
fi

echo "job map     : $JOBS_TSV"
echo "monitor     : squeue -u \$USER -o '%.18i %.30j %.10T %.12r %S'"
if [ "$DRY_RUN" = 1 ]; then
  echo
  echo "DRY RUN -- nothing was submitted."
fi
exit 0
