#!/bin/bash
# Pooled (MindMerger-mirror) M2-ALIGN pipeline as chained SLURM jobs.
#
# Topology -- one job per training stage (languages are POOLED, not per-language):
#
#   stage1_pooled ....... mapping, 11 langs pooled (100k/lang x 3 epochs, 4 GPUs)
#      `-- stage2_pooled . visual grounding, 11 langs pooled (~11k/lang x 3 epochs)
#             `-- stage3b_pooled ... VQA augmentation, 11 langs + English (30k x 12, 3 epochs)
#                    `-- pooled_eval_<BM>_<L> ... per EVAL_BENCHMARKS x eval lang
#
#   baseline_eval_<BM>_<L> ...... (BASELINE_EVAL=1) vanilla Qwen3-VL, no dependency
#   baseline_tt_eval_<BM>_<L> ... (BASELINE_EVAL=1, TRANSLATE_TEST=1) vanilla Qwen3-VL on
#                                 NLLB-translated English questions (xgqa/cvqa* only)
#
# Benchmarks (EVAL_BENCHMARKS, shared by pooled + baseline evals):
#   xgqa, cvqa (open-ended, choice log-likelihood), cvqa_mcq (closed-ended),
#   cvqa_generation, mgsm, msvamp. mgsm/msvamp are text-only and go through
#   the same eval job scripts (no image passed), fanned out over TEXT_LANGS.
#
# Qwen3-VL-8B-Instruct and NLLB stay frozen in every stage; only the mapping
# layer is trained.
#
# Every arrow is `--dependency=afterok` (downstream starts only if upstream
# exits 0). The eval fan runs in parallel once Stage 3b finishes.
#
# The 11 trained languages are fixed inside the *_pooled.sh scripts
# (bn ru de zh pt id ko jv mn si ga). This launcher only chooses which eval
# languages to fan out to and wires the dependencies.
#
# Usage:
#   ./launch.sh                     # train chain + pooled evals + baseline + translate-test
#   BASELINE_EVAL=0 ./launch.sh     # re-launch an arm without re-running the (unchanged) baselines
#   SKIP_TRAIN=1 ./launch.sh        # baseline + translate-test only
#   DRY_RUN=1 ./launch.sh           # print sbatch commands, submit nothing
#   XGQA_LANGS="bn de" CVQA_LANGS="bn jv" ./launch.sh
#   EVAL_BENCHMARKS=cvqa_mcq BASELINE_EVAL=1 ARM=arm1 ./launch.sh
#       # train chain + closed-ended CVQA eval, plus native-res Baseline cvqa_mcq
#   SKIP_TRAIN=1 BASELINE_EVAL=1 BASELINE_VISUAL_PIXELS=65536 EVAL_BENCHMARKS=cvqa_mcq ARM=arm2 ./launch.sh
#       # Baseline only, at Stage3's 256*256 pixel budget
#   STAGE3_JOB=3188207 EVAL_BENCHMARKS=xgqa ARM=arm1 ./launch.sh
#       # pooled eval only, afterok on an already-queued Stage 3b job (no retraining)
#
# Nothing here needs internet; it only calls sbatch. Data/checkpoints are
# assumed to already be on $SCRATCH (see README for download/transfer).

set -euo pipefail

PROJECT_ROOT="$HOME/projects/def-annielee/tajm/M2-ALIGN"
cd "$PROJECT_ROOT"

DATA_ROOT="$SCRATCH/M2-ALIGN"

# ---------------------------------------------------------------------------
# Config (all env-overridable)
# ---------------------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"
COLLECT="${COLLECT:-1}"             # 1 = append a final job that scrapes eval accuracies
INCLUDE_ENGLISH="${INCLUDE_ENGLISH:-1}"   # 1 = English GQA rows in the Stage 3b pool (MindMerger-faithful); 0 = ablation
ACCOUNT="${ACCOUNT:-def-annielee}"
ARM="${ARM:-}"                      # optional label, prefixed to the run dir + job names
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-xgqa cvqa cvqa_mcq mgsm msvamp}"   # any of: xgqa cvqa cvqa_generation cvqa_mcq mgsm msvamp
SKIP_TRAIN="${SKIP_TRAIN:-0}"       # 1 = no Stage 1/2/3b chain and no pooled eval
STAGE3_JOB="${STAGE3_JOB:-}"        # existing stage3b_pooled job id: submit no training, hang pooled eval off it
BASELINE_EVAL="${BASELINE_EVAL:-1}" # 1 = also submit vanilla Qwen3-VL evals (Baseline/job-scripts/evaluate.sh)
TRANSLATE_TEST="${TRANSLATE_TEST:-1}" # 1 = also baseline translate-test on xgqa/cvqa* (needs BASELINE_EVAL=1)
BASELINE_VISUAL_PIXELS="${BASELINE_VISUAL_PIXELS:-}"  # empty = native resolution; 65536 = Stage3's budget

# The 11 pooled training languages -- must match LANG_CODES/LANG_NAMES inside
# the *_pooled.sh scripts. Used here only for the data preflight.
POOLED_CODES="bn ru de zh pt id ko jv mn si ga"
declare -A LANG_NAME=(
  [bn]=Bengali [ru]=Russian [de]=German [zh]=Chinese [pt]=Portuguese
  [id]=Indonesian [ko]=Korean [jv]=Javanese [mn]=Mongolian [si]=Sinhalese [ga]=Irish
)

# Eval fan-out (subset of the trained langs that have eval data for each set).
XGQA_LANGS="${XGQA_LANGS:-bn de id ko pt ru zh}"
CVQA_LANGS="${CVQA_LANGS:-bn ga id jv ko mn pt ru si zh}"
TEXT_LANGS="${TEXT_LANGS:-bn de ru zh en}"   # mgsm/msvamp: trained langs they cover + English

S1="Stage1/job-scripts/train_pooled.sh"
S2="Stage2/job-scripts/train_pooled.sh"
S3="Stage3/job-scripts/train_pooled.sh"
EV="Stage3/job-scripts/evaluate_pooled.sh"
BEV="Baseline/job-scripts/evaluate.sh"

# Eval languages for a benchmark (cvqa variants share the CVQA eval files).
langs_for() {
  case "$1" in
    xgqa) echo "$XGQA_LANGS" ;;
    cvqa|cvqa_generation|cvqa_mcq) echo "$CVQA_LANGS" ;;
    mgsm|msvamp) echo "$TEXT_LANGS" ;;
    *) echo "ERROR: unsupported benchmark in EVAL_BENCHMARKS: $1" >&2; exit 1 ;;
  esac
}
data_dir_for() { case "$1" in cvqa*) echo cvqa ;; *) echo "$1" ;; esac; }
is_vqa() { case "$1" in mgsm|msvamp) return 1 ;; *) return 0 ;; esac; }

for BM in $EVAL_BENCHMARKS; do langs_for "$BM" >/dev/null || exit 1; done

RUN_TAG="${ARM:+${ARM}_}$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$DATA_ROOT/pipeline_runs/pooled_$RUN_TAG"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v sbatch >/dev/null || { echo "ERROR: sbatch not on PATH"; exit 1; }
for f in "$S1" "$S2" "$S3" "$EV" "$BEV"; do
  [ -f "$f" ] || { echo "ERROR: missing job script: $f"; exit 1; }
done

missing=0
warn() { echo "  MISSING: $1"; missing=1; }
if [ "$DRY_RUN" != 1 ]; then
  echo "== Data preflight =="
  if [ "$SKIP_TRAIN" != 1 ] && [ -z "$STAGE3_JOB" ]; then
    [ "$INCLUDE_ENGLISH" != 1 ] || [ -f "$DATA_ROOT/Stage3/data/en.jsonl" ] || warn "Stage3/data/en.jsonl (English rows in Stage 3b pool)"
    [ -d "$DATA_ROOT/Stage3/data/gqa/images" ]     || warn "Stage3/data/gqa/images/"
    for c in $POOLED_CODES; do
      [ -f "$DATA_ROOT/Stage1/data/${LANG_NAME[$c]}_to_English.jsonl" ] || warn "Stage1/data/${LANG_NAME[$c]}_to_English.jsonl"
      [ -f "$DATA_ROOT/Stage2/data/$c/wit_pairs.jsonl" ]                || warn "Stage2/data/$c/wit_pairs.jsonl"
      [ -f "$DATA_ROOT/Stage3/data/$c.jsonl" ]                          || warn "Stage3/data/$c.jsonl"
    done
  fi
  for BM in $EVAL_BENCHMARKS; do
    for L in $(langs_for "$BM"); do
      d="$(data_dir_for "$BM")"
      [ -f "$DATA_ROOT/Stage3/data/$d/$L.jsonl" ] || warn "Stage3/data/$d/$L.jsonl"
      if [ "$BASELINE_EVAL" = 1 ] && [ "$TRANSLATE_TEST" = 1 ] && is_vqa "$BM"; then
        [ -f "$DATA_ROOT/Baseline/data/translate_test/$d/$L.jsonl" ] || warn "Baseline/data/translate_test/$d/$L.jsonl"
      fi
    done
  done
  if [ "$missing" = 1 ]; then
    echo "ERROR: inputs above are missing -- the pooled chain would fail. Fix or use DRY_RUN=1." >&2
    exit 1
  fi
  echo "  ok"
  echo
fi

mkdir -p "$RUN_DIR" "$DATA_ROOT/Baseline/logs"
JOBS_TSV="$RUN_DIR/jobs.tsv"
EVAL_TSV="$RUN_DIR/eval_jobs.tsv"     # phase<TAB>benchmark<TAB>lang<TAB>jobid<TAB>logdir
printf 'phase\tjob\tjobid\n' > "$JOBS_TSV"
: > "$EVAL_TSV"

echo "run tag   : pooled_$RUN_TAG"
echo "run dir   : $RUN_DIR"
echo "benchmarks: $EVAL_BENCHMARKS"
echo "xgqa langs: $XGQA_LANGS"
echo "cvqa langs: $CVQA_LANGS"
echo "train     : $(if [ "$SKIP_TRAIN" = 1 ]; then echo skipped; elif [ -n "$STAGE3_JOB" ]; then echo "reuse stage3b job $STAGE3_JOB"; else echo 'stage1 -> stage2 -> stage3b'; fi)"
echo "text langs: $TEXT_LANGS"
echo "baseline  : $([ "$BASELINE_EVAL" = 1 ] && echo "on (visual_pixels=${BASELINE_VISUAL_PIXELS:-native}, translate-test=$TRANSLATE_TEST)" || echo off)"
echo "dry run   : $DRY_RUN"
echo

join_colon() { local IFS=:; echo "$*"; }

# submit <jobname> <dep-spec|""> <script> <export-kv-csv|""> [extra-sbatch-opts...]
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

ALL_EVAL=()

if [ "$SKIP_TRAIN" != 1 ]; then
if [ -n "$STAGE3_JOB" ]; then
  S3_ID="$STAGE3_JOB"
  printf 'stage3b\treused\t%s\n' "$S3_ID" >> "$JOBS_TSV"
  echo "== Reusing Stage 3b job $S3_ID (no training submitted) =="
  echo
else
# ---------------------------------------------------------------------------
# Phase 1-3: pooled Stage 1 -> Stage 2 -> Stage 3b (linear chain)
# ---------------------------------------------------------------------------
echo "== Pooled training chain =="
S1_ID="$(submit "${ARM:+${ARM}_}stage1_pooled" "" "$S1" "")"
printf 'stage1\tstage1_pooled\t%s\n' "$S1_ID" >> "$JOBS_TSV"
echo "  stage1_pooled -> $S1_ID"

S2_ID="$(submit "${ARM:+${ARM}_}stage2_pooled" "$S1_ID" "$S2" "")"
printf 'stage2\tstage2_pooled\t%s\n' "$S2_ID" >> "$JOBS_TSV"
echo "  stage2_pooled -> $S2_ID   (after $S1_ID)"

S3_ID="$(submit "${ARM:+${ARM}_}stage3b_pooled" "$S2_ID" "$S3" "INCLUDE_ENGLISH=$INCLUDE_ENGLISH")"
printf 'stage3b\tstage3b_pooled\t%s\n' "$S3_ID" >> "$JOBS_TSV"
echo "  stage3b_pooled -> $S3_ID   (after $S2_ID)"
echo
fi

# ---------------------------------------------------------------------------
# Phase 4: pooled Stage 3b eval fan -- depends on Stage 3b
# ---------------------------------------------------------------------------
echo "== Pooled eval =="
for BM in $EVAL_BENCHMARKS; do
  for L in $(langs_for "$BM"); do
    id="$(submit "${ARM:+${ARM}_}pooled_eval_${BM}_$L" "$S3_ID" "$EV" "BENCHMARK=$BM,LANG=$L,CHECKPOINT_STAGE=stage3b")"
    ALL_EVAL+=("$id")
    printf 'pooled\t%s\t%s\t%s\t%s\n' "$BM" "$L" "$id" "$DATA_ROOT/Stage3/logs" >> "$EVAL_TSV"
    echo "  $BM/$L -> $id   (after $S3_ID)"
  done
done
echo
fi

# ---------------------------------------------------------------------------
# Baseline eval fan (optional): vanilla Qwen3-VL, no mapping, no dependency
# ---------------------------------------------------------------------------
if [ "$BASELINE_EVAL" = 1 ]; then
  echo "== Baseline eval (visual_pixels=${BASELINE_VISUAL_PIXELS:-native}) =="
  px_tag="${BASELINE_VISUAL_PIXELS:+_px$BASELINE_VISUAL_PIXELS}"
  for BM in $EVAL_BENCHMARKS; do
    for L in $(langs_for "$BM"); do
      if is_vqa "$BM"; then
        id="$(submit "${ARM:+${ARM}_}baseline_eval_${BM}_$L$px_tag" "" "$BEV" \
                     "BENCHMARK=$BM,LANG=$L,VISUAL_PIXELS=$BASELINE_VISUAL_PIXELS")"
        printf 'baseline%s\t%s\t%s\t%s\t%s\n' "$px_tag" "$BM" "$L" "$id" "$DATA_ROOT/Baseline/logs" >> "$EVAL_TSV"
      else
        id="$(submit "${ARM:+${ARM}_}baseline_eval_${BM}_$L" "" "$BEV" "BENCHMARK=$BM,LANG=$L")"
        printf 'baseline\t%s\t%s\t%s\t%s\n' "$BM" "$L" "$id" "$DATA_ROOT/Baseline/logs" >> "$EVAL_TSV"
      fi
      ALL_EVAL+=("$id")
      echo "  $BM/$L -> $id"
      if [ "$TRANSLATE_TEST" = 1 ] && is_vqa "$BM"; then
        id="$(submit "${ARM:+${ARM}_}baseline_tt_eval_${BM}_$L$px_tag" "" "$BEV" \
                     "BENCHMARK=$BM,LANG=$L,TRANSLATE_TEST=1,VISUAL_PIXELS=$BASELINE_VISUAL_PIXELS")"
        ALL_EVAL+=("$id")
        printf 'baseline_tt%s\t%s\t%s\t%s\t%s\n' "$px_tag" "$BM" "$L" "$id" "$DATA_ROOT/Baseline/logs" >> "$EVAL_TSV"
        echo "  $BM/$L (translate-test) -> $id"
      fi
    done
  done
  echo
fi

# ---------------------------------------------------------------------------
# Phase 5 (optional): scrape all eval accuracies into one table
# ---------------------------------------------------------------------------
if [ "$COLLECT" = 1 ] && [ "${#ALL_EVAL[@]}" -gt 0 ]; then
  COLLECT_SH="$RUN_DIR/collect.sh"
  cat > "$COLLECT_SH" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=pooled_collect_$RUN_TAG
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
  line="\$(grep -oE 'accuracy=[0-9.]+%( \(n=[0-9]+\))?' "\$f" | tail -1 || true)"
  acc="\$(printf '%s' "\$line" | grep -oE '[0-9.]+' | head -1 || true)"
  n="\$(printf '%s' "\$line" | grep -oE 'n=[0-9]+' | grep -oE '[0-9]+' || true)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "\$phase" "\$bm" "\$lang" "\${acc:-MISSING}" "\${n:-}" "\$jid" >> "\$OUT"
done < "$EVAL_TSV"
sort -k1,1 -k2,2 -k3,3 -o "\$OUT" "\$OUT"
echo "=== pooled_$RUN_TAG eval results ==="
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

echo "job map   : $JOBS_TSV"
echo "eval map  : $EVAL_TSV"
echo "monitor   : squeue -u \$USER -o '%.18i %.28j %.10T %.12r %S'"
if [ "$DRY_RUN" = 1 ]; then
  echo
  echo "DRY RUN -- nothing was submitted."
fi
exit 0
