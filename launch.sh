#!/bin/bash
# Pooled (MindMerger-mirror) M2-ALIGN pipeline as chained SLURM jobs.
#
# Topology -- one job per training stage (languages are POOLED, not per-language):
#
#   stage0_sft ................ English-GQA SFT of Qwen3-VL  -> the MonoVQA checkpoint
#     |
#     |-- monovqa_eval_{xgqa,cvqa}_<L> ... condition 2: MonoVQA zero-shot, per eval lang
#     |
#     `-- stage1_pooled ....... mapping, 11 langs pooled (100k/lang x 3 epochs, 4 GPUs)
#            `-- stage2_pooled . visual grounding, 11 langs pooled (~11k/lang x 3 epochs)
#                   `-- stage3b_pooled ... VQA augmentation, 11 langs + English (30k x 12, 3 epochs)
#                          |-- pooled_eval_xgqa_<L> ... condition 3, per eval lang
#                          `-- pooled_eval_cvqa_<L>
#
# Every arrow is `--dependency=afterok` (downstream starts only if upstream
# exits 0). The two eval fans run in parallel once their parent finishes.
#
# The 11 trained languages are fixed inside the *_pooled.sh scripts
# (bn ru de zh pt id ko jv mn si ga). This launcher only chooses which eval
# languages to fan out to and wires the dependencies.
#
# Usage:
#   ./launch.sh                     # LoRA Stage 0, submit everything
#   DRY_RUN=1 ./launch.sh           # print sbatch commands, submit nothing
#   MODE=full ./launch.sh           # full-weight Stage 0 SFT (4 GPUs, ZeRO-3)
#   XGQA_LANGS="bn de" CVQA_LANGS="bn jv" ./launch.sh
#   SKIP_MONOVQA_EVAL=1 ./launch.sh # skip condition-2 eval fan only
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
MODE="${MODE:-lora}"                # Stage 0: lora (1 GPU) | full (4 GPUs, ZeRO-3)
EPOCHS_S0="${EPOCHS_S0:-3}"         # Stage 0 epochs (MetaMath was 3)
DRY_RUN="${DRY_RUN:-0}"
COLLECT="${COLLECT:-1}"             # 1 = append a final job that scrapes eval accuracies
SKIP_MONOVQA_EVAL="${SKIP_MONOVQA_EVAL:-0}"
INCLUDE_ENGLISH="${INCLUDE_ENGLISH:-1}"   # 1 = English GQA rows in the Stage 3b pool (MindMerger-faithful); 0 = ablation
ACCOUNT="${ACCOUNT:-def-annielee}"

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

S0="Stage0/job-scripts/train.sh"
S0_EV="Stage0/job-scripts/evaluate.sh"
S1="Stage1/job-scripts/train_pooled.sh"
S2="Stage2/job-scripts/train_pooled.sh"
S3="Stage3/job-scripts/train_pooled.sh"
EV="Stage3/job-scripts/evaluate_pooled.sh"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$DATA_ROOT/pipeline_runs/pooled_$RUN_TAG"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v sbatch >/dev/null || { echo "ERROR: sbatch not on PATH"; exit 1; }
for f in "$S0" "$S0_EV" "$S1" "$S2" "$S3" "$EV"; do
  [ -f "$f" ] || { echo "ERROR: missing job script: $f"; exit 1; }
done
case "$MODE" in lora|full) ;; *) echo "ERROR: MODE must be lora|full"; exit 1 ;; esac

missing=0
warn() { echo "  MISSING: $1"; missing=1; }
if [ "$DRY_RUN" != 1 ]; then
  echo "== Data preflight =="
  [ -f "$DATA_ROOT/Stage3/data/english.jsonl" ] || warn "Stage3/data/english.jsonl (Stage 0 + English aug)"
  [ -d "$DATA_ROOT/Stage3/data/gqa/images" ]     || warn "Stage3/data/gqa/images/"
  for c in $POOLED_CODES; do
    [ -f "$DATA_ROOT/Stage1/data/${LANG_NAME[$c]}_to_English.jsonl" ] || warn "Stage1/data/${LANG_NAME[$c]}_to_English.jsonl"
    [ -f "$DATA_ROOT/Stage2/data/$c/wit_pairs.jsonl" ]                || warn "Stage2/data/$c/wit_pairs.jsonl"
    [ -f "$DATA_ROOT/Stage3/data/$c.jsonl" ]                          || warn "Stage3/data/$c.jsonl"
  done
  for L in $XGQA_LANGS; do [ -f "$DATA_ROOT/Stage3/data/xgqa/$L.jsonl" ] || warn "Stage3/data/xgqa/$L.jsonl"; done
  for L in $CVQA_LANGS; do [ -f "$DATA_ROOT/Stage3/data/cvqa/$L.jsonl" ] || warn "Stage3/data/cvqa/$L.jsonl"; done
  if [ "$missing" = 1 ]; then
    echo "ERROR: inputs above are missing -- the pooled chain would fail. Fix or use DRY_RUN=1." >&2
    exit 1
  fi
  echo "  ok"
  echo
fi

mkdir -p "$RUN_DIR"
JOBS_TSV="$RUN_DIR/jobs.tsv"
EVAL_TSV="$RUN_DIR/eval_jobs.tsv"     # phase<TAB>benchmark<TAB>lang<TAB>jobid<TAB>logdir
printf 'phase\tjob\tjobid\n' > "$JOBS_TSV"
: > "$EVAL_TSV"

echo "run tag   : pooled_$RUN_TAG"
echo "run dir   : $RUN_DIR"
echo "Stage 0   : MODE=$MODE, EPOCHS=$EPOCHS_S0"
echo "xgqa eval : $XGQA_LANGS"
echo "cvqa eval : $CVQA_LANGS"
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

# ---------------------------------------------------------------------------
# Phase 0: Stage 0 SFT -> MonoVQA checkpoint
# ---------------------------------------------------------------------------
echo "== Stage 0 (MonoVQA SFT) =="
S0_OPTS=()
if [ "$MODE" = full ]; then
  S0_OPTS=(--gres=gpu:4 --cpus-per-task=48 --mem=490G --time=20:00:00)
fi
S0_ID="$(submit "stage0_sft" "" "$S0" "MODE=$MODE,EPOCHS=$EPOCHS_S0" "${S0_OPTS[@]}")"
printf 'stage0\tstage0_sft\t%s\n' "$S0_ID" >> "$JOBS_TSV"
echo "  stage0_sft -> $S0_ID"
echo

# ---------------------------------------------------------------------------
# Phase 1: MonoVQA (condition 2) eval fan -- depends only on Stage 0
# ---------------------------------------------------------------------------
ALL_EVAL=()
if [ "$SKIP_MONOVQA_EVAL" != 1 ]; then
  echo "== MonoVQA eval (condition 2, zero-shot) =="
  for BM in xgqa cvqa; do
    eval "cov=\$${BM^^}_LANGS"
    for L in $cov; do
      id="$(submit "monovqa_eval_${BM}_$L" "$S0_ID" "$S0_EV" "BENCHMARK=$BM,LANG=$L")"
      ALL_EVAL+=("$id")
      printf 'monovqa\t%s\t%s\t%s\t%s\n' "$BM" "$L" "$id" "$DATA_ROOT/Stage0/logs" >> "$EVAL_TSV"
      echo "  $BM/$L -> $id"
    done
  done
  echo
fi

# ---------------------------------------------------------------------------
# Phase 2-4: pooled Stage 1 -> Stage 2 -> Stage 3b (linear chain off Stage 0)
# ---------------------------------------------------------------------------
echo "== Pooled training chain =="
S1_ID="$(submit "stage1_pooled" "$S0_ID" "$S1" "")"
printf 'stage1\tstage1_pooled\t%s\n' "$S1_ID" >> "$JOBS_TSV"
echo "  stage1_pooled -> $S1_ID   (after $S0_ID)"

S2_ID="$(submit "stage2_pooled" "$S1_ID" "$S2" "")"
printf 'stage2\tstage2_pooled\t%s\n' "$S2_ID" >> "$JOBS_TSV"
echo "  stage2_pooled -> $S2_ID   (after $S1_ID)"

S3_ID="$(submit "stage3b_pooled" "$S2_ID" "$S3" "INCLUDE_ENGLISH=$INCLUDE_ENGLISH")"
printf 'stage3b\tstage3b_pooled\t%s\n' "$S3_ID" >> "$JOBS_TSV"
echo "  stage3b_pooled -> $S3_ID   (after $S2_ID)"
echo

# ---------------------------------------------------------------------------
# Phase 5: pooled Stage 3b eval fan (condition 3) -- depends on Stage 3b
# ---------------------------------------------------------------------------
echo "== Pooled eval (condition 3) =="
for BM in xgqa cvqa; do
  eval "cov=\$${BM^^}_LANGS"
  for L in $cov; do
    id="$(submit "pooled_eval_${BM}_$L" "$S3_ID" "$EV" "BENCHMARK=$BM,LANG=$L,CHECKPOINT_STAGE=stage3b")"
    ALL_EVAL+=("$id")
    printf 'pooled\t%s\t%s\t%s\t%s\n' "$BM" "$L" "$id" "$DATA_ROOT/Stage3/logs" >> "$EVAL_TSV"
    echo "  $BM/$L -> $id   (after $S3_ID)"
  done
done
echo

# ---------------------------------------------------------------------------
# Phase 6 (optional): scrape all eval accuracies into one table
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
  line="\$(grep -oE 'accuracy=[0-9.]+% \(n=[0-9]+\)' "\$f" | tail -1 || true)"
  acc="\$(printf '%s' "\$line" | grep -oE '[0-9.]+' | head -1 || true)"
  n="\$(printf '%s' "\$line" | grep -oE 'n=[0-9]+' | grep -oE '[0-9]+' || true)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "\$phase" "\$bm" "\$lang" "\${acc:-MISSING}" "\${n:-}" "\$jid" >> "\$OUT"
done < "$EVAL_TSV"
sort -k1,1 -k2,2 -k3,3 -o "\$OUT" "\$OUT"
echo "=== pooled_$RUN_TAG eval results (monovqa = condition 2, pooled = condition 3) ==="
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
