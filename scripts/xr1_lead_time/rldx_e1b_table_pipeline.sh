#!/bin/bash
# R16 (docs/rldx_e1b_test_table_plan.md): steps 3-8 as one background run, on a node with one GPU.
#
#   cd $R16_REPO
#   nohup setsid bash $R16_LOG/rldx_e1b_table_pipeline.sh $R16_ROOT/session.env >> $R16_LOG/pipeline.log 2>&1 < /dev/null &
#
# Commands, thread settings and outputs are the plan's. What differs:
#   - Step 6 runs both SAFE heads on GPU 0: the LSTM's 15 runs, then the MLP's, three at a time.
#   - Steps 5 and 6 run together. Step 7 starts after step 6, and step 8 after steps 5 and 7.
#   - Every gate is checked here. The first failed gate stops the run.
#   - Gate E is checked again from the merged sets and the export reports.
#
# Reruns:
#   - A passed step writes $R16_ROOT/pipeline/<step>.done, and a rerun skips it.
#   - A step that left output but no .done file has that output moved (not deleted) to
#     partial_<time>/ beside it, then runs again.
#   - Steps 3 and 4 run by hand are kept when their logs show that their gates passed.
set -u -o pipefail
ENV_FILE=${1:?usage: bash rldx_e1b_table_pipeline.sh <R16_ROOT>/session.env}
source "$ENV_FILE"
cd "$R16_REPO" || { echo "STOP: cannot enter $R16_REPO"; exit 1; }
STATE=$R16_ROOT/pipeline
STAMP=$(date +%Y%m%d_%H%M%S)
CPU_PID=""
GPU_PID=""
mkdir -p "$STATE"

say() { echo "[$(date -Iseconds)] $*"; }
die() {
  say "STOP: $*"
  if [ -n "$(jobs -p)" ]; then say "waiting for the step still running in the background to end"; wait; fi
  say "PIPELINE STOPPED"
  exit 1
}
passed() { [ -e "$STATE/$1.done" ] && say "$1 already passed; skipped"; }
mark() { date -Iseconds > "$STATE/$1.done"; say "PASSED $1"; }
count() { grep -c -E "$1" "$2" 2>/dev/null || true; }
aside() {
  local p d n=0 first=""
  for p in "$@"; do
    [ -e "$p" ] || continue
    d=$(dirname "$p")/partial_$STAMP
    mkdir -p "$d" && mv "$p" "$d/" || die "cannot move $p aside"
    n=$((n + 1))
    [ -n "$first" ] || first=$d/$(basename "$p")
  done
  if [ $n = 1 ]; then say "kept earlier output as $first"; fi
  if [ $n -gt 1 ]; then say "kept $n earlier outputs in partial_$STAMP/ folders, e.g. $first"; fi
}

if [ -e "$STATE/pid" ] && kill -0 "$(cat "$STATE/pid")" 2>/dev/null; then
  echo "STOP: a pipeline (PID $(cat "$STATE/pid")) is already running"
  exit 1
fi
echo $$ > "$STATE/pid"
trap 'rm -f "$STATE/pid"' EXIT

# ---- gate E (step 2's output) ------------------------------------------------------------------
gate_e() {
  passed E && return
  local set
  : > "$R16_LOG/gate_e.log"
  for set in calibration test; do
    "$CLIENT_PY" -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir "$E1B_ROOT/merged/$set" \
      > "$R16_LOG/validate_$set.log" 2>&1 || die "merged $set is not VALID (validate_$set.log)"
    grep -E "^(RoboCasa SAFE dataset|Rollouts)" "$R16_LOG/validate_$set.log" | sed "s/^/$set: /" >> "$R16_LOG/gate_e.log"
    grep -q "^RoboCasa SAFE dataset: VALID" "$R16_LOG/validate_$set.log" || die "merged $set is not VALID"
    "$CLIENT_PY" -I -c 'import json, sys
r = json.load(open(sys.argv[1]))
print("%s export: %s rollouts, complete %s" % (sys.argv[2], r.get("num_rollouts"), r.get("complete")))
sys.exit(0 if r.get("num_rollouts") == 800 and r.get("complete") is True else 1)' \
      "$E1B_ROOT/safe_export/$set/conversion_report.json" "$set" >> "$R16_LOG/gate_e.log" 2>&1 \
      || die "the $set export does not hold 800 rollouts (gate_e.log)"
  done
  cat "$R16_LOG/gate_e.log"
  mark E
}

# ---- step 3: frozen SAFE-LSTM scores of E1b (gate S) ------------------------------------------
step_s() {
  passed S && return
  local s x set f
  if grep -q "^SCORING OK" "$R16_LOG/score_safe.log" 2>/dev/null && [ -s "$R16_ROOT/scores/safe_lstm_scoring.json" ]; then
    say "step 3 was run by hand and its log ends in SCORING OK; kept"
  else
    if grep -q -E "STOP|MISMATCH" "$R16_LOG/score_safe.log" 2>/dev/null; then
      die "score_safe.log (the run by hand) reports a failed check; inspect it"
    fi
    aside "$R16_ROOT/frozen_safe" "$R16_ROOT/scores" "$R16_LOG/frozen_safe_external.log" "$R16_LOG/score_safe.log"
    say "step 3: frozen SAFE-LSTM on E1b's exports (GPU 0)"
    for s in 0 1 2; do
      for x in recorded:$RLDX_EXPORT calibration:$E1B_ROOT/safe_export/calibration test:$E1B_ROOT/safe_export/test; do
        echo "=== ${x%%:*} seed $s" >> "$R16_LOG/frozen_safe_external.log"
        CUDA_VISIBLE_DEVICES=0 "$SAFE_PY" -u -m robocasa.recovery.safe.score_seen_checkpoint_external \
          --export-dir "${x#*:}" --safe-repo "$SAFE_REPO" --training-run-dir "$RLDX_ROOT/final/lstm_seed$s" \
          --output-dir "$R16_ROOT/frozen_safe/${x%%:*}/seed$s" --group all --device cuda \
          >> "$R16_LOG/frozen_safe_external.log" 2>&1 || die "frozen SAFE-LSTM, ${x%%:*} seed $s failed (frozen_safe_external.log)"
      done
    done
    "$SAFE_PY" -I scripts/rldx_e1/score_e1.py safe --freeze-manifest "$E1_FREEZE" --calibration-dir "$RLDX_ROOT/calibration/lstm" \
      $(for s in 0 1 2; do echo --recorded-scores $s $R16_ROOT/frozen_safe/recorded/seed$s; done) \
      $(for set in calibration test; do for s in 0 1 2; do echo --e1-scores $set $s $R16_ROOT/frozen_safe/$set/seed$s; done; done) \
      --output-dir "$R16_ROOT/scores" > "$R16_LOG/score_safe.log" 2>&1 \
      || die "score_e1.py safe failed: $(tail -n 1 "$R16_LOG/score_safe.log")"
  fi
  grep -E "^(reproduce|E1 |SCORING OK)" "$R16_LOG/score_safe.log"
  grep -q "^SCORING OK" "$R16_LOG/score_safe.log" || die "score_safe.log has no SCORING OK"
  for set in calibration test; do
    [ -s "$R16_ROOT/scores/${set}_safe_lstm_predictions.jsonl" ] || die "no SAFE-LSTM scores for $set"
    f=${set}_hide_and_seek_predictions.jsonl
    [ -e "$R16_ROOT/scores/$f" ] || cp "$E1B_ROOT/scores/$f" "$R16_ROOT/scores/"
    cmp -s "$E1B_ROOT/scores/$f" "$R16_ROOT/scores/$f" || die "the copy of E1b's $f differs"
    say "Hide-and-Seek $set copied"
  done
  mark S
}

# ---- step 4: ID lists --------------------------------------------------------------------------
step_ids() {
  passed ids && return
  if grep -q "^IDS OK" "$R16_LOG/ids.log" 2>/dev/null && [ -s "$R16_ROOT/ids/test_ids.json" ]; then
    say "step 4 was run by hand and its log ends in IDS OK; kept"
  else
    aside "$R16_ROOT/ids" "$R16_LOG/ids.log"
    "$SAFE_PY" -I scripts/xr1_lead_time/rldx_test_table.py ids --inputs-dir "$RLDXHS_ROOT/inputs" \
      --e1-scores-dir "$R16_ROOT/scores" --output-dir "$R16_ROOT/ids" > "$R16_LOG/ids.log" 2>&1 \
      || die "ids failed: $(tail -n 1 "$R16_LOG/ids.log")"
  fi
  cat "$R16_LOG/ids.log"
  [ "$(grep -c -F '800 rollouts (400 successes, 400 failures), 20 tasks, per-task (successes, failures): [(20, 20)]' \
       "$R16_LOG/ids.log")" = 2 ] || die "the sets are not 400 + 400 rollouts over 20 tasks with 20 + 20 each"
  grep -q "^IDS OK" "$R16_LOG/ids.log" || die "ids.log has no IDS OK"
  mark ids
}

# ---- step 5: CPU scoring (background) ----------------------------------------------------------
cpu_scoring() {
  unset OPENBLAS_NUM_THREADS
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  local SHARDS COMMON
  SHARDS=($(cat "$R16_LOG/e1b_shards.txt"))
  COMMON=(--dataset-dir "$RLDX_RAW" "${SHARDS[@]}"
          --runtime-bundle "$RLDXHS_ROOT/inputs/rldx_bundle.json" --train-ids "$RLDXHS_ROOT/inputs/train_ids.json"
          --cohort calibration "$R16_ROOT/ids/calibration_ids.json" --cohort test "$R16_ROOT/ids/test_ids.json"
          --cohort rec_calibration "$RLDXHS_ROOT/inputs/calibration_ids.json" --cohort rec_test "$RLDXHS_ROOT/inputs/test_ids.json"
          --feature-aggregation last --seeds 0 1 2 --score-train)
  fold() {
    local F=$1 EXTRA
    EXTRA=()
    if [ "$F" = all ]; then
      EXTRA=(--reference-raw rec_calibration "$R7_ROOT/folds/fold_all/calibration_predictions.jsonl"
             --reference-raw rec_test "$R7_ROOT/folds/fold_all/test_predictions.jsonl"
             --reference-raw train "$R7_ROOT/folds/fold_all/train_predictions.jsonl")
    fi
    nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cross_task "${COMMON[@]}" --folds "$FOLDS" --fold "$F" \
      --detectors hide_and_seek hide_and_seek_zero hide_and_seek_first mahalanobis cosine_knn pca_kmeans rnd \
      ${EXTRA[@]+"${EXTRA[@]}"} --output-dir "$R16_ROOT/folds/fold_$F" > "$R16_LOG/score_fold_$F.log" 2>&1
    echo "=== FOLD $F rc=$? $(date -Iseconds)"
  }
  shuffle() {
    nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cohorts "${COMMON[@]}" --detectors hide_and_seek_task_shuffle \
      --reference rec_calibration "$RLDXHS_ROOT/scores/calibration_predictions.jsonl" \
      --reference rec_test "$RLDXHS_ROOT/scores/test_predictions.jsonl" \
      --output-dir "$R16_ROOT/shuffle" > "$R16_LOG/score_shuffle.log" 2>&1
    echo "=== SHUFFLE rc=$? $(date -Iseconds)"
  }
  fold all & fold 0 & fold 1 & fold 2 & fold 3 & shuffle & wait
  echo "=== CPU SCORING DONE $(date -Iseconds)"
}
start_cpu() {
  passed cpu && return
  aside "$R16_ROOT/folds" "$R16_ROOT/shuffle" "$R16_LOG"/score_fold_*.log "$R16_LOG/score_shuffle.log" "$R16_LOG/score_cpu.log"
  say "step 5: CPU scoring started (folds all and 0-3, the shuffle; 8 threads each)"
  cpu_scoring > "$R16_LOG/score_cpu.log" 2>&1 &
  CPU_PID=$!
}
check_cpu() {
  local log=$R16_LOG/score_cpu.log n
  cat "$log"
  [ "$(count '^=== FOLD (all|[0-3]) rc=0 ' "$log")" = 5 ] || die "a fold did not end with rc=0 (score_cpu.log)"
  grep -q "^=== SHUFFLE rc=0 " "$log" || die "the shuffle did not end with rc=0 (score_shuffle.log)"
  grep -q "^=== CPU SCORING DONE" "$log" || die "the CPU scoring did not finish"
  if grep -l MISMATCH "$R16_LOG"/score_fold_*.log "$R16_LOG/score_shuffle.log"; then die "MISMATCH in the logs above"; fi
  grep -q "^REFERENCE OK" "$R16_LOG/score_fold_all.log" || die "fold all does not reproduce R7 (score_fold_all.log)"
  grep -q "^REFERENCE OK" "$R16_LOG/score_shuffle.log" || die "the shuffle does not reproduce R15 (score_shuffle.log)"
  n=$(count '^reference .* -> OK$' "$R16_LOG/score_fold_all.log")
  say "fold all: $n reference checks OK (the plan expects 21)"
  n=$(count '^reference .* -> OK$' "$R16_LOG/score_shuffle.log")
  say "shuffle: $n reference checks OK (the plan expects 2)"
  mark cpu
}

# ---- step 6: SAFE scores of E1b's exports (GPU 0, background) ---------------------------------
gpu_scoring() {
  local M F S RUN X
  for M in lstm indep; do
    for F in all 0 1 2 3; do
      for S in 0 1 2; do
        if [ "$F" = all ]; then RUN=$RLDX_ROOT/final/${M}_seed$S; else RUN=$R7S/train/fold_$F/${M}_seed$S; fi
        ( for X in calibration test; do
            echo "=== START $M fold $F seed $S $X $(date -Iseconds) GPU 0 ==="
            CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R16_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.score_seen_checkpoint_external \
              --export-dir "$E1B_ROOT/safe_export/$X" --safe-repo "$SAFE_REPO" --training-run-dir "$RUN" \
              --output-dir "$R16_ROOT/safe_score/fold_$F/$X/${M}_seed$S" --group all --device cuda --unseen-tasks score \
              > "$R16_LOG/safe_score_${M}_fold${F}_${X}_seed$S.log" 2>&1
            echo "=== END $M fold $F seed $S $X rc=$? $(date -Iseconds) ==="
          done ) &
      done
      wait
    done
  done
  echo "=== GPU SCORING DONE $(date -Iseconds)"
}
start_gpu() {
  passed gpu && return
  aside "$R16_ROOT/safe_score" "$R16_LOG"/safe_score_*.log
  say "step 6: SAFE scoring started on GPU 0 (60 runs: LSTM, then MLP)"
  gpu_scoring > "$R16_LOG/safe_score_gpu0.log" 2>&1 &
  GPU_PID=$!
}
check_gpu() {
  local log=$R16_LOG/safe_score_gpu0.log
  local ok end
  ok=$(count '^=== END .* rc=0 ' "$log")
  end=$(count '^=== END ' "$log")
  say "SAFE scoring: $ok of $end runs ended with rc=0 (60 expected)"
  [ "$ok" = 60 ] && [ "$end" = 60 ] || { grep "^=== END" "$log" | grep -v " rc=0 "; die "a SAFE scoring run failed (safe_score_gpu0.log)"; }
  grep -q "^=== GPU SCORING DONE" "$log" || die "the SAFE scoring did not finish"
  mark gpu
}

# ---- step 7: SAFE fold directories -------------------------------------------------------------
step_safe_folds() {
  passed safe_folds && return
  local log=$R16_LOG/safe_folds.log
  local M F X S R7D bad=0 A
  aside "$R16_ROOT/safe_folds" "$log"
  say "step 7: SAFE fold directories"
  for M in lstm indep; do
    for F in all 0 1 2 3; do
      R7D=$R7S/folds/$M/fold_$F
      if [ $M = lstm ] && [ "$F" = 0 ]; then R7D=$R7S/folds/lstm/fold_0_f32; fi
      A=(--r7-fold-dir "$R7D" --folds "$FOLDS" --cohort calibration "$R16_ROOT/ids/calibration_ids.json"
         --cohort test "$R16_ROOT/ids/test_ids.json" --output-dir "$R16_ROOT/safe_folds/$M/fold_$F")
      for X in calibration test; do for S in 0 1 2; do A+=(--scores $X $S "$R16_ROOT/safe_score/fold_$F/$X/${M}_seed$S"); done; done
      if [ $M = lstm ] && [ "$F" = all ]; then
        for X in calibration test; do A+=(--e1-reference $X "$R16_ROOT/scores/${X}_safe_lstm_predictions.jsonl"); done
      fi
      echo "=== $M fold $F" >> "$log"
      "$SAFE_PY" -I scripts/xr1_lead_time/rldx_test_table.py safe-fold "${A[@]}" >> "$log" 2>&1 || bad=1
    done
  done
  grep -E "^(===|reference|fold|SAFE FOLD OK|REFERENCE MISMATCH|STOP|error)" "$log"
  [ $bad = 0 ] || die "a safe-fold call failed (safe_folds.log)"
  [ "$(count '^SAFE FOLD OK' "$log")" = 10 ] || die "fewer than 10 SAFE FOLD OK (safe_folds.log)"
  [ "$(count '^reference (calibration|test) vs E1: .* -> OK$' "$log")" = 2 ] \
    || die "fold all's SAFE-LSTM scores do not equal step 3's (safe_folds.log)"
  mark safe_folds
}

# ---- step 8: the table -------------------------------------------------------------------------
step_table() {
  passed table && return
  local rc
  aside "$R16_ROOT/rldx_e1b_test_table.json" "$R16_LOG/table.log"
  say "step 8: the table"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
    --safe-dir "$RLDX_ROOT/calibration/lstm" --inputs-dir "$RLDXHS_ROOT/inputs" --folds "$FOLDS" --ids-dir "$R16_ROOT/ids" \
    --e1-scores-dir "$R16_ROOT/scores" --e1-result "$E1B_ROOT/e1b_result.json" --freeze-manifest "$E1_FREEZE" \
    --fold-dirs "$R16_ROOT"/folds/fold_{0,1,2,3,all} --r7-fold-dirs "$R7_ROOT"/folds/fold_{0,1,2,3,all} \
    --fold-dirs "$R16_ROOT"/safe_folds/lstm/fold_{0,1,2,3,all} \
    --r7-fold-dirs "$R7S/folds/lstm/fold_0_f32" "$R7S"/folds/lstm/fold_{1,2,3,all} \
    --fold-dirs "$R16_ROOT"/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs "$R7S"/folds/indep/fold_{0,1,2,3,all} \
    --shuffle-dir "$R16_ROOT/shuffle" --rldxhs-scores-dir "$RLDXHS_ROOT/scores" --r7-table "$R7_ROOT/rldx_unseen_safe.json" \
    --reference-per-seed "$RLDXPS_ROOT/rldx_per_seed.json" --output "$R16_ROOT/rldx_e1b_test_table.json" \
    > "$R16_LOG/table.log" 2>&1
  rc=$?
  cat "$R16_LOG/table.log"
  [ $rc = 0 ] && grep -q "^RLDX-1 TEST TABLE OK" "$R16_LOG/table.log" || die "the table run failed or a gate failed (table.log)"
  mark table
}

finish() {
  local f
  sha256sum -c --quiet "$R16_LOG/frozen_inputs.sha256" || die "a frozen input changed (lines above)"
  say "INPUTS UNCHANGED"
  mkdir -p "$R16_ROOT/evidence"
  for f in "$R16_ROOT/rldx_e1b_test_table.json" "$R16_LOG"/{export,gate_e,validate_calibration,validate_test}.log \
           "$R16_LOG"/{frozen_safe_external,score_safe,ids,score_cpu,score_fold_all,score_shuffle}.log \
           "$R16_LOG"/{safe_score_gpu0,safe_folds,table}.log "$R16_LOG/frozen_inputs.sha256"; do
    cp "$f" "$R16_ROOT/evidence/" || say "NOTE: could not copy $f"
  done
  say "table sha256 $(sha256sum "$R16_ROOT/rldx_e1b_test_table.json" | cut -c1-64)"
  say "R16 PIPELINE DONE"
  cp "$R16_LOG/pipeline.log" "$R16_ROOT/evidence/" 2>/dev/null || true
}

say "R16 pipeline on $(hostname), PID $$"
say "script sha256 $(sha256sum "$0" | cut -c1-64); worktree $(git rev-parse HEAD)"
[ "$(git rev-parse HEAD)" = "$R16_SHA" ] || die "the worktree is not at R16_SHA"
nvidia-smi -L 2>&1 | head -n 4
for f in "$R16_LOG/e1b_shards.txt" "$R16_LOG/frozen_inputs.sha256"; do [ -s "$f" ] || die "$f is missing (step 1)"; done
while pgrep -u "$(id -un)" -f "python.*$R16_ROOT" > /dev/null; do
  say "a command started by hand is still writing to R16_ROOT; waiting"
  sleep 60
done

gate_e
step_s
step_ids
start_cpu
start_gpu
if [ -n "$GPU_PID" ]; then wait "$GPU_PID"; check_gpu; fi
step_safe_folds
if [ -n "$CPU_PID" ]; then say "waiting for the CPU scoring"; wait "$CPU_PID"; check_cpu; fi
step_table
finish
