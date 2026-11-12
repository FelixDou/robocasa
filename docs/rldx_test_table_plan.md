# R15. RLDX-1's Table II on its test set

Written 2026-11-12, before any run. Branch `claude/rldx-test-table` (from `claude/review-analyses`).
Exploratory, like Table II on XR-1.

## Why

In the RSS Stage 1 draft (2026-11-12) both policies use the same three sets:
- **XR-1:** training, validation, and test with a new calibration set.
- **RLDX-1:** training (740 rollouts); validation, the recorded 100 + 100 held-out rollouts with 60
  calibration successes; and test, E1's new collection (`rldx1_e1_20261105_210723`): calibration
  and test sets of 20 successes and 20 failures per task.

Table III already reports RLDX-1's test set (E1). Table II's RLDX-1 block still comes from the
validation set (R7, `rldx_unseen_safe.json`). R15 computes the same block on the test set, with
the same models, so that Table II reports test numbers for both policies.

## Design (fixed now)

- **Rows.**
  - The seven cross-task detectors of R7: Hide-and-Seek, zero-input, first-inference,
    Mahalanobis, cosine k-NN, PCA-KMeans and RND.
  - The two SAFE heads.
  - The same-task shuffle control (seen tasks only).
  - Timers P (seen) and G (seen and unseen).
- **Models.** R7's, unchanged:
  - **Cross-task detectors:** retrained by `score_cross_task` with R7's settings, seeds and
    threads. Retraining is deterministic, so they must reproduce R7's trajectories exactly (G1).
  - **SAFE heads:** R7's 30 checkpoints (fold all: the SAFE20 final runs; folds 0–3: R7's fold
    runs) score E1's two exports.
  - **Shuffle:** `score_cohorts` with §15.22's settings.
- **Fit.** As on XR-1 (Sec. IV of the paper):
  - The test set's calibration successes set the margin and its calibration failures choose the
    fused path (`confirm_lead.fit_fast`).
  - Seen tasks: fold all, per-task normalization, + P.
  - Unseen tasks: per fold, pooled normalization, + G fitted on the seen tasks' calibration
    rollouts and evaluated on the held-out tasks' test rollouts, pooled over folds.
- **R7's variant** (path chosen on in-sample training failures, because the validation set had
  no calibration failures) is also reported, on the same sets, as `table2.train`.
- **Statistics.** Mean and std (ddof 1) over training seeds 0–2, plus the seed average; FPR, gain
  over the timer, TWA.

## Gates (no table without them)

- **G0, code path.** With R7's fold directories and R7's protocol, every row of
  `rldx_unseen_safe.json` (seen and unseen, all nine detectors) and the shuffle row of
  `rldx_per_seed.json` are reproduced exactly.
- **G1, models.**
  - Every new cross-task fold directory reproduces R7's trajectories of the recorded
    calibration, test and training rollouts (max |diff| ≤ 1e-12) and R7's normalizations.
  - Every SAFE fold directory copies R7's training cohort and normalization byte for byte. Its
    scores of E1's exports come from R7's checkpoints (path and SHA-256 checked).
  - The shuffle directory reproduces §15.22's trajectories.
- **G2, E1.**
  - H_k and ρ_k equal the freeze manifest's.
  - P alone on the test set equals E1's P (FPR 0.06, lead 0.2421).
  - The seed-averaged SAFE-LSTM and Hide-and-Seek trajectories on both E1 sets equal E1's
    (≤ 1e-9).
  - `safe-fold` also requires fold all's SAFE-LSTM scores of E1's exports to equal E1's
    (≤ 1e-12).

## Expectations (written before running)

- **P alone:** lead 0.242 at FPR 0.06 (E1).
- **G:** weak again (validation: seen lead 0.032, unseen 0.026), so every score gains a lot over
  G, controls included.
- **Hide-and-Seek and SAFE-LSTM + P:** single-model gains of a few hundredths, within the E1
  ensembles' CIs. With 400 calibration failures choosing the path, they may differ from the
  validation rows.
- **OOD rows:** the OOD rows and the input ablations gain less than Hide-and-Seek.
- **FPR:** unseen FPR may exceed 0.05, as on XR-1 and the validation set.

## Code

- `scripts/xr1_lead_time/rldx_test_table.py` (`ids`, `safe-fold`, `run`), with tests in
  `tests/test_rldx_test_table.py` (6 tests).
- No other file changes. `rldx_unseen.py`, `score_cross_task`, `score_cohorts` and
  `score_seen_checkpoint_external` are used as they are.

## Runbook

Run from a pinned worktree at `R15_SHA`, the full SHA of the pushed commit that contains this
plan. Never pull or check out in the shared checkout `$PROJECT_FS/robocasa`.
- **CPU:** steps 1–3, 5 and 6.
- **GPUs:** two for step 4, which can run while step 3 scores.

### 0. Session

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/26.9.3/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export CLIENT_PY=$STORAGE_BS/envs/robocasa_openpi/bin/python SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export SAFE_REPO=$PROJECT_FS/SAFE
export RLDX_EXPORT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_original_safe35_natural50_20260808_203158/official_safe_supported20_natural50
export RLDX_RAW=$($CLIENT_PY -I -c "import json,sys; print(json.load(open(sys.argv[1]))['source_dataset'])" $RLDX_EXPORT/conversion_report.json)
export RLDX_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_safe20_natural_weighted_20260815_162530
export RLDXHS_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_hide_and_seek_20261023_000418
export RLDXPS_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_per_seed_20261028_145711
export R7_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_unseen_20261029_135501
export R7S=$R7_ROOT/safe FOLDS=$R7_ROOT/rldx_folds.json
export E1_ROOT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_e1_20261105_210723
export E1_FREEZE=$STORAGE_BS/robocasa_checkpoints/safe/e2_freeze_20261029_205743/e2_freeze_manifest.json
export R15_SHA=<full SHA of the commit containing this plan>
export R15_REPO=$PROJECT_FS/robocasa_rldxtest_${R15_SHA:0:8}
export R15_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_test_table_$(date +%Y%m%d_%H%M%S)
export R15_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R15_ROOT)
mkdir -p $R15_ROOT $R15_LOG
for v in PROJECT_FS STORAGE_BS CLIENT_PY SAFE_PY SAFE_REPO RLDX_EXPORT RLDX_RAW RLDX_ROOT RLDXHS_ROOT RLDXPS_ROOT \
         R7_ROOT R7S FOLDS E1_ROOT E1_FREEZE R15_SHA R15_REPO R15_ROOT R15_LOG; do
  printf 'export %s=%q\n' $v "${!v}"; done > $R15_ROOT/session.env
echo "R15_ROOT=$R15_ROOT"; echo "RLDX_RAW=$RLDX_RAW"
```

In a new terminal, run the first four lines, then `source <R15_ROOT>/session.env`.

### 1. Worktree, tests, frozen inputs

```bash
git -C $PROJECT_FS/robocasa fetch origin claude/rldx-test-table
test -e $R15_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R15_REPO $R15_SHA
test "$(git -C $R15_REPO rev-parse HEAD)" = "$R15_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
mkdir -p $R15_REPO/robocasa/models/assets
cp -aln $PROJECT_FS/robocasa/robocasa/models/assets/. $R15_REPO/robocasa/models/assets/
cd $R15_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_rldx_test_table tests.test_xr1_rldx_unseen \
  tests.test_safe_score_cross_task tests.test_xr1_cross_task_safe 2>&1 | grep -E "^(Ran|OK|FAILED)"
R15_INPUTS=($RLDXHS_ROOT/inputs/{rldx_bundle,train_ids,calibration_ids,test_ids}.json
            $RLDXHS_ROOT/scores/{calibration,test,train}_predictions.jsonl $RLDXHS_ROOT/scores/normalizations.json
            $RLDXPS_ROOT/rldx_per_seed.json $R7_ROOT/rldx_unseen_safe.json $FOLDS
            $R7_ROOT/folds/fold_{0,1,2,3,all}/{scoring.json,normalizations.json,{calibration,test,train}_predictions.jsonl}
            $R7S/folds/lstm/fold_0_f32/scoring.json $R7S/folds/lstm/fold_{1,2,3,all}/scoring.json
            $R7S/folds/indep/fold_{0,1,2,3,all}/scoring.json
            $RLDX_ROOT/calibration/lstm/lstm_seed{0,1,2}/normalized_scores.jsonl
            $E1_ROOT/scores/{calibration,test}_{safe_lstm,hide_and_seek}_predictions.jsonl $E1_ROOT/e1_result.json
            $E1_FREEZE $RLDX_RAW/manifest.jsonl)
for f in "${R15_INPUTS[@]}"; do test -s $f && echo "OK $f" || echo "STOP: missing $f"; done > $R15_LOG/inputs_check.log
grep -v '^OK' $R15_LOG/inputs_check.log; grep -c '^OK' $R15_LOG/inputs_check.log
sha256sum "${R15_INPUTS[@]}" > $R15_LOG/frozen_inputs.sha256
ls -d $E1_ROOT/collection/{calibration,test}/shard* | tee $R15_LOG/e1_shards.txt
for x in calibration test; do echo "$x export: $(ls $E1_ROOT/safe_export/$x | head -n 3 | tr '\n' ' ')"; done
nproc; free -g | head -n 2
```

Continue only if all of these hold:
- `HEAD OK` prints;
- the tests end with `OK`;
- no `STOP` line appears and the count is 56;
- the shard list and both exports are present.

### 2. Test-set ID lists (CPU, seconds)

```bash
cd $R15_REPO
"$SAFE_PY" -I scripts/xr1_lead_time/rldx_test_table.py ids --inputs-dir $RLDXHS_ROOT/inputs \
  --e1-scores-dir $E1_ROOT/scores --output-dir $R15_ROOT/ids 2>&1 | tee $R15_LOG/ids.log
```

It must print 800 rollouts (400 successes, 400 failures) over 20 tasks, `[(20, 20)]` for each
set, and `IDS OK`.

### 3. CPU scoring (background, hours)

Five folds of the cross-task detectors and the shuffle control. R7's thread settings (8
OpenMP/MKL threads per job): the reproduction checks depend on them. Six jobs need 48 cores. With
fewer, run `fold all`, `fold 0`, `fold 1` first and the rest after.

```bash
cd $R15_REPO
cat > $R15_LOG/score_cpu.sh <<'EOF'
set -u
source $R15_ROOT/session.env
cd $R15_REPO
unset OPENBLAS_NUM_THREADS
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mapfile -t SHARDS < $R15_LOG/e1_shards.txt
COMMON=(--dataset-dir $RLDX_RAW "${SHARDS[@]}"
        --runtime-bundle $RLDXHS_ROOT/inputs/rldx_bundle.json --train-ids $RLDXHS_ROOT/inputs/train_ids.json
        --cohort calibration $R15_ROOT/ids/calibration_ids.json --cohort test $R15_ROOT/ids/test_ids.json
        --cohort rec_calibration $RLDXHS_ROOT/inputs/calibration_ids.json --cohort rec_test $RLDXHS_ROOT/inputs/test_ids.json
        --feature-aggregation last --seeds 0 1 2 --score-train)
fold() {
  F=$1; EXTRA=()
  if [ $F = all ]; then
    EXTRA=(--reference-raw rec_calibration $R7_ROOT/folds/fold_all/calibration_predictions.jsonl
           --reference-raw rec_test $R7_ROOT/folds/fold_all/test_predictions.jsonl
           --reference-raw train $R7_ROOT/folds/fold_all/train_predictions.jsonl)
  fi
  "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cross_task "${COMMON[@]}" --folds $FOLDS --fold $F \
    --detectors hide_and_seek hide_and_seek_zero hide_and_seek_first mahalanobis cosine_knn pca_kmeans rnd \
    "${EXTRA[@]}" --output-dir $R15_ROOT/folds/fold_$F > $R15_LOG/score_fold_$F.log 2>&1
  echo "=== FOLD $F rc=$? $(date -Iseconds)"
}
shuffle() {
  "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cohorts "${COMMON[@]}" --detectors hide_and_seek_task_shuffle \
    --reference rec_calibration $RLDXHS_ROOT/scores/calibration_predictions.jsonl \
    --reference rec_test $RLDXHS_ROOT/scores/test_predictions.jsonl \
    --output-dir $R15_ROOT/shuffle > $R15_LOG/score_shuffle.log 2>&1
  echo "=== SHUFFLE rc=$? $(date -Iseconds)"
}
fold all & fold 0 & fold 1 & fold 2 & fold 3 & shuffle & wait
echo "=== CPU SCORING DONE"
EOF
nohup nice -n 10 bash $R15_LOG/score_cpu.sh > $R15_LOG/score_cpu.log 2>&1 &
echo "CPU scoring PID $! on $(hostname)"
```

**Progress:** `cat $R15_LOG/score_cpu.log; tail -qn 2 $R15_LOG/score_fold_*.log $R15_LOG/score_shuffle.log`.

It is done when:
- `score_cpu.log` shows five `FOLD ... rc=0` lines, `SHUFFLE rc=0` and `=== CPU SCORING DONE`;
- `score_fold_all.log` ends with 21 `reference ... -> OK` lines and `REFERENCE OK`;
- `score_shuffle.log` ends with two `reference ... -> OK` lines and `REFERENCE OK`.

Any other `rc` or a `MISMATCH` is a stop condition: paste the log. If a reproduction check fails by
about 1e-7, do not change the thread settings yourself.

### 4. SAFE scores of E1's exports (two GPUs; can run during step 3)

All 30 R7 checkpoints score both exports (60 runs). GPU 0 runs the LSTM, GPU 1 the MLP, with
three seeds at a time on each.

```bash
cd $R15_REPO
for G in 0 1; do
  if [ $G = 0 ]; then M=lstm; else M=indep; fi
  cat > $R15_LOG/safe_score_gpu$G.sh <<EOF
#!/bin/bash
cd $R15_REPO
for F in all 0 1 2 3; do
  for S in 0 1 2; do
    if [ \$F = all ]; then RUN=$RLDX_ROOT/final/${M}_seed\$S; else RUN=$R7S/train/fold_\$F/${M}_seed\$S; fi
    ( for X in calibration test; do
        echo "=== START $M fold \$F seed \$S \$X \$(date -Iseconds) GPU $G ==="
        CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R15_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.score_seen_checkpoint_external \\
          --export-dir $E1_ROOT/safe_export/\$X --safe-repo $SAFE_REPO --training-run-dir \$RUN \\
          --output-dir $R15_ROOT/safe_score/fold_\$F/\$X/${M}_seed\$S --group all --device cuda --unseen-tasks score \\
          > $R15_LOG/safe_score_${M}_fold\${F}_\${X}_seed\$S.log 2>&1
        echo "=== END $M fold \$F seed \$S \$X rc=\$? \$(date -Iseconds) ==="
      done ) &
  done
  wait
done
EOF
done
for G in 0 1; do nohup bash $R15_LOG/safe_score_gpu$G.sh > $R15_LOG/safe_score_gpu$G.log 2>&1 & echo "GPU $G PID $! on $(hostname)"; done
```

**Progress:** `grep -c "rc=0" $R15_LOG/safe_score_gpu*.log`. It is done when both logs count 30.
Any other `rc` is a stop condition.

### 5. SAFE fold directories (CPU, minutes; after step 4)

```bash
cd $R15_REPO
for M in lstm indep; do for F in all 0 1 2 3; do
  R7D=$R7S/folds/$M/fold_$F
  if [ $M = lstm ] && [ $F = 0 ]; then R7D=$R7S/folds/lstm/fold_0_f32; fi
  A=(--r7-fold-dir $R7D --folds $FOLDS --cohort calibration $R15_ROOT/ids/calibration_ids.json
     --cohort test $R15_ROOT/ids/test_ids.json --output-dir $R15_ROOT/safe_folds/$M/fold_$F)
  for X in calibration test; do for S in 0 1 2; do A+=(--scores $X $S $R15_ROOT/safe_score/fold_$F/$X/${M}_seed$S); done; done
  if [ $M = lstm ] && [ $F = all ]; then
    for X in calibration test; do A+=(--e1-reference $X $E1_ROOT/scores/${X}_safe_lstm_predictions.jsonl); done
  fi
  "$SAFE_PY" -I scripts/xr1_lead_time/rldx_test_table.py safe-fold "${A[@]}" 2>&1 | tail -n 4
done; done | tee $R15_LOG/safe_folds.log
grep -c "SAFE FOLD OK" $R15_LOG/safe_folds.log
```

The count must be 10. Fold all of the LSTM also prints two `reference ... vs E1 ... -> OK` lines.

### 6. The table (CPU, minutes; after steps 3 and 5)

```bash
cd $R15_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R15_ROOT/ids \
  --e1-scores-dir $E1_ROOT/scores --e1-result $E1_ROOT/e1_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R15_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R15_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R15_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R15_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --output $R15_ROOT/rldx_test_table.json 2>&1 | tee $R15_LOG/table.log
sha256sum -c --quiet $R15_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
sha256sum $R15_ROOT/rldx_test_table.json; echo "R15_ROOT=$R15_ROOT"
```

The log prints, in order:
- **G1:** one `OK` per fold of each family and one for the shuffle;
- **G0:** `9 of 9 rows equal rldx_unseen_safe.json; shuffle OK; G OK`;
- **G2a–c:** three lines, each `OK`;
- **the tables:** two, one per path-choice rule;
- `RLDX-1 TEST TABLE OK`.

Paste the whole `table.log`, `INPUTS UNCHANGED` and the SHA-256, then copy
`rldx_test_table.json` and `table.log` to the Mac.

## Result

(to be written after the run)
