# R16. RLDX-1's Table II on E1b's test set

Written 2026-11-13, before any run. Branch `claude/rldx-e1b-table`: the E1b line
(`claude/rldx-e1b`) merged with R15 (`claude/rldx-test-table`). Exploratory, like Table II on XR-1.

## Why

- **The paper's RLDX-1 rows come from two test sets.**
  - **Table III:** the test of H3, on E1b's new sets (`rldx1_e1b_20261112_195208`).
  - **Table II:** R15's rows, computed on E1's sets (`rldx1_e1_20261105_210723`).
  - So the paper has to describe two RLDX-1 test sets.
- **R16 computes Table II's RLDX-1 block on E1b's sets,** with R15's code and models. Then both
  tables use one test set.
- **E1's 1,600 rollouts then serve only to fix the H3 path.** That is validation data. The
  paper's H3 paragraph still states, in one sentence, that a first test was run on them.

## Design (fixed now)

Everything is R15's (`docs/rldx_test_table_plan.md`) except the sets:

- **Sets:** E1b's calibration and test sets, 400 successes and 400 failures each, over 20 tasks.
- **Rows, models, fit, statistics:** R15's.
  - The seven cross-task detectors are retrained with R7's settings.
  - R7's 30 SAFE checkpoints are used.
  - The shuffle control follows §15.22.
  - The path is chosen on the test set's calibration failures (`table2.calibration`, the
    paper's rows). The `train` variant is reported alongside.
- **SAFE-LSTM scores of E1b's rollouts.** E1b scored Hide-and-Seek only, by design. R16 runs
  E1's frozen SAFE-LSTM scoring (`score_e1.py safe`, the three SAFE20 final runs) on E1b's
  exports. These scores are made after E1b's analysis and play no part in it.
- **Code:**
  - `rldx_test_table.py` runs unchanged, with E1b's scores in place of E1's.
  - E1b's `e1b_result.json` stores P like `e1_result.json`.
  - The output's `plan` field still names R15, because the script is R15's.

## Reporting (fixed now)

- **Table II:** the paper's RLDX-1 block becomes R16's `table2.calibration` rows, whatever they
  show, with the timer leads in the footnote. R15's rows are no longer reported.
- **Data paragraph:** RLDX-1's sets are described as:
  - training;
  - validation: the recorded 100 + 100, plus E1's 1,600 rollouts, on which the H3 path was
    fixed;
  - test: E1b.
- **H3 paragraph:** keeps its sentence on the first test.

## Gates (no table without them)

- **E (exports):**
  - both merged E1b sets are `VALID`;
  - each export holds 800 rollouts.
- **S (frozen SAFE-LSTM):** `score_e1.py safe` passes its reproduction checks and prints
  `SCORING OK`. Those checks are:
  - the final runs' digests;
  - the normalizations;
  - the recorded rollouts' raw scores.
- **G0, G1:** as in R15. They do not depend on the new sets.
- **G2, against E1b:**
  - H_k and ρ_k equal the freeze manifest's.
  - P alone on the test set equals E1b's P: FPR 0.0675, lead 0.2686.
  - The seed-averaged trajectories equal the scores in `$R16_ROOT/scores` (≤ 1e-9):
    - Hide-and-Seek: E1b's frozen scores, copied unchanged;
    - SAFE-LSTM: gate S's output.
  - In `safe-fold`, fold all's SAFE-LSTM scores equal gate S's (≤ 1e-12).
- **Inputs:** the 54 frozen inputs (R15's list with E1b's files in place of E1's) are unchanged at
  the end.

## Expectations (written before running)

- **P alone:** FPR 0.0675 and lead 0.269, as in E1b.
- **G:** lead about 0.16, as on E1's sets (R15), so every row gains less over G than over P.
- **Hide-and-Seek + P:** the largest seen-task gain.
  - Single models with the adaptive path: a few hundredths up to about 0.1 (R15 on E1's sets:
    0.096).
  - E1b's ensemble with its fixed path gained 0.054.
- **SAFE-LSTM:** second at a few hundredths (R15: 0.054).
- **OOD rows and input ablations:** at most about 0.02.
- **Unseen tasks:** FPR may exceed 0.05.

## Cost

- **CPU:** about the same as R15, a few hours with 48 cores for the fold scoring.
- **GPU:**
  - 9 short runs for the frozen SAFE-LSTM scores;
  - R15's 60 runs on two GPUs.
- **Storage:** E1b's merged sets and exports are written under `$E1B_ROOT`, as E1's are, without
  changing any existing E1b file. Everything else goes under `$R16_ROOT`.

## Runbook

Run from a pinned worktree at `R16_SHA`, the full SHA of the pushed commit that contains this
plan. Never pull or check out in the shared checkout `$PROJECT_FS/robocasa`.

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
export E1_FREEZE=$STORAGE_BS/robocasa_checkpoints/safe/e2_freeze_20261029_205743/e2_freeze_manifest.json
export E1B_ROOT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_e1b_20261112_195208
export E1B_REPO=$PROJECT_FS/robocasa_rldx_e1b_bf6f6080
export R16_SHA=<full SHA of the commit containing this plan>
export R16_REPO=$PROJECT_FS/robocasa_rldx_e1btable_${R16_SHA:0:8}
export R16_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_e1b_test_table_$(date +%Y%m%d_%H%M%S)
export R16_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $R16_ROOT)
mkdir -p $R16_ROOT $R16_LOG
for v in PROJECT_FS STORAGE_BS CLIENT_PY SAFE_PY SAFE_REPO RLDX_EXPORT RLDX_RAW RLDX_ROOT RLDXHS_ROOT RLDXPS_ROOT \
         R7_ROOT R7S FOLDS E1_FREEZE E1B_ROOT E1B_REPO R16_SHA R16_REPO R16_ROOT R16_LOG; do
  printf 'export %s=%q\n' $v "${!v}"; done > $R16_ROOT/session.env
echo "R16_ROOT=$R16_ROOT"; echo "RLDX_RAW=$RLDX_RAW"
```

In a new terminal, run the first four lines, then `source <R16_ROOT>/session.env`.

### 1. Worktree, assets, tests, frozen inputs

The worktree takes E1b's asset folder by `mv` (E1b is finished), so no new inodes are spent on
assets.

```bash
git -C $PROJECT_FS/robocasa fetch origin claude/rldx-e1b-table
test -e $R16_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R16_REPO $R16_SHA
test "$(git -C $R16_REPO rev-parse HEAD)" = "$R16_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
if [ -d $E1B_REPO/robocasa/models/assets ]; then
  rm -rf $R16_REPO/robocasa/models/assets && mv $E1B_REPO/robocasa/models/assets $R16_REPO/robocasa/models/assets
fi
git -C $R16_REPO status --short robocasa/models/assets | head -n 3
cd $R16_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_rldx_test_table tests.test_rldx_e1b \
  tests.test_rldx_e1_analysis tests.test_xr1_rldx_unseen tests.test_safe_score_cross_task tests.test_xr1_cross_task_safe \
  2>&1 | grep -E "^(Ran|OK|FAILED)"
R16_INPUTS=($RLDXHS_ROOT/inputs/{rldx_bundle,train_ids,calibration_ids,test_ids}.json
            $RLDXHS_ROOT/scores/{calibration,test,train}_predictions.jsonl $RLDXHS_ROOT/scores/normalizations.json
            $RLDXPS_ROOT/rldx_per_seed.json $R7_ROOT/rldx_unseen_safe.json $FOLDS
            $R7_ROOT/folds/fold_{0,1,2,3,all}/{scoring.json,normalizations.json,{calibration,test,train}_predictions.jsonl}
            $R7S/folds/lstm/fold_0_f32/scoring.json $R7S/folds/lstm/fold_{1,2,3,all}/scoring.json
            $R7S/folds/indep/fold_{0,1,2,3,all}/scoring.json
            $RLDX_ROOT/calibration/lstm/lstm_seed{0,1,2}/normalized_scores.jsonl
            $E1B_ROOT/scores/{calibration,test}_hide_and_seek_predictions.jsonl $E1B_ROOT/e1b_result.json
            $E1_FREEZE $RLDX_RAW/manifest.jsonl)
for f in "${R16_INPUTS[@]}"; do test -s $f && echo "OK $f" || echo "STOP: missing $f"; done > $R16_LOG/inputs_check.log
grep -v '^OK' $R16_LOG/inputs_check.log; grep -c '^OK' $R16_LOG/inputs_check.log
sha256sum "${R16_INPUTS[@]}" > $R16_LOG/frozen_inputs.sha256
ls -d $E1B_ROOT/collection/{calibration,test}/shard* | tee $R16_LOG/e1b_shards.txt
ls $E1B_ROOT/merged $E1B_ROOT/safe_export 2>&1 | head -n 2
nproc; free -g | head -n 2; nvidia-smi -L
```

Continue only if all of these hold:
- `HEAD OK` prints;
- the asset status shows nothing, or only the untracked `README.md`;
- the tests end with `OK`;
- no `STOP` line appears and the count is 54;
- four shards are listed;
- `merged` and `safe_export` do not exist yet.

### 2. E1b's SAFE exports (CPU, minutes)

```bash
cd $R16_REPO
for set in calibration test; do
  $CLIENT_PY -m robocasa.recovery.safe.merge_atomic_datasets --source-dirs $E1B_ROOT/collection/$set/shard* \
    --output-dir $E1B_ROOT/merged/$set 2>&1 | tail -n 2
  $CLIENT_PY -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir $E1B_ROOT/merged/$set | tail -n 1
  $CLIENT_PY -m robocasa.recovery.safe.export_to_official_safe --dataset-dir $E1B_ROOT/merged/$set \
    --output-dir $E1B_ROOT/safe_export/$set --summary-only 2>&1 | tail -n 3
done 2>&1 | tee $R16_LOG/export.log
```

Gate E: both merged sets are `VALID` and each export holds 800 rollouts.

### 3. Frozen SAFE-LSTM scores of E1b (GPU 0, minutes; gate S)

```bash
cd $R16_REPO
for s in 0 1 2; do
  for x in recorded:$RLDX_EXPORT calibration:$E1B_ROOT/safe_export/calibration test:$E1B_ROOT/safe_export/test; do
    CUDA_VISIBLE_DEVICES=0 $SAFE_PY -u -m robocasa.recovery.safe.score_seen_checkpoint_external \
      --export-dir ${x#*:} --safe-repo $SAFE_REPO --training-run-dir $RLDX_ROOT/final/lstm_seed$s \
      --output-dir $R16_ROOT/frozen_safe/${x%%:*}/seed$s --group all --device cuda 2>&1 | tail -n 1
  done
done | tee $R16_LOG/frozen_safe_external.log
$SAFE_PY -I scripts/rldx_e1/score_e1.py safe --freeze-manifest $E1_FREEZE --calibration-dir $RLDX_ROOT/calibration/lstm \
  $(for s in 0 1 2; do echo --recorded-scores $s $R16_ROOT/frozen_safe/recorded/seed$s; done) \
  $(for set in calibration test; do for s in 0 1 2; do echo --e1-scores $set $s $R16_ROOT/frozen_safe/$set/seed$s; done; done) \
  --output-dir $R16_ROOT/scores 2>&1 | tee $R16_LOG/score_safe.log
for set in calibration test; do
  cp $E1B_ROOT/scores/${set}_hide_and_seek_predictions.jsonl $R16_ROOT/scores/ &&
  cmp $E1B_ROOT/scores/${set}_hide_and_seek_predictions.jsonl $R16_ROOT/scores/${set}_hide_and_seek_predictions.jsonl &&
  echo "Hide-and-Seek $set copied"
done
```

Gate S:
- `score_safe.log` shows its reproduction checks and `SCORING OK`;
- both `Hide-and-Seek ... copied` lines print.

### 4. Test-set ID lists (CPU, seconds)

```bash
cd $R16_REPO
"$SAFE_PY" -I scripts/xr1_lead_time/rldx_test_table.py ids --inputs-dir $RLDXHS_ROOT/inputs \
  --e1-scores-dir $R16_ROOT/scores --output-dir $R16_ROOT/ids 2>&1 | tee $R16_LOG/ids.log
```

The output must show:
- 800 rollouts per set (400 successes, 400 failures) over 20 tasks;
- `[(20, 20)]` for each set;
- `IDS OK`.

### 5. CPU scoring (background, hours)

This step is R15's step 3, run on E1b's shards. It uses 8 OpenMP/MKL threads per job, as R7 did;
the reproduction checks depend on that setting. Six jobs need 48 cores. With fewer cores, run
`fold all`, `fold 0` and `fold 1` first and the rest after.

```bash
cd $R16_REPO
cat > $R16_LOG/score_cpu.sh <<'EOF'
set -u
source $R16_ROOT/session.env
cd $R16_REPO
unset OPENBLAS_NUM_THREADS
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mapfile -t SHARDS < $R16_LOG/e1b_shards.txt
COMMON=(--dataset-dir $RLDX_RAW "${SHARDS[@]}"
        --runtime-bundle $RLDXHS_ROOT/inputs/rldx_bundle.json --train-ids $RLDXHS_ROOT/inputs/train_ids.json
        --cohort calibration $R16_ROOT/ids/calibration_ids.json --cohort test $R16_ROOT/ids/test_ids.json
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
    "${EXTRA[@]}" --output-dir $R16_ROOT/folds/fold_$F > $R16_LOG/score_fold_$F.log 2>&1
  echo "=== FOLD $F rc=$? $(date -Iseconds)"
}
shuffle() {
  "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cohorts "${COMMON[@]}" --detectors hide_and_seek_task_shuffle \
    --reference rec_calibration $RLDXHS_ROOT/scores/calibration_predictions.jsonl \
    --reference rec_test $RLDXHS_ROOT/scores/test_predictions.jsonl \
    --output-dir $R16_ROOT/shuffle > $R16_LOG/score_shuffle.log 2>&1
  echo "=== SHUFFLE rc=$? $(date -Iseconds)"
}
fold all & fold 0 & fold 1 & fold 2 & fold 3 & shuffle & wait
echo "=== CPU SCORING DONE"
EOF
nohup nice -n 10 bash $R16_LOG/score_cpu.sh > $R16_LOG/score_cpu.log 2>&1 &
echo "CPU scoring PID $! on $(hostname)"
```

**Progress:** `cat $R16_LOG/score_cpu.log; tail -qn 2 $R16_LOG/score_fold_*.log $R16_LOG/score_shuffle.log`.

It is done when:
- `score_cpu.log` shows five `FOLD ... rc=0` lines, `SHUFFLE rc=0` and `=== CPU SCORING DONE`;
- `score_fold_all.log` ends with 21 `reference ... -> OK` lines and `REFERENCE OK`;
- `score_shuffle.log` ends with two `reference ... -> OK` lines and `REFERENCE OK`.

Any other `rc`, or a `MISMATCH`, is a stop condition: paste the log.

### 6. SAFE scores of E1b's exports (two GPUs; after step 3, during step 5)

This step is R15's step 4. All 30 R7 checkpoints score both exports (60 runs). GPU 0 runs the
LSTM and GPU 1 the MLP, three seeds at a time on each.

```bash
cd $R16_REPO
for G in 0 1; do
  if [ $G = 0 ]; then M=lstm; else M=indep; fi
  cat > $R16_LOG/safe_score_gpu$G.sh <<EOF
#!/bin/bash
cd $R16_REPO
for F in all 0 1 2 3; do
  for S in 0 1 2; do
    if [ \$F = all ]; then RUN=$RLDX_ROOT/final/${M}_seed\$S; else RUN=$R7S/train/fold_\$F/${M}_seed\$S; fi
    ( for X in calibration test; do
        echo "=== START $M fold \$F seed \$S \$X \$(date -Iseconds) GPU $G ==="
        CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R16_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.score_seen_checkpoint_external \\
          --export-dir $E1B_ROOT/safe_export/\$X --safe-repo $SAFE_REPO --training-run-dir \$RUN \\
          --output-dir $R16_ROOT/safe_score/fold_\$F/\$X/${M}_seed\$S --group all --device cuda --unseen-tasks score \\
          > $R16_LOG/safe_score_${M}_fold\${F}_\${X}_seed\$S.log 2>&1
        echo "=== END $M fold \$F seed \$S \$X rc=\$? \$(date -Iseconds) ==="
      done ) &
  done
  wait
done
EOF
done
for G in 0 1; do nohup bash $R16_LOG/safe_score_gpu$G.sh > $R16_LOG/safe_score_gpu$G.log 2>&1 & echo "GPU $G PID $! on $(hostname)"; done
```

**Progress:** `grep -c "rc=0" $R16_LOG/safe_score_gpu*.log`. It is done when both logs count 30.
Any other `rc` is a stop condition.

### 7. SAFE fold directories (CPU, minutes; after step 6)

```bash
cd $R16_REPO
for M in lstm indep; do for F in all 0 1 2 3; do
  R7D=$R7S/folds/$M/fold_$F
  if [ $M = lstm ] && [ $F = 0 ]; then R7D=$R7S/folds/lstm/fold_0_f32; fi
  A=(--r7-fold-dir $R7D --folds $FOLDS --cohort calibration $R16_ROOT/ids/calibration_ids.json
     --cohort test $R16_ROOT/ids/test_ids.json --output-dir $R16_ROOT/safe_folds/$M/fold_$F)
  for X in calibration test; do for S in 0 1 2; do A+=(--scores $X $S $R16_ROOT/safe_score/fold_$F/$X/${M}_seed$S); done; done
  if [ $M = lstm ] && [ $F = all ]; then
    for X in calibration test; do A+=(--e1-reference $X $R16_ROOT/scores/${X}_safe_lstm_predictions.jsonl); done
  fi
  "$SAFE_PY" -I scripts/xr1_lead_time/rldx_test_table.py safe-fold "${A[@]}" 2>&1 | tail -n 4
done; done | tee $R16_LOG/safe_folds.log
grep -c "SAFE FOLD OK" $R16_LOG/safe_folds.log
```

The count must be 10. The LSTM's fold all also prints two `reference ... -> OK` lines.

### 8. The table (CPU, minutes; after steps 5 and 7)

```bash
cd $R16_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_test_table.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS --ids-dir $R16_ROOT/ids \
  --e1-scores-dir $R16_ROOT/scores --e1-result $E1B_ROOT/e1b_result.json --freeze-manifest $E1_FREEZE \
  --fold-dirs $R16_ROOT/folds/fold_{0,1,2,3,all} --r7-fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/lstm/fold_{0,1,2,3,all} \
  --r7-fold-dirs $R7S/folds/lstm/fold_0_f32 $R7S/folds/lstm/fold_{1,2,3,all} \
  --fold-dirs $R16_ROOT/safe_folds/indep/fold_{0,1,2,3,all} --r7-fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --shuffle-dir $R16_ROOT/shuffle --rldxhs-scores-dir $RLDXHS_ROOT/scores --r7-table $R7_ROOT/rldx_unseen_safe.json \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --output $R16_ROOT/rldx_e1b_test_table.json 2>&1 | tee $R16_LOG/table.log
sha256sum -c --quiet $R16_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
mkdir -p $R16_ROOT/evidence && cp $R16_ROOT/rldx_e1b_test_table.json $R16_LOG/{export,frozen_safe_external,score_safe,ids,safe_folds,table}.log \
  $R16_LOG/frozen_inputs.sha256 $R16_ROOT/evidence/ && sha256sum $R16_ROOT/rldx_e1b_test_table.json
```

The log prints, in order:
- **G1:** one `OK` per fold of each family and one for the shuffle;
- **G0:** `9 of 9 rows equal rldx_unseen_safe.json; shuffle OK; G OK`;
- **G2a–c:** three lines, each `OK`. P is `(0.0675, 0.2686)`.
- **the tables:** two, one per path-choice rule;
- `RLDX-1 TEST TABLE OK`.

The script's messages say "E1" for the sets given with `--e1-scores-dir`; here those are E1b's
sets.

Paste the whole `table.log`, `INPUTS UNCHANGED` and the SHA-256. Then, on the Mac:

```bash
rsync -av tsubame:<R16_ROOT>/evidence/ <local worktree>/reports/rldx_e1b_test_table_evidence/
```

## Result

(to be written after the run)
