# OOD baselines per seed (RSS Table I, mean ± std over 3 model seeds)

Purpose: report the four OOD rows of Table I (Mahalanobis, cosine k-NN, PCA-KMeans, RND) as
Hide-and-Seek does, mean ± std over 3 seeds, like the SAFE, ActProbe and Hide-and-Seek rows
([`xr1_per_seed_detection.md`](xr1_per_seed_detection.md)).

Table I's OOD rows come from one run of the Section 15.10 pipeline
([`xr1_offline_detector_baselines.md`](xr1_offline_detector_baselines.md), run
`xr1_offline_baselines_20261017_100820`, code `d71b6d27`) with `--model-seed 0`. The model
seed is not inert there: it picks the 50,000 training rows each baseline is fitted on (when there
are more), the randomized PCA, the k-means initialization, the RND network initialization and
the k-NN reference subsample. Seeds 1 and 2 are added here.

Protocol:
- **Same configuration, new seed.** `develop --fixed-selection <seed-0 bundle> --model-seed S`
  skips the grid search and refits the configuration and aggregation that seed 0 selected, on
  the same fit/validation partitions (meta-split seed 0), so only training randomness changes.
  Task normalization is refit on the same meta-validation rollouts, as for seed 0.
- **Seed-0 gate.** The same command with `--model-seed 0` must reproduce the original
  development run exactly (validation scores max diff 0.0, equal normalization and selection
  score), or it stops. The scorer code has not changed since `d71b6d27`.
- **Same calibration and test.** Each seed is calibrated on the 228-rollout primary calibration
  cohort (event-level FPR <= 0.05) and evaluated once on the 760-rollout primary cohort
  (380/380), exactly as seed 0. Both steps also check that the frozen SAFE/time results are
  reproduced.
- **Summary.** `scripts/xr1_lead_time/ood_per_seed_summary.py` reads the three
  `evaluate/analysis.json` files, checks that seed 0 reproduces Table I (TPR/FPR), and prints
  mean ± std (ddof 1) of bACC, TWA = (1 − DT + 1 − FPR)/2 and DT.

Status of the data: the primary cohort was opened before; this adds per-seed point estimates of
already-reported baselines, not a new test. Nothing is selected from test results.

Cost: CPU only; caches already exist. `develop` with one configuration takes minutes per seed;
`calibrate` (joint threshold searches for the staged detectors) dominates. A `cpu_16`
allocation is enough (the caches are about 0.3–1 GB each); check `maxrss` as below.

## Commands (TSUBAME, CPU only)

`OOD_SHA` is the full SHA of the pushed commit that contains this file.

### 0. Session, pinned worktree, inputs

```bash
qrsh -g tga-shinoda -l cpu_16=1 -l h_rt=8:00:00
```

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_ENV=$STORAGE_BS/envs/vla_safe SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export XR1_MATCHED_FPR_TAG=xr1_prospective_matched_fpr_20260815_163700
export XR1_CAL_ANALYSIS=$STORAGE_BS/robocasa_checkpoints/safe/${XR1_MATCHED_FPR_TAG}_calibration
export XR1_TEST_ANALYSIS=$STORAGE_BS/robocasa_checkpoints/safe/${XR1_MATCHED_FPR_TAG}_prospective_test
export XR1_RUNTIME_BUNDLE=$XR1_CAL_ANALYSIS/runtime_bundle.json
export XR1_FROZEN_ANALYSIS=$XR1_TEST_ANALYSIS/analysis.json
export XR1_BASELINE_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/xr1_offline_baselines_20261017_100820

export OOD_SHA=<OOD_SHA>
export OOD_REPO=$PROJECT_FS/robocasa_oodseed_${OOD_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/time-aware-fusion-paper
test -e $OOD_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $OOD_REPO $OOD_SHA
test "$(git -C $OOD_REPO rev-parse HEAD)" = "$OOD_SHA" || echo "STOP: worktree HEAD is not OOD_SHA"

export OOD_TAG=xr1_ood_per_seed_$(date +%Y%m%d_%H%M%S)
export OOD_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$OOD_TAG
export OOD_LOG=$STORAGE_BS/robocasa_logs/eval/$OOD_TAG
mkdir -p $OOD_ROOT $OOD_LOG
printf 'export %s=%q\n' OOD_SHA "$OOD_SHA" OOD_REPO "$OOD_REPO" OOD_TAG "$OOD_TAG" \
  OOD_ROOT "$OOD_ROOT" OOD_LOG "$OOD_LOG" XR1_BASELINE_ROOT "$XR1_BASELINE_ROOT" > $OOD_LOG/paths.env

for f in $XR1_RUNTIME_BUNDLE $XR1_FROZEN_ANALYSIS \
         $XR1_BASELINE_ROOT/develop/baseline_bundle.json $XR1_BASELINE_ROOT/develop/validation_scores.jsonl \
         $XR1_BASELINE_ROOT/evaluate/analysis.json; do
  test -s $f && echo "OK $f" || echo "STOP: missing $f"
done
for c in development calibration prospective_test; do
  test -s $XR1_BASELINE_ROOT/cache_$c/provenance.json && echo "OK cache_$c" || echo "STOP: missing cache_$c"
done
(cd $XR1_BASELINE_ROOT && find develop calibrate evaluate -type f -print0 | sort -z | xargs -0 sha256sum) \
  > $OOD_LOG/seed0_outputs.sha256
wc -l $OOD_LOG/seed0_outputs.sha256

cd $OOD_REPO
CUDA_VISIBLE_DEVICES="" nice -n 10 "$SAFE_PY" -m unittest tests.test_safe_offline_detector_baselines 2>&1 | tail -n 3
```

Continue only if every input is `OK` and the tests end with `OK` (16 tests).

### 1. Seed-0 gate (minutes)

```bash
cd $OOD_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines develop \
  --development-cache $XR1_BASELINE_ROOT/cache_development \
  --runtime-bundle $XR1_RUNTIME_BUNDLE \
  --fixed-selection $XR1_BASELINE_ROOT/develop/baseline_bundle.json \
  --model-seed 0 \
  --output-dir $OOD_ROOT/seed0/develop 2>&1 | tee $OOD_LOG/develop_seed0.log | grep -E 'fixed selection|error'
```

It must print four lines `fixed selection <baseline>: validation scores max diff 0.0,
normalization equal -> OK` (mahalanobis, knn_cosine, pca_kmeans, rnd) and no `error`. Otherwise
stop and report the log: seed 0 is then not reproducible with this code and the per-seed rows
would not be comparable.

### 2. Seeds 1 and 2: develop, calibrate, evaluate (in parallel)

```bash
cd $OOD_REPO
for S in 1 2; do
  cat > $OOD_LOG/run_seed$S.sh <<EOF
set -e
cd $OOD_REPO
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines develop \\
  --development-cache $XR1_BASELINE_ROOT/cache_development --runtime-bundle $XR1_RUNTIME_BUNDLE \\
  --fixed-selection $XR1_BASELINE_ROOT/develop/baseline_bundle.json --model-seed $S \\
  --output-dir $OOD_ROOT/seed$S/develop
"$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines calibrate \\
  --baseline-bundle $OOD_ROOT/seed$S/develop/baseline_bundle.json \\
  --calibration-cache $XR1_BASELINE_ROOT/cache_calibration \\
  --output-dir $OOD_ROOT/seed$S/calibrate --workers 6
"$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines evaluate \\
  --baseline-calibration $OOD_ROOT/seed$S/calibrate/baseline_calibration.json \\
  --test-cache $XR1_BASELINE_ROOT/cache_prospective_test --frozen-analysis $XR1_FROZEN_ANALYSIS \\
  --output-dir $OOD_ROOT/seed$S/evaluate --test-per-class 10 --bootstrap-samples 2000 --bootstrap-seed 0
echo "SEED $S DONE"
EOF
  nohup nice -n 10 bash $OOD_LOG/run_seed$S.sh > $OOD_LOG/run_seed$S.log 2>&1 &
  echo "seed $S PID: $!"
done
```

Monitor (peak memory of the allocation, then the end of each log):

```bash
qstat -j $JOB_ID | grep -o 'maxrss=[^,]*'; tail -n 3 $OOD_LOG/run_seed1.log $OOD_LOG/run_seed2.log
```

Each log must end with `SEED S DONE`. `calibrate/status.json` must report
`frozen_thresholds_reproduced: true` and `evaluate` must print
`"frozen_results_reproduced": true`; an `error:` line is a stop condition.

### 3. Summary (seconds)

```bash
cd $OOD_REPO
"$SAFE_PY" -I -u scripts/xr1_lead_time/ood_per_seed_summary.py \
  --analysis 0=$XR1_BASELINE_ROOT/evaluate/analysis.json \
             1=$OOD_ROOT/seed1/evaluate/analysis.json 2=$OOD_ROOT/seed2/evaluate/analysis.json \
  --expect mahalanobis=0.666,0.068 knn_cosine=0.582,0.055 pca_kmeans=0.605,0.063 rnd=0.629,0.068 \
  --output $OOD_ROOT/ood_per_seed_summary.json 2>&1 | tee $OOD_LOG/summary.log
(cd $XR1_BASELINE_ROOT && sha256sum -c --quiet $OOD_LOG/seed0_outputs.sha256) && echo "SEED 0 OUTPUTS UNCHANGED"
```

Report back: the four `fixed selection ... -> OK` lines from step 1, the four `reference ...
-> OK` lines, the per-seed and `mean+-std` lines and `REFERENCE OK` from `summary.log`, and
`SEED 0 OUTPUTS UNCHANGED`.

Not covered: Table II's OOD + P rows (held-out set) still use the seed-0 models. Adding seeds
there needs each seed's models scored on the held-out cohorts (`score-baselines`) and then
`per_seed_detection.py --fused`.
