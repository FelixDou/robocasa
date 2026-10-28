# OOD baselines per seed (RSS Tables I and II, mean ± std over 3 model seeds)

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
  development run, or it stops: normalized validation scores within 1e-6 (absolute), task
  normalization and selection score within 1e-6 (relative). The scorer code has not changed
  since `d71b6d27`. A bit-exact check (2026-10-27) passed for Mahalanobis and k-NN, but
  PCA-KMeans differed by 1.9e-8 with an identical selection score: multithreaded k-means sums
  in a thread-count-dependent order, and this run's thread count differs from the original's.
- **Seed 0 end to end.** Seed 0 is refit and run through calibration and evaluation like seeds
  1 and 2, so all three seeds come from the same code and settings, and the summary checks that
  this refit seed 0 reproduces Table I's TPR/FPR.
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

The first, bit-exact attempt (2026-10-27) left a partial `seed0/develop`; keep it as evidence.

```bash
cd $OOD_REPO
test -e $OOD_ROOT/seed0/develop && mv $OOD_ROOT/seed0/develop $OOD_ROOT/seed0/develop_bitexact_failed
test -e $OOD_LOG/develop_seed0.log && mv $OOD_LOG/develop_seed0.log $OOD_LOG/develop_seed0_bitexact_failed.log
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines develop \
  --development-cache $XR1_BASELINE_ROOT/cache_development \
  --runtime-bundle $XR1_RUNTIME_BUNDLE \
  --fixed-selection $XR1_BASELINE_ROOT/develop/baseline_bundle.json \
  --model-seed 0 --reproduction-tolerance 1e-6 \
  --output-dir $OOD_ROOT/seed0/develop 2>&1 | tee $OOD_LOG/develop_seed0.log | grep -E 'fixed selection|error'
```

It must print four lines `fixed selection <baseline>: ... (tolerance 1e-06) -> OK`
(mahalanobis, knn_cosine, pca_kmeans, rnd) and no `error`. Otherwise stop and report the log.

### 2. Seeds 0, 1, 2: develop, calibrate, evaluate (in parallel)

Seed 0 reuses the bundle from step 1; seeds 1 and 2 are refit here.

```bash
cd $OOD_REPO
for S in 0 1 2; do
  if test $S = 0; then DEVELOP="echo seed 0 bundle from step 1"; else DEVELOP="\"$SAFE_PY\" -u -m robocasa.recovery.safe.run_offline_detector_baselines develop --development-cache $XR1_BASELINE_ROOT/cache_development --runtime-bundle $XR1_RUNTIME_BUNDLE --fixed-selection $XR1_BASELINE_ROOT/develop/baseline_bundle.json --model-seed $S --output-dir $OOD_ROOT/seed$S/develop"; fi
  cat > $OOD_LOG/run_seed$S.sh <<EOF
set -e
cd $OOD_REPO
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$DEVELOP
test -s $OOD_ROOT/seed$S/develop/baseline_bundle.json
"$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines calibrate \\
  --baseline-bundle $OOD_ROOT/seed$S/develop/baseline_bundle.json \\
  --calibration-cache $XR1_BASELINE_ROOT/cache_calibration \\
  --output-dir $OOD_ROOT/seed$S/calibrate --workers 5
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

Monitor until the three runs end (peak memory, then the last line of each log, every 2 minutes):

```bash
while pgrep -f "run_seed[012].sh" > /dev/null; do echo "$(date +%H:%M) $(qstat -j $JOB_ID | grep -o 'maxrss=[^,]*')"; tail -qn 1 $OOD_LOG/run_seed{0,1,2}.log; sleep 120; done; tail -n 3 $OOD_LOG/run_seed{0,1,2}.log
```

Each log must end with `SEED S DONE`, and `evaluate` must print
`"frozen_results_reproduced": true`. An `error:` line is a stop condition.

### 3. Summary (seconds)

```bash
cd $OOD_REPO
"$SAFE_PY" -I -u scripts/xr1_lead_time/ood_per_seed_summary.py \
  --analysis 0=$OOD_ROOT/seed0/evaluate/analysis.json \
             1=$OOD_ROOT/seed1/evaluate/analysis.json 2=$OOD_ROOT/seed2/evaluate/analysis.json \
  --expect mahalanobis=0.666,0.068 knn_cosine=0.582,0.055 pca_kmeans=0.605,0.063 rnd=0.629,0.068 \
  --output $OOD_ROOT/ood_per_seed_summary.json 2>&1 | tee $OOD_LOG/summary.log
(cd $XR1_BASELINE_ROOT && sha256sum -c --quiet $OOD_LOG/seed0_outputs.sha256) && echo "SEED 0 OUTPUTS UNCHANGED"
```

Report back: the four `fixed selection ... -> OK` lines from step 1, the four `reference ...
-> OK` lines, the per-seed and `mean+-std` lines and `REFERENCE OK` from `summary.log`, and
`SEED 0 OUTPUTS UNCHANGED`.

## Table II: OOD + P per seed (held-out set)

Table II's OOD + P rows come from report Section 15.14 (`dose_response.py`): the seed-0 models'
trajectories were attached to the held-out E1 cohorts with `score-baselines` and each score was
OR-fused with P, fitted and evaluated like SAFE + P (q = 0.95; 10 + 10 per task in calibration
and test). Here each seed's models from steps 1–2 are attached the same way, and
`scripts/xr1_lead_time/ood_per_seed_fused.py` fuses and evaluates them with the same functions
(`dose_response.fused_gain`). It checks that P alone is identical for every seed
(lead 0.2252, FPR 0.0553), that seed 0 reproduces the original `$E1_OUT/dose_response.json`
exactly at its stored 4-decimal precision, and that it matches Table II's gain and FPR to within
half a unit of the last digit, then prints mean ± std (ddof 1) of FPR, mean lead, gain over P
and TWA. (A first attempt, 2026-10-27, compared Python-rounded values with the table and
stopped on PCA-KMeans: stored FPR 0.0605, 23 of 380, which the table shows as 0.061 but
`round(0.0605, 3)` gives 0.060.) Both held-out cohorts were opened
before; this adds per-seed point estimates, not a new test.

Prerequisite: steps 1–2 done (`$OOD_ROOT/seed{0,1,2}/calibrate/baseline_calibration.json`).

### 4. Attach each seed's trajectories to the held-out cohorts

```bash
source "$(ls -d $STORAGE_BS/robocasa_logs/eval/xr1_ood_per_seed_* | sort | tail -n 1)/paths.env"
export XR1_LCAL_ROOT=$STORAGE_BS/robocasa_rollouts/safe/xr1_lead_calibration_20261020_001108
export XR1_CONFIRM_ROOT=$STORAGE_BS/robocasa_rollouts/safe/xr1_confirmatory_fused_20261017_160220
export E1_OUT=$XR1_LCAL_ROOT/lead_analysis
for f in $E1_OUT/calibration_predictions.jsonl $E1_OUT/test_predictions.jsonl \
         $E1_OUT/calibration_with_baselines.jsonl $E1_OUT/test_with_baselines.jsonl \
         $XR1_LCAL_ROOT/cache_lead_calibration/provenance.json \
         $XR1_CONFIRM_ROOT/cache_confirmatory_test/provenance.json \
         $OOD_ROOT/seed0/calibrate/baseline_calibration.json $OOD_ROOT/seed1/calibrate/baseline_calibration.json \
         $OOD_ROOT/seed2/calibrate/baseline_calibration.json; do
  test -s $f && echo "OK $f" || echo "STOP: missing $f"
done
sha256sum $E1_OUT/calibration_predictions.jsonl $E1_OUT/test_predictions.jsonl \
  $E1_OUT/calibration_with_baselines.jsonl $E1_OUT/test_with_baselines.jsonl > $OOD_LOG/heldout_inputs.sha256

cd $OOD_REPO
cat > $OOD_LOG/attach_heldout.sh <<EOF
set -e
cd $OOD_REPO
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for S in 0 1 2; do
  mkdir -p $OOD_ROOT/seed\$S/heldout
  for cohort in calibration test; do
    if test \$cohort = calibration; then cache=$XR1_LCAL_ROOT/cache_lead_calibration; group=lead_calibration
    else cache=$XR1_CONFIRM_ROOT/cache_confirmatory_test; group=confirmatory_test; fi
    "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines score-baselines \\
      --baseline-calibration $OOD_ROOT/seed\$S/calibrate/baseline_calibration.json \\
      --predictions $E1_OUT/\${cohort}_predictions.jsonl --cache \$cache --cache-group \$group \\
      --output $OOD_ROOT/seed\$S/heldout/\${cohort}_with_baselines.jsonl > /dev/null
    echo "seed \$S \$cohort attached"
  done
done
echo "ATTACH DONE"
EOF
nohup nice -n 10 bash $OOD_LOG/attach_heldout.sh > $OOD_LOG/attach_heldout.log 2>&1 &
echo "attach PID: $!"
```

Monitor with `tail -n 3 $OOD_LOG/attach_heldout.log`; it must end with `ATTACH DONE`.

Optional diagnostic: how far the refit seed-0 trajectories are from the originals used by
Table II (expected 0 or float noise below 1e-6):

```bash
"$SAFE_PY" - $E1_OUT $OOD_ROOT/seed0/heldout <<'PY'
import json, sys, numpy as np
for cohort in ("calibration", "test"):
    load = lambda p: {r["rollout_id"]: r["baseline_scores"] for r in (json.loads(l) for l in open(p) if l.strip())}
    a = load(f"{sys.argv[1]}/{cohort}_with_baselines.jsonl"); b = load(f"{sys.argv[2]}/{cohort}_with_baselines.jsonl")
    assert set(a) == set(b), cohort
    for n in sorted(next(iter(a.values()))):
        d = max(float(np.max(np.abs(np.asarray(a[k][n], float) - np.asarray(b[k][n], float)))) for k in a)
        print(f"{cohort:12s} {n:12s} max |refit - original| {d:.3g}")
PY
```

### 5. Per-seed fused metrics (seconds)

```bash
cd $OOD_REPO
test -s $E1_OUT/dose_response.json && echo "OK dose_response.json" || echo "STOP: missing $E1_OUT/dose_response.json"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/ood_per_seed_fused.py \
  --runtime-bundle $XR1_RUNTIME_BUNDLE --reference $E1_OUT/dose_response.json \
  --seed 0 $OOD_ROOT/seed0/heldout/calibration_with_baselines.jsonl $OOD_ROOT/seed0/heldout/test_with_baselines.jsonl \
  --seed 1 $OOD_ROOT/seed1/heldout/calibration_with_baselines.jsonl $OOD_ROOT/seed1/heldout/test_with_baselines.jsonl \
  --seed 2 $OOD_ROOT/seed2/heldout/calibration_with_baselines.jsonl $OOD_ROOT/seed2/heldout/test_with_baselines.jsonl \
  --expect mahalanobis=0.034,0.055 knn_cosine=0.021,0.058 pca_kmeans=0.022,0.061 rnd=0.020,0.058 \
  --output $OOD_ROOT/ood_per_seed_fused_heldout.json 2>&1 | tee $OOD_LOG/fused_summary.log
sha256sum -c $OOD_LOG/heldout_inputs.sha256 && echo "HELD-OUT INPUTS UNCHANGED"
```

Report back: the `P alone` line, the four `original ... -> OK` and four `reference ... -> OK`
lines, the per-seed and
`mean+-std` lines, `REFERENCE OK` and `HELD-OUT INPUTS UNCHANGED` (and the diagnostic, if run).
