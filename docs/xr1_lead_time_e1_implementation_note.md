# E1: implementation note and runbook for Amendment 1 (lead time over a per-task timer)

This note fixes the details that
[`xr1_confirmatory_amendment_1_lead_time.md`](xr1_confirmatory_amendment_1_lead_time.md)
left open. It does not change the amendment: the hypotheses, comparator, endpoint, margins
and FPR limits there are unchanged, and that file stays byte-identical. The note is
committed before any lead-time, timer or P quantity is computed on the confirmatory cohort.

## Status of the confirmatory cohort when this note was written (2026-10-20)

- Amendment 1 was opened as FelixDou/robocasa PR #3 on 2026-10-17 13:06:46 UTC
  (22:06 JST).
- The confirmatory cohort (base seed 1300007) was first scored by
  `run_offline_detector_baselines confirm`, which wrote `confirm/analysis.json`
  at 2026-10-19 20:51:54 JST. The amendment therefore came before any
  outcome on that cohort was computed.
- That run reported the original recall endpoint (report §15.11), so recall and FPR of the
  frozen time-risk, staged and fused detectors on this cohort have been seen.
- The user confirmed on 2026-10-20 that no timer, P, lead or SAFE + P quantity has been
  computed on the confirmatory cohort.

## Fixed details

1. **Calibration cohort.**
   - Same SAFE-patched Xiaomi-Robotics-1 server, 38 tasks and two shards as the
     confirmatory collection.
   - Collector code from the pinned worktree `robocasa_xr1_confirmatory_4e82e5c8`, so the
     collection code is identical to the confirmatory cohort's.
   - Official Xiaomi seed protocol with base seed **1400007** and 500 attempts per task,
     so seeds lie in [1400007, 1425007).
   - The first 10 successes and 10 failures per task in (seed, reset, rollout ID) order,
     with `--no-retain-only-quota`. Videos are not recorded.
   - Before collection, every Xiaomi manifest on `/gs/bs` is scanned, and the range must
     have no prior use. If a task cannot reach 10/10 within 500 attempts, stop and report.
2. **Scores.**
   - Both cohorts are scored by the frozen seed-0/1/2 SAFE checkpoints
     (`score_seen_checkpoint_external`), with group `lead_calibration` for the new cohort.
   - Both are assembled by `run_prospective_matched_fpr export` with the frozen
     `runtime_bundle.json`. The export computes no threshold, alarm or metric.
   - Export refuses the following:
     - checkpoint or score-group mismatches;
     - incomplete task coverage;
     - seeds outside the declared range;
     - any rollout ID or (task, seed, reset) identity shared with development, the
       August calibration, or the files passed to `--disjoint-from`.
   - The test export must equal `confirm/event_predictions.jsonl` rollout by rollout
     (`--confirm-predictions`).
3. **Analysis code.**
   - The analysis is `scripts/xr1_lead_time/confirm_lead.py`.
   - Its detector definitions are those of the registered `quantile_timer.py`: the same u
     grid, s grid, conformal rank, tie-breaking and epsilon.
   - `confirm_lead.py reproduce` must print `REPRODUCTION OK` on the opened August review
     data before the run. That requires it to match `quantile_timer.json` exactly at every
     q, and to make identical refits on 25 resampled calibration sets.
   - `lead_time.py` and `quantile_timer.py` were only refactored to be importable, and
     their command-line output is byte-identical.
4. **Decision bootstrap (H1 CI).**
   - 2,000 replicates with seed 0.
   - In each replicate the 38 tasks are drawn once with replacement and used for both
     cohorts. Within each drawn task, calibration successes, calibration failures, test
     successes and test failures are each resampled with replacement.
   - P and SAFE + P are refitted on the resampled calibration cohort, both are evaluated
     on the resampled test cohort, and the replicate gain is the mean lead difference over
     the resampled test failures.
   - The 95% CI is the 2.5–97.5 percentile interval.
   - The development quantiles rho_k are frozen and not resampled.
   - The test-only bootstrap with fixed thresholds is also reported, as in the dry run.
5. **H2 (null scores).**
   - Two nulls, each over 20 draws with seed 1:
     - noise: the running maximum of i.i.d. N(0, 1), one value per inference;
     - within-task resampled SAFE: per-seed raw SAFE scores drawn from all inferences of
       the same task in the same cohort, ignoring outcomes, then normalized with the frozen
       per-task normalization and averaged over seeds.
   - Nulls are generated separately for each cohort from its primary rows.
   - Each draw refits the fused path and the conformal margin exactly as SAFE + P, with
     P's rho_k.
   - H2 is supported if, for each null, the **mean** gain over P across the 20 draws is
     below 0.05 H_k. The maximum over draws is reported.
6. **Primary selection.** In both cohorts, the first 10 successes and 10 failures per task
   in (seed, reset, rollout ID) order. Where the input carries a `primary_balanced` flag,
   the selection must agree with it.

Everything else in the output is descriptive:
- the quantiles 0.75, 0.90 and 1.0;
- the global timer, SAFE + global timer, time risk + global timer, and the old joint
  search over SAFE and time risk;
- frozen SAFE + time risk;
- timers tuned on the test cohort to the fused detector's test FPR;
- the atomic and composite strata.

## Run log

**2026-10-21, first run at `cb5ad443`.**
- Collection ended at 1,522 rollouts (925 successes, 597 failures, all 38 tasks at 10/10),
  with 0 errors and every seed in [1400007, 1425007).
- Both exports passed: 760 primary rollouts each, with 1,522 and 1,526 retained.
- `reproduce` printed `REPRODUCTION OK`.
- `run` printed the four per-q point estimates, then stopped with
  `ValueError: Unexpected frozen fused threshold keys: ['early_safe', 'late_time']`.
- The cause was the descriptive "frozen SAFE + time risk" row: the frozen calibration
  stores the fused thresholds under the source names of `prospective_matched_fpr`, which
  this code did not expect.
- No output file was written, and the input checksums were unchanged.
- The printed point estimates were:

| q | P lead (FPR) | SAFE + P lead (FPR) | Gain |
|---|---|---|---|
| 0.75 | 0.2186 (0.0447) | 0.2449 (0.0474) | 0.0263 |
| 0.90 | 0.2229 (0.0605) | 0.2512 (0.0579) | 0.0284 |
| 0.95 (primary) | 0.2252 (0.0553) | 0.2550 (0.0605) | 0.0298 |
| 1.00 | 0.1639 (0.0368) | 0.2376 (0.0500) | 0.0737 |

**The fix.**
- It changes only the key mapping of that descriptive row:
  - `early_safe` maps to the whole SAFE trajectory;
  - `late_time` maps to the whole time-risk trajectory;
  - both alarm on `>=`, as in `prospective_matched_fpr.first_alert`.
- Every primary, bootstrap and null computation is unchanged and deterministic (fixed
  seeds), so the rerun must print the same four lines. A mismatch is a stop condition.

## Runbook

Run on the cluster. This follows
[`cluster_experiment_runbook.md`](cluster_experiment_runbook.md) and the
pinned-worktree pattern; never pull in the shared checkout.

### 1. Collection (two GPUs)

Use the exports, seed-range scan, server and collector commands given for this run:
- worktree `robocasa_xr1_confirmatory_4e82e5c8`;
- ports 10118/10119;
- root `$STORAGE_BS/robocasa_rollouts/safe/xr1_lead_calibration_<timestamp>`, whose
  variables are saved in `session.env`.

Monitor with the one-shot monitor of
[`xr1_confirmatory_fused_test.md`](xr1_confirmatory_fused_test.md) §5, using
`$XR1_LCAL_ROOT`. Resume interrupted shards with `--resume` added to the same collector
command. When all 38 tasks are ready, stop the servers.

### 2. Exports for this stage

`<E1_SHA>` is the full SHA of the pushed commit that contains this note.

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_REPO=$PROJECT_FS/SAFE SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export XR1_CLIENT_ENV=$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365
export WANDB_MODE=disabled WANDB_DISABLED=true
export XR1_CAL_ANALYSIS=$STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_calibration
export XR1_RUNTIME_BUNDLE=$XR1_CAL_ANALYSIS/runtime_bundle.json
export XR1_FUSED_CALIBRATION=$STORAGE_BS/robocasa_checkpoints/safe/xr1_offline_baselines_20261017_100820/calibrate_fused/baseline_calibration.json
export XR1_CONFIRM_ROOT=$STORAGE_BS/robocasa_rollouts/safe/xr1_confirmatory_fused_20261017_160220
export XR1_AUGUST_TEST=$STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_prospective_test/event_predictions.jsonl
source $STORAGE_BS/robocasa_rollouts/safe/xr1_lead_calibration_<timestamp>/session.env
export E1_SHA=<E1_SHA>
export E1_REPO=$PROJECT_FS/robocasa_e1_lead_${E1_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin
test -e $E1_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $E1_REPO $E1_SHA
test "$(git -C $E1_REPO rev-parse HEAD)" = "$E1_SHA" || echo "STOP: worktree HEAD is not E1_SHA"
mkdir -p $E1_REPO/robocasa/models/assets
cp -aln $PROJECT_FS/robocasa/robocasa/models/assets/. $E1_REPO/robocasa/models/assets/
export E1_OUT=$XR1_LCAL_ROOT/lead_analysis; mkdir -p $E1_OUT
```

### 3. Merge, validate, export and score the calibration cohort

```bash
cd $E1_REPO
for shard in shard0 shard1; do
  $XR1_CLIENT_ENV/bin/python -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir $XR1_LCAL_ROOT/$shard
done
$XR1_CLIENT_ENV/bin/python -m robocasa.recovery.safe.merge_atomic_datasets \
  --source-dirs $XR1_LCAL_ROOT/shard0 $XR1_LCAL_ROOT/shard1 --output-dir $XR1_LCAL_ROOT/merged_all_retained
$XR1_CLIENT_ENV/bin/python -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir $XR1_LCAL_ROOT/merged_all_retained
$XR1_CLIENT_ENV/bin/python -m robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir $XR1_LCAL_ROOT/merged_all_retained --output-dir $XR1_LCAL_ROOT/official_safe_all_retained
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 nice -n 10 $SAFE_PY -u -m robocasa.recovery.safe.score_seen_checkpoint_external \
    --export-dir $XR1_LCAL_ROOT/official_safe_all_retained --safe-repo $SAFE_REPO \
    --training-run-dir $XR1_CAL_ANALYSIS/runtime/seed$seed \
    --output-dir $XR1_LCAL_ROOT/frozen_mlp_scores/seed$seed \
    --group lead_calibration --device cuda
done
```

### 4. Assemble both cohorts (no metrics)

```bash
cd $E1_REPO
sha256sum $XR1_RUNTIME_BUNDLE $XR1_FUSED_CALIBRATION $XR1_CONFIRM_ROOT/confirm/event_predictions.jsonl | tee $E1_OUT/inputs_sha256_before.txt
CUDA_VISIBLE_DEVICES="" nice -n 10 $SAFE_PY -u -m robocasa.recovery.safe.run_prospective_matched_fpr export \
  --runtime-bundle $XR1_RUNTIME_BUNDLE --score-root $XR1_LCAL_ROOT/frozen_mlp_scores \
  --score-group lead_calibration --seed-range 1400007 1425007 \
  --disjoint-from $XR1_CONFIRM_ROOT/confirm/event_predictions.jsonl $XR1_AUGUST_TEST \
  --output $E1_OUT/calibration_predictions.jsonl
CUDA_VISIBLE_DEVICES="" nice -n 10 $SAFE_PY -u -m robocasa.recovery.safe.run_prospective_matched_fpr export \
  --runtime-bundle $XR1_RUNTIME_BUNDLE --score-root $XR1_CONFIRM_ROOT/confirmatory_collection/frozen_mlp_scores \
  --score-group confirmatory_test --seed-range 1300007 1325007 \
  --disjoint-from $XR1_AUGUST_TEST \
  --output $E1_OUT/test_predictions.jsonl
```

### 5. Reproduction check, then the one confirmatory analysis

The reproduction check reads the opened August calibration and test predictions (already
opened, so this reveals nothing new). Run the confirmatory analysis only after it prints
`REPRODUCTION OK`.

```bash
cd $E1_REPO
mkdir -p $E1_OUT/august_review_data
ln -sf $XR1_CAL_ANALYSIS/calibration_predictions.jsonl $XR1_AUGUST_TEST $XR1_RUNTIME_BUNDLE $E1_OUT/august_review_data/
nice -n 10 $SAFE_PY -I scripts/xr1_lead_time/confirm_lead.py reproduce $E1_OUT/august_review_data
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
nice -n 10 $SAFE_PY -I -u scripts/xr1_lead_time/confirm_lead.py run \
  --calibration $E1_OUT/calibration_predictions.jsonl \
  --test $E1_OUT/test_predictions.jsonl \
  --confirm-predictions $XR1_CONFIRM_ROOT/confirm/event_predictions.jsonl \
  --runtime-bundle $XR1_RUNTIME_BUNDLE --fused-calibration $XR1_FUSED_CALIBRATION \
  --output $E1_OUT/confirm_lead.json 2>&1 | tee $STORAGE_BS/robocasa_logs/eval/$(basename $XR1_LCAL_ROOT)_confirm_lead.log
sha256sum $XR1_RUNTIME_BUNDLE $XR1_FUSED_CALIBRATION $XR1_CONFIRM_ROOT/confirm/event_predictions.jsonl | diff - $E1_OUT/inputs_sha256_before.txt && echo INPUTS UNCHANGED
```

Report the final `H1:` / `H2:` line, the primary block, and `INPUTS UNCHANGED`. Any
checksum, overlap, seed-range, coverage or reproduction error is a stop condition: do not
edit inputs to make the run pass.
