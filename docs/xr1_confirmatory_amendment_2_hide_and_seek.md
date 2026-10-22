# Amendment 2: Hide-and-Seek + P against the per-task timer P

**Amends:** [`xr1_confirmatory_amendment_1_lead_time.md`](xr1_confirmatory_amendment_1_lead_time.md)
and its implementation note
[`xr1_lead_time_e1_implementation_note.md`](xr1_lead_time_e1_implementation_note.md).
Both files are left unchanged. Amendment 1's H1 and H2 for SAFE have been tested and
stand as reported (report §15.13).

**Date:** 2026-10-22. This amendment is committed **before** any Hide-and-Seek or
control score is computed on either E1 cohort:
- the lead-calibration cohort (seeds [1400007, 1425007),
  `xr1_lead_calibration_20261020_001108`);
- the confirmatory test cohort (seeds [1300007, 1325007),
  `xr1_confirmatory_fused_20261017_160220`).

"Computed" means any trajectory, threshold, alarm or metric, as in Amendment 1. The
detectors had only ever been run with the August cohorts as their evaluation set, and
the score-only tool used below did not exist before this commit.

## What has already been seen on these cohorts

- **SAFE.** Recall and FPR of the frozen time-risk, staged and fused detectors on the
  confirmatory cohort (report §15.11). SAFE + P against P on both cohorts (E1, §15.13):
  gain +0.030, decision CI [+0.007, +0.093], H1 not supported.
- **Videos.** 48 confirmatory failure videos were labeled for the point of no return
  (§15.17).
- **Exploratory baselines.** The dose-response analysis (§15.14) attached the frozen
  success-only baselines (Mahalanobis, kNN, PCA-KMeans, RND) to both cohorts.
- **Not seen.** No Hide-and-Seek, ActProbe or control quantity has been computed on
  either cohort.

## Reason

On the opened August cohorts (report §15.20), Hide-and-Seek + P added +0.083 [+0.022,
+0.143] of lead over P at matched FPR, about ten times SAFE. Two controls with the same
LSTM and training, on zero inputs and on same-task, same-step features of other
rollouts, gained about 0. So the gain was not elapsed time or task identity. That
result is exploratory:
- Hide-and-Seek was one of two supervised baselines compared;
- it used the physical-signals fitting grid and a test-only bootstrap;
- the cohort was already open.

This amendment tests it once, on cohorts Hide-and-Seek has never scored, with the
registered E1 procedure.

## Frozen detector

- **Code.** `hide_and_seek` in `robocasa/recovery/safe/baselines/detectors.py` at the
  commit that contains this file.
- **Settings:**
  - PCA to 128 dimensions;
  - LSTM hidden size 128;
  - peak margin 0.2 and onset weight 1.0;
  - AdamW with learning rate 1e-3, weight decay 1e-4 and batch size 32;
  - 200 epochs, seeds 0, 1 and 2, CPU.
- **Inputs:** the per-inference DiT vector read by the frozen SAFE MLP (feature
  aggregation `last`: last flow step, last action position).
- **Training rollouts:** the 1,824 final-refit training rollouts (`split == "train"` in
  `xr1_pooled_39tasks_final_refits_20260814_231508/indep_seed0/scores.jsonl`), the same
  set as the August runs.
- **Scores:** a running maximum, normalized per task from training maxima and averaged
  over the three seeds, as in `run_baselines`.

**Integrity check.** The detectors are retrained for this run. On the August
calibration (228) and test (1,643) rollouts, Hide-and-Seek, zero-input and shuffle
trajectories must equal those of the §15.20 runs exactly (largest difference 0). If any
differs, stop: the frozen detector was not reproduced.

## Cohorts and scoring

- **Calibration:** the lead-calibration cohort, primary 10 successes and 10 failures per
  task (380/380). Selection, P and the fused path are fitted as in E1.
- **Test:** the confirmatory cohort, primary 380/380, with the E1 checks:
  - it matches `confirm/event_predictions.jsonl` rollout by rollout;
  - it is disjoint from calibration;
  - every frozen task is covered.
- **Scoring:** `robocasa.recovery.safe.baselines.score_cohorts` writes trajectories for
  all retained rollouts of both cohorts. It computes no threshold, alarm or metric.

## Hypotheses and decision rules

**Prerequisite (no decision without it).** The analysis recomputes SAFE + P against P on
the same cohorts with the same code. It must equal the registered E1 result
(`confirm_lead.json`, q = 0.95) to the stored rounding:
- P mean lead;
- SAFE + P mean lead;
- the gain;
- the decision-bootstrap CI.

If it does not, no H1 or H2 decision is reported.

**H1-HS (primary).** Hide-and-Seek + P adds warning lead over P. The detector, timer,
fused path, margin, endpoint and bootstrap are Amendment 1's, with Hide-and-Seek in
place of SAFE:
- P is the per-task quantile timer at q = 0.95, with its conformal margin set on
  calibration successes (alpha = 0.05).
- Hide-and-Seek + P uses the OR rule on the nested path (u, s), chosen on calibration
  failures over the registered grid: u over the 0, 0.05, ..., 1 quantiles of the
  failures' maximum scores, s in {0, 1, 2, 5, 10, 20, 50}.
- The endpoint is the mean of (H_k − T)/H_k over test failures, with a miss counted as 0.
- The decision bootstrap draws the 38 tasks once for both cohorts and resamples rollouts
  within each task and class. It refits P and the fused detector in every replicate
  (2,000 replicates, seed 0).

H1-HS is **supported** if all of these hold:
1. the decision-bootstrap 95% CI of the gain lies above 0;
2. the point gain is at least 0.05 H_k;
3. Hide-and-Seek + P has a test FPR of at most 0.075;
4. its Wilson 95% upper bound is at most 0.10.

**H2-HS.** Uninformative scores fitted the same way do not reach the margin. There are
two nulls, 20 draws each, seed 1:
- the running maximum of i.i.d. N(0, 1);
- within-task resampled Hide-and-Seek: per-seed raw scores drawn from all inferences of
  the same task in the same cohort, ignoring outcomes, then normalized with the frozen
  per-task normalization and averaged over seeds.

H2-HS is **supported** if each null's mean gain over P is below 0.05 H_k.

## Declared secondary analyses

These do not enter H1-HS. Each is reported with its rule regardless of the H1-HS
outcome. A claim about mechanism is made only if H1-HS is supported.

- **S1, episode-difficulty prior.** `hide_and_seek_first` gives every inference the
  rollout's own first-inference features. It keeps the episode's initial appearance and
  the elapsed time, and removes what happens later.
  - Its gain over P is reported with its decision CI.
  - The rule: *the warning develops during the rollout* if the decision-bootstrap CI of
    (Hide-and-Seek + P) − (first + P) lies above 0.
  - Otherwise the gain is reported as consistent with a difficulty prior visible from
    the start.
- **S2, learned-timer replication.** `hide_and_seek_zero` and
  `hide_and_seek_task_shuffle`, as in §15.20.
  - The rule: the §15.20 conclusion (*not elapsed time or task identity*) replicates if
    both decision-bootstrap CIs of Hide-and-Seek + P minus the control + P lie above 0.
- **S3, against SAFE.** (Hide-and-Seek + P) − (SAFE + P), with its decision CI. This is
  descriptive.

All contrasts are paired: they use the same bootstrap replicates and P is refitted once
per replicate. Each detector's gain CI equals what `confirm_lead.boot_reselect` gives for
that score alone (checked by `tests/test_xr1_confirm_hide_and_seek.py`).

**Descriptive only:**
- the quantiles 0.75, 0.90 and 1.0;
- the atomic and composite strata;
- the test-only bootstrap;
- P tuned on the test cohort to Hide-and-Seek + P's test FPR.

## Dry run on the opened August cohorts

`confirm_hide_and_seek.py dry-run` runs the same analysis on the August calibration
(114/114) and test (380/380) cohorts, which are already open. SAFE + P must reproduce
`quantile_timer.json`. Its numbers are recorded here after the run, but they cannot
change anything above. The dry run uses the registered grid and decision bootstrap,
while §15.20 used the physical-signals grid and a test-only bootstrap, so its point gain
may differ from +0.098.

Dry run result: *to be filled in from the run log.*

## Expected outcome, written before scoring

- **The gain will likely shrink.** Hide-and-Seek was picked as the better of two
  baselines on an opened cohort, so the August +0.098 (+0.083 matched) is optimistic.
- **The cohorts differ.** SAFE's gain on the E1 cohorts (+0.030) was larger than on
  August (+0.007), so cohort differences cut both ways.
- **Expectation for H1-HS.** A point gain above 0 is expected. With E1's calibration
  size, the decision CI was about 0.09 wide for SAFE, so passing needs a point gain of
  about 0.05 or more with its lower bound above 0. The study does not predict whether
  that happens.
- **Expectation for S1.** No direction is predicted.

## How the outcome will be reported

- **H1-HS and S1 both supported:** a policy-internal score (Hide-and-Seek's objective on
  SAFE's features) adds at least 0.05 H_k of warning beyond a per-task timer, and the
  warning develops during the rollout.
- **H1-HS supported, S1 not:** the added warning is real but consistent with a
  difficulty prior visible at the first inference.
- **H1-HS not supported:** the exploratory gain did not replicate at the registered
  margin. It is reported as such, with the point gain and CI.

## Stop conditions

Any of these stops the run. Do not edit inputs to make it pass:
- an integrity-check mismatch;
- the SAFE reproduction failing;
- a checksum change in the frozen inputs;
- an overlap, seed-range or coverage error.

## Runbook

Run on the cluster from a pinned worktree, following
[`cluster_experiment_runbook.md`](cluster_experiment_runbook.md). Never pull in the
shared checkout. `AMEND2_SHA` is the full SHA of the pushed commit that contains this
file.

### 1. Environment, pinned code, inputs

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
source "$(ls -d $STORAGE_BS/robocasa_logs/eval/xr1_published_baselines_controls_* | sort | tail -n 1)/paths.env"

export AMEND2_SHA=<AMEND2_SHA>
export AMEND2_REPO=$PROJECT_FS/robocasa_amend2_${AMEND2_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin main
test -e $AMEND2_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $AMEND2_REPO $AMEND2_SHA
test "$(git -C $AMEND2_REPO rev-parse HEAD)" = "$AMEND2_SHA" || echo "STOP: worktree HEAD is not AMEND2_SHA"
cd $AMEND2_REPO

export XR1_LCAL_ROOT=$STORAGE_BS/robocasa_rollouts/safe/xr1_lead_calibration_20261020_001108
export XR1_CONFIRM_ROOT=$STORAGE_BS/robocasa_rollouts/safe/xr1_confirmatory_fused_20261017_160220
export E1_OUT=$XR1_LCAL_ROOT/lead_analysis
export AMEND2_TAG=xr1_amend2_hide_and_seek_$(date +%Y%m%d_%H%M%S)
export AMEND2_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$AMEND2_TAG
export AMEND2_LOG=$STORAGE_BS/robocasa_logs/eval/$AMEND2_TAG
mkdir -p $AMEND2_LOG $AMEND2_ROOT

for d in $XR1_LCAL_ROOT/merged_all_retained $XR1_CONFIRM_ROOT/confirmatory_collection/merged_all_retained; do
  test -f $d/manifest.jsonl && echo "raw OK: $d" || echo "STOP: no manifest in $d"
done
"$SAFE_PY" - "$XR1_RUNTIME_BUNDLE" "$AMEND2_ROOT/august_calibration_ids.json" <<'PY'
import json, sys
ids = json.load(open(sys.argv[1]))["primary_calibration_rollout_ids"]
json.dump(sorted(ids), open(sys.argv[2], "w")); print("August primary calibration IDs:", len(ids))
PY
sha256sum $XR1_RUNTIME_BUNDLE $XR1_TRAIN_IDS $XR1_EVAL_IDS $E1_OUT/calibration_predictions.jsonl \
  $E1_OUT/test_predictions.jsonl $E1_OUT/confirm_lead.json $XR1_CONFIRM_ROOT/confirm/event_predictions.jsonl \
  | tee $AMEND2_LOG/frozen_inputs.sha256
{
  printf 'export %s=%q\n' AMEND2_SHA "$AMEND2_SHA" AMEND2_REPO "$AMEND2_REPO" AMEND2_ROOT "$AMEND2_ROOT" \
    AMEND2_LOG "$AMEND2_LOG" XR1_LCAL_ROOT "$XR1_LCAL_ROOT" XR1_CONFIRM_ROOT "$XR1_CONFIRM_ROOT" E1_OUT "$E1_OUT"
  cat "$XR1_CTRL_LOG/paths.env"
} > $AMEND2_LOG/paths.env

PYTHONPATH=$AMEND2_REPO "$SAFE_PY" -m unittest tests.test_safe_baselines tests.test_safe_score_cohorts \
  tests.test_xr1_published_baselines_lead tests.test_xr1_confirm_hide_and_seek tests.test_xr1_confirm_lead
```

### 2. Score all cohorts (no metrics; several hours on CPU)

```bash
cd $AMEND2_REPO
SCORE_ARGS=(
  --dataset-dir "${XR1_RAW_DIRS[@]}" $XR1_LCAL_ROOT/merged_all_retained $XR1_CONFIRM_ROOT/confirmatory_collection/merged_all_retained
  --runtime-bundle $XR1_RUNTIME_BUNDLE --train-ids $XR1_TRAIN_IDS
  --cohort august_calibration $AMEND2_ROOT/august_calibration_ids.json
  --cohort august_test $XR1_EVAL_IDS
  --cohort lead_calibration $E1_OUT/calibration_predictions.jsonl
  --cohort confirmatory_test $E1_OUT/test_predictions.jsonl
  --detectors hide_and_seek hide_and_seek_first hide_and_seek_zero hide_and_seek_task_shuffle
  --feature-aggregation last --seeds 0 1 2
  --reference august_calibration $XR1_LEAD_ROOT/baselines/calibration_predictions.jsonl
  --reference august_calibration $XR1_CTRL_ROOT/baselines/calibration_predictions.jsonl
  --reference august_test $XR1_LEAD_ROOT/baselines/evaluation_predictions.jsonl
  --reference august_test $XR1_CTRL_ROOT/baselines/evaluation_predictions.jsonl
  --output-dir $AMEND2_ROOT/scores
)
PYTHONPATH=$AMEND2_REPO CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cohorts "${SCORE_ARGS[@]}" > $AMEND2_LOG/score.log 2>&1 &
echo "score PID: $! on $(hostname)"
```

It must end with six `reference ... -> OK` lines and `REFERENCE OK`. Otherwise stop.

### 3. Dry run on the opened August cohorts

```bash
cd $AMEND2_REPO
nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/confirm_hide_and_seek.py dry-run \
  --calibration $XR1_PS_INPUTS/calibration_predictions.jsonl --test $XR1_PS_INPUTS/event_predictions.jsonl \
  --scores-calibration $AMEND2_ROOT/scores/august_calibration_predictions.jsonl \
  --scores-test $AMEND2_ROOT/scores/august_test_predictions.jsonl \
  --normalizations $AMEND2_ROOT/scores/normalizations.json --runtime-bundle $XR1_RUNTIME_BUNDLE \
  --output $AMEND2_ROOT/dry_run_august.json 2>&1 | tee $AMEND2_LOG/dry_run.log
```

### 4. The one confirmatory analysis

```bash
cd $AMEND2_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/confirm_hide_and_seek.py run \
  --calibration $E1_OUT/calibration_predictions.jsonl --test $E1_OUT/test_predictions.jsonl \
  --scores-calibration $AMEND2_ROOT/scores/lead_calibration_predictions.jsonl \
  --scores-test $AMEND2_ROOT/scores/confirmatory_test_predictions.jsonl \
  --normalizations $AMEND2_ROOT/scores/normalizations.json --runtime-bundle $XR1_RUNTIME_BUNDLE \
  --e1-result $E1_OUT/confirm_lead.json --confirm-predictions $XR1_CONFIRM_ROOT/confirm/event_predictions.jsonl \
  --output $AMEND2_ROOT/confirm_hide_and_seek.json 2>&1 | tee $AMEND2_LOG/confirm.log
sha256sum -c $AMEND2_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
```

Report the final `H1:` line, the JSON block above it, and `INPUTS UNCHANGED`.
