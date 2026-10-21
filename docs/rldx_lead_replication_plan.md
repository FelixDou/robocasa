# RLDX-1 offline replication of the lead-over-timer analysis (plan)

**Status: exploratory, second policy, no new rollouts.**
- Purpose: preliminary evidence for H3 (the timer result is not specific to XR-1),
  ahead of the preregistered E2.
- The RLDX-1 SAFE20 data were analysed before (report §8.3), but never for lead time
  or against a timer. No lead, timer or P quantity has been computed on them.
- This plan was committed before the script was run.

## Data

- **Rollouts.** RLDX-1 (`RLWRLD/RLDX-1-FT-RC365`) natural-rate collection,
  support-filtered to 20 tasks with 50 rollouts each: 1,000 rollouts, 562 successes and
  438 failures.
  - Export: `rldx1_original_safe35_natural50_20260808_203158/official_safe_supported20_natural50`.
- **SAFE scores.** Root `rldx1_safe20_natural_weighted_20260815_162530`.
  - Primary model: the LSTM, which the earlier run selected by inner CV.
  - The MLP is reported descriptively.
  - Per-seed per-inference scores, normalized with training-only per-task statistics,
    are read from `<calibration dir>/<model>_seed{0,1,2}/normalized_scores.jsonl`.
  - The SAFE score at inference t is the seed mean of each seed's running maximum of
    `full_scores` (the untruncated trajectory). This is the same construction as XR-1.
- **Splits**, taken unchanged from the earlier calibration manifest:
  - development = `train` (740 rollouts);
  - calibration successes = `calibration` (60, three per task);
  - test = `evaluation` (200, five successes and five failures per task).
  - There are **no held-out calibration failures**. The fused detector's path (u, s) is
    therefore chosen on the *training* failures' scores. These are in-sample for SAFE, so
    the path choice is optimistic. The FPR margin still comes only from the 60 held-out
    calibration successes, as Proposition 1 requires.

## Definitions (as in E1 unless stated)

- **Time limit H_k:** the longest training failure of task k, in inferences. If a record
  carries `online_task_timeout_inferences`, it is reported next to H_k as a check.
- **Rollout length L:** `full_num_inferences`.
- **Lead:** (L − T)/H_k over test failures, with a miss counted as 0. An alarm at the final
  inference counts.
- **P:** the per-task q = 0.95 quantile of training success durations L/H_k, plus a
  conformal margin set on the 60 calibration successes at alpha = 0.05.
- **SAFE + P:** the OR rule, fitted with `confirm_lead.fit_fast`.
- **Other detectors:** the global timer (one c on calibration, `lead_time.search`) and
  SAFE + global timer.
- **Null:** noise + P (running max of i.i.d. N(0, 1)), fitted the same way, 20 draws with
  seed 1.
- **Uncertainty:** test-only task-then-rollout bootstrap (2,000 replicates, seed 0) for
  every gain over P, overall and by task type.

## Checks, reported first

- **C1, do failures run to the time limit?** For training and test failures separately:
  - the fraction with L = H_k;
  - the fraction with L ≥ 0.95 H_k;
  - the quantiles of L/H_k;
  - the number of test failures with L > H_k.

  The lead framing assumes failures end at the limit. If more than 10% of test failures
  end before 0.95 H_k, every recall and lead result is flagged as computed on data where
  failures can end early. Those rollouts are then listed and described, and they become
  the E5 question for this policy.
- **C2, data integrity:**
  - the three seeds agree on IDs, labels and lengths;
  - every task has training successes and failures, 3 calibration successes, and
    5 + 5 test rollouts;
  - the splits are disjoint.
- **C3, sample sizes:** 60 calibration successes give a conformal rank of 58 of 60. The
  realised test FPR is reported with a Wilson interval over 100 test successes.

## Questions and expectations, written before running

- **R1.** Failures mostly run to the limit, as on XR-1 (C1).
- **R2.** P has more lead than the global timer.
- **R3.** SAFE + P gains little over P overall: a point estimate of 0.03 or less, with a CI
  that may include 0.
- **R4.** The gain is larger on atomic than on composite tasks. RLDX SAFE was informative
  mainly on atomic tasks (report §8.3).
- **R5.** Noise + P gains essentially nothing (under 0.01).

All results are exploratory and support, but cannot replace, the preregistered E2.

## Runbook

Run on the cluster from a pinned worktree, CPU only. Step 1 is read-only discovery;
continue only if it shows the expected files.

```bash
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export RLDX_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_safe20_natural_weighted_20260815_162530
ls $RLDX_ROOT; ls $RLDX_ROOT/calibration 2>/dev/null; ls $RLDX_ROOT/calibration/lstm 2>/dev/null | head -20
for s in 0 1 2; do f=$RLDX_ROOT/calibration/lstm/lstm_seed$s/normalized_scores.jsonl; test -s $f && echo "OK $f" || echo "MISSING $f"; done
```

Step 2: `RLDX_SHA` is the full SHA of the pushed commit that contains this plan.

```bash
export RLDX_SHA=<full SHA>
export RLDX_REPO=$PROJECT_FS/robocasa_rldx_lead_${RLDX_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin
test -e $RLDX_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $RLDX_REPO $RLDX_SHA
test "$(git -C $RLDX_REPO rev-parse HEAD)" = "$RLDX_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
export RLDX_OUT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_lead_replication_$(date +%Y%m%d_%H%M%S)
mkdir -p $RLDX_OUT
sha256sum $RLDX_ROOT/calibration/lstm/lstm_seed*/normalized_scores.jsonl $RLDX_ROOT/calibration/lstm/split_manifest.json | tee $RLDX_OUT/inputs_sha256_before.txt
cd $RLDX_REPO
CUDA_VISIBLE_DEVICES="" $SAFE_PY -W ignore -m unittest tests.test_rldx_lead_replication 2>&1 | grep -E "^(Ran|OK|FAILED)"
for model in lstm indep; do
  test -d $RLDX_ROOT/calibration/$model || { echo "no $model calibration dir"; continue; }
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 $SAFE_PY -I -u scripts/xr1_lead_time/rldx_replication.py \
    --calibration-dir $RLDX_ROOT/calibration/$model --model $model \
    --output $RLDX_OUT/rldx_lead_$model.json 2>&1 | tee $RLDX_OUT/rldx_lead_$model.log
done
sha256sum $RLDX_ROOT/calibration/lstm/lstm_seed*/normalized_scores.jsonl $RLDX_ROOT/calibration/lstm/split_manifest.json | diff - $RLDX_OUT/inputs_sha256_before.txt && echo INPUTS UNCHANGED
```
