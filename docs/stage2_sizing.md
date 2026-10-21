# Offline sizing for the RSS 2027 Stage 2 protocol (E1b, E2, E3)

**Status: exploratory sizing on opened cohorts.**
- Every cohort used here is already open for lead time (report §15.13–15.15).
  These numbers size the Stage 2 experiments. They are planning inputs, not evidence
  for any hypothesis, and the paper should cite them only as such.
- No rollout is collected. No registered analysis changes: P and SAFE + P are fitted with
  `confirm_lead.fit_fast` and evaluated with `confirm_lead.evaluate_fast`, exactly as in E1.
- This plan, its expectations and its decision rules were committed before the script was
  run on real data. The code is `scripts/xr1_lead_time/stage2_sizing.py`; synthetic tests are
  in `tests/test_stage2_sizing.py`.
- Stage 1 (two-stage format) is due 2026-12-04. E1b, E2 and E3 are Stage 2 items and need a
  sample-size justification in Stage 1.

## Common definitions

- **Detectors.** P is the per-task quantile timer, q = 0.95, with a conformal margin at
  alpha = 0.05. SAFE + P is the OR rule on the nested path, with (u, s) chosen on calibration
  failures. The development quantiles rho_k are frozen and never resampled.
- **Endpoint.** The mean warning lead (H_k − T)/H_k over test failures, with a miss counted
  as 0. The gain is lead(SAFE + P) − lead(P), paired over test failures.
- **Criteria of a simulated study** (Amendment 1, restricted to the cohort tested):
  - *CI > 0*: the decision-bootstrap 95% CI lower bound is above 0;
  - *point*: the point gain, rounded to 4 decimals as in `confirm_lead`, is at least 0.05;
  - *FPR rule*: fused test FPR ≤ 0.075 and its Wilson upper bound ≤ 0.10.
  Power is reported for "CI > 0", "CI > 0 and point" (the criterion in the task brief) and
  "CI > 0, point and FPR rule" (the full Amendment 1 rule).
- **Sizes.** n = 10, 20, ..., 60 successes and n failures per task, in both the calibration
  and the test cohort.

## Method for E1b and E2: resample, refit, count

**Outer simulation (the sampling distribution of a Stage 2 study).** For each n and each of
R = 2,000 replicates:
1. Draw n successes and n failures per task, with replacement, from the calibration pool, and
   independently from the test pool.
2. Fit P and SAFE + P on the drawn calibration sample (`fit_fast`).
3. Evaluate both on the drawn test sample (`evaluate_fast`); record the gain, the fused FPR
   and its Wilson bound, and whether the fitted path uses SAFE at all.

Two outer modes are run:
- **within task (primary):** the tasks are fixed and only rollouts are resampled within task
  and class. Inference is *conditional on these tasks*, as the brief asks;
- **with tasks:** tasks are also drawn with replacement, once per replicate for both cohorts,
  as in the E1 decision bootstrap. This gives the spread a task-level CI would have.

**The decision CI in a simulated study.** A full nested bootstrap per replicate is too costly
for every cell, so the CI lower bound of study r is approximated by g_r − hw, where hw is the
median minus the 2.5th percentile of the outer distribution at that n (within-task hw for the
primary CI; with-tasks hw for the task-level CI). This is the percentile interval of a
bootstrap whose shape matches the sampling distribution.

**Nested check of that approximation.** At n = 10, 30 and 60, on the plug-in population,
200 outer studies each get a real inner bootstrap: 300 replicates that resample rollouts
within task and class *from the study's own samples* and refit. The pass rates of CI > 0 and
CI > 0 and point are compared with the approximation on the same studies. The inner and outer
SD and half-width are also reported.

**Assumed gains.** The plug-in population (the empirical pools) has its own gain g0(n), which
can differ from the observed small-sample estimate and can rise with n, because more
calibration successes make the conformal margin less conservative. Assumed gains are imposed
two ways:
- **Shift (primary).** The outer gains at size n are moved by γ − mean(g), keeping their
  spread and the per-replicate FPR outcomes. Works for any γ, above or below g0.
- **Dilution (mechanistic check, γ < g0 only).** Each drawn failure, in calibration and
  test, is replaced with probability π = 1 − γ/g0(n) by a *null failure*: a same-task
  success's SAFE trajectory held at its running maximum to H_k. A fraction π of failures then
  carries no SAFE signal, which lowers the gain through the same refit. The realised mean gain
  is reported next to the power, since it is only approximately γ.

**E1b data.**
- The 10 atomic tasks of the E1 cohorts:
  `lead_analysis/calibration_with_baselines.jsonl` and `test_with_baselines.jsonl` under
  `xr1_lead_calibration_20261020_001108`.
- P and SAFE + P are calibrated on atomic calibration rollouts only (the D4 design of §15.14).
- **Pool:** all retained rows of the atomic tasks (primary). Quota collection retains every
  attempt until both quotas are met, so within a class the retained rows are a seed-ordered
  sample, unbiased with respect to outcome, and atomic tasks keep many extra successes.
  Sensitivity: the primary 10/10 rows only (`--pool primary`).
- **Assumed gains:** 0.05 (the margin), 0.0695 (the shrunk midpoint of the D4 sizing rule)
  and 0.089 (the D4 point estimate).
- **Check (stop condition):** the stratum-calibrated gain on the primary rows must reproduce
  D4, `dose_response.json` → safe → atomic → stratum_calibration → gain (+0.089).

**E2 data.**
- RLDX-1 SAFE20, LSTM (`rldx1_safe20_natural_weighted_20260815_162530/calibration/lstm`),
  20 tasks. H_k and rho_k are as in `rldx_replication.py` (training split).
- A fresh cohort's SAFE scores would be held-out, so the pool is the held-out rollouts only:
  the calibration split (3 successes per task) plus the evaluation split (5 + 5 per task),
  i.e. 8 successes and 5 failures per task. The simulated calibration *failures* are therefore
  held-out, which is the Stage 2 design.
- The pool is too small to split, so simulated calibration and test samples are drawn from
  the same pool.
- **Assumed gains:** 0.034 (the observed replication gain) and 0.05.
- **Check (stop condition):** the replication fit (calibration successes + training failures)
  must reproduce `rldx_lead_lstm.json` → primary → gain (+0.034).

## Collection cost

- Per task, attempts until n successes and n failures, everything retained:
  max(n + NB(n, p), n + NB(n, 1 − p)), 20,000 draws, for two cohorts (calibration + test).
- p is the observed success rate:
  - XR-1: all retained rows of both E1 cohorts. These are quota collections, which bias p
    slightly (inverse sampling); the bias is small next to the spread across tasks;
  - RLDX-1: the natural-rate collection, 50 rollouts per task over all splits.
- Reported per n: total rollouts (mean, 95th percentile), total inferences
  (attempts × (p · mean success length + (1 − p) · H_k)) as a GPU-time proxy, and tasks
  that exceed the 500-attempt cap of the E1 protocol in more than 5% of draws.
- A cross-check reads the collection manifests directly (cluster commands, step 3).

## Method for E3: retry-benefit bounds

The alarm times are those of the registered fits:
- XR-1: the E1 primary fit (pooled calibration, 38 tasks) on the E1 test cohort;
- RLDX-1: the replication fit on the evaluation split.

Under no monitor, a failure is known at its time limit H_k. A monitor's failure "alarm" is
min(alarm, H_k): an unalarmed failure is still known at the time limit. The typical success
duration D_k is the median success length (XR-1: all retained successes of both cohorts;
RLDX-1: training successes). p_k is as in the cost section. The retry success rate is
r = p_k by default (`--retry-success same`), i.e. a retry behaves like a fresh attempt.

**Single-retry bound (the brief's bound).** For budget B · H_k:

benefit_k = (1 − p_k) · r · frac_k, where frac_k = the fraction of task-k test failures
with B · H_k − alarm ≥ D_k.

It is reported for B ∈ {1.0, 1.25, 1.5, 2.0}, per task type (task-macro mean), for no monitor,
P and SAFE + P, with the decomposition P − none (timer value) and SAFE + P − P (policy-signal
value).

**Multi-retry Monte Carlo (sharper).** 20,000 episodes per task, detector and budget, with
common random numbers across detectors:
- each attempt succeeds with p_k (first) or r (later);
- a successful attempt takes a sampled test success's duration, unless that success's own
  false alarm comes first, which aborts it;
- a failing attempt ends at a sampled test failure's alarm time;
- the episode wins if a success finishes within the budget.

The success-rate gain over the same draws' first-attempt rate includes the false-alarm cost.
Also reported:
- the false-alarm exposure p_k · FPR_k;
- the episodes per arm needed to detect SAFE + P vs P on success rate at 2 H_k (two-sided
  0.05, power 0.8, unpaired). Because the effect is an upper bound, this n is a lower bound.

## Expectations, written before running

**E1b.**
- The check reproduces +0.089.
- The plug-in gain at n = 10 is near 0.08–0.09 and rises slightly with n.
- The within-task SD at n = 10 is well below the D4 decision SD (0.052, which resamples
  tasks), about 0.03, and shrinks roughly as 1/sqrt(n). The with-tasks SD stays near 0.05
  and barely shrinks.
- At γ = 0.05, "CI > 0 and point" has power of about 0.5 or less at every n. This is
  structural: when the truth equals the margin, the point estimate clears it about half the
  time.
- At γ = 0.0695, 0.8 is reached at about n = 30–40 under the within-task CI, and not by
  n = 60 under the task-level CI.
- At γ = 0.089, 0.8 is reached by about n = 20 (within task).
- The FPR rule binds at small n. With 100 atomic test successes, a Wilson upper bound
  ≤ 0.10 needs at most 4 false positives, which happens about 40% of the time at a true FPR
  of 0.05. The full rule's power is therefore capped near 0.4 at n = 10 and stops binding
  by n ≈ 30.
- The nested check agrees with the approximation within 0.10. The inner bootstrap may be
  somewhat wider than the outer distribution.
- Cost: several atomic tasks rarely fail, so n = 40 needs about 400 or more attempts per
  task per cohort, near the 500 cap, and about 4,000 rollouts in total (as §15.14 estimated).

**E2.**
- The check reproduces +0.034.
- With held-out failures choosing the path, the plug-in gain is at most the observed 0.034,
  likely 0.02–0.04.
- At γ = 0.034, the point ≥ 0.05 criterion loses power as n grows, towards 0, while CI > 0
  reaches 0.8 by about n = 20 within task. E2 should therefore register a superiority
  criterion (CI > 0) as the replication test, not the 0.05 margin.
- The shared, small pool (5 failures per task) makes the SD optimistic.
- Cost: the natural success rate is about 0.56, so a balanced task needs about 2n attempts
  per cohort. Tasks with extreme p dominate the total.

**E3.**
- The single-retry bound at 2 H_k is identical for none, P and SAFE + P. This is structural:
  D_k ≤ H_k, so a retry at the time limit always fits.
- Monitors separate only at tighter budgets (1.0–1.5 H_k). There, P − none carries most of
  the benefit, and SAFE + P − P is small (≤ 0.01 overall, larger on atomic tasks).
- In the multi-retry model at 2 H_k, SAFE + P − P is ≤ about 0.01, so thousands of episodes
  per arm would be needed.
- So E3 is not feasible as a success-rate superiority test of SAFE + P over P. Candidate
  framings are P (any monitor) vs none at a tight budget, or time to success.

## Decision rules, fixed before running

- **E1b.** Propose the smallest n in {10, ..., 60} whose power is at least 0.8 for the full
  rule (CI > 0, point, FPR rule) under the within-task CI at γ = 0.0695 (shift), provided:
  - the nested check agrees with the approximation within 0.10 at the bracketing sizes;
  - the dilution power at the same n is within 0.10 of the shift power.

  Otherwise, cite the nested or dilution number, whichever is lower. If no n reaches 0.8,
  report the power at n = 60 and do not register E1b with the 0.05 margin. The task-level
  power is reported beside it. It is the number that matters for any claim about atomic tasks
  in general.
- **E2.** Propose the smallest n with power ≥ 0.8 for CI > 0 at γ = 0.034 (within task, with
  the FPR rule), and report the margin criterion's power to justify not using it.
- **E3.** If the episodes per arm for SAFE + P vs P exceed 2,000 for either policy, propose E3
  as a demonstration or as a P-vs-none comparison, not as a SAFE + P vs P test.
- **Stop conditions:** any check mismatch (E1b +0.089, E2 +0.034), a changed input checksum,
  or an error. Do not edit inputs to make a run pass.

## Limits of the method

- **Opened cohorts.** All pools come from cohorts whose lead-time results are known, and the
  0.089 atomic gain was a selected subgroup. That is why shifted gains are used, not the
  plug-in alone.
- **Finite pools.** OR fusion acts through the extreme tail of SAFE scores (§15.14–15.15).
  Resampling cannot create a new extreme, so the uncertainty of the fused path, and of the
  conformal margin, is likely understated. This is worst for E2 (5 held-out failures per
  task) and for atomic tasks that rarely fail (about 10 failures per task). Power at large n
  is limited by the number of distinct trajectories, not by n.
- **Within-task inference** is conditional on these tasks. It says nothing about other
  atomic tasks or other policies. The with-tasks columns show how much wider a task-level CI
  would be.
- **Assumed-gain mechanisms.** The shift assumes the sampling distribution's shape does not
  depend on the true gain. The dilution assumes the missing gain comes from SAFE-invisible
  failures. Neither models a gain that is smaller because the task mix differs.
- **The CI approximation** is checked against nested bootstraps only on the plug-in
  population and at three sizes.
- **E2 shares one pool** between simulated calibration and test, so the same trajectory can
  set the path and be scored. This is optimistic.
- **E3 is a shadow-data bound.**
  - Retries are assumed independent with success rate r = p_k. A retry from the failed state
    is probably less likely to succeed, so the bound is generous.
  - The single-retry bound ignores false-alarm cost; the multi-retry model includes it.
  - Closed-loop control would change the state distribution, and so the alarm times.
- **Success rates** come from quota collections (XR-1) or a 50-per-task natural run
  (RLDX-1), so per-task p is noisy. Cost totals are best read to about ±20%.

## Cluster commands

CPU only, from a pinned detached worktree; outputs under `/gs/bs`. `S2_SHA` is the full SHA of
the pushed commit that contains this plan.

**1. Exports and worktree**

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export WANDB_MODE=disabled WANDB_DISABLED=true
export XR1_LCAL_ROOT=$STORAGE_BS/robocasa_rollouts/safe/xr1_lead_calibration_20261020_001108
export E1_OUT=$XR1_LCAL_ROOT/lead_analysis
export XR1_RUNTIME_BUNDLE=$STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_calibration/runtime_bundle.json
export RLDX_CAL=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_safe20_natural_weighted_20260815_162530/calibration/lstm
export RLDX_REP=$(ls -d $STORAGE_BS/robocasa_checkpoints/safe/rldx1_lead_replication_* | tail -1)/rldx_lead_lstm.json
export S2_SHA=<full SHA>
export S2_REPO=$PROJECT_FS/robocasa_stage2_sizing_${S2_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin
test -e $S2_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $S2_REPO $S2_SHA
test "$(git -C $S2_REPO rev-parse HEAD)" = "$S2_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
export S2_TAG=stage2_sizing_$(date +%Y%m%d_%H%M%S)
export S2_OUT=$STORAGE_BS/robocasa_checkpoints/safe/$S2_TAG S2_LOG=$STORAGE_BS/robocasa_logs/eval/$S2_TAG
mkdir -p $S2_OUT $S2_LOG
ls -l $E1_OUT/calibration_with_baselines.jsonl $E1_OUT/test_with_baselines.jsonl $E1_OUT/dose_response.json \
      $XR1_RUNTIME_BUNDLE $RLDX_REP $RLDX_CAL/lstm_seed{0,1,2}/normalized_scores.jsonl
sha256sum $E1_OUT/calibration_with_baselines.jsonl $E1_OUT/test_with_baselines.jsonl $XR1_RUNTIME_BUNDLE \
          $RLDX_CAL/lstm_seed{0,1,2}/normalized_scores.jsonl | tee $S2_OUT/inputs_sha256_before.txt
echo "S2_OUT=$S2_OUT"
```

**2. Tests, then the three runs** (about 10–20 minutes in total with 8 workers; use an
allocation with at least 8 CPU cores; nothing uses a GPU)

```bash
cd $S2_REPO
CUDA_VISIBLE_DEVICES="" $SAFE_PY -W ignore -m unittest tests.test_stage2_sizing tests.test_xr1_confirm_lead \
  tests.test_xr1_dose_response tests.test_rldx_lead_replication 2>&1 | grep -E "^(Ran|OK|FAILED)"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup sh -c "
nice -n 10 $SAFE_PY -I -u scripts/xr1_lead_time/stage2_sizing.py e1b --calibration $E1_OUT/calibration_with_baselines.jsonl \
  --test $E1_OUT/test_with_baselines.jsonl --runtime-bundle $XR1_RUNTIME_BUNDLE --dose-response $E1_OUT/dose_response.json \
  --workers 8 --output $S2_OUT/e1b_all.json > $S2_LOG/e1b_all.log 2>&1;
nice -n 10 $SAFE_PY -I -u scripts/xr1_lead_time/stage2_sizing.py e1b --calibration $E1_OUT/calibration_with_baselines.jsonl \
  --test $E1_OUT/test_with_baselines.jsonl --runtime-bundle $XR1_RUNTIME_BUNDLE --dose-response $E1_OUT/dose_response.json \
  --pool primary --nested-sizes '' --workers 8 --output $S2_OUT/e1b_primary.json > $S2_LOG/e1b_primary.log 2>&1;
nice -n 10 $SAFE_PY -I -u scripts/xr1_lead_time/stage2_sizing.py e2 --calibration-dir $RLDX_CAL --replication-json $RLDX_REP \
  --workers 8 --output $S2_OUT/e2.json > $S2_LOG/e2.log 2>&1;
nice -n 10 $SAFE_PY -I -u scripts/xr1_lead_time/stage2_sizing.py e3 --calibration $E1_OUT/calibration_with_baselines.jsonl \
  --test $E1_OUT/test_with_baselines.jsonl --runtime-bundle $XR1_RUNTIME_BUNDLE --rldx-calibration-dir $RLDX_CAL \
  --output $S2_OUT/e3.json > $S2_LOG/e3.log 2>&1;
echo ALL DONE > $S2_LOG/done.txt" > $S2_LOG/driver.log 2>&1 &
echo "driver PID $!  logs in $S2_LOG"
```

Monitor with `ls $S2_LOG; tail -3 $S2_LOG/*.log`. When `done.txt` exists:

**3. Manifest cross-check, checksum check, and the output to paste back**

```bash
for d in $XR1_LCAL_ROOT/merged_all_retained; do
  $SAFE_PY -I -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; print(sys.argv[1], len(r), 'rows', sum(x['failed'] for x in r), 'failures')" $d/manifest.jsonl
done
sha256sum $E1_OUT/calibration_with_baselines.jsonl $E1_OUT/test_with_baselines.jsonl $XR1_RUNTIME_BUNDLE \
          $RLDX_CAL/lstm_seed{0,1,2}/normalized_scores.jsonl | diff - $S2_OUT/inputs_sha256_before.txt && echo INPUTS UNCHANGED
cat $S2_LOG/driver.log; cat $S2_LOG/e1b_all.log $S2_LOG/e1b_primary.log $S2_LOG/e2.log $S2_LOG/e3.log
```

Paste back the whole output of step 3. The manifest line should read 1,522 rows and 597
failures (report §15.13).

## Results

To be filled from the cluster output.
