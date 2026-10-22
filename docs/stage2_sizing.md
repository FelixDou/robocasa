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

## Results (run 2026-10-21)

### Run record

- Code: commit `9757e024`, pinned worktree `robocasa_stage2_sizing_9757e024`, CPU only,
  8 workers.
- Output: `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/stage2_sizing_20261021_131345/`
  (`e1b_all.json`, `e1b_primary.json`, `e2.json`, `e3.json`, `inputs_sha256_before.txt`).
- Logs: `/gs/bs/tga-shinoda/felid/robocasa_logs/eval/stage2_sizing_20261021_131345/`.
- Tests on the cluster: 16 passed (this file's tests plus the E1, dose-response and RLDX tests).
- **Checks.**
  - E1b reproduces D4: +0.0890 against +0.0890, MATCH (both pools).
  - E2 reproduces the replication: +0.0342 against +0.0342, MATCH.
  - E3's fits reproduce the registered leads: XR-1 P 0.2252 (FPR 0.0553) and
    SAFE + P 0.2550 (0.0605); RLDX-1 0.2711 and 0.3053 (0.03).
  - Manifest cross-check: `merged_all_retained/manifest.jsonl` has 1,522 rows and 597
    failures, matching report §15.13.
  - Input checksums unchanged after the run (`INPUTS UNCHANGED`). The driver log holds only
    nohup's notice, and all four runs finished (`done.txt`).
- **Pools.**
  - E1b, all retained: calibration 467 successes / 103 failures, test 373 / 105, over
    10 atomic tasks. Primary rows: 100 / 100 in each cohort.
  - E2, held-out: 160 successes / 100 failures over 20 tasks.

### E1b: atomic-only XR-1 test

**Table S1 — sampling distribution of the gain (plug-in, all-retained pool).** FPR pass is
the fraction of simulated studies that meet the Amendment 1 FPR rule.

| n per class per task | Mean gain | SD within task | SD with tasks | Mean fused test FPR | FPR pass |
|---:|---:|---:|---:|---:|---:|
| 10 | +0.066 | 0.036 | 0.042 | 0.061 | 0.39 |
| 20 | +0.061 | 0.029 | 0.037 | 0.059 | 0.55 |
| 30 | +0.060 | 0.027 | 0.034 | 0.059 | 0.66 |
| 40 | +0.058 | 0.025 | 0.033 | 0.057 | 0.76 |
| 50 | +0.058 | 0.023 | 0.033 | 0.058 | 0.79 |
| 60 | +0.058 | 0.023 | 0.033 | 0.058 | 0.84 |

**Table S2 — power at the assumed gains (all-retained pool, shift, within-task CI).**
Each cell is "CI > 0 and point ≥ 0.05" / the full Amendment 1 rule (plus the FPR rule).

| n | γ = 0.05 | γ = 0.0695 | γ = 0.089 |
|---:|---|---|---|
| 10 | 0.42 / 0.07 | 0.62 / 0.17 | 0.92 / 0.34 |
| 20 | 0.43 / 0.10 | 0.66 / 0.27 | 0.98 / 0.54 |
| 30 | 0.44 / 0.15 | 0.67 / 0.34 | 1.00 / 0.66 |
| 40 | 0.45 / 0.23 | 0.67 / 0.44 | 1.00 / 0.76 |
| 50 | 0.46 / 0.26 | 0.70 / 0.49 | 1.00 / 0.79 |
| 60 | 0.48 / 0.32 | 0.70 / 0.55 | 1.00 / 0.84 |

CI > 0 alone, γ = 0.0695: 0.77, 0.98, then 1.00 from n = 30. Because CI > 0 passes in
(almost) every study from n = 20, "CI > 0 plus the FPR rule" at γ = 0.0695 equals the FPR
pass rate of Table S1 there (0.55 at n = 20 to 0.84 at n = 60). With the task-level CI,
"CI > 0 and point" is within 0.01 of the within-task value at every n and γ.

**Dilution** is possible only at γ = 0.05 for this pool, because the plug-in gain is below
0.0695. There it gives 0.39–0.43 for "CI > 0 and point", against 0.42–0.48 by shift.

**Primary-pool sensitivity (100 / 100 per cohort).**
- The plug-in gain is higher, +0.100 at n = 10 falling to +0.091 at n = 60, and so is the
  spread: SD within task 0.042–0.027, with tasks 0.052–0.044.
- At γ = 0.0695: "CI > 0 and point" by shift is 0.42, 0.68, 0.70, 0.67, 0.68, 0.72; by
  dilution it is 0.55, 0.79, 0.83, 0.85, 0.87, 0.91. The full rule by shift is 0.13–0.61.
  With the task-level CI it is 0.41 at n = 10, then 0.25–0.38.
- At γ = 0.089, the full rule reaches 0.84 at n = 50.

**Table S3 — nested check of the CI approximation (plug-in, all-retained pool;
200 studies × 300 inner replicates).**

| n | CI > 0, nested / approx | CI > 0 and point, nested / approx | Inner / outer SD |
|---:|---|---|---|
| 10 | 0.99 / 0.69 | 0.57 / 0.58 | 0.034 / 0.036 |
| 30 | 1.00 / 0.93 | 0.57 / 0.57 | 0.025 / 0.027 |
| 60 | 1.00 / 0.96 | 0.55 / 0.55 | 0.021 / 0.023 |

- For the decision criterion ("CI > 0 and point") the approximation agrees within 0.01 at
  every size.
- For CI > 0 alone it is conservative at n = 10 (0.69 against 0.99). The inner percentile
  interval sits higher than g − hw, so CI > 0 alone is understated at small n, not overstated.

### E2: fresh RLDX-1 cohort with held-out calibration failures

**Table S4 — plug-in sampling distribution and power (shift, within-task CI).** The power
cells are "CI > 0 and point ≥ 0.05" / the full rule.

| n | Mean gain | SD within | SD with tasks | FPR pass | CI > 0 at γ = 0.034 | γ = 0.034 | γ = 0.05 |
|---:|---:|---:|---:|---:|---:|---|---|
| 10 | +0.040 | 0.012 | 0.028 | 0.62 | 1.00 | 0.09 / 0.03 | 0.39 / 0.22 |
| 20 | +0.036 | 0.007 | 0.024 | 0.92 | 1.00 | 0.03 / 0.02 | 0.43 / 0.40 |
| 30 | +0.035 | 0.006 | 0.023 | 0.98 | 1.00 | 0.02 / 0.02 | 0.46 / 0.45 |
| 40 | +0.034 | 0.005 | 0.022 | 0.99 | 1.00 | 0.01 / 0.01 | 0.47 / 0.47 |
| 50 | +0.034 | 0.004 | 0.022 | 1.00 | 1.00 | 0.00 / 0.00 | 0.47 / 0.47 |
| 60 | +0.034 | 0.004 | 0.022 | 1.00 | 1.00 | 0.00 / 0.00 | 0.48 / 0.48 |

- **Dilution at γ = 0.034** (π = 0.15 to 0.01) gives the same margin-criterion power: 0.09
  down to 0.00.
- **Nested check:** it agrees exactly on both criteria at n = 10, 30 and 60. The inner SD is
  larger than the outer SD (0.017 against 0.012 at n = 10), the expected sign of the
  shared-pool optimism.
- **The within-task SD is very small** (0.004–0.012), against 0.022–0.028 with tasks. Almost
  all of the uncertainty is between tasks.

**Table S4b — power for CI > 0 alone, task-level CI against within-task CI (from
`e2.json`, `ci_only_task_ci` and `ci_only`; read on the cluster after the run).**

| n | γ = 0.034, task CI | γ = 0.034, within task | γ = 0.05, task CI |
|---:|---:|---:|---:|
| 10 | 0.44 | 1.00 | 0.99 |
| 20 | 0.54 | 1.00 | 1.00 |
| 30 | 0.80 | 1.00 | 1.00 |
| 40 | 0.74 | 1.00 | 1.00 |
| 50 | 0.90 | 1.00 | 1.00 |
| 60 | 0.90 | 1.00 | 1.00 |

The task-level column is not monotone (0.80 at n = 30, 0.74 at n = 40). Each n uses one
half-width estimated from its own with-tasks simulation, and power is a step function of
that half-width, so neighbouring sizes carry Monte Carlo noise of about ±0.05. Read the
column as "about 0.5 up to n = 20, about 0.75–0.8 at n = 30–40, about 0.9 from n = 50".

### Collection cost

**Table S5 — rollouts for calibration + test cohorts, everything retained.** Inferences are
a GPU-time proxy. "Over cap" lists tasks above the 500-attempt cap in more than 5% of draws.

| n | E1b (10 atomic XR-1 tasks): mean (p95) | E1b inferences | Over cap | E2 (20 RLDX-1 tasks): mean (p95) | E2 inferences |
|---:|---|---:|---|---|---:|
| 10 | 1,053 (1,248) | 22,754 | – | 1,253 (1,376) | 103,246 |
| 20 | 2,102 (2,370) | 45,336 | – | 2,497 (2,672) | 205,746 |
| 30 | 3,151 (3,476) | 67,903 | – | 3,745 (3,960) | 308,523 |
| 40 | 4,197 (4,572) | 90,435 | OpenCabinet | 4,989 (5,234) | 411,104 |
| 50 | 5,246 (5,664) | 113,023 | CloseToasterOvenDoor, OpenCabinet | 6,234 (6,512) | 513,602 |
| 60 | 6,293 (6,746) | 135,561 | CloseToasterOvenDoor, OpenCabinet | 7,482 (7,788) | 616,196 |

- **XR-1 atomic success rates:** 0.43–0.91 (OpenCabinet 0.91, CloseToasterOvenDoor 0.89).
- **RLDX-1 natural rates:** 0.30–0.80, mean 0.56.
- XR-1's cost is driven by the rarely failing tasks. RLDX-1's cost per rollout is higher in
  inferences, because its time limits are longer.

### E3: closed-loop retry bounds

**Table S6 — success-rate gain over a single attempt (task-macro means), retry success
r = p_k.** Each cell is the single-retry bound / the multi-retry Monte Carlo.

| Policy, type | Budget | None | P | SAFE + P | P − none | SAFE + P − P |
|---|---:|---|---|---|---|---|
| XR-1, atomic (10) | 1.0 H | 0 / 0 | .037 / .024 | .097 / .068 | .037 / .024 | **.060 / .044** |
| | 1.25 H | 0 / .015 | .150 / .120 | .150 / .125 | .150 / .105 | 0 / .006 |
| | 2.0 H | .174 / .175 | .174 / .205 | .174 / .209 | 0 / .030 | 0 / .003 |
| XR-1, composite (28) | 1.0 H | 0 / 0 | .041 / .003 | .046 / .006 | .041 / .003 | .005 / .003 |
| | 1.25 H | .008 / .011 | .109 / .075 | .112 / .077 | .102 / .065 | .003 / .001 |
| | 2.0 H | .209 / .210 | .209 / .232 | .209 / .233 | 0 / .022 | 0 / .001 |
| RLDX-1, atomic (14) | 1.0 H | 0 / 0 | .134 / .118 | .137 / .130 | .134 / .118 | .003 / .011 |
| | 1.25 H | .042 / .048 | .182 / .176 | .182 / .186 | .140 / .128 | 0 / .010 |
| | 2.0 H | .213 / .212 | .213 / .271 | .213 / .275 | 0 / .059 | 0 / .004 |
| RLDX-1, composite (6) | 1.0 H | 0 / 0 | 0 / −.012 | 0 / −.012 | 0 / −.012 | 0 / 0 |
| | 2.0 H | .231 / .231 | .231 / .228 | .231 / .228 | 0 / −.003 | 0 / 0 |

- **Single-retry bound at 2 H_k:** identical for all three arms on both policies, as
  predicted.
- **Multi-retry at 2 H_k** (the script's episodes-per-arm figure for SAFE + P vs P):
  - XR-1: 93,510 per arm on atomic tasks and 1,520,098 on all tasks;
  - RLDX-1: 73,749 on atomic tasks and 239,474 on all tasks.
- **False-alarm exposure p·FPR:** XR-1 0.032 (P) and 0.036 (SAFE + P); RLDX-1 0.011 for both.
- **Episodes per arm for other contrasts.** These were derived by hand from Table S6 with the
  same two-proportion formula, using the task-macro mean p. They are lower bounds, because
  the effects are upper bounds.
  - SAFE + P vs P, XR-1 atomic, 1.0 H_k: about 1,450 per arm;
  - SAFE + P vs P, RLDX-1 atomic, 1.0 H_k: about 23,000;
  - P vs none, all tasks, 1.25 H_k: about 650 (XR-1) and 480 (RLDX-1);
  - P vs none, all tasks, 2.0 H_k: about 4,500 (XR-1).

### Decisions under the pre-stated rules

- **E1b.**
  - No n ≤ 60 reaches 0.8 for the full rule at γ = 0.0695. The power at n = 60 is 0.55
    (all-retained pool), and "CI > 0 and point" alone levels off near 0.70.
  - Dilution could not be run at this γ, and the nested check agrees on the decision
    criterion.
  - **By the rule, E1b is not registered with the 0.05 margin.**
  - Even at the D4 point estimate (γ = 0.089), the full rule needs n ≈ 60 per class per task
    (about 6,300 rollouts), and the binding part is the FPR rule, not the gain.
- **E2.**
  - The smallest n with power ≥ 0.8 for CI > 0 plus the FPR rule at γ = 0.034 is
    **n = 20** (0.92; CI > 0 passes in every study, so the FPR pass rate decides it). That is
    about 2,500 rollouts (p95 2,672) over 20 tasks.
  - The 0.05 margin has power ≤ 0.09 at γ = 0.034 and does not exceed 0.48 even at a true
    gain of 0.05. **E2 should register CI > 0 (superiority) as its replication criterion.**
  - **n = 20 holds only for inference conditional on these 20 tasks.** With a task-level CI
    (tasks resampled, as in the E1 decision bootstrap), CI > 0 at γ = 0.034 has power 0.54
    at n = 20 and reaches about 0.9 only at **n = 50** (Table S4b; the FPR rule passes in
    every study at that size). That is about 6,200 rollouts (p95 6,512).
  - Registering E2 therefore needs a choice of inference before Stage 1:
    - within-task: n = 20, about 2,500 rollouts, a claim about these tasks;
    - task-level: n = 50, about 6,200 rollouts, a claim that extends to tasks like these.
  - The shared held-out pool (5 failures per task) makes both numbers optimistic.
- **E3.**
  - Episodes per arm for SAFE + P vs P at 2 H_k exceed 2,000 on both policies, by one to
    three orders of magnitude.
  - **By the rule, E3 is proposed as a demonstration or as a P-vs-none comparison**: about
    500–650 episodes per arm at 1.25 H_k on either policy.
  - The one SAFE-specific contrast within reach is XR-1 atomic tasks at a 1.0 H_k budget
    (retry within the original time limit): about 1,450 episodes per arm, on an optimistic
    bound.

### Expectations against results

| Expectation (written before running) | Result |
|---|---|
| E1b check reproduces +0.089 | Yes |
| E1b plug-in gain at n = 10 near 0.08–0.09, rising slightly with n | **No.** 0.066 with the all-retained pool, falling to 0.058 (primary pool 0.100 → 0.091) |
| Within-task SD ≈ 0.03 at n = 10, shrinking as 1/sqrt(n); with-tasks SD ≈ 0.05, flat | 0.036, shrinking more slowly (0.023 at n = 60); with tasks 0.042 → 0.033 |
| γ = 0.05: "CI > 0 and point" ≤ about 0.5 at every n | Yes (0.42–0.48) |
| γ = 0.0695: 0.8 at n ≈ 30–40 within task | **No.** It levels off near 0.70 (0.55 with the FPR rule) |
| γ = 0.089: 0.8 by n ≈ 20 | Yes for "CI > 0 and point" (0.98); the full rule needs n ≈ 60 |
| FPR rule passes about 0.4 at n = 10 and stops binding by n ≈ 30 | 0.39 at n = 10, but **still binding**: 0.66 at n = 30, 0.84 at n = 60 |
| Nested check within 0.10 | Yes on the decision criterion; CI > 0 alone at n = 10 is off by 0.30 (approximation conservative) |
| Cost at n = 40 ≈ 4,000 rollouts, near the cap | 4,197; OpenCabinet over the cap |
| E2 check reproduces +0.034; plug-in gain ≤ 0.034 | Yes; 0.040 at n = 10, 0.034 from n = 40 |
| E2: margin power → 0; CI > 0 at 0.8 by n ≈ 20 | Yes; CI > 0 is already 1.00 at n = 10, and the FPR rule sets n = 20 |
| E3 single-retry bound flat at 2 H_k | Yes, both policies |
| E3: SAFE + P − P ≤ 0.01 at tight budgets, larger on atomic tasks | ≤ 0.013 except at 1.0 H_k on XR-1: **atomic 0.044–0.060**, which lifts the all-task mean to 0.014–0.020 |
| E3 needs thousands of episodes per arm | Tens of thousands to over a million at 2 H_k |

### Reading the results (exploratory)

- **The 0.089 atomic estimate is optimistic.**
  - The pool with all retained successes (467 calibration successes rather than 100) gives a
    plug-in gain of about 0.06 at Stage 2 sizes. The primary rows give about 0.10.
  - More distinct successes put more of them in the extreme tail, which pushes the SAFE
    threshold up and removes early alarms. This is the finite-pool limit stated above, and it
    points the same way as the subgroup selection.
  - A planning value near 0.06 for the atomic gain is more defensible than 0.0695.
- **At a planning gain near 0.06, the 0.05 margin is close to a coin flip at any n.**
- **The test FPR runs above nominal.** It averages 0.057–0.061 rather than 0.05, as in E1
  (0.061). The calibration and test cohorts are therefore not exchangeable in the success
  tail, and the Wilson ≤ 0.10 limit binds more than its 380-success design assumed.
- **For E2, the gain is stable, but nearly all the uncertainty is between tasks.** Going
  from conditional to task-level inference raises the needed n from 20 to about 50.
- **For E3, nearly all of the retry value comes from having a timer.** What the policy signal
  adds over the timer is measurable in success rate only on XR-1 atomic tasks with no extra
  budget.

**What Stage 1 can cite**, as planning statements from opened cohorts:
- E2 with a superiority criterion (CI > 0) at 20 + 20 per task in calibration and test,
  about 2,500 rollouts, for inference conditional on the 20 tasks; or 50 + 50 per task,
  about 6,200 rollouts, for a task-level CI;
- E1b as a superiority (CI > 0) subgroup test, or dropped. Not with a 0.05 margin;
- E3 as P vs none at 1.25 H_k (about 500–650 episodes per arm), with SAFE + P as a third arm
  that is descriptive.

## E2 co-primary sizing (2026-10-23)

**Why.** After the Hide-and-Seek results (report §15.20–15.22), the RSS Stage 1 draft makes
SAFE + P and Hide-and-Seek + P co-primary in E2: on RLDX-1's 20 tasks, each against P, each
supported if its decision CI lies above zero at 97.5% (Bonferroni over the two), under the
E2 FPR rule, with no 0.05 margin. The draft's power figure (0.92) is SAFE alone at 95%. This
section sizes the co-primary design. It is exploratory sizing on opened cohorts, like
everything above. It was written, with the code and tests, before the script was run on
real data.

**Code.** `scripts/xr1_lead_time/e2_coprimary_sizing.py`, tests in
`tests/test_e2_coprimary_sizing.py`. It reuses this document's E2 method:
- **Pool.** The same held-out pool: 3 calibration successes plus 5/5 evaluation rollouts per
  task, shared by the simulated calibration and test samples.
- **Draws.** The same seed and the same cell and replicate numbering. SAFE's simulated
  studies are therefore exactly those of Table S4 (the test checks this on synthetic data).
- **Fits.** `confirm_lead.fit_fast` and `evaluate_fast`.

**What is new.**
- **Paired scoring.** Every simulated study is scored with both detectors on the same drawn
  rollouts. P is fitted once per study.
- **Hide-and-Seek scores.** Its trajectories are the §15.22 run's
  (`rldx1_hide_and_seek_20261023_000418/scores`), for the same held-out rollouts and, for
  the check, the training failures.
- **Two CI levels.** The decision CI is approximated as above (g − hw), at 95% and 97.5%,
  with hw = median minus the 2.5th or 1.25th percentile of the within-task sampling
  distribution. The task-level variant uses the with-tasks distribution.
- **Two FPR rules.**
  - A1: FPR ≤ 0.075 and Wilson upper ≤ 0.10.
  - E2: false alarms do not reject "true FPR ≤ 0.075" in a one-sided exact binomial test at
    level 0.05 (at most 39 of 400 at n = 20).
- **Assumed gains**, imposed by shift:
  - SAFE: 0.034 (§15.15);
  - Hide-and-Seek: 0.083 (§15.22), 0.052 (its gain against P at matched false alarms)
    and 0.05;
  - joint: (0.034, 0.083) and (0.034, 0.052).
- **Reported:** power per detector, for at least one and for both, at n = 10–60 per class
  per task.

**Checks (stop conditions).** On the §15.15 / §15.22 fit (held-out successes, training
failures, evaluation split):
- SAFE + P must reproduce +0.0342 (`rldx_lead_lstm.json`);
- Hide-and-Seek + P must reproduce the gain stored in `rldx_hide_and_seek.json`
  (+0.0834; §15.22 rounds it to +0.083).

**Expectations, written before running.**
- **SAFE at 97.5%.** The within-task SD at n = 20 is 0.007, so the CI half-width barely
  changes and CI > 0 stays at about 1.00. SAFE's power stays the FPR pass rate: 0.92 under
  A1, and at least that under E2.
- **Hide-and-Seek.** The plug-in gain should be well above SAFE's (about 0.07–0.09), so CI > 0
  should be about 1.00 at every n. Its power is then decided by false alarms. On the opened
  cohort it used 4 of 100 against P's 3, so we expect a lower FPR pass rate than SAFE's at
  n = 10, rising with n. We do not predict the value at n = 20.
- **At least one co-primary passing.** Close to 1 at n = 20.
- **Task-level CI.** Much lower power for both, as in Table S4b; we report it without a power
  claim.

### Cluster commands

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
source $STORAGE_BS/robocasa_logs/eval/rldx1_hide_and_seek_20261023_000418/paths.env
export E2CO_SHA=<full SHA of the commit containing this section>
export E2CO_REPO=$PROJECT_FS/robocasa_e2co_${E2CO_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin main
test -e $E2CO_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $E2CO_REPO $E2CO_SHA
test "$(git -C $E2CO_REPO rev-parse HEAD)" = "$E2CO_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
export E2CO_TAG=e2_coprimary_sizing_$(date +%Y%m%d_%H%M%S)
export E2CO_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$E2CO_TAG
export E2CO_LOG=$STORAGE_BS/robocasa_logs/eval/$E2CO_TAG
mkdir -p $E2CO_ROOT $E2CO_LOG
sha256sum $RLDX_CAL/lstm_seed{0,1,2}/normalized_scores.jsonl $RLDXHS_ROOT/scores/{train,calibration,test}_predictions.jsonl \
  $RLDX_LEAD_OUT/rldx_lead_lstm.json $RLDXHS_ROOT/rldx_hide_and_seek.json | tee $E2CO_LOG/frozen_inputs.sha256
cd $E2CO_REPO
"$SAFE_PY" -W ignore -m unittest tests.test_e2_coprimary_sizing tests.test_stage2_sizing 2>&1 | grep -E "^(Ran|OK|FAILED)"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup nice -n 10 \
  "$SAFE_PY" -I -u scripts/xr1_lead_time/e2_coprimary_sizing.py --calibration-dir $RLDX_CAL \
  --hs-scores-dir $RLDXHS_ROOT/scores --replication-json $RLDX_LEAD_OUT/rldx_lead_lstm.json \
  --hs-result $RLDXHS_ROOT/rldx_hide_and_seek.json --workers 8 \
  --output $E2CO_ROOT/e2_coprimary.json > $E2CO_LOG/e2_coprimary.log 2>&1 &
echo "sizing PID: $! on $(hostname)"
```

When it exits: `cat $E2CO_LOG/e2_coprimary.log` and
`sha256sum -c $E2CO_LOG/frozen_inputs.sha256`.

### Results (2026-10-23)

**Run.** Code `a21e0c0c`, folder `e2_coprimary_sizing_<timestamp>` (under
`robocasa_checkpoints/safe/`, log under `robocasa_logs/eval/`). 2,000 replicates per cell,
seed 20261021.

**Checks, both passed.**
- The replication fits reproduce SAFE + P +0.0342 (§15.15) and Hide-and-Seek + P +0.0834
  (`rldx_hide_and_seek.json`).
- All eight frozen input files were unchanged after the run.
- SAFE's plug-in rows equal Table S4 (mean gain, SD, A1 FPR pass at every n), as the shared
  draws require.

**Table S5 — plug-in sampling distribution (within task).** FPR pass is the share of
simulated studies meeting each FPR rule.

| n | SAFE mean gain (SD) | SAFE FPR pass A1 / E2 | Hide-and-Seek mean gain (SD) | Hide-and-Seek FPR pass A1 / E2 |
|---:|---|---|---|---|
| 10 | +0.040 (0.012) | 0.62 / 0.99 | +0.083 (0.017) | 0.61 / 0.99 |
| 20 | +0.036 (0.007) | 0.92 / 1.00 | +0.080 (0.011) | 0.90 / 1.00 |
| 30 | +0.035 (0.006) | 0.98 / 1.00 | +0.080 (0.009) | 0.98 / 1.00 |
| 40 | +0.034 (0.005) | 0.99 / 1.00 | +0.080 (0.008) | 0.99 / 1.00 |
| 50 | +0.034 (0.004) | 1.00 / 1.00 | +0.081 (0.008) | 1.00 / 1.00 |
| 60 | +0.034 (0.004) | 1.00 / 1.00 | +0.081 (0.007) | 1.00 / 1.00 |

**Table S6 — co-primary power at 97.5% (shift, within-task CI).** Each cell is power under
the E2 FPR rule / under the A1 rule.

| n | SAFE, γ = 0.034 | Hide-and-Seek, γ = 0.083 | Hide-and-Seek, γ = 0.052 | Hide-and-Seek, γ = 0.05 |
|---:|---|---|---|---|
| 10 | 0.99 / 0.62 | 0.99 / 0.61 | 0.91 / 0.56 | 0.88 / 0.53 |
| 20 | 1.00 / 0.92 | 1.00 / 0.90 | 0.99 / 0.89 | 0.98 / 0.89 |
| 30 | 1.00 / 0.98 | 1.00 / 0.98 | 1.00 / 0.98 | 1.00 / 0.98 |
| 60 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

- **Joint (97.5%, E2 rule).** At least one co-primary passes in every simulated study at
  every n. Both pass in 0.98 (n = 10) and 1.00 (n ≥ 20) of studies at (0.034, 0.083). At
  (0.034, 0.052), both pass in 0.90 at n = 10 and 0.99 at n = 20.
- **95% against 97.5%.** The CI changes power only at n = 10 and at the lower Hide-and-Seek
  gains. For example, Hide-and-Seek at γ = 0.05 has CI > 0 in 0.93 of studies at 95% and
  0.89 at 97.5%. From n = 20, CI > 0 is at least 0.99 everywhere.
- **Task-level CI (secondary, no power claim), 97.5%, E2 rule, n = 20.**
  - SAFE at 0.034: 0.33.
  - Hide-and-Seek: 0.99 at 0.083, 0.34 at 0.052, 0.27 at 0.05.

**Against the expectations.**
- **SAFE.** It behaved as written: the 97.5% level changes nothing, and power equals the FPR
  pass rate (0.92 under A1, 1.00 under E2 at n = 20).
- **Hide-and-Seek.** The plug-in gain is 0.080–0.083 with CI > 0 in every study, as
  expected. Its FPR pass rate is about SAFE's (0.90 against 0.92 under A1 at n = 20), not
  clearly lower as we expected.
- **At least one co-primary passing.** Equal to 1, as expected.

**Reading.**
- **The co-primary design costs almost nothing at E2's size.** At n = 20, power under the
  E2 rule is 1.00 for SAFE and at least 0.98 for Hide-and-Seek, even at its matched-FPR gain
  (0.052). Under H1's stricter fixed FPR limit it is 0.89–0.92.
- **What actually limits power is the FPR rule.** The E2 rule allows up to 39 of 400 false
  alarms (9.75% observed FPR), so it almost never fails here. Under the A1 rule, false alarms
  decide, at about 0.9 from n = 20.
- **These figures are optimistic.** Calibration and test samples come from one small
  held-out pool (8 successes and 5 failures per task), and inference is conditional on the
  20 tasks. With tasks resampled, power drops sharply for SAFE and for Hide-and-Seek at
  0.052. The task-level CI is reported without a power claim, as before.
