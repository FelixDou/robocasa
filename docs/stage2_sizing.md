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

## E2 rule under null fusions (2026-10-23)

**Summary for the protocol** (run 2026-10-23, `e2_null_rules_<timestamp>`, code `34ee5ac7`).
- Values are at n = 20 per class per task, at 97.5%, within task, over 2,000 simulated
  studies, unless stated.
- Type I error is per detector: the share of null studies that pass.
- "Nested" uses a real inner decision bootstrap (100 studies per cell). "Approx" uses the
  g − hw shortcut, which the nested check shows to be badly anticonservative for the noise
  null (see Results).

| Rule | Type I, noise + P | Type I, resampled SAFE + P | SAFE + P power (γ = 0.034) | Hide-and-Seek + P power (γ = 0.052) | Both pass (0.034, 0.052) | Smallest n with both ≥ 0.8 |
|---|---|---|---:|---:|---:|---:|
| (i) current E2 rule | 0.020 nested (approx 0.217) | 0.000 nested (approx 0.001) | 1.00 | 0.99 | 0.99 | 10 |
| (ii) (i) + CI > 0 against P tuned to the fused FPR | ≤ 0.020 (bound from nested (i); approx 0.055) | ≤ 0.001 | 0.98 | 0.98 | 0.96 | 10 |
| (iii) (i) + Monte Carlo null test (K = 39 per null) | ≤ 0.020 (bound from nested (i); approx 0.039) | 0.000 | 1.00 | 0.99 | 0.99 | 10 |

At n = 10, rule (i)'s nested type I under noise is 0.050 (SE about 0.022), and both pass in
0.90 / 0.74 / 0.81 of studies under (i) / (ii) / (iii). At n = 30 the nested type I is 0.010.

**Recommendation.**
- By the pre-stated rule: **rule (i), smallest n = 10.**
  - Only rule (i) is eligible as the rule is written. Its approximated type I error fails,
    but the nested check puts it on the other side of 0.05, and the rule says the nested
    value then decides.
  - Rules (ii) and (iii) miss by Monte Carlo error on the approximated values (0.055 at
    n = 20; 0.053 at n = 10).
- **Read with the nested check (post hoc), the order is the same.** (ii) and (iii) are
  subsets of (i), so they have no higher type I error than nested (i). All three rules are
  then eligible, and the tie-break picks (i).
- **Two points the rule does not capture** (judgment, not pre-stated):
  - **Use n = 20, not n = 10.** At n = 10 the noise type I error sits at 0.050 with an SE of
    about 0.022, and the family-wise error over two co-primaries could reach about 0.10. At
    n = 20 it is 0.020 (family-wise ≤ 0.04), and both co-primaries pass in 0.99 of studies.
  - **Consider registering (iii) instead of (i).** This simulation cannot produce the
    reviewer's mechanism, a fused detector spending more false alarms on the test cohort
    than calibration allowed: calibration and test share one pool. Rule (iii) guards
    against that because its nulls run on the same cohorts. It costs no power at n ≥ 20
    (0.99 both pass, as for (i)), and it makes the null control part of the decision, as
    the review asks. Rule (ii) has the same protection but costs 0.03 at n = 20, and its
    tuned-P comparator needs a fix first (see Results).
- **Author's decision (2026-10-23): rule (iii) at n = 20 per class per task** is the E2 rule
  for the Stage 2 protocol.

**Why.** The mock review (concern 2, `claude/rss-stage1-mock-review`) argues that E2's rule
can be passed with false-alarm budget alone:
- the within-task CI is narrow (SD 0.007 at n = 20);
- the binomial FPR rule admits up to 39 of 400 false alarms;
- on RLDX-1, two of 20 noise + P draws gained +0.038 and +0.049, at FPR 0.08 (§15.15).

It asks for the rule's type I error under null fusions next to its power. It also asks for
the null control to become part of the decision.

**Status.** This is exploratory sizing on opened cohorts, like everything above. This
section, the code and the tests were committed before the script was run on real data.

### Method

**Code.** `scripts/xr1_lead_time/e2_null_rules.py`, with tests in
`tests/test_e2_null_rules.py`.

**Studies.** At n = 10, 20 and 30 per class per task, 2,000 within-task studies each. These
are exactly the studies of `e2_coprimary_sizing.py`: the same held-out pool (8 successes and
5 failures per task), seed, cells and draws. The test checks this on synthetic data. On the
cluster, the run stops if SAFE's and Hide-and-Seek's mean gain and SD at each n differ from
`e2_coprimary.json`.

In every study, on the same drawn rollouts, P is fitted once. Four detectors are then fused
with P and fitted exactly like SAFE + P (`confirm_lead.fit_fast`):
- **SAFE + P** and **Hide-and-Seek + P**, the co-primaries;
- **(a) noise + P:** the running maximum of i.i.d. N(0, 1), a fresh draw for every simulated
  rollout;
- **(b) within-task resampled SAFE + P:** for each seed, normalized SAFE values drawn with
  replacement from all held-out inferences of the same task, outcome ignored. Each seed's
  running maximum is taken, then the mean over the three seeds. This is the construction of
  E1's H2 null.

The nulls come from a separate random stream, so they do not disturb the shared draws.

**Rule components, for a detector with gain g against P in a study.**
- **Decision CI > 0 at 97.5% (within task).** Approximated as in E2 and the co-primary
  sizing: g − hw > 0, where hw is the median minus the 1.25th percentile of that detector's
  own sampling distribution at that n. For a null, hw comes from the null's own
  distribution, as a study's bootstrap would.
- **E2 FPR rule.** The false alarms must not reject "true FPR ≤ 0.075" in a one-sided exact
  binomial test at level 0.05. That allows at most 21, 39 and 56 false alarms of 200, 400 and
  600 test successes.
- **Tuned-P CI, rule (ii).** This is Amendment 1's registered secondary, made within task:
  - on the study's test cohort, P's margin is chosen from confirm_lead's grid
    (−0.5 to 1.0 in steps of 0.0025);
  - the choice maximizes mean lead with no more false alarms than the fused detector has;
  - the gain against that tuned P gets a test-only bootstrap with thresholds fixed:
    failures resampled within task, 1,000 replicates, 97.5% percentile interval;
  - it must have a lower bound above 0.
- **Monte Carlo null test, rule (iii).**
  - In each study, K = 39 fresh draws of each null are fused with P and refitted on the same
    rollouts.
  - The detector passes if p = (1 + #{null gains ≥ g})/(K + 1) ≤ 0.05 for **both** nulls,
    i.e. if g beats at least 38 of the 39 draws of each.
  - Written this way, rule (iii)'s type I error is at most 0.05 under either null by
    exchangeability. A plain "95th percentile of 20 draws" would allow about 0.09.
  - Ties count against the detector. A fit that falls back to the timer gives a gain of
    exactly 0, which ties with every reference that also falls back, so the test is
    conservative.
  - Checked on synthetic data before the run: over 400 studies with K = 19, a null passed
    the test against its own null in 0.028 (noise) and 0.048 (resampled) of studies.

**Rules.**
- **(i)** CI > 0 and the E2 FPR rule (the current rule).
- **(ii)** (i) and the tuned-P CI > 0.
- **(iii)** (i) and the Monte Carlo null test.

**Outputs.**
- **Type I error** of each rule under each null: the share of null studies passing it, with
  its Monte Carlo SE (about 0.005 at 0.05). Each component's pass rate is reported as well.
- **Co-primary power** of each rule:
  - at the plug-in gains;
  - at SAFE γ = 0.034 and Hide-and-Seek γ = 0.083, 0.052 and 0.05, imposed by shift.

  Under a shift, the decision-CI and tuned-P bounds move by the same amount, and the null
  reference gains do not move. The power that both co-primaries pass is given at
  (0.034, 0.083) and (0.034, 0.052).
- **Nested check of rule (i) for the nulls.** At n = 10, 20 and 30, 100 studies per null get
  a real inner decision bootstrap (200 replicates, rollouts within task and class, refit).
  Its rule (i) pass rate is compared with the approximation on the same studies.

**Type I error as reported** is per detector. A 97.5% two-sided CI is a one-sided test at
0.0125. With two co-primaries, the family-wise error is at most the sum of the two
per-detector rates.

### Recommendation rule, fixed before running

1. A rule is **eligible** if its type I error is at most 0.05 under both nulls at n = 10, 20
   and 30 (point estimates).
   - If the nested check gives rule (i) a type I error on the other side of 0.05 from the
     approximation at any n, the nested value decides whether rule (i) is eligible.
   - Rules (ii) and (iii) only add conditions to (i), so they cannot be less strict than
     their tuned-CI or Monte Carlo component alone.
2. Among eligible rules, **recommend** the one with the highest power that both co-primaries
   pass at the planning gains (SAFE 0.034, Hide-and-Seek 0.052) at n = 20. On a tie, prefer
   the simpler rule, in the order (i), (ii), (iii).
3. Under that rule, report the **smallest n** in {10, 20, 30} at which SAFE (0.034) and
   Hide-and-Seek (0.052) each have power at least 0.8. If none does, say so and give the
   power at n = 30.

Hide-and-Seek's planning gain is 0.052, its gain against P at matched false alarms (§15.22),
not 0.083, its opened point estimate. The script applies this rule mechanically and prints
the result.

### Expectations, written before running

- **Checks.**
  - Both replication gains reproduce (SAFE +0.0342, Hide-and-Seek +0.0834).
  - The co-primary draws match `e2_coprimary.json`.
- **Nulls fused with P.**
  - The mean null gain is small, between 0 and +0.01, like the replication's +0.009 for
    noise.
  - The E2 FPR rule passes in nearly every null study: calibration and test come from one
    pool here, so the fused test FPR stays near 0.05, far below the 9.75% the rule allows.
- **Rule (i).** Its type I error is governed by the null's mean gain relative to its own
  97.5% half-width. We expect it to exceed 0.05 for at least one null at n = 20 or 30. If
  the null mean is positive, it should grow with n, because the half-width shrinks while
  the mean does not. We do not predict its size.
- **Rule (ii).**
  - Type I at most about 0.02 under both nulls: a tuned timer with the same number of false
    alarms should beat randomly placed alarms.
  - SAFE's RLDX-1 gain came at an unchanged FPR (§15.15), so its tuned-P gain should stay
    near its gain, with power ≥ 0.8 at n = 20.
  - Hide-and-Seek at plug-in should pass easily, since its matched-FPR gain is 0.052. Shifted
    to γ = 0.052, its tuned-P gain is near 0.02, so its power there is uncertain.
- **Rule (iii).**
  - Type I at most 0.05 by construction, and lower in practice, because the CI and FPR
    conditions also apply.
  - The bar (the second-largest of 39 null gains) should lie near the null mean plus about
    two null SDs, roughly 0.015–0.03.
  - SAFE at 0.034 should pass in most studies from n = 20, and less often at n = 10.
- **Recommendation.** One of (ii) or (iii). We do not predict which.

### Limits

- **One shared, small pool.** Calibration and test samples are drawn from the same held-out
  pool (5 failures and 8 successes per task). Cohorts are therefore exchangeable here, and
  the fused detector can spend extra false-alarm budget on the test cohort only through
  sampling noise.
  - A real Stage 2 calibration and test cohort can differ. On XR-1 the test FPR ran at
    0.058–0.061 against the 0.05 target.
  - That would raise the type I error of rule (i), which has no paired control, and leave
    rules (ii) and (iii) largely unaffected, since both compare on the same test cohort.
  - These type I estimates are therefore lower bounds for rule (i).
- **Tail resampling.** Resampling cannot create new extreme scores. With only 5 failures per
  task, the co-primaries' sampling SD is likely understated, as in the earlier sections.
- **Inference is conditional on the 20 tasks.**
- **The decision CI is approximated**, and checked by nested bootstraps for rule (i) under
  the nulls only.
- **The shift assumes the deficit against a tuned P is unchanged** when the true gain is
  lower.
- **The nulls fix the scope of "type I error".** Noise and within-task resampled SAFE carry
  no information about which rollout fails. A detector with weak but real signal is not a
  null.

### Cluster commands

CPU only, from a pinned detached worktree, about 30–60 minutes with 8 workers. Outputs go
under `/gs/bs`. `E2N_SHA` is the full SHA of the pushed commit that contains this section.

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
source $STORAGE_BS/robocasa_logs/eval/rldx1_hide_and_seek_20261023_000418/paths.env
export E2CO_JSON=$(ls -d $STORAGE_BS/robocasa_checkpoints/safe/e2_coprimary_sizing_*/e2_coprimary.json | sort | tail -n 1)
echo "co-primary run: $E2CO_JSON"
export E2N_SHA=<full SHA>
export E2N_REPO=$PROJECT_FS/robocasa_e2null_${E2N_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin
test -e $E2N_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $E2N_REPO $E2N_SHA
test "$(git -C $E2N_REPO rev-parse HEAD)" = "$E2N_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
export E2N_TAG=e2_null_rules_$(date +%Y%m%d_%H%M%S)
export E2N_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$E2N_TAG E2N_LOG=$STORAGE_BS/robocasa_logs/eval/$E2N_TAG
mkdir -p $E2N_ROOT $E2N_LOG
sha256sum $RLDX_CAL/lstm_seed{0,1,2}/normalized_scores.jsonl $RLDXHS_ROOT/scores/{train,calibration,test}_predictions.jsonl \
  $RLDX_LEAD_OUT/rldx_lead_lstm.json $RLDXHS_ROOT/rldx_hide_and_seek.json $E2CO_JSON | tee $E2N_LOG/frozen_inputs.sha256
cd $E2N_REPO
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_e2_null_rules tests.test_e2_coprimary_sizing \
  tests.test_stage2_sizing 2>&1 | grep -E "^(Ran|OK|FAILED)"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup nice -n 10 \
  "$SAFE_PY" -I -u scripts/xr1_lead_time/e2_null_rules.py --calibration-dir $RLDX_CAL \
  --hs-scores-dir $RLDXHS_ROOT/scores --replication-json $RLDX_LEAD_OUT/rldx_lead_lstm.json \
  --hs-result $RLDXHS_ROOT/rldx_hide_and_seek.json --coprimary-json $E2CO_JSON --workers 8 \
  --output $E2N_ROOT/e2_null_rules.json > $E2N_LOG/e2_null_rules.log 2>&1 &
echo "PID $! on $(hostname); log $E2N_LOG/e2_null_rules.log"
```

Check progress with `tail -n 3 $E2N_LOG/e2_null_rules.log`: the two check lines print within
a minute, and the tables print at the end. When the process has exited, paste back the
output of:

```bash
cat $E2N_LOG/e2_null_rules.log; sha256sum -c $E2N_LOG/frozen_inputs.sha256
```

Stop conditions: a replication or co-primary-draw mismatch (the script exits with `STOP`),
a failed checksum, or an error. Do not edit inputs to make the run pass.

### Results (2026-10-23)

**Run.**
- Code `34ee5ac7`, worktree `robocasa_e2null_34ee5ac7`.
- Output `e2_null_rules_<timestamp>/e2_null_rules.json` under `robocasa_checkpoints/safe/`;
  log under `robocasa_logs/eval/`.
- 2,000 studies per n, seed 20261021, K = 39 reference draws per null in every study.

**Checks, all passed.**
- The replication gains reproduce SAFE + P +0.0342 and Hide-and-Seek + P +0.0834.
- The co-primary draws match `e2_coprimary_sizing_20261023_011537/e2_coprimary.json`.
- All nine frozen input files are unchanged (`sha256sum -c` OK).
- SAFE's and Hide-and-Seek's mean gains and SDs equal Table S5 at n = 10, 20 and 30.

**Table N1 — detectors fused with P (within task).** Gains are against P; hw is the 97.5%
half-width of the detector's own sampling distribution.

| Detector | n | Mean gain (SD) | hw | Mean fused FPR | Mean gain against tuned P |
|---|---:|---|---:|---:|---:|
| SAFE | 10 / 20 / 30 | +0.040 (0.012) / +0.036 (0.007) / +0.035 (0.006) | 0.015 / 0.011 / 0.009 | 0.052 / 0.050 / 0.049 | +0.031 / +0.033 / +0.033 |
| Hide-and-Seek | 10 / 20 / 30 | +0.083 (0.017) / +0.080 (0.011) / +0.080 (0.009) | 0.030 / 0.025 / 0.021 | 0.053 / 0.050 / 0.049 | +0.073 / +0.078 / +0.081 |
| Noise | 10 / 20 / 30 | +0.0055 (0.0081) / +0.0037 (0.0048) / +0.0031 (0.0040) | 0.010 / 0.007 / 0.007 | 0.054 / 0.053 / 0.052 | −0.006 / −0.003 / −0.001 |
| Resampled SAFE | 10 / 20 / 30 | +0.0000 (0.0003) / 0 / 0 | 0 | 0.045 / 0.047 / 0.047 | −0.002 / +0.000 / +0.001 |

- **Resampled SAFE.** Its fit falls back to the timer in essentially every study, so its
  gain is exactly 0. On this pool it is not an informative null.
- **Noise.** Its small positive gain is a timing effect: the running maximum of noise rises
  with elapsed time and acts as a second, noisier clock.
- **The rule (iii) bar** (the second-largest of 39 reference gains, mean over studies) is
  +0.020, +0.012 and +0.010 for noise at n = 10, 20 and 30, and 0 for resampled SAFE.

**Table N2 — type I error.** Each cell is the share of null studies passing. The Monte Carlo
SE of the approximated rates is 0.004–0.009. Nested uses 100 studies, SE about 0.01–0.02.

| Null | n | Rule (i) approx | Rule (i) nested | Rule (ii) approx | Rule (iii) approx | FPR rule only | Tuned CI only |
|---|---:|---:|---:|---:|---:|---:|---:|
| Noise | 10 | 0.221 | **0.050** | 0.032 | 0.053 | 0.982 | 0.303 |
| Noise | 20 | 0.217 | **0.020** | 0.055 | 0.039 | 0.994 | 0.404 |
| Noise | 30 | 0.157 | **0.010** | 0.033 | 0.036 | 0.999 | 0.472 |
| Resampled SAFE | 10 | 0.001 | 0.000 | 0.001 | 0.000 | 0.995 | 0.601 |
| Resampled SAFE | 20 | 0.001 | 0.000 | 0.001 | 0.000 | 0.999 | 0.778 |
| Resampled SAFE | 30 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.836 |

**Table N3 — co-primary power by rule (i) / (ii) / (iii)** (shift, within-task CI, E2 FPR
rule).

| Detector, assumed gain | n = 10 | n = 20 | n = 30 |
|---|---|---|---|
| SAFE, 0.034 | 0.99 / 0.86 / 0.90 | 1.00 / 0.98 / 1.00 | 1.00 / 0.99 / 1.00 |
| Hide-and-Seek, 0.083 | 0.99 / 0.98 / 0.99 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Hide-and-Seek, 0.052 | 0.91 / 0.85 / 0.91 | 0.99 / 0.98 / 0.99 | 1.00 / 1.00 / 1.00 |
| Hide-and-Seek, 0.05 | 0.88 / 0.82 / 0.88 | 0.98 / 0.97 / 0.98 | 1.00 / 1.00 / 1.00 |
| Both pass, (0.034, 0.083) | 0.98 / 0.84 / 0.89 | 1.00 / 0.98 / 1.00 | 1.00 / 0.99 / 1.00 |
| Both pass, (0.034, 0.052) | 0.90 / 0.74 / 0.81 | 0.99 / 0.96 / 0.99 | 1.00 / 0.99 / 1.00 |

Plug-in power (no shift) is within 0.06 of the SAFE 0.034 and Hide-and-Seek 0.083 rows at n = 10, and within 0.01 from n = 20.

**What the nested check shows.**
- **The g − hw shortcut overstates rule (i)'s type I error under noise by a factor of 5 to
  16:** 0.25 against 0.05 at n = 10, 0.24 against 0.02 at n = 20, 0.16 against 0.01 at
  n = 30.
- **Why.** The noise null's sampling distribution has a point mass at 0 (studies whose fit
  falls back to the timer) plus small positive gains. Its median minus 1.25th percentile is
  therefore tiny, and the shortcut's lower bound sits close to the study's own gain. A real
  inner bootstrap refits on resampled rollouts, often falls back to the timer, and gets a
  lower bound at or below 0.
- **The co-primaries are not affected.** Their approximation matched nested bootstraps
  exactly in the E2 check above (Table S4).
- **Type I conclusions therefore rest on the nested values.** Rules (ii) and (iii) only add
  conditions to (i), so their type I error with a real CI is at most nested (i).

**Against the expectations.**

| Expectation (written before running) | Result |
|---|---|
| Checks pass | Yes |
| Mean null gain 0 to +0.01 | Yes. Noise +0.003 to +0.006; resampled SAFE exactly 0 (timer fallback) |
| E2 FPR rule passes nearly every null study | Yes (0.98–1.00): it offers no protection here |
| Rule (i) type I > 0.05 for at least one null at n = 20 or 30, growing with n | **No.** Nested 0.02 and 0.01, falling with n. Only the shortcut exceeds 0.05 |
| Rule (ii) type I ≤ about 0.02 | Approx 0.032–0.055; ≤ 0.020 at n ≥ 20 by the nested bound |
| SAFE's tuned-P gain stays near its gain; power ≥ 0.8 at n = 20 | Yes (+0.033 against +0.036; 0.98) |
| Hide-and-Seek at shifted 0.052 under (ii): uncertain | 0.98 at n = 20 |
| Rule (iii) type I ≤ 0.05; bar about 0.015–0.03 | 0.036–0.053 (within MC error of its exact 0.05 bound); bar 0.010–0.020 |
| SAFE under (iii) passes in most studies from n = 20 | Yes (1.00; 0.90 at n = 10) |
| Recommendation is (ii) or (iii) | **No.** The pre-stated rule gives (i) |

**Issues found.**
- **A grid artifact in the tuned-P comparator.**
  - Rule (ii) searches P's margin on confirm_lead's grid (steps of 0.0025). The calibrated
    margin of P usually falls between grid points.
  - When the fused fit falls back to the timer (fused = P), the best grid timer with no
    more false alarms is slightly later than P. The per-failure differences are then all
    ≥ 0, and some are positive, so their bootstrap lower bound is above 0.
  - This is why "tuned CI only" passes in 0.30–0.84 of null studies even though the null's
    mean tuned gain is ≤ 0.001.
  - It does not move rule (ii)'s type I error, because rule (i)'s CI blocks those fallback
    studies.
  - If rule (ii) is registered, the tuned timer's candidates must include P's own
    calibrated margin, so that a detector identical to P gets exactly 0.
- **Resampled SAFE is not an informative null on RLDX-1's held-out pool**: it never changes
  the fit. Noise is the binding null here.

**Reading (exploratory).**
- **Concern 2's mechanism does not appear here.** In this exchangeable simulation, E2's
  rule (i) holds its type I error at 0.02 at n = 20 under both nulls.
  - The protection comes from the decision CI, not from the FPR rule, which passes almost
    every null study.
  - The reviewer's example (noise gaining +0.038/+0.049 at FPR 0.08 in §15.15) came from
    60 calibration successes and a test FPR well above P's. At E2's size the fused noise
    FPR averages 0.053, and the 95th percentile of noise gains is +0.013.
- **The simulation cannot test the reviewer's real worry.** That worry is a fused detector
  spending more false alarms on a test cohort that differs from calibration. Rules (ii) and
  (iii) guard against it by construction, at no power cost for (iii) from n = 20. That is
  the case for registering (iii) even though the pre-stated rule picks (i).
- **The power figures for the protocol:**
  - at n = 20, both co-primaries pass in 0.99 (i), 0.96 (ii) and 0.99 (iii) of studies at
    (SAFE 0.034, Hide-and-Seek 0.052);
  - each detector individually is at 0.98 or more.
- **Limits as stated above.** One shared pool, 5 failures per task, inference conditional on
  the 20 tasks, a shift model for assumed gains.

## E3 with the measured same-scene retry rate (2026-10-23)

**Summary for the protocol** (run 2026-10-23, `e3_retry_sizing_<timestamp>`, code `ddab0ec4`).
- Retry success r = 0.245 (measured, post hoc), budget 1.25 H_k, two-sided 0.05, power 0.8.
- "Paired" means McNemar pairs for a design that branches at the alarm.

| Contrast | XR-1 (38 tasks) | RLDX-1 (20 tasks) |
|---|---|---|
| P vs none: success gain | +0.017 | +0.034 (atomic +0.054; composite −0.012) |
| P vs none: episodes per arm, unpaired | 14,014 (was 648 at r = p) | 3,280 (was 484) |
| P vs none: pairs, branching design | 2,306 (159,041 at r = 0.17) | **461** (970 at r = 0.17) |
| SAFE + P vs P, atomic at 1.0 H_k: pairs | 7,845 | 15,300 |
| Reset vs continuation: crossover c\* | 0.053 | 0.104 |
| Reset vs continuation: continuation proxy | 0.092 | 0.082 |
| Reset vs continuation: expected direction | continuation better (c\* < proxy < 0.27) | unknown (proxy < c\* < 0.27) |
| Reset vs continuation: pairs (pre-stated rule) | about 3,000 | about 6,500 |

**Recommendation.**
- **E3 at about 650 per arm is not worthwhile as specified.** With the measured retry rate,
  P vs no monitor needs 3,300–14,000 episodes per arm.
- **The one affordable superiority test** is P vs no monitor on RLDX-1 with a branching
  design. That needs 461 pairs at r = 0.245, or about 1,000 to survive r = 0.17.
  - Its effect is entirely on atomic tasks.
  - Its retry rate is borrowed from XR-1.
- **XR-1 supports no affordable superiority test of any monitor contrast.**
- **Reset against continuation hinges on c**, a continuation success rate nobody has
  measured, which sits close to the crossover on both policies.
  - Measure c first: a cheap continuation pilot whose time limit is raised to 1.25 H_k.
  - Until then, register E3 as an estimation study (both arms' success rates with CIs), not
    as a superiority test.
- Details and the reduced claim are in "Results" below.
- **Author's decision (2026-10-23).** E3 runs on RLDX-1 and asks reset vs continuation at
  1.25 H_k. P vs no monitor is descriptive only, since its no-monitor arm wastes the remaining
  budget. Whether E3 is a test or an estimation study is decided by the RLDX-1 continuation
  and retry pilot: [`rldx_e3_pilot_plan.md`](rldx_e3_pilot_plan.md).
- **Pilot result (2026-10-28; report §15.26).** c = 0.0405 [0.020, 0.070]; the scene gate
  failed, so r stayed at 0.245 / 0.17. The direction is not decided (at r = 0.17 the sign
  flips at c = 0.070), so **E3 is an estimation study at 1,000 pairs (50 per task)**:
  expected reset − continuation +0.028, CI half-width 0.017. Its reset arm must restore the
  failed attempt's scene explicitly (a state saved at reset, or per-episode seeding) and
  check scene identity, because RLDX-1's `official_rldx` protocol does not re-create scenes
  after each task's first.

**Why.** E3's sizing above assumed that a retry after an alarm succeeds at the task's
ordinary success rate (r = p_k). The same-scene retry pilot (report §15.24,
`xr1_retry_pilot_plan.md`) measured otherwise.
- **Status of the pilot.** Post hoc: its registered scene gate failed, and 2 of 194 scenes
  did not reproduce and were excluded.
- **Measured rates.** A retry of an originally failed scene succeeded in 24/98 = **0.245**
  (Wilson [0.170, 0.339]; task cluster [0.190, 0.303]). A retry of an originally successful
  scene succeeded in 66/94 = 0.702.
- **Comparison with p.** The 10 pilot tasks have task-macro p = 0.486, so r_fail / p = 0.504.
  r_fail is below p_k on every task.

The drafts session also proposed **alarm-triggered reset against continuation** (the same
rollout keeps running to 1.25 H_k) as E3's primary contrast. This section sizes both.

**Status.** This is exploratory sizing on opened cohorts and a post hoc pilot. This section
and the code changes were committed before the run on real data.

### Method

The E3 model above is unchanged (`stage2_sizing.py e3`). What changes:

- **Retry success rates**, one run each:
  - `0.245`: the pooled measured rate, for every task and both policies;
  - `0.17`: its Wilson lower bound (sensitivity);
  - `scale:0.504`: r_k = 0.504 × p_k per task, so that tasks keep their relative difficulty;
  - `scale:0.350` (= 0.17 / 0.486): the per-task version of the lower bound;
  - `same` (r = p_k): the old assumption, run only as a check. It must reproduce the
    2026-10-21 E3 tables exactly (`--reference-json`); otherwise the run stops.

  r applies to every retry (a second retry of the same scene is assumed no harder than the
  first). The rate is measured on XR-1 and assumed for RLDX-1.
- **Episodes per arm for P vs none and SAFE + P vs P, at every budget**, two ways:
  - **Unpaired:** the two-proportion formula on task-macro success rates (two-sided 0.05,
    power 0.8), as before. This fits independent runs on matched scenes.
  - **Paired by seed (McNemar, Connor 1987):** for a design that *branches at the alarm*
    (restore the simulator state, then reset in one arm and continue in the other), so the
    two arms share their first attempt up to the alarm.
    - The effect is the common-random-number difference.
    - The discordant total comes from runs whose retries are drawn independently in each
      arm, as two stochastic policy runs would be.
    - Not modelled: a positive correlation between arms' retries on hard scenes. It would
      lower n, so these figures are conservative on that point.
- **Reset against continuation at 1.25 H_k** (P or SAFE + P as the alarm; the continuation
  arm has no monitor):
  - first-attempt successes succeed in the continuation arm, but can be aborted by a false
    alarm in the reset arm;
  - first-attempt failures succeed in the continuation arm with probability c, independently
    of the reset arm's retries;
  - reported for c ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.27, 0.40}: both arms' success rates, their
    difference, and the pairs needed (McNemar, branching design) and episodes per arm
    (unpaired).
- **What the data say about c.** No rollout ran past H_k, so c cannot be measured. Three
  reference points:
  - **Late-success hazard proxy (computed).** Of the rollouts still running at 0.75 H_k, the
    share that succeed by H_k: XR-1 from all retained rows of both E1 cohorts, RLDX-1 from all
    rows. It is likely optimistic for c. Rollouts still running at H_k are the slowest ones,
    and a population's success hazard usually falls with time.
  - **0.27:** the share of §15.17's labeled failures in the category "slow or nearly
    succeeded" (13/48).
  - **0.79:** the share judged "recoverable until timeout" (§15.17). This is a ceiling, not
    an estimate: recoverable does not mean the policy would recover.
- **Crossover c\*.** The continuation success at which reset and continuation tie, read off
  the grid by linear interpolation.

### Expectations, written before running

- **Check.** `same` reproduces the 2026-10-21 tables exactly.
- **P vs none at 1.25 H_k.** Retry gains scale roughly with r / p.
  - At r = 0.245 the XR-1 gain falls from 0.075 to about 0.03, and unpaired n rises from about
    650 to about 3,000–4,000 per arm. RLDX-1 is similar, at about 2,000–3,000.
  - At r = 0.17, roughly twice that again.
  - The per-task scaling (0.504 p_k) gives about the same as r = 0.245.
- **Paired branching design.** Several times smaller than unpaired, because only
  first-attempt failures can be discordant: hundreds to about 1,500 pairs.
- **SAFE + P vs P, XR-1 atomic at 1.0 H_k.** The gain falls from 0.044 to about 0.015.
  Unpaired n is above 10,000; paired, a few thousand.
- **Reset against continuation.**
  - c\* is low, about 0.05–0.12. A reset at the alarm leaves about 0.5 H_k, which fits one
    retry that succeeds about a quarter of the time. Continuation instead gives every failure
    a chance c.
  - The hazard proxy is likely above c\*, so the expected direction favours continuation.
  - Near c\* the needed n is unbounded. At the hazard proxy it should be small (low hundreds).

### Recommendation rule, fixed before running

- **P vs none at 1.25 H_k (r = 0.245):**
  - unpaired n above 2,000 per arm on both policies → E3 at about 650 per arm is not
    worthwhile as specified;
  - paired branching n ≤ 1,000 → propose the branching design at that size (state
    save/restore at the alarm, plus the scene-identity check of §15.24);
  - otherwise, reduce the claim to a description of retry outcomes with CIs and register no
    superiority test.
- **Reset against continuation:**
  - if both the hazard proxy and 0.27 exceed c\* on a policy, the test's expected direction
    there is "continuation better". Register it as two-sided, sized at the pairs needed at
    c = the hazard proxy, and state the expected direction;
  - if c\* lies between them, the direction is unknown in advance. Register it as two-sided,
    sized at the larger of the two n, and say so.
- **SAFE + P vs P** stays descriptive unless its paired n at 1.0 H_k is ≤ 1,000 on XR-1
  atomic tasks.

### Limits

- **Pilot coverage.** r is post hoc, from 98 failed scenes on 10 mid-success XR-1 tasks.
  High- and low-success tasks, and RLDX-1, are extrapolations.
- **Retry duration.** A retry's duration is drawn from the task's ordinary success durations.
  Hard scenes may take longer, which would lower retry benefits further.
- **Continuation success** is assumed and independent of the reset arm's retries.
- **Shadow data and the bound.** The alarm times come from shadow data, and the
  retry-benefit model is the bound of the earlier E3 section, with its limits.

### Cluster commands

CPU only, about 10–20 minutes for all five runs. `E3R_SHA` is the full SHA of the pushed
commit that contains this section.

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export E1_OUT=$STORAGE_BS/robocasa_rollouts/safe/xr1_lead_calibration_20261020_001108/lead_analysis
export XR1_RUNTIME_BUNDLE=$STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_calibration/runtime_bundle.json
export RLDX_CAL=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_safe20_natural_weighted_20260815_162530/calibration/lstm
export E3_REF=$STORAGE_BS/robocasa_checkpoints/safe/stage2_sizing_20261021_131345/e3.json
export E3R_SHA=<full SHA>
export E3R_REPO=$PROJECT_FS/robocasa_e3retry_${E3R_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin
test -e $E3R_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $E3R_REPO $E3R_SHA
test "$(git -C $E3R_REPO rev-parse HEAD)" = "$E3R_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
export E3R_TAG=e3_retry_sizing_$(date +%Y%m%d_%H%M%S)
export E3R_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$E3R_TAG E3R_LOG=$STORAGE_BS/robocasa_logs/eval/$E3R_TAG
mkdir -p $E3R_ROOT $E3R_LOG
sha256sum $E1_OUT/calibration_with_baselines.jsonl $E1_OUT/test_with_baselines.jsonl $XR1_RUNTIME_BUNDLE \
  $RLDX_CAL/lstm_seed{0,1,2}/normalized_scores.jsonl $E3_REF | tee $E3R_LOG/frozen_inputs.sha256
cd $E3R_REPO
CUDA_VISIBLE_DEVICES="" $SAFE_PY -W ignore -m unittest tests.test_stage2_sizing 2>&1 | grep -E "^(Ran|OK|FAILED)"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nohup sh -c "
for r in same 0.245 0.17 scale:0.504 scale:0.350; do
  tag=\$(echo \$r | tr ':.' '__')
  ref=''; test \$r = same && ref='--reference-json $E3_REF'
  nice -n 10 $SAFE_PY -I -u scripts/xr1_lead_time/stage2_sizing.py e3 --calibration $E1_OUT/calibration_with_baselines.jsonl \
    --test $E1_OUT/test_with_baselines.jsonl --runtime-bundle $XR1_RUNTIME_BUNDLE --rldx-calibration-dir $RLDX_CAL \
    --retry-success \$r \$ref --output $E3R_ROOT/e3_r_\$tag.json > $E3R_LOG/e3_r_\$tag.log 2>&1 || exit 1
done
echo ALL DONE > $E3R_LOG/done.txt" > $E3R_LOG/driver.log 2>&1 &
echo "driver PID $!; logs in $E3R_LOG"
```

When `done.txt` exists (or the driver stops), paste back the output of:

```bash
grep -h "check" $E3R_LOG/e3_r_*.log; cat $E3R_LOG/driver.log
for f in 0_245 0_17 scale_0_504 scale_0_350; do echo "=== r = $f"; awk '/episodes per arm \(unpaired/{p=1} p' $E3R_LOG/e3_r_$f.log; done
sha256sum -c $E3R_LOG/frozen_inputs.sha256
```

The driver stops at the first failing run. The `same` run must print
`check: reference E3 tables MATCH`.

### Results (2026-10-23)

**Run.**
- Code `ddab0ec4`, worktree `robocasa_e3retry_ddab0ec4`.
- Five runs: `same`, 0.245, 0.17, `scale:0.504`, `scale:0.350`.
- Output `e3_retry_sizing_<timestamp>/e3_r_*.json` under `robocasa_checkpoints/safe/`; logs
  under `robocasa_logs/eval/`.

**Checks.**
- `same` reproduced the 2026-10-21 E3 tables exactly (`check: reference E3 tables MATCH`).
- Every run reproduced the registered leads: XR-1 P 0.2252 and SAFE + P 0.2550; RLDX-1
  0.2711 and 0.3053.
- All seven input files were unchanged.

**Table R1 — P vs no monitor at 1.25 H_k** (all tasks; atomic in brackets). Each cell is the
success gain, then episodes per arm unpaired / pairs for the branching design.

| Retry success | XR-1 | RLDX-1 |
|---|---|---|
| r = p_k (2026-10-21) | +0.075 → 648 / – | +0.086 → 484 / – |
| 0.245 | +0.017 → 14,014 / 2,306 (atomic +0.023 → 6,035 / 1,155) | +0.034 → 3,280 / 461 (atomic +0.054 → 1,211 / 252) |
| 0.17 | +0.002 → 1,184,927 / 159,041 | +0.021 → 8,919 / 970 (atomic +0.035 → 2,980 / 465) |
| 0.504 · p_k | +0.023 → 7,325 / 1,226 | +0.043 → 2,024 / 333 |
| 0.350 · p_k | +0.007 → 88,030 / 12,019 | +0.028 → 4,954 / 643 |

**At other budgets (r = 0.245):**
- **XR-1 P vs none is negative at 1.0 H_k (−0.014) and about 0 at 1.5 H_k.** P's false
  alarms abort successes (exposure 0.032) at an unchanged rate, while retry gains shrink
  with r. On XR-1 a timer-triggered reset barely pays for its own false alarms.
- **RLDX-1 composite tasks lose with P at 1.0–1.25 H_k (−0.012).** No retry fits there.

**SAFE + P vs P** stays far out of reach. At 1.0 H_k on XR-1 atomic tasks (r = 0.245):
gain +0.007, 63,858 per arm unpaired or 7,845 pairs. RLDX-1 needs at least 11,000 pairs at
every budget.

**Table R2 — reset at the alarm (P) against continuation to 1.25 H_k, r = 0.245.**
- c\* is the continuation success at which the two arms tie.
- "Proxy" is the late-success hazard: of the rollouts still running at 0.75 H_k, the share
  that succeed by H_k.
- Pairs are for the branching design.

| Policy, tasks | Reset − continuation at c = 0 | c\* | Proxy | Pairs at the proxy | Pairs at c = 0.27 |
|---|---:|---:|---:|---:|---:|
| XR-1, all | +0.023 | 0.053 | 0.092 | about 3,000 (continuation better) | 147 |
| XR-1, atomic | +0.028 | 0.097 | 0.091 | over 10^5 (tie) | 407 |
| XR-1, composite | +0.022 | 0.044 | 0.092 | about 1,650 (continuation better) | 116 |
| RLDX-1, all | +0.046 | 0.104 | 0.082 | about 6,500 (reset better) | 225 |
| RLDX-1, atomic | +0.070 | 0.188 | 0.062 | about 350 (reset better) | 1,194 |
| RLDX-1, composite | −0.012 | below 0 | 0.130 | about 90 (continuation better) | 44 |

- **Sensitivity of c\*.** Across the four retry rates, c\* ranges from 0.015 to 0.066 for
  XR-1 (all tasks) and from 0.066 to 0.137 for RLDX-1. Lower r moves c\* down, so
  continuation is favoured more.
- **Reset with SAFE + P** is within about 0.005 of reset with P in every cell.
- **Pairs at the proxy** are read from the grid (linear in the difference, n ∝ 1/Δ²). They
  are approximate.

**Decisions under the pre-stated rule.**
- **P vs none at 1.25 H_k.**
  - Unpaired n exceeds 2,000 on both policies (14,014 and 3,280), so E3 at about 650 per
    arm is not worthwhile as specified.
  - Paired branching n is at most 1,000 on RLDX-1 (461), so propose that design there.
  - It is not, on XR-1 (2,306), so XR-1 gets a description only.
- **Reset against continuation.**
  - XR-1: c\* lies below both the proxy and 0.27, so the expected direction is
    "continuation better". It would be registered two-sided at about 3,000 pairs (at the
    proxy).
  - RLDX-1: c\* lies between them, so the direction is unknown. It would be registered
    two-sided at the larger n, about 6,500 pairs.
- **SAFE + P vs P.** Paired n is 7,845 on XR-1 atomic tasks at 1.0 H_k, above 1,000, so it
  stays descriptive.

**Against the expectations.**

| Expectation (written before running) | Result |
|---|---|
| `same` reproduces the tables | Yes |
| P vs none (r = 0.245): XR-1 gain about 0.03, n about 3,000–4,000; RLDX-1 about 2,000–3,000 | **XR-1 worse:** +0.017, n 14,014, because the false-alarm cost does not scale with r. RLDX-1 as expected (3,280) |
| r = 0.17 about twice the n again | XR-1 far worse (gain about 0); RLDX-1 2.7× |
| Per-task scaling close to 0.245 | Somewhat better, because high-p atomic tasks keep a higher r |
| Branching design several times smaller: hundreds to about 1,500 | 6–7× smaller. RLDX-1 461; XR-1 2,306, above the range |
| SAFE + P vs P (XR-1 atomic, 1.0 H_k): gain about 0.015, unpaired above 10,000 | Smaller: +0.007, 63,858; paired 7,845 |
| c\* about 0.05–0.12 | XR-1 0.04–0.10; RLDX-1 0.10 (all), 0.19 (atomic), below 0 (composite) |
| Proxy above c\*, so continuation favoured, at low-hundreds n | **Partly.** Yes for XR-1 overall. No for RLDX-1: the proxy (0.06–0.08) is lower than expected, below c\*. n at the proxy is in the thousands |

**Reading and recommendation (exploratory).**
- **The pilot changes E3's case qualitatively, not just its size.**
  - With r = p, a timer-triggered reset looked like a 7–9 point success gain.
  - With the measured same-scene rate it is 1.7 points on XR-1, where false alarms eat most
    of it, and 3.4 points on RLDX-1, all on atomic tasks.
  - At the Wilson lower bound (0.17), the XR-1 gain vanishes.
- **Reset against continuation is knife-edge on both policies.** The reset benefit
  (2–5 points at c = 0) is matched by continuation at c\* = 0.05–0.10. The only data proxy
  for c lands at about the same values, 0.06–0.09. Its direction cannot be predicted from
  existing data, and its size is unaffordable unless c turns out far from c\*.
- **c is cheap to measure, and it decides the design.** A continuation pilot is a collection
  run with the time limit raised to 1.25 H_k. Its late successes estimate c directly, with
  no branching needed.
  - About 300 time-limit failures per policy give c ± 0.05, enough to tell whether c sits
    clearly above or below c\*.
  - On the 10 pilot tasks that is about 600 attempts.
- **A sound E3 for Stage 1:**
  1. **Primary (if kept as a test):** P vs no monitor on RLDX-1 at 1.25 H_k, branching at the
     alarm, about 1,000 pairs. That size survives r = 0.17 (970 pairs); 461 suffice at the
     point estimate.
     - The claim is restricted to RLDX-1, and in effect to its atomic tasks.
     - It needs the state save/restore and the scene-identity check of §15.24.
     - Its retry rate is borrowed from XR-1, and the Stage 2 protocol must measure it.
  2. **Reset against continuation:** an estimation study on both policies. Report both
     arms' success rates and their paired difference with CIs, register the continuation
     pilot as its first step, and make no superiority claim in either direction.
  3. **SAFE + P against P, and everything on XR-1:** descriptive.
- **The reduced claim, honestly put.** E3 can show whether a timer-triggered reset helps
  RLDX-1 on its atomic tasks. It can describe, not decide, whether resetting beats simply
  giving the policy more time. It cannot show that a learned monitor improves task success
  over the timer.
- **Limits as stated above.**
  - r is post hoc, from 10 mid-success XR-1 tasks, and assumed constant across retries.
  - Continuation is assumed independent of the retries.
  - The alarm times come from shadow data.

