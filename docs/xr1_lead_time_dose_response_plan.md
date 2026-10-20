# Exploratory follow-up to E1: does the fused gain track length-matched signal?

**Status: exploratory.**
- Both cohorts used here are already open for lead time (report §15.13).
- Nothing in this analysis is confirmatory.
- Its only formal role is to decide whether to preregister a fresh atomic-task test
  (Amendment 2), and to size that test.
- This plan was committed before the analysis was run.

## Questions

1. **Dose-response (D1).** The policy score's signal that a timer cannot have is its
   ability to separate failures from successes *among rollouts still running at the same
   inference*. Does a task's length-matched AUC on the calibration cohort predict that
   task's lead gain of SAFE + P over P on the test cohort?
2. **Other detectors (D2).** Does each frozen success-only baseline (cosine kNN,
   Mahalanobis, PCA-KMeans, RND), fused with P and calibrated exactly like SAFE + P, give
   a similar gain? Overall and by task type.
3. **Detector × stratum (D3).** Across detectors and the atomic and composite strata,
   does a higher length-matched AUC go with a higher fused gain?
4. **Atomic stratum (D4).** What is the gain of SAFE + P (and of kNN + P) on atomic tasks
   when P and the fused path are calibrated on atomic calibration rollouts only, with the
   decision bootstrap? This is the design Amendment 2 would register.

## Definitions

- **Cohorts.**
  - The new calibration cohort and the confirmatory test cohort of §15.13 (primary
    10/10 per task, 38 tasks).
  - Baseline trajectories come from the frozen models in
    `xr1_offline_baselines_20261017_100820` (`score-baselines`, no refit).
  - The test cohort's baseline trajectories must equal those in
    `confirm/event_predictions.jsonl`.
- **Fused detectors.**
  - P at q = 0.95. Each detector d + P is fitted with `confirm_lead.fit_fast`, which is
    identical to the registered `quantile_timer.fit`.
  - Pooled fits use all 38 calibration tasks.
  - Stratum fits use only that stratum's calibration rollouts.
- **Length-matched AUC (LM-AUC).**
  - For a task and landmark f in {0.10, 0.25, 0.50, 0.75}, let k = max(1, ceil(f·H_k)).
  - Among rollouts with length ≥ k, compute the ROC-AUC of the running-max score at
    inference k for failure (ties count one half). This is the definition of the earlier
    at-risk analysis.
  - A task's LM-AUC is the mean over landmarks with both classes present. Stratum and
    overall values are task-macro means.
  - The time-risk score has LM-AUC exactly 0.5 by construction.
- **Gains.**
  - The per-task gain is the mean over that task's test failures of
    lead(d + P) − lead(P).
  - Overall and stratum gains use the test-only task-then-rollout bootstrap (2,000
    replicates, seed 0).
  - D4 also uses the decision bootstrap: tasks drawn once per replicate for both
    cohorts, refitted per replicate, seed 0.
- **D1 statistic.**
  - Spearman's rho between calibration LM-AUC and test gain over the 38 tasks.
  - Two-sided permutation p-value with 10,000 permutations, seed 0.
  - Task-bootstrap 95% CI with 2,000 replicates, seed 0.
  - Computed for every detector; SAFE is the headline.

## Expectations written before running

- D1: rho > 0 for SAFE. With 10 failures per task the per-task gain is noisy, so a modest
  rho (about 0.2–0.4) with a CI that touches 0 would be unsurprising.
- D2: on recall, fused baselines matched fused SAFE (§15.10–15.11), so lead gains within
  about 0.02 of SAFE's are expected.
- D3: atomic strata have higher LM-AUC and higher gains for every detector.

## Decision guide for Amendment 2

Preregister a fresh atomic-task test **only if** the D4 stratum-calibrated SAFE + P gain on
atomic tasks has a decision-bootstrap CI lower bound above 0 **and** a point estimate of at
least 0.05. D1–D3 are supporting evidence and do not gate the decision.

If the decision is to proceed:
- Primary: SAFE + P against P, on atomic tasks only.
- P and the fused path calibrated on fresh atomic calibration rollouts.
- Margin 0.05, with the same FPR limits as Amendment 1.
- kNN + P (label-free) as the declared secondary, if its D4 gain is within 0.02 of SAFE's.
- Sample size: the smallest per-class count per task, from {10, 20, 30, 40}, that gives
  at least 80% approximate power. Power is computed at a *shrunk* assumed gain, the
  midpoint of 0.05 and the D4 point estimate, to allow for selecting the best-looking
  subgroup. The decision-bootstrap SE is scaled by sqrt(10/n).

If the gate fails, the atomic result is reported only as a descriptive subgroup finding.

## Runbook

Use the exports of
[`xr1_lead_time_e1_implementation_note.md`](xr1_lead_time_e1_implementation_note.md) §2,
with `E1_SHA` set to the commit that contains this plan. The steps below add the
baseline feature cache for the new calibration cohort, attach the frozen baseline
trajectories to both cohorts, and run the analysis once.

```bash
export XR1_SAFE_CONFIG=$XR1_CAL_ANALYSIS/runtime/seed0/config.yaml
cd $E1_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nohup nice -n 10 $SAFE_PY -u -m robocasa.recovery.safe.run_offline_detector_baselines extract \
  --export-dir $XR1_LCAL_ROOT/official_safe_all_retained --safe-repo $SAFE_REPO --safe-config $XR1_SAFE_CONFIG \
  --group lead_calibration --output-dir $XR1_LCAL_ROOT/cache_lead_calibration \
  > $XR1_LCAL_LOG/extract_lead_calibration.log 2>&1 &
echo "extract PID $!"
```

When the extract finishes:

```bash
cd $E1_REPO
for cohort in calibration test; do
  if test $cohort = calibration; then cache=$XR1_LCAL_ROOT/cache_lead_calibration; group=lead_calibration
  else cache=$XR1_CONFIRM_ROOT/cache_confirmatory_test; group=confirmatory_test; fi
  CUDA_VISIBLE_DEVICES="" nice -n 10 $SAFE_PY -u -m robocasa.recovery.safe.run_offline_detector_baselines score-baselines \
    --baseline-calibration $XR1_FUSED_CALIBRATION --predictions $E1_OUT/${cohort}_predictions.jsonl \
    --cache $cache --cache-group $group --output $E1_OUT/${cohort}_with_baselines.jsonl | grep -E '"(baselines|rollouts|status)"' -A5
done
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
nice -n 10 $SAFE_PY -I -u scripts/xr1_lead_time/dose_response.py \
  --calibration $E1_OUT/calibration_with_baselines.jsonl --test $E1_OUT/test_with_baselines.jsonl \
  --confirm-predictions $XR1_CONFIRM_ROOT/confirm/event_predictions.jsonl \
  --runtime-bundle $XR1_RUNTIME_BUNDLE --output $E1_OUT/dose_response.json \
  2>&1 | tee $STORAGE_BS/robocasa_logs/eval/xr1_lead_calibration_20261020_001108_dose_response.log
```
