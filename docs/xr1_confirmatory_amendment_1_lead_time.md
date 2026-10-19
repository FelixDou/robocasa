# Amendment 1 to the confirmatory preregistration: warning lead over a per-task timer

**Amends:** [`xr1_confirmatory_fused_test.md`](xr1_confirmatory_fused_test.md) (commit 4e82e5c).
That file is left unchanged on purpose, so the checksum recorded by `preregister` stays valid.

**Date:** 2026-10-17. This amendment was committed **before** any rollout of the confirmatory cohort (base seed 1300007) was opened or scored. Here, opening means computing any outcome, score or alarm on it.

**Analysis code:** [`scripts/xr1_lead_time/quantile_timer.py`](../scripts/xr1_lead_time/quantile_timer.py) implements the comparator and the fused detector below. It uses the shared definitions in [`scripts/xr1_lead_time/lead_time.py`](../scripts/xr1_lead_time/lead_time.py). Before this amendment was merged, it was dry-run on the opened August cohorts; the results are in the "Dry run" section.

**Reason.** On the opened prospective cohort, every failed rollout ran to its task's time limit H_k. A timer that alarms at a fixed fraction c of H_k, with c set on calibration at FPR at most 0.05, then alarms on every failure. Recall can no longer separate a learned monitor from elapsed time, so the primary endpoint moves to warning lead. Nothing about the confirmatory cohort has been seen.

**Unchanged.** The cohort definition (10 successes and 10 failures per task, 38 tasks, first rollouts in seed order), the frozen SAFE checkpoints, the per-task normalization, the time-risk curves, the 5% FPR target, the task-then-rollout bootstrap (2,000 replicates) and the reproducibility and checksum checks.

**Comparators.**
- Primary comparator: the per-task quantile timer P. It alarms at the first inference t with t >= c_k·H_k, where c_k = min(1, rho_k + delta). An alarm at the final inference counts.
  - rho_k is the 95th percentile (numpy `quantile`, linear) of task k's **development** success durations, as fractions of H_k. These durations are read from the active counts of the frozen time-risk curves in `runtime_bundle.json`. Each development failure runs to H_k, so each drop in the active count marks a success ending at that inference.
  - delta is one shared margin. It is set by the split-conformal rule (paper, Proposition 1) on the calibration successes, at alpha = 0.05.
  - The 95th percentile was chosen after a review of the opened cohorts, which compared the longest success (q = 1) with lower quantiles. The quantiles 0.75, 0.90 and 1.0 are reported as a secondary sensitivity analysis and do not enter any decision rule.
- Primary detector: SAFE + P (OR rule), calibrated with Proposition 1 along the nested path delta = lam, tau_s = u + s·lam.
  - (u, s) is chosen on calibration **failures** only, to maximize their mean lead at the conformal lam. The grid is u over the 0, 0.05, ..., 1 quantiles of the failures' maximum SAFE scores, s in {0, 1, 2, 5, 10, 20, 50}, plus the timer-only path.
  - Calibration successes never enter the path choice.
- Secondary comparator: the global timer. It alarms at t >= c·H_k, with one c for all tasks set on calibration.
- Secondary detectors: SAFE + global timer, SAFE + time risk (the original fused design), time risk + timer, and noise + P (20 draws).

**Calibration.**
- Enlarge calibration to 10 successes per task (n = 380), drawn from fresh seeds disjoint from the confirmatory cohort.
- Fit P and SAFE + P with `quantile_timer.py` on the balanced calibration cohort (10 successes and 10 failures per task), as defined above.
- As a sensitivity analysis, also report the old joint search.

**Endpoints.**
- Primary: mean warning lead, the mean of (H_k − T)/H_k over failures with a miss counted as 0. The test is the paired difference, SAFE + P minus P.
- Secondary:
  - recall and FPR, with Wilson intervals;
  - the fraction of failures alarmed by 0.5 H_k and 0.75 H_k;
  - the lead gain against a timer tuned on the test cohort to the fused detector's test FPR.

**Decision rules.**
- **H1 supported** if all of these hold:
  - the bootstrap 95% CI of the lead gain lies above 0;
  - the point gain is at least 0.05 H_k;
  - the fused test FPR is at most 0.075, with a Wilson 95% upper bound of at most 0.10.
- **H2 supported** if noise + P and within-task-resampled SAFE + P each gain less than 0.05 H_k over P.
- The bootstrap resamples the calibration and confirmatory cohorts together and re-selects thresholds in each replicate.
- Every other comparison is descriptive.

**The original endpoint becomes secondary.** The original rule (A.2–A.3 of the preregistration) compared recall of fused SAFE + time against time risk: CI above 0, gain of at least 0.15, FPR at most 0.075 and Wilson upper bound at most 0.10. It is still reported with the same rule, as a secondary result.

**Relation to the frozen protocol.** Data collection (A.5), cohort selection and disjointness (A.6), and the frozen SAFE and baseline artifacts (A.4) are unchanged. `confirm` still runs as written and reports the original endpoint. The lead-time endpoints are computed from the same scored trajectories with the code above. The thresholds for P and SAFE + P come from the enlarged calibration cohort described above, never from the confirmatory cohort.

**Dry run on the opened August cohorts.** Command: `python3 -I scripts/xr1_lead_time/quantile_timer.py <review_data>`. It uses the primary calibration cohort (114 successes) and the August test cohort (380/380). The output is in `scripts/xr1_lead_time/quantile_timer.json`.

| Detector | Test FPR | Mean lead | Gain of SAFE + P over P (95% CI) |
|---|---|---|---|
| P, q = 0.95 (primary) | 0.045 | 0.195 | |
| SAFE + P, q = 0.95 | 0.042 | 0.202 | 0.007 (−0.003 to 0.019) |
| P, q = 0.90 | 0.029 | 0.169 | 0.028 (−0.012 to 0.067) |
| P, q = 1.0 (longest success) | 0.034 | 0.125 | 0.057 (−0.003 to 0.119) |
| P, q = 0.75 | 0.021 | 0.132 | 0.064 (0.029 to 0.099) |

For reference, the global timer reaches 0.090 at FPR 0.047, and SAFE + global timer reaches 0.207 at 0.058.

**Not yet in the code.** `quantile_timer.py` computes the test-only bootstrap (task then rollout, 2,000 replicates, seed 0). The decision rule above also resamples calibration and re-selects thresholds in each replicate; that bootstrap must be added and committed before the confirmatory cohort is scored. For the confirmatory run, the data directory holds the confirmatory `event_predictions.jsonl` and the frozen `runtime_bundle.json`, and the balanced calibration file is passed as the second argument.

**Expected outcome.** On the opened data, SAFE adds 0.007 of lead over P, far below the 0.05 margin. We therefore expect H1 to fail. That result would show that, on this benchmark, the policy's hidden state adds no usable warning over task duration.
