# Amendment 1 to the confirmatory preregistration: warning lead over a per-task timer

**Amends:** [`xr1_confirmatory_fused_test.md`](xr1_confirmatory_fused_test.md) (commit 4e82e5c).
That file is left unchanged on purpose, so the checksum recorded by `preregister` stays valid.

**Date:** 2026-10-17. This amendment was committed **before** any rollout of the confirmatory cohort (base seed 1300007) was opened or scored. Here, opening means computing any outcome, score or alarm on it.

**Analysis code:** [`scripts/xr1_lead_time/lead_time.py`](../scripts/xr1_lead_time/lead_time.py) and [`scripts/xr1_lead_time/per_task_timer.py`](../scripts/xr1_lead_time/per_task_timer.py), as committed with this amendment. They were run on the August cohorts, which are already open. The confirmatory analysis applies the same lead and timer definitions.

**Reason.** On the opened prospective cohort, every failed rollout ran to its task's time limit H_k. A timer that alarms at a fixed fraction c of H_k, with c set on calibration at FPR at most 0.05, then alarms on every failure. Recall can no longer separate a learned monitor from elapsed time, so the primary endpoint moves to warning lead. Nothing about the confirmatory cohort has been seen.

**Unchanged.** The cohort definition (10 successes and 10 failures per task, 38 tasks, first rollouts in seed order), the frozen SAFE checkpoints, the per-task normalization, the time-risk curves, the 5% FPR target, the task-then-rollout bootstrap (2,000 replicates) and the reproducibility and checksum checks.

**Comparators.**
- Primary comparator: the per-task timer. It alarms at the first inference t with t >= c_k·H_k, where c_k = min(1, rho_k + delta). Here rho_k is the longest successful **development** rollout of task k as a fraction of H_k, and delta is one shared margin set on calibration successes for FPR at most 0.05. This was added after the exploratory per-task timer check (`scripts/xr1_lead_time/per_task_timer.py`), which recovered most of SAFE's lead over the single timer.
- Secondary comparator: the global timer. It alarms at t >= c·H_k, with one c for all tasks set on calibration. An alarm at the final inference counts, as in review_controls.py.
- Primary detector: SAFE + per-task timer (OR rule).
- Secondary detectors: SAFE + global timer, SAFE + time risk (the original fused design), time risk + timer, and noise + per-task timer (20 draws).

**Calibration.**
- Enlarge calibration to 10 successes per task (n = 380), drawn from fresh seeds disjoint from the confirmatory cohort.
- Choose the threshold path for SAFE + per-task timer on the development cohort only. Set its single parameter with the split-conformal rule (paper, Proposition 1).
- As a sensitivity analysis, also report the old joint search.

**Endpoints.**
- Primary: mean warning lead, the mean of (H_k − T)/H_k over failures with a miss counted as 0. The test is the paired difference, SAFE + per-task timer minus the per-task timer.
- Secondary:
  - recall and FPR, with Wilson intervals;
  - the fraction of failures alarmed by 0.5 H_k and 0.75 H_k;
  - the lead gain against a timer tuned on the test cohort to the fused detector's test FPR.

**Decision rules.**
- **H1 supported** if all of these hold:
  - the bootstrap 95% CI of the lead gain lies above 0;
  - the point gain is at least 0.05 H_k;
  - the fused test FPR is at most 0.075, with a Wilson 95% upper bound of at most 0.10.
- **H2 supported** if noise + per-task timer and within-task-resampled SAFE + per-task timer each gain less than 0.05 H_k over the per-task timer.
- The bootstrap resamples the calibration and confirmatory cohorts together and re-selects thresholds in each replicate.
- Every other comparison is descriptive.

**The original endpoint becomes secondary.** The original rule (A.2–A.3 of the preregistration) compared recall of fused SAFE + time against time risk: CI above 0, gain of at least 0.15, FPR at most 0.075 and Wilson upper bound at most 0.10. It is still reported with the same rule, as a secondary result.

**Relation to the frozen protocol.** Data collection (A.5), cohort selection and disjointness (A.6), and the frozen SAFE and baseline artifacts (A.4) are unchanged. `confirm` still runs as written and reports the original endpoint. The lead-time endpoints are computed from the same scored trajectories with the code above. The thresholds for the per-task timer and for SAFE + per-task timer come from the enlarged calibration cohort described above, never from the confirmatory cohort.
