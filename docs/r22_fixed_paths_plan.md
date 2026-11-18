# R22. Fixed paths everywhere, and one metric

Written 2026-11-18, before any run. Branch `claude/r22-fixed-paths`, cut from R21
(`claude/r21-native-bands`, 0f653cb0). Exploratory. This is the author's decision after sensei's
revision directions v2 and the discussion of draft v4.

## Why

- **Two designs.** The paper evaluates fused monitors in two ways:
  - H1, H2 and Table II use the *adaptive path*, chosen on the same calibration set that sets its
    margin. That voids Proposition 1 for the chosen monitor, and the expected test FPR rises to
    about 0.055.
  - H3 uses a path fixed on separate RLDX-1 validation rollouts, before its test.
- **The decision (author, 2026-11-18):**
  - Every fused monitor uses a path fixed on validation data, with its margin set on the
    calibration successes and one evaluation on the test set. Proposition 1 then covers every
    result.
  - The paper says nothing more about the adaptive path.
- **One metric (author, same day):** every gain is Δℓ̄, the gain in mean lead over the timer held
  to the monitor's test false alarms (today's "matched" gain). The same-target gain is kept only to
  restate H3's registered rule in one clause.
- **What R8 showed (XR-1, seen tasks, ensembles):**
  - Hide-and-Seek: unchanged (0.072 vs 0.071 at the same target, 0.057 vs 0.056 matched).
  - SAFE-MLP: its frozen path is conservative (16 false alarms against P's 21), and its matched gain
    rises to 0.039 (test-only CI [0.009, 0.072]).

## Rule (fixed now)

For each policy, monitor score (each seed, and the seed average for ensembles), side (seen tasks
against P; unseen tasks against E and G) and fold:

1. **Choose:** run `hk_free.fit` on the *validation* rows of that fold's calibration tasks. Its
   choices are unchanged: the path δ = λ, τ_s = u + sλ, the timer alone allowed, u from the
   validation failures' quantiles, and s from `confirm_lead.S_GRID`. For each candidate, the margin
   comes from Proposition 1 on the validation successes; the path with the largest mean lead on the
   validation failures is kept. The timer is the side's comparator (P or E). G is reported against
   E's paths.
2. **Freeze** (u, s).
3. **Certify:** λ̂ by Proposition 1 on the calibration successes, along the frozen path.
4. **Evaluate** once on the test set:
   - FPR;
   - Δℓ̄: the comparator at the smallest margin with at most the monitor's test false alarms
     (`hk_free.matched_timer`), as in Table II;
   - the same-target gain, kept for the record only.

**Validation sets:**
- **XR-1:** the paper's validation set, the August cohorts. August calibration has 3 + 3 rollouts
  per task (114 + 114) and August test 10 + 10 (380 + 380), 988 rollouts in all. R8 checked them
  disjoint from E1 (gate F2).
- **RLDX-1:** the recorded validation cohorts (R7: 100 + 100 test rollouts and 60 calibration
  successes). H3 stays exactly as registered, with its path frozen on E1's 1,600 further RLDX-1
  rollouts; it is reproduced here, not refitted.
- **Unseen folds:** the validation rows of the fold's seen tasks only, scored by that fold's models
  with pooled normalization. The held-out tasks' validation rows are never read.

**Unchanged** (no path): Table I, Fig. 2, the timer G rows, R21's band rows and FIPER.

## Scores needed, and what exists

| Policy, side | Score | Validation scores |
|---|---|---|
| XR-1 seen | SAFE-MLP (3 seeds) | exist: the frozen final models on the August cohorts (R8's inputs, `$XR1_PS_INPUTS`) |
| XR-1 seen | Hide-and-Seek, zero input, first step, shuffle (3 seeds) | exist: Amendment 2's `august_*_predictions.jsonl` (R8's inputs) |
| XR-1 seen | OOD ×4 (3 seeds, per-task) | exist as the OOD per-seed `calibrate` / `evaluate` outputs on `cache_calibration` / `cache_prospective_test` (the August cohorts); to be confirmed in A5 |
| XR-1 seen | SAFE-LSTM (fold all) | **A2** |
| XR-1 unseen | Hide-and-Seek, zero input, first step (folds 0–3) | **A1** |
| XR-1 unseen | SAFE-LSTM (folds 0–3) | **A2** |
| XR-1 unseen | OOD ×4 (folds 0–3) | **A3** |
| XR-1 unseen | SAFE-MLP (folds 0–3) | **A4** |
| RLDX-1 | every score, seen and unseen | exist: R7's recorded cohorts, reproduced by R16's fold directories (gate G1) |

## Stage A: score XR-1's validation set (cluster)

Every job adds the two August cohorts to an existing scoring run. Each one must also reproduce the
stored E1-cohort trajectories of that run; that is how the retrained or reloaded models are shown to
be the same.

- **A1 (CPU, several hours per fold):** `score_cross_task` for folds 0–3 with Hide-and-Seek, zero
  input and first step.
  - Cohorts: `--cohort august_calibration $AMEND2_ROOT/august_calibration_ids.json` and
    `--cohort august_test $PSEED_ROOT/august_test_balanced_ids.json`, as R6 C2 did for ActProbe's
    fold all.
  - Gate: `--reference-raw` on both E1 cohorts against `$XT_ROOT/fold_F`.
- **A2 (CPU):** `score_cross_task --detectors safe_lstm` for folds 0–3 and all, with the same
  cohorts. Gate: `--reference-raw` against `$R11_ROOT/safe_lstm/fold_F`.
- **A3 (CPU, minutes):** `run_offline_detector_baselines score-fold` for folds 0–3, with R6's
  bundles (`$R6_ROOT/ood/develop/fold_F/seed{0,1,2}`).
  - Cohorts: the two E1 cohorts and the August ones (`cache_calibration`, `cache_prospective_test`).
  - Gate: the E1-cohort rows equal `$R6_ROOT/ood/fold_F`.
- **A4 (GPU):** `score_seen_checkpoint_external` with the saved fold checkpoints
  (`$RS_ROOT/train/fold_F/indep_seed{0,1,2}`) on the August SAFE exports
  (`$XR1_AUGUST_ROOT`, to be located).
  - `cross_task_safe assemble` is extended to accept the two August cohorts as optional extra
    cohorts.
  - Gate: the E1-cohort rows equal `$RS_ROOT/folds/fold_F`.
- **A5 (CPU, minutes):** confirm, or produce, the per-seed seen-task OOD scores on the August
  cohorts with the seed bundles, gated on reproducing the stored E1-cohort files.

## Stage B: the fixed-path analysis (cluster, CPU)

Code: `scripts/xr1_lead_time/r22_fixed_paths.py`, called with `--r22` by `seen_unseen_tables.py`
(XR-1, with the stage A directories) and `rldx_test_table.py` (RLDX-1, validation = recorded
cohorts).

**Outputs** (everything the paper quotes about fused monitors):
- **Table II:** every constant-threshold row on both policies and both sides (OOD ×4, SAFE-MLP,
  SAFE-LSTM, Hide-and-Seek, zero input, first step, shuffle on seen tasks only). Per seed: FPR, Δℓ̄
  and the chosen path; then mean ± std.
- **Table III (XR-1 rows):** ensembles + P (SAFE-MLP, Hide-and-Seek, Mahalanobis as one model, zero
  input, first step, shuffle).
  - False alarms and Δℓ̄, with Table III's CI (tasks, then failures; 2,000 paired resamples,
    conditional on the fitted thresholds and the matched timer).
  - The same-target gain, for the record.
- **Unseen ensembles:** Hide-and-Seek, SAFE-MLP and SAFE-LSTM against E and G, Δℓ̄ with R19's
  full-pipeline bootstrap.
  - Bootstrap: 1,000 replicates, seed 22; tasks, then calibration and test rollouts within task and
    class.
  - In each replicate the path stays frozen, while λ̂ and the timer are refit and the timer is
    re-matched.
- **H2 nulls:** 100 random-noise and 100 within-task resampled fusions per policy's seen side.
  - Each null is built on the validation rows the same way as on the test rows, gets its own frozen
    path, and is evaluated like the monitors.
  - Reported: Hide-and-Seek's and SAFE-MLP's ranks among them, at Δℓ̄.
- **Per task:** Δℓ̄ for atomic and composite tasks (SAFE-MLP, Hide-and-Seek, OOD), and the
  top-5 share.
- **Budget-allocated timer:** per-task margins allocated on the validation set, certified on the
  calibration successes; Hide-and-Seek + P's Δℓ̄ against it.
- **Fig. 1:** the frozen Hide-and-Seek + P path's τ_s and timer alarm on rollout
  `d5be410e4f86cbc95d554fe7`.

## Gates

- **A0 (stage A):** every rerun reproduces its stored E1-cohort trajectories (bit for bit, or
  ≤ 1e-12 where the stored run was compared that way). The August rows are new.
- **V0:** validation, calibration and test rollouts are disjoint (rollout IDs), on both policies.
- **K0:**
  - With the selection switched back to the calibration set (adaptive), the code reproduces every
    stored adaptive row: R14 (XR-1) and R17 (RLDX-1), Table II and III, ≤ 1e-12.
  - This shows that only the selection data change.
- **K8:** on XR-1's seen tasks, the fixed-path SAFE-MLP and Hide-and-Seek ensembles reproduce R8's
  chosen paths and same-target gains at R8's precision.
- **K3:** the RLDX-1 H3 row reproduces E1b's registered result with its frozen path (19 false
  alarms, 0.054, 0.113).
- **N0:** every certified λ̂ equals Proposition 1 on the calibration successes along the frozen path.
- **The scripts' own gates,** and the frozen inputs are unchanged.

## Expectations (written before running)

- **Hide-and-Seek:** about as before on both policies (R8: unchanged on XR-1).
- **SAFE-MLP:** its seen-task Δℓ̄ on XR-1 rises toward R8's 0.039, because its frozen path is
  conservative. Elsewhere, small changes.
- **FPRs:** test FPRs near or below 0.05, without the adaptive inflation to about 0.055.
- **Unseen ensembles:** Hide-and-Seek's gain over E stays positive. Its CIs may widen, since the
  path no longer adapts in each replicate.

## Reporting (fixed now)

- **Source:** every fused number in the paper comes from R22.
- **Method text:** the adaptive path is removed. Sec. IV-C says that each path is chosen on the
  validation set and frozen, and that its margin comes from Proposition 1 on the calibration
  successes.
- **Metric:** every gain is Δℓ̄ (matched), defined once. H3's same-target condition is restated in
  one clause.
- **Completeness:** reported whatever it shows, including SAFE-MLP's conservative-path effect,
  which the text explains.

## Runbook

Stage A first; stage B after its gates hold. (Commands to be added with the code.)

## Result

(to be written after the runs)
