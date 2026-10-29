# Review-response analyses on the E1 cohorts: pre-specified plan

**Date:** 2026-10-28. This plan and its code
(`scripts/xr1_lead_time/review_analyses.py`, `tests/test_xr1_review_analyses.py`) are committed
**before** any of the analyses below is run. No lead–FPR curve, split-path fit, re-split or
Hide-and-Seek per-task gain exists yet.

**Status: exploratory, pre-specified.** Both E1 cohorts are open (E1, Amendment 2, dose-response,
point-of-no-return labels, cross-task). The rules below fix how results will be read, but nothing
here can confirm anything or change a registered decision (E1: H1 not supported for SAFE + P;
Amendment 2: H1-HS supported for Hide-and-Seek + P).

## Why

An external review of the paper proposal (2026-10-28) raised four statistical points that the
existing analyses answer only at one operating point:

1. **Matched false alarms.** The +0.030 (SAFE + P) and +0.071 (Hide-and-Seek + P) gains are at
   test FPRs of 0.061 against P's 0.055. Calibrating every method to a nominal 5% does not make
   realized false-alarm rates identical. The reviewer asks for lead–FPR curves, or a defined
   matched comparison, with thresholds fixed without the test outcomes. → **R1, R2**
2. **Variation across tasks.** Point estimates and pooled CIs do not show whether a gain is broad
   or carried by a few tasks. → **R3**
3. **Scope of the conformal guarantee.** Proposition 1 holds for a path fixed in advance, but the
   registered fit chooses the path with margins computed on the calibration successes, and both
   preregistered tests ended at a test FPR of 0.061, above the 0.05 target. Is that reuse, a
   difference between the cohorts, or chance? → **R4**
4. **Fair information budget for the timer.** The cross-task study covered Hide-and-Seek on
   unseen tasks. SAFE was left out because retraining it per fold needs the official SAFE
   trainer. → **R5** (staged; its code is written after a feasibility check)

Time-weighted accuracy, TWA = ½(lead + 1 − FPR) when every failure ends at H_k (Hide-and-Seek's
Eq. 12 with T_i = H_k), is reported next to every lead, so the tables read in the metric the
field already uses.

## What has already been seen on these cohorts

- **E1 (report §15.13), α = 0.05 only.** P, SAFE + P and its decision CI; quantile sensitivity
  q ∈ {0.75, 0.90, 1.0}; the global timer and SAFE + global timer by joint search; P tuned on
  the test cohort to SAFE + P's FPR (SAFE then gains +0.015 [−0.007, 0.041]); atomic and composite
  strata.
- **Amendment 2 (§15.21), α = 0.05 only.** Hide-and-Seek and its first-inference, zero-input and
  shuffle controls fused with P, decision CIs, contrasts. `confirm_hide_and_seek.json` also holds
  P tuned on the test cohort to Hide-and-Seek's FPR, which §15.21 did not tabulate.
- **Dose-response (§15.14), α = 0.05 only.** Success-only baselines fused with P, overall, by task
  type, and **per task** for SAFE and the baselines.
- **Cross-task (plan `xr1_cross_task_unseen_plan.md`).** Hide-and-Seek, zero-input,
  first-inference and cosine kNN trained without each fold's tasks, against G and P_n.
- **Never computed on these cohorts:** any FPR target other than 0.05; a fused detector tightened
  on the test cohort to P's false alarms; Hide-and-Seek's per-task gains; the per-task
  distribution of false alarms; a path chosen without the certifying successes; re-splits of the
  pooled successes; SAFE trained without a task.

## Data and code

- **Cohorts:** E1 lead calibration (380/380, seeds [1400007, 1425007)) and E1 confirmatory test
  (380/380, seeds [1300007, 1325007)), the primary 10 + 10 per task, selected by
  `confirm_lead.primary`.
- **Scores:** SAFE in the E1 prediction files; Hide-and-Seek and its three controls from
  Amendment 2's `scores/`; the frozen success-only baselines (Mahalanobis, cosine kNN,
  PCA-KMeans, RND) from the dose-response files `*_with_baselines.jsonl`.
- **Detectors:** `confirm_lead.py`'s, unchanged: P (q = 0.95), the nested OR path (u, s) chosen
  on calibration failures, Proposition 1's margin on calibration successes. The only change is
  that α is a parameter: `fit_alpha(c, rk, 0.05)` equals `confirm_lead.fit_fast(c, rk)` (unit
  test and run-time check).
- **G (global timer):** P with ρ_k = 0 for every task, as in the cross-task study.

## Analyses

### R1. Lead–FPR curves

- **Grid:** α ∈ {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10, 0.125, 0.15, 0.20}.
- **At each α:** fit P, G, and every score + P (SAFE, Hide-and-Seek, three controls, four
  baselines) and SAFE + G, Hide-and-Seek + G, all on the calibration cohort only. Evaluate once
  on the test cohort: false alarms, FPR (Wilson 95%), recall, mean lead, TWA.
- **Gains at the same α** (thresholds never see the test outcomes): fused lead minus P's lead,
  with a test-only CI (`quantile_timer.boot`, tasks then failures, 2,000).
- **Gains at matched false alarms** (both use the test cohort and are labeled so):
  - *P loosened:* P's margin tuned on the test cohort to the largest lead with at most the fused
    detector's false alarms. This favors the timer. It is `confirm_lead`'s test-tuned P.
  - *Fused tightened:* the fused detector moved along its own path (u, s fixed, λ on a grid) to
    the largest lead with at most P's false alarms.
- **Decision bootstrap** at α ∈ {0.02, 0.05, 0.10} for SAFE, Hide-and-Seek and the three controls,
  with the contrasts Hide-and-Seek − SAFE and Hide-and-Seek − each control. The design is
  `confirm_hide_and_seek.boot_multi`'s: tasks drawn once for both cohorts, rollouts within task
  and class, every detector refitted, 2,000 replicates, seed 0. At α = 0.05 its CIs must equal
  Amendment 2's (integrity check I3).
- **Noise band:** 20 draws of running-max Gaussian noise fused with P at every α (seed 1).
- **Band summary** over α ∈ {0.02, …, 0.10} (7 points): each failure's gain averaged over the
  band, at the same α and at matched false alarms, with test-only CIs.

**Reading of each detector, fixed now:**

| Reading | Rule |
|---|---|
| earlier at every budget | matched gain > 0 at every α in the band, and the band-average matched CI lies above 0 |
| later than the timer at matched false alarms | band-average matched CI lies below 0 |
| no earlier warning shown | band-average CIs include 0, both at the same α and at matched false alarms |
| budget-dependent | anything else |

### R2. The α = 0.05 table at matched false alarms

R1's α = 0.05 row for every detector, in one table: false alarms, FPR (Wilson), mean lead, TWA,
gain over P with the decision CI, gain over P loosened, gain when tightened to P, and SAFE + G and
Hide-and-Seek + G. Nothing new is fitted. This is the table the paper will cite for "the same FPR".

### R3. Per-task gains

At the registered α = 0.05 fits, for every detector:
- **Per task:** gain over P (the mean over the task's 10 failures), failures with an earlier or a
  later alarm than P, and false alarms of P and of the fused detector (of 10 successes).
- **Summary:**
  - tasks with a positive, zero or negative gain, and an exact two-sided sign test (zeros excluded);
  - median and IQR of the task gains;
  - the top 5 tasks and their share of the summed gain;
  - leave-one-task-out pooled gain, minimum and maximum, with the thresholds held fixed;
  - between-task SD of the task gains, against the RMS within-task standard error;
  - means by task type;
  - false alarms per task (tasks with 0, tasks with ≥ 2, maximum). This shows how marginal the
    FPR guarantee is.
- The Spearman correlation of SAFE's and Hide-and-Seek's task gains (do they help on the same
  tasks?).

**Reading, fixed now:**

| Reading | Rule |
|---|---|
| broad | positive on at least half of the tasks with a nonzero gain, sign test p < 0.05, and not concentrated |
| concentrated | the top 5 tasks carry more than half of the summed gain, or dropping one task halves the pooled gain |
| mixed | anything else with a positive pooled gain |
| no pooled gain | pooled gain ≤ 0 |

### R4. Calibration reuse and the test FPR

**R4a. Path chosen without the certifying successes.**
- Split each task's 10 calibration successes in `confirm_lead.primary` order (seed, reset,
  rollout ID): the first 5 go to selection and the last 5 to certification.
- Choose (u, s) exactly as the registered fit does, using the calibration failures and the 190
  selection successes. Then set the margins of P and of the chosen path from the 190
  certification successes only. Proposition 1 then holds for the chosen path (rank
  ⌈191 × 0.95⌉ = 182 of 190).
- **Report:** test FPR, lead, gain over P from the same half, and gain over the registered P. Do
  this for SAFE, Hide-and-Seek and the three controls.
- **Variability:** repeat with 20 random within-task halvings (seed 3), reporting the median and
  range.
- **Reading for Hide-and-Seek:** "does not depend on reusing calibration successes" if the
  deterministic split's gain over P from the same half is ≥ 0.05 and its test FPR ≤ 0.075.
  Otherwise "sensitive to reusing calibration successes". Either way the registered decision
  stands, and the paper states the result.

**R4b. Random re-splits of the pooled successes.**
- Pool each task's 20 successes from both cohorts. 200 times (seed 4), split them at random into
  10 calibration and 10 test successes per task, keeping each cohort's failures where they are.
- On each re-split, fit three detectors and record their test FPR:
  - P;
  - the registered fused fit;
  - the R4a split-path fit (halves drawn at random within the calibration successes).
- The calibration and test successes are exchangeable by construction. So for P and for the
  split path, the mean test FPR over re-splits should be at most 0.05 up to Monte Carlo error.
  This is a check of the method.
- **Two effects, both in FPR units:**
  - selection effect = mean FPR of the registered fit − mean FPR of the split path;
  - cohort effect = the actual test FPR (0.061) − mean FPR of the registered fit over re-splits.
- The percentile of the actual FPR among the re-splits is also reported.
- **Reading, fixed now:**
  - "neither raises the test FPR by 0.005 or more" if both effects are < 0.005;
  - otherwise "mainly path selection on calibration successes" or "mainly a difference between
    the calibration and test cohorts", whichever effect is larger.

### Expectations, written before running

- **R1.**
  - P's lead rises with α.
  - SAFE: "no earlier warning shown" or "budget-dependent". Its matched gain at 0.05 is already
    known to be +0.015 [−0.007, 0.041].
  - Hide-and-Seek: "earlier at every budget".
  - The three controls and the noise band: "no earlier warning shown".
  - The baselines: like SAFE.
- **R2.** Hide-and-Seek uses 23 false alarms against P's 21, so its matched gain should stay close
  to +0.071, roughly +0.06.
- **R3.**
  - SAFE: "concentrated", since its gain is mostly atomic (+0.081 against +0.012 composite).
  - Hide-and-Seek: no prior; "mixed" is the guess.
  - Most tasks should have 0 or 1 false alarm under P.
- **R4a.** The certification half is half the size, so margins are noisier. Hide-and-Seek's gain
  should stay positive, and whether it stays ≥ 0.05 is open.
- **R4b.** P alone, with no path selection, already ended at 0.055 on the actual split. So a
  cohort effect of 0.005–0.01 is expected, and a selection effect of a similar size.

## Integrity checks (no result without them)

- **I1 (fit).** `fit_alpha(·, 0.05)` equals `confirm_lead.fit_fast` for every detector.
- **I2 (references at α = 0.05).**
  - **E1:** P's mean lead and false alarms, and SAFE + P's mean lead, false alarms and gain,
    must equal `confirm_lead.json`.
  - **Amendment 2:** P, and each of Hide-and-Seek and its three controls (mean lead, false alarms
    and gain), must equal `confirm_hide_and_seek.json`.
  - **Dose-response:** each baseline's gain and false alarms must equal `dose_response.json`.
- **I3 (decision CIs).** With 2,000 replicates, the α = 0.05 CIs of SAFE, Hide-and-Seek and the
  three controls must equal Amendment 2's `gain_decision_ci95`.
- **I4 (inputs).** Checksums of the frozen inputs are unchanged after the run.
- **Also:**
  - normalization: the Hide-and-Seek trajectories equal their normalized raw scores;
  - the baseline files agree with the E1 rows (labels and SAFE trajectories);
  - every test failure ends at H_k, which is what makes TWA = ½(lead + 1 − FPR).

If I1–I3 fail, the analysis writes only the checks and exits with status 1.

## R5. SAFE on unseen tasks (staged)

**Question.** Does SAFE-MLP trained without a task add warning over G on that task, as
Hide-and-Seek does (+0.127 [0.069, 0.211])? How does it compare with Hide-and-Seek on the same
folds, and with P_n for n ∈ {1, 3, 5, all}?

**Why staged.** Three things are not yet known:
- The official SAFE trainer runs on a GPU and may not reproduce the frozen final refit bit for bit.
- `score_seen_checkpoint_external` refuses tasks outside the model's training tasks. It also needs
  a per-task `task_min_step`, which a held-out task does not have.
- A training run's length is unknown.

Stage A answers these questions without training on any fold. The fold code is written after it,
under the rules below, which are fixed now.

**Stage A (feasibility; no fold trained, no lead computed).**
- **A1.** Where SAFE's data and model code read `task_min_step`. Does it change the score
  trajectory, or only the training targets and metrics?
- **A2.** The frozen final refit's recorded settings:
  - export path, selection summary, model configuration, class weighting and loss mode;
  - the split manifest's fields and sizes;
  - the training time per fit;
  - whether the E1 runtime checkpoints (`$XR1_CAL_ANALYSIS/runtime/seed*`) equal the final refit's.
- **A3** (written after A2, one GPU). Retrain seed 0 on the original 1,824 training rollouts with
  those settings, then compare its scores with the frozen `indep_seed0/scores.jsonl`.

**Stage B design, fixed now.**
- **Folds:** `cross_task_folds.json`, unchanged. Fold f trains on the frozen 1,824 training IDs
  minus fold f's tasks (48 per held-out task, as for Hide-and-Seek). It uses the frozen MLP
  configuration (horizon 1.0, diffusion 1.0, lr 1e-4, regularization 1e-3) and seeds 0–2.
- **Scoring:** every E1 rollout, held-out tasks included, through an opt-in scorer path for
  tasks outside training.
  - If A1 shows that `task_min_step` does not change the trajectory, a held-out task's cutoff is
    its rollout length.
  - If A1 shows that it does change the trajectory, stop and re-plan.
- **Normalization:** pooled over the fold's seen-task training scores, the cross-task formula.
- **Analysis:** `cross_task_lead.py`'s rules U1–U7 with SAFE as the monitor, plus
  SAFE − Hide-and-Seek on the same folds.
- **Gates:**
  - **(B-G1)** The frozen checkpoints, rescored by the Stage B scorer, reproduce E1's
    `frozen_mlp_scores` exactly.
  - **(B-G2)** If A3 is bit-exact, the retrained fold all must equal the frozen trajectories.
    Otherwise both of these must hold, and fold all, retrained, becomes the seen reference:
    - fold all's SAFE + P gain (per-task normalization) lies within ±0.010 of +0.030;
    - the Spearman correlation of per-rollout maximum scores with the frozen ones is ≥ 0.98.
  - **(B-G3)** The per-task normalization of fold all equals the runtime bundle's
    `task_normalizations`.
- **Rules:**
  - **U1-S:** SAFE + G over G, unseen arm. Supported if the decision CI lies above 0, the point
    gain is ≥ 0.05, the FPR is ≤ 0.075 and the Wilson upper bound is ≤ 0.10. These are U1's
    criteria.
  - **U5-S:** SAFE + P_n over P_n for n ∈ {1, 3, 5, all}.
  - **U-HS:** (Hide-and-Seek + G) − (SAFE + G), unseen arm, decision CI.
  - The matched-false-alarm version against G is reported with every G comparison.
- **Expectation (written with the design):**
  - U1-S is likely to pass against G, because G is weak and its margin is conservative.
  - Against G at matched false alarms, and against P_all, SAFE's gain should be near its
    seen-task +0.030 or below.
  - The informative number is U-HS.

### Stage B implementation

Added 2026-10-28, after Stage A and before any fold is trained. It changes no rule above. It
fixes the code and the checks that follow from Stage A: A1 means the cutoff does not change a
score, and A3 means retraining is bit-exact.

**Code.**
- **`robocasa/recovery/safe/score_seen_checkpoint_external.py`** gains two opt-in options.
  - `--unseen-tasks score` scores tasks outside the checkpoint's training tasks, with their
    rollout length as the cutoff, and marks each row `training_task`.
  - `--cutoff-rule rollout_length` applies that cutoff to every task. It is used only for check
    B-G1b.
  - The defaults, and the files they write, are unchanged.
- **`robocasa/recovery/safe/cross_task_safe.py`** has three commands:
  - `manifests`: one selection manifest per fold. Each is `seen38_split_seed0.json` without the
    fold's held-out tasks, after checking that the frozen refit's train and test IDs equal the
    manifest's.
  - `assemble`: fold directories in the cross-task format, with detector `safe`.
  - `compare`: score-file equality.
- **`scripts/xr1_lead_time/cross_task_safe.py`**: the analysis. It reuses `cross_task_lead.py`'s
  fold design, fits, few-shot timers and bootstrap draw sequence, unchanged.

**Models.**
- Folds 0–3 × seeds 0–2 are trained with the frozen command of A3 and the fold manifest:
  12 runs, `train_seen_tasks` unchanged.
- Fold all is the frozen final refit. A3 showed that retraining reproduces it exactly.

**Checks, all required before the analysis reports anything:**
- **B-G1a.** Fold all's scores of both E1 cohorts, through the new scorer path
  (`--unseen-tasks score`), equal the E1 rows' `per_seed_raw_safe` exactly (`assemble`, fold all).
- **B-G1b.** Fold all seed 0, scored with `--cutoff-rule rollout_length`, equals its scores with
  training cutoffs (`compare` prints `IDENTICAL`). This is A1 checked empirically.
- **B-G3.** Fold all's per-task normalization equals the runtime bundle's `task_normalizations`
  (`assemble`, fold all).
- **S1.** Every fold's assembled training tasks are exactly its seen tasks. Each score file comes
  from that fold's checkpoint (SHA-256), and its unseen tasks are the fold's held-out tasks.
- **S2.** SAFE fold all with per-task normalization, fused with P, equals E1: P 0.2252,
  21 FP; SAFE + P 0.2550, 23 FP; gain +0.0298.
- **S3.** The Hide-and-Seek fold directories of the cross-task run pass that run's own checks.
  Hide-and-Seek + G's zero-shot gain (+0.127) and its decision CI ([0.069, 0.211], 2,000
  replicates) are reproduced exactly. That shows SAFE and Hide-and-Seek share the cross-task
  run's replicates.

**Nulls for U1-S** (20 draws, seed 1), as in the cross-task study:
- Gaussian noise;
- within-task resampled fold SAFE, pooled-normalized.

If either mean gain over G is ≥ 0.05, U1-S is reported as not interpretable.

## R6. Seen and unseen columns for Tables I and II (written 2026-10-28, before any run)

**Why.** Hide-and-Seek reports every detector on seen and unseen tasks, side by side. The paper's
Tables I and II report seen tasks only, and the unseen results so far cover SAFE and the
Hide-and-Seek family against G. R6 adds an Unseen column group for every row that can have one.

**Protocol (Hide-and-Seek's, with our folds).**
- **Unseen:** the four cross-task folds (`cross_task_folds.json`). Each task is held out once,
  and monitors are trained without it. Thresholds are calibrated on the seen tasks'
  lead-calibration rollouts and applied to the held-out tasks' test rollouts. Outcomes are pooled
  over the folds. Hide-and-Seek instead uses one random 7/3 split per seed; ours reuses the trained
  folds and gives every task an unseen score.
- **Seen:** all 38 tasks, as Tables I and II are now: per-task models and per-task normalization.
  This is stronger than Hide-and-Seek's seen setting, which has no per-task normalization; the
  captions say so.
- **Cohorts:** the held-out E1 cohorts for both column groups (calibration 380/380, test 380/380).
  Table I therefore moves from the validation set to the held-out set. Its seen numbers change.
- **α = 0.05** throughout (Hide-and-Seek averages 0.15–0.25).
- **Statistics:** mean and standard deviation (ddof 1) over training seeds 0–2; timers are
  deterministic. No confidence intervals.

**Table I (standalone).** One threshold per detector and seed: the largest calibration lead with
calibration FPR ≤ 0.05 (`per_seed_detection.threshold`), applied once. Columns: bACC and TWA for
Seen and Unseen (wACC equals bACC on balanced sets and is dropped; the caption says so). TPR, FPR
and DT are kept in the JSON.

**Table II (fused).** Seen: score + P, the published per-seed rows (unchanged). Unseen: score + G
per fold (`cross_task_lead.unseen`: G and the fused path fitted on seen-task calibration rollouts,
pooled normalization), pooled. Columns: FPR, gain over the timer and TWA for each group.

**Rows and score sources.**

| Row | Seen | Unseen |
|---|---|---|
| Mahalanobis, cosine k-NN, PCA-KMeans, RND | Section 15.10 tuned models, per seed (`xr1_ood_per_seed_20261027_212729`) | the same configuration refit per fold on seen tasks, every task-specific reference pooled (`run_offline_detector_baselines develop --fold`, `score-fold`) |
| SAFE-MLP | E1 `per_seed_raw_safe`, bundle normalization | Stage B fold models |
| ActProbe | fold all of a new cross-task run (`score_cross_task --detectors actprobe`) | its folds 0–3 |
| Hide-and-Seek, zero-input, first-inference | Amendment 2 scores | cross-task folds (`xr1_cross_task_20261027_212141`) |
| same-task shuffle | Amendment 2 scores | — (needs the task's own rollouts) |
| global timer | t/H, same threshold rule | the same, fitted per fold |
| time risk, per-task timer P | as now | — (per-task by construction) |

**Pooled variant of the tuned OOD baselines.** Mahalanobis (one Gaussian per task), PCA-KMeans
(32 clusters per task) and k-NN (per-task reference) cannot score a task without its own
successes. Their unseen version keeps every tuned setting but pools the reference over the fold's
seen tasks (one Gaussian, one set of clusters, one reference); RND is unchanged. The partitions
are the frozen fit/validation partitions restricted to the seen tasks. The pooled normalization
uses `fit_safe_normalization`'s formula over the seen tasks' meta-validation trajectories, the
role the per-task normalization has in Section 15.10. The configuration was selected with all
tasks, which can only favor the baselines (as for SAFE in Stage B).

**Gates (no table without them).**
- **T1.** Every fold directory matches the fold file (fold, held-out tasks, training on exactly
  the seen tasks, fold-file SHA-256, seeds 0–2), and every selected rollout is present with the
  right task, label and length.
- **T2.** Seen + P per seed equals the stored runs to 1e-12: run C of
  `xr1_per_seed_detection_20261027_152130` (SAFE, Hide-and-Seek and its three controls) and
  `ood_per_seed_fused_heldout.json` (the four OOD rows).
- **T3.** Seed-averaged unseen + G equals the cross-task results at their stored precision:
  Hide-and-Seek +0.1273, zero-input +0.0331, first-inference 0.0000 (`cross_task_lead.json`) and
  SAFE +0.0166 (`cross_task_safe.json`).
- **T4.** ActProbe fold all reproduces B1's raw August trajectories exactly
  (`--reference-raw`), and its per-task normalization equals B1's.
- **T5.** Every OOD fold directory is the pooled variant of the frozen Section 15.10 bundle
  (its SHA-256).

**Expectations (written before running).**
- Unseen bACC and TWA fall for every learned detector; the global timer barely moves.
- Hide-and-Seek keeps the best unseen TWA among learned detectors, as its unseen + G gain
  (+0.127) suggests.
- The pooled OOD variants lose most of their seen-task advantage, since their per-task references
  carried most of it.
- SAFE's unseen + G gain stays near noise (+0.017 seed-averaged), with single models at or below it.

**Code.** `scripts/xr1_lead_time/seen_unseen_tables.py`; tests
`tests/test_xr1_seen_unseen_tables.py` and `tests/test_safe_offline_detector_folds.py`. The
per-seed scripts and docs of `claude/time-aware-fusion-paper` (`per_seed_detection.py`,
`ood_per_seed_fused.py`, `run_offline_detector_baselines.py`) are copied unchanged into this
branch, then extended with the fold mode.

## R7. RLDX-1 unseen columns for Table II (written 2026-10-29, before any run)

**Why.** After R6, Table II's RLDX-1 block has seen rows only. R7 fills its Unseen (+ G)
columns, as R6 did for XR-1. Table I stays XR-1 only (author's choice, 2026-10-29). Like R5–R6,
this is exploratory on opened cohorts.

**Data and protocol.** These are the RLDX-1 cohorts of report §15.15/§15.22, unchanged:
- 20 tasks; 740 training rollouts; 60 held-out calibration successes (3 per task); 200 test
  rollouts (5 + 5 per task).
- H_k is the longest training failure of task k.
- Everything is fitted with `confirm_lead.fit_fast`. The calibration successes set the FPR,
  and the training failures choose the fused path. On RLDX-1 the training failures are scored
  in-sample.

**Folds.** `rldx_unseen.py folds` deals the 20 tasks into 4 folds:
- the rule is `cross_task_lead.make_folds`, stratified by the bundle's task types, seed
  20261029;
- the file is written in the run folder before anything is trained, and its SHA-256 is
  recorded by every fold directory;
- a fold with no held-out task is refused.

**Arms.**
- **Seen (+ P).** Fold all, per-task normalization, every task. These are the published
  per-seed rows (`rldx1_per_seed_20261028_145711`), recomputed here and gated.
- **Unseen (+ G).** Per fold, G (P with ρ_k = 0) and the fused path are fitted on the seen
  tasks' calibration successes and their training failures. The scores are the fold's models
  with pooled normalization. They are applied to the held-out tasks' test rollouts and pooled
  over folds. Every task is held out once, so the pooled set is all 200 test rollouts.
- **Statistics:** mean and std (ddof 1) over seeds 0–2, plus the seed average.

**Rows.**

| Row | Seen | Unseen |
|---|---|---|
| Hide-and-Seek, zero-input, first-inference | §15.22 scores | `score_cross_task --score-train` per fold |
| Mahalanobis, cosine k-NN, PCA-KMeans, RND (default settings) | `rldx1_per_seed` OOD scores | the same per fold |
| same-task shuffle | §15.22 | — (needs the task's own rollouts) |
| SAFE-LSTM, SAFE-MLP | §15.15 | Stage B′, below |
| G | as published | per fold, fitted on the seen tasks |

The default OOD detectors fit one model over all training successes, with no per-task
reference. Unlike XR-1's tuned ones, they need no pooled variant.

**Code.**
- `score_cross_task --score-train` is new and off by default. It writes the fold's
  (seen-task) training trajectories, as `score_cohorts --score-train` does.
- `scripts/xr1_lead_time/rldx_unseen.py` (`folds`, `run`), with tests in
  `tests/test_xr1_rldx_unseen.py`.
- `rldx_per_seed.py` and its test are copied unchanged from `claude/time-aware-fusion-paper`.

**Gates (no table without them).**
- **R1.** Every fold directory matches the fold file (folds, held-out tasks, training on exactly
  the seen tasks, seeds 0–2, `--score-train`). Fold all reproduces the earlier `score_cohorts`
  trajectories of all seven detectors in all three cohorts (`--reference-raw`; 21 checks).
- **R2.** Seen + P equals `rldx_per_seed.json` exactly (P, lead, FPR, gain), per seed and
  seed-averaged, for every detector.
- **R3.** G is the same timer whichever score it is fused with.

**Expectations (written before running).**
- G is weak on RLDX-1: seen lead 0.032, against P's 0.271. So every score can gain a lot over
  G, controls included. The controls' gains, not zero, are the yardstick.
- Hide-and-Seek gains the most and stays above the zero-input and first-inference controls, as
  on XR-1. The margin should be smaller: its seen gain on RLDX-1 comes from atomic tasks, and
  each fold holds out only a few of them.
- The OOD rows gain little more than the controls. Their unseen FPR may exceed 0.05, as on
  XR-1.
- About 45 seen calibration successes per fold make the 5% threshold coarse (two false alarms
  among them), and one false alarm on the 100 test successes moves the FPR by 0.01.

**SAFE (staged, as R5).**
- **Stage A′** (file reads, then one GPU). Each answer decides how Stage B′ is built:
  - **A′1.** Where are the SAFE20 final models, their raw scores and settings? Do
    `normalized_scores.jsonl`'s `full_raw_scores` equal the final models' scores?
  - **A′2.** What are the training command, selection summary, time per fit, split manifest
    (740 / 260) and official export?
  - **A′3** (written after A′2). Retrain LSTM seed 0 and MLP seed 0 on the 740 training
    rollouts, and compare with the frozen raw trajectories.
- **Stage B′** design, fixed now:
  - **Folds:** R7's.
  - **Training:** both heads with the frozen settings and seeds 0–2, without the fold's tasks
    (24 fits).
  - **Scoring:** every RLDX-1 rollout, through the unseen-task scorer path.
  - **Assembly:** fold directories in the cross-task format (calibration, test and train
    predictions; pooled normalization by the cross-task formula).
  - **Seen rows:** these keep SAFE's own normalization, since they are the published rows.
  - **Gate:** fold all must equal the frozen raw trajectories if A′3 is bit-exact. Otherwise
    the gate follows R5's B-G2.

**Stage B′ implementation** (added 2026-10-29, after A′3 was found bit-exact for both heads,
before any fold is trained). It changes no rule above.
- **Code.**
  - `score_seen_checkpoint_external --group all` is a new label for a whole export with
    several splits. Nothing else changes.
  - `cross_task_safe assemble-export` builds one SAFE head's fold directory in the cross-task
    format, with detector `safe_lstm` or `safe_indep`:
    - the whole-export scores are split by the RLDX-1 ID lists into calibration, test and
      seen-task training cohorts;
    - normalization is pooled by the cross-task formula over the seen tasks' training
      rollouts, and per task by SAFE's own formula (`fit_task_normalization`).
  - `rldx_unseen.py run` takes one `--fold-dirs` per family: the cross-task detectors, then
    one family per SAFE head.
  - Tests are in `tests/test_xr1_rldx_unseen.py`.
- **Models.**
  - The manifests come from `cross_task_safe manifests` on `frozen_outer_split.json`
    unchanged: each fold drops its held-out tasks' training and test rollouts.
  - Folds 0–3 × seeds 0–2 × both heads are trained with the D4 command and the fold manifest
    (24 fits).
  - Fold all is the frozen SAFE20 final models.
- **Scoring:** every model (24 fold models and the 6 frozen ones) scores the whole export
  (1,000 rollouts) with `--group all --unseen-tasks score`.
- **Checks, all required before the table:**
  - **B′-G1** (`assemble-export`, fold all): the scores of all 1,000 rollouts equal the frozen
    `full_raw_scores`, and SAFE's per-task normalization recomputed from the final runs equals
    the frozen one.
  - **B′-G2** (`assemble-export`, every fold): the external scores of each run's training
    rollouts equal the run's own `scores.jsonl`.
  - **B′-G3** (cutoff): fold all seed 0 of each head, scored with
    `--cutoff-rule rollout_length`, equals its training-cutoff scores (`compare` prints
    `IDENTICAL`). XR-1 checked the MLP only.
  - **R1–R3** with the three families. R2 now also requires SAFE-LSTM and SAFE-MLP + P to equal
    `rldx_per_seed.json`.
- **Expectation (written now):**
  - SAFE-MLP gains little over G; its seen gain over P is +0.008.
  - SAFE-LSTM gains less than Hide-and-Seek (+0.196). Whether it beats the OOD rows (+0.10)
    is open.

## Runbook

Run on TSUBAME from a pinned worktree, CPU only for R1–R4, following
[`cluster_experiment_runbook.md`](cluster_experiment_runbook.md). Never pull in the shared
checkout. `RV_SHA` is the full SHA of the pushed commit on branch `claude/review-analyses` that
contains this file.

### 0. Environment, pinned code, inputs, unit tests

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
# Amendment 2 paths: AMEND2_ROOT, E1_OUT, XR1_LCAL_ROOT, XR1_CONFIRM_ROOT, XR1_RUNTIME_BUNDLE, ...
source $STORAGE_BS/robocasa_logs/eval/xr1_amend2_hide_and_seek_20261022_214209/paths.env

export RV_SHA=<RV_SHA>
export RV_REPO=$PROJECT_FS/robocasa_review_${RV_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/review-analyses
test -e $RV_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $RV_REPO $RV_SHA
test "$(git -C $RV_REPO rev-parse HEAD)" = "$RV_SHA" || echo "STOP: worktree HEAD is not RV_SHA"
cd $RV_REPO

export RV_TAG=xr1_review_analyses_$(date +%Y%m%d_%H%M%S)
export RV_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$RV_TAG
export RV_LOG=$STORAGE_BS/robocasa_logs/eval/$RV_TAG
mkdir -p $RV_ROOT $RV_LOG
RV_INPUTS=($E1_OUT/calibration_predictions.jsonl $E1_OUT/test_predictions.jsonl $E1_OUT/confirm_lead.json
           $E1_OUT/calibration_with_baselines.jsonl $E1_OUT/test_with_baselines.jsonl $E1_OUT/dose_response.json
           $AMEND2_ROOT/confirm_hide_and_seek.json $AMEND2_ROOT/scores/normalizations.json
           $AMEND2_ROOT/scores/lead_calibration_predictions.jsonl
           $AMEND2_ROOT/scores/confirmatory_test_predictions.jsonl $XR1_RUNTIME_BUNDLE)
for f in "${RV_INPUTS[@]}"; do test -s $f && echo "OK $f" || echo "STOP: missing $f"; done
sha256sum "${RV_INPUTS[@]}" | tee $RV_LOG/frozen_inputs.sha256
printf 'export %s=%q\n' RV_SHA "$RV_SHA" RV_REPO "$RV_REPO" RV_ROOT "$RV_ROOT" RV_LOG "$RV_LOG" > $RV_LOG/paths.env

PYTHONPATH=$RV_REPO "$SAFE_PY" -m unittest tests.test_xr1_review_analyses tests.test_xr1_confirm_hide_and_seek \
  tests.test_xr1_confirm_lead tests.test_xr1_dose_response 2>&1 | tail -n 3
```

Every input must print `OK`, and the unit tests must end with `OK`.

### 1. R1–R4 (one process, CPU, about an hour)

```bash
cd $RV_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup nice -n 10 \
  "$SAFE_PY" -I -u scripts/xr1_lead_time/review_analyses.py run \
  --calibration $E1_OUT/calibration_predictions.jsonl --test $E1_OUT/test_predictions.jsonl \
  --scores-calibration $AMEND2_ROOT/scores/lead_calibration_predictions.jsonl \
  --scores-test $AMEND2_ROOT/scores/confirmatory_test_predictions.jsonl \
  --normalizations $AMEND2_ROOT/scores/normalizations.json --runtime-bundle $XR1_RUNTIME_BUNDLE \
  --e1-result $E1_OUT/confirm_lead.json --amend2-result $AMEND2_ROOT/confirm_hide_and_seek.json \
  --baselines-calibration $E1_OUT/calibration_with_baselines.jsonl \
  --baselines-test $E1_OUT/test_with_baselines.jsonl --dose-response-result $E1_OUT/dose_response.json \
  --output $RV_ROOT/review_analyses.json > $RV_LOG/review_analyses.log 2>&1 &
echo "analysis PID $! on $(hostname)"
```

The eleven `alpha ...` lines appear within minutes. The decision bootstrap takes most of the
hour. When the process has exited:

```bash
tail -n 80 $RV_LOG/review_analyses.log
sha256sum -c --quiet $RV_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
ls -la $RV_ROOT/review_analyses.json
```

Report the `alpha` lines, the JSON block, the `R4a`/`R4b` line, `REVIEW ANALYSES OK` and
`INPUTS UNCHANGED`. `REPRODUCTION FAILED` is a stop condition: do not edit inputs to make it pass.
The JSON is then copied to the local evidence folder:

```bash
# local machine
rsync -av tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/<RV_TAG>/review_analyses.json \
  reports/safe_robocasa365_research_plan_20260823/evidence/<RV_TAG>/
```

### 2. R5 Stage A (SAFE unseen feasibility; no fold trained)

A1 and A2 read files only. They can run in the same terminal after step 0.

```bash
source $RV_LOG/paths.env; cd $RV_REPO
export SAFE_REPO=$PROJECT_FS/SAFE
export XR1_FINAL_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/xr1_pooled_39tasks_final_refits_20260814_231508
export XR1_CAL_ANALYSIS=$STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_calibration

# A1: where SAFE reads task_min_step
grep -rn "task_min_step" $SAFE_REPO/failure_prob | tee $RV_LOG/a1_task_min_step.txt

# A2: the frozen final refit's settings, and whether E1's runtime checkpoints are the same weights
ls $XR1_FINAL_ROOT $XR1_FINAL_ROOT/indep_seed0 | tee $RV_LOG/a2_listing.txt
sha256sum $XR1_FINAL_ROOT/indep_seed{0,1,2}/model_final.ckpt $XR1_CAL_ANALYSIS/runtime/seed{0,1,2}/model_final.ckpt \
  | tee $RV_LOG/a2_checkpoints.sha256
diff $XR1_FINAL_ROOT/indep_seed0/config.yaml $XR1_CAL_ANALYSIS/runtime/seed0/config.yaml && echo "SAME CONFIG seed0"
"$SAFE_PY" -I - <<'PY' 2>&1 | tee $RV_LOG/a2_settings.txt
import json, os
root = os.environ["XR1_FINAL_ROOT"] + "/indep_seed0"
m = json.load(open(root + "/metrics.json"))
print("metrics keys:", sorted(m))
for k in sorted(m):
    if any(w in k.lower() for w in ("arg", "command", "selection", "export", "class_weight", "loss", "epoch",
                                     "device", "seconds", "elapsed", "online", "split", "seed")):
        print(k, "=", json.dumps(m[k], default=str)[:600])
s = json.load(open(root + "/split_manifest.json"))
print("split keys:", sorted(s), {k: len(v) for k, v in s.items() if isinstance(v, (list, dict))})
with open(root + "/scores.jsonl") as f:
    r = json.loads(f.readline())
print("scores.jsonl fields:", sorted(r))
PY
grep -n -E "data_path|name:|seed|lr|epochs|horizon|diffusion|weight_decay|lambda|batch" \
  $XR1_FINAL_ROOT/indep_seed0/config.yaml | head -n 40 | tee -a $RV_LOG/a2_settings.txt
ls $STORAGE_BS/robocasa_logs/eval | grep -i -E "pooled_39tasks|final_refit" | tee -a $RV_LOG/a2_settings.txt
```

Report the three files: `a1_task_min_step.txt`, `a2_checkpoints.sha256` and `a2_settings.txt`.
The A3 retraining command and the Stage B code follow from them, before any fold is trained.

### 3. R5 Stage B (SAFE per fold; two GPUs, then CPU)

The steps take roughly 1–2 h of training, 1–2 h of scoring and 1 h of analysis. The first load of
each export from `/gs/bs` takes 40–70 minutes, and later loads on the same node are much faster.
Use one allocation with two GPUs and enough time left. A background job dies when its allocation
ends. `RS_SHA` is the full SHA of the pushed commit on `claude/review-analyses` that contains this
step.

**B0. Environment, pinned code, inputs, unit tests.**

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python SAFE_REPO=$PROJECT_FS/SAFE
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
source $STORAGE_BS/robocasa_logs/eval/xr1_amend2_hide_and_seek_20261022_214209/paths.env
export XR1_FINAL_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/xr1_pooled_39tasks_final_refits_20260814_231508
export XR1_POOL=$STORAGE_BS/robocasa_rollouts/safe/xr1_safe_pooled_39tasks_50each_seed0_20260812_221815
export XR1_SAFE_EXPORT=$XR1_POOL/official_safe_39tasks_50each_seed0 XR1_SEEN38_SPLIT=$XR1_POOL/seen38_split_seed0.json
export XR1_SAFE_CV_SUMMARY=$STORAGE_BS/robocasa_checkpoints/safe/xr1_safe_pooled_39tasks_50each_seed0_20260812_221815_full_cv/cv_selection_summary.json
export XT_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/xr1_cross_task_20261027_212141

export RS_SHA=<RS_SHA>
export RS_REPO=$PROJECT_FS/robocasa_xtsafe_${RS_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/review-analyses
test -e $RS_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $RS_REPO $RS_SHA
test "$(git -C $RS_REPO rev-parse HEAD)" = "$RS_SHA" && echo "HEAD OK" || echo "STOP: worktree HEAD is not RS_SHA"
cd $RS_REPO
export RS_TAG=xr1_cross_task_safe_$(date +%Y%m%d_%H%M%S)
export RS_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$RS_TAG RS_LOG=$STORAGE_BS/robocasa_logs/eval/$RS_TAG
mkdir -p $RS_ROOT $RS_LOG
export XR1_LCAL_EXPORT=$("$SAFE_PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['export_dir'])" \
  $XR1_LCAL_ROOT/frozen_mlp_scores/seed0/provenance.json)
export XR1_TEST_EXPORT=$("$SAFE_PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['export_dir'])" \
  $XR1_CONFIRM_ROOT/confirmatory_collection/frozen_mlp_scores/seed0/provenance.json)
echo "lead calibration export: $XR1_LCAL_EXPORT"; echo "confirmatory test export: $XR1_TEST_EXPORT"
RS_INPUTS=($E1_OUT/calibration_predictions.jsonl $E1_OUT/test_predictions.jsonl $E1_OUT/confirm_lead.json
           $XR1_RUNTIME_BUNDLE $XR1_SEEN38_SPLIT $XR1_SAFE_CV_SUMMARY $XR1_FINAL_ROOT/indep_seed{0,1,2}/scores.jsonl
           $XR1_FINAL_ROOT/indep_seed{0,1,2}/model_final.ckpt $XT_ROOT/cross_task_lead.json
           $RS_REPO/scripts/xr1_lead_time/cross_task_folds.json)
for f in "${RS_INPUTS[@]}"; do test -s $f && echo "OK $f" || echo "STOP: missing $f"; done
for F in 0 1 2 3 all; do test -s $XT_ROOT/fold_$F/scoring.json && echo "OK H&S fold $F" || echo "STOP: no H&S fold $F"; done
sha256sum "${RS_INPUTS[@]}" | tee $RS_LOG/frozen_inputs.sha256
printf 'export %s=%q\n' RS_SHA "$RS_SHA" RS_REPO "$RS_REPO" RS_ROOT "$RS_ROOT" RS_LOG "$RS_LOG" \
  XR1_LCAL_EXPORT "$XR1_LCAL_EXPORT" XR1_TEST_EXPORT "$XR1_TEST_EXPORT" > $RS_LOG/paths.env
PYTHONPATH=$RS_REPO "$SAFE_PY" -m unittest tests.test_xr1_cross_task_safe tests.test_xr1_cross_task_lead \
  tests.test_safe_score_cross_task 2>&1 | tail -n 3
nvidia-smi --query-gpu=index,memory.used --format=csv
```

**B1. Fold manifests.** It must print `MANIFESTS OK` and `[48]` dropped per held-out task in
every fold.

```bash
cd $RS_REPO
PYTHONPATH=$RS_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.cross_task_safe manifests \
  --selection-manifest $XR1_SEEN38_SPLIT --folds $RS_REPO/scripts/xr1_lead_time/cross_task_folds.json \
  --frozen-scores $XR1_FINAL_ROOT/indep_seed0/scores.jsonl --output-dir $RS_ROOT/manifests | tee $RS_LOG/manifests.log
```

**B2. Train 12 models.** GPU 0 trains folds 0 and 1, GPU 1 folds 2 and 3, three seeds each, one
after another.

```bash
cd $RS_REPO
for G in 0 1; do
  if [ $G = 0 ]; then FS="0 1"; else FS="2 3"; fi
  cat > $RS_LOG/train_gpu$G.sh <<EOF
#!/bin/bash
cd $RS_REPO
for F in $FS; do for S in 0 1 2; do
  echo "=== START fold \$F seed \$S \$(date -Iseconds) on \$(hostname) GPU $G ==="
  CUDA_VISIBLE_DEVICES=$G WANDB_MODE=disabled "$SAFE_PY" -u -m robocasa.recovery.safe.train_seen_tasks \\
    --export-dir $XR1_SAFE_EXPORT --safe-repo $SAFE_REPO \\
    --output-dir $RS_ROOT/train/fold_\$F/indep_seed\$S \\
    --model indep --seed \$S --split-seed 0 \\
    --selection-manifest $RS_ROOT/manifests/fold_\$F.json --selection-summary $XR1_SAFE_CV_SUMMARY \\
    --class-weighting official_inverse_frequency --epochs 1000 --device cuda
  echo "=== END fold \$F seed \$S rc=\$? \$(date -Iseconds) ==="
done; done
EOF
done
cat $RS_LOG/train_gpu0.sh
for G in 0 1; do nohup bash $RS_LOG/train_gpu$G.sh > $RS_LOG/train_gpu$G.log 2>&1 & echo "GPU $G PID $! on $(hostname)"; done
```

Check progress with `grep -h "=== " $RS_LOG/train_gpu*.log; tail -c 200 $RS_LOG/train_gpu0.log; echo`.
Training is done when both logs have six `END ... rc=0` lines. Any other `rc` is a stop condition.

**B3. Score both E1 cohorts with every fold model** (fold all uses the frozen refit). GPU 0
scores the lead-calibration cohort and GPU 1 the confirmatory test cohort.

```bash
cd $RS_REPO
for G in 0 1; do
  if [ $G = 0 ]; then C=lead_calibration; X=$XR1_LCAL_EXPORT; else C=confirmatory_test; X=$XR1_TEST_EXPORT; fi
  cat > $RS_LOG/score_gpu$G.sh <<EOF
#!/bin/bash
cd $RS_REPO
for F in all 0 1 2 3; do for S in 0 1 2; do
  if [ \$F = all ]; then RUN=$XR1_FINAL_ROOT/indep_seed\$S; else RUN=$RS_ROOT/train/fold_\$F/indep_seed\$S; fi
  echo "=== START $C fold \$F seed \$S \$(date -Iseconds) on \$(hostname) GPU $G ==="
  CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$RS_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.score_seen_checkpoint_external \\
    --export-dir $X --safe-repo $SAFE_REPO --training-run-dir \$RUN \\
    --output-dir $RS_ROOT/score/fold_\$F/$C/seed\$S --group $C --device cuda --unseen-tasks score
  echo "=== END $C fold \$F seed \$S rc=\$? \$(date -Iseconds) ==="
done; done
EOF
done
cat $RS_LOG/score_gpu1.sh
for G in 0 1; do nohup bash $RS_LOG/score_gpu$G.sh > $RS_LOG/score_gpu$G.log 2>&1 & echo "GPU $G PID $! on $(hostname)"; done
```

Scoring is done when both logs have 15 `END ... rc=0` lines.

**B4. Cutoff check (B-G1b).** It must print `IDENTICAL`.

```bash
cd $RS_REPO
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$RS_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.score_seen_checkpoint_external \
  --export-dir $XR1_TEST_EXPORT --safe-repo $SAFE_REPO --training-run-dir $XR1_FINAL_ROOT/indep_seed0 \
  --output-dir $RS_ROOT/score/cutoff_check/confirmatory_test/seed0 --group confirmatory_test --device cuda \
  --unseen-tasks score --cutoff-rule rollout_length > $RS_LOG/cutoff_check.log 2>&1
PYTHONPATH=$RS_REPO "$SAFE_PY" -m robocasa.recovery.safe.cross_task_safe compare \
  $RS_ROOT/score/fold_all/confirmatory_test/seed0 $RS_ROOT/score/cutoff_check/confirmatory_test/seed0 \
  | tee $RS_LOG/cutoff_compare.log
```

**B5. Assemble the five fold directories (B-G1a, B-G3, S1).** Every fold must print
`ASSEMBLY OK`. Fold all must also print two `reference ... -> OK` lines and `bundle normalization
... -> OK`.

```bash
cd $RS_REPO
for F in all 0 1 2 3; do
  A=()
  for S in 0 1 2; do
    if [ $F = all ]; then RUN=$XR1_FINAL_ROOT/indep_seed$S; else RUN=$RS_ROOT/train/fold_$F/indep_seed$S; fi
    A+=(--training-run $S $RUN --scores lead_calibration $S $RS_ROOT/score/fold_$F/lead_calibration/seed$S
        --scores confirmatory_test $S $RS_ROOT/score/fold_$F/confirmatory_test/seed$S)
  done
  if [ $F = all ]; then
    A+=(--reference-predictions lead_calibration $E1_OUT/calibration_predictions.jsonl
        --reference-predictions confirmatory_test $E1_OUT/test_predictions.jsonl)
  else
    A+=(--fold-manifest $RS_ROOT/manifests/fold_$F.json)
  fi
  PYTHONPATH=$RS_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.cross_task_safe assemble --fold $F \
    --folds $RS_REPO/scripts/xr1_lead_time/cross_task_folds.json --runtime-bundle $XR1_RUNTIME_BUNDLE \
    "${A[@]}" --output-dir $RS_ROOT/folds/fold_$F 2>&1 | tail -n 5
done | tee $RS_LOG/assemble.log
```

**B6. The analysis (CPU, about an hour).**

```bash
cd $RS_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup nice -n 10 \
  "$SAFE_PY" -I -u scripts/xr1_lead_time/cross_task_safe.py run \
  --calibration $E1_OUT/calibration_predictions.jsonl --test $E1_OUT/test_predictions.jsonl \
  --safe-fold-dirs $RS_ROOT/folds/fold_0 $RS_ROOT/folds/fold_1 $RS_ROOT/folds/fold_2 $RS_ROOT/folds/fold_3 $RS_ROOT/folds/fold_all \
  --hs-fold-dirs $XT_ROOT/fold_0 $XT_ROOT/fold_1 $XT_ROOT/fold_2 $XT_ROOT/fold_3 $XT_ROOT/fold_all \
  --runtime-bundle $XR1_RUNTIME_BUNDLE --e1-result $E1_OUT/confirm_lead.json \
  --cross-task-result $XT_ROOT/cross_task_lead.json \
  --output $RS_ROOT/cross_task_safe.json > $RS_LOG/analysis.log 2>&1 &
echo "analysis PID $! on $(hostname)"
```

When it ends:

```bash
tail -n 70 $RS_LOG/analysis.log
sha256sum -c --quiet $RS_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
```

Report:
- B1's fold lines;
- the `END` lines of B2 and B3 (`grep -h "=== END" $RS_LOG/*_gpu*.log`);
- B4's `IDENTICAL`;
- B5's log;
- the analysis tail, ending in `CROSS-TASK SAFE OK`;
- `INPUTS UNCHANGED`.

### 4. R6: seen and unseen columns (CPU, one large allocation)

ActProbe peaks at about 95 GB per run (B1), so use a node with at least about 300 GB and run at
most three ActProbe folds at a time, e.g. `qrsh -g tga-shinoda -l node_h=1 -l h_rt=12:00:00`.
Everything below is CPU only. `R6_SHA` is the full SHA of the pushed commit on
`claude/review-analyses` that contains this step.

**C0. Environment, pinned code, inputs, unit tests.**

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
source $STORAGE_BS/robocasa_logs/eval/xr1_amend2_hide_and_seek_20261022_214209/paths.env
export CK=$STORAGE_BS/robocasa_checkpoints/safe
export XT_ROOT=$CK/xr1_cross_task_20261027_212141 RS_ROOT=$CK/xr1_cross_task_safe_20261028_193716
export PSEED_ROOT=$CK/xr1_per_seed_detection_20261027_152130 OOD_ROOT=$CK/xr1_ood_per_seed_20261027_212729
export XR1_BASELINE_ROOT=$CK/xr1_offline_baselines_20261017_100820
export R6_LCAL_CACHE=$STORAGE_BS/robocasa_rollouts/safe/xr1_lead_calibration_20261020_001108/cache_lead_calibration
export R6_TEST_CACHE=$STORAGE_BS/robocasa_rollouts/safe/xr1_confirmatory_fused_20261017_160220/cache_confirmatory_test

export R6_SHA=<R6_SHA>
export R6_REPO=$PROJECT_FS/robocasa_seenunseen_${R6_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/review-analyses
test -e $R6_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R6_REPO $R6_SHA
test "$(git -C $R6_REPO rev-parse HEAD)" = "$R6_SHA" && echo "HEAD OK" || echo "STOP: worktree HEAD is not R6_SHA"
cd $R6_REPO
export R6_TAG=xr1_seen_unseen_$(date +%Y%m%d_%H%M%S)
export R6_ROOT=$CK/$R6_TAG R6_LOG=$STORAGE_BS/robocasa_logs/eval/$R6_TAG
mkdir -p $R6_ROOT $R6_LOG
export FOLDS=$R6_REPO/scripts/xr1_lead_time/cross_task_folds.json
printf 'export %s=%q\n' R6_SHA "$R6_SHA" R6_REPO "$R6_REPO" R6_TAG "$R6_TAG" R6_ROOT "$R6_ROOT" R6_LOG "$R6_LOG" \
  > $R6_LOG/paths.env

for f in $E1_OUT/calibration_predictions.jsonl $E1_OUT/test_predictions.jsonl $XR1_RUNTIME_BUNDLE $XR1_TRAIN_IDS \
         $PSEED_ROOT/august_test_balanced_ids.json $AMEND2_ROOT/august_calibration_ids.json \
         $AMEND2_ROOT/scores/lead_calibration_predictions.jsonl $AMEND2_ROOT/scores/confirmatory_test_predictions.jsonl \
         $AMEND2_ROOT/scores/normalizations.json $XT_ROOT/cross_task_lead.json $RS_ROOT/cross_task_safe.json \
         $PSEED_ROOT/per_seed_fused_heldout.json $PSEED_ROOT/actprobe_scores/normalizations.json \
         $PSEED_ROOT/actprobe_scores/august_calibration_predictions.jsonl \
         $PSEED_ROOT/actprobe_scores/august_test_predictions.jsonl \
         $OOD_ROOT/ood_per_seed_fused_heldout.json $XR1_BASELINE_ROOT/develop/baseline_bundle.json \
         $XR1_BASELINE_ROOT/cache_development/provenance.json \
         $R6_LCAL_CACHE/provenance.json $R6_TEST_CACHE/provenance.json \
         $OOD_ROOT/seed{0,1,2}/heldout/{calibration,test}_with_baselines.jsonl \
         $XT_ROOT/fold_{0,1,2,3}/scoring.json $RS_ROOT/folds/fold_{0,1,2,3}/scoring.json; do
  test -s $f && echo "OK $f" || echo "STOP: missing $f"
done > $R6_LOG/inputs_check.log
grep -v '^OK' $R6_LOG/inputs_check.log; grep -c '^OK' $R6_LOG/inputs_check.log
echo "XR1_RAW_DIRS: ${#XR1_RAW_DIRS[@]} directories"
sha256sum $E1_OUT/calibration_predictions.jsonl $E1_OUT/test_predictions.jsonl $XR1_RUNTIME_BUNDLE \
  $AMEND2_ROOT/scores/*_predictions.jsonl $AMEND2_ROOT/scores/normalizations.json \
  $XT_ROOT/cross_task_lead.json $RS_ROOT/cross_task_safe.json $PSEED_ROOT/per_seed_fused_heldout.json \
  $PSEED_ROOT/actprobe_scores/*.json* $OOD_ROOT/ood_per_seed_fused_heldout.json \
  $OOD_ROOT/seed{0,1,2}/heldout/*_with_baselines.jsonl $XR1_BASELINE_ROOT/develop/baseline_bundle.json \
  | tee $R6_LOG/frozen_inputs.sha256 | wc -l
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -m unittest tests.test_safe_offline_detector_folds \
  tests.test_safe_offline_detector_baselines tests.test_xr1_seen_unseen_tables \
  tests.test_xr1_cross_task_safe tests.test_xr1_cross_task_lead 2>&1 | tail -n 3
free -g | head -n 2; nproc
```

No `STOP` line may print, the count must be 34, `XR1_RAW_DIRS` must be nonempty, and the tests
must end with `OK`.

**C1. OOD pooled refits per fold (12 `develop` runs, then 4 `score-fold` runs).** One script per
fold, all four in the background.

```bash
cd $R6_REPO
for F in 0 1 2 3; do
  cat > $R6_LOG/ood_fold$F.sh <<EOF
set -e
cd $R6_REPO
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for S in 0 1 2; do
  "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines develop \\
    --development-cache $XR1_BASELINE_ROOT/cache_development --runtime-bundle $XR1_RUNTIME_BUNDLE \\
    --fixed-selection $XR1_BASELINE_ROOT/develop/baseline_bundle.json --model-seed \$S \\
    --baselines mahalanobis knn_cosine pca_kmeans rnd --folds $FOLDS --fold $F \\
    --output-dir $R6_ROOT/ood/develop/fold_$F/seed\$S > /dev/null
  echo "fold $F seed \$S developed"
done
"$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines score-fold \\
  --bundle 0=$R6_ROOT/ood/develop/fold_$F/seed0/baseline_bundle.json \\
  --bundle 1=$R6_ROOT/ood/develop/fold_$F/seed1/baseline_bundle.json \\
  --bundle 2=$R6_ROOT/ood/develop/fold_$F/seed2/baseline_bundle.json --folds $FOLDS \\
  --cohort lead_calibration $E1_OUT/calibration_predictions.jsonl $R6_LCAL_CACHE lead_calibration \\
  --cohort confirmatory_test $E1_OUT/test_predictions.jsonl $R6_TEST_CACHE confirmatory_test \\
  --output-dir $R6_ROOT/ood/fold_$F
echo "=== OOD FOLD $F DONE"
EOF
  nohup nice -n 10 bash $R6_LOG/ood_fold$F.sh > $R6_LOG/ood_fold$F.log 2>&1 &
  echo "OOD fold $F PID $! on $(hostname)"
done
```

Progress: `tail -qn 1 $R6_LOG/ood_fold?.log`. Each log must end with `=== OOD FOLD F DONE`; an
`error:` or `Traceback` line is a stop condition.

**C2. ActProbe per fold (5 runs, about 95 GB and 1–2 hours each).** Fold all also scores the
August cohorts and must reproduce B1 exactly (`REFERENCE OK`). The script runs fold all, 0 and 1
together, then 2 and 3.

```bash
cd $R6_REPO
cat > $R6_LOG/actprobe.sh <<'EOF'
set -u
# Arrays such as XR1_RAW_DIRS are not inherited by a child shell, so read the paths here.
source $STORAGE_BS/robocasa_logs/eval/xr1_amend2_hide_and_seek_20261022_214209/paths.env
cd $R6_REPO
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
COMMON=(--dataset-dir "${XR1_RAW_DIRS[@]}" $XR1_LCAL_ROOT/merged_all_retained $XR1_CONFIRM_ROOT/confirmatory_collection/merged_all_retained
        --runtime-bundle $XR1_RUNTIME_BUNDLE --train-ids $XR1_TRAIN_IDS --folds $FOLDS
        --cohort lead_calibration $E1_OUT/calibration_predictions.jsonl
        --cohort confirmatory_test $E1_OUT/test_predictions.jsonl
        --detectors actprobe --feature-aggregation last --seeds 0 1 2)
one() {
  F=$1; EXTRA=()
  if [ $F = all ]; then
    EXTRA=(--cohort august_calibration $AMEND2_ROOT/august_calibration_ids.json
           --cohort august_test $PSEED_ROOT/august_test_balanced_ids.json
           --reference-raw august_calibration $PSEED_ROOT/actprobe_scores/august_calibration_predictions.jsonl
           --reference-raw august_test $PSEED_ROOT/actprobe_scores/august_test_predictions.jsonl)
  fi
  "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cross_task "${COMMON[@]}" --fold $F "${EXTRA[@]}" \
    --output-dir $R6_ROOT/actprobe/fold_$F > $R6_LOG/actprobe_fold_$F.log 2>&1
  echo "=== ACTPROBE FOLD $F rc=$? $(date -Iseconds)"
}
one all & one 0 & one 1 & wait
one 2 & one 3 & wait
echo "=== ACTPROBE DONE"
EOF
nohup nice -n 10 bash $R6_LOG/actprobe.sh > $R6_LOG/actprobe.log 2>&1 &
echo "ActProbe PID $! on $(hostname)"
```

Progress: `cat $R6_LOG/actprobe.log; tail -qn 2 $R6_LOG/actprobe_fold_*.log`. Done when
`actprobe.log` shows five `rc=0` lines and `=== ACTPROBE DONE`, and `actprobe_fold_all.log` ends
with two `reference ... -> OK` lines and `REFERENCE OK`. Any other `rc` is a stop condition.

**C2b. Rerun of fold all (2026-10-29).**

*What went wrong.* The first run, `xr1_seen_unseen_20261028_235911` at `8ef49963`, passed
`$XR1_EVAL_IDS` as the August test cohort. That is the full prospective-test file, 1,643
rollouts. B1 had scored `$PSEED_ROOT/august_test_balanced_ids.json`, 760 rollouts, sha256
`f8cbd6d3…`. The reference check counts a rollout missing from B1 as `inf`, so fold all
reported `max diff inf -> MISMATCH`, rc=1. T4 then failed, and no tables were written.

*What still matched.* Checked locally on the copied files, B1's 760 rollouts are all in the
new file and match exactly (difference 0.0, same seeds, lengths and labels). August
calibration (228) and the normalization also matched exactly.

*What gets rerun.* Fold all is rerun with B1's balanced list. The first run's folder, including
its `fold_all` and failed `seen_unseen_tables.json`, is kept as evidence. Folds 0–3 and the OOD
folds are reused unchanged.

```bash
cd $R6_REPO
export AUG_TEST_IDS=$PSEED_ROOT/august_test_balanced_ids.json
echo "f8cbd6d306b914163b0ed41dc7614937693b8eb26bc82874f350de2ab819a00c  $AUG_TEST_IDS" | sha256sum -c -
if test -e $R6_ROOT/actprobe/fold_all_b1ids; then echo "STOP: $R6_ROOT/actprobe/fold_all_b1ids exists"; else
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup nice -n 10 \
    "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cross_task \
    --dataset-dir "${XR1_RAW_DIRS[@]}" $XR1_LCAL_ROOT/merged_all_retained $XR1_CONFIRM_ROOT/confirmatory_collection/merged_all_retained \
    --runtime-bundle $XR1_RUNTIME_BUNDLE --train-ids $XR1_TRAIN_IDS --folds $FOLDS \
    --cohort lead_calibration $E1_OUT/calibration_predictions.jsonl \
    --cohort confirmatory_test $E1_OUT/test_predictions.jsonl \
    --cohort august_calibration $AMEND2_ROOT/august_calibration_ids.json --cohort august_test $AUG_TEST_IDS \
    --reference-raw august_calibration $PSEED_ROOT/actprobe_scores/august_calibration_predictions.jsonl \
    --reference-raw august_test $PSEED_ROOT/actprobe_scores/august_test_predictions.jsonl \
    --detectors actprobe --feature-aggregation last --seeds 0 1 2 --fold all \
    --output-dir $R6_ROOT/actprobe/fold_all_b1ids > $R6_LOG/actprobe_fold_all_b1ids.log 2>&1 &
  echo "ActProbe fold all PID $! on $(hostname)"
fi
```

The log must end with two `reference ... -> OK` lines (228 and 760 rollouts) and `REFERENCE OK`.
C3 then reads `fold_all_b1ids` in place of `fold_all` and writes `seen_unseen_tables_b1ids.json`
and `tables_b1ids.log`, since the script refuses to overwrite an existing output.

**C3. The tables (CPU, minutes).** Only after C1 and C2 are done.

```bash
cd $R6_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/seen_unseen_tables.py run \
  --calibration $E1_OUT/calibration_predictions.jsonl --test $E1_OUT/test_predictions.jsonl \
  --runtime-bundle $XR1_RUNTIME_BUNDLE --folds $FOLDS \
  --safe-fold-dirs $RS_ROOT/folds/fold_{0,1,2,3} --xt-fold-dirs $XT_ROOT/fold_{0,1,2,3} \
  --actprobe-fold-dirs $R6_ROOT/actprobe/fold_{0,1,2,3,all} --ood-fold-dirs $R6_ROOT/ood/fold_{0,1,2,3} \
  --ood-source-bundle $XR1_BASELINE_ROOT/develop/baseline_bundle.json \
  --amend2-scores $AMEND2_ROOT/scores/lead_calibration_predictions.jsonl $AMEND2_ROOT/scores/confirmatory_test_predictions.jsonl \
  --amend2-normalizations $AMEND2_ROOT/scores/normalizations.json \
  --ood-seen 0 $OOD_ROOT/seed0/heldout/calibration_with_baselines.jsonl $OOD_ROOT/seed0/heldout/test_with_baselines.jsonl \
  --ood-seen 1 $OOD_ROOT/seed1/heldout/calibration_with_baselines.jsonl $OOD_ROOT/seed1/heldout/test_with_baselines.jsonl \
  --ood-seen 2 $OOD_ROOT/seed2/heldout/calibration_with_baselines.jsonl $OOD_ROOT/seed2/heldout/test_with_baselines.jsonl \
  --run-c $PSEED_ROOT/per_seed_fused_heldout.json --ood-fused $OOD_ROOT/ood_per_seed_fused_heldout.json \
  --cross-task-result $XT_ROOT/cross_task_lead.json --cross-task-safe-result $RS_ROOT/cross_task_safe.json \
  --actprobe-reference-normalizations $PSEED_ROOT/actprobe_scores/normalizations.json \
  --output $R6_ROOT/seen_unseen_tables.json 2>&1 | tee $R6_LOG/tables.log
sha256sum -c --quiet $R6_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
sha256sum $R6_ROOT/seen_unseen_tables.json
```

Report the whole `tables.log` (it ends in `SEEN/UNSEEN TABLES OK` or `REPRODUCTION FAILED`),
`INPUTS UNCHANGED` and the SHA-256. Then copy the JSON to the local evidence folder:

```bash
# local machine
rsync -av tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/<R6_TAG>/seen_unseen_tables.json \
  reports/safe_robocasa365_research_plan_20260823/evidence/<R6_TAG>/
```

### 5. R7: RLDX-1 unseen columns (CPU, one large allocation; Stage A′ file reads)

`R7_SHA` is the full SHA of the pushed commit on `claude/review-analyses` that contains this
step. The RLDX-1 OOD scoring of `rldx1_per_seed` peaked at 23 GB. Five fold processes at once
need about 120 GB and 40 cores, so use a `node_h` allocation.

**D0. Environment, pinned code, inputs, folds, unit tests.**

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
source $STORAGE_BS/robocasa_logs/eval/rldx1_per_seed_20261028_145711/paths.env
echo "RLDXPS_ROOT=$RLDXPS_ROOT RLDXHS_ROOT=$RLDXHS_ROOT RLDX_ROOT=$RLDX_ROOT RLDX_RAW=$RLDX_RAW SAFE_PY=$SAFE_PY"

export R7_SHA=<R7_SHA>
export R7_REPO=$PROJECT_FS/robocasa_rldxunseen_${R7_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/review-analyses
test -e $R7_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R7_REPO $R7_SHA
test "$(git -C $R7_REPO rev-parse HEAD)" = "$R7_SHA" && echo "HEAD OK" || echo "STOP: worktree HEAD is not R7_SHA"
cd $R7_REPO
export R7_TAG=rldx1_unseen_$(date +%Y%m%d_%H%M%S)
export R7_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$R7_TAG R7_LOG=$STORAGE_BS/robocasa_logs/eval/$R7_TAG
mkdir -p $R7_ROOT $R7_LOG
export FOLDS=$R7_ROOT/rldx_folds.json
printf 'export %s=%q\n' R7_SHA "$R7_SHA" R7_REPO "$R7_REPO" R7_TAG "$R7_TAG" R7_ROOT "$R7_ROOT" R7_LOG "$R7_LOG" \
  FOLDS "$FOLDS" RLDXPS_ROOT "$RLDXPS_ROOT" RLDXHS_ROOT "$RLDXHS_ROOT" RLDX_ROOT "$RLDX_ROOT" RLDX_RAW "$RLDX_RAW" \
  SAFE_PY "$SAFE_PY" > $R7_LOG/paths.env
echo "R7_TAG=$R7_TAG"

R7_INPUTS=($RLDXHS_ROOT/inputs/rldx_bundle.json $RLDXHS_ROOT/inputs/{train,calibration,test}_ids.json
           $RLDXHS_ROOT/scores/{train,calibration,test}_predictions.jsonl $RLDXHS_ROOT/scores/normalizations.json
           $RLDXPS_ROOT/ood_scores/{train,calibration,test}_predictions.jsonl $RLDXPS_ROOT/ood_scores/normalizations.json
           $RLDXPS_ROOT/rldx_per_seed.json
           $RLDX_ROOT/calibration/lstm/lstm_seed{0,1,2}/normalized_scores.jsonl
           $RLDX_ROOT/calibration/indep/indep_seed{0,1,2}/normalized_scores.jsonl $RLDX_RAW/manifest.jsonl)
for f in "${R7_INPUTS[@]}"; do test -s $f && echo "OK $f" || echo "STOP: missing $f"; done > $R7_LOG/inputs_check.log
grep -v '^OK' $R7_LOG/inputs_check.log; grep -c '^OK' $R7_LOG/inputs_check.log
sha256sum "${R7_INPUTS[@]}" > $R7_LOG/frozen_inputs.sha256

"$SAFE_PY" -I scripts/xr1_lead_time/rldx_unseen.py folds --inputs-dir $RLDXHS_ROOT/inputs --write $FOLDS \
  | tee $R7_LOG/folds.log
CUDA_VISIBLE_DEVICES="" "$SAFE_PY" -W ignore -m unittest tests.test_safe_score_cross_task tests.test_xr1_rldx_unseen \
  tests.test_rldx_per_seed tests.test_rldx_hide_and_seek tests.test_rldx_lead_replication 2>&1 | tail -n 3
free -g | head -n 2; nproc
```

Continue only if all of these hold:
- `HEAD OK` is printed, no `STOP` line appears, and the count is 20;
- `folds.log` lists four folds whose tasks add up to 20;
- the tests end with `OK` (15 tests).

**D1. Score the five folds (background).**

```bash
cd $R7_REPO
cat > $R7_LOG/score.sh <<'EOF'
set -u
source $R7_LOG/paths.env
cd $R7_REPO
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
COMMON=(--dataset-dir $RLDX_RAW --runtime-bundle $RLDXHS_ROOT/inputs/rldx_bundle.json
        --train-ids $RLDXHS_ROOT/inputs/train_ids.json --folds $FOLDS
        --cohort calibration $RLDXHS_ROOT/inputs/calibration_ids.json --cohort test $RLDXHS_ROOT/inputs/test_ids.json
        --detectors hide_and_seek hide_and_seek_zero hide_and_seek_first mahalanobis cosine_knn pca_kmeans rnd
        --feature-aggregation last --seeds 0 1 2 --score-train)
one() {
  F=$1; EXTRA=()
  if [ $F = all ]; then
    for c in calibration test train; do
      EXTRA+=(--reference-raw $c $RLDXHS_ROOT/scores/${c}_predictions.jsonl
              --reference-raw $c $RLDXPS_ROOT/ood_scores/${c}_predictions.jsonl)
    done
  fi
  "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cross_task "${COMMON[@]}" --fold $F "${EXTRA[@]}" \
    --output-dir $R7_ROOT/folds/fold_$F > $R7_LOG/score_fold_$F.log 2>&1
  echo "=== FOLD $F rc=$? $(date -Iseconds)"
}
one all & one 0 & one 1 & one 2 & one 3 & wait
echo "=== SCORING DONE"
EOF
nohup nice -n 10 bash $R7_LOG/score.sh > $R7_LOG/score.log 2>&1 &
echo "scoring PID $! on $(hostname)"
```

Progress: `cat $R7_LOG/score.log; tail -qn 2 $R7_LOG/score_fold_*.log`. It is done when:
- `score.log` shows five `rc=0` lines and `=== SCORING DONE`;
- `score_fold_all.log` ends with 21 `reference ... -> OK` lines and `REFERENCE OK`.

Any other `rc` is a stop condition.

**D2. Stage A′ (file reads; can run while D1 scores).**

```bash
source $R7_LOG/paths.env; cd $R7_REPO
ls $RLDX_ROOT | tee $R7_LOG/a_listing.txt
find $RLDX_ROOT -maxdepth 3 \( -name model_final.ckpt -o -name scores.jsonl -o -name metrics.json \
  -o -name config.yaml -o -name "*selection*.json" -o -name split_manifest.json \) | sort | tee -a $R7_LOG/a_listing.txt
ls $STORAGE_BS/robocasa_logs/eval | grep -i -E "safe20|supported20" | tee -a $R7_LOG/a_listing.txt
"$SAFE_PY" -I - <<'PY' 2>&1 | tee $R7_LOG/a_settings.txt
import glob, json, os
import numpy as np
root = os.environ["RLDX_ROOT"]
words = ("arg", "command", "selection", "export", "weight", "loss", "epoch", "second", "elapsed", "split",
         "seed", "config", "horizon", "diffusion", "lr", "lambda", "device")
for model in ("lstm", "indep"):
    for path in sorted(glob.glob(f"{root}/**/{model}_seed0/metrics.json", recursive=True)):
        m = json.load(open(path))
        print("==", path, "\nkeys:", sorted(m))
        for k in sorted(m):
            if any(w in k.lower() for w in words):
                print("  ", k, "=", json.dumps(m[k], default=str)[:400])
    norm = f"{root}/calibration/{model}/{model}_seed0/normalized_scores.jsonl"
    rows = [json.loads(line) for line in open(norm) if line.strip()]
    print(model, "normalized rows", len(rows), "fields", sorted(rows[0]))
    finals = sorted(glob.glob(f"{root}/**/{model}_seed0/scores.jsonl", recursive=True))
    print(model, "final scores files:", finals)
    if finals:
        fin = {}
        for line in open(finals[0]):
            if line.strip():
                r = json.loads(line)
                fin[r["rollout_id"]] = r
        diffs = [float(np.max(np.abs(np.asarray(r["full_raw_scores"]) - np.asarray(fin[r["rollout_id"]]["scores"]))))
                 for r in rows if r["rollout_id"] in fin]
        print(model, "rows matched", len(diffs), "of", len(rows), "max |full_raw_scores - final scores|",
              max(diffs) if diffs else None)
        print(model, "final scores fields:", sorted(next(iter(fin.values()))))
s = json.load(open(f"{root}/calibration/lstm/split_manifest.json"))
print("split manifest keys:", sorted(s), {k: len(v) for k, v in s.items() if isinstance(v, (list, dict))})
PY
```

Report `a_listing.txt` and `a_settings.txt`. A′3 and the Stage B′ code are written from them.

**D4. Stage A′3: retrain LSTM seed 0 and MLP seed 0 (two GPUs at once).**

```bash
source $R7_LOG/paths.env; cd $R7_REPO
export SAFE_REPO=$PROJECT_FS/SAFE
export RLDX_EXPORT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_original_safe35_natural50_20260808_203158/official_safe_supported20_natural50
git -C $SAFE_REPO rev-parse HEAD
test -s $RLDX_EXPORT/conversion_report.json && echo "EXPORT OK" || echo "STOP: no export"
nvidia-smi --query-gpu=index,memory.used --format=csv
mkdir -p $R7_ROOT/safe_retrain_check
for M in lstm indep; do
  if [ $M = lstm ]; then G=0; else G=1; fi
  CUDA_VISIBLE_DEVICES=$G WANDB_MODE=disabled nohup "$SAFE_PY" -u -m robocasa.recovery.safe.train_seen_tasks \
    --export-dir $RLDX_EXPORT --safe-repo $SAFE_REPO --output-dir $R7_ROOT/safe_retrain_check/${M}_seed0 \
    --model $M --seed 0 --split-seed 0 \
    --selection-manifest $RLDX_ROOT/protocol/frozen_outer_split.json \
    --selection-summary $RLDX_ROOT/cv/cv_selection_summary.json \
    --class-weighting official_inverse_frequency --epochs 1000 --device cuda \
    > $R7_LOG/a3_retrain_${M}_seed0.log 2>&1 &
  echo "$M PID $! on GPU $G ($(hostname))"
done
```

The SAFE commit must be `b6036abe07b2b2bb9996afb2c07f13d6a9f507c0`, and `EXPORT OK` must print.
Progress: `tail -c 300 $R7_LOG/a3_retrain_*_seed0.log; echo`. Each run ends with a JSON block,
like the original final logs.

**D5. Stage A′3 comparison (after both runs end).**

```bash
source $R7_LOG/paths.env; cd $R7_REPO
for M in lstm indep; do
  echo "== $M"
  sha256sum $RLDX_ROOT/final/${M}_seed0/model_final.ckpt $R7_ROOT/safe_retrain_check/${M}_seed0/model_final.ckpt
  diff $RLDX_ROOT/final/${M}_seed0/config.yaml $R7_ROOT/safe_retrain_check/${M}_seed0/config.yaml && echo "SAME CONFIG $M"
done 2>&1 | tee $R7_LOG/a3_compare.txt
"$SAFE_PY" -I - <<'PY' 2>&1 | tee -a $R7_LOG/a3_compare.txt
import json, os
import numpy as np
root, new = os.environ["RLDX_ROOT"] + "/final", os.environ["R7_ROOT"] + "/safe_retrain_check"
def load(path):
    with open(path) as stream:
        return {r["rollout_id"]: r for r in (json.loads(line) for line in stream if line.strip())}
for m in ("lstm", "indep"):
    a, b = load(f"{root}/{m}_seed0/scores.jsonl"), load(f"{new}/{m}_seed0/scores.jsonl")
    diffs = [float(np.max(np.abs(np.asarray(a[k]["scores"]) - np.asarray(b[k]["scores"]))))
             if len(a[k]["scores"]) == len(b[k]["scores"]) else float("inf") for k in a if k in b]
    ma, mb = (json.load(open(f"{x}/{m}_seed0/metrics.json")) for x in (root, new))
    keys = ("selected_hyperparameters", "task_min_steps", "training_class_weights", "training_objective", "counts")
    print(m, "rollouts", len(a), len(b), "same ids", set(a) == set(b), "max |diff|", max(diffs) if diffs else None,
          "metrics equal", {k: ma.get(k) == mb.get(k) for k in keys})
PY
```

Report `a3_compare.txt`, plus the first and last 5 lines of each retraining log.

**D6. Stage B′ setup: pinned code, frozen SAFE inputs, tests, fold manifests.** Use a node with
two GPUs (`node_h`). `R7B_SHA` is the full SHA of the pushed commit that contains this step. In a
new shell, run the first five lines of D0 (environment and the per-seed `paths.env`) first.

```bash
source "$(ls -d $STORAGE_BS/robocasa_logs/eval/rldx1_unseen_* | sort | tail -n 1)/paths.env"
export SAFE_REPO=$PROJECT_FS/SAFE WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export RLDX_EXPORT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_original_safe35_natural50_20260808_203158/official_safe_supported20_natural50
export R7B_SHA=<R7B_SHA>
export R7B_REPO=$PROJECT_FS/robocasa_rldxsafe_${R7B_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/review-analyses
test -e $R7B_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $R7B_REPO $R7B_SHA
test "$(git -C $R7B_REPO rev-parse HEAD)" = "$R7B_SHA" && echo "HEAD OK" || echo "STOP: worktree HEAD is not R7B_SHA"
cd $R7B_REPO
export R7S=$R7_ROOT/safe
mkdir -p $R7S
printf 'export %s=%q\n' R7B_SHA "$R7B_SHA" R7B_REPO "$R7B_REPO" R7S "$R7S" SAFE_REPO "$SAFE_REPO" \
  RLDX_EXPORT "$RLDX_EXPORT" > $R7_LOG/paths_safe.env
SAFE_INPUTS=($RLDX_ROOT/protocol/frozen_outer_split.json $RLDX_ROOT/cv/cv_selection_summary.json
             $RLDX_ROOT/final/{lstm,indep}_seed{0,1,2}/{model_final.ckpt,scores.jsonl,metrics.json,config.yaml})
for f in "${SAFE_INPUTS[@]}"; do test -s $f && echo "OK $f" || echo "STOP: missing $f"; done | grep -c '^OK'
sha256sum "${SAFE_INPUTS[@]}" > $R7_LOG/frozen_safe.sha256
git -C $SAFE_REPO rev-parse HEAD
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$R7B_REPO "$SAFE_PY" -W ignore -m unittest tests.test_xr1_rldx_unseen \
  tests.test_xr1_cross_task_safe tests.test_safe_score_cross_task 2>&1 | tail -n 3
ls -l --time-style=+%H:%M:%S $R7_ROOT/safe_retrain_check/*_seed0/{config.yaml,metrics.json}
PYTHONPATH=$R7B_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.cross_task_safe manifests \
  --selection-manifest $RLDX_ROOT/protocol/frozen_outer_split.json --folds $FOLDS \
  --frozen-scores $RLDX_ROOT/final/lstm_seed0/scores.jsonl --output-dir $R7S/manifests | tee $R7_LOG/safe_manifests.log
nvidia-smi --query-gpu=index,memory.used --format=csv
```

Continue only if all of these hold:
- `HEAD OK` prints, the count is 26 and the SAFE commit is `b6036abe…`;
- the tests end with `OK`;
- the manifests print `MANIFESTS OK`, with one dropped count per held-out task, the same in
  every fold (37 if every task has 37 training rollouts).

The `ls` line gives A′3's time per fit (`config.yaml` written at the start, `metrics.json` at
the end).

**D7. Train 24 fold models.** GPU 0 trains the LSTM, GPU 1 the MLP. On each GPU, the three seeds
of a fold run at once, and the folds follow one another.

```bash
source $R7_LOG/paths.env; source $R7_LOG/paths_safe.env; cd $R7B_REPO
for G in 0 1; do
  if [ $G = 0 ]; then M=lstm; else M=indep; fi
  cat > $R7_LOG/safe_train_gpu$G.sh <<EOF
#!/bin/bash
cd $R7B_REPO
for F in 0 1 2 3; do
  for S in 0 1 2; do
    ( echo "=== START $M fold \$F seed \$S \$(date -Iseconds) GPU $G ==="
      CUDA_VISIBLE_DEVICES=$G WANDB_MODE=disabled PYTHONPATH=$R7B_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.train_seen_tasks \\
        --export-dir $RLDX_EXPORT --safe-repo $SAFE_REPO --output-dir $R7S/train/fold_\$F/${M}_seed\$S \\
        --model $M --seed \$S --split-seed 0 --selection-manifest $R7S/manifests/fold_\$F.json \\
        --selection-summary $RLDX_ROOT/cv/cv_selection_summary.json \\
        --class-weighting official_inverse_frequency --epochs 1000 --device cuda \\
        > $R7_LOG/safe_train_${M}_fold\${F}_seed\$S.log 2>&1
      echo "=== END $M fold \$F seed \$S rc=\$? \$(date -Iseconds) ===" ) &
  done
  wait
done
EOF
done
for G in 0 1; do nohup bash $R7_LOG/safe_train_gpu$G.sh > $R7_LOG/safe_train_gpu$G.log 2>&1 & echo "GPU $G PID $! on $(hostname)"; done
```

Progress: `grep -h "=== END" $R7_LOG/safe_train_gpu*.log | sort | uniq -c | head; grep -c "rc=0" $R7_LOG/safe_train_gpu*.log`.
Training is done when both logs have 12 `END ... rc=0` lines. Any other `rc` is a stop condition.

**D8. Score the whole export with all 30 models** (fold all uses the frozen models). GPU 0
scores the LSTM, GPU 1 the MLP, three seeds at once.

```bash
source $R7_LOG/paths.env; source $R7_LOG/paths_safe.env; cd $R7B_REPO
for G in 0 1; do
  if [ $G = 0 ]; then M=lstm; else M=indep; fi
  cat > $R7_LOG/safe_score_gpu$G.sh <<EOF
#!/bin/bash
cd $R7B_REPO
for F in all 0 1 2 3; do
  for S in 0 1 2; do
    if [ \$F = all ]; then RUN=$RLDX_ROOT/final/${M}_seed\$S; else RUN=$R7S/train/fold_\$F/${M}_seed\$S; fi
    ( echo "=== START score $M fold \$F seed \$S \$(date -Iseconds) GPU $G ==="
      CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$R7B_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.score_seen_checkpoint_external \\
        --export-dir $RLDX_EXPORT --safe-repo $SAFE_REPO --training-run-dir \$RUN \\
        --output-dir $R7S/score/fold_\$F/${M}_seed\$S --group all --device cuda --unseen-tasks score \\
        > $R7_LOG/safe_score_${M}_fold\${F}_seed\$S.log 2>&1
      echo "=== END score $M fold \$F seed \$S rc=\$? \$(date -Iseconds) ===" ) &
  done
  wait
done
EOF
done
for G in 0 1; do nohup bash $R7_LOG/safe_score_gpu$G.sh > $R7_LOG/safe_score_gpu$G.log 2>&1 & echo "GPU $G PID $! on $(hostname)"; done
```

Scoring is done when both logs have 15 `END ... rc=0` lines
(`grep -c "rc=0" $R7_LOG/safe_score_gpu*.log`).

**D9. Cutoff check (B′-G3).** It must print `IDENTICAL` twice.

```bash
source $R7_LOG/paths.env; source $R7_LOG/paths_safe.env; cd $R7B_REPO
for M in lstm indep; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R7B_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.score_seen_checkpoint_external \
    --export-dir $RLDX_EXPORT --safe-repo $SAFE_REPO --training-run-dir $RLDX_ROOT/final/${M}_seed0 \
    --output-dir $R7S/score/cutoff_check/${M}_seed0 --group all --device cuda --unseen-tasks score \
    --cutoff-rule rollout_length > $R7_LOG/safe_cutoff_${M}.log 2>&1
  echo "== $M"
  PYTHONPATH=$R7B_REPO "$SAFE_PY" -m robocasa.recovery.safe.cross_task_safe compare \
    $R7S/score/fold_all/${M}_seed0 $R7S/score/cutoff_check/${M}_seed0
done 2>&1 | tee $R7_LOG/safe_cutoff_compare.log
```

**D10. Assemble the ten fold directories (B′-G1, B′-G2).** It must count 10 `ASSEMBLY OK`.
Fold all of each head also prints four `reference ... -> OK` lines.

```bash
source $R7_LOG/paths.env; source $R7_LOG/paths_safe.env; cd $R7B_REPO
for M in lstm indep; do for F in all 0 1 2 3; do
  A=(--fold $F --folds $FOLDS --head $M --runtime-bundle $RLDXHS_ROOT/inputs/rldx_bundle.json
     --cohort train $RLDXHS_ROOT/inputs/train_ids.json --cohort calibration $RLDXHS_ROOT/inputs/calibration_ids.json
     --cohort test $RLDXHS_ROOT/inputs/test_ids.json --output-dir $R7S/folds/$M/fold_$F)
  for S in 0 1 2; do
    if [ $F = all ]; then
      RUN=$RLDX_ROOT/final/${M}_seed$S
      A+=(--frozen-normalized $S $RLDX_ROOT/calibration/$M/${M}_seed$S/normalized_scores.jsonl)
    else
      RUN=$R7S/train/fold_$F/${M}_seed$S
    fi
    A+=(--training-run $S $RUN --scores $S $R7S/score/fold_$F/${M}_seed$S)
  done
  if [ $F != all ]; then A+=(--fold-manifest $R7S/manifests/fold_$F.json); fi
  PYTHONPATH=$R7B_REPO "$SAFE_PY" -u -m robocasa.recovery.safe.cross_task_safe assemble-export "${A[@]}" 2>&1 | tail -n 7
done; done | tee $R7_LOG/safe_assemble.log
grep -c "ASSEMBLY OK" $R7_LOG/safe_assemble.log
```

**D11. The table with the SAFE rows.**

```bash
source $R7_LOG/paths.env; source $R7_LOG/paths_safe.env; cd $R7B_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_unseen.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS \
  --fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} \
  --fold-dirs $R7S/folds/lstm/fold_{0,1,2,3,all} --fold-dirs $R7S/folds/indep/fold_{0,1,2,3,all} \
  --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json --output $R7_ROOT/rldx_unseen_safe.json 2>&1 | tee $R7_LOG/unseen_safe.log
sha256sum -c --quiet $R7_LOG/frozen_inputs.sha256 && sha256sum -c --quiet $R7_LOG/frozen_safe.sha256 && echo INPUTS UNCHANGED
sha256sum $R7_ROOT/rldx_unseen_safe.json
```

Report:
- `safe_manifests.log`;
- the two training and two scoring `rc=0` counts;
- `safe_cutoff_compare.log`, `safe_assemble.log` and the whole `unseen_safe.log`, which ends
  in `RLDX-1 UNSEEN TABLE OK`;
- `INPUTS UNCHANGED` and the SHA-256.

Then copy `rldx_unseen_safe.json` to the evidence folder, as in D3.

**D3. The table (CPU, minutes; after D1).**

```bash
source $R7_LOG/paths.env; cd $R7_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/rldx_unseen.py run \
  --safe-dir $RLDX_ROOT/calibration/lstm --inputs-dir $RLDXHS_ROOT/inputs --folds $FOLDS \
  --fold-dirs $R7_ROOT/folds/fold_{0,1,2,3,all} --reference-per-seed $RLDXPS_ROOT/rldx_per_seed.json \
  --output $R7_ROOT/rldx_unseen.json 2>&1 | tee $R7_LOG/unseen.log
sha256sum -c --quiet $R7_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
sha256sum $R7_ROOT/rldx_unseen.json $FOLDS; echo "R7_TAG=$R7_TAG"
```

Report the whole `unseen.log` (it ends in `RLDX-1 UNSEEN TABLE OK` or `REPRODUCTION FAILED`),
`INPUTS UNCHANGED`, both SHA-256 values and `R7_TAG`. Then copy both files to the local evidence
folder:

```bash
# local machine
mkdir -p reports/safe_robocasa365_research_plan_20260823/evidence/<R7_TAG>
for f in rldx_unseen.json rldx_folds.json; do
  rsync -av tsubame:/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/<R7_TAG>/$f \
    reports/safe_robocasa365_research_plan_20260823/evidence/<R7_TAG>/
done
```

## Result

Recorded 2026-10-28.
- The R1–R4 sections above were not changed after the run.
- The runbook's checksum line was fixed before the run (`22f42dbc`). The analysis code is
  unchanged from `4a558327`.
- After the R1–R4 run and Stage A, two things were added before any fold is trained: "Stage B
  implementation" under R5, and runbook step 3.
- R5 Stage B ran from `9d0353be` with no code change. Its result follows R5 Stage A below.
- R6 ran from `8ef49963` with no code change. Only its runbook was fixed (C2b, the August
  test list) before fold all was rerun. Its result follows R5 Stage B below.
- R7's CPU rows ran from `9fef3adf` with no code change. Their result is the last section below.

**Run.**
- Folder: `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_review_analyses_20261028_155557`
  (`review_analyses.json`).
- Logs: `/gs/bs/tga-shinoda/felid/robocasa_logs/eval/xr1_review_analyses_20261028_155557`.
- Code: pinned worktree `robocasa_review_22f42dbc`, CPU only, about 10 minutes.

**Integrity checks.** All passed:
- **I1:** `fit_alpha(·, 0.05)` equals `fit_fast` for all nine detectors.
- **I2:** E1, Amendment 2 and dose-response reproduced exactly.
- **I3:** the α = 0.05 decision CIs equal Amendment 2's.
- **I4:** `INPUTS UNCHANGED`.
- **Also:** normalization difference 0.0; every test failure ends at H_k.

### R1. Lead–FPR curves

All gains are over P at the same α, in units of H_k. The columns:
- *same-α:* thresholds from calibration only;
- *P loosened:* P tuned on the test cohort to the fused detector's false alarms (favors the timer);
- *tightened:* the fused detector moved along its own path to P's false alarms (this loosens it
  when it has fewer false alarms than P, as Hide-and-Seek at α = 0.04);
- "—": no point on the path reaches P's false alarms.

| α | P: FP, lead | SAFE + P: FP; same-α / P loosened / tightened | Hide-and-Seek + P: FP; same-α / P loosened / tightened |
|---|---|---|---|
| 0.01 | 2, 0.040 | 2; +0.026 / −0.010 / — | 2; +0.028 / −0.008 / +0.061 |
| 0.02 | 3, 0.100 | 11; +0.057 / −0.024 / −0.001 | 3; +0.031 / +0.027 / +0.065 |
| 0.03 | 7, 0.155 | 13; +0.036 / +0.007 / −0.008 | 9; +0.048 / +0.036 / — |
| 0.04 | 19, 0.208 | 19; +0.016 / +0.017 / +0.015 | 13; +0.040 / +0.064 / +0.101 |
| **0.05** | **21, 0.225** | **23; +0.030 / +0.015 / +0.016** | **23; +0.071 / +0.056 / +0.067** |
| 0.06 | 25, 0.249 | 25; +0.019 / +0.017 / +0.021 | 24; +0.064 / +0.066 / +0.065 |
| 0.075 | 29, 0.273 | 30; +0.023 / +0.024 / +0.013 | 33; +0.086 / +0.057 / +0.070 |
| 0.10 | 32, 0.293 | 32; +0.037 / +0.037 / +0.030 | 42; +0.102 / +0.058 / +0.072 |
| 0.125 | 33, 0.299 | 45; +0.070 / +0.027 / +0.031 | 46; +0.115 / +0.069 / +0.060 |
| 0.15 | 33, 0.299 | 51; +0.090 / +0.027 / +0.026 | 53; +0.140 / +0.072 / +0.064 |
| 0.20 | 33, 0.299 | 75; +0.116 / +0.020 / −0.004 | 79; +0.180 / +0.077 / +0.060 |

- **P stops loosening at 33 false alarms.** For α ≥ 0.125 it stays at 33 because its margin
  cannot go below 0 (the 95th percentile of success durations). Above 0.10, same-α gains
  therefore compare against a timer that cannot spend its budget.
- **Noise + P** gains at most +0.006 on average for α ≤ 0.10 (maximum +0.017 over 20 draws).
  At α = 0.125, 0.15 and 0.20 it gains +0.014, +0.033 and +0.064. All of that comes from extra
  false alarms (73 on average at 0.20).
- **The global timer G** is conservative: 8 false alarms at α = 0.05.

**Band summary, α = 0.02–0.10.** Each failure's gain is averaged over the 7 targets. CIs are
test-only.

| Detector | Same-α | Matched (P loosened) | Matched ≤ 0 at | Reading |
|---|---|---|---|---|
| SAFE | +0.031 [+0.006, +0.059] | +0.013 [−0.011, +0.041] | 0.02 | budget-dependent |
| **Hide-and-Seek** | +0.063 [+0.006, +0.115] | **+0.052 [−0.005, +0.105]** | none | budget-dependent |
| first-inference | +0.003 [−0.001, +0.008] | −0.006 [−0.010, −0.000] | 5 of 7 | later than the timer at matched false alarms |
| zero-input | +0.007 [−0.003, +0.024] | −0.005 [−0.016, +0.011] | 5 of 7 | no earlier warning shown |
| same-task shuffle | +0.009 [−0.007, +0.027] | −0.020 [−0.036, −0.003] | 6 of 7 | later than the timer at matched false alarms |
| cosine kNN | +0.025 [+0.005, +0.050] | +0.007 [−0.013, +0.033] | 0.02 | budget-dependent |
| **Mahalanobis** | +0.040 [+0.016, +0.066] | **+0.024 [+0.000, +0.051]** | none | **earlier at every budget** |
| PCA-KMeans | +0.026 [+0.002, +0.054] | +0.008 [−0.016, +0.036] | 0.02 | budget-dependent |
| RND | +0.025 [+0.008, +0.047] | +0.004 [−0.013, +0.025] | 0.02, 0.10 | budget-dependent |

**Decision bootstrap (refits, 2,000 replicates).** Gains at the same α, with 95% CIs.

| α | SAFE | Hide-and-Seek | H&S − SAFE | H&S − first | H&S − zero | H&S − shuffle |
|---|---|---|---|---|---|---|
| 0.02 | +0.057 [+0.005, +0.120] | +0.031 [+0.012, +0.172] | [−0.059, +0.093] | [+0.002, +0.163] | [−0.000, +0.129] | [−0.025, +0.118] |
| 0.05 | +0.030 [+0.007, +0.093] | +0.071 [+0.022, +0.157] | [−0.011, +0.103] | [+0.014, +0.153] | [+0.013, +0.131] | [+0.004, +0.127] |
| 0.10 | +0.037 [+0.016, +0.098] | +0.102 [+0.057, +0.161] | [+0.021, +0.094] | [+0.038, +0.140] | [+0.049, +0.132] | [+0.036, +0.126] |

H&S − SAFE excludes 0 only at α = 0.10. There Hide-and-Seek uses 42 false alarms and SAFE 32, so
the contrast is partly a difference in false alarms. At matched false alarms the two gains are
+0.058 and +0.037.

**Reading.**
- **Hide-and-Seek.** Its matched point gain is positive at every target in the band, under
  both matched procedures: +0.027 to +0.066 with P loosened, +0.065 to +0.101 tightened. But the
  band-average CI with P loosened includes 0 (lower bound −0.005). By the rule fixed in advance
  it is "budget-dependent", not the expected "earlier at every budget".
- **SAFE.** Its same-α gains are mostly bought with false alarms: matched, it gains +0.013 in the
  band.
- **The controls** gain nothing, or lose, at matched false alarms, although their same-α gains
  reach +0.05 to +0.09 at loose targets.
- **Mahalanobis** is the one detector "earlier at every budget", by a small margin (+0.024; at
  α = 0.05, +0.030 [+0.010, +0.053] with P loosened).

### R2. The α = 0.05 table

| Detector | FP (FPR, Wilson) | Lead | TWA | Gain [decision CI] | P loosened [test-only CI] | Tightened to 21 FP |
|---|---|---|---|---|---|---|
| P | 21 (0.055) | 0.225 | 0.585 | — | — | — |
| G | 8 (0.021) | 0.123 | 0.551 | — | — | — |
| SAFE + P | 23 (0.061, ≤ 0.089) | 0.255 | 0.597 | +0.030 [+0.007, +0.093] | +0.015 [−0.007, +0.041] | +0.016 [−0.004, +0.042] |
| Hide-and-Seek + P | 23 (0.061, ≤ 0.089) | 0.296 | 0.618 | +0.071 [+0.022, +0.157] | +0.056 [−0.014, +0.120] | +0.067 [−0.003, +0.130] |
| first-inference + P | 21 (0.055) | 0.226 | 0.586 | +0.001 [−0.015, +0.036] | −0.003 | +0.006 |
| zero-input + P | 23 (0.061) | 0.237 | 0.588 | +0.012 [+0.001, +0.044] | −0.003 | −0.003 |
| shuffle + P | 24 (0.063) | 0.236 | 0.587 | +0.011 [+0.000, +0.064] | −0.011 | −0.013 |
| Mahalanobis + P | 21 (0.055) | 0.259 | 0.602 | +0.034 | +0.030 [+0.010, +0.053] | +0.037 [+0.018, +0.060] |
| cosine kNN + P | 22 (0.058) | 0.246 | 0.594 | +0.021 | +0.009 | +0.018 |
| PCA-KMeans + P | 23 (0.061) | 0.247 | 0.593 | +0.022 | +0.007 | +0.018 |
| RND + P | 22 (0.058) | 0.245 | 0.593 | +0.020 | +0.007 | +0.009 |
| SAFE + G | 17 | 0.217 | 0.586 | +0.094 over G, −0.008 over P | | |
| Hide-and-Seek + G | 24 | 0.295 | 0.616 | +0.172 over G, +0.070 over P | | |

### R3. Per-task gains (α = 0.05)

| Detector | + / 0 / − tasks | Sign test p | Median [IQR] | Top-5 share | Leave-one-task-out min | Between-task SD vs within SE | Atomic / composite mean (positive) | Reading |
|---|---|---|---|---|---|---|---|---|
| SAFE | 15 / 3 / 20 | 0.50 | −0.009 [−0.015, +0.032] | 0.88 | +0.022 | 0.078 vs 0.013 | +0.081 (7/10) / +0.012 (8/28) | concentrated |
| Hide-and-Seek | 21 / 1 / 16 | 0.51 | +0.054 [−0.092, +0.260] | 0.78 | +0.058 | 0.218 vs 0.042 | +0.160 (8/10) / +0.039 (13/28) | concentrated |
| Mahalanobis | 16 / 19 / 3 | 0.004 | 0.000 [0.000, +0.031] | 0.71 | +0.027 | 0.066 vs 0.022 | +0.053 (7/10) / +0.027 (9/28) | concentrated |

The other detectors are also "concentrated" (the controls have almost no gain to concentrate).

- **The task gains are bimodal.** On 21 tasks Hide-and-Seek + P alarms earlier than P on most
  failures (top: PickPlaceCounterToCabinet +0.55, StoreLeftoversInBowl +0.43,
  PickPlaceCounterToStove +0.38). Its gain is negative on 16 tasks, and on 14 of them it alarms
  **later than P on all 10 failures**, by up to 0.21 H_k. Several tasks sit at exactly −0.213.
  - **The mechanism:** the fused path relaxes its timer margin (λ above P's δ) to spend the
    false-alarm budget on the score.
  - Where the score fires early, the gain is large. Where it does not, the relaxed timer alarms
    later than P alone.
  - The pooled gain is a reallocation of the false-alarm budget across tasks, not an earlier
    alarm on every task.
- **The gain survives dropping any one task:** the leave-one-task-out minimum is +0.058, still
  ≥ 0.05. But five tasks carry 78% of the summed gain.
- **SAFE and Hide-and-Seek help on the same tasks** (Spearman 0.71 over task gains).
- **False alarms per task (P):** 24 tasks have 0, 6 have ≥ 2, and the most is 3 of 10. Under
  Hide-and-Seek + P the most is 4 of 10 (PickPlaceCounterToCabinet). The 5% bound is marginal over
  tasks, not per task.

### R4. Calibration reuse and the test FPR

**R4a. Path chosen on half of the calibration successes, margin set on the other half.**
All columns are at α = 0.05.

| Detector | FP (FPR) | Lead | Gain over P from the same half (23 FP, lead 0.239) | Gain over the registered P | 20 random halvings: median FPR; median gain [range] |
|---|---|---|---|---|---|
| SAFE | 23 (0.061) | 0.260 | +0.021 [−0.002, +0.048] | +0.035 | 0.054; +0.005 [−0.067, +0.039] |
| Hide-and-Seek | 23 (0.061) | 0.304 | **+0.065 [−0.001, +0.127]** | +0.079 | 0.037; **+0.018 [−0.250, +0.115]** |
| first-inference | 22 | 0.235 | −0.004 | +0.010 | 0.057; −0.006 |
| zero-input | 26 | 0.251 | +0.012 | +0.025 | 0.065; −0.012 |
| shuffle | 21 | 0.196 | −0.043 | −0.029 | 0.054; −0.019 |

- **By the rule fixed in advance**, Hide-and-Seek "does not depend on reusing calibration
  successes": on the deterministic split its gain is +0.065, with FPR 0.061.
- **Across random halvings, however**, its median gain is only +0.018, with a range from −0.25 to
  +0.11. With 190 successes for path selection, the chosen path is unstable. When the selected
  score threshold rarely fires, the relaxed timer costs lead everywhere (see R3).

**R4b. 200 random re-splits of the pooled successes.** Test FPR at α = 0.05.

| | Mean over re-splits (SD) | P(> 0.05) | P(> 0.075) | Actual split | Percentile of actual |
|---|---|---|---|---|---|
| P (no path selection) | 0.047 (0.016) | 0.35 | 0.04 | 0.055 | 70 |
| SAFE + P, registered fit | 0.054 (0.016) | 0.53 | 0.12 | 0.061 | 62 |
| SAFE + P, split path | 0.043 (0.020) | 0.28 | 0.06 | 0.061 | 84 |
| Hide-and-Seek + P, registered fit | 0.055 (0.016) | 0.57 | 0.13 | 0.061 | 66 |
| Hide-and-Seek + P, split path | 0.045 (0.018) | 0.36 | 0.05 | 0.061 | 81 |

- **The effects:**

  | Detector | Selection effect | Cohort effect |
  |---|---|---|
  | SAFE | 0.012 | 0.006 |
  | Hide-and-Seek | 0.010 | 0.006 |

  The reading for both is "mainly path selection on calibration successes".
- **About 0.003 of the selection effect is the smaller certification set**: the bound is
  1 − 182/191 = 0.047, against 1 − 362/381 = 0.050.
- **Against P**, which has the same calibration set and no selection, the registered fits run
  about 0.008 higher.
- **The method check passed**: with exchangeable successes, P and the split path stay at or below
  0.05 on average.
- **The 0.061 on the actual split** is at the 62nd–66th percentile for the registered fits, so it
  is within chance for them.
- **Choosing the path on the calibration successes** raises the expected test FPR from about 0.047
  to 0.055, and the chance of exceeding 0.075 from 4% to 12–13%.

### Expectations against results

| Expectation | Result |
|---|---|
| R1 SAFE: "no earlier warning shown" or "budget-dependent" | budget-dependent (matched +0.013) |
| R1 Hide-and-Seek: "earlier at every budget" | **no**: budget-dependent (matched point > 0 at every target; band CI lower bound −0.005) |
| R1 controls and noise: no earlier warning | yes; first-inference and shuffle are slightly later than P at matched false alarms |
| R1 baselines like SAFE | yes, except Mahalanobis (earlier at every budget, +0.024) |
| R2 Hide-and-Seek matched ≈ +0.06 | +0.056 (P loosened), +0.067 (tightened) |
| R3 SAFE concentrated | yes (top-5 share 0.88) |
| R3 Hide-and-Seek mixed | **no**: concentrated (0.78); bimodal |
| R3 most tasks 0–1 false alarm under P | yes (32 of 38) |
| R4a Hide-and-Seek positive, ≥ 0.05 open | +0.065 on the deterministic split; median +0.018 over random halvings |
| R4b cohort effect 0.005–0.01 and a similar selection effect | cohort 0.006, selection 0.010–0.012 |

### What this changes for the paper

- **Report gains at matched false alarms, not only at the same α.** At loose targets, same-α
  gains of the controls and of noise reach +0.06 to +0.09. Even at α = 0.05, SAFE's and
  Hide-and-Seek's same-α gains exceed their matched gains by 0.015, because both use two more
  false alarms than P.
- **Hide-and-Seek's earlier warning holds at equal false alarms** (+0.056 to +0.067 at 0.05,
  positive at every target from 0.02 to 0.10). But its uncertainty is task-driven: the gain is
  bimodal across tasks, concentrated in five, and on 16 tasks the fused detector warns later
  than P alone. The paper should show the per-task distribution and explain the budget
  reallocation.
- **SAFE adds about +0.015 at equal false alarms.** Against G it looks useful (+0.094); against P
  it does not (−0.008).
- **Conformal scope:**
  - the bound is marginal over tasks (up to 4 of 10 successes alarmed in one task);
  - choosing the path on the calibration successes costs about 0.008 in expected FPR;
  - the split path restores the bound, but it is unstable at 190 successes.
- **For E2:** choose the fused path on a separate selection set, sized so that the certification
  set keeps about 380 successes. Do not halve the calibration set. This resolves the draft's TODO
  on held-out path selection.

### R5 Stage A

- **A1.** In the SAFE repo, `task_min_step` is set by the data loaders (`set_task_min_step`, the
  earliest termination per task) and read only in `utils/metrics.py` and `utils/video.py`. The use
  in `utils/routines.py` is commented out. The model and the dataset masks do not read it. A
  held-out task can therefore be scored with its rollout length as the cutoff (Stage B rule).
- **A2.** The E1 runtime checkpoints are the final refit (SHA-256 identical for seeds 0–2; same
  config). The refit used:
  - `--selection-manifest .../seen38_split_seed0.json` (1,824 train, 76 test, 38 tasks);
  - `--selection-summary .../xr1_safe_pooled_39tasks_50each_seed0_20260812_221815_full_cv/cv_selection_summary.json`;
  - official inverse-frequency weighting, no online mode;
  - MLP settings: lr 1e-4, weight decay 0.01, λ_reg 1e-3, batch 512, 1,000 epochs.

  The original log shows 49 minutes for three seeds (23:15–00:04 on 2026-08-14/15).
- **A3. Retraining is bit-exact.**
  - **Run:** seed 0 retrained on r1n8 (H100), with the same export, selection manifest, selection
    summary and settings, from the pinned worktree `22f42dbc`. Output:
    `$RV_ROOT/safe_retrain_check/indep_seed0`; log: `$RV_LOG/a3_retrain_seed0.log`.
  - **Checkpoint:** SHA-256 identical to the frozen `indep_seed0/model_final.ckpt` (`c8b0124e…`).
  - **Config:** identical.
  - **Scores:** all 1,900 score trajectories equal (maximum absolute difference 0.0).
  - **Metrics:** selected hyperparameters, task cutoffs, class weights and objective are equal.
  - **Time:** 18:33–19:16 JST, 43 minutes. Almost all of it is loading the 1,950-rollout export
    (about 1.3–2.2 s per rollout from `/gs/bs`).
  - **A second launch, not used:** a duplicate launch at 18:42 (`a3_retrain_r2.sh`) had no GPU
    visible (`CUDA_VISIBLE_DEVICES` empty), so it cannot train. Its log is
    `a3_retrain_seed0_r2*.log` and its output folder `indep_seed0_r2*`. It plays no part in the
    result.
  - **Consequence for Stage B:** gate B-G2 is the exact form. Retrained fold all must equal the
    frozen trajectories, and the frozen trajectories are the seen reference.

### R5 Stage B

**SAFE does not carry warning to unseen tasks. U1-S is not supported.** SAFE + G gains +0.017
over G, below the 0.05 margin. Hide-and-Seek + G gains +0.127 on the same folds and replicates.
Hide-and-Seek − SAFE is +0.111 [0.036, 0.166].

**Run.**
- Folder: `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_cross_task_safe_20261028_193716`
  (`cross_task_safe.json`, SHA-256 `113864b4…`, copied to
  `reports/safe_robocasa365_research_plan_20260823/evidence/xr1_cross_task_safe_20261028_193716/`).
- Logs: `/gs/bs/tga-shinoda/felid/robocasa_logs/eval/xr1_cross_task_safe_20261028_193716`.
- Code: pinned worktree `robocasa_xtsafe_9d0353be` on r1n8 (2× H100), all five fold directories
  assembled by `9d0353be`.
- 12 fold models (folds 0–3 × seeds 0–2); fold all is the frozen refit. Both E1 cohorts scored
  with every model. Decision bootstrap 2,000 replicates, seed 0.

**Checks.**
- **B0:** 20 unit tests passed; both GPUs empty at the start.
- **B1:** folds 0 and 1 have 28 seen tasks and 1,344 training rollouts; folds 2 and 3 have 29 and
  1,392. Exactly 48 training rollouts are removed per held-out task. `MANIFESTS OK`.
- **B2, B3:** all 12 trainings and 30 scorings ended with exit code 0 (`42 ok`).
- **B-G1a:** fold all's scores, through `--unseen-tasks score`, equal E1's `per_seed_raw_safe`
  on every row of the E1 prediction files (maximum difference 0.0). That is 1,522 calibration-campaign
  and 1,526 test-campaign rollouts, a superset of the 760 + 760 cohort rollouts.
- **B-G1b:** fold all seed 0, scored with every cutoff set to the rollout length, equals its
  scores with training cutoffs on all 1,526 test-campaign rollouts (maximum difference 0.0,
  `IDENTICAL`). The cutoff does not change a score, as A1 found.
- **B-G3:** fold all's per-task normalization equals the runtime bundle's (maximum difference 0.0).
- **S1:** every fold assembled with its own seen tasks, checkpoints and held-out tasks.
- **S2:** E1 reproduced exactly: P 0.2252, 21 FP; SAFE + P 0.2550, 23 FP; gain +0.0298.
- **S3:** Hide-and-Seek's zero-shot gain (+0.1273) and decision CI ([0.0687, 0.2105]) equal the
  cross-task run's. SAFE and Hide-and-Seek therefore share its replicates.
- **Inputs:** `INPUTS UNCHANGED` after the analysis. The hashes recorded in the JSON also equal
  those of the R1–R4 and cross-task runs: both E1 prediction files, the runtime bundle,
  `confirm_lead.json`, the fold file and `cross_task_lead.json`.
- **Copy:** the local `cross_task_safe.json` has the cluster file's SHA-256.

**Unseen arm, α = 0.05** (380 failures and 380 successes, pooled over the four folds).

| | Lead | FP (FPR) | Gain over G | Decision CI |
|---|---|---|---|---|
| G | 0.126 | 9 (0.024) | | |
| SAFE + G | 0.143 | 20 (0.053) | +0.017 | [0.003, 0.091] |
| Hide-and-Seek + G | 0.253 | 18 (0.047) | +0.127 | [0.069, 0.211] |
| Hide-and-Seek − SAFE | | | +0.111 | [0.036, 0.166] |

- **U1-S.** CI above 0: yes. Point ≥ 0.05: **no** (0.017). FPR ≤ 0.075: yes (0.053). Wilson upper
  ≤ 0.10: yes (0.080). Not supported. The test-only bootstrap CI is [−0.022, +0.060].
- **Nulls** (20 draws each; the reading is interpretable because both means are below 0.05):
  - Gaussian noise: mean gain +0.005, maximum +0.019. One draw of 20 reaches SAFE's +0.017.
  - SAFE resampled within task: mean +0.002, maximum +0.009. No draw reaches +0.017.
  - SAFE's gain is above both null means and above every resampled draw, so it carries some
    temporal signal. It is the size of the best noise draw.
- **At matched false alarms** (G loosened on the test cohort, which favors the timer; test-only
  bootstrap):
  - SAFE + G (20 FP) against G at 19 FP (lead 0.221): **−0.079 [−0.119, −0.034]**. SAFE's gain at
    α = 0.05 is paid for with 11 more false alarms than G, and those buy less lead than loosening
    G does.
  - Hide-and-Seek + G (18 FP) against G at 18 FP (lead 0.203): **+0.051 [−0.002, +0.104]**.
- **U5-S, timers from n successes of the task** (gain over P_n and Hide-and-Seek − SAFE, decision
  CIs):

  | n | P_n | SAFE + P_n | Gain | Hide-and-Seek − SAFE |
  |---|---|---|---|---|
  | 1 | 0.169 | 0.182 | +0.013 [0.001, 0.077] | +0.086 [0.018, 0.155] |
  | 3 | 0.186 | 0.195 | +0.010 [0.000, 0.074] | +0.088 [0.020, 0.158] |
  | 5 | 0.201 | 0.209 | +0.008 [−0.001, 0.069] | +0.080 [0.024, 0.155] |
  | all | 0.228 | 0.238 | +0.009 [0.000, 0.047] | +0.059 [0.027, 0.144] |

  SAFE adds about +0.01 at every n. Hide-and-Seek stays ahead of SAFE at every n.
- **U-HS:** +0.111 [0.036, 0.166]. Hide-and-Seek warns earlier than SAFE on unseen tasks.
- **U4-S (transfer loss):** unseen gain +0.017, seen gain +0.011; difference +0.006
  [−0.028, +0.041]. No loss is detected because, with the unseen arm's pooled normalization,
  SAFE gains little over G on seen tasks too (9 FP against 8). With E1's per-task normalization
  it gains +0.094 over G (R2) but stays 0.008 below P; over P it gains +0.030 (S2), +0.015 at
  matched false alarms (R2).

**Descriptive.**
- **Per fold** (gain over G; FP of G → SAFE + G): fold 0 0.000 (5 → 5); fold 1 +0.011 (1 → 7);
  fold 2 +0.023 (3 → 3); fold 3 +0.036 (0 → 5). In fold 0 the chosen path makes SAFE's score
  almost unused (u = 1.45). The 11 extra false alarms are in folds 1 and 3.
- **By task type:** atomic tasks (100 failures) −0.041, composite tasks (280 failures) +0.037. The
  fused detector warning later than G on atomic tasks fits R3's mechanism (the fused path relaxes
  the timer's margin); this run does not test it.

**Expectations against results.**
- "U1-S is likely to pass against G": **wrong**. The CI and FPR criteria pass; the point gain is a
  third of the margin.
- "Against G at matched false alarms, and against P_all, SAFE's gain should be near +0.030 or
  below": held (−0.079 and +0.009).
- "The informative number is U-HS": +0.111 [0.036, 0.166].

**Caveats.**
- The fold models are trained on 1,344–1,392 rollouts instead of 1,824. Their MLP settings are the
  frozen ones, chosen by cross-validation on the pool that contains the held-out tasks. That can
  only favor SAFE.
- The matched-false-alarm CIs are test-only. G is loosened on the test cohort.

**What this changes for the paper.**
- **On unseen tasks, Hide-and-Seek transfers and SAFE does not.** Hide-and-Seek: +0.127 over G,
  +0.051 at matched false alarms. SAFE: +0.017 (the size of a noise score's), −0.079 at matched
  false alarms. The gap is +0.111 [0.036, 0.166] and holds with P_n for every n.
- **Beyond the per-task timer, SAFE adds little on any task.** Over P: +0.030 seen (+0.015 at
  matched false alarms), about +0.01 unseen. Over G the seen gain depends on normalization:
  +0.011 with the unseen arm's pooled normalization, +0.094 with E1's per-task normalization
  (R2); unseen it is +0.017. This supports the review's wording, "limited additional warning after
  controlling for duration", rather than "SAFE mostly tracks time".
- **Report matched false alarms beside every G comparison.** Against G alone, SAFE's α = 0.05 CI
  excludes 0.

### R6

**On unseen tasks, Hide-and-Seek is the only learned detector that beats the global timer.**
Standalone, every other learned detector's unseen TWA (0.500 to 0.547) is at or below G's
(0.551). Fused with G, their gains come with two to three times G's false alarms, and their TWA
(0.549 to 0.554) equals G's. Hide-and-Seek has standalone TWA 0.595 and gains
+0.122 ± 0.005 over G at FPR 0.059.

**Run.**
- Folder: `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_seen_unseen_20261028_235911`
  (`seen_unseen_tables_b1ids.json`, SHA-256 `d927f67e…`, copied to
  `reports/safe_robocasa365_research_plan_20260823/evidence/xr1_seen_unseen_20261028_235911/`).
- Logs: `/gs/bs/tga-shinoda/felid/robocasa_logs/eval/xr1_seen_unseen_20261028_235911`.
- Code: pinned worktree `robocasa_seenunseen_8ef49963` on r1n8, CPU only, with no code change.
- The first table run failed T4 because the runbook passed the wrong August test list (C2b).
  Fold all was rerun with B1's list as `fold_all_b1ids`. The first `fold_all` and its failed
  `seen_unseen_tables.json` are kept.

**Checks.**
- **C1, C2:** the four OOD folds ended with `=== OOD FOLD F DONE`, and ActProbe folds 0–3 with
  rc=0.
- **Fold all rerun:** `REFERENCE OK`. August calibration (228 rollouts) and August test (760)
  match B1 with maximum difference 0.0.
- **T1:** every fold directory matches the fold file. ActProbe has folds 0–3 and all; SAFE, the
  cross-task folds and OOD have folds 0–3.
- **T2:** seen + P equals run C and `ood_per_seed_fused_heldout.json` for all nine rows
  (maximum difference 0.0).
- **T3:** the seed-averaged unseen + G gains equal the stored ones: Hide-and-Seek 0.1273,
  zero-input 0.0331, first-inference 0.0000, SAFE 0.0166.
- **T4:** reference differences 0.0 and 0.0; normalization relative difference 0.0.
- **T5:** every OOD fold is the pooled variant of bundle `c62d225a…`.
- **Inputs:** `INPUTS UNCHANGED`. The local copy has the cluster file's SHA-256.

**Table I, standalone** (one threshold per detector and seed: largest calibration lead at
calibration FPR ≤ 0.05; held-out test cohort, 380 + 380; mean ± sd over seeds 0–2).

| | Seen bACC | Seen TWA | Seen TPR / FPR | Unseen bACC | Unseen TWA | Unseen TPR / FPR |
|---|---|---|---|---|---|---|
| Mahalanobis | 0.818 ± 0.001 | 0.582 ± 0.001 | 0.689 / 0.053 | 0.726 ± 0.000 | 0.543 ± 0.000 | 0.518 / 0.066 |
| Cosine k-NN | 0.788 ± 0.000 | 0.577 ± 0.000 | 0.637 / 0.061 | 0.729 ± 0.000 | 0.544 ± 0.000 | 0.526 / 0.068 |
| PCA-KMeans | 0.796 ± 0.001 | 0.578 ± 0.001 | 0.657 / 0.064 | 0.728 ± 0.001 | 0.542 ± 0.001 | 0.523 / 0.067 |
| RND | 0.793 ± 0.002 | 0.572 ± 0.001 | 0.650 / 0.065 | 0.728 ± 0.001 | 0.547 ± 0.001 | 0.525 / 0.068 |
| SAFE-MLP | 0.763 ± 0.003 | 0.582 ± 0.011 | 0.568 / 0.043 | 0.713 ± 0.003 | 0.541 ± 0.001 | 0.472 / 0.046 |
| ActProbe | 0.654 ± 0.003 | 0.558 ± 0.006 | 0.356 / 0.048 | 0.500 ± 0.000 | 0.500 ± 0.001 | 0.009 / 0.009 |
| Hide-and-Seek | 0.742 ± 0.033 | 0.599 ± 0.005 | 0.532 / 0.048 | 0.790 ± 0.002 | 0.595 ± 0.005 | 0.642 / 0.061 |
| Time risk | 0.837 | 0.545 | 0.711 / 0.037 | — | — | — |
| Global timer G | 0.988 | 0.550 | 1.000 / 0.024 | 0.988 | 0.551 | 1.000 / 0.024 |
| Per-task timer P | 0.972 | 0.585 | 1.000 / 0.055 | — | — | — |

- Seen DT: P 0.775, Hide-and-Seek 0.754, the OOD rows 0.780 to 0.792, SAFE 0.792, ActProbe
  0.836, G 0.877.
- ActProbe's seen-task thresholds almost never fire on unseen tasks. Seeds 0 and 2 raise no
  alarm at all, and seed 1 raises 10 on failures and 10 on successes.

**Table II, fused** (seen: score + P, the published per-seed rows; unseen: score + G per fold,
pooled; mean ± sd over seeds 0–2. P: lead 0.225, 21 FP. G unseen: lead 0.126, 9 FP).

| | Seen FPR | Seen gain over P | Seen TWA | Unseen FPR (FP) | Unseen gain over G | Unseen TWA |
|---|---|---|---|---|---|---|
| Mahalanobis | 0.055 | +0.034 ± 0.000 | 0.602 | 0.073 (27–28) | +0.044 ± 0.001 | 0.549 |
| Cosine k-NN | 0.058 | +0.021 ± 0.000 | 0.594 | 0.079 (30) | +0.053 ± 0.000 | 0.550 |
| PCA-KMeans | 0.060 | +0.022 ± 0.001 | 0.594 | 0.071 (27) | +0.053 ± 0.001 | 0.554 |
| RND | 0.060 | +0.021 ± 0.001 | 0.593 | 0.073 (27–28) | +0.050 ± 0.001 | 0.551 |
| SAFE-MLP | 0.058 | +0.024 ± 0.015 | 0.596 | 0.053 (20) | +0.028 ± 0.009 | 0.551 |
| Hide-and-Seek | 0.055 | +0.039 ± 0.006 | 0.605 | 0.059 (21–24) | +0.122 ± 0.005 | 0.595 |
| zero-input | 0.062 | +0.013 ± 0.003 | 0.588 | 0.037 (13–15) | +0.020 ± 0.008 | 0.554 |
| first-inference | 0.055 | +0.001 ± 0.001 | 0.586 | 0.027 (10–11) | +0.002 ± 0.003 | 0.550 |
| same-task shuffle | 0.058 | +0.005 ± 0.001 | 0.586 | — | — | — |
| G | 0.024 | −0.102 | 0.550 | 0.024 (9) | — | 0.551 |

**Expectations against results.**
- "Unseen bACC and TWA fall for every learned detector; the global timer barely moves": **partly
  wrong**. They fall for the OOD rows, SAFE and ActProbe, and G moves by 0.001. Hide-and-Seek's
  bACC rises (0.742 → 0.790; its seen seeds spread 0.713 to 0.778), and its TWA stays
  (0.599 → 0.595).
- "Hide-and-Seek keeps the best unseen TWA among learned detectors": held (0.595 standalone and
  fused; next best 0.547 and 0.554).
- "The pooled OOD variants lose most of their seen-task advantage": held. Their standalone TWA
  goes from 0.572–0.582 to 0.542–0.547, below G. Their + G gains (+0.044 to +0.053) come with 27
  to 30 false alarms against G's 9, and their TWA equals G's.
- "SAFE's unseen + G gain stays near noise (+0.017 seed-averaged), with single models at or
  below it": **wrong on the second part**. Single models gain +0.028 ± 0.009, more than the seed
  average, all at 20 FP. The noise maximum in Stage B was +0.019.

**What this changes for the paper.**
- Tables I and II get a Seen | Unseen column group each. Table I moves to the held-out set, so
  its seen numbers change: OOD bACC 0.79–0.82 (was 0.76–0.80), G 0.988 (was 0.976), P 0.972
  (was 0.978); Hide-and-Seek TWA 0.599 (was 0.595), P 0.585 (was 0.575).
- On unseen tasks the standalone ranking is clearer than on seen ones. Only Hide-and-Seek's TWA
  is above G's, and the timer still has the best bACC.
- The unseen + G gains of the OOD rows are spent false alarms, not earlier warning: TWA equal to
  G's at three times its FPR. The table needs the FPR column beside the gain.
- Table IV becomes redundant. Its inferential numbers stay in the text: Hide-and-Seek − SAFE
  +0.111 [0.036, 0.166]; G at matched false alarms (SAFE −0.079, Hide-and-Seek +0.051); P_n.

### R7 (CPU rows)

**On RLDX-1's unseen tasks, Hide-and-Seek again adds the most over G.** Its gain is
+0.196 ± 0.013, against +0.010 and +0.038 for the zero-input and first-inference controls.
Unlike on XR-1, the default OOD detectors also gain over G (+0.099 to +0.115). They do so at
FPR 0.010–0.040, and their TWA (0.544 to 0.564) is above G's (0.513). G is very weak here:
lead 0.026 with no false alarm.

**Run.**
- Folder: `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/rldx1_unseen_20261029_135501`
  (`rldx_unseen.json`, SHA-256 `58a68f5b…`, copied to
  `reports/safe_robocasa365_research_plan_20260823/evidence/rldx1_unseen_20261029_135501/`;
  `rldx_folds.json`, SHA-256 `079ce858…`, also copied).
- Logs: `/gs/bs/tga-shinoda/felid/robocasa_logs/eval/rldx1_unseen_20261029_135501`.
- Code: pinned worktree `robocasa_rldxunseen_9fef3adf` on r4n5, CPU only.
  - The five folds were scored at once. The run folder was created at 13:55, and the last
    fold (all) finished at 14:28.
  - The table took minutes.
- **Folds:** 6, 6, 4 and 4 tasks. RLDX-1 has 14 atomic and 6 composite tasks, so the folds
  are 4 + 2, 4 + 2, 3 + 1 and 3 + 1. The rule deals each task type from fold 0, so folds 0 and
  1 get the remainders.

**Checks.** All passed:
- Five scoring runs with rc=0.
- **R1:** fold all ends with `REFERENCE OK`, and all 21 reference checks pass.
  - 18 are exactly 0.
  - PCA-KMeans differs by at most 1.4e-14 in the three cohorts. That is floating-point
    rounding, within the 1e-12 tolerance, and R2 is still exact.
  - The fold directories match the fold file. The folds' tasks are recorded in
    `rldx_unseen.json` (`checks.R1.folds`).
- **Timer and calibration:** G's λ is 0.947–0.973 per fold. Each fold calibrates on 42–48
  seen-task successes and 225–293 seen-task training failures.
- **R2:** seen + P equals `rldx_per_seed.json`, with P at 0.2711 and FPR 0.03.
- **R3:** one G for every score.
- `INPUTS UNCHANGED`.

**Table II, RLDX-1, unseen + G** (held-out test, 100 + 100, pooled over folds; mean ± sd over
seeds 0–2; seed-averaged score in the last column). G alone: FPR 0.000 (0 FP), lead 0.026,
TWA 0.513.

| | FPR (FP per seed) | Gain over G | TWA | Seed average |
|---|---|---|---|---|
| Hide-and-Seek | 0.057 ± 0.032 (7, 8, 2) | +0.196 ± 0.013 | 0.583 ± 0.021 | +0.191 at 0.050 |
| zero-input | 0.023 ± 0.040 (7, 0, 0) | +0.010 ± 0.024 | 0.506 ± 0.008 | +0.030 at 0.070 |
| first-inference | 0.090 ± 0.010 (10, 9, 8) | +0.038 ± 0.025 | 0.487 ± 0.012 | +0.033 at 0.100 |
| Mahalanobis | 0.030 (3, 3, 3) | +0.112 ± 0.000 | 0.554 ± 0.000 | +0.114 at 0.030 |
| Cosine k-NN | 0.010 (1, 1, 1) | +0.112 ± 0.000 | 0.564 ± 0.000 | +0.112 at 0.010 |
| PCA-KMeans | 0.040 (4, 4, 4) | +0.115 ± 0.004 | 0.550 ± 0.002 | +0.113 at 0.050 |
| RND | 0.037 ± 0.012 (5, 3, 3) | +0.099 ± 0.013 | 0.544 ± 0.011 | +0.097 at 0.050 |

The seen + P rows are the published ones, reproduced exactly (R2). Same-task shuffle is seen
only.

**Expectations against results.**
- "Every score can gain a lot over G, controls included": **wrong**. The controls gain +0.010
  and +0.038. The first-inference control's TWA (0.487) is below G's: it buys little lead with
  8–10 false alarms.
- "Hide-and-Seek gains the most and stays above the controls": held.
- "Its margin should be smaller than on XR-1": **wrong**. Hide-and-Seek − zero-input is +0.186
  here, against +0.102 on XR-1 (R6 single models).
- "The OOD rows gain little more than the controls, and their FPR may exceed 0.05": **wrong**.
  They gain +0.10 to +0.115 at FPR ≤ 0.040.
- "The 5% threshold is coarse": held. G raised no false alarm on 100 test successes.

**Caveats.**
- G uses no false alarm, while the fused rows use 1 to 8. No matched-false-alarm comparison
  was run. On XR-1, loosening G from 9 to 18–19 false alarms raised its lead from 0.126 to
  0.20–0.22 (R5).
- Single-seed FPR varies widely for the sequence models: Hide-and-Seek's seeds have 2 to 8
  false alarms, and zero-input's 0 to 7.
- 100 test failures; folds of 4 to 6 tasks.

**SAFE (Stage A′, file reads).**
- **A′1.** The final models are in `$RLDX_ROOT/final/{lstm,indep}_seed{s}`, with
  `metrics.json` and `scores.jsonl`. `normalized_scores.jsonl`'s `full_raw_scores` equal the
  final models' `scores` on all 1,000 rollouts, for both heads (seed 0; maximum difference
  0.0). The frozen raw trajectories are therefore the seen reference.
- **A′2.** The training settings are:
  - export `.../rldx1_original_safe35_natural50_20260808_203158/official_safe_supported20_natural50`;
  - selection manifest `$RLDX_ROOT/protocol/frozen_outer_split.json`;
  - selection summary `$RLDX_ROOT/cv/cv_selection_summary.json`;
  - official inverse-frequency weighting (class weights 2.183 and 1.836), rollout split unit.

  The split manifest has 740 train, 60 calibration and 200 evaluation rollouts. The training
  logs are `rldx1_safe20_natural_weighted_20260815_162530_{final_lstm,final_indep,master}.log`.
  The exact command and the time per fit come from them before A′3.
- **A′2, continued** (`a2b_commands.txt`).
  - **Pipeline:** one driver ran on 2026-08-15 from 16:25 to 20:46. It did a 6-fit smoke run,
    810 inner-CV fits, 6 final refits (3 seeds × 2 heads), calibration and an audit
    (`COMPLETE, FROZEN, AND HEALTHY`). Its log prints stage banners only, and the driver
    script is not in the repo, so the exact final command is not recorded.
  - **Final settings** (`metrics.json`):
    - LSTM: h 0.0, d 1.0, lr 3e-5, λ_reg 1e-2.
    - MLP: h concat-2, d concat-2, lr 1e-3, λ_reg 1e-3.
    - Both: official loss (`pinned_safe_output`, focal γ 2.0 recorded) and official
      inverse-frequency weights.
    - Train 740 (402 S / 338 F), test 260; `task_min_step` from the training split.
  - **Files:** each `final/<head>_seed<s>` has `config.yaml`, `metrics.json`,
    `model_final.ckpt`, `scores.jsonl`, `split_manifest.json` and `train_history.json`.
  - **Split:** `frozen_outer_split.json` has train 740 and test 260 (60 calibration + 200
    evaluation), with 20 included tasks.
- **A′3 (written now, two GPUs).** The command is rebuilt from `train_seen_tasks`' options:
  - the recorded selection manifest and selection summary;
  - the trainer's defaults for everything else recorded: class weighting, loss mode, focal γ,
    split seed 0, 1,000 epochs, no online mode.

  The check is the proof that the command is right. All of these must hold:
  - `config.yaml` is identical;
  - the checkpoint SHA-256 is identical;
  - all 1,000 trajectories are equal;
  - selected hyperparameters, task cutoffs, class weights, objective and counts are equal.

  If `config.yaml` differs, its diff names the setting to fix, and nothing else follows until
  A′3 is exact. If the config is equal but the scores differ, Stage B′ uses R5's approximate
  B-G2 instead.
- **A′3. Retraining is bit-exact for both heads.**
  - **Run:** LSTM seed 0 on GPU 0 and MLP seed 0 on GPU 1, on r4n5, from the pinned worktree
    `9fef3adf`, with the D4 command. Output: `$R7_ROOT/safe_retrain_check/{lstm,indep}_seed0`.
  - **Checkpoints:** SHA-256 identical to the frozen ones (LSTM `8bccbab9…`, MLP `8ca1aa49…`).
  - **Config:** `config.yaml` identical for both heads.
  - **Scores:** all 1,000 trajectories equal (maximum absolute difference 0.0), same IDs.
  - **Metrics:** selected hyperparameters, task cutoffs, class weights, objective and counts are
    equal.
  - **Consequence for Stage B′:** the gate is the exact form. Fold all is the frozen SAFE20
    models, and the rebuilt command trains the folds.
