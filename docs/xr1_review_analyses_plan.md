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

## Result

Recorded 2026-10-28.
- The R1–R4 sections above were not changed after the run.
- The runbook's checksum line was fixed before the run (`22f42dbc`). The analysis code is
  unchanged from `4a558327`.
- After the R1–R4 run and Stage A, two things were added before any fold is trained: "Stage B
  implementation" under R5, and runbook step 3.

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
