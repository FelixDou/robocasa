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
- **Expectation:**
  - U1-S is likely to pass against G, because G is weak and its margin is conservative.
  - Against G at matched false alarms, and against P_all, SAFE's gain should be near its
    seen-task +0.030 or below.
  - The informative number is U-HS.

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
tail -n 45 $RV_LOG/review_analyses.log
sha256sum -c $RV_LOG/frozen_inputs.sha256 | tail -n 3 && echo INPUTS UNCHANGED
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

## Result

Not run yet.
