# Cross-task (unseen-task) evaluation on XR-1: pre-specified interpretation plan

**Date:** 2026-10-27. This plan, the fold assignment and the code are committed **before**
any detector is trained on a subset of the XR-1 tasks. No cross-task score, timer fit,
alarm or metric exists yet.

**Status: exploratory, pre-specified.** Both cohorts used here were opened by earlier
analyses (E1, Amendment 2, dose-response). The rules below fix how the results will be read,
but the analysis cannot confirm anything. It does not change the E1 or Amendment 2 results.
If it suggests an effect, the confirmatory test belongs in Stage 2 on fresh rollouts.

## Question

Hide-and-Seek (Park et al., NeurIPS 2026) and SAFE report detectors on **seen** tasks and
on **unseen** tasks. Their unseen protocol holds out tasks from detector training and tuning,
deploys zero-shot, and calibrates thresholds on seen-task successes. Our comparison so far
covers seen tasks only, where the per-task timer P is available. P needs successful rollouts
of each task. On a task the monitor never saw, with no rollouts of it, the only timer left is
a **global timer G**: one fraction of the time limit H_k for every task. G is much weaker than
P. On XR-1, P leads the global timer by about 0.10 of H_k (E1, descriptive). On RLDX-1, P's
lead is 0.271 against the global timer's 0.032 (report §15.15).

Three questions:
1. **Zero-shot.** On held-out tasks, does Hide-and-Seek, trained without them, add warning
   lead over G at the same false-alarm budget?
2. **Few-shot.** If the deployer has n successful rollouts of the new task, a per-task timer
   P_n can be built. How fast does P_n remove the monitor's gain?
3. **Transfer.** The seen-task gain of Hide-and-Seek over P (+0.071, Amendment 2) could be
   partly a learned per-task duration. How much of the gain over G survives on unseen tasks?

"Unseen" means unseen by the **detector**, as in Hide-and-Seek. The policy (XR-1) is the same
released checkpoint throughout. Whether the policy itself was trained on a task is not varied
here. That is a different meaning (RoboCasa365's `composite_unseen` split).

## What has already been seen on these cohorts

- **E1 (report §15.13).** P, SAFE + P and, as secondaries, the global timer fitted by joint
  search on calibration (not by Proposition 1), SAFE + global timer and the time risk.
- **Amendment 2 (§15.21).** Hide-and-Seek and its first-inference, zero-input and shuffle
  controls, all trained on all 38 tasks, each fused with P.
- **Dose-response (§15.14).** Frozen success-only baselines attached to both cohorts.
- **Point of no return (§15.17).** 48 confirmatory failure videos labeled.
- **Not seen.** No detector trained on a subset of tasks. No timer built from n successes. No
  monitor fused with G under Proposition 1. No score with pooled normalization.

## Data

- **Training rollouts:** the 1,824 final-refit training rollouts (`split == "train"` in
  `xr1_pooled_39tasks_final_refits_20260814_231508/indep_seed0/scores.jsonl`), in their file
  order. For fold f, the training rollouts of fold f's tasks are dropped.
- **Calibration:** the E1 lead-calibration cohort, primary 10 successes and 10 failures per task
  (380/380, seeds [1400007, 1425007)).
- **Test:** the E1 confirmatory test cohort, primary 380/380 (seeds [1300007, 1325007)).
- **Development success durations** for P and P_n: the runtime bundle's time-risk active counts,
  as in `quantile_timer.dev_fractions` (6 to 45 successes per task, median 27).

## Folds

Four folds, stratified by task type. Within atomic tasks and then within composite tasks:
sort names, permute with `numpy.random.default_rng(20261027)`, deal round-robin into folds
0–3. The result is committed as
[`scripts/xr1_lead_time/cross_task_folds.json`](../scripts/xr1_lead_time/cross_task_folds.json).
Each task is held out exactly once, so every test rollout is scored by the one fold that never
saw its task. Four folds hold out 24–26% of tasks per fold, close to Hide-and-Seek's 30%, while
keeping at least 28 tasks for training and calibration.

| Fold | Held-out atomic | Held-out composite |
|---|---|---|
| 0 | CloseBlenderLid, CloseToasterOvenDoor, SlideDishwasherRack | BreadSelection, GarnishPancake, GetToastedBread, KettleBoiling, PrepareCoffee, SetUpCuttingStation, StoreLeftoversInBowl |
| 1 | CoffeeSetupMug, PickPlaceCounterToStove, TurnOnSinkFaucet | CategorizeCondiments, LoadDishwasher, MakeIceLemonade, ScrubCuttingBoard, SearingMeat, SteamInMicrowave, StirVegetables |
| 2 | TurnOffStove, TurnOnMicrowave | ArrangeBreadBasket, CuttingToolSelection, PortionHotDogs, PreSoakPan, RecycleBottlesByType, WashFruitColander, WeighIngredients |
| 3 | OpenCabinet, PickPlaceCounterToCabinet | ArrangeTea, DeliverStraw, PackIdenticalLunches, RinseSinkBasin, StackBowlsCabinet, WaffleReheat, WashLettuce |

A fifth run, **fold all**, holds out nothing. It gives the seen reference and the
reproduction check.

## Detectors

Code: `robocasa/recovery/safe/baselines/detectors.py`, unchanged, at the commit that contains
this file. Settings are Amendment 2's frozen settings:
- PCA to 128 dimensions;
- LSTM hidden size 128;
- margin 0.2, onset weight 1.0;
- AdamW, learning rate 1e-3, weight decay 1e-4, batch 32;
- 200 epochs, seeds 0, 1 and 2, CPU;
- feature aggregation `last`.

| Detector | Role |
|---|---|
| `hide_and_seek` | **primary monitor** |
| `hide_and_seek_zero` | control: same LSTM on zero inputs, so a learned elapsed-time signal |
| `hide_and_seek_first` | control: the rollout's first-inference features repeated, so a difficulty prior |
| `cosine_knn` | success-only monitor (no failure labels), descriptive |

**Not included, with reasons:**
- **SAFE-MLP.** Retraining it per fold needs the external official SAFE trainer. Its seen-task
  gain over P was below the margin (+0.030, E1), so it is not the monitor whose transfer is in
  question.
- **The same-task shuffle control.** It needs training rollouts of the scored task, so it is
  undefined on a held-out task. The scorer refuses it for any fold except fold all.
- **ActProbe, Mahalanobis, PCA-KMeans, RND.** Cosine kNN stands for the success-only family.
  ActProbe reduces to chunk magnitude on XR-1.

**Scoring** (`robocasa.recovery.safe.baselines.score_cross_task`, one call per fold). It trains
each detector on the fold's training rollouts and writes every cohort rollout's raw per-seed
trajectory. It computes no threshold, alarm or metric.

**Normalization.** A zero-shot monitor has no statistics for a task it has never seen. Each
fold therefore stores a **pooled** normalization: one location (median) and scale
(max of 1.4826·MAD, SD, 1e-6) over the training maxima of all seen-task training rollouts.
This is `fit_safe_normalization`'s formula, pooled over tasks. The score is the running maximum,
normalized with the pooled statistics of each seed, then averaged over seeds. Every monitor in
every arm below uses the pooled normalization, except the per-task reproduction check.

## Timers and fusion

All timers and fused detectors use Proposition 1 as registered in E1 (`confirm_lead.fit_fast`):
- conformal margin on calibration successes, alpha = 0.05;
- the nested OR path (u, s), chosen on calibration failures;
- u over the 0, 0.05, …, 1 quantiles of the failures' maximum scores;
- s in {0, 1, 2, 5, 10, 20, 50}.

| Timer | rho_k (fraction of H_k) |
|---|---|
| **G** (global) | 0 for every task. The conformal margin then makes G one fraction of H_k, set on calibration successes. |
| **P_n**, n = 1, 3, 5 | 0.95 quantile of n development successes of task k, drawn without replacement (all of them if fewer). The same rule applies to seen and held-out tasks, so the margin is calibrated for the noise of an n-success estimate. |
| **P** | 0.95 quantile of all development successes of task k (the registered P) |

**Unseen arm (per fold, then pooled).** For fold f:
1. Fit the timer and the fused path on the calibration rollouts of fold f's **seen** tasks only.
2. Apply them to the test rollouts of fold f's **held-out** tasks.

Leads and false alarms are pooled over the four folds. The pooled test set is the whole test
cohort (380 failures, 380 successes), each rollout scored by a model that never saw its task.

**Seen reference (fold all).** The same scores, timers and fusion with every task in training,
calibration and test.

**Endpoint.** The mean warning lead (H_k − T)/H_k over pooled test failures, with a miss counted
as 0. A gain is the mean over failures of (fused lead − timer lead), with both fitted on the same
calibration rows.

**Decision bootstrap.** 2,000 replicates, seed 0. Each replicate:
1. draws the 38 tasks with replacement once;
2. sends each drawn task's calibration rollouts to the folds that see it and its test rollouts
   to the fold that holds it out;
3. resamples rollouts within each task and class;
4. refits every timer and fused path in every fold.

P_n draws its n successes afresh in each replicate, so its CIs include that noise. Point
estimates for P_n average 20 draws (seed 2). Contrasts are per-replicate differences on the same
replicates. This is `confirm_hide_and_seek.boot_multi`'s design, extended to folds.

## Integrity checks (no result without them)

- **I1 (folds).** `cross_task_lead.py folds` on both E1 prediction files must print `FOLDS OK`.
  The fold file must equal the rule above for the cohorts' task types.
- **I2 (training reproduction).** Fold all must reproduce the Amendment 2 raw per-seed
  trajectories of Hide-and-Seek, the zero-input control and the first-inference control on
  both E1 cohorts exactly (largest difference ≤ 1e-12; six `reference ... -> OK` lines and
  `REFERENCE OK`). This shows that the fold code trains the registered detectors.
- **I3 (analysis reproduction).** Fold all with per-task normalization, fused with P, must equal
  `confirm_hide_and_seek.json` to its stored rounding:
  - P's mean lead and false alarms;
  - each of the three detectors' test mean lead, false alarms and gain.

  The per-task normalization must equal Amendment 2's `normalizations.json`. If I3 fails, the
  analysis writes only the checks and exits with status 1.
- **I4 (inputs).** Checksums of the frozen inputs are unchanged after the run.
- **Coverage.** Each fold's training tasks are exactly its seen tasks. Held-out flags match the
  fold file. Every primary rollout has a finite trajectory of the right length in every fold.

## Rules

**U1 (primary): zero-shot gain over G.** Hide-and-Seek + G over G, unseen arm. It is
**supported** if all of these hold (Amendment 1's criteria):
1. the decision-bootstrap 95% CI of the gain lies above 0;
2. the point gain is at least 0.05 H_k;
3. the pooled unseen test FPR of Hide-and-Seek + G is at most 0.075;
4. its Wilson 95% upper bound is at most 0.10.

**Null sanity for U1.** Two uninformative scores are fitted exactly like Hide-and-Seek + G in
the unseen arm, 20 draws each, seed 1:
- the running maximum of i.i.d. N(0, 1);
- within-task resampled fold Hide-and-Seek: per-seed raw scores drawn from all inferences of
  the same task in the same cohort, then pooled-normalized.

If either null's mean gain over G is 0.05 or more, U1 is reported as **not interpretable**,
whatever its value. A local check on the opened August timing data (SAFE scores standing in for
a detector, 5 draws) gave +0.0005 for noise and +0.009 for resampled scores, so the margin
should be meaningful.

**U2: learned elapsed-time signal.** (Hide-and-Seek + G) − (zero-input + G), unseen arm.
- Rule: the gain is **beyond a learned elapsed-time signal** if its decision CI lies above 0.
- This control matters more here than on seen tasks. With pooled normalization, the zero-input
  LSTM is a learned timer in absolute inferences, which G (a fraction of H_k) does not have.

**U3: difficulty prior.** (Hide-and-Seek + G) − (first-inference + G), unseen arm. The rule:
the warning **develops during the rollout** if its decision CI lies above 0.

**U4: transfer loss.** The zero-shot gain over G minus the seen-reference gain over G (fold all,
pooled normalization). Both are measured on the same test rollouts.
- Rule: **transfer loss detected** if the decision CI lies below 0.
- Otherwise the result is reported as no loss detected, not as equivalence.

**U5: few-shot curve.** For n = 1, 3, 5 and all, the following are reported with decision CIs:
- the gain of Hide-and-Seek + P_n over P_n (monitor zero-shot, timer from n successes);
- P_n's lead minus G's lead (how fast the timer catches up);
- (Hide-and-Seek + P_n) − (Hide-and-Seek + G).

Reading: the **crossover** is the smallest n whose point gain is below 0.05 H_k, or "not
reached". Also reported: the n = all gain minus the Amendment 2-style seen gain (fold all,
per-task normalization, + P), with its CI.

**U6: success-only monitor (descriptive).** Cosine kNN + G over G, unseen arm, and
Hide-and-Seek − kNN, with decision CIs.

**U7: does the FPR calibration transfer?** The FPR guarantee holds for tasks exchangeable with
the calibration tasks, and held-out tasks need not be.
- Reported: pooled unseen test FPR of G and Hide-and-Seek + G, and per-fold FPRs.
- Rule: **calibration transfers** if both pooled Wilson upper bounds are at most 0.10.

**Descriptive only:**
- the atomic and composite strata;
- per-fold leads and paths;
- the test-only bootstrap of the U1 gain;
- G tuned on the test cohort to Hide-and-Seek + G's test FPR, which favors the timer;
- the seen reference with P (pooled and per-task normalization).

## Expected outcome, written before scoring

- **G will be far below P.** On the opened August test cohort (local stand-in, timers only) G
  led by 0.093 and P by 0.197. On the E1 cohort, P led by 0.225 (§15.13).
- **Zero-shot Hide-and-Seek.** Earlier transfer was weak. SAFE on π0 RoboCasa rollouts had an
  unseen-task ROC-AUC of 0.45–0.58 against about 0.79 seen (report §4.3). Hide-and-Seek's own
  unseen balanced accuracy is up to 14 points below seen (VLABench), and slightly above seen on
  two of five splits. But G is weak, and a monitor needs little task knowledge to beat it. No
  direction is predicted for U1.
- **U2 is the one to watch.** On seen tasks, the zero-input control matched Hide-and-Seek's
  ROC-AUC (time explains separation). On unseen tasks, a learned absolute-time signal may gain
  over G by itself. No direction is predicted.
- **Few-shot.** P needs only a 95th percentile of success durations, so P_n at n = 3–5 likely
  recovers most of P's lead over G. The crossover is expected at small n, but not predicted.
- **FPR.** Pooled over folds, unseen-task FPR may exceed 0.05, because tasks differ in duration
  spread. No direction is predicted.

## How the outcome will be reported

| Outcome | Paper statement (labeled exploratory) |
|---|---|
| U1 and U2 supported | On tasks the monitor never saw and with no timing data for them, Hide-and-Seek adds at least 0.05 H_k of warning over a global timer, beyond a learned elapsed-time signal. Report the few-shot crossover n. |
| U1 supported, U2 not | The zero-shot gain over a global timer is matched by the zero-input control. It is generic elapsed-time information, not evidence from the policy's features. |
| U1 not supported, FPR within limits | Zero-shot, Hide-and-Seek does not add a margin's worth of warning even over the weakest timer. |
| U1 fails on FPR (U7 not transferring) | Calibrated on other tasks, neither the timer nor the fused detector keeps its false-alarm rate on new tasks. Report the fold FPRs. |
| Null sanity fails | U1 is not interpretable at this margin. Report the values only. |

In every case, the paper gets a Seen/Unseen table in Hide-and-Seek's layout:
- **Seen:** fold all, + G and + P.
- **Unseen:** pooled folds, + G and + P_n.

A Stage 2 confirmatory item is proposed only if U1 and U2 are both supported.

## Stop conditions

Any of these stops the run. Do not edit inputs or code to make it pass:
- I1, I2 or I3 failing;
- a checksum change in the frozen inputs;
- a coverage, overlap or held-out-flag error;
- `Disk quota exceeded`.

## Runbook

Run on TSUBAME from a pinned worktree, CPU only, following
[`cluster_experiment_runbook.md`](cluster_experiment_runbook.md). Never pull in the shared
checkout. `XT_SHA` is the full SHA of the pushed commit that contains this file, on branch
`claude/xr1-unseen-tasks`.

### 0. Environment, pinned code, inputs, unit tests

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
# Amendment 2 paths: AMEND2_ROOT, E1_OUT, XR1_LCAL_ROOT, XR1_CONFIRM_ROOT, XR1_RAW_DIRS,
# XR1_RUNTIME_BUNDLE, XR1_TRAIN_IDS.
source $STORAGE_BS/robocasa_logs/eval/xr1_amend2_hide_and_seek_20261022_214209/paths.env

export XT_SHA=<XT_SHA>
export XT_REPO=$PROJECT_FS/robocasa_xtask_${XT_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/xr1-unseen-tasks
test -e $XT_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $XT_REPO $XT_SHA
test "$(git -C $XT_REPO rev-parse HEAD)" = "$XT_SHA" || echo "STOP: worktree HEAD is not XT_SHA"
cd $XT_REPO

export XT_TAG=xr1_cross_task_$(date +%Y%m%d_%H%M%S)
export XT_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/$XT_TAG
export XT_LOG=$STORAGE_BS/robocasa_logs/eval/$XT_TAG
mkdir -p $XT_ROOT $XT_LOG
XT_INPUTS=($E1_OUT/calibration_predictions.jsonl $E1_OUT/test_predictions.jsonl $XR1_RUNTIME_BUNDLE
           $XR1_TRAIN_IDS $AMEND2_ROOT/confirm_hide_and_seek.json $AMEND2_ROOT/scores/normalizations.json
           $AMEND2_ROOT/scores/lead_calibration_predictions.jsonl
           $AMEND2_ROOT/scores/confirmatory_test_predictions.jsonl
           $XT_REPO/scripts/xr1_lead_time/cross_task_folds.json)
for f in "${XT_INPUTS[@]}"; do test -s $f && echo "OK $f" || echo "STOP: missing $f"; done
for d in "${XR1_RAW_DIRS[@]}" $XR1_LCAL_ROOT/merged_all_retained $XR1_CONFIRM_ROOT/confirmatory_collection/merged_all_retained; do
  test -f $d/manifest.jsonl && echo "raw OK: $d" || echo "STOP: no manifest in $d"
done
sha256sum "${XT_INPUTS[@]}" | tee $XT_LOG/frozen_inputs.sha256
printf 'export %s=%q\n' XT_SHA "$XT_SHA" XT_REPO "$XT_REPO" XT_ROOT "$XT_ROOT" XT_LOG "$XT_LOG" > $XT_LOG/paths.env

PYTHONPATH=$XT_REPO "$SAFE_PY" -m unittest tests.test_safe_score_cross_task tests.test_xr1_cross_task_lead \
  tests.test_safe_score_cohorts tests.test_xr1_confirm_hide_and_seek
```

The unit tests must all pass, including the Hide-and-Seek reproduction test (it needs PyTorch,
so it is skipped on machines without it).

### 1. Folds (I1)

```bash
cd $XT_REPO
for f in $E1_OUT/calibration_predictions.jsonl $E1_OUT/test_predictions.jsonl; do
  "$SAFE_PY" -I scripts/xr1_lead_time/cross_task_lead.py folds --predictions $f
done 2>&1 | tee $XT_LOG/folds.log
```

It must print `FOLDS OK` twice.

### 2. Score the five folds (no metrics; several hours on CPU)

Each fold runs as its own process with 8 threads, as Amendment 2's scoring did. Five at once
need about 40 cores. Check `nproc` and `free -g` first. With fewer cores, start fold all first
and the others as cores free up.

```bash
cd $XT_REPO
nproc; free -g
XT_ARGS=(
  --dataset-dir "${XR1_RAW_DIRS[@]}" $XR1_LCAL_ROOT/merged_all_retained $XR1_CONFIRM_ROOT/confirmatory_collection/merged_all_retained
  --runtime-bundle $XR1_RUNTIME_BUNDLE --train-ids $XR1_TRAIN_IDS
  --folds $XT_REPO/scripts/xr1_lead_time/cross_task_folds.json
  --cohort lead_calibration $E1_OUT/calibration_predictions.jsonl
  --cohort confirmatory_test $E1_OUT/test_predictions.jsonl
  --detectors hide_and_seek hide_and_seek_zero hide_and_seek_first cosine_knn
  --feature-aggregation last --seeds 0 1 2
)
for F in all 0 1 2 3; do
  EXTRA=()
  if [ "$F" = all ]; then
    EXTRA=(--reference-raw lead_calibration $AMEND2_ROOT/scores/lead_calibration_predictions.jsonl
           --reference-raw confirmatory_test $AMEND2_ROOT/scores/confirmatory_test_predictions.jsonl)
  fi
  PYTHONPATH=$XT_REPO CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup nice -n 10 \
    "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.score_cross_task "${XT_ARGS[@]}" --fold $F "${EXTRA[@]}" \
    --output-dir $XT_ROOT/fold_$F > $XT_LOG/score_fold_$F.log 2>&1 &
  echo "fold $F PID $! on $(hostname)"
done
```

Monitor with `tail -n 2 $XT_LOG/score_fold_*.log`. Fold all must end with six
`reference ... -> OK` lines and `REFERENCE OK` (I2). Otherwise stop.

### 3. The analysis (about 20 minutes)

```bash
cd $XT_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup nice -n 10 \
  "$SAFE_PY" -I -u scripts/xr1_lead_time/cross_task_lead.py run \
  --calibration $E1_OUT/calibration_predictions.jsonl --test $E1_OUT/test_predictions.jsonl \
  --fold-dirs $XT_ROOT/fold_0 $XT_ROOT/fold_1 $XT_ROOT/fold_2 $XT_ROOT/fold_3 $XT_ROOT/fold_all \
  --runtime-bundle $XR1_RUNTIME_BUNDLE --amend2-result $AMEND2_ROOT/confirm_hide_and_seek.json \
  --amend2-normalizations $AMEND2_ROOT/scores/normalizations.json \
  --output $XT_ROOT/cross_task_lead.json > $XT_LOG/analysis.log 2>&1 &
echo "analysis PID $!"
```

When it ends:

```bash
tail -n 70 $XT_LOG/analysis.log
sha256sum -c $XT_LOG/frozen_inputs.sha256 && echo INPUTS UNCHANGED
```

Report the `Amendment 2 reproduction` line, the per-detector and few-shot lines, the JSON
block, the final `U1:` line and `INPUTS UNCHANGED`.

## Result

Recorded 2026-10-28. Nothing above this section was changed after scoring.

**Run.**
- Folder: `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_cross_task_20261027_212141`.
- Logs: `/gs/bs/tga-shinoda/felid/robocasa_logs/eval/xr1_cross_task_20261027_212141`.
- Code: every fold scored at `152fbb86`, the commit of this plan.
- Output: `cross_task_lead.json` (2,000 decision-bootstrap replicates).

**Integrity checks.** All passed.
- **I1:** `FOLDS OK` on both cohorts. Each fold dropped exactly 48 training rollouts per held-out
  task (1,344 or 1,392 left; 1,824 for fold all).
- **I2:** fold all's raw trajectories equal Amendment 2's for all three Hide-and-Seek variants on
  both cohorts, with max difference 0.0.
- **I3:** the Amendment 2 table is reproduced exactly: P 0.2252; Hide-and-Seek + P 0.2961, gain
  +0.0708; zero-input +0.0121; first-inference +0.0010; normalization difference 0.0.
- **I4:** `INPUTS UNCHANGED`.
- Every fold has the full 6,088 + 6,104 rows (4 detectors × 1,522 and × 1,526 rollouts).

**Operational restarts.** No code, input or argument changed. All runs used `152fbb86`.
- **What ran out of memory:** with `last` aggregation, `load_baseline_rollout` kept each
  rollout's whole raw feature tensor alive (a NumPy view). Each process held 111–126 GB, and the
  allocation's limit is estimated at 340–460 GB.
- **First launch (all five folds at once):** fold all and fold 0 were killed by the memory limit
  while loading data. They wrote nothing.
- **Early relaunch of all and 0:** this pushed memory over the limit again, and fold 3 was killed
  after `hide_and_seek seed 0`. Its partial output is kept as `fold_3.oomkilled/`. The two
  relaunched folds were stopped by hand.
- **Final runs:** fold 3 was rerun from scratch. Fold 0 started when folds 1 and 2 had exited, and
  fold all when fold 3 had exited.
- **Logs:** the killed and stopped attempts keep their logs as `*.oomkilled.log` and
  `*.stopped_early.log`.
- **Later fix:** the loader now copies the slice (`3c5b6bba`, committed after scoring and used by
  no fold here). It was verified on 2026-10-28 (`loader_copy_check_20261028_130325`): fold all
  re-scored with the fix equals this run's fold all bit for bit, with max difference 0.0 for all
  four detectors on both cohorts. Peak memory was 5.0 GB instead of about 126 GB, and the run time
  was unchanged (8,178 s vs 8,007 s).

**Primary and secondary rules.** All gains are in units of H_k. Brackets are decision-bootstrap
95% CIs.

| Rule | Result | Verdict |
|---|---|---|
| **U1** Hide-and-Seek + G over G, unseen | lead 0.253 vs 0.126, gain **+0.127 [0.069, 0.211]**; FPR 0.047 (18 FP), Wilson ≤ 0.074 | **supported** (all 4 criteria) |
| Null sanity | noise +0.005 mean (max +0.019); resampled H&S +0.003 (max +0.021) | interpretable |
| **U2** − zero-input + G | +0.094 [0.038, 0.165] (both at 18 FP) | beyond a learned elapsed-time signal |
| **U3** − first-inference + G | +0.127 [0.067, 0.210]; the first-inference control adds 0.000 over G | develops during the rollout |
| **U4** unseen − seen gain over G | 0.127 − 0.122 = +0.005 [−0.048, 0.100] | no transfer loss detected (not equivalence) |
| **U5** few-shot crossover | gain over P_n ≥ 0.068 for every n | not reached |
| **U6** cosine kNN + G over G | +0.011 [−0.002, 0.066]; H&S − kNN +0.116 [0.042, 0.188] | descriptive |
| **U7** FPR on held-out tasks | G 0.024 (≤ 0.044); H&S + G 0.047 (≤ 0.074) | calibration transfers |

The zero-input control alone gains +0.033 [0.0025, 0.087] over G: a learned elapsed-time signal,
below the margin.

**Seen/Unseen table (Hide-and-Seek's layout).** Mean warning lead (FPR), with the gain over the
timer in the same row.

| Monitor | Seen (fold all) | Unseen (pooled folds) |
|---|---|---|
| G | 0.123 (0.021) | 0.126 (0.024) |
| Hide-and-Seek + G | 0.245 (0.047), +0.122 [0.022, 0.208] | 0.253 (0.047), +0.127 [0.069, 0.211] |
| P (seen) / P_n, n = all (unseen) | 0.225 (0.055) | 0.228 (0.058) |
| Hide-and-Seek + P / + P_all | 0.296 (0.061), +0.071 [0.022, 0.160] | 0.296 (0.063), +0.068 [0.048, 0.160] |

Seen + P uses per-task normalization (Amendment 2). With pooled normalization the seen gain over
P is +0.064, at FPR 0.071 (Wilson upper bound 0.101).

**Few-shot (U5).** P_n is the mean over 20 draws of n development successes per task.

| n | P_n lead (FPR) | H&S + P_n lead (FPR) | gain over P_n | P_n − G | (H&S + P_n) − (H&S + G) |
|---|---|---|---|---|---|
| 1 | 0.169 (0.041) | 0.268 (0.052) | +0.099 [0.047, 0.188] | +0.043 [−0.038, 0.118] | +0.014 [−0.028, 0.061] |
| 3 | 0.186 (0.048) | 0.284 (0.061) | +0.098 [0.047, 0.189] | +0.060 [−0.030, 0.131] | +0.030 [−0.016, 0.072] |
| 5 | 0.201 (0.052) | 0.289 (0.064) | +0.088 [0.045, 0.189] | +0.075 [−0.025, 0.141] | +0.035 [−0.013, 0.081] |
| all | 0.228 (0.058) | 0.296 (0.063) | +0.068 [0.048, 0.160] | +0.102 [0.007, 0.147] | +0.043 [0.003, 0.087] |

Gain at n = all minus the seen gain over P (per-task normalization): −0.003 [−0.077, 0.101].

**Descriptive.**
- **Matched false alarms (favors the timer).** G uses 9 FP; Hide-and-Seek + G uses 18. G tuned on
  the test cohort to 18 FP leads by 0.203. Against it, Hide-and-Seek's gain is
  **+0.051 [−0.002, 0.104]** (test-only bootstrap). Part of the 0.127 comes from G leaving its
  false-alarm budget unused, because the Proposition 1 margin makes it conservative. At equal
  false alarms, the gain over a global timer is about the margin and not clearly above 0.
- **Task type.** Atomic: 100 failures, gain +0.043 (0.124 → 0.167). Composite: 280 failures,
  gain +0.158 (0.127 → 0.284). This is against G, so it is not comparable to the seen-task strata
  against P (§15.13).
- **Per fold** (held-out tasks only):

  | Fold | Tasks | G lead (FPR) | H&S + G lead (FPR) | Gain | Zero-input gain | kNN gain |
  |---|---|---|---|---|---|---|
  | 0 | 10 | 0.169 (0.050) | 0.223 (0.040) | +0.054 | 0.000 | +0.015 |
  | 1 | 10 | 0.111 (0.010) | 0.175 (0.030) | +0.064 | +0.017 | +0.005 |
  | 2 | 9 | 0.110 (0.033) | 0.365 (0.100) | +0.255 | +0.077 | +0.026 |
  | 3 | 9 | 0.110 (0.000) | 0.262 (0.022) | +0.152 | +0.045 | −0.001 |

  - All four folds gain at least +0.054.
  - Fold 2 has the largest gain, and also an FPR of 0.10 (9 of the 18 pooled false alarms; Wilson
    up to 0.179).
  - In fold 3, the fitted path caps the timer at the time limit (`lam` 1.109 ≥ 1), so all warning
    comes from the score. 33 of its 90 failures are flagged only at the final inference, with no
    lead. The seen fold all path does the same (`lam` 1.092): with pooled normalization, the fit
    drops G.
- **Test-only bootstrap of U1:** +0.127 [0.073, 0.184].

**Reading, by the table written before scoring.** This is the row "U1 and U2 supported":

> On tasks the monitor never saw and with no timing data for them, Hide-and-Seek adds at least
> 0.05 H_k of warning over a global timer, beyond a learned elapsed-time signal. The few-shot
> crossover is not reached: even with every development success of the new task, Hide-and-Seek
> trained without the task adds +0.068 over the per-task timer, the same as its seen-task gain
> (+0.071).

Caveats that go with that statement:
- **Exploratory only:** both cohorts were already opened.
- **Matched false alarms:** with G tuned to the same 18 false alarms, the gain is about the margin
  (+0.051, CI touching 0).
- **No loss is not equivalence:** U4's CI is wide.
- **Fold 2:** it carries the largest gain at FPR 0.10.

**Stage 2.** By the plan's rule, a Stage 2 confirmatory item is proposed: zero-shot Hide-and-Seek
on held-out tasks. Given the false-alarm caveat, its primary comparison should be against the
per-task timer from n successes (H&S + P_n over P_n), where the false-alarm rates nearly match.
The comparison against G would be secondary, reported together with the matched-FPR comparison.
