# When do XR-1 failures become hopeless? Point-of-no-return labeling (plan)

**Status: exploratory, on the confirmatory cohort.** That cohort is already open for lead
time (report §15.13).
- Purpose: preliminary evidence for E5. Every failure runs to the time limit, but the
  failure may have been decided much earlier. If it was, lead relative to that moment is
  the target a monitor should be judged on, and the timer's late alarm is a real limitation.
- This plan, with its definitions and expectations, was committed before any video was
  labeled.

## Sample

- **Frame:** 48 failures from the primary confirmatory test cohort (380 failures).
  - 2 per atomic task (10 tasks, 20 videos);
  - 1 per composite task (28 tasks, 28 videos).
- **Selection:** uniformly at random within each task, with a fixed seed (0), by
  `scripts/xr1_lead_time/no_return_sample.py`.
- **Blinding:** the labeler sees only the video, the task name and its instruction, and the
  time limit. Detector alarm times go to a separate key file, which is opened only after
  every label is saved. Videos carry anonymous IDs in random order.

## What is labeled (per video)

- **t_e, first visible error:** the first moment the robot does something that deviates from
  a successful execution (a wrong grasp, a slip, a collision, heading to the wrong object),
  even if recovery is still possible.
- **t_u, point of no return:** the earliest moment after which, in the labeler's judgment,
  the policy cannot complete the task before the time limit without outside intervention.
  Examples:
  - the object dropped out of reach or off the counter;
  - an object knocked over or wedged so that it cannot be re-grasped;
  - an irreversible wrong action (wrong item placed in a closed container);
  - the gripper or arm stuck against the scene for good.
- **Recoverable until timeout.** If the robot stays in a recoverable state until the end
  (still trying, slow, oscillating near the goal), check this box; then t_u = end of video.
- **Category**, one of:
  - dropped or lost object;
  - knocked over or displaced scene object;
  - wrong object or wrong target;
  - stuck or collision;
  - never made progress or wandered;
  - slow or nearly succeeded;
  - other, with a note.
- **Confidence:** high, medium or low, for t_u.

Times are marked on the video timeline and converted with the recorded settings: frame
index × video frame stride (2) = environment step, and environment step / 16 = inference.

## Analysis (`scripts/xr1_lead_time/no_return_analysis.py`)

All measures use the inference index and the fraction of the time limit H_k.

1. **Distribution of t_u / H_k**, overall and by task type.
   - The fraction of failures hopeless by 25%, 50% and 75% of H_k.
   - The fraction recoverable until timeout.
2. **Same for t_e / H_k**, and the gap t_u − t_e.
3. **Alarm timing against t_u**, for P and SAFE + P (q = 0.95) fitted exactly as in E1 on the
   E1 calibration cohort.
   - **Wasted time:** T_alarm − t_u as a fraction of H_k.
   - **Pre-emptive alarms:** the fraction of failures alarmed at or before t_u.
   - **Usable lead:** H_k − max(T_alarm, t_u), the part of the remaining time after the
     failure is decided that a retry could still use.
4. **Robustness:** results restricted to high-confidence labels.
5. **Optional reliability check:** a second labeler on 12 of the 48 videos (6 atomic,
   6 composite). Report the median absolute difference in t_u / H_k.

Every number is descriptive. With 48 videos, quantiles are reported with a bootstrap over
videos (2,000 replicates, seed 0).

## Expectations, written before labeling

- The majority of failures are decided well before the limit: median t_u / H_k ≤ 0.6.
- Atomic failures are decided earlier than composite ones. Atomic tasks are short and often
  fail by a drop or a missed grasp; composite tasks often fail by slowly running out of time.
- Both P and SAFE + P alarm after t_u for most failures, with wasted time around
  0.2–0.4 H_k. SAFE + P reduces wasted time mainly on atomic tasks.
- At least 15% of failures are "recoverable until timeout" or "slow / nearly succeeded".
  For those, the timer is the right monitor.

## Runbook

**1. On the cluster: sample and bundle the videos.** Use a pinned worktree at the commit
containing this plan, CPU only.

```bash
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export NR_SHA=<full SHA>
export NR_REPO=$PROJECT_FS/robocasa_no_return_${NR_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin
test -e $NR_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $NR_REPO $NR_SHA
export E1_OUT=$STORAGE_BS/robocasa_rollouts/safe/xr1_lead_calibration_20261020_001108/lead_analysis
export XR1_CONFIRM_ROOT=$STORAGE_BS/robocasa_rollouts/safe/xr1_confirmatory_fused_20261017_160220
export XR1_RUNTIME_BUNDLE=$STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_calibration/runtime_bundle.json
export NR_OUT=$STORAGE_BS/robocasa_rollouts/safe/xr1_no_return_labeling
cd $NR_REPO
CUDA_VISIBLE_DEVICES="" $SAFE_PY -I -u scripts/xr1_lead_time/no_return_sample.py \
  --calibration $E1_OUT/calibration_predictions.jsonl --test $E1_OUT/test_predictions.jsonl \
  --runtime-bundle $XR1_RUNTIME_BUNDLE --collection-root $XR1_CONFIRM_ROOT/confirmatory_collection \
  --output $NR_OUT
du -sh $NR_OUT/bundle; ls $NR_OUT
```

**2. Copy only the bundle (not `key/`) to the Mac.**

```bash
rsync -av <cluster-login>:/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_no_return_labeling/bundle/ ~/Desktop/xr1_no_return_bundle/
```

**3. Label.** Open `scripts/xr1_lead_time/no_return_labeler.html` in a browser and choose the
bundle folder.
- Labels autosave in the browser.
- **Export labels** downloads `no_return_labels.json`.

**4. Analyse,** only after every video is labeled. Copy `no_return_labels.json` next to the
key on the cluster (or copy the key to the Mac), then run:

```bash
$SAFE_PY -I scripts/xr1_lead_time/no_return_analysis.py --labels no_return_labels.json \
  --key $NR_OUT/key/key.json --output $NR_OUT/no_return_results.json
```
