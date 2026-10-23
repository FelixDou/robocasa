# XR-1 same-scene retry pilot (plan)

**Status: pilot, for sizing the RSS Stage 2 E3 experiment.** It tests nothing that Stage 2
registers.
- E3's sample size (at least about 650 episodes per arm) assumed that a retry after an alarm
  succeeds at the task's ordinary success rate p_k, as if it were a fresh attempt. The mock
  review (concern 5) points out that a retry restarts the *same* scene. If failures
  concentrate on hard scenes, retry success r is lower than p_k and E3 is undersized.
- This pilot estimates r. It was committed before any rollout was collected.

## Design

- **Scenes.**
  - The XR-1 confirmatory collection (`xr1_confirmatory_fused_20261017_160220`): official
    Xiaomi seed protocol, base seed 1300007, `--num-rollouts 500`, 38 tasks, quota 10 + 10,
    all attempts retained.
  - Under that protocol an attempt's seed is `base + task_index × 500 + episode`, and its
    rollout ID is a hash of (task, seed, settings). Re-running the collector with the same
    settings therefore re-creates the same scenes, under the same rollout IDs, in the same
    order.
  - The policy samples its action chunks stochastically, so each re-run is a genuine new
    attempt at the same scene.
  - The collector and the XR-1 policy client are unchanged since the confirmatory commit
    `4e82e5c8`. The only addition is the opt-in `--attempt-limits` option, which stops a task
    after a given number of leading attempts without changing seeds or IDs.
- **Tasks**, by a rule fixed here:
  - Among the 38 tasks, the 10 whose all-retained confirmatory success rate is closest to
    0.5, ties broken by task name.
  - E3 plans "8 to 10 tasks with mid-range success". The atomic/composite mix is reported.
- **Attempts per task.** Every original attempt up to and including the 10th failure, in
  attempt (episode) order.
  - This re-attempts 10 originally failed scenes per task (100 in all), plus the
    originally successful scenes in between, which serve as a control.
  - Expected total: about 200 rollouts.
- **Collector settings:** identical to the confirmatory collection:
  - same server and checkpoint, policy config, `replan_steps` 16, split `pretrain`;
  - SAFE features and actions recorded;
  - videos at 256 × 384 with frame stride 2, so first frames are comparable;
  - no quotas, `--attempt-limits` from the selection file;
  - new ports and a new output root under `/gs/bs`.

## Validity gates (checked before any outcome is read)

1. **Identity.** Every pilot rollout ID exists in the confirmatory manifests, with the same
   task, seed and episode index.
2. **Scene.** The first video frame of each pilot rollout matches the original's: mean
   absolute pixel difference ≤ 2.0 on the 0–255 scale, for at least 95% of pairs.
   - If this fails, the scenes were not reproduced and the pilot is void. It is reported as
     such, and E3 keeps its current size with the caveat.

## Endpoints

- **r_fail:** success rate on retry of originally failed scenes, pooled.
  - Wilson 95% interval, plus a task-cluster bootstrap interval (2,000 replicates, seed 0).
- **r_succ:** success rate on retry of originally successful scenes (control).
- **p:** the selected tasks' confirmatory success rate (task-macro). The independence
  assumption predicts r_fail ≈ r_succ ≈ p.
- **Per-task table:** p_k, retries of failed scenes and their successes, retries of
  successful scenes and their successes.
- **Use:** E3's sample size is recomputed with r = r_fail (and its lower 95% bound as a
  sensitivity) in place of r = p_k, by the Stage 2 sizing code.

## Expectations, written before collecting

- Failures concentrate on hard scenes, so r_fail < p: roughly half of p is plausible.
  r_succ > p.
- If r_fail is close to 0, alarm-triggered reset-and-retry can barely help on XR-1. Then E3's
  primary contrast should be reconsidered: continuation, or retry with a perturbation.

## Runbook

Two GPUs, from a pinned worktree at the commit that contains this plan. The collector
settings are copied from the confirmatory runbook (`xr1_confirmatory_fused_test.md` §4–5),
except: no quotas, `--attempt-limits`, new ports 10120/10121, and a new output root.

```bash
module load miniconda; eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export RP_SHA=<full SHA of the commit containing this runbook>
export ROBOCASA_REPO=$PROJECT_FS/robocasa_retry_pilot_${RP_SHA:0:8}
export XR1_SAFE_REPO=$PROJECT_FS/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe
export XR1_SERVER_ENV=$STORAGE_BS/envs/xiaomi_robotics_1_server XR1_CLIENT_ENV=$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365
export XR1_SAFE_CHECKPOINT=$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe
export HF_HOME=$STORAGE_BS/hf_home TRANSFORMERS_CACHE=$STORAGE_BS/hf_home/transformers WANDB_MODE=disabled WANDB_DISABLED=true
export XR1_POLICY_CONFIG='{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}'
export XR1_CONFIRM_COLL=$STORAGE_BS/robocasa_rollouts/safe/xr1_confirmatory_fused_20261017_160220/confirmatory_collection
export RP_ROOT=$STORAGE_BS/robocasa_rollouts/safe/xr1_retry_pilot_$(date +%Y%m%d_%H%M%S)
export RP_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $RP_ROOT)
mkdir -p $RP_ROOT $RP_LOG; env | grep -E '^(RP_|ROBOCASA_REPO|XR1_CONFIRM_COLL)' > $RP_ROOT/session.env

# Worktree, assets, tests.
git -C $PROJECT_FS/robocasa fetch origin
test -e $ROBOCASA_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $ROBOCASA_REPO $RP_SHA
mkdir -p $ROBOCASA_REPO/robocasa/models/assets
cp -aln $PROJECT_FS/robocasa/robocasa/models/assets/. $ROBOCASA_REPO/robocasa/models/assets/
cd $ROBOCASA_REPO && $XR1_CLIENT_ENV/bin/python -W ignore -m unittest tests.test_xr1_retry_pilot tests.test_safe_atomic_collection 2>&1 | tail -1

# Plan: task selection and attempt limits.
python3 -I scripts/xr1_retry_pilot/plan_pilot.py $XR1_CONFIRM_COLL --output $RP_ROOT

# Servers (GPUs 0 and 1), then the two collectors.
export XR1_PORT0=10120 XR1_PORT1=10121
cd $XR1_SAFE_REPO
CUDA_VISIBLE_DEVICES=0 nohup $XR1_SERVER_ENV/bin/python -u deploy/server.py --model $XR1_SAFE_CHECKPOINT --host 127.0.0.1 --port $XR1_PORT0 > $RP_LOG/server_$XR1_PORT0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $XR1_SERVER_ENV/bin/python -u deploy/server.py --model $XR1_SAFE_CHECKPOINT --host 127.0.0.1 --port $XR1_PORT1 > $RP_LOG/server_$XR1_PORT1.log 2>&1 &
until ss -ltn | grep -q ":$XR1_PORT0\b" && ss -ltn | grep -q ":$XR1_PORT1\b"; do sleep 10; done
cd $ROBOCASA_REPO
for shard in 0 1; do
  if test $shard = 0; then port=$XR1_PORT0; else port=$XR1_PORT1; fi
  mapfile -t tasks < $RP_ROOT/tasks_shard$shard.txt
  CUDA_VISIBLE_DEVICES=$shard MUJOCO_EGL_DEVICE_ID=$shard \
  nohup $XR1_CLIENT_ENV/bin/python -u -m robocasa.recovery.safe.collect_atomic_rollouts \
    --output-dir $RP_ROOT/collection/shard$shard --tasks "${tasks[@]}" \
    --num-rollouts 500 --seed 1300007 --seed-protocol official_xiaomi \
    --attempt-limits $RP_ROOT/attempt_limits.json \
    --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
    --model-family xiaomi_robotics_1 --policy-name Xiaomi-Robotics-1-RoboCasa365 \
    --checkpoint $XR1_SAFE_CHECKPOINT --policy-config "$XR1_POLICY_CONFIG" \
    --host 127.0.0.1 --port $port --split pretrain --replan-steps 16 \
    --record-safe-features --record-actions --record-videos \
    --video-height 256 --video-width 384 --video-frame-stride 2 \
    --no-record-subtask-trace --continue-on-error --max-errors 5 \
    > $RP_LOG/collection_shard$shard.log 2>&1 &
done

# When both collectors have exited: stop the servers, then analyse (gates first).
pkill -f "deploy/server.py.*--port 1012[01]"
$XR1_CLIENT_ENV/bin/python -I scripts/xr1_retry_pilot/analyze_pilot.py $RP_ROOT/collection $XR1_CONFIRM_COLL \
  --selection $RP_ROOT/selection.json --output $RP_ROOT/retry_pilot.json 2>&1 | tee $RP_LOG/analyze.log
```

## Outcome of the registered gates, and a post hoc analysis (2026-10-23)

**Run.** `xr1_retry_pilot_20261023_103019`: 194 of 194 attempts (shard 0: 96, shard 1: 98),
0 errors.
- The limits file was split into one file per shard, with the same values, because the
  collector rejects limits for tasks it is not collecting. This is a runbook detail, not a
  change of design.

**Gate 1 (identity): passed.** Every pilot rollout ID matches its original, with the same
task, seed and episode.

**Gate 2 (scene): failed under the registered rule. The pilot is void as specified.**
- Only 93.8% of first-frame pairs (182 of 194) were within the mean absolute difference of
  2.0; the rule required 95%. No retry outcome was read before this verdict.
- The diagnostic looked at frames only:
  - every pair differs slightly (minimum 0.78, median 1.31, 95th percentile 2.08), which
    looks like rendering or video-compression noise;
  - 10 of the 12 exceedances lie between 2.0 and 4.0, in RinseSinkBasin (6) and
    CloseBlenderLid (4), the two tasks with the highest baseline differences;
  - the other 2 are gross: TurnOffStove episode 17 (47.5) and CloseBlenderLid episode 8
    (22.2). These are different scenes.
  - There is no difference between 4.0 and 22.2. The 2.0 tolerance, chosen before any data,
    was too tight for these renderings.

**Post hoc analysis (added after the gate failed and after the frame diagnostic, before any
outcome was read).**
- Exclude only pairs whose first-frame difference exceeds 10, i.e. the two mismatched
  scenes, and compute the planned endpoints on the remaining 192 pairs
  (`analyze_pilot.py --post-hoc-exclude-above 10`). The output is labelled post hoc.
- Its use is limited to informing the E3 design qualitatively. The RSS protocol states that
  E3's same-scene retry success is measured in Stage 2; nothing is registered from this pilot.
- The two mismatches are themselves a finding: seed replay reproduced 192 of 194 scenes. An
  E3 reset by seed replay therefore needs a scene-identity check, or a reset that restores the
  recorded initial state.
