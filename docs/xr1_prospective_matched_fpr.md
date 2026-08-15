# Xiaomi prospective matched-FPR shadow experiment

This runbook executes the experiment frozen in
`docs/safe_robocasa_experiment_report.md` section 15.6. It uses original
rollout-level SAFE only. It never enables Subtask-SAFE or changes Xiaomi policy
actions in response to an alarm.

The two scientific phases must remain ordered:

1. collect and score calibration rollouts, then freeze the runtime bundle;
2. only afterward collect and score the disjoint prospective shadow test.

## 1. Fresh session and preflight

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_SOURCE_REPO="$PROJECT_FS/robocasa"
export XR1_CODE_BRANCH=codex/safe-xiaomi-robotics-1
export SAFE_REPO="$PROJECT_FS/SAFE"
export XR1_SAFE_REPO="$PROJECT_FS/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe"
export XR1_SERVER_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_server"
export XR1_CLIENT_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365"
export SAFE_ENV="$STORAGE_BS/envs/vla_safe"
export XR1_SAFE_CHECKPOINT="$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe"
export XR1_FINAL_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_pooled_39tasks_final_refits_20260814_231508"

export HF_HOME="$STORAGE_BS/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_ENABLED=0

# Do not alter the shared checkout, which may be running an unrelated branch.
# Resolve the published Xiaomi branch once and create a detached, immutable
# worktree for this experiment.
git -C "$ROBOCASA_SOURCE_REPO" fetch origin "$XR1_CODE_BRANCH"
export XR1_CODE_COMMIT="$(git -C "$ROBOCASA_SOURCE_REPO" rev-parse FETCH_HEAD)"
export ROBOCASA_REPO="$PROJECT_FS/robocasa_xr1_prospective_${XR1_CODE_COMMIT:0:8}"

if test -e "$ROBOCASA_REPO"; then
  test -f "$ROBOCASA_REPO/.git" || {
    echo "STOP: existing worktree path is not a Git checkout: $ROBOCASA_REPO"
    exit 1
  }
else
  git -C "$ROBOCASA_SOURCE_REPO" worktree add \
    --detach "$ROBOCASA_REPO" "$XR1_CODE_COMMIT"
fi

test "$(git -C "$ROBOCASA_REPO" rev-parse HEAD)" = "$XR1_CODE_COMMIT" || {
  echo "STOP: Xiaomi worktree is not at the resolved branch commit"
  exit 1
}

# Git worktrees omit ignored simulator assets. Hard-link the shared asset tree
# without changing or duplicating the source assets.
export XR1_SHARED_ASSETS="$ROBOCASA_SOURCE_REPO/robocasa/models/assets"
export XR1_ASSETS="$ROBOCASA_REPO/robocasa/models/assets"
test -d "$XR1_SHARED_ASSETS" || {
  echo "STOP: shared RoboCasa assets are missing"
  exit 1
}
mkdir -p "$XR1_ASSETS"
cp -aln "$XR1_SHARED_ASSETS/." "$XR1_ASSETS/"

cd "$ROBOCASA_REPO"
echo "XR1 code commit: $(git rev-parse HEAD)"

test -x "$XR1_SERVER_ENV/bin/python"
test -x "$XR1_CLIENT_ENV/bin/python"
test -x "$SAFE_ENV/bin/python"
test -f "$XR1_SAFE_REPO/deploy/server.py"
test -f "$XR1_SAFE_CHECKPOINT/modeling_mibot.py"
test -f "$XR1_FINAL_ROOT/indep_seed0/model_final.ckpt"
test -f "$XR1_FINAL_ROOT/indep_seed1/model_final.ckpt"
test -f "$XR1_FINAL_ROOT/indep_seed2/model_final.ckpt"
test -f robocasa/recovery/safe/score_seen_checkpoint_external.py
test -f robocasa/recovery/safe/run_prospective_matched_fpr.py

grep -n "safe_feature_steps" "$XR1_SAFE_CHECKPOINT/modeling_mibot.py"
grep -n "request_safe_features" "$XR1_SAFE_REPO/deploy/server.py"

df -h "$STORAGE_BS"
df -ih "$STORAGE_BS"
nvidia-smi
```

Create one durable experiment root and freeze the 38-task list from the final
refit rather than reselecting tasks from results:

```bash
export XR1_MATCHED_FPR_TAG=xr1_prospective_matched_fpr_$(date +%Y%m%d_%H%M%S)
export XR1_MATCHED_FPR_ROOT="$STORAGE_BS/robocasa_rollouts/safe/$XR1_MATCHED_FPR_TAG"
export XR1_MATCHED_FPR_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval/$XR1_MATCHED_FPR_TAG"
mkdir -p "$XR1_MATCHED_FPR_ROOT" "$XR1_MATCHED_FPR_LOG_ROOT"

"$XR1_CLIENT_ENV/bin/python" - \
  "$XR1_FINAL_ROOT/indep_seed0/split_manifest.json" \
  "$XR1_MATCHED_FPR_ROOT" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
tasks = sorted(manifest["task_names"])
if len(tasks) != 38 or "PickPlaceSinkToCounter" in tasks:
    raise SystemExit(f"unexpected frozen task set: {len(tasks)} tasks")

# Greedy balancing uses the registered rollout horizons as compute weights.
from robocasa.recovery.safe.atomic_tasks import registered_safe_task_horizons
horizons = registered_safe_task_horizons()
shards = [[], []]
loads = [0, 0]
for task in sorted(tasks, key=lambda name: (-horizons[name], name)):
    shard = min(range(2), key=lambda index: (loads[index], index))
    shards[shard].append(task)
    loads[shard] += horizons[task]

root = pathlib.Path(sys.argv[2])
for index, values in enumerate(shards):
    (root / f"tasks_shard{index}.txt").write_text("\n".join(sorted(values)) + "\n")
(root / "frozen_tasks.json").write_text(json.dumps({
    "tasks": tasks,
    "shards": shards,
    "horizon_loads": loads,
    "source_split_manifest": str(pathlib.Path(sys.argv[1]).resolve()),
}, indent=2, sort_keys=True) + "\n")
print("tasks:", len(tasks))
print("horizon loads:", loads)
print("shard sizes:", [len(values) for values in shards])
PY

printf 'export XR1_CODE_COMMIT=%q\nexport ROBOCASA_REPO=%q\nexport XR1_MATCHED_FPR_TAG=%q\nexport XR1_MATCHED_FPR_ROOT=%q\nexport XR1_MATCHED_FPR_LOG_ROOT=%q\n' \
  "$XR1_CODE_COMMIT" "$ROBOCASA_REPO" "$XR1_MATCHED_FPR_TAG" \
  "$XR1_MATCHED_FPR_ROOT" "$XR1_MATCHED_FPR_LOG_ROOT" \
  > "$STORAGE_BS/robocasa_rollouts/safe/xr1_prospective_matched_fpr_latest.env"
```

Stop if the task count is not exactly 38 or if available storage is below
100 GiB.

## 2. Start and verify two Xiaomi SAFE servers

```bash
export XR1_PORT0=10106
export XR1_PORT1=10107
export XR1_SERVER0_LOG="$XR1_MATCHED_FPR_LOG_ROOT/server_${XR1_PORT0}.log"
export XR1_SERVER1_LOG="$XR1_MATCHED_FPR_LOG_ROOT/server_${XR1_PORT1}.log"

cd "$XR1_SAFE_REPO"
CUDA_VISIBLE_DEVICES=0 \
nohup "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
  --model "$XR1_SAFE_CHECKPOINT" \
  --host 127.0.0.1 \
  --port "$XR1_PORT0" \
  > "$XR1_SERVER0_LOG" 2>&1 &
export XR1_SERVER0_PID=$!

CUDA_VISIBLE_DEVICES=1 \
nohup "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
  --model "$XR1_SAFE_CHECKPOINT" \
  --host 127.0.0.1 \
  --port "$XR1_PORT1" \
  > "$XR1_SERVER1_LOG" 2>&1 &
export XR1_SERVER1_PID=$!

printf 'export XR1_PORT0=%q\nexport XR1_PORT1=%q\nexport XR1_SERVER0_PID=%q\nexport XR1_SERVER1_PID=%q\n' \
  "$XR1_PORT0" "$XR1_PORT1" "$XR1_SERVER0_PID" "$XR1_SERVER1_PID" \
  >> "$STORAGE_BS/robocasa_rollouts/safe/xr1_prospective_matched_fpr_latest.env"

until ss -ltn | grep -q ":$XR1_PORT0" && \
      ss -ltn | grep -q ":$XR1_PORT1"; do
  date
  tail -n 20 "$XR1_SERVER0_LOG" 2>/dev/null
  tail -n 20 "$XR1_SERVER1_LOG" 2>/dev/null
  sleep 10
done

grep -q "Model loaded" "$XR1_SERVER0_LOG"
grep -q "Model loaded" "$XR1_SERVER1_LOG"
ss -ltnp | grep -E ":$XR1_PORT0|:$XR1_PORT1"
nvidia-smi
```

Run one short rollout through each server before collection:

```bash
export XR1_SMOKE_ROOT="$XR1_MATCHED_FPR_ROOT/server_smoke"
mkdir -p "$XR1_SMOKE_ROOT"
cd "$ROBOCASA_REPO"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_SMOKE_ROOT/server0" \
  --tasks CloseBlenderLid \
  --num-rollouts 1 \
  --seed 490007 \
  --seed-protocol official_xiaomi \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 \
  --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 --port "$XR1_PORT0" \
  --split pretrain --horizon 20 --replan-steps 16 \
  --record-safe-features --record-actions --no-record-videos \
  --no-record-subtask-trace --max-errors 1

CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_SMOKE_ROOT/server1" \
  --tasks TurnOnMicrowave \
  --num-rollouts 1 \
  --seed 490008 \
  --seed-protocol official_xiaomi \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 \
  --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 --port "$XR1_PORT1" \
  --split pretrain --horizon 20 --replan-steps 16 \
  --record-safe-features --record-actions --no-record-videos \
  --no-record-subtask-trace --max-errors 1

for shard in server0 server1; do
  "$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.validate_atomic_dataset \
    --dataset-dir "$XR1_SMOKE_ROOT/$shard"
done
```

Both validators must print `RoboCasa SAFE dataset: VALID` and `Official SAFE
loader compatible: True`.

## 3. Calibration collection: exactly 3/3 primary, keep all overshoot

```bash
mapfile -t XR1_TASKS0 < "$XR1_MATCHED_FPR_ROOT/tasks_shard0.txt"
mapfile -t XR1_TASKS1 < "$XR1_MATCHED_FPR_ROOT/tasks_shard1.txt"
export XR1_CAL_ROOT="$XR1_MATCHED_FPR_ROOT/calibration_collection"
mkdir -p "$XR1_CAL_ROOT/shard0" "$XR1_CAL_ROOT/shard1"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_CAL_ROOT/shard0" \
  --tasks "${XR1_TASKS0[@]}" \
  --num-rollouts 250 --seed 500007 --seed-protocol official_xiaomi \
  --success-quota 3 --failure-quota 3 --no-retain-only-quota \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 --port "$XR1_PORT0" --split pretrain --replan-steps 16 \
  --record-safe-features --record-actions --record-videos \
  --video-height 256 --video-width 384 --video-frame-stride 2 \
  --no-record-subtask-trace --continue-on-error --max-errors 5 \
  > "$XR1_MATCHED_FPR_LOG_ROOT/calibration_shard0.log" 2>&1 &
export XR1_CAL_PID0=$!

CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_CAL_ROOT/shard1" \
  --tasks "${XR1_TASKS1[@]}" \
  --num-rollouts 250 --seed 500007 --seed-protocol official_xiaomi \
  --success-quota 3 --failure-quota 3 --no-retain-only-quota \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 --port "$XR1_PORT1" --split pretrain --replan-steps 16 \
  --record-safe-features --record-actions --record-videos \
  --video-height 256 --video-width 384 --video-frame-stride 2 \
  --no-record-subtask-trace --continue-on-error --max-errors 5 \
  > "$XR1_MATCHED_FPR_LOG_ROOT/calibration_shard1.log" 2>&1 &
export XR1_CAL_PID1=$!

printf 'export XR1_CAL_ROOT=%q\nexport XR1_CAL_PID0=%q\nexport XR1_CAL_PID1=%q\n' \
  "$XR1_CAL_ROOT" "$XR1_CAL_PID0" "$XR1_CAL_PID1" \
  >> "$STORAGE_BS/robocasa_rollouts/safe/xr1_prospective_matched_fpr_latest.env"
```

One-shot monitoring command (rerun it; it does not trap the terminal in
`watch`):

```bash
"$XR1_CLIENT_ENV/bin/python" - "$XR1_CAL_ROOT" 3 <<'PY'
import collections
import json
import pathlib
import sys

root, target = pathlib.Path(sys.argv[1]), int(sys.argv[2])
counts = collections.defaultdict(lambda: [0, 0])
errors = 0
for shard in (root / "shard0", root / "shard1"):
    manifest = shard / "manifest.jsonl"
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                counts[row["task_name"]][int(not row["failed"])] += 1
    error_path = shard / "errors.jsonl"
    if error_path.is_file():
        errors += sum(bool(line.strip()) for line in error_path.read_text().splitlines())
print(f"{'TASK':32s} {'SUCCESS':>8s} {'FAILURE':>8s} {'READY':>7s}")
for task, (failures, successes) in sorted(counts.items()):
    print(f"{task:32s} {successes:8d} {failures:8d} {str(min(successes, failures) >= target):>7s}")
print("tasks seen:", len(counts), "errors:", errors)
print("ready:", sum(min(values) >= target for values in counts.values()), "/ 38")
PY

ps -fp "$XR1_CAL_PID0" "$XR1_CAL_PID1"
tail -n 30 "$XR1_MATCHED_FPR_LOG_ROOT/calibration_shard0.log"
tail -n 30 "$XR1_MATCHED_FPR_LOG_ROOT/calibration_shard1.log"
du -sh "$XR1_CAL_ROOT"
```

## 4. Merge, export, score, and freeze calibration

Only continue when all 38 tasks have at least three successes and three
failures and both collector processes have stopped normally.

```bash
for shard in shard0 shard1; do
  "$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.validate_atomic_dataset \
    --dataset-dir "$XR1_CAL_ROOT/$shard"
done

export XR1_CAL_MERGED="$XR1_CAL_ROOT/merged_all_retained"
export XR1_CAL_EXPORT="$XR1_CAL_ROOT/official_safe_all_retained"

cd "$ROBOCASA_REPO"
"$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.merge_atomic_datasets \
  --source-dirs "$XR1_CAL_ROOT/shard0" "$XR1_CAL_ROOT/shard1" \
  --output-dir "$XR1_CAL_MERGED"

"$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$XR1_CAL_MERGED"

"$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir "$XR1_CAL_MERGED" \
  --output-dir "$XR1_CAL_EXPORT"

export XR1_CAL_SCORE_ROOT="$XR1_CAL_ROOT/frozen_mlp_scores"
mkdir -p "$XR1_CAL_SCORE_ROOT"

for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 "$SAFE_ENV/bin/python" -u -m \
    robocasa.recovery.safe.score_seen_checkpoint_external \
    --export-dir "$XR1_CAL_EXPORT" \
    --safe-repo "$SAFE_REPO" \
    --training-run-dir "$XR1_FINAL_ROOT/indep_seed$seed" \
    --output-dir "$XR1_CAL_SCORE_ROOT/seed$seed" \
    --group calibration \
    --device cuda
done

export XR1_CAL_ANALYSIS="$STORAGE_BS/robocasa_checkpoints/safe/${XR1_MATCHED_FPR_TAG}_calibration"
"$SAFE_ENV/bin/python" -u -m \
  robocasa.recovery.safe.run_prospective_matched_fpr calibrate \
  --final-root "$XR1_FINAL_ROOT" \
  --calibration-score-root "$XR1_CAL_SCORE_ROOT" \
  --output-dir "$XR1_CAL_ANALYSIS" \
  --seeds 0 1 2 \
  --target-fpr 0.05 \
  --safe-end 0.25 \
  --time-start 0.50 \
  --min-per-class 3

export XR1_RUNTIME_BUNDLE="$XR1_CAL_ANALYSIS/runtime_bundle.json"
test -f "$XR1_RUNTIME_BUNDLE"
"$SAFE_ENV/bin/python" - "$XR1_RUNTIME_BUNDLE" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("phase:", d["phase"])
print("tasks:", len(d["task_horizons"]))
print("primary calibration:", d["calibration_counts"]["primary_balanced"])
for detector, selection in d["threshold_selection"].items():
    print(detector, selection["thresholds"], selection["validation_metrics"])
PY

printf 'export XR1_CAL_EXPORT=%q\nexport XR1_CAL_SCORE_ROOT=%q\nexport XR1_CAL_ANALYSIS=%q\nexport XR1_RUNTIME_BUNDLE=%q\n' \
  "$XR1_CAL_EXPORT" "$XR1_CAL_SCORE_ROOT" "$XR1_CAL_ANALYSIS" "$XR1_RUNTIME_BUNDLE" \
  >> "$STORAGE_BS/robocasa_rollouts/safe/xr1_prospective_matched_fpr_latest.env"
```

Do not collect the prospective test unless this phase reports 228 primary
calibration rollouts, 38 tasks, and event-level validation FPR no greater than
0.05 for every selected detector.

## 5. Prospective shadow collection: exactly 10/10 primary, keep all overshoot

Use the same two verified servers, but the disjoint base seed 700007:

```bash
mapfile -t XR1_TASKS0 < "$XR1_MATCHED_FPR_ROOT/tasks_shard0.txt"
mapfile -t XR1_TASKS1 < "$XR1_MATCHED_FPR_ROOT/tasks_shard1.txt"
export XR1_TEST_ROOT="$XR1_MATCHED_FPR_ROOT/prospective_test_collection"
mkdir -p "$XR1_TEST_ROOT/shard0" "$XR1_TEST_ROOT/shard1"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_TEST_ROOT/shard0" \
  --tasks "${XR1_TASKS0[@]}" \
  --num-rollouts 500 --seed 700007 --seed-protocol official_xiaomi \
  --success-quota 10 --failure-quota 10 --no-retain-only-quota \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 --port "$XR1_PORT0" --split pretrain --replan-steps 16 \
  --record-safe-features --record-actions --record-videos \
  --video-height 256 --video-width 384 --video-frame-stride 2 \
  --no-record-subtask-trace --continue-on-error --max-errors 5 \
  > "$XR1_MATCHED_FPR_LOG_ROOT/prospective_test_shard0.log" 2>&1 &
export XR1_TEST_PID0=$!

CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_TEST_ROOT/shard1" \
  --tasks "${XR1_TASKS1[@]}" \
  --num-rollouts 500 --seed 700007 --seed-protocol official_xiaomi \
  --success-quota 10 --failure-quota 10 --no-retain-only-quota \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 --port "$XR1_PORT1" --split pretrain --replan-steps 16 \
  --record-safe-features --record-actions --record-videos \
  --video-height 256 --video-width 384 --video-frame-stride 2 \
  --no-record-subtask-trace --continue-on-error --max-errors 5 \
  > "$XR1_MATCHED_FPR_LOG_ROOT/prospective_test_shard1.log" 2>&1 &
export XR1_TEST_PID1=$!

printf 'export XR1_TEST_ROOT=%q\nexport XR1_TEST_PID0=%q\nexport XR1_TEST_PID1=%q\n' \
  "$XR1_TEST_ROOT" "$XR1_TEST_PID0" "$XR1_TEST_PID1" \
  >> "$STORAGE_BS/robocasa_rollouts/safe/xr1_prospective_matched_fpr_latest.env"
```

Use the section 3 one-shot monitor with `"$XR1_TEST_ROOT" 10`.

## 6. Merge, score with frozen copies, and evaluate without retuning

```bash
for shard in shard0 shard1; do
  "$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.validate_atomic_dataset \
    --dataset-dir "$XR1_TEST_ROOT/$shard"
done

export XR1_TEST_MERGED="$XR1_TEST_ROOT/merged_all_retained"
export XR1_TEST_EXPORT="$XR1_TEST_ROOT/official_safe_all_retained"

cd "$ROBOCASA_REPO"
"$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.merge_atomic_datasets \
  --source-dirs "$XR1_TEST_ROOT/shard0" "$XR1_TEST_ROOT/shard1" \
  --output-dir "$XR1_TEST_MERGED"

"$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$XR1_TEST_MERGED"

"$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir "$XR1_TEST_MERGED" \
  --output-dir "$XR1_TEST_EXPORT"

export XR1_TEST_SCORE_ROOT="$XR1_TEST_ROOT/frozen_mlp_scores"
mkdir -p "$XR1_TEST_SCORE_ROOT"

for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 "$SAFE_ENV/bin/python" -u -m \
    robocasa.recovery.safe.score_seen_checkpoint_external \
    --export-dir "$XR1_TEST_EXPORT" \
    --safe-repo "$SAFE_REPO" \
    --training-run-dir "$XR1_CAL_ANALYSIS/runtime/seed$seed" \
    --output-dir "$XR1_TEST_SCORE_ROOT/seed$seed" \
    --group prospective_test \
    --device cuda
done

export XR1_TEST_ANALYSIS="$STORAGE_BS/robocasa_checkpoints/safe/${XR1_MATCHED_FPR_TAG}_prospective_test"
"$SAFE_ENV/bin/python" -u -m \
  robocasa.recovery.safe.run_prospective_matched_fpr evaluate \
  --runtime-bundle "$XR1_RUNTIME_BUNDLE" \
  --evaluation-score-root "$XR1_TEST_SCORE_ROOT" \
  --output-dir "$XR1_TEST_ANALYSIS" \
  --test-per-class 10 \
  --bootstrap-samples 2000 \
  --bootstrap-seed 0
```

Compact result audit:

```bash
"$SAFE_ENV/bin/python" - "$XR1_TEST_ANALYSIS/analysis.json" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("status:", d["status"])
print("counts:", d["counts"]["primary_balanced"])
for detector, row in d["overall"].items():
    print(detector)
    print("  ROC/AP:", row["roc_auc"], row["average_precision"])
    print("  TPR/FPR:", row["true_positive_rate"], row["false_positive_rate"])
    print("  Wilson TPR/FPR:", row["tpr_wilson_95"], row["fpr_wilson_95"])
    print("  adjusted detection:", row["missed_failure_adjusted_detection_fraction"])
    print("  recall by landmark:", row["failure_recall_by_landmark"])
print("paired staged minus time:", d["paired_staged_minus_time"])
print("bootstrap:", d["task_rollout_bootstrap"])
print("FPR drift:", d["calibration_to_test_fpr_drift"])
print("success criteria:", d["success_criteria"])
PY
```

The primary result is `counts.primary_balanced`; `all_retained_secondary`
contains every retained overshoot rollout under the same frozen thresholds.
Because collection stops when both outcome quotas are reached, this secondary
stream documents stopping cost and sensitivity but is not an unbiased estimate
of deployment prevalence.
Do not tune or rerun thresholds based on either result.
