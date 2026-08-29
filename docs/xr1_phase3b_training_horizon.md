# XR1 Phase 3B training-horizon continuation

This protocol extends the already validated Phase 2 branches without changing
parents, snapshots, candidate seeds, nominal seeds, policy, checkpoint, runtime
bundle, or the original first 64 environment steps. Read
`docs/cluster_experiment_runbook.md` and
`docs/xr1_phase2_complete_snapshot_replay.md` first.

The execution order is fixed:

1. run the four-snapshot engineering sentinel;
2. require every replay-integrity gate to pass;
3. run the complete 20-snapshot screen even if sentinel recovery outcomes are
   all zero;
4. analyze the full result and follow its registered routing decision.

The sentinel is not a scientific stopping screen. It can block the full run
only for provenance, support, replay-prefix, repeat, restore, or artifact
failures.

## Fresh-session exports and preflight

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export XR1_SAFE_REPO="$PROJECT_FS/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe"
export XR1_SERVER_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_server"
export XR1_CLIENT_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365"
export XR1_SAFE_CHECKPOINT="$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe"
export XR1_PHASE2_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase2_shared_policy_2tasks_5each_20260825_014645"
export XR1_PHASE3B_PORT=10306
export XR1_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval"

export HF_HOME="$STORAGE_BS/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

cd "$ROBOCASA_REPO"
git pull --ff-only origin main

test -x "$XR1_SERVER_ENV/bin/python"
test -x "$XR1_CLIENT_ENV/bin/python"
test -f "$XR1_SAFE_REPO/deploy/server.py"
test -f "$XR1_PHASE2_ROOT/plan.json"
test -f "$XR1_PHASE2_ROOT/analysis.json"
test -f "$XR1_PHASE2_ROOT/branch_records.jsonl"
test -f "$XR1_PHASE2_ROOT/parent_records.jsonl"

export XR1_PHASE2_MODEL_PATH="$($XR1_CLIENT_ENV/bin/python - "$XR1_PHASE2_ROOT/plan.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["checkpoint_provenance"]["path"])
PY
)"

if [ ! -d "$XR1_PHASE2_MODEL_PATH" ]; then
  mkdir -p "$(dirname "$XR1_PHASE2_MODEL_PATH")"
  rsync -a --info=progress2 "$XR1_SAFE_CHECKPOINT/" "$XR1_PHASE2_MODEL_PATH/"
fi

df -h "$STORAGE_BS"
nvidia-smi
```

Stop if storage has no safe margin, the frozen source is incomplete, or the
node-local checkpoint cannot be recreated at the exact registered path.

## Start and verify the frozen Xiaomi server

```bash
mkdir -p "$XR1_LOG_ROOT"
export XR1_PHASE3B_SERVER_LOG="$XR1_LOG_ROOT/xr1_phase3b_server_$(date +%Y%m%d_%H%M%S).log"

cd "$XR1_SAFE_REPO"
CUDA_VISIBLE_DEVICES=0 nohup "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
  --model "$XR1_PHASE2_MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$XR1_PHASE3B_PORT" \
  > "$XR1_PHASE3B_SERVER_LOG" 2>&1 &
export XR1_PHASE3B_SERVER_PID=$!

until "$XR1_CLIENT_ENV/bin/python" - "$XR1_PHASE3B_PORT" <<'PY'
import socket, sys
with socket.socket() as connection:
    raise SystemExit(connection.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
do
  tail -n 20 "$XR1_PHASE3B_SERVER_LOG" 2>/dev/null || true
  sleep 10
done

grep -q 'Model loaded' "$XR1_PHASE3B_SERVER_LOG"
ps -fp "$XR1_PHASE3B_SERVER_PID"
ss -ltnp | grep ":$XR1_PHASE3B_PORT"
```

## Register and run the four-snapshot sentinel

The deterministic selector uses the lowest reset-index valid parent for each
task and keeps both its prefix and landmark snapshots. It reuses all seven
original branches per snapshot, for 28 continuations total. ArrangeTea runs to
480 environment steps and CuttingToolSelection to 256.

```bash
export XR1_PHASE3B_SENTINEL="$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_sentinel_$(date +%Y%m%d_%H%M%S)"
export XR1_PHASE3B_SENTINEL_LOG="$XR1_LOG_ROOT/$(basename "$XR1_PHASE3B_SENTINEL").log"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.run_phase3b_training_horizon_continuation \
  --phase2-run-dir "$XR1_PHASE2_ROOT" \
  --output-dir "$XR1_PHASE3B_SENTINEL" \
  --scope sentinel \
  --model-path "$XR1_PHASE2_MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$XR1_PHASE3B_PORT" \
  > "$XR1_PHASE3B_SENTINEL_LOG" 2>&1 &
export XR1_PHASE3B_SENTINEL_PID=$!

printf 'export XR1_PHASE3B_SENTINEL=%q\nexport XR1_PHASE3B_SENTINEL_LOG=%q\nexport XR1_PHASE3B_SENTINEL_PID=%q\n' \
  "$XR1_PHASE3B_SENTINEL" "$XR1_PHASE3B_SENTINEL_LOG" "$XR1_PHASE3B_SENTINEL_PID" \
  > "$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_sentinel_latest.env"
```

Monitor from another shell:

```bash
source "$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_sentinel_latest.env"
while kill -0 "$XR1_PHASE3B_SENTINEL_PID" 2>/dev/null; do
  clear
  date
  "$XR1_CLIENT_ENV/bin/python" -m \
    robocasa.recovery.print_phase3b_training_horizon \
    --run-dir "$XR1_PHASE3B_SENTINEL"
  tail -n 20 "$XR1_PHASE3B_SENTINEL_LOG"
  sleep 30
done

"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.print_phase3b_training_horizon \
  --run-dir "$XR1_PHASE3B_SENTINEL"
```

After engineering validity passes, freeze the sentinel diagnostic analysis:

```bash
"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.analyze_phase3b_training_horizon \
  --run-dir "$XR1_PHASE3B_SENTINEL" \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 0
```

Do not use sentinel completion, headroom, or routing output to cancel the full
screen.

## Register and run all twenty snapshots

```bash
export XR1_PHASE3B_FULL="$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_full20_$(date +%Y%m%d_%H%M%S)"
export XR1_PHASE3B_FULL_LOG="$XR1_LOG_ROOT/$(basename "$XR1_PHASE3B_FULL").log"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.run_phase3b_training_horizon_continuation \
  --phase2-run-dir "$XR1_PHASE2_ROOT" \
  --output-dir "$XR1_PHASE3B_FULL" \
  --scope full \
  --sentinel-run-dir "$XR1_PHASE3B_SENTINEL" \
  --model-path "$XR1_PHASE2_MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$XR1_PHASE3B_PORT" \
  > "$XR1_PHASE3B_FULL_LOG" 2>&1 &
export XR1_PHASE3B_FULL_PID=$!

printf 'export XR1_PHASE3B_FULL=%q\nexport XR1_PHASE3B_FULL_LOG=%q\nexport XR1_PHASE3B_FULL_PID=%q\n' \
  "$XR1_PHASE3B_FULL" "$XR1_PHASE3B_FULL_LOG" "$XR1_PHASE3B_FULL_PID" \
  > "$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_full_latest.env"
```

Use the same monitor with `XR1_PHASE3B_FULL` and `XR1_PHASE3B_FULL_PID`. After
all 140 long-horizon branches pass the engineering audit:

```bash
"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.analyze_phase3b_training_horizon \
  --run-dir "$XR1_PHASE3B_FULL" \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 0
```

The analysis writes `scientific_analysis.json`, snapshot/candidate JSONL, CSV,
task-macro estimands, genuine-replan completion curves, continuation gates,
and one registered routing decision. A critic is unlocked only if every full-
screen critic gate passes; otherwise the next registered route is a fresh
replan or higher-level operator screen.

## Resume

Restart the server, source the latest environment file, and rerun the identical
command with `--resume`. Completed branch IDs are skipped. The registration is
immutable and the runner re-hashes every frozen Phase 2 manifest, snapshot,
payload, runtime, checkpoint, and server entrypoint before continuing. A
first-64 mismatch fails closed and is retained in `errors.jsonl`; use a fresh
output root after any non-transient scientific-integrity failure.
