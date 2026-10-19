#!/usr/bin/env bash
# XR-1 stage-gated re-prompt trial: plan | smoke | start | resume | status | analyze.
# One worker = one XR-1 server GPU + one simulator GPU + one unique port; seeds
# are split round-robin over workers and every arm of a seed runs on one worker.
# See docs/stage_gated_reprompt.md.
set -euo pipefail
action=${1:-status}
case "$action" in plan|smoke|start|resume|status|analyze|worker) ;; *)
  echo "Usage: $0 {plan|smoke|start|resume|status|analyze}"; exit 1 ;; esac

export STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
export ROBOCASA_REPO=${ROBOCASA_REPO:-/gs/fs/tga-shinoda/felid/robocasa}
export XR1_CLIENT_ENV=${XR1_CLIENT_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365}
export XR1_SERVER_ENV=${XR1_SERVER_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_server}
export XR1_SAFE_REPO=${XR1_SAFE_REPO:-/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe}
export XR1_SAFE_CHECKPOINT=${XR1_SAFE_CHECKPOINT:-$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe}
export GATE_MODEL=${GATE_MODEL:-$XR1_SAFE_CHECKPOINT}
export GATE_CONFIG=${GATE_CONFIG:-$ROBOCASA_REPO/scripts/stage_gated_reprompt_xr1.json}
export GATE_RUN_PREFIX=${GATE_RUN_PREFIX:-xr1_stage_gate}
# Space-separated "simulator_gpu:server_gpu" pairs; one worker (seed shard) per pair.
export GATE_GPU_PAIRS=${GATE_GPU_PAIRS:-"0:1 2:3"}
# Distinct from the recoverability ladder's 10340 so both can run at once.
export GATE_PORT_BASE=${GATE_PORT_BASE:-10360}
# Smoke seeds are disjoint from the trial's base seed 400007.
export GATE_SMOKE_BASE_SEED=${GATE_SMOKE_BASE_SEED:-499007}
latest="$STORAGE_BS/robocasa_rollouts/recovery/xr1_stage_gate_latest.env"
recovery_root="$STORAGE_BS/robocasa_rollouts/recovery"
log_dir="$STORAGE_BS/robocasa_logs/eval"
py="$XR1_CLIENT_ENV/bin/python"
module=robocasa.recovery.stage_gated_reprompt

if [[ "$action" == plan ]]; then
  cd "$ROBOCASA_REPO"
  extra=()
  [[ -z "${GATE_CALIBRATION_DIR:-}" ]] || extra+=(--calibration-dir "$GATE_CALIBRATION_DIR")
  pairs=($GATE_GPU_PAIRS)
  exec "$py" -m "$module" plan --config "$GATE_CONFIG" --num-workers "${#pairs[@]}" ${extra[@]+"${extra[@]}"}
fi

if [[ "$action" == status || "$action" == analyze ]]; then
  [[ -n "${GATE_ROOT:-}" ]] || source "$latest"
  cd "$ROBOCASA_REPO"
  if [[ "$action" == analyze ]]; then
    # shellcheck disable=SC2206
    runs=(${GATE_ANALYZE_RUNS:-"$GATE_ROOT"/shard*})
    exec "$py" -m "$module" analyze --run-dirs "${runs[@]}" --output-dir "${GATE_ANALYSIS_DIR:-$GATE_ROOT/analysis}"
  fi
  echo "run: $GATE_ROOT"
  for shard in "$GATE_ROOT"/shard*; do
    echo "== $(basename "$shard")"
    test -f "$shard/status.json" && cat "$shard/status.json" || echo "no status yet"
    test -f "$shard/errors.jsonl" && echo "errors: $(wc -l < "$shard/errors.jsonl")"
    test -f "$shard/worker_failure.json" && { echo "WORKER FAILED:"; cat "$shard/worker_failure.json"; }
  done
  ls -1 "$log_dir"/"$(basename "$GATE_ROOT")"_* 2>/dev/null | tail -8
  exit 0
fi

if [[ "$action" == worker ]]; then
  shard=$2; nshards=$3; gpu=$4; server_gpu=$5; port=$6; shift 6
  out="$GATE_ROOT/shard$shard"
  mkdir -p "$out"
  exec 9>"$out/.worker.lock"
  flock -n 9 || { echo "Another worker owns $out"; exit 1; }
  server_pid=""
  cleanup() {
    local code=$?
    trap - EXIT
    if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
    if [[ "$code" != 0 ]]; then
      printf '{"exit_code": %d, "worker_log": "%s"}\n' "$code" "${GATE_WORKER_LOG:-unknown}" > "$out/worker_failure.json"
    fi
    exit "$code"
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  "$py" - "$STORAGE_BS" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 20 * 2**30:
    raise SystemExit("Less than 20 GiB free on durable storage; inspect before running")
PY
  if ss -ltn | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "Port $port is occupied; choose another GATE_PORT_BASE"; exit 1
  fi
  server_log="$log_dir/$(basename "$GATE_ROOT")_server${shard}_$(date +%Y%m%d_%H%M%S).log"
  (
    export CUDA_VISIBLE_DEVICES="$server_gpu"
    cd "$XR1_SAFE_REPO"
    exec "$XR1_SERVER_ENV/bin/python" -u deploy/server.py --model "$GATE_MODEL" --host 127.0.0.1 --port "$port"
  ) > "$server_log" 2>&1 &
  server_pid=$!
  echo "server_pid=$server_pid server_log=$server_log"
  deadline=$((SECONDS + 1800))
  until "$py" - "$port" <<'PY'
import socket, sys
with socket.socket() as connection:
    raise SystemExit(connection.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
  do
    kill -0 "$server_pid" 2>/dev/null || { tail -60 "$server_log"; exit 1; }
    (( SECONDS < deadline )) || { echo "Server startup timed out"; tail -60 "$server_log"; exit 1; }
    sleep 10
  done
  grep -q 'Model loaded' "$server_log" || { echo "Missing model-ready message"; tail -60 "$server_log"; exit 1; }
  cd "$ROBOCASA_REPO"
  # robosuite requires the EGL device id to be one of the CUDA_VISIBLE_DEVICES ids.
  export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu"
  "$py" -u -m "$module" collect --config "$GATE_CONFIG" --output-dir "$out" \
    --shard-index "$shard" --num-shards "$nshards" --host 127.0.0.1 --port "$port" \
    --policy-arg "model_path=$GATE_MODEL" --policy-arg "policy_checkpoint=$GATE_MODEL" "$@"
  exit 0
fi

# smoke | start | resume
test -x "$py" && test -x "$XR1_SERVER_ENV/bin/python"
test -f "$XR1_SAFE_REPO/deploy/server.py" && test -f "$GATE_CONFIG"
grep -q 'input_data.pop("sampling_seed"' "$XR1_SAFE_REPO/deploy/server.py" || {
  echo "XR-1 server lacks the sampling_seed patch (docs/safe_xiaomi_robotics_1.md)"; exit 1; }
export HF_HOME="$STORAGE_BS/hf_home" TRANSFORMERS_CACHE="$STORAGE_BS/hf_home/transformers"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
extra=()
pairs=($GATE_GPU_PAIRS)
stamp=$(date +%Y%m%d_%H%M%S)
if [[ "$action" == smoke ]]; then
  export GATE_ROOT=${GATE_ROOT:-$recovery_root/${GATE_RUN_PREFIX}_smoke_$stamp}
  [[ ! -e "$GATE_ROOT" ]] || { echo "GATE_ROOT exists; use resume or a new root"; exit 1; }
  pairs=("${pairs[0]}")
  extra=(--tasks ArrangeTea --seeds-per-task 2 --base-seed "$GATE_SMOKE_BASE_SEED")
elif [[ "$action" == start ]]; then
  export GATE_ROOT=${GATE_ROOT:-$recovery_root/${GATE_RUN_PREFIX}_$stamp}
  [[ ! -e "$GATE_ROOT" ]] || { echo "GATE_ROOT exists; use resume or a new root"; exit 1; }
else
  [[ -n "${GATE_ROOT:-}" ]] || source "$latest"
  test -d "$GATE_ROOT" || { echo "Missing $GATE_ROOT"; exit 1; }
  # Resume must use the same number of workers: seeds are sharded by worker count.
  nprev=$(ls -d "$GATE_ROOT"/shard* 2>/dev/null | wc -l)
  [[ "$nprev" == "${#pairs[@]}" ]] || { echo "GATE_GPU_PAIRS has ${#pairs[@]} pairs but $GATE_ROOT has $nprev shards"; exit 1; }
  for shard in "$GATE_ROOT"/shard*; do rm -f "$shard/worker_failure.json"; done
  extra=(--resume)
  if [[ -f "$GATE_ROOT/collect_args.txt" ]]; then
    mapfile -t saved < "$GATE_ROOT/collect_args.txt"
    extra+=("${saved[@]}")
  fi
fi
case "$GATE_ROOT" in "$STORAGE_BS"/*) ;; *) echo "GATE_ROOT must be under $STORAGE_BS"; exit 1 ;; esac
mkdir -p "$GATE_ROOT" "$log_dir"
if [[ "$action" == smoke ]]; then
  printf '%s\n' ${extra[@]+"${extra[@]}"} > "$GATE_ROOT/collect_args.txt"
fi
for name in STORAGE_BS ROBOCASA_REPO GATE_ROOT GATE_CONFIG GATE_MODEL GATE_GPU_PAIRS GATE_PORT_BASE GATE_RUN_PREFIX XR1_CLIENT_ENV XR1_SERVER_ENV XR1_SAFE_REPO; do
  printf 'export %s=%q\n' "$name" "${!name}"
done > "$GATE_ROOT/run.env"
cp "$GATE_ROOT/run.env" "$latest"
git -C "$ROBOCASA_REPO" rev-parse HEAD > "$GATE_ROOT/git_commit.txt"
nshards=${#pairs[@]}
for shard in "${!pairs[@]}"; do
  gpu=${pairs[$shard]%%:*}; server_gpu=${pairs[$shard]##*:}
  [[ "$gpu" != "$server_gpu" ]] || { echo "Use separate simulator and server GPUs ($gpu)"; exit 1; }
  port=$((GATE_PORT_BASE + shard))
  export GATE_WORKER_LOG="$log_dir/$(basename "$GATE_ROOT")_shard${shard}_$(date +%Y%m%d_%H%M%S).log"
  nohup bash "$ROBOCASA_REPO/scripts/run_stage_gated_reprompt_xr1.sh" worker \
    "$shard" "$nshards" "$gpu" "$server_gpu" "$port" ${extra[@]+"${extra[@]}"} > "$GATE_WORKER_LOG" 2>&1 &
  echo "shard=$shard pid=$! sim_gpu=$gpu server_gpu=$server_gpu port=$port log=$GATE_WORKER_LOG"
done
echo "GATE_ROOT=$GATE_ROOT"
