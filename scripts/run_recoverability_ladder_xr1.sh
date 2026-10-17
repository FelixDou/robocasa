#!/usr/bin/env bash
# XR-1 recoverability ladder (schema 2):
#   plan | smoke | naturals | audit | calibrate-stages | start | resume | status | analyze.
# One worker = one XR-1 server GPU + one simulator GPU + one unique port.
# See docs/recoverability_ladder.md.
set -euo pipefail
action=${1:-status}
case "$action" in plan|smoke|naturals|audit|calibrate-stages|start|resume|status|analyze|worker) ;; *)
  echo "Usage: $0 {plan|smoke|naturals|audit|calibrate-stages|start|resume|status|analyze}"; exit 1 ;; esac

export STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
export ROBOCASA_REPO=${ROBOCASA_REPO:-/gs/fs/tga-shinoda/felid/robocasa}
export XR1_CLIENT_ENV=${XR1_CLIENT_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365}
export XR1_SERVER_ENV=${XR1_SERVER_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_server}
export XR1_SAFE_REPO=${XR1_SAFE_REPO:-/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe}
export XR1_SAFE_CHECKPOINT=${XR1_SAFE_CHECKPOINT:-$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe}
export LADDER_MODEL=${LADDER_MODEL:-$XR1_SAFE_CHECKPOINT}
export LADDER_CONFIG=${LADDER_CONFIG:-$ROBOCASA_REPO/scripts/recoverability_ladder_xr1_pilot.json}
export LADDER_RUN_PREFIX=${LADDER_RUN_PREFIX:-xr1_ladder_v2}
# Space-separated "simulator_gpu:server_gpu" pairs; one worker (task shard) per pair.
export LADDER_GPU_PAIRS=${LADDER_GPU_PAIRS:-"0:1"}
export LADDER_PORT_BASE=${LADDER_PORT_BASE:-10340}
# Natural-parent (no-branch) phase that feeds audit-predicates and calibrate-stages.
export LADDER_NATURAL_PARENTS=${LADDER_NATURAL_PARENTS:-12}
export LADDER_NATURAL_BASE_SEED=${LADDER_NATURAL_BASE_SEED:-320007}
latest="$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_latest.env"
recovery_root="$STORAGE_BS/robocasa_rollouts/recovery"
log_dir="$STORAGE_BS/robocasa_logs/eval"
py="$XR1_CLIENT_ENV/bin/python"
module=robocasa.recovery.recoverability_ladder

if [[ "$action" == plan ]]; then
  cd "$ROBOCASA_REPO"
  extra=()
  [[ -z "${LADDER_CALIBRATION_DIR:-}" ]] || extra+=(--calibration-dir "$LADDER_CALIBRATION_DIR")
  # shellcheck disable=SC2206
  [[ -z "${LADDER_REFERENCE_RUNS:-}" ]] || { refs=($LADDER_REFERENCE_RUNS); extra+=(--reference-run "${refs[@]}"); }
  pairs=($LADDER_GPU_PAIRS)
  exec "$py" -m "$module" plan --config "$LADDER_CONFIG" --num-workers "${#pairs[@]}" "${extra[@]}"
fi

if [[ "$action" == status || "$action" == analyze || "$action" == audit || "$action" == calibrate-stages ]]; then
  [[ -n "${LADDER_ROOT:-}" ]] || source "$latest"
  cd "$ROBOCASA_REPO"
  if [[ "$action" == analyze ]]; then
    # shellcheck disable=SC2206
    runs=(${LADDER_ANALYZE_RUNS:-"$LADDER_ROOT"/shard*})
    extra=()
    [[ -z "${LADDER_SCORES_JSONL:-}" ]] || extra+=(--scores-jsonl "$LADDER_SCORES_JSONL")
    exec "$py" -m "$module" analyze --run-dirs "${runs[@]}" --output-dir "${LADDER_ANALYSIS_DIR:-$LADDER_ROOT/analysis}" "${extra[@]}"
  fi
  if [[ "$action" == audit ]]; then
    # shellcheck disable=SC2206
    runs=(${LADDER_AUDIT_RUNS:-"$LADDER_ROOT"/shard*})
    exec "$py" -m "$module" audit-predicates --run-dirs "${runs[@]}" --config "$LADDER_CONFIG" \
      --output-dir "${LADDER_AUDIT_DIR:-$LADDER_ROOT/predicate_audit}"
  fi
  if [[ "$action" == calibrate-stages ]]; then
    # shellcheck disable=SC2206
    runs=(${LADDER_CALIBRATION_RUNS:-"$LADDER_ROOT"/shard*})
    exec "$py" -m "$module" calibrate-stages --run-dirs "${runs[@]}" --config "$LADDER_CONFIG" \
      --output "${LADDER_STAGE_CALIBRATION:-$LADDER_ROOT/stage_calibration.json}"
  fi
  echo "run: $LADDER_ROOT"
  for shard in "$LADDER_ROOT"/shard*; do
    echo "== $(basename "$shard")"
    test -f "$shard/status.json" && cat "$shard/status.json" || echo "no status yet"
    test -f "$shard/errors.jsonl" && echo "errors: $(wc -l < "$shard/errors.jsonl")"
    test -f "$shard/worker_failure.json" && { echo "WORKER FAILED:"; cat "$shard/worker_failure.json"; }
  done
  ls -1 "$log_dir"/"$(basename "$LADDER_ROOT")"_* 2>/dev/null | tail -8
  exit 0
fi

if [[ "$action" == worker ]]; then
  shard=$2; nshards=$3; gpu=$4; server_gpu=$5; port=$6; shift 6
  out="$LADDER_ROOT/shard$shard"
  mkdir -p "$out"
  exec 9>"$out/.worker.lock"
  flock -n 9 || { echo "Another worker owns $out"; exit 1; }
  server_pid=""
  cleanup() {
    local code=$?
    trap - EXIT
    if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
    if [[ "$code" != 0 ]]; then
      printf '{"exit_code": %d, "worker_log": "%s"}\n' "$code" "${LADDER_WORKER_LOG:-unknown}" > "$out/worker_failure.json"
    fi
    exit "$code"
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  "$py" - "$STORAGE_BS" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 50 * 2**30:
    raise SystemExit("Less than 50 GiB free on durable storage; inspect before running")
PY
  if ss -ltn | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "Port $port is occupied; choose another LADDER_PORT_BASE"; exit 1
  fi
  server_log="$log_dir/$(basename "$LADDER_ROOT")_server${shard}_$(date +%Y%m%d_%H%M%S).log"
  (
    export CUDA_VISIBLE_DEVICES="$server_gpu"
    cd "$XR1_SAFE_REPO"
    exec "$XR1_SERVER_ENV/bin/python" -u deploy/server.py --model "$LADDER_MODEL" --host 127.0.0.1 --port "$port"
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
  export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID=0
  "$py" -u -m "$module" collect --config "$LADDER_CONFIG" --output-dir "$out" \
    --shard-index "$shard" --num-shards "$nshards" --host 127.0.0.1 --port "$port" \
    --policy-arg "model_path=$LADDER_MODEL" --policy-arg "policy_checkpoint=$LADDER_MODEL" "$@"
  exit 0
fi

# smoke | naturals | start | resume
test -x "$py" && test -x "$XR1_SERVER_ENV/bin/python"
test -f "$XR1_SAFE_REPO/deploy/server.py" && test -f "$LADDER_CONFIG"
grep -q 'input_data.pop("sampling_seed"' "$XR1_SAFE_REPO/deploy/server.py" || {
  echo "XR-1 server lacks the sampling_seed patch (docs/safe_xiaomi_robotics_1.md)"; exit 1; }
export HF_HOME="$STORAGE_BS/hf_home" TRANSFORMERS_CACHE="$STORAGE_BS/hf_home/transformers"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
extra=()
pairs=($LADDER_GPU_PAIRS)
stamp=$(date +%Y%m%d_%H%M%S)
if [[ "$action" == smoke ]]; then
  export LADDER_ROOT=${LADDER_ROOT:-$recovery_root/${LADDER_RUN_PREFIX}_smoke_$stamp}
  pairs=("${pairs[0]}")
  extra=(--tasks CuttingToolSelection --parents-per-task 3 --failed-parents-per-task 1 --success-controls-per-task 0 --k-branches 1)
elif [[ "$action" == naturals ]]; then
  export LADDER_ROOT=${LADDER_ROOT:-$recovery_root/${LADDER_RUN_PREFIX}_naturals_$stamp}
  [[ ! -e "$LADDER_ROOT" ]] || { echo "LADDER_ROOT exists; use resume or a new root"; exit 1; }
  extra=(--parents-only --parents-per-task "$LADDER_NATURAL_PARENTS" --base-seed "$LADDER_NATURAL_BASE_SEED")
elif [[ "$action" == start ]]; then
  export LADDER_ROOT=${LADDER_ROOT:-$recovery_root/${LADDER_RUN_PREFIX}_pilot_$stamp}
  [[ ! -e "$LADDER_ROOT" ]] || { echo "LADDER_ROOT exists; use resume or a new root"; exit 1; }
else
  [[ -n "${LADDER_ROOT:-}" ]] || source "$latest"
  test -d "$LADDER_ROOT" || { echo "Missing $LADDER_ROOT"; exit 1; }
  for shard in "$LADDER_ROOT"/shard*; do rm -f "$shard/worker_failure.json"; done
  extra=(--resume)
  # A resumed naturals run must repeat its parents-only arguments.
  if [[ -f "$LADDER_ROOT/collect_args.txt" ]]; then
    mapfile -t saved < "$LADDER_ROOT/collect_args.txt"
    extra+=("${saved[@]}")
  fi
fi
case "$LADDER_ROOT" in "$STORAGE_BS"/*) ;; *) echo "LADDER_ROOT must be under $STORAGE_BS"; exit 1 ;; esac
# The stage-overdue detector needs a calibration; without one it is the v1 stall timer.
frozen_calibration="$LADDER_ROOT/stage_calibration.frozen.json"
if [[ "$action" == start ]]; then
  if [[ -n "${LADDER_STAGE_CALIBRATION:-}" ]]; then
    test -f "$LADDER_STAGE_CALIBRATION" || { echo "Missing $LADDER_STAGE_CALIBRATION"; exit 1; }
  elif [[ "${LADDER_ALLOW_UNCALIBRATED:-0}" != 1 ]]; then
    echo "Set LADDER_STAGE_CALIBRATION (calibrate-stages output) or LADDER_ALLOW_UNCALIBRATED=1"; exit 1
  fi
fi
mkdir -p "$LADDER_ROOT" "$log_dir"
if [[ "$action" != resume && -n "${LADDER_STAGE_CALIBRATION:-}" ]]; then
  cp "$LADDER_STAGE_CALIBRATION" "$frozen_calibration"
fi
[[ ! -f "$frozen_calibration" ]] || extra+=(--stage-calibration "$frozen_calibration")
if [[ "$action" == naturals ]]; then
  printf '%s\n' --parents-only --parents-per-task "$LADDER_NATURAL_PARENTS" --base-seed "$LADDER_NATURAL_BASE_SEED" > "$LADDER_ROOT/collect_args.txt"
fi
for name in STORAGE_BS ROBOCASA_REPO LADDER_ROOT LADDER_CONFIG LADDER_MODEL LADDER_GPU_PAIRS LADDER_PORT_BASE LADDER_RUN_PREFIX XR1_CLIENT_ENV XR1_SERVER_ENV XR1_SAFE_REPO; do
  printf 'export %s=%q\n' "$name" "${!name}"
done > "$LADDER_ROOT/run.env"
cp "$LADDER_ROOT/run.env" "$latest"
git -C "$ROBOCASA_REPO" rev-parse HEAD > "$LADDER_ROOT/git_commit.txt"
nshards=${#pairs[@]}
for shard in "${!pairs[@]}"; do
  gpu=${pairs[$shard]%%:*}; server_gpu=${pairs[$shard]##*:}
  [[ "$gpu" != "$server_gpu" ]] || { echo "Use separate simulator and server GPUs ($gpu)"; exit 1; }
  port=$((LADDER_PORT_BASE + shard))
  export LADDER_WORKER_LOG="$log_dir/$(basename "$LADDER_ROOT")_shard${shard}_$(date +%Y%m%d_%H%M%S).log"
  nohup bash "$ROBOCASA_REPO/scripts/run_recoverability_ladder_xr1.sh" worker \
    "$shard" "$nshards" "$gpu" "$server_gpu" "$port" "${extra[@]}" > "$LADDER_WORKER_LOG" 2>&1 &
  echo "shard=$shard pid=$! sim_gpu=$gpu server_gpu=$server_gpu port=$port log=$LADDER_WORKER_LOG"
done
echo "LADDER_ROOT=$LADDER_ROOT"
