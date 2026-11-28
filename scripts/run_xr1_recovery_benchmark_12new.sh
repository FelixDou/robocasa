#!/usr/bin/env bash
# Collect synchronized passive XR-1 rollouts across the 12 new shared-primitive
# Group-A/B tasks that complete the 18-task Test-1/Test-2 benchmark alongside
# the 6 already-recorded pilot tasks (xr1_recovery_pilot6_full_20261128_142543).
#
# Pre-registered 18-task split:
#   - 10 Train/Dev: 6 pilot tasks + LoadDishwasher, GetToastedBread, PreSoakPan, SteamInMicrowave
#   - 8 Held-Out Test-1: BreadSelection, PackIdenticalLunches, StackBowlsCabinet,
#                        WashFruitColander, ArrangeTea, KettleBoiling, SearingMeat, WaffleReheat
#
# Usage (from the RoboCasa checkout):
#   bash scripts/run_xr1_recovery_benchmark_12new.sh --plan
#   bash scripts/run_xr1_recovery_benchmark_12new.sh --smoke          # 2/task = 24 rollouts (~10 min on 2 GPUs)
#   PILOT_GPUS="0 1" NUM_ROLLOUTS=20 bash scripts/run_xr1_recovery_benchmark_12new.sh
#   PILOT_GPUS="0 1 2 3" NUM_ROLLOUTS=20 bash scripts/run_xr1_recovery_benchmark_12new.sh
#
# Optional environment overrides:
#   PILOT_GPUS         Space-separated GPU indices (default: auto-detected)
#   BENCHMARK_TASKS    Optional space-separated subset of tasks to run
#   NUM_ROLLOUTS       Attempts per task (default: 20; --smoke forces 2)
#   SEED               Base official_xiaomi seed (default: 2900007; --smoke: 2750007)
#   VIDEO_FRAME_STRIDE Video frame stride (default: 2 = 10 fps; --smoke: 2 = 10 fps)
#   PILOT_OUT_DIR      Resume into an existing run root under /gs/bs/.../safe/
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODE="full"
if [[ $# -eq 1 && "$1" == "--plan" ]]; then
  MODE="plan"
elif [[ $# -eq 1 && "$1" == "--smoke" ]]; then
  MODE="smoke"
elif [[ $# -ne 0 ]]; then
  echo "Usage: bash $0 [--plan | --smoke]" >&2
  exit 2
fi

if [[ -n "${BENCHMARK_TASKS:-}" ]]; then
  read -r -a TASKS <<< "$BENCHMARK_TASKS"
else
  # Order interleaves heavy multi-object and shorter tasks for even shard load balancing
  TASKS=(
    LoadDishwasher
    BreadSelection
    GetToastedBread
    PackIdenticalLunches
    PreSoakPan
    StackBowlsCabinet
    SteamInMicrowave
    WashFruitColander
    ArrangeTea
    KettleBoiling
    SearingMeat
    WaffleReheat
  )
fi

STORAGE="${STORAGE_BS:-/gs/bs/tga-shinoda/felid}"
CLIENT="${XR1_CLIENT_PY:-$STORAGE/envs/xiaomi_robotics_1_robocasa365/bin/python}"
SERVER="${XR1_SERVER_ENV:-$STORAGE/envs/xiaomi_robotics_1_server}/bin/python"
XR1="${XR1_SAFE_REPO:-/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe}"
MODEL="${XR1_SAFE_CHECKPOINT:-$STORAGE/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe}"
COLLECTOR="robocasa.recovery.safe.collect_atomic_rollouts"
CLIENT_PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$MODE" == "smoke" ]]; then
  NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
  SEED="${SEED:-2750007}"
  VIDEO_FRAME_STRIDE="${VIDEO_FRAME_STRIDE:-2}"
else
  NUM_ROLLOUTS="${NUM_ROLLOUTS:-20}"
  SEED="${SEED:-2900007}"
  VIDEO_FRAME_STRIDE="${VIDEO_FRAME_STRIDE:-2}"
fi

COMMON_ARGS=(
  --num-rollouts "$NUM_ROLLOUTS"
  --seed "$SEED"
  --seed-protocol official_xiaomi
  --env-interface gym
  --split pretrain
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy
  --model-family xiaomi_robotics_1
  --policy-name Xiaomi-Robotics-1-RoboCasa365
  --checkpoint "$MODEL"
  --host 127.0.0.1
  --replan-steps 16
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}'
  --record-actions
  --record-safe-features
  --record-videos
  --record-subtask-trace
  --video-camera-name obs_grid
  --video-width 768
  --video-height 256
  --video-fps 20
  --video-frame-stride "$VIDEO_FRAME_STRIDE"
  --no-continue-on-error
)

if [[ "$MODE" == "plan" ]]; then
  exec env PYTHONPATH="$CLIENT_PYTHONPATH" "$CLIENT" -m "$COLLECTOR" \
    --tasks "${TASKS[@]}" "${COMMON_ARGS[@]}" --output-dir planned --dry-run
fi

export HF_HOME="$STORAGE/hf_home"
export TRANSFORMERS_CACHE="$STORAGE/hf_home/transformers"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_DEVICE_ORDER=PCI_BUS_ID

test -x "$CLIENT" && test -x "$SERVER" && test -f "$XR1/deploy/server.py" || {
  echo "Check XR-1 client, server, and Xiaomi-Robotics-1-safe repo paths." >&2
  exit 1
}
test -f "$MODEL/model-00003-of-00003.safetensors" || {
  echo "Missing XR-1 weights in $MODEL" >&2
  exit 1
}
test -f "$REPO/robocasa/models/assets/fixtures/windows/Window047/model.xml" || {
  echo "Missing MuJoCo assets in $REPO/robocasa/models/assets (symlink from main checkout first)." >&2
  exit 1
}
"$CLIENT" -c 'import imageio_ffmpeg; imageio_ffmpeg.get_ffmpeg_exe()'

# Resolve GPU list
GPUS=()
if [[ -n "${PILOT_GPUS:-}" ]]; then
  read -r -a GPUS <<< "$PILOT_GPUS"
else
  mask="${CUDA_VISIBLE_DEVICES-}"
  if [[ -n "$mask" ]]; then
    IFS=',' read -r -a raw_devs <<< "$mask"
    for d in "${raw_devs[@]}"; do
      if [[ "$d" == GPU-* ]]; then
        GPUS+=("$(nvidia-smi -i "$d" --query-gpu=index --format=csv,noheader)")
      else
        GPUS+=("$d")
      fi
    done
  else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  fi
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs detected." >&2
  exit 1
fi
for g in "${GPUS[@]}"; do
  [[ "$g" =~ ^[0-9]+$ ]] || { echo "Expected numeric GPU index, got '$g'." >&2; exit 1; }
done

NUM_SHARDS="${#GPUS[@]}"
if (( NUM_SHARDS > ${#TASKS[@]} )); then
  NUM_SHARDS="${#TASKS[@]}"
fi

LOGDIR="$STORAGE/robocasa_logs/eval"
PARENT="$STORAGE/robocasa_rollouts/safe"
mkdir -p "$LOGDIR" "$PARENT"

RESUME_FLAG=()
if [[ -n "${PILOT_OUT_DIR:-}" ]]; then
  OUT_ROOT="$PILOT_OUT_DIR"
  mkdir -p "$OUT_ROOT"
  RESUME_FLAG=(--resume)
else
  PREFIX="xr1_recovery_bench12new_${MODE}_$(date +%Y%m%d_%H%M%S)"
  OUT_ROOT="$PARENT/$PREFIX"
  mkdir -p "$OUT_ROOT"
fi
TAG="${OUT_ROOT##*/}"
MASTER_LOG="$LOGDIR/${TAG}_master.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=== XR-1 12-new-task benchmark collector ($MODE) ==="
echo "Run root:  $OUT_ROOT"
echo "Code:      $(git rev-parse --short HEAD) ($REPO)"
echo "Tasks:     ${#TASKS[@]} (${TASKS[*]})"
echo "GPUs:      ${GPUS[*]:0:$NUM_SHARDS} (shards=$NUM_SHARDS)"
echo "Rollouts:  $NUM_ROLLOUTS/task, seed=$SEED, video_stride=$VIDEO_FRAME_STRIDE"
echo "Resume:    ${RESUME_FLAG[*]:-no}"

echo "=== Preflight benchmark18 unit tests ==="
env PYTHONPATH="$CLIENT_PYTHONPATH" "$CLIENT" -m unittest discover -s "$REPO/tests" -p "test_long_horizon_benchmark18.py" -v

SERVER_PIDS=()
WORKER_PIDS=()
cleanup() {
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Allocate distinct free localhost ports
mapfile -t SHARD_PORTS < <("$CLIENT" - "$NUM_SHARDS" <<'PY'
import socket, sys
n = int(sys.argv[1])
socks = []
try:
    for _ in range(n):
        sk = socket.socket()
        sk.bind(("127.0.0.1", 0))
        print(sk.getsockname()[1])
        socks.append(sk)
finally:
    for sk in socks:
        sk.close()
PY
)

for ((s = 0; s < NUM_SHARDS; s++)); do
  gpu="${GPUS[$s]}"
  port="${SHARD_PORTS[$s]}"
  shard_dir="$OUT_ROOT/shard${s}"
  mkdir -p "$shard_dir"
  server_log="$LOGDIR/${TAG}_shard${s}_server.log"
  collector_log="$LOGDIR/${TAG}_shard${s}_collector.log"

  shard_tasks=()
  for ((t = s; t < ${#TASKS[@]}; t += NUM_SHARDS)); do
    shard_tasks+=("${TASKS[$t]}")
  done
  echo "Shard $s: GPU=$gpu port=$port tasks=(${shard_tasks[*]}) dir=$shard_dir"

  (
    cd "$XR1"
    exec env CUDA_VISIBLE_DEVICES="$gpu" "$SERVER" -u deploy/server.py \
      --model "$MODEL" --host 127.0.0.1 --port "$port"
  ) >"$server_log" 2>&1 &
  spid=$!
  SERVER_PIDS+=("$spid")

  (
    deadline=$((SECONDS + 1800))
    until grep -q 'Model loaded' "$server_log" && "$CLIENT" -c \
      'import socket,sys; socket.create_connection(("127.0.0.1",int(sys.argv[1])),2).close()' "$port" 2>/dev/null; do
      if ! kill -0 "$spid" 2>/dev/null || (( SECONDS >= deadline )); then
        echo "Shard $s: XR-1 server failed to become ready" >&2
        tail -40 "$server_log" >&2 || true
        exit 1
      fi
      sleep 5
    done
    echo "Shard $s: server ready on port $port"
    env PYTHONPATH="$CLIENT_PYTHONPATH" CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
      "$CLIENT" -u -m "$COLLECTOR" \
      --tasks "${shard_tasks[@]}" \
      "${COMMON_ARGS[@]}" \
      ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} \
      --port "$port" \
      --output-dir "$shard_dir"
    env PYTHONPATH="$CLIENT_PYTHONPATH" "$CLIENT" -m robocasa.recovery.safe.validate_atomic_dataset \
      --dataset-dir "$shard_dir"
    env PYTHONPATH="$CLIENT_PYTHONPATH" "$CLIENT" -m scripts.xr1_lead_time.verify_recovery_recording \
      "$shard_dir" --tasks "${shard_tasks[@]}" --expected-per-task "$NUM_ROLLOUTS" --stride "$VIDEO_FRAME_STRIDE"
  ) >"$collector_log" 2>&1 &
  WORKER_PIDS+=("$!")
done

FAIL=0
for ((s = 0; s < NUM_SHARDS; s++)); do
  if ! wait "${WORKER_PIDS[$s]}"; then
    echo "FAIL: shard $s failed; inspect $LOGDIR/${TAG}_shard${s}_collector.log" >&2
    FAIL=1
  else
    echo "PASS: shard $s completed and verified"
  fi
done

for spid in "${SERVER_PIDS[@]}"; do
  kill "$spid" 2>/dev/null || true
done
wait "${SERVER_PIDS[@]}" 2>/dev/null || true
trap - EXIT

if (( FAIL != 0 )); then
  exit 1
fi

TAR_PATHS=()
for ((s = 0; s < NUM_SHARDS; s++)); do
  TAR_PATHS+=("shard${s}/manifest.jsonl" "shard${s}/summary.json" "shard${s}/subtasks")
done
tar -C "$OUT_ROOT" -czf "$OUT_ROOT/metadata_review.tar.gz" "${TAR_PATHS[@]}"
ls -lh "$OUT_ROOT/metadata_review.tar.gz"
echo "Metadata review bundle: $OUT_ROOT/metadata_review.tar.gz"
echo "All shards done: $OUT_ROOT"
