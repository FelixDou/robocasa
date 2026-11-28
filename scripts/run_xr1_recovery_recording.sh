#!/usr/bin/env bash
# Run from the RoboCasa checkout on one allocated GPU. --plan only needs Python and NumPy.
set -euo pipefail
cd "$(dirname "$0")/.."
storage=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
client=${XR1_CLIENT_PY:-$storage/envs/xiaomi_robotics_1_robocasa365/bin/python}
server=${XR1_SERVER_ENV:-$storage/envs/xiaomi_robotics_1_server}/bin/python
xr1=${XR1_SAFE_REPO:-/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe}
model=${XR1_SAFE_CHECKPOINT:-$storage/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe}
args=(--tasks ArrangeBreadBasket CuttingToolSelection --num-rollouts 2
      --seed 2600007 --seed-protocol official_xiaomi --env-interface gym --split pretrain
      --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy
      --model-family xiaomi_robotics_1 --policy-name Xiaomi-Robotics-1-RoboCasa365
      --checkpoint "$model" --host 127.0.0.1 --replan-steps 16
      --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}'
      --record-actions --record-safe-features --record-videos --record-subtask-trace
      --video-camera-name obs_grid --video-width 768 --video-height 256
      --video-fps 20 --video-frame-stride 1 --no-continue-on-error)
collector=robocasa.recovery.safe.collect_atomic_rollouts
if [[ $# == 1 && $1 == --plan ]]; then
  exec "$client" -m "$collector" "${args[@]}" --output-dir planned --dry-run
fi
if [[ $# != 0 ]]; then echo "Usage: bash $0 [--plan]" >&2; exit 2; fi
export HF_HOME="$storage/hf_home" TRANSFORMERS_CACHE="$storage/hf_home/transformers"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
test -x "$client" && test -x "$server" && test -f "$xr1/deploy/server.py" || {
  echo "Check the XR-1 client environment, server environment and patched server path." >&2; exit 1; }
test -f "$model/model-00003-of-00003.safetensors" || { echo "Missing XR-1 weights in $model" >&2; exit 1; }
# Use the allocated device even when it is identified by GPU UUID.
mask=${CUDA_VISIBLE_DEVICES-}
device=${mask%%,*}
if [[ -z "$device" ]]; then
  device=$(nvidia-smi --query-gpu=index --format=csv,noheader)
elif [[ "$device" == GPU-* ]]; then
  device=$(nvidia-smi -i "$device" --query-gpu=index --format=csv,noheader)
fi
[[ "$device" =~ ^[0-9]+$ ]] || { echo "Expected a full allocated GPU, got '$device'." >&2; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$device" MUJOCO_EGL_DEVICE_ID="$device"
logdir="$storage/robocasa_logs/eval"
parent="$storage/robocasa_rollouts/safe"
mkdir -p "$logdir" "$parent"
out=$(mktemp -d "$parent/xr1_recovery_recording_XXXXXXXX")
tag=${out##*/}
server_log="$logdir/${tag}_server.log"
exec > >(tee "$logdir/${tag}_collector.log") 2>&1
echo "Run directory: $out"
echo "Server log: $server_log"
"$client" -c 'import imageio_ffmpeg; imageio_ffmpeg.get_ffmpeg_exe()'
port=$("$client" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
echo "GPU $device, port $port, code $(git rev-parse --short HEAD)"
(
  cd "$xr1"
  exec "$server" -u deploy/server.py --model "$model" --host 127.0.0.1 --port "$port"
) >"$server_log" 2>&1 &
pid=$!
stop_server() { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline=$((SECONDS + 1800))
until grep -q 'Model loaded' "$server_log" && "$client" -c \
  'import socket,sys; socket.create_connection(("127.0.0.1",int(sys.argv[1])),2).close()' "$port" 2>/dev/null; do
  if ! kill -0 "$pid" 2>/dev/null || (( SECONDS >= deadline )); then
    echo "XR-1 failed to become ready"; tail -40 "$server_log"; exit 1
  fi
  sleep 5
done
"$client" -u -m "$collector" "${args[@]}" --port "$port" --output-dir "$out"
stop_server
trap - EXIT
"$client" -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir "$out"
"$client" -m scripts.xr1_lead_time.verify_recovery_recording "$out"
tar -C "$out" -czf "$out/review.tar.gz" manifest.jsonl summary.json videos subtasks
echo "Review package: $out/review.tar.gz"
