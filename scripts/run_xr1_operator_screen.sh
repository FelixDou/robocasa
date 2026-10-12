#!/usr/bin/env bash
# One-GPU XR1 operator sentinel -> full screen -> paired analysis.
set -euo pipefail

action=${1:-status}
export STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
export ROBOCASA_REPO=${ROBOCASA_REPO:-/gs/fs/tga-shinoda/felid/robocasa}
latest="$STORAGE_BS/robocasa_rollouts/recovery/xr1_operator_latest.env"

if [[ "$action" == status || "$action" == resume ]]; then
  if [[ -z "${XR1_OPERATOR_ROOT:-}" ]]; then
    test -f "$latest" || { echo "No saved operator run: $latest"; exit 1; }
    source "$latest"
  fi
fi

export XR1_SAFE_REPO=${XR1_SAFE_REPO:-/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe}
export XR1_SERVER_ENV=${XR1_SERVER_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_server}
export XR1_CLIENT_ENV=${XR1_CLIENT_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365}
export XR1_SAFE_CHECKPOINT=${XR1_SAFE_CHECKPOINT:-$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe}
export XR1_PHASE3B_ROOT=${XR1_PHASE3B_ROOT:-$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_terminal_camera_v7_full20_20261012_091253}
export XR1_OPERATOR_PORT=${XR1_OPERATOR_PORT:-10316}
export XR1_OPERATOR_REUSE_SERVER=${XR1_OPERATOR_REUSE_SERVER:-0}
export XR1_OPERATOR_GPU=${XR1_OPERATOR_GPU:-0}
export HF_HOME="$STORAGE_BS/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES="$XR1_OPERATOR_GPU" MUJOCO_EGL_DEVICE_ID=0
cd "$ROBOCASA_REPO"

if [[ "$action" == status ]]; then
  "$XR1_CLIENT_ENV/bin/python" - "$XR1_OPERATOR_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
print("run:", root)
for scope in ("sentinel", "full"):
    path = root / scope / "status.json"
    if path.exists():
        data = json.loads(path.read_text())
        print(scope, data.get("status"), "branches", data.get("branches"), "/", data.get("expected_branches"))
        if data.get("error"):
            print("error:", data["error"])
    else:
        print(scope, "not started")
path = root / "full/scientific_analysis.json"
if path.exists():
    data = json.loads(path.read_text())
    print("decision:", data["decision"])
    print("best fixed:", data["best_fixed_operator"])
    print("operator gains:", {k: round(v["completion"], 4) for k, v in data["contrasts_vs_continue"].items()})
    print("oracle routing gain:", data["oracle_routing_minus_best_fixed"])
PY
  echo "driver log: ${XR1_OPERATOR_LOG:-unknown}"
  exit 0
fi

if [[ "$action" == start || "$action" == resume ]]; then
  test -x "$XR1_CLIENT_ENV/bin/python"
  test -x "$XR1_SERVER_ENV/bin/python"
  test -f "$XR1_PHASE3B_ROOT/registration.json"
  test -f "$XR1_SAFE_REPO/deploy/server.py"
  export XR1_OPERATOR_ROOT=${XR1_OPERATOR_ROOT:-$STORAGE_BS/robocasa_rollouts/recovery/xr1_operator_$(date +%Y%m%d_%H%M%S)}
  if [[ "$action" == start && -e "$XR1_OPERATOR_ROOT" ]]; then
    echo "Output root already exists; use resume or a new XR1_OPERATOR_ROOT"
    exit 1
  fi
  mkdir -p "$XR1_OPERATOR_ROOT" "$STORAGE_BS/robocasa_logs/eval"
  export XR1_OPERATOR_LOG="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_OPERATOR_ROOT")_driver_$(date +%Y%m%d_%H%M%S).log"
  for name in STORAGE_BS ROBOCASA_REPO XR1_OPERATOR_ROOT XR1_OPERATOR_LOG XR1_PHASE3B_ROOT XR1_OPERATOR_PORT XR1_OPERATOR_GPU XR1_SAFE_REPO XR1_SERVER_ENV XR1_CLIENT_ENV XR1_SAFE_CHECKPOINT; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done > "$latest"
  if [[ "$action" == start ]]; then
    if [[ -n "${XR1_OPERATOR_SPEC:-}" ]]; then
      cp "$XR1_OPERATOR_SPEC" "$XR1_OPERATOR_ROOT/operator_spec.json"
    else
      "$XR1_CLIENT_ENV/bin/python" - "$XR1_OPERATOR_ROOT/operator_spec.json" <<'PY'
import json, pathlib, sys
from robocasa.recovery.operator_screen import DEFAULT_SPEC
pathlib.Path(sys.argv[1]).write_text(json.dumps(DEFAULT_SPEC, indent=2) + "\n")
PY
    fi
  fi
  nohup bash "$ROBOCASA_REPO/scripts/run_xr1_operator_screen.sh" worker > "$XR1_OPERATOR_LOG" 2>&1 &
  echo "driver_pid=$!"
  echo "run=$XR1_OPERATOR_ROOT"
  echo "log=$XR1_OPERATOR_LOG"
  exit 0
fi

[[ "$action" == worker ]] || { echo "Usage: $0 {start|resume|status}"; exit 1; }
# Lock the whole run, including server startup. Per-scope collectors also lock.
exec 9>"$XR1_OPERATOR_ROOT/.driver.lock"
flock -n 9 || { echo "Another driver owns this run"; exit 1; }
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$XR1_CLIENT_ENV/bin/python" - "$STORAGE_BS" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free
print("durable free GiB:", round(free / 2**30, 1), flush=True)
if free < 10 * 2**30:
    raise SystemExit("Less than 10 GiB free; inspect storage before starting")
PY
df -h "$STORAGE_BS" /tmp
nvidia-smi
export XR1_PHASE2_ROOT
XR1_PHASE2_ROOT=$("$XR1_CLIENT_ENV/bin/python" - "$XR1_PHASE3B_ROOT/registration.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["phase2_run_dir"])
PY
)
export XR1_OPERATOR_MODEL
XR1_OPERATOR_MODEL=$("$XR1_CLIENT_ENV/bin/python" - "$XR1_PHASE2_ROOT/plan.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["checkpoint_provenance"]["path"])
PY
)
if [[ "$XR1_OPERATOR_MODEL" == /tmp/ut06746/* ]]; then
  mkdir -p "$XR1_OPERATOR_MODEL"
  rsync -a --info=progress2 "$XR1_SAFE_CHECKPOINT/" "$XR1_OPERATOR_MODEL/"
fi
test -f "$XR1_OPERATOR_MODEL/modeling_mibot.py"

common=(--phase3b-run-dir "$XR1_PHASE3B_ROOT" --model-path "$XR1_OPERATOR_MODEL"
        --host 127.0.0.1 --port "$XR1_OPERATOR_PORT" --operator-spec "$XR1_OPERATOR_ROOT/operator_spec.json")
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.run_operator_screen \
  "${common[@]}" --scope sentinel --output-dir "$XR1_OPERATOR_ROOT/preflight" --dry-run

if [[ "$XR1_OPERATOR_REUSE_SERVER" != 1 ]]; then
  if ss -ltn | awk '{print $4}' | grep -Eq ":${XR1_OPERATOR_PORT}$"; then
    echo "Port $XR1_OPERATOR_PORT is occupied; choose a unique XR1_OPERATOR_PORT"
    exit 1
  fi
  server_log="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_OPERATOR_ROOT")_server_$(date +%Y%m%d_%H%M%S).log"
  (
    cd "$XR1_SAFE_REPO"
    exec "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
      --model "$XR1_OPERATOR_MODEL" --host 127.0.0.1 --port "$XR1_OPERATOR_PORT"
  ) > "$server_log" 2>&1 &
  server_pid=$!
  deadline=$((SECONDS + 1800))
  until "$XR1_CLIENT_ENV/bin/python" - "$XR1_OPERATOR_PORT" <<'PY'
import socket, sys
with socket.socket() as connection:
    raise SystemExit(connection.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
  do
    kill -0 "$server_pid" 2>/dev/null || { tail -60 "$server_log"; exit 1; }
    (( SECONDS < deadline )) || { echo "Server startup timed out"; tail -60 "$server_log"; exit 1; }
    tail -5 "$server_log"
    sleep 10
  done
  grep -q 'Model loaded' "$server_log" || { echo "Missing model-ready message"; tail -60 "$server_log"; exit 1; }
  kill -0 "$server_pid"
fi

"$XR1_CLIENT_ENV/bin/python" - "$XR1_OPERATOR_PORT" <<'PY'
import socket, sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=10):
    pass
print("Frozen Xiaomi server is listening", flush=True)
PY

for scope in sentinel full; do
  arguments=("${common[@]}" --scope "$scope" --output-dir "$XR1_OPERATOR_ROOT/$scope")
  if [[ "$scope" == full ]]; then
    arguments+=(--sentinel-run-dir "$XR1_OPERATOR_ROOT/sentinel")
  fi
  if [[ -f "$XR1_OPERATOR_ROOT/$scope/registration.json" ]]; then
    arguments+=(--resume)
  fi
  "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.run_operator_screen "${arguments[@]}"
done
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.analyze_operator_screen \
  --run-dir "$XR1_OPERATOR_ROOT/full" --bootstrap-replicates 2000 --bootstrap-seed 0
echo "Operator screen complete: $XR1_OPERATOR_ROOT/full/scientific_analysis.json"
