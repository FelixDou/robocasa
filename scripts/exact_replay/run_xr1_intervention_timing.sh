#!/usr/bin/env bash
# One GPU: audited stage-entry replay -> 60-run sentinel -> 220 total -> analysis.
set -euo pipefail
action=${1:-status}
export STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
export ROBOCASA_REPO=${ROBOCASA_REPO:-/gs/fs/tga-shinoda/felid/robocasa}
latest="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_latest.env"
if [[ "$action" == status || "$action" == resume ]]; then
  if [[ -z "${XR1_TIMING_ROOT:-}" ]]; then
    test -f "$latest" || { echo "No saved timing run: $latest"; exit 1; }
    source "$latest"
  fi
fi
export XR1_SAFE_REPO=${XR1_SAFE_REPO:-/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe}
export XR1_SERVER_ENV=${XR1_SERVER_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_server}
export XR1_CLIENT_ENV=${XR1_CLIENT_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365}
export XR1_SAFE_CHECKPOINT=${XR1_SAFE_CHECKPOINT:-$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe}
export XR1_TIMING_GPU=${XR1_TIMING_GPU:-0} XR1_TIMING_PORT=${XR1_TIMING_PORT:-10318}
export HF_HOME="$STORAGE_BS/hf_home" TRANSFORMERS_CACHE="$STORAGE_BS/hf_home/transformers"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES="$XR1_TIMING_GPU" MUJOCO_EGL_DEVICE_ID=0
cd "$ROBOCASA_REPO"

if [[ "$action" == status ]]; then
  "$XR1_CLIENT_ENV/bin/python" - "$XR1_TIMING_ROOT/full" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
print("run:", root)
for name in ("status.json", "sentinel_analysis.json", "analysis.json", "scientific_analysis.json"):
    path = root / name
    if path.exists():
        data = json.loads(path.read_text())
        print(name, {k: data[k] for k in ("status", "scope", "branches", "expected_branches", "engineering_all_pass", "current_branch", "error", "decision", "recommended_fresh_validation_cell") if k in data})
        if "contrasts_vs_continue" in data:
            print("completion gains:", {c: round(v["completion"], 4) for c, v in data["contrasts_vs_continue"].items()})
PY
  echo "driver log: ${XR1_TIMING_LOG:-unknown}"
  exit 0
fi

if [[ "$action" == start || "$action" == resume ]]; then
  test -x "$XR1_CLIENT_ENV/bin/python"
  test -x "$XR1_SERVER_ENV/bin/python"
  test -f "$XR1_SAFE_REPO/deploy/server.py"
  export XR1_TIMING_ROOT=${XR1_TIMING_ROOT:-$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_$(date +%Y%m%d_%H%M%S)}
  if [[ "$action" == resume ]]; then
    test -f "$XR1_TIMING_ROOT/timing_spec.json"
    if [[ -z "${XR1_TIMING_SOURCE:-}" && -f "$XR1_TIMING_ROOT/full/registration.json" ]]; then
      XR1_TIMING_SOURCE=$("$XR1_CLIENT_ENV/bin/python" - "$XR1_TIMING_ROOT/full/registration.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["operator_run_dir"])
PY
)
    fi
  elif [[ -e "$XR1_TIMING_ROOT" ]]; then
    echo "Output root already exists; use resume or a new XR1_TIMING_ROOT"; exit 1
  fi
  if [[ -z "${XR1_TIMING_SOURCE:-}" ]]; then
    operator_latest="$STORAGE_BS/robocasa_rollouts/recovery/xr1_operator_latest.env"
    test -f "$operator_latest" || { echo "Set XR1_TIMING_SOURCE to the completed operator run's full directory"; exit 1; }
    XR1_TIMING_SOURCE=$(source "$operator_latest"; printf '%s/full' "$XR1_OPERATOR_ROOT")
  fi
  export XR1_TIMING_SOURCE
  test -f "$XR1_TIMING_SOURCE/scientific_analysis.json"
  "$XR1_CLIENT_ENV/bin/python" - "$STORAGE_BS" "$XR1_TIMING_ROOT" <<'PY'
from pathlib import Path
import sys
if not Path(sys.argv[2]).resolve().is_relative_to(Path(sys.argv[1]).resolve()):
    raise SystemExit("Timing outputs must be under durable STORAGE_BS, never /gs/fs")
PY
  mkdir -p "$XR1_TIMING_ROOT" "$STORAGE_BS/robocasa_logs/eval"
  export XR1_TIMING_LOG="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_TIMING_ROOT")_driver_$(date +%Y%m%d_%H%M%S).log"
  if [[ "$action" == start ]]; then
    "$XR1_CLIENT_ENV/bin/python" - "$XR1_TIMING_ROOT/timing_spec.json" <<'PY'
import json, pathlib, sys
from robocasa.recovery.exact_replay.intervention_timing import DEFAULT_SPEC
pathlib.Path(sys.argv[1]).write_text(json.dumps(DEFAULT_SPEC, indent=2) + "\n")
PY
  fi
  for name in STORAGE_BS ROBOCASA_REPO XR1_TIMING_ROOT XR1_TIMING_SOURCE XR1_TIMING_LOG XR1_TIMING_PORT XR1_TIMING_GPU XR1_SAFE_REPO XR1_SERVER_ENV XR1_CLIENT_ENV XR1_SAFE_CHECKPOINT; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done > "$XR1_TIMING_ROOT/run.env"
  cp "$XR1_TIMING_ROOT/run.env" "$latest"
  nohup bash "$ROBOCASA_REPO/scripts/exact_replay/run_xr1_intervention_timing.sh" worker > "$XR1_TIMING_LOG" 2>&1 &
  echo "driver_pid=$!"
  echo "run=$XR1_TIMING_ROOT"
  echo "source=$XR1_TIMING_SOURCE"
  echo "log=$XR1_TIMING_LOG"
  exit 0
fi

[[ "$action" == worker ]] || { echo "Usage: $0 {start|resume|status}"; exit 1; }
exec 9>"$XR1_TIMING_ROOT/.driver.lock"
flock -n 9 || { echo "Another timing driver owns this run"; exit 1; }
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
export XR1_TIMING_MODEL
XR1_TIMING_MODEL=$("$XR1_CLIENT_ENV/bin/python" - "$XR1_TIMING_SOURCE/registration.json" <<'PY'
import json, pathlib, sys
registration = json.loads(pathlib.Path(sys.argv[1]).read_text())
plan = json.loads((pathlib.Path(registration["phase2_run_dir"]) / "plan.json").read_text())
print(plan["checkpoint_provenance"]["path"])
PY
)
if [[ "$XR1_TIMING_MODEL" == /tmp/ut06746/* ]]; then
  mkdir -p "$XR1_TIMING_MODEL"
  rsync -a --info=progress2 "$XR1_SAFE_CHECKPOINT/" "$XR1_TIMING_MODEL/"
fi
test -f "$XR1_TIMING_MODEL/modeling_mibot.py"
common=(--operator-run-dir "$XR1_TIMING_SOURCE" --model-path "$XR1_TIMING_MODEL"
        --host 127.0.0.1 --port "$XR1_TIMING_PORT" --timing-spec "$XR1_TIMING_ROOT/timing_spec.json"
        --output-dir "$XR1_TIMING_ROOT/full")
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.exact_replay.run_intervention_timing \
  "${common[@]}" --scope sentinel --dry-run
if ss -ltn | awk '{print $4}' | grep -Eq ":${XR1_TIMING_PORT}$"; then
  echo "Port $XR1_TIMING_PORT is occupied; choose a unique XR1_TIMING_PORT"; exit 1
fi
server_log="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_TIMING_ROOT")_server_$(date +%Y%m%d_%H%M%S).log"
(
  cd "$XR1_SAFE_REPO"
  exec "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
    --model "$XR1_TIMING_MODEL" --host 127.0.0.1 --port "$XR1_TIMING_PORT"
) > "$server_log" 2>&1 &
server_pid=$!
deadline=$((SECONDS + 1800))
until "$XR1_CLIENT_ENV/bin/python" - "$XR1_TIMING_PORT" <<'PY'
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
echo "Frozen Xiaomi policy server ready on $XR1_TIMING_PORT"
for scope in sentinel full; do
  arguments=("${common[@]}" --scope "$scope")
  if [[ -f "$XR1_TIMING_ROOT/full/registration.json" ]]; then
    arguments+=(--resume)
  fi
  "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.exact_replay.run_intervention_timing "${arguments[@]}"
done
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.exact_replay.analyze_intervention_timing \
  --run-dir "$XR1_TIMING_ROOT/full" --bootstrap-replicates 2000 --bootstrap-seed 0
echo "Timing study complete: $XR1_TIMING_ROOT/full/scientific_analysis.json"
