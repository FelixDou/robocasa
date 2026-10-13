#!/usr/bin/env bash
# Two isolated nominal replays; no intervention branches or source-root writes.
set -euo pipefail
action=${1:-status}
export STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
export ROBOCASA_REPO=${ROBOCASA_REPO:-/gs/fs/tga-shinoda/felid/robocasa}
latest="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_replay_latest.env"
if [[ ( "$action" == status || "$action" == resume ) && -z "${XR1_REPLAY_ROOT:-}" ]]; then
  source "$latest"
fi
if [[ "$action" == status ]]; then
  "${XR1_CLIENT_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365}/bin/python" - "$XR1_REPLAY_ROOT/diagnostic" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
if (root/"status.json").exists():
    print("status:", json.loads((root/"status.json").read_text()))
if (root/"report.json").exists():
    data = json.loads((root/"report.json").read_text())
    if data.get("protocol") == "xr1_gc_guarded_all_parent_replay_audit_v1":
        print("all-parent audit:", {k: data[k] for k in ("status", "parents", "expected_parents", "nominal_replays", "parents_by_task", "engineering_all_pass", "scientific_trajectories_collected", "full_timing_study_complete")})
        for row in data["parent_reports"]:
            print("parent", row["parent_id"], row["engineering_gates"])
        print("report:", root/"report.json")
        raise SystemExit(0)
    print("finding:", data["finding"])
    print("RNG runtime initialization:", data.get("rng_runtime_init", "after_reset"))
    print("GPU assignment:", data.get("device_assignment"))
    collection = data.get("garbage_collection")
    if collection is not None:
        print("GC mode:", collection["mode"])
        divergence = data.get("first_repeat_divergence") or {}
        boundary = divergence.get("step")
        for i, trace in enumerate(collection["attempts"]):
            print("GC attempt", i, {k: v for k, v in trace.items() if k != "events"})
            if boundary is not None:
                nearby = [e for e in trace["events"] if e["nominal"] and abs(e["environment_step"]-boundary) <= 1]
                print("GC events near first difference:", nearby[:20])
    if "startup_events" in data:
        print("startup events:", data["startup_events"])
    for i, attempt in enumerate(data["attempts"]):
        print("attempt", i, {k: attempt[k] for k in ("prefix_replay_accepted", "environment_steps", "source_endpoint_channels", "final_ordered_stage", "final_at_inference_boundary", "request_repeat_probe")})
        print("initial XML:", {k: attempt["initial_model_xml_audit"][k] for k in ("available", "compatible")})
        print("replay error:", attempt["replay_error"]["message"] if attempt["replay_error"] else None)
    print("initial repeats:", data["repeated_initial_state_audit"]["channels"])
    print("endpoint repeats:", data["repeated_endpoint_audit"]["channels"])
    print("first repeat divergence:", data["first_repeat_divergence"])
    window = data.get("observation_window_comparison")
    if window is not None:
        print("requested raw window complete:", window["requested_window_complete"])
        print("observation trace rows:", window["trace_rows"])
        print("first observation difference (environment step):", window["first_observation_difference_step"])
        print("first difference by observation key:", window["first_difference_by_observation_key"])
        print("first difference by captured state channel:", window["first_difference_by_captured_state_channel"])
        print("first observation difference has raw state:", window["first_difference_raw_state_captured"])
        print("all-observation archive counts:", window.get("raw_observation_counts"))
        print("first observation difference has raw pixels:", window.get("first_difference_raw_observation_captured"))
        details = window["first_observation_difference"]
        if details is not None:
            print("state channels at first observation difference:", details["state_audit"]["channels"] if details["state_audit"] is not None else "not captured")
            print("progress channels at first observation difference:", details.get("progress_channel_matches"))
            print("observation differences:", details["changed_observation_keys"])
            print("first-difference camera statistics:", details.get("camera_statistics"))
        print("window camera statistics:", window["camera_statistics"])
    print("divergence from previous failed run:", data["first_divergence_from_failed_attempt"])
    print("report:", root/"report.json")
PY
  echo "driver log: ${XR1_REPLAY_LOG:-unknown}"
  exit 0
fi
export XR1_CLIENT_ENV=${XR1_CLIENT_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365}
export XR1_SERVER_ENV=${XR1_SERVER_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_server}
export XR1_SAFE_REPO=${XR1_SAFE_REPO:-/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe}
export XR1_SAFE_CHECKPOINT=${XR1_SAFE_CHECKPOINT:-$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe}
export XR1_REPLAY_SOURCE=${XR1_REPLAY_SOURCE:-${XR1_TIMING_ROOT:?Source the failed timing run.env or set XR1_REPLAY_SOURCE}/full}
export XR1_REPLAY_PARENT=${XR1_REPLAY_PARENT:-f9cd6c2c907171c2f8fea437}
export XR1_REPLAY_GPU=${XR1_REPLAY_GPU:-0} XR1_REPLAY_PORT=${XR1_REPLAY_PORT:-10319}
export XR1_REPLAY_SERVER_GPU=${XR1_REPLAY_SERVER_GPU:-$XR1_REPLAY_GPU}
export XR1_REPLAY_ALL_PARENTS=${XR1_REPLAY_ALL_PARENTS:-0}
[[ "$XR1_REPLAY_ALL_PARENTS" =~ ^[01]$ ]] || {
  echo "XR1_REPLAY_ALL_PARENTS must be 0 or 1"; exit 1;
}
if [[ "$action" == resume && "$XR1_REPLAY_ALL_PARENTS" != 1 ]]; then
  echo "Resume is available only for the all-parent audit"; exit 1
fi
export XR1_REPLAY_RNG_RUNTIME_INIT=${XR1_REPLAY_RNG_RUNTIME_INIT:-after_reset}
export XR1_REPLAY_GC_MODE=${XR1_REPLAY_GC_MODE:-default}
case "$XR1_REPLAY_GC_MODE" in
  default|disabled) ;;
  *) echo "XR1_REPLAY_GC_MODE must be default or disabled"; exit 1 ;;
esac
export XR1_REPLAY_RECORD_ALL_OBSERVATIONS=${XR1_REPLAY_RECORD_ALL_OBSERVATIONS:-0}
[[ "$XR1_REPLAY_RECORD_ALL_OBSERVATIONS" =~ ^[01]$ ]] || {
  echo "XR1_REPLAY_RECORD_ALL_OBSERVATIONS must be 0 or 1"; exit 1;
}
export XR1_REPLAY_OBS_WINDOW_START=${XR1_REPLAY_OBS_WINDOW_START:-}
export XR1_REPLAY_OBS_WINDOW_END=${XR1_REPLAY_OBS_WINDOW_END:-}
if [[ -n "$XR1_REPLAY_OBS_WINDOW_START" || -n "$XR1_REPLAY_OBS_WINDOW_END" ]]; then
  [[ "$XR1_REPLAY_OBS_WINDOW_START" =~ ^[0-9]+$ && "$XR1_REPLAY_OBS_WINDOW_END" =~ ^[0-9]+$ ]] || {
    echo "Set both XR1_REPLAY_OBS_WINDOW_START and XR1_REPLAY_OBS_WINDOW_END to environment step numbers"; exit 1;
  }
fi
case "$XR1_REPLAY_RNG_RUNTIME_INIT" in
  after_reset|before_env) ;;
  *) echo "XR1_REPLAY_RNG_RUNTIME_INIT must be after_reset or before_env"; exit 1 ;;
esac
export CUDA_VISIBLE_DEVICES="$XR1_REPLAY_GPU" MUJOCO_EGL_DEVICE_ID=0
export HF_HOME="$STORAGE_BS/hf_home" TRANSFORMERS_CACHE="$STORAGE_BS/hf_home/transformers"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
cd "$ROBOCASA_REPO"
if [[ "$action" == start || "$action" == resume ]]; then
  export XR1_REPLAY_ROOT=${XR1_REPLAY_ROOT:-$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_replay_$(date +%Y%m%d_%H%M%S)}
  test -x "$XR1_CLIENT_ENV/bin/python"
  test -x "$XR1_SERVER_ENV/bin/python"
  test -f "$XR1_REPLAY_SOURCE/registration.json"
  test -f "$XR1_SAFE_REPO/deploy/server.py"
  "$XR1_CLIENT_ENV/bin/python" - "$STORAGE_BS" "$XR1_REPLAY_ROOT" "$XR1_REPLAY_SOURCE" "$action" <<'PY'
from pathlib import Path
import sys
storage, root, source = (Path(p).resolve() for p in sys.argv[1:4])
if not root.is_relative_to(storage) or root == storage:
    raise SystemExit("Diagnostic outputs must be in a new directory under durable STORAGE_BS")
if root == source or root.is_relative_to(source) or source.is_relative_to(root):
    raise SystemExit("Diagnostic root must be separate from the frozen timing root")
if root.exists() and sys.argv[4] == "start":
    raise SystemExit("Diagnostic root already exists; select a new XR1_REPLAY_ROOT")
if sys.argv[4] == "resume" and not (root/"diagnostic/registration.json").is_file():
    raise SystemExit("Missing all-parent audit registration for resume")
PY
  mkdir -p "$XR1_REPLAY_ROOT" "$STORAGE_BS/robocasa_logs/eval"
  export XR1_REPLAY_LOG="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_REPLAY_ROOT")_driver_$(date +%Y%m%d_%H%M%S)_$$.log"
  for name in STORAGE_BS ROBOCASA_REPO XR1_REPLAY_SOURCE XR1_REPLAY_ROOT XR1_REPLAY_PARENT XR1_REPLAY_GPU XR1_REPLAY_SERVER_GPU XR1_REPLAY_ALL_PARENTS XR1_REPLAY_PORT XR1_REPLAY_RNG_RUNTIME_INIT XR1_REPLAY_GC_MODE XR1_REPLAY_RECORD_ALL_OBSERVATIONS XR1_REPLAY_OBS_WINDOW_START XR1_REPLAY_OBS_WINDOW_END XR1_REPLAY_LOG XR1_CLIENT_ENV XR1_SERVER_ENV XR1_SAFE_REPO XR1_SAFE_CHECKPOINT; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done > "$XR1_REPLAY_ROOT/run.env"
  cp "$XR1_REPLAY_ROOT/run.env" "$latest"
  nohup bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" worker > "$XR1_REPLAY_LOG" 2>&1 &
  echo "driver_pid=$!"
  echo "run=$XR1_REPLAY_ROOT"
  echo "log=$XR1_REPLAY_LOG"
  exit 0
fi
[[ "$action" == worker ]] || { echo "Usage: $0 {start|resume|status}"; exit 1; }
exec 9>"$XR1_REPLAY_ROOT/.driver.lock"
flock -n 9 || { echo "Another driver owns this diagnostic root"; exit 1; }
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
if shutil.disk_usage(sys.argv[1]).free < 10 * 2**30:
    raise SystemExit("Less than 10 GiB free; inspect durable storage before starting")
PY
export XR1_REPLAY_MODEL
XR1_REPLAY_MODEL=$("$XR1_CLIENT_ENV/bin/python" - "$XR1_REPLAY_SOURCE/registration.json" <<'PY'
import json, pathlib, sys
registration = json.loads(pathlib.Path(sys.argv[1]).read_text())
plan = json.loads((pathlib.Path(registration["phase2_run_dir"])/"plan.json").read_text())
print(plan["checkpoint_provenance"]["path"])
PY
)
if [[ "$XR1_REPLAY_MODEL" == /tmp/ut06746/* ]]; then
  mkdir -p "$XR1_REPLAY_MODEL"
  rsync -a --info=progress2 "$XR1_SAFE_CHECKPOINT/" "$XR1_REPLAY_MODEL/"
fi
replay_module=robocasa.recovery.diagnose_intervention_timing_replay
common=(--timing-run-dir "$XR1_REPLAY_SOURCE"
        --output-dir "$XR1_REPLAY_ROOT/diagnostic" --model-path "$XR1_REPLAY_MODEL"
        --host 127.0.0.1 --port "$XR1_REPLAY_PORT" --rng-runtime-init "$XR1_REPLAY_RNG_RUNTIME_INIT"
        --gc-mode "$XR1_REPLAY_GC_MODE")
if [[ "$XR1_REPLAY_ALL_PARENTS" == 1 ]]; then
  replay_module=robocasa.recovery.audit_intervention_timing_parents
  if [[ -f "$XR1_REPLAY_ROOT/diagnostic/registration.json" ]]; then
    common+=(--resume)
  fi
else
  common+=(--parent-id "$XR1_REPLAY_PARENT")
fi
if [[ -n "$XR1_REPLAY_OBS_WINDOW_START" ]]; then
  common+=(--observation-window "$XR1_REPLAY_OBS_WINDOW_START" "$XR1_REPLAY_OBS_WINDOW_END")
fi
if [[ "$XR1_REPLAY_RECORD_ALL_OBSERVATIONS" == 1 ]]; then
  common+=(--record-all-observations)
fi
"$XR1_CLIENT_ENV/bin/python" -u -m "$replay_module" "${common[@]}" --dry-run
if ss -ltn | awk '{print $4}' | grep -Eq ":${XR1_REPLAY_PORT}$"; then
  echo "Port $XR1_REPLAY_PORT is occupied; choose a unique XR1_REPLAY_PORT"; exit 1
fi
server_log="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_REPLAY_ROOT")_server_$(date +%Y%m%d_%H%M%S)_$$.log"
(
  export CUDA_VISIBLE_DEVICES="$XR1_REPLAY_SERVER_GPU"
  cd "$XR1_SAFE_REPO"
  exec "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
    --model "$XR1_REPLAY_MODEL" --host 127.0.0.1 --port "$XR1_REPLAY_PORT"
) > "$server_log" 2>&1 &
server_pid=$!
export XR1_REPLAY_SERVER_PID="$server_pid"
deadline=$((SECONDS + 1800))
until "$XR1_CLIENT_ENV/bin/python" - "$XR1_REPLAY_PORT" <<'PY'
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
"$XR1_CLIENT_ENV/bin/python" -u -m "$replay_module" "${common[@]}"
