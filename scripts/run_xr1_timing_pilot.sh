#!/usr/bin/env bash
# Bounded timing pilot: reuse valid v4 evidence, exclude invalid parents, run all timing cells.
set -euo pipefail
action=${1:-status}
export STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
export ROBOCASA_REPO=${ROBOCASA_REPO:-/gs/fs/tga-shinoda/felid/robocasa}
latest="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_pilot_latest.env"
case "$action" in start|resume|status|worker|review) ;; *) echo "Usage: $0 {start|resume|status|review review.json}"; exit 1 ;; esac
if [[ "$action" != start && "$action" != worker && -z "${XR1_PILOT_ROOT:-}" ]]; then
  source "$latest"
fi
export XR1_CLIENT_ENV=${XR1_CLIENT_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365}
export XR1_SERVER_ENV=${XR1_SERVER_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_server}
export XR1_SAFE_REPO=${XR1_SAFE_REPO:-/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe}
export XR1_SAFE_CHECKPOINT=${XR1_SAFE_CHECKPOINT:-$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe}
export XR1_PILOT_SOURCE=${XR1_PILOT_SOURCE:-$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_rngfix_20261012_194551/full}
export XR1_PILOT_GPU=${XR1_PILOT_GPU:-0} XR1_PILOT_SERVER_GPU=${XR1_PILOT_SERVER_GPU:-1}
export XR1_PILOT_PORT=${XR1_PILOT_PORT:-10332}
export XR1_PILOT_REFERENCES=${XR1_PILOT_REFERENCES:-$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_references_observer_v4_20261013_220743/references}
export XR1_REFS_SERVER_GPU="$XR1_PILOT_SERVER_GPU"
if [[ "$action" == status ]]; then
  "$XR1_CLIENT_ENV/bin/python" - "$XR1_PILOT_ROOT/pilot" "${XR1_PILOT_LOG:-unknown}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
status = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {"status": "starting"}
print("run:", root)
print("status:", status)
failure = root.parent/"driver_failure.json"
if failure.exists():
    saved = json.loads(failure.read_text())
    if saved.get("driver_log", sys.argv[2]) == sys.argv[2]:
        print("driver status: failed", saved)
        print("Inspect the driver log even if collection status still says starting/running.")
    else:
        print("previous driver failure preserved:", saved)
for name in ("report.json", "scientific_analysis.json"):
    if (root/name).exists():
        data = json.loads((root/name).read_text())
        print(name, {k: data[k] for k in ("status", "coverage_by_task", "analysis_parents_by_task", "image_exclusions_by_task", "excluded_parents", "timing_errors_by_operator", "timing_collection_complete", "pilot_scientific_analysis_ready", "decision") if k in data})
for phase in ("references", "qualification"):
    ledger = root/(phase+"_records.jsonl")
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    print(phase, "attempted:", len(rows), "excluded:", sum(r["kind"] == "excluded" for r in rows))
    for row in rows:
        if row["kind"] == "excluded":
            print("excluded:", row["parent"]["task_name"], row["parent"]["environment_seed"], row["reason"])
bid = status.get("current_branch")
if bid:
    folder = (root/"branch_attempts"/bid).resolve()
    if not folder.is_relative_to(root):
        raise SystemExit("Branch path escaped the pilot root")
    for attempt in sorted(folder.glob("attempt_*/status.json")):
        print("branch progress:", json.loads(attempt.read_text()))
if (root/"camera_review.html").exists():
    print("camera gallery:", root/"camera_review.html")
PY
  echo "driver log: ${XR1_PILOT_LOG:-unknown}"
  exit 0
fi
cd "$ROBOCASA_REPO"
if [[ "$action" == review ]]; then
  test -n "${2:-}" || { echo "Provide the completed image review JSON path"; exit 1; }
  "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.run_intervention_timing_pilot \
    --timing-run-dir "$XR1_PILOT_SOURCE" --output-dir "$XR1_PILOT_ROOT/pilot" \
    --model-path "$XR1_PILOT_MODEL" --review-file "$2"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$XR1_PILOT_GPU" MUJOCO_EGL_DEVICE_ID=0
export HF_HOME="$STORAGE_BS/hf_home" TRANSFORMERS_CACHE="$STORAGE_BS/hf_home/transformers"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
if [[ "$action" == start || "$action" == resume ]]; then
  export XR1_PILOT_ROOT=${XR1_PILOT_ROOT:-$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_pilot_$(date +%Y%m%d_%H%M%S)}
  test -x "$XR1_CLIENT_ENV/bin/python"
  test -x "$XR1_SERVER_ENV/bin/python"
  test -f "$XR1_PILOT_SOURCE/registration.json"
  test -f "$XR1_SAFE_REPO/deploy/server.py"
  [[ "$XR1_PILOT_GPU" != "$XR1_PILOT_SERVER_GPU" ]] || { echo "Use separate simulator and server GPUs"; exit 1; }
  "$XR1_CLIENT_ENV/bin/python" - "$STORAGE_BS" "$XR1_PILOT_ROOT" "$XR1_PILOT_SOURCE" "$action" <<'PY'
from pathlib import Path
import json, sys
storage, root, source = (Path(p).resolve() for p in sys.argv[1:4])
if not root.is_relative_to(storage) or root == storage:
    raise SystemExit("Pilot outputs must be in a new directory under durable STORAGE_BS")
if root == source or root.is_relative_to(source) or source.is_relative_to(root):
    raise SystemExit("Pilot root must be separate from the frozen source")
if root.exists() and sys.argv[4] == "start":
    raise SystemExit("Pilot root exists; choose a new XR1_PILOT_ROOT")
if sys.argv[4] == "resume" and not (root/"pilot/registration.json").is_file():
    raise SystemExit("Missing pilot registration for resume")
if sys.argv[4] == "resume" and (root/"pilot/fatal_errors.jsonl").exists():
    raise SystemExit("Recorded fatal pilot failure: preserve this root and diagnose; do not retry")
if sys.argv[4] == "resume" and (root/"driver_failure.json").exists():
    failure = json.loads((root/"driver_failure.json").read_text())
    if failure.get("exit_code") not in (130, 143):
        raise SystemExit("Recorded driver failure: preserve this root and diagnose; only interrupted runs may resume")
PY
  export XR1_PILOT_MODEL
  XR1_PILOT_MODEL=$("$XR1_CLIENT_ENV/bin/python" - "$XR1_PILOT_SOURCE/registration.json" <<'PY'
import json, pathlib, sys
registration = json.loads(pathlib.Path(sys.argv[1]).read_text())
plan = json.loads((pathlib.Path(registration["phase2_run_dir"])/"plan.json").read_text())
print(plan["checkpoint_provenance"]["path"])
PY
)
  mkdir -p "$XR1_PILOT_ROOT" "$STORAGE_BS/robocasa_logs/eval"
  export XR1_PILOT_LOG="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_PILOT_ROOT")_driver_$(date +%Y%m%d_%H%M%S)_$$.log"
  export XR1_PILOT_RESUME=0
  [[ "$action" != resume ]] || export XR1_PILOT_RESUME=1
  for name in STORAGE_BS ROBOCASA_REPO XR1_PILOT_SOURCE XR1_PILOT_REFERENCES XR1_PILOT_ROOT XR1_PILOT_GPU XR1_PILOT_SERVER_GPU XR1_PILOT_PORT XR1_PILOT_MODEL XR1_PILOT_LOG XR1_PILOT_RESUME XR1_CLIENT_ENV XR1_SERVER_ENV XR1_SAFE_REPO XR1_SAFE_CHECKPOINT; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done > "$XR1_PILOT_ROOT/run.env"
  cp "$XR1_PILOT_ROOT/run.env" "$latest"
  nohup bash "$ROBOCASA_REPO/scripts/run_xr1_timing_pilot.sh" worker > "$XR1_PILOT_LOG" 2>&1 &
  echo "driver_pid=$!"
  echo "run=$XR1_PILOT_ROOT"
  echo "log=$XR1_PILOT_LOG"
  exit 0
fi
exec 9>"$XR1_PILOT_ROOT/.driver.lock"
flock -n 9 || { echo "Another driver owns this pilot root"; exit 1; }
server_pid=""
cleanup() {
  local code=$?
  trap - EXIT
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ "$code" != 0 ]]; then
    "$XR1_CLIENT_ENV/bin/python" - "$XR1_PILOT_ROOT" "$code" <<'PY'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
log = os.environ["XR1_PILOT_LOG"]
record = json.dumps({"exit_code": int(sys.argv[2]), "inspect_driver_log": True, "driver_log": log})
failures = root/"driver_failures"
failures.mkdir(exist_ok=True)
(failures/(pathlib.Path(log).name+".json")).write_text(record)
(root/"driver_failure.json").write_text(record)
print("Pilot driver failed; inspect the saved driver log and status.")
PY
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$XR1_CLIENT_ENV/bin/python" - "$STORAGE_BS" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 50 * 2**30:
    raise SystemExit("Less than 50 GiB free for full state/image traces; inspect durable storage")
PY
if [[ "$XR1_PILOT_MODEL" == /tmp/ut06746/* ]]; then
  mkdir -p "$XR1_PILOT_MODEL"
  rsync -a --info=progress2 "$XR1_SAFE_CHECKPOINT/" "$XR1_PILOT_MODEL/"
fi
common=(--timing-run-dir "$XR1_PILOT_SOURCE" --reference-run-dir "$XR1_PILOT_REFERENCES" --output-dir "$XR1_PILOT_ROOT/pilot"
        --model-path "$XR1_PILOT_MODEL" --host 127.0.0.1 --port "$XR1_PILOT_PORT")
[[ "$XR1_PILOT_RESUME" != 1 ]] || common+=(--resume)
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.run_intervention_timing_pilot "${common[@]}" --dry-run
if ss -ltn | awk '{print $4}' | grep -Eq ":${XR1_PILOT_PORT}$"; then
  echo "Port $XR1_PILOT_PORT is occupied; choose a unique XR1_PILOT_PORT"; exit 1
fi
server_log="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_PILOT_ROOT")_server_$(date +%Y%m%d_%H%M%S)_$$.log"
(
  export CUDA_VISIBLE_DEVICES="$XR1_PILOT_SERVER_GPU"
  cd "$XR1_SAFE_REPO"
  exec "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
    --model "$XR1_PILOT_MODEL" --host 127.0.0.1 --port "$XR1_PILOT_PORT"
) > "$server_log" 2>&1 &
server_pid=$!
export XR1_PILOT_SERVER_PID="$server_pid" XR1_REFS_SERVER_PID="$server_pid"
deadline=$((SECONDS + 1800))
until "$XR1_CLIENT_ENV/bin/python" - "$XR1_PILOT_PORT" <<'PY'
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
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.run_intervention_timing_pilot "${common[@]}"
