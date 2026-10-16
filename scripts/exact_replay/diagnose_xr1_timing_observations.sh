#!/usr/bin/env bash
# Fixed recorded actions, two observation comparisons and a separate render probe.
set -euo pipefail
action=${1:-status}
export STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
export ROBOCASA_REPO=${ROBOCASA_REPO:-/gs/fs/tga-shinoda/felid/robocasa}
latest="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_observations_latest.env"
if [[ "$action" == status && -z "${XR1_OBS_ROOT:-}" ]]; then
  source "$latest"
fi
export XR1_CLIENT_ENV=${XR1_CLIENT_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365}
if [[ "$action" == status ]]; then
  "$XR1_CLIENT_ENV/bin/python" - "$XR1_OBS_ROOT/diagnostic" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
if (root/"status.json").exists():
    print("status:", json.loads((root/"status.json").read_text()))
else:
    print("No diagnostic status yet; inspect the driver log.")
if (root/"report.json").exists():
    report = json.loads((root/"report.json").read_text())
    print("finding:", report["finding"])
    print("policy calls:", report["policy_calls"])
    if "cases" in report:
        print("startup order evidence:", report["startup_order_evidence"])
        print("CUDA initialization order changed:", report["cuda_initialization_order_changed"])
        for name, case in report["cases"].items():
            print("case:", name, case["finding"])
            print("first observation difference:", case["repeat_comparison"]["first_observation_difference_step"])
            print("first environment difference:", case["repeat_comparison"]["first_non_observation_difference_step"])
            print("startup events:", case["startup_events"])
            print("first-step camera statistics:", case["camera_statistics_at_step_1"])
        for name in ("first_environment_cross_case_comparison", "warmed_control_vs_first_early_environment"):
            comparison = report[name]
            print(name + ":", {k: comparison[k] for k in (
                "first_observation_difference_step", "first_non_observation_difference_step")})
        print("interpretation limit:", report["interpretation_limit"])
        print("report:", root/"report.json")
        raise SystemExit(0)
    comparison = report["repeat_comparison"]
    for field in ("first_observation_difference_step", "first_non_observation_difference_step",
                  "first_difference_by_state_channel", "first_difference_by_observation_key"):
        print(field + ":", comparison[field])
    if comparison["first_observation_difference"]:
        print("first observation difference:", comparison["first_observation_difference"])
    print("difference from recorded source:", report["first_observation_difference_from_recorded_source"])
    print("render probe prefix:", {k: report["render_probe_prefix_comparison"][k] for k in (
        "first_observation_difference_step", "first_non_observation_difference_step")})
    probe = report["render_probe"]
    print("render probe:", {k: v for k, v in probe.items() if k != "renders"})
    for row in probe.get("renders", []):
        print("rerender", row["render_index"], "changed keys:", row["changed_observation_keys"])
    print("report:", root/"report.json")
PY
  echo "driver log: ${XR1_OBS_LOG:-unknown}"
  exit 0
fi
export XR1_OBS_SOURCE=${XR1_OBS_SOURCE:-${XR1_REPLAY_ROOT:?Source the completed replay run.env or set XR1_OBS_SOURCE}/diagnostic}
export XR1_OBS_GPU=${XR1_OBS_GPU:-0}
export XR1_OBS_PROBE_STEP=${XR1_OBS_PROBE_STEP:-233}
export XR1_OBS_COMPARE_INIT_ORDER=${XR1_OBS_COMPARE_INIT_ORDER:-0}
extra=()
if [[ "$XR1_OBS_COMPARE_INIT_ORDER" == 1 ]]; then
  extra+=(--compare-init-order)
elif [[ "$XR1_OBS_COMPARE_INIT_ORDER" != 0 ]]; then
  echo "XR1_OBS_COMPARE_INIT_ORDER must be 0 or 1"; exit 1
fi
export CUDA_VISIBLE_DEVICES="$XR1_OBS_GPU" MUJOCO_EGL_DEVICE_ID=0
export HF_HOME="$STORAGE_BS/hf_home" TRANSFORMERS_CACHE="$STORAGE_BS/hf_home/transformers"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
cd "$ROBOCASA_REPO"
if [[ "$action" == start ]]; then
  export XR1_OBS_ROOT=${XR1_OBS_ROOT:-$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_observations_$(date +%Y%m%d_%H%M%S)}
  test -x "$XR1_CLIENT_ENV/bin/python"
  "$XR1_CLIENT_ENV/bin/python" - "$STORAGE_BS" "$XR1_OBS_ROOT" "$XR1_OBS_SOURCE" <<'PY'
from pathlib import Path
import shutil, sys
storage, root, source = (Path(p).resolve() for p in sys.argv[1:])
if not root.is_relative_to(storage) or root == storage:
    raise SystemExit("Outputs must be in a new directory under durable STORAGE_BS")
if root == source or root.is_relative_to(source) or source.is_relative_to(root):
    raise SystemExit("Output root must be separate from the source diagnostic")
if root.exists():
    raise SystemExit("Output root exists; select a new XR1_OBS_ROOT")
if shutil.disk_usage(storage).free < 10 * 2**30:
    raise SystemExit("Less than 10 GiB free; inspect durable storage before starting")
PY
  "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.exact_replay.diagnose_intervention_timing_observations \
    --source-diagnostic-dir "$XR1_OBS_SOURCE" --output-dir "$XR1_OBS_ROOT/diagnostic" \
    --probe-step "$XR1_OBS_PROBE_STEP" "${extra[@]}" --dry-run
  mkdir -p "$XR1_OBS_ROOT" "$STORAGE_BS/robocasa_logs/eval"
  export XR1_OBS_LOG="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_OBS_ROOT")_driver.log"
  for name in STORAGE_BS ROBOCASA_REPO XR1_OBS_SOURCE XR1_OBS_ROOT XR1_OBS_GPU XR1_OBS_PROBE_STEP XR1_OBS_COMPARE_INIT_ORDER XR1_OBS_LOG XR1_CLIENT_ENV; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done > "$XR1_OBS_ROOT/run.env"
  cp "$XR1_OBS_ROOT/run.env" "$latest"
  nohup bash "$ROBOCASA_REPO/scripts/exact_replay/diagnose_xr1_timing_observations.sh" worker > "$XR1_OBS_LOG" 2>&1 &
  echo "driver_pid=$!"
  echo "run=$XR1_OBS_ROOT"
  echo "log=$XR1_OBS_LOG"
  exit 0
fi
[[ "$action" == worker ]] || { echo "Usage: $0 {start|status}"; exit 1; }
exec 9>"$XR1_OBS_ROOT/.driver.lock"
flock -n 9 || { echo "Another driver owns this diagnostic root"; exit 1; }
exec "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.exact_replay.diagnose_intervention_timing_observations \
  --source-diagnostic-dir "$XR1_OBS_SOURCE" --output-dir "$XR1_OBS_ROOT/diagnostic" \
  --probe-step "$XR1_OBS_PROBE_STEP" "${extra[@]}"
