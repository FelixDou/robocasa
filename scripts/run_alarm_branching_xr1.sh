#!/usr/bin/env bash
# Phase A of docs/recovery_harvest_finetune_plan.md: recoverability at the Hide-and-Seek + P alarm.
#   export | score | plan | start | resume | status | analyze | audit | decide
# audit/decide are step R0 of the redesign (failure audit; simulator only, no policy server).
# export and score are CPU-only (SAFE env). start/resume reuse the ladder worker
# (one XR-1 server GPU + one simulator GPU + one port per worker).
set -euo pipefail
action=${1:-status}
case "$action" in export|score|plan|start|resume|status|analyze|audit|decide) ;; *)
  echo "Usage: $0 {export|score|plan|start|resume|status|analyze|audit|decide}"; exit 1 ;; esac

export STORAGE_BS=${STORAGE_BS:-/gs/bs/tga-shinoda/felid}
export ROBOCASA_REPO=${ROBOCASA_REPO:-/gs/fs/tga-shinoda/felid/robocasa}
export SAFE_PY=${SAFE_PY:-$STORAGE_BS/envs/vla_safe/bin/python}
export XR1_CLIENT_ENV=${XR1_CLIENT_ENV:-$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365}
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
# Inputs (defaults: the Amendment 2 run, the E1 lead calibration and the ladder v2 pilot).
export AMEND2_ROOT=${AMEND2_ROOT:-$STORAGE_BS/robocasa_checkpoints/safe/xr1_amend2_hide_and_seek_20261022_214209}
export E1_OUT=${E1_OUT:-$STORAGE_BS/robocasa_rollouts/safe/xr1_lead_calibration_20261020_001108/lead_analysis}
export ALARM_SOURCE_ROOT=${ALARM_SOURCE_ROOT:-$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_v2_pilot_20261017_164537}
export ALARM_K=${ALARM_K:-8}
export ALARM_OFFSETS=${ALARM_OFFSETS:-"8 16"}
# Fixed branch budget (fraction of the time limit), independent of the alarm's clock.
export ALARM_BUDGET_MODE=${ALARM_BUDGET_MODE:-fixed_fraction}
export ALARM_BUDGET_FRACTION=${ALARM_BUDGET_FRACTION:-0.5}
export LADDER_GPU_PAIRS=${LADDER_GPU_PAIRS:-"0:1 2:3"}
export LADDER_PORT_BASE=${LADDER_PORT_BASE:-10380}
latest="$STORAGE_BS/robocasa_rollouts/recovery/xr1_alarm_phaseA_latest.env"
log_dir="$STORAGE_BS/robocasa_logs/eval"
mkdir -p "$log_dir"
cd "$ROBOCASA_REPO"
# shellcheck disable=SC2206
sources=(${ALARM_SOURCE_RUNS:-"$ALARM_SOURCE_ROOT"/shard*})
# shellcheck disable=SC2206
offsets=($ALARM_OFFSETS)

cpu_env() { export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; }

if [[ "$action" == export ]]; then
  export ALARM_SPEC=${ALARM_SPEC:-$STORAGE_BS/robocasa_checkpoints/safe/hs_p_alarm_spec_$(date +%Y%m%d_%H%M%S)}
  [[ ! -e "$ALARM_SPEC" ]] || { echo "ALARM_SPEC exists: $ALARM_SPEC"; exit 1; }
  cpu_env
  log="$log_dir/$(basename "$ALARM_SPEC")_export.log"
  reuse=()
  # Detectors trained by an earlier export (its detectors/ dir) skip retraining.
  [[ -z "${ALARM_REUSE_DETECTORS:-}" ]] || reuse=(--reuse-detectors "$ALARM_REUSE_DETECTORS")
  PYTHONPATH="$ROBOCASA_REPO" nohup nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.hs_p_alarm export "${reuse[@]}" \
    --amendment-scores-dir "$AMEND2_ROOT/scores" --reference-cohort confirmatory_test \
    --e1-calibration "$E1_OUT/calibration_predictions.jsonl" \
    --hs-calibration-scores "$AMEND2_ROOT/scores/lead_calibration_predictions.jsonl" \
    --hs-result "$AMEND2_ROOT/confirm_hide_and_seek.json" --output-dir "$ALARM_SPEC" > "$log" 2>&1 &
  echo "export pid=$! log=$log ALARM_SPEC=$ALARM_SPEC"
  exit 0
fi

if [[ "$action" == score ]]; then
  : "${ALARM_SPEC:?set ALARM_SPEC to the export output}"
  grep -q '"kind": "hs_p_alarm_spec"' "$ALARM_SPEC/alarm_spec.json" || { echo "No spec in $ALARM_SPEC"; exit 1; }
  export ALARM_ROOT=${ALARM_ROOT:-$STORAGE_BS/robocasa_rollouts/recovery/xr1_alarm_phaseA_$(date +%Y%m%d_%H%M%S)}
  mkdir -p "$ALARM_ROOT"
  cpu_env
  PYTHONPATH="$ROBOCASA_REPO" nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.hs_p_alarm score-ladder \
    --spec-dir "$ALARM_SPEC" --run-dir "${sources[@]}" --output "$ALARM_ROOT/alarms.jsonl" \
    2>&1 | tee "$log_dir/$(basename "$ALARM_ROOT")_score.log"
  for name in STORAGE_BS ROBOCASA_REPO ALARM_SPEC ALARM_ROOT ALARM_SOURCE_ROOT ALARM_K ALARM_OFFSETS ALARM_BUDGET_MODE ALARM_BUDGET_FRACTION LADDER_GPU_PAIRS LADDER_PORT_BASE; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done > "$ALARM_ROOT/run.env"
  printf 'export ALARM_SOURCE_RUNS=%q\n' "${sources[*]}" >> "$ALARM_ROOT/run.env"
  cp "$ALARM_ROOT/run.env" "$latest"
  echo "ALARM_ROOT=$ALARM_ROOT"
  exit 0
fi

[[ -n "${ALARM_ROOT:-}" ]] || source "$latest"
# shellcheck disable=SC2206
sources=(${ALARM_SOURCE_RUNS:-"$ALARM_SOURCE_ROOT"/shard*})
alarms="$ALARM_ROOT/alarms.jsonl"
branch_root="$ALARM_ROOT/branches"
module=robocasa.recovery.alarm_branching
py="$XR1_CLIENT_ENV/bin/python"

if [[ "$action" == plan ]]; then
  exec "$py" -m "$module" plan --source-run "${sources[@]}" --alarms "$alarms" \
    --k-branches "$ALARM_K" --offsets "${offsets[@]}" \
    --budget-mode "$ALARM_BUDGET_MODE" --budget-fraction "$ALARM_BUDGET_FRACTION"
fi
if [[ "$action" == status ]]; then
  echo "ALARM_ROOT=$ALARM_ROOT"
  for shard in "$branch_root"/shard*; do
    echo "== $(basename "$shard")"
    test -f "$shard/status.json" && cat "$shard/status.json" || echo "no status yet"
    test -f "$shard/errors.jsonl" && echo "errors: $(wc -l < "$shard/errors.jsonl")"
    test -f "$shard/worker_failure.json" && { echo "WORKER FAILED:"; cat "$shard/worker_failure.json"; }
  done
  exit 0
fi
if [[ "$action" == analyze ]]; then
  exec "$py" -m "$module" analyze --run-dirs "$branch_root"/shard* --output-dir "$ALARM_ROOT/gate_a" \
    --k-branches "$ALARM_K"
fi

audit_dir="$ALARM_ROOT/r0_audit"
if [[ "$action" == audit ]]; then
  # One simulator GPU (rendering for the camera images); no policy server.
  gpu=${ALARM_AUDIT_GPU:-0}
  resume=()
  [[ ! -e "$audit_dir/audit.json" ]] || resume=(--resume)
  log="$log_dir/$(basename "$ALARM_ROOT")_r0_audit_$(date +%Y%m%d_%H%M%S).log"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" nohup "$py" -u -m robocasa.recovery.failure_audit audit \
    --source-run "${sources[@]}" --alarms "$alarms" --branch-runs "$branch_root"/shard* \
    --output-dir "$audit_dir" "${resume[@]}" > "$log" 2>&1 &
  echo "audit pid=$! gpu=$gpu log=$log out=$audit_dir"
  exit 0
fi
if [[ "$action" == decide ]]; then
  review=()
  [[ -z "${ALARM_REVIEW:-}" ]] || review=(--review "$ALARM_REVIEW")
  exec "$py" -m robocasa.recovery.failure_audit decide --audit-dir "$audit_dir" "${review[@]}"
fi

# start | resume: one ladder worker per GPU pair, writing to $branch_root/shard<i>.
test -f "$alarms" || { echo "Missing $alarms (run score first)"; exit 1; }
extra=(--source-run "${sources[@]}" --alarms "$alarms" --k-branches "$ALARM_K" --offsets "${offsets[@]}"
       --budget-mode "$ALARM_BUDGET_MODE" --budget-fraction "$ALARM_BUDGET_FRACTION")
if [[ "$action" == start ]]; then
  [[ ! -e "$branch_root" ]] || { echo "$branch_root exists; use resume"; exit 1; }
else
  test -d "$branch_root" || { echo "Missing $branch_root"; exit 1; }
  for shard in "$branch_root"/shard*; do rm -f "$shard/worker_failure.json"; done
  extra+=(--resume)
fi
mkdir -p "$branch_root"
git -C "$ROBOCASA_REPO" rev-parse HEAD > "$branch_root/git_commit.txt"
export LADDER_ROOT="$branch_root" LADDER_WORKER_KIND=alarm_branch
# shellcheck disable=SC2206
pairs=($LADDER_GPU_PAIRS)
nshards=${#pairs[@]}
for shard in "${!pairs[@]}"; do
  gpu=${pairs[$shard]%%:*}; server_gpu=${pairs[$shard]##*:}
  [[ "$gpu" != "$server_gpu" ]] || { echo "Use separate simulator and server GPUs ($gpu)"; exit 1; }
  port=$((LADDER_PORT_BASE + shard))
  export LADDER_WORKER_LOG="$log_dir/$(basename "$ALARM_ROOT")_branch_shard${shard}_$(date +%Y%m%d_%H%M%S).log"
  nohup bash "$ROBOCASA_REPO/scripts/run_recoverability_ladder_xr1.sh" worker \
    "$shard" "$nshards" "$gpu" "$server_gpu" "$port" "${extra[@]}" > "$LADDER_WORKER_LOG" 2>&1 &
  echo "shard=$shard pid=$! sim_gpu=$gpu server_gpu=$server_gpu port=$port log=$LADDER_WORKER_LOG"
done
