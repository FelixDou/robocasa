# XR1 Phase 2: complete simulator-policy snapshot replay

This experiment validates the identification layer required before collecting
counterfactual recovery data. It does **not** measure recovery quality. A branch
is admissible only when the simulator, wrapper episode counters, environment
RNG, Python/NumPy/Torch RNG, current observation, semantic state, Xiaomi image
and proprioception queues, cached action plan, instruction, counters, and
sampling schedule all round-trip.

Read `docs/cluster_experiment_runbook.md` and
`docs/safe_xiaomi_robotics_1.md` before running this protocol.

## Frozen protocol

- Tasks: `ArrangeTea` and `CuttingToolSelection`.
- Frozen segment / live semantic trigger mappings:
  - `ArrangeTea::PickPlaceCabinetToCounter_2_place` / `mug_on_tray`
  - `CuttingToolSelection::OpenDrawer_1` / `drawer_open`
- Eligible parents: five per task, using a new seed/reset family. Up to 15
  predeclared identities per task may be attempted because a nominal parent can
  terminate before reaching both diagnostic boundaries; ineligible attempts
  are retained and never contribute partial branches.
- Snapshots: active-stage policy-inference prefix 2 and the first inference at
  or beyond 25% of the fit-only successful-stage median horizon stored in the
  frozen stage-aware runtime bundle.
- Per snapshot: two identical-seed repeats, four explicitly seeded candidate
  branches, and one environment-only negative control.
- Expected support: 10 parents, 20 snapshots, 120 primary branches, 140 total
  branches.
- Default suffix: 64 environment steps. All real policy requests, full returned
  action chunks, actions, transitions, semantic traces, and outcomes are kept.

The environment-only control is diagnostic and is excluded from the primary
validity gates.

Two reachability pilots on 2026-08-24 exposed a namespace mismatch before any
valid Phase 2 result was opened.  Frozen training segments use names such as
`PickPlaceCabinetToCounter_2_place`, whereas live predicate tracking reports
`mug_on_tray`.  The apparent 0/11 reachability result is therefore invalid as
policy evidence: an instrumented parent subsequently spent 2,804 environment
steps and 175 policy inferences in `mug_on_tray`.  Those pilots remain retained
as invalid-instrumentation evidence and are never pooled with the corrected
protocol.  The runner now requires both a frozen `--target-stage` for its
fit-only horizon and a live `--trigger-stage` for snapshot capture.  Every
parent record also stores its observed stage sequence, per-stage visits,
environment steps, policy inferences, maximum consecutive inferences, terminal
outcome, and termination reason.

The original CuttingToolSelection placement boundary
(`PickPlaceDrawerToCounter_2_place` / `correct_tool_on_cutting_board`) reached
zero eligible parents in 8 bounded attempts. It was therefore replaced, in a
new protocol rather than by editing the failed run, with the earlier validated
`OpenDrawer_1` / `drawer_open` boundary. The resulting claim is explicitly
about this early stage and not the unreachable placement stage.

## Implemented artifacts

- `robocasa/recovery/full_snapshot.py`: versioned, checksummed snapshot files;
  exact restore and tolerance audit.
- `robocasa/recovery/xiaomi_robotics_1_policy.py`: explicit policy
  `get_state()` / `set_state()`, one-shot inference seeds, and request hashes.
- `robocasa/recovery/counterfactual_branch.py`: paired branch executor and
  preregistered engineering gates.
- `robocasa/recovery/run_phase2_snapshot_replay.py`: task/stage planner,
  collector, branch runner, evidence ledger, and final analysis.
- `robocasa/recovery/print_phase2_snapshot_replay.py`: safe running/final
  status printer; a failed scientific gate does not terminate the shell.

Snapshot and branch payloads use pickle and must only be loaded from trusted
experiment directories.

Branch schema 8 records a digest for every top-level observation field at
every suffix step. Raw observation sequences are retained only for same-seed
repeat branches, which keeps candidate storage bounded while allowing exact
per-camera and per-state diagnosis if the strict observation gate fails. This
instrumentation does not weaken the gate: full same-seed observation sequences
must still match before a Phase 2 replay run is valid.

The audit independently gates both the first request and the complete request
sequence. This prevents a replay with a matching initial cached action chunk
but a divergent later visual replan from passing request reproducibility.

For exact visual replay, `--canonical-camera-observations` bypasses RoboSuite's
cached multi-camera observable images. Each named camera is rendered directly
and synchronously from the current simulator state; the final image from two
consecutive readbacks is copied into the Xiaomi observation. The render mode
and repeat count are frozen in the run plan and checked on resume.

Each branch runs in a fresh environment and renderer by default
(`--fresh-branch-contexts`). The complete causal snapshot is restored only
after that context is initialized. This prevents sequential branches from
sharing offscreen-renderer or observable-cache history. The Xiaomi server
services one persistent client connection, so branches reuse the nominal
policy wrapper rather than opening competing sockets. Its captured queues,
action plan, instruction, counters, and sampling state are restored before
every branch, preserving policy-state isolation without a second client. The
environment-isolation mode and the `shared_restored` policy-connection mode are
frozen in the plan and checked on resume. Disabling fresh environments is a
diagnostic ablation only; it is not admissible for the Phase 2 validity claim.

Environment construction and reset run under a temporary deterministic Python
and NumPy seed derived from the frozen environment identity. This covers legacy
fixture code that samples visual properties, including counter collision-geom
colors, from module-level RNGs instead of the environment RNG. The caller RNG
streams are restored immediately afterward, and exact model-XML equality is
still required by the restore gate.

## Fresh-session exports and preflight

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export XR1_SAFE_REPO="$PROJECT_FS/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe"
export XR1_SERVER_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_server"
export XR1_CLIENT_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365"
export XR1_SAFE_CHECKPOINT="$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe"
export XR1_STAGE_RUNTIME="$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_context_full_20260824_154110/runtime_bundle.json"
export XR1_PHASE2_PORT=10306
export XR1_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval"

export HF_HOME="$STORAGE_BS/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

cd "$ROBOCASA_REPO"
git pull --ff-only origin codex/safe-xiaomi-robotics-1

test -x "$XR1_SERVER_ENV/bin/python"
test -x "$XR1_CLIENT_ENV/bin/python"
test -f "$XR1_SAFE_REPO/deploy/server.py"
test -f "$XR1_SAFE_CHECKPOINT/modeling_mibot.py"
test -f "$XR1_STAGE_RUNTIME"
grep -q 'sampling_seed' "$XR1_SAFE_REPO/deploy/server.py"
grep -q '"status": "frozen"' "$XR1_STAGE_RUNTIME"
df -h "$STORAGE_BS"
nvidia-smi
```

Do not continue if storage is full, the sampling-seed patch is absent, or the
runtime bundle is not frozen.

## Start and verify the Xiaomi server

```bash
export XR1_PHASE2_SERVER_LOG="$XR1_LOG_ROOT/xr1_phase2_server_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$XR1_LOG_ROOT"

cd "$XR1_SAFE_REPO"
CUDA_VISIBLE_DEVICES=0 nohup "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
  --model "$XR1_SAFE_CHECKPOINT" \
  --host 127.0.0.1 \
  --port "$XR1_PHASE2_PORT" \
  > "$XR1_PHASE2_SERVER_LOG" 2>&1 &
export XR1_PHASE2_SERVER_PID=$!

until "$XR1_CLIENT_ENV/bin/python" - "$XR1_PHASE2_PORT" <<'PY'
import socket
import sys
with socket.socket() as connection:
    raise SystemExit(connection.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
do
  tail -n 20 "$XR1_PHASE2_SERVER_LOG" 2>/dev/null
  sleep 10
done

grep -q 'Model loaded' "$XR1_PHASE2_SERVER_LOG"
ps -fp "$XR1_PHASE2_SERVER_PID"
ss -ltnp | grep ":$XR1_PHASE2_PORT"
nvidia-smi
```

## Deterministic dry run

```bash
export XR1_PHASE2_DRY="$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase2_snapshot_dry_$(date +%Y%m%d_%H%M%S)"
cd "$ROBOCASA_REPO"

"$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.run_phase2_snapshot_replay \
  --output-dir "$XR1_PHASE2_DRY" \
  --runtime-bundle "$XR1_STAGE_RUNTIME" \
  --tasks ArrangeTea CuttingToolSelection \
  --target-stage ArrangeTea=PickPlaceCabinetToCounter_2_place \
  --target-stage CuttingToolSelection=OpenDrawer_1 \
  --trigger-stage ArrangeTea=mug_on_tray \
  --trigger-stage CuttingToolSelection=drawer_open \
  --model-path "$XR1_SAFE_CHECKPOINT" \
  --checkpoint XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 \
  --server-repository "$XR1_SAFE_REPO" \
  --host 127.0.0.1 --port "$XR1_PHASE2_PORT" \
  --dry-run
```

Require 10 parents, 20 snapshots, 120 primary branches, and 140 total branches.
The dry run does not create its output directory or contact the server.

## One-parent live smoke

Use one task and one parent first. This is 2 snapshots and 14 total branches.

```bash
export XR1_PHASE2_SMOKE="$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase2_snapshot_smoke_$(date +%Y%m%d_%H%M%S)"
export XR1_PHASE2_SMOKE_LOG="$XR1_LOG_ROOT/$(basename "$XR1_PHASE2_SMOKE").log"
cd "$ROBOCASA_REPO"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
"$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.run_phase2_snapshot_replay \
  --output-dir "$XR1_PHASE2_SMOKE" \
  --runtime-bundle "$XR1_STAGE_RUNTIME" \
  --tasks ArrangeTea \
  --target-stage ArrangeTea=PickPlaceCabinetToCounter_2_place \
  --trigger-stage ArrangeTea=mug_on_tray \
  --num-parents-per-task 1 \
  --model-path "$XR1_SAFE_CHECKPOINT" \
  --checkpoint XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 \
  --server-repository "$XR1_SAFE_REPO" \
  --host 127.0.0.1 --port "$XR1_PHASE2_PORT" \
  --canonical-camera-observations \
  --fresh-branch-contexts \
  --seed 910007 \
  --ordinary-sampling-seed-base 6000000 \
  --candidate-seed-base 7000000 \
  2>&1 | tee "$XR1_PHASE2_SMOKE_LOG"

"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.print_phase2_snapshot_replay \
  --run-dir "$XR1_PHASE2_SMOKE"
```

Require every gate to pass before the full run. A gate failure is a stop and
diagnose result; do not immediately rerun or scale it away.

## Combine task-specific validation roots

If a predeclared live trigger is unreachable, preserve that run and declare a
replacement task-stage protocol in a new output root. Do not edit or merge the
source JSONL files. After both task-specific cohorts are complete, the combined
auditor selects only the explicitly named task from each source, verifies every
snapshot and branch payload, checks protocol compatibility and identifier
uniqueness, and reruns all engineering gates without copying raw artifacts.

```bash
export XR1_PHASE2_COMBINED="$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase2_combined_$(date +%Y%m%d_%H%M%S)"

"$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.audit_combined_phase2_snapshot_replay \
  --source "ArrangeTea=$XR1_PHASE2_ARRANGE_ROOT" \
  --source "CuttingToolSelection=$XR1_PHASE2_CUTTING_ROOT" \
  --parents-per-task 5 \
  --output-dir "$XR1_PHASE2_COMBINED"

"$XR1_CLIENT_ENV/bin/python" -m json.tool \
  "$XR1_PHASE2_COMBINED/analysis.json"
```

The output contains hashes and absolute paths for each immutable source, but no
snapshot or branch payload copies. A stopped multi-task source is admissible
only when the declared task itself has exactly the requested completed-parent,
snapshot, and branch support and all referenced artifacts are present.
Sources collected with and without fresh branch contexts, or with different
branch policy-connection modes, are intentionally protocol-incompatible and
cannot be combined.

## Full bounded validation

```bash
export XR1_PHASE2_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase2_snapshot_2tasks_5each_$(date +%Y%m%d_%H%M%S)"
export XR1_PHASE2_LOG="$XR1_LOG_ROOT/$(basename "$XR1_PHASE2_ROOT").log"
cd "$ROBOCASA_REPO"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.run_phase2_snapshot_replay \
  --output-dir "$XR1_PHASE2_ROOT" \
  --runtime-bundle "$XR1_STAGE_RUNTIME" \
  --tasks ArrangeTea CuttingToolSelection \
  --target-stage ArrangeTea=PickPlaceCabinetToCounter_2_place \
  --target-stage CuttingToolSelection=OpenDrawer_1 \
  --trigger-stage ArrangeTea=mug_on_tray \
  --trigger-stage CuttingToolSelection=drawer_open \
  --num-parents-per-task 5 \
  --max-parent-attempts-per-task 15 \
  --model-path "$XR1_SAFE_CHECKPOINT" \
  --checkpoint XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 \
  --server-repository "$XR1_SAFE_REPO" \
  --host 127.0.0.1 --port "$XR1_PHASE2_PORT" \
  --canonical-camera-observations \
  --fresh-branch-contexts \
  --seed 920007 \
  --ordinary-sampling-seed-base 6000000 \
  --candidate-seed-base 7000000 \
  > "$XR1_PHASE2_LOG" 2>&1 &
export XR1_PHASE2_PID=$!

printf 'XR1_PHASE2_ROOT=%s\nXR1_PHASE2_LOG=%s\nXR1_PHASE2_PID=%s\n' \
  "$XR1_PHASE2_ROOT" "$XR1_PHASE2_LOG" "$XR1_PHASE2_PID" \
  > "$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase2_latest.env"
```

Monitor without using a nested `watch` process:

```bash
source "$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase2_latest.env"
while kill -0 "$XR1_PHASE2_PID" 2>/dev/null; do
  clear
  date
  "$XR1_CLIENT_ENV/bin/python" -m \
    robocasa.recovery.print_phase2_snapshot_replay \
    --run-dir "$XR1_PHASE2_ROOT"
  tail -n 20 "$XR1_PHASE2_LOG"
  sleep 30
done

"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.print_phase2_snapshot_replay \
  --run-dir "$XR1_PHASE2_ROOT"
```

Stop the monitor with Ctrl-C; it does not stop the experiment.

If the allocation ends, restart the Xiaomi server in a new allocation, source
`xr1_phase2_latest.env`, repeat the fresh-session exports, and resume the same
frozen output root:

```bash
cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.run_phase2_snapshot_replay \
  --output-dir "$XR1_PHASE2_ROOT" \
  --runtime-bundle "$XR1_STAGE_RUNTIME" \
  --tasks ArrangeTea CuttingToolSelection \
  --target-stage ArrangeTea=PickPlaceCabinetToCounter_2_place \
  --target-stage CuttingToolSelection=OpenDrawer_1 \
  --trigger-stage ArrangeTea=mug_on_tray \
  --trigger-stage CuttingToolSelection=drawer_open \
  --num-parents-per-task 5 --max-parent-attempts-per-task 15 \
  --model-path "$XR1_SAFE_CHECKPOINT" \
  --checkpoint XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 \
  --server-repository "$XR1_SAFE_REPO" \
  --host 127.0.0.1 --port "$XR1_PHASE2_PORT" \
  --canonical-camera-observations \
  --fresh-branch-contexts \
  --seed 920007 \
  --ordinary-sampling-seed-base 6000000 \
  --candidate-seed-base 7000000 \
  --resume \
  >> "$XR1_PHASE2_LOG" 2>&1 &
export XR1_PHASE2_PID=$!
```

Resume validates the immutable plan and runtime hash, skips completed branch
IDs, completes a two-snapshot parent interrupted during branch execution, and
retains every earlier error. Add `--retry-errors` only after diagnosing a
transient parent error; the original error remains in the final error rate.

## Positive and negative meaning

A pass establishes that same-state branch contrasts are interpretable and that
the Xiaomi proposal distribution supplies non-identical action chunks. It does
not establish useful recovery opportunity; that is Phase 3.

Any failed exact request, action-chunk, first-transition, restore, alignment, or
support gate invalidates counterfactual collection. A suffix agreement below
95%, candidate diversity below 90%, restore-induced semantic regression, or
branch error rate at least 2% also stops progression to Phase 3. Preserve all
snapshots, branch payloads, status files, and errors for diagnosis.
