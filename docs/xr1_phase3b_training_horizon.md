# XR1 Phase 3B training-horizon continuation

> **Frozen 2026-10-16.** This track is closed and must not be extended; see
> [exact_replay_frozen.md](exact_replay_frozen.md). Code moved to
> `robocasa/recovery/exact_replay/` and `scripts/exact_replay/`; existing
> registrations must be resumed or reviewed from commit `4d65b542`.

This protocol extends the already validated Phase 2 branches without changing
parents, snapshots, candidate seeds, nominal seeds, policy, checkpoint, runtime
bundle, or the original first 64 environment steps. Read
`docs/cluster_experiment_runbook.md` and
`docs/xr1_phase2_complete_snapshot_replay.md` first.

The execution order is fixed:

1. run the four-snapshot engineering sentinel;
2. require every replay-integrity gate to pass;
3. run the complete 20-snapshot screen even if sentinel recovery outcomes are
   all zero;
4. analyze the full result and follow its registered routing decision.

## Completed result (12 October 2026)

The corrected engineering sentinel and full run completed successfully. The
full artifact is:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_phase3b_terminal_camera_v7_full20_20261012_091253
```

The engineering audit covered 10 parents, 20 snapshots, 140 long-horizon
branches, and 140 same-runtime prefix references with zero errors. Every
engineering gate passed. The scientific gate was estimable and failed:

- mixed candidate outcomes occurred in 1/20 snapshots and one ArrangeTea
  parent only;
- task-macro random-of-four stage completion was 0.2375 and oracle-best-of-four
  was 0.2500, for 0.0125 oracle headroom over random;
- nominal stage completion was 0.2000, so oracle-minus-nominal was 0.0500;
- candidate full-task success and safety termination were both 0.0000;
- CuttingToolSelection had no stage completions at its 256-step horizon.

The frozen routing decision is `higher_level_operator_screen`. Do not increase
candidate count, train a local candidate critic, or launch the formal five-task
candidate screen from this proposal family. Its higher-level follow-up has
now also completed: 160 scientific suffixes passed engineering, but
continuation reached 22.5% completion versus 17.5% for every intervention,
with zero oracle routing gain. See `docs/xr1_operator_screen.md` for the
completed result and `docs/xr1_earlier_intervention_timing_study.md` for the
next proposed common-stage-entry timing diagnostic. No oracle rewind anchors
were available, so the earlier 180-suffix proposal was not executed.

The sentinel is not a scientific stopping screen. It can block the full run
only for provenance, support, replay-prefix, repeat, restore, or artifact
failures.

Phase 2 schema-5 snapshots contain a legacy in-memory checksum that is
process-dependent for some diagnostic values. Phase 3B therefore keeps the
general snapshot loader strict but uses a registered compatibility contract:
the compressed snapshot must exactly match the SHA-256 frozen in the
checksummed Phase 3B registration, and its embedded snapshot ID, parent, task,
protocol, schema, and payload-checksum-derived identity must all agree. Every
accepted snapshot is written to `snapshot_integrity_audits.jsonl`; the
engineering sentinel fails unless this ledger covers all four snapshots.

The same legacy issue affects the per-step causal-environment hash because a
RoboSuite controller buffer can be a NumPy `dtype=object` array. The old digest
hashed its process-local object pointers. Phase 3B does not waive the 64-step
prefix gate. For every registered branch it now performs two passes:

1. a new 64-step same-runtime prefix reference using the frozen snapshot,
   branch kind, treatment seed, policy state, and action-generation protocol;
2. the registered training-horizon continuation.

The reference must match the frozen Phase 2 requests, actions, observations
consumed by actions 0--63, semantic trace, declared seed, and raw first causal
transition. The legacy causal hash must either match or have the complete
64-of-64 process-local mismatch signature. For the non-scientific
`environment_only` diagnostic alone, the unconsumed observation at index 64
may differ only in all three registered camera fields; this exact condition is
recorded explicitly. No state or proprioceptive difference is accepted. The
continuation must then match every fresh reference channel exactly, including
observation 64 and all 64 causal-environment states under the corrected
logical-value object-array digest (`stable_digest_schema_version=2`).
References are retained under `prefix_references/`; the full run remains
blocked by any failed qualification, missing reference, or
reference-to-continuation mismatch.

The frozen Phase 2 XML can also differ in whether mesh assets explicitly carry
`content_type="model/obj"`. Current MuJoCo serialization can omit or insert this
redundant MIME hint after `reset_from_xml_string()` when the same mesh already
has an `.obj` filename. The restore audit accepts only that exact bidirectional
normalization, and records omission, insertion, and total counts together with
the name digest, XML hashes, and a small name sample in
`environment_model_xml_audit`. Element order, tags, children, text, asset
paths, and every other attribute remain exact; any geometry, physics, visual,
path, or other XML change still fails closed. The final engineering analysis
separately gates complete coverage and reports the strict-versus-compatible
restore counts and total accepted MIME normalizations.

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
export XR1_PHASE2_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase2_shared_policy_2tasks_5each_20260825_014645"
export XR1_PHASE3B_PORT=10306
export XR1_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval"

export HF_HOME="$STORAGE_BS/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

cd "$ROBOCASA_REPO"
git pull --ff-only origin main

test -x "$XR1_SERVER_ENV/bin/python"
test -x "$XR1_CLIENT_ENV/bin/python"
test -f "$XR1_SAFE_REPO/deploy/server.py"
test -f "$XR1_PHASE2_ROOT/plan.json"
test -f "$XR1_PHASE2_ROOT/analysis.json"
test -f "$XR1_PHASE2_ROOT/branch_records.jsonl"
test -f "$XR1_PHASE2_ROOT/parent_records.jsonl"

export XR1_PHASE2_MODEL_PATH="$($XR1_CLIENT_ENV/bin/python - "$XR1_PHASE2_ROOT/plan.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["checkpoint_provenance"]["path"])
PY
)"

if [ ! -d "$XR1_PHASE2_MODEL_PATH" ]; then
  mkdir -p "$(dirname "$XR1_PHASE2_MODEL_PATH")"
  rsync -a --info=progress2 "$XR1_SAFE_CHECKPOINT/" "$XR1_PHASE2_MODEL_PATH/"
fi

df -h "$STORAGE_BS"
nvidia-smi
```

Stop if storage has no safe margin, the frozen source is incomplete, or the
node-local checkpoint cannot be recreated at the exact registered path.

## Start and verify the frozen Xiaomi server

```bash
mkdir -p "$XR1_LOG_ROOT"
export XR1_PHASE3B_SERVER_LOG="$XR1_LOG_ROOT/xr1_phase3b_server_$(date +%Y%m%d_%H%M%S).log"

cd "$XR1_SAFE_REPO"
CUDA_VISIBLE_DEVICES=0 nohup "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
  --model "$XR1_PHASE2_MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$XR1_PHASE3B_PORT" \
  > "$XR1_PHASE3B_SERVER_LOG" 2>&1 &
export XR1_PHASE3B_SERVER_PID=$!

until "$XR1_CLIENT_ENV/bin/python" - "$XR1_PHASE3B_PORT" <<'PY'
import socket, sys
with socket.socket() as connection:
    raise SystemExit(connection.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
do
  tail -n 20 "$XR1_PHASE3B_SERVER_LOG" 2>/dev/null || true
  sleep 10
done

grep -q 'Model loaded' "$XR1_PHASE3B_SERVER_LOG"
ps -fp "$XR1_PHASE3B_SERVER_PID"
ss -ltnp | grep ":$XR1_PHASE3B_PORT"
```

## Register and run the four-snapshot sentinel

The deterministic selector uses the lowest reset-index valid parent for each
task and keeps both its prefix and landmark snapshots. It reuses all seven
original branches per snapshot, for 28 continuations total. ArrangeTea runs to
480 environment steps and CuttingToolSelection to 256.

```bash
export XR1_PHASE3B_SENTINEL="$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_sentinel_$(date +%Y%m%d_%H%M%S)"
export XR1_PHASE3B_SENTINEL_LOG="$XR1_LOG_ROOT/$(basename "$XR1_PHASE3B_SENTINEL").log"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.exact_replay.run_phase3b_training_horizon_continuation \
  --phase2-run-dir "$XR1_PHASE2_ROOT" \
  --output-dir "$XR1_PHASE3B_SENTINEL" \
  --scope sentinel \
  --model-path "$XR1_PHASE2_MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$XR1_PHASE3B_PORT" \
  > "$XR1_PHASE3B_SENTINEL_LOG" 2>&1 &
export XR1_PHASE3B_SENTINEL_PID=$!

printf 'export STORAGE_BS=%q\nexport ROBOCASA_REPO=%q\nexport XR1_CLIENT_ENV=%q\nexport XR1_PHASE2_ROOT=%q\nexport XR1_PHASE2_MODEL_PATH=%q\nexport XR1_PHASE3B_PORT=%q\nexport XR1_PHASE3B_SENTINEL=%q\nexport XR1_PHASE3B_SENTINEL_LOG=%q\nexport XR1_PHASE3B_SENTINEL_PID=%q\n' \
  "$STORAGE_BS" "$ROBOCASA_REPO" "$XR1_CLIENT_ENV" "$XR1_PHASE2_ROOT" \
  "$XR1_PHASE2_MODEL_PATH" "$XR1_PHASE3B_PORT" "$XR1_PHASE3B_SENTINEL" \
  "$XR1_PHASE3B_SENTINEL_LOG" "$XR1_PHASE3B_SENTINEL_PID" \
  > "$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_sentinel_latest.env"
```

Monitor from another shell:

The printer reports `initializing_registration` while frozen source hashes are
still being built; this is startup progress rather than a failure.

```bash
export STORAGE_BS=/gs/bs/tga-shinoda/felid
source "$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_sentinel_latest.env"
cd "$ROBOCASA_REPO"
while kill -0 "$XR1_PHASE3B_SENTINEL_PID" 2>/dev/null; do
  clear
  date
  "$XR1_CLIENT_ENV/bin/python" -m \
    robocasa.recovery.exact_replay.print_phase3b_training_horizon \
    --run-dir "$XR1_PHASE3B_SENTINEL"
  tail -n 20 "$XR1_PHASE3B_SENTINEL_LOG"
  sleep 30
done

"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.exact_replay.print_phase3b_training_horizon \
  --run-dir "$XR1_PHASE3B_SENTINEL"
```

After engineering validity passes, freeze the sentinel diagnostic analysis:

```bash
"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.exact_replay.analyze_phase3b_training_horizon \
  --run-dir "$XR1_PHASE3B_SENTINEL" \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 0
```

Do not use sentinel completion, headroom, or routing output to cancel the full
screen.

## Register and run all twenty snapshots

```bash
export XR1_PHASE3B_FULL="$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_full20_$(date +%Y%m%d_%H%M%S)"
export XR1_PHASE3B_FULL_LOG="$XR1_LOG_ROOT/$(basename "$XR1_PHASE3B_FULL").log"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.exact_replay.run_phase3b_training_horizon_continuation \
  --phase2-run-dir "$XR1_PHASE2_ROOT" \
  --output-dir "$XR1_PHASE3B_FULL" \
  --scope full \
  --sentinel-run-dir "$XR1_PHASE3B_SENTINEL" \
  --model-path "$XR1_PHASE2_MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$XR1_PHASE3B_PORT" \
  > "$XR1_PHASE3B_FULL_LOG" 2>&1 &
export XR1_PHASE3B_FULL_PID=$!

printf 'export STORAGE_BS=%q\nexport ROBOCASA_REPO=%q\nexport XR1_CLIENT_ENV=%q\nexport XR1_PHASE2_ROOT=%q\nexport XR1_PHASE2_MODEL_PATH=%q\nexport XR1_PHASE3B_PORT=%q\nexport XR1_PHASE3B_SENTINEL=%q\nexport XR1_PHASE3B_FULL=%q\nexport XR1_PHASE3B_FULL_LOG=%q\nexport XR1_PHASE3B_FULL_PID=%q\n' \
  "$STORAGE_BS" "$ROBOCASA_REPO" "$XR1_CLIENT_ENV" "$XR1_PHASE2_ROOT" \
  "$XR1_PHASE2_MODEL_PATH" "$XR1_PHASE3B_PORT" "$XR1_PHASE3B_SENTINEL" \
  "$XR1_PHASE3B_FULL" "$XR1_PHASE3B_FULL_LOG" "$XR1_PHASE3B_FULL_PID" \
  > "$STORAGE_BS/robocasa_checkpoints/safe/xr1_phase3b_full_latest.env"
```

Use the same monitor with `XR1_PHASE3B_FULL` and `XR1_PHASE3B_FULL_PID`. After
all 140 long-horizon branches pass the engineering audit:

```bash
"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.exact_replay.analyze_phase3b_training_horizon \
  --run-dir "$XR1_PHASE3B_FULL" \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 0
```

The analysis writes `scientific_analysis.json`, snapshot/candidate JSONL, CSV,
task-macro estimands, genuine-replan completion curves, continuation gates,
and one registered routing decision. The completed candidate gate failed and
routed to the higher-level operator screen, which subsequently also failed.
Its results and the next proposed timing diagnostic are linked in the
completed-result section above; neither critic nor router training is unlocked.

## Resume

Restart the server, source the latest environment file, and rerun the identical
command with `--resume`. Completed branch IDs are skipped. The registration is
immutable and the runner re-hashes every frozen Phase 2 manifest, snapshot,
payload, runtime, checkpoint, and server entrypoint before continuing. A
first-64 mismatch fails closed and is retained in `errors.jsonl`; use a fresh
output root after any non-transient scientific-integrity failure.
