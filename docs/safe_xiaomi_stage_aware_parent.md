# Full-parent stage-aware SAFE for Xiaomi composite tasks

This experiment tests whether composite-task SAFE improves when training keeps
the complete parent trajectory and uses the outcome of the active semantic
stage. It does **not** reset the temporal representation at stage boundaries.
Every row is one genuine Xiaomi policy inference.

The label contract is:

- a completed semantic stage is a stage success (`failure=0`);
- the terminal active stage of a failed parent is a stage failure (`failure=1`);
- a terminal failed stage reached only after the final cached action chunk has
  no causal policy inference and is reported as censored / non-estimable;
- future stages that were never attempted are censored and create no rows;
- the final parent outcome is retained only as an auxiliary/baseline target.

The frozen comparison contains:

| Arm | Input / target |
|---|---|
| `terminal` | full-parent causal SAFE summary, parent terminal target |
| `stage` | full-parent causal SAFE summary, eventual active-stage target |
| `multihorizon` | stage target plus within-4/8/16-inference heads |
| `conditioned` | multihorizon plus task, stage, elapsed-stage and progress conditioning |
| `context` | conditioned arm plus causal Xiaomi observation/state history |
| `prototype` | success-stage prototype distance |
| `time_only` | stage-conditioned elapsed-time baseline |

Stage support, duration scales, feature normalizers, regularization, refit
epochs, score normalization, prototypes, time curves, and thresholds are all
fit without the locked outer parents. Thresholds use successful calibration
**parents** as the conformal unit, so the target FPR is a successful-parent FPR
rather than a per-stage FPR. The locked 80-parent pilot outer split is never
scored. A performance claim requires a new seed/reset-identity-disjoint raw
collection and the frozen `evaluate` phase.

## 1. Fresh-session setup

Read `docs/cluster_experiment_runbook.md` first. The commands below assume the
Xiaomi SAFE checkpoint/server patch from `docs/safe_xiaomi_robotics_1.md` is
already installed.

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export XR1_SAFE_REPO="$PROJECT_FS/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe"
export SAFE_REPO="$PROJECT_FS/SAFE"
export XR1_SERVER_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_server"
export XR1_CLIENT_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365"
export SAFE_ENV="$STORAGE_BS/envs/vla_safe"
export XR1_PY="$XR1_CLIENT_ENV/bin/python"
export SAFE_PY="$SAFE_ENV/bin/python"
export XR1_SAFE_CHECKPOINT="$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe"
export XR1_SAFE38_FINAL="$STORAGE_BS/robocasa_checkpoints/safe/xr1_pooled_39tasks_final_refits_20260814_231508"
export XR1_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval"

mkdir -p "$XR1_LOG_ROOT" /tmp/ut06746
cd "$ROBOCASA_REPO"
git pull --ff-only origin codex/safe-xiaomi-robotics-1

test -x "$XR1_PY"
test -x "$SAFE_PY"
test -f "$XR1_SAFE_CHECKPOINT/config.json"
test -d "$SAFE_REPO/.git"
test -f "$XR1_SAFE38_FINAL/indep_seed0/model_final.ckpt"
test -f robocasa/recovery/safe/stage_aware_parent_safe.py
test -f robocasa/recovery/safe/run_stage_aware_parent_safe.py
"$SAFE_PY" -c 'import numpy, torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
"$SAFE_PY" -m robocasa.recovery.safe.run_stage_aware_parent_safe --help
df -h "$STORAGE_BS"
df -ih "$STORAGE_BS"
```

Stop before training or collection if CUDA is unavailable, the checkpoint is
missing, or storage is unexpectedly full.

## 2. Developmental run on the existing five-task collection

This is the fastest first run. It compares `terminal`, `stage`,
`multihorizon`, `conditioned`, `prototype`, and `time_only` using the existing
250 raw parents and their existing 170/80 parent split. The old tensors do not
contain the newly added observation/state channel, so `context` is deliberately
omitted. This run must not be presented as a new outer-test result.

```bash
export XR1_EXISTING_COLLECTION="$STORAGE_BS/robocasa_rollouts/safe/xr1_dense_subtask_live_5tasks_50each_20260817_195216"
export XR1_EXISTING_RAW="$XR1_EXISTING_COLLECTION/subtask_training/merged_5tasks_50each"
export XR1_EXISTING_SPLIT="$XR1_EXISTING_COLLECTION/subtask_training/parent_rollout_split.json"

test -f "$XR1_EXISTING_RAW/manifest.jsonl"
test -f "$XR1_EXISTING_SPLIT"

export XR1_STAGE_DRY="$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_aware_parent_dry_$(date +%Y%m%d_%H%M%S)"
cd "$ROBOCASA_REPO"
"$SAFE_PY" -u -m robocasa.recovery.safe.run_stage_aware_parent_safe develop \
  --dataset-dir "$XR1_EXISTING_RAW" \
  --selection-manifest "$XR1_EXISTING_SPLIT" \
  --output-dir "$XR1_STAGE_DRY" \
  --tasks \
    ArrangeBreadBasket ArrangeTea BreadSelection \
    CuttingToolSelection GarnishPancake \
  --arms terminal stage multihorizon conditioned prototype \
  --failure-horizons 4 8 16 \
  --prefixes 1 2 4 8 \
  --primary-prefixes 1 2 \
  --dry-run

"$SAFE_PY" -m json.tool "$XR1_STAGE_DRY/status.json"
"$SAFE_PY" -m json.tool "$XR1_STAGE_DRY/protocol.json" | sed -n '1,180p'
```

Require `status=dry_run_valid`, `opened_outer_scored=false`, five tasks, and a
nonempty fixed stage set.

Run a bounded one-seed/one-regularizer GPU smoke:

```bash
export XR1_STAGE_SMOKE="$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_aware_parent_smoke_$(date +%Y%m%d_%H%M%S)"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 "$SAFE_PY" -u -m \
  robocasa.recovery.safe.run_stage_aware_parent_safe develop \
  --dataset-dir "$XR1_EXISTING_RAW" \
  --selection-manifest "$XR1_EXISTING_SPLIT" \
  --output-dir "$XR1_STAGE_SMOKE" \
  --tasks \
    ArrangeBreadBasket ArrangeTea BreadSelection \
    CuttingToolSelection GarnishPancake \
  --arms terminal stage multihorizon conditioned prototype \
  --failure-horizons 4 8 16 \
  --regularizations 0.001 \
  --seeds 0 \
  --epochs 5 --patience 3 --batch-size 256 \
  --device cuda

"$SAFE_PY" -m robocasa.recovery.safe.print_stage_aware_parent_safe \
  --run-dir "$XR1_STAGE_SMOKE"
```

Only after that smoke completes, launch the full developmental comparison:

```bash
export XR1_STAGE_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_aware_parent_5tasks_$(date +%Y%m%d_%H%M%S)"
export XR1_STAGE_LOG="$XR1_LOG_ROOT/$(basename "$XR1_STAGE_ROOT").log"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 \
nohup "$SAFE_PY" -u -m robocasa.recovery.safe.run_stage_aware_parent_safe develop \
  --dataset-dir "$XR1_EXISTING_RAW" \
  --selection-manifest "$XR1_EXISTING_SPLIT" \
  --output-dir "$XR1_STAGE_ROOT" \
  --tasks \
    ArrangeBreadBasket ArrangeTea BreadSelection \
    CuttingToolSelection GarnishPancake \
  --arms terminal stage multihorizon conditioned prototype \
  --failure-horizons 4 8 16 \
  --prefixes 1 2 4 8 --primary-prefixes 1 2 \
  --regularizations 0.00001 0.0001 0.001 0.01 \
  --seeds 0 1 2 \
  --epochs 500 --patience 50 --batch-size 256 \
  --device cuda \
  > "$XR1_STAGE_LOG" 2>&1 &
export XR1_STAGE_PID=$!

printf 'export XR1_STAGE_ROOT=%q\nexport XR1_STAGE_LOG=%q\nexport XR1_STAGE_PID=%q\n' \
  "$XR1_STAGE_ROOT" "$XR1_STAGE_LOG" "$XR1_STAGE_PID" \
  > "$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_aware_parent_latest.env"
```

One-shot monitor (rerun it; do not use `watch` if the terminal does not accept
Ctrl-C):

```bash
source "$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_aware_parent_latest.env"
date
ps -fp "$XR1_STAGE_PID" || true
printf 'screen fits: '; grep -c '^SCREEN ' "$XR1_STAGE_LOG" 2>/dev/null || true
printf 'refits: '; grep -c '^REFIT ' "$XR1_STAGE_LOG" 2>/dev/null || true
"$SAFE_PY" -m json.tool "$XR1_STAGE_ROOT/status.json" 2>/dev/null || true
tail -n 40 "$XR1_STAGE_LOG" 2>/dev/null
du -sh "$XR1_STAGE_ROOT" 2>/dev/null
```

The expected full count is 48 neural screens (four neural arms, four
regularizers, three seeds) and 12 neural refits. The prototype and time-only
arms do not train neural checkpoints.

Print final developmental results:

```bash
source "$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_aware_parent_latest.env"
"$SAFE_PY" -m robocasa.recovery.safe.print_stage_aware_parent_safe \
  --run-dir "$XR1_STAGE_ROOT"
```

### 2.1 One-time retrospective check on the locked 80 parents

After the runtime bundle is frozen, score the exact locked outer IDs once.
These parents were not loaded by the stage-aware fit, selection, or calibration
phases, so this is a useful held-out sanity check. Their outcomes were already
examined in earlier terminal-versus-subtask experiments, however, so this run
is explicitly retrospective and cannot replace the new prospective collection.
Do not refit, change stages, change normalization, update thresholds, or decide
whether to run the prospective test from this result.

```bash
export XR1_STAGE_RUNTIME="$XR1_STAGE_ROOT/runtime_bundle.json"
export XR1_STAGE_LOCKED_OUTER="$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_aware_locked_outer_$(date +%Y%m%d_%H%M%S)"

test -f "$XR1_STAGE_RUNTIME"
test ! -e "$XR1_STAGE_LOCKED_OUTER"

# The runtime bundle is authoritative. Deriving this path prevents accidentally
# passing the parent collection directory, which has no top-level manifest.
export XR1_EXISTING_RAW="$("$SAFE_PY" - "$XR1_STAGE_RUNTIME" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["source_dataset"])
PY
)"

test -f "$XR1_EXISTING_RAW/manifest.jsonl" || {
  echo "STOP: frozen source manifest is missing: $XR1_EXISTING_RAW/manifest.jsonl"
  exit 1
}

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 "$SAFE_PY" -u -m \
  robocasa.recovery.safe.run_stage_aware_parent_safe evaluate \
  --dataset-dir "$XR1_EXISTING_RAW" \
  --runtime-bundle "$XR1_STAGE_RUNTIME" \
  --output-dir "$XR1_STAGE_LOCKED_OUTER" \
  --opened-outer \
  --bootstrap-replicates 2000 --bootstrap-seed 0 \
  --device cuda

"$SAFE_PY" -m robocasa.recovery.safe.print_stage_aware_parent_safe \
  --run-dir "$XR1_STAGE_LOCKED_OUTER"
```

Require `evaluation_scope=retrospective_locked_opened_outer`,
`prospective_claim=false`, exactly 80 parents, and
`thresholds_updated_on_test=false`.

## 3. Fresh context-enabled training collection

The `context` arm cannot be reconstructed from old tensors. The updated Xiaomi
client now stores its causal, zero-padded observation state history once per
true policy inference under
`auxiliary__observation_state_history`. Collect a fresh five-task dataset with
semantic traces. Keep all 50 rollouts per task.

Start two already-patched Xiaomi SAFE servers:

```bash
export XR1_PORT0=10206
export XR1_PORT1=10207
export XR1_CONTEXT_TAG=xr1_stage_context_$(date +%Y%m%d_%H%M%S)
export XR1_SERVER0_LOG="$XR1_LOG_ROOT/${XR1_CONTEXT_TAG}_server0.log"
export XR1_SERVER1_LOG="$XR1_LOG_ROOT/${XR1_CONTEXT_TAG}_server1.log"

cd "$XR1_SAFE_REPO"
CUDA_VISIBLE_DEVICES=0 nohup "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
  --model "$XR1_SAFE_CHECKPOINT" --host 127.0.0.1 --port "$XR1_PORT0" \
  > "$XR1_SERVER0_LOG" 2>&1 &
export XR1_SERVER0_PID=$!

CUDA_VISIBLE_DEVICES=1 nohup "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
  --model "$XR1_SAFE_CHECKPOINT" --host 127.0.0.1 --port "$XR1_PORT1" \
  > "$XR1_SERVER1_LOG" 2>&1 &
export XR1_SERVER1_PID=$!

until ss -ltn | grep -q ":$XR1_PORT0" && ss -ltn | grep -q ":$XR1_PORT1"; do
  date
  tail -n 10 "$XR1_SERVER0_LOG" 2>/dev/null
  tail -n 10 "$XR1_SERVER1_LOG" 2>/dev/null
  sleep 10
done
grep -q 'Model loaded' "$XR1_SERVER0_LOG"
grep -q 'Model loaded' "$XR1_SERVER1_LOG"
ss -ltnp | grep -E ":$XR1_PORT0|:$XR1_PORT1"
nvidia-smi
```

Run one short infrastructure smoke and verify the auxiliary tensor exists:

```bash
export XR1_CONTEXT_SMOKE="$STORAGE_BS/robocasa_rollouts/safe/xr1_stage_context_smoke_$(date +%Y%m%d_%H%M%S)"
cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
"$XR1_PY" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_CONTEXT_SMOKE" \
  --tasks ArrangeTea --num-rollouts 1 \
  --seed 880007 --seed-protocol official_xiaomi \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 \
  --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 --port "$XR1_PORT0" \
  --split pretrain --horizon 20 --replan-steps 16 \
  --record-safe-features --record-actions --no-record-videos \
  --record-subtask-trace --max-errors 1

"$XR1_PY" - "$XR1_CONTEXT_SMOKE" <<'PY'
import json
import pathlib
import sys
import numpy as np
root = pathlib.Path(sys.argv[1])
record = json.loads(next(line for line in (root / "manifest.jsonl").read_text().splitlines() if line))
with np.load(root / record["tensor_path"], allow_pickle=False) as payload:
    value = payload["auxiliary__observation_state_history"]
    print("context shape:", value.shape)
    assert value.ndim == 2 and value.shape[0] == record["valid_sequence_length"]
    assert np.isfinite(value).all()
print("XR1 CONTEXT COLLECTION SMOKE: VALID")
PY
```

Launch 50 parents per task. Do not set success/failure quotas and do not discard
rollouts.

```bash
export XR1_CONTEXT_RAW="$STORAGE_BS/robocasa_rollouts/safe/xr1_stage_context_5tasks_50each_$(date +%Y%m%d_%H%M%S)"
export XR1_CONTEXT_LOG0="$XR1_LOG_ROOT/${XR1_CONTEXT_TAG}_collect0.log"
export XR1_CONTEXT_LOG1="$XR1_LOG_ROOT/${XR1_CONTEXT_TAG}_collect1.log"
mkdir -p "$XR1_CONTEXT_RAW/shard0" "$XR1_CONTEXT_RAW/shard1"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup "$XR1_PY" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_CONTEXT_RAW/shard0" \
  --tasks ArrangeBreadBasket BreadSelection GarnishPancake \
  --num-rollouts 50 --seed 900007 --seed-protocol official_xiaomi \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 --port "$XR1_PORT0" --split pretrain --replan-steps 16 \
  --record-safe-features --record-actions --no-record-videos \
  --record-subtask-trace --continue-on-error --max-errors 5 \
  > "$XR1_CONTEXT_LOG0" 2>&1 &
export XR1_CONTEXT_PID0=$!

CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
nohup "$XR1_PY" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_CONTEXT_RAW/shard1" \
  --tasks ArrangeTea CuttingToolSelection \
  --num-rollouts 50 --seed 900007 --seed-protocol official_xiaomi \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 --port "$XR1_PORT1" --split pretrain --replan-steps 16 \
  --record-safe-features --record-actions --no-record-videos \
  --record-subtask-trace --continue-on-error --max-errors 5 \
  > "$XR1_CONTEXT_LOG1" 2>&1 &
export XR1_CONTEXT_PID1=$!

printf 'export XR1_CONTEXT_RAW=%q\nexport XR1_CONTEXT_LOG0=%q\nexport XR1_CONTEXT_LOG1=%q\nexport XR1_CONTEXT_PID0=%q\nexport XR1_CONTEXT_PID1=%q\nexport XR1_PORT0=%q\nexport XR1_PORT1=%q\n' \
  "$XR1_CONTEXT_RAW" "$XR1_CONTEXT_LOG0" "$XR1_CONTEXT_LOG1" \
  "$XR1_CONTEXT_PID0" "$XR1_CONTEXT_PID1" "$XR1_PORT0" "$XR1_PORT1" \
  > "$STORAGE_BS/robocasa_rollouts/safe/xr1_stage_context_latest.env"
```

One-shot collection monitor:

```bash
source "$STORAGE_BS/robocasa_rollouts/safe/xr1_stage_context_latest.env"
date
ps -fp "$XR1_CONTEXT_PID0" "$XR1_CONTEXT_PID1" || true
for shard in shard0 shard1; do
  printf '%s rollouts: ' "$shard"
  wc -l < "$XR1_CONTEXT_RAW/$shard/manifest.jsonl" 2>/dev/null || echo 0
done
tail -n 15 "$XR1_CONTEXT_LOG0" 2>/dev/null
tail -n 15 "$XR1_CONTEXT_LOG1" 2>/dev/null
du -sh "$XR1_CONTEXT_RAW" 2>/dev/null
```

After both collectors stop, validate, merge by hard link, and freeze a raw
parent split without materializing segmented SAFE records:

```bash
source "$STORAGE_BS/robocasa_rollouts/safe/xr1_stage_context_latest.env"
for shard in shard0 shard1; do
  "$XR1_PY" -m robocasa.recovery.safe.validate_atomic_dataset \
    --dataset-dir "$XR1_CONTEXT_RAW/$shard"
done

export XR1_CONTEXT_MERGED="$XR1_CONTEXT_RAW/merged_250"
"$XR1_PY" -m robocasa.recovery.safe.merge_atomic_datasets \
  --source-dirs "$XR1_CONTEXT_RAW/shard0" "$XR1_CONTEXT_RAW/shard1" \
  --output-dir "$XR1_CONTEXT_MERGED"

export XR1_CONTEXT_SPLIT_ROOT="$XR1_CONTEXT_RAW/full_parent_split"
"$SAFE_PY" -m robocasa.recovery.safe.run_stage_aware_parent_safe split \
  --dataset-dir "$XR1_CONTEXT_MERGED" \
  --output-dir "$XR1_CONTEXT_SPLIT_ROOT" \
  --tasks \
    ArrangeBreadBasket ArrangeTea BreadSelection \
    CuttingToolSelection GarnishPancake \
  --train-fraction 0.68 --split-seed 0 --require-context

export XR1_CONTEXT_SPLIT="$XR1_CONTEXT_SPLIT_ROOT/parent_rollout_split.json"
"$SAFE_PY" -m json.tool "$XR1_CONTEXT_SPLIT_ROOT/status.json"
```

The expected split is 250 total parents, approximately 170 development and 80
locked holdout. Exact counts can differ slightly because each task/outcome
stratum is rounded independently.

### 3.1 Mandatory frozen confirmation, including the SAFE38 baseline

Do this before fitting the context arm. The new trace-enabled collection is a
valid prospective test for the already frozen five-task stage-aware bundle.
The older SAFE38 rollouts cannot provide stage ground truth because they were
recorded without semantic traces. They can nevertheless be used cleanly by
applying the already frozen 38-task MLP to this new five-task export. That
external detector is a secondary fixed-prefix comparator: it is never refit,
normalized, or thresholded on the prospective parents.

The current development thresholds had only 18 successful calibration parents,
so matched-FPR event results remain descriptive. The primary confirmation is
the prefix-1/2 task-stage macro ROC-AUC and its predeclared ranking gates.

```bash
export XR1_FROZEN_STAGE_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_aware_parent_5tasks_20260823_162943"
export XR1_FROZEN_STAGE_RUNTIME="$XR1_FROZEN_STAGE_ROOT/runtime_bundle.json"
export XR1_CONTEXT_EXPORT="$XR1_CONTEXT_RAW/official_safe_all_retained"
export XR1_SAFE38_SCORE_ROOT="$XR1_CONTEXT_RAW/frozen_safe38_indep_scores"
export XR1_FROZEN_CONFIRM="$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_aware_frozen_confirmation_$(date +%Y%m%d_%H%M%S)"

test -f "$XR1_FROZEN_STAGE_RUNTIME"
test -f "$XR1_CONTEXT_MERGED/manifest.jsonl"
test ! -e "$XR1_CONTEXT_EXPORT"
test ! -e "$XR1_FROZEN_CONFIRM"

cd "$ROBOCASA_REPO"
"$XR1_PY" -m robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir "$XR1_CONTEXT_MERGED" \
  --output-dir "$XR1_CONTEXT_EXPORT"

mkdir -p "$XR1_SAFE38_SCORE_ROOT"
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 "$SAFE_PY" -u -m \
    robocasa.recovery.safe.score_seen_checkpoint_external \
    --export-dir "$XR1_CONTEXT_EXPORT" \
    --safe-repo "$SAFE_REPO" \
    --training-run-dir "$XR1_SAFE38_FINAL/indep_seed$seed" \
    --output-dir "$XR1_SAFE38_SCORE_ROOT/seed$seed" \
    --group prospective_test \
    --allow-task-subset \
    --device cuda
done

CUDA_VISIBLE_DEVICES=0 "$SAFE_PY" -u -m \
  robocasa.recovery.safe.run_stage_aware_parent_safe evaluate \
  --dataset-dir "$XR1_CONTEXT_MERGED" \
  --runtime-bundle "$XR1_FROZEN_STAGE_RUNTIME" \
  --output-dir "$XR1_FROZEN_CONFIRM" \
  --external-scores \
    "safe38_indep=$XR1_SAFE38_SCORE_ROOT/seed0,$XR1_SAFE38_SCORE_ROOT/seed1,$XR1_SAFE38_SCORE_ROOT/seed2" \
  --bootstrap-replicates 2000 --bootstrap-seed 0 \
  --device cuda

"$SAFE_PY" -m robocasa.recovery.safe.print_stage_aware_parent_safe \
  --run-dir "$XR1_FROZEN_CONFIRM"
```

The evaluator requires exact rollout-ID agreement across all three SAFE38
score files, checks seed/reset identity disjointness from development, averages
the frozen seed scores inference-by-inference, and labels them explicitly as
fixed-prefix-only external scores. The four primary gates remain conditioned
ROC at least 0.60, conditioned-minus-terminal at least +0.05, and conditioned
beating terminal and time-only on at least four of five tasks. SAFE38 is
reported alongside those gates but does not replace either comparator.

Run the same dry-run and smoke gates as Section 2, now using
`$XR1_CONTEXT_MERGED`, `$XR1_CONTEXT_SPLIT`, and all six arms. Then launch the
full run with:

```bash
export XR1_CONTEXT_TRAIN="$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_context_full_$(date +%Y%m%d_%H%M%S)"
export XR1_CONTEXT_TRAIN_LOG="$XR1_LOG_ROOT/$(basename "$XR1_CONTEXT_TRAIN").log"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 \
nohup "$SAFE_PY" -u -m robocasa.recovery.safe.run_stage_aware_parent_safe develop \
  --dataset-dir "$XR1_CONTEXT_MERGED" \
  --selection-manifest "$XR1_CONTEXT_SPLIT" \
  --output-dir "$XR1_CONTEXT_TRAIN" \
  --tasks \
    ArrangeBreadBasket ArrangeTea BreadSelection \
    CuttingToolSelection GarnishPancake \
  --arms terminal stage multihorizon conditioned context prototype \
  --failure-horizons 4 8 16 \
  --prefixes 1 2 4 8 --primary-prefixes 1 2 \
  --regularizations 0.00001 0.0001 0.001 0.01 \
  --seeds 0 1 2 --epochs 500 --patience 50 --batch-size 256 \
  --device cuda \
  > "$XR1_CONTEXT_TRAIN_LOG" 2>&1 &
export XR1_CONTEXT_TRAIN_PID=$!

printf 'export XR1_CONTEXT_TRAIN=%q\nexport XR1_CONTEXT_TRAIN_LOG=%q\nexport XR1_CONTEXT_TRAIN_PID=%q\n' \
  "$XR1_CONTEXT_TRAIN" "$XR1_CONTEXT_TRAIN_LOG" "$XR1_CONTEXT_TRAIN_PID" \
  > "$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_context_train_latest.env"
```

Expected: 60 neural screens and 15 neural refits. Do not inspect or score the
locked raw holdout.

## 4. Frozen prospective evaluation

After development completes, freeze
`$XR1_CONTEXT_TRAIN/runtime_bundle.json`. Collect a second five-task dataset
with the same commands from Section 3 but a new output root and base seed
`950007`. Do not reuse `900007`, and do not select rollouts by outcome. Validate
and merge its two shards into `$XR1_PROSPECTIVE_MERGED`.

The evaluator rejects overlapping rollout IDs and overlapping
`(task, environment_seed, environment_reset_index)` identities. It never
updates a model, normalizer, prototype, time curve, or threshold.

```bash
export XR1_RUNTIME="$XR1_CONTEXT_TRAIN/runtime_bundle.json"
export XR1_PROSPECTIVE_RESULT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_stage_context_prospective_$(date +%Y%m%d_%H%M%S)"

test -f "$XR1_RUNTIME"
test -f "$XR1_PROSPECTIVE_MERGED/manifest.jsonl"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 "$SAFE_PY" -u -m \
  robocasa.recovery.safe.run_stage_aware_parent_safe evaluate \
  --dataset-dir "$XR1_PROSPECTIVE_MERGED" \
  --runtime-bundle "$XR1_RUNTIME" \
  --output-dir "$XR1_PROSPECTIVE_RESULT" \
  --bootstrap-replicates 2000 --bootstrap-seed 0 \
  --device cuda

"$SAFE_PY" -m robocasa.recovery.safe.print_stage_aware_parent_safe \
  --run-dir "$XR1_PROSPECTIVE_RESULT"
```

The primary success criterion is not merely ROC-AUC. At the frozen matched
successful-parent FPR, the selected stage-aware arm should have a paired
miss-adjusted detection fraction at least 0.05 earlier than `time_only`, with a
task/parent bootstrap 95% interval below zero, while its failed-stage TPR is no
more than 0.05 below time-only. The report also retains stage-level balanced
accuracy, per-task/per-stage metrics, 10/25/50% failure recall, every stage
event prediction, and every parent prediction.
