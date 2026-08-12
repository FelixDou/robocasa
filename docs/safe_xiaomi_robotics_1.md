# Original SAFE for Xiaomi-Robotics-1 on RoboCasa365

## Scope

This integration adapts the original binary-outcome SAFE pipeline to the
released Xiaomi-Robotics-1 RoboCasa365 policy. A rollout is labeled only by
official task success (`0` for success, `1` for natural failure). It does not
record, train, calibrate, or evaluate Subtask-SAFE.

The integration reuses the existing SAFE MLP/LSTM, functional conformal
calibration, official-loader export, and evaluation tools. Xiaomi features,
normalization, detector weights, thresholds, and calibration records remain a
separate model family; do not reuse π0 or RLDX detector artifacts.

Pinned released artifacts:

- SAFE source: `b6036abe07b2b2bb9996afb2c07f13d6a9f507c0`
- Xiaomi source: `4da1db0a4deefa6de7ebb4ef0b8754017290f5f7`
- checkpoint: `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`
- checkpoint revision: `0d1aa76d0d82debc9b611e4d1e231096434d5be4`
- Transformers: `4.57.1`

## Feature contract

XR-1 generates an action chunk with a five-step flow-matching DiT. At every
flow step, the integration records the final-layer action-token hidden states:

```python
hidden_states = self.dit(...)
hidden_states = hidden_states[:, -action_length:, :]
safe_features = hidden_states.detach().float()
output = self.action_output_layer(hidden_states)
```

The feature identifier is:

```text
dit_action_tokens_pre_action_output_layer
```

For the released RoboCasa365 checkpoint, one inference is expected to produce
float32 `(5, 30, 1024)` features: five flow steps, 30 action positions, and a
1,024-wide DiT state. Dimensions are validated from the live response rather
than hard-coded in the collector. A rollout stores:

```text
(num_policy_inferences, flow_steps, action_horizon, feature_dim)
```

The captured copy is converted to float32 only after the unchanged BF16 hidden
state has passed through `action_output_layer`, so feature capture does not
alter the action computation.

## Isolated patched source and checkpoint

Read `docs/cluster_experiment_runbook.md` first. Keep source on `/gs/fs` and
checkpoints, data, and logs on `/gs/bs`. Preserve the clean Xiaomi reproduction
checkout and official checkpoint by creating derived SAFE variants:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export XR1_REPO="$PROJECT_FS/robocasa_benchmark_repos/Xiaomi-Robotics-1"
export XR1_SAFE_REPO="$PROJECT_FS/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe"
export XR1_COMMIT=4da1db0a4deefa6de7ebb4ef0b8754017290f5f7

export XR1_SERVER_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_server"
export XR1_CLIENT_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365"
export XR1_CHECKPOINT="$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365"
export XR1_SAFE_CHECKPOINT="$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe"

export XR1_SERVER_PATCH="$ROBOCASA_REPO/patches/xiaomi_robotics_1_safe_server_4da1db0.patch"
export XR1_MODEL_PATCH="$ROBOCASA_REPO/patches/xiaomi_robotics_1_safe_model_0d1aa76.patch"

test "$(git -C "$XR1_REPO" rev-parse HEAD)" = "$XR1_COMMIT"
test -x "$XR1_SERVER_ENV/bin/python"
test -x "$XR1_CLIENT_ENV/bin/python"
test -f "$XR1_CHECKPOINT/modeling_mibot.py"
test -f "$XR1_SERVER_PATCH"
test -f "$XR1_MODEL_PATCH"
```

Create an isolated source worktree once:

```bash
if test -d "$XR1_SAFE_REPO/.git" || test -f "$XR1_SAFE_REPO/.git"; then
  test "$(git -C "$XR1_SAFE_REPO" rev-parse HEAD)" = "$XR1_COMMIT"
else
  git -C "$XR1_REPO" worktree add --detach "$XR1_SAFE_REPO" "$XR1_COMMIT"
fi

if git -C "$XR1_SAFE_REPO" apply --reverse --check "$XR1_SERVER_PATCH" 2>/dev/null; then
  echo "XR-1 SAFE server patch is already applied"
else
  git -C "$XR1_SAFE_REPO" apply --check "$XR1_SERVER_PATCH"
  git -C "$XR1_SAFE_REPO" apply "$XR1_SERVER_PATCH"
fi
```

Create a derived checkpoint using hard links for the immutable weight shards,
then break the link for the one Python file that will be patched:

```bash
if test ! -d "$XR1_SAFE_CHECKPOINT"; then
  cp -al "$XR1_CHECKPOINT" "$XR1_SAFE_CHECKPOINT"
  cp --remove-destination \
    "$XR1_CHECKPOINT/modeling_mibot.py" \
    "$XR1_SAFE_CHECKPOINT/modeling_mibot.py"
fi

if patch --batch --dry-run -R -p1 -d "$XR1_SAFE_CHECKPOINT" \
    < "$XR1_MODEL_PATCH" >/dev/null 2>&1; then
  echo "XR-1 SAFE model patch is already applied"
else
  patch --batch --dry-run -p1 -d "$XR1_SAFE_CHECKPOINT" \
    < "$XR1_MODEL_PATCH"
  patch --batch -p1 -d "$XR1_SAFE_CHECKPOINT" \
    < "$XR1_MODEL_PATCH"
fi

test -f "$XR1_SAFE_CHECKPOINT/model-00001-of-00003.safetensors"
grep -n "safe_feature_steps" "$XR1_SAFE_CHECKPOINT/modeling_mibot.py"
grep -n "request_safe_features" "$XR1_SAFE_REPO/deploy/server.py"
```

`cp -al` is safe here only because the patched Python file is immediately
replaced with a private inode before editing. The large safetensor shards stay
shared and unchanged.

## Start the opt-in SAFE server

Use a port distinct from any reproduction run. Normal requests still receive
the released action-only tensor; only requests with
`request_safe_features=True` receive the extended dictionary.

```bash
export XR1_SAFE_PORT=10096
export XR1_SAFE_SERVER_TAG=xr1_safe_server_$(date +%Y%m%d_%H%M%S)
export XR1_SAFE_SERVER_LOG="$STORAGE_BS/robocasa_logs/eval/${XR1_SAFE_SERVER_TAG}_${XR1_SAFE_PORT}.log"

mkdir -p "$STORAGE_BS/robocasa_logs/eval"

CUDA_VISIBLE_DEVICES=0 \
nohup "$XR1_SERVER_ENV/bin/python" -u \
  "$XR1_SAFE_REPO/deploy/server.py" \
  --model "$XR1_SAFE_CHECKPOINT" \
  --host 127.0.0.1 \
  --port "$XR1_SAFE_PORT" \
  > "$XR1_SAFE_SERVER_LOG" 2>&1 &

echo "server_pid=$!"
until "$XR1_CLIENT_ENV/bin/python" - "$XR1_SAFE_PORT" <<'PY'
import socket
import sys
with socket.socket() as connection:
    raise SystemExit(connection.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
do
  tail -30 "$XR1_SAFE_SERVER_LOG" 2>/dev/null || true
  sleep 10
done

grep -q "Model loaded" "$XR1_SAFE_SERVER_LOG"
```

## One-rollout original-SAFE smoke

This matches Xiaomi preprocessing: `pretrain` scenes, observation history 4 at
interval 2, crop ratio 0.95, 16 executed actions per query, target50 task
indexing, and base seed 7. The short horizon is only an infrastructure smoke;
task failure is an acceptable outcome.

```bash
export XR1_SAFE_SMOKE="$STORAGE_BS/robocasa_rollouts/safe/xr1_safe_smoke_$(date +%Y%m%d_%H%M%S)"
export XR1_SAFE_SMOKE_LOG="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_SAFE_SMOKE").log"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
"$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_SAFE_SMOKE" \
  --tasks CloseBlenderLid \
  --num-rollouts 1 \
  --seed 7 \
  --seed-protocol official_xiaomi \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 \
  --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 \
  --port "$XR1_SAFE_PORT" \
  --split pretrain \
  --horizon 20 \
  --replan-steps 16 \
  --record-safe-features \
  --record-actions \
  --no-record-videos \
  --no-record-subtask-trace \
  --max-errors 1 \
  2>&1 | tee "$XR1_SAFE_SMOKE_LOG"

"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$XR1_SAFE_SMOKE"
```

Inspect the actual tensor before any larger collection:

```bash
"$XR1_CLIENT_ENV/bin/python" - "$XR1_SAFE_SMOKE" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
record = json.loads(next(line for line in (root / "manifest.jsonl").read_text().splitlines() if line))
with np.load(root / record["tensor_path"], allow_pickle=False) as payload:
    features = payload["features"]
    print("model_family:", record["model_family"])
    print("feature_layer:", record["feature_layer"])
    print("shape:", features.shape)
    print("dtype:", features.dtype)
    print("finite:", np.isfinite(features).all())
    print("nonzero:", np.any(features))
    print("inference steps:", payload["inference_environment_steps"].tolist())
PY
```

The smoke passes only when the validator prints both `RoboCasa SAFE dataset:
VALID` and `Official SAFE loader compatible: True`, the tensor is finite and
nonzero, its expected tail shape is `(5, 30, 1024)`, and inference steps are
spaced by at least 16 environment actions.

## Plan collection from the official 2,500-rollout results

Use Xiaomi's task-level `eval_robocasa365/summary.json`, not a reduced local
pilot, to select collection tasks. The planner validates the pinned release
totals (50 tasks, 50 trials per task, and 1,432 successes), checks every task
and horizon against the local target50 registry, and uses an inclusive official
success-count interval. This experiment selects tasks with 5 through 45
successes out of 50, collects 50 new rollouts per task, keeps both classes at
their natural rate, and records no Subtask-SAFE trace.

```bash
export XR1_OFFICIAL_SUMMARY="$XR1_REPO/eval_robocasa365/summary.json"
export XR1_COLLECTION_TAG=xr1_safe_official_5to45_50each_$(date +%Y%m%d_%H%M%S)
export XR1_COLLECTION_ROOT="$STORAGE_BS/robocasa_rollouts/safe/$XR1_COLLECTION_TAG"
export XR1_COLLECTION_LATEST_ENV="$STORAGE_BS/robocasa_rollouts/safe/xr1_safe_official_5to45_latest.env"

cd "$ROBOCASA_REPO"
"$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.safe.plan_xiaomi_official_collection \
  --official-summary "$XR1_OFFICIAL_SUMMARY" \
  --output-dir "$XR1_COLLECTION_ROOT" \
  --latest-env "$XR1_COLLECTION_LATEST_ENV" \
  --min-successes 5 \
  --max-successes 45 \
  --rollouts-per-task 50 \
  --num-shards 2
```

The plan is valid only if it reports 39 eligible tasks, 1,950 expected new
rollouts, and a passing storage preflight. Preserve any earlier pilot-derived
plan as provenance, but do not launch it.

Launch one collector per verified SAFE server. Use a new base seed so these
rollouts cannot collide with the released evaluation or earlier SAFE batches.
There are deliberately no class quotas and no `--retain-only-quota`: every
valid success and failure is retained.

```bash
source "$XR1_COLLECTION_LATEST_ENV"
mapfile -t XR1_SHARD0_TASKS < "$XR1_SHARD0_TASK_FILE"
mapfile -t XR1_SHARD1_TASKS < "$XR1_SHARD1_TASK_FILE"

export XR1_COLLECTION_BASE_SEED=100007
export XR1_SHARD0_LOG="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_COLLECTION_ROOT")_shard0.log"
export XR1_SHARD1_LOG="$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_COLLECTION_ROOT")_shard1.log"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_SHARD0_DIR" \
  --tasks "${XR1_SHARD0_TASKS[@]}" \
  --num-rollouts "$XR1_COLLECTION_ROLLOUTS_PER_TASK" \
  --seed "$XR1_COLLECTION_BASE_SEED" \
  --seed-protocol official_xiaomi \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 \
  --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 \
  --port 10096 \
  --split pretrain \
  --replan-steps 16 \
  --record-safe-features \
  --record-actions \
  --no-record-videos \
  --no-record-subtask-trace \
  --continue-on-error \
  --max-errors 50 \
  > "$XR1_SHARD0_LOG" 2>&1 &
export XR1_SHARD0_PID=$!

CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$XR1_SHARD1_DIR" \
  --tasks "${XR1_SHARD1_TASKS[@]}" \
  --num-rollouts "$XR1_COLLECTION_ROLLOUTS_PER_TASK" \
  --seed "$XR1_COLLECTION_BASE_SEED" \
  --seed-protocol official_xiaomi \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --model-family xiaomi_robotics_1 \
  --policy-name Xiaomi-Robotics-1-RoboCasa365 \
  --checkpoint "$XR1_SAFE_CHECKPOINT" \
  --policy-config '{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}' \
  --host 127.0.0.1 \
  --port 10097 \
  --split pretrain \
  --replan-steps 16 \
  --record-safe-features \
  --record-actions \
  --no-record-videos \
  --no-record-subtask-trace \
  --continue-on-error \
  --max-errors 50 \
  > "$XR1_SHARD1_LOG" 2>&1 &
export XR1_SHARD1_PID=$!

printf 'export XR1_SHARD0_PID=%q\nexport XR1_SHARD1_PID=%q\nexport XR1_SHARD0_LOG=%q\nexport XR1_SHARD1_LOG=%q\n' \
  "$XR1_SHARD0_PID" "$XR1_SHARD1_PID" "$XR1_SHARD0_LOG" "$XR1_SHARD1_LOG" \
  > "$XR1_COLLECTION_ROOT/runtime.env"
```

Ports `10096` and `10097` must already be listening and each server log must
contain `Model loaded` before starting the collectors. Do not aim two parallel
collectors at one Xiaomi server.

## Dataset and detector workflow

After the one-rollout smoke, collect naturally successful and failed target50
rollouts with the same policy module and feature identity. Keep
`--no-record-subtask-trace`; class balance is enforced only with rollout-level
`--success-quota`, `--failure-quota`, and `--retain-only-quota`. Preserve and
resume partial shards rather than deleting them.

Export a completed validated dataset to the original SAFE loader layout:

```bash
"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir "$XR1_SAFE_DATASET" \
  --output-dir "$XR1_SAFE_EXPORT"
```

The report format is
`official_safe_xiaomi_robotics_1_env_records_policy_records`. The official
loader continues to receive each inference tensor under `pre_velocity`; that
field is a loader compatibility name, while `feature_layer` and
`model_family=xiaomi_robotics_1` preserve its true XR-1 identity.

For a multi-batch pool, `--rollouts-per-task 50 --selection-seed 0` selects
exactly 50 rollouts per task while preserving each pooled task's natural outcome
rate as closely as integer counts allow. Selection is deterministic within each
task and outcome class. This mode is mutually exclusive with the exact
success/failure balance options.

Build a fixed seen-task split before hyperparameter selection:

```bash
"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.safe.build_seen_task_split \
  --export-dir "$XR1_SAFE_EXPORT" \
  --output "$XR1_SAFE_SPLIT" \
  --test-per-class 1 \
  --num-inner-folds 3 \
  --seed 0
```

The builder includes only tasks with at least four examples of both outcomes:
one of each outcome remains untouched for the outer test and three remain in
the outer-training pool for outcome-stratified three-fold CV. Excluded tasks
remain in the immutable 50-per-task export and are named in the split manifest;
they are excluded only from model selection and held-out evaluation. Pass the
result with `--selection-manifest` to both `run_seen_cv_grid` and
`train_seen_tasks`.

Train MLP/LSTM and calibrate functional conformal thresholds from the exported
XR-1 dataset with the existing SAFE tools. Treat the result as a new XR-1
detector experiment until its task split, sample counts, seeds, hyperparameter
selection, calibration set, and held-out evaluation have all been recorded.

## Online failure-detection prefix comparison

Variable final rollout length is not a causal online feature: successful
rollouts stop at first success while failures usually reach the task timeout.
The seen-task CV and final-refit tools therefore expose two original-SAFE-only
views. Neither mode reads or creates subtask annotations.

`matched_success_length` pairs successes and failures within each task and
split, then truncates each failure to its paired success length. The complete
success prefix is retained. The two outcome classes consequently have exactly
the same inference-length distribution per task.

`fixed_landmark` estimates each task timeout from failures in the training
split only. It evaluates a declared fraction of that timeout and retains only
rollouts still active at that causal time. Training is rebalanced to equal
success/failure support within each task after this at-risk filter; evaluation
retains the natural at-risk support. A separate detector experiment is required
for every landmark fraction; do not select a landmark using the outer test.

Both transformations happen only after the source rollout split. The source
500-rollout export remains immutable, outer identities stay fixed, and all
generated labels remain the original binary final rollout outcome.

Use the same frozen outer split as the completed unmodified Xiaomi refit:

```bash
export XR1_SAFE_EXPORT="$STORAGE_BS/robocasa_rollouts/safe/xr1_safe_balanced_25x25_seed0_20260807_122434/official_safe_25x25_seed0"
export XR1_ORIGINAL_FINAL_ROOT=/path/to/completed/xiaomi/final_refits
export XR1_OUTER_SPLIT="$XR1_ORIGINAL_FINAL_ROOT/indep_seed0/split_manifest.json"
export SAFE_REPO="$PROJECT_FS/SAFE"

test -f "$XR1_SAFE_EXPORT/conversion_report.json"
test -f "$XR1_OUTER_SPLIT"
test -f "$SAFE_REPO/failure_prob/train.py"
```

Run a three-fold, one-configuration smoke for each view before a full grid. The
smoke intentionally does not open the outer test:

```bash
export XR1_ONLINE_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_online_safe_$(date +%Y%m%d_%H%M%S)"

for mode in matched_success_length fixed_landmark; do
  for model in indep lstm; do
    "$XR1_CLIENT_ENV/bin/python" -u -m \
      robocasa.recovery.safe.run_seen_cv_grid \
      --export-dir "$XR1_SAFE_EXPORT" \
      --safe-repo "$SAFE_REPO" \
      --output-root "$XR1_ONLINE_ROOT/${mode}_smoke" \
      --model "$model" \
      --outer-split-manifest "$XR1_OUTER_SPLIT" \
      --train-per-class 17 \
      --online-safe-mode "$mode" \
      --online-safe-seed 0 \
      --online-landmark-fraction 0.5 \
      --horizon-selectors 1.0 \
      --diffusion-selectors 1.0 \
      --learning-rates 1e-4 \
      --regularization 1e-3 \
      --num-folds 3 \
      --epochs 1 \
      --device cuda \
      --fail-fast
  done
done
```

Each smoke must complete six fits per mode. Inspect each `metrics.json` and
confirm:

- `outer_test_scored` is false;
- `online_safe.protocol.subtask_safe` is false;
- source train and validation parents are disjoint;
- matched-length per-task duration ROC-AUC is exactly 0.5;
- a fixed landmark reports both outcomes for every retained task.

For the full comparison, use distinct roots and the complete existing selector,
learning-rate, and regularization grids. Summarize each mode independently:

```bash
"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.safe.summarize_seen_cv \
  --root "$XR1_ONLINE_ROOT/matched_success_length_full"

"$XR1_CLIENT_ENV/bin/python" -m \
  robocasa.recovery.safe.summarize_seen_cv \
  --root "$XR1_ONLINE_ROOT/fixed_landmark_0p50_full"
```

Use each root's own `cv_selection_summary.json` for its final three-seed refits.
Pass the identical online options, outer split, and `--train-per-class 17` to
`train_seen_tasks`. The final command rejects a selection summary produced by
a different online mode, seed, or landmark fraction.

Start with the preregistered 0.50 landmark. Fractions 0.25 and 0.75 are
separate sensitivity experiments. A fraction is unsupported when a task has no
eventually successful rollout still active at that point; reduce the fraction
rather than silently dropping that task from the primary ten-task comparison.

## Causal elapsed-time and SAFE hybrid

When elapsed time is allowed as an online feature, the relevant question is
whether SAFE improves on the strongest causal time-only detector. Do not use a
rollout's final duration as an input: it is unknown until the rollout has
finished. The hybrid analyzer instead compares three deployable scores:

- `time_only`: a monotone, task-conditioned estimate of eventual failure among
  training rollouts still active at the current inference index;
- `safe_only`: the running maximum SAFE score with task normalization fitted
  from source-training rollouts only;
- `safe_time_task`: logistic risk from the current SAFE prefix, elapsed
  progress, time-only survival risk, and task identity.

Complete source-training rollouts are deterministically divided into meta-fit
and threshold-validation sets before prefix rows are created. Each rollout has
equal total weight during hybrid fitting. Thresholds maximize validation
balanced accuracy, and the source test split remains evaluation-only within
the command. No subtask, progress-predicate, failure-onset, future-score, or
final-duration field is used.

Run the comparison on the original, unmodified Xiaomi final-refit scores rather
than either duration-controlled refit:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_online_safe_latest.env

export XR1_ORIGINAL_FINAL_ROOT="$(dirname "$(dirname "$XR1_OUTER_SPLIT")")"
export XR1_TIME_HYBRID_ROOT="$XR1_ONLINE_ROOT/original_natural_time_safe_hybrid"

"$SAFE_PY" -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/analyze_time_safe_hybrid.py" \
  --final-root "$XR1_ORIGINAL_FINAL_ROOT" \
  --output-dir "$XR1_TIME_HYBRID_ROOT" \
  --models indep lstm \
  --seeds 0 1 2 \
  --validation-per-class 5 \
  --split-seed 0 \
  --regularizations 0.01 0.1 1.0 10.0 \
  --landmark-fractions 0.10 0.25 0.50 0.75 1.00 \
  --formats png pdf
```

The output contains `analysis.json`, `per_seed_event_metrics.csv`,
`landmark_metrics.csv`, `event_predictions.jsonl`, per-model event-comparison
figures, and a `runtime/<model>_seed<seed>/` directory with the fitted hybrid
model plus its training-derived time curves, task normalization, horizons, and
thresholds. Event metrics use the first threshold crossing during each
naturally terminated rollout. Landmark rows include only test rollouts still
active at that causal time.

The existing Xiaomi outer test was opened by the preceding detector analyses.
Consequently this run is a post-hoc comparison and must not be presented as a
new confirmatory test. Freeze the chosen time/hybrid protocol and collect a new
independent natural-rate test pool before making a final performance claim.

## Outcome-only causal-prefix residual SAFE

The natural event comparison can be very accurate while detecting failures too
late for recovery. The causal-prefix residual experiment therefore optimizes a
different, preregistered target: task-macro ROC-AUC among rollouts still active
at 25 and 50 percent of a task timeout estimated from meta-fit failures only.

The experiment remains normal outcome SAFE. It uses the final binary rollout
outcome and does not read or construct Subtask-SAFE annotations, predicate
progress, failure-onset labels, future observations, or final rollout duration
as an online input. Complete source rollouts are divided into meta-fit,
threshold-validation, and fixed outer-test parents before any prefix examples
are created.

For each supported task and landmark at 10, 25, 50, and 75 percent, training
failures are deterministically downsampled to the number of naturally at-risk
successes. Unsupported task/landmark strata are excluded from feature-residual
training rather than teaching the residual a deterministic time label. Every
retained source parent has equal total weight across its prefixes.

The runner compares:

- `time_only`: monotone task-conditioned failure risk among meta-fit rollouts
  still active at the current inference;
- `prefix_safe`: an MLP over the current Xiaomi hidden vector, the change from
  the previous inference, recent-window mean, and recent-window slope;
- `residual_safe_time`: the same MLP added as a logit residual to the frozen
  task-conditioned time-only risk.

Regularization and early stopping use meta-validation only. Model selection
uses the task-macro mean of the 25 and 50 percent landmark ROC-AUC values.
Online thresholds are selected on complete meta-validation trajectories at a
fixed empirical false-positive-rate cap. The fixed outer test is
evaluation-only within the command.

Run a bounded single-seed smoke first:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_online_safe_latest.env

export XR1_CAUSAL_PREFIX_SMOKE="$XR1_ONLINE_ROOT/causal_prefix_residual_smoke"

"$SAFE_PY" -u -m robocasa.recovery.safe.run_causal_prefix_residual \
  --export-dir "$XR1_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --output-dir "$XR1_CAUSAL_PREFIX_SMOKE" \
  --outer-split-manifest "$XR1_OUTER_SPLIT" \
  --seeds 0 \
  --landmarks 0.10 0.25 0.50 0.75 \
  --primary-landmarks 0.25 0.50 \
  --horizon-selector 1.0 \
  --diffusion-selector 1.0 \
  --regularizations 0.0001 \
  --epochs 20 \
  --patience 5 \
  --device cuda
```

After the smoke produces `analysis.json` with `status=complete`, run the full
three-seed developmental comparison in a new timestamped durable directory:

```bash
export XR1_CAUSAL_PREFIX_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_causal_prefix_residual_$(date +%Y%m%d_%H%M%S)"

"$SAFE_PY" -u -m robocasa.recovery.safe.run_causal_prefix_residual \
  --export-dir "$XR1_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --output-dir "$XR1_CAUSAL_PREFIX_ROOT" \
  --outer-split-manifest "$XR1_OUTER_SPLIT" \
  --seeds 0 1 2 \
  --landmarks 0.10 0.25 0.50 0.75 \
  --primary-landmarks 0.25 0.50 \
  --horizon-selector 1.0 \
  --diffusion-selector 1.0 \
  --temporal-window 4 \
  --hidden-dim 128 \
  --dropout 0.1 \
  --learning-rate 0.0003 \
  --regularizations 0.00001 0.0001 0.001 0.01 \
  --epochs 1000 \
  --patience 100 \
  --target-fpr 0.05 \
  --device cuda
```

The output contains `analysis.json`, `split_manifest.json`,
`selection_audit.csv`, `per_seed_event_metrics.csv`, `landmark_metrics.csv`,
`event_predictions.jsonl`, training histories, and deployable per-seed runtime
directories. Advance the residual model only if it improves over `time_only`
at the fixed 25/50 percent causal risk sets and improves detection time at the
same validation-selected false-positive-rate target. Because the existing
outer test has already been inspected, a fresh natural-rate test collection is
still required for a confirmatory performance claim.

## Staged early SAFE with a late elapsed-time fallback

The causal-prefix result shows useful Xiaomi representation signal at 10 and 25
percent, but the prefix-only detector misses too many failures to deploy alone.
`run_early_safe_time_cascade` evaluates a staged policy that preserves that
early signal while retaining elapsed time as a late safety net:

1. an outcome-only SAFE head trained only at 10 percent may alarm at that
   checkpoint;
2. a separate outcome-only SAFE head trained only at 25 percent may alarm at
   that checkpoint;
3. task-conditioned elapsed-time risk is ineligible before 50 percent and acts
   as the fallback thereafter.

The two SAFE heads have separate weights and feature scalers. They share one
early-SAFE operating threshold; the late time fallback has a second threshold.
Those two thresholds are searched jointly on complete meta-validation parents
under one event-level 5 percent FPR cap. Feasible pairs are ranked by failure
recall, then lower missed-failure-adjusted detection fraction, then lower FPR.
The fixed outer test is used only after model and threshold selection.

The comparison reports `early_safe`, `staged_safe_time`, and `time_only`
separately. `time_only` is allowed to inspect its causal time-risk trajectory
from the beginning, making it a deliberately strong reference rather than the
50-percent fallback component. Every event prediction records its first alarm
inference, horizon fraction, and whether SAFE or time caused the alarm.

Run a bounded one-seed smoke before the full experiment:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_online_safe_latest.env

export XR1_EARLY_CASCADE_SMOKE="$XR1_ONLINE_ROOT/early_safe_time_cascade_smoke"

"$SAFE_PY" -u -m robocasa.recovery.safe.run_early_safe_time_cascade \
  --export-dir "$XR1_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --output-dir "$XR1_EARLY_CASCADE_SMOKE" \
  --outer-split-manifest "$XR1_OUTER_SPLIT" \
  --seeds 0 \
  --early-landmarks 0.10 0.25 \
  --time-fallback 0.50 \
  --horizon-selector 1.0 \
  --diffusion-selector 1.0 \
  --regularizations 0.0001 \
  --epochs 20 \
  --patience 5 \
  --target-fpr 0.05 \
  --device cuda
```

After the smoke completes, use a new durable timestamped output directory for
the three-seed comparison:

```bash
export XR1_EARLY_CASCADE_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_early_safe_time_cascade_$(date +%Y%m%d_%H%M%S)"

"$SAFE_PY" -u -m robocasa.recovery.safe.run_early_safe_time_cascade \
  --export-dir "$XR1_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --output-dir "$XR1_EARLY_CASCADE_ROOT" \
  --outer-split-manifest "$XR1_OUTER_SPLIT" \
  --seeds 0 1 2 \
  --early-landmarks 0.10 0.25 \
  --time-fallback 0.50 \
  --horizon-selector 1.0 \
  --diffusion-selector 1.0 \
  --temporal-window 4 \
  --hidden-dim 128 \
  --dropout 0.1 \
  --learning-rate 0.0003 \
  --regularizations 0.00001 0.0001 0.001 0.01 \
  --epochs 1000 \
  --patience 100 \
  --target-fpr 0.05 \
  --device cuda
```

The output includes `analysis.json`, a parent-disjoint `split_manifest.json`,
per-head `selection_audit.csv`, joint `threshold_selection.csv`, event and
landmark metrics, per-rollout prediction traces, and a deployable runtime bundle
for each seed. The primary comparison is staged versus time-only recall and
missed-failure-adjusted detection fraction at the same validation FPR target.
Because this reuses the already inspected outer test, it remains developmental;
freeze the cascade before evaluating a new confirmatory test collection.
