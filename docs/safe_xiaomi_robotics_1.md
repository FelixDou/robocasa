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

Train MLP/LSTM and calibrate functional conformal thresholds from the exported
XR-1 dataset with the existing SAFE tools. Treat the result as a new XR-1
detector experiment until its task split, sample counts, seeds, hyperparameter
selection, calibration set, and held-out evaluation have all been recorded.
