# SAFE-style failure detection for RLDX-1 on RoboCasa

## Scope and scientific status

This integration applies the existing RoboCasa raw-SAFE collection, storage,
training, calibration, evaluation, and visualization pipeline to RLDX-1. The
rollout label remains binary task outcome only: success is `0` and natural
failure is `1`. It does not use subtask labels, simulator state, expert
demonstrations, recovery trajectories, or manufactured frame-level failure
onsets.

The result is **SAFE-style RLDX-1**, not the exact official π0 feature. SAFE's
official π0 baseline uses π0 `pre_velocity`. RLDX-1 has a different action
architecture, so this integration chooses the closest structural analogue and
records that different identity in every manifest. π0 and RLDX-1 rollouts are
therefore incompatible by construction and cannot be silently mixed.

Source revisions inspected:

- SAFE: `b6036abe07b2b2bb9996afb2c07f13d6a9f507c0`
- RLDX-1: `ef05cd4ae634ff97d672d42275febbc0b92cc192`
- RLDX-1 checkpoint: `RLWRLD/RLDX-1-FT-RC365`

## RLDX-1 feature contract

RLDX-1 predicts an action chunk with flow matching. At every Euler denoising
step, MSAT produces state/action token outputs `ao`, and the
embodiment-specific `action_decoder` maps the action-token suffix to action
velocity. The captured latent is:

```python
ao[:, -action_horizon:]
```

immediately before:

```python
pred_velocity = action_decoder(ao, embodiment_id)[:, -action_horizon:]
```

The feature layer identifier is:

```text
action_model_msat_action_suffix_pre_action_decoder
```

One real RLDX inference returns float32:

```text
(denoising_steps, action_horizon, MSAT_output_dim)
```

The dimensions are read from the running checkpoint and validated at runtime;
they are not hard-coded in RoboCasa. A rollout stores:

```text
(num_policy_inferences, denoising_steps, action_horizon, MSAT_output_dim)
```

No denoising or horizon aggregation occurs during collection. The existing
SAFE aggregation selectors (`mean`, `first`, `last`, mixed selectors, and
`concat-N`) remain downstream training choices.

Capturing the tensor only detaches and copies `ao`; it does not cast or replace
the tensor used by `action_decoder`, so requesting features does not change the
action computation.

## Server and client protocol

`patches/rldx1_safe_features_ef05cd4.patch` applies to the pinned RLDX commit.
A request opts in through the existing ZeroMQ `options` dictionary:

```python
{"request_safe_features": True}
```

The normal `(action, info)` response remains backward compatible. When
requested, `info` additionally contains:

```python
{
    "safe_features": np.ndarray,
    "safe_feature_metadata": {
        "schema_version": 1,
        "model_family": "rldx1",
        "model_id": "RLDX",
        "checkpoint": "RLWRLD/RLDX-1-FT-RC365",
        "feature_layer": "action_model_msat_action_suffix_pre_action_decoder",
        "feature_shape": [denoising_steps, action_horizon, hidden_dim],
        "feature_dtype": "float32",
        "action_horizon": action_horizon,
        "flow_steps": denoising_steps,
        "aggregation": "raw",
    },
}
```

`RLDXZeroMQPolicy` validates this payload, strips the single-environment batch
axis, and creates one inference record only when it contacts the RLDX server.
Actions served from its local `execution_horizon` cache produce no duplicate
feature records. `reset()` clears the action cache, feature queue, environment
step, inference index, and RLDX memory session.

## Apply the RLDX patch

Use source storage under `/gs/fs` and generated data under `/gs/bs`:

```bash
export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export RLDX_REPO="$PROJECT_FS/RLDX-1"
export RLDX_COMMIT=ef05cd4ae634ff97d672d42275febbc0b92cc192
export RLDX_SAFE_PATCH="$ROBOCASA_REPO/patches/rldx1_safe_features_ef05cd4.patch"

test "$(git -C "$RLDX_REPO" rev-parse HEAD)" = "$RLDX_COMMIT" || {
  echo "RLDX checkout is not at the pinned benchmark commit"
  exit 1
}
test -f "$RLDX_SAFE_PATCH" || exit 1

if git -C "$RLDX_REPO" apply --reverse --check "$RLDX_SAFE_PATCH" 2>/dev/null; then
  echo "RLDX SAFE patch is already applied"
else
  git -C "$RLDX_REPO" apply --check "$RLDX_SAFE_PATCH" &&
  git -C "$RLDX_REPO" apply "$RLDX_SAFE_PATCH"
fi

cd "$RLDX_REPO"
python -m py_compile \
  rldx/model/core/rldx.py \
  rldx/policy/policy_runtime.py \
  rldx/policy/rldx_policy.py \
  rldx/policy/step_request.py

grep -n "request_safe_features" \
  rldx/policy/step_request.py \
  rldx/policy/policy_runtime.py
grep -n "safe_feature_steps" rldx/model/core/rldx.py
```

## Start the patched server

Follow `docs/cluster_experiment_runbook.md` for the full environment block.
The patched server uses the normal RLDX port family:

```bash
export RLDX_SERVER_TAG=rldx1_safe_$(date +%Y%m%d_%H%M%S)
export RLDX_SERVER_LOG="$STORAGE_BS/robocasa_logs/eval/${RLDX_SERVER_TAG}_20100.log"

cd "$RLDX_REPO"
CUDA_VISIBLE_DEVICES=0 \
nohup uv run python -u rldx/eval/run_rldx_server.py \
  --model-path RLWRLD/RLDX-1-FT-RC365 \
  --embodiment-tag GENERAL_EMBODIMENT \
  --host 127.0.0.1 \
  --port 20100 \
  --use-sim-policy-wrapper \
  > "$RLDX_SERVER_LOG" 2>&1 &

echo "server_pid=$!"
until ss -ltn | grep -q ':20100'; do
  tail -30 "$RLDX_SERVER_LOG" 2>/dev/null || true
  sleep 10
done
```

Do not enable RLDX's optional compiled inference paths for the first SAFE
collection. The patch instruments the eager `RLDXActionModel` path, while an
optimized replacement may bypass or specialize that method. Establish the
eager action-equivalence smoke first; compiled SAFE capture would require a
separate equivalence test.

## One-rollout protocol smoke test

Run this bounded test before collecting a balanced dataset:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate "$STORAGE_BS/envs/robocasa_openpi"

export RLDX_SAFE_SMOKE="$STORAGE_BS/robocasa_rollouts/safe/rldx1_safe_smoke_$(date +%Y%m%d_%H%M%S)"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$RLDX_SAFE_SMOKE" \
  --tasks TurnOnSinkFaucet \
  --num-rollouts 1 \
  --seed 7 \
  --seed-protocol official_rldx \
  --policy-module robocasa.recovery.rldx_zmq_policy:make_policy \
  --model-family rldx1 \
  --policy-name RLDX-1-FT-RC365 \
  --checkpoint RLWRLD/RLDX-1-FT-RC365 \
  --policy-config '{"embodiment_tag":"GENERAL_EMBODIMENT"}' \
  --host 127.0.0.1 \
  --port 20100 \
  --split target \
  --replan-steps 8 \
  --record-safe-features \
  --record-actions \
  --no-record-videos \
  --max-errors 1

python -m robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$RLDX_SAFE_SMOKE"
```

Inspect the actual feature contract:

```bash
python - "$RLDX_SAFE_SMOKE" <<'PY'
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

The smoke is successful only if the dataset validator reports `VALID`, the
features are finite and nonzero, and inference steps are spaced by at least the
configured eight cached actions.

## Mixed atomic/composite pilot collection

The collector module keeps its historical
`collect_atomic_rollouts` name, but accepts registered atomic and composite
RoboCasa tasks. Without `--horizon`, it reads each task's official horizon from
the current dataset registry. Do not force one common horizon across composite
tasks.

The first RLDX pilot uses five atomic tasks and five composite-seen tasks. They
span moderate success rates, longer-horizon failures, and a composite task
(`LoadDishwasher`) whose ordered progress is substantially higher than its
binary success rate:

| Type | Task | Pilot target | Official horizon |
|---|---|---:|---:|
| Atomic | `CoffeeSetupMug` | 10 success + 10 failure | 600 |
| Atomic | `CloseToasterOvenDoor` | 10 success + 10 failure | 450 |
| Atomic | `PickPlaceDrawerToCounter` | 10 success + 10 failure | 750 |
| Atomic | `PickPlaceCounterToStove` | 10 success + 10 failure | 600 |
| Atomic | `TurnOnSinkFaucet` | 10 success + 10 failure | 600 |
| Composite | `PreSoakPan` | 10 success + 10 failure | 2400 |
| Composite | `ScrubCuttingBoard` | 10 success + 10 failure | 1200 |
| Composite | `WashLettuce` | 10 success + 10 failure | 1650 |
| Composite | `StackBowlsCabinet` | 10 success + 10 failure | 2100 |
| Composite | `LoadDishwasher` | 10 success + 10 failure | 1800 |

Run one immutable shard per server/GPU. The assignment below mixes atomic and
composite tasks so the sum of expected simulator steps is approximately
balanced between the two GPUs. Forty-five attempts per task is the pilot cap;
`--retain-only-quota` stores exactly 10 examples of each class when the target
is reached:

```bash
export RLDX_SAFE_TAG=rldx1_safe_mixed10_10x10_$(date +%Y%m%d_%H%M%S)
export RLDX_SAFE_ROOT="$STORAGE_BS/robocasa_rollouts/safe/$RLDX_SAFE_TAG"
mkdir -p "$RLDX_SAFE_ROOT"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$RLDX_SAFE_ROOT/shard0" \
  --tasks LoadDishwasher PreSoakPan CoffeeSetupMug \
          CloseToasterOvenDoor PickPlaceDrawerToCounter \
  --num-rollouts 45 \
  --seed 7 \
  --seed-protocol official_rldx \
  --success-quota 10 \
  --failure-quota 10 \
  --retain-only-quota \
  --policy-module robocasa.recovery.rldx_zmq_policy:make_policy \
  --model-family rldx1 \
  --policy-name RLDX-1-FT-RC365 \
  --checkpoint RLWRLD/RLDX-1-FT-RC365 \
  --policy-config '{"embodiment_tag":"GENERAL_EMBODIMENT"}' \
  --host 127.0.0.1 \
  --port 20100 \
  --split target \
  --replan-steps 8 \
  --record-safe-features \
  --record-actions \
  --record-videos \
  --video-frame-stride 2 \
  --max-errors 3 \
  --resume

CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
python -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$RLDX_SAFE_ROOT/shard1" \
  --tasks StackBowlsCabinet WashLettuce ScrubCuttingBoard \
          PickPlaceCounterToStove TurnOnSinkFaucet \
  --num-rollouts 45 \
  --seed 7 \
  --seed-protocol official_rldx \
  --success-quota 10 \
  --failure-quota 10 \
  --retain-only-quota \
  --policy-module robocasa.recovery.rldx_zmq_policy:make_policy \
  --model-family rldx1 \
  --policy-name RLDX-1-FT-RC365 \
  --checkpoint RLWRLD/RLDX-1-FT-RC365 \
  --policy-config '{"embodiment_tag":"GENERAL_EMBODIMENT"}' \
  --host 127.0.0.1 \
  --port 20101 \
  --split target \
  --replan-steps 8 \
  --record-safe-features \
  --record-actions \
  --record-videos \
  --video-frame-stride 2 \
  --max-errors 3 \
  --resume
```

Validate both completed shards, then merge them into a new directory:

```bash
python -m robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$RLDX_SAFE_ROOT/shard0"
python -m robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$RLDX_SAFE_ROOT/shard1"

python -m robocasa.recovery.safe.merge_atomic_datasets \
  --source-dirs "$RLDX_SAFE_ROOT/shard0" "$RLDX_SAFE_ROOT/shard1" \
  --output-dir "$RLDX_SAFE_ROOT/merged"

python -m robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$RLDX_SAFE_ROOT/merged"
```

The merged pilot is healthy only when it contains 200 valid rollouts, 100
successes, 100 failures, all ten tasks at exactly 10/10, zero collection
errors, no duplicate rollout or task/reset identities, and no missing requested
artifacts. If a task does not reach both quotas within 45 attempts, preserve
the partial shard and resume it with a larger `--num-rollouts`; do not lower the
class quota or delete the evidence.

The resulting dataset can then be exported with
`robocasa.recovery.safe.export_to_official_safe` and passed to the existing
official SAFE MLP/LSTM grid, conformal calibration, reporting, and score-video
tools. The pinned official loader does not hard-code π0's 1024-wide latent: it
selects the requested horizon and diffusion indices, stacks the resulting
vectors, and sets `dim_features` from the loaded tensor's final dimension.
Select and calibrate RLDX hyperparameters independently of π0; do not reuse a
π0 predictor or mix the two feature families.

## Ten-task RLDX training protocol

For the balanced 10-task pilot, use the same tasks in training and test while
holding out examples before any hyperparameter selection:

- outer training pool: 7 successes + 7 failures per task, 140 rollouts total;
- untouched outer test: 3 successes + 3 failures per task, 60 rollouts total;
- hyperparameter selection: three outcome-stratified folds inside only the
  140-rollout outer training pool;
- final estimation: refit three model seeds on all 140 training rollouts and
  evaluate the fixed 60-rollout test once;
- threshold calibration: not performed on the 60 test rollouts. Collect or
  reserve a separate successful-rollout calibration set before reporting a
  conformal operating point.

The 10-task RLDX export is:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/rldx1_safe_mixed10_10x10_20260726_190751/official_safe_10x10
```

Activate the SAFE environment and keep all generated files under `/gs/bs`:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa_safe_integration"
export SAFE_REPO="$PROJECT_FS/SAFE"
export SAFE_ENV="$STORAGE_BS/envs/vla_safe"
export SAFE_OFFICIAL="$STORAGE_BS/robocasa_rollouts/safe/rldx1_safe_mixed10_10x10_20260726_190751/official_safe_10x10"
export SAFE_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval"

conda activate "$SAFE_ENV"

export XDG_CACHE_HOME="$STORAGE_BS/xdg_cache"
export MPLCONFIGDIR="$STORAGE_BS/matplotlib_config"
export WANDB_CACHE_DIR="$STORAGE_BS/wandb_cache"
export TMPDIR=/tmp/ut06746/safe_rldx_training
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_ENABLED=0
export PYTHONNOUSERSITE=1

mkdir -p \
  "$SAFE_LOG_ROOT" \
  "$XDG_CACHE_HOME" \
  "$MPLCONFIGDIR" \
  "$WANDB_CACHE_DIR" \
  "$TMPDIR"

test "$(git -C "$SAFE_REPO" rev-parse HEAD)" = \
  "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
test -f "$SAFE_OFFICIAL/conversion_report.json"
```

First run one bounded selector pair per architecture for two epochs. These two
processes use separate GPUs and should produce three fold metrics each:

```bash
export SAFE_RLDX_SMOKE_TAG=safe_rldx1_seen10_cv_smoke_$(date +%Y%m%d_%H%M%S)
export SAFE_RLDX_SMOKE_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/$SAFE_RLDX_SMOKE_TAG"
mkdir -p "$SAFE_RLDX_SMOKE_ROOT"

CUDA_VISIBLE_DEVICES=0 nohup python -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$SAFE_RLDX_SMOKE_ROOT" \
  --model indep \
  --train-per-class 7 \
  --split-seed 0 \
  --inner-seed 0 \
  --num-folds 3 \
  --horizon-selectors 1.0 \
  --diffusion-selectors 0.0 \
  --learning-rates 3e-4 \
  --regularization 1e-3 \
  --epochs 2 \
  --device cuda \
  --resume \
  > "$SAFE_LOG_ROOT/${SAFE_RLDX_SMOKE_TAG}_indep.log" 2>&1 &
echo "indep_pid=$!"

CUDA_VISIBLE_DEVICES=1 nohup python -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$SAFE_RLDX_SMOKE_ROOT" \
  --model lstm \
  --train-per-class 7 \
  --split-seed 0 \
  --inner-seed 0 \
  --num-folds 3 \
  --horizon-selectors 1.0 \
  --diffusion-selectors concat-2 \
  --learning-rates 1e-3 \
  --regularization 1e-2 \
  --epochs 2 \
  --device cuda \
  --resume \
  > "$SAFE_LOG_ROOT/${SAFE_RLDX_SMOKE_TAG}_lstm.log" 2>&1 &
echo "lstm_pid=$!"
```

The smoke passes only with six completed fits, no `failure.json`, finite loss,
and a plan that records 140 outer-training and 60 untouched outer-test
rollouts:

```bash
echo "Completed: $(find "$SAFE_RLDX_SMOKE_ROOT" -mindepth 2 -name metrics.json | wc -l) / 6"
echo "Failures:  $(find "$SAFE_RLDX_SMOKE_ROOT" -name failure.json | wc -l)"
tail -30 "$SAFE_LOG_ROOT/${SAFE_RLDX_SMOKE_TAG}_indep.log"
tail -30 "$SAFE_LOG_ROOT/${SAFE_RLDX_SMOKE_TAG}_lstm.log"

python "$ROBOCASA_REPO/robocasa/recovery/safe/summarize_seen_cv.py" \
  --root "$SAFE_RLDX_SMOKE_ROOT" \
  --expected-folds 0 1 2 \
  --quiet

python - "$SAFE_RLDX_SMOKE_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
for model in ("indep", "lstm"):
    plan = json.loads((root / f"cv_plan_{model}.json").read_text())
    print(
        model,
        "tasks=", plan["num_tasks"],
        "outer_train=", plan["outer_train_rollouts"],
        "outer_test=", plan["outer_test_rollouts"],
        "test_used=", plan["outer_test_used"],
    )
PY
```

After this smoke passes, launch the full official 405-fit grid per architecture
with the same split seeds and 1,000 epochs. Do not reuse the π0 hyperparameters:

```bash
export SAFE_RLDX_CV_TAG=safe_rldx1_seen10_official_cv_2gpu_$(date +%Y%m%d_%H%M%S)
export SAFE_RLDX_CV_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/$SAFE_RLDX_CV_TAG"
mkdir -p "$SAFE_RLDX_CV_ROOT"

CUDA_VISIBLE_DEVICES=0 nohup python -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$SAFE_RLDX_CV_ROOT" \
  --model indep \
  --train-per-class 7 \
  --split-seed 0 \
  --inner-seed 0 \
  --num-folds 3 \
  --epochs 1000 \
  --device cuda \
  --resume \
  > "$SAFE_LOG_ROOT/${SAFE_RLDX_CV_TAG}_indep.log" 2>&1 &
echo "indep_pid=$!"

CUDA_VISIBLE_DEVICES=1 nohup python -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$SAFE_RLDX_CV_ROOT" \
  --model lstm \
  --train-per-class 7 \
  --split-seed 0 \
  --inner-seed 0 \
  --num-folds 3 \
  --epochs 1000 \
  --device cuda \
  --resume \
  > "$SAFE_LOG_ROOT/${SAFE_RLDX_CV_TAG}_lstm.log" 2>&1 &
echo "lstm_pid=$!"
```

## Leakage-free seen-task calibration for the 25/25 dataset

After selecting and refitting the final MLP on the 25-success/25-failure
dataset, calibrate a conformal operating threshold without reusing an
evaluation rollout. The final training command uses 17 successes and 17
failures per task, leaving 8 successes and 8 failures per task held out.
This calibration stage then uses:

- 3 of the 8 held-out successes per task for calibration, 30 rollouts total;
- the official SAFE 30/70 reference/calibration split inside those 30
  successful rollouts, yielding 9 reference and 21 nonconformity rollouts;
- the remaining 5 successes and all 8 failures per task for final threshold
  evaluation, yielding 130 rollouts total;
- task affine normalization fitted only from the 340 training rollouts;
- the official functional threshold with `extend` alignment and `tfunc`
  modulation.

This is a seen-task, task-conditioned protocol. It must not be used to claim
generalization to an unseen task, because that task would not have a fitted
normalization statistic. The stage is CPU-only and needs neither a policy
server nor a GPU. Alpha 0.15 is fixed before evaluation; the other alpha rows
are sensitivity analyses and must not be used to choose a better-looking test
operating point.

In a new cluster session, activate the SAFE environment and point
`SAFE_RLDX25_FINAL_ROOT` at the directory containing `indep_seed0`,
`indep_seed1`, and `indep_seed2`:

```bash
set -euo pipefail

module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa_safe_integration"
export SAFE_ENV="$STORAGE_BS/envs/vla_safe"

# Replace the final path component with the completed final-training directory.
export SAFE_RLDX25_FINAL_ROOT=/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/REPLACE_WITH_FINAL_DIRECTORY
export SAFE_RLDX25_CALIB_ROOT="${SAFE_RLDX25_FINAL_ROOT}_task_normalized_conformal"

export XDG_CACHE_HOME="$STORAGE_BS/xdg_cache"
export MPLCONFIGDIR="$STORAGE_BS/matplotlib_config"
export TMPDIR=/tmp/ut06746/safe_rldx_calibration
export PYTHONNOUSERSITE=1

conda activate "$SAFE_ENV"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR" "$SAFE_RLDX25_CALIB_ROOT"

for seed in 0 1 2; do
  test -f "$SAFE_RLDX25_FINAL_ROOT/indep_seed${seed}/scores.jsonl"
done

python -u "$ROBOCASA_REPO/robocasa/recovery/safe/calibrate_seen_tasks.py" \
  --final-root "$SAFE_RLDX25_FINAL_ROOT" \
  --output-dir "$SAFE_RLDX25_CALIB_ROOT" \
  --model indep \
  --seeds 0 1 2 \
  --calibration-successes-per-task 3 \
  --split-seed 0 \
  --conformal-seed 0 \
  --reference-fraction 0.3 \
  --alphas 0.05 0.10 0.15 0.20 \
  --selected-alpha 0.15 \
  --modulation tfunc
```

Audit the split and print the operating-point trade-off:

```bash
python - "$SAFE_RLDX25_CALIB_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "split_manifest.json").read_text())
summary = json.loads((root / "summary.json").read_text())

print("Split counts:", manifest["counts"])
print("Tasks:", len(manifest["task_names"]))
print(
    "Calibration/evaluation disjoint:",
    not (
        set(manifest["calibration_success_ids"])
        & set(manifest["evaluation_ids"])
    ),
)

print("\nalpha    TPR    FPR bal_acc det_time")
for alpha, metrics in summary["aggregate"]["by_alpha"].items():
    print(
        f"{float(alpha):4.2f}",
        f"{metrics['true_positive_rate_mean']:6.3f}",
        f"{metrics['false_positive_rate_mean']:6.3f}",
        f"{metrics['balanced_accuracy_mean']:7.3f}",
        f"{metrics['normalized_detection_time_mean']:8.3f}",
    )

print("\nSelected alpha:", summary["selected_alpha"])
print("Report:", root / "summary.json")
print("Trade-off plot:", root / "conformal_tradeoff.png")
print("Per-task plot:", root / "per_task_balanced_accuracy.png")
PY
```

To inspect causal detections on videos for one fitted seed, render normalized
evaluation trajectories with the matching calibrated threshold:

```bash
export SAFE_RLDX25_VIDEO_ROOT="$SAFE_RLDX25_CALIB_ROOT/videos_indep_seed0_alpha0p15"

python -u "$ROBOCASA_REPO/robocasa/recovery/safe/render_score_videos.py" \
  --scores "$SAFE_RLDX25_CALIB_ROOT/indep_seed0/normalized_scores.jsonl" \
  --calibration "$SAFE_RLDX25_CALIB_ROOT/indep_seed0/alpha_0p15/calibration.json" \
  --split evaluation \
  --max-videos 20 \
  --output-dir "$SAFE_RLDX25_VIDEO_ROOT"
```

The yellow curve is the normalized SAFE score and the red curve is the
functional threshold. The status changes from `MONITORING` to `ALERT` at the
first causal crossing; the overlay never uses a future score.

The result is healthy only if the split reports 340 training, 30 calibration
successes, 9 reference successes, 21 nonconformity successes, and 130
evaluation rollouts consisting of 50 successes and 80 failures. The
calibration and evaluation ID sets must be disjoint.

## Atomic-versus-composite SAFE ablation

Use a two-stage ablation to distinguish threshold heterogeneity from negative
training transfer:

1. calibrate the already-trained mixed MLP separately on atomic and composite
   tasks, without retraining;
2. select and train independent atomic-only and composite-only MLPs, then
   calibrate each predictor on its matching task type.

Both stages reuse the exact outer train/test rollout IDs from the completed
mixed final refit. The per-task calibration selector is independent of which
other tasks are present, so mixed and split predictors also use identical
calibration and evaluation rollout IDs within each task type. Alpha 0.15
remains fixed before evaluation.

Set the shared paths:

```bash
export SAFE_OFFICIAL="$STORAGE_BS/robocasa_rollouts/safe/rldx1_safe_mixed10_extra15x15_20260727_163929/official_safe_25x25"
export SAFE_MIXED_FINAL="$STORAGE_BS/robocasa_checkpoints/safe/safe_rldx1_25x25_final_2gpu_20260730_104400"
export SAFE_MIXED_SPLIT="$SAFE_MIXED_FINAL/indep_seed0/split_manifest.json"

export SAFE_TYPE_TAG=safe_rldx1_25x25_type_ablation_$(date +%Y%m%d_%H%M%S)
export SAFE_TYPE_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/$SAFE_TYPE_TAG"
export SAFE_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval"

mkdir -p \
  "$SAFE_TYPE_ROOT/mixed_calibration" \
  "$SAFE_TYPE_ROOT/cv" \
  "$SAFE_TYPE_ROOT/final" \
  "$SAFE_TYPE_ROOT/split_calibration" \
  "$SAFE_LOG_ROOT"

for seed in 0 1 2; do
  test -f "$SAFE_MIXED_FINAL/indep_seed${seed}/scores.jsonl"
done
test -f "$SAFE_MIXED_SPLIT"
```

First calibrate the mixed MLP separately by task type:

```bash
for task_type in atomic composite; do
  python -u "$ROBOCASA_REPO/robocasa/recovery/safe/calibrate_seen_tasks.py" \
    --final-root "$SAFE_MIXED_FINAL" \
    --output-dir "$SAFE_TYPE_ROOT/mixed_calibration/$task_type" \
    --model indep \
    --task-type "$task_type" \
    --seeds 0 1 2 \
    --calibration-successes-per-task 3 \
    --split-seed 0 \
    --conformal-seed 0 \
    --reference-fraction 0.3 \
    --alphas 0.05 0.10 0.15 0.20 \
    --selected-alpha 0.15 \
    --modulation tfunc \
    > "$SAFE_LOG_ROOT/${SAFE_TYPE_TAG}_mixed_${task_type}_calibration.log" 2>&1
done
```

Each type-specific calibration must contain 170 training rollouts, 15
calibration successes split into 4 reference and 11 nonconformity rollouts,
and 65 evaluation rollouts consisting of 25 successes and 40 failures.

Before the full sweep, run one two-epoch, three-fold smoke on each GPU:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$SAFE_TYPE_ROOT/cv_smoke/atomic" \
  --model indep \
  --task-type atomic \
  --outer-split-manifest "$SAFE_MIXED_SPLIT" \
  --train-per-class 17 \
  --split-seed 0 \
  --inner-seed 0 \
  --num-folds 3 \
  --horizon-selectors concat-2 \
  --diffusion-selectors concat-2 \
  --learning-rates 1e-3 \
  --regularization 1e-3 \
  --epochs 2 \
  --device cuda \
  --resume \
  > "$SAFE_LOG_ROOT/${SAFE_TYPE_TAG}_atomic_smoke.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$SAFE_TYPE_ROOT/cv_smoke/composite" \
  --model indep \
  --task-type composite \
  --outer-split-manifest "$SAFE_MIXED_SPLIT" \
  --train-per-class 17 \
  --split-seed 0 \
  --inner-seed 0 \
  --num-folds 3 \
  --horizon-selectors concat-2 \
  --diffusion-selectors concat-2 \
  --learning-rates 1e-3 \
  --regularization 1e-3 \
  --epochs 2 \
  --device cuda \
  --resume \
  > "$SAFE_LOG_ROOT/${SAFE_TYPE_TAG}_composite_smoke.log" 2>&1 &
```

The smoke passes with three metrics per type, no `failure.json`, five selected
tasks, 170 outer-training rollouts, and 80 untouched outer-test rollouts.

Launch the full 405-fit MLP sweep per type only after that smoke passes:

```bash
for spec in "0 atomic" "1 composite"; do
  set -- $spec
  gpu="$1"
  task_type="$2"

  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u \
    "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
    --export-dir "$SAFE_OFFICIAL" \
    --safe-repo "$SAFE_REPO" \
    --output-root "$SAFE_TYPE_ROOT/cv/$task_type" \
    --model indep \
    --task-type "$task_type" \
    --outer-split-manifest "$SAFE_MIXED_SPLIT" \
    --train-per-class 17 \
    --split-seed 0 \
    --inner-seed 0 \
    --num-folds 3 \
    --epochs 1000 \
    --device cuda \
    --resume \
    > "$SAFE_LOG_ROOT/${SAFE_TYPE_TAG}_${task_type}_cv.log" 2>&1 &

  echo "$task_type pid=$!"
done
```

Summarize each completed sweep:

```bash
for task_type in atomic composite; do
  python "$ROBOCASA_REPO/robocasa/recovery/safe/summarize_seen_cv.py" \
    --root "$SAFE_TYPE_ROOT/cv/$task_type" \
    --expected-folds 0 1 2 \
    --quiet
done
```

Refit three MLP seeds per type with the selected type-specific
hyperparameters:

```bash
for spec in "0 atomic" "1 composite"; do
  set -- $spec
  gpu="$1"
  task_type="$2"

  CUDA_VISIBLE_DEVICES="$gpu" nohup bash -c '
    set -euo pipefail
    task_type="$1"
    for seed in 0 1 2; do
      python -u "$ROBOCASA_REPO/robocasa/recovery/safe/train_seen_tasks.py" \
        --export-dir "$SAFE_OFFICIAL" \
        --safe-repo "$SAFE_REPO" \
        --output-dir "$SAFE_TYPE_ROOT/final/$task_type/indep_seed${seed}" \
        --model indep \
        --task-type "$task_type" \
        --outer-split-manifest "$SAFE_MIXED_SPLIT" \
        --seed "$seed" \
        --split-seed 0 \
        --train-per-class 17 \
        --epochs 1000 \
        --device cuda \
        --selection-summary "$SAFE_TYPE_ROOT/cv/$task_type/cv_selection_summary.json" \
        --resume
    done
  ' bash "$task_type" \
    > "$SAFE_LOG_ROOT/${SAFE_TYPE_TAG}_${task_type}_final.log" 2>&1 &

  echo "$task_type final pid=$!"
done
```

Finally calibrate the two split predictors:

```bash
for task_type in atomic composite; do
  python -u "$ROBOCASA_REPO/robocasa/recovery/safe/calibrate_seen_tasks.py" \
    --final-root "$SAFE_TYPE_ROOT/final/$task_type" \
    --output-dir "$SAFE_TYPE_ROOT/split_calibration/$task_type" \
    --model indep \
    --task-type "$task_type" \
    --seeds 0 1 2 \
    --calibration-successes-per-task 3 \
    --split-seed 0 \
    --conformal-seed 0 \
    --reference-fraction 0.3 \
    --alphas 0.05 0.10 0.15 0.20 \
    --selected-alpha 0.15 \
    --modulation tfunc \
    > "$SAFE_LOG_ROOT/${SAFE_TYPE_TAG}_split_${task_type}_calibration.log" 2>&1
done
```

For each type, the mixed and split calibration manifests must have identical
`calibration_success_ids` and `evaluation_ids`. A gain from mixed training to
split training therefore cannot be attributed to a different held-out split.

## Validation status and remaining live check

The RLDX patch applies cleanly to the pinned commit and its modified files pass
`py_compile`. RoboCasa unit tests cover action-only backward compatibility,
missing and invalid feature payloads, cached-action versus real-inference
records, reset behavior, RLDX model-family provenance, dataset validation, and
official-loader export. A live GPU feature capture still requires the cluster
checkpoint, RLDX server, RoboCasa simulator, and assets, so the one-rollout
command above is the required final runtime check before a large collection.

## Natural-outcome-rate and official-loss experiment

The 25-success/25-failure export is intentionally balanced. The paper-like
experiment reconstructs each task's policy success rate from valid manifest
records plus `skipped_class_quota_reached` records. Those discarded rollouts
were executed and labeled by the simulator. Errors, `skipped_quota_reached`,
and other administrative skips are not outcomes and are excluded.

The builder preserves the existing 160-rollout test set exactly and creates
five immutable resamples for three controlled regimes:

- `matched_weighted`: matched-size 50/50 training with official SAFE
  inverse-frequency weights;
- `natural_weighted`: natural-rate training with official weights;
- `natural_unweighted`: the same natural-rate IDs with equal, scale-matched
  class multipliers. This removes inverse-frequency scaling while preserving
  the unchanged official loss and its mean monitor-loss scale relative to
  regularization.

Consequently, `natural_weighted - matched_weighted` isolates prevalence at
fixed sample count and `natural_weighted - natural_unweighted` isolates the
paper's loss weighting.

### New-session setup and split construction

```bash
set -euo pipefail

module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/vla_safe

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa_safe_integration"
export SAFE_REPO="$PROJECT_FS/SAFE"
export SAFE_OFFICIAL="$STORAGE_BS/robocasa_rollouts/safe/rldx1_safe_mixed10_extra15x15_20260727_163929/official_safe_25x25"
export SAFE_RLDX25_FINAL_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/safe_rldx1_25x25_final_2gpu_20260730_104400"
export SAFE_OUTER_SPLIT="$SAFE_RLDX25_FINAL_ROOT/indep_seed0/split_manifest.json"
export RLDX_FIRST_ROOT="$STORAGE_BS/robocasa_rollouts/safe/rldx1_safe_mixed10_10x10_20260726_190751"
export RLDX_EXTRA_ROOT="$STORAGE_BS/robocasa_rollouts/safe/rldx1_safe_mixed10_extra15x15_20260727_163929"

export SAFE_NATURAL_TAG=safe_rldx1_natural_rate_screen_$(date +%Y%m%d_%H%M%S)
export SAFE_NATURAL_PLAN_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/${SAFE_NATURAL_TAG}_plan"
export SAFE_NATURAL_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/$SAFE_NATURAL_TAG"
export SAFE_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval"
export XDG_CACHE_HOME="$STORAGE_BS/xdg_cache"
export MPLCONFIGDIR="$STORAGE_BS/matplotlib_config"
export WANDB_CACHE_DIR="$STORAGE_BS/wandb_cache"
export TMPDIR=/tmp/ut06746/safe_natural_rate
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_ENABLED=0
export PYTHONNOUSERSITE=1

mkdir -p \
  "$SAFE_NATURAL_PLAN_ROOT" "$SAFE_NATURAL_ROOT" "$SAFE_LOG_ROOT" \
  "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" "$TMPDIR"

cd "$ROBOCASA_REPO"

python -u -m robocasa.recovery.safe.build_natural_rate_experiment \
  --collection-dataset "$RLDX_FIRST_ROOT/shard0" \
  --collection-dataset "$RLDX_FIRST_ROOT/shard1" \
  --collection-dataset "$RLDX_EXTRA_ROOT/shard0" \
  --collection-dataset "$RLDX_EXTRA_ROOT/shard1" \
  --official-export "$SAFE_OFFICIAL" \
  --outer-split-manifest "$SAFE_OUTER_SPLIT" \
  --output-dir "$SAFE_NATURAL_PLAN_ROOT" \
  --subset-seeds 0 1 2 3 4 \
  > "$SAFE_NATURAL_PLAN_ROOT/build_stdout.json"
```

Freeze the hyperparameters already selected by the completed 25x25
training-only inner CV:

```bash
python - "$SAFE_NATURAL_PLAN_ROOT/frozen_selection_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = {
    "schema_version": 1,
    "task_type_filter": "all",
    "selection_source": "completed 25x25 training-only inner CV",
    "best_by_model": {
        "indep": {
            "horizon_selector": "concat-2",
            "diffusion_selector": "concat-2",
            "learning_rate": 1e-3,
            "lambda_reg": 1e-3,
        },
        "lstm": {
            "horizon_selector": "concat-2",
            "diffusion_selector": "concat-2",
            "learning_rate": 1e-4,
            "lambda_reg": 1e-2,
        },
    },
}
Path(sys.argv[1]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

export SAFE_FROZEN_SELECTION="$SAFE_NATURAL_PLAN_ROOT/frozen_selection_summary.json"
```

Audit the rate estimates and test-set invariant before training:

```bash
python - "$SAFE_NATURAL_PLAN_ROOT" "$SAFE_OUTER_SPLIT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
outer = json.loads(Path(sys.argv[2]).read_text())
audit = json.loads((root / "natural_rate_audit.json").read_text())
plan = json.loads((root / "experiment_plan.json").read_text())

print("Observed genuine outcomes:", audit["num_observed_outcomes"])
print("Common samples/task:", plan["samples_per_task"])
for task, values in audit["per_task"].items():
    print(
        f"{task:32s} S={values['successes']:3d} F={values['failures']:3d} "
        f"p_success={values['natural_success_rate']:.3f}"
    )
fixed_test = sorted(map(str, outer["test"]))
for record in plan["manifests"]:
    manifest = json.loads(Path(record["path"]).read_text())
    assert manifest["test"] == fixed_test
    assert not set(manifest["train"]) & set(manifest["test"])
print("VERDICT: IMMUTABLE SPLITS HEALTHY")
PY
```

### Two-GPU frozen-hyperparameter screen

Each GPU runs 45 fits: three regimes, five subset seeds, and three model seeds.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m \
  robocasa.recovery.safe.run_natural_rate_screen \
  --experiment-plan "$SAFE_NATURAL_PLAN_ROOT/experiment_plan.json" \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --selection-summary "$SAFE_FROZEN_SELECTION" \
  --output-root "$SAFE_NATURAL_ROOT" \
  --model indep \
  --subset-seeds 0 1 2 3 4 \
  --model-seeds 0 1 2 \
  --epochs 1000 \
  --device cuda \
  --resume \
  > "$SAFE_LOG_ROOT/${SAFE_NATURAL_TAG}_indep.log" 2>&1 &
echo "MLP PID=$!"

CUDA_VISIBLE_DEVICES=1 nohup python -u -m \
  robocasa.recovery.safe.run_natural_rate_screen \
  --experiment-plan "$SAFE_NATURAL_PLAN_ROOT/experiment_plan.json" \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --selection-summary "$SAFE_FROZEN_SELECTION" \
  --output-root "$SAFE_NATURAL_ROOT" \
  --model lstm \
  --subset-seeds 0 1 2 3 4 \
  --model-seeds 0 1 2 \
  --epochs 1000 \
  --device cuda \
  --resume \
  > "$SAFE_LOG_ROOT/${SAFE_NATURAL_TAG}_lstm.log" 2>&1 &
echo "LSTM PID=$!"
```

Monitor and summarize:

```bash
jobs -l
echo "Completed: $(find "$SAFE_NATURAL_ROOT" -mindepth 2 -name metrics.json | wc -l) / 90"
echo "Failures:  $(find "$SAFE_NATURAL_ROOT" -name failure.json | wc -l)"
tail -20 "$SAFE_LOG_ROOT/${SAFE_NATURAL_TAG}_indep.log"
tail -20 "$SAFE_LOG_ROOT/${SAFE_NATURAL_TAG}_lstm.log"

test "$(find "$SAFE_NATURAL_ROOT" -mindepth 2 -name metrics.json | wc -l)" -eq 90
test "$(find "$SAFE_NATURAL_ROOT" -name failure.json | wc -l)" -eq 0

python -m robocasa.recovery.safe.summarize_natural_rate_screen \
  --root "$SAFE_NATURAL_ROOT" \
  --expected-subset-seeds 0 1 2 3 4 \
  --expected-model-seeds 0 1 2 \
  --quiet

python - "$SAFE_NATURAL_ROOT/natural_rate_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
for key, result in summary["groups"].items():
    roc = result["test_roc_auc"]
    prc = result["test_prc_auc"]
    print(
        f"{key:28s} ROC={roc['mean']:.3f} +/- {roc['std']:.3f} "
        f"PRC={prc['mean']:.3f} +/- {prc['std']:.3f} "
        f"n={result['counts']['train']}"
    )
print(json.dumps(summary["paired_comparisons"], indent=2))
PY
```

Treat subset seeds, rather than the three optimization seeds within one
subset, as the independent resampling units. Run a new inner-CV sweep only if
the frozen `natural_weighted` regime is consistently promising.

### Detailed task-normalized and per-task analysis

After all 90 fits complete, compute raw pooled, training-only task-z pooled,
macro-task, atomic/composite, and per-task results. Model seeds are averaged
inside each subset before the five subset seeds are summarized.

```bash
export SAFE_NATURAL_ANALYSIS="$SAFE_NATURAL_ROOT/detailed_analysis"

python -u -m robocasa.recovery.safe.analyze_natural_rate_screen \
  --root "$SAFE_NATURAL_ROOT" \
  --output-dir "$SAFE_NATURAL_ANALYSIS" \
  --expected-subset-seeds 0 1 2 3 4 \
  --expected-model-seeds 0 1 2 \
  --formats png pdf \
  --quiet

find "$SAFE_NATURAL_ANALYSIS" -maxdepth 1 -type f -print | sort
```

The durable outputs are:

- `detailed_summary.json`;
- `run_metrics.jsonl`;
- `group_metrics.csv`;
- `per_task_metrics.csv`;
- `roc_aggregation_comparison.{png,pdf}`;
- `per_task_roc_indep.{png,pdf}`;
- `per_task_roc_lstm.{png,pdf}`.

## One normal SAFE model per task: balanced versus natural rate

This experiment is rollout-level SAFE and is deliberately unrelated to
Subtask-SAFE. It trains an independent monitor for each of the ten original
RLDX tasks. The primary comparison keeps the training size fixed at 22
rollouts per task and reuses the same frozen 8-success/8-failure test episodes:

- `matched_weighted`: 11 successes and 11 failures with official SAFE weights;
- `natural_weighted`: 22 examples sampled at the task's observed policy outcome
  rate, also with official SAFE inverse-frequency weights.

The MLP (`indep`) is the preregistered primary architecture because it was the
stronger normal-SAFE RLDX model. Hyperparameters remain frozen at the completed
all-task training-only CV choice (`concat-2`, `concat-2`, `1e-3`, `1e-3`), so
the task test sets are never used for model or hyperparameter selection. Five
training-subset resamples and three optimization seeds produce 300 fits. Model
seeds are averaged within each subset; the five subset seeds are the reported
training-resampling units. Reusing the same test episodes does not create 15
independent test sets.

The workflow is:

1. rebuild or reuse `build_natural_rate_experiment`'s immutable all-task plan;
2. derive task-only manifests with `build_task_specific_rate_experiment`;
3. run disjoint five-task shards with `run_task_specific_rate_screen`;
4. summarize task ROC-AUC/AP, task-type macro metrics, and paired
   natural-minus-balanced effects with `summarize_task_specific_rate_screen`.

The durable outputs are `task_specific_summary.json`,
`task_specific_metrics.csv`, and balanced/natural ROC-AUC and AP figures in PNG
and PDF. Because each fitted monitor sees only one task, task-z normalization is
neither required nor applied.
