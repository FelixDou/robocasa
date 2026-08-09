# Stage-adapter Subtask-SAFE diagnostic

This is the bounded follow-up to the negative causal Subtask-SAFE result. It
uses the existing official Subtask-SAFE export; it does not collect new
rollouts, score the frozen outer test, or integrate recovery.

The experiment asks a narrower question: after controlling for semantic stage
and within-stage time, do stored RLDX action features add causal early-warning
signal?

## What is implemented

`robocasa.recovery.safe.run_subtask_stage_adapter` performs the complete
developmental protocol:

1. load the official Subtask-SAFE export and its parent-rollout outer split;
2. keep every outer-test parent locked and record only its identity/count;
3. split outer-training parents into meta-fit, selection, and untouched
   diagnostic sets, stratified by parent task and parent outcome;
4. select stages, stage horizons, feature scaling, and elapsed-time curves from
   meta-fit parents only;
5. build causal features at relative stage landmarks 10%, 25%, and 50% using
   current, delta, recent-mean, and recent-slope summaries;
6. compare a shared stage-conditioned model, shared trunk with small per-stage
   adapters, and isolated per-stage models;
7. compare balanced BCE with weak temporal localization constraints;
8. compare SAFE alone with a residual correction over stage-conditioned time
   risk;
9. select one entire pipeline on selection parents, then evaluate it once on
   diagnostic parents;
10. fit per-stage online thresholds on selection successes and report the first
    causal alarm on diagnostic segments.

The unit of resampling and splitting is always the parent rollout. The default
fit-only support floor is nine successful and five failed stage segments. Any
requested stage below that floor is reported as excluded rather than borrowing
diagnostic or outer-test examples.

## Continuation gate

The result passes only when all default conditions hold on the one-time
developmental diagnostic:

- macro stage ROC-AUC at 25% and 50% is at least 0.65;
- the parent-bootstrap lower confidence bound for improvement over
  stage-conditioned elapsed time is positive, and the point improvement is at
  least 0.05;
- event TPR is at least 0.40;
- macro and worst-stage FPR are at most 0.05;
- mean normalized lead among detected failures is at least 0.25.

Passing this gate justifies a fresh confirmatory parent pool. It is not itself
a confirmatory claim because the old outer identities have already informed
earlier Subtask-SAFE work.

## Cluster preflight

Start from the canonical exports in `docs/cluster_experiment_runbook.md`, then
activate the SAFE training environment and set the immutable inputs. Replace
`SUBTASK_SAFE_EXPORT` only if the intended augmented export has a different
path.

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/vla_safe

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export SAFE_REPO="$PROJECT_FS/SAFE"
export ROBOCASA_LOG_ROOT="$STORAGE_BS/robocasa_logs"
export ROBOCASA_CKPT_ROOT="$STORAGE_BS/robocasa_checkpoints"
export SUBTASK_SAFE_EXPORT="$STORAGE_BS/robocasa_rollouts/safe/rldx1_subtask_safe_official_20260803_212413"
export SUBTASK_PARENT_SPLIT="$SUBTASK_SAFE_EXPORT/parent_rollout_split.json"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_ENABLED=0

mkdir -p "$ROBOCASA_LOG_ROOT/eval" "$ROBOCASA_CKPT_ROOT/safe"
cd "$ROBOCASA_REPO"

test -f "$SUBTASK_SAFE_EXPORT/conversion_report.json"
test -f "$SUBTASK_PARENT_SPLIT"
test -f "$SAFE_REPO/failure_prob/train.py"
git -C "$SAFE_REPO" rev-parse HEAD
python -m robocasa.recovery.safe.run_subtask_stage_adapter --help
```

The SAFE repository commit is verified by the runner against the same pinned
official commit used by the existing SAFE tooling. The command stops before
training on an incompatible repository, invalid parent split, missing stage
support, or unavailable CUDA device.

## Bounded smoke

Run one adapter/residual configuration in the foreground first. The four stage
names below are the target-aware stages from the updated Subtask-SAFE analysis;
fit-only support rules may still exclude a deficient stage.

```bash
export STAGE_ADAPTER_SMOKE="$ROBOCASA_CKPT_ROOT/safe/subtask_stage_adapter_smoke_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=0 python -u -m robocasa.recovery.safe.run_subtask_stage_adapter \
  --export-dir "$SUBTASK_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --output-dir "$STAGE_ADAPTER_SMOKE" \
  --selection-manifest "$SUBTASK_PARENT_SPLIT" \
  --stages \
    LoadDishwasher::dishwasher_closed \
    PreSoakPan::TurnOnSinkFaucet_3 \
    ScrubCuttingBoard::cutting_board_scrubbed \
    WashLettuce::lettuce_rinsed \
  --architectures stage_adapter \
  --objectives temporal_contrastive \
  --detectors stage_safe_time \
  --seeds 0 \
  --regularizations 1e-4 \
  --epochs 20 \
  --patience 5 \
  --bootstrap-replicates 100 \
  --device cuda
```

Check that it finishes and that the protocol did not touch the outer test:

```bash
python - "$STAGE_ADAPTER_SMOKE" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
status = json.loads((root / "status.json").read_text())
analysis = json.loads((root / "analysis.json").read_text())
print(json.dumps(status, indent=2, sort_keys=True))
print(json.dumps(analysis["claims"], indent=2, sort_keys=True))
print(json.dumps(analysis["continuation_gate"], indent=2, sort_keys=True))
assert analysis["claims"]["outer_test_scored"] is False
assert analysis["claims"]["outer_test_used_for_training_or_selection"] is False
PY
```

## Full developmental screen

Use a fresh output directory. This screens 3 architectures, 2 objectives, 2
detectors, 4 regularizations, and 3 seeds. Only the selected pipeline reaches
the diagnostic parents.

```bash
export STAGE_ADAPTER_ROOT="$ROBOCASA_CKPT_ROOT/safe/subtask_stage_adapter_$(date +%Y%m%d_%H%M%S)"
export STAGE_ADAPTER_LOG="$ROBOCASA_LOG_ROOT/eval/$(basename "$STAGE_ADAPTER_ROOT").log"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 nohup python -u -m robocasa.recovery.safe.run_subtask_stage_adapter \
  --export-dir "$SUBTASK_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --output-dir "$STAGE_ADAPTER_ROOT" \
  --selection-manifest "$SUBTASK_PARENT_SPLIT" \
  --stages \
    LoadDishwasher::dishwasher_closed \
    PreSoakPan::TurnOnSinkFaucet_3 \
    ScrubCuttingBoard::cutting_board_scrubbed \
    WashLettuce::lettuce_rinsed \
  --architectures shared_one_hot stage_adapter stage_specific \
  --objectives bce temporal_contrastive \
  --detectors stage_safe stage_safe_time \
  --seeds 0 1 2 \
  --regularizations 1e-5 1e-4 1e-3 1e-2 \
  --landmarks 0.10 0.25 0.50 \
  --primary-landmarks 0.25 0.50 \
  --epochs 1000 \
  --patience 100 \
  --bootstrap-replicates 2000 \
  --device cuda \
  > "$STAGE_ADAPTER_LOG" 2>&1 &

echo "pid=$!"
echo "log=$STAGE_ADAPTER_LOG"
echo "output=$STAGE_ADAPTER_ROOT"
```

Monitor without starting a duplicate job:

```bash
ps -ef | grep '[r]un_subtask_stage_adapter'
tail -100 "$STAGE_ADAPTER_LOG"
cat "$STAGE_ADAPTER_ROOT/status.json"
```

## Result artifacts

The output root contains:

| Artifact | Meaning |
|---|---|
| `protocol.json` | immutable arguments, SAFE commit, support, horizons, and split audit |
| `split_manifest.json` | parent IDs in fit, selection, diagnostic, and locked outer test |
| `selection_audit.csv` | every pipeline/seed selection result |
| `selection_aggregate.csv` | seed-aggregated pipeline ranking |
| `runtime/` | selected or screened model states and preprocessing contracts |
| `diagnostic_landmark_metrics.csv` | within-stage causal ROC/AP at relative landmarks |
| `diagnostic_event_metrics.csv` | TPR, FPR, lead, and balanced accuracy per stage |
| `diagnostic_event_predictions.jsonl` | first alarms for auditable video alignment |
| `thresholds.json` | selection-only per-stage operating thresholds |
| `analysis.json` | selected pipeline, bootstrap comparison, claims, and gate |
| `status.json` | initializing, running, failed, or complete state |

If the gate fails, do not tune against the diagnostic result. The stored RLDX
features have then failed this bounded representation/objective test; the next
scientific change is earlier-layer visual/state feature capture with a fresh
parent allocation.

## Observation-conditioned representation follow-up

The augmented stage-adapter screen failed its online event-TPR gate. Do not
retune thresholds, smoothing, or model selection on those consumed diagnostic
parents. The implemented follow-up instead captures an observation-conditioned
RLDX representation on a fresh parent pool.

The new `action_observation_context` mode preserves the existing action-token
latent byte-for-byte in the original `features` tensor and separately stores,
once per genuine policy inference:

- an attention-mask-aware mean of the vision/language backbone tokens after
  RLDX memory processing; and
- a mean of the robot-state tokens after the embodiment-specific state encoder.

The context is stored as `(inferences, context_dim)` rather than broadcast over
the denoising-step and action-horizon axes. This avoids multiplying its storage
cost while recording exact context-component slices. A single collected
rollout can therefore be exported as `action`, `observation_context`, or `all`,
producing a paired representation ablation with identical actions, parents,
labels, and inference timestamps. The `all` export selects the final denoising
step, mean-pools its action horizon, concatenates the context, and adds singleton
SAFE axes `(1, 1, combined_dim)`. This exactly matches loading the raw `action`
view with `--diffusion-selector 1.0 --horizon-selector mean`.

This is a Subtask-SAFE representation experiment. It does not redefine the
separate raw rollout-level SAFE baseline.

### Apply the incremental RLDX patch

The observation patch applies after the existing raw RLDX SAFE patch at the
pinned benchmark commit:

```bash
export RLDX_BASE_SAFE_PATCH="$ROBOCASA_REPO/patches/rldx1_safe_features_ef05cd4.patch"
export RLDX_OBSERVATION_PATCH="$ROBOCASA_REPO/patches/rldx1_safe_observation_context_ef05cd4.patch"

test "$(git -C "$RLDX_REPO" rev-parse HEAD)" = \
  ef05cd4ae634ff97d672d42275febbc0b92cc192

if git -C "$RLDX_REPO" apply --reverse --check "$RLDX_BASE_SAFE_PATCH" 2>/dev/null; then
  echo "base SAFE patch already applied"
else
  git -C "$RLDX_REPO" apply --check "$RLDX_BASE_SAFE_PATCH" &&
  git -C "$RLDX_REPO" apply "$RLDX_BASE_SAFE_PATCH"
fi

if git -C "$RLDX_REPO" apply --reverse --check "$RLDX_OBSERVATION_PATCH" 2>/dev/null; then
  echo "observation-context patch already applied"
else
  git -C "$RLDX_REPO" apply --check "$RLDX_OBSERVATION_PATCH" &&
  git -C "$RLDX_REPO" apply "$RLDX_OBSERVATION_PATCH"
fi

cd "$RLDX_REPO"
python -m py_compile \
  rldx/model/core/rldx.py \
  rldx/policy/policy_runtime.py \
  rldx/policy/rldx_policy.py \
  rldx/policy/step_request.py
```

Start a fresh eager RLDX server following `docs/cluster_experiment_runbook.md`.
Compiled inference is still outside this feature-capture contract.

### One-rollout feature-contract smoke

Use a fresh output directory under `/gs/bs`. One smoke verifies wiring and
shape only; it is not usable training evidence.

```bash
export OBS_CONTEXT_SMOKE_TAG="rldx1_subtask_observation_context_smoke_$(date +%Y%m%d_%H%M%S)"
export OBS_CONTEXT_SMOKE="$STORAGE_BS/robocasa_rollouts/safe/$OBS_CONTEXT_SMOKE_TAG"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$OBS_CONTEXT_SMOKE" \
  --tasks PreSoakPan \
  --num-rollouts 1 \
  --seed 7 \
  --seed-protocol official_rldx \
  --policy-module robocasa.recovery.rldx_zmq_policy:make_policy \
  --model-family rldx1 \
  --safe-feature-mode action_observation_context \
  --policy-name RLDX-1-FT-RC365 \
  --checkpoint RLWRLD/RLDX-1-FT-RC365 \
  --policy-config '{"embodiment_tag":"GENERAL_EMBODIMENT","safe_feature_mode":"action_observation_context"}' \
  --host 127.0.0.1 \
  --port 20100 \
  --split target \
  --replan-steps 8 \
  --record-safe-features \
  --record-actions \
  --record-subtask-trace \
  --record-videos \
  --max-errors 1

python -m robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$OBS_CONTEXT_SMOKE"
```

Inspect the saved component contract:

```bash
python - "$OBS_CONTEXT_SMOKE" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
record = json.loads(next(line for line in (root / "manifest.jsonl").read_text().splitlines() if line))
with np.load(root / record["tensor_path"], allow_pickle=False) as payload:
    features = payload["features"]
    context = payload["observation_context"]
print("layer:", record["feature_layer"])
print("observation layer:", record["observation_feature_layer"])
print("schema:", record["feature_schema_version"])
print("action shape:", features.shape)
print("observation shape:", context.shape)
print("observation components:", json.dumps(record["observation_components"], indent=2))
print("finite:", bool(np.isfinite(features).all() and np.isfinite(context).all()))
print("nonzero:", bool(np.any(features) and np.any(context)))
PY
```

Expected schema is `2`. The raw action tensor remains four-dimensional, the
context is two-dimensional, and the named observation slices must be contiguous
and cover the context feature dimension exactly.

### Paired exports after fresh collection

After collecting and validating the new seed-disjoint parent pool, materialize
three views from the same dual-stream dataset:

```bash
export FRESH_DUAL_DATASET=/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/REPLACE_WITH_FRESH_DUAL_DATASET
export FRESH_EXPORT_ROOT="$STORAGE_BS/robocasa_rollouts/safe/$(basename "$FRESH_DUAL_DATASET")_views"

for view in action observation_context all; do
  python -u -m robocasa.recovery.safe.export_subtask_safe \
    --dataset-dir "$FRESH_DUAL_DATASET" \
    --output-dir "$FRESH_EXPORT_ROOT/$view" \
    --feature-view "$view" \
    --split-seed 0
done

python - "$FRESH_EXPORT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
splits = []
for view in ("action", "observation_context", "all"):
    payload = json.loads((root / view / "parent_rollout_split.json").read_text())
    payload.pop("export_dir", None)
    splits.append(payload)
assert splits[0] == splits[1] == splits[2]
print("paired parent split: VERIFIED")
PY
```

The split contents are deterministic for the same source and seed, although
their `export_dir` provenance field differs. Run the developmental screen on
each view without inspecting or reusing the previous diagnostic parents. The
`action` view preserves the raw tensor. For the paired screen, run all three
views with `--horizon-selector mean --diffusion-selector 1.0`; this makes the
action component of `all` identical to the selected `action` view. The compact
`observation_context` and `all` exports already have singleton denoising and
horizon axes, so those selectors are an identity for them.

At a 5% per-stage FPR target, plan at least 20 successful calibration segments
per stage for empirical resolution. A decision-grade fresh evaluation needs at
least 59 successful segments per stage to put the one-sided 95% upper bound
below 5% when zero false alarms are observed; target 60 or more, plus 30–50
failed evaluation segments per stage. Freeze the feature view, training
pipeline, and online threshold rule before that final evaluation.
