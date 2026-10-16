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
