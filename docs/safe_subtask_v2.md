# Causal Subtask-SAFE v2

> Historical protocol note: the v2 gate output called its training-only
> per-stage target prevalence an "elapsed-time" baseline. That comparator was a
> valid stage-prior control but was mislabeled. Subtask-SAFE v3 reports the stage
> prior and duration-normalized progress separately and uses parent-disjoint
> held-out threshold calibration. See `docs/safe_subtask_v3.md`.

This protocol tests whether RLDX policy features predict failure **causally**, while
the current semantic subtask is still active. It is deliberately separate from the
retrospective whole-segment Subtask-SAFE result.

The implementation enforces four controls:

1. Parent rollouts are split before semantic segments or causal prefixes are used.
2. Training uses fixed prefixes (`1 2 4 8 16`) or deterministic random prefixes;
   test always uses the declared fixed prefixes.
3. Stage support and the one-hot stage catalog are learned from training data only.
4. A final gate report compares SAFE to a training-only elapsed-time baseline on
   identical segment and parent identities.

## New commands

- `robocasa.recovery.safe.plan_subtask_safe_collection` reports supported stages
  and targeted success/failure deficits.
- `robocasa.recovery.safe.run_seen_cv_grid` accepts causal-prefix and conditioning
  arguments.
- `robocasa.recovery.safe.train_seen_tasks` performs the final causal refit and
  writes source-segment/prefix identities to `scores.jsonl`.
- `robocasa.recovery.safe.evaluate_causal_subtask_gates` produces paired causal
  metrics, parent bootstrap confidence intervals, operating points, a plot, and a
  machine-readable continuation decision.

## Cluster setup

Run from a fresh allocation after pulling the integration branch. Large outputs
remain under `/gs/bs`.

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/vla_safe

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa_safe_integration"
export SAFE_REPO="$PROJECT_FS/SAFE"
export SAFE_ENV="$STORAGE_BS/envs/vla_safe"
export SAFE_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval"

export SUBTASK_SAFE_EXPORT="$STORAGE_BS/robocasa_rollouts/safe/rldx1_subtask_safe_official_20260803_212413"
export SUBTASK_PARENT_SPLIT="$SUBTASK_SAFE_EXPORT/parent_rollout_split.json"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROBOCASA_REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROBOCASA_REPO"
```

## 1. Select supported stages and quantify collection deficits

The defaults select stages with at least 10 successful and 5 failed training
segments, then target 25/15 train success/failure segments and 10/10 held-out
segments. These are segment deficits, not exact rollout counts.

```bash
export SUBTASK_V2_PLAN="$STORAGE_BS/robocasa_checkpoints/safe/rldx1_subtask_v2_plan"

python -u -m robocasa.recovery.safe.plan_subtask_safe_collection \
  --export-dir "$SUBTASK_SAFE_EXPORT" \
  --selection-manifest "$SUBTASK_PARENT_SPLIT" \
  --output-dir "$SUBTASK_V2_PLAN" \
  --min-train-successes 10 \
  --min-train-failures 5 \
  --target-train-successes 25 \
  --target-train-failures 15 \
  --target-test-successes 10 \
  --target-test-failures 10
```

Inspect `collection_plan.csv`. Collection should continue as natural complete
parent rollouts; do not independently sample stage segments. Export and regenerate
the parent split after adding data.

## 2. Causal inner-CV experiment

Use separate output roots for each treatment. The primary detector is `indep`
(the official SAFE MLP). The first comparison isolates whether explicit elapsed
progress helps beyond policy features and active-stage identity.

```bash
export CAUSAL_TAG="rldx1_subtask_v2_fixed_stage_$(date +%Y%m%d_%H%M%S)"
export CAUSAL_CV_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/$CAUSAL_TAG"

CUDA_VISIBLE_DEVICES=0 nohup "$SAFE_ENV/bin/python" -u -m \
  robocasa.recovery.safe.run_seen_cv_grid \
  --export-dir "$SUBTASK_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$CAUSAL_CV_ROOT" \
  --model indep \
  --selection-manifest "$SUBTASK_PARENT_SPLIT" \
  --task-type composite \
  --class-weighting official_inverse_frequency \
  --causal-prefix-mode fixed \
  --causal-prefix-horizons 1 2 4 8 16 \
  --causal-selection-prefix 8 \
  --causal-conditioning subtask_one_hot \
  --min-stage-successes 10 \
  --min-stage-failures 5 \
  --num-folds 3 \
  --epochs 1000 \
  --device cuda \
  --resume \
  > "$SAFE_LOG_ROOT/${CAUSAL_TAG}_indep.log" 2>&1 &
```

Run the elapsed-progress treatment in a distinct root by changing
`--causal-conditioning` to `subtask_one_hot_elapsed`. Run the random-prefix
ablation in another root with:

```text
--causal-prefix-mode random --random-prefixes-per-segment 3
```

Random prefix sampling is deterministic for the split seed and source segment.
The fixed-prefix treatment should be used for the final continuation-gate refit.

Summarize a completed grid:

```bash
python -m robocasa.recovery.safe.summarize_seen_cv \
  --root "$CAUSAL_CV_ROOT" \
  --expected-folds 0 1 2 \
  --quiet
```

## 3. Final fixed-prefix refits

The final command must use the exact causal protocol recorded by the selected CV
summary. A mismatch fails before training.

```bash
export CAUSAL_SELECTION="$CAUSAL_CV_ROOT/cv_selection_summary.json"
export CAUSAL_FINAL_ROOT="${CAUSAL_CV_ROOT}_final"

for SEED in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 "$SAFE_ENV/bin/python" -u -m \
    robocasa.recovery.safe.train_seen_tasks \
    --export-dir "$SUBTASK_SAFE_EXPORT" \
    --safe-repo "$SAFE_REPO" \
    --output-dir "$CAUSAL_FINAL_ROOT/indep_seed${SEED}" \
    --model indep \
    --seed "$SEED" \
    --split-seed 0 \
    --task-type composite \
    --selection-manifest "$SUBTASK_PARENT_SPLIT" \
    --selection-summary "$CAUSAL_SELECTION" \
    --class-weighting official_inverse_frequency \
    --causal-prefix-mode fixed \
    --causal-prefix-horizons 1 2 4 8 16 \
    --causal-conditioning subtask_one_hot \
    --min-stage-successes 10 \
    --min-stage-failures 5 \
    --epochs 1000 \
    --device cuda \
    --resume
done
```

Use one GPU per model or treatment when two GPUs are available; do not launch two
memory-heavy processes on the same device.

## 4. Apply the continuation gates

```bash
export CAUSAL_GATE_ROOT="$CAUSAL_FINAL_ROOT/causal_gate_analysis"

python -u -m robocasa.recovery.safe.evaluate_causal_subtask_gates \
  --final-root "$CAUSAL_FINAL_ROOT" \
  --output-dir "$CAUSAL_GATE_ROOT" \
  --models indep \
  --seeds 0 1 2 \
  --prefixes 1 2 4 8 16 \
  --target-prefix 8 \
  --min-roc 0.65 \
  --min-delta 0.05 \
  --target-fpr 0.10 \
  --min-tpr 0.40 \
  --min-lead 0.25 \
  --bootstrap-replicates 2000
```

The model proceeds to online recovery integration only if all gates pass:

- prefix-8 ROC-AUC is at least 0.65;
- SAFE beats elapsed time by at least 0.05 ROC-AUC;
- the paired parent-bootstrap 95% interval for that difference excludes zero;
- TPR is at least 0.40 at FPR no greater than 0.10;
- detected failures retain at least 25% normalized lead time.

If the gates fail, the prescribed next action is a representation/objective change
(for example, an earlier RLDX feature layer, temporal deltas, or direct language
conditioning), not indiscriminate additional rollout collection.

Generated artifacts are `summary.json`, `prefix_metrics.csv`,
`operating_points.csv`, `gate_report.md`, and `causal_prefix_vs_elapsed.png`.
