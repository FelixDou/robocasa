# Collecting atomic RoboCasa rollouts for official SAFE π0

For the distinct RLDX-1 feature contract and ZeroMQ collection path, see
[`safe_rldx1.md`](safe_rldx1.md). RLDX-1 and π0 feature datasets must not be
mixed.

## Scope

This integration collects natural π0 task successes and natural π0 task failures in registered RoboCasa atomic environments. RoboCasa's task-success predicate is the only label authority: `failure_label` is `0` when the task succeeds and `1` otherwise. Reaching the rollout horizon without success is a failure. No frame-level failure onset, subtask label, recovery behavior, expert conversion, model training, calibration, or detector evaluation is part of this collection path.

## Official sources and inspected revisions

The implementation was checked against these upstream repositories and revisions on 2026-07-13:

- SAFE: `https://github.com/vla-safe/SAFE`, commit `b6036abe07b2b2bb9996afb2c07f13d6a9f507c0`
- SAFE's OpenPI fork: `https://github.com/vla-safe/openpi`, commit `9c99ed53f6a0c9be93a1c63cee5792620777d96b`
- RoboCasa's OpenPI fork: `https://github.com/robocasa-benchmark/openpi`, commit `5a6beda9ff99da30b4e1b59320f6a32971d7c397`
- this RoboCasa checkout at inspection time: `4a3bf301474ff254a67ce98f88ad71f01676a4bb`

Exact source snapshots used during integration were under `/tmp/safe_sources`. They are inspection snapshots, not durable clones. No persistent SAFE or OpenPI checkout was found beside this RoboCasa repository. The server-side integration is therefore supplied as `patches/openpi_safe_features_5a6beda.patch`, which applies to the pinned RoboCasa OpenPI commit. SAFE's relevant loader is `failure_prob/data/pizero.py`; the OpenPI feature is defined in `src/openpi/models/pi0.py` and exposed in `src/openpi/policies/policy.py`.

## Exact latent and inference association

Official SAFE records π0's `pre_velocity`: the final action-expert suffix hidden state for every predicted action position immediately before `action_out_proj`. It is not projected velocity and not an image embedding. For the official π0 configuration, each real policy inference produces float32 features with shape:

```text
(flow_steps=10, action_horizon=50, action_expert_width=1024)
```

RoboCasa preserves these raw axes. A rollout feature tensor therefore has shape `(num_policy_inferences, flow_steps, action_horizon, width)`. The official loader may later select or aggregate flow and horizon positions; collection does not average either axis.

`OpenPIWebsocketPolicy` requests the feature explicitly with `request_safe_features=true`. The server responds with normal actions plus `safe_features` and `safe_feature_metadata`. The client validates shape, floating dtype, finite values, nonzero content, action-horizon agreement, and feature identity. Missing features are an error only when SAFE collection is enabled; ordinary action-only calls remain supported.

The client creates one record when it actually calls `client.infer`. It does not create records while consuming cached actions. With `replan_steps=5`, inference steps are expected to be `0, 5, 10, ...`, with one feature record per inference rather than five copies. Every record also stores the full predicted action chunk because the official SAFE π0 loader consumes both `pre_velocity` and `actions`. `reset()` clears episode state, and an instruction change clears cached actions and restarts the inference-record sequence.

## Collector and labels

The dedicated entry point is:

```text
python -m robocasa.recovery.safe.collect_atomic_rollouts
```

It validates task names and resolves each task's official horizon by parsing `ATOMIC_TASK_DATASETS` without importing RoboSuite. `--horizon` is an explicit all-task override; it is required with `--allow-unregistered-atomic-tasks`. Important options include `--tasks`, `--num-rollouts`, `--seed`, `--seed-end`, `--seed-protocol`, `--output-dir`, `--policy-module`, repeatable `--policy-arg`, `--host`, `--port`, `--policy-name`, `--checkpoint`, `--policy-config`, `--horizon`, `--replan-steps`, `--record-videos`, `--video-frame-stride`, `--record-actions`, `--record-safe-features`, `--continue-on-error`, `--resume`, `--success-quota`, `--failure-quota`, and `--dry-run`. Boolean options support their `--no-...` form.

The default `rollout_index` seed protocol creates a fresh environment with seed `seed + rollout_index`. For evaluation-parity experiments, `--seed-protocol official_openpi --seed 7` instead creates one environment per task and obtains episodes by repeatedly calling `reset()`, as the pinned official OpenPI RoboCasa evaluator does. Each manifest row records the shared base seed and its distinct `environment_reset_index`. The official evaluator also records every second environment step and writes at 20 FPS; use `--video-frame-stride 2 --video-fps 20` to match its video timing. For example, a full 1050-step `OpenCabinet` rollout becomes 526 frames, or approximately 26.3 seconds.

The collector obtains the final task result from the simulator success predicate. It records executed actions separately from the full action chunks predicted at inference time. It only appends a manifest row after the video, executed-action file, and SAFE tensor file requested for that rollout have been finalized.

## Dataset schema and directory layout

Each JSONL manifest row contains:

```text
rollout_id, task_name, task_instruction, environment_seed
seed_protocol, environment_reset_index, environment_split
success, failed, failure_label, termination_reason
num_environment_steps, num_policy_inferences, inference_environment_steps
safe_feature_path, safe_feature_shape, safe_feature_dtype
safe_feature_layer, safe_feature_aggregation
policy_name, policy_checkpoint, policy_config
replan_steps, action_horizon, rollout_horizon
action_path, video_path, video_frame_stride
safe_repository_commit, openpi_repository_commit, robocasa_commit, created_at
schema_version, feature_schema_version, collection_complete
```

The directory structure is:

```text
dataset/
  manifest.jsonl
  summary.json
  errors.jsonl                         # present when collection/resume errors occur
  skipped.jsonl                        # present for quota skips or completed resume entries
  rollouts/<rollout_id>.npz
  actions/<task>/<rollout_id>.npz      # optional executed actions
  videos/<task>/<rollout_id>.mp4       # optional
  incomplete/<rollout_id>/...          # quarantined partial artifacts
```

Each rollout NPZ contains raw `features`, `inference_environment_steps`, `valid_length`, `rollout_id`, serialized feature metadata, the binary failure flag, and `policy_action_chunks`. The feature and action axes remain available to the official loader. Files use temporary names and atomic replacement where practical; the append-only manifest is fsynced after each complete rollout.

Rollout IDs are deterministic hashes of task, seed protocol, seed/reset index, policy/checkpoint identity, policy config, horizon, `replan_steps`, and OpenPI revision. Resume refuses incompatible configuration, never overwrites a completed rollout, and never appends a second valid row for it. Existing artifacts without a valid manifest row are moved under `incomplete/` before recollection. Errors and every skipped episode are journaled. Success/failure quotas stop future attempts after both requested class counts are reached; already completed valid rollouts are never discarded. An official-protocol comparison should start from a fresh policy server and run uninterrupted when possible, because resuming can reconstruct the environment reset sequence but cannot replay policy-server random-number consumption from skipped episodes.

## Prepare the SAFE-enabled OpenPI server

Apply the companion patch in a checkout of the pinned RoboCasa OpenPI revision:

```bash
export ROBOCASA_REPO=/gs/fs/tga-shinoda/felid/robocasa
export OPENPI_REPO=/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/openpi

git -C "$OPENPI_REPO" checkout 5a6beda9ff99da30b4e1b59320f6a32971d7c397
git -C "$OPENPI_REPO" apply "$ROBOCASA_REPO/patches/openpi_safe_features_5a6beda.patch"
cd "$OPENPI_REPO"
python -m py_compile \
  src/openpi/models/pi0.py \
  src/openpi/policies/policy.py \
  src/openpi/policies/policy_config.py
```

Start that checkout's normal `scripts/serve_policy.py` entry point with the π0 RoboCasa config and checkpoint. The specific checkpoint path remains installation-dependent; set it once for all commands:

```bash
export PI0_CHECKPOINT=/tmp/ut06746/pi0_74999
export SAFE_DATASET=/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/pi0_74999_atomic
cd /gs/fs/tga-shinoda/felid/robocasa
```

After copying the durable checkpoint to the node-local path as described in the cluster runbook, start the patched server:

```bash
cd "$OPENPI_REPO"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -u scripts/serve_policy.py \
  --port=8120 policy:checkpoint \
  --policy.config=pi0_robocasa_pretrain_human300 \
  --policy.dir="$PI0_CHECKPOINT"
```

## Future collection commands

Official OpenPI-parity smoke test (base seed 7, repeated resets, official horizon, and official video subsampling):

```bash
python -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$SAFE_DATASET" \
  --tasks OpenCabinet \
  --num-rollouts 10 \
  --seed 7 \
  --seed-protocol official_openpi \
  --policy-name pi0_robocasa_pretrain_human300 \
  --checkpoint "$PI0_CHECKPOINT" \
  --policy-config '{"config_name":"pi0_robocasa_pretrain_human300","step":74999}' \
  --host 127.0.0.1 \
  --port 8120 \
  --split pretrain \
  --replan-steps 5 \
  --record-actions \
  --record-videos \
  --video-frame-stride 2 \
  --video-fps 20 \
  --record-safe-features
```

Planning is simulator-free and does not contact the OpenPI server:

```bash
python -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$SAFE_DATASET" \
  --tasks TurnOnSinkFaucet \
  --num-rollouts 1 \
  --seed 13000 \
  --policy-name pi0_robocasa_pretrain_human300 \
  --checkpoint "$PI0_CHECKPOINT" \
  --host 127.0.0.1 \
  --port 8120 \
  --split target \
  --replan-steps 5 \
  --record-videos \
  --dry-run
```

One-rollout live protocol smoke test, to run only when RoboSuite, RoboCasa assets, the checkpoint, and patched server are available:

```bash
python -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$SAFE_DATASET" \
  --tasks TurnOnSinkFaucet \
  --num-rollouts 1 \
  --seed 13000 \
  --policy-name pi0_robocasa_pretrain_human300 \
  --checkpoint "$PI0_CHECKPOINT" \
  --policy-config '{"config_name":"pi0_robocasa_pretrain_human300","step":74999}' \
  --host 127.0.0.1 \
  --port 8120 \
  --split target \
  --replan-steps 5 \
  --record-actions \
  --record-videos \
  --record-safe-features
```

Multi-task collection over registered atomic tasks:

```bash
python -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$SAFE_DATASET" \
  --tasks OpenMicrowave CloseMicrowave TurnOnSinkFaucet TurnOnStove \
  --num-rollouts 100 \
  --seed 20000 \
  --policy-name pi0_robocasa_pretrain_human300 \
  --checkpoint "$PI0_CHECKPOINT" \
  --policy-config '{"config_name":"pi0_robocasa_pretrain_human300","step":74999}' \
  --host 127.0.0.1 \
  --port 8120 \
  --split target \
  --replan-steps 5 \
  --record-actions \
  --record-videos
```

The same collection with per-task class stopping conditions:

```bash
python -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$SAFE_DATASET" \
  --tasks OpenMicrowave CloseMicrowave TurnOnSinkFaucet TurnOnStove \
  --num-rollouts 100 \
  --seed 20000 \
  --success-quota 20 \
  --failure-quota 20 \
  --policy-name pi0_robocasa_pretrain_human300 \
  --checkpoint "$PI0_CHECKPOINT" \
  --policy-config '{"config_name":"pi0_robocasa_pretrain_human300","step":74999}' \
  --host 127.0.0.1 \
  --port 8120 \
  --split target \
  --replan-steps 5 \
  --record-actions \
  --record-videos
```

Resume with the same identity and artifact configuration:

```bash
python -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$SAFE_DATASET" \
  --tasks OpenMicrowave CloseMicrowave TurnOnSinkFaucet TurnOnStove \
  --num-rollouts 100 \
  --seed 20000 \
  --success-quota 20 \
  --failure-quota 20 \
  --policy-name pi0_robocasa_pretrain_human300 \
  --checkpoint "$PI0_CHECKPOINT" \
  --policy-config '{"config_name":"pi0_robocasa_pretrain_human300","step":74999}' \
  --host 127.0.0.1 \
  --port 8120 \
  --split target \
  --replan-steps 5 \
  --record-actions \
  --record-videos \
  --resume
```

## Validation and official SAFE export

Validate the source dataset. Invalid data produces exit status 1; `--json-output` also writes the structured report:

```bash
python -m robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$SAFE_DATASET" \
  --json-output "$SAFE_DATASET/validation.json"
```

The official SAFE π0 loader requires `env_records/*.pkl` and a globally ordered `policy_records/*meta.pkl` file per policy inference. The deterministic adapter validates the source first, materializes `pre_velocity` and predicted `actions` into that exact layout, preserves rollout IDs and labels, and symlinks videos when present. Per-inference feature arrays must be materialized because the unmodified official loader opens each pickle; the report records this duplication.

Preview and then export:

```bash
python -m robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir "$SAFE_DATASET" \
  --output-dir "${SAFE_DATASET}_official" \
  --dry-run

python -m robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir "$SAFE_DATASET" \
  --output-dir "${SAFE_DATASET}_official"
```

To preserve the complete source while producing an exact deterministic class
balance for official training, select equal per-task class counts during export:

```bash
python -m robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir "$SAFE_DATASET" \
  --output-dir "${SAFE_DATASET}_official_balanced_10x10" \
  --successes-per-task 10 \
  --failures-per-task 10 \
  --selection-seed 0
```

The conversion report records the source counts, selected counts, and selection
seed. The source manifest and artifacts are never modified.

Resume an interrupted export with the same source manifest:

```bash
python -m robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir "$SAFE_DATASET" \
  --output-dir "${SAFE_DATASET}_official" \
  --resume
```

## Validate with the unmodified official SAFE loader

The source validator proves that the RoboCasa tensors and metadata can be
exported, but the final compatibility gate is loading the materialized files
through the official SAFE `failure_prob.data.pizero` implementation. Run this
inside the dedicated SAFE environment after installing the pinned SAFE commit:

```bash
export SAFE_REPO=/gs/fs/tga-shinoda/felid/SAFE
export SAFE_OFFICIAL="${SAFE_DATASET}_official_balanced_10x10"

cd /gs/fs/tga-shinoda/felid/robocasa
python -m robocasa.recovery.safe.validate_official_export \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --horizon-selector 0.0 \
  --diffusion-selector 0.0 \
  --expected-rollouts 100 \
  --expected-successes 50 \
  --expected-failures 50 \
  --expected-task CloseFridge \
  --expected-task OpenDrawer \
  --expected-task PickPlaceCounterToCabinet \
  --expected-task PickPlaceCounterToStove \
  --expected-task TurnOnSinkFaucet \
  --json-output "$SAFE_OFFICIAL/official_loader_validation.json"
```

This invokes the upstream loader itself, checks the post-aggregation tensors,
and fails with status 1 if counts, task coverage, shapes, or finite-value checks
do not match.

## Official SAFE training smoke

The official pi0 experiment is a validation-selected grid, not a single fixed
hyperparameter configuration. Before launching that grid, verify the complete
training path with a two-epoch SAFE-MLP run. Keep Hydra, W&B, logs, and weights
under `/gs/bs` because the upstream defaults are relative to its source tree.

```bash
export RUN_TAG="safe_pi0_robocasa_smoke_$(date +%Y%m%d_%H%M%S)"
export RUN_ROOT="/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/$RUN_TAG"
mkdir -p "$RUN_ROOT"

cd "$SAFE_REPO"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled WANDB_DISABLED=true \
python -u -m failure_prob.train \
  dataset=pizero \
  dataset.data_path="$SAFE_OFFICIAL" \
  dataset.horizon_idx_rel=0.0 \
  dataset.diff_idx_rel=0.0 \
  model=indep \
  model.lr=1e-4 \
  model.lambda_reg=1e-2 \
  model.n_epochs=2 \
  train.seed=0 \
  train.eval_save_logs=true \
  train.eval_save_ckpt=true \
  train.logs_save_path="$RUN_ROOT/artifacts" \
  train.wandb_dir="$RUN_ROOT/wandb" \
  "hydra.run.dir=$RUN_ROOT/hydra" \
  2>&1 | tee "$RUN_ROOT/train.log"

test -s "$RUN_ROOT/artifacts/model_final.ckpt"
test -s "$RUN_ROOT/artifacts/config.yaml"
grep -E 'Loaded 100 rollouts|train:|val_seen:|val_unseen:' "$RUN_ROOT/train.log"
```

The pinned official pi0 SAFE-MLP sweep uses 1,000 epochs, batch size 512,
Adam, a two-layer width-256 MLP, horizon and diffusion selectors
`0.0,1.0,concat-2`, learning rates
`1e-5,3e-5,1e-4,3e-4,1e-3`, regularization
`1e-3,1e-2,1e-1`, and seeds `0,1,2`. That is 405 fits for SAFE-MLP alone.
Select the final configuration by validation-seen ROC-AUC as in the official
procedure. Do not describe the two-epoch command above as the final model.

With only five tasks, the official 30% unseen-task split leaves three seen and
two unseen tasks. This is a useful pilot of cross-task transfer, but it is not a
high-powered unseen-task benchmark; report the actual task IDs assigned to each
split for every seed.

## Offline official evaluation and model selection

The upstream trainer logs validation and conformal tables to W&B but does not
persist them when W&B is disabled. The RoboCasa evaluator imports the pinned
official loader, model, scalar metrics, and functional conformal implementation
and writes durable local artifacts instead. It reproduces the official
`train`/`val_seen`/`val_unseen` split for the checkpoint seed, calibrates on
successful `val_seen` trajectories, edge-extends trajectories, uses the
official 30/70 regression/modulation partition, and detects with `score >=
upper_band`.

Evaluate the completed two-epoch smoke checkpoints before starting the grid:

```bash
cd /gs/fs/tga-shinoda/felid/robocasa

python -u -m robocasa.recovery.safe.evaluate_official_safe \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --checkpoint /gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_pi0_indep_smoke_20260716_145732/artifacts/model_final.ckpt \
  --config /gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_pi0_indep_smoke_20260716_145732/artifacts/config.yaml \
  --output-dir /gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_pi0_indep_smoke_20260716_145732/evaluation \
  --device cuda
```

Every evaluation directory contains:

```text
metrics.json                 # official scalar metrics plus duration diagnostics
split_manifest.json          # exact rollout IDs and per-task class counts
scores.jsonl                 # one score trajectory per rollout
functional_conformal.csv     # official alpha sweep and confusion/detection metrics
functional_bands.npz         # saved model/constant/time-only upper bands
provenance.json              # commits, hashes, selectors, model config, seed
```

`metrics.json` includes constant-score and causal absolute-time baselines. The
time-only curve increases with inference index, so it exposes the case where
failures are predictable merely because they run longer. The official
`falert_early` metric is the matched per-task earliest-stop sensitivity check;
the duration-only ROC-AUC is reported separately and triggers a warning at
ROC-AUC at least 0.75 (or at most 0.25).

## Resumable official MLP/LSTM grid

The grid runner expands exactly the pinned SAFE pi0 grid: two models, three
action-horizon selectors, three diffusion selectors, five learning rates,
three regularization values, and three seeds, for 810 fits. Each fit uses the
unmodified official model, loss, optimizer, class weights, batch size 512, and
1,000 epochs, followed by the offline evaluator above. It never re-exports or
modifies the dataset.

Preview the complete plan without training:

```bash
export SAFE_GRID_ROOT=/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_pi0_official_grid_$(date +%Y%m%d_%H%M%S)

cd /gs/fs/tga-shinoda/felid/robocasa
python -m robocasa.recovery.safe.run_official_grid \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --robocasa-repo "$PWD" \
  --output-root "$SAFE_GRID_ROOT" \
  --dry-run
```

On a two-GPU allocation, launch deterministic disjoint shards. Each shard has
405 fits and can be restarted with `--resume`. A numerically invalid grid point
is recorded in that run's `failure.json` and does not terminate the shard;
subsequent resumes skip it unless `--retry-errors` is requested:

The driver invokes the evaluator by file path rather than importing the root
`robocasa` package. Official SAFE evaluation therefore does not require
RoboSuite or simulator dependencies in the dedicated SAFE environment.

```bash
export GRID_LOG_ROOT=/gs/bs/tga-shinoda/felid/robocasa_logs/eval
mkdir -p "$SAFE_GRID_ROOT" "$GRID_LOG_ROOT"

CUDA_VISIBLE_DEVICES=0 nohup python -u -m robocasa.recovery.safe.run_official_grid \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --robocasa-repo /gs/fs/tga-shinoda/felid/robocasa \
  --output-root "$SAFE_GRID_ROOT" \
  --num-shards 2 \
  --shard-index 0 \
  --resume \
  > "$GRID_LOG_ROOT/$(basename "$SAFE_GRID_ROOT")_shard0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u -m robocasa.recovery.safe.run_official_grid \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --robocasa-repo /gs/fs/tga-shinoda/felid/robocasa \
  --output-root "$SAFE_GRID_ROOT" \
  --num-shards 2 \
  --shard-index 1 \
  --resume \
  > "$GRID_LOG_ROOT/$(basename "$SAFE_GRID_ROOT")_shard1.log" 2>&1 &
```

After all shards finish, reproduce official selection by averaging each
configuration over seeds 0, 1, and 2 and maximizing `falert_early_roc_auc` on
`val_seen`. `val_unseen` is reported but never used for selection:

```bash
python -m robocasa.recovery.safe.summarize_official_grid \
  --output-root "$SAFE_GRID_ROOT" \
  --quiet
```

This writes `selection_summary.json` and `selection_summary.csv`. A
configuration is eligible for selection only after all three seeds complete.

Once all 810 evaluations are present, build the compact final scientific
report for the selected MLP and LSTM configurations. Invoke this script by file
path in the SAFE environment so RoboSuite is not required:

```bash
export SAFE_FINAL_REPORT="$SAFE_GRID_ROOT/final_report"

python "$ROBOCASA_REPO/robocasa/recovery/safe/report_official_grid.py" \
  --grid-root "$SAFE_GRID_ROOT" \
  --output-dir "$SAFE_FINAL_REPORT" \
  --expected-seeds 0 1 2 \
  --quiet
```

The final directory contains `final_report.json`,
`selected_conformal_summary.csv`, validation/held-out AUC, conformal tradeoff,
per-task, representative score/band, ROC/PR, and detection-time plots. The
report foregrounds matched-earliest metrics because full termination time is a
perfect or near-perfect outcome proxy in this RoboCasa collection.

## All-five-seen 7+7 training / 3+3 test protocol

When the deployment task set is known, use the outcome-stratified same-task
protocol rather than interpreting the official held-out-task split. For each of
the five tasks, the fixed split places seven successes and seven failures in
training and leaves the remaining three successes and three failures for test:
70 training rollouts and 30 evaluation-only rollouts total. The same split is
shared across model seeds. Hyperparameters are frozen from the completed
official 810-fit sweep:

- SAFE-MLP: horizon `1.0`, diffusion `0.0`, learning rate `3e-4`, regularization `1e-3`.
- SAFE-LSTM: horizon `1.0`, diffusion `concat-2`, learning rate `1e-3`, regularization `1e-2`.

This experiment reports matched-earliest ROC-AUC and PRC-AUC overall and per
task. It deliberately does not fit a conformal threshold: using any of the 30
test rollouts for calibration would contaminate the requested six-rollout
evaluation set, while calibrating on model-training successes would give an
optimistic false-alarm estimate. Collect fresh same-task successes later when
an honest deployment threshold is needed.

Run the three MLP seeds on GPU 0 and the three LSTM seeds on GPU 1. Invoke the
standalone file so the SAFE environment does not need RoboSuite:

```bash
export SAFE_SEEN_ROOT=/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_all5_seen_7x7_train_3x3_test_$(date +%Y%m%d_%H%M%S)
export SAFE_SEEN_LOG_ROOT=/gs/bs/tga-shinoda/felid/robocasa_logs/eval
mkdir -p "$SAFE_SEEN_ROOT" "$SAFE_SEEN_LOG_ROOT"

CUDA_VISIBLE_DEVICES=0 nohup bash -lc '
  set -euo pipefail
  for seed in 0 1 2; do
    python -u "$ROBOCASA_REPO/robocasa/recovery/safe/train_seen_tasks.py" \
      --export-dir "$SAFE_OFFICIAL" \
      --safe-repo "$SAFE_REPO" \
      --output-dir "$SAFE_SEEN_ROOT/indep_seed${seed}" \
      --model indep \
      --seed "$seed" \
      --split-seed 0 \
      --train-per-class 7 \
      --epochs 1000 \
      --device cuda \
      --resume
  done
' > "$SAFE_SEEN_LOG_ROOT/$(basename "$SAFE_SEEN_ROOT")_indep.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash -lc '
  set -euo pipefail
  for seed in 0 1 2; do
    python -u "$ROBOCASA_REPO/robocasa/recovery/safe/train_seen_tasks.py" \
      --export-dir "$SAFE_OFFICIAL" \
      --safe-repo "$SAFE_REPO" \
      --output-dir "$SAFE_SEEN_ROOT/lstm_seed${seed}" \
      --model lstm \
      --seed "$seed" \
      --split-seed 0 \
      --train-per-class 7 \
      --epochs 1000 \
      --device cuda \
      --resume
  done
' > "$SAFE_SEEN_LOG_ROOT/$(basename "$SAFE_SEEN_ROOT")_lstm.log" 2>&1 &
```

After six `metrics.json` files appear, aggregate the same-task test result:

```bash
python "$ROBOCASA_REPO/robocasa/recovery/safe/summarize_seen_tasks.py" \
  --root "$SAFE_SEEN_ROOT" \
  --expected-seeds 0 1 2
```

### Score overlays on rollout videos

`render_score_videos.py` maps video frames back to environment steps and then
to genuine pi0 inference calls. The overlay shows the current causal SAFE
score, maximum score so far, complete growing score trace, task, model, seed,
and ground-truth outcome. It explicitly labels the visualization as score-only;
it does not display a test-fitted threshold.

Render six balanced test examples from a chosen run:

```bash
python "$ROBOCASA_REPO/robocasa/recovery/safe/render_score_videos.py" \
  --scores "$SAFE_SEEN_ROOT/lstm_seed0/scores.jsonl" \
  --output-dir "$SAFE_SEEN_ROOT/lstm_seed0/score_videos" \
  --split test \
  --max-videos 6
```

Omit `--max-videos` to render all 30 held-out videos.

### Final all-seen result figures

After the leakage-free inner-CV sweep and six final refits complete, create a
reproducible static figure set directly from the saved metrics and score
trajectories:

```bash
export SAFE_FINAL_PLOTS="$SAFE_FINAL_ROOT/result_plots"

python "$ROBOCASA_REPO/robocasa/recovery/safe/plot_seen_results.py" \
  --final-root "$SAFE_FINAL_ROOT" \
  --cv-summary "$SAFE_CV_ROOT/cv_selection_summary.json" \
  --output-dir "$SAFE_FINAL_PLOTS" \
  --formats png pdf
```

The command writes five figures in PNG and PDF format: aggregate test ROC/PRC,
per-seed sensitivity, the inner-CV-to-test gap, SAFE versus the episode-duration
baseline, and seed-0 held-out score trajectories. It also writes
`per_seed_metrics.csv`, `summary_metrics.csv`, and `plot_manifest.json` so every
plotted value remains auditable. Error bars are population standard deviations
across the three training seeds. No threshold is fitted on test rollouts, and
per-task AUC is intentionally omitted because each task has only three test
successes and three test failures.

## Training-only inner-CV sweep for the all-five-seen protocol

The first all-seen run reused hyperparameters selected by the official
held-out-task protocol. To tune specifically for the 7+7 setting without
leaking the fixed 3+3 test rollouts, run three-fold inner cross-validation only
inside the 70-rollout training pool. Every task/outcome group contributes
seven training-pool examples, partitioned 3/2/2 across the validation folds.
The outer 30 rollouts are neither scored nor used for selection.

The sweep keeps the official SAFE search space and model implementation:
three horizon selectors, three diffusion selectors, five learning rates, three
regularization values, and three folds. This is 405 fits per architecture and
810 total. Matched-horizon cutoffs are frozen from the 70 outer-training
rollouts only. Each GPU worker loads a feature-selector pair once and evaluates
all 45 associated fits, avoiding 405 reloads of the 21 GB export.

Before the full sweep, run a two-epoch, one-configuration-per-model smoke. It
must create six fold metrics (three MLP and three LSTM) and zero failures:

```bash
export SAFE_CV_SMOKE_ROOT=/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_all5_seen_innercv_smoke_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 python -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
  --export-dir "$SAFE_OFFICIAL" --safe-repo "$SAFE_REPO" \
  --output-root "$SAFE_CV_SMOKE_ROOT" --model indep \
  --horizon-selectors 1.0 --diffusion-selectors 0.0 \
  --learning-rates 3e-4 --regularization 1e-3 \
  --num-folds 3 --epochs 2 --device cuda --resume &

CUDA_VISIBLE_DEVICES=1 python -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
  --export-dir "$SAFE_OFFICIAL" --safe-repo "$SAFE_REPO" \
  --output-root "$SAFE_CV_SMOKE_ROOT" --model lstm \
  --horizon-selectors 1.0 --diffusion-selectors concat-2 \
  --learning-rates 1e-3 --regularization 1e-2 \
  --num-folds 3 --epochs 2 --device cuda --resume &

wait
test "$(find "$SAFE_CV_SMOKE_ROOT" -mindepth 2 -name metrics.json | wc -l)" -eq 6
test "$(find "$SAFE_CV_SMOKE_ROOT" -name failure.json | wc -l)" -eq 0
```

```bash
export SAFE_CV_TAG=safe_all5_seen_innercv_$(date +%Y%m%d_%H%M%S)
export SAFE_CV_ROOT=/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/$SAFE_CV_TAG
export SAFE_CV_LOG_ROOT=/gs/bs/tga-shinoda/felid/robocasa_logs/eval
mkdir -p "$SAFE_CV_ROOT" "$SAFE_CV_LOG_ROOT"

CUDA_VISIBLE_DEVICES=0 nohup python -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$SAFE_CV_ROOT" \
  --model indep \
  --train-per-class 7 \
  --split-seed 0 \
  --inner-seed 0 \
  --num-folds 3 \
  --epochs 1000 \
  --device cuda \
  --resume \
  > "$SAFE_CV_LOG_ROOT/${SAFE_CV_TAG}_indep.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u \
  "$ROBOCASA_REPO/robocasa/recovery/safe/run_seen_cv_grid.py" \
  --export-dir "$SAFE_OFFICIAL" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$SAFE_CV_ROOT" \
  --model lstm \
  --train-per-class 7 \
  --split-seed 0 \
  --inner-seed 0 \
  --num-folds 3 \
  --epochs 1000 \
  --device cuda \
  --resume \
  > "$SAFE_CV_LOG_ROOT/${SAFE_CV_TAG}_lstm.log" 2>&1 &
```

After 810 successful fold metrics are present, select without printing all
configurations:

```bash
python "$ROBOCASA_REPO/robocasa/recovery/safe/summarize_seen_cv.py" \
  --root "$SAFE_CV_ROOT" \
  --expected-folds 0 1 2 \
  --quiet
```

Then refit three seeds per architecture on all 70 outer-training rollouts by
passing `--selection-summary "$SAFE_CV_ROOT/cv_selection_summary.json"` to
`train_seen_tasks.py`. Evaluate the fixed 30-rollout test set once through
those final refits. The earlier all-seen result is exploratory because its
matched cutoff was derived before this stricter training-only cutoff rule; the
inner-CV refit supersedes it.

## Oracle-guided semantic Subtask-SAFE recording

The first Subtask-SAFE stage uses RoboCasa's curated ordered semantic-subtask
mapping as the oracle current-subtask source. A subtask is a natural-language
unit with a stable ID and one or more associated runtime predicates:

```text
(subtask_id, natural-language instruction, predicate_names)
```

The source definitions are `_TASK_SUBTASK_GROUP_OVERRIDES` and
`_COMPOSITE_ATOMIC_TASK_OVERRIDES` in
`robocasa/recovery/eval_composite_predicates.py`. They are converted by
`mapped_subtask_sequence()` in
`robocasa/recovery/create_recovery_failure_dataset.py` into a canonical
observation-safe sequence. Raw predicates are timestamping evidence, not
Subtask-SAFE training units. This stage does not yet predict subtask identity
or continuous progress. Add this flag to a normal SAFE collection:

```bash
--record-subtask-trace
```

For each retained rollout, the collector evaluates the existing predicate
tracker at reset and after every environment action, then maps its observations
onto the ordered natural-language subtask sequence. It writes a separate
schema-v3 artifact under:

```text
subtasks/<task_name>/<rollout_id>.json
```

The original SAFE feature tensor and official-loader compatibility are
unchanged. The Subtask-SAFE artifact contains:

- every ordered semantic subtask's ID, natural-language instruction, predicate
  set, `source_subtask_ids` provenance, and whether its predicates are required
  by the official task-success condition;
- the environment step where every semantic subtask first becomes complete;
- the oracle current semantic subtask aligned to every genuine policy
  inference;
- one segment for every observed-active semantic subtask;
- a binary label with semantics
  `active_subtask_eventually_fails_before_completion`.

A completed segment has label `0`. Only one segment of an unsuccessful rollout
has label `1`: the active terminal subtask, or, when every ordered subtask was
observed complete but official task success was never reached, the earliest
completed subtask whose predicates are false at the terminal state. This
second case records that a transient completion regressed before overall task
completion; `terminal_failure_reason` and
`terminal_unsatisfied_predicate_names` preserve that distinction. Later
subtasks genuinely completed in the meantime remain successful segments.
Never-entered future subtasks are not samples. A subtask already true at reset,
or co-completed with another semantic subtask without an observed active state,
is recorded under
`excluded_completed_subtasks` and is not a training sample. Ordered first
completion remains monotonic for trace alignment. Terminal regression does not
rewrite inference ownership or create a duplicate segment; it changes the
original segment's outcome to failure because its completion did not persist.

Transient grasp predicates are useful timestamps but are not official success
requirements. A policy can occasionally complete the following placement by
pushing an object, or a brief grasp can fall between observed simulator states.
When an optional transient predicate was never observed but the immediately
following semantic subtask completes, the transient unit is recorded under
`excluded_bypassed_subtasks`. Its ambiguous inference interval is not labeled
as either success or failure. Required semantic subtasks are never bypassed.

Before recording, the canonicalizer reviews all 32 registered composite tasks
with these observability rules:

- deterministic setup conditions are context, not attempted subtasks;
- a separate pick unit exists only when a distinct `*_grasped` predicate can
  timestamp it;
- a separate release unit exists only when `gripper_released` or another
  release predicate can timestamp it;
- adjacent units with the same predicate signature are merged, while retaining
  all original IDs in `source_subtask_ids`;
- a pick/place atomic step without a grasp predicate becomes one natural
  pick-and-place unit rather than three indistinguishable samples.

The deterministic setup exclusions are `cabinets_open` for
`GatherTableware`, `dishwasher_rack_accessible` for `LoadDishwasher`, freezer
and preloaded-container context for `SeparateFreezerRack`, the setup-opened
cabinet for `SearingMeat`, the setup-opened drawer for
`SetUpCuttingStation`, and `cabinet_open` for `StackBowlsCabinet`.

For example, the effective `LoadDishwasher` sequence is now:

```text
Pick the cup from the counter.
Place the cup on the dishwasher rack.
Pick the bowl from the counter.
Place the bowl on the dishwasher rack.
Close the dishwasher.
```

Subtask-SAFE schema v1 represented required predicates as if each predicate
were a subtask. Schema v2 added natural-language groups but still retained
unobservable setup and duplicate-predicate units. The schema-v3 validator
intentionally rejects both older formats; rerun collection in a fresh output
directory after pulling this change.

The validator reports the number of recorded rollouts and usable successful
and failed segments:

```bash
python -m robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$SUBTASK_SAFE_SMOKE"
```

The first live smoke should use a registered composite task and a new output
directory. A trace is considered usable only when the current subtask is
available at every environment state and at least one real policy inference is
associated with the segment. Cached action consumption never creates duplicate
SAFE features.

### Composite-first semantic coverage audit

Subtask outcomes should be collected at the natural full-rollout rate. Do not
stop collection when a subtask class reaches a quota and do not duplicate rare
segments. Split by rollout before extracting segments, then compute any
inverse-frequency loss weights from the training split only.

Before scaling collection, audit the semantic subtask granularity:

```bash
python -m robocasa.recovery.safe.audit_subtask_safe_dataset \
  --dataset-dir "$SUBTASK_SAFE_DATASET" \
  --task-type composite \
  --target-successes 30 \
  --target-failures 20 \
  --json-output "$SUBTASK_SAFE_DATASET/subtask_coverage.json" \
  --csv-output "$SUBTASK_SAFE_DATASET/subtask_coverage.csv"
```

Pass multiple shard directories after `--dataset-dir` to audit them together.
Use `--allow-partial` only while a collector is still running; every other
validator error remains fatal. The audit reports each `(task, semantic
subtask)` pair separately, including its natural-language instruction,
predicate set, usable successes, usable failures, labeled segments with no
policy inference, completed subtasks excluded because no active state was
observed, rollouts where the subtask was not reached, and the remaining target
deficits. Its collection priority ranks the summed failure deficit first
because one failed rollout contributes at most one terminal failed subtask.
These deficits are segment counts, not exact additional-rollout requirements.

The first discovery batch should cover the five composite pilot tasks with 10
natural rollouts per task:

- `LoadDishwasher`
- `PreSoakPan`
- `ScrubCuttingBoard`
- `StackBowlsCabinet`
- `WashLettuce`

Inspect the semantic audit before changing mappings or scaling toward 50
natural rollouts per task. A practical initial target is at least 30 usable
successful and 20 usable failed segments for every retained `(task, subtask)`
pair. Keep atomic controls in a separate dataset and model; `CoffeeSetupMug`,
`PickPlaceCounterToStove`, and `PickPlaceDrawerToCounter` are useful controls
at 20--30 natural rollouts each. Atomic and composite models should not be
mixed until their separate behavior is understood.

### Leakage-safe segment export and official SAFE training

`export_subtask_safe` materializes only segments with a binary label and at
least one genuine policy inference. Every segment becomes one pseudo-rollout
for the pinned official SAFE loader, with the natural-language subtask as its
`task_description`. The export excludes never-entered, already-completed,
bypassed-optional, unlabeled, and labeled-without-inference intervals.

The exporter creates the outer split before materializing segments. Parent
rollouts are stratified by parent task and official rollout outcome; every
segment from a parent remains in the same split. The resulting
`parent_rollout_split.json` contains both segment IDs and parent-rollout IDs,
and the training code verifies both sets before accepting it.

```bash
export SUBTASK_SAFE_EXPORT="$STORAGE_BS/robocasa_rollouts/safe/subtask_safe_export_$(date +%Y%m%d_%H%M%S)"

python -u -m robocasa.recovery.safe.export_subtask_safe \
  --dataset-dir "$SUBTASK_SAFE_DATASET" \
  --output-dir "$SUBTASK_SAFE_EXPORT" \
  --train-fraction 0.7 \
  --split-seed 0

python -u -m robocasa.recovery.safe.validate_official_export \
  --export-dir "$SUBTASK_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --expected-rollouts 426 \
  --expected-successes 341 \
  --expected-failures 85
```

The exact expected counts above describe the 150-rollout composite pilot and
should be changed for another source dataset. Inspect the immutable split:

```bash
python - "$SUBTASK_SAFE_EXPORT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
split = json.loads((root / "parent_rollout_split.json").read_text())
print("protocol:", split["protocol"])
print("train:", split["counts"]["train"])
print("test:", split["counts"]["test"])
print("parent leakage:", bool(set(split["parent_train"]) & set(split["parent_test"])))
print("segment leakage:", bool(set(split["train"]) & set(split["test"])))
PY
```

For hyperparameter selection, pass the generated file as an exact selection
manifest. `run_seen_cv_grid` detects its `split_unit: parent_rollout`, creates
inner folds from parent rollouts rather than segments, and computes official
inverse-frequency class weights from each inner training fold only:

```bash
python -u -m robocasa.recovery.safe.run_seen_cv_grid \
  --export-dir "$SUBTASK_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --output-root "$SUBTASK_SAFE_CV_ROOT" \
  --model indep \
  --task-type composite \
  --selection-manifest "$SUBTASK_SAFE_EXPORT/parent_rollout_split.json" \
  --class-weighting official_inverse_frequency \
  --split-seed 0 \
  --inner-seed 0 \
  --num-folds 3 \
  --epochs 1000 \
  --device cuda \
  --resume
```

Run `--model lstm` on a second GPU while writing to the same grid root. After
both architectures complete, summarize the grid and refit each selected model
on the full outer training segment set:

```bash
python -m robocasa.recovery.safe.summarize_seen_cv \
  --root "$SUBTASK_SAFE_CV_ROOT" \
  --expected-folds 0 1 2 \
  --quiet

python -u -m robocasa.recovery.safe.train_seen_tasks \
  --export-dir "$SUBTASK_SAFE_EXPORT" \
  --safe-repo "$SAFE_REPO" \
  --output-dir "$SUBTASK_SAFE_FINAL_ROOT/indep_seed0" \
  --model indep \
  --seed 0 \
  --split-seed 0 \
  --task-type composite \
  --selection-manifest "$SUBTASK_SAFE_EXPORT/parent_rollout_split.json" \
  --selection-summary "$SUBTASK_SAFE_CV_ROOT/cv_selection_summary.json" \
  --class-weighting official_inverse_frequency \
  --epochs 1000 \
  --device cuda \
  --resume
```

Repeat the final refit for seeds 1 and 2 and for `lstm`. Score files retain the
parent rollout, parent task, subtask ID, instruction, original environment-step
alignment, and semantic segment boundaries, so later overlay videos can place
the Subtask-SAFE score on the correct portion of the original rollout.

### Parent-aware Subtask-SAFE evaluation and calibration

Semantic segments from one environment rollout are correlated and must never
be treated as independent calibration/evaluation units. Run the dedicated
analysis after all six final refits. It reports natural single-class subtask
support without inventing an AUC, resamples complete parent rollouts for 95%
bootstrap intervals, and compares SAFE against a causal elapsed-time hazard
fitted exclusively from training segment survival:

```bash
export SUBTASK_ANALYSIS_ROOT="$SUBTASK_SAFE_FINAL_ROOT/parent_causal_analysis"

python -u -m robocasa.recovery.safe.analyze_subtask_safe_results \
  --final-root "$SUBTASK_SAFE_FINAL_ROOT" \
  --output-dir "$SUBTASK_ANALYSIS_ROOT" \
  --models indep lstm \
  --seeds 0 1 2 \
  --prefix-horizons 1 2 4 8 16 32 64 128 \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 0 \
  --formats png pdf
```

At a prefix horizon, only segments that are still active after that many
genuine policy inferences are eligible. The elapsed baseline is therefore
causal: at inference `k` it uses only the semantic task identity, the fact that
the subtask remains active, and training data. It never uses the held-out final
segment length. The analysis writes exact per-seed, per-subtask, and prefix CSV
files plus support, prefix, and parent-bootstrap figures.

Functional conformal calibration also assigns complete held-out parents. All
successful segments from selected calibration parents may contribute to the
threshold; their failure segments are consumed but excluded. Every segment
from every remaining parent is evaluation-only. Reference and nonconformity
successes are themselves parent-disjoint:

```bash
export SUBTASK_CALIBRATION_ROOT="$SUBTASK_SAFE_FINAL_ROOT/parent_calibration"

python -u -m robocasa.recovery.safe.calibrate_seen_tasks \
  --final-root "$SUBTASK_SAFE_FINAL_ROOT" \
  --output-dir "$SUBTASK_CALIBRATION_ROOT" \
  --model indep \
  --seeds 0 1 2 \
  --task-type composite \
  --split-unit parent_rollout \
  --calibration-parent-fraction 0.4 \
  --split-seed 0 \
  --conformal-seed 0 \
  --reference-fraction 0.3 \
  --alphas 0.05 0.10 0.15 0.20 \
  --selected-alpha 0.15 \
  --modulation tfunc
```

Each alpha directory contains both the Subtask-SAFE calibration and a separately
calibrated training-only elapsed-hazard baseline. `detection_events.csv`
records causal detection environment steps and lead time before each failed
semantic segment ends. The selected alpha remains predeclared; the other alpha
values are sensitivity analyses rather than post-hoc operating-point choices.

Render one annotated video per evaluation parent using normalized scores and
the calibration for the same model seed:

```bash
python -u -m robocasa.recovery.safe.render_score_videos \
  --scores "$SUBTASK_CALIBRATION_ROOT/indep_seed0/normalized_scores.jsonl" \
  --calibration "$SUBTASK_CALIBRATION_ROOT/indep_seed0/alpha_0p15/calibration.json" \
  --output-dir "$SUBTASK_CALIBRATION_ROOT/indep_seed0/alpha_0p15/parent_videos" \
  --split evaluation \
  --group-by-parent \
  --max-videos 10
```

The overlay identifies the parent task and outcome, active natural-language
subtask, segment outcome, elapsed environment steps, causal SAFE risk, growing
score trace, conformal threshold, and alert state. SAFE output is called a risk
score, not a calibrated probability. The semantic subtask boundary remains an
oracle diagnostic input; online subtask recognition is a separate future
module.

## Local structural validation

These checks require no GPU, RoboSuite, simulator, checkpoint, or server:

```bash
python -m py_compile \
  robocasa/recovery/openpi_websocket_policy.py \
  robocasa/recovery/safe/*.py \
  tests/test_safe_*.py

python -m unittest -v \
  tests.test_safe_openpi_policy \
  tests.test_safe_dataset \
  tests.test_safe_collect_rollouts \
  tests.test_safe_subtask_safe \
  tests.test_safe_atomic_collection \
  tests.test_safe_official_export \
  tests.test_safe_official_training \
  tests.test_safe_cli
```

## Future live dependencies and limitations

A live smoke test still requires a RoboCasa environment with RoboSuite and assets, the π0 RoboCasa checkpoint, a checkout of the pinned OpenPI revision with the companion patch applied, and a reachable WebSocket server. The normal unit tests replace the simulator and policy with deterministic mocks and do not claim real SAFE performance.

The current policy transports send raw latent features with each genuine
inference. That is reliable and gives RoboCasa a direct outcome association,
but it increases response size. The Subtask-SAFE extension records registered
atomic or composite ordered natural-language subtask transitions, while raw
predicates remain diagnostic evidence. It intentionally does not invent a
frame-level failure onset, predict subtask identity or continuous progress, or
trigger recovery. The dedicated segment exporter trains the original official
SAFE MLP or LSTM on oracle-delimited subtask intervals; online subtask
recognition and recovery remain future stages.
