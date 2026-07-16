# Collecting atomic RoboCasa rollouts for official SAFE π0

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
405 fits and can be restarted with `--resume`:

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
  --output-root "$SAFE_GRID_ROOT"
```

This writes `selection_summary.json` and `selection_summary.csv`. A
configuration is eligible for selection only after all three seeds complete.

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
  tests.test_safe_atomic_collection \
  tests.test_safe_official_export \
  tests.test_safe_official_training \
  tests.test_safe_cli
```

## Future live dependencies and limitations

A live smoke test still requires a RoboCasa environment with RoboSuite and assets, the π0 RoboCasa checkpoint, a checkout of the pinned OpenPI revision with the companion patch applied, and a reachable WebSocket server. The normal unit tests replace the simulator and policy with deterministic mocks and do not claim real SAFE performance.

The current WebSocket transport sends the raw latent back with each inference. That is reliable and gives RoboCasa a direct outcome association, but it increases inference response size. The integration supports JAX π0 only, matching the inspected official feature. It intentionally does not create frame-level labels, convert expert demonstrations, collect composite tasks, or trigger recovery. Training uses the pinned unmodified official SAFE repository; the in-tree trainer remains an isolated structural baseline rather than the source of the official experiment result.
