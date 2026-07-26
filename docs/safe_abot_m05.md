# SAFE-style failure detection for ABot-M0.5 on RoboCasa

## Scope and scientific status

This integration applies the existing RoboCasa raw-SAFE collection, storage,
training, calibration, evaluation, and visualization pipeline to
`acvlab/ABot-M0.5-RoboCasa365`. Natural task success is label `0`; natural task
failure is label `1`. It does not use simulator state, subtask annotations,
expert trajectories, or manufactured frame-level failure labels.

This is **SAFE-style ABot-M0.5**, not the official SAFE π0 feature. ABot has a
different action architecture and chunk protocol, so its rollouts carry
`model_family=abot_m05` and cannot be silently mixed with π0 or RLDX-1
rollouts. The existing SAFE MLP and LSTM are reused as detector baselines, but
ABot needs its own training and conformal calibration.

Pinned sources and checkpoint:

- SAFE: `b6036abe07b2b2bb9996afb2c07f13d6a9f507c0`
- ABot-Manipulation: `7642747ed2817b241dde5df06e17ee80192718ad`
- checkpoint: `acvlab/ABot-M0.5-RoboCasa365`

## Feature contract

ABot uses a dual-stream Mixture-of-Transformers. At every effective action
denoising update, the integration captures the normalized action-stream tokens
immediately before:

```python
latent_hidden_states = self.action_proj_out(latent_hidden_states)
```

The layer identifier is:

```text
action_stream_post_norm_pre_action_proj_out
```

One inference returns float32:

```text
(flow_steps, action_horizon, action_stream_width)
```

For the released RoboCasa configuration this is expected to be
`(50, 32, 768)`, but every dimension is read from and checked against the live
server response. The final transformer invocation updates ABot's KV cache but
does not perform a denoising step, so it is deliberately excluded. Collection
stores raw features without reducing the flow or action axes.

Feature capture only detaches the pre-projection tensor. The original tensor
continues unchanged into `action_proj_out`, so the opt-in request does not
alter the action computation.

## Official ABot chunk and seed behavior

`ABotM05WebsocketPolicy` preserves the released client protocol:

- the first prediction has 32 actions but executes only the second 16-action
  frame;
- later predictions execute the full 32-action chunk;
- one observation keyframe is retained every four executed actions;
- completed chunks are sent back in the KV-cache prefill request before the
  next prediction;
- cached actions do not create duplicate SAFE inference records.

Consequently, inference environment steps begin `0, 16, 48, 80, ...`, not
`0, 32, 64, ...`. This first half-chunk is intentional.

`--seed-protocol official_abot --seed 7` creates one environment per task and
calls `reset(seed=7)`, `reset(seed=8)`, and so on, matching the released ABot
evaluator. Each row records both the effective seed and reset index.

## Prepare a separate patched ABot checkout

Keep the official ABot evaluation checkout clean and put the SAFE worktree
under source storage:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export ABOT_REPO="$PROJECT_FS/robocasa_benchmark_repos/ABot-Manipulation"
export ABOT_SAFE_REPO="$PROJECT_FS/robocasa_benchmark_repos/ABot-Manipulation-safe-7642747"
export ABOT_COMMIT=7642747ed2817b241dde5df06e17ee80192718ad
export ABOT_SAFE_PATCH="$ROBOCASA_REPO/patches/abot_m05_safe_features_7642747.patch"

test "$(git -C "$ABOT_REPO" rev-parse HEAD)" = "$ABOT_COMMIT" || exit 1
test -f "$ABOT_SAFE_PATCH" || exit 1

if [ ! -e "$ABOT_SAFE_REPO/.git" ]; then
  git -C "$ABOT_REPO" worktree add --detach "$ABOT_SAFE_REPO" "$ABOT_COMMIT"
fi

if git -C "$ABOT_SAFE_REPO" apply --reverse --check \
  "$ABOT_SAFE_PATCH" 2>/dev/null; then
  echo "ABot SAFE patch is already applied"
else
  git -C "$ABOT_SAFE_REPO" apply --check "$ABOT_SAFE_PATCH" &&
  git -C "$ABOT_SAFE_REPO" apply "$ABOT_SAFE_PATCH"
fi

/gs/bs/tga-shinoda/felid/envs/abot_m05/bin/python -m py_compile \
  "$ABOT_SAFE_REPO/wam/modules/model.py" \
  "$ABOT_SAFE_REPO/wam/pipelines/core/pipeline_output.py" \
  "$ABOT_SAFE_REPO/wam/pipelines/core/pipeline_wam.py" \
  "$ABOT_SAFE_REPO/wam/pipelines/core/server_policy.py"
```

## Start the patched server

Follow `docs/cluster_experiment_runbook.md` for the standard storage and
environment checks. Run the ordinary ABot preflight on its clean checkout
before applying or serving the separate SAFE worktree.

Stage the checkpoint to node-local storage in every new allocation as described
by `robocasa/scripts/abot_m05/evaluate_cluster.sh`, then start one server:

```bash
export ABOT_ENV="$STORAGE_BS/envs/abot_m05"
export ABOT_CHECKPOINT_ROOT="$STORAGE_BS/robocasa_checkpoints/abot_m05/ABot-M0.5-RoboCasa365"
export ABOT_LOCAL_CHECKPOINT_ROOT=/tmp/ut06746/abot_m05/ABot-M0.5-RoboCasa365
export ABOT_PORT=35056
export ABOT_MASTER_PORT=35100
export ABOT_SERVER_TAG=abot_m05_safe_$(date +%Y%m%d_%H%M%S)
export ABOT_SERVER_LOG="$STORAGE_BS/robocasa_logs/eval/${ABOT_SERVER_TAG}.log"
export ABOT_SERVER_OUTPUT="$STORAGE_BS/robocasa_rollouts/safe_server/$ABOT_SERVER_TAG"

mkdir -p \
  "$ABOT_LOCAL_CHECKPOINT_ROOT" \
  "$ABOT_SERVER_OUTPUT" \
  "$(dirname "$ABOT_SERVER_LOG")"

rsync -a --info=progress2 \
  "$ABOT_CHECKPOINT_ROOT/" \
  "$ABOT_LOCAL_CHECKPOINT_ROOT/"

test -f \
  "$ABOT_LOCAL_CHECKPOINT_ROOT/checkpoint_step/transformer/diffusion_pytorch_model.safetensors" \
  || exit 1

cd "$ABOT_SAFE_REPO"
nohup env \
  PYTHONPATH="$ABOT_SAFE_REPO" \
  CUDA_VISIBLE_DEVICES=0 \
  WANDB_MODE=disabled \
  WANDB_DISABLED=true \
  WAN22_PRETRAINED_PATH="$ABOT_LOCAL_CHECKPOINT_ROOT/base_checkpoint" \
  WAN22_PRETRAINED_MODEL_NAME_OR_PATH="$ABOT_LOCAL_CHECKPOINT_ROOT/base_checkpoint" \
  ROBOCASA_POSTTRAIN_MODEL_PATH_TEST="$ABOT_LOCAL_CHECKPOINT_ROOT/checkpoint_step" \
  "$ABOT_ENV/bin/python" -u -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$ABOT_MASTER_PORT" \
    wam/server.py \
    --config-name robocasa_train_test_atomic_target \
    --port "$ABOT_PORT" \
    --save_root "$ABOT_SERVER_OUTPUT" \
    --save_pred_video 0 \
    --attn-mode torch \
  > "$ABOT_SERVER_LOG" 2>&1 &

export ABOT_SERVER_PID=$!
echo "server_pid=$ABOT_SERVER_PID"
echo "server_log=$ABOT_SERVER_LOG"

until ss -ltn | grep -q ":$ABOT_PORT"; do
  kill -0 "$ABOT_SERVER_PID" 2>/dev/null || {
    tail -80 "$ABOT_SERVER_LOG"
    exit 1
  }
  tail -20 "$ABOT_SERVER_LOG" 2>/dev/null || true
  sleep 10
done
```

Use the eager `torch` attention path for the first smoke. FlashAttention can be
tested separately after the raw feature and action contracts pass.

## One-rollout collection smoke

Run the simulator from the normal RoboCasa client environment while exposing
the ABot websocket client package through `PYTHONPATH`:

```bash
export ROBOCASA_ENV="$STORAGE_BS/envs/robocasa_openpi"
export SAFE_SMOKE="$STORAGE_BS/robocasa_rollouts/safe/abot_m05_sink_smoke_$(date +%Y%m%d_%H%M%S)"

cd "$ROBOCASA_REPO"
PYTHONPATH="$ABOT_SAFE_REPO:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
"$ROBOCASA_ENV/bin/python" -u -m \
  robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir "$SAFE_SMOKE" \
  --tasks TurnOnSinkFaucet \
  --num-rollouts 1 \
  --seed 7 \
  --seed-protocol official_abot \
  --policy-module robocasa.recovery.abot_websocket_policy:make_policy \
  --model-family abot_m05 \
  --policy-name acvlab/ABot-M0.5-RoboCasa365 \
  --checkpoint "$ABOT_LOCAL_CHECKPOINT_ROOT/checkpoint_step" \
  --policy-config \
    '{"frame_chunk_size":2,"action_per_frame":16,"action_num_inference_steps":50,"attention":"torch"}' \
  --host 127.0.0.1 \
  --port "$ABOT_PORT" \
  --split pretrain \
  --replan-steps 32 \
  --record-safe-features \
  --record-actions \
  --record-videos \
  --video-frame-stride 2 \
  --max-errors 1

"$ROBOCASA_ENV/bin/python" -m \
  robocasa.recovery.safe.validate_atomic_dataset \
  --dataset-dir "$SAFE_SMOKE"
```

The required smoke verdict is `Atomic SAFE dataset: VALID` and
`Official SAFE loader compatible: True`. Inspect `summary.json`,
`manifest.jsonl`, and the server log before increasing the rollout count. A
success is useful but not required for protocol validation; balanced training
still requires both natural outcome classes.

## Export and train

The ABot dataset uses the same export, official SAFE MLP/LSTM training,
conformal calibration, reporting, and video-overlay tools as π0 and RLDX:

```bash
python -m robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir "$ABOT_DATASET" \
  --output-dir "$ABOT_OFFICIAL_EXPORT" \
  --resume
```

The export format is
`official_safe_abot_m05_env_records_policy_records`. The official SAFE loader
reads `pre_velocity` dynamically, so the ABot action-stream width requires no
new detector module. Do not reuse π0 or RLDX detector weights or conformal
thresholds: train and calibrate on ABot rollouts.

Engineering compatibility does not guarantee π0-level detection quality.
Compare held-out ROC-AUC, PRC-AUC, conformal TPR/FPR, and detection time before
claiming similar behavior.
