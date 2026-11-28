#!/usr/bin/env bash
# Run the camera perception floor check on the 120-rollout synchronized XR-1 pilot:
#   1. Extract strictly causal frozen SigLIP-So400M/14-224 (3-camera) + XR-1 DiT
#      + proprioception + executed-action features (1 GPU, ~1-2 min).
#   2. Run 5-fold cross-fitted relation probing, LOTO universal primitive transfer,
#      component substitution, and fully-observed camera-monitor lead evaluation
#      for both linear (hidden_dim=0) and 1-layer MLP (hidden_dim=128) probes.
#
# Usage (from the RoboCasa checkout on a 1-GPU node):
#   bash scripts/run_xr1_perception_floor.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

STORAGE="${STORAGE_BS:-/gs/bs/tga-shinoda/felid}"
PY="${PERCEPTION_PY:-$STORAGE/envs/xiaomi_robotics_1_robocasa365/bin/python}"
RUN_DIR="${PILOT6_RUN_DIR:-$STORAGE/robocasa_rollouts/safe/xr1_recovery_pilot6_full_20261128_142543}"
VISION_SNAPSHOT="${VISION_SNAPSHOT:-$STORAGE/hf_home/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c}"
VISION_WEIGHTS="${VISION_WEIGHTS:-$STORAGE/hf_home/hub/models--lerobot--pi0_libero_finetuned_v044/snapshots/45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f/model.safetensors}"

export HF_HOME="$STORAGE/hf_home"
export TRANSFORMERS_CACHE="$STORAGE/hf_home/transformers"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PY" || { echo "Missing Python env: $PY" >&2; exit 1; }
test -d "$RUN_DIR" || { echo "Missing pilot run dir: $RUN_DIR" >&2; exit 1; }
test -d "$VISION_SNAPSHOT" || { echo "Missing cached PaliGemma config dir: $VISION_SNAPSHOT" >&2; exit 1; }
test -f "$VISION_WEIGHTS" || { echo "Missing cached SigLIP safetensors: $VISION_WEIGHTS" >&2; exit 1; }

TAG="xr1_perception_floor_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${PERCEPTION_OUT_ROOT:-$STORAGE/robocasa_rollouts/safe/$TAG}"
LOGDIR="$STORAGE/robocasa_logs/eval"
mkdir -p "$OUT_ROOT" "$LOGDIR"
LOG="$LOGDIR/${TAG}.log"
exec > >(tee "$LOG") 2>&1

echo "=== XR-1 Perception Floor Check ==="
echo "Repo:       $REPO ($(git rev-parse --short HEAD))"
echo "Run input:  $RUN_DIR"
echo "Vision:     $VISION_SNAPSHOT"
echo "Output:     $OUT_ROOT"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-<auto>}"

echo "=== Preflight unit tests (PyTorch + OpenCV) ==="
"$PY" -m unittest discover -s "$REPO/tests" -p "test_long_horizon_perception_floor.py" -v

BUNDLE_DIR="$OUT_ROOT/features"
if [[ ! -f "$BUNDLE_DIR/perception_features.npz" ]]; then
  "$PY" -u -m scripts.xr1_lead_time.extract_perception_features \
    --run "$RUN_DIR" \
    --vision-snapshot "$VISION_SNAPSHOT" \
    --vision-weights "$VISION_WEIGHTS" \
    --batch-size 128 \
    --output "$BUNDLE_DIR"
else
  echo "Reusing existing features bundle: $BUNDLE_DIR"
fi

echo -e "\n=== Evaluating linear probes (hidden_dim=0) ==="
"$PY" -u -m scripts.xr1_lead_time.eval_perception_floor \
  --bundle "$BUNDLE_DIR" \
  --hidden-dim 0 \
  --epochs 100 \
  --output "$OUT_ROOT/eval_linear"

echo -e "\n=== Evaluating 1-hidden-layer MLP probes (hidden_dim=128) ==="
"$PY" -u -m scripts.xr1_lead_time.eval_perception_floor \
  --bundle "$BUNDLE_DIR" \
  --hidden-dim 128 \
  --epochs 100 \
  --output "$OUT_ROOT/eval_mlp128"

tar -C "$OUT_ROOT" -czf "$OUT_ROOT/perception_floor_results.tar.gz" \
  eval_linear/perception_floor_summary.json \
  eval_mlp128/perception_floor_summary.json
ls -lh "$OUT_ROOT/perception_floor_results.tar.gz"
echo "Results bundle: $OUT_ROOT/perception_floor_results.tar.gz"
echo "PASS: perception floor check complete"
