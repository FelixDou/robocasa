#!/usr/bin/env bash
# Fast (<15s, ZERO new rollouts, ZERO VLM re-inference) Test-2 continuation-aware
# pilot check on the 120-rollout synchronized XR-1 dataset.
#
# Usage:
#   bash scripts/run_xr1_test2_continuation_pilot.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

STORAGE="${STORAGE_BS:-/gs/bs/tga-shinoda/felid}"
PY="${PERCEPTION_PY:-$STORAGE/envs/xiaomi_robotics_1_robocasa365/bin/python}"
BUNDLE_DIR="${PERCEPTION_BUNDLE_DIR:-$STORAGE/robocasa_rollouts/safe/xr1_perception_floor_20261128_191229/features}"
ZS_VLM_DIR="${ZS_VLM_DIR:-$STORAGE/robocasa_rollouts/safe/xr1_vlm_dest_full_20261128_193617/vlm_scores}"
LORA_VLM_DIR="${LORA_VLM_DIR:-$STORAGE/robocasa_rollouts/safe/xr1_vlm_lora_crosstask_20261128_202023/vlm_scores}"

export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PY" || { echo "Missing Python env: $PY" >&2; exit 1; }
test -f "$BUNDLE_DIR/perception_features.npz" || { echo "Missing bundle: $BUNDLE_DIR" >&2; exit 1; }
test -f "$ZS_VLM_DIR/vlm_destination_scores.npz" || { echo "Missing ZS VLM scores: $ZS_VLM_DIR" >&2; exit 1; }
test -f "$LORA_VLM_DIR/vlm_destination_scores.npz" || { echo "Missing LoRA VLM scores: $LORA_VLM_DIR" >&2; exit 1; }

TAG="xr1_test2_continuation_pilot_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$STORAGE/robocasa_rollouts/safe/$TAG"
LOGDIR="$STORAGE/robocasa_logs/eval"
mkdir -p "$OUT_ROOT" "$LOGDIR"
LOG="$LOGDIR/${TAG}.log"
exec > >(tee "$LOG") 2>&1

echo "=== XR-1 Test-2 Continuation-Aware Pilot Check ==="
echo "Repo:   $REPO ($(git rev-parse --short HEAD))"
echo "Output: $OUT_ROOT"

"$PY" -m unittest discover -s "$REPO/tests" -p "test_long_horizon_test2_continuation.py" -v

echo -e "\n=== With Base Zero-Shot Qwen3-VL trajectories ==="
"$PY" -u -m scripts.xr1_lead_time.eval_test2_continuation_pilot \
  --bundle "$BUNDLE_DIR" \
  --vlm-scores "$ZS_VLM_DIR" \
  --output "$OUT_ROOT/with_zs_vlm"

echo -e "\n=== With 3-Fold Cross-Task LoRA Unseen-Task VLM trajectories ==="
"$PY" -u -m scripts.xr1_lead_time.eval_test2_continuation_pilot \
  --bundle "$BUNDLE_DIR" \
  --vlm-scores "$LORA_VLM_DIR" \
  --output "$OUT_ROOT/with_lora_vlm"

tar -C "$OUT_ROOT" -czf "$OUT_ROOT/test2_continuation_pilot.tar.gz" \
  with_zs_vlm/test2_continuation_pilot.json \
  with_lora_vlm/test2_continuation_pilot.json
ls -lh "$OUT_ROOT/test2_continuation_pilot.tar.gz"
echo "PASS: Test-2 continuation pilot check complete"
