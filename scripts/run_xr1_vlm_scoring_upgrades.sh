#!/usr/bin/env bash
# Fast (<30s, ZERO VLM re-inference) scoring-upgrade ablation on the cached
# 120-rollout Qwen3-VL-4B-Instruct logits and perception features bundle.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_xr1_vlm_scoring_upgrades.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

STORAGE="${STORAGE_BS:-/gs/bs/tga-shinoda/felid}"
PY="${PERCEPTION_PY:-$STORAGE/envs/xiaomi_robotics_1_robocasa365/bin/python}"
BUNDLE_DIR="${PERCEPTION_BUNDLE_DIR:-$STORAGE/robocasa_rollouts/safe/xr1_perception_floor_20261128_191229/features}"
VLM_SCORES_DIR="${VLM_SCORES_DIR:-$STORAGE/robocasa_rollouts/safe/xr1_vlm_dest_full_20261128_193617/vlm_scores}"

export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PY" || { echo "Missing Python env: $PY" >&2; exit 1; }
test -f "$BUNDLE_DIR/perception_features.npz" || { echo "Missing bundle: $BUNDLE_DIR" >&2; exit 1; }
test -f "$VLM_SCORES_DIR/vlm_destination_scores.npz" || { echo "Missing VLM scores: $VLM_SCORES_DIR" >&2; exit 1; }

TAG="xr1_vlm_scoring_upgrades_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${UPGRADES_OUT_ROOT:-$STORAGE/robocasa_rollouts/safe/$TAG}"
LOGDIR="$STORAGE/robocasa_logs/eval"
mkdir -p "$OUT_ROOT" "$LOGDIR"
LOG="$LOGDIR/${TAG}.log"
exec > >(tee "$LOG") 2>&1

echo "=== XR-1 Zero-Re-Inference VLM Scoring Upgrades ==="
echo "Repo:       $REPO ($(git rev-parse --short HEAD))"
echo "Bundle:     $BUNDLE_DIR"
echo "VLM scores: $VLM_SCORES_DIR"
echo "Output:     $OUT_ROOT"

echo "=== Preflight unit tests ==="
"$PY" -m unittest discover -s "$REPO/tests" -p "test_long_horizon_vlm_scoring_upgrades.py" -v

echo -e "\n=== Evaluating with grasp_stream=xr1_dit ==="
"$PY" -u -m scripts.xr1_lead_time.eval_vlm_scoring_upgrades \
  --bundle "$BUNDLE_DIR" \
  --vlm-scores "$VLM_SCORES_DIR" \
  --grasp-stream xr1_dit \
  --grasp-hidden-dim 128 \
  --grasp-epochs 100 \
  --output "$OUT_ROOT/eval_dit"

echo -e "\n=== Evaluating with grasp_stream=proprio_actions ==="
"$PY" -u -m scripts.xr1_lead_time.eval_vlm_scoring_upgrades \
  --bundle "$BUNDLE_DIR" \
  --vlm-scores "$VLM_SCORES_DIR" \
  --grasp-stream proprio_actions \
  --grasp-hidden-dim 128 \
  --grasp-epochs 100 \
  --output "$OUT_ROOT/eval_proprio"

tar -C "$OUT_ROOT" -czf "$OUT_ROOT/vlm_scoring_upgrades.tar.gz" \
  eval_dit/vlm_scoring_upgrades_summary.json \
  eval_proprio/vlm_scoring_upgrades_summary.json
ls -lh "$OUT_ROOT/vlm_scoring_upgrades.tar.gz"
echo "Results bundle: $OUT_ROOT/vlm_scoring_upgrades.tar.gz"
echo "PASS: scoring upgrades complete"
