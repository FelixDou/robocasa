#!/usr/bin/env bash
# Fast (~20s, ZERO VLM re-inference) evaluation of grasp-anchored inferred stage
# clocks + union-bound channel-separated split-conformal calibration across BOTH
# zero-shot Qwen3-VL-4B and 3-fold cross-task LoRA unseen-task Qwen3-VL-4B.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_xr1_grasp_anchored_monitor.sh
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
test -f "$ZS_VLM_DIR/vlm_destination_scores.npz" || { echo "Missing zero-shot VLM scores: $ZS_VLM_DIR" >&2; exit 1; }
test -f "$LORA_VLM_DIR/vlm_destination_scores.npz" || { echo "Missing LoRA VLM scores: $LORA_VLM_DIR" >&2; exit 1; }

TAG="xr1_grasp_anchored_monitor_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$STORAGE/robocasa_rollouts/safe/$TAG"
LOGDIR="$STORAGE/robocasa_logs/eval"
mkdir -p "$OUT_ROOT" "$LOGDIR"
LOG="$LOGDIR/${TAG}.log"
exec > >(tee "$LOG") 2>&1

echo "=== XR-1 Grasp-Anchored Zero-Privileged Monitor Comparison ==="
echo "Repo:       $REPO ($(git rev-parse --short HEAD))"
echo "Output:     $OUT_ROOT"

"$PY" -m unittest discover -s "$REPO/tests" -p "test_long_horizon_grasp_anchored.py" -v

for spec in \
  "zeroshot_vlm:$ZS_VLM_DIR:proprio_actions" \
  "zeroshot_vlm:$ZS_VLM_DIR:xr1_dit" \
  "lora_unseen_vlm:$LORA_VLM_DIR:proprio_actions" \
  "lora_unseen_vlm:$LORA_VLM_DIR:xr1_dit"; do
  IFS=: read -r lbl vdir gstream <<< "$spec"
  echo -e "\n================================================================"
  "$PY" -u -m scripts.xr1_lead_time.eval_grasp_anchored_monitor \
    --bundle "$BUNDLE_DIR" \
    --vlm-scores "$vdir" \
    --label "$lbl" \
    --grasp-stream "$gstream" \
    --output "$OUT_ROOT/${lbl}_${gstream}"
done

tar -C "$OUT_ROOT" -czf "$OUT_ROOT/grasp_anchored_results.tar.gz" \
  zeroshot_vlm_proprio_actions/grasp_anchored_summary.json \
  zeroshot_vlm_xr1_dit/grasp_anchored_summary.json \
  lora_unseen_vlm_proprio_actions/grasp_anchored_summary.json \
  lora_unseen_vlm_xr1_dit/grasp_anchored_summary.json
ls -lh "$OUT_ROOT/grasp_anchored_results.tar.gz"
echo "PASS: grasp-anchored monitor evaluation complete"
