#!/usr/bin/env bash
# Zero-shot Qwen3-VL-4B-Instruct destination verification + decoupled hybrid
# monitor evaluation (fast XR-1 DiT/proprio grasp trigger + Qwen3-VL spatial
# destination verifier) on the 120-rollout synchronized XR-1 pilot.
#
# Usage (from the RoboCasa checkout on a 1-GPU node):
#   bash scripts/run_xr1_vlm_destination_check.sh
#   VLM_SMOKE=1 bash scripts/run_xr1_vlm_destination_check.sh   # 2 rollouts/task smoke
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

STORAGE="${STORAGE_BS:-/gs/bs/tga-shinoda/felid}"
PY="${PERCEPTION_PY:-$STORAGE/envs/xiaomi_robotics_1_robocasa365/bin/python}"
RUN_DIR="${PILOT6_RUN_DIR:-$STORAGE/robocasa_rollouts/safe/xr1_recovery_pilot6_full_20261128_142543}"
BUNDLE_DIR="${PERCEPTION_BUNDLE_DIR:-$STORAGE/robocasa_rollouts/safe/xr1_perception_floor_20261128_191229/features}"
QWEN_DIR="${QWEN3_VL_DIR:-$STORAGE/robocasa_checkpoints/lingbot_vla_v2/downloads/Qwen3-VL-4B-Instruct}"
STRIDE="${VLM_DECISION_STRIDE:-2}"
BATCH_SIZE="${VLM_BATCH_SIZE:-16}"

export HF_HOME="$STORAGE/hf_home"
export TRANSFORMERS_CACHE="$STORAGE/hf_home/transformers"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PY" || { echo "Missing Python env: $PY" >&2; exit 1; }
test -d "$RUN_DIR" || { echo "Missing pilot run dir: $RUN_DIR" >&2; exit 1; }
test -f "$BUNDLE_DIR/perception_features.npz" || { echo "Missing perception bundle: $BUNDLE_DIR" >&2; exit 1; }
test -d "$QWEN_DIR" || { echo "Missing local Qwen3-VL-4B-Instruct dir: $QWEN_DIR" >&2; exit 1; }

if [[ "${VLM_SMOKE:-0}" == "1" ]]; then
  TAG="xr1_vlm_dest_smoke_$(date +%Y%m%d_%H%M%S)"
  EXTRA_ARGS=(--max-rollouts-per-task 2)
else
  TAG="xr1_vlm_dest_full_$(date +%Y%m%d_%H%M%S)"
  EXTRA_ARGS=()
fi

OUT_ROOT="${VLM_OUT_ROOT:-$STORAGE/robocasa_rollouts/safe/$TAG}"
LOGDIR="$STORAGE/robocasa_logs/eval"
mkdir -p "$OUT_ROOT" "$LOGDIR"
LOG="$LOGDIR/${TAG}.log"
exec > >(tee "$LOG") 2>&1

echo "=== XR-1 Zero-Shot Qwen3-VL Destination Verifier & Hybrid Monitor ==="
echo "Repo:       $REPO ($(git rev-parse --short HEAD))"
echo "Run input:  $RUN_DIR"
echo "Bundle:     $BUNDLE_DIR"
echo "Model:      $QWEN_DIR"
echo "Stride/BS:  stride=$STRIDE batch_size=$BATCH_SIZE smoke=${VLM_SMOKE:-0}"
echo "Output:     $OUT_ROOT"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-<auto>}"

echo "=== Preflight unit tests ==="
"$PY" -m unittest discover -s "$REPO/tests" -p "test_long_horizon_vlm_verifier.py" -v

VLM_SCORES_DIR="$OUT_ROOT/vlm_scores"
echo -e "\n=== Step 1: Zero-shot Qwen3-VL-4B-Instruct scoring ==="
"$PY" -u -m scripts.xr1_lead_time.extract_vlm_destinations \
  --run "$RUN_DIR" \
  --model-dir "$QWEN_DIR" \
  --decision-stride "$STRIDE" \
  --batch-size "$BATCH_SIZE" \
  --output "$VLM_SCORES_DIR" \
  "${EXTRA_ARGS[@]}"

if [[ "${VLM_SMOKE:-0}" != "1" ]]; then
  echo -e "\n=== Step 2: Hybrid monitor evaluation (xr1_dit grasp + Qwen3-VL dest) ==="
  "$PY" -u -m scripts.xr1_lead_time.eval_vlm_hybrid_monitor \
    --bundle "$BUNDLE_DIR" \
    --vlm-scores "$VLM_SCORES_DIR" \
    --grasp-stream xr1_dit \
    --grasp-hidden-dim 128 \
    --grasp-epochs 100 \
    --output "$OUT_ROOT/eval_hybrid_dit"

  echo -e "\n=== Step 3: Hybrid monitor evaluation (proprio_actions grasp + Qwen3-VL dest) ==="
  "$PY" -u -m scripts.xr1_lead_time.eval_vlm_hybrid_monitor \
    --bundle "$BUNDLE_DIR" \
    --vlm-scores "$VLM_SCORES_DIR" \
    --grasp-stream proprio_actions \
    --grasp-hidden-dim 128 \
    --grasp-epochs 100 \
    --output "$OUT_ROOT/eval_hybrid_proprio"

  tar -C "$OUT_ROOT" -czf "$OUT_ROOT/vlm_hybrid_results.tar.gz" \
    vlm_scores/vlm_destination_index.json \
    eval_hybrid_dit/vlm_hybrid_summary.json \
    eval_hybrid_proprio/vlm_hybrid_summary.json
  ls -lh "$OUT_ROOT/vlm_hybrid_results.tar.gz"
  echo "Results bundle: $OUT_ROOT/vlm_hybrid_results.tar.gz"
else
  echo "Smoke mode complete; inspect $VLM_SCORES_DIR/vlm_destination_index.json"
fi

echo "PASS: Qwen3-VL destination verification complete"
