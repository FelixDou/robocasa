#!/usr/bin/env bash
# 3-fold cross-task LoRA adaptation of Qwen3-VL-4B-Instruct:
#   - Train rank-16 LoRA on 4 tasks, evaluate STRICTLY ZERO-SHOT on the 2 held-out
#     unseen tasks across 3 disjoint folds (all 6 tasks / 120 rollouts).
#   - Feed the 100% unseen-task VLM scores into both eval_vlm_hybrid_monitor.py
#     and eval_vlm_scoring_upgrades.py (apples-to-apples with Section 9).
#
# Usage (1 GPU, ~12-16 min total):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_xr1_vlm_lora_cross_task.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

STORAGE="${STORAGE_BS:-/gs/bs/tga-shinoda/felid}"
PY="${PERCEPTION_PY:-$STORAGE/envs/xiaomi_robotics_1_robocasa365/bin/python}"
RUN_DIR="${PILOT6_RUN_DIR:-$STORAGE/robocasa_rollouts/safe/xr1_recovery_pilot6_full_20261128_142543}"
BUNDLE_DIR="${PERCEPTION_BUNDLE_DIR:-$STORAGE/robocasa_rollouts/safe/xr1_perception_floor_20261128_191229/features}"
QWEN_DIR="${QWEN3_VL_DIR:-$STORAGE/robocasa_checkpoints/lingbot_vla_v2/downloads/Qwen3-VL-4B-Instruct}"

export HF_HOME="$STORAGE/hf_home"
export TRANSFORMERS_CACHE="$STORAGE/hf_home/transformers"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PY" || { echo "Missing Python env: $PY" >&2; exit 1; }
test -d "$RUN_DIR" || { echo "Missing pilot run dir: $RUN_DIR" >&2; exit 1; }
test -f "$BUNDLE_DIR/perception_features.npz" || { echo "Missing perception bundle: $BUNDLE_DIR" >&2; exit 1; }
test -d "$QWEN_DIR" || { echo "Missing local Qwen3-VL-4B-Instruct dir: $QWEN_DIR" >&2; exit 1; }

TAG="xr1_vlm_lora_crosstask_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${LORA_OUT_ROOT:-$STORAGE/robocasa_rollouts/safe/$TAG}"
LOGDIR="$STORAGE/robocasa_logs/eval"
mkdir -p "$OUT_ROOT" "$LOGDIR"
LOG="$LOGDIR/${TAG}.log"
exec > >(tee "$LOG") 2>&1

echo "=== XR-1 3-Fold Cross-Task Qwen3-VL-4B LoRA (Strict Unseen-Task Zero-Shot) ==="
echo "Repo:       $REPO ($(git rev-parse --short HEAD))"
echo "Run input:  $RUN_DIR"
echo "Bundle:     $BUNDLE_DIR"
echo "Model:      $QWEN_DIR"
echo "Output:     $OUT_ROOT"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-<auto>}"

echo "=== Preflight unit tests ==="
"$PY" -m unittest discover -s "$REPO/tests" -p "test_long_horizon_vlm_lora.py" -v

echo -e "\n=== Step 1: 3-Fold Cross-Task LoRA Training & Unseen-Task Zero-Shot Scoring ==="
"$PY" -u -m scripts.xr1_lead_time.train_eval_vlm_lora_cross_task \
  --run "$RUN_DIR" \
  --model-dir "$QWEN_DIR" \
  --decision-stride "${VLM_DECISION_STRIDE:-2}" \
  --eval-batch-size "${VLM_BATCH_SIZE:-16}" \
  --rank "${LORA_RANK:-16}" \
  --alpha "${LORA_ALPHA:-32}" \
  --epochs "${LORA_EPOCHS:-2}" \
  --micro-batch-size "${LORA_MICRO_BS:-4}" \
  --grad-accum "${LORA_GRAD_ACCUM:-4}" \
  --lr "${LORA_LR:-1e-4}" \
  --weight-decay "${LORA_WD:-0.05}" \
  --output "$OUT_ROOT"

VLM_SCORES_DIR="$OUT_ROOT/vlm_scores"

echo -e "\n=== Step 2: Hybrid monitor evaluation (xr1_dit grasp + LoRA unseen-task VLM) ==="
"$PY" -u -m scripts.xr1_lead_time.eval_vlm_hybrid_monitor \
  --bundle "$BUNDLE_DIR" \
  --vlm-scores "$VLM_SCORES_DIR" \
  --grasp-stream xr1_dit \
  --grasp-hidden-dim 128 \
  --grasp-epochs 100 \
  --output "$OUT_ROOT/eval_hybrid_dit"

echo -e "\n=== Step 3: Hybrid monitor evaluation (proprio_actions grasp + LoRA unseen-task VLM) ==="
"$PY" -u -m scripts.xr1_lead_time.eval_vlm_hybrid_monitor \
  --bundle "$BUNDLE_DIR" \
  --vlm-scores "$VLM_SCORES_DIR" \
  --grasp-stream proprio_actions \
  --grasp-hidden-dim 128 \
  --grasp-epochs 100 \
  --output "$OUT_ROOT/eval_hybrid_proprio"

echo -e "\n=== Step 4: Full scoring upgrade & post-release diagnostics (proprio_actions) ==="
"$PY" -u -m scripts.xr1_lead_time.eval_vlm_scoring_upgrades \
  --bundle "$BUNDLE_DIR" \
  --vlm-scores "$VLM_SCORES_DIR" \
  --grasp-stream proprio_actions \
  --grasp-hidden-dim 128 \
  --grasp-epochs 100 \
  --output "$OUT_ROOT/eval_upgrades_proprio"

echo -e "\n=== Step 5: Full scoring upgrade & post-release diagnostics (xr1_dit) ==="
"$PY" -u -m scripts.xr1_lead_time.eval_vlm_scoring_upgrades \
  --bundle "$BUNDLE_DIR" \
  --vlm-scores "$VLM_SCORES_DIR" \
  --grasp-stream xr1_dit \
  --grasp-hidden-dim 128 \
  --grasp-epochs 100 \
  --output "$OUT_ROOT/eval_upgrades_dit"

tar -C "$OUT_ROOT" -czf "$OUT_ROOT/vlm_lora_crosstask_results.tar.gz" \
  vlm_scores/vlm_destination_index.json \
  eval_hybrid_dit/vlm_hybrid_summary.json \
  eval_hybrid_proprio/vlm_hybrid_summary.json \
  eval_upgrades_proprio/vlm_scoring_upgrades_summary.json \
  eval_upgrades_dit/vlm_scoring_upgrades_summary.json
ls -lh "$OUT_ROOT/vlm_lora_crosstask_results.tar.gz"
echo "Results bundle: $OUT_ROOT/vlm_lora_crosstask_results.tar.gz"
echo "PASS: 3-fold cross-task Qwen3-VL LoRA evaluation complete"
