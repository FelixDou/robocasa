#!/usr/bin/env bash
set -euo pipefail

# Master resumable 18-task (360-rollout) Group-A/B Test-1 + LLM-Task-Graph +
# Powered Equal-Duration Test-2 evaluation pipeline on TSUBAME.
#
# Combines:
#   - PILOT6_RUN_DIR  (6 pilot tasks, 120 rollouts)
#   - BENCH12_RUN_DIR (12 new shared-primitive tasks, 240 rollouts)
#
# Usage:
#   bash scripts/run_xr1_benchmark18_eval_pipeline.sh
#
# Resuming an interrupted run:
#   BENCH18_EVAL_ROOT=/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_benchmark18_eval_<ts> \
#     bash scripts/run_xr1_benchmark18_eval_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/gs/bs/tga-shinoda/felid/envs/xiaomi_robotics_1_robocasa365/bin/python}"
PILOT6_RUN_DIR="${PILOT6_RUN_DIR:-/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_recovery_pilot6_full_20261128_142543}"
PILOT6_ZS_VLM_DIR="${PILOT6_ZS_VLM_DIR:-/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_vlm_dest_full_20261128_193617}"

if [[ -z "${BENCH12_RUN_DIR:-}" ]]; then
  BENCH12_RUN_DIR="$(ls -1dt /gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_recovery_bench12new_full_* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${BENCH12_RUN_DIR}" || ! -d "${BENCH12_RUN_DIR}" ]]; then
  echo "ERROR: BENCH12_RUN_DIR not found. Set BENCH12_RUN_DIR explicitly." >&2
  exit 1
fi

MODEL_DIR="${MODEL_DIR:-/gs/bs/tga-shinoda/felid/robocasa_checkpoints/lingbot_vla_v2/downloads/Qwen3-VL-4B-Instruct}"
VISION_SNAPSHOT="${VISION_SNAPSHOT:-/gs/bs/tga-shinoda/felid/hf_home/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c}"
VISION_WEIGHTS="${VISION_WEIGHTS:-/gs/bs/tga-shinoda/felid/hf_home/hub/models--lerobot--pi0_libero_finetuned_v044/snapshots/45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f/model.safetensors}"

TS="$(date +%Y%m%d_%H%M%S)"
EVAL_ROOT="${BENCH18_EVAL_ROOT:-/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_benchmark18_eval_${TS}}"
mkdir -p "${EVAL_ROOT}"

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export HF_HOME="/gs/bs/tga-shinoda/felid/hf_home"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

HELDOUT8=(
  BreadSelection
  PackIdenticalLunches
  StackBowlsCabinet
  WashFruitColander
  ArrangeTea
  KettleBoiling
  SearingMeat
  WaffleReheat
)

echo "========================================================================"
echo "=== XR-1 18-TASK BENCHMARK EVALUATION PIPELINE (TEST-1 + LLM + TEST-2) ==="
echo "Repo:          ${REPO_ROOT} ($(git rev-parse --short HEAD))"
echo "Pilot-6 dir:   ${PILOT6_RUN_DIR}"
echo "Bench-12 dir:  ${BENCH12_RUN_DIR}"
echo "Qwen3-VL dir:  ${MODEL_DIR}"
echo "Eval root:     ${EVAL_ROOT}"
echo "GPU:           ${CUDA_VISIBLE_DEVICES}"
echo "========================================================================"

echo ""
echo "=== Step 0: Preflight Unit Tests & 360-Rollout Dataset Verification ==="
"${PYTHON}" -u -m unittest discover -s tests -p "test_long_horizon_benchmark18*.py" -v
"${PYTHON}" -u -m unittest tests/test_long_horizon_vlm_lora.py -v

"${PYTHON}" -u - "${PILOT6_RUN_DIR}" "${BENCH12_RUN_DIR}" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
from scripts.xr1_lead_time import analyze_recovery_pilot6 as pilot6

roots = [Path(p) for p in sys.argv[1:]]
manifests = []
for r in roots:
    if (r / "manifest.jsonl").exists():
        manifests.append(r / "manifest.jsonl")
    else:
        manifests.extend(sorted(r.glob("shard*/manifest.jsonl")))
seen = set()
counts = Counter()
succ = Counter()
for mp in manifests:
    for line in mp.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row["rollout_id"]
        if rid in seen:
            raise SystemExit(f"ERROR: Duplicate rollout_id {rid}")
        seen.add(rid)
        t = row["task_name"]
        counts[t] += 1
        if not row.get("failed", False):
            succ[t] += 1

expected = set(pilot6.BENCHMARK18_ALL_TASKS)
missing_tasks = sorted(expected - set(counts))
under = {t: counts[t] for t in sorted(expected) if counts[t] < 20}
print(f"Verified total={len(seen)} rollouts across {len(counts)} tasks (successes={sum(succ.values())}, failures={len(seen)-sum(succ.values())})")
for t in pilot6.BENCHMARK18_ALL_TASKS:
    split = "TRAIN10" if t in pilot6.BENCHMARK18_TRAIN_TASKS else "HELDOUT8"
    print(f"  [{split:8s}] {t:24s} total={counts[t]:2d}  succ={succ[t]:2d}  fail={counts[t]-succ[t]:2d}")
if missing_tasks or under or len(seen) != 360:
    raise SystemExit(f"ERROR: Incomplete benchmark dataset! total={len(seen)}, missing={missing_tasks}, under_20={under}")
print("PASS: Exact 360/360 rollouts across all 18 tasks (20/task, 0 duplicates).")
PY

echo ""
echo "=== Step 1: Oracle Hierarchy & Test-2 Readiness (All 18 Tasks & Held-Out 8 Tasks) ==="
if [[ ! -f "${EVAL_ROOT}/oracle_all18/summary.json" ]]; then
  rm -rf "${EVAL_ROOT}/oracle_all18"
  "${PYTHON}" -u -m scripts.xr1_lead_time.analyze_recovery_pilot6 \
    --run "${PILOT6_RUN_DIR}" "${BENCH12_RUN_DIR}" \
    --output "${EVAL_ROOT}/oracle_all18"
else
  echo "Skipping oracle_all18 (already exists)"
fi

if [[ ! -f "${EVAL_ROOT}/oracle_heldout8/summary.json" ]]; then
  rm -rf "${EVAL_ROOT}/oracle_heldout8"
  "${PYTHON}" -u -m scripts.xr1_lead_time.analyze_recovery_pilot6 \
    --run "${PILOT6_RUN_DIR}" "${BENCH12_RUN_DIR}" \
    --tasks "${HELDOUT8[@]}" \
    --output "${EVAL_ROOT}/oracle_heldout8"
else
  echo "Skipping oracle_heldout8 (already exists)"
fi

echo ""
echo "=== Step 2: Extract Perception Features (360 Rollouts: Proprio + XR-1 DiT + SigLIP 3-Cam) ==="
if [[ ! -f "${EVAL_ROOT}/features/perception_features.npz" ]]; then
  rm -rf "${EVAL_ROOT}/features"
  "${PYTHON}" -u -m scripts.xr1_lead_time.extract_perception_features \
    --run "${PILOT6_RUN_DIR}" "${BENCH12_RUN_DIR}" \
    --output "${EVAL_ROOT}/features" \
    --vision-snapshot "${VISION_SNAPSHOT}" \
    --vision-weights "${VISION_WEIGHTS}" \
    --batch-size 128
else
  echo "Skipping perception feature extraction (already exists)"
fi

echo ""
echo "=== Step 3: Base Zero-Shot Qwen3-VL-4B-Instruct Scoring (All 18 Tasks, Reusing 120 Pilot-6 Checkpoints) ==="
if [[ ! -f "${EVAL_ROOT}/vlm_zeroshot_all18/vlm_destination_scores.npz" ]]; then
  mkdir -p "${EVAL_ROOT}/vlm_zeroshot_all18/per_rollout"
  if [[ -d "${PILOT6_ZS_VLM_DIR}/per_rollout" ]]; then
    cp -n "${PILOT6_ZS_VLM_DIR}/per_rollout/"*.json "${EVAL_ROOT}/vlm_zeroshot_all18/per_rollout/" 2>/dev/null || true
    echo "Seeded $(ls -1 "${EVAL_ROOT}/vlm_zeroshot_all18/per_rollout/"*.json 2>/dev/null | wc -l) cached pilot6 checkpoints"
  fi
  "${PYTHON}" -u -m scripts.xr1_lead_time.extract_vlm_destinations \
    --run "${PILOT6_RUN_DIR}" "${BENCH12_RUN_DIR}" \
    --output "${EVAL_ROOT}/vlm_zeroshot_all18" \
    --model-dir "${MODEL_DIR}" \
    --decision-stride 2 \
    --batch-size 16
else
  echo "Skipping base zero-shot VLM scoring (already exists)"
fi

echo ""
echo "=== Step 4: 10-Train -> 8-Held-Out Unseen-Task Qwen3-VL-4B LoRA ==="
if [[ ! -f "${EVAL_ROOT}/vlm_lora_10to8/vlm_scores/vlm_destination_scores.npz" ]]; then
  "${PYTHON}" -u -m scripts.xr1_lead_time.train_eval_vlm_lora_10to8 \
    --run "${PILOT6_RUN_DIR}" "${BENCH12_RUN_DIR}" \
    --output "${EVAL_ROOT}/vlm_lora_10to8" \
    --model-dir "${MODEL_DIR}" \
    --decision-stride 2 \
    --eval-batch-size 16 \
    --rank 16 \
    --alpha 32.0 \
    --epochs 2 \
    --micro-batch-size 4 \
    --grad-accum 4 \
    --lr 1e-4
else
  echo "Skipping 10-train -> 8-heldout LoRA (already exists)"
fi

echo ""
echo "=== Step 5a: 100% Automated LLM-Generated Task Graphs (Zero-Shot Qwen3-VL, 8 Held-Out Tasks) ==="
if [[ ! -f "${EVAL_ROOT}/vlm_llm_graph_zs_heldout8/vlm_destination_scores.npz" ]]; then
  "${PYTHON}" -u -m scripts.xr1_lead_time.generate_llm_task_graphs \
    --run "${PILOT6_RUN_DIR}" "${BENCH12_RUN_DIR}" \
    --tasks "${HELDOUT8[@]}" \
    --output "${EVAL_ROOT}/vlm_llm_graph_zs_heldout8" \
    --model-dir "${MODEL_DIR}" \
    --decision-stride 2 \
    --batch-size 16
else
  echo "Skipping LLM-graph zero-shot scoring (already exists)"
fi

echo ""
echo "=== Step 5b: 100% Automated LLM-Generated Task Graphs (10-Train LoRA Qwen3-VL, 8 Held-Out Tasks) ==="
if [[ ! -f "${EVAL_ROOT}/vlm_llm_graph_lora10_heldout8/vlm_destination_scores.npz" ]]; then
  "${PYTHON}" -u -m scripts.xr1_lead_time.generate_llm_task_graphs \
    --run "${PILOT6_RUN_DIR}" "${BENCH12_RUN_DIR}" \
    --tasks "${HELDOUT8[@]}" \
    --output "${EVAL_ROOT}/vlm_llm_graph_lora10_heldout8" \
    --model-dir "${MODEL_DIR}" \
    --existing-graphs "${EVAL_ROOT}/vlm_llm_graph_zs_heldout8/llm_task_graphs.json" \
    --lora-checkpoint "${EVAL_ROOT}/vlm_lora_10to8/lora_adapter_train10.pt" \
    --binary-prompt \
    --decision-stride 2 \
    --batch-size 16
else
  echo "Skipping LLM-graph LoRA-10 scoring (already exists)"
fi

echo ""
echo "=== Step 6: Test-1 10-Train -> 8-Held-Out Unseen-Task Monitor Evaluation ==="
for STREAM in proprio_actions xr1_dit; do
  OUT_T1="${EVAL_ROOT}/test1_${STREAM}"
  if [[ ! -f "${OUT_T1}/test1_benchmark18_summary.json" ]]; then
    rm -rf "${OUT_T1}"
    "${PYTHON}" -u -m scripts.xr1_lead_time.eval_test1_benchmark18 \
      --bundle "${EVAL_ROOT}/features" \
      --zs-vlm "${EVAL_ROOT}/vlm_zeroshot_all18" \
      --lora-vlm "${EVAL_ROOT}/vlm_lora_10to8/vlm_scores" \
      --llm-zs-vlm "${EVAL_ROOT}/vlm_llm_graph_zs_heldout8" \
      --llm-lora-vlm "${EVAL_ROOT}/vlm_llm_graph_lora10_heldout8" \
      --grasp-stream "${STREAM}" \
      --output "${OUT_T1}"
  else
    echo "Skipping test1_${STREAM} (already exists)"
  fi
done

echo ""
echo "=== Step 7: Powered 18-Task Equal-Duration Test-2 Continuation Evaluation ==="
if [[ ! -f "${EVAL_ROOT}/test2_all18/test2_benchmark18_summary.json" ]]; then
  rm -rf "${EVAL_ROOT}/test2_all18"
  "${PYTHON}" -u -m scripts.xr1_lead_time.eval_test2_benchmark18 \
    --bundle "${EVAL_ROOT}/features" \
    --vlm-scores "${EVAL_ROOT}/vlm_zeroshot_all18" \
    --output "${EVAL_ROOT}/test2_all18"
else
  echo "Skipping test2_all18 (already exists)"
fi

echo ""
echo "=== Step 8: Package Lightweight Summary Bundle ==="
tar -czf "${EVAL_ROOT}/benchmark18_eval_summaries.tar.gz" -C "${EVAL_ROOT}" \
  oracle_all18/summary.json \
  oracle_heldout8/summary.json \
  vlm_llm_graph_zs_heldout8/llm_task_graphs.json \
  test1_proprio_actions/test1_benchmark18_summary.json \
  test1_xr1_dit/test1_benchmark18_summary.json \
  test2_all18/test2_benchmark18_summary.json

ls -lh "${EVAL_ROOT}/benchmark18_eval_summaries.tar.gz"
echo "=== ALL 18-TASK BENCHMARK EVALUATION STEPS COMPLETE: ${EVAL_ROOT} ==="
