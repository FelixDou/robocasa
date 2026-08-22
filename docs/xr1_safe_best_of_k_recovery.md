# Bounded XR-1 SAFE best-of-K recovery pilot

## What this implementation tests

During the original high-level rollout, best-of-K stays dormant and the policy
makes one action-only server request per genuine inference. With
`sampling_seed_base` omitted, this path is backward compatible; the pilot
enables it only to match stochastic chunks across experimental arms. Once
`run_recovery_after_failed_rollout` starts the bounded retry, the policy:

1. requests `K` chunks from the same observation and language instruction;
2. uses explicit, recorded sampling seeds;
3. extracts the frozen horizon `1.0`, diffusion `1.0` SAFE feature token;
4. scores every candidate with the frozen independent-MLP checkpoints for
   seeds 0, 1, and 2;
5. applies each seed's frozen training-task normalization with the
   mathematically equivalent saturation-safe centered sigmoid transform and
   averages them;
6. commits only the candidate selected by `lowest_safe`, `random`, or
   `highest_safe`.

The simulator client uses Python 3.11, while pinned official SAFE's training
configuration dataclasses cannot be imported there because they use mutable
dataclass defaults. Recovery scoring therefore constructs only the official
independent model's inference projector locally. Its `projector` name, layer
order, activations, accumulation behavior, and state-dict keys match pinned
SAFE, and every frozen checkpoint is still loaded with `strict=True`. The
runtime rejects configurations with anything other than
`n_history_steps=1`, verifies the official SAFE commit, and preserves all
existing artifact hashes. Provenance records
`model_loader_protocol=official_safe_indep_inference_compat_v1`.

The candidate score is the normalized single-inference contribution. Frozen
checkpoints use a sigmoid output, and recovery candidates can drive that
float32 probability to exactly `1.0`. Direct probability-space ranking then
collapses distinct candidates into numerical ties. Protocol
`xr1_safe_single_inference_stable_centered_sigmoid_ensemble_v2` instead uses
the pre-sigmoid logit `z` and computes `-sigmoid(-z) / scale` in float64 for
each seed. This removes only `(1 - location) / scale`, which is constant across
the K candidates, so it preserves the frozen normalized detector's exact
candidate ordering in real arithmetic without changing checkpoints or fitting
new parameters. Official float32 probabilities, logits, legacy normalized
probabilities, and stable centered contributions are all retained in traces.

For the independent model, the executed prefix is an additive constant shared by all
candidates before the prospective detector's running-maximum operation. The
pilot intentionally does not reconstruct that alarm history, which can flatten
candidate differences after an earlier peak. This is therefore a within-state
ranking probe, not the calibrated SAFE alarm score and not a Subtask-SAFE
model.

The frozen detector's paired elapsed-time term is intentionally excluded from
candidate selection: all K chunks are proposed at the same environment step,
so that term is identical and has no ranking information.

Every genuine recovery inference writes a compact selection record into
`subtask.safe_best_of_k.records`, including candidate scores, sampling seeds,
per-seed raw probabilities, pre-sigmoid logits, legacy normalized probability
scores, stable centered scores, selected index, score spread, pairwise action
diversity, and frozen bundle/checkpoint provenance. The selected
candidate's full feature tensor remains available through the normal Xiaomi
inference-record interface.

## Required frozen artifacts

Read `docs/cluster_experiment_runbook.md` and complete the server/model patch
setup in `docs/safe_xiaomi_robotics_1.md`. The best-of-K server patch must be
applied after the base SAFE server patch.

```bash
export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export SAFE_REPO="$PROJECT_FS/SAFE"
export XR1_SAFE_REPO="$PROJECT_FS/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe"
export XR1_SAFE_CHECKPOINT="$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe"
export XR1_RUNTIME_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_calibration"
export XR1_RUNTIME_BUNDLE="$XR1_RUNTIME_ROOT/runtime_bundle.json"
export XR1_SAFE_PORT=10096
export XR1_CLIENT_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365"

test "$(git -C "$SAFE_REPO" rev-parse HEAD)" = b6036abe07b2b2bb9996afb2c07f13d6a9f507c0
test -f "$XR1_RUNTIME_BUNDLE"
for seed in 0 1 2; do
  test -f "$XR1_RUNTIME_ROOT/runtime/seed${seed}/model_final.ckpt"
  test -f "$XR1_RUNTIME_ROOT/runtime/seed${seed}/config.yaml"
done
grep -n "sampling_seed" "$XR1_SAFE_REPO/deploy/server.py"

PYTHONPATH="$ROBOCASA_REPO" "$XR1_CLIENT_ENV/bin/python" - <<'PY'
from robocasa.recovery.safe.xr1_best_of_k import (
    MODEL_LOADER_PROTOCOL,
    SCORING_PROTOCOL,
)
print("SAFE client compatibility loader:", MODEL_LOADER_PROTOCOL)
print("SAFE recovery scoring protocol:", SCORING_PROTOCOL)
PY
```

Start and verify the SAFE-capable XR-1 server before launching the simulator,
using the command in `docs/safe_xiaomi_robotics_1.md`. Do not proceed unless
the model-loaded line, listening port, server process, GPU process, storage,
and log path all pass the runbook preflight.

## One-rollout infrastructure smoke

This run is intentionally bounded. It proves wiring only; if the original
rollout succeeds, no recovery or best-of-K selection should occur.

```bash
module load miniconda
eval "$('/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda' shell.bash hook)"

export PILOT_TAG=xr1_safe_bok_smoke_$(date +%Y%m%d_%H%M%S)
export PILOT_OUTPUT="$STORAGE_BS/robocasa_rollouts/recovery/${PILOT_TAG}.json"
export PILOT_VIDEO_DIR="$STORAGE_BS/robocasa_rollouts/recovery/${PILOT_TAG}_videos"
export PILOT_LOG="$STORAGE_BS/robocasa_logs/eval/${PILOT_TAG}.log"

mkdir -p "$(dirname "$PILOT_OUTPUT")" "$PILOT_VIDEO_DIR" "$(dirname "$PILOT_LOG")"
cd "$ROBOCASA_REPO"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 WANDB_MODE=disabled \
"$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.evaluate_recovery_benchmark \
  --output "$PILOT_OUTPUT" \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --policy-arg model_path="$XR1_SAFE_CHECKPOINT" \
  --policy-arg host=127.0.0.1 \
  --policy-arg port="$XR1_SAFE_PORT" \
  --policy-arg replan_steps=16 \
  --policy-arg safe_best_of_k=4 \
  --policy-arg safe_candidate_strategy=lowest_safe \
  --policy-arg safe_candidate_seed=7000 \
  --policy-arg sampling_seed_base=1000 \
  --policy-arg safe_runtime_bundle="$XR1_RUNTIME_BUNDLE" \
  --policy-arg safe_repo="$SAFE_REPO" \
  --policy-arg safe_device=cpu \
  --env-interface gym \
  --task-set atomic_seen \
  --envs CloseBlenderLid \
  --split pretrain \
  --modes env_to_last_good \
  --recovery-level atomic \
  --num-rollouts 1 \
  --seed 7 \
  --include-trace \
  --video-dir "$PILOT_VIDEO_DIR" \
  --video-render-source obs \
  2>&1 | tee "$PILOT_LOG"
```

After the smoke, require a completed JSON result, no server/client traceback,
distinct sampling seeds, and non-identical candidates before interpreting SAFE
ranking. A zero-record result is expected when no recovery was attempted.

```bash
"$XR1_CLIENT_ENV/bin/python" - "$PILOT_OUTPUT" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
records = []
for mode in payload.get("modes", {}).values():
    for rollout in mode.get("rollouts", []):
        records.extend(rollout.get("subtask", {}).get("safe_best_of_k", {}).get("records", []))
print("selection_records:", len(records))
for record in records:
    print({
        "scores": record["scores"],
        "selected_index": record["selected_index"],
        "sampling_seeds": record["sampling_seeds"],
        "score_spread": record["score_spread"],
        "all_identical": record["action_diversity"]["all_identical"],
    })
PY
```

## Pilot comparison

Freeze task/seed allocation before scaling. Run matched recovery attempts for:

- `safe_best_of_k=1`: ordinary single-chunk recovery;
- `safe_best_of_k=4`, `safe_candidate_strategy=random`: sampling-only control;
- `safe_best_of_k=4`, `safe_candidate_strategy=lowest_safe`: proposed method;
- optionally `safe_best_of_k=4`, `safe_candidate_strategy=highest_safe`: negative control.

The primary outcome is atomic recovery success under the same recovery mode,
horizon, initial-state seed, ordinary-inference seed schedule, and
candidate-seed schedule. Keep `sampling_seed_base` and `safe_candidate_seed`
identical for the K=4 random and lowest-SAFE arms. This makes the high-level
trajectory and first recovery candidate set identical. After the arms select
different actions, their states and later candidate chunks can diverge; treat
the full recovery outcomes as matched by task and initial seed, not as fully
paired candidate decisions. Also report how often candidate actions are
identical, SAFE score spread, server errors, and runtime. Do not claim recovery
utility from retrospective SAFE separation alone.

## Ten-rollout-per-task screening run

Start with the two K=4 arms below over exactly the 38 tasks represented in the
frozen runtime bundle. This is 380 high-level rollouts per arm. Run the arms
sequentially against one server; simultaneous clients require separate Xiaomi
servers, GPUs, ports, logs, and output files.

```bash
mapfile -t XR1_FROZEN_TASKS < <(
  "$XR1_CLIENT_ENV/bin/python" - "$XR1_RUNTIME_BUNDLE" <<'PY'
import json
import sys

bundle = json.load(open(sys.argv[1]))
task_sets = [
    set(bundle["task_normalizations"][str(seed)])
    for seed in bundle["model_seeds"]
]
if any(task_set != task_sets[0] for task_set in task_sets[1:]):
    raise SystemExit("Frozen ensemble seeds disagree on task coverage")
tasks = sorted(task_sets[0])
if len(tasks) != 38:
    raise SystemExit(f"Expected 38 frozen tasks, found {len(tasks)}")
print("\n".join(tasks))
PY
)
printf 'frozen_tasks=%s\n' "${#XR1_FROZEN_TASKS[@]}"
printf '%s\n' "${XR1_FROZEN_TASKS[@]}"

export SCREEN_TAG=xr1_safe_bok_10pertask_$(date +%Y%m%d_%H%M%S)
export ARM=random  # Run random first; then set ARM=lowest_safe and repeat.
export ARM_OUTPUT="$STORAGE_BS/robocasa_rollouts/recovery/${SCREEN_TAG}_${ARM}.json"
export ARM_LOG="$STORAGE_BS/robocasa_logs/eval/${SCREEN_TAG}_${ARM}.log"

test "$ARM" = random || test "$ARM" = lowest_safe
test ! -e "$ARM_OUTPUT"
cd "$ROBOCASA_REPO"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 WANDB_MODE=disabled \
nohup "$XR1_CLIENT_ENV/bin/python" -u -m \
  robocasa.recovery.evaluate_recovery_benchmark \
  --output "$ARM_OUTPUT" \
  --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
  --policy-arg model_path="$XR1_SAFE_CHECKPOINT" \
  --policy-arg host=127.0.0.1 \
  --policy-arg port="$XR1_SAFE_PORT" \
  --policy-arg replan_steps=16 \
  --policy-arg safe_best_of_k=4 \
  --policy-arg safe_candidate_strategy="$ARM" \
  --policy-arg safe_candidate_seed=7000 \
  --policy-arg sampling_seed_base=1000 \
  --policy-arg safe_runtime_bundle="$XR1_RUNTIME_BUNDLE" \
  --policy-arg safe_repo="$SAFE_REPO" \
  --policy-arg safe_device=cpu \
  --env-interface gym \
  --envs "${XR1_FROZEN_TASKS[@]}" \
  --split pretrain \
  --modes env_to_last_good \
  --recovery-level atomic \
  --num-rollouts 10 \
  --seed 7 \
  --include-trace \
  > "$ARM_LOG" 2>&1 &

echo "pid=$!"
echo "output=$ARM_OUTPUT"
echo "log=$ARM_LOG"
```

Before launching the second arm, require the first output to have
`partial=false`, 380 rollout records, zero errors, and a nonzero number of
recovery selection records. Ten rollouts per task is a screening sample:
analyze the aggregate task-seed-matched recovery difference and uncertainty,
but do not interpret individual task rates as stable estimates.
