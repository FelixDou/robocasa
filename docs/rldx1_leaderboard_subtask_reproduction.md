# RLDX-1 leaderboard reproduction with subtask tracking

This run reproduces RLDX-1 on the 50-task RoboCasa365 target benchmark while
recording the same runtime predicate progress used by the existing pi0
subtask evaluator. It preserves the official RLDX policy server, batched
action-chunk execution, `n_action_steps=8`, `n_envs=5`, per-task horizons,
episode seeds, and videos. The only added output is a resumable
`subtask_rollouts.json` for every task.

The leaderboard submission to target is:

| Split | Published success |
|---|---:|
| Atomic-Seen (18 tasks) | 67.6% |
| Composite-Seen (16 tasks) | 27.9% |
| Composite-Unseen (16 tasks) | 8.5% |
| Overall 50-task average | 36.0% |

It used RLDX commit `ef05cd4ae634ff97d672d42275febbc0b92cc192`,
checkpoint `RLWRLD/RLDX-1-FT-RC365`, 50 episodes per task, batch size 192,
and 250,000 training steps. The checkpoint has a non-commercial RLWRLD model
license; accept its Hugging Face terms before launching the server.

## 1. Fresh-terminal setup

Use an allocated GPU node and keep source on `/gs/fs` and all generated
artifacts on `/gs/bs`:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export RLDX_REPO="$PROJECT_FS/RLDX-1"
export RLDX_SIM_PY="$RLDX_REPO/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
export ROBOCASA_LOG_ROOT="$STORAGE_BS/robocasa_logs"
export ROBOCASA_ROLLOUT_ROOT="$STORAGE_BS/robocasa_rollouts"
export HF_HOME="$STORAGE_BS/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export UV_CACHE_DIR="$STORAGE_BS/uv_cache"
export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_ENABLED=0

mkdir -p \
  "$ROBOCASA_LOG_ROOT/eval" \
  "$ROBOCASA_ROLLOUT_ROOT" \
  "$HF_HOME" \
  "$UV_CACHE_DIR"

test -x "$RLDX_SIM_PY" || { echo "Missing RLDX simulator environment"; exit 1; }
```

Pull this evaluator into the cluster checkout, then verify the RLDX source
revision. Use a clean RLDX worktree before detaching at the submission commit.

```bash
cd "$ROBOCASA_REPO"
git pull origin main

cd "$RLDX_REPO"
git status --short
git fetch origin
git switch --detach ef05cd4ae634ff97d672d42275febbc0b92cc192
git rev-parse HEAD
```

## 2. Start one RLDX server

```bash
export RLDX_SERVER_LOG="$ROBOCASA_LOG_ROOT/eval/rldx1_rc365_server_20100_$(date +%Y%m%d_%H%M%S).log"

cd "$RLDX_REPO"
CUDA_VISIBLE_DEVICES=0 \
nohup uv run python -u rldx/eval/run_rldx_server.py \
  --model-path RLWRLD/RLDX-1-FT-RC365 \
  --embodiment-tag GENERAL_EMBODIMENT \
  --device cuda \
  --host 127.0.0.1 \
  --port 20100 \
  --use-sim-policy-wrapper \
  > "$RLDX_SERVER_LOG" 2>&1 &

echo "server_pid=$!"
echo "server_log=$RLDX_SERVER_LOG"

until ss -ltn | grep -q ':20100'; do
  tail -30 "$RLDX_SERVER_LOG" 2>/dev/null || true
  sleep 10
done
```

Do not launch the simulator until the port is listening and the server log
confirms the model loaded.

## 3. One-task smoke test

The smoke test uses one simulator environment and two episodes. It keeps the
full primitive-step trace so the artifact proves that subtask payloads reach
the evaluator through RLDX's `MultiStepWrapper`.

```bash
export RUN_TAG=rldx1_rc365_subtasks_smoke_$(date +%Y%m%d_%H%M%S)
export RLDX_SUBTASK_ROOT="$ROBOCASA_ROLLOUT_ROOT/$RUN_TAG"
export SMOKE_LOG="$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}.log"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
"$RLDX_SIM_PY" -u robocasa/recovery/evaluate_rldx_official_subtasks.py \
  --rldx-repo "$RLDX_REPO" \
  --output-dir "$RLDX_SUBTASK_ROOT" \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 20100 \
  --envs MakeIceLemonade \
  --split target \
  --n-episodes 2 \
  --n-envs 1 \
  --n-action-steps 8 \
  --sync-envs \
  --include-trace \
  --require-leaderboard-commit \
  2>&1 | tee "$SMOKE_LOG"
```

Validate both binary task success and subtask availability:

```bash
cd "$ROBOCASA_REPO"
"$RLDX_SIM_PY" robocasa/recovery/print_subtask_rollout_summary.py \
  "$RLDX_SUBTASK_ROOT/MakeIceLemonade/subtask_rollouts.json"

"$RLDX_SIM_PY" robocasa/recovery/print_subtask_accuracy.py \
  "$RLDX_SUBTASK_ROOT/MakeIceLemonade/subtask_rollouts.json"

"$RLDX_SIM_PY" - "$RLDX_SUBTASK_ROOT/MakeIceLemonade/subtask_rollouts.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
print("path:", path)
print("partial:", data["partial"])
print("summary:", data["summary"])
assert len(data["rollouts"]) == 2
assert data["summary"]["subtask_eval_unavailable_rollouts"] == 0
assert all(rollout["subtask_trace"] for rollout in data["rollouts"])
PY
```

## 4. Two-GPU, 10-episode-per-task starter evaluation

This profile evaluates all 50 tasks with 10 episodes per task: 500 rollouts
total. Two disjoint task shards assign 25 tasks to each GPU. Each RLDX server
receives batches from five simulator environments, preserving the official
`n_envs=5` episode layout.

Start one server per GPU on distinct ports:

```bash
export SERVER_TAG=rldx1_rc365_2gpu_servers_$(date +%Y%m%d_%H%M%S)

for spec in "0 20100" "1 20101"; do
  set -- $spec
  gpu=$1
  port=$2
  log="$ROBOCASA_LOG_ROOT/eval/${SERVER_TAG}_${port}.log"

  cd "$RLDX_REPO"
  CUDA_VISIBLE_DEVICES="$gpu" \
  nohup uv run python -u rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1-FT-RC365 \
    --embodiment-tag GENERAL_EMBODIMENT \
    --device cuda \
    --host 127.0.0.1 \
    --port "$port" \
    --use-sim-policy-wrapper \
    > "$log" 2>&1 &
  echo "gpu=$gpu port=$port pid=$! log=$log"
done

until ss -ltn | grep -q ':20100' && ss -ltn | grep -q ':20101'; do
  ss -ltn | grep -E ':20100|:20101' || true
  tail -10 "$ROBOCASA_LOG_ROOT/eval/${SERVER_TAG}_20100.log" 2>/dev/null || true
  tail -10 "$ROBOCASA_LOG_ROOT/eval/${SERVER_TAG}_20101.log" 2>/dev/null || true
  sleep 10
done
```

Launch the two task shards:

```bash
export RUN_TAG=rldx1_rc365_target50_10eps_2gpu_$(date +%Y%m%d_%H%M%S)
export RLDX_SUBTASK_ROOT="$ROBOCASA_ROLLOUT_ROOT/$RUN_TAG"
mkdir -p "$RLDX_SUBTASK_ROOT"

for spec in "0 20100 0" "1 20101 1"; do
  set -- $spec
  gpu=$1
  port=$2
  shard=$3
  log="$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}_shard${shard}.log"

  cd "$ROBOCASA_REPO"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
  nohup "$RLDX_SIM_PY" -u \
    robocasa/recovery/evaluate_rldx_official_subtasks.py \
      --rldx-repo "$RLDX_REPO" \
      --output-dir "$RLDX_SUBTASK_ROOT" \
      --policy-client-host 127.0.0.1 \
      --policy-client-port "$port" \
      --task-set target50 \
      --split target \
      --n-episodes 10 \
      --n-envs 5 \
      --n-action-steps 8 \
      --no-include-trace \
      --num-shards 2 \
      --shard-index "$shard" \
      --max-errors 3 \
      --require-leaderboard-commit \
      > "$log" 2>&1 &
  echo "gpu=$gpu shard=$shard pid=$! log=$log"
done

echo "RUN_TAG=$RUN_TAG"
echo "RLDX_SUBTASK_ROOT=$RLDX_SUBTASK_ROOT"
```

Monitor both shards:

```bash
tail -f \
  "$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}_shard0.log" \
  "$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}_shard1.log"
```

The task JSONs still contain final predicates, ordered completion, maximum
progress, first blocker, stuck subtask, and failure modes. `--no-include-trace`
only omits the large per-step trace from disk.

## 5. Full four-GPU target50 evaluation

Use four RLDX servers and four disjoint task shards. Each shard writes unique
task directories and its own summary, so completed task JSONs are safe to
resume. The full run keeps videos because RLDX's official video wrapper also
owns deterministic per-episode seeding. `--no-include-trace` avoids storing
millions of per-step trace rows; completion, blocker, failure-mode, and maximum
ordered-progress summaries are still computed from every primitive step.

Start four servers:

```bash
export SERVER_TAG=rldx1_rc365_4gpu_servers_$(date +%Y%m%d_%H%M%S)

for spec in "0 20100" "1 20101" "2 20102" "3 20103"; do
  set -- $spec
  gpu=$1
  port=$2
  log="$ROBOCASA_LOG_ROOT/eval/${SERVER_TAG}_${port}.log"

  cd "$RLDX_REPO"
  CUDA_VISIBLE_DEVICES="$gpu" \
  nohup uv run python -u rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1-FT-RC365 \
    --embodiment-tag GENERAL_EMBODIMENT \
    --device cuda \
    --host 127.0.0.1 \
    --port "$port" \
    --use-sim-policy-wrapper \
    > "$log" 2>&1 &
  echo "gpu=$gpu port=$port pid=$! log=$log"
done

until ss -ltn | grep -q ':20100' \
  && ss -ltn | grep -q ':20101' \
  && ss -ltn | grep -q ':20102' \
  && ss -ltn | grep -q ':20103'; do
  ss -ltn | grep -E ':20100|:20101|:20102|:20103' || true
  sleep 10
done
```

Launch the four disjoint shards:

```bash
export RUN_TAG=rldx1_rc365_target50_subtasks_$(date +%Y%m%d_%H%M%S)
export RLDX_SUBTASK_ROOT="$ROBOCASA_ROLLOUT_ROOT/$RUN_TAG"
mkdir -p "$RLDX_SUBTASK_ROOT"

for spec in "0 20100 0" "1 20101 1" "2 20102 2" "3 20103 3"; do
  set -- $spec
  gpu=$1
  port=$2
  shard=$3
  log="$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}_shard${shard}.log"

  cd "$ROBOCASA_REPO"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
  nohup "$RLDX_SIM_PY" -u \
    robocasa/recovery/evaluate_rldx_official_subtasks.py \
      --rldx-repo "$RLDX_REPO" \
      --output-dir "$RLDX_SUBTASK_ROOT" \
      --policy-client-host 127.0.0.1 \
      --policy-client-port "$port" \
      --task-set target50 \
      --split target \
      --n-episodes 50 \
      --n-envs 5 \
      --n-action-steps 8 \
      --no-include-trace \
      --num-shards 4 \
      --shard-index "$shard" \
      --max-errors 3 \
      --require-leaderboard-commit \
      > "$log" 2>&1 &
  echo "gpu=$gpu shard=$shard pid=$! log=$log"
done

echo "RUN_TAG=$RUN_TAG"
echo "RLDX_SUBTASK_ROOT=$RLDX_SUBTASK_ROOT"
```

Monitor without deleting partial outputs:

```bash
tail -f "$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}_shard"*.log

find "$RLDX_SUBTASK_ROOT" -name subtask_rollouts.json -print | sort
ps -ef | grep -E '[r]un_rldx_server.py|[e]valuate_rldx_official_subtasks.py'
ss -ltnp | grep -E ':20100|:20101|:20102|:20103' || true
nvidia-smi
```

Rerunning the same shard command with the same output root resumes from the
contiguous `(env_idx, episode_idx)` records already present. The evaluator
refuses to merge a partial file if its checkpoint, commit, seed, split,
horizon, vector width, action-chunk size, video mode, or trace mode changed.

## 6. Aggregate and inspect the result

After every shard exits, aggregate all task files without contacting a policy
server:

```bash
cd "$ROBOCASA_REPO"
"$RLDX_SIM_PY" robocasa/recovery/evaluate_rldx_official_subtasks.py \
  --rldx-repo "$RLDX_REPO" \
  --output-dir "$RLDX_SUBTASK_ROOT" \
  --task-set target50 \
  --aggregate-only
```

The main artifact is:

```text
$RLDX_SUBTASK_ROOT/benchmark_summary.json
```

Require 50 complete tasks, no missing tasks, no task errors, the requested
rollout count per task, no duplicate episode keys, and zero
subtask-unavailable rollouts before comparing to the published split scores.
Use `EXPECTED_EPISODES=10` for the two-GPU starter run or `50` for the full
leaderboard reproduction.

```bash
export EXPECTED_EPISODES=10

EXPECTED_EPISODES="$EXPECTED_EPISODES" \
"$RLDX_SIM_PY" - "$RLDX_SUBTASK_ROOT/benchmark_summary.json" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
expected_episodes = int(os.environ["EXPECTED_EPISODES"])
print("overall:", 100 * data["overall_mean_task_success_rate"])
for group, summary in data["groups"].items():
    print(group, 100 * summary["mean_task_success_rate"], summary)
print("missing_tasks:", data["missing_tasks"])
print("errors:", data["errors"])
assert data["num_tasks"] == 50
assert data["num_complete_tasks"] == 50
assert not data["missing_tasks"]
assert not data["errors"]
assert all(task["num_rollouts"] == expected_episodes for task in data["tasks"])
assert all(
    task["subtask_eval_unavailable_rollouts"] == 0 for task in data["tasks"]
)
PY
```

Generate the per-task bottleneck tables used for pi0 comparisons:

```bash
find "$RLDX_SUBTASK_ROOT" -name subtask_rollouts.json -print0 \
  | xargs -0 "$RLDX_SIM_PY" robocasa/recovery/print_subtask_accuracy.py \
      --out-json "$RLDX_SUBTASK_ROOT/subtask_accuracy.json" \
      --out-bottleneck-csv "$RLDX_SUBTASK_ROOT/subtask_bottlenecks.csv" \
      --out-bottleneck-md "$RLDX_SUBTASK_ROOT/subtask_bottlenecks.md"
```

Compare the split-level success rates to 67.6%, 27.9%, and 8.5%, and compare
the overall unweighted task average to 36.0%. Keep deviations separate from
subtask progress: a rollout can make meaningful ordered progress and still
fail the official sparse task predicate.
