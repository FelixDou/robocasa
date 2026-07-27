# ABot-M0.5 RoboCasa365 evaluation

These scripts reproduce the released ABot-M0.5 evaluation while preserving the
cluster storage contract:

- ABot source code: `/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/ABot-Manipulation`
- ABot environment: `/gs/bs/tga-shinoda/felid/envs/abot_m05`
- durable checkpoint: `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/abot_m05`
- node-local serving copy: `/tmp/ut06746/abot_m05`
- results: `/gs/bs/tga-shinoda/felid/robocasa_rollouts/abot_m05`
- logs: `/gs/bs/tga-shinoda/felid/robocasa_logs/eval`

The ABot source is pinned to commit
`7642747ed2817b241dde5df06e17ee80192718ad`. The model repository is
`acvlab/ABot-M0.5-RoboCasa365`, not the stale `acvlab/abot-m0.5` identifier
shown in the upstream getting-started guide.

## 1. Accept the checkpoint gate

Visit
[acvlab/ABot-M0.5-RoboCasa365](https://huggingface.co/acvlab/ABot-M0.5-RoboCasa365),
log in, and accept its access conditions. Do not place a Hugging Face token in
this repository.

## 2. Prepare source and environments

Start from a fresh cluster terminal:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export HF_HOME="$STORAGE_BS/hf_home"

cd "$ROBOCASA_REPO"

# First pass: clone the pinned ABot source and build its environment.
bash robocasa/scripts/abot_m05/setup_cluster.sh --skip-download

# Authenticate interactively after accepting the gated model.
HF_HOME="$HF_HOME" \
  "$STORAGE_BS/envs/abot_m05/bin/hf" auth login

# Second pass: download and verify the 34 GB checkpoint.
bash robocasa/scripts/abot_m05/setup_cluster.sh --skip-env
```

The setup stops before download if storage has less than 60 GiB free. A quota
error is a stop-and-inspect condition; check both `df -h` and `lfs quota -h`
instead of retrying.

`torch` attention is the default and does not require compiling FlashAttention.
Use `--with-flash-attn` during setup only when that backend is specifically
needed.

## 3. Validate before allocating a long run

In a GPU allocation:

```bash
cd /gs/fs/tga-shinoda/felid/robocasa

bash robocasa/scripts/abot_m05/evaluate_cluster.sh \
  preflight \
  --gpus 0
```

Preflight checks the exact ABot commit, clean upstream checkout, server and
client imports, compatible Click/Typer CLI packages, RoboCasa 1.0.1, and all
required checkpoint artifacts.

If preflight reports that the client Click is incompatible, repair only the
RoboCasa client environment and verify its dependency set:

```bash
CLIENT_PYTHON=/gs/bs/tga-shinoda/felid/envs/robocasa_openpi/bin/python

"$CLIENT_PYTHON" -m pip install click==8.2.1
"$CLIENT_PYTHON" -m pip check
"$CLIENT_PYTHON" -c \
  "import click, typer; print('click', click.__version__, 'typer', typer.__version__)"
```

Typer's installed implementation subclasses the generic `click.Choice`, which
requires Click 8.2 or newer. The conservative 8.2.1 pin avoids pulling a newer
Click release into the shared RoboCasa/OpenPI environment.

## 4. One-episode smoke

The smoke test starts the server before the simulator client, evaluates
`CloseFridge`, records a video and JSON result, and shuts the server down:

```bash
bash robocasa/scripts/abot_m05/evaluate_cluster.sh \
  smoke \
  --gpus 0 \
  --task CloseFridge \
  --run-tag abot_m05_smoke_$(date +%Y%m%d_%H%M%S)
```

By default, the checkpoint is incrementally copied from durable storage to
node-local `/tmp` before serving. Use `--no-local-copy` only when `/tmp` cannot
hold 40 GiB and loading directly from `/gs/bs` is acceptable.

## 5. Pilot and full evaluation

The leaderboard protocol uses the `pretrain` scene split and 50 episodes for
each of 50 tasks. The wrapper forces `SPLIT=pretrain` and explicitly overrides
the inconsistent 20/50/20 defaults in the upstream split scripts.

Five-episode pilot:

```bash
bash robocasa/scripts/abot_m05/evaluate_cluster.sh \
  all \
  --gpus 0,1,2,3 \
  --episodes 5 \
  --run-tag abot_m05_pilot5_$(date +%Y%m%d_%H%M%S)
```

Two-GPU, 10-episode-per-task run with semantic subtask progress:

```bash
export RUN_TAG=abot_m05_subtask10_$(date +%Y%m%d_%H%M%S)
export LOG="/gs/bs/tga-shinoda/felid/robocasa_logs/eval/${RUN_TAG}.log"

nohup bash robocasa/scripts/abot_m05/evaluate_cluster.sh \
  all \
  --gpus 0,1 \
  --episodes 10 \
  --subtask-progress \
  --run-tag "$RUN_TAG" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "log=$LOG"
```

`--subtask-progress` installs a client-only Gym hook through `PYTHONPATH`; it
does not modify the pinned ABot checkout. Each batch writes
`subtask_progress.json` beside its `robocasa_eval.json`. After all splits
finish, `subtask_progress_summary.json` reports availability, mean maximum
ordered progress, stuck subtasks, and failure modes. Official binary success
remains unchanged, while semantic progress sidecars are required and validated
for all 18 Atomic-Seen and 32 composite tasks.

Full 2,500-rollout reproduction:

```bash
export RUN_TAG=abot_m05_full50_$(date +%Y%m%d_%H%M%S)
export LOG="/gs/bs/tga-shinoda/felid/robocasa_logs/eval/${RUN_TAG}.log"

nohup bash robocasa/scripts/abot_m05/evaluate_cluster.sh \
  all \
  --gpus 0,1,2,3 \
  --episodes 50 \
  --run-tag "$RUN_TAG" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "log=$LOG"
```

The three splits run sequentially so they do not compete for the same GPUs.
Reusing the same `--run-tag` resumes the official scheduler from completed
per-batch JSON files.

Monitor with:

```bash
tail -f "$LOG"
nvidia-smi

RUN_ROOT="/gs/bs/tga-shinoda/felid/robocasa_rollouts/abot_m05/$RUN_TAG"

# Per-task aggregates only. A raw robocasa_eval.json count also includes
# batch-level files and therefore double-counts completed tasks.
find "$RUN_ROOT" -type f -name robocasa_eval.json \
  ! -path '*/batches/*'

find "$RUN_ROOT" -type f \
  \( -name summary.json -o -name overall_summary.json \
     -o -name subtask_progress_summary.json \)
```

After all three splits finish, the wrapper validates exactly 18 Atomic-Seen,
16 Composite-Seen, and 16 Composite-Unseen task results with the requested
episode count. It writes `overall_summary.json` and exits nonzero instead of
silently reporting a partial run.

## 6. Analyze ordered subtask progress

After a complete `--subtask-progress` run, generate validated rollout-, task-,
and split-level statistics plus PNG and SVG figures on a login or CPU node:

```bash
RUN_ROOT=/gs/bs/tga-shinoda/felid/robocasa_rollouts/abot_m05/abot_m05_subtask10_20260725_200251
CLIENT_PYTHON=/gs/bs/tga-shinoda/felid/envs/robocasa_openpi/bin/python

"$CLIENT_PYTHON" \
  robocasa/scripts/abot_m05/analyze_subtask_progress.py \
  "$RUN_ROOT" \
  --expected-episodes 10
```

The command refuses partial or unbalanced input before writing results. Outputs
are placed in `$RUN_ROOT/subtask_analysis`:

- `subtask_progress_statistics.json` with fixed-task bootstrap intervals
- rollout-, task-, split-, stuck-subtask-, and failure-mode CSV tables
- split success versus progress, task-level gap, progress-survival, and
  stuck-subtask figures as both PNG and SVG
- `README.md` with a compact split summary and interpretation guardrail

The confidence intervals resample rollouts within each fixed benchmark task.
Official binary task success remains the leaderboard metric; ordered progress
is a diagnostic that reveals partial completion hidden by binary success.

The pinned upstream scheduler can return status 1 after a transient failed
attempt even when its retries completed every episode. The harness validates
the aggregated split in that case and continues only when task uniqueness,
task count, episode count, success count, and success rate are all complete.
Reusing the same run tag remains safe: completed episodes are discovered from
their preserved batch JSONs and skipped.

Expected leaderboard reference for the artifact named ABot-M0.5:

| Split | Reported success |
|---|---:|
| Atomic-Seen | 75.6% |
| Composite-Seen | 37.7% |
| Composite-Unseen | 3.3% |
| Overall task average | 40.3% |

The ABot paper/repository also associates 46.6% with “ABot-M0.5,” while the
RoboCasa leaderboard attributes that score to ABot-M0.6. Preserve per-task and
split summaries so the released checkpoint can be compared against both claims.
