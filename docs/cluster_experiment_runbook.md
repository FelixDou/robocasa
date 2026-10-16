# RoboCasa Cluster Experiment Runbook

This runbook records the user's cluster layout and the commands repeatedly used
for RoboCasa, OpenPI, RLDX, recovery evaluation, and failure-dataset creation.
It is written for future Codex tasks and for copy-paste use in a fresh terminal.

## 1. Local versus cluster execution

Code is edited in the local macOS checkout:

```text
/Users/felixdoublet/Desktop/PhD/Shinoda_lab/Code/robocasa
```

Experiments run in a GPU allocation on the cluster. Source repositories live on
`/gs/fs`; large or generated artifacts live on `/gs/bs`. Changes move from the
local machine to the cluster through GitHub:

```bash
# Local machine, after requested implementation and validation
git add path/to/changed_file.py
git commit -m "..."
git push origin main

# Cluster
cd /gs/fs/tga-shinoda/felid/robocasa
git pull origin main
```

Do not assume the cluster can see uncommitted or unpushed local changes.

## 2. Storage layout

| Purpose | Canonical location |
|---|---|
| RoboCasa source | `/gs/fs/tga-shinoda/felid/robocasa` |
| RLDX source | `/gs/fs/tga-shinoda/felid/RLDX-1` |
| OpenPI source | `/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/openpi` |
| Conda environment | `/gs/bs/tga-shinoda/felid/envs/robocasa_openpi` |
| Durable OpenPI reproduction checkpoint | `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/repro/openpi/pi0_robocasa_pretrain_human300/pi0_robocasa_pretrain_human300_repro_20260505_233725/74999` |
| Node-local OpenPI checkpoint | `/tmp/ut06746/pi0_74999` |
| Durable checkpoints | `/gs/bs/tga-shinoda/felid/robocasa_checkpoints` |
| Downloaded RoboCasa datasets | `/gs/bs/tga-shinoda/felid/robocasa_datasets` |
| Generated failure datasets | `/gs/bs/tga-shinoda/felid/robocasa_failure_datasets` |
| Rollouts and recovery results | `/gs/bs/tga-shinoda/felid/robocasa_rollouts` |
| Logs | `/gs/bs/tga-shinoda/felid/robocasa_logs` |
| Hugging Face cache | `/gs/bs/tga-shinoda/felid/hf_home` |
| UV cache | `/gs/bs/tga-shinoda/felid/uv_cache` |

Use `/gs/fs` only for source code. Large outputs under `/gs/fs` have previously
caused quota failures.

## 3. Canonical fresh-terminal exports

Run this in each new cluster terminal:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi

export USER_ID=ut06746
export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export WORK_ROOT=/work/6/ut06746

export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export RLDX_REPO="$PROJECT_FS/RLDX-1"
export OPENPI_REPO="$PROJECT_FS/robocasa_benchmark_repos/openpi"

export ROBOCASA_LOG_ROOT="$STORAGE_BS/robocasa_logs"
export ROBOCASA_ROLLOUT_ROOT="$STORAGE_BS/robocasa_rollouts"
export ROBOCASA_FAILURE_DATASET_ROOT="$STORAGE_BS/robocasa_failure_datasets"
export ROBOCASA_DATASET_BASE="$STORAGE_BS/robocasa_datasets"
export ROBOCASA_CKPT_ROOT="$STORAGE_BS/robocasa_checkpoints"

export PI0_REPRO_SOURCE="$ROBOCASA_CKPT_ROOT/repro/openpi/pi0_robocasa_pretrain_human300/pi0_robocasa_pretrain_human300_repro_20260505_233725/74999"
export PI0_REPRO_LOCAL=/tmp/ut06746/pi0_74999

export RLDX_SIM_PY="$RLDX_REPO/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"

export UV_CACHE_DIR="$STORAGE_BS/uv_cache"
export UV_LINK_MODE=copy
export HF_HOME="$STORAGE_BS/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export NO_ALBUMENTATIONS_UPDATE=1

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_ENABLED=0

mkdir -p \
  "$ROBOCASA_LOG_ROOT/eval" \
  "$ROBOCASA_ROLLOUT_ROOT" \
  "$ROBOCASA_FAILURE_DATASET_ROOT" \
  "$ROBOCASA_DATASET_BASE" \
  "$UV_CACHE_DIR" \
  "$HF_HOME"

echo "ROBOCASA_REPO=$ROBOCASA_REPO"
echo "OPENPI_REPO=$OPENPI_REPO"
echo "RLDX_REPO=$RLDX_REPO"
echo "ROBOCASA_LOG_ROOT=$ROBOCASA_LOG_ROOT"
echo "ROBOCASA_ROLLOUT_ROOT=$ROBOCASA_ROLLOUT_ROOT"
echo "ROBOCASA_FAILURE_DATASET_ROOT=$ROBOCASA_FAILURE_DATASET_ROOT"
echo "PI0_REPRO_SOURCE=$PI0_REPRO_SOURCE"
```

Validate the environment before launching expensive work:

```bash
which python
python --version
python - <<'PY'
import numpy
print("numpy", numpy.__version__)
PY

test -d "$ROBOCASA_REPO/.git" && echo "RoboCasa repo OK"
test -f "$OPENPI_REPO/scripts/serve_policy.py" && echo "OpenPI server OK"
test -d "$PI0_REPRO_SOURCE/params" && echo "OpenPI checkpoint OK"
test -x "$RLDX_SIM_PY" && echo "RLDX simulator Python OK"
```

## 4. Server architecture and ports

### OpenPI / pi0

The OpenPI process owns the pi0 model and exposes a WebSocket policy service.
The RoboCasa simulator is a separate process and calls it through localhost.

```text
GPU 0: OpenPI checkpoint server, port 8120
          ^ WebSocket policy requests
          |
        RoboCasa simulator / dataset shard 0

GPU 1: optional second OpenPI server, port 8121
          ^ WebSocket policy requests
          |
        RoboCasa simulator / dataset shard 1
```

When GPU count is limited, the model server and simulator can share the same
GPU. This was the working pattern for two parallel shards: server 8120 plus
simulator shard 0 on GPU 0, and server 8121 plus simulator shard 1 on GPU 1.

### RLDX

RLDX uses its own policy server and ports:

```text
GPU 0: RLDX server, TCP/ZMQ port 20100 <- RoboCasa evaluator
GPU 1: RLDX server, TCP/ZMQ port 20101 <- optional second evaluator
GPU 2: RLDX server, TCP/ZMQ port 20102 <- optional third evaluator
```

Do not mix OpenPI ports `812x` and RLDX ports `2010x`.

## 5. Copy the pi0 checkpoint to node-local storage

`/tmp` is local to the allocated node and is empty in a new allocation. Copy
the checkpoint before starting OpenPI:

```bash
mkdir -p /tmp/ut06746
mkdir -p "$PI0_REPRO_LOCAL"

rsync -a --info=progress2 \
  "$PI0_REPRO_SOURCE/" \
  "$PI0_REPRO_LOCAL/"

test -d "$PI0_REPRO_LOCAL/params" || {
  echo "Checkpoint copy is incomplete"
  exit 1
}
```

If `rsync` appears quiet, inspect it from another terminal rather than starting
a second copy:

```bash
ps -ef | grep '[r]sync'
du -sh "$PI0_REPRO_LOCAL"
find "$PI0_REPRO_LOCAL" -maxdepth 2 -type f | head
```

## 6. Start OpenPI policy servers

### One GPU / one server

```bash
export SERVER_TAG=pi0_74999_server_$(date +%Y%m%d_%H%M%S)
export SERVER_LOG="$ROBOCASA_LOG_ROOT/eval/${SERVER_TAG}_8120.log"

pkill -f "scripts/serve_policy.py.*8120" 2>/dev/null || true

cd "$OPENPI_REPO"
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
nohup python -u scripts/serve_policy.py \
  --port=8120 policy:checkpoint \
  --policy.config=pi0_robocasa_pretrain_human300 \
  --policy.dir="$PI0_REPRO_LOCAL" \
  > "$SERVER_LOG" 2>&1 &

echo "server_pid=$!"
echo "server_log=$SERVER_LOG"

until ss -ltn | grep -q ':8120'; do
  date
  tail -20 "$SERVER_LOG" 2>/dev/null || true
  sleep 10
done

echo "OpenPI server is listening on 8120"
```

Expected final server message:

```text
server listening on 0.0.0.0:8120
```

### Two GPUs / two parallel servers

```bash
export SERVER_TAG=pi0_74999_2gpu_servers_$(date +%Y%m%d_%H%M%S)

for spec in "0 8120" "1 8121"; do
  set -- $spec
  gpu=$1
  port=$2
  log="$ROBOCASA_LOG_ROOT/eval/${SERVER_TAG}_${port}.log"

  pkill -f "scripts/serve_policy.py.*${port}" 2>/dev/null || true

  cd "$OPENPI_REPO"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  nohup python -u scripts/serve_policy.py \
    --port="$port" policy:checkpoint \
    --policy.config=pi0_robocasa_pretrain_human300 \
    --policy.dir="$PI0_REPRO_LOCAL" \
    > "$log" 2>&1 &

  echo "gpu=$gpu port=$port pid=$! log=$log"
done

until ss -ltn | grep -q ':8120' && ss -ltn | grep -q ':8121'; do
  date
  ss -ltn | grep -E ':8120|:8121' || true
  sleep 10
done
```

Server health checks:

```bash
ss -ltnp | grep -E ':8120|:8121' || true
ps -ef | grep '[s]cripts/serve_policy.py'
nvidia-smi
tail -60 "$ROBOCASA_LOG_ROOT/eval/${SERVER_TAG}_8120.log"
```

## 7. pi0 smoke test

Always run a small test before a full evaluation or dataset collection:

```bash
cd "$ROBOCASA_REPO"
git pull origin main

export RUN_TAG=pi0_failure_dataset_smoke_$(date +%Y%m%d_%H%M%S)
export DATASET_DIR="$ROBOCASA_FAILURE_DATASET_ROOT/$RUN_TAG"
export LOG="$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}.log"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python -u robocasa/recovery/create_recovery_failure_dataset.py \
  --output-dir "$DATASET_DIR" \
  --policy-module robocasa.recovery.openpi_websocket_policy:make_policy \
  --policy-name pi0_repro_74999 \
  --policy-arg host=127.0.0.1 \
  --policy-arg port=8120 \
  --envs OpenDrawer PickPlaceDrawerToCounter MakeIceLemonade \
  --split target \
  --num-rollouts 2 \
  --seed 0 \
  --stop-after-failures 3 \
  --max-failed-per-task 1 \
  --include-trace \
  --record-actions \
  --record-videos \
  --video-render-source sim \
  2>&1 | tee "$LOG"
```

Inspect the manifest:

```bash
python - "$DATASET_DIR/recovery_failure_dataset.json" <<'PY'
import collections
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
samples = d.get("samples", [])
print("manifest:", p)
print("partial:", d.get("partial"))
print("samples:", len(samples))
print("errors:", len(d.get("errors", [])))
print("tasks:", collections.Counter(s.get("task_name") for s in samples))
for sample in samples:
    print(
        sample.get("sample_id"),
        "completed=", [x.get("subtask_id") for x in sample.get("completed_subtasks", [])],
        "failed=", (sample.get("failed_subtask") or {}).get("subtask_id"),
    )
PY
```

## 8. Create a balanced 500-sample failure dataset on two GPUs

Each shard records five failures for each of the 50 target tasks: 250 samples
per GPU and 500 total. Use distinct seeds, output directories, ports, and logs.

```bash
cd "$ROBOCASA_REPO"
git pull origin main

export RUN_TAG=pi0_target50_failure_dataset_500samples_2gpu_$(date +%Y%m%d_%H%M%S)
export DATASET_DIR="$ROBOCASA_FAILURE_DATASET_ROOT/$RUN_TAG"
mkdir -p "$DATASET_DIR/gpu0" "$DATASET_DIR/gpu1"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup python -u robocasa/recovery/create_recovery_failure_dataset.py \
  --output-dir "$DATASET_DIR/gpu0" \
  --policy-module robocasa.recovery.openpi_websocket_policy:make_policy \
  --policy-name pi0_repro_74999 \
  --policy-arg host=127.0.0.1 \
  --policy-arg port=8120 \
  --task-set all_target \
  --split target \
  --num-rollouts 80 \
  --seed 0 \
  --stop-after-failures 250 \
  --max-failed-per-task 5 \
  --include-trace \
  --record-actions \
  --record-videos \
  --video-render-source sim \
  > "$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}_gpu0.log" 2>&1 &
echo "gpu0_pid=$!"

CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
nohup python -u robocasa/recovery/create_recovery_failure_dataset.py \
  --output-dir "$DATASET_DIR/gpu1" \
  --policy-module robocasa.recovery.openpi_websocket_policy:make_policy \
  --policy-name pi0_repro_74999 \
  --policy-arg host=127.0.0.1 \
  --policy-arg port=8121 \
  --task-set all_target \
  --split target \
  --num-rollouts 80 \
  --seed 10000 \
  --stop-after-failures 250 \
  --max-failed-per-task 5 \
  --include-trace \
  --record-actions \
  --record-videos \
  --video-render-source sim \
  > "$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}_gpu1.log" 2>&1 &
echo "gpu1_pid=$!"

echo "RUN_TAG=$RUN_TAG"
echo "DATASET_DIR=$DATASET_DIR"
```

Monitor from another terminal after recreating the exports:

```bash
tail -f \
  "$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}_gpu0.log" \
  "$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}_gpu1.log"
```

The known 1,500-sample dataset is rooted at:

```text
/gs/bs/tga-shinoda/felid/robocasa_failure_datasets/pi0_target50_failure_dataset_500samples_2gpu_20260706_132505
```

Its merged manifest is:

```text
merged_1500/recovery_failure_dataset.json
```

## 9. pi0 recovery evaluation

The generic evaluator supports `env_to_last_good`, `eef_to_last_good`, and
`continue_from_failure`. Use atomic recovery prompts while preserving subtask
diagnostics:

```bash
cd "$ROBOCASA_REPO"

export RUN_TAG=pi0_atomic_recovery_smoke_$(date +%Y%m%d_%H%M%S)
export RECOVERY_ROOT="$ROBOCASA_ROLLOUT_ROOT/$RUN_TAG"
mkdir -p "$RECOVERY_ROOT/env_to_last_good"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python -u robocasa/recovery/evaluate_recovery_benchmark.py \
  --policy-module robocasa.recovery.openpi_websocket_policy:make_policy \
  --policy-arg host=127.0.0.1 \
  --policy-arg port=8120 \
  --modes env_to_last_good \
  --envs PickPlaceDrawerToCounter SteamInMicrowave MakeIceLemonade \
  --split target \
  --num-rollouts 1 \
  --recovery-level atomic \
  --recovery-horizon-use-original-task \
  --include-trace \
  --split-recovery-video \
  --video-render-source sim \
  --output "$RECOVERY_ROOT/env_to_last_good/results.json" \
  --video-dir "$RECOVERY_ROOT/videos" \
  2>&1 | tee "$ROBOCASA_LOG_ROOT/eval/${RUN_TAG}.log"
```

For a full run replace `--envs ...` with `--task-set all_target` and set
`--num-rollouts 5`. Parallel modes need separate servers/ports or must run
sequentially. Do not point simultaneous clients at one stateful server unless
that behavior has been explicitly validated.

Video rendering has historically been sensitive to EGL state and recovery
resets. Keep quantitative results and video quality as separate validation
checks. Useful alternatives are:

```text
--video-render-source obs_exact
--video-render-source sim
--split-recovery-video --no-merge-split-video
```

## 10. RLDX leaderboard reproduction

The official non-recovery wrapper can start its own RLDX server:

```bash
cd "$RLDX_REPO"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
nohup bash run_scripts/eval/robocasa_365/eval_robocasa365.sh \
  --model-path RLWRLD/RLDX-1-FT-RC365 \
  --task-set atomic_seen \
  --n-episodes 5 \
  --n-envs 1 \
  --server-port 20100 \
  --server-warmup-sec 30 \
  > "$ROBOCASA_LOG_ROOT/eval/rldx_atomic_seen_5eps_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Repeat with `composite_seen` and `composite_unseen` on distinct GPUs and ports.

Known five-episode, no-recovery reproduction results for
`RLWRLD/RLDX-1-FT-RC365` were:

| Task set | Tasks | Mean success |
|---|---:|---:|
| `atomic_seen` | 18 | 57.8% |
| `composite_seen` | 16 | 22.5% |
| `composite_unseen` | 16 | 10.0% |
| Overall task average | 50 | 31.2% |

## 11. Start RLDX servers for recovery evaluation

Start the model server in the RLDX repository and use the official simulation
Python for the RoboCasa evaluator:

```bash
cd "$RLDX_REPO"
export RLDX_SERVER_TAG=rldx_recovery_servers_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 \
nohup uv run python -u rldx/eval/run_rldx_server.py \
  --model-path RLWRLD/RLDX-1-FT-RC365 \
  --embodiment-tag GENERAL_EMBODIMENT \
  --host 127.0.0.1 \
  --port 20100 \
  --use-sim-policy-wrapper \
  > "$ROBOCASA_LOG_ROOT/eval/${RLDX_SERVER_TAG}_20100.log" 2>&1 &

until ss -ltn | grep -q ':20100'; do
  tail -20 "$ROBOCASA_LOG_ROOT/eval/${RLDX_SERVER_TAG}_20100.log" 2>/dev/null || true
  sleep 10
done
```

Official-path recovery smoke test:

```bash
cd "$ROBOCASA_REPO"

export RUN_TAG=rldx_official_recovery_smoke_$(date +%Y%m%d_%H%M%S)
export RECOVERY_ROOT="$ROBOCASA_ROLLOUT_ROOT/$RUN_TAG"
mkdir -p "$RECOVERY_ROOT"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
"$RLDX_SIM_PY" robocasa/recovery/evaluate_rldx_official_recovery.py \
  --rldx-repo "$RLDX_REPO" \
  --output "$RECOVERY_ROOT/results.json" \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 20100 \
  --envs PickPlaceCounterToCabinet \
  --split target \
  --modes env_to_last_good \
  --num-rollouts 5 \
  --subtask-horizon 40
```

Use `--match-recovery-horizon-to-no-progress` instead of a fixed horizon when
that is the intended experiment. Leave `--compat-normalize-policy-io` disabled
for the official RLDX rollout path.

## 12. Download official RoboCasa datasets to `/gs/bs`

The downloader follows `robocasa.macros.DATASET_BASE_PATH`. The cluster checkout
should contain this local-only configuration in `robocasa/macros_private.py`:

```python
DATASET_BASE_PATH = "/gs/bs/tga-shinoda/felid/robocasa_datasets"
```

Then run:

```bash
cd "$ROBOCASA_REPO"
nohup python -u -m robocasa.scripts.download_datasets \
  --split target \
  --source human \
  > "$ROBOCASA_LOG_ROOT/eval/download_target_human_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Downloaded target datasets are LeRobot/Parquet, not HDF5. Inspect:

```text
$ROBOCASA_DATASET_BASE/v1.0/target/atomic/*/*/lerobot/
$ROBOCASA_DATASET_BASE/v1.0/target/composite/*/*/lerobot/
```

Composite Parquet files contain frame-level fields including
`annotation.human.subtask`, `annotation.human.subtask_name`,
`annotation.human.subtask_stage`, and `subtask_idx`. Standalone atomic datasets
do not ship those stage columns.

## 13. Analyze a merged failure dataset

```bash
cd "$ROBOCASA_REPO"
git pull origin main

export BASE="$ROBOCASA_FAILURE_DATASET_ROOT/pi0_target50_failure_dataset_500samples_2gpu_20260706_132505"
export MANIFEST="$BASE/merged_1500/recovery_failure_dataset.json"
export ANALYSIS_DIR="$BASE/merged_1500/analysis"

python -u robocasa/recovery/analyze_recovery_failure_dataset.py \
  --manifest "$MANIFEST" \
  --output-dir "$ANALYSIS_DIR" \
  --top-n 25 \
  --formats png pdf

find "$ANALYSIS_DIR" -maxdepth 1 -type f -print | sort
```

The analyzer writes CSV tables, PNG/PDF figures, and `summary.md`. Interpret
ordered subtask progress separately from task success. In the current merged
1,500-sample dataset, all failures were recorded at the rollout horizon and the
stored composite `atomic_sequence` was not multi-step, so exact failure-time and
atomic-progress analyses are not yet supported by that manifest.

## 14. Standard result inspection

Generic JSON summary:

```bash
python - "$RESULT_JSON" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
print("path:", p)
print("partial:", d.get("partial"))
print("top-level summary:", d.get("summary"))
print("errors:", len(d.get("errors", [])))
print("samples:", len(d.get("samples", [])))

for mode, payload in d.get("modes", {}).items():
    print("\nmode:", mode)
    print("summary:", payload.get("summary"))
    print("errors:", payload.get("num_errors", 0))
    print("records:", len(payload.get("rollouts", [])))
PY
```

Background-process checks:

```bash
jobs -l
ps -ef | grep -E '[s]erve_policy.py|[r]un_rldx_server.py|[e]valuate_|[c]reate_recovery_failure_dataset.py'
ss -ltnp | grep -E ':8120|:8121|:20100|:20101|:20102' || true
nvidia-smi
```

## 15. Storage and quota diagnostics

On `Disk quota exceeded`, stop launching new work and inspect both bytes and
inodes:

```bash
df -h /gs/fs/tga-shinoda/felid /gs/bs/tga-shinoda/felid
df -ih /gs/fs/tga-shinoda/felid /gs/bs/tga-shinoda/felid
du -sh /gs/fs/tga-shinoda/felid /gs/bs/tga-shinoda/felid
du -h --max-depth=1 /gs/bs/tga-shinoda/felid | sort -h
quota -s 2>/dev/null || true
lfs quota -h /gs/fs/tga-shinoda/felid 2>/dev/null || true
```

Do not delete partial datasets or logs until their value has been assessed.

## 16. Common mistakes

- Empty exports in a new terminal produce paths such as `/eval/...` and
  `/rldx_recovery...`. Re-run the canonical export block before any launch.
- `bash: : command not found` commonly means a required variable such as
  `SIM_PY` expanded to an empty string.
- A server log may not exist immediately after `nohup`; wait briefly, then use
  `ls` and `tail`.
- Python 3.9 cannot parse OpenPI's `match` statements. Activate the
  `robocasa_openpi` environment and verify Python 3.11.
- RoboCasa currently requires NumPy 2.2.5 in the relevant simulator checkout.
- Never start a client just because a server process exists; verify the port is
  listening and the log says the model loaded.
- Do not reuse `$RUN_TAG`, `$RECOVERY_ROOT`, or `$DATASET_DIR` from another
  terminal unless they have been explicitly exported again.
- `nohup` redirects output; no terminal output is expected. Monitor the log in
  another terminal.
- `^C` only interrupts the foreground process. It does not necessarily stop a
  background server; inspect `ps`, `jobs`, and listening ports.
- Warnings about `mimicgen`, `robosuite_models`, or `mink` are usually
  non-fatal for the PandaOmron RoboCasa tasks used here. Tracebacks, nonzero
  error counts, corrupt JSON, quota errors, and server protocol errors are not.

## 17. Security and reproducibility

- Never commit GitHub credentials, SSH private keys, Hugging Face tokens, or
  W&B API keys.
- Record the git commit, checkpoint path, task set, split, seed, policy port,
  GPU mapping, command, log path, and output manifest for every experiment.
- Use unique timestamped run tags.
- Keep source manifests immutable after merge when possible; write repaired or
  merged variants to new directories and retain backup names when applying a
  label repair.
