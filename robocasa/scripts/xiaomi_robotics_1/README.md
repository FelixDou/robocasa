# Xiaomi-Robotics-1 RoboCasa365 evaluation

This harness reproduces evaluation from Xiaomi's released RoboCasa365
checkpoint. It does not reproduce proprietary pre-training.

Pinned artifacts:

- Xiaomi source commit: `4da1db0a4deefa6de7ebb4ef0b8754017290f5f7`
- checkpoint repository: `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`
- checkpoint revision: `0d1aa76d0d82debc9b611e4d1e231096434d5be4`
- RoboCasa version: `1.0.1`

The official protocol is `pretrain` scenes, `target50`, 50 trials per task,
seed 7, observation history 4 at interval 2, 16 executed actions per query,
and crop ratio 0.95. Xiaomi's released reference contains 1,432 successes in
2,500 episodes (57.28%). This differs slightly from the 57.4% leaderboard row;
the harness reports deltas against both references.

## Storage layout

Source remains on `/gs/fs`; every large or generated artifact is placed on
`/gs/bs`:

| Purpose | Path |
|---|---|
| Xiaomi source | `/gs/fs/tga-shinoda/felid/robocasa_benchmark_repos/Xiaomi-Robotics-1` |
| model-server environment | `/gs/bs/tga-shinoda/felid/envs/xiaomi_robotics_1_server` |
| simulator environment | `/gs/bs/tga-shinoda/felid/envs/xiaomi_robotics_1_robocasa365` |
| checkpoint | `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365` |
| results and scheduler state | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/xiaomi_robotics_1` |
| logs | `/gs/bs/tga-shinoda/felid/robocasa_logs/eval` |

The dedicated simulator environment is cloned from the known-working
`robocasa_openpi` environment, then repointed at this RoboCasa checkout and
pinned to Transformers 4.57.1. This avoids modifying the OpenPI environment.

## 1. Fetch this branch on the cluster

The local branch must be committed and pushed before these files exist on the
cluster. After publication:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

cd /gs/fs/tga-shinoda/felid/robocasa
git fetch origin codex/xiaomi-robotics-1-eval
git switch codex/xiaomi-robotics-1-eval
git pull --ff-only origin codex/xiaomi-robotics-1-eval
```

## 2. Prepare source, environments, and checkpoint

Run setup on a login or CPU node with network access. The public checkpoint is
about 10.1 GB and does not require Hugging Face authentication.

```bash
cd /gs/fs/tga-shinoda/felid/robocasa

bash robocasa/scripts/xiaomi_robotics_1/setup_cluster.sh --dry-run
bash robocasa/scripts/xiaomi_robotics_1/setup_cluster.sh
```

Setup requires at least 45 GiB free under `/gs/bs` before it starts. If it
reports a quota problem, inspect bytes and inodes instead of retrying:

```bash
df -h /gs/fs/tga-shinoda/felid /gs/bs/tga-shinoda/felid
df -ih /gs/fs/tga-shinoda/felid /gs/bs/tga-shinoda/felid
lfs quota -h /gs/fs/tga-shinoda/felid 2>/dev/null || true
```

Each setup component is reusable:

```bash
bash robocasa/scripts/xiaomi_robotics_1/setup_cluster.sh \
  --skip-source --skip-server-env --skip-client-env
```

## 3. Preflight in a GPU allocation

Preflight does not load the 5.44B-parameter model. It verifies both Python
environments, CUDA visibility, exact source/checkpoint pins, RoboCasa 1.0.1,
the checkpoint processor's `robocasa365` normalization, and free ports.

```bash
cd /gs/fs/tga-shinoda/felid/robocasa

bash robocasa/scripts/xiaomi_robotics_1/evaluate_cluster.sh \
  preflight \
  --gpus 0
```

## 4. One-rollout smoke

The smoke loads one model server, waits until it reports `Model loaded`, and
then runs Xiaomi's short one-trial `CloseBlenderLid` smoke. A task failure is
not an infrastructure failure; the purpose is to prove model loading,
processor compatibility, socket inference, simulator stepping, and merging.

```bash
export RUN_TAG=xr1_rc365_smoke_$(date +%Y%m%d_%H%M%S)

bash robocasa/scripts/xiaomi_robotics_1/evaluate_cluster.sh \
  smoke \
  --gpus 0 \
  --run-tag "$RUN_TAG"
```

Inspect:

```bash
tail -100 "/gs/bs/tga-shinoda/felid/robocasa_logs/eval/${RUN_TAG}_server_10086.log"
tail -100 "/gs/bs/tga-shinoda/felid/robocasa_logs/eval/${RUN_TAG}_client.log"
python -m json.tool \
  "/gs/bs/tga-shinoda/felid/robocasa_rollouts/xiaomi_robotics_1/${RUN_TAG}/reproduction_summary.json"
```

## 5. Bounded pilot

Run one complete task with the official horizon and 50 trials before spending
on all target50 tasks:

```bash
export RUN_TAG=xr1_rc365_close_blender_lid_50_$(date +%Y%m%d_%H%M%S)
export LOG="/gs/bs/tga-shinoda/felid/robocasa_logs/eval/${RUN_TAG}_wrapper.log"

nohup bash robocasa/scripts/xiaomi_robotics_1/evaluate_cluster.sh \
  run \
  --gpus 0 \
  --episodes 50 \
  --task CloseBlenderLid \
  --run-tag "$RUN_TAG" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "log=$LOG"
```

## 6. Full 2,500-rollout reproduction

Xiaomi's reference uses eight independent model servers. The same protocol can
run with fewer GPUs; it will simply take longer. Use one listed GPU per worker.

```bash
export RUN_TAG=xr1_rc365_full50_$(date +%Y%m%d_%H%M%S)
export LOG="/gs/bs/tga-shinoda/felid/robocasa_logs/eval/${RUN_TAG}_wrapper.log"

nohup bash robocasa/scripts/xiaomi_robotics_1/evaluate_cluster.sh \
  run \
  --gpus 0,1,2,3,4,5,6,7 \
  --episodes 50 \
  --run-tag "$RUN_TAG" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "log=$LOG"
```

The wrapper starts every server first, verifies every port, launches the
simulator workers, preserves worker errors and partial scheduler state, and
strictly validates the final task/episode/seed inventory. It refuses to reuse
an existing run tag so partial evidence cannot be overwritten accidentally.

Monitor from another terminal:

```bash
tail -f "$LOG"
nvidia-smi
ss -ltnp | grep -E ':1008[6-9]|:1009[0-3]' || true

RUN_ROOT="/gs/bs/tga-shinoda/felid/robocasa_rollouts/xiaomi_robotics_1/$RUN_TAG"
SCHEDULER_ROOT="/gs/bs/tga-shinoda/felid/robocasa_rollouts/xiaomi_robotics_1/scheduler/$RUN_TAG"

find "$SCHEDULER_ROOT/results" -type f -name '*.json' | wc -l
find "$SCHEDULER_ROOT/errors" -type f -name '*.json' | wc -l
find "$RUN_ROOT" -maxdepth 2 -type f -print | sort
```

Do not call a partial run a reproduction. A valid full result must contain 50
tasks, 50 episodes per task, 2,500 unique scheduler results, no pending/running
jobs, no error records, and the expected global episode seeds.

## 7. Revalidate an existing completed run

```bash
bash robocasa/scripts/xiaomi_robotics_1/evaluate_cluster.sh \
  summarize \
  --episodes 50 \
  --run-tag "$RUN_TAG"
```

The primary output is:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/xiaomi_robotics_1/<run-tag>/reproduction_summary.json
```

It contains official binary success at overall and split levels, completeness
checks, artifact provenance, Xiaomi released-reference deltas, and leaderboard
deltas. Videos are disabled by default; add `--save-failure-videos` only when
qualitative debugging is needed because video volume can be substantial.
