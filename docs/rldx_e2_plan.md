# E2: closed-loop reset and retry on new RLDX-1 scenes (plan and runbook)

- **Paper:** E2 ("Closed loop: reset and retry") of the RSS 2027 paper; it estimates H4.
- **Written:** 2026-11-06, after E1 and before any E2 rollout.
- **Author's decision (2026-11-06):** E2 resets at **SAFE-LSTM + P**, the monitor E1 confirmed on new
  RLDX-1 rollouts (H3 held for it, not for Hide-and-Seek + P). The Stage 1 text named Hide-and-Seek + P;
  it is kept here as a descriptive condition, computed from the same rollouts at no extra cost.
- **Code:** `scripts/rldx_e2/` (`plan_e2.py`, `collect_e2.py`, `check_e2.py`, `analyze_e2.py`),
  `scripts/rldx_e1/score_e1.py` (`--set-names e2`), tests in `tests/test_rldx_e2.py`.

## Question

Within a budget of 1.25 H per scene, does resetting and retrying when SAFE-LSTM + P alarms give a
higher success rate than resetting when the timer P alarms? E2 estimates the difference with a 95% CI.
It does not test it: the reset simulation predicts about +0.006 (CI half-width about 0.02).

## Design

- **Scenes.** 50 new scenes per task on the 20 RLDX-1 tasks, 1,000 in all, collected with E1's settings
  (the recorded collection's) and a new base seed, 1401. Seeds 1411 … 1491 are reserved for re-running
  the tasks of a stopped shard. The smoke run uses seed 1102.
- **Two attempts per scene.** The collector runs unchanged with the official RLDX-1 protocol (one
  environment per task, repeated resets) and 100 rollouts per task. Only the environment's `reset()` is
  replaced (`collect_e2.py`):
  - reset 2j builds scene j and keeps its full simulator state, model XML and episode metadata;
  - reset 2j+1 restores that state through the environment's `reset_to`, the RoboCasa playback path,
    and the collector resets the policy as for any rollout.
  - Both attempts run to the budget B = ceil(1.25 H) control steps (`--horizon-scale 1.25`), with SAFE
    features recorded.
- **Why one shared retry.** The Stage 1 design branches each monitor from the shared first attempt at
  its alarm and retries "with a seed that differs from the first attempt's but is shared by all
  monitors". RLDX-1's policy server samples without a seed we can set, so the shared retry is
  implemented directly: one retry per scene, from the restored initial state, used by every rule. The
  policy does not observe elapsed time, so a retry started at a later alarm is the same retry with less
  budget left.
- **Single reset.** A retry is not reset again. The reset simulation allowed repeated resets; with 1.25 H
  a second retry rarely has time to succeed.
- **No resume.** Each task is one collector run in its own directory. An error stops it
  (`--no-continue-on-error`), and `--resume` is refused: a replayed reset sequence would not rebuild
  the same scenes. An interrupted task is collected again from scratch with the next reserved seed;
  its partial attempts are not used.

## Reset rules (fixed by `plan_e2.py`)

Alarms are computed on the first attempt with E1's code (`confirm_lead.alarm_vec`). Each is what an
online monitor would raise, because the scores are causal: SAFE-LSTM is a forward LSTM and the score is
a running maximum.

| Rule | Alarm | Fixed from | Latency charged |
|---|---|---|---|
| none | never | — | — |
| P | timer at ceil(min(1, ρ_k + λ̂) H_k) | λ̂ = 0.0739, E1's margin on its 400 new calibration successes; ρ_k from the freeze manifest | 0 |
| SAFE-LSTM + P (primary) | P, or the score crossing u = 7.9314 | E1: λ̂ = 0.0739; path from the freeze manifest | 1 control step (50 ms) |
| R_opt | time only, at ceil(c_k H_k) | per-task cutoffs c_k chosen in the reset simulation on recorded training rollouts at 1.25 H (r = 0.245; `reset.json` `3dac8155…`) | 0 |
| Hide-and-Seek + P (descriptive) | P, or the score crossing −0.17 + 20 λ̂ | E1: λ̂ = 0.0956; path from the freeze manifest | 1 control step |

**Scene outcome under rule c.** a is c's alarm inference on the first attempt A, and t_a is the control
step at which that inference happens:
- if c does not alarm during A, the scene succeeds if A succeeded;
- otherwise c resets at t_a. The scene succeeds if the retry R succeeds within B − t_a − d_c steps,
  where d_c is c's latency. A success of A at or after the alarm inference is pre-empted, as in E1's
  false-alarm count;
- under "none", the scene succeeds if A succeeds within B.

## Gates

1. **Smoke** (seed 1102, two scenes of one task):
   - the retry restored its first attempt's state: same state hash, residual ≤ 1e-6;
   - the first observations match: images within 2.0 mean absolute difference, other arrays within
     1e-4, same instruction;
   - the smoke rollouts, which run past H, score with SAFE-LSTM over their whole length.
2. **Collection** (`check_e2.py`):
   - E1's settings and feature checks, with E2's seeds and budget;
   - every task complete in exactly one directory, with resets 0 … 99 once each;
   - the pair checks above for every scene. A failing pair is excluded; more than 2% excluded stops E2.
3. **Scoring** (`score_e1.py`, unchanged checks): the frozen models reproduce their recorded scores
   exactly before any E2 score is written.

## Analysis (`analyze_e2.py`)

- **Primary:** the success rate under SAFE-LSTM + P minus that under P, over the usable scenes, with a
  95% CI. Scenes are resampled within task (2,000 replicates, seed 20261106).
- **Also reported:**
  - each rule's success rate with its CI;
  - P − none, SAFE-LSTM + P − none, R_opt − P, and SAFE-LSTM + P − R_opt;
  - resets, unnecessary resets (of first attempts that succeeded within B), and resets that turned a
    success into a failure;
  - the latency sensitivity, charging a full inference (8 steps) per learned-monitor reset;
  - the split by task type;
  - the restore check: the retry's success after a successful and after a failed first attempt;
  - Hide-and-Seek + P − P, as a descriptive result.
- **Prediction (reset simulation at r = 0.245, 1.25 H):**
  - none 0.012;
  - P 0.046;
  - SAFE-LSTM + P 0.052;
  - R_opt 0.051.

  These are success gained over the first attempt alone, a different scale from E2's success rates.
  The difference between monitors is what carries over.

## Runbook

Run from a pinned worktree at the commit that contains this plan. `E2_SHA` is that commit's full SHA.
- **GPUs:** four for steps 4–6.
- **CPU only:** steps 0–3 and 7–8 (scoring uses one GPU).
- **Never** pull or check out in the shared checkout `$PROJECT_FS/robocasa`.

### 0. Session

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/26.9.3/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export RLDX_REPO=$PROJECT_FS/RLDX-1
export CLIENT_PY=$STORAGE_BS/envs/robocasa_openpi/bin/python SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export UV_CACHE_DIR=$STORAGE_BS/uv_cache UV_LINK_MODE=copy HF_HOME=$STORAGE_BS/hf_home
export TRANSFORMERS_CACHE=$HF_HOME/transformers WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export SAFE35_ROOT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_original_safe35_natural50_20260808_203158
export RLDX_EXPORT=$SAFE35_ROOT/official_safe_supported20_natural50
export RLDX_RAW=$($CLIENT_PY -I -c "import json,sys; print(json.load(open(sys.argv[1]))['source_dataset'])" $RLDX_EXPORT/conversion_report.json)
export E1_FREEZE=$STORAGE_BS/robocasa_checkpoints/safe/e2_freeze_20261029_205743/e2_freeze_manifest.json
export RLDX_SNAPSHOT=$HF_HOME/hub/models--RLWRLD--RLDX-1-FT-RC365/snapshots/587e9ecdcc5e7184fcc17f58713908edff5af041
export E1_ROOT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_e1_20261105_210723
export RESET_SIM=$STORAGE_BS/robocasa_checkpoints/safe/xr1_list_c_20261103_211633/reset.json
export E2_SHA=<full SHA of the commit containing this plan>
export E2_REPO=$PROJECT_FS/robocasa_rldx_e2_${E2_SHA:0:8}
export E2_ROOT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_e2_$(date +%Y%m%d_%H%M%S)
export E2_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $E2_ROOT)
mkdir -p $E2_ROOT $E2_LOG; env | grep -E '^(E1_|E2_|RLDX_|RESET_SIM|SAFE35_ROOT|CLIENT_PY|SAFE_PY)' > $E2_ROOT/session.env
```

In a new terminal, re-run the exports, but replace the `E2_ROOT`, `E2_LOG` and `mkdir` lines with
`source <E2_ROOT>/session.env`.

### 1. Worktree, assets, tests

```bash
git -C $PROJECT_FS/robocasa fetch origin
test -e $E2_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $E2_REPO $E2_SHA
test "$(git -C $E2_REPO rev-parse HEAD)" = "$E2_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
mkdir -p $E2_REPO/robocasa/models/assets
cp -aln $PROJECT_FS/robocasa/robocasa/models/assets/. $E2_REPO/robocasa/models/assets/
cd $E2_REPO && CUDA_VISIBLE_DEVICES="" $SAFE_PY -W ignore -m unittest tests.test_rldx_e2 tests.test_rldx_e1_plan tests.test_rldx_e1_analysis tests.test_safe_atomic_collection 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only with `OK`.

### 2. Frozen inputs

```bash
cd $E2_REPO
sha256sum $E1_FREEZE | cut -c1-16                 # must print 917c06e6ab701f9f
sha256sum $RESET_SIM | cut -c1-16                 # must print 3dac815549293724
sha256sum $E1_ROOT/e1_result.json $E1_ROOT/plan.json | tee $E2_LOG/e1_inputs.sha256
git -C $RLDX_REPO rev-parse HEAD                  # must be ef05cd4a...
git -C $RLDX_REPO diff | cmp - $E2_REPO/patches/rldx1_e1_audited_worktree_ef05cd4.diff \
  && echo "RLDX diff = audited diff" || echo "STOP: RLDX working tree differs from the audited diff"
test -d $RLDX_SNAPSHOT && echo "policy snapshot 587e9ecd OK" || echo "STOP: snapshot missing"
```

The snapshot's content digest was checked for E1. Nothing has changed it since.

### 3. Plan

```bash
cd $E2_REPO
$CLIENT_PY -I scripts/rldx_e2/plan_e2.py --e1-plan $E1_ROOT/plan.json --e1-result $E1_ROOT/e1_result.json \
  --freeze-manifest $E1_FREEZE --reset-sim $RESET_SIM --seed-roots $STORAGE_BS/robocasa_rollouts/safe \
  --output $E2_ROOT 2>&1 | tee $E2_LOG/plan.log
```

Continue only with `PLAN OK`. It must print:
- P λ 0.0739 and SAFE-LSTM + P λ 0.0739 with u 7.9314;
- seed 1401;
- 2,000 rollouts.

### 4. Servers (GPUs 0–3, ports 20150–20153)

```bash
cd $RLDX_REPO
while IFS=$'\t' read -r set shard gpu port; do
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup uv run python -u rldx/eval/run_rldx_server.py \
    --model-path $RLDX_SNAPSHOT --embodiment-tag GENERAL_EMBODIMENT \
    --host 127.0.0.1 --port $port --use-sim-policy-wrapper > $E2_LOG/server_$port.log 2>&1 < /dev/null &
  echo "server $port GPU $gpu PID $!"
done < $E2_ROOT/shards.tsv
until [ "$(ss -ltn | grep -cE ':2015[0-3]\b')" -ge 4 ]; do sleep 10; done; echo "servers up"
ps -eo args | grep "[r]un_rldx_server.py" | grep -c "$RLDX_SNAPSHOT"    # must print 4
```

### 5. Smoke (gate 1; port 20150, seed 1102, two scenes; not part of E2)

```bash
cd $E2_REPO
mapfile -t FLAGS < $E2_ROOT/collector_flags_e2.txt
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 $CLIENT_PY -u scripts/rldx_e2/collect_e2.py \
  --output-dir $E2_ROOT/smoke --tasks TurnOnSinkFaucet "${FLAGS[@]}" --seed 1102 --num-rollouts 4 \
  --host 127.0.0.1 --port 20150 2>&1 | tail -n 3 | tee $E2_LOG/smoke.log
$CLIENT_PY -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir $E2_ROOT/smoke | tail -n 2
$CLIENT_PY -I scripts/rldx_e2/check_e2.py --plan $E2_ROOT/plan.json --smoke smoke=$E2_ROOT/smoke 2>&1 | tee -a $E2_LOG/smoke.log
$CLIENT_PY -I -c "import json,sys; [print(r['reset_index'], r['role'], r.get('restore_residual'), (r.get('first_obs_distance') or {}).get('image_max'), (r.get('first_obs_distance') or {}).get('state_max')) for r in map(json.loads, open(sys.argv[1]))]" $E2_ROOT/smoke/pairs.jsonl
```

Paste the output back. Continue only with `CHECK OK (smoke)`, residuals of 0 and small image
differences.

**Scoring past H.** E2's rollouts run to 1.25 H, longer than any rollout SAFE-LSTM has scored, so the
smoke rollouts are scored once (CPU, seed 0):

```bash
cd $E2_REPO
$CLIENT_PY -m robocasa.recovery.safe.export_to_official_safe --dataset-dir $E2_ROOT/smoke \
  --output-dir $E2_ROOT/smoke_export --summary-only 2>&1 | tail -n 2
CUDA_VISIBLE_DEVICES="" $SAFE_PY -u -m robocasa.recovery.safe.score_seen_checkpoint_external \
  --export-dir $E2_ROOT/smoke_export --safe-repo $PROJECT_FS/SAFE \
  --training-run-dir $STORAGE_BS/robocasa_checkpoints/safe/rldx1_safe20_natural_weighted_20260815_162530/final/lstm_seed0 \
  --output-dir $E2_ROOT/smoke_scores --group all --device cpu --allow-task-subset 2>&1 | tail -n 1
$CLIENT_PY -I -c "import json,sys; m={r['rollout_id']: r['valid_sequence_length'] for r in map(json.loads, open(sys.argv[1]))}; s={r['rollout_id']: r['num_inferences'] for r in map(json.loads, open(sys.argv[2]))}; print({k: (m[k], s.get(k)) for k in m}); print('LENGTHS OK' if all(s.get(k) == v for k, v in m.items()) else 'STOP: scores shorter than rollouts')" $E2_ROOT/smoke/manifest.jsonl $E2_ROOT/smoke_scores/scores.jsonl
```

Continue only with `LENGTHS OK`. If the export or scoring refuses the 4-rollout set, paste the error.

### 6. Collect (about 12–16 h on four GPUs; may span several sessions)

Each GPU runs its shard's tasks one at a time, each task in its own directory
(`collection/<task>`), so the end of a session loses only the tasks in progress. A task whose
earlier directory was interrupted is collected again with the next reserved seed (1411, 1421, …).
Its partial directory is kept under `collection_incomplete/` and never used.

Create the queue script once:

```bash
cat > $E2_ROOT/run_tasks.sh <<'SH'
#!/bin/bash
# One E2 collector per task, in order, on one GPU (docs/rldx_e2_plan.md, step 6). Usage: run_tasks.sh SHARD GPU PORT
set -u
source $E2_ROOT/session.env
shard=$1 gpu=$2 port=$3
mapfile -t FLAGS < $E2_ROOT/collector_flags_e2.txt
cd $E2_REPO
while read -r task; do
  [ -z "$task" ] && continue
  d=$E2_ROOT/collection/$task
  if [ -e "$d" ]; then echo "$(date +%F_%T) $task: directory exists, skipped"; continue; fi
  tries=$(ls -d $E2_ROOT/collection_incomplete/${task}_try* 2>/dev/null | wc -l)
  seed=$((1401 + 10 * tries))
  echo "$(date +%F_%T) $task: start, seed $seed"
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu $CLIENT_PY -u scripts/rldx_e2/collect_e2.py \
    --output-dir $d --tasks $task "${FLAGS[@]}" --seed $seed --host 127.0.0.1 --port $port \
    > $E2_LOG/collection_$task.log 2>&1 < /dev/null
  echo "$(date +%F_%T) $task: exit $?"
done < $E2_ROOT/tasks_e2_shard$shard.txt
SH
chmod +x $E2_ROOT/run_tasks.sh
```

**Start (or continue) the queues.** In a new session, first re-run steps 0 (with `source
<E2_ROOT>/session.env`) and 4 (servers), then move interrupted directories aside:

```bash
mkdir -p $E2_ROOT/collection $E2_ROOT/collection_incomplete
for d in $E2_ROOT/collection/*/; do
  [ -d "$d" ] || continue; t=$(basename $d); n=$(cat $d/manifest.jsonl 2>/dev/null | wc -l)
  if ! grep -q '"partial": false' $d/summary.json 2>/dev/null || [ "$n" -ne 100 ]; then
    k=$(ls -d $E2_ROOT/collection_incomplete/${t}_try* 2>/dev/null | wc -l)
    mv $d $E2_ROOT/collection_incomplete/${t}_try$((k + 1)); echo "moved aside: $t ($n rollouts)"
  fi
done
while IFS=$'\t' read -r set shard gpu port; do
  E2_ROOT=$E2_ROOT setsid nohup bash $E2_ROOT/run_tasks.sh $shard $gpu $port \
    >> $E2_LOG/queue_shard$shard.log 2>&1 < /dev/null &
  echo "queue shard$shard GPU $gpu started"
done < $E2_ROOT/shards.tsv
```

**Progress** (finished tasks have 100 rollouts):

```bash
for d in $E2_ROOT/collection/*/; do echo "$(basename $d): $(cat $d/manifest.jsonl 2>/dev/null | wc -l)/100"; done
tail -n 2 $E2_LOG/queue_shard*.log
```

**If a task exits with a nonzero code** (an error, not the session ending), paste its
`collection_<task>.log`. The queue moves on to the next task; the failed one is moved aside by the
sweep above and collected again in the next round.

### 7. Gate 2, when every collector has exited

```bash
pkill -f "run_rldx_server.py.*--port 2015[0-3]"
cd $E2_REPO
$CLIENT_PY -I scripts/rldx_e2/check_e2.py --plan $E2_ROOT/plan.json --output $E2_ROOT/pairs.json \
  $(for d in $E2_ROOT/collection/*/; do echo "e2=${d%/}"; done) 2>&1 | tee $E2_LOG/check.log
```

Paste `check.log` back. Continue only with `CHECK OK`.

### 7a. Instruction fix and rerun (deviation, decided 2026-11-07 before any outcome was read)

**What gate 2 found.** On the first full collection (`rldx1_e2_20261106_143309`, code `0810b17f`),
30 of 1,000 pairs were excluded for "instruction differs", more than 2%, so E2 stopped. The printed
examples were all OpenDrawer.
- **Restore:** the restore itself was exact: residual 0, largest first-image difference 0.022.
- **Cause:** OpenDrawer's instruction is "Open the {side} drawer.", with the side drawn when the robot
  is placed. The gym wrapper's `reset_to` reloads the model, which places the robot again and draws
  the side again.
- **Effect:** the robot pose and the target drawer come back from the saved state, so they are exact.
  Only the instruction word changed, so on those retries the policy was told to open the wrong drawer.

**Fix (`collect_e2.py`).** After `reset_to`, the retry is given the first attempt's own instruction
(`ep_meta["lang"]` saved with its state) through the wrapper's `set_task_description`.
- The override lasts the whole retry, and the next ordinary reset clears it.
- Each retry row records `instruction_restored`.
- Tests: `test_retry_keeps_the_first_attempts_instruction` and
  `test_without_the_override_the_check_catches_it`.

**Rerun rule (fixed before any E2 outcome or score was read).** Every task with at least one
"instruction differs" pair is collected again, whole, with the fix and the first reserved rerun seed
1411.
- **Kept, never used:** its first directory goes to `collection_excluded/<task>_instruction`.
- **Unchanged:** the other tasks, gate 2, the analysis and every threshold.
- **Not taken:** dropping the affected pairs would select scenes (a mismatch can only happen where
  the robot can start on either side), and would break the 2% rule.

**Commands.** `E2F_SHA` is the commit containing this step.

```bash
# 0. Environment as in step 0 (source session.env), then the fixed worktree.
export E2F_SHA=<E2F_SHA>
export E2_REPO=$PROJECT_FS/robocasa_rldx_e2_${E2F_SHA:0:8}
git -C $PROJECT_FS/robocasa fetch origin claude/rldx-e1
test -e $E2_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $E2_REPO $E2F_SHA
test "$(git -C $E2_REPO rev-parse HEAD)" = "$E2F_SHA" && echo "HEAD OK" || echo "STOP: wrong HEAD"
mkdir -p $E2_REPO/robocasa/models/assets
cp -aln $PROJECT_FS/robocasa/robocasa/models/assets/. $E2_REPO/robocasa/models/assets/
echo "export E2_REPO=$E2_REPO E2F_SHA=$E2F_SHA" > $E2_ROOT/session_fix.env
cd $E2_REPO && CUDA_VISIBLE_DEVICES="" $SAFE_PY -W ignore -m unittest tests.test_rldx_e2 2>&1 | grep -E "^(Ran|OK|FAILED)"

# 1. Exclusions by task and reason (reads only pairs.jsonl: no outcome).
$CLIENT_PY -I scripts/rldx_e2/check_e2.py --plan $E2_ROOT/plan.json --breakdown \
  $(for d in $E2_ROOT/collection/*/; do echo "e2=${d%/}"; done) | tee $E2_LOG/breakdown_before_fix.log
grep '^RERUN' $E2_LOG/breakdown_before_fix.log | cut -d' ' -f2 > $E2_ROOT/rerun_tasks.txt; cat $E2_ROOT/rerun_tasks.txt

# 2. Move each such task aside (kept, never used).
mkdir -p $E2_ROOT/collection_excluded
while read -r t; do mv $E2_ROOT/collection/$t $E2_ROOT/collection_excluded/${t}_instruction && echo "moved $t"; done < $E2_ROOT/rerun_tasks.txt

# 3. Servers (step 4), then one collector per task, round-robin over the four GPUs.
cd $RLDX_REPO
while IFS=$'\t' read -r set shard gpu port; do
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup uv run python -u rldx/eval/run_rldx_server.py \
    --model-path $RLDX_SNAPSHOT --embodiment-tag GENERAL_EMBODIMENT \
    --host 127.0.0.1 --port $port --use-sim-policy-wrapper > $E2_LOG/server_rerun_$port.log 2>&1 < /dev/null &
done < $E2_ROOT/shards.tsv
until [ "$(ss -ltn | grep -cE ':2015[0-3]\b')" -ge 4 ]; do sleep 10; done; echo "servers up"
cd $E2_REPO
mapfile -t FLAGS < $E2_ROOT/collector_flags_e2.txt
i=0
while read -r t; do
  gpu=$((i % 4)); port=$((20150 + i % 4)); i=$((i + 1))
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu setsid nohup $CLIENT_PY -u scripts/rldx_e2/collect_e2.py \
    --output-dir $E2_ROOT/collection/$t --tasks $t "${FLAGS[@]}" --seed 1411 --host 127.0.0.1 --port $port \
    > $E2_LOG/collection_${t}_rerun.log 2>&1 < /dev/null &
  echo "$t on GPU $gpu, seed 1411"
done < $E2_ROOT/rerun_tasks.txt
```

The script assigns more than one task to a GPU only when more than four tasks need a rerun.
When every rerun directory shows 100 rollouts, run step 7 from `$E2_REPO` (this worktree), then
steps 8–9 unchanged.

### 8. Export and scores (SAFE-LSTM on one GPU; Hide-and-Seek on CPU)

```bash
cd $E2_REPO
export SAFE_REPO=$PROJECT_FS/SAFE
export RLDX_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_safe20_natural_weighted_20260815_162530
export RLDX_CAL=$RLDX_ROOT/calibration/lstm
export FREEZE_SCORES=$(dirname $E1_FREEZE)/scores_run2
$CLIENT_PY -m robocasa.recovery.safe.merge_atomic_datasets --source-dirs $E2_ROOT/collection/*/ \
  --output-dir $E2_ROOT/merged/e2 2>&1 | tail -n 2
$CLIENT_PY -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir $E2_ROOT/merged/e2 | tail -n 1
$CLIENT_PY -m robocasa.recovery.safe.export_to_official_safe --dataset-dir $E2_ROOT/merged/e2 \
  --output-dir $E2_ROOT/safe_export/e2 --summary-only 2>&1 | tail -n 3
for s in 0 1 2; do
  for x in recorded:$RLDX_EXPORT e2:$E2_ROOT/safe_export/e2; do
    CUDA_VISIBLE_DEVICES=0 $SAFE_PY -u -m robocasa.recovery.safe.score_seen_checkpoint_external \
      --export-dir ${x#*:} --safe-repo $SAFE_REPO --training-run-dir $RLDX_ROOT/final/lstm_seed$s \
      --output-dir $E2_ROOT/safe_scores/${x%%:*}/seed$s --group all --device cuda 2>&1 | tail -n 1
  done
done | tee $E2_LOG/safe_external.log
$SAFE_PY -I scripts/rldx_e1/score_e1.py safe --set-names e2 --freeze-manifest $E1_FREEZE --calibration-dir $RLDX_CAL \
  $(for s in 0 1 2; do echo --recorded-scores $s $E2_ROOT/safe_scores/recorded/seed$s; done) \
  $(for s in 0 1 2; do echo --e1-scores e2 $s $E2_ROOT/safe_scores/e2/seed$s; done) \
  --output-dir $E2_ROOT/scores 2>&1 | tee $E2_LOG/score_safe.log
unset OPENBLAS_NUM_THREADS
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=$E2_REPO $SAFE_PY -I scripts/rldx_e1/score_e1.py hs \
  --set-names e2 --freeze-manifest $E1_FREEZE --freeze-scores-dir $FREEZE_SCORES --recorded-dataset $RLDX_RAW \
  --output-dir $E2_ROOT/scores $(for d in $E2_ROOT/collection/*/; do echo "e2=${d%/}"; done) 2>&1 | tee $E2_LOG/score_hs.log
```

Each scoring command prints its reproduction checks and `SCORING OK`, or stops before writing
anything.

### 9. Analysis (CPU, minutes)

```bash
cd $E2_REPO
$CLIENT_PY -I scripts/rldx_e2/analyze_e2.py --plan $E2_ROOT/plan.json --pairs $E2_ROOT/pairs.json \
  --scores-dir $E2_ROOT/scores --output $E2_ROOT/e2_result.json 2>&1 | tee $E2_LOG/analyze.log
```

Paste `analyze.log` back and copy `e2_result.json` to the Mac.
