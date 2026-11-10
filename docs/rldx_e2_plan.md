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

## Result (2026-11-09)

**Run.**
- Root `rldx1_e2_20261106_143309`, collected with code `0810b17f`.
- OpenDrawer recollected with code `6a3a40f3` and seed 1411 (step 7a). Its first directory is in
  `collection_excluded/OpenDrawer_instruction`. No other task was rerun.
- Steps 7–9 ran from `6a3a40f3`.

**Gates.**
- **Before the fix:** the breakdown showed OpenDrawer with 30 of 50 pairs "instruction differs", and no
  other exclusion.
- **After it:** `CHECK OK`, with 1,000 of 1,000 pairs usable, restore residual 0 and the largest
  first-image difference 0.022. All 50 OpenDrawer retries record `instruction_restored`.
- **Scoring:** both steps end in `SCORING OK`. SAFE-LSTM reproduces its normalization and the
  recorded scores with difference 0. Hide-and-Seek reproduces the recorded calibration (60) and test
  (200) scores with difference 0. 2,000 rollouts were scored.

**Hashes.** `e2_result.json` `0df32ba9…`, `pairs.json` `2ecf49a1…`, `plan.json` `0121ce9e…`. The
copies, with the logs, are in the paper worktree:
`reports/paper_time_aware_fusion/rss2027_stage2/evidence/e2/`.

**Success within 1.25 H_k** (1,000 scenes; 95% CIs from scenes resampled within task):

| Rule | Success | Resets | Unnecessary | Lost a success |
|---|---|---:|---:|---:|
| No reset | 0.508 [0.481, 0.533] | 0 | 0 | 0 |
| P | 0.500 [0.475, 0.525] | 540 | 48 | 45 |
| SAFE-LSTM + P (primary) | 0.496 [0.471, 0.521] | 547 | 55 | 49 |
| Hide-and-Seek + P (descriptive) | 0.504 [0.478, 0.529] | 539 | 47 | 42 |
| Success-tuned timer (R_opt) | 0.496 [0.471, 0.521] | 541 | 49 | 42 |

**Paired differences** ("only" = scenes won by one rule and lost by the other):

| Contrast | Δ | 95% CI | Only first / only second |
|---|---:|---|---|
| **SAFE-LSTM + P − P (primary)** | **−0.004** | **[−0.008, −0.001]** | 0 / 4 |
| P − none | −0.008 | [−0.026, +0.010] | 37 / 45 |
| SAFE-LSTM + P − none | −0.012 | [−0.030, +0.006] | 37 / 49 |
| R_opt − P | −0.004 | [−0.012, +0.004] | 8 / 12 |
| SAFE-LSTM + P − R_opt | 0.000 | [−0.009, +0.008] | 12 / 12 |
| Hide-and-Seek + P − P (descriptive) | +0.004 | [−0.001, +0.010] | 6 / 2 |

- **Main comparison.** SAFE-LSTM + P and P give the same outcome on 996 scenes. On the other 4,
  SAFE-LSTM's earlier reset lost a success.
- **Latency.** Charging a full inference (8 steps) per learned-monitor reset changes nothing
  (SAFE-LSTM + P −0.004, Hide-and-Seek + P +0.004).
- **By type.**
  - Atomic (700 scenes): none 0.510, P 0.521, SAFE-LSTM + P 0.516, Hide-and-Seek + P 0.526, R_opt 0.519.
  - Composite (300 scenes): none 0.503, P 0.450, SAFE-LSTM + P 0.450, Hide-and-Seek + P 0.453, R_opt 0.443.
- **Restore check.**
  - First attempts succeed in 0.508 of scenes.
  - The full-budget retry succeeds in 0.628 of scenes whose first attempt succeeded, and in 0.148 of
    those whose first attempt failed.

**Against the plan.**
- *SAFE − P ≈ +0.006 (half-width about 0.02):* observed −0.004. That is inside the predicted range,
  with the opposite sign.
  - The CI is narrower than sized because the rules share the retry and differ on only 4 scenes.
- *Retry success r = 0.245 (XR-1 pilot):* observed 0.148 after a failure, about 40% lower. This is
  why resets do not pay.
  - On composite tasks, resetting at P loses 0.053 against not resetting.
- *R_opt matches the fused monitors at 1.25 H_k:* yes. SAFE-LSTM + P − R_opt = 0.000.

**Conclusion.** On this policy and budget, acting on an alarm by resetting and retrying does not
raise success.
- The timer's resets already cost as many successes as they recover (37 won, 45 lost against no
  reset).
- SAFE-LSTM's extra warning changes the outcome of 4 scenes in 1,000, all for the worse.
- The limiting factor is the retry's success after a failure (15%), not the time left when the
  alarm comes.

## Follow-up (written 2026-11-09, after the result; descriptive)

`scripts/rldx_e2/followup_e2.py`, with tests in `tests/test_rldx_e2.py::FollowupTest`. CPU, minutes.
- **Inputs.** E2's own files. No rollout is new.
- **Alarms.** Every rule's alarm comes from `analyze_e2.alarms`, so its timing is exactly the one
  behind the result.
- **Gate F0.** The script first recomputes E2's published success rates and stops if any differs.

- **F1. Ceiling of reset-and-retry.** A perfect detector resets exactly the first attempts that would
  fail, and never a success. It does so at P's alarm (oracle@P), at SAFE-LSTM + P's (oracle@SAFE,
  with its one step of latency), or at the first inference (oracle@0, where the retry gets the whole
  budget). Each is reported against no reset and against P, with 95% CIs.
  - This bounds what any detector could gain from this recovery action at each timing.
- **F2. Is failure a property of the scene?**
  - Per task and type: first-attempt success, and full-budget retry success after a failed and
    after a successful first attempt.
  - The retry success after a failure that independence within the task would predict.
  - The Mantel–Haenszel odds ratio (strata = tasks) of retry success given first-attempt success,
    with a bootstrap CI. OR ≫ 1 means some scenes are hard for this policy.
- **F3. Stopping at the alarm instead of restarting** (handing over, say). For P, SAFE-LSTM + P and
  Hide-and-Seek + P:
  - robot time saved on failed first attempts, (B − t_a) control steps at 20 Hz;
  - successes stopped, meaning those whose alarm came before the success;
  - differences against P per scene, with 95% CIs.

**Expectations (written before running).**
- **F1.**
  - oracle@0 is about 0.508 + 0.492 × 0.148 ≈ 0.58, i.e. +0.07 over no reset.
  - oracle@P gains less, about +0.02 to +0.04, because its retry has only the budget left after the
    timer's alarm.
  - oracle@SAFE is within 0.01 of oracle@P.
  - So even a perfect detector at the timer's timing gains a few points at most.
- **F2.** OR well above 1, likely 5–15, with retry success after a failure (0.148) well below the
  independence prediction. Composite tasks are more dependent than atomic ones.
- **F3.**
  - P saves roughly a third of each failed attempt's length.
  - SAFE-LSTM + P saves about 1 s more per failure (E1: +1.4 s) and stops about 7 more successes,
    from 55 against 48 unnecessary resets in E2.

**Command** (from the step 7a worktree, after step 9):

```bash
cd $E2_REPO
$CLIENT_PY -I scripts/rldx_e2/followup_e2.py --plan $E2_ROOT/plan.json --pairs $E2_ROOT/pairs.json \
  --scores-dir $E2_ROOT/scores --result $E2_ROOT/e2_result.json --output $E2_ROOT/e2_followup.json \
  2>&1 | tee $E2_LOG/followup.log
sha256sum $E2_ROOT/e2_followup.json
```

### Follow-up result (2026-11-09) and a concern about the retry

Run from `5cc80fad`; `e2_followup.json` `223c8f4a…`. F0 OK.
- **F1.**
  - oracle@P 0.545: +0.037 [0.026, 0.049] over no reset, +0.045 [0.033, 0.058] over P.
  - oracle@SAFE is identical to oracle@P.
  - oracle@0 0.581: +0.073 [0.058, 0.089] over no reset.
- **F2.** Mantel–Haenszel OR 4.25 [3.05, 6.01]. Retry success after a failure is 0.148; independence
  within the task predicts 0.232.
- **F3.**
  - P saves 23.0 s per failed attempt (43% of its length) and stops 48 successes.
  - SAFE-LSTM + P saves 0.41 s more per scene [0.33, 0.49] and stops 0.007 more successes per scene
    [0.002, 0.013].
  - Hide-and-Seek + P saves 0.23 s more [0.10, 0.37] and stops −0.001 [−0.006, 0.004].

**Concern.** The full-budget retry starts from the restored first-attempt state with the same policy
and budget, so its success should match the first attempt's. Instead it is 0.392 against 0.508
(composite 0.273 against 0.503). On five tasks, no retry succeeds after a successful first attempt:

| Task | Retries succeeding after a successful first attempt |
|---|---|
| StackBowlsCabinet | 0 of 33 |
| NavigateKitchen | 0 of 22 |
| LoadDishwasher | 0 of 17 |
| PreSoakPan | 0 of 10 |
| OpenCabinet | 0 of 8 |

The likely cause is the OpenDrawer mechanism with an unchanged instruction. `reset_to` reloads the
model, which can draw some of the task's Python-side choices again (targets, fixture references, the
robot's base anchor), so the success check or the controller no longer matches the restored scene.
**Until the restore check below passes, the E2 result and the follow-up are provisional.**

## Restore check (written 2026-11-09, before running)

`scripts/rldx_e2/restore_check.py`, with tests in `tests/test_rldx_e2.py::RestoreCheckTest`. No
policy is involved.
- **Setup.** Each task's environment is built exactly as E2's collector built it: `make_env` with the
  plan's split and seed, `reset()` without a seed, and `collect_e2`'s PairedReset installed.
- **Per scene.** The first attempt and the retry each record the task's Python-side attributes and
  ep_meta, then apply the same 60 random actions (seeded, 0.3 of the action range). They record the
  simulator state, the success flag and `_check_success()` at every step.
- **Verdict.** A task is SAME only if no attribute or ep_meta entry differs, the instructions match,
  the states never diverge (1e-6), and the success checks agree at every step.
- **Expectations.**
  - The five tasks above are DIFFERENT, with a redrawn target or fixture reference among the
    attribute differences, or a state divergence that points to the controller.
  - Tasks whose retries matched their first attempts (e.g. RinseSinkBasin, TurnOnMicrowave) are SAME.

```bash
cd $E2_REPO     # a worktree at the commit containing this section, assets hard-linked
MUJOCO_GL=egl PYTHONPATH=$E2_REPO $CLIENT_PY -u scripts/rldx_e2/restore_check.py --plan $E2_ROOT/plan.json \
  --output $E2_ROOT/restore_check.json --render 2>&1 | grep -v -i warn | tee $E2_LOG/restore_check.log
```

The output decides the next step:
- **Tasks found DIFFERENT:** a targeted restore fix with a test, then the affected tasks are
  recollected (as in step 7a). The analysis is redone only after the check passes on every task.
- **Every task SAME:** the cause lies outside the environment (policy server, timing), and that is
  examined next.

### Restore check, first runs (2026-11-10)

- **`b247038d`:** after an action-space fix (the first run errored on RoboCasa's Dict action space),
  every task was DIFFERENT.
  - The restored state equals the first attempt's at step 0, but under the same actions the two
    diverge from step 1 in all 20 tasks (about 0.066 on the calm tasks, up to 4.3).
  - ep_meta `object_cfgs` differ in 13 tasks; the printout was truncated.
  - The attribute snapshot read the gym wrapper, so it showed only the instruction override.
- **Version 2:**
  - reads the RoboCasa task itself;
  - diffs every leaf of ep_meta and of each robot's composite and part controllers;
  - adds a control that restores the same state a second time. If the two restores agree with each
    other but not with the first attempt, the restore differs. If they disagree with each other, the
    simulation itself is not repeatable.

### Restore check, version 2 (2026-11-10, `c9c94183`; StackBowlsCabinet, NavigateKitchen, RinseSinkBasin, TurnOnMicrowave)

- **The simulation is repeatable.** Restoring the same state twice gives identical trajectories and
  identical controllers in every scene.
- **The arm and torso controllers' goals are not restored.** After `reset_to`, `goal_ori`, `goal_pos`
  and the torso goal are the same constant in every task and scene: the default pose that the
  rebuild inside `reset_to` set before the state was loaded. A fresh build sets them to the actual
  pose. So under the same actions the retry moves differently from the first step (about 0.066).
- **Task attributes are drawn again.** NavigateKitchen's `target_pos` (what `_check_success`
  measures against) changed by up to 2 m with the same target fixture and instruction, so its
  retries could not succeed. That matches 0 of 22.
- **StackBowlsCabinet** showed no attribute difference beyond the controllers. Its success check
  uses fixture interior sites; the rebuild below makes the question moot.

**Conclusion.** `reset_to` reloads the model through a fresh reset, so every Python-side choice made
while building (targets, sides, base anchors, controller goals) can differ from the restored scene.
The E2 retries are not valid second attempts. **E2's first collection (`rldx1_e2_20261106_143309`)
is kept as a record but not used.**

## Retry by rebuilding (2026-11-10)

`collect_e2.PairedReset` now defaults to mode `rebuild`.
- **How it works.** Before each first attempt it saves the state of every random generator the scene
  is built from: the environment's `rng`, numpy's global state and Python's `random`. For the retry
  it winds them back and calls the same normal reset. The scene, task attributes and controllers are
  then the first attempt's by construction.
- **How it is verified.** check_e2 requires the rebuilt state and model XML to match the first
  attempt's exactly (residual 0).
- **Old mode.** Mode `restore` remains for the record.
- **Tests:** `RebuildModeTest`, plus the restore check in both modes.

**Next, in order.**
1. **The restore check in rebuild mode, on all 20 tasks** (`--mode rebuild`, the default). Every
   task must be SAME.
2. **A new E2 collection** into a new root, with everything else unchanged: the same plan, rules,
   tasks, 50 scenes per task, budget and analysis. The seed must be unused, so 1501 rather than
   1401: the first collection's scenes have been seen.
3. **Steps 7–9 and the follow-up** on that collection.

### Restore check in rebuild mode (2026-11-10, `84f8b291`, all 20 tasks)

18 of 20 tasks SAME.
- **RinseSinkBasin, scene 0:** a sink reset-region side flipped. Two rebuilds also differed from
  each other, so the build draws randomness that was not wound back.
- **TurnOnElectricKettle, scene 1:** the kettle, which anchors the robot's start, was placed 2 m
  elsewhere. The two rebuilds agreed with each other by chance.
- **Common cause.** RoboCasa falls back to an unseeded `np.random.default_rng()` where no generator
  is passed: `Fixture`, `fixture_stack`, `others`, `windows`, object sampling and placement samplers.
  These fallbacks draw from the operating system.

**Fix (`seeded_fallbacks`).** During each build, and its rebuild, an unseeded `default_rng()` is
seeded from a sequence derived from the scene's own generator state, and numpy's function is put
back afterwards.
- The first build and the rebuild draw the same values, and different scenes still differ.
- The check now also compares the scene model XML.
- Tests: `FallbackTest`. The check is to be run again on all 20 tasks; all must be SAME.

### Restore check in rebuild mode, run 2 (2026-11-10, `584f8cee`, 3 scenes per task)

12 of 20 tasks SAME.
- **Two rebuilds disagreed** in PreSoakPan, ScrubCuttingBoard, NavigateKitchen and RinseSinkBasin
  scene 1; PreSoakPan even got another layout. So generators that persist across resets were not
  wound back: only the env's `rng` was.
- **The first build differed from rebuilds that agreed** in RinseSinkBasin scene 0, WashLettuce and
  TurnOnElectricKettle scene 2.
- **CloseFridge (all scenes) and CloseToasterOvenDoor:** identical states, but different model XML.

**Change.** Every numpy Generator and RandomState reachable from the environment, or held at module
level in RoboCasa, robosuite or gym, is now wound back (`find_generators`). When the model XML differs,
the check prints the changed lines. Tests: `PersistentGeneratorTest`.

### Restore check in rebuild mode, run 3 (2026-11-10, `88c96e80`), and a change of approach

12 of 20 tasks SAME, even with every reachable generator wound back. Rebuilds of the same scene still
disagreed with each other in several tasks, including a different layout. **An exact rebuild is not
reachable in RoboCasa**, so the rebuild mode is set aside.
- CloseFridge and CloseToasterOvenDoor had identical states, and their model XML differed only in mesh
  attributes (`content_type`, ordering).

**A second defect in E2's first collection.** `reset_to` stores the scene's metadata on the task
(`_ep_meta`). Nothing clears it, and RoboCasa reuses it to build the next scene: layout, style,
fixtures, objects and robot start. robomimic's wrapper clears it on every reset; the gym wrapper E2
used does not. So in E2's first collection the scenes after a retry largely repeated the previous one.
- **Evidence:** first-attempt success was more than 3 SE from the task's expected rate in 10 of 20
  tasks, at the extremes, e.g. OpenStandMixerHead 0.06 against 0.80 and TurnOnMicrowave 0.88 against
  0.30.

## Retry by a fixed restore (2026-11-10; default mode `restore`)

The retry restores the first attempt's XML and state through `reset_to` (exact physical state). Then:
1. Every plain value (numbers, strings, arrays) of the task, the robots, the composite and part
   controllers that differs from the first attempt's is set back (`copy_back`). This fixes the
   controller goals, NavigateKitchen's `target_pos` and OpenDrawer's side.
2. The stored metadata is cleared (`clear_ep_meta`), so the next scene is built fresh.
3. The retry gets the first attempt's instruction.

Each retry row records `values_restored` and `ep_meta_cleared`. check_e2 now stops if any scene starts
from the same state as an earlier scene of its task, and `--breakdown` counts repeated start states.
The restore check now reports scenes repeating the previous one. Mode `restore_legacy` is E2's first
collection.

**Next.**
1. `check_e2 --breakdown` on E2's first collection, to count its repeated scenes (for the record).
2. The restore check in mode `restore` on all 20 tasks with 3 scenes. Every task must be SAME, with no
   repeated scene.
3. The new collection (seed 1501, new root), then steps 7–9 and the follow-up.

### Restore check, fixed restore, run 1 (2026-11-10, `d36e935f`)

- **Scene freshness fixed:** no scene repeats the previous one in any task.
- **Still 0 of 20 SAME.** After the restore, the controller goals were still the default constant.
  `reset_to` rebuilds the robots and controllers as new objects, so the values were being copied onto
  the old ones.
- **Change:** `copy_back` now resolves the task, each robot, its composite controller and its named
  part controllers by position after the restore (test: `RebuiltControllerTest`).
- **Check treats two harmless differences as equal:**
  - the model XML that MuJoCo re-serializes (float rounding at the 6th digit, an optional mesh
    `content_type` attribute), compared at 4 decimals;
  - a fixture reference stored as `(fixture, False)` on a fresh build and as the fixture after a
    restore.
