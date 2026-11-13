# E1b: a second test of H3 for Hide-and-Seek on new RLDX-1 rollouts (plan and runbook)

**Status.** Written 2026-11-12, before any E1b rollout exists. Branch `claude/rldx-e1b` (from
`claude/rldx-e1`). The author decided to run it after E1 and R15.

## Why

- **E1** (`docs/rldx_e1_plan.md`) tested H3 with every path fixed on the validation set, whose
  calibration set has only 60 successes.
  - SAFE-LSTM + P passed.
  - Hide-and-Seek + P did not: gains 0.032 and 0.052, both CIs include zero.
- **R15** (branch `claude/rldx-test-table`): with its path re-chosen on E1's own calibration
  rollouts, the Hide-and-Seek ensemble gained 0.094 with 23 false alarms (P: 24).
- **The question:** was Hide-and-Seek's failure the score or the path fixed on 60 successes?
  R15 cannot answer it, because it chose the path on the rollouts it evaluates.
- **E1b:** Hide-and-Seek + P with its path fixed on E1's 1,600 rollouts, tested on new rollouts
  with E1's rule.

## Decision after the first freeze (2026-11-12, before any E1b rollout)

The first freeze, at code `6322c89a` (`e1b_freeze.json`, sha256 `1e5ab16d…`), chose new paths
for both detectors. Its preview on E1's data (paths chosen on E1's calibration set, the rule
applied to E1's test set) gave:

| Detector | Path | False alarms (P: 24) | Gain, 97.5% CI | Matched gain, 97.5% CI | Preview |
|---|---|---|---|---|---|
| Hide-and-Seek | u 0.2301, s 5 | 23 | 0.094 [0.040, 0.152] | 0.101 [0.044, 0.147] | passes |
| SAFE-LSTM | u 2.9249, s 20 | 32 | 0.053 [−0.004, 0.100] | 0.022 [−0.024, 0.064] | fails |

The new path helps Hide-and-Seek and hurts SAFE-LSTM, which passed E1 with its own path.

**Author's decision:** E1b tests Hide-and-Seek + P alone. SAFE-LSTM keeps its E1 result and is
not scored on E1b's rollouts.

This decision used E1's data only. The rule is unchanged, including the 97.5% intervals E1 used
for two detectors, which is conservative for one. The first freeze is kept and not used; the
second freeze (`e1b_freeze_hs.json`) chooses Hide-and-Seek's path only, by the same rule, on the
same data.

## Fixed before collection

Everything is E1's except the tested detector, its path and the null fusions' data.

- **Kept from E1:**
  - Models: Hide-and-Seek, seeds 0–2, from the E1 freeze manifest
    (`e2_freeze_20261029_205743/e2_freeze_manifest.json`, sha256 `917c06e6ab701f9f…`).
  - H_k and ρ_k, the policy snapshot (`587e9ecd`) and the audited RLDX-1 source.
  - The collector settings: 20 successes and 20 failures per task in each set.
  - The rule:
    - (a) the 97.5% CI of the gain over P lies above zero;
    - (b) the same at matched false alarms;
    - (c) the exact binomial bound on test false alarms;
    - (d) the gain beats 38 of 39 random and 38 of 39 resampled fusions.
  - 2,000 paired bootstrap replicates and 97.5% intervals (E1's level).
- **New path:** chosen by `scripts/rldx_e1/e1b.py freeze` with `confirm_lead.fit_fast` on all
  1,600 E1 rollouts (calibration and test sets). Each candidate's margin is set on E1's 800
  successes; E1's 800 failures choose the path. Only (u, s) is frozen. The margin is set on
  E1b's new calibration successes, so Proposition 1 applies.
- **Null fusions:** paths chosen on null scores of the same 1,600 rollouts. The resampled null
  draws from E1's normalized SAFE-LSTM values (E1 used the validation set for both).
- **New sets:**
  - calibration base seed 1601, test base seed 1701 (smoke: 1105);
  - the planner refuses any seed used by an earlier RLDX-1 collection.
- **Analysis seed:** 20261112.
- **Design check (not a result):** `freeze --design-check` chooses the path on E1's calibration
  set alone and applies the rule to E1's test set. It previews the test out of sample and is
  reported as a preview.

## Reporting (fixed now)

- E1's result stays as it is. E1b is reported as a second, separately planned test, whatever
  its outcome.
- If E1b passes, the paper can say that a test with the path fixed on 1,600 rollouts confirms
  Hide-and-Seek's early warning, and that the first test, with the path fixed on 60 successes,
  did not.
- If E1b fails, R15's 0.094 is a property of E1's rollouts, and the paper says so.
- The first test's SAFE-LSTM result is part of E1 and stays reported with it.

## Gates

1. **Freeze:** `FREEZE OK`, and the freeze file is copied to the Mac before step 4.
2. **Smoke and collection:** as in E1 (`CHECK OK (smoke)`, `CHECK OK`, every shard `VALID`).
3. **Scoring:** as in E1. Each monitor first reproduces its recorded scores exactly.
4. **Analysis:**
   - **R:** E1's gate on the recorded rollouts.
   - **F:** E1's score files are the ones the freeze read, the freeze tests Hide-and-Seek only,
     and the path recomputed from them equals the frozen one.
   - **S:** the plan used E1b's seeds, and no E1b rollout is an E1 rollout.

## Code

- `scripts/rldx_e1/e1b.py` (`freeze`, `analyze`), with tests in `tests/test_rldx_e1b.py` (4
  tests).
  - It runs `analyze_e1.py`'s point estimate, bootstrap, null references and decision for
    Hide-and-Seek alone.
  - A test checks that these equal `analyze_e1.analyze`'s Hide-and-Seek and P numbers on the
    same data.
- `plan_e1.py`, `check_e1_collection.py` and `score_e1.py hs` are reused unchanged.

## Cost

- **Collection:** the same as E1, about 2,500 rollouts. E1 took about a day on four GPUs. On two
  GPUs, plan with `--shards-per-set 1` (about twice as long).
- **Scoring:** Hide-and-Seek only, on CPU, as in E1 (no GPU, no SAFE export).

## Runbook

Run from a pinned worktree at `E1B_SHA`, the full SHA of the pushed commit that contains this
plan. Never pull or check out in the shared checkout `$PROJECT_FS/robocasa`.

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
export RLDX_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_safe20_natural_weighted_20260815_162530
export E1_FREEZE=$STORAGE_BS/robocasa_checkpoints/safe/e2_freeze_20261029_205743/e2_freeze_manifest.json
export RLDX_SNAPSHOT=$HF_HOME/hub/models--RLWRLD--RLDX-1-FT-RC365/snapshots/587e9ecdcc5e7184fcc17f58713908edff5af041
export E1_ROOT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_e1_20261105_210723
export E1B_SHA=<full SHA of the commit containing this plan>
export E1B_REPO=$PROJECT_FS/robocasa_rldx_e1b_${E1B_SHA:0:8}
export E1B_ROOT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_e1b_$(date +%Y%m%d_%H%M%S)
export E1B_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $E1B_ROOT)
mkdir -p $E1B_ROOT $E1B_LOG
env | grep -E '^(E1_|E1B_|RLDX_|SAFE35_ROOT|CLIENT_PY|SAFE_PY)' | sed 's/^/export /' > $E1B_ROOT/session.env
echo "E1B_ROOT=$E1B_ROOT"; test -f $RLDX_RAW/manifest.jsonl && echo "raw manifest OK"
```

In a new terminal, run the first four lines and then `source <E1B_ROOT>/session.env`.

### 1. Worktree, assets, tests

```bash
git -C $PROJECT_FS/robocasa fetch origin claude/rldx-e1b
test -e $E1B_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $E1B_REPO $E1B_SHA
test "$(git -C $E1B_REPO rev-parse HEAD)" = "$E1B_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
mkdir -p $E1B_REPO/robocasa/models/assets
cp -aln $PROJECT_FS/robocasa/robocasa/models/assets/. $E1B_REPO/robocasa/models/assets/
cd $E1B_REPO && CUDA_VISIBLE_DEVICES="" $SAFE_PY -W ignore -m unittest tests.test_rldx_e1b tests.test_rldx_e1_plan \
  tests.test_rldx_e1_analysis tests.test_e2_frozen_rule tests.test_xr1_fixed_path tests.test_rldx_continuation_pilot \
  tests.test_safe_atomic_collection 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only with `OK`.

### 2. Frozen inputs (record, then check)

```bash
cd $E1B_REPO
sha256sum $E1_FREEZE | tee $E1B_LOG/freeze_sha256.txt | cut -c1-16    # must print 917c06e6ab701f9f
sha256sum $E1_ROOT/scores/*_predictions.jsonl $E1_ROOT/scores/*_scoring.json $E1_ROOT/e1_result.json \
  | tee $E1B_LOG/e1_inputs.sha256
test -d $RLDX_SNAPSHOT && echo "policy snapshot 587e9ecd OK" || echo "STOP: snapshot missing"
git -C $RLDX_REPO rev-parse HEAD | tee $E1B_LOG/rldx_repo_head.txt      # must be ef05cd4a...
git -C $RLDX_REPO diff > $E1B_LOG/rldx_repo_diff.patch
cmp $E1B_LOG/rldx_repo_diff.patch $E1B_REPO/patches/rldx1_e1_audited_worktree_ef05cd4.diff \
  && echo "RLDX diff = audited diff" || echo "STOP: RLDX working tree differs from the audited diff"
```

Continue only if the freeze hash is `917c06e6ab701f9f`, the RLDX commit is `ef05cd4a…` and
`RLDX diff = audited diff` prints. E1's snapshot digest check does not need repeating if the
snapshot directory is unchanged (`ls -l --time-style=+%F $RLDX_SNAPSHOT`).

### 3. Freeze the paths (CPU, minutes)

```bash
cd $E1B_REPO
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $SAFE_PY -I -u scripts/rldx_e1/e1b.py freeze --freeze-manifest $E1_FREEZE \
  --e1-scores-dir $E1_ROOT/scores --design-check --output $E1B_ROOT/e1b_freeze_hs.json 2>&1 | tee $E1B_LOG/freeze_hs.log
```

It prints the frozen path (and E1's for comparison), the design check and `FREEZE OK`.
Paste `freeze_hs.log` back and copy `e1b_freeze_hs.json` to the Mac before step 4.

### 4. Plan (reads the recorded collection; no monitor is scored)

```bash
cd $E1B_REPO
$CLIENT_PY -I scripts/rldx_e1/plan_e1.py $RLDX_RAW --collection-root $SAFE35_ROOT \
  --freeze-manifest $E1_FREEZE --seed-roots $STORAGE_BS/robocasa_rollouts/safe \
  --calibration-seed 1601 --test-seed 1701 --output $E1B_ROOT 2>&1 | tee $E1B_LOG/plan.log
cat $E1B_ROOT/shards.tsv
```

Continue only with `PLAN OK`. With two GPUs, add `--shards-per-set 1`.

### 5. Servers (one per shard, ports 20150–20153)

```bash
cd $RLDX_REPO
while IFS=$'\t' read -r set shard gpu port; do
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup uv run python -u rldx/eval/run_rldx_server.py \
    --model-path $RLDX_SNAPSHOT --embodiment-tag GENERAL_EMBODIMENT \
    --host 127.0.0.1 --port $port --use-sim-policy-wrapper > $E1B_LOG/server_$port.log 2>&1 < /dev/null &
  echo "server $port GPU $gpu PID $!"
done < $E1B_ROOT/shards.tsv
NSH=$(wc -l < $E1B_ROOT/shards.tsv)
until [ "$(ss -ltn | grep -cE ':2015[0-9]\b')" -ge "$NSH" ]; do sleep 10; done; echo "servers up"
ps -eo args | grep "[r]un_rldx_server.py" | grep -c "$RLDX_SNAPSHOT"    # must equal the number of shards
tail -n 5 $E1B_LOG/server_20150.log
```

### 6. Smoke (port 20150, seed 1105, not part of E1b)

```bash
cd $E1B_REPO
mapfile -t FLAGS < $E1B_ROOT/collector_flags_calibration.txt
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 $CLIENT_PY -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir $E1B_ROOT/smoke --tasks TurnOnSinkFaucet "${FLAGS[@]}" --seed 1105 --num-rollouts 2 \
  --host 127.0.0.1 --port 20150 --max-errors 1 2>&1 | tail -n 5 | tee $E1B_LOG/smoke.log
$CLIENT_PY -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir $E1B_ROOT/smoke | tail -n 3
$CLIENT_PY -I scripts/rldx_e1/check_e1_collection.py --plan $E1B_ROOT/plan.json --smoke --smoke-seed 1105 \
  smoke=$E1B_ROOT/smoke 2>&1 | tee -a $E1B_LOG/smoke.log
```

Continue only with `VALID` and `CHECK OK (smoke)`.

### 7. Collect (add `--resume` to resume an interrupted shard)

```bash
cd $E1B_REPO
while IFS=$'\t' read -r set shard gpu port; do
  mapfile -t FLAGS < $E1B_ROOT/collector_flags_$set.txt
  mapfile -t tasks < $E1B_ROOT/tasks_${set}_shard$shard.txt
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu nohup $CLIENT_PY -u -m robocasa.recovery.safe.collect_atomic_rollouts \
    --output-dir $E1B_ROOT/collection/$set/shard$shard --tasks "${tasks[@]}" "${FLAGS[@]}" \
    --host 127.0.0.1 --port $port --continue-on-error --max-errors 5 \
    > $E1B_LOG/collection_${set}_shard$shard.log 2>&1 < /dev/null &
  echo "$set shard$shard GPU $gpu PID $!"
done < $E1B_ROOT/shards.tsv
```

**Progress:**

```bash
for d in $E1B_ROOT/collection/*/shard*; do echo "$d: $(wc -l < $d/manifest.jsonl 2>/dev/null) retained, $(wc -l < $d/errors.jsonl 2>/dev/null) errors"; done
tail -n 2 $E1B_LOG/collection_*.log
```

Do not score any monitor on these rollouts before step 9.

### 8. Gate 2, when every collector has exited

```bash
pkill -f "run_rldx_server.py.*--port 2015[0-9]"
cd $E1B_REPO
for d in $E1B_ROOT/collection/*/shard*; do echo "$d"; $CLIENT_PY -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir $d | tail -n 1; done
$CLIENT_PY -I scripts/rldx_e1/check_e1_collection.py --plan $E1B_ROOT/plan.json \
  $(for d in $E1B_ROOT/collection/*/shard*; do echo "$(basename $(dirname $d))=$d"; done) 2>&1 | tee $E1B_LOG/check.log
```

Continue only with `CHECK OK` and every shard `VALID`.

### 9. Hide-and-Seek scores (CPU)

```bash
cd $E1B_REPO
unset OPENBLAS_NUM_THREADS
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=$E1B_REPO $SAFE_PY -I scripts/rldx_e1/score_e1.py hs \
  --freeze-manifest $E1_FREEZE --freeze-scores-dir $(dirname $E1_FREEZE)/scores_run2 --recorded-dataset $RLDX_RAW \
  --output-dir $E1B_ROOT/scores \
  $(for d in $E1B_ROOT/collection/*/shard*; do echo "$(basename $(dirname $d))=$d"; done) 2>&1 | tee $E1B_LOG/score_hs.log
```

It prints the reproduction checks on the recorded rollouts, 800 rollouts scored per set, and
`SCORING OK`. SAFE-LSTM is not scored on E1b's rollouts.

### 10. Analysis (CPU, minutes)

```bash
cd $E1B_REPO
CUDA_VISIBLE_DEVICES="" $SAFE_PY -I scripts/rldx_e1/e1b.py analyze --freeze-manifest $E1_FREEZE \
  --e1b-freeze $E1B_ROOT/e1b_freeze_hs.json --calibration-dir $RLDX_ROOT/calibration/lstm \
  --hs-scores-dir $(dirname $E1_FREEZE)/scores_run2 \
  --e1-scores-dir $E1_ROOT/scores --e1b-scores-dir $E1B_ROOT/scores --plan $E1B_ROOT/plan.json \
  --output $E1B_ROOT/e1b_result.json 2>&1 | tee $E1B_LOG/analyze.log
sha256sum -c --quiet $E1B_LOG/e1_inputs.sha256 && echo "E1 INPUTS UNCHANGED"
```

It prints gates R, F and S, then the Hide-and-Seek line (false alarms against the bound, the gain
and matched gain with their 97.5% CIs, the null p-values, conditions (a)–(d), and whether H3
holds), and `ANALYSIS DONE`. Paste `analyze.log` back and copy `e1b_result.json` to the Mac.

## Result (2026-11-13, run `rldx1_e1b_20261112_195208`, code `bf6f6080`)

**H3 holds for Hide-and-Seek + P.** Every gate passed:

- **Smoke:** `CHECK OK (smoke)`.
- **Collection:** every shard `VALID`, `CHECK OK`, 400 successes and 400 failures in each set,
  and no errors.
- **Scoring:** the recorded calibration (60) and test (200) scores were reproduced exactly
  (max |diff| 0), followed by `SCORING OK`.
- **Analysis:** gates R, F and S passed, and E1's inputs were unchanged.

E1b's test set (400 successes, 400 failures), with the path frozen in `e1b_freeze_hs.json`
(u 0.2301, s 5) and the margin set on E1b's calibration successes:

| Monitor | False alarms (bound 39) | Mean lead | Gain, 97.5% CI | Matched gain, 97.5% CI | Null p (noise, resampled) | (a) (b) (c) (d) |
|---|---|---|---|---|---|---|
| P | 27 | 0.269 | — | — | — | — |
| Hide-and-Seek + P | 19 | 0.322 | 0.054 [0.001, 0.129] | 0.113 [0.049, 0.141] | 0.025, 0.025 | all true |

- **The null fusions:** the gain beats all 39 noise fusions and all 39 resampled fusions.
- **Reported alongside:**
  - **By type:** atomic 0.092, composite −0.036.
  - **In seconds:** 1.21 s.
  - **Run to the limit:** every failure ran to its limit, and one test success
    (`0a81d1cf4038bd83ee32936c`) ended at its limit.
- **Reading:**
  - The test with the path fixed on E1's 1,600 rollouts credits Hide-and-Seek's early warning;
    E1's test, with the path fixed on 60 successes, did not.
  - At the same target, the interval's lower end (0.0007) is barely above zero. The matched
    interval is clearly above zero, because Hide-and-Seek + P raises fewer false alarms than P.
  - As on XR-1, the gain comes from the atomic tasks.
- **Paper reporting (author's decision, stated on 2026-11-12 before collection):**
  - The paper reports H3 for Hide-and-Seek: E1's test, then E1b.
  - SAFE-LSTM's E1 result stays in E1's record and is not discussed in the paper.
  - This supersedes the last bullet of "Reporting".
- **Evidence:** `reports/rldx_e1b_evidence/` contains:
  - `e1b_result.json` (sha256 `bd0aab11e5e56b54…`);
  - both freeze files;
  - the smoke, check, scoring, freeze and analysis logs.
