# E1: frozen monitors on new RLDX-1 rollouts (plan and collection runbook)

**Status.** Written 2026-11-05, before any E1 rollout exists.
- **Paper:** E1 ("Second policy") of the RSS 2027 paper; it tests H3.
- **Why now:** the author decided to start it before the Stage 1 review, so its data exist
  well before the Stage 2 deadline (16 April 2027).
- **Design:** the one stated in the Stage 1 draft and in R9/R10 of
  `docs/xr1_review_analyses_plan.md` (branch `claude/review-analyses`).
- **If the Extension Charter asks for changes,** these rollouts stay valid new data, and the
  changes are added on top.

This document covers the collection, the scoring and the analysis. The code for all three was
committed before any E1 rollout existed (scoring and analysis on 2026-11-05, in the commit after
the collection plan). No monitor is scored on an E1 rollout unless it first reproduces its
recorded scores exactly (steps 9–10).

## Question

Do SAFE-LSTM + P and Hide-and-Seek + P, fixed on recorded RLDX-1 rollouts, warn earlier than
the per-task timer P on new rollouts, at the same FPR target and at matched false alarms (H3)?

## Frozen before collection

Every item comes from the E1 freeze manifest (R10 part B,
`e2_freeze_20261029_205743/e2_freeze_manifest.json`, sha256 `917c06e6ab701f9f…`; the run
was then called "E2").
- **Tasks:** the 20 RLDX-1 tasks of the recorded collection (14 atomic, 6 composite), with
  their time limits $H_k$ (in policy inferences of 8 control steps) and P's $\rho_k$.
- **Detectors:**
  - SAFE-LSTM: seeds 0–2, `rldx1_safe20_natural_weighted_20260815_162530/final/lstm_seed*`.
  - Hide-and-Seek: seeds 0–2, `e2_freeze_20261029_205743/models_run2/hide_and_seek_seed*.npz`.
  - The score normalization of each.
- **Fused paths:**
  - SAFE-LSTM + P: $u = 7.9314$, $s = 0$.
  - Hide-and-Seek + P: $u = -0.17$, $s = 20$.
  - P: $\tau_s = \infty$.
  - The null fusions' paths are chosen on null scores of the recorded data, as in R10.
- **Policy:** `RLWRLD/RLDX-1-FT-RC365`, revision `587e9ecdcc5e7184fcc17f58713908edff5af041`.
  - This is the revision the recorded rollouts used.
  - The current Hub revision (`c53d0bc0`) changes the processor's `image_max_area`, so the
    server loads the 587e9ecd snapshot by path, offline.
- **RLDX-1 source:** commit `ef05cd4a` with the cluster checkout's working-tree diff, audited on
  2026-11-05 and stored as `patches/rldx1_e1_audited_worktree_ef05cd4.diff` (sha256
  `1a723b1c0dba9491…`). It is not the SAFE patch alone, so the patch no longer reverse-applies
  cleanly. It contains:
  - `patches/rldx1_safe_features_ef05cd4.patch` (the recorded rollouts' feature capture);
  - compatibility edits that leave computation unchanged:
    - a dataclass `kw_only` fix;
    - an import fallback in the official evaluator;
    - `python -u` in its script.
  - the `max_area is None` guard in `resize_preserve_aspect_area_then_crop`. With the 587e9ecd
    snapshot's null `image_max_area`, it keeps the full image area; without it, that snapshot
    cannot run. So the recorded rollouts used it too.
  - an optional `safe_feature_mode = "action_observation_context"` (2026-08-09, after the
    recorded collection started). It is used only when a request asks for it; E1's client never
    does. In the default `action` mode, actions, features and metadata equal the SAFE patch's.

## Rollouts

- **Two new sets:** a calibration set and a test set.
  - Each has 20 successes and 20 failures for each of the 20 tasks (800 per set).
  - Collected with per-class quotas (`--success-quota 20 --failure-quota 20
    --retain-only-quota`).
  - The collector keeps attempting until both classes are full. It discards later examples of
    a full class, but records their outcome.
- **New scenes:**
  - Base seed 1201 for the calibration set and 1301 for the test set, under the `official_rldx`
    protocol (one environment per task, reset between attempts).
  - The recorded collection used base seed 20.
  - The planner stops if either seed was used by any RLDX-1 collection it finds.
- **Same settings as the recorded collection.** Everything else is copied from it by the
  planner and checked on every rollout:
  - seed protocol, split (`target`), replan steps (8), policy label and config;
  - feature recording, actions, videos (frame size and stride);
  - registry time limits, at 1.0 $H_k$.
- **Attempt cap:** 300 per task and set. The planner stops if a task exceeds it in more than
  1% of simulated collections.
- **Cost (from the recorded success rates):** about 2,500 rollouts and 1.8 million control
  steps. That is about 2.3 times the E3 pilot, which took 26 hours on two GPUs: about 1–1.5
  days on four GPUs, or 2.5 days on two. The planner prints the exact estimate.

## Gates

1. **Smoke** (1 task, 2 attempts, base seed 1101, not part of E1).
   - `check_e1_collection.py --smoke` prints `CHECK OK (smoke)`.
   - The dataset validator prints `VALID`.
   - Every server was started offline from the 587e9ecd snapshot path.
2. **Collection.** On all four shards, `check_e1_collection.py` prints `CHECK OK`:
   - settings and feature contract equal the recorded collection's;
   - every rollout has its set's seed;
   - no rollout ID repeats, within E1 or against the recorded rollouts;
   - exactly 20 + 20 per task and set;
   - no collection errors.
   - Each shard is also `VALID`.

   If a task misses its quota at the cap, preserve the shard and resume it with a larger
   `--num-rollouts`. Never lower a quota or delete evidence.

## Analysis (fixed before collection; `scripts/rldx_e1/analyze_e1.py`)

- **Per detector,** on the new sets:
  - $\hat\lambda$ is set on the new calibration successes by Proposition 1 at $\alpha = 0.05$,
    with the path fixed. If $\hat\lambda = \infty$, the monitor is disabled.
  - P's margin is set the same way.
  - Both are evaluated on the new test set.
- **H3 holds for a detector only if all four conditions hold (R10):**
  - (a) the 97.5% CI of its gain in mean lead over P lies above zero;
  - (b) the 97.5% CI of its gain over P loosened on the test set to its own false alarms lies
    above zero (matched);
  - (c) its test false alarms are at most 39 of 400 (exact binomial rule);
  - (d) its gain beats at least 38 of 39 random-score fusions and 38 of 39 resampled-score
    fusions run on the same sets.
- **CIs:** bootstrap with calibration and test rollouts resampled within task and class,
  $\hat\lambda$ re-set in each replicate, path fixed. 97.5% is Bonferroni over the two
  detectors.
- **Simulated on the recorded rollouts (R10):** type I error 0 of 2,000 under both nulls;
  power 1.00 for SAFE-LSTM and 0.94 for Hide-and-Seek (nested) at 20 + 20 per task.
- **Gate R** comes before any E1 number is computed. On the recorded rollouts, the frozen paths
  certified on the recorded calibration successes must give the manifest's margins (0.0421,
  0.068) and replication gains (0.0342, 0.0834). The manifest's $H_k$ and $\rho_k$ must equal
  those recomputed from the recorded rows.
- **Random draws** are fixed by seed 20261105: stream 1 for the bootstrap, stream 2 for the
  null fusions.
- **Reported alongside, not decisive:**
  - the per-type split;
  - seconds (0.4 s per RLDX-1 inference);
  - whether every failure ran to its limit.
- **Not part of E1:** Hide-and-Seek's input ablations were not frozen with models. If wanted,
  they are an exploratory addition after E1.

## Code

- `scripts/rldx_e1/plan_e1.py` writes the collection plan:
  - collector flags per set;
  - tasks per shard, balanced by expected control steps;
  - GPU and port per shard.

  It reuses `scripts/rldx_continuation_pilot/plan_pilot.py` to read and check the recorded
  collection.
- `scripts/rldx_e1/check_e1_collection.py` holds gates 1 and 2.
- `scripts/rldx_e1/score_e1.py` scores the new sets with the frozen monitors:
  - `hs` reloads the three Hide-and-Seek models. It writes nothing until they reproduce the
    freeze run's scores exactly on every recorded calibration and test rollout.
  - `safe` assembles SAFE-LSTM scores from `score_seen_checkpoint_external` outputs. It writes
    nothing until the final runs' digests match the manifest, their per-task normalization
    equals the recorded one, and the re-scored recorded export equals its recorded raw scores.
- `scripts/rldx_e1/analyze_e1.py` holds gate R and rule (a)–(d).
- `tests/test_rldx_e1_plan.py` and `tests/test_rldx_e1_analysis.py` are simulator-free.
- The collector is `robocasa.recovery.safe.collect_atomic_rollouts`, unchanged.

## Runbook

Run everything from a pinned worktree at the commit that contains this plan. `E1_SHA` is
that commit's full SHA (the author receives it with this plan).
- **GPUs:** four for steps 4–6. With two GPUs, plan with `--shards-per-set 1`.
- **CPU only:** steps 0–3 and 7.
- **Recovery runs:** never pull or check out in the shared checkout
  `$PROJECT_FS/robocasa`, which they use.

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
export E1_SHA=<full SHA of the commit containing this plan>
export E1_REPO=$PROJECT_FS/robocasa_rldx_e1_${E1_SHA:0:8}
export E1_ROOT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_e1_$(date +%Y%m%d_%H%M%S)
export E1_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $E1_ROOT)
mkdir -p $E1_ROOT $E1_LOG; env | grep -E '^(E1_|RLDX_|SAFE35_ROOT|CLIENT_PY|SAFE_PY)' > $E1_ROOT/session.env
echo "RLDX_RAW=$RLDX_RAW"; test -f $RLDX_RAW/manifest.jsonl && echo "raw manifest OK"
```

In a new terminal, re-run the exports above, but replace the three `E1_ROOT`, `E1_LOG` and
`mkdir` lines with `source <E1_ROOT>/session.env`.

### 1. Worktree, assets, tests

```bash
git -C $PROJECT_FS/robocasa fetch origin
test -e $E1_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $E1_REPO $E1_SHA
test "$(git -C $E1_REPO rev-parse HEAD)" = "$E1_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
mkdir -p $E1_REPO/robocasa/models/assets
cp -aln $PROJECT_FS/robocasa/robocasa/models/assets/. $E1_REPO/robocasa/models/assets/
cd $E1_REPO && CUDA_VISIBLE_DEVICES="" $SAFE_PY -W ignore -m unittest tests.test_rldx_e1_plan tests.test_rldx_e1_analysis tests.test_e2_frozen_rule tests.test_xr1_fixed_path tests.test_rldx_continuation_pilot tests.test_safe_atomic_collection 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Continue only with `OK`. One test module prints a JSON object, so read the `Ran` and `OK` lines, not
the last line.

### 2. Frozen inputs (record, then check)

```bash
cd $E1_REPO
sha256sum $E1_FREEZE | tee $E1_LOG/freeze_sha256.txt | cut -c1-16    # must print 917c06e6ab701f9f
test -d $RLDX_SNAPSHOT && echo "policy snapshot 587e9ecd OK" || echo "STOP: snapshot missing"
git -C $RLDX_REPO rev-parse HEAD | tee $E1_LOG/rldx_repo_head.txt      # must be ef05cd4a...
git -C $RLDX_REPO status --short > $E1_LOG/rldx_repo_status.txt
git -C $RLDX_REPO diff > $E1_LOG/rldx_repo_diff.patch
cmp $E1_LOG/rldx_repo_diff.patch $E1_REPO/patches/rldx1_e1_audited_worktree_ef05cd4.diff \
  && echo "RLDX diff = audited diff" || echo "STOP: RLDX working tree differs from the audited diff"
CUDA_VISIBLE_DEVICES="" $SAFE_PY -I -c "import sys; sys.path.insert(0, 'scripts/xr1_lead_time'); import e2_freeze_manifest as m; print(m.tree_hash(sys.argv[1])['digest'])" $RLDX_SNAPSHOT \
  | tee $E1_LOG/snapshot_digest.txt   # must print f1a6048dfce7c138dc20337ac93ce112e055b4184bae7347d67b65698b6c3eb6
cat $E1_LOG/rldx_repo_status.txt
```

Paste the output back. Continue only if all of these hold:
- the freeze hash is `917c06e6ab701f9f`;
- the RLDX commit is `ef05cd4a…`;
- `RLDX diff = audited diff` prints;
- the snapshot digest is `f1a6048d…` (reading 13.8 GB takes a few minutes).

The status may also list the untracked `external_dependencies/robocasa365`.

### 3. Plan (reads the recorded collection; no monitor is scored)

```bash
cd $E1_REPO
$CLIENT_PY -I scripts/rldx_e1/plan_e1.py $RLDX_RAW --collection-root $SAFE35_ROOT \
  --freeze-manifest $E1_FREEZE --seed-roots $STORAGE_BS/robocasa_rollouts/safe \
  --output $E1_ROOT 2>&1 | tee $E1_LOG/plan.log
cat $E1_ROOT/shards.tsv
```

Continue only with `PLAN OK`. Paste `plan.log` back. With two GPUs, add `--shards-per-set 1`.

### 4. Servers (one per shard: GPUs 0–3, ports 20150–20153)

```bash
cd $RLDX_REPO
while IFS=$'\t' read -r set shard gpu port; do
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup uv run python -u rldx/eval/run_rldx_server.py \
    --model-path $RLDX_SNAPSHOT --embodiment-tag GENERAL_EMBODIMENT \
    --host 127.0.0.1 --port $port --use-sim-policy-wrapper > $E1_LOG/server_$port.log 2>&1 < /dev/null &
  echo "server $port GPU $gpu PID $!"
done < $E1_ROOT/shards.tsv
NSH=$(wc -l < $E1_ROOT/shards.tsv)
until [ "$(ss -ltn | grep -cE ':2015[0-9]\b')" -ge "$NSH" ]; do sleep 10; done; echo "servers up"
ps -eo args | grep "[r]un_rldx_server.py" | grep -c "$RLDX_SNAPSHOT"    # must equal the number of shards
tail -n 5 $E1_LOG/server_20150.log
```

If a server rejects the local snapshot path, stop and paste its log. Do not fall back to the Hub
name, which would load the current revision.

### 5. Smoke (gate 1; port 20150, seed 1101, not part of E1)

```bash
cd $E1_REPO
mapfile -t FLAGS < $E1_ROOT/collector_flags_calibration.txt
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 $CLIENT_PY -u -m robocasa.recovery.safe.collect_atomic_rollouts \
  --output-dir $E1_ROOT/smoke --tasks TurnOnSinkFaucet "${FLAGS[@]}" --seed 1101 --num-rollouts 2 \
  --host 127.0.0.1 --port 20150 --max-errors 1 2>&1 | tail -n 5 | tee $E1_LOG/smoke.log
$CLIENT_PY -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir $E1_ROOT/smoke | tail -n 3
$CLIENT_PY -I scripts/rldx_e1/check_e1_collection.py --plan $E1_ROOT/plan.json --smoke --smoke-seed 1101 \
  smoke=$E1_ROOT/smoke 2>&1 | tee -a $E1_LOG/smoke.log
```

Continue only with `VALID` and `CHECK OK (smoke)`.

### 6. Collect (add `--resume` to resume an interrupted shard)

```bash
cd $E1_REPO
while IFS=$'\t' read -r set shard gpu port; do
  mapfile -t FLAGS < $E1_ROOT/collector_flags_$set.txt
  mapfile -t tasks < $E1_ROOT/tasks_${set}_shard$shard.txt
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu nohup $CLIENT_PY -u -m robocasa.recovery.safe.collect_atomic_rollouts \
    --output-dir $E1_ROOT/collection/$set/shard$shard --tasks "${tasks[@]}" "${FLAGS[@]}" \
    --host 127.0.0.1 --port $port --continue-on-error --max-errors 5 \
    > $E1_LOG/collection_${set}_shard$shard.log 2>&1 < /dev/null &
  echo "$set shard$shard GPU $gpu PID $!"
done < $E1_ROOT/shards.tsv
```

**Progress** (retained rollouts, errors, last log lines):

```bash
for d in $E1_ROOT/collection/*/shard*; do echo "$d: $(wc -l < $d/manifest.jsonl 2>/dev/null) retained, $(wc -l < $d/errors.jsonl 2>/dev/null) errors"; done
tail -n 2 $E1_LOG/collection_*.log
```

The progress lines show retained counts only. Do not score any monitor on these rollouts.

### 7. Gate 2, when every collector has exited

```bash
pkill -f "run_rldx_server.py.*--port 2015[0-9]"
cd $E1_REPO
for d in $E1_ROOT/collection/*/shard*; do echo "$d"; $CLIENT_PY -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir $d | tail -n 1; done
$CLIENT_PY -I scripts/rldx_e1/check_e1_collection.py --plan $E1_ROOT/plan.json \
  $(for d in $E1_ROOT/collection/*/shard*; do echo "$(basename $(dirname $d))=$d"; done) 2>&1 | tee $E1_LOG/check.log
```

Paste `check.log` back. Continue to scoring only with `CHECK OK` and every shard `VALID`.

### 8. SAFE exports (CPU)

Merge each set's shards, validate the result, and export it to SAFE's loader layout. This is the
format the recorded rollouts were scored from.

```bash
cd $E1_REPO
for set in calibration test; do
  $CLIENT_PY -m robocasa.recovery.safe.merge_atomic_datasets --source-dirs $E1_ROOT/collection/$set/shard* \
    --output-dir $E1_ROOT/merged/$set 2>&1 | tail -n 2
  $CLIENT_PY -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir $E1_ROOT/merged/$set | tail -n 1
  $CLIENT_PY -m robocasa.recovery.safe.export_to_official_safe --dataset-dir $E1_ROOT/merged/$set \
    --output-dir $E1_ROOT/safe_export/$set --summary-only 2>&1 | tail -n 3
done
```

Continue only if both merged sets are `VALID` and each export holds 800 rollouts.

### 9. Scores with the frozen monitors (one GPU for SAFE; Hide-and-Seek on CPU)

SAFE-LSTM: every frozen final run scores the recorded export (its reproduction check) and both
E1 exports. Nothing is fitted.

```bash
cd $E1_REPO
export SAFE_REPO=$PROJECT_FS/SAFE
export RLDX_ROOT=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_safe20_natural_weighted_20260815_162530
export RLDX_CAL=$RLDX_ROOT/calibration/lstm
export FREEZE_SCORES=$(dirname $E1_FREEZE)/scores_run2
for s in 0 1 2; do
  for x in recorded:$RLDX_EXPORT calibration:$E1_ROOT/safe_export/calibration test:$E1_ROOT/safe_export/test; do
    CUDA_VISIBLE_DEVICES=0 $SAFE_PY -u -m robocasa.recovery.safe.score_seen_checkpoint_external \
      --export-dir ${x#*:} --safe-repo $SAFE_REPO --training-run-dir $RLDX_ROOT/final/lstm_seed$s \
      --output-dir $E1_ROOT/safe_scores/${x%%:*}/seed$s --group all --device cuda 2>&1 | tail -n 1
  done
done | tee $E1_LOG/safe_external.log
```

Then assemble both monitors' E1 scores. Each command prints its reproduction checks and
`SCORING OK`, or stops before writing anything.

```bash
cd $E1_REPO
$SAFE_PY -I scripts/rldx_e1/score_e1.py safe --freeze-manifest $E1_FREEZE --calibration-dir $RLDX_CAL \
  $(for s in 0 1 2; do echo --recorded-scores $s $E1_ROOT/safe_scores/recorded/seed$s; done) \
  $(for set in calibration test; do for s in 0 1 2; do echo --e1-scores $set $s $E1_ROOT/safe_scores/$set/seed$s; done; done) \
  --output-dir $E1_ROOT/scores 2>&1 | tee $E1_LOG/score_safe.log
unset OPENBLAS_NUM_THREADS
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=$E1_REPO $SAFE_PY -I scripts/rldx_e1/score_e1.py hs \
  --freeze-manifest $E1_FREEZE --freeze-scores-dir $FREEZE_SCORES --recorded-dataset $RLDX_RAW --output-dir $E1_ROOT/scores \
  $(for d in $E1_ROOT/collection/*/shard*; do echo "$(basename $(dirname $d))=$d"; done) 2>&1 | tee $E1_LOG/score_hs.log
```

Hide-and-Seek uses the freeze run's thread settings (8 OpenMP/MKL threads, CPU). If a
reproduction check fails by about 1e-7, paste the log; do not loosen the check yourself.

### 10. Analysis (CPU, minutes)

```bash
cd $E1_REPO
CUDA_VISIBLE_DEVICES="" $SAFE_PY -I scripts/rldx_e1/analyze_e1.py --freeze-manifest $E1_FREEZE \
  --calibration-dir $RLDX_CAL --hs-scores-dir $FREEZE_SCORES --e1-scores-dir $E1_ROOT/scores \
  --plan $E1_ROOT/plan.json --output $E1_ROOT/e1_result.json 2>&1 | tee $E1_LOG/analyze.log
```

It prints gate R first (both lines `OK`), then one line per detector: false alarms against the
bound, the gain and matched gain with their 97.5% CIs, the null p-values, conditions (a)–(d), and
whether H3 holds. Paste `analyze.log` back and copy `e1_result.json` to the Mac.
