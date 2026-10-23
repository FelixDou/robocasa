# RLDX-1 continuation and same-scene retry pilot for E3 (plan)

**Status: pilot, for designing the RSS Stage 2 E3 experiment.** It tests nothing that Stage 2
registers. This plan, its decision rule and its code were committed before any pilot rollout
was collected.

## Why

E3 asks whether resetting a rollout when the per-task timer P alarms beats letting the policy
keep running, at the same total budget of 1.25 H_k. The Stage 2 sizing
([`stage2_sizing.md`](stage2_sizing.md), "E3 with the measured same-scene retry rate") left
two unknowns, and E3's design depends on both:

- **c, the continuation success.** Of the rollouts that have not succeeded by the time limit
  H_k, the share that succeed by 1.25 H_k. No rollout has ever run past H_k, so c has never
  been measured. On RLDX-1 the sizing puts the crossover (reset and continuation tie) at
  c\* = 0.104 (r = 0.245). The only proxy, the late-success hazard, is 0.082. At the proxy,
  a test would need about 6,500 pairs.
- **r, the same-scene retry success.** The sizing borrowed r = 0.245 from the XR-1 pilot
  (report §15.24). RLDX-1's r is unmeasured.

The decision taken for E3 (2026-10-23): E3 runs on RLDX-1, its question is reset vs
continuation, and it is registered as a test only if this pilot shows the difference is
decidable within 1,000 pairs. Otherwise it is an estimation study.

## Design

One collection measures both unknowns.

- **Scenes.** The RLDX-1 SAFE20 natural-rate collection
  (`rldx1_original_safe35_natural50_20260808_203158`, filtered to the 20 supported tasks):
  50 scenes per task, 1,000 in all, 562 successes and 438 failures.
  - Its seed protocol `official_rldx` seeds each task's environment once and resets it 50
    times. Scenes are built only at reset: the scene RNG (`Kitchen.rng`) is used for layout,
    style, textures and object placement, not during steps. Re-running the collector with the
    same seed and settings therefore re-creates the same scenes, in the same order.
  - The policy samples stochastically, so each re-run is a new attempt at the same scene.
- **One change: the time limit.** Every task runs to ceil(1.25 × H_k), with H_k the registry
  horizon used originally. This is the new opt-in collector option `--horizon-scale 1.25`.
  - Seeds and the reset sequence are unchanged.
  - Rollout IDs differ, because the horizon is part of the rollout identity. Scenes are
    therefore paired by (task, reset index).
  - The RoboCasa gym wrapper never truncates (`ignore_done=True`), and rollouts stop at
    success. A pilot rollout's outcome at H_k is therefore exactly its outcome under the
    original limit.
- **All other collector settings are copied from the original collection** by
  `plan_pilot.py`: seed, protocol, split, replan steps, policy, checkpoint, policy config,
  action and video recording, frame stride, and the frame size read from an original video.
  SAFE features are recorded, as the collector requires.
- **Size.** All 1,000 scenes. Expected yield, if the outcome rates hold:
  - about 440 rollouts without success by H_k, so c to about ±0.03;
  - about 430 same-scene retries of originally failed scenes, for r.
- **Cost.** About 1.1× the original collection's env steps: failures run 25% longer. That is
  roughly 90,000 policy inferences, under half of E2's collection. `plan_pilot.py` prints the
  exact figure. Two GPUs, one RLDX server and one simulator shard each, with shards balanced
  by expected steps.
- **Scenes reused.** These are old scenes (base seed 7) whose outcomes are already known.
  Stage 2's E3 must use a fresh base seed.

## Gates (checked before any outcome is read)

1. **Identity and coverage.**
   - Every pilot rollout is a planned scene, with the planned seed, protocol, split, policy
     and replan steps, and a horizon of ceil(1.25 H_k).
   - No scene appears twice.
   - At least 95% of scenes are collected overall, and at least 80% of each task's.
   - Failing this stops the analysis; resume the collection.
2. **Scene.** A pair is the same scene if the mean absolute difference of the first video
   frames is at most **10** (0–255 scale).
   - The XR-1 pilot's tolerance of 2.0 was too tight for these renderings. Same-scene pairs
     there differed by 0.8–4.0; the two different scenes by 22 and 47.
   - If at least 95% of pairs are the same scene, r is measured, using same-scene pairs only.
   - Otherwise **r is void**, and the sizing uses XR-1's 0.245 and its lower bound 0.17.
   - c does not need pairing, and is reported either way.

## Endpoints

- **c (primary):** of the pilot rollouts with no success by H_k, the share that succeed by
  1.25 H_k.
  - Pooled, with Wilson and task-cluster bootstrap 95% intervals (2,000 replicates, seed 0).
  - The **decision interval** is the union of the two intervals.
  - Also reported by task type, by the scene's original outcome, and as the distribution of
    L / H_k for late successes.
- **r_fail:** success by H_k on same-scene retries of originally failed scenes, with the same
  intervals. **r_succ:** the same on originally successful scenes (control).
- **p:** the original success rate and the pilot's success by H_k, task-macro (a
  stationarity check).
- **Direct paired estimate (exploratory).** Each same-scene pair holds two independent
  attempts at one scene: the pilot rollout (run to 1.25 H_k) and the original (run to H_k).
  P alarms at the fraction f_k = min(1, ρ_k + λ) of the time limit, refitted exactly as in the
  E3 sizing and checked against its RLDX-1 lead of 0.2711. Then:
  - **continuation:** the pilot rollout succeeds by 1.25 H_k;
  - **reset:** the pilot rollout succeeds before P's alarm, or, once alarmed, the original
    attempt succeeds within the remaining budget. A retry still running at its own alarm
    counts as a failure, because no third attempt exists; the share is reported.
  - **no monitor (descriptive):** success by H_k, or a retry that fits the 0.25 H_k left.

  Reported: reset − continuation (task-macro, task-cluster CI, discordant counts, exact
  McNemar p), overall and by type. This is a small E3 on old scenes. It is exploratory and
  is used only as the cross-check in the rule below.

## Decision rule, fixed before collecting

Applied by `decide.py` to RLDX-1 over all 20 tasks, with P as the alarm. It uses two
`stage2_sizing.py e3` runs: one at the pilot's r (or 0.245 if r is void) and one at its Wilson
lower bound (or 0.17). Each run adds c's decision interval and point to the default c grid.

1. **The direction is decided** if reset − continuation has the same nonzero sign at both ends
   of c's decision interval, under both retry rates.
2. **n** is the largest number of pairs needed at those four points (McNemar, two-sided 0.05,
   power 0.8). It is rounded up to a multiple of 20 tasks, with a floor of 400 (20 per task).
3. **REGISTER** if the direction is decided and n ≤ 1,000: reset vs continuation becomes a
   two-sided superiority test at n pairs, with the expected direction stated.
4. **ESTIMATION** otherwise. E3 reports both arms and their paired difference with 95% CIs, at
   1,000 pairs (50 per task), with no superiority claim. The expected half-width is reported.
5. **Cross-check.** A REGISTER verdict is downgraded to ESTIMATION if the direct paired
   estimate's 95% interval lies entirely on the other side of zero from the expected
   direction.

Either way, in Stage 2's E3:
- P vs no monitor is not a registered test. The no-monitor arm stops at H_k and wastes the
  remaining budget. It is reported descriptively, from the continuation arm's outcome at H_k.
- SAFE + P and Hide-and-Seek + P as triggers are descriptive.

## Expectations, written before collecting

- **Gates.** Both pass. Seed replay reproduced 192 of 194 XR-1 scenes.
- **c:** between 0.02 and 0.08, below the hazard proxy 0.082. Rollouts still running at the
  limit are the slowest ones, and the success hazard falls with time. We expect more late
  successes on atomic than on composite tasks.
- **r_fail:** between 0.20 and 0.35, about half of p (0.56), as on XR-1. **r_succ** above p.
- **p:** the pilot's success by H_k within ±0.07 of the original 0.562 (task-macro).
- **Decision.** Most likely **ESTIMATION.**
  - A rough linear reading of the sizing grid (Δ(c) ≈ 0.046 · (1 − c / 0.104) at r = 0.245)
    gives n ≤ 1,000 only if c lies clearly below about 0.03 (reset better) or above about
    0.18 (continuation better).
  - We expect c's interval to reach into the band between.
- **Direct estimate.** Reset − continuation within ±0.05, with a CI that includes 0.

## Limits

- **One policy, old scenes.** The scenes' original outcomes were opened long ago. The pilot
  only measures c and r; it is not a test.
- **c is pooled** across tasks, as the sizing model uses it. Task variation enters only
  through the cluster interval.
- **The direct estimate uses the original rollout as the retry.** It is exchangeable with a
  new attempt on the same scene, but its retry is truncated at H_k. That is harmless: the
  remaining budget is below H_k whenever f_k > 0.25.
- **P's alarm is a fraction of the step limit.** P was fitted in inference units; converting
  at 8 steps per inference is approximate by at most one inference.
- **The robocasa code differs from the original collection's commit.** If scene generation
  changed, the scene gate catches it. c is unaffected.

## Code

- `robocasa/recovery/safe/collect_atomic_rollouts.py`: new opt-in `--horizon-scale`.
  - Default runs are unchanged.
  - The scale and the base horizons are recorded in the config and checked on resume.
- `scripts/rldx_continuation_pilot/plan_pilot.py`: checks the source and writes the plan
  files.
- `scripts/rldx_continuation_pilot/analyze_pilot.py`: gates, then c, r and p.
- `scripts/rldx_continuation_pilot/direct_e3.py`: the direct paired estimate.
- `scripts/rldx_continuation_pilot/decide.py`: the rule above.
- Tests: `tests/test_rldx_continuation_pilot.py` and
  `tests/test_safe_atomic_collection.py` (`--horizon-scale`).

## Runbook

Run it from a pinned worktree at the commit that contains this plan. `EP_SHA` is that
commit's full SHA. Two GPUs.

### 0. Session

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate /gs/bs/tga-shinoda/felid/envs/robocasa_openpi
export PROJECT_FS=/gs/fs/tga-shinoda/felid STORAGE_BS=/gs/bs/tga-shinoda/felid
export RLDX_REPO=$PROJECT_FS/RLDX-1
export CLIENT_PY=$STORAGE_BS/envs/robocasa_openpi/bin/python SAFE_PY=$STORAGE_BS/envs/vla_safe/bin/python
export UV_CACHE_DIR=$STORAGE_BS/uv_cache UV_LINK_MODE=copy HF_HOME=$STORAGE_BS/hf_home
export TRANSFORMERS_CACHE=$HF_HOME/transformers WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export SAFE35_ROOT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_original_safe35_natural50_20260808_203158
export RLDX_EXPORT=$SAFE35_ROOT/official_safe_supported20_natural50
export RLDX_CAL=$STORAGE_BS/robocasa_checkpoints/safe/rldx1_safe20_natural_weighted_20260815_162530/calibration/lstm
export RLDX_RAW=$($CLIENT_PY -I -c "import json,sys; print(json.load(open(sys.argv[1]))['source_dataset'])" $RLDX_EXPORT/conversion_report.json)
echo "RLDX_RAW=$RLDX_RAW"; test -f $RLDX_RAW/manifest.jsonl && echo "raw manifest OK"
export EP_SHA=<full SHA of the commit containing this plan>
export EP_REPO=$PROJECT_FS/robocasa_rldx_e3pilot_${EP_SHA:0:8}
export EP_ROOT=$STORAGE_BS/robocasa_rollouts/safe/rldx1_e3_pilot_$(date +%Y%m%d_%H%M%S)
export EP_LOG=$STORAGE_BS/robocasa_logs/eval/$(basename $EP_ROOT)
mkdir -p $EP_ROOT $EP_LOG; env | grep -E '^(EP_|RLDX_|SAFE35_ROOT|CLIENT_PY|SAFE_PY)' > $EP_ROOT/session.env
```

In a new terminal, re-run the exports above, but replace the three `EP_ROOT`, `EP_LOG` and
`mkdir` lines with `source <EP_ROOT>/session.env`.

### 1. Worktree, assets, tests

```bash
git -C $PROJECT_FS/robocasa fetch origin
test -e $EP_REPO || git -C $PROJECT_FS/robocasa worktree add --detach $EP_REPO $EP_SHA
test "$(git -C $EP_REPO rev-parse HEAD)" = "$EP_SHA" && echo "worktree OK" || echo "STOP: wrong HEAD"
mkdir -p $EP_REPO/robocasa/models/assets
cp -aln $PROJECT_FS/robocasa/robocasa/models/assets/. $EP_REPO/robocasa/models/assets/
cd $EP_REPO && $CLIENT_PY -W ignore -m unittest tests.test_rldx_continuation_pilot tests.test_safe_atomic_collection 2>&1 | tail -1
```

### 2. Plan (reads the original collection; no outcome endpoint)

```bash
cd $EP_REPO
$CLIENT_PY -I scripts/rldx_continuation_pilot/plan_pilot.py $RLDX_RAW --collection-root $SAFE35_ROOT \
  --output $EP_ROOT 2>&1 | tee $EP_LOG/plan.log
```

Continue only with `PLAN OK`. Paste `plan.log` back. If it prints `STOP`, or warns that original
videos are missing, stop and report.

### 3. Servers (GPUs 0 and 1, ports 20140 and 20141)

```bash
git -C $RLDX_REPO log -1 --format=%H > $EP_LOG/rldx_repo_head.txt
git -C $RLDX_REPO status --short > $EP_LOG/rldx_repo_status.txt
cd $RLDX_REPO
for g in 0 1; do
  port=$((20140 + g))
  CUDA_VISIBLE_DEVICES=$g nohup uv run python -u rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1-FT-RC365 --embodiment-tag GENERAL_EMBODIMENT \
    --host 127.0.0.1 --port $port --use-sim-policy-wrapper > $EP_LOG/server_$port.log 2>&1 &
done
until ss -ltn | grep -q ':20140\b' && ss -ltn | grep -q ':20141\b'; do sleep 10; done; echo "servers up"
```

### 4. Collect (add `--resume` to resume an interrupted shard)

```bash
cd $EP_REPO
mapfile -t FLAGS < $EP_ROOT/collector_flags.txt
for shard in 0 1; do
  port=$((20140 + shard)); mapfile -t tasks < $EP_ROOT/tasks_shard$shard.txt
  CUDA_VISIBLE_DEVICES=$shard MUJOCO_EGL_DEVICE_ID=$shard nohup $CLIENT_PY -u -m robocasa.recovery.safe.collect_atomic_rollouts \
    --output-dir $EP_ROOT/collection/shard$shard --tasks "${tasks[@]}" "${FLAGS[@]}" \
    --attempt-limits $EP_ROOT/attempt_limits_shard$shard.json \
    --host 127.0.0.1 --port $port --continue-on-error --max-errors 5 \
    > $EP_LOG/collection_shard$shard.log 2>&1 &
  echo "shard $shard PID $!"
done
```

Progress:

```bash
for s in 0 1; do echo "shard$s: $(wc -l < $EP_ROOT/collection/shard$s/manifest.jsonl 2>/dev/null) rollouts, $(wc -l < $EP_ROOT/collection/shard$s/errors.jsonl 2>/dev/null) errors"; tail -n 2 $EP_LOG/collection_shard$s.log; done
```

After the first few rollouts, check that the horizon took effect. This prints each shard's
config horizons, which must be 1.25× the plan's H:

```bash
grep -o '"task_horizons": {[^}]*}' $EP_ROOT/collection/shard0/summary.json
```

### 5. Analyse (gates first), then size and decide

When both collectors have exited:

```bash
pkill -f "run_rldx_server.py.*--port 2014[01]"
cd $EP_REPO
$CLIENT_PY -I scripts/rldx_continuation_pilot/analyze_pilot.py $EP_ROOT/collection --selection $EP_ROOT/selection.json \
  --output $EP_ROOT/pilot.json 2>&1 | tee $EP_LOG/analyze.log
```

Continue only if gate 1 passed. A failed scene gate is not a stop: c is still valid, and the
fallback r values are used.

```bash
export E1_OUT=$STORAGE_BS/robocasa_rollouts/safe/xr1_lead_calibration_20261020_001108/lead_analysis
export XR1_RUNTIME_BUNDLE=$STORAGE_BS/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_calibration/runtime_bundle.json
read R RLOW CONT < <($CLIENT_PY -I -c "import json,sys; s=json.load(open(sys.argv[1]))['sizing_inputs']; print(s['retry_success'], s['retry_success_low'], ','.join(str(x) for x in s['continuation']))" $EP_ROOT/pilot.json)
echo "r=$R r_low=$RLOW c=$CONT"
sha256sum $E1_OUT/calibration_with_baselines.jsonl $E1_OUT/test_with_baselines.jsonl $XR1_RUNTIME_BUNDLE \
  $RLDX_CAL/lstm_seed{0,1,2}/normalized_scores.jsonl $EP_ROOT/pilot.json $EP_ROOT/selection.json | tee $EP_LOG/frozen_inputs.sha256
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for pair in "r:$R" "rlow:$RLOW"; do
  tag=${pair%%:*}; r=${pair#*:}
  nice -n 10 $SAFE_PY -I -u scripts/xr1_lead_time/stage2_sizing.py e3 --calibration $E1_OUT/calibration_with_baselines.jsonl \
    --test $E1_OUT/test_with_baselines.jsonl --runtime-bundle $XR1_RUNTIME_BUNDLE --rldx-calibration-dir $RLDX_CAL \
    --retry-success $r --continuation 0,0.05,0.1,0.15,0.2,0.27,0.4,$CONT --output $EP_ROOT/e3_$tag.json > $EP_LOG/e3_$tag.log 2>&1 \
    || echo "STOP: e3 $tag failed"
done
$SAFE_PY -I scripts/rldx_continuation_pilot/direct_e3.py --pilot-json $EP_ROOT/pilot.json --pilot-collection $EP_ROOT/collection \
  --selection $EP_ROOT/selection.json --rldx-calibration-dir $RLDX_CAL --output $EP_ROOT/direct.json 2>&1 | tee $EP_LOG/direct.log
$SAFE_PY -I scripts/rldx_continuation_pilot/decide.py --pilot $EP_ROOT/pilot.json --e3 $EP_ROOT/e3_r.json \
  --e3-low $EP_ROOT/e3_rlow.json --direct $EP_ROOT/direct.json --output $EP_ROOT/decision.json 2>&1 | tee $EP_LOG/decide.log
sha256sum -c $EP_LOG/frozen_inputs.sha256
```

Paste back `analyze.log`, the `check` lines of `e3_r.log` and `e3_rlow.log`, `direct.log`,
`decide.log` and the checksum result. The RLDX-1 check must print P lead 0.2711. Do not edit
inputs to make a run pass.
