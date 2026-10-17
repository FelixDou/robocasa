# Confirmatory prospective test of fused SAFE + time (preregistration and runbook)

**Status: final preregistration. The hypotheses and success criteria (A.2–A.6)
were reviewed and approved by the user on 2026-10-17.** No rollout may be
collected for this test until the cluster worktree is pinned to the commit
containing this version of the file (Section 2) and `preregister` has run
(Section 3).

## Part A. Preregistration

### A.1 Background

Section 15.10 of
[`safe_robocasa_experiment_report.md`](safe_robocasa_experiment_report.md)
found, post hoc on the already-opened August cohort, that unstaged fusion of a
policy-internal score with elapsed time beat the preregistered staged detector.
`fused_safe_time` reached recall 0.955 at FPR 0.058 against 0.632 at 0.047 for
`time_only`. Fused success-only baselines were statistically indistinguishable
from fused SAFE. Every fused detector was defined after that cohort was
opened, and every test FPR (0.058–0.068) exceeded calibration's 0.044. This
test collects a fresh, identity-disjoint cohort and evaluates the frozen
detectors once.

### A.2 Hypotheses

- **Primary.** `fused_safe_time` has higher failure recall than `time_only` at
  the frozen operating points.
- **Pre-declared secondary.** `fused_knn_cosine_time` (fitted on successful
  rollouts only, so label-free) has higher failure recall than `time_only` at
  the frozen operating points.

The two hypotheses are judged separately. The secondary does not rescue a
failed primary, and no multiplicity correction is applied to it.

### A.3 Success criteria

Each hypothesis passes only if all three criteria hold. All three are
reported whatever the outcome.

| Criterion | Definition |
|---|---|
| (a) | Paired task/rollout bootstrap 95% CI of the recall difference (detector minus `time_only`) has lower bound > 0. 2,000 replicates, seed 0, the same task-then-rollout resampling as the August study. |
| (b) | Point recall gain over `time_only` ≥ 0.15. |
| (c) | Observed test FPR ≤ 0.075, and its Wilson 95% upper bound ≤ 0.10. |

These values are encoded in `CONFIRMATORY_CRITERIA` in
`robocasa/recovery/safe/run_offline_detector_baselines.py` and copied into the
preregistration JSON. The evaluator reads them only from that JSON.

### A.4 Frozen detectors (no refit, no recalibration)

| Component | Frozen source |
|---|---|
| SAFE MLP seeds 0/1/2, task normalization, time-risk curves, task horizons | `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_calibration/runtime_bundle.json` and its `runtime/seed*/` copies |
| Baseline models, aggregations, normalization | `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_offline_baselines_20261017_100820/develop/baseline_bundle.json` and `develop/models/*.pkl` |
| All thresholds | `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_offline_baselines_20261017_100820/calibrate_fused/baseline_calibration.json` |

The `preregister` step records the sha256 of every file above, the sha256 of
this document, and the code commit. `confirm` re-verifies all of them and
stops on any change. Thresholds were fixed on the August 228-rollout
calibration cohort; the calibration FPR of every detector was 5/114.

### A.5 Data collection

- Policy: Xiaomi-Robotics-1 RoboCasa365 (the same SAFE-patched checkpoint and
  server as August), official Xiaomi seed protocol, `replan_steps` 16, SAFE
  features, actions and videos recorded as in August.
- Tasks: the same frozen 38-task list and two shards as August
  (`xr1_prospective_matched_fpr_20260815_163700/tasks_shard{0,1}.txt`).
- Quota: the first 10 successes and first 10 failures per task, with
  `--no-retain-only-quota`. Every attempt, overshoot rollout and error is kept.
  At most 500 attempts per task.
- Seeds: base environment seed **1300007**, 500 attempts per task. Seeds are
  therefore drawn from [1300007, 1325007). Prior Xiaomi collections in the
  docs used bases 7, 100007, 310007, 490007/8, 500007, 696001, 700007, 880007,
  900007, 910007, 920007 and 950007/11/12/22. The August test occupied
  700007–725006 and the August calibration 500007–512506. Before collection,
  `preregister` scans every Xiaomi manifest found on `/gs/bs` and refuses the
  range if any prior rollout used a seed in it.
- If any task cannot reach 10/10 within 500 attempts, stop and report. Do not
  raise the cap, drop the task, or change the quota without a written
  amendment committed before scoring.

### A.6 Analysis plan

- Primary cohort: per task, the first 10 successes and 10 failures in
  (environment seed, reset index, rollout ID) order, as in August. That is 760
  rollouts (380/380). The all-retained stream is reported as a secondary,
  descriptive cohort.
- Scores: SAFE trajectories come from the frozen seed checkpoints
  (`score_seen_checkpoint_external`, group `confirmatory_test`). Baseline
  features come from the official loader (`run_offline_detector_baselines
  extract`). Each detector raises its first alarm when its frozen threshold is
  crossed, exactly as in Section 15.10.
- Disjointness, enforced by `confirm`: no rollout ID or (task, environment
  seed, reset index) identity may appear in development, calibration, the
  August test, the baseline fit/validation sets, or any registered exclusion
  source. Every seed must lie in the preregistered range.
- Descriptive only: every other frozen detector (the other fused detectors,
  `staged_safe_time`, `safe_only`, the standalone and staged baselines),
  miss-adjusted detection, recall by the 10/25/50% landmarks, at-risk
  landmark ROC-AUC, atomic/composite strata with per-stratum paired bootstrap,
  and calibration-to-test FPR drift.
- `confirm` runs once into a new directory and refuses an existing one.
  Nothing is selected, refit, or recalibrated from these results.

### A.7 Commit rule

1. The user reviews A.2–A.6 and this file is finalized.
2. The final file is committed and pushed to `origin/main`. Record its full
   SHA as `XR1_CONFIRM_SHA`.
3. The cluster worktree is pinned to that SHA, and `preregister` is run
   (Section 3) **before the first collection command**.
4. The sha256 of `preregistration.json` is reported to the session or report
   before collection starts.

Any later change to this document invalidates the preregistration checksum,
and `confirm` will refuse to run.

## Part B. Cluster runbook

Read [`cluster_experiment_runbook.md`](cluster_experiment_runbook.md) first.
This run uses two GPUs alongside the recovery experiments. Use only the
dedicated worktree created below; never pull or check out in the shared
`/gs/fs/tga-shinoda/felid/robocasa` checkout, and never touch the exact-replay
paths.

### 1. Fresh session exports

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_SOURCE_REPO="$PROJECT_FS/robocasa"
export SAFE_REPO="$PROJECT_FS/SAFE"
export XR1_SAFE_REPO="$PROJECT_FS/robocasa_benchmark_repos/Xiaomi-Robotics-1-safe"
export XR1_SERVER_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_server"
export XR1_CLIENT_ENV="$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365"
export SAFE_ENV="$STORAGE_BS/envs/vla_safe"
export SAFE_PY="$SAFE_ENV/bin/python"
export XR1_SAFE_CHECKPOINT="$STORAGE_BS/robocasa_checkpoints/xiaomi_robotics_1/Xiaomi-Robotics-1-RoboCasa365-safe"
export HF_HOME="$STORAGE_BS/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_ENABLED=0

export XR1_AUGUST_TAG=xr1_prospective_matched_fpr_20260815_163700
export XR1_AUGUST_ROOT="$STORAGE_BS/robocasa_rollouts/safe/$XR1_AUGUST_TAG"
export XR1_CAL_ANALYSIS="$STORAGE_BS/robocasa_checkpoints/safe/${XR1_AUGUST_TAG}_calibration"
export XR1_AUGUST_TEST_ANALYSIS="$STORAGE_BS/robocasa_checkpoints/safe/${XR1_AUGUST_TAG}_prospective_test"
export XR1_RUNTIME_BUNDLE="$XR1_CAL_ANALYSIS/runtime_bundle.json"
export XR1_BASELINE_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_offline_baselines_20261017_100820"
export XR1_FUSED_CALIBRATION="$XR1_BASELINE_ROOT/calibrate_fused/baseline_calibration.json"

# Full SHA of the pushed commit that contains the final version of this file.
export XR1_CONFIRM_SHA=<full 40-character SHA>
```

### 2. Dedicated worktree pinned to the preregistered commit

```bash
git -C "$ROBOCASA_SOURCE_REPO" fetch origin main
git -C "$ROBOCASA_SOURCE_REPO" merge-base --is-ancestor "$XR1_CONFIRM_SHA" FETCH_HEAD \
  || echo "STOP: XR1_CONFIRM_SHA is not on origin/main"
export ROBOCASA_REPO="$PROJECT_FS/robocasa_xr1_confirmatory_${XR1_CONFIRM_SHA:0:8}"

if test -e "$ROBOCASA_REPO"; then
  echo "worktree exists: $ROBOCASA_REPO"
else
  git -C "$ROBOCASA_SOURCE_REPO" worktree add --detach "$ROBOCASA_REPO" "$XR1_CONFIRM_SHA"
fi
test "$(git -C "$ROBOCASA_REPO" rev-parse HEAD)" = "$XR1_CONFIRM_SHA" \
  || echo "STOP: worktree HEAD is not the preregistered commit"
test -z "$(git -C "$ROBOCASA_REPO" status --porcelain --untracked-files=no)" \
  || echo "STOP: worktree has local changes"

# Worktrees omit ignored simulator assets; hard-link them without copying.
mkdir -p "$ROBOCASA_REPO/robocasa/models/assets"
cp -aln "$ROBOCASA_SOURCE_REPO/robocasa/models/assets/." "$ROBOCASA_REPO/robocasa/models/assets/"

cd "$ROBOCASA_REPO"
CUDA_VISIBLE_DEVICES="" nice -n 10 "$SAFE_PY" -m unittest \
  tests.test_safe_offline_detector_baselines tests.test_safe_prospective_matched_fpr

export XR1_CONFIRM_TAG=xr1_confirmatory_fused_$(date +%Y%m%d_%H%M%S)
export XR1_CONFIRM_ROOT="$STORAGE_BS/robocasa_rollouts/safe/$XR1_CONFIRM_TAG"
export XR1_CONFIRM_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval/$XR1_CONFIRM_TAG"
mkdir -p "$XR1_CONFIRM_ROOT" "$XR1_CONFIRM_LOG_ROOT"
cp "$XR1_AUGUST_ROOT/tasks_shard0.txt" "$XR1_AUGUST_ROOT/tasks_shard1.txt" \
   "$XR1_AUGUST_ROOT/frozen_tasks.json" "$XR1_CONFIRM_ROOT/"

CUDA_VISIBLE_DEVICES="" nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines \
  synthetic-smoke --output-dir "$XR1_CONFIRM_ROOT/synthetic_smoke"

printf 'export XR1_CONFIRM_SHA=%q\nexport ROBOCASA_REPO=%q\nexport XR1_CONFIRM_TAG=%q\nexport XR1_CONFIRM_ROOT=%q\nexport XR1_CONFIRM_LOG_ROOT=%q\n' \
  "$XR1_CONFIRM_SHA" "$ROBOCASA_REPO" "$XR1_CONFIRM_TAG" "$XR1_CONFIRM_ROOT" "$XR1_CONFIRM_LOG_ROOT" \
  > "$STORAGE_BS/robocasa_rollouts/safe/xr1_confirmatory_fused_latest.env"
df -h "$STORAGE_BS"; df -ih "$STORAGE_BS"
```

Continue only if the tests pass and the smoke's status includes
`confirmatory_verdict`. Stop if available storage is below 150 GiB.

### 3. Exclusion sources and preregistration (before any collection)

Find every prior Xiaomi rollout manifest and the three baseline caches:

```bash
"$SAFE_PY" - "$STORAGE_BS" "$XR1_CONFIRM_ROOT/exclusion_sources.txt" <<'PY'
import json, pathlib, sys

root, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sources = []
for path in sorted((root / "robocasa_rollouts").rglob("manifest.jsonl")):
    with path.open() as stream:
        first = next((line for line in stream if line.strip()), None)
    if first and json.loads(first).get("model_family") == "xiaomi_robotics_1":
        sources.append(path)
baseline = root / "robocasa_checkpoints/safe/xr1_offline_baselines_20261017_100820"
for name in ("cache_development", "cache_calibration", "cache_prospective_test"):
    sources.append(baseline / name / "rollouts.jsonl")
august = root / "robocasa_checkpoints/safe/xr1_prospective_matched_fpr_20260815_163700_prospective_test"
sources.append(august / "event_predictions.jsonl")
output.write_text("".join(f"{path}\n" for path in sources))
for path in sources:
    seeds = []
    for line in path.read_text().splitlines():
        if line.strip():
            seed = json.loads(line).get("environment_seed")
            if seed is not None:
                seeds.append(int(seed))
    print(f"{min(seeds) if seeds else '-':>9} {max(seeds) if seeds else '-':>9} {len(seeds):6d}  {path}")
print("sources:", len(sources))
PY
```

Inspect the printed seed ranges. Then freeze the preregistration. It refuses
to overwrite an existing file and refuses a seed range that overlaps any
source:

```bash
cd "$ROBOCASA_REPO"
mapfile -t XR1_EXCLUSIONS < "$XR1_CONFIRM_ROOT/exclusion_sources.txt"
CUDA_VISIBLE_DEVICES="" nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines preregister \
  --baseline-calibration "$XR1_FUSED_CALIBRATION" \
  --preregistration-doc "$ROBOCASA_REPO/docs/xr1_confirmatory_fused_test.md" \
  --code-commit "$XR1_CONFIRM_SHA" \
  --base-environment-seed 1300007 \
  --num-rollouts-per-task 500 \
  --exclusion-sources "${XR1_EXCLUSIONS[@]}" \
  --output "$XR1_CONFIRM_ROOT/preregistration.json"

sha256sum "$XR1_CONFIRM_ROOT/preregistration.json" | tee "$XR1_CONFIRM_ROOT/preregistration.sha256"
"$SAFE_PY" - "$XR1_CONFIRM_ROOT/preregistration.json" > "$XR1_CONFIRM_ROOT/frozen_inputs.sha256" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
for row in d["frozen_artifacts"]:
    print(f"{row['sha256']}  {row['path']}")
print(f"{d['preregistration_doc_sha256']}  {d['preregistration_doc']}")
PY
sha256sum -c --quiet "$XR1_CONFIRM_ROOT/frozen_inputs.sha256" && echo "frozen inputs OK"
```

Report the `preregistration.json` sha256 before continuing.

### 4. Ports and servers (GPUs 0 and 1)

The recovery ladder uses 10340/10341 and pi0 uses 8120/8121. This test uses
10116/10117. Confirm they are free:

```bash
export XR1_PORT0=10116
export XR1_PORT1=10117
if ss -ltn | grep -qE ":(${XR1_PORT0}|${XR1_PORT1})\b"; then
  echo "STOP: port ${XR1_PORT0} or ${XR1_PORT1} is in use"; ss -ltnp | grep -E ":(${XR1_PORT0}|${XR1_PORT1})\b"
else
  echo "ports free"
fi
nvidia-smi
```

Start the servers (the block re-checks the ports and starts nothing if either
is taken):

```bash
export XR1_SERVER0_LOG="$XR1_CONFIRM_LOG_ROOT/server_${XR1_PORT0}.log"
export XR1_SERVER1_LOG="$XR1_CONFIRM_LOG_ROOT/server_${XR1_PORT1}.log"
cd "$XR1_SAFE_REPO"
if ss -ltn | grep -qE ":(${XR1_PORT0}|${XR1_PORT1})\b"; then
  echo "STOP: ports are in use; servers not started"
else
  CUDA_VISIBLE_DEVICES=0 nohup "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
    --model "$XR1_SAFE_CHECKPOINT" --host 127.0.0.1 --port "$XR1_PORT0" \
    > "$XR1_SERVER0_LOG" 2>&1 &
  export XR1_SERVER0_PID=$!
  CUDA_VISIBLE_DEVICES=1 nohup "$XR1_SERVER_ENV/bin/python" -u deploy/server.py \
    --model "$XR1_SAFE_CHECKPOINT" --host 127.0.0.1 --port "$XR1_PORT1" \
    > "$XR1_SERVER1_LOG" 2>&1 &
  export XR1_SERVER1_PID=$!
fi
printf 'export XR1_PORT0=%q\nexport XR1_PORT1=%q\nexport XR1_SERVER0_PID=%q\nexport XR1_SERVER1_PID=%q\n' \
  "$XR1_PORT0" "$XR1_PORT1" "$XR1_SERVER0_PID" "$XR1_SERVER1_PID" \
  >> "$STORAGE_BS/robocasa_rollouts/safe/xr1_confirmatory_fused_latest.env"

until ss -ltn | grep -q ":$XR1_PORT0\b" && ss -ltn | grep -q ":$XR1_PORT1\b"; do
  date; tail -n 5 "$XR1_SERVER0_LOG" "$XR1_SERVER1_LOG" 2>/dev/null; sleep 10
done
grep -q "Model loaded" "$XR1_SERVER0_LOG" && grep -q "Model loaded" "$XR1_SERVER1_LOG" && echo "both servers loaded"
```

One short smoke rollout per server. Smoke seeds 1290007 and 1290008 lie
outside the preregistered range, and the smoke data are not part of the
cohort:

```bash
export XR1_POLICY_CONFIG='{"source_checkpoint":"XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365","checkpoint_revision":"0d1aa76d0d82debc9b611e4d1e231096434d5be4","crop_ratio":0.95,"observation_history":4,"observation_interval":2}'
cd "$ROBOCASA_REPO"
for shard in 0 1; do
  if test "$shard" = 0; then port=$XR1_PORT0; task=CloseBlenderLid; seed=1290007; else port=$XR1_PORT1; task=TurnOnMicrowave; seed=1290008; fi
  CUDA_VISIBLE_DEVICES=$shard MUJOCO_EGL_DEVICE_ID=$shard \
  "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
    --output-dir "$XR1_CONFIRM_ROOT/server_smoke/server$shard" \
    --tasks "$task" --num-rollouts 1 --seed "$seed" --seed-protocol official_xiaomi \
    --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
    --model-family xiaomi_robotics_1 --policy-name Xiaomi-Robotics-1-RoboCasa365 \
    --checkpoint "$XR1_SAFE_CHECKPOINT" --policy-config "$XR1_POLICY_CONFIG" \
    --host 127.0.0.1 --port "$port" --split pretrain --horizon 20 --replan-steps 16 \
    --record-safe-features --record-actions --no-record-videos \
    --no-record-subtask-trace --max-errors 1
  "$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.validate_atomic_dataset \
    --dataset-dir "$XR1_CONFIRM_ROOT/server_smoke/server$shard"
done
```

Both validators must print `RoboCasa SAFE dataset: VALID` and
`Official SAFE loader compatible: True`.

### 5. Collection: exactly 10/10 primary, keep everything

```bash
cd "$ROBOCASA_REPO"
mapfile -t XR1_TASKS0 < "$XR1_CONFIRM_ROOT/tasks_shard0.txt"
mapfile -t XR1_TASKS1 < "$XR1_CONFIRM_ROOT/tasks_shard1.txt"
export XR1_TEST_ROOT="$XR1_CONFIRM_ROOT/confirmatory_collection"
mkdir -p "$XR1_TEST_ROOT/shard0" "$XR1_TEST_ROOT/shard1"

if ! test -s "$XR1_CONFIRM_ROOT/preregistration.json"; then
  echo "STOP: run preregister (section 3) before collecting"
else for shard in 0 1; do
  if test "$shard" = 0; then port=$XR1_PORT0; tasks=("${XR1_TASKS0[@]}"); else port=$XR1_PORT1; tasks=("${XR1_TASKS1[@]}"); fi
  CUDA_VISIBLE_DEVICES=$shard MUJOCO_EGL_DEVICE_ID=$shard \
  nohup "$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.safe.collect_atomic_rollouts \
    --output-dir "$XR1_TEST_ROOT/shard$shard" \
    --tasks "${tasks[@]}" \
    --num-rollouts 500 --seed 1300007 --seed-protocol official_xiaomi \
    --success-quota 10 --failure-quota 10 --no-retain-only-quota \
    --policy-module robocasa.recovery.xiaomi_robotics_1_policy:make_policy \
    --model-family xiaomi_robotics_1 --policy-name Xiaomi-Robotics-1-RoboCasa365 \
    --checkpoint "$XR1_SAFE_CHECKPOINT" --policy-config "$XR1_POLICY_CONFIG" \
    --host 127.0.0.1 --port "$port" --split pretrain --replan-steps 16 \
    --record-safe-features --record-actions --record-videos \
    --video-height 256 --video-width 384 --video-frame-stride 2 \
    --no-record-subtask-trace --continue-on-error --max-errors 5 \
    > "$XR1_CONFIRM_LOG_ROOT/collection_shard$shard.log" 2>&1 &
  echo "collector shard$shard PID: $!"
done; fi
echo "export XR1_TEST_ROOT=$XR1_TEST_ROOT" >> "$STORAGE_BS/robocasa_rollouts/safe/xr1_confirmatory_fused_latest.env"
```

One-shot monitor (rerun as needed):

```bash
"$XR1_CLIENT_ENV/bin/python" - "$XR1_TEST_ROOT" 10 <<'PY'
import collections, json, pathlib, sys
root, target = pathlib.Path(sys.argv[1]), int(sys.argv[2])
counts = collections.defaultdict(lambda: [0, 0])
errors = 0
for shard in (root / "shard0", root / "shard1"):
    manifest = shard / "manifest.jsonl"
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                counts[row["task_name"]][int(not row["failed"])] += 1
    error_path = shard / "errors.jsonl"
    if error_path.is_file():
        errors += sum(bool(line.strip()) for line in error_path.read_text().splitlines())
print(f"{'TASK':32s} {'SUCCESS':>8s} {'FAILURE':>8s} {'READY':>7s}")
for task, (failures, successes) in sorted(counts.items()):
    print(f"{task:32s} {successes:8d} {failures:8d} {str(min(successes, failures) >= target):>7s}")
print("tasks seen:", len(counts), "errors:", errors)
print("ready:", sum(min(values) >= target for values in counts.values()), "/ 38")
PY
tail -n 5 "$XR1_CONFIRM_LOG_ROOT"/collection_shard*.log
du -sh "$XR1_TEST_ROOT"
```

Continue only when all 38 tasks are ready and both collectors have exited
normally. Then stop the servers (`kill "$XR1_SERVER0_PID" "$XR1_SERVER1_PID"`)
and confirm the ports are released.

### 6. Merge, validate, export, and score

```bash
cd "$ROBOCASA_REPO"
for shard in shard0 shard1; do
  "$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir "$XR1_TEST_ROOT/$shard"
done
export XR1_TEST_MERGED="$XR1_TEST_ROOT/merged_all_retained"
export XR1_TEST_EXPORT="$XR1_TEST_ROOT/official_safe_all_retained"
"$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.merge_atomic_datasets \
  --source-dirs "$XR1_TEST_ROOT/shard0" "$XR1_TEST_ROOT/shard1" --output-dir "$XR1_TEST_MERGED"
"$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.validate_atomic_dataset --dataset-dir "$XR1_TEST_MERGED"
"$XR1_CLIENT_ENV/bin/python" -m robocasa.recovery.safe.export_to_official_safe \
  --dataset-dir "$XR1_TEST_MERGED" --output-dir "$XR1_TEST_EXPORT"

# SAFE trajectories from the frozen runtime copies (GPU 0, after the servers stopped).
export XR1_TEST_SCORE_ROOT="$XR1_TEST_ROOT/frozen_mlp_scores"
mkdir -p "$XR1_TEST_SCORE_ROOT"
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.score_seen_checkpoint_external \
    --export-dir "$XR1_TEST_EXPORT" --safe-repo "$SAFE_REPO" \
    --training-run-dir "$XR1_CAL_ANALYSIS/runtime/seed$seed" \
    --output-dir "$XR1_TEST_SCORE_ROOT/seed$seed" \
    --group confirmatory_test --device cuda
done

# Baseline features with the official loader and the frozen SAFE configuration.
export XR1_SAFE_CONFIG="$XR1_CAL_ANALYSIS/runtime/seed0/config.yaml"
CUDA_VISIBLE_DEVICES="" nohup nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines extract \
  --export-dir "$XR1_TEST_EXPORT" --safe-repo "$SAFE_REPO" --safe-config "$XR1_SAFE_CONFIG" \
  --group confirmatory_test --output-dir "$XR1_CONFIRM_ROOT/cache_confirmatory_test" \
  > "$XR1_CONFIRM_LOG_ROOT/extract_confirmatory.log" 2>&1 &
echo "extract PID: $!"
```

### 7. One frozen evaluation

```bash
cd "$ROBOCASA_REPO"
sha256sum -c --quiet "$XR1_CONFIRM_ROOT/preregistration.sha256" && echo "preregistration unchanged"
sha256sum -c --quiet "$XR1_CONFIRM_ROOT/frozen_inputs.sha256" && echo "frozen inputs unchanged"
CUDA_VISIBLE_DEVICES="" nohup nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.run_offline_detector_baselines confirm \
  --preregistration "$XR1_CONFIRM_ROOT/preregistration.json" \
  --safe-score-root "$XR1_TEST_SCORE_ROOT" \
  --test-cache "$XR1_CONFIRM_ROOT/cache_confirmatory_test" \
  --output-dir "$XR1_CONFIRM_ROOT/confirm" \
  > "$XR1_CONFIRM_LOG_ROOT/confirm.log" 2>&1 &
echo "confirm PID: $!"
```

Any checksum, disjointness, seed-range, or task-coverage error is a stop
condition. Report it; do not edit inputs to make the run pass.

### 8. Verdict table and integrity checks

```bash
"$SAFE_PY" - "$XR1_CONFIRM_ROOT/confirm/analysis.json" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("status:", d["status"], "| counts:", d["counts"]["primary_balanced"], "| tasks:", d["counts"]["tasks"])
print("preregistration sha256:", d["preregistration_sha256"], "| code:", d["code_commit"],
      "| running code matches:", d["code_commit_matches"])
for role, v in d["verdict"].items():
    b = v["recall_delta_bootstrap"]
    print(f"\n{role}: {v['detector']} vs {v['reference']}  ALL PASS = {v['all_pass']}")
    print(f"  (a) recall delta {b['mean']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] -> {v['checks']['a_bootstrap_ci95_lower_above_zero']}")
    print(f"  (b) point gain {v['point_recall_gain']:+.3f} -> {v['checks']['b_point_recall_gain_at_least_0p15']}")
    print(f"  (c) FPR {v['test_fpr']:.3f} Wilson [{v['test_fpr_wilson_95'][0]:.3f}, {v['test_fpr_wilson_95'][1]:.3f}] -> {v['checks']['c_test_fpr_controlled']}")
print(f"\n{'detector':34s} {'TPR':>6s} {'FPR':>6s} {'BA':>6s} {'adj':>6s} {'r@25':>6s}")
for name, row in sorted(d["overall"].items()):
    print(f"{name:34s} {row['true_positive_rate']:6.3f} {row['false_positive_rate']:6.3f} "
          f"{row['balanced_accuracy']:6.3f} {row['missed_failure_adjusted_detection_fraction']:6.3f} "
          f"{row['failure_recall_by_landmark']['0.25']:6.3f}")
for task_type, rows in d["by_task_type"].items():
    print("\n", task_type)
    for name in ("time_only", "staged_safe_time", "fused_safe_time", "fused_knn_cosine_time"):
        row = rows[name]
        boot = d["by_task_type_paired"][task_type].get(f"{name}__vs__time_only")
        delta = "" if boot is None else (
            f"dTPR vs time {boot['bootstrap']['tpr_delta']['mean']:+.3f} "
            f"[{boot['bootstrap']['tpr_delta']['ci95'][0]:+.3f}, {boot['bootstrap']['tpr_delta']['ci95'][1]:+.3f}]")
        print(f"  {name:28s} TPR {row['true_positive_rate']:.3f} FPR {row['false_positive_rate']:.3f} {delta}")
PY

sha256sum -c --quiet "$XR1_CONFIRM_ROOT/preregistration.sha256" && echo "preregistration unchanged after evaluation"
sha256sum -c --quiet "$XR1_CONFIRM_ROOT/frozen_inputs.sha256" && echo "frozen inputs unchanged after evaluation"
```

Report the primary and secondary verdicts, with all three criteria each,
exactly as computed.
