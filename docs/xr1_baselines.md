# Published baselines on the frozen Xiaomi matched-FPR protocol

`robocasa.recovery.safe.baselines` reimplements published failure detectors on
the same RoboCasa365 / Xiaomi-Robotics-1 rollouts that SAFE uses, and scores
them under the frozen August protocol (same training rollouts, task horizons,
elapsed-time risk curves, balanced calibration rollouts, 5% FPR target and
SAFE/time stage windows). No new collection or policy inference is needed: the
detectors read the stored DiT features and the recorded action chunks.

## Detectors

| Name | Follows | Inputs | Training labels |
| --- | --- | --- | --- |
| `action_consistency` | STAC-Single (SAFE baselines), ACC in VLA-FAIL | Overlap disagreement between consecutive action chunks, EMA | None |
| `actprobe` | ActProbe (Huang et al., 2026) | Chunk consistency, chunk magnitude, executed magnitude (only chunk magnitude when chunks do not overlap), task code; LSTM | Success/fail per rollout |
| `mahalanobis` | Embedding baseline in SAFE; LLMD in VLA-FAIL | Features, PCA + Ledoit-Wolf covariance | Successes only |
| `cosine_knn` | Embedding baseline in SAFE | Raw features, top-5 cosine distance | Successes only |
| `pca_kmeans` | Embedding baseline in SAFE and FAIL-Detect | Features, PCA + 32 k-means centroids | Successes only |
| `rnd` | OOD half of FIPER; RND in FAIL-Detect | Features, random network distillation | Successes only |
| `hide_and_seek` | Hide-and-Seek (Park et al., 2026) | Features (PCA 128); LSTM with peak-margin and proxy-onset losses | Success/fail per rollout |

Not reproducible from stored rollouts: full STAC, FIPER action-chunk entropy,
Tri-Info and ACE need several action samples per inference, and LogpZO needs a
flow density model. Those would need a new multi-sample collection.

Each detector is trained with seeds 0, 1 and 2. Scores are turned into a
running maximum, normalized per task from training maxima and averaged over
seeds, exactly as the SAFE scores in `run_prospective_matched_fpr`.

## Which detectors to run on XR-1

The released XR-1 RoboCasa365 checkpoint predicts 16-step chunks and replans
every 16 steps, so consecutive chunks never overlap. On this layout:

- `action_consistency` is undefined. The runner skips it and records the
  reason under `skipped` in `results.json` and in `summary.md`.
- `actprobe` drops the consistency and executed-magnitude signals (the latter
  equals the chunk magnitude) and keeps chunk magnitude plus the task code.
  `configs[seed].action_signals` records the columns it used.
- `mahalanobis`, `cosine_knn`, `pca_kmeans` and `rnd` were already compared,
  with a development grid, by the offline suite in
  [`xr1_offline_detector_baselines.md`](xr1_offline_detector_baselines.md).
  Rerunning them here only duplicates that result.

So the new XR-1 comparison is `actprobe` and `hide_and_seek`, the two
baselines trained with failure labels. Pass `--feature-aggregation last` so
`hide_and_seek` reads the same per-inference vector as the frozen SAFE MLP
(last flow step, last action position).

## Run on TSUBAME

Read [`cluster_experiment_runbook.md`](cluster_experiment_runbook.md) first.
The job reads stored rollouts only and needs no policy server or GPU. Run it
from a pinned worktree so the shared checkout used by the recovery runs is not
touched.

### 1. Environment, pinned code and frozen inputs

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export SAFE_PY="$STORAGE_BS/envs/vla_safe/bin/python"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0

git -C "$PROJECT_FS/robocasa" fetch origin main
export BASELINES_SHA="$(git -C "$PROJECT_FS/robocasa" rev-parse FETCH_HEAD)"
export BASELINES_REPO="$PROJECT_FS/robocasa_baselines_${BASELINES_SHA:0:8}"
test -d "$BASELINES_REPO" || git -C "$PROJECT_FS/robocasa" worktree add --detach "$BASELINES_REPO" "$BASELINES_SHA"
cd "$BASELINES_REPO"
echo "code: $(git rev-parse HEAD)"

export XR1_FINAL_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/xr1_pooled_39tasks_final_refits_20260814_231508"
export XR1_PMF_TAG=xr1_prospective_matched_fpr_20260815_163700
export XR1_RUNTIME_BUNDLE="$STORAGE_BS/robocasa_checkpoints/safe/${XR1_PMF_TAG}_calibration/runtime_bundle.json"
export XR1_EVAL_IDS="$STORAGE_BS/robocasa_checkpoints/safe/${XR1_PMF_TAG}_prospective_test/event_predictions.jsonl"
export XR1_TRAIN_IDS="$XR1_FINAL_ROOT/indep_seed0/scores.jsonl"
for f in "$XR1_RUNTIME_BUNDLE" "$XR1_EVAL_IDS" "$XR1_TRAIN_IDS"; do test -f "$f" || echo "STOP: missing $f"; done
```

The raw rollout folders are read from the official exports, whose
`conversion_report.json` records each `source_dataset`. The export paths come
from the offline-baseline run (section 3 of
[`xr1_offline_detector_baselines.md`](xr1_offline_detector_baselines.md)):

```bash
source "$STORAGE_BS/robocasa_checkpoints/safe/xr1_offline_baselines_latest.env"
mapfile -t XR1_RAW_DIRS < <(
  for export_dir in "$XR1_DEV_EXPORT" "$XR1_CAL_EXPORT" "$XR1_TEST_EXPORT"; do
    "$SAFE_PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["source_dataset"])' \
      "$export_dir/conversion_report.json"
  done
)
printf '%s\n' "${XR1_RAW_DIRS[@]}"
for d in "${XR1_RAW_DIRS[@]}"; do test -f "$d/manifest.jsonl" || echo "STOP: no manifest in $d"; done

export XR1_BASELINES_TAG=xr1_published_baselines_$(date +%Y%m%d_%H%M%S)
export XR1_BASELINES_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/$XR1_BASELINES_TAG"
export XR1_BASELINES_LOG="$STORAGE_BS/robocasa_logs/eval/$XR1_BASELINES_TAG"
mkdir -p "$XR1_BASELINES_LOG"
sha256sum "$XR1_RUNTIME_BUNDLE" "$XR1_EVAL_IDS" "$XR1_TRAIN_IDS" > "$XR1_BASELINES_LOG/frozen_inputs.sha256"
{
  printf 'export %s=%q\n' BASELINES_REPO "$BASELINES_REPO" XR1_BASELINES_ROOT "$XR1_BASELINES_ROOT" \
    XR1_BASELINES_LOG "$XR1_BASELINES_LOG" XR1_RUNTIME_BUNDLE "$XR1_RUNTIME_BUNDLE" \
    XR1_EVAL_IDS "$XR1_EVAL_IDS" XR1_TRAIN_IDS "$XR1_TRAIN_IDS"
  printf 'XR1_RAW_DIRS=('; printf '%q ' "${XR1_RAW_DIRS[@]}"; printf ')\n'
} > "$XR1_BASELINES_LOG/paths.env"

"$SAFE_PY" -m unittest tests.test_safe_baselines
```

Continue only if the tests pass. If the job is resumed in a new terminal, run
the module and conda lines, then `source "$XR1_BASELINES_LOG/paths.env"`.

### 2. Smoke run

One seed and five epochs. Rollout IDs are checked against the manifests before
any tensor is read, so ID or path mismatches stop the job within seconds.
Loading the tensors still reads every rollout once.

```bash
cd "$BASELINES_REPO"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
nohup nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.run_baselines \
  --dataset-dir "${XR1_RAW_DIRS[@]}" \
  --runtime-bundle "$XR1_RUNTIME_BUNDLE" \
  --train-ids "$XR1_TRAIN_IDS" \
  --evaluation-ids "$XR1_EVAL_IDS" \
  --output-dir "${XR1_BASELINES_ROOT}_smoke" \
  --detectors action_consistency actprobe hide_and_seek \
  --feature-aggregation last \
  --seeds 0 --epochs 5 \
  > "$XR1_BASELINES_LOG/smoke.log" 2>&1 &
echo "smoke PID: $!"
```

Expected in `smoke.log`: `Skipping action_consistency: action_horizon=16 and
replan_steps=16 ...`, then a summary table with `actprobe` and `hide_and_seek`
rows and `evaluation_rollouts` 760 in `results.json`.

### 3. Full run

```bash
cd "$BASELINES_REPO"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
nohup nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.run_baselines \
  --dataset-dir "${XR1_RAW_DIRS[@]}" \
  --runtime-bundle "$XR1_RUNTIME_BUNDLE" \
  --train-ids "$XR1_TRAIN_IDS" \
  --evaluation-ids "$XR1_EVAL_IDS" \
  --output-dir "$XR1_BASELINES_ROOT" \
  --detectors actprobe hide_and_seek \
  --feature-aggregation last \
  --seeds 0 1 2 \
  > "$XR1_BASELINES_LOG/full.log" 2>&1 &
echo "full PID: $!"
```

On a GPU node, drop `CUDA_VISIBLE_DEVICES=""` and add `--device cuda` to
speed up the LSTM training.

```bash
tail -n 20 "$XR1_BASELINES_LOG/full.log"
cat "$XR1_BASELINES_ROOT/summary.md"
sha256sum -c "$XR1_BASELINES_LOG/frozen_inputs.sha256"
```

## Outputs

- `summary.md`: one row per detector variant with ROC-AUC, TPR, FPR, balanced
  accuracy, missed-failure-adjusted detection fraction and recall by 25% of the
  horizon, plus the time-only reference.
- `results.json`: configs, calibration thresholds, calibration and evaluation
  metrics, and paired detection-time comparisons against time-only.
- `evaluation_predictions.jsonl` and `calibration_predictions.jsonl`: per-rollout
  normalized score and time-risk trajectories for every detector, on the
  evaluation and calibration rollouts. (Runs made before the calibration file
  was added, such as the first XR-1 run of 2026-10-22, wrote only the evaluation
  file.)

Each detector is reported twice: alone (`baseline only`) and in your staged
design (`staged baseline time`: the baseline may alarm before 25% of the
horizon, elapsed time after 50%). The second row answers whether the time
fallback helps every detector or only SAFE.

The evaluation cohort was already opened for SAFE, so these numbers compare
methods on a shared test set; they are not a new confirmatory test.

## Lead over the per-task timer P and fusion with time

`scripts/xr1_lead_time/published_baselines_lead.py` scores the baselines of one
run with the two protocols already used for SAFE on the same August cohorts
(calibration 114/114, test 380/380, already opened, so the results are
descriptive):

- **A. Lead over P** (report section 15.16, `physical_signals.analyse`): each
  score fused with the per-task 95th-percentile timer P on the nested (u, s)
  path. It reports the lead gain over P with a task-then-rollout bootstrap CI,
  the gain over P loosened to the same number of test false alarms (matched
  FPR), and a noise null fitted the same way (100 draws by default).
- **B. Fused with the global time risk** (sections 15.10–15.11): the score OR
  the frozen time risk over the whole rollout, thresholds chosen jointly on
  calibration at the 5% target. It reports TPR, FPR, detection timing, and
  paired task-then-rollout bootstrap CIs (2000 replicates, seed 0) against
  time-only and against fused SAFE.

Before writing results it reproduces P alone (lead 0.1952, 17 false alarms),
SAFE + P (+0.0069), fused SAFE + time (0.955 / 0.058) and the run's own
baseline-only rows, and exits with `REPRODUCTION FAILED` otherwise.

It needs `calibration_predictions.jsonl`, so a run made before that file was
added must be repeated. The repeated run also checks determinism:
`--compare-run` reports the largest difference between the two runs'
evaluation trajectories.

### Commands

In a fresh terminal, restore the earlier run's variables, then pin the new code:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export SAFE_PY="$STORAGE_BS/envs/vla_safe/bin/python"

export XR1_BASELINES_LOG="$(ls -d "$STORAGE_BS"/robocasa_logs/eval/xr1_published_baselines_* | grep -v _lead | sort | tail -n 1)"
source "$XR1_BASELINES_LOG/paths.env"
export XR1_FIRST_RUN="$XR1_BASELINES_ROOT"
test -f "$XR1_FIRST_RUN/results.json" && echo "first run: $XR1_FIRST_RUN"

git -C "$PROJECT_FS/robocasa" fetch origin main
export LEAD_SHA="$(git -C "$PROJECT_FS/robocasa" rev-parse FETCH_HEAD)"
export LEAD_REPO="$PROJECT_FS/robocasa_baselines_lead_${LEAD_SHA:0:8}"
test -d "$LEAD_REPO" || git -C "$PROJECT_FS/robocasa" worktree add --detach "$LEAD_REPO" "$LEAD_SHA"
cd "$LEAD_REPO" && echo "code: $(git rev-parse HEAD)"
test -f scripts/xr1_lead_time/published_baselines_lead.py && echo "code OK"

export XR1_LEAD_TAG=xr1_published_baselines_lead_$(date +%Y%m%d_%H%M%S)
export XR1_LEAD_ROOT="$STORAGE_BS/robocasa_checkpoints/safe/$XR1_LEAD_TAG"
export XR1_LEAD_LOG="$STORAGE_BS/robocasa_logs/eval/$XR1_LEAD_TAG"
export XR1_PS_INPUTS="$STORAGE_BS/robocasa_checkpoints/safe/xr1_physical_signals_20261021/inputs"
mkdir -p "$XR1_LEAD_LOG"
sha256sum "$XR1_PS_INPUTS"/calibration_predictions.jsonl "$XR1_PS_INPUTS"/event_predictions.jsonl \
  "$XR1_PS_INPUTS"/runtime_bundle.json "$XR1_RUNTIME_BUNDLE" | tee "$XR1_LEAD_LOG/frozen_inputs.sha256"
{
  printf 'export %s=%q\n' LEAD_REPO "$LEAD_REPO" XR1_LEAD_ROOT "$XR1_LEAD_ROOT" XR1_LEAD_LOG "$XR1_LEAD_LOG" \
    XR1_PS_INPUTS "$XR1_PS_INPUTS" XR1_FIRST_RUN "$XR1_FIRST_RUN"
  cat "$XR1_BASELINES_LOG/paths.env"
} > "$XR1_LEAD_LOG/paths.env"

PYTHONPATH="$LEAD_REPO" "$SAFE_PY" -m unittest tests.test_safe_baselines tests.test_xr1_published_baselines_lead
```

The hashes must start with `a08794b0`, `0e87bb23`, `c4378ba0` and
`c4378ba0`. To resume in a new terminal, run the module and conda lines, then
`source` the newest `xr1_published_baselines_lead_*/paths.env` and `cd "$LEAD_REPO"`.

Repeat the baseline run with the same arguments (about one hour on CPU):

```bash
cd "$LEAD_REPO"
PYTHONPATH="$LEAD_REPO" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
nohup nice -n 10 "$SAFE_PY" -u -m robocasa.recovery.safe.baselines.run_baselines \
  --dataset-dir "${XR1_RAW_DIRS[@]}" \
  --runtime-bundle "$XR1_RUNTIME_BUNDLE" --train-ids "$XR1_TRAIN_IDS" --evaluation-ids "$XR1_EVAL_IDS" \
  --output-dir "$XR1_LEAD_ROOT/baselines" \
  --detectors actprobe hide_and_seek \
  --feature-aggregation last --seeds 0 1 2 \
  > "$XR1_LEAD_LOG/baselines.log" 2>&1 &
echo "baselines PID: $! on $(hostname)"
sleep 60; ps -p $! -o etime,%cpu,rss && tail -n 5 "$XR1_LEAD_LOG/baselines.log"
```

When `$XR1_LEAD_ROOT/baselines/summary.md` exists, compare it with the first
run's `summary.md` and run the analysis:

```bash
cd "$LEAD_REPO"
diff "$XR1_FIRST_RUN/summary.md" "$XR1_LEAD_ROOT/baselines/summary.md" && echo "summary identical to first run"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup nice -n 10 "$SAFE_PY" -I -u scripts/xr1_lead_time/published_baselines_lead.py \
  --baseline-run "$XR1_LEAD_ROOT/baselines" \
  --calibration "$XR1_PS_INPUTS/calibration_predictions.jsonl" \
  --test "$XR1_PS_INPUTS/event_predictions.jsonl" \
  --bundle "$XR1_PS_INPUTS/runtime_bundle.json" \
  --compare-run "$XR1_FIRST_RUN" \
  --workers 8 --noise-draws 100 \
  --out "$XR1_LEAD_ROOT/published_baselines_lead.json" \
  > "$XR1_LEAD_LOG/lead.log" 2>&1 &
echo "lead PID: $! on $(hostname)"
```

When it exits, inspect the result and verify the frozen inputs:

```bash
grep -A40 "=== A." "$XR1_LEAD_LOG/lead.log"
tail -n 3 "$XR1_LEAD_LOG/lead.log"
sha256sum -c "$XR1_LEAD_LOG/frozen_inputs.sha256"
```

The last log line must be `REPRODUCTION OK`.
