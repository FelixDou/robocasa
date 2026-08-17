# Xiaomi dense subtask-outcome SAFE pilot

This is a developmental five-composite-task experiment. It reuses the exact
Xiaomi rollouts selected for the 39-task SAFE training export, but it never
modifies that source dataset. Recorded environment-step actions are replayed
through RoboCasa's existing ordered semantic predicate evaluator, then each
genuine policy inference receives the outcome of its active semantic subtask.

The stored user-facing label is:

- `subtask_success_label=1`: the active semantic subtask completed;
- `subtask_success_label=0`: the active semantic subtask was the terminal
  failed subtask.

The complementary `failure_label` is stored for SAFE risk training. Future
subtasks that were never attempted do not create samples. Completed earlier
subtasks in a failed composite rollout remain positive.

## Why these five tasks

The initial pilot uses:

| Task | Existing selected outcomes | Semantic stages |
|---|---:|---:|
| `ArrangeBreadBasket` | 25 success / 25 failure | 3 |
| `ArrangeTea` | 27 / 23 | 3 |
| `BreadSelection` | 17 / 33 | 2 |
| `CuttingToolSelection` | 22 / 28 | 2 |
| `GarnishPancake` | 31 / 19 | 3 |

They have relatively balanced parent outcomes and curated ordered predicate
mappings. The optional ten-task expansion adds `DeliverStraw`,
`GetToastedBread`, `KettleBoiling`, `MakeIceLemonade`, and
`WashFruitColander`, but only after the five-task replay gate passes.

## Scientific gate before training

Action replay is not assumed to be deterministic merely because actions were
recorded. Every accepted annotation must satisfy all of the following:

1. the source action artifact exists and contains exactly `num_env_steps`
   actions;
2. environment recreation uses the recorded task, split, Xiaomi seed, and
   reset seed;
3. the replayed terminal success exactly matches the immutable source label;
4. every source SAFE inference step aligns to a replayed environment state;
5. the semantic record passes the existing Subtask-SAFE schema validator;
6. the output contains at least one completed-stage and one failed-stage
   inference sample for every retained task;
7. model splits will be made by parent rollout, never by dense sample.

Any terminal mismatch is an error, not a label to keep. The source features,
actions, videos, and manifest are read-only; annotations live in a separate
output root with source hashes.

## Fresh-session preflight

No Xiaomi policy server is needed because replay uses stored actions.

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export PROJECT_FS=/gs/fs/tga-shinoda/felid
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO="$PROJECT_FS/robocasa"
export XR1_PY="$STORAGE_BS/envs/xiaomi_robotics_1_robocasa365/bin/python"
export XR1_POOL_ROOT="$STORAGE_BS/robocasa_rollouts/safe/xr1_safe_pooled_39tasks_50each_seed0_20260812_221815"
export XR1_SOURCE="$XR1_POOL_ROOT/source_pool_all_2642"
export XR1_LOG_ROOT="$STORAGE_BS/robocasa_logs/eval"

mkdir -p "$XR1_LOG_ROOT"
cd "$ROBOCASA_REPO"
git pull --ff-only origin codex/safe-xiaomi-robotics-1

test -x "$XR1_PY"
test -f "$XR1_SOURCE/manifest.jsonl"
"$XR1_PY" -m robocasa.recovery.safe.replay_dense_subtask_labels --help
```

Resolve the official export report that freezes the exact selected 1,950
source rollout IDs:

```bash
export XR1_SELECTION_REPORT="$($XR1_PY - "$XR1_POOL_ROOT" "$XR1_SOURCE" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2]).resolve()
matches = []
for path in root.rglob("conversion_report.json"):
    try:
        report = json.loads(path.read_text())
    except Exception:
        continue
    if (
        report.get("complete") is True
        and int(report.get("num_rollouts", -1)) == 1950
        and pathlib.Path(report.get("source_dataset", "")).resolve() == source
        and len(report.get("mapping", [])) == 1950
    ):
        matches.append(path)
if len(matches) != 1:
    raise SystemExit(f"Expected exactly one selected export report, found {matches}")
print(matches[0])
PY
)"

echo "XR1_SELECTION_REPORT=$XR1_SELECTION_REPORT"
test -f "$XR1_SELECTION_REPORT"
```

Dry-run the exact five-task selection and require 250 source parents with no
missing action artifact:

```bash
"$XR1_PY" -m robocasa.recovery.safe.replay_dense_subtask_labels \
  --dataset-dir "$XR1_SOURCE" \
  --selection-report "$XR1_SELECTION_REPORT" \
  --output-dir /tmp/ut06746/xr1_dense_subtask_dry_run \
  --tasks \
    ArrangeBreadBasket ArrangeTea BreadSelection \
    CuttingToolSelection GarnishPancake \
  --dry-run \
  | tee /tmp/ut06746/xr1_dense_subtask_dry_run.json
```

The dry-run must print `status: dry_run_valid`, `rollouts: 250`, five tasks,
and `missing_action_artifacts: []`.

## Ten-rollout replay smoke

Replay one source success and one source failure for every task in the
foreground. This exercises environment recreation, action decoding, predicate
tracking, label alignment, and output validation.

```bash
export XR1_DENSE_SMOKE="$STORAGE_BS/robocasa_rollouts/safe/xr1_dense_subtask_5tasks_smoke_$(date +%Y%m%d_%H%M%S)"

cd "$ROBOCASA_REPO"
"$XR1_PY" -u -m robocasa.recovery.safe.replay_dense_subtask_labels \
  --dataset-dir "$XR1_SOURCE" \
  --selection-report "$XR1_SELECTION_REPORT" \
  --output-dir "$XR1_DENSE_SMOKE" \
  --tasks \
    ArrangeBreadBasket ArrangeTea BreadSelection \
    CuttingToolSelection GarnishPancake \
  --successes-per-task 1 \
  --failures-per-task 1
```

Audit the smoke:

```bash
"$XR1_PY" - "$XR1_DENSE_SMOKE" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "annotation_summary.json").read_text())
print(json.dumps(summary, indent=2, sort_keys=True))
assert summary["status"] == "complete"
assert summary["rollouts"] == 10
assert summary["errors"] == 0
assert summary["dense_success_samples"] > 0
assert summary["dense_failure_samples"] > 0
assert len(summary["support"]) == 5
print("XR1 DENSE SUBTASK REPLAY SMOKE: VALID")
PY
```

## Full 250-rollout annotation

Only after the smoke passes, run all exact selected parents. This is CPU
simulator replay; it does not load Xiaomi weights or contact a policy server.

```bash
export XR1_DENSE_ROOT="$STORAGE_BS/robocasa_rollouts/safe/xr1_dense_subtask_5tasks_$(date +%Y%m%d_%H%M%S)"
export XR1_DENSE_LOG="$XR1_LOG_ROOT/$(basename "$XR1_DENSE_ROOT").log"

cd "$ROBOCASA_REPO"
nohup "$XR1_PY" -u -m robocasa.recovery.safe.replay_dense_subtask_labels \
  --dataset-dir "$XR1_SOURCE" \
  --selection-report "$XR1_SELECTION_REPORT" \
  --output-dir "$XR1_DENSE_ROOT" \
  --tasks \
    ArrangeBreadBasket ArrangeTea BreadSelection \
    CuttingToolSelection GarnishPancake \
  > "$XR1_DENSE_LOG" 2>&1 &

export XR1_DENSE_PID=$!
printf 'export XR1_DENSE_ROOT=%q\nexport XR1_DENSE_LOG=%q\nexport XR1_DENSE_PID=%q\n' \
  "$XR1_DENSE_ROOT" "$XR1_DENSE_LOG" "$XR1_DENSE_PID" \
  > "$STORAGE_BS/robocasa_rollouts/safe/xr1_dense_subtask_latest.env"

echo "pid=$XR1_DENSE_PID"
echo "log=$XR1_DENSE_LOG"
echo "output=$XR1_DENSE_ROOT"
```

One-shot monitoring command:

```bash
source "$STORAGE_BS/robocasa_rollouts/safe/xr1_dense_subtask_latest.env"

date
ps -fp "$XR1_DENSE_PID" || true
printf 'annotated rollouts: '
wc -l < "$XR1_DENSE_ROOT/annotation_manifest.jsonl" 2>/dev/null || echo 0
tail -n 30 "$XR1_DENSE_LOG" 2>/dev/null
du -sh "$XR1_DENSE_ROOT" 2>/dev/null
```

## Training decision after annotation

The summary reports parent-level success/failure support for every semantic
stage as well as dense sample counts. Retain only stages with both outcomes in
all parent-grouped partitions. The initial training screen will compare:

1. the old terminal rollout label broadcast to every inference;
2. dense active-subtask outcome labels;
3. dense outcome labels plus task/stage conditioning;
4. within-stage time-only.

It will use one sample per genuine inference, parent-and-stage-balanced loss,
parent-grouped fit/selection/diagnostic splits, and causal 10%, 25%, and 50%
within-stage evaluation. The current opened prospective cohort is development
data only; any positive model result needs a new frozen calibration and a new
seed-disjoint prospective test.
