# Recoverability ladder

## Why

The bit-exact snapshot-replay track (Phase 2/3B, operator screen, timing
pilot) covered 2 tasks and 10 parents. It lost many parents to attrition and
found no usable recovery headroom. The June pi0 study
(`outputs/manual-20260603-recovery-results/SUMMARY.md`) did show a gradient:
an oracle world reset to the last-good subtask recovered 16% of trials,
EEF-only reset 4.8%, and continuing 2%.

The ladder measures where that recovery value lives, across many tasks and
parents, with cheap approximate branching. Each failed parent is attributed to
one of: stochastic, robot-state, instruction, world-state, or competence
failure.

It deliberately does **not** use `full_snapshot.py` or the Phase 2/3B/timing
replay stack. Residual nondeterminism is recorded but never gated on.

Code: `robocasa/recovery/recoverability_ladder.py` (collector, planner,
analyzer), `scripts/run_recoverability_ladder_xr1.sh` (cluster launcher),
`scripts/recoverability_ladder_xr1_pilot.json` (pilot config),
`tests/test_recoverability_ladder.py` (mock env/policy tests).

## Protocol

1. **Parent rollout.** Run the policy naturally up to the official task
   horizon. At every genuine policy inference (`at_inference_boundary` or an
   empty `action_plan`), keep a snapshot: `sim.get_state().flatten()`, `qpos`,
   `qvel`, and the ordered-subtask count. One terminal snapshot is also kept.
   The model XML (gzip) and `ep_meta` are saved once per parent. With
   `record_policy_features`, the parent's XR-1 SAFE features are saved (float16)
   for later offline scoring.
2. **Ordered progress.** The env's own `get_subtask_progress()` predicates
   (`eval_composite_predicates.py`) feed an incremental copy of the ordered
   prefix logic in `build_subtask_trace`. A unit test checks the two agree.
3. **Detection time (anchor a).** Choose the detector with `detector`:
   - `stall` (default): fires at the first inference after `stall_fraction × H`
     steps with no ordered progress.
   - `time`: fires at `time_fraction × H`.
   - `end`: fires at the timeout state.

   If a detector never fires, it falls back to the terminal state. All three
   detector times are recorded for every parent.
4. **Anchors.** Anchors are restore targets, counted in inferences, and are
   always at or before detection:
   - `detection`
   - `last_good`: the first inference at or after the last ordered-subtask
     completion. If no subtask completed, this is the initial state.
   - `last_good_minus_2`: `anchor_back_inferences` (default 2) inferences
     before `last_good`.
5. **Rungs.** Every branch first restores the detection state, which is where
   an online recovery would actually start.

   | rung | world | robot | prompt |
   |---|---|---|---|
   | `resample_continue` | detection | detection | task |
   | `robot_reset` | detection | anchor (`eef_to_last_good`) | task |
   | `subtask_reprompt` | detection | detection | current ordered subtask until it completes, then task |
   | `world_reset` | anchor (`env_to_last_good`, oracle) | anchor | task |

   State changes go through `recovery_rollout.apply_recovery_mode`, which also
   provides the `evaluate_recovery_benchmark.py` modes. Reprompt targets come
   from `_resolve_recovery_target` (default `reprompt_level=atomic`, as in the
   June study).
6. **Cells.** A cell is a (rung, anchor) pair. The default cells are:
   - `resample_continue/detection`
   - `subtask_reprompt/detection`
   - `robot_reset/{last_good, last_good_minus_2}`
   - `world_reset/{last_good, last_good_minus_2}`

   A world reset to the detection state is the same intervention as
   resampling, so it is not a separate cell. When two cells give the same
   intervention for a parent, only one runs and the other is recorded as an
   alias. For example, if `last_good == detection`, then
   `world_reset/last_good` becomes an alias of `resample_continue/detection`.
   The analyzer resolves aliases, so every cell has an estimate for every
   parent.
7. **Branches.** Each canonical cell runs `k_branches` (default K=4) times.
   - **Seeds.** Each branch gets a fresh sampling seed:
     `stable_seed(parent, rung, anchor, k)`. XR-1 uses it as its
     `sampling_seed_base`, which needs the patched server. pi0/RLDX adapters
     have no seed hook, so their server sampling is fresh but uncontrolled.
     This is recorded as `seed_control`.
   - **Policy context.** By default the policy context is reset at the start of
     each branch (`policy_context=reset`), the same way for every rung. XR-1's
     history is only 7 steps long. `restore` uses `get_state/set_state` when
     the adapter supports it.
   - **Budget.** Every branch gets the same budget,
     `ceil(budget_fraction × H)` steps (default 0.5H), and stops early on
     success.
   - **Outcomes.** Each branch records full-task success, steps used, the
     ordered count at the end, `stages_advanced` relative to detection,
     `target_completed`, reprompt completion time, and restore residuals.
8. **Harm control.** The first `success_controls_per_task` successful parents
   get the same cells. Their branch time is the last inference before the
   median detection fraction of failed parents of that task. If no failed
   parent exists yet, it uses `time_fraction`. If the time detector fires
   inside a success, that is recorded as `detector_fired_on_success`.

Collection stops for a task once it reaches `failed_parents_per_task` failed
parents and `success_controls_per_task` controls, or `parents_per_task`
attempts. Parents beyond those quotas are kept as `unbranched`; they still
contribute to the base success rate.

## Outputs (per shard directory)

```text
plan.json            frozen config, horizons, budgets, git commit (resume refuses changes)
parents.jsonl        one row per completed parent (anchors, detectors, scores, cell schedule)
branches.jsonl       one row per branch (rung, anchor, k, seed, outcome, residuals)
errors.jsonl         per-parent tracebacks (collection stops at max_errors)
status.json          counts and measured seconds per env step
snapshots/<id>.npz   steps, states, qpos, qvel, ordered_count, is_inference
models/<id>.xml.gz   model XML (+ <id>.ep_meta.json)
features/<id>.npz    optional parent SAFE features aligned to inference steps
```

A parent row is written only after all of its branches finish. Each branch row
carries the parent's `attempt_uid`. Branches from a crashed parent attempt are
therefore ignored by the analyzer, and that parent is rerun on resume.

## Analysis

`python -m robocasa.recovery.recoverability_ladder analyze` writes
`summary.md`, `summary.json`, `ladder.csv`, `parent_cells.csv`, `decay.csv`,
and `score_correlations.csv`:

- **Ladder.** For each role (failure, success control), task (and ALL), and
  cell: mean over parents of P(success) and P(stage advance). Each parent's
  value is the mean of its K branches. CIs are a parent-cluster bootstrap,
  stratified by task for ALL.
- **Paired contrasts.** Each cell minus `resample_continue/detection`, paired
  within parent.
- **Attribution.** The first ladder rung with P ≥ `--threshold` (0.5) names
  the failure class. The order is resample → `stochastic`, robot reset at
  last-good → `robot_state`, reprompt → `instruction`, world reset at
  last-good → `world_state`. If none reaches the threshold, the class is
  `competence`.
- **Decay.** P(success) for `world_reset` versus anchor distance in
  inferences, with resample as distance 0, in buckets 0, 1–2, 3–5, 6–10, and
  >10.
- **Score at detection vs recoverability.** Spearman correlation, with a
  parent bootstrap CI, between each score and two targets: P(resample) and the
  parent's best-cell P. Built-in scores are `time` (t/H), `stall` (steps since
  progress / H), and `ordered_progress`. External SAFE scores join with
  `--scores-jsonl` rows `{"parent_id", "snapshot_index", "score", "name"}`.
  Only rows whose `snapshot_index` equals the parent's detection index are
  used. Producing these from `features/*.npz` with a frozen XR-1 SAFE model is
  a separate step.

## Five-task XR-1 pilot

The tasks come from XR-1 natural-rate 50-rollout selections
(`docs/safe_xiaomi_dense_subtask.md`). All five have base success between 20%
and 80% and curated ordered predicates:

| task | expected success | H | branch budget |
|---|---:|---:|---:|
| ArrangeBreadBasket | 0.50 | 4350 | 2175 |
| ArrangeTea | 0.54 | 2250 | 1125 |
| BreadSelection | 0.34 | 1950 | 975 |
| CuttingToolSelection | 0.44 | 1200 | 600 |
| GarnishPancake | 0.62 | 2700 | 1350 |

The pilot config collects up to 24 parents per task and stops at 8 failed
parents plus 2 success controls. It uses the 6 default cells with K=4, the
stall detector at 0.15H, a branch budget of 0.5H, base seed 300007 (separate
from earlier collections), and SAFE features for parents only.

**Dry-run estimate** (`plan`). These numbers assume 0.06 s per environment
step and must be recalibrated from the smoke run:

| | env steps | worker-hours | GPU-hours (2 GPU/worker) | wall (2 workers) |
|---|---:|---:|---:|---:|
| expected (30% of branches succeed at mid-budget) | 1.44 M | 24.5 | 49 | 12.2 h |
| upper bound (no aliasing, every branch full budget) | 1.66 M | 28.2 | 56 | 14.1 h |

Storage is about 4.1 GiB, almost all of it parent SAFE features. Branch steps
are about 90% of the cost. `budget_fraction`, `k_branches`, and
`failed_parents_per_task` are the main levers for cost.

## Cluster commands

Read `docs/cluster_experiment_runbook.md` first. XR-1 server/client setup and
patches are in `docs/safe_xiaomi_robotics_1.md`. The server must carry the
`sampling_seed` patch, and the launcher checks for it.

Local (after review):

```bash
git push origin main
```

Cluster, fresh terminal:

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO=/gs/fs/tga-shinoda/felid/robocasa
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
cd "$ROBOCASA_REPO" && git pull origin main
```

1. Dry run and GPU-hour estimate (no simulator, no GPU):

```bash
LADDER_GPU_PAIRS="0:1 2:3" bash scripts/run_recoverability_ladder_xr1.sh plan
```

2. Smoke test: CuttingToolSelection, up to 3 parents, 1 failed parent, K=1,
   one worker. Use simulator GPU 0, server GPU 1, port 10340:

```bash
LADDER_GPU_PAIRS="0:1" bash scripts/run_recoverability_ladder_xr1.sh smoke
bash scripts/run_recoverability_ladder_xr1.sh status
```

Pass the smoke only if all of these hold:
- the server log shows `Model loaded`;
- `errors.jsonl` is absent;
- `status.json` reports `measured_seconds_per_env_step`;
- `branches.jsonl` has `seed_control=policy_sampling_seed_base` and
  `source_restore_residual` near 0.

3. Recalibrate the estimate with the measured throughput:

```bash
source "$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_latest.env"
LADDER_CALIBRATION_DIR="$LADDER_ROOT/shard0" LADDER_GPU_PAIRS="0:1 2:3" \
  bash scripts/run_recoverability_ladder_xr1.sh plan
```

4. Pilot with two workers. Shards are task-balanced by horizon. Ports are
   10340 and 10341; logs go to `$STORAGE_BS/robocasa_logs/eval`, and outputs to
   `$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_pilot_<ts>/shard{0,1}`:

```bash
unset LADDER_ROOT
LADDER_GPU_PAIRS="0:1 2:3" bash scripts/run_recoverability_ladder_xr1.sh start
```

5. Monitor, and resume after an interruption. Resume refuses a changed plan
   and reruns only unfinished parents. Use the same `LADDER_GPU_PAIRS` count
   as the original start:

```bash
bash scripts/run_recoverability_ladder_xr1.sh status
tail -f "$STORAGE_BS"/robocasa_logs/eval/xr1_ladder_pilot_*_shard0_*.log
LADDER_GPU_PAIRS="0:1 2:3" bash scripts/run_recoverability_ladder_xr1.sh resume
```

6. Analyze all shards. Optionally join offline SAFE scores:

```bash
bash scripts/run_recoverability_ladder_xr1.sh analyze
# LADDER_SCORES_JSONL=/gs/bs/.../safe_scores.jsonl bash scripts/run_recoverability_ladder_xr1.sh analyze
cat "$LADDER_ROOT/analysis/summary.md"
```

Do not delete partial shards or logs. A `worker_failure.json` or a nonempty
`errors.jsonl` is a stop-and-inspect condition.

The collector is policy-agnostic. For pi0 or RLDX, use a config with
`policy_module` set to `robocasa.recovery.openpi_websocket_policy:make_policy`
or `robocasa.recovery.rldx_zmq_policy:make_policy`. Run `python -m
robocasa.recovery.recoverability_ladder collect --config ... --output-dir ...
--host 127.0.0.1 --port 8120` against an already verified server, on runbook
ports (`812x` for OpenPI, `2010x` for RLDX).

## Caveats

- **Approximate restore.** Restores reset MuJoCo state only. Controller
  internals, the env step counter, and renderer caches are not restored.
  `source_restore_residual` and `anchor_restore_residual` quantify the
  simulator part and are reported in `summary.md`.
- **Robot-only reset.** It uses the evaluator's joint-prefix selection
  (`robot0_` and the robot model naming prefixes). A held object is not moved
  with the robot.
- **Single-stage tasks.** For atomic (single-stage) tasks, `subtask_reprompt`
  with `reprompt_level=atomic` reduces to the task instruction. Such branches
  are flagged `reprompt_equivalent_to_task_prompt`.
- **Time detector.** When `detector=time`, detection can precede the last
  progress. Anchors are therefore clipped to progress at or before detection.

## Local tests

```bash
/path/to/python-with-numpy -m unittest tests.test_recoverability_ladder
```
