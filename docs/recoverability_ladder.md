# Recoverability ladder

> **v1 pilot results (17 October 2026):**
> [recoverability_ladder_pilot_20261016.md](recoverability_ladder_pilot_20261016.md).
> **v2 pilot results (19 October 2026):**
> [recoverability_ladder_v2_pilot_20261017.md](recoverability_ladder_v2_pilot_20261017.md).
> This document describes **schema 2 (v2)**, which fixes the validity problems
> A–F found there. Schema-1 runs stay readable by `analyze`,
> `calibrate-stages` and `audit-predicates`.
>
> **Next step:** prospective stage-gated re-prompt trial on fresh seeds,
> [stage_gated_reprompt.md](stage_gated_reprompt.md).

## Why

The bit-exact snapshot-replay track (Phase 2/3B, operator screen, timing
pilot) covered 2 tasks and 10 parents. It lost many parents to attrition and
found no usable recovery headroom. The June pi0 study
(`outputs/manual-20260603-recovery-results/SUMMARY.md`) did show a gradient:
an oracle world reset to the last-good subtask recovered 16% of trials,
EEF-only reset 4.8%, and continuing 2%.

The ladder measures where recovery value lives, across many tasks and
parents, with cheap approximate branching. From each failed natural rollout
(parent), branches restart from saved simulator states under escalating
interventions (resample, robot reset, subtask re-prompt, oracle world reset).

It deliberately does **not** use `full_snapshot.py` or the frozen exact-replay
stack ([exact_replay_frozen.md](exact_replay_frozen.md)). Residual
nondeterminism is recorded but never gated on.

Code: `robocasa/recovery/recoverability_ladder.py` (collector, planner,
audit, stage calibration, analyzer), `scripts/run_recoverability_ladder_xr1.sh`
(cluster launcher), `scripts/recoverability_ladder_xr1_pilot.json` (five-task
v2 pilot), `scripts/recoverability_ladder_xr1_expansion.json` (optional five
more tasks), `tests/test_recoverability_ladder.py` (mock env/policy tests).

## What changed in v2

| v1 problem | v2 fix |
|---|---|
| A. CuttingToolSelection `drawer_open` never completes before success | `audit-predicates` flags such predicates from parent records; raw per-predicate values are saved at every snapshot; ladder-level `exclude_predicates` / `required_subtask_overrides` (shared predicates unchanged) |
| B. The stall detector is a timer (fired before success in 31/36 successes) | `stage_overdue`: fires when time in the current ordered stage exceeds the 0.9 quantile of that stage's duration in successful parents (`calibrate-stages`); v1 stall rule as fallback below 3 examples; the rule that fired is recorded |
| C. Ordered progress latches transient truths | `regression` detector (a latched required predicate false for ≥ 16 steps); durable count next to the latched count |
| D. Anchors too far back to measure decay | resample and reprompt branches at stage entry, +4, +8, +16 inferences and detection; time curves in the analyzer |
| E. Small-sample attribution | per-(task, stage) ladder tables, paired contrasts and per-parent cell tables; attribution kept but labelled exploratory |
| F. Budget ceiling (0.5 H) | `budget_mode=remaining_horizon` (H minus branch-start step); default in the v2 pilot |

## Protocol

1. **Parent rollout.** Run the policy naturally up to the official task
   horizon H. At every genuine policy inference (`at_inference_boundary` or an
   empty `action_plan`) keep a snapshot: `sim.get_state().flatten()`, `qpos`,
   `qvel`, the latched and durable ordered counts, and the raw boolean value
   of **every** predicate returned by `get_subtask_progress()` (required,
   transient and excluded ones). One terminal snapshot is also kept. The model
   XML (gzip) and `ep_meta` are saved once per parent; with
   `record_policy_features`, the parent's XR-1 SAFE features too.
2. **Ordered progress.** The env's predicates (`eval_composite_predicates.py`)
   feed an incremental copy of the ordered-prefix logic of
   `build_subtask_trace` (a unit test checks they agree).
   - The ladder's ordered list is the env's `required_predicates`, replaced by
     `required_subtask_overrides[task]` if given, minus
     `exclude_predicates[task]`. Predicate values and task success are never
     changed. Unknown names stop the parent with an error.
   - **Latched count** (v1 `ordered_count`): stages ever completed in order.
     Stages already true at reset now latch at step 0.
   - **Durable count**: latched stages whose predicate is still true at the
     snapshot. Predicates whose stage is in `regression_ignore_stages`
     (default `precondition`, `release`) or listed in
     `regression_ignore_predicates[task]` need not persist and always count.
   - **Regression events**: every time a latched required predicate becomes
     false (start step, end step, eligibility), at step resolution.
3. **Detection.** All detectors are computed and recorded for every parent;
   `detector` picks the one used for branching:
   - `stage_overdue`: at each inference, the time since the current latched
     stage began is compared with `max(q, stage_overdue_min_steps)`, where `q`
     is the `stage_quantile` (0.9) of that stage's durations in successful
     parents of the calibration file. With fewer than `stage_min_examples` (3)
     durations, or no calibration, the v1 rule applies to that stage
     (`stall_fraction × H` steps in the stage). The parent records
     `detection_rule` = `stage_quantile` or `stall_fallback`.
   - `regression`: the first inference at which an eligible regression has
     lasted `regression_min_steps` (16) and is still open.
   - `overdue_or_regression` (v2 default): the earlier of the two; ties go to
     `regression`, which names the undone stage.
   - `stall`, `time`, `end`: unchanged from v1.

   A detector that never fires falls back to the terminal state
   (`terminal_fallback`). `stage_at_detection` is the regressed predicate for
   regression detections and the current latched stage otherwise.
4. **Anchors** (snapshot indices = inferences, all at or before detection):
   `detection`; `last_good` = `stage_entry` (the first inference at or after
   the last ordered completion before detection; the initial state if none);
   `stage_entry_plus_<n>` = stage entry + n inferences, clipped to detection;
   `last_good_minus_2` (v1 only).
5. **Rungs.**

   | rung | branch starts from | robot | prompt | budget clock |
   |---|---|---|---|---|
   | `resample_continue` | anchor state | anchor | task | anchor step |
   | `subtask_reprompt` | anchor state | anchor | current (or regressed) stage until it holds, then task | anchor step |
   | `robot_reset` | detection state | anchor joints (`eef_to_last_good`) | task | detection step |
   | `world_reset` | anchor state (`env_to_last_good`, oracle) | anchor | task | detection step |

   Resample and reprompt branches at earlier anchors are counterfactual
   branch points: "what if the episode had been branched here". Robot and
   world resets are interventions applied at detection, so their clock (and
   remaining time) is the detection time. Reprompt targets come from
   `_resolve_recovery_target` (`reprompt_level=atomic`); after a regression
   the target is the regressed predicate and completes when it is true again.
6. **Cells** (`cells` for failed parents, `control_cells` for success
   controls). v2 pilot, 12 failure cells: `resample_continue` and
   `subtask_reprompt` at `stage_entry`, `+4`, `+8`, `+16` and `detection`, plus
   `robot_reset/last_good` and `world_reset/last_good`. Two cells alias when
   they start from the same state with the same robot state, prompt and
   budget; only one runs and the analyzer resolves the alias. Anchors clipped
   to detection therefore cost nothing. With `fixed_fraction`,
   `world_reset/last_good` is the same intervention as
   `resample_continue/stage_entry`; with `remaining_horizon` it is not (less
   time is left at detection).
7. **Budget.** `budget_mode=fixed_fraction` (v1): `ceil(budget_fraction × H)`
   for every branch. `remaining_horizon`: `H − clock step`, i.e. the time the
   natural rollout had left at the branch's clock, capped at H and at
   `max_budget_steps` (scalar or per task). Branches record `budget_mode`,
   `budget_start_step` and `budget_steps`. Branches stop early on success.
8. **Branch details.** Seeds `stable_seed(parent, rung, anchor, k)` (XR-1
   `sampling_seed_base`, needs the patched server; recorded as
   `seed_control`). Policy context reset per branch (`policy_context=reset`).
   Outcomes: success, steps, latched and durable counts, stages advanced from
   the start state and from detection, target completion, reprompt time,
   restore residuals.
9. **Harm control.** The first `success_controls_per_task` successful parents
   get `control_cells` at the last inference before the median detection
   fraction of the task's failed parents. Every detector's firing on
   successful parents is recorded (false alarms).
10. **Quotas.** A task stops at `failed_parents_per_task` failures plus
    `success_controls_per_task` controls, or `parents_per_task` attempts.
    Other parents are `unbranched`. `parents_only=true` (`--parents-only`)
    runs natural parents with no branching: the input for the audit and the
    stage calibration.

## Predicate audit and the CuttingToolSelection fix (problem A)

`audit-predicates` reads parent records (any schema) and reports, per task
and required predicate:

- `true_at_reset`: raw value true at reset in ≥ 50% of parents;
- `unreached_before_success`: in successful parents, first raw true only at
  the success step (schema 2), or the **first** ordered stage that latched
  only at the success step (`blocker`, schema 1 too), in ≥ 25% of successes,
  for a stage before the task's final non-release stage (the last real stage
  normally completes at success);
- `regresses_after_latch`: false again for ≥ `regression_min_steps` after
  latching; `transient_in_successes` if this happens in ≥ 25% of successful
  parents (the predicate need not persist; candidate for
  `regression_ignore_predicates`).

It writes `predicate_audit.{md,json}` with a suggested config snippet.

On the v1 pilot (latch mode, no raw traces):

| task | predicate | finding |
|---|---|---|
| CuttingToolSelection | `drawer_open` | first stage latched only at success in **7/7** successes → flagged |
| ArrangeBreadBasket | `cabinet_open` | same in **4/9** successes → flagged |
| ArrangeTea | `cabinet_closed` | 6/6, but it is the final real stage → not flagged |
| ArrangeTea, GarnishPancake | — | 1 and 4 failures with every stage latched (problem C) |

In ArrangeBreadBasket the reprompt "Open the cabinet." never beat resampling
for the 5 failures stuck at that stage (v1 per-parent cells), consistent with
partly opened (or one-door-open) cabinets: `is_open` needs every door joint at
0.90.

**Choice: ladder-level exclusion, shared predicates unchanged.** The v2 pilot
sets `exclude_predicates = {"CuttingToolSelection": ["drawer_open"],
"ArrangeBreadBasket": ["cabinet_open"]}`. The CTS stages become
`correct_tool_on_cutting_board → wrong_tool_not_on_cutting_board →
gripper_released`, so detection and reprompts target the tool placement
instead of "Open the drawer.". ABB loses its separate cabinet stage: opening is
part of `bread_in_basket`. The ABB exclusion goes beyond the CTS fix that was
asked for and is a judgement call; to revert it, delete that one line (the
v1 data then calibrates ABB directly).

Changing `eval_composite_predicates.py` instead (for example a lower
`is_open` threshold) would silently relabel every existing SAFE/subtask
dataset that uses these predicates: Subtask-SAFE labels, the dense
subtask-outcome annotations (`docs/safe_xiaomi_dense_subtask.md`, which
already handles this case with `downstream_subtask_observed` evidence) and
recovery targets. That was not done; it would need review first.

The audit only sees what the parents record. Schema-1 runs give the latch
view only; the v2 naturals phase records raw traces, so run the audit on it
before branching (especially for the expansion tasks).

## Stage calibration (problem B)

`calibrate-stages --run-dirs ... --config CONFIG --output FILE` collects, for
each successful parent and each stage of the ladder's ordered list (after the
config's overrides and exclusions), the stage duration: its latch step minus
the previous stage's latch step (0 for the first). Sources:

- `progress_events` when the parent ran with the same ordered list (schema 1
  and 2; schema-1 stages already true at reset count as 0);
- otherwise the latches are recomputed under the new list from the parent's
  predicate trace at snapshot resolution (schema 2 only);
- otherwise the parent is skipped (`required_mismatch_without_predicate_trace`).

So the v1 pilot is a valid source for ArrangeTea, BreadSelection and
GarnishPancake, but not for CuttingToolSelection or ArrangeBreadBasket under
the exclusions (v1 recorded no raw traces), and BreadSelection has only 2 v1
successes (< 3, stall fallback). The **naturals phase** (12 natural parents
per task, no branching, ~2 h) supplies raw traces for all five tasks; build
the calibration from the v1 pilot plus the naturals. The collector stores the
file's SHA-256 in `plan.json`; resume refuses a different file. Thresholds are
applied at collection time with the run's `stage_quantile`,
`stage_min_examples` and `stage_overdue_min_steps`.

Replaying the stage-overdue rule on v1 parents with a v1 calibration
(in-sample, so optimistic) cut false alarms on successful parents from 31/36
(stall) to 11/36, and moved detection later (median detection fraction
ArrangeBreadBasket 0.39, ArrangeTea 0.11, BreadSelection 0.26 (fallback),
CuttingToolSelection 0.73, GarnishPancake 0.76).

## Outputs (per shard directory)

```text
plan.json                frozen config, horizons, budget mode, calibration SHA-256, git commit
parents.jsonl            one row per parent: detectors, rule, stage at detection, anchors,
                         latched/durable counts per snapshot, predicate trace, regressions, cells
branches.jsonl           one row per branch: rung, anchor, start/robot/clock index, budget, outcome
errors.jsonl             per-parent tracebacks (collection stops at max_errors)
status.json              counts and measured seconds per env step
snapshots/<id>.npz       steps, states, qpos, qvel, ordered_count, durable_count, is_inference,
                         predicate_names, predicate_values [snapshots x predicates]
models/<id>.xml.gz       model XML (+ <id>.ep_meta.json)
features/<id>.npz        optional parent SAFE features aligned to inference steps
```

A parent row is written only after all of its branches finish. Each branch
row carries the parent's `attempt_uid`, so branches from a crashed attempt
are ignored and that parent reruns on resume.

## Analysis

`analyze --run-dirs ...` writes `summary.md`, `summary.json` and CSVs
(`parent_cells`, `ladder`, `stage_ladder`, `stage_contrasts`, `time_curve`,
`detector_health`, `decay`, `score_correlations`). `summary.md` contains:

- **Detector health**: per task, how often each detector fired before success
  on successful parents; failures that still made progress after detection;
  failures with every stage latched; detection-rule counts.
- **Progress at detection**: mean latched and durable progress of failed
  parents, and how many have durable < latched.
- **P(success) by cell** (failures, pooled), **harm control** (success
  controls) and **paired contrasts** vs `resample_continue/detection`.
- **Ladder by task and stage at detection**: P(success) per cell with parent
  bootstrap CI, and the paired contrast vs resampling at detection, for each
  (task, stage-at-detection) group.
- **Recoverability over time**: P(success) of resample and reprompt branches
  versus inferences since stage entry (bins 0, 1–4, 5–8, 9–16, >16) and versus
  inferences before detection (0, 1–4, 5–8, 9–16, 17–32, >32), per task and
  pooled (task-stratified), with parent-cluster bootstrap CIs and the mean
  budget per bin. Under `remaining_horizon`, earlier anchors also have more
  time left; the budget column shows by how much.
- **Per-parent cells**: one row per failed parent with P(success) of every
  cell (`R` resample, `P` reprompt, `RR` robot reset, `WR` world reset; `@d`
  detection, `@e` stage entry, `@e+n`, `@lg` last good).
- v1 views kept: decay by world-reset anchor distance, score-vs-recoverability
  Spearman (now with `durable_progress` and `stage_overdue_ratio`; external
  SAFE scores via `--scores-jsonl`), and the **exploratory** first-rung
  attribution.

Each parent's value is the mean of its K branches; CIs resample parents
(stratified by task for pooled rows). Mixed schemas or budget modes are
flagged in `summary.md`; analyze v1 and v2 separately for claims.

## Configs and dry-run estimates

**v2 pilot** (`scripts/recoverability_ladder_xr1_pilot.json`): the five v1
tasks, K=4, 8 failed parents and 4 success controls per task, 12 failure
cells, 4 control cells, `overdue_or_regression`, `remaining_horizon`, the two
exclusions above, base seed 310007 (naturals 320007), SAFE features for
parents only. ArrangeBreadBasket branches are capped at 2200 steps
(`max_budget_steps`): its H=4350 is 2.5× its longest v1 natural success
(1727 steps from reset), so the cap keeps every branch longer than any
observed full success.

Planning anchors (`expected_detection_fraction`,
`expected_stage_entry_fraction`) come from the v1 replay above. One worker
(2 GPUs), 0.053 s per env step (v1 measured), 20 s per reset:

| task | H | parents | branches | env steps | worker-hours |
|---|---:|---:|---:|---:|---:|
| ArrangeBreadBasket | 4350 | 16 | 416 | 0.86 M | 12.7 |
| ArrangeTea | 2250 | 18 | 384 | 0.74 M | 10.9 |
| BreadSelection | 1950 | 13 | 448 | 0.63 M | 9.3 |
| CuttingToolSelection | 1200 | 15 | 448 | 0.32 M | 4.8 |
| GarnishPancake | 2700 | 22 | 448 | 0.58 M | 8.6 |
| **total (expected)** | | 84 | 2144 | 3.12 M | **46.4 (92.8 GPU-h)** |
| upper bound (no aliasing, every branch full budget) | | | 2240 | 3.72 M | 55.2 (110.3 GPU-h) |

Naturals phase: 12 parents per task, 0.12 M steps, about 2.1 worker-hours.
Smoke: about 0.2 worker-hours.

**Does it fit 24 h with at most one resume?** Only just, in expectation.
Two allocations give about 47 usable hours (0.5 h server start-up each). The
branched pilot's expected 46.4 h fits with no margin; naturals plus pilot
(≈ 49 h) do not, and the upper bound (55 h) needs a third allocation. The v1
planner over-predicted the v1 run by 21% (1.41 M planned vs 1.16 M measured
steps), mostly because it did not model aliasing; v2 models aliasing, so do
not count on that margin. Decide after the naturals phase with
`plan --reference-run` (measured anchors):

| variant | expected worker-h | allocations (expected / upper) |
|---|---:|---|
| shipped (ABB cap 2200) | 46.4 | 2 / 3 |
| no ABB cap (full remaining horizon) | 55.3 | 3 / 3 |
| ABB cap 1800 | 44.2 | 2 / 3 |
| `success_controls_per_task: 2` | 44.0 | 2 / 3 |
| `budget_mode: fixed_fraction` (v1 budget, problem F returns) | 35.4 | 2 / 2 |

**Optional expansion** (`scripts/recoverability_ladder_xr1_expansion.json`):
DeliverStraw (0.36), KettleBoiling (0.58), MakeIceLemonade (0.38),
WashFruitColander (0.62) from the planned expansion of
`docs/safe_xiaomi_dense_subtask.md`, plus SetUpCuttingStation (0.64) instead of
GetToastedBread, whose toaster timer and `env.toaster_on` latch live outside
MuJoCo and cannot be restored by snapshots. Rates are fresh XR-1 counts out of
50 (`docs/safe_robocasa_experiment_report.md` §13.4). All five have curated
predicates and MuJoCo-only success. DeliverStraw `drawer_open` and
MakeIceLemonade `fridge_open` are 0.90 `is_open` preconditions on fixtures that
start closed (the ABB pattern): run its naturals and the audit before
branching. Estimate with generic anchors (0.35 H detection, 0.15 H stage
entry): 57.2 worker-hours expected (114.5 GPU-h), 64.5 upper: three
allocations, or two runs of 2–3 tasks each.

## Cluster commands

Read `docs/cluster_experiment_runbook.md` first. XR-1 server/client setup and
patches are in `docs/safe_xiaomi_robotics_1.md`; the server must carry the
`sampling_seed` patch (the launcher checks). One worker = simulator GPU 0 +
XR-1 server GPU 1 on port 10340 (`LADDER_GPU_PAIRS="0:1"`). Outputs go to
`$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_v2_*`, logs to
`$STORAGE_BS/robocasa_logs/eval`. Nothing is written under `/gs/fs`.

Local, after review:

```bash
git push origin main
```

Cluster, fresh terminal (in every allocation):

```bash
module load miniconda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export STORAGE_BS=/gs/bs/tga-shinoda/felid
export ROBOCASA_REPO=/gs/fs/tga-shinoda/felid/robocasa
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_ENABLED=0
export LADDER_GPU_PAIRS="0:1"
export V1_PILOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_pilot_20261016_170336/shard0"
cd "$ROBOCASA_REPO" && git pull origin main
mkdir -p "$STORAGE_BS/robocasa_logs/eval"
```

1. Dry run (no simulator, no GPU) and the v1 audit (latch view):

```bash
bash scripts/run_recoverability_ladder_xr1.sh plan | tail -40
LADDER_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_pilot_20261016_170336" \
  LADDER_AUDIT_RUNS="$V1_PILOT" LADDER_AUDIT_DIR="$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_v1_predicate_audit" \
  bash scripts/run_recoverability_ladder_xr1.sh audit
```

2. Smoke: CuttingToolSelection, 3 parents, 1 failed parent, K=1:

```bash
bash scripts/run_recoverability_ladder_xr1.sh smoke
bash scripts/run_recoverability_ladder_xr1.sh status
```

Pass only if: the server log shows `Model loaded`; there is no `errors.jsonl`;
`status.json` has `measured_seconds_per_env_step`; branches have
`seed_control=policy_sampling_seed_base`, `source_restore_residual` near 0 and
`budget_mode=remaining_horizon`; the CTS parent's `required_subtasks` no
longer contains `drawer_open`.

3. Naturals phase (≈ 2 h): natural parents only, for the audit and the stage
calibration:

```bash
bash scripts/run_recoverability_ladder_xr1.sh naturals
bash scripts/run_recoverability_ladder_xr1.sh status
tail -f "$STORAGE_BS"/robocasa_logs/eval/xr1_ladder_v2_naturals_*_shard0_*.log
```

4. When the naturals run is `complete`: audit, then build the calibration
from the v1 pilot plus the naturals, then re-plan with measured throughput and
anchors:

```bash
source "$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_latest.env"
export NATURALS_ROOT="$LADDER_ROOT"
bash scripts/run_recoverability_ladder_xr1.sh audit
cat "$NATURALS_ROOT/predicate_audit/predicate_audit.md"
LADDER_CALIBRATION_RUNS="$V1_PILOT $NATURALS_ROOT/shard0" \
  LADDER_STAGE_CALIBRATION="$NATURALS_ROOT/stage_calibration.json" \
  bash scripts/run_recoverability_ladder_xr1.sh calibrate-stages
LADDER_CALIBRATION_DIR="$NATURALS_ROOT/shard0" LADDER_REFERENCE_RUNS="$NATURALS_ROOT/shard0" \
  bash scripts/run_recoverability_ladder_xr1.sh plan | grep -A8 '"expected"'
```

Stop and review if the audit flags a predicate the config does not exclude,
if a stage the calibration summary marks `(stall fallback)` matters, or if
`allocations_needed` exceeds 2 (apply a lever from the table above, then
commit the config change locally and pull it).

5. Branched pilot. The calibration file is copied to
`<run root>/stage_calibration.frozen.json`, which resume reuses; `start` refuses to run without one unless
`LADDER_ALLOW_UNCALIBRATED=1`. In a fresh terminal, first
`export NATURALS_ROOT=$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_v2_naturals_<ts>`:

```bash
unset LADDER_ROOT
LADDER_STAGE_CALIBRATION="$NATURALS_ROOT/stage_calibration.json" \
  bash scripts/run_recoverability_ladder_xr1.sh start
source "$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_latest.env" && export PILOT_ROOT="$LADDER_ROOT"
```

6. Monitor; in the next allocation, resume (reruns only unfinished parents,
refuses a changed plan or calibration; use the same `LADDER_GPU_PAIRS` count):

```bash
bash scripts/run_recoverability_ladder_xr1.sh status
tail -f "$STORAGE_BS"/robocasa_logs/eval/xr1_ladder_v2_pilot_*_shard0_*.log
bash scripts/run_recoverability_ladder_xr1.sh resume
```

After an allocation ends, `LADDER_ROOT` comes from
`xr1_ladder_latest.env` (the last smoke/naturals/start). To resume a specific
run, export it first:
`export LADDER_ROOT=$STORAGE_BS/robocasa_rollouts/recovery/xr1_ladder_v2_pilot_<ts>`.

7. Analyze (optionally with offline SAFE scores):

```bash
bash scripts/run_recoverability_ladder_xr1.sh analyze
cat "$LADDER_ROOT/analysis/summary.md"
# LADDER_SCORES_JSONL=/gs/bs/.../safe_scores.jsonl bash scripts/run_recoverability_ladder_xr1.sh analyze
```

8. Optional expansion: same steps with the expansion config and its own
prefix (with `PILOT_ROOT` exported as in step 5); then analyze both runs
together:

```bash
export LADDER_CONFIG="$ROBOCASA_REPO/scripts/recoverability_ladder_xr1_expansion.json"
export LADDER_RUN_PREFIX=xr1_ladder_v2_expansion LADDER_NATURAL_BASE_SEED=340007
bash scripts/run_recoverability_ladder_xr1.sh naturals
# ... audit, calibrate-stages (naturals only), plan, start as in steps 4-6 ...
LADDER_ANALYZE_RUNS="$PILOT_ROOT/shard0 $LADDER_ROOT/shard0" LADDER_ANALYSIS_DIR="$LADDER_ROOT/analysis_combined" \
  bash scripts/run_recoverability_ladder_xr1.sh analyze
```

Do not delete partial shards or logs. A `worker_failure.json` or a nonempty
`errors.jsonl` is a stop-and-inspect condition. Concurrent runs need distinct
`LADDER_PORT_BASE` values (for example 10350).

The collector is policy-agnostic. For pi0 or RLDX, use a config with
`policy_module` set to `robocasa.recovery.openpi_websocket_policy:make_policy`
or `robocasa.recovery.rldx_zmq_policy:make_policy` and run `python -m
robocasa.recovery.recoverability_ladder collect --config ... --output-dir ...
--host 127.0.0.1 --port 8120` against an already verified server, on runbook
ports (`812x` for OpenPI, `2010x` for RLDX).

## Caveats

- **Approximate restore.** Restores reset MuJoCo state only. Controller
  internals, the env step counter, renderer caches and Python-side fixture
  state (toaster, microwave and coffee-machine timers) are not restored.
  Prefer tasks whose success depends on MuJoCo state only.
- **Robot-only reset.** Joint-prefix selection (`robot0_` and the robot model
  prefixes); a held object is not moved with the robot.
- **Calibration is from successes only.** Stage-overdue thresholds describe
  how long successful episodes stay in a stage; a stage rarely visited by
  successes falls back to the stall rule. Calibration in-sample to the same
  run would understate false alarms; the naturals use separate seeds.
- **Remaining horizon.** Under `remaining_horizon`, earlier anchors have more
  time; the time curve measures recoverability with the natural time limit,
  not state quality alone. The ABB cap bounds this for ABB.
- **Durable count exemptions.** Preconditions and releases are exempt by
  stage. Predicates without a `stage` are treated as must-persist.
- **Single-stage tasks.** With `reprompt_level=atomic`, a reprompt on an
  atomic task is the task instruction (`reprompt_equivalent_to_task_prompt`).

## Design choices for review

1. CTS `drawer_open` **and** ABB `cabinet_open` are excluded at ladder level;
   the shared predicate is unchanged. ABB is an extension of the brief based
   on the v1 audit (4/9 successes; reprompt never helped there).
2. Stage-overdue thresholds come from a **separate naturals phase** plus the
   v1 pilot, not from the branched run itself, because v1 has no raw traces
   for the excluded tasks and only 2 BreadSelection successes.
3. Stage time is measured from the latch of the current **latched** stage;
   regressions are a separate detector, and the regressed predicate becomes
   `stage_at_detection` and the reprompt target.
4. Durable count exempts `precondition` and `release` stages (a closed drawer
   after the pick, or a regrasp, is not lost progress).
5. Robot and world resets use the **detection clock** under
   `remaining_horizon` (they are interventions applied at detection); the
   time-curve resample cell at stage entry is the world reset with the
   anchor clock.
6. Success controls run only the 4 intervention cells at detection/last good
   (harm control), not the time curve.
7. ABB branches are capped at 2200 steps to approach the two-allocation
   budget; GetToastedBread is replaced by SetUpCuttingStation in the
   expansion.
8. The planner places anchors at median detection/stage-entry fractions
   (from the v1 replay, or `--reference-run`) and reports allocations needed.

## Local tests

```bash
/path/to/python-with-numpy -m unittest tests.test_recoverability_ladder
```
