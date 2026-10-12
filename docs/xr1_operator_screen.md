# XR1 higher-level recovery-operator screen

Read `docs/cluster_experiment_runbook.md` first. This protocol implements the
operator fallback after the completed negative Phase 3B candidate screen.
It reuses the same 20 exact snapshots, 10 parents and two tasks. It does not
modify or resume any Phase 2/3B source directory.

## Allocation and evidence boundary

The full default allocation is **160 scientific suffixes**:
20 snapshots × four operators × two continuation-seed schedules. Each task
has one supervised target stage and two snapshot boundaries. This is a
developmental capability screen conditional on those two tasks, not a
generalization, end-to-end deployment, or natural-prevalence experiment.

| Operator | Frozen treatment |
|---|---|
| `continue` | Preserve policy context and original instruction. |
| `history_reset_replan` | Clear image/state history and action cache once; preserve sampling configuration, seed schedule and counters; replan under the original instruction. |
| `stage_retry` | Clear context, use a fixed supervisor-supplied subgoal for 64 environment steps, then clear context and return to the original instruction. |
| `retreat_reset_replan` | Open the gripper with zero arm/base command for eight steps, then issue a +0.1 normalized vertical arm command with open gripper for 16 steps; clear context and replan. |

The fixed goals are “Pick the mug from the cabinet and place it on the tray.”
for ArrangeTea and “Open the drawer.” for CuttingToolSelection. No simulator
predicate selects the prompt or decides when to switch it. The operator is
conditioned on a supervisor-supplied goal; this does **not** validate a causal
stage estimator. The retreat is a bounded proposed maneuver, not a claim of
collision-free execution; the engineering sentinel must validate it on the
actual robot/controller stack before scaling.

The action cache is already empty at these snapshots. Therefore the proposed
cache-only `flush_and_replan` arm would duplicate continuation and is replaced
by the explicitly distinct **history-reset** treatment.

The retreat fails closed unless the live right-arm controller exposes a
six-dimensional delta-pose command in the base/world frame. Absolute-pose or
unknown controllers require inspection and a separately registered maneuver.

All four arms receive the same environment-step endpoint: 480 steps for
ArrangeTea and 256 for CuttingToolSelection, including scripted retreat steps.
Every actual policy request receives seed `schedule_base + snapshot_index ×
1000 + request_index`; the two schedule bases are 81000000 and 82000000.
Snapshots keep their global index across sentinel/full scopes. Histories,
instructions and the number of calls can differ after treatment; these are
paired seed schedules, not identical post-treatment trajectories.

## Engineering sentinel and registration

Before the full screen, select the lowest-reset-index eligible parent in each
task and both boundaries: four snapshots. Run both seed schedules for every
operator and duplicate schedule 0 for every operator. This is **48 engineering
executions**, including 16 repeat executions, followed by the 160-suffix full
screen. The sentinel only gates integrity, never recovery efficacy.

Registration freezes the operator specification, source manifests and hashes,
all source snapshot/branch hashes, source Phase 3B decision, execution and
analysis code, snapshot selection, seed schedules, endpoints and allocations.
Baseline simulator/policy/observation restoration is audited before a treatment
can mutate context or take an action. Every branch uses a fresh environment
and renderer and the shared restored policy connection.

The sentinel requires exact repeated request/action/observation/causal-state
sequences, matching semantic outcomes, valid restoration, complete artifacts,
correct per-request seeds and budgets, zero technical errors, and a genuinely
changed first request for history reset versus continuation. Safety and
regression are analyzed as scientific outcomes, not hidden by the integrity
gate. Full collection is blocked if the sentinel fails.

Completed branches are checked and skipped on resume. An atomically written
payload interrupted before its ledger append is integrity-checked and recovered
without overwriting it. Duplicate/unknown IDs, modified registrations, changed
code/specifications, missing payloads, corrupt hashes and changed source files
fail closed. Technical errors remain in `errors.jsonl`; diagnose them and use a
new output root. An allocation interruption without a technical error can resume
the same root. Concurrent writers are blocked by run and collector locks.

## Cluster commands: automatic one-GPU sequence

After the implementation is committed and pushed, in a GPU allocation:

```bash
cd /gs/fs/tga-shinoda/felid/robocasa
git pull --ff-only origin main

# Default GPU 0, new Xiaomi server on port 10316, unique durable output root.
bash scripts/run_xr1_operator_screen.sh start
```

To use a custom bounded specification, set `XR1_OPERATOR_SPEC` to a JSON file
before `start`. The launcher copies it into the new run and registration locks
it; changing the specification after registration prevents resume.

The launcher uses the canonical Xiaomi server/client environments, recreates
the exact node-local checkpoint path from the frozen Phase 2 plan, verifies
checkpoint/server/runtime provenance, checks storage, starts and verifies the
server, runs the engineering sentinel, runs all 20 snapshots only after it
passes, and writes the scientific analysis. W&B is disabled. Logs are in
`/gs/bs/tga-shinoda/felid/robocasa_logs/eval`; results are in
`/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery`.

For a second allocation or GPU, set the GPU and an unused port before starting:

```bash
export XR1_OPERATOR_GPU=1
export XR1_OPERATOR_PORT=10317
bash scripts/run_xr1_operator_screen.sh start
```

An already running server may be reused only after verifying that it serves the
frozen checkpoint with the frozen server entrypoint:

```bash
export XR1_OPERATOR_PORT=10306
export XR1_OPERATOR_REUSE_SERVER=1
bash scripts/run_xr1_operator_screen.sh start
```

Owned servers are stopped when the driver finishes or fails; reused servers
are left running. One stateful server must have only one active simulator
client. The launcher refuses an occupied port unless reuse was requested.

Monitor from a new shell:

```bash
export STORAGE_BS=/gs/bs/tga-shinoda/felid
source "$STORAGE_BS/robocasa_rollouts/recovery/xr1_operator_latest.env"
cd "$ROBOCASA_REPO"
bash scripts/run_xr1_operator_screen.sh status
tail -f "$XR1_OPERATOR_LOG"
```

After an allocation ends, request a new GPU allocation and resume:

```bash
cd /gs/fs/tga-shinoda/felid/robocasa
bash scripts/run_xr1_operator_screen.sh resume
```

For concurrent jobs, select the desired root explicitly with
`export XR1_OPERATOR_ROOT=/gs/bs/.../your_run`; otherwise status/resume selects
the most recently launched run recorded in `xr1_operator_latest.env`.

## Analysis and routing

The primary outcome is active-stage completion at any point within the common
budget. Also report completion retained at the branch endpoint, previous-stage
regression, declared safety events, full-task success, policy calls and latency.
Durability at a suffix endpoint does not establish next-stage or eventual-task
durability. These parents lack eventual terminal-control labels, so eventual
successful-parent harm and deployment failure prevalence remain unestimable.

Each snapshot/operator outcome averages its two seeds. Tasks receive equal
weight. The analysis compares operators against continuation and estimates
oracle state-conditioned selection **after averaging seeds**, relative to the
best fixed deployable operator. Rewind diagnostics never enter these gates.
Parent-then-snapshot bootstrap preserves all arms and seeds together, with tasks
fixed; intervals are developmental and conditional on the two studied tasks.
Best-fixed and oracle routing estimates use the same development data and need
fresh-parent validation before any efficacy claim.

A fixed operator passes when completion and durable-completion gains are each
at least 0.10 task-macro, the completion sign is positive in both tasks, at
least three parents benefit, regression increases by at most 0.05 and no
declared safety events occur. Router development additionally requires at
least 0.10 completion and durable advantage over the strongest fixed operator,
positive task signs, multi-parent support and the same harm guardrails. A
passing fixed operator without routing headroom routes to fresh-parent fixed
operator validation. Otherwise the result routes to mechanism redesign or a
separately registered earlier-snapshot study. These are compute-allocation
gates, not powered confirmatory tests.

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_operator_latest.env
cd "$ROBOCASA_REPO"
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.analyze_operator_screen \
  --run-dir "$XR1_OPERATOR_ROOT/full" \
  --bootstrap-replicates 2000 --bootstrap-seed 0
```

Outputs include `registration.json`, `status.json`, `branch_records.jsonl`,
compressed branch payloads with actions/features/traces, integrity/error
ledgers, `analysis.json`, `scientific_analysis.json`, and branch/snapshot
results in CSV/JSONL. Video rendering is not part of this collector.

## Optional genuine oracle anchors

The existing snapshots do not contain earlier last-good states. The default
therefore has **160**, not 180, scientific executions. Do not substitute the
current snapshot or an arbitrary reset and call it a rewind upper bound.

For an explicitly oracle-assisted diagnostic, the Python CLI accepts
`--oracle-anchors /absolute/anchors.json`. The manifest must cover all 20 target
snapshot IDs and reference trusted earlier complete snapshots of the same
parent, task, environment seed and model XML:

```json
{
  "target_snapshot_id": {
    "path": "/gs/bs/.../earlier_complete_snapshot.pkl.gz",
    "sha256": "64-character compressed-file SHA-256",
    "snapshot_id": "earlier_snapshot_id"
  }
}
```

Supply the same manifest to sentinel and full CLI calls. One
`oracle_full_state_rewind` branch per target adds 20 scientific suffixes. This
restores simulator **and policy** to the supplied earlier anchor and is labeled
accordingly; it is not the old environment-only restore diagnostic and is not
a mathematically guaranteed upper bound. No anchor collection is implemented
or inferred from the available Phase 2 snapshots.

## Local validation

```bash
.venv_video/bin/python -m unittest \
  tests.test_operator_screen \
  tests.test_phase2_snapshot_replay \
  tests.test_phase3b_training_horizon \
  tests.test_safe_xiaomi_policy
bash -n scripts/run_xr1_operator_screen.sh
```

These verify protocol logic and mocked execution. Actual Xiaomi/MuJoCo/EGL
execution remains a cluster sentinel check.
