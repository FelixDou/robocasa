# XR1 common-stage-entry intervention timing study

Status: **implemented; simulator-free checks pass; live replay and scientific
collection remain pending on the cluster**.
Read `docs/cluster_experiment_runbook.md` before any cluster operation.

The first live attempt, `xr1_timing_20261012_192313`, stopped before intervention
collection. Its first ArrangeTea reference matched simulator state, controller,
policy state, observations, semantics, global RNG, counters, and XML exactly;
only one environment generator's state and PCG increment differed. That run
remains failed and preserved. The amended RNG reconstruction below has local
regression coverage and requires a new live sentinel in a new output root.

## Why this is the next experiment

The completed 160-branch operator screen passed every engineering gate but
found task-macro stage completion of 22.5% for continuation and 17.5% for each
intervention. Oracle operator routing added zero completion gain. No parent
benefited on average. The registered decision was
`mechanism_redesign_or_earlier_snapshot_study`.

The first existing snapshot is at target-stage inference prefix 2; the other
is a later fit-only landmark. The result does not establish whether the
operators lack capability or whether useful intervention must occur earlier.
The next experiment tests timing from a shared initial simulator-policy state.
SAFE alarms remain outside this diagnostic.

## Stage 0: provenance, predicate, and replay audit

Preserve the Phase 2, Phase 3B, and operator-screen roots unchanged. Inspect
the current branch/snapshot results, source parent records, snapshot metadata,
and predicate transitions. In particular, distinguish true loss of an ordered
completed-stage prefix from an initially true relational predicate normally
becoming false during progress. The existing `regression` output alone does
not make that distinction.

Recreate the same 10 nominal parents from their frozen environment and
ordinary-policy sampling provenance. Capture complete states at the first
eligible inference boundary of the target-stage visit that led to the
original prefix-2 snapshot. Verify the original prefix-2 reference using complete
policy state (including the latest request and inferred action records),
observations, semantics, simulator, controller, RNG, counters, and model-XML
audits. The source snapshots do not contain the entire original nominal
request/action history; endpoint agreement does not independently audit every
historical request. The new replay stores every request and consumed action
for inspection. Earlier
states must be genuinely captured. A late state cannot be renamed, reset to
an arbitrary pose, or presented as a last-good rewind.

If replay does not reproduce the source, stop and diagnose the mismatch.
Any newly collected cohort must use a new registration and be labeled as
such; it cannot silently replace these paired parents.

The old source has no reset-time global RNG checkpoint. The collector reuses
the source endpoint's global streams and accepts replay only if they remain
unchanged through every nominal step and exactly match the source at the
reference. If replay advances a module-level RNG, collection stops instead
of inventing its missing initial state.

Environment RNGs use an explicit, selective idle-stream reconstruction contract.
`RoboCasaGymEnv.reset(seed=...)` seeds the underlying simulator's `rng`, while
Gymnasium's separate `np_random` property can initialize an entropy-seeded
generator on first access by snapshot capture. This behavior is documented in
the [Gymnasium source](https://gymnasium.farama.org/v1.2.3/_modules/gymnasium/core/).
After the ordinary seeded reset, before any nominal action or entry snapshot,
the collector checks generator topology and identifies PCG64/PCG64DXSM streams
whose fixed increment differs from the source. An ordinary draw advances the
state but does not change that increment; see the
[NumPy PCG64 state documentation](https://numpy.org/doc/1.24/reference/random/bit_generators/pcg64.html).
Only these distinctly initialized streams receive their frozen source state.
They must then remain unchanged after every nominal step. Normally seeded
simulator generators retain their reset states and may advance naturally
during replay; their endpoint must still match the source exactly. Any use of
a reconstructed stream, topology change, or remaining source mismatch rejects
the replay. It never patches an RNG at the endpoint or excludes it from strict
equality. Audits report each generator's reset-state hash, source endpoint
hash, initial equality, reconstruction decision, and number of idle checks.
Reconstructed streams are explicitly labeled; their source endpoint does not
independently record their original reset-time state.

Full snapshot restoration and the final reference audit still require exact
environment RNG equality. The old Phase 2/3B and operator execution modules
are unchanged, preserving their frozen code hashes.

## Scientific allocation

- Same two tasks and 10 parent identities, explicitly development-only.
- One complete common stage-entry snapshot per parent.
- Two frozen continuation-seed schedules per parent.
- Three intervention timings: immediately, after one genuine policy
  inference, and after two genuine policy inferences from that common base.
- Three intervention operators, unchanged from the completed screen:
  `history_reset_replan`, `stage_retry`, `retreat_reset_replan`.
- One shared `continue` trajectory per parent and seed.

Total: **10 × 2 × (1 + 3 × 3) = 200 unique scientific trajectories**.
Repeating `continue` in each timing cell would allocate 240 runs but does not
create additional independent controls. Engineering repeats are additional.

Stage retry retains the same fixed supervisor-supplied task goals and 64-step
duration; this does not validate a deployable stage estimator. Retreat retains
its bounded 24-step sequence and controller compatibility guard. The study
changes timing and the common-base endpoint design, not the operators.

## Pairing and endpoint

For each parent and seed, freeze the complete nominal request/seed schedule.
All intervention regimes follow that seed's identical nominal prefix until
their assigned application time. Record every genuine request and consumed
action; cached environment steps do not count as additional inferences.
After intervention, branches may diverge and remain paired by initial parent
and seed, not by pretending their later states are equal.

Every regime ends at a common absolute environment-step endpoint measured
from stage entry: 480 steps for ArrangeTea and 256 for CuttingToolSelection.
Waiting and scripted actions consume this budget. Equal post-intervention
suffixes would grant delayed regimes more total time and are not the primary
design. Existing operator-screen outcomes cannot be pooled as late controls:
their start states, endpoints, and request schedules differ.

If the target stage completes or becomes ineligible before the assigned
intervention, retain the assignment, log the eligibility reason, and continue
under the frozen no-intervention fallback. Analyze all assigned regimes;
do not select only parents still failing at a later time.

## Sentinel and collection

Use the lowest-reset-index registered parent in each task for the engineering
sentinel, covering all regimes and both seeds, plus exact repeats. Gate only
on provenance, full restore, prefix equality, seed accounting, control reuse,
timing eligibility, endpoints, payload integrity, and zero technical errors.
After it passes, collect every registered parent regardless of sentinel
efficacy. Use a new durable root under
`/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery`; preserve source roots,
partial evidence, and errors.

One immutable registration allocates 220 executions: 200 unique primaries
plus 20 same-seed repeats (all ten regimes on schedule 0 for both sentinel
parents). The sentinel executes 40 primaries plus those 20 repeats. The full
stage reuses that evidence and adds 160 primaries. All files share `full/`;
`sentinel_analysis.json` covers the sentinel and `analysis.json` covers all
220 executions. Science uses only the 200 primaries. Nominal parent replay
requests are additional setup work, excluded from these branch counts.

History reset is audited as an applied operation. Request changes are reported
separately: an immediate reset with empty history can legitimately leave the
first request unchanged. Early scripted termination can legitimately produce
zero policy requests; actions and declared termination are still audited.

## Outcomes and decision

Primary: task-macro active-stage completion within the common endpoint for
each operator/timing regime minus its paired continuation control. Secondary:
completion retained at the endpoint, full-task success, predicate-level and
audited ordered-prefix regression, declared safety events, calls, and latency.
Compare early versus delayed execution of the same operator to test timing.

Average seeds before state-level selection. Keep both tasks fixed and
resample parents, retaining all timings, operators, and seeds together. These
are exploratory estimates from already opened development identities. Report
all nine intervention cells, not only the winner.

Use the existing practical screening guardrails for a candidate cell:
completion and retained-completion gains of at least 10 percentage points,
positive completion signs in both tasks, at least three benefiting parents,
regression increase no greater than five points, and zero declared safety
events. Preserve the old predicate flag as a sensitivity metric if the
ordered-prefix audit changes its interpretation. Selection among timing
cells requires fresh-parent validation before any efficacy claim.

If earlier intervention helps and delayed execution of the same operator
loses benefit, develop that executor and timing rule on a fresh cohort. If
useful operator value remains absent, redesign the recovery mechanism or
scope the contribution to a diagnostic causal benchmark. Do not train a
router or launch a broad surface, flow-guidance, or SAFE-trigger trial on the
basis of these failed operators.

## Run on the cluster

After pulling the implementation, run in a GPU allocation. The launcher checks
the completed negative operator screen, recreates the exact registered model
path under `/tmp/ut06746` when needed, checks live provenance, starts a private
policy server and verifies readiness, then runs the sentinel, full collection,
and paired analysis. It uses one visible GPU, port 10318, disables W&B, and
keeps logs under durable storage. The old operator and Phase 2/3B roots stay
unchanged. No remote experiment has been executed by the local implementation.

```bash
cd /gs/fs/tga-shinoda/felid/robocasa
git pull --ff-only origin main
export XR1_TIMING_GPU=0
export XR1_TIMING_PORT=10318
bash scripts/run_xr1_intervention_timing.sh start
```

By default `XR1_TIMING_SOURCE` is read from `xr1_operator_latest.env` and points
to its `full` directory. To override, export that exact completed `full` path
before `start`. For concurrent runs, set distinct ports and `XR1_TIMING_ROOT`
paths. Every run writes its own `run.env`; the latest file is a convenience.

Monitor and inspect results without a GPU:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_latest.env
bash /gs/fs/tga-shinoda/felid/robocasa/scripts/run_xr1_intervention_timing.sh status
tail -f "$XR1_TIMING_LOG"
cat "$XR1_TIMING_ROOT/full/scientific_analysis.json"
```

After a clean interruption or allocation expiry, resume in a new GPU allocation
using that run's saved environment; the launcher recreates the node-local model
and verifies frozen hashes before reusing completed trajectories:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_latest.env
export XR1_TIMING_GPU=0
bash /gs/fs/tga-shinoda/felid/robocasa/scripts/run_xr1_intervention_timing.sh resume
```

A recorded technical error or provenance/engineering failure stops collection.
Preserve its root and logs for diagnosis; do not delete error files to force
resume. A corrected implementation requires a new registration and output root.

For the initial RNG-only failure, pull the patch and start a new run using the
same source operator screen. Do not resume the old registration:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_20261012_192313/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_TIMING_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_rngfix_$(date +%Y%m%d_%H%M%S)"
export XR1_TIMING_GPU=0
bash "$ROBOCASA_REPO/scripts/run_xr1_intervention_timing.sh" start
```

Implementation: `intervention_timing.py`, `run_intervention_timing.py`, and
`analyze_intervention_timing.py` under `robocasa/recovery/`; launcher under
`scripts/`. Local checks:

```bash
.venv_video/bin/python -m unittest tests.test_intervention_timing tests.test_operator_screen tests.test_phase2_snapshot_replay tests.test_phase3b_training_horizon tests.test_safe_xiaomi_policy
bash -n scripts/run_xr1_intervention_timing.sh
```
