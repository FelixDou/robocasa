# XR1 common-stage-entry intervention timing study

Status: **live 60-execution sentinel passed; full collection stopped at 80/220
executions on a nominal source-replay mismatch; scientific analysis pending**.
Read `docs/cluster_experiment_runbook.md` before any cluster operation.

The first live attempt, `xr1_timing_20261012_192313`, stopped before intervention
collection. Its first ArrangeTea reference matched simulator state, controller,
policy state, observations, semantics, global RNG, counters, and XML exactly;
only one environment generator's state and PCG increment differed. That run
remains failed and preserved. The amended RNG reconstruction below has local
regression coverage and requires a new live sentinel in a new output root.

The amended run, `xr1_timing_rngfix_20261012_194551`, passed all 60 sentinel
executions and reached 80/220 executions (60 unique primaries plus 20 repeats).
It stopped while reconstructing ArrangeTea parent `f9cd6c2c907171c2f8fea437`
(reset index 4, environment seed 950011), before any intervention trajectory
for that parent. Source snapshot `d6c936ebf70e1dd79eee5559` is at environment
step 400, stage `mug_on_tray`, local inference prefix 2, ordinary inference
index 25, sampling base 6940000. The saved replay stayed at `kettle_on_tray`
through step 399. At the endpoint its environment RNG hash matched the source,
but observation and policy-state hashes did not. This is a trajectory mismatch,
not evidence that the prefix eligibility condition should be relaxed. The
existing collector stops before producing a complete endpoint audit for this
failure; its progress trace and the 80 completed executions remain preserved.
The cause of the trajectory mismatch has not yet been established.

The isolated diagnostic `xr1_timing_replay_20261013_012318` completed two
400-step replays. Every audited initial-state channel matched between attempts,
but both failed to reconstruct the historical prefix. The first recorded
difference was the observation returned after action index 232 (environment
step 233). Policy-state hashes first differed at action index 233; request,
action, and causal-environment hashes first differed at index 240. The first
different request was ordinary call index 15. All 25 paired calls used matching
sampling seeds, and every pair with matching request hashes had matching full
response hashes. Environment RNG hashes matched throughout. Ordered-stage
labels first differed at index 309, and local inference counts at 320.

This places the first observed mismatch in observation generation, before the
next policy inference and physical-trajectory divergence. It does not yet
identify a camera, observable, renderer mechanism, or unrecorded simulator
state, and matching requests do not prove determinism for untested inputs.
The full timing study remains incomplete; inspect the observation path with
the separate recorded-action diagnostic below before another full launch.

The recorded-action run `xr1_timing_observations_20261013_015311` completed
without policy calls. All audited environment channels, model XML, RNG, and
non-camera observations matched throughout the two 400-step repeats. All three
camera streams first differed at environment step 1, affecting over 99.9% of
pixels, with mean absolute RGB differences of approximately 128--139 levels.
The first environment's camera means dropped from approximately 131--169 at
reset to 28--35 after its first step; approximately 56% of pixels became dark,
and the RGB channel means became nearly equal. This dark, nearly achromatic
output persisted at the sampled step 233. The second environment and separate
third probe had matching, normally colored images at the sampled steps 1 and
233. The third environment's four rerenders matched exactly. That probe did
not test the outlying first environment. These results strongly suggest
rendering corruption in this diagnostic, but do not establish its mechanism
or that it is identical to the earlier nominal step-232 observation incident.

The controlled run `xr1_timing_init_order_20261013_071529` completed both
fresh-process cases, each with two 400-step recorded-action replays and a
separate 233-step probe. Startup events showed that the control remained CUDA
uninitialized through reset and RNG restoration; global RNG inspection then
initialized CUDA. Its first environment again had camera differences from
step 1, while audited non-observation channels matched through all 400 steps.
In the early-initialization case, CUDA was initialized before environment
creation, and the two complete repeats had no observation or audited
environment differences. The first early-initialized environment also matched
the control's second, already warmed environment throughout the full trace.
At step 1 its left-camera dark-pixel fraction was 0.06%, compared with 57.35%
for the outlying control. The third probe completed in both cases; the pasted
status excerpt does not include its detailed rerender audit.

Early RNG-runtime initialization therefore removed the first-environment
divergence in this one controlled process pair. This supports a workaround
for the fixed-action diagnostic; it does not isolate the underlying graphics
mechanism or establish that the historical policy-driven prefix is reproducible.
The next check below keeps all source-replay gates intact and runs that
policy-driven replay with early initialization.

The nominal run `xr1_timing_replay_earlycuda_20261013_073123`, launched with
early initialization requested, completed two 400-step replays with 25 nominal
requests each. Neither passed the historical source-prefix audit. The first
ended at `kettle_on_tray`; the second ended at `mug_on_tray` and matched source
subtask evaluation, but still differed in simulator/environment state,
controller, policy, and observations. Both matched source global RNG,
environment RNG, environment control state, and step counter. Both repeated
first-request probes returned identical actions and full responses for seed
6940000 and request hash
`703480442efd9e874629c1bbc56113fb6d5e5f84a4e2d8ac1078cda36ddf7101`.
The subsequent trace scan confirmed CUDA was initialized before **both**
environment creations and all 11 initial repeat-audit channels matched.
Both progress logs contain 400 rows and both request logs contain 25 rows.
The first repeat difference was exclusively `next_observation_sha256` at
action index 233: the returned observation at **environment step 234**.
Observation-input and policy-state hashes first differed at action index 234;
requests, actions, and causal-environment hashes first differed at index 240.
Environment RNG hashes matched throughout. Ordered-stage labels first differed
at index 308 and local inference counts at 320. Request and response hashes
first differed at ordinary call index 15; all sampling seeds matched, and
every paired matching request had a matching full response.

Early initialization therefore did not remove this policy-driven observation
divergence. There is no observed same-request/different-response pair in these
tested calls, but that does not prove server determinism for untested inputs.
The first differing observation key and underlying mechanism are still unknown.
The next diagnostic below retains the policy loop and captures raw observations
and complete audited state around environment step 234. Historical replay
remains unresolved; the earlier fixed-action startup-order result remains
scoped to its controlled pair.

The instrumented run `xr1_timing_replay_window_20261013_121357` completed
another two 400-step, 25-request replays with CUDA initialized before both
environments and all initial repeat channels matching. Both failed the
historical prefix and ended at `kettle_on_tray`. Their final subtask evaluations
matched each other, while environment, integration, controller, policy, and
observation state did not. Both repeated first-request probes passed.

The first within-pair difference moved to action index 246 (environment step
247), exclusively in the returned observation hash. All three camera keys
first differed at environment step 247. The five numeric state-observation
keys first differed at 257. At the first image mismatch the recorded action,
policy-state, request, environment RNG, and causal-environment hashes still
matched. This localizes the first observed discrepancy to camera output under
matching recorded causal-state fingerprints; it does not identify a renderer
mechanism or exclude unrecorded state. The older failed-run comparison instead
reported a causal hash difference at action index 0; that separate cross-run
comparison must not be confused with this pair's step-247 onset.

The two full per-key traces each contain 401 observations. Complete raw states
were retained only at steps 232--236, where all audited channels and camera
statistics matched and the images had normal color. The first mismatch at
247 lies outside that window: its raw pixels and full state audit are absent.
Exact state audits at 232--236 cannot be extrapolated to 247. The next capture
retains raw observations over the entire trace, so another shift in onset will
not prevent measuring the first image difference. Full state capture remains
separately scoped to the requested window.

The full-observation run `xr1_timing_replay_raw_20261013_124815` completed two
400-step, 25-request attempts, with early CUDA initialization confirmed and all
initial repeat channels exact. Both still failed historical prefix replay and
ended at `kettle_on_tray`; both first-request probes passed. Each raw-observation
archive contains all 401 observations. The first difference occurred after
action index 236, at environment step 237, in all three camera streams.
Numeric state observations first differed at step 241. At step 237 every other
recorded progress channel matched, including the action, policy, request,
environment RNG, and causal-environment hashes. Full raw-state audits remain
limited to steps 232--236, where all channels matched.

All 65,536 pixels in each 256-by-256 camera frame differed at step 237, with
full-frame bounding boxes. Mean absolute RGB errors were 85.60 for left,
81.11 for right, and 87.02 for wrist; maximum errors were 255, 255, and 246.
Attempt 0's channel means became nearly equal: left
`[115.62, 117.00, 115.22]`, right `[116.00, 115.41, 114.81]`, and wrist
`[103.61, 103.85, 103.24]`. Attempt 1 retained colored-scene means of
`[169.77, 145.94, 131.23]`, `[170.36, 150.03, 137.27]`, and
`[140.41, 131.09, 129.34]`. Whole-frame disagreement under matching recorded
causal-state fingerprints strongly supports a rendering-path failure. These
statistics do not identify its graphics/context mechanism, establish that every
pixel became grayscale, or prove equality of unrecorded state. The next control
keeps rendering on GPU 0 and moves only the policy server to GPU 1 to test
whether sharing a GPU contributes. The user confirmed a two-GPU allocation is
available. Historical replay remains unresolved.

The separated-GPU run `xr1_timing_replay_splitgpu_20261013_130517` confirmed
server visibility `1`, simulator visibility `0`, and EGL device `0`, with CUDA
initialized before both environments. Both 400-step, 25-request attempts
failed historical prefix replay; the first ended at `kettle_on_tray`, the
second at `mug_on_tray`. Both first-request probes passed and all initial repeat
channels matched. The first within-pair discrepancy remained the returned
camera observations at environment step 237, after action index 236. Every
other progress channel still matched there; numeric state observations first
differed at 241. All pixels in all three frames changed, with mean RGB errors
88.16 (left), 85.74 (right), and 85.96 (wrist). Attempt 1's complete step-237
observation hash exactly matched the healthy attempt-1 observation from the
same-GPU run; attempt 0's corrupted image hashes differed between the runs.
Thus separating the model server from rendering did not eliminate the failure
in this test. It does not rule out every CUDA/GL interaction inside the simulator
process, which still initializes and inspects CUDA RNG.

The next hypothesis concerns delayed renderer cleanup in the simulator process.
The local gym-wrapper constructor resets its underlying environment before
the nominal replay performs another seeded reset. Upstream robosuite's
[`MjRenderContext`](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/master/robosuite/utils/binding_utils.py)
holds a simulator reference, registers itself with that simulator, and frees
graphics resources in its destructor. This makes cleanup of unreachable cycles
a plausible trigger, but it is not evidence that collection caused this
incident or that the cluster's installed destructor matches upstream. The new
diagnostic records the installed cleanup source and collection events, and can
defer automatic cyclic collection until after environment cleanup.

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

### Diagnose one failed nominal parent

Use `scripts/diagnose_xr1_timing_replay.sh` in a GPU allocation for an isolated
diagnostic. It leaves the timing collector, launcher, source registration,
errors, and all completed trajectories unchanged. No recovery branches are
collected and no historical source is replaced.

The diagnostic validates the original registration and live checkpoint/server
provenance, starts a private policy server on port 10319, and reconstructs only
the selected parent twice. Each attempt uses a fresh environment and policy,
matching the original Phase 2 per-parent lifecycle. It captures initial and
endpoint diagnostic state, raw nominal actions/inferences, request/response
hashes, stage transitions, and complete endpoint comparisons even when the
prefix guard rejects replay. A cached endpoint is stored as diagnostic data;
its cache is never cleared to manufacture an eligible branch snapshot. Initial
model XML is compared with the historical source. Initial and endpoint states
and every progress channel are also compared across the two attempts and with
the previously saved failed trace.

After capturing each nominal endpoint, repeat that attempt's first explicitly
seeded request and compare its actions and full response. The extra server
call is labeled as a probe and excluded from nominal request counts. It cannot
change the already captured trajectory. Matching probes do not prove that
every historical model invocation was deterministic. An isolated success
also does not establish why the earlier run failed or permit resuming its
failed registration.

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_rngfix_20261012_194551/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_REPLAY_SOURCE="$XR1_TIMING_ROOT/full"
export XR1_REPLAY_PARENT=f9cd6c2c907171c2f8fea437
export XR1_REPLAY_GPU=0
export XR1_REPLAY_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_replay_$(date +%Y%m%d_%H%M%S)"
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" status
tail -n 50 "$XR1_REPLAY_LOG"
```

The durable root is `xr1_timing_replay_<timestamp>/diagnostic/`, with its own
registration, artifact hashes, and `report.json`. A `diagnostic_complete`
status means the comparisons finished; it does not mean source replay passed
or the timing study is complete. Diagnose the reported divergence before
restarting the full study or considering a separately registered new cohort.

### Isolate observation generation using recorded actions

`scripts/diagnose_xr1_timing_observations.sh` uses the completed nominal
diagnostic's attempt-0 actions. It requires a GPU allocation for rendering, but
starts no model server and loads no policy checkpoint. It validates the frozen
timing registration, source snapshot identity/file hash, diagnostic input
artifact hashes, and all 400 recorded action hashes before creating output.
It never changes the previous diagnostic, timing ledger, errors, or sources.

Two fresh environments execute that identical action list. Each must first
match the recorded attempt's initial environment, observations, semantics,
RNG, and counters exactly under the existing narrow XML compatibility audit.
The diagnostic records every returned observation separately by key, complete
environment-state channels, and raw compressed state/observation envelopes.
It compares camera pixels, changed-pixel counts, error magnitudes, and bounding
boxes at the first mismatch. Model XML hashes are recorded separately from
physical-state hashes; only the model field is removed from the latter, and
XML compatibility is still checked explicitly. No policy state is invented,
no snapshot is made eligible, and no recovery trajectories are collected.

A third fresh environment executes the same actions through environment step
233 (immediately after action index 232), then calls the existing canonical
camera-render path four times without stepping. These extra render calls occur
only in that separate probe environment. Each checks that audited environment
state, semantics, counters, and RNG remain unchanged. The probe prefix is also
compared with the first replay, so a different prefix is visible. The returned
step observation is compared with the first rerender, then consecutive
rerenders are compared. No forced raw-observable refresh is used.

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_20261013_012318/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_OBS_SOURCE="$XR1_REPLAY_ROOT/diagnostic"
export XR1_OBS_GPU=0
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_observations.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_observations_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_observations.sh" status
tail -n 50 "$XR1_OBS_LOG"
```

Outputs live in a new durable `xr1_timing_observations_<timestamp>/diagnostic/`
root. Each environment's `observations.jsonl` contains per-key and state-channel
hashes; `state_<step>.pkl.gz` retains raw frames and full diagnostic state.
The separate probe saves `render_<index>.pkl.gz`. `report.json` contains the
comparisons and an artifact-hash ledger. `diagnostic_complete` means this
diagnostic finished; the timing experiment remains incomplete. Failure writes
`status.json` and preserves partial states and traces.

Camera-only differences with matching audited environment channels isolate the
camera observation path. Differences in state observations point to their
construction or observable caches. Environment differences under identical
actions require investigating simulation or missing causal state. A matching
pair means the incident was not reproduced in those two instrumented replays;
it does not certify a fix or justify restarting the full study. Raw-state
capture adds I/O time, which can affect whether a timing-sensitive renderer
problem recurs. SAFE-trigger recovery remains outside this diagnostic.

### Compare CUDA/RNG-runtime initialization order

The observation diagnostic currently restores and inspects source global RNG
state after reset has already rendered the first observation. Its CUDA RNG
inspection path can initialize CUDA after a renderer already exists;
[PyTorch documents eager CUDA initialization for the RNG getter](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.get_rng_state.html).
Subsequent environments share an already initialized runtime.
The controlled cluster pair described above supports early initialization for
the first-environment failure. Historical nominal prefix replay still requires
the separate policy-driven check below.

Set `XR1_OBS_COMPARE_INIT_ORDER=1` to run a controlled comparison. The launcher
starts the following two cases sequentially, each in a fresh Python interpreter:

- `after_reset`: preserve the existing initialization order.
- `before_env`: call the existing global RNG inspection path before environment
  creation, initializing its optional Torch/CUDA runtime early. This early call
  does not install source RNG state. Both cases keep the original post-reset
  source RNG restoration and exact initial-state audit.

Each case performs the same two 400-action replays and separate step-233
render probe. This requires 2,066 simulator steps across six fresh environments,
with no policy server or checkpoint. Startup events inspect whether Torch is
loaded and CUDA is initialized without themselves importing Torch or
initializing CUDA. Events before and after RNG restoration and inspection
distinguish those two operations. The report checks that the control remained
uninitialized through reset and initialized by RNG inspection, while the early
case initialized before environment creation. It compares first environments
across cases, and compares the early-initialized
first environment with the control's second, already warmed environment. It
also reports first-step camera color/darkness statistics automatically.

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_observations_20261013_015311/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_OBS_COMPARE_INIT_ORDER=1
export XR1_OBS_GPU=0
export XR1_OBS_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_init_order_$(date +%Y%m%d_%H%M%S)"
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_observations.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_observations_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_observations.sh" status
tail -n 50 "$XR1_OBS_LOG"
```

The parent output contains `after_reset/`, `before_env/`, and a combined
`report.json`. Every child retains raw frames, state audits, startup events,
and artifact hashes. A child failure preserves its files and writes a failed
parent status. If the recorded events do not show the intended change in CUDA
initialization order, the report says this hypothesis was not exercised. If
the control reproduces the mismatch, early initialization removes it, and the unchanged environment
and warmed-control comparisons all pass, the conclusion remains scoped to
this one fresh-process pair. It does not certify historical nominal prefix
replay, solve the original step-232 incident, or authorize resuming the failed
timing registration. The frozen timing collector is unchanged.

### Verify policy-driven replay after early initialization

Set `XR1_REPLAY_RNG_RUNTIME_INIT=before_env` for the isolated nominal diagnostic
to call global RNG inspection before loading the policy factory or creating
either environment. This early call does not install source RNG state. Source
restoration, policy requests and sampling seeds, prefix eligibility, and full
endpoint audits still use the existing frozen replay implementation unchanged.
The default remains `after_reset` for control runs. Dry-run validation does not
initialize CUDA, construct environments, or issue policy requests.

The diagnostic records the requested mode, initialization metadata, and startup
events, including CUDA state before each environment is created. A warmup
failure preserves its registration and writes a failed status. It performs
two fresh policy-driven replays of the original ArrangeTea parent, with the
existing post-endpoint request probes. This requires the private policy server
and checkpoint, unlike the recorded-action diagnostic.

Run in a GPU allocation with a new output root and an available private port:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_20261013_012318/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_REPLAY_RNG_RUNTIME_INIT=before_env
export XR1_REPLAY_GPU=0
export XR1_REPLAY_PORT=10320
export XR1_REPLAY_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_replay_earlycuda_$(date +%Y%m%d_%H%M%S)"
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" status
tail -n 50 "$XR1_REPLAY_LOG"
```

Check that CUDA is initialized before both environment creations, the two
complete progress traces match, both initial and endpoint repeat audits are
exact, and **both** attempts report `prefix_replay_accepted: true` against the
historical source. The existing repeated seeded-request probes must also pass.
A repeatable trajectory that still misses the historical endpoint
is a distinct outcome, not a passing source replay. Even a successful isolated
replay leaves the failed full registration preserved and the timing study
incomplete; it provides evidence for a separately registered collector change.

### Capture the policy-driven observation boundary

The early-initialized nominal pair still diverged after action index 233, at
environment step 234. The prior early-initialized recorded-action pair matched
through 400 steps. Capture the difference inside the policy-driven replay
before drawing conclusions from another offline action replay.

Set `XR1_REPLAY_OBS_WINDOW_START=232` and `XR1_REPLAY_OBS_WINDOW_END=236`.
This optional instrumentation retains both original 400-step policy loops,
all source-prefix gates, and the existing post-endpoint request probes. It adds
per-key observation hashes at reset and after every action (401 rows per
complete attempt). At the five selected environment steps it also captures
raw observations, policy state including its real cache, simulator integration,
controller, environment control, semantics, counters, and RNG. These captures
occur after step evaluation and before the next observation is appended to
policy history. Environment step 234 is immediately after action index 233.

Extra hashes and raw states are buffered in memory and written/compressed only
after that attempt's nominal endpoint is captured. There are no extra renders,
policy requests, source restorations, or branch snapshots. The existing
inspection RNG-idle guard remains active. CPU hashing and state inspection
still add work and can affect reproduction of a timing-sensitive fault; a
matching instrumented pair would not establish a fix.

Run in a GPU allocation. The launcher starts and verifies a private server;
the total remains 800 simulator steps, 50 nominal policy requests, and two
post-endpoint request probes for this parent:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_earlycuda_20261013_073123/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_REPLAY_RNG_RUNTIME_INIT=before_env
export XR1_REPLAY_OBS_WINDOW_START=232
export XR1_REPLAY_OBS_WINDOW_END=236
export XR1_REPLAY_GPU=0
export XR1_REPLAY_PORT=10321
export XR1_REPLAY_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_replay_window_$(date +%Y%m%d_%H%M%S)"
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" status
tail -n 50 "$XR1_REPLAY_LOG"
```

Each attempt retains `observation_window/trace.jsonl` and five
`observation_window/state_<step>.pkl.gz` diagnostic envelopes. The report's
`observation_window_comparison` records the first difference by observation
key over the full trace, exact state-channel audits throughout the raw window,
per-camera color/darkness statistics, and pixel differences at the first
observation mismatch when its raw state was captured. The artifact hash ledger
includes these files. If the first mismatch moves outside the selected window,
the full per-key trace still identifies it and the report explicitly marks its
raw state unavailable; window state agreement cannot be extrapolated there.

Camera-only observation differences with matching audited simulator/controller/
RNG state localize the observed failure to the camera path. State-observation
differences instead require investigating their construction or caches.
Physical-state differences at the same boundary require revisiting simulation
and state coverage. None of these diagnostic outcomes completes the timing
study or tests SAFE-trigger recovery.

### Retain raw observations when the first mismatch moves

The 232--236 window missed the next pair's first camera difference at step 247.
Set `XR1_REPLAY_RECORD_ALL_OBSERVATIONS=1` to retain every returned observation
at reset and after all 400 actions. This uses the private copies already made
by the diagnostic step wrapper, retaining them in host memory without another
copy, renderer call, or simulator-state getter. The requested full-state window
can stay at 232--236. Pixel capture no longer depends on that window.

After each nominal endpoint is captured, the diagnostic writes
`observation_window/all_observations.pkl.gz`. Postprocessing validates every
saved observation against the complete per-step hash trace and computes pixel
counts, bounding boxes, RGB errors, and camera color/darkness statistics at the
first differing observation anywhere in the trace. The existing progress
channels at the corresponding action index are compared separately. Outside
the full-state window, `state_audit` remains explicitly unavailable; matching
causal fingerprints exclude only the declared downstream controller caches and
are not promoted to a full state-channel audit. The report distinguishes raw
pixel availability from full raw-state availability.

The option defaults off. It can also run without a full-state window. Each
complete attempt retains 401 raw observations; the attempt report records their
count and array bytes. Compression and additional file writes happen after the
nominal endpoint, and the source-prefix gates, action/request schedule, cache,
and post-endpoint probes remain unchanged. Retaining extra host memory and
instrumentation can still affect whether the incident recurs.

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_window_20261013_121357/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_REPLAY_RNG_RUNTIME_INIT=before_env
export XR1_REPLAY_RECORD_ALL_OBSERVATIONS=1
export XR1_REPLAY_OBS_WINDOW_START=232
export XR1_REPLAY_OBS_WINDOW_END=236
export XR1_REPLAY_GPU=0
export XR1_REPLAY_PORT=10322
export XR1_REPLAY_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_replay_raw_$(date +%Y%m%d_%H%M%S)"
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" status
```

This remains two 400-step policy-driven replays and the existing two
post-endpoint request probes. A matching instrumented pair means the observed
failure was not reproduced; it does not certify historical prefix replay or
complete the timing study.

### Separate the policy-server and rendering GPUs

The current same-GPU run exhibits large image differences before recorded
causal-state or action differences. The launcher now accepts
`XR1_REPLAY_SERVER_GPU`, defaulting to `XR1_REPLAY_GPU` to preserve the existing
layout. `XR1_REPLAY_GPU` controls the simulator process's CUDA visibility, with
logical EGL device 0. Only the policy-server subprocess receives
`XR1_REPLAY_SERVER_GPU`. Registration and report record both visibility masks,
the EGL device, and process IDs; `run.env` preserves the selected server GPU.
This changes no frozen collector, source snapshot, prefix gate, seed, render
path, or policy request schedule.

With two allocated GPUs, keep the simulator on physical GPU 0 and move the
model server to GPU 1. Retain early RNG-runtime initialization, the same raw
observations, and the original full-state window. Use the completed
`xr1_timing_replay_raw_20261013_124815` run as the same-GPU reference:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_raw_20261013_124815/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_REPLAY_GPU=0
export XR1_REPLAY_SERVER_GPU=1
export XR1_REPLAY_RNG_RUNTIME_INIT=before_env
export XR1_REPLAY_RECORD_ALL_OBSERVATIONS=1
export XR1_REPLAY_OBS_WINDOW_START=232
export XR1_REPLAY_OBS_WINDOW_END=236
export XR1_REPLAY_PORT=10323
export XR1_REPLAY_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_replay_splitgpu_$(date +%Y%m%d_%H%M%S)"
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" status
```

The new diagnostic still performs two 400-step replays, 50 nominal requests,
and two existing post-endpoint probes. Verify the logged GPU assignment and
initial repeat audit. If camera divergence persists, splitting server and
renderer did not eliminate the incident in this test. If both complete traces
match, that supports the separated layout as a workaround for the tested pair;
it does not identify the underlying mechanism. Compare identical-request full
response hashes with the same-GPU reference before attributing trajectory
changes to GPU sharing, since the model server now runs on another device.
Repeatability and acceptance against the historical source remain separate:
both source-prefix audits must pass to claim historical replay, and even an
isolated success leaves the failed timing registration preserved.

### Test delayed cyclic cleanup during replay

`XR1_REPLAY_GC_MODE=disabled` disables automatic cyclic garbage collection before
each fresh environment is constructed and restores its previous enabled state
only after the environment and policy close attempts. It does not force a
collection, change source restoration, or modify renderer code. The default
mode traces collections while preserving the caller's GC setting. As described
in [Python's GC documentation](https://docs.python.org/3.11/library/gc.html),
disabling automatic collection does not prevent explicit `gc.collect()` calls;
the diagnostic logs those too. Reference-count destruction is not disabled.

`gc_attempt_<index>.json` retains start/stop events with generation, replay
phase, environment-step position, and the automatic-GC enabled flag. The report
records nominal collection counts and restoration of the previous setting;
status shows events near the first mismatch. An event during stepping is labeled
with the step before that action returns. `renderer_cleanup_source.json`
records the installed binding file path, SHA-256, and renderer destructor source
when available. Reading that source occurs before environment creation; no
dependency is edited. Both files enter the artifact hash ledger.

Keep the tested separated-GPU layout and all raw-observation capture:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_splitgpu_20261013_130517/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_REPLAY_GPU=0
export XR1_REPLAY_SERVER_GPU=1
export XR1_REPLAY_GC_MODE=disabled
export XR1_REPLAY_RNG_RUNTIME_INIT=before_env
export XR1_REPLAY_RECORD_ALL_OBSERVATIONS=1
export XR1_REPLAY_PORT=10324
export XR1_REPLAY_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_replay_gcdisabled_$(date +%Y%m%d_%H%M%S)"
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" status
```

This still performs two 400-step replays and the existing seeded-request probes.
Confirm automatic GC was disabled and inspect any nominal collection events.
If no nominal collections occur and image corruption persists, automatic cyclic
collection is not necessary to produce the failure in that case. If explicit
collections occur, the broader cleanup hypothesis was not fully suppressed.
If both repeats match, this supports a diagnostic workaround; the prior
separated-GPU control did not log GC, so a matched `default`-mode trace and
cleanup/source inspection remain necessary before claiming a mechanism.
Both historical prefix audits still need to pass independently. The failed
timing registration and all prior evidence remain preserved.

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
.venv_video/bin/python -m unittest tests.test_intervention_timing_observation_diagnostic tests.test_intervention_timing_replay_diagnostic tests.test_intervention_timing tests.test_operator_screen tests.test_phase2_snapshot_replay tests.test_phase3b_training_horizon tests.test_safe_xiaomi_policy
bash -n scripts/run_xr1_intervention_timing.sh
bash -n scripts/diagnose_xr1_timing_replay.sh
bash -n scripts/diagnose_xr1_timing_observations.sh
```
