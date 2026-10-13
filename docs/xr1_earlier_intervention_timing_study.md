# XR1 common-stage-entry intervention timing study

Status: **historical endpoint camera corruption confirmed for one source;
the original timing study remains stopped at 80/220 executions; a new
reference cohort is required before the timing comparison can proceed**.
Read `docs/cluster_experiment_runbook.md` before any cluster operation.

Historical-source preflight: **the corrected GC-scope audit accepted 4/10 parents, then
stopped on historical-source disagreement for ArrangeTea parent
`6cfb095ec52c00970abf8340`**. Its two 304-step replays matched each other exactly,
including all 305 observations, and the GC guard passed with zero collections.
The originally failing parent also passed within this run on `r5n4`. The subsequent
fresh-process check reproduced the source mismatch, so it does not require
preceding parents. Offline inspection found identical current request/response
and observation traces across runs. The supplied endpoint comparison now
visually confirms corrupted historical camera images and coherent images in
both current attempts. The saved cross-run initial/endpoint XML audits accept
193 redundant OBJ mesh MIME omissions; this makes their raw hashes differ
without an audited physics difference. Complete intermediate causal states
were not retained, so exact physical agreement at every intermediate step
is not established. Full-study validation remains pending. The next experiment
is the separately registered clean-reference cohort specified below; another
identical historical replay is not the next step.

The first fresh-reference run, `xr1_timing_references_20261013_192435`,
captured its first nominal common entry at step 304 with 19 requests, but
stopped before the second trajectory began: a generic observation helper
returned `None` through the Gym wrapper. Zero parent pairs passed, and camera
health remains unreviewed. The new collector's v2 fix explicitly retrieves
and checks live observations beneath wrappers and audits the complete reset
state before continuation. It requires a new output root; see the
[fresh-reference run instructions](xr1_fresh_timing_references.md).

The v2 run `xr1_timing_references_obsfix_20261013_201241` completed both
first-parent trajectories at step 304 with 19 requests each. Initial restore,
complete initial-state equality, GC, and early-CUDA checks passed. Complete
state, transition/inference, and encoded request/response repeatability gates
failed, leaving zero accepted pairs and no interventions. The supplied status
does not localize the first divergence. Inspect the saved `repeat_audit.json`
and transition traces using the CPU-only command in the fresh-reference
instructions before any additional collection; preserve the failed run.

The saved audit subsequently localized differences to controller/control and
observation state at step 1, policy state at step 2, request/response call row 1,
consumed actions at transition 16, and declared physical state at step 17.
No identical-request/different-response pair was observed. Counter samples
showed 1 versus 0 and 0.05 versus 0.0. A local regression reproduced stale
counter aliases created when the legacy setter writes forwarded wrapper
attributes. The new collector's v3 fix preserves control-attribute ownership;
it does not establish or fix the remaining observation-state discrepancy.
Inspect the saved control paths, controller-buffer identity, all camera keys,
and observable implementation before another GPU run.

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

The GC-disabled run `xr1_timing_replay_gcdisabled_20261013_135416` completed
two 400-step, 25-request attempts. **Both reported `prefix_replay_accepted:
true`**, with no replay error and all 11 source endpoint channels accepted under
the existing audit/XML compatibility contract. Both reached `mug_on_tray` at
the expected inference boundary. Initial and endpoint repeat audits were exact,
every progress field matched across both complete repeats, and all 401 returned
observations matched by every key. Both first-request probes returned exact
actions and full responses. The existing older-failed-trace comparison still
reported a causal hash difference at action index 0; that comparison is separate
from the two accepted historical endpoint audits and the exact current pair.

Automatic GC was disabled at environment creation and through cleanup in both
attempts. No nominal cyclic collection was observed, including explicit
collections, and the caller's original enabled state was restored each time.
The simulator stayed on GPU 0 and server on GPU 1, with early CUDA initialization
confirmed before both environments. The installed renderer source was available
at `robocasa_benchmark_repos/robosuite/robosuite/utils/binding_utils.py`, with
SHA-256 `c8463cddb558f22247b1c8f202443ddeabe1469e5433b5b9d0eb7b938f32f480`.
Its destructor calls `self.con.free()` before `self.gl_ctx.free()` and contains
no call to make its own GL context current first. This is consistent with the
delayed-cleanup hypothesis; the supplied evidence does not identify which
object was finalized at the earlier camera failure.

This is the first isolated two-repeat success against the failed historical
parent, not a completed timing study or a SAFE-trigger recovery result. The
next check restores default GC with the same current diagnostic code, tracing
and GPU layout. That matched control is required before attributing the
successful replay specifically to suppressing automatic cyclic collection.

The same-code default-mode run `xr1_timing_replay_gccontrol_20261013_155951`
completed both 400-step attempts but failed both historical prefix audits.
Initial repeat channels matched and both seeded first-request probes passed.
Attempt 0 recorded a generation-2 collection of 17 objects during the progress
callback labeled environment step 194. Its next returned camera observation,
at environment step 195 (after action index 194), was the first repeat mismatch.
The contemporaneous action, policy, request, environment RNG, and causal-state
hashes still matched. Numeric state observations first differed at step 209.
Camera mean RGB errors were 97.83 (left), 104.55 (right), and 83.93 (wrist),
with almost all pixels changed. Attempt 1 had no generation-2 event in the
displayed step-193--195 neighborhood. Total nominal collection stops were
1,151 and 1,146. The onset moved earlier than previous controls, consistent
with instrumentation changing collection timing; this is not a fixed-step
task failure.

Together with the zero-collection, two-repeat source acceptance under disabled
GC, this strengthens the delayed cyclic-cleanup hypothesis and supports a
bounded GC workaround. It does not identify the finalized object or prove the
exact GL resource deletion mechanism. The disabled-GC result was pasted from
`r2n5`, while the default-GC control came from `r5n4`: code/configuration were
matched, but node hardware/runtime was not held fixed. New diagnostics record
the client hostname so that this distinction stays visible. The next stage
audits all ten original parents using the passing diagnostic lifecycle on the
current allocation, before changing the frozen timing collector or collecting
interventions.

The all-parent run `xr1_timing_parent_audit_20261013_164436` stopped on its first
parent, `b1455183d9fabd7eeb4c4bf5` (ArrangeTea, seed 950007, reset index 0),
on node `r5n4`. Both 336-step attempts passed the historical source-prefix
audit, all 11 endpoint channels, initial/endpoint repeat audits, complete
progress comparison, and both seeded full-response probes. All 337 observations
matched by every key and raw hash. Early CUDA initialization was confirmed and
automatic GC was disabled at environment creation and at the end of cleanup,
with the caller's original setting restored afterward.

The sole failed gate was `gc_control_exercised_without_nominal_collections`:
each attempt recorded one collection stop during the v1 nominal scope. That
scope began at `nominal_setup`, before the frozen collector resets the
environment. The initial status omitted events because the repeats had no
divergence, so the count alone did not locate the event within the active
trajectory. The status display was extended to show failed-parent gates and
nominal collection stops even for matching repeats, without running a simulator
or changing saved evidence. The original cohort remains failed with zero
accepted parents and zero intervention trajectories.

Subsequent saved-event inspection located both collection stops at
`nominal_setup`, environment step 0, generation 2, with automatic GC disabled.
They collected 6,186 and 7,617 objects respectively, with zero uncollectable
objects. This phase precedes the first post-reset subtask evaluation: policy
reset, environment reset, and source RNG setup run there. Neither trace
contained a collection during the active trajectory. The v1 audit's zero-GC
failure therefore came from including reset/setup in its nominal scope; it was
not an observed repeatability or historical-source mismatch.

The v2 all-parent audit records
`gc_nominal_scope: first_post_reset_subtask_evaluation_through_source_endpoint_audit`
in its registration and requires that scope in each child report and GC trace.
Reset/setup events remain recorded and have a separate count; they are no
longer counted as active-rollout collections. A collection during the initial
post-reset evaluation or first action still fails, even at environment step 0.
Every recorded collection must also show automatic GC disabled. Existing source,
observation, request, and repeat gates are retained. The original failed audit,
its recorded counts, and all artifacts stay unchanged; the correction requires
a new root and does not retroactively accept its first parent. It does not
change collection behavior, renderer code, or the frozen timing collector.

The corrected run `xr1_timing_parent_audit_gc_scope_20261013_171134` accepted
four parents on `r5n4`: ArrangeTea `b1455183d9fabd7eeb4c4bf5`,
CuttingToolSelection `f56bbdc19d00beb376c8a1b3`, ArrangeTea
`3094d859730a906172f552ef`, and the previously failing ArrangeTea
`f9cd6c2c907171c2f8fea437`. Each accepted parent passed both source-prefix
audits and all repeat, observation, request-probe, runtime, and GC gates.
This confirms acceptance of the previously failing parent on the same node as
the earlier default-GC failure, while the exact renderer mechanism remains
unproven.

The fifth attempted parent, `6cfb095ec52c00970abf8340` (ArrangeTea, seed 950012,
reset index 5, snapshot `64a6f92ef6c5730f2856e421`), failed only
`both_source_prefix_audits_accepted`. Both attempts completed 304 steps and 19
nominal requests, reached `mug_on_tray` at the required inference boundary,
and reported no replay exception. Initial and endpoint repeats matched in all
11 channels; complete progress and all 305 per-key/raw observations were exact.
Both seeded first-request probes passed. No reset/setup or active-rollout
collections occurred, and early CUDA initialization was confirmed. Relative to
the historical source, both endpoints differed in environment state, simulator
integration state, controller state, policy state, and observation. Semantics,
RNG, control state, tracker state, and the step counter matched. The supplied
status contains channel flags but no numeric mismatch magnitudes.

This is a repeatable current trajectory that misses the historical reference,
not a collection-scope failure. Keep the four accepted records and the failed
child artifacts unchanged; do not drop this parent, relax exact source gates,
or resume the failed cohort. Historical snapshots store the endpoint and latest
policy records, not the complete nominal request/action history, so the first
historical divergence cannot be localized from this output. The next
608-step check runs only this parent twice in a new simulator process with a
fresh private policy server, retaining the same seeds, checkpoint, GPU mapping,
early initialization, GC policy, and full observation capture. Its outcome
tests dependence on prior execution history in the client and/or server; it
does not by itself isolate one process or establish a specific cause. Status
now displays bounded samples from the saved source-endpoint mismatches and
latest policy-record mismatches when a source audit fails.

The isolated run `xr1_timing_replay_6cf_isolated_20261013_174013` completed on
the same node, `r5n4`, using fresh simulator and policy-server processes. Both
304-step, 19-request attempts again failed historical source agreement while
matching each other in every initial/endpoint channel, complete progress, and
all 305 per-key/raw observations. Both first-request probes passed. Early CUDA
initialization preceded both environments, and neither setup nor active replay
recorded a collection. The mismatch therefore does not require running the
four preceding parents in the current process. Exact cross-run trace comparison
with the cohort remains an offline check, rather than an inference from equal
endpoint error magnitudes.

Both isolated attempts reported historical endpoint qpos error 0.4411556,
qvel error 0.4923724, control-input error 78.33905, and qacc-warmstart error
22.01044. Base-position and base-rotation observation errors were approximately
0.08606 and 0.17164. Retained camera arrays differed by up to 255 intensity
levels. Latest inferred actions differed by 1.99609375 and features by 13.75,
but their request hashes also differed: historical
`3ed58e0a76c0f1fa788b6f26185089430fac5ccbd0ecd3d144f78e38d75da81c`, replay
`3a54bbe5f32c4f7da014ba140f81699896ce37bce19921cc8f599ee34e94604b`.
These are substantial state/trajectory differences. The action/feature mismatch
does not demonstrate nondeterministic responses to identical historical inputs.
A maximum pixel error alone cannot distinguish graphics corruption from a
different physical scene or viewpoint.

The next check uses only saved data: verify the diagnostic artifacts and frozen
source file, compare the complete current cohort/isolated traces, and inspect
historical endpoint cameras and retained image queues beside replay images.
`inspect_intervention_timing_source.py` performs this CPU-only inspection,
exports native RGB comparison panels and per-frame statistics, and reports
saved latest request/inference identities. It does not reconstruct a historical
encoded request from later endpoint queues, infer unavailable frame timestamps,
certify image health from statistics, change source eligibility, or collect new
nominal/intervention trajectories. The historical first divergence remains
unlocalized without a full historical nominal trace. Preserve the four accepted
cohort records and both failed source comparisons.

The offline run `xr1_timing_source_inspection_20261013_181125` accepted the
registered source file integrity and verified 305 raw observations in each
isolated attempt. Both corresponding cohort-versus-isolated comparisons had
exact initial and endpoint state audits, identical 19-row request/response
traces, and identical complete 305-row per-key observation traces. There were
no identical-request/different-response pairs. However, each comparison first
differed in `causal_environment_sha256` at progress row/action index 0. This
opaque hash difference remains a limitation: it must not be discarded or
represented as full causal-trace agreement. The saved initial/endpoint audit's
`all_exact` permits the declared narrow XML compatibility rule; inspect its
`model_xml_audit` before attributing the hash difference to serialization or
physical state. The intervening complete causal states were not retained.

Historical right-camera mean RGB was [156.08, 155.97, 156.22], with a mean
absolute per-pixel red/green gap of 111.28 and zero percent exactly gray pixels.
The historical wrist statistics were nearly identical: [156.10, 155.98, 156.23]
and gap 111.29. Balanced channel averages therefore do not imply grayscale
pixels. Replay right-camera mean RGB was [125.81, 124.54, 124.76], with gap 2.81
and 22.15% exactly gray pixels; the replay left-camera gap was 2.55, compared
with 87.17 historically. These unusual historical color distributions and
similar camera summaries motivate visual inspection of `endpoint_cameras.png`
and the retained history panels. They do not, by themselves, prove corruption,
identical right/wrist frames, or which trajectory's images are correct.

The latest source/replay requests both occurred at environment step 288,
inference index 18, with sampling seed 6960018 and ordinary seed base 6960000.
Their hashes differed as recorded above. Thus the latest observed historical
input mismatch was already present at step 288, but the earliest historical
divergence remains unknown. The historical gate stays blocked; no parent is
substituted and no new GPU rollout is warranted solely to repeat these checks.

### Confirmed historical endpoint corruption and XML compatibility

On 2026-10-13 the user supplied a screenshot of the exported endpoint camera
panel. All three historical cameras show multicolored pixel/stripe corruption;
both replay rows show coherent kitchen views. Together with the registered
source-file integrity check and the inspection tool's direct, unscaled uint8
RGB export, this establishes that the saved historical endpoint at environment
step 304 contains corrupted camera observations. It is not merely a large
pixel difference caused by a different robot pose. The source file still
matches its registered bytes: file integrity does not certify image quality.

The [supplied comparison screenshot](assets/xr1_timing_historical_endpoint_6cf_20261013.png)
is preserved byte-for-byte (1212 x 1266 pixels; SHA-256
`a91f1453ed1ed8e49c9b4fb1cddfa26d10967fcb34ff5b405ee7e4119c169662`).
This repository artifact is a screenshot of the panel, not the original
cluster PNG or a new simulator observation. The retained image-history panels
have not been supplied for visual review.

For both corresponding cohort/isolated attempts, all four saved XML audits
(initial and endpoint for each attempt) report `strict_exact=false`,
`compatible=true`, `compatibility_mode=mesh_obj_content_type_omission`, and
193 allowed mesh content-type normalizations. The existing compatibility rule
allows only omission of redundant `content_type="model/obj"` on an unchanged
`.obj` mesh; other attributes and XML structure must agree. This explains a
raw fingerprint difference at the saved states because those fingerprints
include XML. It is separate from the historical pixel corruption. It does
not establish that every intervening causal state was identical, and does
not justify excluding arbitrary XML or physical-state differences from audits.

The screenshot establishes corruption at the historical endpoint only. It
does not date the onset, establish what images were encoded into the request
at step 288, prove a particular CUDA/GC/destructor mechanism, or show that all
historical parents are affected. Matching requests and responses across the
tested current runs remain evidence of repeatability for those calls only.
The 160-branch operator screen's recorded outcomes are preserved; image
quality and the affected snapshot's use in that screen need review before
treating its negative result as an operator-capability finding under clean
camera inputs. One corrupted source does not invalidate every earlier result.

### Next experiment: a separately registered clean-reference cohort

Keep the original snapshots, registrations, failed audits, and 80 collected
timing executions unchanged. Do not resume the old collector, waive historical
source agreement, or replace only the failed parent. The existing collector
is bound to the historical prefix-2 references and cannot implement this new
cohort by changing an output path alone. The new collector and registration
implement the nominal-reference preflight in steps 1--4 below; live cluster
validation is pending. Use the [fresh-reference run instructions](xr1_fresh_timing_references.md).
It stops at `awaiting_image_review` after ten passing pairs. Integration of
the timing grid in step 5 follows an accepted reference cohort; the old
timing launcher cannot consume these new references.

1. Freeze the same ten task/environment-seed/reset-index assignments, checkpoint,
   policy configuration, target stages, and ordinary sampling seeds. Label this
   as a new cohort using the previously selected seeds, not an independent
   sample or a reconstruction of the historical trajectories. Retain early
   CUDA initialization, separate renderer/server GPUs, and the corrected GC
   guard through environment cleanup.
2. Generate fresh nominal trajectories with fresh policy history. Capture the
   actual reset-time simulator, controller, policy, global/environment RNG,
   observation, and evaluation states. Capture the first eligible target-stage
   inference boundary while the target is incomplete, using a frozen nominal
   search budget. Do not restore the historical endpoint's corrupted images
   or infer an earlier state from that endpoint. If a target is not reached,
   record that outcome without replacing the seed or extending its budget
   after inspecting outcomes.
3. Repeat each nominal trajectory from its newly captured initial state in a
   fresh environment. Preserve every consumed action, encoded request hash,
   response hash, raw camera observation, and complete causal state sufficient
   to audit any first difference. Require repeat agreement at the common entry
   and throughout the recorded prefix. Audit any XML normalization explicitly
   under the existing narrow rule rather than dropping XML from comparison.
4. Review camera panels from reset through entry for all ten parents, alongside
   the repeat audits. Do not certify health from RGB statistics alone. This
   stage collects nominal reference trajectories and no intervention outcomes.
   Any failed repeat or corrupted image blocks the timing comparison.
5. Only after this preflight passes, register and run the paired timing grid
   from the new common-entry states: continuation and the three existing
   operators at delays 0/1/2, two continuation schedules, the same fixed task
   budgets, and sentinel repeats. Report this as a new timing study; do not
   merge its results into the unfinished original 220-execution registration.

This preserves the scientific question of whether earlier intervention helps,
while making valid current observations the experimental starting point.
SAFE-triggered recovery remains deferred.

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

## Stage 0: provenance, predicate, and replay audit (original blocked protocol)

The following stages document the frozen original protocol and implementation.
They remain useful for interpreting preserved artifacts, but their historical
source-replay requirement is blocked by the confirmed corrupted endpoint.
Use the new-cohort specification above for subsequent implementation.

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
records separate reset/setup and nominal collection counts, the explicit nominal
scope, and restoration of the previous setting. Nominal tracing begins at the
first subtask evaluation after reset/source RNG setup and ends after the frozen
source endpoint audit. It includes the initial evaluation and first action at
step 0. Status shows setup events, nominal phase counts, and events near the
first mismatch, or the first 20 nominal collection stops when there is no mismatch.
An event during stepping
is labeled with the step before that action returns. `renderer_cleanup_source.json`
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

### Confirm the GC effect with the matched default-mode control

The disabled-GC pair above passed both historical prefix audits and matched
throughout, while the earlier separated-GPU pair diverged. That earlier control
did not have the current GC observer or installed-source inspection. Use the
same current implementation and restore only the default GC mode in a new
process/output root. No implementation change is needed:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_gcdisabled_20261013_135416/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_REPLAY_GC_MODE=default
export XR1_REPLAY_PORT=10325
export XR1_REPLAY_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_replay_gccontrol_$(date +%Y%m%d_%H%M%S)"
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" status
```

Inspect collection events near the first image mismatch and compare source
acceptance and repeatability with the disabled-GC run. A reproduced failure
associated with collection strengthens this hypothesis; it still does not
identify a particular destructor call. If the default-mode pair also passes,
the effect has not been reproduced under the matched instrumentation and needs
further controlled evidence. Only after that check should a bounded GC policy
be applied to a separately registered timing collector and validated across the
registered parents and intervention branches. Do not resume the old failed
registration or treat this one-parent success as full-study completion.

### Audit every registered parent with the guarded lifecycle

Set `XR1_REPLAY_ALL_PARENTS=1` to run
`audit_intervention_timing_parents.py` through the existing launcher. It requires
the original ten unique parents, five per task, and validates their registered
snapshot files and source identities. Each parent receives two fresh nominal
policy/environment replays with early RNG-runtime initialization and automatic
cyclic GC disabled through cleanup. The fixed raw-state window is cleared
because source prefixes have different lengths; every returned observation is
still retained and checked. No intervention trajectories or replacement source
parents are collected, and the frozen timing execution modules stay unchanged.

Each parent must pass both original source-prefix audits, complete initial and
endpoint repeat audits, progress and per-key/raw observation comparisons,
seeded-request probes, and the recorded GC/runtime policy. The v2 GC guard
requires zero collections from the first post-reset subtask evaluation through
the source endpoint audit, with automatic GC disabled during all recorded
collections; explicit reset/setup collections remain visible separately.
A repeatable pair that misses the historical source fails. Collection stops at the first failed
parent and preserves its diagnostics and all preceding accepted parents.

Run on the current two-GPU allocation, using the same source registration and
a new audit root for the corrected GC scope:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_parent_audit_20261013_164436/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_REPLAY_ALL_PARENTS=1
export XR1_REPLAY_GC_MODE=disabled
export XR1_REPLAY_RNG_RUNTIME_INIT=before_env
export XR1_REPLAY_RECORD_ALL_OBSERVATIONS=1
export XR1_REPLAY_GPU=0
export XR1_REPLAY_SERVER_GPU=1
unset XR1_REPLAY_OBS_WINDOW_START XR1_REPLAY_OBS_WINDOW_END
export XR1_REPLAY_PORT=10327
export XR1_REPLAY_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_parent_audit_gc_scope_$(date +%Y%m%d_%H%M%S)"
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" status
tail -n 50 "$XR1_REPLAY_LOG"
```

The cohort report lives at `$XR1_REPLAY_ROOT/diagnostic/report.json`. Success is
`status: all_parents_pass`, with 10 parents, 20 nominal replays, five accepted
parents per task, and every engineering gate true. The source-step total is
printed during dry-run validation. Each parent retains two complete diagnostic
attempts, raw images, source audits, GC events, and installed renderer source;
the parent ledger binds reports by file hash and those reports transitively
bind their artifacts. This is an engineering preflight, with zero scientific
intervention trajectories and `full_timing_study_complete: false`.

An interrupted audit without a recorded error can resume with the same saved
`run.env` and `bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" resume`.
The source/configuration/code and every completed parent's artifact hashes are
revalidated; accepted parents are skipped, and an interrupted uncommitted
attempt gets a separate new directory. Old files and logs are preserved.
A recorded failure blocks automatic retry and requires diagnosis before a new
root. An all-parent pass supports applying this bounded lifecycle to a new
timing registration, followed by the sentinel and full branch audits. It does
not resume or complete the old failed registration, or test SAFE-trigger
recovery.

### Isolate the repeatable historical mismatch after four accepted parents

The existing single-parent launcher starts a fresh simulator process and its
own fresh private policy server. No execution-module change is needed. Keep the
current allocation/node if available so that process history is the intended
changed factor; the reported hostname makes a node change visible.

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_parent_audit_gc_scope_20261013_171134/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_REPLAY_ALL_PARENTS=0
export XR1_REPLAY_PARENT=6cfb095ec52c00970abf8340
export XR1_REPLAY_PORT=10328
export XR1_REPLAY_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_replay_6cf_isolated_$(date +%Y%m%d_%H%M%S)"
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" start

source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_latest.env
bash "$ROBOCASA_REPO/scripts/diagnose_xr1_timing_replay.sh" status
tail -n 50 "$XR1_REPLAY_LOG"
```

For a passing pair, require both `prefix_replay_accepted` values true, exact
initial/endpoint/progress/observation repeats, exact seeded request probes, and
zero active-rollout collections under the registered scope. A fresh-process
pass would show that the mismatch was not invariant to the preceding execution
history; compare against the preserved cohort trace before changing lifecycle.
A repeatable source mismatch again would require inspecting saved source audit
details, latest request/inference records, and raw observations. It would not
certify that the historical source is defective. A divergent fresh pair would
reopen the current-run repeatability investigation. All outcomes leave the full
timing study incomplete; this check collects no interventions and does not make
the old failed cohort resumable.

### Inspect the saved historical source without a GPU run

This command reads the isolated diagnostic and the failed cohort child. It
checks every registered diagnostic artifact hash before unpickling, validates
the frozen timing registration and source identity/file hash, and verifies
complete raw-observation support and hashes. It compares both corresponding
current attempts' initial states, endpoints, progress, request/response hashes,
and all observation hashes across runs. No policy-model loading, simulator,
policy server, CUDA RNG inspection, or network connection is required.

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_6cf_isolated_20261013_174013/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_SOURCE_INSPECT_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_source_inspection_$(date +%Y%m%d_%H%M%S)"
set -o pipefail
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.inspect_intervention_timing_source \
  --diagnostic-dir "$XR1_REPLAY_ROOT/diagnostic" \
  --comparison-diagnostic-dir /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_parent_audit_gc_scope_20261013_171134/diagnostic/parents/6cfb095ec52c00970abf8340/attempt_g9lwwpxh/diagnostic \
  --output-dir "$XR1_SOURCE_INSPECT_ROOT" \
  2>&1 | tee "$STORAGE_BS/robocasa_logs/eval/$(basename "$XR1_SOURCE_INSPECT_ROOT").log"
```

The new output directory contains `report.json`, `endpoint_cameras.png`, and
one `history_camera_<index>.png` per camera. Rows distinguish historical source
and both replay attempts; labels identify actual saved queue indices. Pixels
are exported without resizing or recoloring. History panels show at most eight
representative frames per row, while statistics retain every saved queue frame.
Console output summarizes source/replay camera statistics, latest policy records,
and cohort comparisons. PNG hashes are stored in the report. Comparison results
must be read together with exact step-support/row counts; missing rows do not
count as agreement. The report's scope remains saved endpoints and retained
queues, not identification of the first historical mismatch or proof of which
trajectory is correct. Existing output roots and paths overlapping frozen inputs
are rejected. This inspection leaves the historical replay gate blocked.

For the completed inspection, show the saved XML compatibility details without
rerunning any inspection or simulator:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_replay_6cf_isolated_20261013_174013/run.env
export XR1_SOURCE_INSPECT_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_source_inspection_20261013_181125"
"$XR1_CLIENT_ENV/bin/python" - "$XR1_SOURCE_INSPECT_ROOT/report.json" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
for i, attempt in enumerate(report["comparison_with_cohort"]["attempts"]):
    for name in ("initial_audit", "endpoint_audit"):
        xml = attempt[name].get("model_xml_audit") or {}
        print(i, name, {k: xml.get(k) for k in (
            "strict_exact", "compatible", "compatibility_mode",
            "allowed_mesh_content_type_normalizations")})
PY
```

The user subsequently supplied the endpoint panel screenshot, reviewed above:
its historical camera row is corrupted. The original native cluster PNG and
`history_camera_1.png` (the right-camera history) have not been reviewed locally.
The screenshot does not justify changing the frozen reference or accepting
the historical source-replay gate.

Historical command for the initial RNG-only failure, retained for provenance.
Do not execute this launcher to restart the study after the confirmed source
corruption; it still requires agreement with the old references:

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
.venv_video/bin/python -m unittest tests.test_intervention_timing_source_inspection tests.test_intervention_timing_parent_audit tests.test_intervention_timing_observation_diagnostic tests.test_intervention_timing_replay_diagnostic tests.test_intervention_timing tests.test_operator_screen tests.test_phase2_snapshot_replay tests.test_phase3b_training_horizon tests.test_safe_xiaomi_policy
bash -n scripts/run_xr1_intervention_timing.sh
bash -n scripts/diagnose_xr1_timing_replay.sh
bash -n scripts/diagnose_xr1_timing_observations.sh
```
