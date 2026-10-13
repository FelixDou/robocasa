# XR1 fresh common-entry reference cohort

Implementation status: **v4 accepted the first ArrangeTea pair, then failed
CuttingToolSelection's live reset-observation check. One of ten pairs is
accepted; inspect the saved reset audit before another collection**.
Protocol: `xr1_fresh_common_entry_reference_cohort_v4`.
Read [the cluster runbook](cluster_experiment_runbook.md) before any cluster operation.

The user has now authorized continuing under a new
[timing pilot with audited exclusions](xr1_timing_audit_exclude_pilot.md).
Use that launcher to reuse the accepted v4 pair, retain the documented reset
rejection, and screen the remaining seeds without aborting at each invalid
parent. This document and its no-exclusion rules retain the historical v4
contract; the failed v4 registration is not resumed or rewritten.

The saved historical cameras for ArrangeTea parent `6cfb095ec52c00970abf8340`
are corrupted. The original timing run remains stopped at 80/220 executions.
This collector creates a new reference cohort from the same ten previously
selected task/seed/reset-index assignments. It does not reconstruct the old
trajectories or change their eligibility. Its new parent IDs include the new
protocol; metadata retains the legacy identities for provenance. This is a
study on previously selected seeds, not an independent benchmark sample.

The collector loads registered historical files only to verify identity and
read fixed checkpoint, policy configuration, target stages, and ordinary
sampling seeds. It never restores their observations, cached image history,
actions, or endpoint RNG streams. It captures the actual new reset state and
then the first eligible inference boundary in the target stage while that
target remains incomplete. The nominal search budget is frozen to the original
Phase 2 plan's `parent_horizon`. An unavailable entry is recorded, blocks the
cohort, and does not cause seed replacement or a longer search.

Each parent has two fresh environment/policy instances. The second restores
the first's new reset snapshot, including actual global and environment RNG
streams. Complete declared simulator, integration, controller, control, RNG,
policy, semantic, raw observation, and observable sampling/cache states are saved at **every environment
step**, including reset and entry. Consumed actions, inference records, full
encoded requests/responses, and their hashes are retained. Per-step state
audits apply the existing narrow OBJ MIME compatibility rule explicitly; raw
state hashes are retained but are not substituted for portable comparisons.
V4 retains the full controller comparison and separately gates its existing
declared causal scope, as detailed below. No numeric tolerance is introduced.

CUDA RNG inspection is initialized before environment creation. Automatic
cyclic GC is disabled before each creation until both environment and policy
cleanup finish. Explicit setup collections remain logged. Any collection from
the first post-setup subtask evaluation through entry capture rejects the
parent; the original GC setting is restored after cleanup. GPU mappings,
hostname, process IDs, startup events, and installed renderer cleanup source
are recorded. Only the launcher's private server is stopped on exit.

## First live run and wrapper fix

`xr1_timing_references_20261013_192435` stopped with zero completed parent
pairs. Its first ArrangeTea trajectory (legacy parent
`b1455183d9fabd7eeb4c4bf5`, v1 parent `6b7a52d961efd3a7eaccf53a`, seed 950007)
captured the common entry at environment step 304 after 19 policy requests.
The second attempt failed at step zero with
`AttributeError: 'NoneType' object has no attribute 'get'` in `camera_format`.
No repeat or image-health result was established for this parent, and no
interventions were collected.

The collector used the legacy `_get_obs_after_state_change` helper directly.
That helper can reach its fallback through a non-forwarding Gym wrapper;
called without a fallback, it returns `None`. The older snapshot restore
helper also calls it with the saved observation as a fallback, so a reported
observation match on this path does not independently verify live pixels.

V2 leaves frozen helpers unchanged. It restores simulator/controller/policy/RNG
state, then resolves the actual `get_current_observation` method through the
environment wrapper graph and requires a live RGB observation dictionary.
It compares this observation with the new reset snapshot at zero tolerance,
records the getter path and `snapshot_fallback_used=false`, and rejects any
missing getter, missing observation, or pixel mismatch. A complete reset-state
audit after live observation retrieval and semantic evaluation must also pass
before the second trajectory's first policy call. No observation gate is waived.

Preserve the failed v1 root and its 304-step trace. Its registration binds the
old protocol/code and is not resumable with this fix. The following historical
launch command produced the v2 run examined below; inspect its saved audit
before starting another collection:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_references_20261013_192435/run.env
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
export XR1_REFS_ROOT="$STORAGE_BS/robocasa_rollouts/recovery/xr1_timing_references_obsfix_$(date +%Y%m%d_%H%M%S)"
export XR1_REFS_PORT=10328
bash "$ROBOCASA_REPO/scripts/collect_xr1_timing_references.sh" start
```

## V2 first-parent repeat failure: inspect saved data next

`xr1_timing_references_obsfix_20261013_201241` on `r5n4` completed both
304-step, 19-request nominal trajectories for ArrangeTea seed 950007, v2
parent `6c84f0411b020f6889786e66`. Both captured a common entry. The initial
restore, complete initial-state audit, GC guard, and early-CUDA gates passed.
The failed gates were `every_complete_state_repeats_exact`,
`consumed_actions_and_inferences_repeat_exact`, and
`encoded_requests_and_responses_repeat_exact`. The cohort stopped with zero
accepted parent pairs and no intervention trajectories.

This confirms the wrapper setup fix allowed the second trajectory to proceed.
Equal lengths, request counts, and entry timing do not establish trajectory
equality. The supplied status does not identify the first differing state
channel, observation, request, response, or consumed action. It therefore does
not establish a renderer mechanism, simulator nondeterminism, or a response
difference for identical policy inputs. Retain the failed root and its strict
gates. The next step reads the existing JSON audits; no simulator, policy
server, checkpoint loading, or new GPU run is needed.

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_references_obsfix_20261013_201241/run.env
export XR1_REFS_PAIR="$XR1_REFS_ROOT/references/parents/6c84f0411b020f6889786e66/collection_xmaaqmxz"
"$XR1_CLIENT_ENV/bin/python" - "$XR1_REFS_PAIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
audit = json.loads((root/"repeat_audit.json").read_text())
first = audit.get("first_state_difference")
print("state support:", audit["state_support_exact"], audit["state_counts"])
by_channel = {}
for row in audit["state_audits"]:
    for channel, exact in row["channels"].items():
        if not exact:
            by_channel.setdefault(channel, row["environment_step"])
print("first difference by state channel:", by_channel)
print("first differing state step:", first["environment_step"] if first else None)
if first:
    print("state channels:", first["audit"]["channels"])
    for channel, rows in first["audit"]["mismatches"].items():
        if rows:
            print(channel, "mismatches:", len(rows), "sample:", rows[:3])
    print("XML audit:", first["audit"].get("model_xml_audit"))
print("request/response comparison:", audit["request_comparison"])
traces = [[json.loads(line) for line in (root/f"attempt_{i}/transitions.jsonl").read_text().splitlines() if line.strip()]
          for i in range(2)]
print("transition row counts:", list(map(len, traces)))
for key in ("action_sha256", "requests_sha256", "inference_sha256", "reward_sha256", "done", "info_sha256"):
    different = next((a["step"] for a, b in zip(*traces) if a.get(key) != b.get(key)), None)
    print("first transition difference", key, different)
print("camera gallery:", root/"camera_review.html")
PY
```

State steps are observations before the next action; transition step `s`
applies the action from state `s` to state `s+1`. Request comparison rows are
zero-based policy-call indices. This distinction matters when identifying
whether a camera, policy response, or physical transition differed first.

## Step-one divergence and counter ownership

The subsequent saved-audit summary localized the first difference to state
step 1: controller state, environment control state, and observations. Policy
state first differed at step 2. Encoded requests and full responses first
differed at policy-call row 1; consumed actions, request records, and inference
records first differed at transition step 16. Environment state and simulator
integration state first differed at state step 17, after that changed action.
No identical-request/different-response pair was recorded; info hashes matched.

At step 1 the supplied control samples were 1 versus 0 and 0.05 versus 0.0.
The controller's `root[7].attributes.current` differed by approximately
2.17e-19. Sample numeric observation errors were approximately 4.38e-7 for
base rotation, 3.73e-5 for gripper qpos, and 2.65e-6 for relative end-effector
position. The summary did not include the affected control object paths, the
controller buffer's identity, or all camera-key comparisons. These numeric
samples do not establish camera corruption or a physical-state difference
at step 1. The declared physical channels remained equal through state 16.

Source inspection and a local regression test found a concrete counter-restore
defect. `RoboCasaGymEnv.__getattr__` forwards reads to its inner environment,
while the legacy `_restore_environment_control_state` uses `setattr` on every
captured object path. Setting a forwarded `timestep` or `cur_time` on the
wrapper creates a separate stored value. Initial equality can still pass at
zero, but the inner environment advances while the wrapper's copy stays zero.
This reproduces the supplied counter pattern; the saved paths below are
needed to identify the affected live objects. It does not establish the cause
of the observation differences.

V3 records which control attributes are declared by each instance/class and
which are dynamically forwarded. It checks the ownership schema before
restore, removes only new alias shadows created by the legacy setter, and
re-audits values and ownership afterward. Real stored attributes, descriptors,
and their independent values remain restored. Frozen shared snapshot helpers
are unchanged. No numeric tolerance or repeatability gate is relaxed. New
v3 snapshots retain this ownership metadata; future timing continuations must
use the ownership-aware restore path rather than the legacy setter alone.

The following read-only inspection was run on the failed v2 data. It resolved
the control paths, controller-buffer identity, and per-key observation differences:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_references_obsfix_20261013_201241/run.env
export XR1_REFS_PAIR="$XR1_REFS_ROOT/references/parents/6c84f0411b020f6889786e66/collection_xmaaqmxz"
cd "$ROBOCASA_REPO"
git pull --ff-only origin main
"$XR1_CLIENT_ENV/bin/python" -u -m robocasa.recovery.inspect_intervention_timing_restore \
  --pair-dir "$XR1_REFS_PAIR" --step 1
```

This reads trusted saved experiment dictionaries and the installed source
file; it does not create an environment, load a checkpoint, query RNGs, or
modify artifacts.

## Saved-state inspection and v4 observation runtime

The supplied state-1 inspection confirms `env.unwrapped` was shadowing
`timestep`/`cur_time`: attempt 0 reports 1/0.05, attempt 1 reports 0/0,
while both actual inner environments (`env.unwrapped.env`) report 1/0.05.
The v3 ownership repair remains included in v4; v3 has not been separately
validated live.

All three 256x256x3 cameras are **byte-exact at state 1**. All five numeric
proprioceptive keys differ; maximum absolute errors are:

| Observation | Maximum absolute difference |
|---|---:|
| Base position | 3.5762786865234375e-7 |
| Base rotation | 4.38201823271811e-7 |
| Relative end-effector position | 2.652406692504883e-6 |
| Relative end-effector rotation | 3.6656856536865234e-6 |
| Gripper qpos | 3.730505704879761e-5 |

This localizes the first observed input difference to proprioception. It does
not establish image health at later steps or identify the exact unrecorded
sampling phase retrospectively. The installed `observables.py` SHA-256 is
`83e7bc67745477d207c09c4a5eb86e280909b06531ab7346595a45a84200dcfd`.
Its supplied `Observable.update` increments `_time_since_last_sample` even
with `force=True`, changes `_sampled`, resamples `_current_delay`, and updates
both `_current_observed_value` and the dependency cache. Its supplied `reset`
does not explicitly reset `_sampled`. The frozen FullSnapshot implementation
captures none of these observable/cache fields. Consequently the v2 forced
live verification can pass at reset while changing later sample timing.
This is a concrete missing-state mechanism; its causal contribution to the
failed live pair still requires the new run.

V4 records the four mutable sampling fields, all other plain observable
configuration, callback/class source identities, and each actual owner's
`_obs_cache` in reset/entry metadata and every diagnostic state. Forwarded
aliases are not separate owners. Restoration keeps the new simulator's live
sensor functions. It rejects changed topology/configuration, missing runtime
state, non-plain state, callable objects, and unsupported stateful filter/
corruption/delay closures. It restores sampling/cache state before live reset
verification, verifies actual pixels and proprioception at zero tolerance,
then restores that sampling/cache state and both RNG families again before
the next action. It returns the verified live observation; no saved-image
fallback is introduced. The complete initial audit includes observer runtime
and must still be fully exact before any nominal request. The verification
audit retains before/after-forced-read/restored hashes and differences so that
the phase perturbation is directly inspectable. Nominal rollout steps gain no
extra sensor reads. Future timing executors must restore this metadata too;
the frozen snapshot helper alone is insufficient.

The tiny controller difference is record 7,
`robots[0].recent_ee_forcetorques['right']`, a DeltaBuffer with approximately
2.1684e-19 difference. `full_snapshot.py` already explicitly excludes
`.recent_ee_forcetorques[` from its causal transition fingerprint. V4 applies
that **pre-existing, named exclusion prospectively** to repeat acceptance,
with a new registration and parent identities. `repeat_audit.json` still
retains full `controller_state` comparisons, `all_state_channels_exact`, and
the first full mismatch; it separately records causal acceptance and the
first causal mismatch. A passing v4 pair can therefore have
`all_state_channels_exact=false` solely for this named downstream buffer.
The new gate is `every_declared_causal_state_repeats_exact`. Every other
controller, physical, observation, observable-runtime, policy, semantic,
counter, and RNG channel remains exact, as do consumed actions and full
encoded requests/responses. There is no tolerance for small proprioceptive
differences. V2 remains failed under its original, stricter gate; its artifacts
and registration are not rewritten or reused as accepted v4 references.

The v4 live experiment reported below stopped at the first failed pair,
preserving all states and the new sampler audit. All ten passing pairs plus
separate image review are still required before a new timing grid. This is
an engineering experiment with no interventions and no SAFE detector recovery
claim.

## First v4 live result: one accepted pair, reset observation failure

`xr1_timing_references_observer_v4_20261013_220743` on `r5n4` accepted
ArrangeTea parent `58f0ab7314bd1b19d88ec731` (seed 950007, legacy parent
`b1455183d9fabd7eeb4c4bf5`). Both nominal attempts reached entry at environment
step 304 with 19 requests. The accepted pair passed the registered v4
causal-state, observable-runtime, transition, encoded request/response,
initial-restore, GC, and startup checks. This is the first accepted fresh
nominal pair. It does not certify the full-controller exactness flag, which
the pasted log did not include, or camera health by visual inspection.

The next parent, CuttingToolSelection seed 950022, reset index 0,
`a86aedada9ea0a3d7fd66354` (legacy parent `f56bbdc19d00beb376c8a1b3`), captured
its first common entry at step zero. This is permitted by the registered entry
definition; it used zero nominal requests. Its second attempt failed at reset
with `New reset-state live observation differs; inspect initial_restore_audit.json`.
The traceback places the failure after the simulator/controller/control/RNG
restore phase and observer-runtime restoration, at the exact live observation
comparison. The final complete reset-state/semantic audit and the nominal loop
were not reached. The second attempt's saved `status: running` is a stale
attempt marker; the enclosing `status: failed` and driver exit code 1 are
authoritative. The cohort has **1/10 accepted pairs**, not a passing cohort.
No interventions were collected and image review remains pending.

The pasted log does not identify the differing observation keys, magnitudes,
or camera pixels. It cannot distinguish stale sampled proprioception from a
render mismatch or another observation path issue. A cached reset observation
differing from a forced measurement is one possibility, not a proven cause.
The rejection precedes nominal policy calls for this parent, so it is not a
recorded divergence in a policy response or rollout action. Preserve this
failed registration and its accepted first pair; do not skip the failed seed,
relax observation tolerances, resume the failed run, or launch an identical
cohort before inspection.

Read the existing audit without a simulator, CUDA/RNG inspection, policy
server, checkpoint loading, or artifact modification:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_references_observer_v4_20261013_220743/run.env
export XR1_REFS_PAIR="$XR1_REFS_ROOT/references/parents/a86aedada9ea0a3d7fd66354/collection_m_ec8h1l"
"$XR1_CLIENT_ENV/bin/python" - "$XR1_REFS_PAIR/attempt_1/initial_restore_audit.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
audit = json.loads(path.read_text())
print("audit:", path)
print("restore flags:", {k: v for k, v in audit.items() if k.endswith("_exact") or k == "valid"})
print("live verification:", audit.get("observation_verification"))
print("observation mismatches:", json.dumps(audit.get("observation_mismatches", []), indent=2))
runtime = audit.get("observation_runtime_verification", {})
saved = runtime.get("saved_sha256")
print("sampler/cache restored:", saved is not None and saved == runtime.get("restored_sha256"))
changes = runtime.get("forced_read_mismatches", [])
print("forced-read runtime changes:", len(changes))
print(json.dumps(changes[:12], indent=2))
PY
```

Interpret the observation mismatch paths before selecting a repair. Sampler
state changing during a forced read is expected and is separately restored;
it is not by itself evidence of the cause of a live observation mismatch.

## Launch on the two-GPU allocation

The following describes the launcher used for v4. The existing live v4 root
is failed; diagnose the saved audit above before proposing a replacement run.

The launcher uses the existing absolute client/server environment paths. It
copies the registered model path under `/tmp/ut06746` from durable checkpoint
storage when needed, verifies frozen provenance, starts a private server,
checks its listening port and model-ready log message, and then starts the
simulator client. Outputs and logs go to `/gs/bs`.

```bash
cd /gs/fs/tga-shinoda/felid/robocasa
git pull --ff-only origin main

export XR1_REFS_SOURCE=/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_rngfix_20261012_194551/full
export XR1_REFS_GPU=0
export XR1_REFS_SERVER_GPU=1
export XR1_REFS_PORT=10331
export XR1_REFS_ROOT=/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_references_observer_v4_$(date +%Y%m%d_%H%M%S)

bash scripts/collect_xr1_timing_references.sh start
```

Choose another unused port for concurrent work. The launcher requires separate
simulator/server device selections and at least 50 GiB free before copying or
collection. Full state and request traces can be large; this check is a minimum,
not an estimate of total usage. A quota error means stop and inspect storage.

## Monitor and determine completion

In another terminal on the allocation:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_references_latest.env
bash "$ROBOCASA_REPO/scripts/collect_xr1_timing_references.sh" status
tail -n 60 "$XR1_REFS_LOG"
```

Status also prints a current attempt's recorded environment step and any failed
parent gates. A launcher/server failure appears as `driver status: failed` even
if no collection registration was created. Progress reaches the log every 16
nominal environment steps and after each accepted parent.

| Status | Meaning |
|---|---|
| `collecting_references` | Collection is active; the parent count is completed pairs. |
| `interrupted` | Completed parents and partial attempts are preserved; an unfailed run may resume. |
| `failed` or `driver status: failed` | Inspect the saved failure, partial artifacts, and driver log. |
| `awaiting_image_review` | Ten parents / twenty nominal repetitions passed engineering checks; image health is pending. |
| `references_ready` | Engineering checks and explicit image review passed; a subsequent timing study may consume these references. |
| `image_review_rejected` | At least one reviewed parent has unacceptable images; timing remains blocked. |

These statuses describe nominal references. **No intervention trajectories are
collected, and the timing experiment is not complete.** A genuine entry at
reset is valid and may contain zero nominal policy requests.

Resume an interrupted, unfailed run with the same code and configuration:

```bash
source /path/to/this/reference_run/run.env
bash "$ROBOCASA_REPO/scripts/collect_xr1_timing_references.sh" resume
```

Resume verifies the registration, every committed parent's artifact hashes,
and entry-snapshot checksums. It retains incomplete attempts and creates a new
attempt for an uncommitted parent. Recorded collection failures cannot be
retried in place. Preserve failed roots and diagnose them; do not remove error
records or change the registration to make a retry eligible.

## Inspect cameras and record the result

Open `$XR1_REFS_ROOT/references/camera_review.html`, retaining its relative
`parents` directory when transferring the gallery from the cluster. Each
parent's slider has every saved frame from reset through entry, displaying both
attempts and all three cameras at native pixel values. The HTML scales the
panel for the viewport; the underlying PNG pixels are unmodified. Arrow keys
advance one frame. Review every frame, rather than only the reset/endpoint.
Exact repeated images can still be corrupted, so engineering repeatability
does not certify camera quality.

After inspection, make a copy of `image_review_template.json` and fill each
parent's `images_acceptable` boolean and `reviewed_all_frames` boolean. Set
`images_acceptable` to false for corrupted images and record the issue in
`notes`. The protocol, registration hash, and ledger hash must remain unchanged.
The default null/false template is intentionally not an accepted assessment.

```bash
source /path/to/this/reference_run/run.env
bash "$ROBOCASA_REPO/scripts/collect_xr1_timing_references.sh" review /path/to/completed_image_review.json
```

Recording the review is CPU-only: it checks the cohort and artifact hashes and
records one immutable assessment. Rejected or incomplete image review cannot
make references eligible. Downstream code must call
`validated_references(root)` to verify the complete cohort, artifacts, entry
snapshots, and accepted assessment before using them.

The next live step after acceptance is a **newly registered** paired timing
grid from these common-entry states. Integration of that grid with the new
reference protocol remains a subsequent step; do not point the old timing
launcher at this directory or merge results into its unfinished registration.
SAFE-triggered recovery remains deferred.

## Local validation

```bash
.venv_video/bin/python -m unittest tests.test_intervention_timing_fresh_references tests.test_intervention_timing_source_inspection tests.test_intervention_timing_parent_audit tests.test_intervention_timing_observation_diagnostic tests.test_intervention_timing_replay_diagnostic tests.test_intervention_timing tests.test_operator_screen tests.test_phase2_snapshot_replay tests.test_phase3b_training_horizon tests.test_safe_xiaomi_policy
bash -n scripts/collect_xr1_timing_references.sh
```

These are simulator-free tests and shell validation. They do not establish
live renderer health, GPU repeatability, or intervention efficacy.

The initial implementation's combined suite ran 117 tests: 116 passed and one was skipped.
Launcher tests exercised provenance-before-server ordering, separate GPU
exports, occupied-port rejection, and private-server cleanup using local
stand-ins. A synthetic native RGB panel was visually inspected, and slider,
arrow navigation, and image paths were checked on the generated gallery.
The v2 fix adds regression coverage for a non-forwarding Gym-style wrapper,
live pixels differing after restore, and an unavailable live observation.
The updated combined suite ran 120 tests: 119 passed and one was skipped.
The v3 ownership fix's combined suite ran 122 tests: 121 passed and one was
skipped. Its regression reproduces stale forwarded counters under the legacy
setter and checks transparent aliases, class-declared state, and descriptors
under the corrected restore. The saved-state inspection command was checked
on synthetic records without changing them or executing the inspected source.
V4 adds a faithful sampler regression showing that a forced read can leave
reset observations and physics exact while changing the next proprioceptive
sample. Restoring sampler/cache state repairs that difference. The paired
cohort test also exercises a delayer consuming global NumPy RNG and checks
complete sampler/cache/RNG equality after verification. Negative cases reject
changed sampler phases, dependency caches, tiny proprioceptive errors, causal
controller changes, missing observer ownership, changed configuration, and
unsupported callbacks. The force/torque exception is tested separately from
full-state diagnostics, including cohort/parent/ledger reporting when that
buffer alone differs. A lifetime regression checks source caching does not
retain closed simulators through sensor closures. The v4 combined suite ran
128 tests: 127 passed and one was skipped. Shell syntax, Python compilation,
and whitespace checks passed. Live v4 accepted one pair and then failed at the
next parent's reset-observation check; cohort validation and camera review
remain incomplete.
