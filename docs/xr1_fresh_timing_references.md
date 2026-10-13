# XR1 fresh common-entry reference cohort

Implementation status: **v2 completed both first-parent nominal prefixes,
but their repeatability gates failed; first-divergence inspection is pending**.
Protocol: `xr1_fresh_common_entry_reference_cohort_v2`.
Read [the cluster runbook](cluster_experiment_runbook.md) before any cluster operation.

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
policy, semantic, and raw observation states are saved at **every environment
step**, including reset and entry. Consumed actions, inference records, full
encoded requests/responses, and their hashes are retained. Per-step state
audits apply the existing narrow OBJ MIME compatibility rule explicitly; raw
state hashes are retained but are not substituted for portable comparisons.

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

## Launch on the two-GPU allocation

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
export XR1_REFS_PORT=10327
export XR1_REFS_ROOT=/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_references_$(date +%Y%m%d_%H%M%S)

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
The v2 live cohort remains failed at its first pair; per-step divergence
details are the next inspection, rather than another collection launch.
