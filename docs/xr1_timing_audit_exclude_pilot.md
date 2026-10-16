# XR1 timing pilot with audited parent exclusions

> **Frozen 2026-10-16.** This track is closed and must not be extended; see
> [exact_replay_frozen.md](exact_replay_frozen.md). Code moved to
> `robocasa/recovery/exact_replay/` and `scripts/exact_replay/`; existing
> registrations must be resumed or reviewed from commit `4d65b542`.

Protocol: `xr1_common_entry_timing_audit_and_exclude_pilot_v1`.
First live pilot: **72/72 assigned executions collected, all engineering gates
passed, zero timing errors; nominal image review is pending. Only three
CuttingToolSelection parents qualified; no ArrangeTea parents qualified.**
Read [the cluster runbook](cluster_experiment_runbook.md) before cluster operations.

The user authorized continuing timing research with engineering exclusions,
instead of requiring every selected seed to pass. This is a new registered
pilot. Original timing runs, their 80 collected executions, failed v4 roots,
and existing registrations remain unchanged. Historical timing outcomes are
not merged into this new sample. The accepted v4 reference pair is reusable
because its complete engineering evidence is validated, copied and bound by
hash before the new run starts.

## First live pilot result

Run: `/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_pilot_20261013_233646/pilot`
on `r5n4`. Collection finished at `2026-10-13T15:58:16.456292+00:00`
(2026-10-14 00:58 JST), with status
`timing_complete_awaiting_image_review`.

| Task | Attempted parents | Eligible parents | Excluded parents | Timing engineering passed |
|---|---:|---:|---:|---:|
| ArrangeTea | 5 | 0 | 5 | 0 |
| CuttingToolSelection | 5 | 3 | 2 | 3 |
| Total | 10 | 3 | 7 | 3 |

All 72 assigned executions completed: 60 primary trajectories (three parents,
ten cells, two continuation schedules) and 12 engineering repeats. Every
retained parent's engineering gates passed, with zero timing errors and no
saved prequalification controls belonging to excluded parents. The status
field `successful_assigned_branches=72` counts completed executions, not
successful robot tasks. The pasted status contains no recovery efficacy
estimates.

Four parents were excluded during reference screening for live reset
observation mismatch: ArrangeTea seeds 950010 and 950012, and
CuttingToolSelection seeds 950022 (the reused v4 rejection) and 950025.
Three additional ArrangeTea parents, seeds 950007, 950011 and 950013, failed
the common-entry sampled-observation/runtime/semantic restore check before
any successful control suffix was recorded. These are distinct gate failures;
the status alone does not identify every differing channel or establish that
all seven exclusions share the historical camera-corruption cause. In
particular, seed 950007's accepted v4 nominal reference pair did not guarantee
that its entry snapshot would pass the separate timing restore check.

The audit-and-exclude pipeline therefore completed on its eligible assignments,
but attrition was 70%, and an entire task was lost. This is not evidence of
rare sample corruption or a completed two-task timing comparison.
`timing_collection_complete=true` and `engineering_all_pass=true` coexist
with `pilot_scientific_analysis_ready=false` and
`full_timing_study_complete=false`. SAFE-triggered recovery was not tested.

Next, inspect every nominal camera frame of the three eligible parents and
record the image review using the commands below. If all three pass, review
will emit paired descriptive outcomes for CuttingToolSelection only, with
status `complete_limited_coverage` and decision
`insufficient_valid_task_coverage`. It will not emit the registered two-task
bootstrap or a passing operator recommendation. Read these existing results
before allocating more GPU time or expanding the fixed seed pool. The absent
ArrangeTea task cannot be repaired by silently replacing seeds, weakening a
restore gate, or labeling an observed-task average as a two-task average.

Evidence identifiers:

- Registration SHA-256: `137cfd263d7847ef572133fed4e6572ffdf3531dbc90c965fb51ba8bb8f8e604`.
- Assignment SHA-256: `ac4561076711d85d7d7bdcec275ac29c915f1e22653771527239f223b7e8924b`.
- Branch-ledger SHA-256: `c4ba3cf6a2f64fc8fdb1921efe7ec41afe1853eba2e8d579a7ea358942baa0af`.
- Driver log: `/gs/bs/tga-shinoda/felid/robocasa_logs/eval/xr1_timing_pilot_20261013_233646_driver_20261013_233647_3496843.log`.

## Fixed pool and eligibility

Attempt the same ten previously selected seeds, five per task, once each.
There are no replacement seeds, no repeated attempts until one passes, and
no selection based on recovery success. The launcher defaults to the failed
v4 run `xr1_timing_references_observer_v4_20261013_220743/references` as an
evidence source. It imports its one committed, validated ArrangeTea pair and
the documented CuttingToolSelection reset-observation rejection before any
nominal requests. It screens the remaining eight parents using the unchanged
v4 reference algorithm. If the v4 evidence does not meet these conditions it
is not silently treated as a pass. Reused evidence must match the fixed seed,
task, reset index, policy settings, code, and checkpoint; only the transport
host/port may change. Old files are read only and copied, not hard-linked.

A reference exception with a known reset-state/observation mismatch, or failed
reference engineering gates, excludes the whole parent. Its partial traces,
reason, stage, task, seed, and artifact hashes remain in the new run. Unknown
configuration/code exceptions stop the run. Quota, storage, transport, import,
and out-of-memory failures also stop the run; they are never counted as bad
samples. This distinction prevents an unavailable server or full filesystem
from being disguised as ten invalid seeds.

Each passing reference then undergoes two full `continue`, schedule-0 suffixes
from its actual common entry, before any recovery treatment runs for any
parent. They must repeat exactly, including consumed actions, observations,
declared causal environment, semantics, policy, sampler/cache state, full
encoded requests/responses, and seed schedule. A failed control preflight
excludes that parent; its control outcomes are not included in scientific
comparisons. The same rules apply regardless of whether the robot succeeds.
An ordinary unsuccessful rollout remains a valid outcome.

## Restore and timing contracts

The entry restore preserves real control-attribute ownership, complete
observable sampling/cache state, both RNG families, controller state, and
simulator integration state. It reconstructs the policy observation through
the actual Gym adapter with `_get_observations(force_update=False)`. It does
not substitute the saved observation or take a new forced measurement. A
sampled observation can intentionally lag the current instantaneous physical
state; advancing its sampler during verification would change the experiment.
The reconstructed observation and all complete declared state, including
sampler/cache/RNG and semantics, must equal the snapshot at zero tolerance.
An impure getter that changes the sampler or other declared state is rejected.
This verifies restoration of sampled state, not fresh-render image health.

Nominal reference repeat acceptance retains v4's explicitly named downstream
force/torque buffer exclusion and its full diagnostic comparison. Entry
restoration itself still requires the complete declared snapshot state exact.
Automatic cyclic GC is disabled before each environment creation through
environment/policy cleanup. Explicit reset/restore setup collections remain
logged and do not count as suffix collections; the active scope begins after
restore, before sampled observation reconstruction and semantic evaluation.
Any collection in that active scope rejects the execution. CUDA RNG inspection initializes before environment
creation. Simulator and policy server use separate GPUs; a fresh environment
and policy connection is created per suffix. No frozen shared module is edited.

After all ten eligibility decisions are committed, an immutable
`assignment.json` binds all ten timing cells and both continuation schedules
for every eligible parent. Cells are `continue`, plus each of three operators
at delays 0, 1, and 2 genuine inferences. The common absolute suffix budgets
remain 480 environment steps for ArrangeTea and 256 for CuttingToolSelection.
Original global parent indices determine request seed bases; exclusions never
renumber them. Completed/ineligible intervention fallbacks remain assigned
outcomes. This remains supervisor-stage timing, not SAFE-detector recovery.

The first eligible parent of each task is selected for complete schedule-0
engineering repeats before looking at any treatment outcomes. Every other
eligible parent retains its schedule-0 `continue` repeat from preflight.
Thus the all-valid maximum is 200 primary trajectories plus 28 repeats,
228 assigned executions. The registration enumerates 300 potential execution
identities so a rejected original sentinel can be replaced deterministically;
it does not require or execute all 300. The actual denominator is frozen after
eligibility. Controls belonging to excluded preflight parents are reported
separately from assigned scientific trajectories.

An exception during a timing branch is recorded by parent, timing cell,
operator and stage, with its partial requests and diagnostics. Remaining
branches continue. Such errors are not converted into pre-intervention
exclusions or robot failures. Missing cells, failed prefixes, or failed repeats
remain visible and block scientific inference. Complete valid parents may
have descriptive summaries after image review, with the affected parents and
error rates reported explicitly; they do not establish an unbiased treatment
effect when missingness could depend on treatment.

## Launch and monitor

Use a fresh two-GPU allocation or the existing allocation with both GPUs free:

```bash
cd /gs/fs/tga-shinoda/felid/robocasa
git pull --ff-only origin main

export XR1_PILOT_SOURCE=/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_rngfix_20261012_194551/full
export XR1_PILOT_REFERENCES=/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_references_observer_v4_20261013_220743/references
export XR1_PILOT_GPU=0
export XR1_PILOT_SERVER_GPU=1
export XR1_PILOT_PORT=10332
export XR1_PILOT_ROOT=/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_pilot_$(date +%Y%m%d_%H%M%S)
bash scripts/exact_replay/run_xr1_timing_pilot.sh start
```

The launcher validates source provenance before starting a private policy
server, verifies readiness before the simulator, uses unbuffered Python and
durable logs, and cleans up only its own server. It copies a node-local model
from durable storage if the registered checkpoint path requires it. It requires
50 GiB free as a minimum storage guard, not an estimate. Inspect quota failures
before launching any further work.

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_pilot_latest.env
bash "$ROBOCASA_REPO/scripts/exact_replay/run_xr1_timing_pilot.sh" status
tail -n 60 "$XR1_PILOT_LOG"
```

Status shows attempted/excluded references, qualification decisions, current
timing progress, task coverage, and errors by operator. The new run is under
`$XR1_PILOT_ROOT/pilot`; it is not the original `full/` timing registration.

| Status | Meaning |
|---|---|
| `screening_references` | Reusing/screening the fixed ten-parent pool. |
| `qualifying_parents` | Testing complete restored `continue` repeats before treatment. |
| `running_timing` | Running the frozen eligible timing assignments. |
| `timing_complete_awaiting_image_review` | Assigned trajectories and engineering checks passed; nominal image review is pending. |
| `timing_complete_with_errors_awaiting_image_review` | Collection finished, with explicit timing errors or failed engineering checks. |
| `no_eligible_parents` | All candidates were excluded; this is not a successful timing comparison. |
| `complete` / `complete_with_exclusions` | Image review and minimum coverage support the revised exploratory analysis. |
| `complete_limited_coverage` | Insufficient valid task coverage; only limited descriptive output is available. |
| `complete_with_timing_errors` | Late errors remain; scientific inference is blocked. |
| `interrupted` | Partial attempts and committed decisions are preserved and resumable. |
| `failed` / driver failure | Infrastructure, configuration, or integrity failure; inspect the retained evidence. |

Resume an interrupted pilot with its saved environment:

```bash
source /path/to/pilot_run/run.env
bash "$ROBOCASA_REPO/scripts/exact_replay/run_xr1_timing_pilot.sh" resume
```

Resume revalidates code, source/evidence hashes, parent markers, complete branch
payloads, encoded request artifacts, terminal errors, and assignment. Committed
exclusions and terminal timing errors are not retried. Interrupted uncommitted
attempts get new directories; payloads written before a ledger append can be
recovered. Recorded fatal errors block automatic resume. Never delete error
records or modify a registration to make it pass.

## Image review and scientific interpretation

Timing collection may finish before human review, so the automatic pipeline
does not stop between the nominal screen and suffix execution. Scientific
analysis remains blocked until explicit nominal image review. Open
`$XR1_PILOT_ROOT/pilot/camera_review.html` and inspect every slider frame for
each eligible parent, before reading recovery outcomes. The gallery contains
only pre-intervention nominal images. Assess these on their quality, not on
whether a recovery operator succeeded. Copy `image_review_template.json` to a
separate file, set each `reviewed_all_frames=true`, and set each
`images_acceptable` to a boolean; record corruption in `notes`.

```bash
source /path/to/pilot_run/run.env
bash "$ROBOCASA_REPO/scripts/exact_replay/run_xr1_timing_pilot.sh" review /path/to/completed_image_review.json
```

Review is CPU-only and records one immutable assessment. A rejected nominal
image excludes that parent from paired analysis using pre-treatment evidence;
its collected trajectories remain on disk and in coverage reporting. The
review does not certify every suffix camera frame. Full encoded requests and
responses are retained for later inspection, and suffix errors stay explicit.

For the completed first pilot, use its saved environment rather than whichever
run the latest environment file may subsequently identify:

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_pilot_20261013_233646/run.env
cp -n "$XR1_PILOT_ROOT/pilot/image_review_template.json" "$XR1_PILOT_ROOT/completed_image_review.json"
nano "$XR1_PILOT_ROOT/completed_image_review.json"
# After inspecting every frame, fill each parent's two booleans and notes.
bash "$ROBOCASA_REPO/scripts/exact_replay/run_xr1_timing_pilot.sh" review "$XR1_PILOT_ROOT/completed_image_review.json"
bash "$ROBOCASA_REPO/scripts/exact_replay/run_xr1_timing_pilot.sh" status
```

The top-level gallery links to each eligible reference's gallery under
`references/`. When transferring it to another computer, preserve these
relative paths and each linked gallery's `camera_panels/` directory. Snapshot
pickles and recovery branch payloads are not needed for viewing the galleries.
Set `reviewed_all_frames=true` only after reviewing that parent's complete
slider; set `images_acceptable=false` and explain the issue if any frame is
corrupted. Preserve the protocol and three binding hashes.

After review, print the available descriptive cell outcomes without treating
missing-task coverage as a passing scientific gate:

```bash
"$XR1_CLIENT_ENV/bin/python" - "$XR1_PILOT_ROOT/pilot/scientific_analysis.json" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("decision:", data["decision"])
print("parents by task:", data["parents_by_task"])
print("analysis ready:", data["pilot_scientific_analysis_ready"])
for task, cells in data.get("tasks", {}).items():
    print("\n", task, "(descriptive, engineering/image-qualified parents only)")
    for cell, values in cells.items():
        print(cell, {key: values[key] for key in (
            "completion", "durable_completion", "ordered_prefix_regression",
            "safety", "task_success", "intervention_executed")})
PY
```

At least two valid parents per task are required for the two-task summary and
parent bootstrap. This minimum avoids a single-parent resampling calculation;
it is not a statistical power claim. Missing tasks or insufficient coverage
must not be presented as the planned two-task result. Any late timing error or
failed branch engineering gate blocks inference/recommendation, even if other
parents have descriptive outcomes. Summaries retain paired cells and both
averaged seed schedules, use task macro averages, report exclusions and late
errors, and remain conditional on previously selected seeds passing the
engineering/image checks. Fresh independent validation remains necessary.

## Local checks

```bash
.venv_video/bin/python -m unittest tests.exact_replay.test_intervention_timing_pilot tests.exact_replay.test_intervention_timing_fresh_references tests.exact_replay.test_intervention_timing_source_inspection tests.exact_replay.test_intervention_timing_parent_audit tests.exact_replay.test_intervention_timing_observation_diagnostic tests.exact_replay.test_intervention_timing_replay_diagnostic tests.exact_replay.test_intervention_timing tests.exact_replay.test_operator_screen tests.exact_replay.test_phase2_snapshot_replay tests.exact_replay.test_phase3b_training_horizon tests.test_safe_xiaomi_policy
bash -n scripts/exact_replay/run_xr1_timing_pilot.sh
```

These are simulator-free protocol/execution tests and shell checks, not live
GPU validation or evidence of recovery efficacy.

The combined suite ran 142 tests: 141 passed and one was skipped. Fourteen
new pilot tests cover complete timing grids (including executed delayed
interventions), negative robot outcomes, reference/control exclusions,
sentinel promotion, fixed seed indices, sampled-state restoration, late
errors, interrupted commits/resume, reused v4 evidence without source
modification, changed encoded artifacts/assignment, missing task coverage,
immutable image review, setup versus active GC, and launcher GPU/provenance/
private-server lifecycle. Python compilation, shell syntax, and whitespace
checks also passed. Frozen execution and reference modules remain unchanged.
