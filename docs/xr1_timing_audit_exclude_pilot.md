# XR1 timing pilot with audited parent exclusions

Protocol: `xr1_common_entry_timing_audit_and_exclude_pilot_v1`.
Implementation and local regression checks are complete; live cluster validation is pending.
Read [the cluster runbook](cluster_experiment_runbook.md) before cluster operations.

The user authorized continuing timing research with engineering exclusions,
instead of requiring every selected seed to pass. This is a new registered
pilot. Original timing runs, their 80 collected executions, failed v4 roots,
and existing registrations remain unchanged. Historical timing outcomes are
not merged into this new sample. The accepted v4 reference pair is reusable
because its complete engineering evidence is validated, copied and bound by
hash before the new run starts.

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
bash scripts/run_xr1_timing_pilot.sh start
```

The launcher validates source provenance before starting a private policy
server, verifies readiness before the simulator, uses unbuffered Python and
durable logs, and cleans up only its own server. It copies a node-local model
from durable storage if the registered checkpoint path requires it. It requires
50 GiB free as a minimum storage guard, not an estimate. Inspect quota failures
before launching any further work.

```bash
source /gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_timing_pilot_latest.env
bash "$ROBOCASA_REPO/scripts/run_xr1_timing_pilot.sh" status
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
bash "$ROBOCASA_REPO/scripts/run_xr1_timing_pilot.sh" resume
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
bash "$ROBOCASA_REPO/scripts/run_xr1_timing_pilot.sh" review /path/to/completed_image_review.json
```

Review is CPU-only and records one immutable assessment. A rejected nominal
image excludes that parent from paired analysis using pre-treatment evidence;
its collected trajectories remain on disk and in coverage reporting. The
review does not certify every suffix camera frame. Full encoded requests and
responses are retained for later inspection, and suffix errors stay explicit.

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
.venv_video/bin/python -m unittest tests.test_intervention_timing_pilot tests.test_intervention_timing_fresh_references tests.test_intervention_timing_source_inspection tests.test_intervention_timing_parent_audit tests.test_intervention_timing_observation_diagnostic tests.test_intervention_timing_replay_diagnostic tests.test_intervention_timing tests.test_operator_screen tests.test_phase2_snapshot_replay tests.test_phase3b_training_horizon tests.test_safe_xiaomi_policy
bash -n scripts/run_xr1_timing_pilot.sh
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
