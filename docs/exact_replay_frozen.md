# Exact-replay track: frozen (16 October 2026)

The exact complete-snapshot replay track is **frozen**. Do not extend it, add
diagnostics to it, collect new parents with it, or resume its registrations
except to close out the timing pilot image review below.

## What is frozen

| Stage | Doc | Outcome |
|---|---|---|
| Phase 2 exact snapshot replay | [xr1_phase2_complete_snapshot_replay.md](xr1_phase2_complete_snapshot_replay.md) | Engineering passed (10 parents, 20 snapshots, 140 branches) |
| Phase 3 short-horizon opportunity | [xr1_phase3_candidate_opportunity.md](xr1_phase3_candidate_opportunity.md) | 0/80 completions; no headroom |
| Phase 3B training-horizon continuation | [xr1_phase3b_training_horizon.md](xr1_phase3b_training_horizon.md) | 1/20 mixed snapshots; oracle-minus-random 0.0125 |
| Higher-level operator screen | [xr1_operator_screen.md](xr1_operator_screen.md) | Continue 22.5% vs 17.5% for every intervention; zero routing gain |
| Common-entry timing study and references | [xr1_earlier_intervention_timing_study.md](xr1_earlier_intervention_timing_study.md), [xr1_fresh_timing_references.md](xr1_fresh_timing_references.md) | Stopped at 80/220; replaced by the pilot |
| Audit-and-exclude timing pilot | [xr1_timing_audit_exclude_pilot.md](xr1_timing_audit_exclude_pilot.md) | 72/72 executions on 3 CuttingToolSelection parents; 7/10 parents excluded; no ArrangeTea |

## Why it is frozen

- Every scientific gate failed on two tasks and ten parents, and none of the
  states had meaningful outcome variance. CuttingToolSelection never completed
  its target stage in any arm.
- Bit-exact determinism is not needed for the scientific question (does an
  intervention at a given state raise P(success)?). Common random numbers
  reduce variance but have cost seven weeks and 70% parent attrition.
- The successor is the scaled, approximately branched recoverability-ladder
  study, which reuses `full_snapshot`, `recovery_rollout`,
  `evaluate_recovery_benchmark` and the policy adapters, not this package.

## Where the code lives

- Modules: `robocasa/recovery/exact_replay/`
- Launchers: `scripts/exact_replay/`
- Tests: `tests/exact_replay/` (run as `python -m unittest tests.exact_replay.test_<name>`)
- Byte-identical code that produced the registered results: git tag
  `exact-replay-frozen-20261016` (commit `4d65b542`, also on `origin/main`).

`full_snapshot.py` stays in `robocasa/recovery/` because the Xiaomi policy
adapter and the new study use it.

Registered runs hash their code by repository-relative path. They therefore
**cannot be resumed or reviewed from the moved code**. To touch any existing
registration, use a checkout of the frozen commit:

```bash
cd /gs/fs/tga-shinoda/felid/robocasa
git fetch origin
git worktree add /gs/fs/tga-shinoda/felid/robocasa_exact_replay_frozen 4d65b542
export ROBOCASA_REPO=/gs/fs/tga-shinoda/felid/robocasa_exact_replay_frozen
```

## Final close-out: timing pilot image review

This is the only remaining action on the track. It is CPU-only and records the
pilot as `complete_limited_coverage` with decision
`insufficient_valid_task_coverage` (only CuttingToolSelection qualified).
The commands are in
[xr1_timing_audit_exclude_pilot.md](xr1_timing_audit_exclude_pilot.md#image-review-and-scientific-interpretation);
run them with `ROBOCASA_REPO` pointing at the frozen checkout above, and
`scripts/run_xr1_timing_pilot.sh` (the frozen path) rather than
`scripts/exact_replay/run_xr1_timing_pilot.sh`.

The `images_acceptable` judgement must come from actually viewing every
nominal frame in `camera_review.html`; do not fill it in without that review.
After recording it, append the final status and the per-cell descriptive
outcomes to the pilot doc and close this track.
