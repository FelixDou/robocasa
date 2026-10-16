# Exact-replay track: frozen (16 October 2026)

The exact complete-snapshot replay track is **frozen and closed**. Do not
extend it, add diagnostics to it, collect new parents with it, or resume its
registrations. Its last open item, the timing pilot image review, was recorded
on 16 October 2026 (status `complete_limited_coverage`).

## What is frozen

| Stage | Doc | Outcome |
|---|---|---|
| Phase 2 exact snapshot replay | [xr1_phase2_complete_snapshot_replay.md](xr1_phase2_complete_snapshot_replay.md) | Engineering passed (10 parents, 20 snapshots, 140 branches) |
| Phase 3 short-horizon opportunity | [xr1_phase3_candidate_opportunity.md](xr1_phase3_candidate_opportunity.md) | 0/80 completions; no headroom |
| Phase 3B training-horizon continuation | [xr1_phase3b_training_horizon.md](xr1_phase3b_training_horizon.md) | 1/20 mixed snapshots; oracle-minus-random 0.0125 |
| Higher-level operator screen | [xr1_operator_screen.md](xr1_operator_screen.md) | Continue 22.5% vs 17.5% for every intervention; zero routing gain |
| Common-entry timing study and references | [xr1_earlier_intervention_timing_study.md](xr1_earlier_intervention_timing_study.md), [xr1_fresh_timing_references.md](xr1_fresh_timing_references.md) | Stopped at 80/220; replaced by the pilot |
| Audit-and-exclude timing pilot | [xr1_timing_audit_exclude_pilot.md](xr1_timing_audit_exclude_pilot.md) | `complete_limited_coverage`: 3 CuttingToolSelection parents, 0% completion in all 10 cells (256-step budget from reset); 7/10 parents excluded; no ArrangeTea |

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

## Final close-out: timing pilot image review (done)

Recorded on 16 October 2026 from the pre-move cluster checkout. All three eligible
parents passed image review (each has a single reset-state frame). The pilot
ended `complete_limited_coverage` with decision
`insufficient_valid_task_coverage` and zero completion in every cell. It is
uninformative about timing because the entry point is the episode start and
the 256-step budget is a floor effect. Full numbers are in
[xr1_timing_audit_exclude_pilot.md](xr1_timing_audit_exclude_pilot.md#final-result-16-october-2026).
