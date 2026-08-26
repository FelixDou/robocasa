# XR1 Phase 3 counterfactual candidate-opportunity screen

Phase 3 starts with a read-only analysis of the validated Phase 2 branch set.
It asks a prerequisite question before any critic is trained: did the four
explicitly seeded Xiaomi candidates from the same complete snapshot produce
different recovery outcomes often enough for candidate selection to matter?

This is simulator-oracle evidence. It does not establish that SAFE or a learned
critic can identify the useful candidate online.

## Frozen analysis contract

- Input must be a zero-error Phase 2 run for which every engineering validity
  gate passed and `branch_policy_connection_mode` is `shared_restored`.
- The two same-seed repeats are a reproducibility control and collapse to one
  nominal outcome. Their outcomes must agree exactly.
- The environment-only branch is a restore diagnostic and is excluded from
  candidate-utility estimates.
- The four `candidate` branches form one paired candidate set per snapshot.
- The primary outcome is completion of the active trigger-stage predicate
  within the Phase 2 frozen suffix horizon.
- Candidate payloads are not copied. The manifest stores their immutable paths,
  compressed-file hashes, scientific-payload hashes, and action/feature hashes.
- Task-macro estimates give every task equal weight. Uncertainty bootstraps
  task, then parent, then snapshot.
- Optional SAFE ranking uses the already-frozen three-seed independent SAFE
  ensemble. No checkpoint, normalization, split, or threshold is updated.

## Estimands

For each snapshot, the analyzer reports:

- mixed-outcome indicator;
- expected random-of-four stage completion;
- oracle-best-of-four stage completion and oracle headroom;
- nominal completion and random-minus-nominal benefit;
- harm among nominally successful controls;
- predicate-regression rate;
- pairwise candidate action and SAFE-feature distances;
- when frozen SAFE scoring is enabled, lowest-SAFE completion and its paired
  contrast against random and highest-SAFE selection.

## Continuation gates

The formal five-task Phase 3 gate is:

1. at least 20% of snapshots have mixed candidate outcomes;
2. mixed outcomes occur in at least three of five tasks; and
3. oracle-best-of-four improves completion by at least 15 percentage points
   over expected random-of-four selection.

The completed Phase 2 run contains only two tasks. It can therefore estimate
the first and third criteria but cannot satisfy or fail the formal task-support
criterion. The bounded pilot says `CONTINUE` only if both estimable criteria
pass. A `REDESIGN` result stops critic training and permits one predeclared
change to proposal diversity, branch timing, suffix horizon, or recovery
operator before recollection.

## Command

The command is read-only with respect to the Phase 2 source:

```bash
python -u -m robocasa.recovery.analyze_phase3_candidate_opportunity \
  --phase2-run-dir "$XR1_PHASE2_ROOT" \
  --output-dir "$XR1_PHASE3_ROOT" \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 0 \
  --safe-runtime-bundle "$XR1_SAFE_RUNTIME" \
  --safe-repo "$SAFE_REPO" \
  --safe-device cpu
```

Omit both SAFE arguments to run the proposal-oracle analysis without testing
SAFE selection. The oracle opportunity decision is unchanged by this optional
scorer.

## Outputs

- `candidate_opportunity.jsonl`: one immutable candidate reference and outcome
  per explicit candidate branch;
- `snapshot_opportunity.csv`: paired snapshot-level opportunity metrics;
- `task_summary.csv`: equal-snapshot summary within each task;
- `analysis.json`: source hashes, task-macro results, hierarchical intervals,
  continuation gates, and the bounded scale decision.

The tool refuses to overwrite a non-empty output directory and rehashes the
Phase 2 plan, analysis, branch manifest, and error ledger after analysis to
prove the source was unchanged.
