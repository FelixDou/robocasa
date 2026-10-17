# Recoverability-ladder pilot: XR-1, five tasks (16–17 October 2026)

Protocol: [recoverability_ladder.md](recoverability_ladder.md). Run:
`/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_ladder_pilot_20261016_170336`
(commit `b7f1f6d6`, one worker, about 17 h). Local copy without snapshots,
features or models: `outputs/xr1_ladder_pilot_20261016_170336/`.

## Run health

77 parents (40 failures, 10 success controls, 27 unbranched), 968 branches,
0 errors, restore residual 0.0 for every branch, seed control
`policy_sampling_seed_base` throughout. 0.053 s per environment step.

| Task | Parents | Base success |
|---|---:|---:|
| ArrangeBreadBasket | 17 | 0.53 |
| ArrangeTea | 14 | 0.43 |
| BreadSelection | 11 | 0.18 |
| CuttingToolSelection | 15 | 0.47 |
| GarnishPancake | 20 | 0.60 |

## Headline numbers (failed parents, all tasks, K=4)

| Cell | P(success) | Paired vs resample |
|---|---:|---:|
| resample / detection | 0.150 | — |
| robot reset / last good | 0.212 | +0.062 [0.000, 0.125] |
| robot reset / last good − 2 | 0.231 | +0.081 [0.013, 0.150] |
| subtask reprompt / detection | 0.225 | +0.075 [−0.019, 0.156] |
| world reset / last good | 0.200 | +0.050 [−0.013, 0.106] |
| world reset / last good − 2 | 0.219 | +0.069 [0.000, 0.138] |

On success controls, resampling at the matched time succeeded 0.575, and
most interventions lowered success (for example world reset / last good − 2:
0.375; reprompt: 0.450).

**The averages hide the result.** Across tasks and stages the effects have
opposite signs, and two validity problems (below) dilute them.

## Findings

### 1. Instruction-recoverable stalls (strongest result)

At BreadSelection's second stage (jam), the policy stalls under the composite
instruction but completes the task when re-prompted with the stage
instruction "Move jam to cutting board."

| BreadSelection parents stalled at stage 2 | Resample | Reprompt |
|---|---:|---:|
| 5 parents × 4 branches | 0/20 | 19/20 |

Every parent shows it (0.75, 1.0, 1.0, 1.0, 1.0). ArrangeTea's mug stage shows
the same direction, weaker: 0/16 → 4/16 across 4 parents. The policy has the
skill; it loses it under the long instruction at that state.

### 2. World-state failures

At GarnishPancake's strawberry stage (3 parents), resampling, robot reset and
reprompt all gave 0/12 each (robot reset 0/24 over two anchors). Resetting the
world to the last-good state gave 4/12 at last good and 9/12 at last good − 2.
The robot cannot recover by itself; something in the scene must be undone.

### 3. Interventions are not uniformly safe

On success controls, interventions often lowered success (GarnishPancake
reprompt 1.00 → 0.38; ArrangeBreadBasket world reset 1.00 → 0.62). At failed
CuttingToolSelection states, reprompt lowered success from 0.28 to 0.03 (an
artifact, see validity problem A). Unconditional intervention is harmful; it
needs a gate.

### 4. The same score predicts opposite intervention values

Spearman correlation at detection, n=40:

| Score | vs P(resample) | vs P(best cell) |
|---|---:|---:|
| elapsed time | −0.37 [−0.62, −0.08] | +0.34 [0.03, 0.62] |
| ordered progress | −0.39 [−0.57, −0.18] | +0.38 [0.08, 0.62] |
| stall length | −0.03 | −0.10 |

Later or more advanced failures are less recoverable by resampling but more
recoverable by some intervention. This is direct, if small-sample, evidence
for the thesis that failure likelihood and intervention value diverge.

## Validity problems (must be fixed before scaling)

**A. CuttingToolSelection's `drawer_open` stage cannot be reached.**
`CuttingToolSelection` uses `ManipulateDrawer(behavior="close")`, which opens
the drawer at reset with `partial_open=True`, i.e. to 27–30% of its range.
The `drawer_open` predicate uses `Fixture.is_open(th=0.90)`. The tools are
reachable without opening further, so the first ordered stage is never
completed during normal execution. In all 7 successful parents, the first
ordered progress coincides with task success. Consequences here: every CTS
failure was "detected" at step 192 with no progress; the reprompt said
"Open the drawer." to a robot facing an open drawer and never handed back the
task instruction. The same predicate was the CTS target stage
(`OpenDrawer_1` / `drawer_open`) in the frozen exact-replay track, which
plausibly explains its zero CTS completions there
([exact_replay_frozen.md](exact_replay_frozen.md)).

**B. The stall detector is a timer.** It fires after 0.15 H without ordered
progress. It fired before success in 31/36 successful parents. Of the 40
failure parents, only 19 were detected at a genuine stall; 13 made further
ordered progress after detection (the failure had not happened yet), and 8
are the CTS artifact. A detector calibrated per stage from successful runs is
needed.

**C. Ordered progress latches transient truths.** Five failed parents
(ArrangeTea 1, GarnishPancake 4) have every required stage latched yet failed,
because a predicate was true transiently (for example the strawberry over the
pancake while still held) and later undone. Failure localization needs
durable completion or regression detection.

**D. Anchors do not measure decay.** `last_good` is the initial state for 21
of 40 parents, and every anchor is more than 10 inferences before detection,
so the decay table has only the 0 and >10 buckets. World reset to the initial
state is a restart, not a recovery.

**E. Small-sample attribution.** With K=4 and a 0.5 threshold, the
first-rung attribution (55% "competence") is noisy and inflated by A and B.
Per-parent cell estimates are more informative than the class counts.

**F. Budget ceiling.** Even success controls only reach 0.575 under resampling
with a 0.5 H budget and a reset policy context (CuttingToolSelection 0.38,
BreadSelection 0.12, ArrangeTea 0.38). This compresses all rung differences.

## Decision

The approximate-branching design works and is cheap enough to scale. The
pilot already shows two distinct recovery signatures (instruction stalls,
world-state failures) and harm from ungated intervention. Do not scale the
current configuration: fix A–D first, then rerun.

The fixes are implemented as schema 2; see "What changed in v2" in
[recoverability_ladder.md](recoverability_ladder.md).
