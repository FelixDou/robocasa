# Which states should be re-prompted? Offline analysis of the stage-gated trial (3 November 2026)

Question: at ArrangeTea's mug stage, re-prompting at stage entry rescued 9
baseline failures but broke 18 baseline successes
([trial results](stage_gated_reprompt_results_20261030.md)). Is there a
signal, available before or during the stage, that separates the two?

Data: the 83 ArrangeTea seeds in which the gate fired, and the 82 such
BreadSelection seeds (`outputs/xr1_stage_gate_20261030_195444/`). Arms are
identical until the first switch, so each seed gives a paired outcome:
helped (+1), harmed (−1) or unchanged. The trial recorded predicate traces
and timing, but no policy features or simulator states, so the SAFE score at
stage entry is not available here.

## Signals at stage entry: none separates help from harm

| Signal at mug-stage entry | Finding |
|---|---|
| Kettle placement flicker before entry | Never occurred (0 in all 83 seeds) |
| Gripper holding an object | Never (released in all 83) |
| Entry time (time to finish the kettle) | Predicts **risk**, not **benefit**: baseline success 0.67 (entry ≤ 250 steps) vs 0.43 (later), but re-prompting lowers success in both groups (0.67 → 0.52, 0.43 → 0.37) |

Failure risk at stage entry does not say whether to intervene, consistent
with the ladder's opposite-sign correlations.

## Time stuck in the stage separates them

Steps from mug-stage entry to mug placement, in the baseline arm:

| Seeds | Baseline duration |
|---|---|
| All 47 baseline successes | 396–708 (median 502, 90th percentile 617) |
| 18 harmed | 446–703, all inside the normal range |
| 9 helped | 492, 492, 595, and 6 never completed |

Under the immediate re-prompt, 10 of the 18 harmed seeds never placed the
mug at all ("Move mug to tray." derails a normal fetch from the cabinet).

A gate that re-prompts only after the stage has run longer than the 90th
percentile of successful durations (617 steps) would, in the baseline arm:

- leave **16 of the 18** harmed seeds untouched;
- still reach **6 of the 9** helped seeds (the 6 that never completed);
- touch 28 of 83 rollouts, only 3 of which succeeded at baseline;
- leave the 6 reached seeds 390–960 steps of horizon after a re-prompt that
  takes as long as the one at entry did.

The trial only re-prompted at entry, so the effect of a **delayed**
re-prompt is not identified here. The ladder supports it: from failing
ArrangeTea mug-stage states, re-prompt minus resample was +0.20 to +0.35 up
to 16 inferences after entry and +0.15 [0.00, 0.45] at detection (5 parents).

## BreadSelection needs the opposite timing

Successful baseline jam stages are long: median 877 steps, 90th percentile
1,396. Delaying the BreadSelection re-prompt by that much would leave almost
no time: 24 of the 25 helped seeds it would still reach would run out of
horizon. There the immediate re-prompt is right: it harms only 4 of 28
baseline successes, and all 4 had unusually long baseline jam stages
(1,047–1,388 steps).

## Why the ladder mispredicted ArrangeTea

The ladder's ArrangeTea success controls were all branched at the cabinet
stage (steps 832–1,120), never at the mug stage. It therefore never measured
mug-stage harm on rollouts that would have succeeded, only re-prompt value
from failing states. For future ladders, harm controls must be branched at
the candidate intervention stage.

## Conclusion and proposal

The intervention decision depends on stage **and** time in stage:

- a stage where the policy usually fails (BreadSelection jam, completed in
  43% of entries at baseline): re-prompt at entry;
- a stage the policy usually completes (ArrangeTea mug, 72% of entries):
  re-prompt only once the stage is overdue against successful durations,
  i.e. the ladder v2 `stage_overdue` rule used as a gate, not as an alarm.

**Proposed second prospective trial (stall-gated re-prompt):** fresh seeds;
ArrangeTea and BreadSelection; arms `baseline`, `entry_gate` (replicates
trial 1) and `stall_gate` (ArrangeTea mug re-prompted after 617 steps in the
stage, frozen from this trial's baseline arm; BreadSelection jam at entry).
Primary: `stall_gate − baseline` on ArrangeTea; secondary: macro over both
tasks and `stall_gate − entry_gate`. Record policy features at stage entry
and at the gate decision so a SAFE-based gate can be evaluated offline
afterwards. A rough projection of the ArrangeTea effect is +0.04 to +0.09; size the
trial for that with the trial planner's power table before running.
