# Recoverability-ladder v2 pilot: XR-1, five tasks (17–19 October 2026)

Protocol: [recoverability_ladder.md](recoverability_ladder.md) (schema 2).
v1 results: [recoverability_ladder_pilot_20261016.md](recoverability_ladder_pilot_20261016.md).
Run: `/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_ladder_v2_pilot_20261017_164537`
(two workers, one resume; calibration from the v1 pilot plus
`xr1_ladder_v2_naturals_20261017_153727`). Local copy without snapshots,
features or models: `outputs/xr1_ladder_v2_pilot_20261017_164537/`.

## Run health

76 parents (40 failures, 20 success controls, 16 unbranched), 2,020 branches,
0 errors, no duplicate parents after the resume (51 branches of interrupted
attempts are ignored by design). Restore residual 0 in 2,019 branches and
0.041 in one. Budget `remaining_horizon` throughout; 0.055 s per step.

## Detector health

| | stall (v1) | stage_overdue | regression |
|---|---:|---:|---:|
| False alarms on successful parents | 35/36 | 11/36 | 1/36 |

Failures detected by `stage_quantile`: 34; by `regression`: 6. Progress after
detection fell from 13/40 (v1, plus 8 CTS artifacts) to 10/40, but 6 of these
are BreadSelection, whose calibration rests on 3 successes (6/6 false alarms).
CuttingToolSelection's tool stage takes ~980 steps in successful runs, so its
overdue threshold fires at about 0.93 H with about 80 steps left.

## Findings

Contrasts are paired within parent, K=4 per cell, parent-bootstrap 95% CIs.
`R` = resample (fresh seed, reset policy context), `P` = re-prompt with the
stage instruction, `WR` = oracle world reset to the stage-entry state with the
detection clock, `@e` = stage entry, `@e+n` = n inferences later,
`@d` = detection.

### 1. Re-prompting helps or hurts depending on the stage (replicates v1)

P minus R from the **same** state:

| Task / stage at detection | n | @e | @e+4 | @e+8 | @e+16 | @d |
|---|---:|---:|---:|---:|---:|---:|
| ArrangeTea / mug_on_tray | 5 | **+0.20** [0.10, 0.25] | **+0.25** [0.05, 0.50] | **+0.35** [0.10, 0.60] | +0.20 [0.00, 0.50] | +0.15 [0.00, 0.45] |
| GarnishPancake / strawberry_on_pancake | 7 | **−0.46** [−0.71, −0.21] | **−0.36** [−0.57, −0.14] | **−0.43** [−0.68, −0.18] | **−0.25** [−0.54, −0.04] | 0.00 |
| ArrangeBreadBasket / bread_in_basket | 6 | **−0.25** [−0.50, −0.04] | **−0.25** [−0.42, −0.08] | −0.12 | +0.08 | −0.04 |
| All failures pooled | 40 | **−0.14** [−0.24, −0.03] | −0.06 | −0.07 | −0.04 | +0.01 |

The ArrangeTea mug result replicates v1 (0/16 → 4/16), and so does the
GarnishPancake harm (v1 strawberry-stage re-prompt: 0/12). Pooled re-prompting
is harmful, so it must be gated by stage. The v1 BreadSelection jam effect
(0/20 → 19/20) was not retested: only one v2 failure was detected at the jam
stage, because detection now fires at the croissant stage.

### 2. World-state failures and the cost of time (GarnishPancake)

At the strawberry stage (7 parents):

- WR − R@d = **+0.29** [0.11, 0.46]: undoing the world damage restores
  recoverability that the robot cannot recover from by itself (replicates v1).
- R@e − WR = +0.18 [−0.07, 0.46]: the same stage-entry state with the time the
  natural rollout had left at stage entry versus at detection; this is the cost
  of the delay alone.

Robot-only reset has zero value again (pooled RR − R@d = 0.000 [−0.04, 0.04]),
as in v1 and the frozen operator screen.

### 3. Recoverability decays before the detector fires

Resampling from failed parents' states recovered 0.31 [0.21, 0.41] at stage
entry and 0.09 [0.04, 0.16] at detection (paired +0.21 [0.11, 0.32]). Part of
the decay is time: the mean remaining budget falls from 1,904 to 1,381 steps,
and all of CuttingToolSelection's decay is time (R@e − WR = +0.38, 80 steps
left at detection). GarnishPancake's strawberry stage decays to 0.00 with
about 1,470 steps left, more than twice the stage's median duration in
successes, so that decay is the state, not the clock.

### 4. Harm on successful parents

Success controls: resampling 0.90, re-prompt 0.86, robot reset 0.81, world
reset 0.80. The v2 budget fix removed v1's ceiling (v1 resample: 0.575).

## Interpretation

1. Intervention value changes sign with task stage, and in the same way across
   two independent cohorts: re-prompting helps at ArrangeTea's mug stage and
   hurts at GarnishPancake's strawberry stage and ArrangeBreadBasket's bread
   stage. A detector that only says "this will fail" cannot choose among these.
2. Some failures are world-state failures that the robot cannot undo; those
   need a different recovery skill, not a better trigger.
3. A late detector arrives after most of the recoverability is gone, partly
   because the state has degraded and partly because time has run out.

## Recommended next steps

1. **Prospective stage-gated re-prompt trial (first end-to-end test).** On
   fresh seeds, compare plain XR-1 with XR-1 that switches to the stage
   instruction at stage entry only for stages with positive evidence
   (ArrangeTea mug; BreadSelection jam from v1) and never elsewhere, using
   simulator predicates for the stage (oracle stage). The endpoint is full-task
   success over all rollouts, so harm on would-be successes is included. This
   needs no branching.
2. Retest BreadSelection's jam stage directly: anchor at jam-stage entry.
3. Drop robot reset from future ladders; it has shown no value three times.
4. Before scaling the ladder to more tasks, calibrate stage durations with
   more natural successes per task (the BreadSelection and CuttingToolSelection
   thresholds rest on 3 successes each).
