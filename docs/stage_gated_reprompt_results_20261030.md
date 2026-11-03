# Stage-gated re-prompt trial: results (XR-1, run 30–31 October 2026)

Protocol and pre-registered analysis: [stage_gated_reprompt.md](stage_gated_reprompt.md).
Ladder evidence behind the whitelist:
[v1](recoverability_ladder_pilot_20261016.md), [v2](recoverability_ladder_v2_pilot_20261017.md).
Run: `/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_stage_gate_20261030_195444`
(two workers, commit `d7e2216e`). Local copy: `outputs/xr1_stage_gate_20261030_195444/`.

## Run health

300 complete seed pairs (100 per task) × 3 arms = 900 rollouts, 0 errors,
0 initial-state mismatches. On the control task, gated and baseline rollouts
had identical state traces in 96/100 seeds and the same outcome in 98/100;
the 2 discordant pairs without any switch bound simulator/server
nondeterminism at about 2%. In the smoke run, arms diverged exactly at the
inference after the first switch.

## Primary endpoint (pre-registered)

Full-task success, `gated_reprompt − baseline`, macro over the whitelisted
tasks (ArrangeTea, BreadSelection), 200 pairs:

**+0.080 [+0.010, +0.155]** (paired bootstrap 95% CI); exact McNemar on the
pooled pairs: 38 helped / 22 harmed, **p = 0.052**.

The pre-registered test (exact McNemar, α = 0.05) **narrowly fails**. The
bootstrap interval excludes zero, but the primary decision rule was McNemar,
so the primary hypothesis is not confirmed. The pooled number also hides two
opposite task effects.

## Per task

| Task (whitelisted stage) | Baseline | Gated | Δ [95% CI] | Helped / harmed | Holm p | Rescue among baseline failures | Harm among baseline successes |
|---|---:|---:|---|---:|---:|---:|---:|
| BreadSelection (jam) | 0.28 | **0.53** | **+0.25 [+0.15, +0.35]** | 29 / 4 | < 0.0001 | 29/72 (0.40) | 4/28 (0.14) |
| ArrangeTea (mug) | 0.49 | 0.40 | −0.09 [−0.19, +0.01] | 9 / 18 | 0.12 | 9/51 (0.18) | 18/49 (0.37) |
| GarnishPancake (control, none) | 0.71 | 0.71 | 0.00 [−0.03, +0.03] | 1 / 1 | — | — | — |

BreadSelection also succeeds faster under gating (restricted mean steps
−393 [−495, −299]); its jam stage completes in 0.56 of rollouts instead of
0.35.

### Re-prompting at every stage (secondary arm)

| Task | Baseline | Always re-prompt | Δ [95% CI] | Harm among baseline successes |
|---|---:|---:|---|---:|
| GarnishPancake | 0.71 | **0.11** | **−0.60 [−0.70, −0.48]** | 64/71 (0.90) |
| ArrangeTea | 0.49 | 0.35 | −0.14 [−0.26, −0.01] | 29/49 (0.59) |
| BreadSelection | 0.28 | 0.52 | +0.24 [+0.13, +0.34] | 7/28 (0.25) |

Over all three tasks, gating beats re-prompting everywhere by **+0.22
[+0.16, +0.28]** (96 helped / 30 harmed, p < 0.0001). At GarnishPancake's
strawberry stage, "Move strawberry to pancake." almost never completes the
stage (1% of switches; 97/98 hit the 1,280-step cap).

## How the ladder predictions transferred

| Stage | Ladder (same-state, failed parents) | Trial (all rollouts, fresh seeds) | Transferred? |
|---|---|---|---|
| BreadSelection jam | +: 0/20 → 19/20 (v1) | +0.25 | **Yes** |
| GarnishPancake strawberry | −0.25 to −0.46 (v2), 0/12 (v1) | −0.60 (always arm) | **Yes** |
| ArrangeTea mug | +0.20 to +0.35 (v2), 0/16 → 4/16 (v1) | −0.09 | **No: sign flips** |

The ladder measured re-prompting from states of rollouts that went on to
fail. At ArrangeTea's mug stage those states were poor: resampling from them
reached full-task success 5–15% of the time, versus 49% for baseline
rollouts in this trial. The trial re-prompts every rollout that reaches the
stage. There,
re-prompting rescues 18% of would-be failures but breaks 37% of would-be
successes. At BreadSelection's jam stage the policy fails often enough
(baseline completes it in 35%) that rescue dominates.

## Interpretation

1. **Stage-level gating works where the stage is a reliable failure point.**
   BreadSelection: +25 points absolute, 29 rescued for 4 harmed. Gating also
   prevents the large harm of re-prompting everywhere (GarnishPancake −60
   points).
2. **Stage alone is not a sufficient gate.** At ArrangeTea the same stage
   contains good and bad states, and the intervention's value has opposite
   signs on them. Counterfactual value measured on failing states does not
   transfer to all states. The gate has to condition on the current state's
   failure risk, not only on the stage.
3. This is the bridge to the detection work: a failure predictor (SAFE,
   time, or a learned value) is useful as an input to a **state-and-stage
   gate**, not as a trigger by itself.

## Next steps

1. Report the trial as pre-registered: primary not confirmed (p = 0.052),
   BreadSelection confirmed, ArrangeTea prediction failed with a mechanism.
2. **State-conditioned gate (offline first).** Using the ladder branches and
   these trial rollouts, test whether a state signal at stage entry (time in
   stage, SAFE score, policy-feature value head) separates states where
   re-prompting helps from those where it hurts at ArrangeTea's mug stage.
   If yes, a second prospective trial with a state-and-stage gate.
3. Keep BreadSelection's jam re-prompt as the positive end-to-end result and
   GarnishPancake as the harm example for the paper's recovery section.
