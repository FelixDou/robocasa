# Stage-gated re-prompt trial 2: results (XR-1, run 4–5 November 2026)

Design and pre-registered analysis: [stage_gated_reprompt_trial2.md](stage_gated_reprompt_trial2.md).
Why this trial: [trial 1 results](stage_gated_reprompt_results_20261030.md),
[offline analysis](stage_gate_state_signal_analysis_20261103.md).
Run: `/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_stage_gate_trial2_20261104_104336`
(two workers, commit `23b498ce`, one resume). Local copy without features:
`outputs/xr1_stage_gate_trial2_20261104_104336/`.

## Run health

- 400 complete seed pairs (seeds 410007–410206 for each task) × 3 arms =
  1,200 rollouts; 0 errors; 0 initial-state mismatches; every rollout under
  `policy_sampling_seed_base`.
- The resume left 3 orphan rows from seeds in progress at the kill. Their
  re-runs had identical state traces, and the analyzer uses only the rows
  listed in `pairs.jsonl`.
- Every stall switch happened exactly 617 steps after mug-stage entry
  (58/58). BreadSelection stall switches happened at entry (163/163).
- Every rollout that reached a whitelisted stage has policy features
  (inference 0 of the stage is recorded unswitched in all 173 ArrangeTea
  stall-arm rollouts).
- **Nondeterminism:** 4 of 800 non-baseline rollouts, all in the stall arm,
  diverged from baseline before any switch:
  - ArrangeTea 410055 and 410094, inside the mug stage;
  - BreadSelection 410014 and 410151, at step 96.

  One of the three ArrangeTea stall "harms" (410055) is one of these. The two
  BreadSelection cases account for the 2/0 `gated − stall` difference there.
  This rate (0.5%) is below trial 1's control-task bound (about 2%).

## Primary endpoint (pre-registered)

Full-task success, `stall_gated_reprompt − baseline` on ArrangeTea, 200 pairs:

**+0.005 [−0.020, +0.030]** (paired bootstrap 95% CI); exact McNemar
4 helped / 3 harmed, **p = 1.00**.

**The primary hypothesis is not confirmed.** The stall gate removed the
harm of re-prompting at ArrangeTea's mug stage, but rescued almost nothing.

## Per task and arm

| Task | Arm | Success | Δ vs baseline [95% CI] | Helped / harmed | McNemar p | Rescue among baseline failures | Harm among baseline successes |
|---|---|---:|---|---:|---:|---:|---:|
| ArrangeTea | baseline | 0.535 | — | — | — | — | — |
| ArrangeTea | entry gate | 0.475 | −0.060 [−0.145, +0.025] | 32 / 44 | 0.21 | 32/93 (0.34) | 44/107 (0.41) |
| ArrangeTea | stall gate | 0.540 | +0.005 [−0.020, +0.030] | 4 / 3 | 1.00 | 4/93 (0.04) | 3/107 (0.03) |
| BreadSelection | baseline | 0.330 | — | — | — | — | — |
| BreadSelection | entry gate | 0.580 | **+0.260 [+0.190, +0.330]** | 61 / 9 | < 0.0001 | 61/134 (0.46) | 9/66 (0.14) |
| BreadSelection | stall gate (delay 0) | 0.570 | **+0.250 [+0.180, +0.320]** | 59 / 9 | < 0.0001 | 59/134 (0.44) | 9/66 (0.14) |

Secondary contrasts:

- **Macro over both tasks**, stall gate − baseline: +0.128 [+0.090, +0.165],
  63 / 12, p < 0.0001. BreadSelection drives this.
- **Value of waiting at ArrangeTea**, entry gate − stall gate: −0.065
  [−0.150, +0.025], 33 / 46, p = 0.18.
- **Speed at BreadSelection:** the entry gate shortens restricted mean steps
  by −384 [−457, −311].

## Replication of trial 1

| | Trial 1 (100 seeds) | Trial 2 (200 seeds) | Pooled |
|---|---|---|---|
| BreadSelection jam, entry gate | +0.25, 29 / 4 | +0.26, 61 / 9 | 90 / 13 |
| ArrangeTea mug, entry gate | −0.09, 9 / 18 | −0.06, 32 / 44 | 41 / 62, McNemar p = 0.048 |

Both trial-1 effects replicate in sign and size on fresh seeds. The pooled
ArrangeTea p-value is a post hoc pooling, reported for description only.

The stall threshold is also stable: the 90th percentile of baseline
mug-stage durations among successes is 619 in trial 2, versus 617 frozen from
trial 1 (median 490 vs 502).

## Why the stall gate did not help

The trial's paired arms separate **which seeds** are re-prompted from **when**
they are re-prompted. Here are ArrangeTea outcomes by the stall gate's
decision (the decision is the same in every arm, because arms are identical
until the switch):

| Seeds (by the stall gate's decision) | n | Baseline | Entry gate (helped / harmed) | Stall gate (helped / harmed) |
|---|---:|---:|---|---|
| Mug placed within 617 steps (not re-prompted) | 115 | 92 | 62 (11 / **41**) | 92 (0 / 0) |
| Still in mug stage at 617 steps (re-prompted) | 58 | 12 | **30 (21 / 3)** | 13 (4 / 3) |
| Never reached the mug stage | 27 | 3 | 3 | 3 |

1. **The selection is right.** Seeds that will overrun the stage are the ones
   re-prompting helps: on those 58 seeds, re-prompting at entry rescued 21
   and broke 3. Seeds that finish on time are the ones it harms: 41 broken
   for 11 rescued.
2. **The timing is wrong.** Re-prompting the same 58 seeds 617 steps later
   rescued 4. By then the rollout is stuck: after the late switch, 20 of 58
   never placed the mug within the 1,280-step cap, 7 ran out of horizon, 18
   placed it but failed a later stage, and 13 succeeded. The median switch
   came at step 889 of 2,250.
3. **The remaining harm comes from interrupting late placements.** Two of the
   three stall harms switched about 60 steps before baseline would have
   placed the mug (switch at 964 vs placement at 1,026; 1,058 vs 1,083). The
   third is the nondeterministic seed above.

The offline projection (+0.04 to +0.09) assumed that a delayed re-prompt
recovers stuck rollouts about as well as one at entry. It did not. This also
qualifies the ladder: its +0.15 for re-prompting at the detection anchor came
from 5 parents. On 58 prospective seeds, a late re-prompt rescued 4/46
baseline failures (0.09).

## What would have worked: an entry-time stall predictor

Re-prompting at entry only on the seeds the stall gate later selected would
have given ArrangeTea success **0.625**, i.e. **+0.090 [+0.045, +0.135]**
over baseline. This uses future information (whether the stage will
overrun), so it is an **upper bound** for a gate that must decide at stage
entry. The best arm per seed would reach 0.705.

So the question for a third trial is whether the state at mug-stage entry
predicts an overrun. Trial 1 found that entry time alone does not separate
help from harm. Trial 2 recorded, for all 173 ArrangeTea seeds that entered
the stage, the XR-1 SAFE token at the stage's first inference before any
switch (stall arm). For an entry-time rule, the counterfactual outcome of
re-prompting is observed exactly: it is the entry-gate arm of the same seed.
That supports an exact cross-validated offline evaluation:

- **label:** the stage overruns 617 steps (58 positives / 173);
- **predictor:** SAFE features at inference 0, and inferences 0–4 for a
  slightly later decision (that counterfactual is not observed, so it is
  estimate-only);
- **policy value:** entry-gate outcome where the predictor fires, baseline
  outcome elsewhere, under seed-level cross-validation; compare against the
  0.090 upper bound and against the stall gate.

## Interpretation

1. **BreadSelection jam re-prompt at entry is now a replicated result:**
   +0.25 and +0.26 in two prospective trials on fresh seeds (90 rescued / 13
   harmed pooled).
2. **ArrangeTea: intervention value depends on the state, and the useful
   window is early.** Time in stage identifies the right seeds only after the
   point where re-prompting still works. A gate needs a *predictive* signal at
   entry, not a *diagnostic* one after the stall.
3. This links detection to recovery more precisely than trial 1 did. The
   detector's job at an intervention point is to predict **whether the
   intervention will help**, early enough for it to work. A late, accurate
   failure alarm (stage overdue) has little intervention value here.

## Next steps

1. Report trial 2 as pre-registered: primary not confirmed (+0.005,
   p = 1.00). The secondary replication of both trial-1 effects holds.
2. **Offline entry-time stall predictor** from the recorded SAFE features
   (features stay on the cluster under the run's `features/`; only the
   ArrangeTea stall-arm files are needed). If its cross-validated policy
   value is clearly above zero, freeze the predictor and threshold and run a
   third prospective trial (`baseline` vs `predicted_stall_entry_gate` on
   ArrangeTea, fresh seeds).
   **Done (5 November):** no entry-time gate passed; best Δ −0.000, stall
   AUROC ≤ 0.64, while about 0.87 would be needed
   ([stage_gate_entry_predictor.md](stage_gate_entry_predictor.md#results-5-november-2026)).
   No third trial.
