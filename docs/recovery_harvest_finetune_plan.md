# Alarm-triggered recovery from self-harvested data (XR-1): plan

Written 5 November 2026, before any collection, code or scoring for it.
Status: **draft**, not yet frozen. Gates and thresholds below become binding
when this file is committed with the code it names.

Evidence behind it:
[ladder v1](recoverability_ladder_pilot_20261016.md),
[ladder v2](recoverability_ladder_v2_pilot_20261017.md),
[re-prompt trial 1](stage_gated_reprompt_results_20261030.md),
[trial 2](stage_gated_reprompt_trial2_results_20261105.md),
[entry-time gate](stage_gate_entry_predictor.md#results-5-november-2026),
[frozen exact-replay screens](exact_replay_frozen.md),
report sections 15.17, 15.21, 15.24 and 15.26
([`safe_robocasa_experiment_report.md`](safe_robocasa_experiment_report.md)).

## Question

When Hide-and-Seek + P raises an alarm, can XR-1 be switched into a
**recovery mode**, learned from its own successful continuations out of
similar states, that raises full-task success without hurting rollouts that
would have succeeded?

At deployment nothing privileged is used: the alarm reads the policy's own
features and the elapsed time, and the recovery mode is the same network with a
different prompt. Simulator snapshots are used only to harvest training data.

## Why this, and not inference-time fixes

What the current recovery operators achieve from failing XR-1 states:

| Operator | Result | Source |
|---|---|---|
| Pick the best of 4 action chunks at a failure snapshot | +0.0125 even with an oracle; mixed outcomes in 1/20 snapshots | Phase 3B (frozen) |
| Robot (arm) rewind | zero value, three times | ladder v1, v2, operator screen |
| Re-prompt with the stage instruction | +0.25 / +0.26 at BreadSelection's jam stage (replicated); harmful at ArrangeTea, GarnishPancake, ArrangeBreadBasket; no entry signal predicts benefit (AUROC 0.60–0.64, about 0.87 needed) | trials 1–2, entry-gate analysis |
| Re-prompt once the stage has stalled | rescued 4 of 58 | trial 2 |
| Retry the failed scene from reset | 0.245 (fresh scenes: 0.486) | 15.24 |
| Keep running 25% past the limit | 0.04 | 15.26 |
| Oracle world reset to stage entry | +0.29 at GarnishPancake's strawberry stage | ladder v2 |

Two observations from the ladder v2 branches (computed 5 November from
`outputs/xr1_ladder_v2_pilot_20261017_164537/`, failed parents, K = 4 resamples
per cell):

| Anchor | Mean success | Best of 4 continuations (oracle) | States with mixed outcomes |
|---|---:|---:|---:|
| Stage entry | 0.30 | 0.55 | 0.48 |
| Ladder detection | 0.09 | 0.20 | 0.17 |

1. Choosing among the policy's own samples has headroom only while the
   policy still produces successes from the state. Ranking cannot create
   behaviour the policy does not have.
2. The ladder still produced **276 successful continuations out of 1,412**
   resample and re-prompt branches from failed parents. 32 of the 40 failed
   parents had at least one. These are on-policy recovery demonstrations from
   failure-prone states, with paired failures from the same states.

**Caveat on the detector.** The ladder's detection anchor is its own
stage-overdue / regression rule, not Hide-and-Seek + P. That rule fired on 12
of 36 successful parents. On failed parents it fired at a median of 0.11–0.48
of the time limit on four tasks (0.93 on CuttingToolSelection). Hide-and-Seek +
P has a 0.061 false-alarm rate and a mean lead of 0.296 of the time limit
(confirmatory cohort, misses counted as 0, report 15.21). It therefore usually
fires later and much more selectively. **Recoverability at its alarm has not
been measured**, so Phase A measures it before anything is harvested.

## Related work and how this differs

| Work | What it does | Difference here |
|---|---|---|
| [Kintsugi-VLA](https://arxiv.org/html/2609.31048) (Sep 2026) | Branches from failed-rollout snapshots; a privileged PPO expert supplies recovery demos; one task, SmolVLA | No expert: demos are the policy's own successful branches; states chosen by a deployable detector; long-horizon composite tasks |
| [RaC](https://arxiv.org/pdf/2509.07953) (2025) | Human recovery and correction data; success scales with recovery maneuvers | Automatic, from simulator branches |
| [IntervenGen](https://arxiv.org/html/2405.01472v1) (2024) | Transforms a few human corrections into many with MimicGen-style replay | Fallback if the policy cannot produce its own recoveries (Phase A gate) |
| [π\*0.6 / RECAP](https://arxiv.org/abs/2511.14759) (2025) | Advantage-conditioned VLA: a good/bad indicator in the prompt, "good" at inference | Model M3 below, with outcome labels from paired same-state branches |
| [Backtrack](https://arxiv.org/abs/2610.32941) (Oct 2026) | Failure head on frozen-VLA features, then arm rewind and retry; LIBERO-Long | Arm rewind had zero value on RoboCasa365 three times |
| [RESCUE](https://arxiv.org/abs/2610.13516) (Oct 2026) | Counterfactual evaluation of monitor alarms on LIBERO; detection quality does not imply intervention value | Same lesson; here it is acted on with a learned recovery mode and a prospective trial |
| [DiVeR](https://arxiv.org/html/2610.04933v1) (Oct 2026) | Verifier best-of-16 at every step; +5.9 (π0), +11.6 (π0.5) on RoboCasa atomic tasks | Always-on selection; this plan intervenes only after an alarm |

## Scope

- **Policy:** XR-1 (main), the policy behind the detector, the ladder and both
  trials. Its fine-tuning is new to this project; Step 0 sets it up.
- **Tasks (pilot):** the five ladder tasks, all among the detector's 38 frozen
  tasks: ArrangeBreadBasket, ArrangeTea, BreadSelection, CuttingToolSelection,
  GarnishPancake. Base success 0.18–0.60 (ladder v1).
- **Alarm:** the frozen Hide-and-Seek + P of Amendment 2 (detector, per-task
  normalization, P timer at q = 0.95 with its conformal margin, chosen fused
  path (u, s)), read from the Amendment 2 run. Nothing is refitted.
- **Branching:** approximate state branching as in the ladder (no exact replay;
  [exact_replay_frozen.md](exact_replay_frozen.md)).

## Step 0: feasibility (before any code beyond what is listed)

XR-1 has never been fine-tuned in this project. The only earlier fine-tune is
π0 through OpenPI, on the full RoboCasa train set, which took several days.
Fine-tuning on a dataset of about 1,000 episodes is expected to be affordable,
but the XR-1 pipeline has to be set up first:

- Find the XR-1 training entry point and its data format in the Xiaomi-Robotics-1
  release, and check it can start from the RoboCasa365 checkpoint used here.
- Smoke-train on a few harvested-format episodes. Record the steps per hour,
  GPU memory, and the projected time for about 1,000 episodes.
- Sanity check: a checkpoint saved after a negligible update (for example a
  few steps at a tiny learning rate) must match base XR-1's success on about
  20 natural rollouts per task within noise. This shows the save/load and
  serving path does not change the policy.
- Confirm that the recorded ladder v2 features give the 1,024-d selector that
  Hide-and-Seek reads (last flow step, last action position), as the
  entry-gate `extract` step does for the trial features.
- If XR-1 cannot be fine-tuned, stop and revise. The fallback is π0 through
  OpenPI, whose fine-tuning is known to work here. It needs Hide-and-Seek
  retrained on π0 rollouts with failure labels first.

## Phase A: recoverability at the Hide-and-Seek + P alarm (no new parents)

**A0 (CPU, offline).** Score the 76 ladder v2 parents with the frozen
Hide-and-Seek + P and record, per parent, the first alarm inference `a`,
which component fired (score or timer), and `a` against the ladder's anchors.

Integrity checks, all required (the export step stops on any failure):
- the retrained detector seeds, after saving and reloading, give the same
  alarm as the stored Amendment 2 trajectory on every rollout of the
  confirmatory test cohort, with raw scores within 1e-5 and recomputed
  normalizations within 1e-6. The spec uses the stored normalizations.
  Whether the reproduction was bit-exact is recorded;
- *Amended 5 November 2026, before any ladder parent was scored:* the first
  export required bit-exact reproduction (at most 1e-12). It stopped with the
  recomputed normalizations 3.0e-9 from the stored ones, which is float32
  rounding from retraining on a different CPU, so the criterion is now
  alarm identity;
- P and the fused path (u, s, lam), refitted with the registered
  `confirm_lead.py` code on the Amendment 2 calibration cohort, equal the
  values stored in `confirm_hide_and_seek.json`;
- on the ladder parents, only the alarm and its timing are read. AUROC and
  FPR are reported for description, not used for any decision.

**Float16 caveat (resolved).** The ladder v2 collector stored policy features
in float16, while Hide-and-Seek was trained and confirmed on float32. On the
1,526 confirmatory test rollouts, float16 rounding changed no alarm, so the A0
alarms on the v2 parents stand. New collections still set
`policy_feature_dtype: float32`.

**A0 result (5 November 2026).** Export `hs_p_alarm_spec_20261105_165327`:
- reproduction within float32 rounding (raw 5.8e-10, normalizations 3.0e-9)
  and 0 of 1,526 reference alarms changed;
- on all retained reference rollouts, 605/605 failures and 88/921 successes
  got an alarm.

Ladder v2 parents (`xr1_alarm_phaseA_20261105_170817/alarms.jsonl`):
- every one of the 40 failures got an alarm, and 1 of 36 successes did
  (GarnishPancake). The ladder's stage-overdue rule had 12 of 36;
- median alarm time on failures, as a fraction of the time limit, and the
  component that fired:

| Task | Median alarm time | Component | Alarm vs ladder detection |
|---|---:|---|---|
| ArrangeBreadBasket | 0.62 | score (8/8) | later in 8/8 (median +102 inferences) |
| ArrangeTea | 0.84 | score 5, timer 3 | later in 8/8 (+78) |
| BreadSelection | 0.99 | timer (8/8) | later in 8/8 (+108) |
| CuttingToolSelection | 0.99 | timer (8/8) | later in 8/8 (+4) |
| GarnishPancake | 0.51 | score (8/8) | earlier in 3, later in 5 (+9.5) |

On these tasks, Hide-and-Seek + P buys its precision with lateness. On
BreadSelection and CuttingToolSelection the alarm is effectively the timeout.

**A1 (simulator).** For every failed parent with an alarm, branch from its
saved snapshots at `a`, `a − 8` and `a − 16` inferences (clipped at stage entry),
K = 8 resamples each, policy context reset. Success controls with a false
alarm get the same cells (harm data). Estimated cost: about 1,000 branches,
15–20 worker-hours.

*Amended 5 November 2026, after A0 and before any branch:* the budget is a
fixed 0.5 × the time limit from the branch start (with the source run's
2,200-step cap on ArrangeBreadBasket), not the remaining horizon. With alarms
at 0.84–0.99 of the limit on three tasks, a remaining-horizon budget would
leave about 1–16% of the episode. Gate A would then measure the clock rather
than whether the state can be recovered from, and harvesting recovery data does
not depend on the benchmark clock. Each branch also records whether it
succeeded within the time the natural rollout had left; `analyze` reports
that deployment view next to Gate A.

The environment is rebuilt as the ladder built it (one env per task, then
`reset(seed)` for every attempt seed in order). A parent is branched only if
its reset state equals its saved first snapshot (max difference ≤ 1e-6);
mismatches are recorded and excluded from Gate A.

**Gate A** (decides where Phase B harvests):

| Share of alarmed failure states with ≥ 1 success in 8 resamples (fixed budget) | Decision |
|---|---|
| ≥ 0.30 at `a` or `a − 8` | harvest at `a`, `a − 8`, `a − 16` |
| ≥ 0.30 only at `a − 16` | harvest at `a − 16` only; the recovery mode must then generalize to later states |
| < 0.30 at every anchor | on-policy harvesting cannot supply alarm-state data; stop and redesign around expert data (IntervenGen-style) |

For reference, ladder v2 at its own detection anchor: 0.20 with K = 4.

## Phase B: harvest

- **Parents:** fresh seeds (new base seed), natural XR-1 rollouts with
  features, until 30 failed parents per task have an alarm (expected about
  40–120 parents per task, depending on the task's success rate; about 10–15
  worker-hours).
- **Branches:** the Gate A anchors, K = 8 resamples under the task instruction.
  Re-prompt branches (stage instruction) are a tagged secondary source and
  are relabelled with the task instruction only in the ablation model.
- **Recorded per branch, success or failure:** the observations exactly as
  the policy received them, the executed actions, instruction, anchor, parent,
  seed and outcome.
- **Expected yield:** at 15–20% branch success, 3,600 branches give about
  540–720 recovery demonstrations (about 110–145 per task) and about 2,900
  failed continuations from the same states. Cost about 60–75 worker-hours
  (about 15–19 h on 4 workers; ladder v2 averaged 77 s per branch).
- **Replay data:** the policy's own successful natural parents from Phase B,
  to keep normal behaviour (self-distillation, no human demos).

Dataset health checks before training (all required): sample count per task
and per anchor, task coverage and balance, zero collection errors, no
duplicate branch IDs, every artifact present and readable, restore residuals,
and no parent shared between the harvest and the held-out sets.

## Phase C: training (two models, one recovery prompt)

Recovery mode is a fixed suffix appended to the task instruction (exact text
frozen in the code). Without the suffix the model should behave like base
XR-1.

| Model | Training data | Use |
|---|---|---|
| **M1: filtered BC** | successful branches with the suffix + successful naturals without it (mix frozen in the config) | suffix after the alarm |
| **M3: advantage-conditioned** | successful branches with a "good" indicator, failed branches from the same states with a "bad" indicator, naturals without either | "good" indicator after the alarm |
| M1-rp (ablation, if compute allows) | M1 + relabelled re-prompt successes | as M1 |

Every model starts from the same XR-1 checkpoint and uses one frozen training
budget.

## Phase D: held-out states (Gate B)

- **States:** 20 alarmed failed parents per task from new seeds (never used
  for training), branched at `a`. Harm states: false-alarm successes of the
  same cohort at their alarm, or at the matched time if there are too few.
- **Arms per state, K = 4 each, paired by state:** base XR-1 (task
  instruction), M1 (recovery prompt), M3 (recovery prompt).
- **Primary:** recovery success, model minus base. Parent bootstrap 95% CI,
  Holm across M1 and M3.

**Gate B:** a model passes if both hold:
1. on alarmed failure states, its gain over base is at least +0.10 with the
   CI above 0;
2. on harm states, its point loss against base is at most 0.10 and the CI
   lower bound is above −0.20.

If both models pass, the larger gain goes to Phase E. If neither passes,
report and stop, with the per-anchor and per-task tables.

Why +0.10: assuming the alarm catches about 90% of failures and failure rates
are around 0.5, a +0.10 gain on alarmed states projects to about +0.04 on
full-task success, minus harm on false alarms (0.061 × about 0.5 × the harm
rate). With trial-like discordance (about 12% of pairs), that is roughly the
smallest effect a 600-pair trial detects.

## Phase E: prospective trial (Gate C, preregistered after Gate B)

Fresh seeds; the five tasks; paired arms from the same environment seed and
sampling-seed schedule, as in the re-prompt trials:

| Arm | Model | Prompt |
|---|---|---|
| `baseline` | base XR-1 | task instruction |
| `alarm_recovery` | chosen model | task instruction; recovery prompt from the first Hide-and-Seek + P alarm to the end |
| `ft_plain` | chosen model | task instruction throughout (drift control) |
| `always_recovery` | chosen model | recovery prompt throughout |

- **Primary:** full-task success, `alarm_recovery − baseline`, macro over
  tasks. Exact McNemar on pooled pairs at α = 0.05, and paired bootstrap CI.
- **Secondary:**
  - `ft_plain − baseline` (does fine-tuning change normal behaviour);
  - `alarm_recovery − always_recovery` (value of waiting for the alarm);
  - rescue among baseline failures, harm among baseline successes;
  - per task, and by which alarm component fired.
- **Size:** fixed before the run with the trial planner from the Gate B
  estimates. The rough projection is 120–200 seeds per task.
- **Online alarm:** the client computes Hide-and-Seek + P live. Its alarm
  inference must match the offline scorer on every smoke rollout before the
  trial starts.

## Code to write (with simulator-free tests)

1. **Done (Phase A):** `robocasa/recovery/hs_p_alarm.py`. It exports the frozen
   Hide-and-Seek + P (PCA, LSTM seeds 0–2, normalization, P timers, fused path)
   with the reproduction checks, and scores ladder parents offline
   (`score-ladder`). Still to do: the online scorer for the XR-1 client
   (Phase E).
2. **Done (Phase A):** `robocasa/recovery/alarm_branching.py`. It branches
   ladder parents from saved snapshots at alarm-relative anchors (`alarm`,
   `alarm_minus_n`) through the ladder's `run_branch`, with a scene-identity
   check, resume and sharding, and computes Gate A (`analyze`). Phases B and
   D reuse it. Re-prompt branches are not supported yet: the saved
   snapshots do not record regressed predicates.
3. Ladder `record_branch_trajectories`: per-branch observation/action files,
   and a converter to the XR-1 training format from Step 0. The ladder now
   accepts `policy_feature_dtype` (`float16` by default, `float32` for new
   runs).
4. Harvest dataset builder with the health checks above.
5. XR-1 training configs for M1, M3 and M1-rp.
6. An `alarm_switch` gate in the stage-gated trial collector.

Tests (simulator- and PyTorch-free; the detector save/load test runs only
where PyTorch is installed):

```bash
python -m unittest tests.test_hs_p_alarm tests.test_alarm_branching tests.test_recoverability_ladder
```

### Phase A runbook (cluster)

`scripts/run_alarm_branching_xr1.sh` wraps every step. Defaults: Amendment 2
run `xr1_amend2_hide_and_seek_20261022_214209`, E1 lead calibration
`xr1_lead_calibration_20261020_001108/lead_analysis`, ladder v2 pilot
`xr1_ladder_v2_pilot_20261017_164537` (shards 0 and 1), K = 8, offsets 8 and
16, fixed budget 0.5 H (`ALARM_BUDGET_MODE`, `ALARM_BUDGET_FRACTION`), workers
`0:1 2:3`, ports from 10380.

```bash
cd /gs/fs/tga-shinoda/felid/robocasa && git pull origin main
export PYTHONPATH=$PWD
/gs/bs/tga-shinoda/felid/envs/vla_safe/bin/python -m unittest tests.test_hs_p_alarm tests.test_alarm_branching
```

A0, step 1: export the frozen alarm. This is CPU-only and runs in the
background; expect a few hours, comparable to the Amendment 2 scoring.

```bash
bash scripts/run_alarm_branching_xr1.sh export
```

The log must end with `EXPORT OK`. To rerun without retraining, point
`ALARM_REUSE_DETECTORS` at the `detectors/` folder of an earlier export
(it must be a new `ALARM_SPEC`). Before going on, record the `float16`
block it prints, then set `ALARM_SPEC` to the printed directory.

A0, step 2: score the ladder v2 parents (CPU, minutes).

```bash
export ALARM_SPEC=/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/hs_p_alarm_spec_<stamp>
bash scripts/run_alarm_branching_xr1.sh score
```

Check `alarms_summary.json`:
- `length_mismatches` and `feature_snapshot_mismatches` should be 0;
- the per-task counts give failures with an alarm, false alarms, and where the
  alarm falls relative to the ladder's own detection.

A1: plan, then branch (GPU; one allocation with four GPUs).

```bash
bash scripts/run_alarm_branching_xr1.sh plan
bash scripts/run_alarm_branching_xr1.sh start
bash scripts/run_alarm_branching_xr1.sh status
```

Resume after a stop with `resume`, never with `start`. Gate A:

```bash
bash scripts/run_alarm_branching_xr1.sh analyze
```

Outputs:
- the spec under `$STORAGE_BS/robocasa_checkpoints/safe/hs_p_alarm_spec_*`;
- alarms, branches and `gate_a/` under
  `$STORAGE_BS/robocasa_rollouts/recovery/xr1_alarm_phaseA_*`;
- logs under `$STORAGE_BS/robocasa_logs/eval`.

Every run follows
[`cluster_experiment_runbook.md`](cluster_experiment_runbook.md) for storage:
- harvested data under `$STORAGE_BS/robocasa_failure_datasets/`;
- checkpoints under `$STORAGE_BS/robocasa_checkpoints/`;
- rollouts under `$STORAGE_BS/robocasa_rollouts/recovery/`;
- logs under `$STORAGE_BS/robocasa_logs/eval`.

Runs go from a pinned worktree if other recovery runs share the checkout.

## Not in this plan

- Using the Hide-and-Seek score as flow guidance. It predicts failure, not
  which action helps, and gradients through a detector invite out-of-
  distribution actions.
- Best-of-K chunk reranking. It has an existing, never-run pilot
  ([xr1_safe_best_of_k_recovery.md](xr1_safe_best_of_k_recovery.md)) and
  little headroom at failure states.
- World-state failures that the policy never recovers from by itself
  (GarnishPancake's strawberry stage). These need expert data and are only
  reached if Gate A fails or as a follow-up.
- A learned stage estimator. It is not needed: neither the alarm nor the
  recovery mode reads simulator predicates.

## Open items

- Step 0 answers: XR-1 fine-tuning entry point, data format, run time and the
  sanity-check result.
- Exact recovery-prompt text and the replay mixing ratio, frozen with the
  training code.
- Whether ladder v2 snapshots on `/gs/bs` are complete for all 76 parents
  (A1 needs them).
