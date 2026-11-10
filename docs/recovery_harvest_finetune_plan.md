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

**Gate A result (6 November 2026): fail, redesign around expert data.**
Run `xr1_alarm_phaseA_20261105_170817`, commit `8c9a5304`, fixed budget
0.5 H, K = 8. 41 parents branched (40 alarmed failures, 1 false-alarm
success), 952 branches, 0 errors, 41/41 scene checks matched, restore
residual 0.

| Anchor | Share with ≥ 1 success [95% CI] | Mean success | Share within remaining time |
|---|---|---|---|
| `a` | 0.200 [0.075, 0.325] | 0.106 | 0.075 |
| `a − 8` | 0.200 [0.075, 0.325] | 0.100 | 0.125 |
| `a − 16` | 0.200 [0.075, 0.325] | 0.091 | 0.125 |

What the result shows:

- **Branching earlier does not help.** The share is 0.20 at every anchor, as
  in ladder v2 at its own anchor with K = 4. Doubling K did not raise it.
- **Recoverability belongs to the episode, not the moment.** 11 of 40 failed
  parents have a success at some anchor. Five of them recover often (4–8 of
  8 at an anchor); the other six have 1–2 scattered successes. 29 parents
  never recover.
- **By task, any anchor:** ArrangeBreadBasket 2/8, ArrangeTea 0/8,
  BreadSelection 4/8, CuttingToolSelection 3/8, GarnishPancake 2/8. ArrangeTea
  branches all run to the budget; 85–100% make no further subtask progress.
  Per-task shares were not preregistered and are not used for decisions.
- **Consequence.** Own-branch harvesting would draw most recovery data from
  five episodes and nothing from the 29 states the policy never leaves.
  Fine-tuning on it would mostly teach recoveries the policy already makes by
  resampling.

## Redesign after Gate A: expert recoveries at the alarm states

Written 6 November 2026, after Gate A and before any of the steps below. The
question stays the same; only the source of recovery data changes. The
alarm, the tasks, the fixed branch budget and Phases C–E are unchanged,
except that Phase C trains on the data generated here.

**R0: failure audit (simulator restore of the saved snapshots, no policy).**
For each of the 40 alarmed failure states, read from the simulator state at
`a`: the current stage, where each task object is (on a support surface, on
the floor, outside the scene), whether it is tipped or grasped, and whether
any completed subtask has been undone. Camera images at the alarm and at the
end go with each state for a manual check. Classify each state as

1. recovered by resampling (the 11 above);
2. physically recoverable but never recovered (target reachable and
   resting, nothing irreversible);
3. not recoverable within the episode (object fallen, out of reach, or an
   irreversible event such as a dropped garnish).

Class 2 is the headroom expert data can address. **Gate R0:** if class 2
holds fewer than 10 of the 29 never-recovered states, expert data cannot
change the picture either; stop and report detection plus resampling (or
reset) as the recovery result.

**R1: source of expert actions (cluster check, no simulation).** Check in
this order and use the first that works for all five tasks:

1. *Segment replay (IntervenGen/MimicGen-style).* The RoboCasa v1.0
   composite human demos for the five tasks carry subtask boundaries
   (`annotation.human.subtask_idx`). This needs replayable simulator states
   or object poses per frame to make the segments object-relative. The
   RoboCasa MimicGen fork covers 64 of 65 atomic tasks but no composite
   task, so the subtask-to-object mapping is written here.
2. *Privileged scripted expert.* Per stuck subtask (pick, place, open, close),
   a motion script on privileged object poses through the existing IK stack.
   Only the subtasks that class-2 states are stuck on need scripts.

*R1 design (11 November 2026, before any R1 output).* Upstream RoboCasa
released its subtask annotation pipeline on 14 October 2026: a registered
labeler per composite task (all 300, including the five here) that replays a
demo's simulator states and writes per-frame labels (atomic-task name,
pick/place/navigate/execute stage, instruction). It was merged into this
fork; its only environment changes are an `ArrangeDrinkware` layout
exclusion and a duplicate `Sink` method removed (the two were identical), so
the five pilot tasks are untouched. Its labels need the whole demo and
assume it succeeds, so they are used only on expert demos. Our predicates
stay the stage definition everywhere else (ladder, alarm, R0, SAFE).

The LeRobot folders of the target human demos ship
`extras/episode_*/{states.npz, ep_meta.json, model.xml.gz}`, so the first R1
question is answered: the demos are replayable. `expert_segments index`
replays the first 100 demos per task with upstream's `reset_to`, and records
per frame upstream's labels, our predicates and every task object's pose.

Stage-to-segment join: the frame where a stage's predicate first becomes
true and stays true for 5 frames selects the upstream segment that completes
it. That is the containing segment, or the previous one when the frame is in
`done` or within 10 frames and the first quarter of a segment. The expert
segment is that segment plus the earlier segments of the same subtask (back
to its pick, navigation included). The moved object is the task object that
moves most over it (none below 3 cm: a fixture subtask). The join does not
use the ladder's order, because demonstrators reorder stages (ArrangeTea's
mug before the kettle). The ladder-ordered window is reported next to it.

**R1 decision rule (frozen with this code):** segment replay is the R1
source if every (task, stage) that holds a class-2 state has at least 10
usable segments that agree on the upstream subtask and the moved object, with
agreement at least 0.8 on each. Usable means a demo that ends in task success
and passes the labeler's own checks. Otherwise go to option 2 for the failing
stages. `expert_segments coverage` applies this to the R0 classes and writes
`stage_segments.jsonl`, the source table for R2.

The index also keeps upstream's instruction for every segment of the group
and each task object's display name. `coverage` reports, per stage, the
modal instruction sequence with object names as `{object_key}` (for example
`pick up the {obj} from the cabinet` → `place the {obj} on the tray`). These
are candidates for the stage instructions of later experiments (re-prompt,
the recovery-mode prompt). They are not used in R1's decision, and the
finished re-prompt trials keep the instructions they ran with.

**R2: generation at the alarm states.** From each class-2 state at `a`:
an optional release-and-retreat prefix (open the gripper, lift, return to a
pre-grasp height) when the target is mis-grasped; the expert segment for the
current subtask only; then XR-1 under the task instruction for the remaining
subtasks. Success is full-task success within the fixed 0.5 H budget.
N = 10 attempts per state (different source segments or script seeds).

**Gate R** (thresholds frozen with the R2 code, before it runs): the share of
class-2 states with at least one generated success. Below 0.5, the generator
is the problem; fix it before generating training data.

**What changes downstream.**

- Phase B keeps its parent collection (fresh seeds until 30 alarmed failed
  parents per task). Training data at those states comes from R2. Own-branch
  successes stay as a tagged secondary source.
- Phase C trains M1 on generated recoveries plus own-branch successes, and M3
  adds the failed own resamples from the same states as "bad". Recorded
  observations are rendered as the policy receives them; the actions are the
  executed ones.
- Gate B compares against resampling at the alarm, the deployable baseline
  measured here: share within remaining time 0.075 at `a`.

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
7. **Done (R0):** `robocasa/recovery/failure_audit.py`. `audit` rebuilds each
   alarmed failure's scene (simulator only), restores its first, alarm and
   terminal snapshots, and records per task object the position,
   uprightness, grasp, counter contact and floor or out-of-scene status; at
   the alarm it also records live predicates, undone subtasks and the
   policy's camera images. Provisional class: 1 if an A1 branch succeeded, 3
   if a task object is on the floor (z < 0.3 m) or outside the scene, else
   2. Tipped objects, grasps and undone subtasks are flags only. `decide`
   applies the hand-filled `manual_class` column of `review.csv` (class 1
   cannot be changed) and computes Gate R0.
8. **Done (R1):** `robocasa/recovery/expert_segments.py`. `index` replays the
   target human demos (CPU, sharded, resumable) with upstream's labelers and
   our predicates; `coverage` joins them with the R0 classes, writes
   `r1_coverage.{json,md}` and `stage_segments.jsonl`, and applies the R1
   decision rule. Demo health in the report: clean demos, labeler flags,
   frame/row mismatches and agreement with the labels upstream published in
   the parquet `subtask_idx` column.

Tests (simulator- and PyTorch-free; the detector save/load test runs only
where PyTorch is installed):

```bash
python -m unittest tests.test_hs_p_alarm tests.test_alarm_branching tests.test_failure_audit tests.test_recoverability_ladder tests.test_expert_segments
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

R0 failure audit (one GPU for rendering, no policy server; about one
minute per state):

```bash
bash scripts/run_alarm_branching_xr1.sh audit
```

Then open `$ALARM_ROOT/r0_audit/review.md` and its images, fill
`manual_class` (1, 2 or 3) in `review.csv` only where the images contradict
the provisional class, and run:

```bash
ALARM_REVIEW=$ALARM_ROOT/r0_audit/review.csv bash scripts/run_alarm_branching_xr1.sh decide
```

R1 expert segments (CPU only, no GPU or policy server; 5 shards by default,
`R1_SHARDS`, `R1_MAX_EPISODES`). The index can run before R0 is decided; the
decision needs `decide` first.

```bash
bash scripts/run_alarm_branching_xr1.sh segments
bash scripts/run_alarm_branching_xr1.sh segments-status
bash scripts/run_alarm_branching_xr1.sh r1
```

`r1` writes `$ALARM_ROOT/r1_segments/coverage/r1_coverage.md`. Check the
demo-health table first: errors 0, row mismatches 0, and agreement with the
published labels near 1 (a low value means the replay differs from
upstream's).

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
  (GarnishPancake's strawberry stage). Gate A failed, so these are now in
  scope through the redesign (R0 class 2); class 3 stays out.
- A learned stage estimator. It is not needed: neither the alarm nor the
  recovery mode reads simulator predicates.

## Open items

- Step 0 answers: XR-1 fine-tuning entry point, data format, run time and the
  sanity-check result.
- Exact recovery-prompt text and the replay mixing ratio, frozen with the
  training code.
- Whether ladder v2 snapshots on `/gs/bs` are complete for all 76 parents
  (A1 needs them). Answered: all 41 branched parents had complete snapshots.
- R1 answers: whether the composite human demos for the five tasks include
  replayable states or object poses (answered: yes, per-frame simulator
  states in the LeRobot `extras/`), and which subtasks the class-2 states
  are stuck on (from `r1`).
