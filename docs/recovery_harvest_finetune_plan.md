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

**Step 0 findings (26 November 2026, before any training code).** The cluster
checkout is pinned at `4da1db0`, the 2026-08-03 release commit that added the
post-training code. File contents were read from that commit on GitHub and
from the released RoboCasa365 checkpoint on Hugging Face.

- *Entry point.* `xr1/scripts/train.sh` runs `torchrun xr1/tools/train.py`
  with Hydra configs in `xr1/configs/` (`data/load_washer.yaml`,
  `model/posttrain.yaml`, `trainer/deepspeed.yaml`): PyTorch Lightning with
  DeepSpeed, bf16-mixed, FusedAdam, 500 warmup steps to a peak learning rate
  of 2e-5, 10,000 steps, batch 48. No option freezes part of the model. It
  logs to W&B, which is disabled here.
- *Starting checkpoint.* The trainer loads only a `.pt` state dict
  (`{"module": ...}`). `xr1/tools/weight_convert.py` converts the released
  RoboCasa365 folder into one. There is no converter back, and every
  collector here serves the Hugging Face folder through `deploy/server.py`,
  so a reverse converter is needed: strip the `model.` prefix, load the
  weights into the RoboCasa365 model, and save a derived folder with the
  SAFE-patched `modeling_mibot.py`.
- *Environment.* The server environment has no DeepSpeed, Lightning, Hydra or
  W&B. Training gets its own environment, so the serving one stays unchanged.
- *Data: the released trainer does not reproduce the RoboCasa365 inputs.* Its
  only dataset type (`json_dataset.py`, written for Xiaomi's two-arm robot)
  gives one still frame per camera, a one-step state scaled to [-1, 1] by
  q01/q99, and a 30-step action chunk z-scored with dataset statistics, in a
  left/right-arm layout. The RoboCasa365 checkpoint, as served and evaluated
  here, receives:
  - each of the three cameras as a four-frame video (frames t−6, t−4, t−2, t;
    0.95 centre crop; no resize and no colour jitter);
  - a four-step 14-d state (end-effector position and axis-angle, gripper,
    base position and axis-angle), zero-padded to 60 and not normalized
    (`state_length: 4` in its config);
  - a 16-step chunk whose dims 0–11 are RoboCasa's own 12-d action, with
    identity statistics (mean 0, std 1 on dims 0–11). The processor's
    `action_config` has one robot type, `robocasa365`.

  Fine-tuning through the released dataset type would train the checkpoint on
  inputs it never saw. A RoboCasa365 dataset class that builds each sample
  exactly as the eval client and the processor do is required. Whether the
  training model (`xr1/mibot/models/VLA/XR1.py`) and its collate accept video
  inputs and a four-step state is not yet known. Read by eye and still to
  confirm on the cluster: the shape of `action_config`, how
  `chat_template.jinja` inserts the state, and the second half of
  `modeling_mibot.py`.

Step 0 from here, in order (steps 2–5 superseded on 26 November by decision A
below):

1. Read `XR1.py` and `custom_collate.py` at `4da1db0`: video inputs,
   `state_length`, and the training loss.
2. Training environment, and weight conversion in both directions.
3. The RoboCasa365 dataset class, fed with observations as the policy
   receives them (Phase B records exactly these).
4. *Input-match check, before any update.* With the released weights and
   fixed flow noise, the training code path must predict the same action
   chunk as the served model on the same inputs, within bf16 rounding
   (tolerance fixed with the code), on at least 20 samples per pilot task.
5. The negligible-update sanity check above.

If step 1 shows that the training model cannot take these inputs without
being rewritten, or step 4 has not passed by 15 January 2027, Step 0 fails
and the fallback above applies (the roadmap also lists RLDX-1 before π0).

**Step 1 result (26 November 2026).** `XR1.py` and `custom_collate.py` at
`4da1db0` were read in full, and so was the checkpoint's `modeling_mibot.py`.
The cluster confirmed the versions (torch 2.8.0 and transformers 4.57.1 in the
server environment, as `xr1/assets/requirements.txt` pins), `action_config`
of shape 16 × 60, and that the checkpoint's chat template never mentions the
state. The released training model is not the model the RoboCasa365
checkpoint is served with:

- *Extra heads and losses.* Training adds choice and score heads
  (`state_projector_choice`, `action_projector_choice`,
  `score_projector_choice`) and two VLM embeddings (`action_embed`,
  `score_embed`), with losses on them next to the flow-matching loss. The
  checkpoint's weight index appears to have none of them (to confirm on the
  cluster), so `weight_convert.py`'s output would fail the trainer's strict
  load.
- *State.* Training also gives the state to the VLM as `<state>` tokens. The
  served model gives it only to the DiT.
- *DiT positions.* Training shifts the action tokens' positions by +10; the
  served model does not.
- *Video.* The VLM accepts video inputs, but the collate forwards only image
  tensors and computes positions for images only, so video items would be
  dropped without an error.

Shared by both: the modules the served model uses (`vlm`, `dit`,
`state_projector`, `action_projector`, `action_output_layer`, `t_embedder`,
`t_projector`, `sink`) and the flow-matching formulation (t ~ Beta(1.5, 1),
x_t = (1 − t)·noise + t·action, target action − noise).

Under the rule above this is "cannot take these inputs without being
rewritten". Two options were left: (A) a training forward on the served model
itself, or (B) the fallback now. (B) would redo the alarm, Gate A, R0 and R1
on a new policy before any recovery data exists.

**Decision (26 November 2026, author): (A).** Step 0 continues with our own
training forward on the served model, replacing steps 2–5 above:

2. *Reference sources.* Copy the served checkpoint's Python and config files
   and Xiaomi's eval client (`eval_robocasa365/entry.py`, `deploy/client.py`)
   next to the code, so the trainer is written against the exact files the
   server runs.
3. *Training forward* (a new module under `robocasa/recovery/`). Load the
   RoboCasa365 folder with `AutoModel`, as the server does. Run the VLM as the
   served forward does, without gradients. Sample t ~ Beta(1.5, 1) and noise,
   and call the model's own `dit_forward` on x_t. The loss is the
   flow-matching part of the released loss (0.5 · MSE + 1.0 · FFT magnitude)
   between its output and action − noise, over the active action dims (0–11).
   The VLM is frozen and the DiT and its projectors are trained; the
   trainable set and the optimizer are fixed with the code. Checkpoints
   are saved as a derived Hugging Face folder (the trained weights plus the
   SAFE-patched `modeling_mibot.py`), so every collector serves them
   unchanged and no `.pt` conversion is needed.
4. *Sample builder.* One builder for every training source. It turns recorded
   camera frames, state and actions into the exact processor call that
   `entry.py` makes: three four-frame videos (t−6, t−4, t−2, t, 0.95 crop), the
   same text, a four-step 14-d state padded to 60, and a 16 × 60 action target
   with dims 0–11 active. For Step 0 the source is the RoboCasa365 human
   demos of the five pilot tasks, which R1 already replays.
5. *Input-match check (gate).* On recorded frames, the sample builder and
   Xiaomi's client code must give identical processor outputs (token ids,
   video tensors and grids, state, action mask) on at least 20 samples per
   pilot task.
6. *Loss check (descriptive).* The released weights' loss on held-out demos,
   next to the same loss with actions taken from another episode, to show
   that the loss reads the task.
7. *Negligible-update sanity check*, as above, through the saved derived
   folder and the normal server.
8. *Smoke training:* steps per hour, GPU memory and the projected time for
   about 1,000 episodes.

The fail rule is unchanged: if step 5 or step 7 has not passed by 15 January
2027, Step 0 fails and the fallback applies.

*Code (26 November 2026, before any run).* `robocasa/recovery/xr1_samples.py`
(sample builder, LeRobot reading, the gate's comparison) and
`robocasa/recovery/xr1_finetune.py` (subcommands `input-match`, `loss-check`,
`train`, `diff-weights`), with simulator-free tests (`tests/test_xr1_samples.py`;
`tests/test_xr1_finetune.py`, whose loss and optimizer tests need PyTorch).
Choices fixed with this code:

- The reference for the gate is our deployed client,
  `XiaomiRobotics1Policy._make_request`, which every collector here uses and
  which reproduces Xiaomi's `entry.py` line for line. The gate feeds it the
  episode's observations frame by frame and compares its processor outputs
  with the builder's, tensor for tensor (`input_ids`, `attention_mask`,
  `pixel_values_videos`, `video_grid_thw`, `state`, `action_mask`): 5 held-out
  episodes × 4 frames per task, all identical.
- Demo actions are reordered from the LeRobot layout (base motion first,
  `meta/modality.json`) to the environment's 12-d layout that the model
  outputs and `convert_action` executes. States go through the client's own
  `observation_to_state`.
- A sample needs a full 16-step future, so the last 15 frames of an episode
  are never a sample's first frame and nothing is padded.
- Held-out demos: 10% of episodes, chosen by a hash of task and episode, never
  by frame.
- Training: the action expert (`dit`, `state_projector`, `action_projector`,
  `action_output_layer`, `t_embedder`, `t_projector`, `sink`), parameters in
  bfloat16 as served with float32 master weights; AdamW with the released
  schedule (warmup over 500 steps from 5e-7 to 2e-5, cosine to 5e-6), betas
  (0.9, 0.95), weight decay 0.1 on matrices, gradient clipping at 1.0.
  Gradients are averaged across GPUs; no DeepSpeed, Lightning or Hydra.
- Saved checkpoints copy every non-weight file from a template folder (the
  SAFE-patched checkpoint), and saving fails if the weight key set differs from
  the template's.

Not covered by the gate: whether the demo videos and states equal what the
environment gives the policy (image orientation, colour, the state fields).
The loss check shows it indirectly; a replay comparison is added if the loss
check looks wrong.

*Input-match gate (26 November 2026, commit `d2159376`): passed.* On the
cluster the tests passed with PyTorch (18), and `input-match` (output under
`robocasa_rollouts/recovery/xr1_step0/input_match_*`) found 20 of 20 samples
identical on each of the five tasks. Both the builder and the client give the
processor pre-sampled frames without video metadata, so the processor writes
its frame timestamps at its default 24 fps (it warns) in serving and in
training alike; training sees the same timestamp text as the server. The
demo folders are found without the dataset registry, whose import needs
robosuite: by the one dated `v1.0/target/composite/<Task>/*/lerobot` folder,
recorded in `input_match.json`.

*Loss check (26 November 2026, descriptive).* Released weights, 32 held-out
samples per task (8 episodes × 4 frames), the same noise and t for both
columns. Mean loss with the sample's own actions against actions from another
held-out episode of the same task (`loss_check_20261126_163933`):

| Task | Own | Other episode | Ratio | Own lower |
|---|---:|---:|---:|---:|
| ArrangeBreadBasket | 0.460 | 1.430 | 3.1 | 31/32 |
| ArrangeTea | 0.424 | 0.935 | 2.2 | 31/32 |
| BreadSelection | 0.392 | 0.830 | 2.1 | 25/32 |
| CuttingToolSelection | 0.302 | 1.182 | 3.9 | 32/32 |
| GarnishPancake | 0.508 | 1.444 | 2.8 | 28/32 |

The demos as built here read as the model's own inputs: with a wrong image
orientation, state field or action order, the own loss would sit near the
other-episode loss. This covers the gap the gate leaves, indirectly.

*Negligible-update run (26 November 2026).* 20 steps at learning rate 1e-12,
batch 4, one GPU: loss 0.25–0.75 (MSE 0.03–0.19), gradient norm 12–24 before
clipping, peak GPU memory 22.2 GB. Saved to
`robocasa_checkpoints/xr1_finetune/negligible_20261126_163933/checkpoints/step_000020`.
After the first step (10 s, including data-worker start), 76 samples took
4.5 s (about 17 samples per second), all from the workers' first buffers, so
whether video decoding keeps up over a long run is not measured yet.
`diff-weights` against the template: the same 1,120 weight names, the VLM's
713 tensors unchanged, and 50 tensors of the action expert changed by at most
2.9e-11 (bfloat16 resolves such steps only for weights near zero). Only the
trainable modules moved, and saving writes what was trained.

*Smoke training, step 8 (26 November 2026).* 300 steps, batch 8, one GPU, the
released schedule (learning rate 1.2e-5 at step 300):
`smoke_20261126_164734`. From step 25 to 300 every 25 steps took 7.9–8.0 s:
about 25 samples (3.1 steps) per second, with data loading keeping up over
2,400 samples (the workers' first buffers hold 192). Peak memory 22.2 GB, as
at batch 4. Loss 0.23–0.72 without drift, gradient norm 6–11 before
clipping. The released recipe's 480,000 samples (10,000 steps × 48) would take
about 5.3 h on one GPU.

Open for Step 0: the paired rollouts of step 7 (20 natural rollouts per task,
base and saved folder on the same seeds). They need two GPUs: the ladder
script keeps the simulator and the server apart.

*Step 7 design (26 November 2026, before any paired rollout).* The ladder
script's `naturals` mode with the pilot config (five tasks, pretrain split),
20 natural rollouts per task, base seed 560007 (environment seeds
560007–560026, sampling seeds derived from them as in every ladder run), once
with the SAFE-patched base checkpoint and once with the negligible-update
folder `negligible_20261126_163933/checkpoints/step_000020`, both served by the
same patched server. `xr1_finetune compare-rollouts` pairs the two runs by task
and environment seed. **Rule (frozen with that code):** the saved folder passes
if its pooled success rate over all pairs is within 0.10 of the base's and no
task differs by more than 0.25 (5 of 20). The weights differ by at most
2.9e-11, so a failure points at the serving path, not at training.

*Step 7 launched (26 November 2026, 23:11, node r9n7, at commit bf05db7).*
Both runs share GPU pair 0:1 (simulator on 0, both servers on 1), one shard
each: base on port 10460 into `robocasa_rollouts/recovery/xr1_step0_naturals_base`,
negligible folder on port 10470 into
`robocasa_rollouts/recovery/xr1_step0_naturals_negligible`. Logs in
`robocasa_logs/eval`: `xr1_step0_naturals_base_shard0_20261126_231110.log` and
`xr1_step0_naturals_negligible_shard0_20261126_231116.log`.

*Step 7 result (27 November 2026): passed.* `compare-rollouts` (written at
`robocasa_rollouts/recovery/xr1_step0_naturals_negligible/paired_comparison.json`)
paired all 100 rollouts, with none unpaired on either side. The two folders
succeeded on exactly the same pairs: no pair succeeded with only one of them.

| Task | Pairs | Base | Negligible folder | Difference |
|---|---:|---:|---:|---:|
| ArrangeBreadBasket | 20 | 11 | 11 | 0 |
| ArrangeTea | 20 | 12 | 12 | 0 |
| BreadSelection | 20 | 9 | 9 | 0 |
| CuttingToolSelection | 20 | 9 | 9 | 0 |
| GarnishPancake | 20 | 12 | 12 | 0 |
| Pooled | 100 | 53 | 53 | 0 |

Identical outcomes on every pair are what weights 2.9e-11 apart should give,
so the derived folder is served exactly like the base checkpoint. The base's
53/100 on these seeds is a separate, descriptive number for the pretrain split.

**Step 0 passed (27 November 2026).** Both gates (step 5, input match; step 7,
paired rollouts) passed before the 15 January 2027 deadline. XR-1 stays the
policy and the fallback is not used. Next: R2's checks on the cluster
(`probe-controls`, then self-replay and transfer), then Gate R.

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

**R0 decision (11 November 2026).** Gate R0 passed: 26 of the 29
never-recovered states are class 2 (threshold 10), 3 are class 3. All 40
scene checks matched and there were 0 errors. `decide` ran without manual
overrides; the review images were checked on 26 November (below).

**First R1 coverage run (11 November 2026).** The index replayed 100 demos
per task with 0 errors. All 500 demos succeed and none is flagged by
upstream's checks, and the replay agrees exactly with the labels upstream
published (1.0 on every task). Under the rule above, 4 stuck stages failed:

- `CuttingToolSelection:correct_tool_on_cutting_board` (5 states): the
  upstream subtask agreed in 100/100, but the moved object was the peeler in
  60 and the knife in 40. The correct tool depends on the food in the scene,
  so no single object can reach 0.8.
- `ArrangeBreadBasket:basket_on_dining_counter` (3 states): the subtask
  agreed in 99/99, but the bread rides in the basket and moves as far, so
  the most-moved object was the basket in only 0.535. The instruction always
  names the basket.
- `GarnishPancake:all_latched` (2) and `BreadSelection:all_latched` (1):
  every ladder stage had latched at the alarm without task success, so the
  state has no current stage.

**Amendment 1 (11 November 2026, after that run and before any R2 code).**
The first two failures come from the object rule, not from missing demos, so:

1. A segment's object is the first task object its instructions name
   (`pick up the {basket}`), and the most-moved object only when none is named
   (fixture subtasks).
2. A stage passes if at least 0.8 of its usable segments agree on the
   upstream subtask, objects that are targets in at least 0.2 of them cover at
   least 0.8 of them, and each such object has at least 10 segments of its
   own. R2 draws segments for the object the stuck state needs.
3. A class-2 `all_latched` state takes the first required stage undone at
   its alarm (R0 `regressed_at_alarm`). A state with nothing undone has no stage
   to replay. It is listed separately (decision
   `segment_replay_except_unstaged_states`), and its failure is checked on the
   R0 images before deciding whether R2 can address it.

The original rule's decision is still computed and reported next to the
amended one. In the first run, the unstaged state is GarnishPancake
`cf428856ba92f99d9d7d`: every stage latched, nothing undone, no flags.

The index also keeps upstream's instruction for every segment of the group
and each task object's display name. `coverage` reports, per stage, the
modal instruction sequence with object names as `{object_key}` (for example
`pick up the {obj} from the cabinet` → `place the {obj} on the tray`). These
are candidates for the stage instructions of later experiments (re-prompt,
the recovery-mode prompt). They are not used in R1's decision, and the
finished re-prompt trials keep the instructions they ran with.

**R0 image review (26 November 2026, after the first R1 run and before any R2
code).** All 29 never-recovered states were checked against their alarm and
final images and the simulator poses. No class changed: 26 class 2 and 3
class 3, so Gate R0 stands. `review.csv` keeps `manual_class` empty and has a
note per state. Findings:

- The policy's cameras show the task objects in about half of the states; at
  the alarm the robot often faces a cabinet or the fridge. The other states
  rest on the simulator poses.
- The three class-3 states have their objects on the floor (the kettle is
  visible in `b9268e5b`; basket and bread are at z ≤ 0.06 m in the other two).
- Two GarnishPancake states are uncertain and stay class 2: `863712cc`
  (strawberry in no camera, at 0.83 m, 16 cm below the pancake and 0.29 m
  from it) and `1e18b02a` (strawberry at 0.40 m, probably still in the
  fridge; on the floor by the end of the episode).
- `cf428856`, the unstaged state of Amendment 1, has a stage: the strawberry
  lies on the plate 8 cm from the pancake's centre and `strawberry_on_pancake`
  is false at the alarm. R0's undone list reads predicates only at saved
  snapshots, and this stage latched and came undone between two of them.
- The `tipped` flag carries no information for bread, croissants, knives and
  strawberries, which have no upright pose. It matters for the mug, kettle,
  jam and basket.

**Amendment 2 (26 November 2026, after the image review and before any R1
rerun or R2 code).** An `all_latched` class-2 state takes the stage undone at
its alarm (Amendment 1). If none is recorded, it takes the first required,
non-release stage that is false in R0's live predicates at the alarm
(`live_predicates_at_alarm`). On the current audit this changes only
`cf428856`, which gets `strawberry_on_pancake` (now 6 class-2 states); the
other `all_latched` states get the same stage either way. `coverage` reports
the class-2 states by stage source (`alarm`, `undone`, `live`, `none`).

**Amended R1 run (26 November 2026, commit `2c665401`).** `decide` was rerun
with the annotated review sheet (`r0_audit/review_20261126.csv`): 0 manual
overrides, the same 26 class 2 and 3 class 3, Gate R0 passed. `r1` then gave
**`segment_replay`**: all 26 class-2 states have a stage (23 from the alarm,
2 undone, 1 live), and no stuck stage fails the amended rule. So Amendment 1
fixed both object-rule failures (`correct_tool_on_cutting_board`,
`basket_on_dining_counter`), and Amendment 2 left no unstaged state. The
first report is kept in `r1_segments/coverage_before_amend2/`; the amended
one, with the per-stage table, is in `r1_segments/coverage/`. R1's source for
R2 is segment replay for every stuck stage. The R2 gaps listed below still
apply, since R1 only checks that enough consistent demo segments exist.

**R2: generation at the alarm states.** From each class-2 state at `a`:
an optional release-and-retreat prefix (open the gripper, lift, return to a
pre-grasp height) when the target is mis-grasped; the expert segment for the
current subtask only; then XR-1 under the task instruction for the remaining
subtasks. Success is full-task success within the fixed 0.5 H budget.
N = 10 attempts per state (different source segments or script seeds).

**Gate R** (thresholds frozen with the R2 code, before it runs): the share of
class-2 states with at least one generated success. Below 0.5, the generator
is the problem; fix it before generating training data.

*R2 design (draft, 11 November 2026; binding once committed with the R2
code).* The robot is a mobile manipulator, so plain MimicGen replay (fixed base,
object-relative arm poses) does not cover the stages that need driving
(`strawberry_on_pancake`, `basket_on_dining_counter`: 8 of the 25 staged
states). The design uses four facts from the code and the data:

- The alarm states and the demos are in different kitchens. The 5 pilot tasks
  have target-split human demos only (kitchens 1–10, target object
  instances), while the ladder runs XR-1 in the pretrain split (kitchens
  11–60, pretrain object instances). The R2a check confirmed it: 0 of the
  ~500 demos per task share a kitchen with any class-2 state. No demo
  world-frame base path can be reused, and the arm motions must transfer
  across kitchens and object instances.
- RoboCasa computes the base pose in front of a fixture with
  `EnvUtils.compute_robot_base_placement_pose(env, fixture, ref_object)`.
  NavigateKitchen uses it as its goal and counts success as base within
  0.20 m and `cos(yaw error) >= 0.98`. This gives a kitchen-independent
  goal for driving.
- `stage_segments.jsonl` splits each expert segment into upstream steps
  (pick, navigate, place, execute) with frame bounds, the target object and
  the instructions.
- The environment's 12-d action is an arm pose delta in the base frame,
  gripper, base velocity, torso and a base/arm mode switch. An executor that
  tracks pose targets produces actions in exactly the space XR-1 acts in, so
  the generated data can train XR-1 directly.

R2 has three parts:

1. **R2a source matching (no simulator).** For each class-2 state, read the
   R0 facts at the alarm (what the gripper holds, whether the target is
   tipped). Choose:
   - the first step to run: `pick` if the target is not held, the step after
     the pick if it is, with release-and-retreat first if the gripper holds
     anything else;
   - up to 10 source segments for the stage and the state's target object
     category, ranked by how close the target's pose relative to the robot
     base at the step's first frame is to the alarm state's (kitchen is not
     a criterion, since none match);
   - the stage's reference for each step: the target object for `pick`, and
     for `place` the receptacle named in the place instruction (or the
     stage predicate's fixture, e.g. the dining counter).
2. **R2b executor (simulator).** Restore the alarm snapshot. Then:
   - optional release-and-retreat: open the gripper, lift 10 cm, back off;
   - `navigate` (scripted, not replayed): in base mode, turn toward, drive to
     and align with `compute_robot_base_placement_pose(env, fixture,
     ref_object)` for the fixture holding the place reference, with a
     proportional controller on the base-velocity action; done when the
     NavigateKitchen test above holds;
   - `pick` / `place`: take the source end-effector poses relative to the
     step's reference at the step's first frame, transform them into the
     alarm scene, approach the first pose by interpolation, then track the
     poses with the arm's pose-delta action and the source's gripper
     commands;
   - stop the expert once the stage predicate holds for 5 steps, or after the
     source segment's length plus 50% (navigation: 300 steps).

   Then XR-1 takes over: task instruction, context reset, until success or
   the fixed budget (0.5 H from the alarm, expert time included). The
   observations are recorded as XR-1 receives them, with the executed actions,
   the source segment and the step boundaries.
3. **Executor checks before Gate R**:
   - *self-replay* (tests the controller): from a target demo's own state at
     the segment start, replaying its own segment must complete the stage in
     at least 0.9 of 20 demos per stage;
   - *pretrain transfer* (tests what the alarm states need: other kitchens,
     other object instances, scripted driving): 20 fresh pretrain-split
     resets per task, seeds disjoint from Phase A and Phase B; the executor
     runs every stage in order with sources from the target demos; per
     stage, the completion rate among runs that reached it must be at least
     0.7.

   Below these, the executor is the problem, and the alarm states are not run.

Gate R then uses N = 10 attempts per class-2 state (distinct source segments
where available) and reports two rates. The **expert rate** is the share of
attempts where the expert completes the stuck stage, which separates executor
failures from continuation failures. The **full rate** is the share of
attempts ending in full-task success; Gate R uses this one, at the 0.5
threshold above.

Open before the code:
- the arm controller's delta scaling and frame (OSC pose, base frame), read
  from the composite controller config at run time and tested in self-replay;
- whether pick grasps transfer from target to pretrain object instances of
  the same category; the pretrain-transfer check measures it, and a failing
  category falls back to a scripted top-down grasp on the object's bounding
  box;
- the GarnishPancake state `cf428856ba92f99d9d7d`: resolved by Amendment 2
  (stage `strawberry_on_pancake`);
- *removing the wrong tool*: in 4 of the 5 CuttingToolSelection class-2
  states the wrong tool lies on the cutting board
  (`wrong_tool_not_on_cutting_board` undone), for example the knife next to
  the carrot while the peeler is still in the drawer. Success needs it off the
  board, and no human demo has that step, since demonstrators never put the
  wrong tool there. R2 needs a pick from the board and a place on the counter
  for it, from pick segments of the same tool category or a script;
- *non-task objects in the gripper*: R0's `holding` lists task objects only.
  In `9aaa49f0` the robot holds a piece of ham from the fridge at the alarm
  (and in `4e582d6b` it grasps a coconut later). The release-and-retreat test
  must read the gripper itself (fingers closed on any body), not `holding`;
- *mugs on the tray's edge*: in 4 of the 7 ArrangeTea class-2 states the mug
  rests on the tray's rim (`mug_on_tray` false), and in one it stands on the
  counter beside the tray. The stage's demo segments pick the mug from a
  cabinet, so the pick must transfer to a mug on the counter or the tray. The
  pretrain-transfer check does not test this start, because its runs reach
  the stage with the mug still in the cabinet.

**R2 code (27 November 2026, before any R2 output; this makes the design
binding).** `robocasa/recovery/expert_recovery.py` implements R2a–R2b, the two
executor checks and Gate R with the thresholds above. Where the draft left a
choice open, the code fixes it as follows:

- *Facts at the alarm.* R2a reads them from the restored alarm snapshot, not
  from the R0 records, because R0 stores no orientations, no base pose and
  only task-object grasps. "Holds something" means both finger pads touch the
  same non-robot body, so the ham in `9aaa49f0` triggers the release.
- *Stage table.* For each (task, stage) the code lists the moved object and
  the reference, taken from the predicates. Two cases are special:
  - The correct and wrong tool come from the scene's food (`_CUTTING_MAP`).
  - Fixture stages (`cabinet_open`, `cabinet_closed`, `drawer_open`,
    `fridge_open`) navigate to the fixture first, then replay relative to the
    robot base.
  A stage stuck at `gripper_released` gets only the release.
- *Reference frames.* Each step is replayed in a frame that keeps a
  reference's position and heading and ignores its tilt:
  - pick: the target object;
  - place on an object: the receptacle;
  - place on a fixture (dining counter) or a fixture step: the robot base,
    after navigation.
- *Controller.* `probe-controls` measures the gains: the motion per step for
  each unit action, fitted by least squares for the arm in the base frame and
  for the base in its own frame. A passing probe has R² ≥ 0.9, condition
  number < 20 and no motion leaking between arm and base mode; the executor
  refuses gains from a probe that did not pass. The probe also measures the
  torso's effect on the base height, in both modes. This replaces reading the
  composite controller's config. All poses come from the observation's state
  keys, which the policy and the LeRobot demos share. Self-replay checks that
  shared convention at every segment start (difference ≤ 1e-3, otherwise the
  demo is reported as a mismatch). This settles the first open item above.

  *Probe amended (27 November 2026, after the first probe and before any
  executor run).* The first probe (ArrangeTea, pretrain seed 0) had R² 0.99,
  0.97 and 0.95 (arm position, arm rotation, base), condition numbers 1.3–3.1,
  arm-mode commands moving the base by at most 2e-5 m per step, and torso
  0.0077 m per step per unit in both modes. It failed only on base-mode commands
  moving the hand in the base frame: 0.013 m in one step against 5e-3.
  That rule measured the step-to-step change while the base accelerates, which
  cannot tell a lagging arm from one that holds its world pose. The executor
  re-approaches the first arm frame after base motion (beyond TRACK_TOL), so
  a transient lag does not matter; an arm left at its world pose would
  (carrying during navigation). The rule now: after each base-mode run, 20
  steps of zero motion, and the hand must be back within 5e-3 m of its
  starting pose in the base frame. A zero-motion run of the same length
  checks the restore itself (also within 5e-3 m). The per-step value is still
  reported, with each run's base travel, turn and settled offset. The other
  thresholds are unchanged.

  *Amended again (27 November 2026, after the second probe, still before any
  executor run).* The second probe gave the same gains, the largest base
  travel 0.062 m and turn 0.17 rad, and the hand 7.3 mm off after the base runs
  and 7.3 mm off after the zero-motion run. The arm is not left behind
  (that would be about 6 cm). The 7 mm happens with no command at all, so
  it is drift, not a leak from base mode, and a fixed 5e-3 m on the
  zero-motion run tested the wrong thing. What the executor needs instead:
  - *Base runs against the zero-motion run.* After each base run plus 20
    steps of zero motion, the hand must be within 5e-3 m of where the
    zero-motion run's hand is after the same number of steps (a difference
    of vectors, not of norms).
  - *Drift over a navigation.* The executor sends no arm command while the
    base drives, for up to 300 steps (`NAVIGATE_MAX_STEPS`). Zero-motion runs
    of 300 steps in each mode must keep the hand within 0.02 m (TRACK_TOL)
    of its start in the base frame. The probe reports the offset every 10
    steps and the final offset vector, so a drift that keeps growing is
    visible as such.

  *Third probe (27 November 2026): the 7 mm comes from the restore.* With
  the zero-motion runs first, straight after the environment reset, the hand
  drifted 8e-6 m over 300 steps in either mode. The same run made after
  the motion runs (second probe) drifted 7.3 mm, and every base run, each
  restored after the previous one, ended 7.3 mm off. `restore_world` (the
  evaluator's `env_to_last_good`, through the wrapper's `reset_to` with
  states only) sets the simulator state and does not touch the robot's
  controllers, so their goals, and any frame they keep, carry over from the
  previous run. Gate R restores alarm snapshots this way on a reused
  environment, and so did every ladder branch. `probe-restore`
  (`r2-probe-restore`, diagnostic only) measures the hand's motion after a
  restore that follows each kind of motion (arm, base, turn, torso, none),
  with and without `reset_controllers` (each part controller's goal set to
  the current state), checks whether a forward base command still points
  forward after the base has turned, and lists the controllers' goal and
  frame attributes. Until it has run, the probe stays failed and no executor
  check runs.
- *Ranking and navigation.* Sources are ranked by d = the distance between
  the first executed step's reference positions in the base frame, plus
  0.1 m per radian of heading difference. If even the best d exceeds 0.25 m,
  the robot first drives to the fixture that holds that reference. The goal is
  that fixture's `compute_robot_base_placement_pose` without a reference
  object, slid along the fixture's front until the robot faces the object's
  current position. With a reference object, the helper uses the object's
  initial placement instead. The navigation controller:
  - is a holonomic proportional controller, limited to 0.02 m and 0.05 rad
    per step;
  - drives on to 5 cm and 0.05 rad, or stops at 300 steps;
  - reports the NavigateKitchen test (0.20 m, cos ≥ 0.98).
- *Tracking.* The arm approaches a step's first pose at most 1 cm and
  0.05 rad per step, then plays the source frames, holding each frame until
  the end effector is within 2 cm of it. Frames where the demonstrator
  switched to base mode inside a step are replayed on the base: the demo's
  base pose, moved through the step's reference frame, within 2 cm and
  0.05 rad. The torso follows the demo's base height relative to the
  reference (beyond 1 cm), in whichever mode the probe found it moves, and
  is left alone if the probe found none. The expert cap counts the executed
  source frames (scripted release and set-aside count as 60 each) and adds
  50%. Navigation is capped separately, per phase.
- *Wrong tool on the board (gap 1).* The wrong tool is picked with the pick
  step of a demo whose correct tool it was. It is then lifted 15 cm, moved
  30 cm sideways away from the board, lowered 10 cm and released. This
  happens before the correct tool's pick, or after its place if it is
  already held.
- *Mug on the tray's edge (gap 3).* There is no special code: the pick is
  object-relative, so it starts wherever the mug is. Gate R records each
  object's pose in the base frame, so results can be split by where the mug
  started.
- *Gate R statistic.* Gate R is the share of class-2 states with at least one
  attempt ending in full-task success, threshold 0.5. "Gate R uses this one"
  above means the full outcome decides, not the expert one. The expert rate
  and full rate per attempt are reported but not gated.
- *Pretrain-transfer seeds.* 710007–710026; Phase B must not use them. A
  transfer run tries stages in the ladder's order and stops at the first that
  fails. Reset *i* uses the *i*-th ranked source.
- *Recording (27 November 2026).* Gate R records every attempt as XR-1 saw it,
  unless `--no-record` is passed. Frame *t* is the observation before action
  *t*, and each attempt keeps:
  - the three camera videos (mp4, 20 fps);
  - the named states and the executed 12-d actions, with gripper and control
    mode stored as −1/1 (the demos' raw robosuite convention, whatever the
    executor or XR-1 emitted);
  - a phase per frame (expert or policy).
  `export-lerobot` turns the recorded attempts (successes by default) into one
  LeRobot folder per task. The folder uses the demos' modality layout and adds
  an `annotation.r2.phase` column, so the Step 0 sample builder and trainer
  read it unchanged (`--lerobot-dir TASK=PATH`). Which frames train which
  model is decided in Phase C. Recording Gate R attempts does not make them
  training data. They are Phase A states, and whether they may be used is a
  Phase C decision.

The same commit fixes `stage_segments.jsonl`. Each segment's own `stage` key
(upstream's pick/place label) overwrote the predicate name in the row, so
rows written before 27 November carry the upstream label. The label is now
`upstream_stage`. R2 reads the index directly, not that file.

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
   and a converter to the XR-1 training format from Step 0. **Done for R2**
   (`AttemptRecorder` and `export-lerobot` in `expert_recovery.py`); Phase B's
   own-branch resamples can reuse the recorder. The ladder now
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
9. **Written (R2):** `robocasa/recovery/expert_recovery.py`. Commands:
   - `probe-controls`: measures the action gains;
   - `self-replay` and `transfer`: the executor checks (simulator, no
     policy);
   - `gate-r`: expert plus XR-1 at the class-2 alarm states (simulator and
     policy server);
   - `analyze`: recomputes the summaries.
   The executor runs on a world adapter. Its tests run it on a kinematic
   stand-in with a source recorded in another scene and rotated gains.

Tests (simulator- and PyTorch-free; the detector save/load test runs only
where PyTorch is installed):

```bash
python -m unittest tests.test_hs_p_alarm tests.test_alarm_branching tests.test_failure_audit tests.test_recoverability_ladder tests.test_expert_segments tests.test_expert_recovery
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

R2 runs in order and does not move past a step that fails. The probe, the
self-replay and the transfer check each use one GPU for rendering and no
policy server (`R2_GPU`, default 0). Gate R uses ladder workers
(`R2_GPU_PAIRS`, ports from `R2_PORT_BASE`, default 10480). It refuses to
start until both executor checks have passed.

```bash
bash scripts/run_alarm_branching_xr1.sh r2-probe          # must print "ok": true
bash scripts/run_alarm_branching_xr1.sh r2-self-replay
bash scripts/run_alarm_branching_xr1.sh r2-transfer
bash scripts/run_alarm_branching_xr1.sh r2-status
bash scripts/run_alarm_branching_xr1.sh r2-analyze
bash scripts/run_alarm_branching_xr1.sh r2-gate
```

Outputs are under `$ALARM_ROOT/r2/`:
- `control_probe.json`;
- `self-replay/self_replay_summary.json`;
- `transfer/transfer_summary.json`;
- `gate_r/shard*/`, with `gate_r.{json,md}` per shard. `r2-analyze` pools the
  shards. Recordings are in `gate_r/shard*/recordings/<state>/k<NN>/`.
  `r2-export` writes the LeRobot folders under
  `$STORAGE_BS/robocasa_failure_datasets/r2_recoveries_<stamp>/`.

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

- Step 0 answers: XR-1 fine-tuning entry point and data format (answered 26
  November, see Step 0 findings); whether the training model takes the
  RoboCasa365 inputs, the input-match check, run time and the sanity-check
  result (answered 27 November: Step 0 passed).
- Exact recovery-prompt text and the replay mixing ratio, frozen with the
  training code.
- Whether ladder v2 snapshots on `/gs/bs` are complete for all 76 parents
  (A1 needs them). Answered: all 41 branched parents had complete snapshots.
- R1 answers: whether the composite human demos for the five tasks include
  replayable states or object poses (answered: yes, per-frame simulator
  states in the LeRobot `extras/`), and which subtasks the class-2 states
  are stuck on (answered 26 November: nine stages across the five tasks, all
  with enough segments; `r1_coverage.md`).
