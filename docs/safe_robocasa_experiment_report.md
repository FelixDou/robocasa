# SAFE on RoboCasa: Complete Experimental Report

**Status:** technical research record
**Period covered:** July 13–August 15, 2026
**Models covered:** π0 RoboCasa, RLDX-1, Xiaomi-Robotics-1, and the engineering-only ABot-M0.5 adapter
**Primary task:** predict eventual rollout failure from policy-internal features, following the official SAFE method as closely as possible

## Executive summary

We built and validated an end-to-end SAFE pipeline for RoboCasa:

1. capture the correct policy-internal feature once per genuine policy inference;
2. collect success and failure rollouts without storing repeated features for cached actions;
3. export data in the official SAFE `env_records` / `policy_records` format;
4. train the official independent MLP and LSTM detectors;
5. select hyperparameters without touching the held-out outer test set;
6. evaluate ROC-AUC and PRC-AUC, test duration-only leakage, calibrate conformal operating points, and render score trajectories over videos.

The main scientific result is not that atomic and composite tasks require separate global detectors. Training separate atomic-only and composite-only rollout detectors did **not** improve matched held-out performance relative to the mixed detector. Instead, atomic rollouts were much easier to classify:

- mixed RLDX MLP, task-normalized discrimination:
  - atomic ROC-AUC: **0.827**
  - composite ROC-AUC: **0.667**
- at conformal \(\alpha=0.15\):
  - atomic: TPR **0.275**, FPR **0.027**, balanced accuracy **0.624**
  - composite: TPR **0.167**, FPR **0.067**, balanced accuracy **0.550**

This supported the hypothesis that a single terminal success/failure label is too coarse for long-horizon composite tasks. We therefore implemented **Subtask-SAFE**, whose target is: “does the active natural-language subtask eventually fail before completion?” Semantic stage boundaries were derived from ordered RoboCasa task decompositions, duplicate-predicate stages were merged, and train/test splitting was performed by parent rollout before segment extraction.

Subtask-SAFE produced a promising retrospective segment-level result but failed the stricter causal test:

- 150 parent rollouts yielded 426 usable segments: 341 successful and 85 failed;
- independent MLP held-out segment ROC-AUC: **0.734 ± 0.002**;
- LSTM held-out segment ROC-AUC: **0.601 ± 0.041**;
- after training-only task/subtask normalization and parent-aware bootstrap, MLP ROC-AUC fell to **0.558**, 95% CI **[0.428, 0.676]**;
- at the first online inference of each active segment, MLP ROC-AUC was **0.389**, while the task-conditioned elapsed-time baseline was **0.742**;
- conformal MLP operation at \(\alpha=0.15\) achieved TPR **0.121**, FPR **0.119**, and balanced accuracy **0.501**.

The raw segment ROC therefore overstates deployable early-warning performance. Subtask identity, duration, and unequal stage priors explain much of the apparent discrimination. The current RLDX features contain useful retrospective information, especially for atomic tasks, but do not yet provide a reliable causal early-failure signal at composite subtask onset.

The best current RLDX result is the independent MLP trained on 25 successes and 25 failures for each of ten tasks:

- raw pooled ROC-AUC: **0.575 ± 0.004**
- training-only task-normalized ROC-AUC: **0.713 ± 0.017**
- macro per-task ROC-AUC: **0.715 ± 0.020**

The LSTM was consistently weaker. A controlled natural-prevalence experiment also showed that neither natural outcome sampling nor the paper's inverse-frequency loss weighting materially changed MLP performance. The dominant limitations are therefore representation, temporal target alignment, task/stage score offsets, and sparse failure support—not the original 25/25 balancing choice alone.

The Xiaomi-Robotics-1 study extended original binary-outcome SAFE to a much
broader natural-outcome dataset. The released policy was evaluated on the 39
official RoboCasa365 tasks whose published Xiaomi success count lay inclusively
between 5 and 45 out of 50. A fresh collection produced 1,950 valid rollouts,
and merging it with 692 earlier Xiaomi rollouts produced a 2,642-rollout source
pool. A deterministic seed-0 selection retained exactly 50 rollouts per task:
1,105 successes and 845 failures. One task with only one retained failure was
excluded from model development, leaving 38 tasks, 1,824 outer-training
rollouts, and a fixed 76-rollout outer test with one success and one failure per
task.

The leakage-free 810-fit Xiaomi grid selected an independent MLP and an LSTM
without scoring the outer test. On the final three-seed refits, raw pooled
ROC-AUC was **0.532 ± 0.016** for the MLP and **0.628 ± 0.008** for the
LSTM, while completed-rollout duration alone reached **0.804**. Training-only
task normalization changed the scientific ranking: normalized MLP ROC-AUC
rose to **0.692 ± 0.013** with average precision **0.733**, outperforming the
normalized LSTM at **0.646 ± 0.017**. In causal trajectory replay, the MLP
SAFE score achieved task-macro ROC-AUC **0.671** at 10% and **0.626** at 25%
of the training-derived task horizon, while task-conditioned elapsed time was
exactly **0.500** within task at both landmarks. At the validation-selected
event threshold, MLP SAFE detected 98.3% of failures at an adjusted mean
detection fraction of 0.668, earlier than time-only at 0.820, but its false
positive rate was also higher: 13.2% versus 5.3%. This is the first evidence in
the report of useful early within-task SAFE signal beyond elapsed time, but it
is post-hoc and has not yet survived a matched-FPR prospective test.

## 1. Scope, labels, and evaluation rules

### 1.1 Raw SAFE target

The primary target remained the official binary rollout label:

- `success`: the RoboCasa task success predicate became true;
- `failure`: the rollout reached its horizon without success.

Ordered subtask annotations were not used for the rollout-level baselines. They were later introduced as a separate Subtask-SAFE target; raw rollout-level and subtask-level claims are kept distinct throughout this report.

### 1.2 Unit of observation

One SAFE feature record is stored per **real policy inference**. If an action chunk is reused for several simulator steps, the cached action consumption does not create duplicate SAFE feature records.

### 1.3 Models

We used the detector families in the official SAFE repository:

- `indep`: independent MLP detector;
- `lstm`: temporal LSTM detector.

The detector input is derived from the policy-internal feature tensor with SAFE’s horizon and diffusion selectors.

### 1.4 Selection and held-out evaluation

The final π0 and RLDX experiments used:

- an outer train/test split fixed across model seeds;
- inner three-fold cross-validation on the outer training set;
- hyperparameter selection from inner validation ROC-AUC only;
- final refits with seeds 0, 1, and 2;
- no use of the outer test set during model selection;
- rollout-duration baseline checks;
- task-normalized analysis using statistics fitted from training data only.

Reported `±` values are population standard deviations across the three model seeds unless stated otherwise.

### 1.5 Important metric distinction

Three different aggregation views appear in this report:

- **raw pooled ROC/PRC:** all held-out rollouts pooled with their raw detector scores;
- **training-only task-z pooled ROC/PRC:** scores normalized per task with center and scale estimated from training data only;
- **macro task ROC:** ROC-AUC calculated per task and averaged.

Raw pooled results can be depressed by task-specific score offsets even when the detector ranks success and failure reasonably within each task. The task-normalized and macro metrics diagnose that effect, but they do not replace the operational calibration experiment.

## 2. SAFE feature implementations

### 2.1 π0 RoboCasa

The π0 integration follows the official SAFE `pre_velocity` feature:

- action-expert suffix hidden states;
- captured immediately before `action_out_proj`;
- one tensor per genuine inference;
- `float32`;
- expected inference shape:
  \[
  (10\ \text{flow steps}, 50\ \text{action tokens}, 1024\ \text{channels}).
  \]

Relevant versions:

- official SAFE repository commit: `b6036abe07b2b2bb9996afb2c07f13d6a9f507c0`
- official SAFE OpenPI reference commit: `9c99ed53f6a0c9be93a1c63cee5792620777d96b`
- pinned RoboCasa OpenPI commit: `5a6beda9ff99da30b4e1b59320f6a32971d7c397`
- companion patch: `patches/openpi_safe_features_5a6beda.patch`

The collector also records the executed actions and can record videos. Rollout metadata contains the task, environment seed, reset seed, split, horizon, model/checkpoint provenance, feature layer, and artifact paths.

### 2.2 RLDX-1

RLDX-1 has no module literally named `action_expert`, but its action model contains an equivalent action-token stream. The selected feature is:

- `ao[:, -action_horizon:]`;
- immediately before the embodiment-specific action decoder;
- layer identifier:
  `action_model_msat_action_suffix_pre_action_decoder`;
- stored as `float32`;
- observed tail shape:
  \[
  (4\ \text{diffusion steps}, 16\ \text{action tokens}, 1024\ \text{channels}).
  \]

Relevant versions:

- RLDX repository commit: `ef05cd4ae634ff97d672d42275febbc0b92cc192`
- checkpoint: `RLWRLD/RLDX-1-FT-RC365`
- companion patch: `patches/rldx1_safe_features_ef05cd4.patch`

The simulator-facing client uses ZeroMQ and preserves the same “one feature per real inference” rule. A language batching mismatch was also corrected:

- flat request fields: `[instruction]`;
- nested direct-policy `language`: `[[instruction]]`.

π0 calibration parameters were never reused for RLDX; each model family was trained and calibrated from its own captured features.

### 2.3 ABot-M0.5

An ABot-M0.5 SAFE adapter was implemented on the separate branch:

- branch: `codex/safe-abot-m05`
- commit: `99801d0`
- feature: normalized action-stream tokens immediately before `action_proj_out`;
- expected feature shape:
  \[
  (50\ \text{diffusion steps}, 32\ \text{action tokens}, 768\ \text{channels}).
  \]

This work is an **engineering integration only** in the present report. No completed ABot SAFE collection, detector training, calibration, or scientific result was recorded, so ABot is excluded from all performance comparisons.

### 2.4 Xiaomi-Robotics-1

The Xiaomi integration preserves the original SAFE binary final-rollout label
and does not use Subtask-SAFE supervision. XR-1 generates each action chunk
with a five-step flow-matching DiT. At every genuine policy inference, the
integration records the final-layer action-token hidden states immediately
before `action_output_layer`:

- feature identifier: `dit_action_tokens_pre_action_output_layer`;
- expected inference shape: `(5, 30, 1024)`;
- stored dtype: `float32`;
- stored rollout shape:
  `(num_policy_inferences, flow_steps, action_horizon, feature_dim)`.

The hidden-state copy is converted to `float32` only after the unchanged BF16
tensor has passed through the action-output layer, so feature capture does not
alter policy actions. Feature capture, environment stepping, and video storage
remain one record per genuine inference rather than one record per cached
action.

Pinned artifacts:

- official SAFE repository:
  `b6036abe07b2b2bb9996afb2c07f13d6a9f507c0`;
- Xiaomi source:
  `4da1db0a4deefa6de7ebb4ef0b8754017290f5f7`;
- checkpoint: `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`;
- checkpoint revision:
  `0d1aa76d0d82debc9b611e4d1e231096434d5be4`;
- Transformers: `4.57.1`;
- RoboCasa integration branch: `codex/safe-xiaomi-robotics-1`;
- pooled-data implementation commit: `5190c96e`.

The clean Xiaomi source and checkpoint were preserved. SAFE used an isolated
source worktree, a derived hard-linked checkpoint with an independently
patched `modeling_mibot.py`, and explicit server-side socket closure between
tasks. The socket lifecycle fix was required for reliable task transitions in
long resumable collections.

## 3. Chronology and operational validation

### 3.1 Initial π0 feature-server failure

The first atomic smoke produced no valid rollouts because the OpenPI server did not return `safe_features`. The client correctly raised:

> SAFE feature collection was requested, but the OpenPI server did not return `safe_features`.

The running server was using an incompatible checkout. Applying the companion patch to the pinned OpenPI commit and restarting the server fixed the protocol.

### 3.2 Corrected π0 sink smoke

Dataset:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/safe_sink_corrected_smoke_20260713_175820
```

Result:

- 3 valid rollouts;
- 2 successes, 1 failure;
- seeds 0, 1, and 2;
- rollout lengths 600, 342, and 242 simulator steps;
- official SAFE loader compatible.

This established that the patched server, collector, feature records, labels, and loader were working together. The 600-step horizon came from the then-current official RoboCasa evaluation configuration. Differences between 26-second and 52-second videos were attributable to recording stride/FPS rather than a different simulator horizon.

### 3.3 Incorrect first five-task collection

Dataset:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/safe_atomic_seen5_10x10_pi0_74999_20260713_172407
```

It stopped after collecting ten `OpenCabinet` failures and no successes. This was not accepted as a detector dataset. The collection protocol was corrected to match the official split, task set, reset/seed procedure, and task horizons.

### 3.4 Healthy π0 five-task collection

Source dataset:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/safe_atomic_seen5_10x10_official_seed7_20260713_212804
```

Tasks:

- `CloseFridge`
- `OpenDrawer`
- `PickPlaceCounterToCabinet`
- `PickPlaceCounterToStove`
- `TurnOnSinkFaucet`

Audit:

| Task | Successes | Failures |
|---|---:|---:|
| CloseFridge | 18 | 10 |
| OpenDrawer | 10 | 14 |
| PickPlaceCounterToCabinet | 10 | 13 |
| PickPlaceCounterToStove | 10 | 11 |
| TurnOnSinkFaucet | 12 | 10 |
| **Total** | **60** | **58** |

Additional checks:

- 118 valid rollouts;
- 0 errors;
- official seed protocol with base seed 7;
- 382 skipped outcomes after quotas were met;
- no missing tasks;
- no duplicate rollout IDs;
- no duplicate task/seed/reset episodes;
- all requested artifacts present;
- official SAFE loader compatible.

### 3.5 Balanced π0 export

Official-format export:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/safe_atomic_seen5_10x10_official_seed7_20260713_212804_official_balanced_10x10_seed0
```

It contains 100 rollouts: exactly ten successes and ten failures per task. This deterministic balanced subset was used to make the small-data comparison interpretable. The original naturally imbalanced 118-rollout source remains preserved.

Export log:

```text
/gs/bs/tga-shinoda/felid/robocasa_logs/eval/safe_atomic_seen5_10x10_official_seed7_20260713_212804_official_balanced_10x10_seed0_export.log
```

### 3.6 Attempted π0 expansion to 17 atomic tasks

We prepared a broader π0 collection for all 17 atomic-seen tasks with nonzero measured policy success, excluding `NavigateKitchen` because its measured success rate was 0%. The target was 20 successes and 20 failures per task, or 680 retained rollouts.

The collector was extended to:

- retain only the requested class quotas;
- discard excess majority-class outcomes;
- preserve skipped counts;
- merge hard-linked shards;
- mark unmet quotas as partial;
- stop after a configurable error ceiling.

The initial broad run:

```text
safe_atomic17_incremental_20x20_seed8_20260717_224824
```

was intentionally stopped and not used. Both top-up shards had zero valid rollouts and accumulated 713 errors because:

1. the dedicated SAFE worktree was missing ignored RoboCasa object assets, including `Stool015/model.xml`;
2. neither patched OpenPI server was running.

The failure motivated the `--max-errors` guard and explicit dual-server/asset preflight. A corrected relaunch root was prepared with the pattern:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/safe_atomic17_incremental_20x20_seed8_fixed_*
```

No healthy 680-rollout merged audit or detector result from that expansion was recorded before work shifted to RLDX. It must therefore be treated as an incomplete collection attempt, not as a completed experiment.

### 3.7 Official evaluator environment repair

The first attempt to evaluate completed SAFE grid checkpoints inside the minimal `vla_safe` environment failed with:

```text
ModuleNotFoundError: No module named 'robosuite'
```

The offline evaluator did not actually require a simulator, but importing `robocasa` pulled RoboSuite at package import time. The evaluator was refactored so simulator-independent SAFE evaluation could run without RoboSuite. The repaired grid subsequently completed all 810 evaluations with zero failures.

### 3.8 RLDX dual-server smoke

Dataset:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/rldx1_dual_safe_smoke_fixed_20260726_190445
```

Each of ports 20100 and 20101 produced one valid, loader-compatible failed rollout. This was an infrastructure smoke, not a success-rate or detector-quality result.

### 3.9 Healthy RLDX ten-task, 10/10-per-task dataset

Dataset root:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/rldx1_safe_mixed10_10x10_20260726_190751
```

Merged dataset:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/rldx1_safe_mixed10_10x10_20260726_190751/merged_10x10
```

Official export:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/rldx1_safe_mixed10_10x10_20260726_190751/official_safe_10x10
```

The five atomic tasks were:

- `CloseToasterOvenDoor`
- `CoffeeSetupMug`
- `PickPlaceCounterToStove`
- `PickPlaceDrawerToCounter`
- `TurnOnSinkFaucet`

The five composite tasks were:

- `LoadDishwasher`
- `PreSoakPan`
- `ScrubCuttingBoard`
- `StackBowlsCabinet`
- `WashLettuce`

Audit:

- 200 rollouts;
- 100 successes and 100 failures;
- exactly 10 successes and 10 failures per task;
- 23,395 policy records;
- 200 video links, none broken;
- no missing tasks, duplicate IDs, duplicate task/seed/reset episodes, or missing artifacts;
- 5.74 GiB of materialized feature data;
- official SAFE loader compatible.

### 3.10 RLDX expansion to 25/25 per task

The second collection added 15 new successes and 15 new failures for every task. Both five-task shards completed with:

- 150 retained rollouts per shard;
- 75 successes and 75 failures per shard;
- no errors;
- all per-task quotas reached.

Merged dataset:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/rldx1_safe_mixed10_extra15x15_20260727_163929/merged_25x25
```

Official export:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/rldx1_safe_mixed10_extra15x15_20260727_163929/official_safe_25x25
```

Audit:

- 500 rollouts;
- 250 successes and 250 failures;
- exactly 25 successes and 25 failures per task;
- 58,834 policy records;
- complete conversion report;
- approximately 15 GiB on disk.

An early inspection script printed `INVALID` because it expected legacy summary keys such as `environment_records` and `labels`. The conversion itself was complete; the report schema, not the export, caused that false alarm.

## 4. π0 experiments

### 4.1 Training smoke and duration leakage

Independent MLP smoke:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_pi0_indep_smoke_20260716_145732
```

LSTM smoke:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_pi0_lstm_smoke_20260716_145834
```

Both official models trained and saved checkpoints. The duration-only baseline reached:

- train ROC-AUC: 0.969;
- seen validation ROC-AUC: 1.000;
- unseen validation ROC-AUC: 1.000.

This is evidence of strong episode-length leakage in termination-time evaluation. Failures often run to the full horizon while successful episodes terminate earlier. Consequently, matched-earliest or otherwise time-controlled evaluation is necessary before attributing performance to policy features.

### 4.2 Two-run grid pilot

Root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_pi0_grid_pilot_2gpu_20260716_152847
```

The 1,000-epoch MLP fit took roughly 3 seconds and the LSTM roughly 9 seconds. This was expected: the dataset was small and batch size 512 reduced each epoch to approximately one optimization batch. Fast execution was not evidence that training had been skipped.

### 4.3 Full official 810-fit grid

Root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_pi0_official_grid_2gpu_20260716_153526
```

Audit:

- 810/810 evaluations;
- 270 hyperparameter configurations;
- three model seeds per configuration;
- zero failures.

Selected configurations:

| Model | Horizon selector | Diffusion selector | Learning rate | Regularization | Train ROC | Seen ROC | Unseen ROC |
|---|---|---|---:|---:|---:|---:|---:|
| MLP | 1.0 | 0.0 | 3e-4 | 1e-3 | 0.9990 | 0.7989 ± 0.0807 | 0.4533 ± 0.0368 |
| LSTM | 1.0 | concat-2 | 1e-3 | 1e-2 | 0.9866 | 0.7917 ± 0.0694 | 0.5808 ± 0.0796 |

Interpretation:

- policy features contained substantial seen-task discrimination;
- cross-task transfer was weak;
- the near-perfect train scores indicated considerable overfitting;
- the small dataset and task-dependent distributions made a single global detector difficult;
- unseen-task results were diagnostic, not the target deployment condition.

### 4.4 Official-grid conformal operating points

Final report root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_pi0_official_grid_2gpu_20260716_153526/final_report
```

Matched-earliest conformal results:

| Model | Alpha | TPR | FPR | Balanced accuracy | Relative detection time |
|---|---:|---:|---:|---:|---:|
| MLP | 0.05 | 0.267 | 0.333 | 0.467 | 0.791 |
| MLP | 0.10 | 0.267 | 0.350 | 0.458 | 0.787 |
| MLP | 0.15 | 0.300 | 0.383 | 0.458 | 0.765 |
| MLP | 0.20 | 0.317 | 0.383 | 0.467 | 0.748 |
| LSTM | 0.05 | 0.117 | 0.133 | 0.492 | 0.890 |
| LSTM | 0.10 | 0.133 | 0.150 | 0.492 | 0.872 |
| LSTM | 0.15 | 0.183 | 0.150 | 0.517 | 0.827 |
| LSTM | 0.20 | 0.200 | 0.183 | 0.508 | 0.810 |

Neither model supplied a strong operating point: the MLP generated too many false alarms, while the LSTM reduced false alarms at the cost of detecting few failures and detecting them late.

### 4.5 Initial all-five-seen experiment

Root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_all5_seen_7x7_train_3x3_test_20260717_010030
```

Split:

- 70 training rollouts: 7 successes and 7 failures per task;
- 30 test rollouts: 3 successes and 3 failures per task.

Results:

| Model | Test ROC-AUC | Test PRC-AUC |
|---|---:|---:|
| MLP | 0.506 ± 0.009 | 0.541 ± 0.019 |
| LSTM | 0.702 ± 0.031 | 0.675 ± 0.025 |

This was an exploratory result using frozen hyperparameters. It was superseded by the leakage-free inner-CV experiment below.

Score-overlay videos for the initial LSTM experiment:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_all5_seen_7x7_train_3x3_test_20260717_010030/lstm_seed0/score_videos/*.mp4
```

### 4.6 Leakage-free all-five-seen inner CV and final refit

The timestamped CV root was not printed in the retained terminal output. It can be located with:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_all5_seen_innercv_*/
```

Audit:

- 810/810 inner-CV fits;
- 270 configurations;
- zero failures;
- outer test set never scored during selection.

Selected hyperparameters:

| Model | Horizon selector | Diffusion selector | Learning rate | Regularization | Inner validation ROC |
|---|---|---|---:|---:|---:|
| MLP | 1.0 | concat-2 | 3e-5 | 1e-3 | 0.7193 ± 0.0209 |
| LSTM | 0.0 | 0.0 | 1e-3 | 1e-2 | 0.8170 ± 0.0341 |

Final refit directory:

```text
<PI0_INNER_CV_ROOT>/final_refits
```

Final results on the common 30-rollout held-out set:

| Model | Test ROC-AUC | Test PRC-AUC |
|---|---:|---:|
| MLP | 0.600 ± 0.004 | 0.605 ± 0.002 |
| LSTM | 0.599 ± 0.039 | 0.637 ± 0.031 |

The large inner-CV-to-test gaps show that model selection remained noisy at this sample size. Neither detector provided robust held-out discrimination.

## 5. RLDX-1 experiments

### 5.1 Ten-task experiment with 10 successes and 10 failures per task

CV root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_seen10_official_cv_2gpu_20260727_135354
```

Final root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_seen10_final_refit_20260727_155824
```

Split:

- 140 training rollouts: 70 successes and 70 failures;
- 60 test rollouts: 30 successes and 30 failures;
- six held-out rollouts per task.

Selected inner-CV configurations:

| Model | Horizon selector | Diffusion selector | Learning rate | Regularization | Inner validation ROC |
|---|---|---|---:|---:|---:|
| MLP | concat-2 | 1.0 | 1e-3 | 1e-3 | 0.5482 ± 0.0066 |
| LSTM | 1.0 | 0.0 | 3e-4 | 1e-2 | 0.6220 ± 0.1099 |

Final results:

| Model | Raw pooled ROC | Raw pooled PRC | Task-z pooled ROC | Task-z pooled PRC | Macro task ROC |
|---|---:|---:|---:|---:|---:|
| MLP | 0.543 ± 0.007 | 0.566 ± 0.012 | 0.674 ± 0.023 | 0.708 ± 0.016 | 0.670 ± 0.037 |
| LSTM | 0.520 ± 0.025 | 0.522 ± 0.044 | 0.537 ± 0.026 | 0.558 ± 0.026 | 0.522 ± 0.036 |

Interpretation:

- raw pooled performance was close to chance;
- within-task MLP ranking was meaningfully better after training-only task normalization;
- score scale and offset varied strongly by task;
- the MLP was more stable and effective than the LSTM;
- only six held-out rollouts per task made individual task estimates very uncertain.

### 5.2 Ten-task experiment with 25 successes and 25 failures per task

The timestamped CV root was not printed in the retained output. It can be found with:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_25x25_cv_2gpu_*/
```

The CV sweep completed:

- 810/810 fits;
- 405 MLP and 405 LSTM fits;
- zero failures.

Final root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_25x25_final_2gpu_20260730_104400
```

Split:

- 340 training rollouts: 170 successes and 170 failures;
- 160 test rollouts: 80 successes and 80 failures;
- 16 held-out rollouts per task;
- identical test identities across all six final runs.

Selected configurations:

| Model | Horizon selector | Diffusion selector | Learning rate | Regularization |
|---|---|---|---:|---:|
| MLP | concat-2 | concat-2 | 1e-3 | 1e-3 |
| LSTM | concat-2 | concat-2 | 1e-4 | 1e-2 |

Final aggregate results:

| Model | Raw pooled ROC | Raw pooled PRC | Task-z pooled ROC | Task-z pooled PRC | Macro task ROC |
|---|---:|---:|---:|---:|---:|
| MLP | 0.575 ± 0.004 | 0.573 ± 0.005 | 0.713 ± 0.017 | 0.723 ± 0.031 | 0.715 ± 0.020 |
| LSTM | 0.554 ± 0.006 | 0.581 ± 0.014 | 0.547 ± 0.010 | 0.551 ± 0.017 | 0.576 ± 0.007 |

An earlier aggregate summary reported MLP PRC-AUC `0.565 ± 0.007`; the table above uses the later direct recalculation from saved held-out scores (`0.573 ± 0.005`). The ROC conclusions are unchanged.

MLP per-task ROC-AUC:

| Task | Type | ROC-AUC |
|---|---|---:|
| CloseToasterOvenDoor | atomic | 0.792 ± 0.091 |
| CoffeeSetupMug | atomic | 0.719 ± 0.013 |
| LoadDishwasher | composite | 0.578 ± 0.013 |
| PickPlaceCounterToStove | atomic | 0.786 ± 0.070 |
| PickPlaceDrawerToCounter | atomic | 0.807 ± 0.032 |
| PreSoakPan | composite | 0.620 ± 0.037 |
| ScrubCuttingBoard | composite | 0.573 ± 0.087 |
| StackBowlsCabinet | composite | 0.776 ± 0.060 |
| TurnOnSinkFaucet | atomic | 0.771 ± 0.052 |
| WashLettuce | composite | 0.724 ± 0.019 |

LSTM per-task ROC-AUC:

| Task | Type | ROC-AUC |
|---|---|---:|
| CloseToasterOvenDoor | atomic | 0.448 ± 0.045 |
| CoffeeSetupMug | atomic | 0.672 ± 0.056 |
| LoadDishwasher | composite | 0.667 ± 0.070 |
| PickPlaceCounterToStove | atomic | 0.615 ± 0.103 |
| PickPlaceDrawerToCounter | atomic | 0.656 ± 0.089 |
| PreSoakPan | composite | 0.469 ± 0.068 |
| ScrubCuttingBoard | composite | 0.500 ± 0.026 |
| StackBowlsCabinet | composite | 0.526 ± 0.065 |
| TurnOnSinkFaucet | atomic | 0.620 ± 0.032 |
| WashLettuce | composite | 0.589 ± 0.027 |

Scaling from 10/10 to 25/25 per task improved the MLP:

- task-z pooled ROC: 0.674 → 0.713;
- macro task ROC: 0.670 → 0.715;
- smaller uncertainty from 16 rather than six held-out rollouts per task.

It did not make the LSTM competitive. The MLP remained the preferred detector.

## 6. RLDX task-normalized conformal calibration

Calibration report root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_25x25_final_2gpu_20260730_104400_calibration_20260730_113807
```

Data partition:

- 340 training rollouts;
- 30 calibration successes;
  - 9 reference successes;
  - 21 nonconformity successes;
- 130 evaluation rollouts;
  - 50 successes;
  - 80 failures;
- train, calibration, and evaluation identities all disjoint.

Global MLP operating points:

| Alpha | TPR | FPR | Balanced accuracy | Relative detection time |
|---:|---:|---:|---:|---:|
| 0.05 | 0.133 | 0.033 | 0.550 | 0.769 |
| 0.10 | 0.142 | 0.033 | 0.554 | 0.766 |
| 0.15 | 0.196 | 0.040 | 0.578 | 0.716 |
| 0.20 | 0.337 | 0.080 | 0.629 | 0.766 |

At the preselected \(\alpha=0.15\):

| Task | TPR | FPR | Balanced accuracy |
|---|---:|---:|---:|
| CloseToasterOvenDoor | 0.708 | 0.200 | 0.754 |
| CoffeeSetupMug | 0.250 | 0.067 | 0.592 |
| LoadDishwasher | 0.000 | 0.067 | 0.467 |
| PickPlaceCounterToStove | 0.333 | 0.067 | 0.633 |
| PickPlaceDrawerToCounter | 0.042 | 0.000 | 0.521 |
| PreSoakPan | 0.000 | 0.000 | 0.500 |
| ScrubCuttingBoard | 0.083 | 0.000 | 0.542 |
| StackBowlsCabinet | 0.167 | 0.000 | 0.583 |
| TurnOnSinkFaucet | 0.083 | 0.000 | 0.542 |
| WashLettuce | 0.292 | 0.000 | 0.646 |

Interpretation:

- calibration controlled false alarms reasonably well;
- failure recall remained low;
- detection was generally late;
- performance varied sharply by task;
- a single rollout-level operating point is not sufficient for reliable composite-task intervention.

## 7. Atomic-versus-composite ablation

The timestamped root was not printed in the retained terminal record. Locate it with:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_25x25_type_ablation_*/
```

Its structure is:

```text
<SAFE_TYPE_ROOT>/
├── mixed_calibration/{atomic,composite}/
├── cv/{atomic,composite}/
├── final/{atomic,composite}/
└── split_calibration/{atomic,composite}/
```

### 7.1 Mixed detector evaluated separately by task type

At \(\alpha=0.15\):

| Task type | Task-z ROC | Task-z PRC | TPR | FPR | Balanced accuracy | Relative detection time |
|---|---:|---:|---:|---:|---:|---:|
| Atomic | 0.827 | 0.891 | 0.275 | 0.027 | 0.624 | 0.650 |
| Composite | 0.667 | 0.773 | 0.167 | 0.067 | 0.550 | 0.954 |

Atomic rollouts were substantially easier to rank and failures were detected earlier. Composite detections occurred very near the end of the rollout.

### 7.2 Separate model selection

Inner-CV selections:

| Type | Horizon selector | Diffusion selector | Learning rate | Regularization | Inner validation ROC |
|---|---|---|---:|---:|---:|
| Atomic | concat-2 | 1.0 | 1e-3 | 1e-3 | 0.6665 ± 0.0116 |
| Composite | concat-2 | concat-2 | 1e-3 | 1e-2 | 0.6601 ± 0.0219 |

Final raw held-out results:

| Type-specific detector | Test ROC-AUC | Test PRC-AUC | Train | Test |
|---|---:|---:|---:|---:|
| Atomic MLP | 0.741 ± 0.018 | 0.707 ± 0.022 | 170 | 80 |
| Composite MLP | 0.586 ± 0.005 | 0.578 ± 0.013 | 170 | 80 |

Both tests used 40 successes and 40 failures. Their identities were exact atomic/composite subsets of the original mixed-model test split.

### 7.3 Matched mixed-versus-split calibration

Using identical calibration and evaluation identities:

| Type | Model | Task-z ROC | Task-z PRC | TPR | FPR | Balanced accuracy | Relative detection time |
|---|---|---:|---:|---:|---:|---:|---:|
| Atomic | Mixed | 0.827 | 0.891 | 0.275 | 0.027 | 0.624 | 0.650 |
| Atomic | Split | 0.821 | 0.880 | 0.217 | 0.133 | 0.542 | 0.229 |
| Composite | Mixed | 0.667 | 0.773 | 0.167 | 0.067 | 0.550 | 0.954 |
| Composite | Split | 0.668 | 0.777 | 0.192 | 0.107 | 0.543 | 0.928 |

Differences, split minus mixed:

- atomic:
  - ROC −0.007;
  - PRC −0.011;
  - balanced accuracy −0.083;
- composite:
  - ROC +0.001;
  - PRC +0.004;
  - balanced accuracy −0.007.

### 7.4 Interpretation

The ablation rejects the simple explanation that mixed atomic/composite training confused the detector:

- the separate atomic model did not improve ranking or calibrated operation;
- the separate composite model was effectively tied in ranking and slightly worse in balanced accuracy;
- the mixed model therefore already benefited from the additional cross-type data.

However, the experiment strongly supports a different diagnosis:

- atomic tasks have a relatively coherent action objective and terminal label;
- composite tasks contain several semantically different phases;
- the same rollout-level failure label combines failures at different stages;
- early composite features can be healthy for the current subtask even when a later subtask eventually fails;
- failure evidence for some composites only appears near the end, explaining the 0.954 relative detection time.

The evidence therefore motivates **composite decomposition**, not separate global task-type models.

## 8. Natural-prevalence and inverse-frequency-loss experiment

The original SAFE work trains on naturally imbalanced policy outcomes and uses inverse-frequency loss weighting. To test whether our balanced 25/25 design hid the RLDX signal, we constructed immutable natural-rate subsets from all genuine observed outcomes while preserving the same 160-rollout held-out test set.

Experiment root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_natural_rate_screen_20260730_161528
```

The evidence pool contained 689 genuine outcomes. Each of five subset seeds contributed 23 training rollouts per task. Across all model and subset seeds, 90 runs completed without failure.

Observed policy outcome rates ranged from 0.294 success for `LoadDishwasher` to 0.731 for `PickPlaceCounterToStove`. Three matched regimes were compared:

- balanced/matched with official inverse-frequency weighting: 110 successes, 110 failures;
- natural prevalence without weighting: 113 successes, 107 failures;
- natural prevalence with official inverse-frequency weighting: 113 successes, 107 failures.

### 8.1 Aggregate results

| Model/regime | Raw ROC | Task-z ROC | Macro task ROC | Raw AP | Task-z AP | Macro task AP |
|---|---:|---:|---:|---:|---:|---:|
| MLP, matched weighted | 0.566 ± 0.006 | 0.691 ± 0.012 | 0.694 ± 0.019 | 0.572 ± 0.012 | 0.713 ± 0.010 | 0.743 ± 0.014 |
| MLP, natural unweighted | **0.573 ± 0.004** | **0.700 ± 0.008** | 0.714 ± 0.018 | 0.568 ± 0.005 | 0.721 ± 0.006 | 0.748 ± 0.017 |
| MLP, natural weighted | 0.571 ± 0.004 | 0.699 ± 0.004 | **0.715 ± 0.007** | 0.563 ± 0.005 | **0.726 ± 0.009** | **0.751 ± 0.013** |
| LSTM, matched weighted | **0.546 ± 0.021** | **0.538 ± 0.028** | **0.556 ± 0.026** | **0.579 ± 0.024** | **0.556 ± 0.030** | 0.645 ± 0.026 |
| LSTM, natural unweighted | 0.525 ± 0.020 | 0.522 ± 0.017 | 0.553 ± 0.016 | 0.564 ± 0.017 | 0.532 ± 0.015 | **0.647 ± 0.014** |
| LSTM, natural weighted | 0.532 ± 0.023 | 0.529 ± 0.020 | 0.554 ± 0.020 | 0.569 ± 0.020 | 0.531 ± 0.014 | 0.645 ± 0.015 |

Paired effects were small relative to variation across subset seeds. For the MLP, natural minus matched was +0.008 ± 0.012 in task-z ROC and +0.021 ± 0.013 in macro-task ROC. Official weighting minus unweighted natural training was −0.001 ± 0.007 and +0.001 ± 0.013, respectively. The LSTM remained weak under all regimes.

### 8.2 Interpretation

- Balanced selection was not the main cause of weak raw pooled performance.
- Natural prevalence produced, at most, a small MLP improvement.
- The official inverse-frequency loss neither materially helped nor harmed the MLP at this mild global imbalance.
- Task-aware aggregation remained far more consequential than sampling or loss weighting.
- Atomic MLP macro ROC reached 0.770 in the natural-weighted condition, versus 0.660 for composites.

The controlled conclusion is therefore not that class weighting is unnecessary in general. It is that, for these available outcomes and fixed test identities, weighting could not overcome the representation and stage-alignment limitations.

## 9. Subtask-SAFE implementation and dataset

### 9.1 Prediction target and semantic segmentation

Subtask-SAFE changes the binary unit from a full rollout to the active ordered natural-language subtask:

> Does the active subtask eventually fail before completion?

A successful parent rollout contributes one successful sample for every observed completed subtask. A failed parent rollout contributes successful samples for its completed prefix and one failed sample for the active subtask that never completed. Unattempted later subtasks are excluded. Completed stages without an observed active interval, optional bypasses, and segments without policy inference are recorded but excluded from model training.

The first predicate-level decomposition was rejected because predicates are not equivalent to natural-language subtasks. The corrected schema:

- uses the ordered high-level task decomposition;
- associates each natural-language instruction with its completion predicates;
- merges adjacent stages that reuse the same completion predicate;
- removes stages already satisfied at reset;
- preserves source-stage provenance for every merge.

For example, `LoadDishwasher` was reduced to the five effective online stages: pick cup, place cup on rack, pick bowl, place bowl on rack, and close dishwasher.

### 9.2 Collection and coverage

Source dataset:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/rldx1_composite_subtask_seed8_9_10_150_20260803_210535
```

Official Subtask-SAFE export:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/rldx1_subtask_safe_official_20260803_212413
```

Collection covered five composite tasks and 150 parent rollouts:

| Quantity | Count |
|---|---:|
| Successful parent rollouts | 65 |
| Failed parent rollouts | 85 |
| Semantic task/subtask identities | 18 |
| Usable successful segments | 341 |
| Usable failed segments | 85 |
| Total usable segments | 426 |
| Official policy records | 27,420 |
| Segments without inference | 3 |
| Completed without activation | 12 |
| Optional bypasses excluded | 1 |

Failure support was highly nonuniform. Some early or mechanically easy stages had zero or one observed failure, while `ScrubCuttingBoard::cutting_board_scrubbed` and `StackBowlsCabinet::larger_bowl_placed_in_cabinet` had 15 and 17 failures. This is a natural consequence of prefix completion: every failed rollout contributes at most one failed segment but may contribute several successful prefix segments.

### 9.3 Leakage-free parent split

Parent rollouts were split before segment extraction:

| Split | Parent rollouts | Segments | Success segments | Failure segments |
|---|---:|---:|---:|---:|
| Outer training | 100 | 292 | 236 | 56 |
| Held-out test | 50 | 134 | 105 | 29 |

All segments from one parent remain in exactly one split. Inner CV also uses parent-rollout groups. The outer test set was not used for hyperparameter selection.

## 10. Subtask-SAFE training and retrospective evaluation

### 10.1 Inner cross-validation

Grid root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/rldx1_subtask_safe_cv_2gpu_20260803_222020
```

All 810 fits completed: 270 configurations across two model families and three folds.

| Model | Horizon selector | Diffusion selector | Learning rate | Regularization | Inner ROC |
|---|---|---|---:|---:|---:|
| MLP | concat-2 | concat-2 | 1e-3 | 1e-3 | **0.749 ± 0.037** |
| LSTM | concat-2 | 1.0 | 3e-5 | 1e-2 | 0.722 ± 0.009 |

### 10.2 Final refits

Final root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/rldx1_subtask_safe_final_2gpu_20260804_150002
```

All six final refits completed. The inverse-frequency class weights were `[5.1228, 1.2321]` because only 56 of the 292 training segments were failures.

| Model | Held-out ROC-AUC | Held-out PRC-AUC | Inner-CV ROC |
|---|---:|---:|---:|
| MLP | **0.734 ± 0.002** | **0.433 ± 0.012** | 0.749 ± 0.037 |
| LSTM | 0.601 ± 0.041 | 0.328 ± 0.056 | 0.722 ± 0.009 |

These are retrospective full-segment metrics: a detector may use all inferences collected before the segment completes or times out. They show that the independent detector ranks full failed versus successful segments, but do not establish an online early-warning capability.

## 11. Parent-aware causal controls and calibration

### 11.1 Task/subtask normalization and hierarchical bootstrap

Evaluation that respects task/subtask identity and parent-rollout dependence changes the conclusion:

| Model/method | ROC-AUC | Average precision |
|---|---:|---:|
| MLP raw SAFE | 0.734 ± 0.002 | 0.446 ± 0.013 |
| MLP training-only task/subtask-z SAFE | 0.561 ± 0.025 | 0.251 ± 0.010 |
| LSTM raw SAFE | 0.601 ± 0.041 | 0.345 ± 0.053 |
| LSTM training-only task/subtask-z SAFE | 0.463 ± 0.047 | 0.237 ± 0.025 |
| Task-conditioned elapsed time | 0.742 | 0.431 |

Parent-rollout hierarchical bootstrap estimates were:

- MLP SAFE: mean ROC 0.558, 95% CI [0.428, 0.676];
- LSTM SAFE: mean ROC 0.466, 95% CI [0.317, 0.625];
- elapsed time: mean ROC 0.740, 95% CI [0.646, 0.827].

The raw MLP ROC is inflated by between-stage score offsets and stage-specific failure prevalence. Training-only z-normalization removes this cross-stage shortcut but cannot fix weak within-stage ranking.

### 11.2 Causal prefix evaluation

For each active segment, scores were truncated after the first \(k\) genuine policy inferences. The eligible sample count falls with \(k\), so later-prefix results refer to progressively longer segments.

| Prefix inferences | Active segments | MLP SAFE ROC | LSTM SAFE ROC | Elapsed-time ROC |
|---:|---:|---:|---:|---:|
| 1 | 134 | 0.389 | 0.340 | **0.742** |
| 2 | 127 | 0.347 | 0.363 | **0.723** |
| 4 | 120 | 0.330 | 0.380 | **0.704** |
| 8 | 102 | 0.322 | 0.397 | **0.685** |
| 16 | 68 | 0.374 | 0.463 | **0.668** |
| 32 | 25 | 0.426 | 0.478 | **0.614** |

No segment contained 64 genuine policy inferences. The current SAFE score is below chance at the earliest prefixes, whereas elapsed time remains informative. This is the most deployment-relevant negative result in the study.

### 11.3 Parent-disjoint conformal calibration

Calibration used 15 parent rollouts and evaluation used 35 different parent rollouts. Reference and nonconformity calibration parents were also disjoint. Failure segments were excluded from success-only conformal fitting.

| Method | Alpha | TPR | FPR | Balanced accuracy | Lead environment steps among detections |
|---|---:|---:|---:|---:|---:|
| SAFE | 0.05 | 0.030 | 0.040 | 0.495 | 1516.0 |
| SAFE | 0.10 | 0.061 | 0.080 | 0.491 | 1522.0 |
| SAFE | 0.15 | 0.121 | 0.119 | 0.501 | 1661.0 |
| SAFE | 0.20 | 0.167 | 0.204 | 0.481 | 1642.9 |
| Elapsed time | 0.05 | 0.409 | 0.090 | 0.660 | 1758.2 |
| Elapsed time | 0.10 | 0.500 | 0.149 | 0.675 | 1715.3 |
| Elapsed time | 0.15 | 0.500 | 0.149 | 0.675 | 1715.3 |
| Elapsed time | 0.20 | 0.591 | 0.239 | 0.676 | 1645.8 |

Lead time is conditional on detected failures and must be read together with TPR. At \(\alpha=0.15\), SAFE's apparently large lead applies to only 12.1% of failures; balanced accuracy is essentially chance.

### 11.4 Scientific interpretation

Subtask decomposition corrected the semantic label problem and produced a valid, leakage-free dataset. It did **not** establish that current RLDX action features predict active-subtask failure early enough for intervention. The retrospective MLP result mainly reflects information accumulated late in the segment plus cross-stage score structure. The correct next step is not a larger repetition of the same experiment; it is to improve the causal representation and sampling design.

## 12. Figure and result artifact index

### 12.1 π0 official grid report

Root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_pi0_official_grid_2gpu_20260716_153526/final_report
```

Files:

```text
conformal_tradeoff.png
detection_times_seed0.png
per_task_balanced_accuracy.png
roc_pr_seed0.png
score_trajectories_threshold.png
selected_auc.png
selected_conformal_summary.csv
final_report.json
```

### 12.2 π0 leakage-free all-seen figures

Root:

```text
<PI0_INNER_CV_ROOT>/final_refits/result_plots
```

Files, each generated as PNG and PDF:

```text
test_auc_summary.{png,pdf}
per_seed_test_auc.{png,pdf}
cv_to_test_gap.{png,pdf}
safe_vs_duration_baseline.{png,pdf}
test_score_trajectories_seed0.{png,pdf}
```

Supporting files:

```text
per_seed_metrics.csv
summary_metrics.csv
plot_manifest.json
```

### 12.3 π0 score-overlay videos

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_all5_seen_7x7_train_3x3_test_20260717_010030/lstm_seed0/score_videos/*.mp4
```

### 12.4 RLDX rollout-level and natural-rate figures

Root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_25x25_final_2gpu_20260730_104400_calibration_20260730_113807
```

Figures:

```text
conformal_tradeoff.png
per_task_balanced_accuracy.png
```

Configured video output path:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_25x25_final_2gpu_20260730_104400_calibration_20260730_113807/videos_indep_seed0_alpha0p15/*.mp4
```

Natural-rate figures:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_natural_rate_screen_20260730_161528/detailed_analysis/roc_aggregation_comparison.{png,pdf}
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_natural_rate_screen_20260730_161528/detailed_analysis/per_task_roc_indep.{png,pdf}
```

### 12.5 Atomic/composite calibration figures

Mixed detector:

```text
<SAFE_TYPE_ROOT>/mixed_calibration/atomic/conformal_tradeoff.png
<SAFE_TYPE_ROOT>/mixed_calibration/atomic/per_task_balanced_accuracy.png
<SAFE_TYPE_ROOT>/mixed_calibration/composite/conformal_tradeoff.png
<SAFE_TYPE_ROOT>/mixed_calibration/composite/per_task_balanced_accuracy.png
```

Type-specific detectors:

```text
<SAFE_TYPE_ROOT>/split_calibration/atomic/conformal_tradeoff.png
<SAFE_TYPE_ROOT>/split_calibration/atomic/per_task_balanced_accuracy.png
<SAFE_TYPE_ROOT>/split_calibration/composite/conformal_tradeoff.png
<SAFE_TYPE_ROOT>/split_calibration/composite/per_task_balanced_accuracy.png
```

The confirmed ablation root is:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/safe_rldx1_25x25_type_ablation_20260730_115819
```

### 12.6 Subtask-SAFE figures and videos

Final figures:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/rldx1_subtask_safe_final_2gpu_20260804_150002/figures/
```

Parent-aware causal analysis:

```text
.../parent_causal_analysis/per_subtask_support.{png,pdf}
.../parent_causal_analysis/per_subtask_roc.{png,pdf}
.../parent_causal_analysis/causal_prefix_roc.{png,pdf}
.../parent_causal_analysis/parent_bootstrap_roc.{png,pdf}
```

Parent-disjoint calibration:

```text
.../parent_calibration/conformal_tradeoff.png
.../parent_calibration/per_task_balanced_accuracy.png
.../parent_calibration/detection_events.csv
```

The artifact bundle contains 64 MP4 files, including parent-calibrated overlays under `parent_calibration/indep_seed0/alpha_0p15/parent_videos/`.

## 13. Xiaomi-Robotics-1 original SAFE experiments

### 13.1 Scope and collection lineage

All Xiaomi experiments in this section use the original binary SAFE target:
eventual official RoboCasa task success or natural failure. They do not use
subtask labels, progress predicates, recovery annotations, or failure-onset
supervision.

The first collection covered ten tasks chosen near the policy's 50% success
region:

- atomic: `CoffeeSetupMug`, `TurnOnMicrowave`, `TurnOffStove`,
  `CloseBlenderLid`, and `SlideDishwasherRack`;
- composite: `DeliverStraw`, `ArrangeTea`, `WashFruitColander`,
  `GarnishPancake`, and `WaffleReheat`.

The first 500-rollout batch retained every rollout and contained 266 successes
and 234 failures. Additional natural rollouts were collected only until every
task had at least 25 examples of both outcomes, while still retaining every
rollout. This produced a 692-rollout ten-task pool with 419 successes and 273
failures. A deterministic seed-0 selection then created the balanced
500-rollout training export with exactly 25 successes and 25 failures per task:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_safe_balanced_25x25_seed0_20260807_122434/official_safe_25x25_seed0
```

The export contained 40,160 policy records, loaded with the pinned official
SAFE loader, and preserved Xiaomi feature provenance separately from the
loader-compatibility field named `pre_velocity`.

### 13.2 Initial ten-task detector results

The complete 810-fit grid searched three horizon selectors, three diffusion
selectors, five learning rates, three regularization values, and three inner
folds. The outer split remained untouched during selection.

| Model | Horizon | Diffusion | Learning rate | Regularization | Inner ROC-AUC |
|---|---|---|---:|---:|---:|
| MLP | 1.0 | 1.0 | 1e-4 | 1e-3 | 0.5345 ± 0.0085 |
| LSTM | concat-2 | 1.0 | 1e-3 | 1e-2 | 0.5947 ± 0.0288 |

Three final refits used a common 340-rollout training pool and a fixed
160-rollout outer test with eight successes and eight failures per task.

| Model | Outer ROC-AUC | Outer AUPRC | Completed-duration ROC-AUC |
|---|---:|---:|---:|
| MLP | 0.5170 ± 0.0012 | not retained in the terminal summary | 0.7431 |
| LSTM | 0.5499 ± 0.0241 | not retained in the terminal summary | 0.7431 |

Training-only task normalization and functional conformal evaluation at the
preselected `alpha=0.15` yielded:

| Model | ROC-AUC | AUPRC | TPR | FPR | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| MLP | 0.6156 ± 0.0316 | 0.7222 ± 0.0209 | 0.1042 ± 0.0664 | 0.1067 ± 0.0618 | 0.4988 ± 0.0140 |
| LSTM | 0.5489 ± 0.0313 | 0.6724 ± 0.0180 | 0.3042 ± 0.0948 | 0.2600 ± 0.0566 | 0.5221 ± 0.0383 |

The elapsed-time baseline had balanced accuracy 0.5 in this calibrated
evaluation view, but completed rollout duration remained a strong retrospective
ranker. This discrepancy motivated explicit causal duration controls.

### 13.3 Initial duration-controlled and causal-prefix studies

Two original-SAFE-only transformations were evaluated without Subtask-SAFE:

- `matched_success_length` paired and truncated failures to successful rollout
  lengths within task and split;
- `fixed_landmark_0p25` retained rollouts still active at a training-derived
  25% task landmark.

| Treatment | Model | Selected configuration `(h,d,lr,reg)` | Outer ROC-AUC | Outer AUPRC | Duration ROC-AUC |
|---|---|---|---:|---:|---:|
| Matched success length | MLP | `(1.0,1.0,3e-4,1e-3)` | 0.5456 ± 0.0029 | 0.5689 ± 0.0024 | 0.5000 |
| Matched success length | LSTM | `(concat-2,0.0,1e-3,1e-1)` | 0.4733 ± 0.0275 | 0.5039 ± 0.0162 | 0.5000 |
| Fixed landmark 0.25 | MLP | `(1.0,1.0,3e-4,1e-3)` | 0.5002 ± 0.0039 | 0.5519 ± 0.0069 | 0.4709 |
| Fixed landmark 0.25 | LSTM | `(0.0,1.0,1e-5,1e-2)` | 0.5298 ± 0.0060 | 0.5423 ± 0.0092 | 0.4709 |

The natural-duration hybrid analysis used only causal current-prefix SAFE,
task identity, current inference index, and training-derived time risk. On the
160-rollout outer test, the MLP event-level comparison was:

| Detector | ROC-AUC | TPR | FPR | Balanced accuracy | Mean detected-failure fraction |
|---|---:|---:|---:|---:|---:|
| Time only | 0.9844 | 1.0000 | 0.0500 | 0.9750 | 0.7645 |
| SAFE only | 1.0000 | 0.9875 | 0.0000 | 0.9938 ± 0.0051 | 0.9043 ± 0.0007 |
| SAFE + time + task | 1.0000 | 0.9792 ± 0.0212 | 0.0000 | 0.9896 ± 0.0106 | 0.9187 ± 0.0144 |

The LSTM SAFE-only and hybrid ROC-AUC values were 0.9097 ± 0.0436 and
0.9218 ± 0.0375, but their false-positive rates were 0.1500 ± 0.0568
and 0.1292 ± 0.0482. High event AUC did not imply early recovery utility:
the time-only detector usually alarmed earlier.

An outcome-only causal-prefix residual experiment therefore optimized fixed
10%, 25%, 50%, and 75% prefixes and selected thresholds at a 5% validation FPR
target. Its event-level results were:

| Detector | ROC-AUC | TPR | FPR | Balanced accuracy | Detection fraction |
|---|---:|---:|---:|---:|---:|
| Prefix SAFE | 0.7583 ± 0.0146 | 0.2417 | 0.0667 | 0.5875 ± 0.0184 | 0.5307 |
| Residual SAFE + time | 0.9840 ± 0.0066 | 0.9667 | 0.0292 | 0.9688 ± 0.0051 | 0.7794 |
| Time only | 0.9869 | 1.0000 | 0.0250 | 0.9875 | 0.7946 |

Prefix SAFE's task-macro ROC-AUC was 0.6104 at 10%, 0.6082 at 25%,
0.5260 at 50%, and 0.7153 at 75%; the last value had very sparse at-risk
support. A staged early-SAFE/late-time cascade then achieved event ROC-AUC
0.9661, TPR 1.0, FPR 0.0458, and adjusted detection fraction 0.7901. The
time-only comparator achieved ROC-AUC 0.9869, FPR 0.0250, and adjusted
detection fraction 0.7946. Across three seeds and 240 failed predictions, the
early SAFE stage alarmed before time-only only twice and increased false
alarms. The original ten-task dataset was therefore insufficient for a useful
early cascade.

### 13.4 Official-result task selection and 39-task collection

The broader study used the official Xiaomi RoboCasa365 release results rather
than the earlier ten-episode pilot. The inclusive eligibility rule was an
official success count between 5 and 45 out of 50. This selected 39 tasks and
excluded tasks reported at 0/50 or 50/50. Every newly collected rollout was
retained; no online balancing or deletion was performed.

| Task | Type | Official success | Fresh success | Fresh failure |
|---|---|---:|---:|---:|
| CategorizeCondiments | composite | 5 | 7 | 43 |
| PrepareCoffee | composite | 8 | 9 | 41 |
| PortionHotDogs | composite | 10 | 8 | 42 |
| SearingMeat | composite | 13 | 15 | 35 |
| PackIdenticalLunches | composite | 14 | 14 | 36 |
| WeighIngredients | composite | 14 | 15 | 35 |
| BreadSelection | composite | 15 | 17 | 33 |
| CuttingToolSelection | composite | 17 | 22 | 28 |
| RecycleBottlesByType | composite | 17 | 9 | 41 |
| CloseBlenderLid | atomic | 18 | 17 | 33 |
| GetToastedBread | composite | 18 | 20 | 30 |
| MakeIceLemonade | composite | 19 | 19 | 31 |
| StirVegetables | composite | 19 | 24 | 26 |
| WaffleReheat | composite | 21 | 29 | 21 |
| ArrangeTea | composite | 23 | 27 | 23 |
| DeliverStraw | composite | 23 | 18 | 32 |
| CoffeeSetupMug | atomic | 25 | 20 | 30 |
| WashFruitColander | composite | 27 | 31 | 19 |
| TurnOnMicrowave | atomic | 28 | 33 | 17 |
| GarnishPancake | composite | 29 | 31 | 19 |
| ArrangeBreadBasket | composite | 30 | 25 | 25 |
| RinseSinkBasin | composite | 30 | 35 | 15 |
| TurnOffStove | atomic | 30 | 30 | 20 |
| SteamInMicrowave | composite | 32 | 35 | 15 |
| LoadDishwasher | composite | 36 | 36 | 14 |
| SetUpCuttingStation | composite | 36 | 32 | 18 |
| StackBowlsCabinet | composite | 36 | 40 | 10 |
| KettleBoiling | composite | 37 | 29 | 21 |
| PreSoakPan | composite | 38 | 38 | 12 |
| ScrubCuttingBoard | composite | 40 | 35 | 15 |
| SlideDishwasherRack | atomic | 40 | 42 | 8 |
| StoreLeftoversInBowl | composite | 42 | 43 | 7 |
| TurnOnSinkFaucet | atomic | 42 | 38 | 12 |
| WashLettuce | composite | 42 | 36 | 14 |
| PickPlaceCounterToCabinet | atomic | 43 | 44 | 6 |
| PickPlaceSinkToCounter | atomic | 44 | 49 | 1 |
| CloseToasterOvenDoor | atomic | 45 | 44 | 6 |
| OpenCabinet | atomic | 45 | 46 | 4 |
| PickPlaceCounterToStove | atomic | 45 | 44 | 6 |

Collection root:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_safe_official_5to45_50each_20260808_215640
```

The two resumable shards produced exactly 1,950 unique valid rollouts:
1,106 successes and 844 failures. Deep validation found no missing artifacts,
duplicate rollout IDs, duplicate task/seed/reset identities, or official-loader
incompatibility. Twelve retained historical error records documented recovered
attempts but did not invalidate any of the 1,950 accepted rollouts. The two
shards occupied approximately 36 GiB.

### 13.5 Pooling, deterministic selection, and fixed split

The 1,950 fresh rollouts were merged with the complete earlier 692-rollout
ten-task pool. The resulting immutable source pool contained 2,642 rollouts:

```text
/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_safe_pooled_39tasks_50each_seed0_20260812_221815/source_pool_all_2642
```

A deterministic within-task, within-outcome seed-0 selection retained exactly
50 rollouts per task while preserving the source pool's natural outcome rate as
closely as possible. The selected 1,950-rollout export contained 1,105
successes and 845 failures and passed the official SAFE loader.

The fixed seen-task split required at least four examples of each outcome:
three for inner-fold support and one untouched outer-test example. The selected
`PickPlaceSinkToCounter` group had 49 successes and one failure and was excluded
from model development but preserved in the immutable export. The final split
was therefore:

| Split property | Value |
|---|---:|
| Included tasks | 38 |
| Excluded tasks | 1 |
| Outer training rollouts | 1,824 |
| Outer training successes | 1,018 |
| Outer training failures | 806 |
| Outer test rollouts | 76 |
| Outer test successes | 38 |
| Outer test failures | 38 |
| Held-out support per task | 1 success, 1 failure |

All six final refits used identical train and test identities. This split is
adequate for pooled and equal-task ranking metrics, but not for per-task
conformal calibration or precise per-task performance estimates.

### 13.6 Hyperparameter selection and final refits

The full grid used the pinned official SAFE objective with official
inverse-frequency class weighting:

- models: independent MLP and LSTM;
- horizon selectors: `0.0`, `1.0`, `concat-2`;
- diffusion selectors: `0.0`, `1.0`, `concat-2`;
- learning rates: `1e-5`, `3e-5`, `1e-4`, `3e-4`, `1e-3`;
- regularization: `1e-3`, `1e-2`, `1e-1`;
- inner folds: 3;
- epochs per fit: 1,000;
- configurations: 270;
- completed fits: 810/810;
- failure records: 0;
- outer test used for selection: false.

| Model | Horizon | Diffusion | Learning rate | Regularization | Inner ROC-AUC |
|---|---|---|---:|---:|---:|
| MLP | 1.0 | 1.0 | 1e-4 | 1e-3 | 0.6542 ± 0.0245 |
| LSTM | concat-2 | concat-2 | 3e-4 | 1e-1 | 0.7415 ± 0.0123 |

Each selected configuration was refitted on all 1,824 outer-training rollouts
with seeds 0, 1, and 2 and then evaluated once on the common 76-rollout test.

| Model | Seed ROC-AUC values | Raw ROC-AUC | Official AUPRC | Duration ROC-AUC |
|---|---|---:|---:|---:|
| MLP | 0.5440, 0.5433, 0.5087 | 0.5320 ± 0.0165 | 0.5108 ± 0.0069 | 0.8040 |
| LSTM | 0.6226, 0.6399, 0.6226 | 0.6283 ± 0.0082 | 0.6329 ± 0.0234 | 0.8040 |

The inner-to-outer gaps were approximately 0.122 for the MLP and 0.113 for the
LSTM. Completed duration remained substantially more discriminative than raw
SAFE.

Final-refit root:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_pooled_39tasks_final_refits_20260814_231508
```

### 13.7 Training-only normalization and task-type comparison

For every detector seed and task, the maximum early SAFE score was centered and
scaled using the 1,824-rollout training split only. The frozen transformation
was then applied to all 76 outer-test rollouts. This is a post-hoc ranking
diagnostic, not a test-fitted threshold.

| Model | Raw ROC-AUC | Task-z ROC-AUC | Raw average precision | Task-z average precision | Macro task ROC-AUC |
|---|---:|---:|---:|---:|---:|
| MLP | 0.5320 ± 0.0165 | **0.6924 ± 0.0133** | 0.5282 | **0.7334** | **0.6886** |
| LSTM | 0.6283 ± 0.0082 | 0.6457 ± 0.0168 | 0.6400 | 0.6632 | 0.6404 |

Normalization reversed the raw model ordering: the MLP was the strongest
within-task detector, while the raw LSTM benefited more from between-task score
offsets.

The included set contained ten atomic and 28 composite tasks. Each contributed
one held-out success and failure.

| Model | Type | Tasks | Raw ROC-AUC | Task-z ROC-AUC | Task-z AP | Macro task ROC-AUC | Duration ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP | Atomic | 10 | 0.5733 ± 0.0450 | **0.7300 ± 0.0535** | **0.7917 ± 0.0531** | 0.6000 | 1.0000 |
| MLP | Composite | 28 | 0.5393 ± 0.0175 | **0.6801 ± 0.0250** | **0.7286 ± 0.0152** | **0.7202 ± 0.0168** | 0.9120 |
| LSTM | Atomic | 10 | 0.6267 ± 0.0981 | 0.6300 ± 0.1225 | 0.7267 ± 0.0978 | 0.6333 ± 0.1247 | 1.0000 |
| LSTM | Composite | 28 | 0.6599 ± 0.0266 | 0.6505 ± 0.0208 | 0.6471 ± 0.0189 | 0.6429 ± 0.0505 | 0.9120 |

For the MLP, atomic minus composite task-z pooled ROC-AUC was
`+0.0499 ± 0.0781`, but the equal-task macro difference was
`-0.1202 ± 0.0168`. A 20,000-sample task bootstrap gave a 95% interval
`[-0.4155, +0.1655]`. For the LSTM, the macro difference was
`-0.0095 ± 0.1662`, with bootstrap interval `[-0.2643, +0.2452]`.
Neither architecture therefore supports a reliable atomic-versus-composite
performance difference at the present test size.

### 13.8 Causal online replay on the broad test

The frozen score trajectories were replayed one inference at a time. Available
inputs were limited to the current SAFE prefix, current inference index, known
task identity, training-derived task horizon, and whether the rollout remained
active. Forbidden inputs were final rollout duration, future SAFE scores,
subtask progress, and failure onset.

The 1,824-rollout source training split was divided deterministically before
prefix expansion:

- meta-fit: 1,748 rollouts;
- threshold validation: 76 rollouts, one per task and outcome;
- fixed outer test: 76 rollouts;
- split seed: 0;
- model seeds: 0, 1, 2;
- hybrid regularization candidates: `0.01`, `0.1`, `1.0`, `10.0`;
- landmarks: 10%, 25%, 50%, 75%, and 100%;
- threshold rule: maximum validation balanced accuracy.

`time_only` was a monotone task-conditioned estimate of eventual failure among
training rollouts still active at the current inference. `safe_only` used the
running maximum SAFE score with meta-fit task normalization.
`safe_time_task` used causal SAFE, elapsed risk, progress, and task identity.

| Model | Detector | Event ROC-AUC | TPR | FPR | Balanced accuracy | Detected-failure fraction | Miss-adjusted fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP | Time only | 0.9647 | 0.8421 | 0.0526 | 0.8947 | 0.7863 | 0.8200 |
| MLP | SAFE only | 0.9513 ± 0.0109 | **0.9825 ± 0.0248** | 0.1316 | 0.9254 ± 0.0124 | **0.6623 ± 0.0206** | **0.6677 ± 0.0282** |
| MLP | SAFE + time + task | **0.9675 ± 0.0069** | 0.9649 ± 0.0248 | 0.0526 | **0.9561 ± 0.0124** | 0.8423 ± 0.0125 | 0.8481 ± 0.0084 |
| LSTM | Time only | 0.9647 | 0.8421 | 0.0526 | 0.8947 | 0.7863 | 0.8200 |
| LSTM | SAFE only | 0.8142 ± 0.0322 | 0.8947 ± 0.0568 | 0.3772 ± 0.0541 | 0.7588 ± 0.0328 | 0.3464 ± 0.0315 | 0.4152 ± 0.0486 |
| LSTM | SAFE + time + task | 0.8633 ± 0.0190 | 0.7807 ± 0.0895 | 0.2719 ± 0.0447 | 0.7544 ± 0.0224 | 0.4345 ± 0.0402 | 0.5559 ± 0.0739 |

The MLP SAFE-only detector alarmed approximately 15.2 percentage points of
the task horizon earlier than time-only after missed failures were counted at
the end, but it also produced five false alarms among 38 successes instead of
two. The MLP hybrid matched time-only's false-positive rate and improved
recall, but alarmed later and was therefore less attractive for recovery.

At fixed causal landmarks, task-conditioned time gives the same score to the
active success and failure within a task and consequently has task-macro
ROC-AUC 0.5. SAFE showed earlier within-task ranking signal:

| Model | Detector | 10% pooled / macro | 25% pooled / macro | 50% pooled / macro | 75% pooled / macro |
|---|---|---:|---:|---:|---:|
| MLP | SAFE only | 0.502 / **0.671** | 0.552 / **0.626** | 0.632 / 0.556 | 0.800 / 0.417 |
| MLP | SAFE + time + task | 0.519 / 0.536 | 0.502 / 0.505 | 0.523 / 0.556 | 0.208 / 0.417 |
| MLP | Time only | 0.513 / 0.500 | 0.494 / 0.500 | 0.518 / 0.500 | 0.145 / 0.500 |
| LSTM | SAFE only | 0.574 / **0.667** | 0.585 / **0.616** | 0.648 / 0.778 | 0.741 / 0.500 |
| LSTM | SAFE + time + task | 0.627 / 0.667 | 0.591 / 0.616 | 0.637 / 0.778 | 0.487 / 0.500 |
| LSTM | Time only | 0.513 / 0.500 | 0.494 / 0.500 | 0.518 / 0.500 | 0.145 / 0.500 |

Task-level AUC support across three seeds corresponded to approximately 37
tasks at 10%, 33 at 25%, nine at 50%, and only two at 75%. The 50% and 75%
values are therefore descriptive only. The preregistered next comparison should
focus on 10% and 25%.

Artifact roots:

```text
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_pooled_39tasks_final_refits_20260814_231508/task_normalized_outer_test
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_pooled_39tasks_final_refits_20260814_231508/task_normalized_outer_test/by_task_type
/gs/bs/tga-shinoda/felid/robocasa_checkpoints/safe/xr1_pooled_39tasks_final_refits_20260814_231508/causal_online_replay_20260815_003121
```

### 13.9 Xiaomi conclusion

The broad Xiaomi experiment changes the conclusion drawn from the initial
ten-task study. The initial data did not support a useful early SAFE cascade;
the 38-task development set does contain early within-task signal. The
training-normalized MLP is the best SAFE candidate even though the raw LSTM has
higher pooled ROC-AUC. At 10% and 25% of task horizon, MLP SAFE ranks failures
above successes more often than task-conditioned time within task. However,
the current global threshold obtains earlier and more complete detection by
accepting a 13.2% false-positive rate. A matched-5%-FPR prospective evaluation
is required before claiming operational superiority over elapsed time.

The online replay is explicitly post-hoc because the 76-rollout outer test had
already been opened by the raw, normalized, and task-type analyses. Runtime
bundles are deployable engineering artifacts, but their reported test metrics
must not be treated as a new confirmatory result.

## 14. Overall interpretation

### 14.1 What worked

- Correct official-style policy features can be captured from π0 and an architecturally analogous RLDX action stream.
- The collection/export pipeline is reproducible and loader-compatible.
- Leakage-free inner CV and fixed held-out tests are operational.
- RLDX MLP features contain useful within-task failure information.
- More data improved RLDX MLP discrimination and reduced uncertainty.
- Training-only task normalization recovered a substantial amount of signal hidden by task-specific offsets.
- Conformal calibration produced low-FPR operating points with explicitly separated train, calibration, and evaluation sets.
- Score-over-video tooling makes detection timing inspectable.
- Natural-language ordered subtask segmentation and parent-rollout-grouped splitting are operational.
- Subtask-SAFE has a complete official-loader-compatible export and reproducible 810-fit selection protocol.
- Parent-aware bootstrap, causal-prefix evaluation, and parent-disjoint calibration prevent an optimistic retrospective metric from being mistaken for deployable performance.
- Xiaomi-Robotics-1 feature capture, resumable multi-task collection, official-loader export, pooled selection, and six final runs with three seeds per architecture are operational over 39 tasks.
- Training-only task normalization exposed substantially stronger Xiaomi MLP discrimination than raw pooled evaluation.
- The broad Xiaomi MLP contains early within-task signal at 10% and 25% causal landmarks where task-conditioned elapsed time alone cannot rank the two outcomes.

### 14.2 What did not work

- π0 results on 100 rollouts were unstable and did not yield a useful conformal operating point.
- RLDX raw pooled scores remained only modestly above chance.
- The LSTM did not outperform the independent MLP in the larger RLDX experiment.
- A single global threshold was inadequate without task-aware normalization.
- Rollout duration was a severe leakage source in naïve termination-time evaluation.
- Separate atomic-only and composite-only training did not improve matched performance.
- Composite failure detection was weak and late.
- Natural prevalence and inverse-frequency loss weighting did not materially improve discrimination.
- Subtask-SAFE's raw segment ROC did not survive task/subtask normalization and parent-aware resampling.
- RLDX Subtask-SAFE did not beat the elapsed-time baseline at any causal prefix or calibrated operating point.
- Xiaomi SAFE has not yet beaten time-only at a matched false-positive rate in a prospective test.
- The Xiaomi global MLP threshold obtained early high recall by accepting a 13.2% false-positive rate, and the low-FPR SAFE/time hybrid alarmed later than time-only.

### 14.3 Scientific conclusion

The present evidence supports the following hierarchy of conclusions:

1. **Strongly supported:** atomic failure prediction is materially easier than composite failure prediction in the current RLDX SAFE setup.
2. **Supported:** task-dependent score offsets harm pooled evaluation and calibration.
3. **Supported:** the independent MLP is preferable to the LSTM for these RLDX features and sample sizes.
4. **Rejected by the ablation:** simply training separate atomic and composite global detectors solves the problem.
5. **Supported:** semantic subtask decomposition fixes label validity and enables clean parent-grouped evaluation.
6. **Rejected for the current representation:** subtask decomposition alone yields a reliable early-warning detector.
7. **Strongly supported:** retrospective full-segment ROC is insufficient; causal-prefix, elapsed-time controls, and parent-aware uncertainty are mandatory.
8. **Supported for Xiaomi:** broad training-only task-normalized MLP scores contain early within-task failure information at 10% and 25% of task horizon.
9. **Not yet established for Xiaomi:** that early signal provides better recovery utility than task-conditioned time at the same prospective false-alarm budget.

## 15. Recommended next experiments

### 15.1 Preserve the validated baselines

Do not replace the raw SAFE result. Preserve:

- the rollout-level success/failure target;
- the 500-rollout RLDX official export;
- the fixed outer split;
- the selected mixed MLP;
- raw, task-z, macro-task, conformal, duration, and detection-time metrics.

Preserve both the 500-rollout RLDX baseline and the 426-segment Subtask-SAFE export. Future claims should always include raw, task/subtask-normalized, macro, parent-bootstrap, causal-prefix, conformal, and elapsed-time results.

### 15.2 Collect for failure-stage support

Do not collect an equal number of additional parent rollouts blindly. Collect toward deficient active failure stages, while retaining natural successes. Stages with fewer than 10 test failures should remain exploratory or be merged only when semantically defensible.

### 15.3 Improve the causal representation

Evaluate:

- stage-conditioned heads or learned task/subtask embeddings instead of post-hoc z-normalization;
- a discrete-time hazard objective with explicit exposure time;
- short causal windows sampled at matched within-subtask progress;
- state/proprioception and progress features alongside the action-token representation;
- leave-one-layout or leave-one-seed-family validation to test robustness.

Any candidate must be compared against task/subtask-conditioned elapsed time and evaluated at inference 1, 2, 4, 8, and 16—not only after the complete segment.

### 15.4 Required comparisons

Use the same composite held-out identities and compare:

- mixed raw rollout-level SAFE;
- composite-only raw rollout-level SAFE;
- the current Subtask-SAFE MLP;
- task/subtask-conditioned or hierarchical Subtask-SAFE;
- episode-duration and time-within-subtask baselines;
- oracle stage boundaries versus boundaries available online.

Report:

- ROC-AUC and PRC-AUC per subtask and macro averaged;
- conformal TPR/FPR at \(\alpha \in \{0.05,0.10,0.15,0.20\}\);
- detection time relative to active-subtask duration and whole-rollout duration;
- lead time before the first irreversible or terminal subtask failure;
- calibration coverage by task and subtask;
- score-over-video overlays with stage boundaries.

### 15.5 Data recommendation

The current 150-parent dataset is enough to reject the claim that decomposition alone solves early warning, but not to train well-powered stage-specific heads. A practical target is:

- at least 25 successful stage transitions per stage;
- at least 25 failures attributed to each stage used for evaluation;
- a minimum of 10–15 held-out examples of each class per evaluated stage after all splitting.

If a stage is rare, merge scientifically equivalent stages or treat it as exploratory rather than leaking test examples into training.

### 15.6 Prospective Xiaomi matched-FPR shadow evaluation

The next Xiaomi experiment must be frozen before collecting or inspecting any
new test outcomes. Its purpose is not another retrospective AUC comparison; it
is to determine whether normalized MLP SAFE alarms materially earlier than a
causal task-conditioned elapsed-time detector at the same false-positive
budget.

#### Frozen detector candidates

Use only the already selected MLP configuration and preserve all three final
model seeds. Do not select a seed from the opened 76-rollout test. At each real
XR-1 policy inference:

1. calculate each seed's current SAFE score;
2. apply that seed's training-only task normalization;
3. take the running maximum independently for each seed;
4. average the three normalized running maxima to form the MLP ensemble score.

Compare three preregistered detectors:

- `time_only`: task-conditioned causal survival risk fitted from the original
  outer-training trajectories;
- `safe_only`: the three-seed normalized MLP ensemble;
- `staged_safe_time`: SAFE eligible from rollout start through the 25%
  training-derived task landmark, followed by a time-only fallback eligible
  from 50% onward.

The previous unrestricted `safe_time_task` logistic hybrid remains a secondary
diagnostic because it improved final classification but alarmed later than
time-only. The LSTM is also secondary because its early alarms were accompanied
by an unacceptable false-positive rate.

#### Independent calibration collection

Collect a new calibration batch before the prospective test. Use new,
non-overlapping environment/reset seed ranges and retain every rollout,
including quota overshoot and errors. Target at least three successes and three
failures for each of the 38 included tasks: 228 primary calibration rollouts.
The target is outcome coverage, not a deployment-prevalence estimate. The
complete sequential stream is retained, but quota-based stopping means its
class ratio is only a stopping-cost and sensitivity diagnostic, not an
unbiased prevalence estimate.

Do not refit the SAFE networks on this batch. Fit only:

- the final task normalization if the training-frozen transformation is being
  checked for drift; the primary analysis must keep the original
  transformation unchanged;
- global event thresholds;
- the staged detector's SAFE and time thresholds.

Select every threshold under the same empirical calibration constraint:
`FPR <= 0.05`. Among feasible thresholds, maximize failure recall; break ties
by lower miss-adjusted detection fraction and then by the more conservative
threshold. With 114 successful calibration rollouts, FPR resolution is about
0.88 percentage points. Record the entire feasible threshold curve rather than
only the winner.

For the staged detector, search threshold pairs jointly under the same overall
5% rollout-level FPR cap. A rollout is a false positive if either the early SAFE
stage or late time stage alarms. No task, task-type, landmark, threshold, or
fallback time may be changed after prospective test collection begins.

#### Prospective shadow test

After calibration is frozen, collect a disjoint shadow-test batch with at least
ten successes and ten failures per task: 760 primary outcome-quota rollouts.
Continue keeping all attempts so quota cost, errors, and seed coverage remain
auditable. Evaluate the detector causally at every recorded inference using
only the prefix available then; deterministic post-collection replay is valid
because thresholds are already frozen and no detector may stop, retry, recover,
or otherwise alter the Xiaomi policy. Shadow mode preserves the
counterfactual terminal outcome needed to score an early alarm.

For every genuine policy inference, record:

- task and rollout identity;
- environment step and policy-inference index;
- three per-seed raw SAFE scores;
- three normalized running maxima and their ensemble mean;
- causal time-only risk;
- each detector's threshold state and first crossing;
- final unmodified task outcome and full horizon;
- video aligned to inference indices.

Calibration and prospective-test manifests must be disjoint by rollout ID,
environment seed, reset seed, and collection seed family. The original
training, opened outer test, new calibration, and new prospective test must be
four explicitly named provenance groups.

#### Primary endpoints and success criteria

The primary population is all 38 tasks pooled with one vote per rollout. Report
atomic and composite strata as secondary views. At the frozen operating point,
report:

- rollout-level TPR and FPR with Wilson 95% intervals;
- miss-adjusted first-detection fraction, with missed failures assigned 1.0;
- recall by 10%, 25%, and 50% of the training-derived task horizon;
- paired counts of failures where SAFE is earlier, tied, later, or missed
  relative to time-only;
- task-macro causal ROC-AUC at 10% and 25%;
- task- and rollout-bootstrap 95% intervals for detector differences;
- calibration-to-test FPR drift;
- results separately for atomic and composite tasks.

The staged detector should be considered a successful recovery candidate only
if all of the following hold prospectively:

1. observed rollout-level FPR is at most 5%, or its Wilson interval remains
   compatible with the calibration target;
2. TPR is not more than five percentage points below time-only;
3. miss-adjusted detection fraction is at least 0.10 lower than time-only;
4. at least 25% of all failures are detected by the 25% landmark;
5. paired task/rollout bootstrap favors earlier staged detection without a
   material increase in false alarms.

If SAFE-only is earlier but violates the FPR constraint, report that negative
result rather than retuning on the test. If the staged detector passes, freeze
its runtime bundle and proceed to a separately powered intervention experiment
that randomizes shadow versus detector-triggered recovery. Do not estimate
recovery benefit from the same shadow dataset.

Implementation entry points added for this protocol:

- `robocasa.recovery.safe.score_seen_checkpoint_external` applies one frozen
  final-refit checkpoint to a new official SAFE export without training or
  calibration and preserves full causal score trajectories plus seed/reset
  provenance;
- `robocasa.recovery.safe.run_prospective_matched_fpr calibrate` fits only the
  frozen training-derived normalization, time-risk curves, and matched-FPR
  thresholds, then writes checksummed runtime artifacts;
- `robocasa.recovery.safe.run_prospective_matched_fpr evaluate` verifies those
  checksums and calibration/test disjointness, applies the frozen thresholds,
  and reports Wilson intervals, landmark recall, paired timing, task-type
  strata, and task/rollout bootstrap differences.

## 16. Limitations

- The π0 and RLDX experiments use different policies, feature shapes, task sets, and data volumes; their absolute metrics are not direct model-family comparisons.
- Most primary rollout-level datasets are deliberately balanced; the separate natural-rate experiment only approximates deployment prevalence from the finite collected outcome pool.
- Only three detector seeds were used.
- π0 held-out sets were very small.
- Per-task RLDX uncertainty remains nontrivial even with 16 test rollouts per task.
- The broad Xiaomi outer test contains only one success and one failure per task; its per-task metrics and atomic/composite differences have wide sampling uncertainty.
- Xiaomi threshold validation reused score trajectories from the detector's source training pool, so its operating-point results are exploratory even though the outer test was not used for threshold selection.
- The Xiaomi causal online result is replay on completed trajectories, not a prospective live shadow test or an intervention experiment.
- Xiaomi task-macro causal support falls from approximately 37 tasks at 10% to nine at 50% and two at 75% because naturally completed successes leave the at-risk set.
- Task normalization assumes the deployed task identity is known.
- Conformal validity is conditional on exchangeability assumptions that can be violated by task, seed, layout, or policy drift.
- Composite stage decomposition is a hypothesis motivated by the results, not yet an experimentally demonstrated improvement.
- Segment classes are naturally imbalanced and failure support is sparse for many stages.
- Segments within one parent are dependent; only the final analyses account for this with grouped splits and hierarchical bootstrap.
- Causal-prefix sample support falls to 25 segments at 32 inferences and zero at 64.
- The elapsed-time baseline is task/subtask-conditioned; its strength may partly reflect policy-specific stage difficulty and horizons, but that is exactly the shortcut a deployable detector must beat.
- Subtask progress uses environment predicates unavailable to a real robot unless separately estimated online.

## 17. Reproducibility pointers

Primary documentation:

- `docs/safe_robocasa.md`
- `docs/safe_rldx1.md`
- `docs/safe_xiaomi_robotics_1.md`
- `docs/safe_abot_m05.md` on `codex/safe-abot-m05`

Key entry points:

- `robocasa.recovery.safe.collect_atomic_rollouts`
- `robocasa.recovery.safe.collect_mixed_rollouts`
- `robocasa.recovery.safe.validate_atomic_dataset`
- `robocasa.recovery.safe.export_official_safe`
- `robocasa.recovery.safe.run_seen_cv_grid`
- `robocasa.recovery.safe.summarize_seen_cv`
- `robocasa.recovery.safe.train_seen_tasks`
- `robocasa.recovery.safe.summarize_seen_tasks`
- `robocasa.recovery.safe.calibrate_seen_tasks`
- `robocasa.recovery.safe.plot_seen_results`
- `robocasa.recovery.safe.render_score_videos`
- `robocasa.recovery.safe.audit_subtask_safe_dataset`
- `robocasa.recovery.safe.export_subtask_official_safe`
- `robocasa.recovery.safe.analyze_subtask_parent_causal`
- `robocasa.recovery.safe.calibrate_subtask_parent`
- `robocasa.recovery.safe.build_seen_task_split`
- `robocasa.recovery.safe.analyze_time_safe_hybrid`
- `robocasa.recovery.safe.run_causal_prefix_residual`
- `robocasa.recovery.safe.run_early_safe_time_cascade`

Key implementation commits:

```text
8e44a98  Integrate SAFE atomic rollout pipeline
fa9bd74  Fix SAFE action parity and split provenance
0fbc50d  Use official horizons for SAFE atomic tasks
7918a9f  Match official OpenPI rollout protocol
5cbba67  Export balanced SAFE atomic datasets
f649fa3  Validate official SAFE loader compatibility
92697e5  Evaluate official SAFE models offline
1d17b28  Decouple SAFE evaluation from RoboSuite
76fa346  Report selected official SAFE results
223c94f  Train SAFE on all seen tasks and render videos
2da4950  Select SAFE hyperparameters with training-only CV
276a308  Plot final SAFE seen-task results
2562c0c  Add SAFE feature collection for RLDX-1
5e02344  Fix RLDX direct-policy language batching
110e89a  Generalize SAFE seen-task training for RLDX
1c92c1b  Add task-normalized SAFE calibration
b67e43c  Add SAFE task-type ablation support
99801d0  Add SAFE feature collection for ABot-M0.5
2c482e1  Add parent-aware Subtask-SAFE causal evaluation
181332a5 Adapt SAFE collection for Xiaomi-Robotics-1
335f61f2 Fix Xiaomi SAFE model patch application
60293fae Preserve Xiaomi seed batches when merging SAFE data
a9caaa60 Add normalized SAFE duration diagnostics
106e820a Add causal online SAFE prefix protocols
f2ab81ba Add causal SAFE time hybrid evaluation
d6a70b01 Add causal prefix residual SAFE training
7e78daa2 Add staged early SAFE time cascade
5e785a88 Plan Xiaomi SAFE collection from official results
5be7e439 Close SAFE policy sockets between tasks
5190c96e Prepare pooled Xiaomi SAFE training data
```

The legacy report bundle contains 20,857 archive entries and all archived
checksums passed. It predates the Xiaomi 39-task experiment. Xiaomi's durable
cluster roots listed in Section 13 are therefore the authoritative locations
for its manifests, metrics, runtime bundles, figures, and score trajectories;
they should be added to the next packaged report snapshot.
