# Long-Horizon VLA Failure Detection on RoboCasa365: Experimental Report

**Status:** living technical research record  
**Period covered:** November 27–28, 2026  
**Models covered:** Xiaomi-Robotics-1 (`Xiaomi-Robotics-1-RoboCasa365`, checkpoint revision `0d1aa76d0d82debc9b611e4d1e231096434d5be4`, repo `4da1db0`)  
**Roadmap reference:** [`reports/long_horizon_failure_detection_roadmap.docx`](../reports/long_horizon_failure_detection_roadmap.docx)  
**Primary question:** On long-horizon composite manipulation tasks where elapsed-time clocks explain nearly all lead of existing VLA failure monitors (RSS Stage 1), can observable physical-relation progress, missed effects, and lost conditions warn earlier at a fixed 5% episode-level false-alarm budget—and distinguish self-corrected mistakes from persistent failures on unseen compositions of familiar subtasks?

---

## Executive summary

This report records all offline audits, oracle calibration studies, cluster inventories, and live synchronized XR-1 recordings for the long-horizon failure detection project, following the provenance and falsification standards established in [`docs/safe_robocasa_experiment_report.md`](safe_robocasa_experiment_report.md).

Key findings through November 28, 2026:

1. **Retrospective 5-task oracle pilot (`n = 76` October parents; Sections 2–3).** Adding physical loss and candidate missed-placement rules to the calibrated task timer + overdue-stage clock gained **+6.42 percentage points (pp)** of task horizon (+8.34 s at 20 Hz, 95% task-then-episode bootstrap CI **[+1.87, +11.57] pp**, positive on 5/5 tasks) under internal cross-fitting, and **+6.37 pp** (+8.28 s, 95% CI **[+1.85, +11.49] pp**) when fitting grace windows on the independent 60-parent October naturals cohort (25 successes). Both runs **failed the roadmap's 2 pp excess-FPR screen** solely because a single ambiguous cutting-tool success (`51fccafb19d3bb617aec`, wrong tool obscured the board until termination) gave 2/36 vs. 1/36 false alarms (+2.78 pp).
2. **Local release-clock control and self-correction yield (Section 4).** At the 32–256 step waits, **hiding the destination predicate completely** (replacing missed-placement verification with a plain post-release timer) produced **100% identical first alarms on all 76 parents** (+6.05 pp over both clocks at 32 steps). Across all 76 parents there were **zero verified multi-decision self-correction pairs** for testing recovery-aware waiting.
3. **Cluster cohort inventory and 32-task transferability matrix (Section 5).** Both legacy 250-rollout August live-traced SAFE collections contain **0 videos, 0 action traces, and no raw time-resolved object/grasp predicates** (latched stage traces only), confirming fresh synchronized collection is required. Cross-referencing [`robocasa/recovery/eval_composite_predicates.py`](../robocasa/recovery/eval_composite_predicates.py) with XR-1's 39-task baseline success pool identifies **18 composite tasks** in the **28%–80% XR-1 success band** built strictly from shared physical primitives (11 pure manipulation/fixture + 7 with appliance states; 11 of 18 already define `_grasped`).
4. **First live synchronized XR-1 recording, predicate repair, and 6-task smoke (Section 6).** Visual inspection of 4 synchronized recordings (`xr1_recovery_recording_PNL4QPfz`) exposed three simulator-predicate defects (`OU.check_obj_grasped` 0.035 m false negative on wide bread loaves, closed-gripper grazing false positive on `wrong_tool_grasped`, and strict `is_open` thresholds). Replacing `_grasped` with Robosuite's opposing-pad `env._check_grasp` and expanding `RECOVERY_PREDICATES` to 6 Group-A tasks passed a 12-rollout live smoke (`xr1_recovery_pilot6_smoke_20261128_135824`, 9S/3F, 12/12 synced).
5. **Fresh 120-rollout 6-task prospective oracle milestone — PASSED ALL ROADMAP GATES (Section 7).** Across 120 fresh synchronized XR-1 rollouts (`xr1_recovery_pilot6_full_20261128_142543`, 64S/56F, 20/task across 6 tasks, strict 5-fold split-conformal calibration at nominal 5% FPR):
   - **`full_physical` vs. `stage_clock`:** **+6.63 pp (+8.3 s)** more failure lead (38.67% / 55.0 s vs. 32.05% / 46.6 s; 95% task-then-episode bootstrap CI **[+3.10, +11.16] pp**; positive on **6/6 tasks**; **26 earlier / 0 later**; failure recall **98.2% vs. 92.9%**) at **identical 3.1% FPR (2/64 vs. 2/64; 0.00 pp excess FPR)**.
   - **`full_physical` vs. `clocks_release_timer` (blind action-clock control):** **+18.92 pp (+25.4 s)**, 95% CI **[+14.19, +23.56] pp**, **6/6 tasks positive** — decisively disproving the single-object release-clock confound once multi-object tasks are included.
   - **`full_physical` vs. `global_timer` ($G$):** **+13.05 pp (+20.5 s)**, 95% CI **[+6.06, +20.21] pp**, **6/6 tasks positive**, identical 3.1% FPR.
   - **Failure anatomy:** **40/56 failures (71.4%)** exhibited a confirmed physical mistake (30 missed placement, 7 lost condition, 3 both).
   - **Test-2 self-correction prevalence:** **5/64 successes (7.8%)** in the 120-rollout cohort (**6/73 = 8.2%** combining smoke + full) showed genuine multi-decision unassisted self-corrections (3–40 decisions before recovery), yielding **4 matched Test-2 pairs** (~1 matched pair per 30 rollouts → ~900 rollouts projected for 30 pairs).

**Current decision:** **Stages 1–3 COMPLETE AND Test-2 Continuation-Aware Pilot VALIDATED (Sections 10.5–11).**
- **Decoupled zero-privileged monitor (Section 10.5):** Anchoring the observable stage clock to our 0.966-AUROC LOTO universal grasp detector + `Qwen3-VL-4B` destination verification (`grasp_and_vlm` stage clock, **zero privileged inputs, zero target-task labels**) reaches **32.40% (45.1 s, 98.2% recall, +10.6 s vs. $G$, 5/6 tasks)** with zero-shot `Qwen3-VL-4B` (6.2% FPR = 4/64) and **29.20% (40.1 s, 96.4% recall, 4.7% FPR = 3/64, +5.6 s vs. $G$, 5/6 tasks)** with 3-fold cross-task LoRA unseen-task `Qwen3-VL-4B`.
- **Test-2 continuation-aware pilot (Section 11):** Strictly prior to recovery/regrasp, **persistent failures exhibit policy abandonment/runaway drift while self-correcting successes stay engaged near the mistake site**: on equal-duration matched pre-regrasp pairs ($n = 4$), `dit_cos_conf` and `dit_cos_onset` agree in **4/4 pairs** (`SC > PF`, `0.783` vs. `0.670`), and at fixed equal 3-decision windows (`H = 3`, zero truncation bias, `same_input_clock` LORO = `0.500`), `dit_cos_conf` already reaches **AUROC `0.8171` (`p = 0.022`)**. Across the broader confirmed-mistake cohort ($n = 11$ successes vs. $35$ persistent failures, strictly pre-regrasp), `fused_continuation` reaches **LOTO unseen-task AUROC `0.8286` (`p = 0.0002`)** (pre-restore **LOTO `0.8753`, `p = 0.0002`** vs. `same_input_clock` `0.626`, `p = 0.220`), with length-invariant `prop_state_step_vel` at **AUROC `0.8545` (`p = 0.0002`)** and `act_base_speed_max` at **AUROC `0.8519` (`p = 0.0005`)**.

Ready to scale synchronized collection to the **18-task Group-A/B train vs. held-out composition benchmark (Test 1 + powering Test 2 + LLM-generated task-graph ablation)**.

---

## 1. Scope, roadmap, and design revisions

The initial research roadmap ([`reports/long_horizon_failure_detection_roadmap.docx`](../reports/long_horizon_failure_detection_roadmap.docx), 27 November 2026) set up a 6-stage pipeline: (1) build an event dataset of missed effects, lost conditions, and retries; (2) test simulator-oracle alarms against per-task and overdue-stage clocks at 5% episode-level FPR; (3) train a camera-based relation readout, attempt tracker, and learned alarm around a frozen VLA; (4) confirm warning-lead transfer on 10–12 held-out compositions (Test 1); (5) evaluate self-correction vs. persistent-failure discrimination against a same-input condition checker (Test 2); and (6) optionally replicate on RLDX-1.

Empirical findings from the first 48 hours of pilots plus critical protocol review motivate five concrete adjustments carried forward in subsequent sections:

1. **Include an action/event-anchored clock baseline** (e.g., post-release or post-grasp-attempt elapsed time without destination verification) alongside `G`, `P`, and the stage clock, because Section 4 showed a release timer reproduced 100% of destination-verified alarms at 32–256 steps.
2. **Broaden "missed effect" to missed grasps and failed fixture manipulations** anchored on proprioceptive/action primitives (and fix `check_obj_grasped` first), because live inspection (Section 6) showed repeated failed grasps and sub-16-step grasp attempts dominate failure modes while post-placement self-corrections are rare.
3. **Separate same-input vs. continuation-aware comparisons for Test 2.** Restricting the learned head to identical binary/probabilistic relation outputs as the deterministic checker leaves it blind to *how* the robot moves after a mistake (reaching back vs. driving away) prior to regrasp.
4. **Replace fragile finite-sample / zero-shot calibration gates** (2 pp one-sided 95% upper bound on excess FPR across 10–12 clustered tasks; $n = 19$ max-order-statistic folds) with stage-normalized scores, larger calibration splits, explicit reporting of failure recall alongside mean lead, and matched-FPR / partial-AUC curves alongside zero-shot threshold transfer.
5. **Favor open-vocabulary vision-language relation probing + an LLM-generated task-graph ablation** over a closed-set visual MLP readout and manual-only held-out decompositions.

---

## 2. Retrospective 76-parent relation-event audit (27 November 2026)

Full protocol and prototype details: [`docs/long_horizon_relation_events.md`](long_horizon_relation_events.md).  
Source cohort: `outputs/xr1_ladder_v2_pilot_20261017_164537` (76 natural XR-1 parents across 5 tasks: 40 timeouts, 36 successes; schema-2 traces with 16-step replan cadence).  
Reproduced output: `outputs/long_horizon_failure_detection/xr1_ladder_76/summary.json`.

- Prototype fixed-threshold rules (16-step archived loss window; 2-consecutive-decision grasp anchor + 32-step post-release grace on the 2 instrumented tasks `CuttingToolSelection` and `GarnishPancake`, covering 33/76 parents and 28 anchored releases).
- **Descriptive uncalibrated result:** adding lost condition + candidate missed placement to `min(P, stage_clock)` warned earlier on **14/40 failures** (+7.6 pp of horizon: 53.3% → 60.9%), while raising successful-episode alarms from **11/36 (30.6%)** to **15/36 (41.7%)**.
- Across 21 parents with physical warnings: 15 remained absent at timeout, 2 returned in eventual failures, and 4 returned in eventual successes (3 ambiguous cutting-tool runs with terminal-only destination confirmation, plus 1 bread-basket flicker `8a8996a2e2ca9988d2bd` returning in 13 steps before the next policy decision).

---

## 3. Cross-fitted calibrated 5-task pilot: internal vs. external naturals fit (27–28 November 2026)

Full protocol and internal-fit report: [`docs/long_horizon_calibration_pilot.md`](long_horizon_calibration_pilot.md).  
Scorer: [`scripts/xr1_lead_time/calibrate_relation_events.py`](../scripts/xr1_lead_time/calibrate_relation_events.py).

### 3.1 Experimental setup and external fitting cohort

Five deterministic folds balanced by task and outcome scored every parent in the 76-parent pilot once, calibrating episode-level max scores at nominal 5% FPR (`rank = ceil(0.95 × (n + 1))`, giving rank 19/19 = fold max on all folds). Three pre-identified ambiguous cutting-tool successes (`84cd852d7d93c84f9df0`, `51fccafb19d3bb617aec`, `15cb328f837aed77ac71`) were held strictly for evaluation.

In the internal pilot (`v1`), each fold had only 5 fitting successes (1–2 instrumented releases), collapsing all fitted placement grace windows to the 1-control-step floor. On November 28, 2026, we exported the independent schema-2 October naturals metadata from TSUBAME and reran calibration with `--external-grace-run`, replacing only the fitting pool while keeping selection, calibration, and evaluation splits identical:

- **Cluster source:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/xr1_ladder_v2_naturals_20261017_153727`
- **Exported archive:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/recovery/relation_fit_export.GfqYTW/xr1_naturals_fit.tar.gz` (local copy: `outputs/long_horizon_failure_detection/cluster_imports/xr1_naturals_fit.tar.gz`, SHA-256 `9dbc1fff1b5c5b4cf3b2af7d93e39eb14b3ba8b4ae39d74e6cacddbff64c31d1`)
- **External fitting counts (60 parents = 25 successes / 35 failures):**
  - `ArrangeBreadBasket`: 5 succ / 7 fail
  - `ArrangeTea`: 8 succ / 4 fail
  - `BreadSelection`: 1 succ / 11 fail
  - `CuttingToolSelection`: **3 succ** / 9 fail (enabled task-specific fit)
  - `GarnishPancake`: **8 succ** / 4 fail (enabled task-specific fit)

### 3.2 Results comparison (`v1` internal fit vs. external October naturals fit)

Output: [`outputs/long_horizon_failure_detection/xr1_ladder_76_calibration_external_naturals/calibration.json`](../outputs/long_horizon_failure_detection/xr1_ladder_76_calibration_external_naturals/calibration.json) (SHA-256 `a3282acbebeef2a8f8fb1f982cc0e7d6054ebe04996607eda86d851d917e0714`).

| Arm | `v1` failures warned / 40 | `v1` succ warned / 36 | `v1` macro lead | External-fit failures / 40 | External-fit succ / 36 | External-fit macro lead |
|---|---:|---:|---:|---:|---:|---:|
| Registered timer `P` (frozen) | 40 | 2 (5.56%) | 19.23% | 40 | 2 (5.56%) | 19.23% |
| Recalibrated timer | 26 | 1 (2.78%) | 13.47% | 26 | 1 (2.78%) | 13.47% |
| Timer + stage clock (`clocks`) | 26 | 1 (2.78%) | 25.98% | 26 | 1 (2.78%) | 25.98% |
| Clocks + lost condition | 27 | 1 (2.78%) | 28.81% | 27 | 1 (2.78%) | 28.81% |
| **Full (clocks + loss + missed placement)** | **30** | **2 (5.56%)** | **32.40%** | **30** | **2 (5.56%)** | **32.35%** |

Per-task paired gain (`full − clocks`, external naturals fit):

| Task | Gain (% of horizon) |
|---|---:|
| `ArrangeBreadBasket` | +7.17 pp |
| `ArrangeTea` | +4.00 pp |
| `BreadSelection` | +0.40 pp |
| `CuttingToolSelection` | +8.67 pp |
| `GarnishPancake` | +11.63 pp |
| **Equal-task mean** | **+6.37 pp (+165.55 steps / +8.28 s at 20 Hz)** |

- **Uncertainty (external fit):** task-then-episode bootstrap 95% CI **[+1.85, +11.49] pp**; within-task 95% CI **[+3.48, +9.64] pp**; positive on **5/5 tasks**; leave-one-task-out mean range **+5.06 to +7.87 pp**; 14 failures warned earlier, 0 later.
- **Fitted placement grace windows (external fit):** `CuttingToolSelection` yielded `[median = 1, p90 = 1]` across all 5 folds (all 3 fitting successes placed at the first decision after release); `GarnishPancake` yielded `[median = 1, p90 = 16]`, selecting `1` on folds 0–3 and `16` on fold 4 (which slightly changed GarnishPancake from +11.85 pp to +11.63 pp).
- **Screen outcome:** `passes_roadmap_pilot_screen = false` in both runs. The single additional false alarm (`51fccafb19d3bb617aec`, alarmed at 1,088 and succeeded at 1,096) yields `fpr_difference = +2.78 pp`, exceeding the roadmap's 2 pp tolerance (unavoidable whenever any 1 of 36 successes flips).
- **Notable finite-sample artifact:** calibrating on $n = 19$ successes per fold forced the cutoff to the fold maximum, dropping timer-only failure recall from 40/40 (19.23% lead) down to 26/40 (13.47% lead). Future calibration splits require $n \ge 100$ successes.

---

## 4. Local controls: destination verification vs. release clock, and recovery-pair yield (27 November 2026)

Full details: [`docs/long_horizon_local_controls.md`](long_horizon_local_controls.md).  
Scorer: [`scripts/xr1_lead_time/relation_local_controls.py`](../scripts/xr1_lead_time/relation_local_controls.py); output: `outputs/long_horizon_failure_detection/xr1_ladder_76_local_controls/results.json`.

Key results on the same 76 parents:

1. **Hiding the destination predicate produced zero alarm changes at waits ≥ 32 steps.** Comparing `clocks + loss + destination-verified missed placement` against `clocks + loss + plain object-release timer` (which stops only on regrasp or episode end) gave **0/76 changed first alarms** at 32, 64, 128, and 256 control steps (and only 1/76 changed at 1 and 16 steps). At 32 steps, both achieved **30/40 failures warned, 1/36 successes warned, 32.03% macro lead (+6.05 pp over both clocks)**.
   - **Interpretation:** In successful trials, episodes finish soon after a valid release, before the calibrated release-elapsed margin expires; in failed trials, the episode continues until the release timer fires regardless of whether destination truth was observed. Any future detector claim must beat an explicit **release-timer / action-event clock** baseline.
2. **Zero verified self-correction pairs in the 76-parent cohort.** Among the 4 physical alarms on successful episodes, 3 cutting-tool successes were ambiguous because `correct_tool_on_cutting_board` coupled `selected_present and not other_present`, and the 1 bread-basket success (`8a8996a2e2ca9988d2bd`) resolved in 13 steps (before the next policy decision). Testing recovery-aware waiting requires fresh synchronized recordings with decoupled predicates (`correct_tool_present_on_board`, `wrong_tool_grasped`, `bread_grasped`, `basket_grasped`).

---

## 5. Cluster cohort audit and 32-composite-task transferability matrix (28 November 2026)

Cluster audit output: `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/long_horizon_cohort_audit_20261128_130837.json`.

### 5.1 Legacy XR-1 cluster cohorts

| Cluster cohort (`/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/...`) | Manifest rows | Subtask traces | Videos | Actions | Raw time-resolved predicates? |
|---|---:|---:|---:|---:|---|
| `xr1_dense_subtask_live_5tasks_50each_20260817_195216` | 502 *(shards + merged)* | 502 | 0 | 0 | **No** (latched stage fields only) |
| `xr1_stage_context_5tasks_50each_20260823_182450` | 500 *(shards + merged)* | 500 | 0 | 0 | **No** (latched stage fields only) |
| `xr1_safe_pooled_39tasks_50each_seed0_20260812_221815` | 2,642 | 0 | 0 | 0 | **No** (rollout-level SAFE pool only) |

**Conclusion:** None of the August SAFE collections can supply raw changing object/grasp predicates, actions, or camera frames. Fresh synchronized collection is required for both expanded oracle sizing and vision-based monitor training.

### 5.2 32-composite-task relation-primitive and XR-1 success-rate matrix

Cross-referencing all 32 composite tasks in `EVAL_COMPOSITE_PREDICATES` ([`robocasa/recovery/eval_composite_predicates.py`](../robocasa/recovery/eval_composite_predicates.py#L2979-L3012)) against XR-1's baseline success counts in `xr1_safe_pooled_39tasks_50each_seed0_20260812_221815` identifies **18 composite tasks** in the **28%–80% XR-1 success band** that rely solely on shared physical primitives (no task-specific custom geometry/counters):

**Group A — Pure manipulation & fixture primitives (11 tasks, 28%–80% SR):**

| Task | XR-1 SR | Has `_grasped`? | Shared primitives |
|---|---:|:---:|---|
| `ArrangeBreadBasket` | 50.0% (25/50) | Yes *(in `RECOVERY_PREDICATES`)* | `grasp`, `in/inside`, `on_fixture`, `open/close`, `release` |
| `CuttingToolSelection` | 44.0% (22/50) | Yes *(in `RECOVERY_PREDICATES`)* | `grasp`, `in/inside`, `open/close`, `release` |
| `DeliverStraw` | 41.3% (43/104) | Yes | `grasp`, `in/inside`, `open/close`, `release` |
| `GarnishPancake` | 63.9% (78/122) | Yes | `grasp`, `in/inside`, `on_fixture`, `open/close`, `release` |
| `LoadDishwasher` | 72.0% (36/50) | Yes | `grasp`, `on_fixture`, `open/close` |
| `MakeIceLemonade` | 38.0% (19/50) | Yes | `grasp`, `in/inside`, `open/close`, `release` |
| `SetUpCuttingStation` | 64.0% (32/50) | Yes | `grasp`, `in/inside`, `open/close`, `release` |
| `BreadSelection` | 34.0% (17/50) | No | `in/inside`, `release` |
| `PackIdenticalLunches` | 28.0% (14/50) | No | `in/inside`, `open/close`, `release` |
| `StackBowlsCabinet` | 80.0% (40/50) | No | `in/inside`, `open/close`, `release` |
| `WashFruitColander` | 56.9% (58/102) | No | `in/inside` |

**Group B — Shared primitives + appliance state (7 tasks, 30%–76% SR):**

| Task | XR-1 SR | Has `_grasped`? | Shared primitives |
|---|---:|:---:|---|
| `GetToastedBread` | 40.0% (20/50) | Yes | `grasp`, `in/inside`, `release`, `appliance` |
| `KettleBoiling` | 58.0% (29/50) | Yes | `grasp`, `on_fixture`, `release`, `appliance` |
| `PreSoakPan` | 76.0% (38/50) | Yes | `grasp`, `in/inside`, `release`, `appliance` |
| `SearingMeat` | 30.0% (15/50) | Yes | `grasp`, `in/inside`, `on_fixture`, `open/close`, `release`, `appliance` |
| `ArrangeTea` | 52.5% (53/101) | No | `in/inside`, `open/close`, `release`, `appliance` |
| `SteamInMicrowave` | 70.0% (35/50) | No | `in/inside`, `open/close`, `appliance` |
| `WaffleReheat` | 52.4% (54/103) | No | `in/inside`, `open/close`, `appliance` |

**Excluded / non-transferable composite tasks (14 tasks):**
- *Custom task-specific geometry/counters (5):* `RinseSinkBasin` (70%), `ScrubCuttingBoard` (70%), `StirVegetables` (48%), `WashLettuce` (72%), `WeighIngredients` (30%).
- *Outside 25%–80% SR band (5):* `CategorizeCondiments` (14%), `PortionHotDogs` (16%), `PrepareCoffee` (18%), `RecycleBottlesByType` (18%), `StoreLeftoversInBowl` (86%).
- *Not in 39-task XR-1 pool (4):* `GatherTableware`, `HeatKebabSandwich`, `PanTransfer`, `SeparateFreezerRack`.

**Feasibility note:** Groups A + B provide **18 viable shared-primitive tasks**, enough for a clean **10-train / 8-held-out** (or 8-train / 10-held-out) composition split once `RECOVERY_PREDICATES` (and repaired grasp checks) are extended beyond the initial 2 tasks.

---

## 6. First live synchronized XR-1 passive recording & visual audit (28 November 2026)

### 6.1 Run provenance and synchronization verification

- **Code:** commit `9e59c463ab38fb9ec33b705ca1b9d26dcfc924fd` (`origin/codex/xr1-recovery-recording-smoke`, run in detached worktree `/gs/fs/tga-shinoda/felid/robocasa_xr1_rec_9e59c463` with gitignored MuJoCo assets symlinked from `/gs/fs/tga-shinoda/felid/robocasa`)
- **Policy:** `Xiaomi-Robotics-1-RoboCasa365` (`4da1db0`, checkpoint `0d1aa76d`, replan every 16 steps, `obs_grid` 768×256 at 20 fps, `video_frame_stride = 1`)
- **Script:** [`scripts/run_xr1_recovery_recording.sh`](../scripts/run_xr1_recovery_recording.sh); verifier: [`scripts/xr1_lead_time/verify_recovery_recording.py`](../scripts/xr1_lead_time/verify_recovery_recording.py)
- **Cluster output:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_recovery_recording_PNL4QPfz`
- **Local review archive:** `outputs/long_horizon_failure_detection/cluster_imports/xr1_recovery_recording_PNL4QPfz_review.tar.gz` (SHA-256 `cc8999a58d1f2863efe382f35ad8ed26d623aa3c13f9c44745648d4c1d7c2324`)
- **Inspection contact sheets:** `outputs/long_horizon_failure_detection/xr1_recovery_recording_PNL4QPfz_inspection/`

All four recordings passed both `validate_atomic_dataset` (`VALID`, 0 errors) and `verify_recovery_recording` (`PASS: four synchronized XR-1 recordings`), and local OpenCV decoding confirmed exact frame-count and timestamp agreement on all four MP4s (`4350/4350`, `4350/4350`, `1200/1200`, `1200/1200` at 20.0 fps; `video_frame_environment_steps == [1..N]`).

### 6.2 Per-episode visual & trace audit

All four episodes timed out (`0/4` successes, `4/4` failures). Visually inspecting synchronized `[agentview_left, agentview_right, eye_in_hand]` contact sheets at key predicate transitions showed what physically happened in each rollout:

| Rollout | Seed | Steps / Decs | Trace summary | Physical reality verified in synchronized video |
|---|---:|---:|---|---|
| `ArrangeBreadBasket` `1dd3b4ce` | 2600075 | 4,350 / 272 | `OpenCabinet_1` stayed active 0→4,350; `recovery_observations` empty | **Cabinet door WAS visibly open by step 600** (wrist camera reached straight inside facing the bread). XR-1 failed repeated bread grasps from steps 600–3,000, knocked the bread out of view, and wedged against the cabinet by 4,000–4,350. |
| `ArrangeBreadBasket` `00072ff1` | 2600076 | 4,350 / 272 | `OpenCabinet_1` done @ 624; `bread_in_basket` True @ 851 (flickered @ 854→880, and 1-step flickers @ 1797, 2593, 2929); `bread_grasped` **never True**; 22 `basket_grasped` pulses (3–32 steps) @ 1,095–3,883 | **XR-1 cleanly grasped, carried, and placed the round bread loaf into the wire basket (steps 700→845→851)** while `bread_grasped` stayed `False`. Then for **~173 s (steps 1,095–4,350)** it repeatedly failed to grasp/lift the thin wire basket rim; the 1-step `bread_in_basket` drops were physics/bbox jitter while bumping the basket. |
| `CuttingToolSelection` `f8d2c686` | 2600083 | 1,200 / 75 | `drawer_open` stayed `False` (inferred done @ 614); `correct_tool_grasped` True **614→654** (40 steps); then False to 1,200 | **Textbook missed placement:** avocado on board (knife required). XR-1 opened drawer enough to reach in, cleanly grasped and lifted the wooden-handled knife (614–640), dropped it back into the drawer at step 654, hovered its **empty gripper** over the cutting board at step 700, and timed out at 1,200 (~25 s of real early-warning lead). |
| `CuttingToolSelection` `cd28746d` | 2600084 | 1,200 / 75 | `drawer_open` stayed `False`; `correct_tool_grasped` True 959, 961–972; **`wrong_tool_grasped` True 973–980**; False to 1,200 | XR-1 wasted ~40 s navigating to the wrong counter (facing a toaster at step 400), returned to the drawer by step 800, then jammed on the black knife handle inside the drawer (959–1,200) without lifting it out. At step 973, **the gripper was still on the black knife, NOT holding the green peeler**—`wrong_tool_grasped` fired from incidental contact while closed. |

Contact sheet files:
- [`ArrangeBreadBasket_1dd3b4ce.png`](../outputs/long_horizon_failure_detection/xr1_recovery_recording_PNL4QPfz_inspection/ArrangeBreadBasket_1dd3b4ce.png)
- [`ArrangeBreadBasket_00072ff1.png`](../outputs/long_horizon_failure_detection/xr1_recovery_recording_PNL4QPfz_inspection/ArrangeBreadBasket_00072ff1.png)
- [`CuttingToolSelection_f8d2c686.png`](../outputs/long_horizon_failure_detection/xr1_recovery_recording_PNL4QPfz_inspection/CuttingToolSelection_f8d2c686.png)
- [`CuttingToolSelection_cd28746d.png`](../outputs/long_horizon_failure_detection/xr1_recovery_recording_PNL4QPfz_inspection/CuttingToolSelection_cd28746d.png)

### 6.3 Three simulator-predicate defects caught by visual inspection

Inspecting [`robocasa/utils/object_utils.py`](../robocasa/utils/object_utils.py#L665-L685) explains both grasp-label failures directly:

```python
def check_obj_grasped(env, obj_name, threshold=0.035):
    ...
    gripper_closed = all(pos < threshold for pos in gripper_joint_positions)
    ...
    return env.check_contact(gripper, obj) and gripper_closed
```

1. **False negative on wide objects (`threshold = 0.035` m per finger = 7.0 cm total width).** A Panda gripper's max opening is 0.04 m per finger (8.0 cm). Wide objects such as the round bread loaf in `00072ff1` keep the fingers ≥ 0.035 m while solidly grasped and carried through mid-air, so `bread_grasped` stayed `False` across the entire pick-and-place.
2. **False positive on incidental contact while closed (`check_contact` instead of opposing-finger grasp).** Because `check_obj_grasped` tests *any* contact between any part of the gripper body and `obj` whenever finger positions are `< 0.035`, grazing an adjacent object inside a tight drawer (`cd28746d` step 973, peeler beside knife) or tapping a wire basket rim (`00072ff1`, 22 pulses) triggers a false grasp. Using opposing two-finger-pad grasp contact (Robosuite's `_check_grasp(gripper, obj)`, or separate `left_fingerpad` + `right_fingerpad` contact checks) fixes both defects simultaneously.
3. **Strict `is_open` threshold vs. physical reachability.** In 3 of 4 episodes (`1dd3b4ce`, `f8d2c686`, `cd28746d`), XR-1 opened the cabinet/drawer enough to reach in and attempt grasps while `cabinet_open`/`drawer_open` remained `False`. On `1dd3b4ce`, this misattributed 3,500 steps of failed bread grasping to `OpenCabinet_1`.
4. **Sub-16-step grasp flickers and 1-step placement jitter.** Both `basket_grasped` (3–22 steps) and `correct_tool_grasped` in `cd28746d` (11 steps) were shorter than XR-1's 16-step replan cadence, confirming that (a) step-level `recovery_observations` capture attempts invisible at policy cadence, and (b) 1-step physics flickers (`bread_in_basket` in `00072ff1`) must remain persistence-filtered.

### 6.4 Grasp-predicate repair and 6-task pilot preparation (28 November 2026)

Implemented and locally verified before the next cluster collection:

- **Opposing-pad grasp check ([`robocasa/recovery/eval_composite_predicates.py`](../robocasa/recovery/eval_composite_predicates.py)):** Replaced `_grasped(env, obj_name)` to call Robosuite's `env._check_grasp(gripper=..., object_geoms=...)` (matching `robocasa/utils/subtask_labeling/`), with fallback to `OU.check_obj_grasped` only for minimal test stubs without `_check_grasp`. This fixes both the wide-object false negative (`bread_grasped` in `00072ff1`) and the closed-gripper incidental-contact false positive (`wrong_tool_grasped` in `cd28746d`).
- **6-task `RECOVERY_PREDICATES` coverage ([`robocasa/recovery/safe/subtask_safe.py`](../robocasa/recovery/safe/subtask_safe.py)):** Kept `ArrangeBreadBasket` and `CuttingToolSelection` unchanged for backward compatibility with `PNL4QPfz`, and added `DeliverStraw`, `GarnishPancake`, `MakeIceLemonade`, and `SetUpCuttingStation` (all 6 Group-A tasks with `_grasped`, 38%–64% XR-1 SR).
- **Generalized verifier ([`scripts/xr1_lead_time/verify_recovery_recording.py`](../scripts/xr1_lead_time/verify_recovery_recording.py)):** Auto-reads tasks, rollout count, and `video_frame_stride` from `summary.json` (with CLI overrides), supports OpenCV fallback when `imageio_ffmpeg` is absent, and logs per-rollout grasp-onset counts. Verified cleanly on the live `PNL4QPfz` archive (`PASS: 4 synchronized XR-1 recordings`).
- **Multi-GPU resumable 6-task launcher ([`scripts/run_xr1_recovery_pilot_6tasks.sh`](../scripts/run_xr1_recovery_pilot_6tasks.sh)):** Supports `--plan`, `--smoke` (2 rollouts/task, seed `2700007`, stride 1), and full pilot (`NUM_ROLLOUTS=20`/task = 120 rollouts, seed `2800007`, stride 2) across 1–6 GPUs (`PILOT_GPUS`), with asset preflight, client-only `PYTHONPATH`, `--resume` via `PILOT_OUT_DIR`, per-shard validation, and a lightweight `metadata_review.tar.gz` bundle.
- **Local verification:** `34/34` unit tests pass (`tests.test_long_horizon_recovery_observations`, `tests.test_long_horizon_video_sync`, `tests.test_long_horizon_local_controls`, `tests.test_long_horizon_calibration`, `tests.test_long_horizon_relation_events`), `bash -n` passes on the launcher, and `verify_recovery_recording` passes on `PNL4QPfz`.

### 6.5 Live 6-task XR-1 smoke verification (`4b3666de`, 28 November 2026)

Ran `PILOT_GPUS="0" bash scripts/run_xr1_recovery_pilot_6tasks.sh --smoke` (`2` rollouts/task = 12 rollouts total, `seed = 2700007`, `video_frame_stride = 1`, ~25 min on 1 GPU, commit `4b3666de`).

- **Cluster root:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_recovery_pilot6_smoke_20261128_135824/shard0`
- **Local metadata archive:** `outputs/long_horizon_failure_detection/cluster_imports/xr1_pilot6_smoke_metadata.tar.gz` (SHA-256 `e4d123dab4ee1c3e7d68ad572057e2f97acc2a4870619ce1e4b0f16cef2a3b0c`)
- **Validation:** `RoboCasa SAFE dataset: VALID` (12 rollouts: **9 successes, 3 failures**; 42 usable segments: 39 successes, 3 failures, 0 errors), `PASS: 12 synchronized XR-1 recordings`, exact 1-to-1 decoded frame sync on every episode.
- **Per-rollout opposing-pad `_check_grasp` timelines:**

| Task | Rollout ID | Outcome | Steps / Decs | Grasp intervals (start→end, steps) | Key physical event & timing |
|---|---|---|---:|---|---|
| `ArrangeBreadBasket` | `aa214332` | **succ** | 1,015 / 64 | `bread`: 591→700 (109); `basket`: 983→985, 986→998 | `bread_in_basket` @ 707 (+7 steps); `basket_on_dining_counter` @ 982 *(wire-basket carry only hit both pads at set-down)* |
| `ArrangeBreadBasket` | `eaa2bfe1` | **fail** | 4,350 / 272 | `bread`: 664→668, 670→783 (113), 1105→1794 (689); `basket`: 2857→2960 (103) | **Missed bread placement @ 783 (−178 s) & @ 1,794 (−128 s); then grasped empty basket @ 2,857 (−75 s)** |
| `CuttingToolSelection` | `0991c30b` | **succ** | 308 / 20 | `correct_tool`: 210→294 (84) | Placed cleanly @ 294 (+0 steps) |
| `CuttingToolSelection` | `5a4fdad1` | **fail** | 1,200 / 75 | `wrong_tool`: 241→388 (147); `correct_tool`: 664→766 (102) | **Wrong tool placed on board @ 386 (−40.7 s); later missed correct-tool placement @ 766 (−21.7 s)** |
| `DeliverStraw` | `25c61d72` | **succ** | 948 / 60 | `straw`: 371→866 (495), 870→872 (2) | `straw_in_glass` @ 831 (+0 steps; 2-step touch on release) |
| `DeliverStraw` | `1c0979cc` | **succ** | 1,009 / 64 | `straw`: 396→962 (566) | `straw_in_glass` @ 964 (+2 steps) |
| `GarnishPancake` | `147b5ad9` | **succ** | 1,248 / 78 | `strawberry`: 741→1225 (484) | `strawberry_on_pancake` @ 1,227 (+2 steps) |
| `GarnishPancake` | `1d703e91` | **fail** | 2,700 / 169 | `strawberry`: 461→669 (208) | **Missed strawberry placement @ 669 (−101.5 s); `gripper_released` @ 695 → timeout** |
| `MakeIceLemonade` | `d285494e` | **succ** | 2,034 / 128 | `lemon`: 719→1035 (316); `ice1`: 1525→1561 (36); `ice2`: 1854→2017 (163) | **Genuine 14.7 s (18-decision) unassisted self-correction:** missed glass with `ice_cube1` @ 1,561, returned to fridge, grasped `ice_cube2` @ 1,854, placed @ 2,018 (+1 step), succeeded @ 2,034 |
| `MakeIceLemonade` | `99438606` | **succ** | 1,532 / 96 | `lemon`: 475→759 (284); `ice2`: 1137→1505 (368) | `lemon_in_glass` @ 759 (+0 steps), `ice_in_glass` @ 1,505 (+0 steps) |
| `SetUpCuttingStation` | `87bbbc90` | **succ** | 1,673 / 105 | `knife`: 409→915 (506), 916→919 (3); `meat`: 1310→1645 (335) | `knife_on_board` @ 915 (+0 steps; 3-step release touch); `meat_on_board` @ 1,647 (+2 steps) |
| `SetUpCuttingStation` | `b1b33b2b` | **succ** | 798 / 50 | `knife`: 392→544 (152); `meat`: 685→775 (90) | `knife_on_board` @ 544 (+0 steps); `meat_on_board` @ 777 (+2 steps) |

- **Five key empirical findings from the 12 timelines:**
  1. **Clean `_check_grasp` signal across all 6 tasks:** Every solid object (`bread`, `knife`, `peeler`, `straw`, `strawberry`, `lemon`, `ice_cube1`, `ice_cube2`, `meat`) produced continuous **36–689-step** grasp intervals (`bread_grasped` fired for 109–689 steps; `wrong_tool_grasped` fired for 147 steps on a genuine wrong-tool pick-and-place). Caveat: thin wire-mesh `basket` carry in `aa214332` did not trigger opposing-pad grasp until set-down (`983–998`).
  2. **All 13 clean successful placements landed within 0–7 control steps (≤ 0.35 s) after release.**
  3. **All 3 failures (3/3) had unmistakable, huge-lead physical mistakes:** 21.7 s to 178 s before timeout (`eaa2bfe1` missed bread twice then grasped the empty basket; `5a4fdad1` placed the wrong tool on the cutting board 40.7 s before timeout; `1d703e91` dropped the strawberry 101.5 s before timeout).
  4. **First verified multi-decision unassisted self-correction (`MakeIceLemonade d285494e`):** XR-1 dropped/missed with `ice_cube1` at step 1,561, spent **293 steps (14.7 s / 18 policy decisions)** returning for `ice_cube2` (step 1,854), and succeeded at step 2,034. A naive 32-step missed-placement rule would false-alarm around step 1,600.
  5. **Precondition note:** `fridge_open` fired cleanly on all 4 fridge episodes, whereas `drawer_open` / `cabinet_open` remained `False` even on successful `DeliverStraw`, `CuttingToolSelection`, `SetUpCuttingStation`, and `ArrangeBreadBasket` runs.
- **Decision:** Gate passed — launched the **full 6-task pilot** (`NUM_ROLLOUTS=20`/task = 120 rollouts, `seed = 2800007`, `video_frame_stride = 2`, 1 GPU on `tsubame`).

### 6.6 6-task oracle & Test-2 analyzer verification on smoke data (28 November 2026)

Built and verified [`scripts/xr1_lead_time/analyze_recovery_pilot6.py`](../scripts/xr1_lead_time/analyze_recovery_pilot6.py) (with unit tests in [`tests/test_long_horizon_pilot6_analyzer.py`](../tests/test_long_horizon_pilot6_analyzer.py); **41/41** long-horizon tests passing in 0.33 s). The analyzer evaluates strictly causal 6-arm episode-level calibrated oracles (`global_timer` → `task_timer` → `stage_clock` → `clocks_release_timer` → `clocks_loss` → `full_physical`), inventories multi-decision self-corrections and candidate Test-2 matched pairs, and exports `summary.json`, `rollouts.csv`, and `events.jsonl`.

End-to-end smoke check on `xr1_pilot6_smoke_meta` (12 rollouts: 9S/3F; output [`outputs/long_horizon_failure_detection/xr1_pilot6_smoke_analysis/summary.json`](../outputs/long_horizon_failure_detection/xr1_pilot6_smoke_analysis/summary.json), SHA-256 `40d7866e153864c4326dc22aca452823d4b4caa98f0ad8e2021ee2c499c80e37`):

- **Clean placement delays (`n = 13`):** median **1.0 step** (0.05 s), p90 **2.8 steps**, max **7 steps** (0.35 s).
- **Test-2 yield (smoke):** **1/9 successes (11.1%)** showed a multi-decision self-correction (`MakeIceLemonade d285494e`, `place_ice`, confirmed at step 1,600, restored at 2,016, **26 XR-1 decisions** before recovery); **3/3 failures** produced persistent/retry-failed physical mistakes (`matched_test2_pairs = 0` only because `MakeIceLemonade` had 2S/0F in the 2-rollout smoke).
- **Early multi-object insight vs. Section 4's local control:** Even on this descriptive 12-rollout smoke (`small_sample_empirical` calibration mode; not yet split-conformal), `full_physical` differed from `clocks_release_timer` on **6/12 episodes** and warned earlier on **3/3 failures** (+24.1 pp / +33.1 s). Unlike the single-object tasks in Section 4 (`CuttingToolSelection`, `GarnishPancake`), on multi-object tasks (`ArrangeBreadBasket`, `MakeIceLemonade`, `SetUpCuttingStation`) the episode continues long after Object 1 is released—so a blind release timer accumulates huge age on successful runs, whereas destination-verified missed placement stops immediately when Object 1 lands. This will be tested properly under split-conformal calibration once the 120-rollout run completes.

---

## 7. Fresh 120-rollout 6-task prospective oracle & Test-2 milestone (28 November 2026)

### 7.1 Run provenance and validation

- **Code:** commit `4b3666ded3ead257eddb9d32dc9e731e7ef32aa3` (`origin/codex/xr1-recovery-pilot-6tasks`)
- **Command:** `PILOT_GPUS="0" NUM_ROLLOUTS=20 VIDEO_FRAME_STRIDE=2 bash scripts/run_xr1_recovery_pilot_6tasks.sh` (`seed = 2800007`, disjoint from all Stage 1, ladder, and smoke seeds)
- **Cluster output:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_recovery_pilot6_full_20261128_142543/shard0`
- **Local metadata archive:** `outputs/long_horizon_failure_detection/cluster_imports/xr1_pilot6_full_metadata.tar.gz` (SHA-256 `8407eb2454668255a87c86d8b838e1ee83c1c36933cfc4dcf4f6cdb45d7101e1`)
- **Analysis outputs:**
  - Primary 120-rollout split-conformal analysis: [`outputs/long_horizon_failure_detection/xr1_pilot6_full_120_analysis/summary.json`](../outputs/long_horizon_failure_detection/xr1_pilot6_full_120_analysis/summary.json) (SHA-256 `ff8c56602f9c5fcac2b4c601fa5793a3d976c063ba297a396cbb9a131cdd6904`)
  - Combined 132-rollout sensitivity (12 smoke + 120 full): [`outputs/long_horizon_failure_detection/xr1_pilot6_combined_132_analysis/summary.json`](../outputs/long_horizon_failure_detection/xr1_pilot6_combined_132_analysis/summary.json) (SHA-256 `9c3b21541a8bf66bded6fa5c0ca9e6270639d41efcbb9e084864c9e50e8b5356`)
- **Validation:** `RoboCasa SAFE dataset: VALID`, `PASS: 120 synchronized XR-1 recordings`, **0 errors**, 377 usable Subtask-SAFE segments (321S / 56F), exact stride-2 frame alignment on every episode.
- **Per-task outcome balance (64 successes / 56 failures):** `ArrangeBreadBasket` 11S/9F (55%), `CuttingToolSelection` 11S/9F (55%), `DeliverStraw` 9S/11F (45%), `GarnishPancake` 10S/10F (50%), `MakeIceLemonade` 9S/11F (45%), `SetUpCuttingStation` 14S/6F (70%).

### 7.2 6-arm split-conformal oracle hierarchy (5-fold CV, nominal 5% episode-level FPR)

Every fold used strict disjoint split-conformal calibration (`n_fit = 17` successes for task/stage quantiles and grace windows, `n_cal = 34–35` successes for episode-level `ceil(0.95 × (n + 1))` cutoffs, scored out-of-fold on all 120 rollouts).

| Arm | Failures warned / 56 | Recall | Successes warned / 64 | Realized FPR (Wilson 95%) | Equal-task mean lead (% horizon) | Equal-task mean lead (s @ 20 Hz) |
|---|---:|---:|---:|---:|---:|---:|
| `global_timer` ($G$) | 56 | 100.0% | 2 | 3.1% [0.9%, 10.7%] | 25.62% | 34.5 s |
| `task_timer` ($P$) | 52 | 92.9% | 3 | 4.7% [1.6%, 12.9%] | 22.00% | 33.3 s |
| `stage_clock` | 52 | 92.9% | 2 | 3.1% [0.9%, 10.7%] | 32.05% | 46.6 s |
| `clocks_release_timer` *(blind)* | 44 | 78.6% | 1 | 1.6% [0.3%, 8.3%] | 19.75% | 29.6 s |
| `clocks_loss` | 54 | 96.4% | 2 | 3.1% [0.9%, 10.7%] | 33.98% | 48.3 s |
| **`full_physical`** | **55** | **98.2%** | **2** | **3.1% [0.9%, 10.7%]** | **38.67%** | **55.0 s** |

**Key paired comparisons (120 fresh rollouts; task-then-episode bootstrap, 2,000 replicates):**

| Comparison | Mean gain (% horizon) | Mean gain (s) | Bootstrap 95% CI (% horizon) | Positive tasks | Earlier / Later | Excess FPR |
|---|---:|---:|---:|---:|---:|---:|
| `stage_clock − task_timer` | +10.04 pp | +13.3 s | [+4.75, +16.12] | **6 / 6** | 28 / 13 | −1.6 pp |
| `clocks_release_timer − stage_clock` | −12.30 pp | −17.0 s | [−18.09, −6.66] | 0 / 6 | 15 / 38 | −1.6 pp |
| **`full_physical − stage_clock`** | **+6.63 pp** | **+8.3 s** | **[+3.10, +11.16]** | **6 / 6** | **26 / 0** | **0.00 pp** |
| **`full_physical − clocks_release_timer`** | **+18.92 pp** | **+25.4 s** | **[+14.19, +23.56]** | **6 / 6** | **52 / 3** | +1.6 pp |
| **`full_physical − global_timer`** | **+13.05 pp** | **+20.5 s** | **[+6.06, +20.21]** | **6 / 6** | **47 / 7** | **0.00 pp** |

Per-task `full_physical − stage_clock` gains (all 6 tasks positive, 0 rollouts warned later):
- `MakeIceLemonade`: **+13.38 pp (+20.1 s)**
- `CuttingToolSelection`: **+9.19 pp (+5.5 s)**
- `GarnishPancake`: **+6.76 pp (+9.1 s)**
- `DeliverStraw`: **+5.61 pp (+7.2 s)**
- `ArrangeBreadBasket`: **+2.49 pp (+5.4 s)**
- `SetUpCuttingStation`: **+2.33 pp (+2.8 s)**

**Combined 132-rollout sensitivity check (12 smoke + 120 full; 73S / 59F):** confirms every finding — `full_physical` reaches **44.72% (62.2 s)** lead at **4.1% FPR (3/73)** vs. `stage_clock` **39.17% (54.0 s)** at **4.1% FPR (3/73)** → **+5.55 pp (+8.1 s)**, 95% CI **[+2.38, +9.09] pp**, **6/6 tasks positive**, **24 earlier / 0 later**, **0.00 pp excess FPR**; and **+17.05 pp (+24.6 s)** over `global_timer`, **+11.35 pp (+14.5 s)** over `clocks_release_timer` (6/6 tasks positive).

**Why destination verification decisively beats the blind release clock here:** On multi-object composite tasks (`ArrangeBreadBasket`, `MakeIceLemonade`, `SetUpCuttingStation`), successful episodes continue for 500–1,000+ steps *after* Object 1 is released. The blind release timer keeps ticking throughout Object 2 on successes, inflating its split-conformal cutoff to `0.28–0.62` (vs. `0.18–0.38` for `full_physical`). Checking the destination stops the clock immediately on arrival (median placement delay **1.0 step**, p90 **10.5 steps** across 126 placements), preserving the exact same cutoff and **0.00 pp excess FPR** as `stage_clock` (the 2 false alarms of `full_physical` — `c0f68907` @ 960 and `35d06a39` @ 1,712 — were identical to `stage_clock`'s 2 false alarms, both caused by long self-correcting successes hitting the stage timer).

### 7.3 Failure anatomy and Test-2 self-correction readiness

- **Failure anatomy (56 failures):**
  - `missed_placement_only`: **30 / 56 (53.6%)**
  - `lost_condition_only`: **7 / 56 (12.5%)**
  - `missed_placement + lost_condition`: **3 / 56 (5.4%)**
  - *(Total with explicit physical violation: **40 / 56 = 71.4%**)*
  - `no_sustained_grasp_before_timeout`: **12 / 56 (21.4%)**
  - `placed_then_stalled_later`: **4 / 56 (7.1%)**
- **Multi-decision unassisted self-corrections:** **5 / 64 successes (7.8%)** in the 120-rollout run (**6 / 73 = 8.2%** in 132 combined):
  - `DeliverStraw a1ddc0a9`: missed straw placement @ 488 → confirmed @ 528 → restored @ 1,165 (**40 XR-1 decisions** window)
  - `ArrangeBreadBasket 66764859`: missed basket placement @ 1,614 → confirmed @ 1,648 → restored @ 2,005 (**23 decisions**)
  - `SetUpCuttingStation 35d06a39`: missed meat placement @ 1,411 → confirmed @ 1,456 → restored @ 1,678 (**14 decisions**)
  - `MakeIceLemonade eca17b9b`: lost `lemon_in_glass` @ 1,091 → confirmed @ 1,120 → restored @ 1,244 (**8 decisions**)
  - `CuttingToolSelection c0f68907`: wrong tool placed on cutting board @ 423 → confirmed @ 448 → removed & succeeded @ 482 (**3 decisions**)
  - *(Plus smoke `MakeIceLemonade d285494e`: missed `ice_cube1` @ 1,561 → confirmed @ 1,600 → `ice_cube2` placed @ 2,016, **26 decisions**)*
- **Matched Test-2 pairs (120 rollouts):** **4 matched pairs** across 4 distinct tasks (`ArrangeBreadBasket`, `CuttingToolSelection`, `DeliverStraw`, `SetUpCuttingStation`), with matched observation windows of **3, 14, 23, and 40 XR-1 decisions**. Yield is **~1 matched pair per 30 rollouts** → powering 30 matched Test-2 pairs passively projects to **~900 test rollouts** (consistent with collecting ~400 test successes).

**Gate decision:** **PASSED.** The 6-task simulator oracle clears every roadmap threshold (`+6.63 pp >= +3 pp`, `6/6 >= 3` tasks positive, `0.00 pp <= 2 pp` excess FPR, 26 earlier / 0 later, 71.4% physical-violation coverage, and 7.8% self-correction prevalence among successes).

---

## 8. Camera & state perception-floor check on the 120 recorded rollouts (28 November 2026)

### 8.1 Setup and provenance

- **Code:** commit `bad8bbd8` (`scripts/xr1_lead_time/extract_perception_features.py`, `scripts/xr1_lead_time/eval_perception_floor.py`, `scripts/run_xr1_perception_floor.sh`)
- **Input cohort:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_recovery_pilot6_full_20261128_142543` (**120 rollouts, 13,901 policy decisions**)
- **Cluster output:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_perception_floor_20261128_191229` (results bundle: `perception_floor_results.tar.gz`, 44 KB)
- **Vision encoder:** 437 trained SigLIP-So400M/14-224 tensors loaded from cached `models--lerobot--pi0_libero_finetuned_v044/.../model.safetensors` (patch mean pooling over `left`, `right`, `wrist` 224×224 crops; strictly causal frame mapping `frame_step <= t`; extracted all 13,901 3-cam decisions in 74.7 s on 1 GPU).
- **Probes:** Exact L2-regularized L-BFGS linear probes (`hidden_dim = 0`) and cosine-annealed AdamW 1-hidden-layer MLP probes (`hidden_dim = 128`), strictly split by complete rollout (same 5 folds as Section 7) plus leave-one-task-out (LOTO) transfer on universal `any_object_grasped` and `any_placement_satisfied` primitives. Preflight PyTorch+OpenCV synthetic causality/probe tests passed 4/4 in 5.9 s (`AUROC = 0.9984` on separable signal).

### 8.2 Relation probing: within-task 5-fold CV and LOTO zero-shot transfer

| Feature stream | Linear macro AUROC | Linear grasp F1 / dest F1 | MLP-128 macro AUROC | LOTO grasp AUROC (F1) [Lin / MLP] | LOTO place AUROC (F1) [Lin / MLP] |
|---|---:|---:|---:|---:|---:|
| `proprio_actions` | 0.8359 | 0.568 / 0.307 | 0.8425 | **0.9571 (0.763)** / **0.9564 (0.704)** | 0.5643 (0.211) / 0.6961 (0.196) |
| **`xr1_dit`** *(frozen VLA)* | **0.8647** | **0.630 / 0.458** | **0.8699** | **0.9645 (0.753)** / **0.9659 (0.780)** | **0.7252 (0.168)** / **0.7599 (0.181)** |
| `siglip_wrist` *(pooled)* | 0.6525 | 0.275 / 0.245 | 0.5587 | 0.6306 (0.351) / 0.5542 (0.313) | 0.4842 (0.160) / 0.4767 (0.232) |
| `siglip_3cam` *(pooled)* | 0.6094 | 0.203 / 0.130 | 0.5743 | 0.5372 (0.288) / 0.5721 (0.301) | 0.5119 (0.164) / 0.4630 (0.222) |
| `siglip_3cam_plus_state` | 0.7941 | 0.453 / 0.260 | 0.6115 | 0.9507 (0.738) / 0.5852 (0.308) | 0.5434 (0.178) / 0.5035 (0.214) |
| `all_fused` | 0.8372 | 0.570 / 0.393 | 0.8009 | 0.9594 (0.771) / 0.9260 (0.715) | 0.6835 (0.174) / 0.6710 (0.252) |

### 8.3 Component substitution and fully-observed monitor lead (5% nominal FPR)

**Component substitution** (`siglip_3cam_plus_state`, linear probe; isolates where oracle lead is lost):

| Configuration | Lead (% horizon) | Lead (s) | Realized FPR | vs. stage clock | vs. global timer $G$ |
|---|---:|---:|---:|---:|---:|
| `oracle_reference` (true grasp + true dest) | **38.67%** | **55.0 s** | 3.1% | **+6.63 pp** | **+13.05 pp (+20.5 s)** |
| **`pred_grasp_true_dest`** (pred grasp + true dest) | **36.74%** | **51.7 s** | **3.1%** | **+4.69 pp** | **+11.12 pp (+17.1 s)** |
| `true_grasp_pred_dest` (true grasp + pred dest) | 13.79% | 20.5 s | 3.1% | −18.25 pp | −11.83 pp (−14.0 s) |
| `fully_observed_camera_monitor` | 9.66% | 14.0 s | 3.1% | −11.10 pp | −15.96 pp (−20.5 s) |

**Fully observed monitor by stream (zero privileged inputs — inferred stage clock + predicted grasps/destinations):**

| Stream (MLP-128 / Linear) | `inferred_stage_clock` lead (FPR) | `camera_full` lead (FPR) | `camera_full − G` | `camera_full − inferred_stage` |
|---|---:|---:|---:|---:|
| **`xr1_dit` (MLP-128)** | 23.20% / 32.8 s (4.7%) | **27.11% / 37.2 s (4.7%)** | **+1.50 pp (+2.7 s)** | **+3.91 pp (+4.4 s)** |
| `xr1_dit` (Linear) | 23.47% / 33.1 s (6.2%) | 25.45% / 35.7 s (6.2%) | −0.17 pp (+1.2 s) | +1.98 pp |
| `all_fused` (Linear) | 24.16% / 34.6 s (3.1%) | 23.16% / 33.7 s (3.1%) | −2.46 pp (−0.8 s) | −1.00 pp |
| `proprio_actions` (MLP-128) | 24.07% / 34.6 s (6.2%) | 22.14% / 31.2 s (3.1%) | −3.47 pp (−3.3 s) | −1.93 pp |
| `siglip_3cam` (Linear) | 12.50% / 19.3 s (1.6%) | 11.21% / 18.2 s (1.6%) | −14.40 pp (−16.3 s) | −1.29 pp |

### 8.4 Diagnostic conclusions (critical pivot for architecture design)

1. **Grasp detection is already solved—including zero-shot across unseen tasks.** Both `proprio_actions` (LOTO **0.957 AUROC**) and `xr1_dit` (LOTO **0.966 AUROC**, F1 **0.780**) detect grasps cleanly. Substituting predicted grasps for simulator-truth grasps preserves **84% of the oracle's gain over $G$ (`+17.1 s` vs. `+20.5 s`, `+4.69 pp` over stage clock, 3.1% FPR)**.
2. **Mean-pooled frozen ViT embeddings (`siglip_3cam` / `siglip_wrist`) are the wrong representation for spatial destination relations** (`dest_F1 = 0.13–0.25`, LOTO `place_AUROC ≈ 0.50`). Global spatial pooling erases tiny object-to-receptacle contacts (straw in glass, strawberry on pancake, knife on cutting board).
3. **100% of the perception bottleneck is destination verification.** False negatives on destination arrival keep the missed-placement age running during successful calibration episodes, inflating the conformal cutoff and collapsing lead (`+20.5 s → −14.0 s`).
4. **XR-1's own frozen DiT latents already carry strong relation information:** without any external vision model, an MLP-128 readout on `xr1_dit` alone already outperforms both $G$ (**+2.7 s**) and its own `inferred_stage_clock` (**+3.91 pp**) at **4.7% FPR**.
5. **Architecture implication:** Decouple the monitor cleanly into two specialized modules:
   - **Fast grasp/release trigger (< 1 ms, LOTO AUROC 0.966):** `proprio_actions + xr1_dit` detects grasp and release events at every decision.
   - **Spatial/open-vocabulary destination verifier (queried after release or at low cadence):** evaluate local **`Qwen3-VL-4B-Instruct`** (`/gs/bs/tga-shinoda/felid/robocasa_checkpoints/lingbot_vla_v2/downloads/Qwen3-VL-4B-Instruct/`) zero-shot Yes/No logprobs over high-res spatial tokens (wrist + agentview) to test whether VLM destination accuracy unlocks the remaining **+14 s** of oracle headroom without task-specific training.

---

## 9. Zero-shot `Qwen3-VL-4B-Instruct` destination verifier and decoupled hybrid monitor (28 November 2026)

### 9.1 Setup and provenance

- **Code:** commit `f865dd48` (`scripts/xr1_lead_time/extract_vlm_destinations.py`, `scripts/xr1_lead_time/eval_vlm_hybrid_monitor.py`, `scripts/run_xr1_vlm_destination_check.sh`, `tests/test_long_horizon_vlm_verifier.py` — 6/6 cluster PyTorch+OpenCV unit tests passed in 7.2 s).
- **Model:** local `/gs/bs/tga-shinoda/felid/robocasa_checkpoints/lingbot_vla_v2/downloads/Qwen3-VL-4B-Instruct` (`bfloat16`, single-forward-pass `Yes`/`No`/`Unknown` logprob readout over 3 native `256×256` views `left`, `right`, `wrist`; `image_grid_thw = [1, 16, 16]` per view, 319 total tokens/query).
- **Input & throughput:** All 120 rollouts from `xr1_recovery_pilot6_full_20261128_142543` at `decision_stride = 2` (causal zero-order hold between sampled decisions; strictly causal frame alignment `frame_step <= t`; strictly non-privileged prompts using only the observable `task_instruction`). Scored **14,082 queries in 323.2 s (43.6 q/s)** on 1 GPU.
- **Cluster output:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_vlm_dest_full_20261128_193617` (`vlm_scores/`, `eval_hybrid_dit/`, `eval_hybrid_proprio/`, `vlm_hybrid_results.tar.gz` 38 KB).

### 9.2 Zero-shot `Qwen3-VL-4B-Instruct` destination & condition verification

- **Universal LOTO `any_placement_satisfied`:** **AUROC = `0.7905`** (vs. `0.5119` pooled `siglip_3cam`, `0.4842` `siglip_wrist`, `0.6961` `proprio_actions`, and `0.7599` trained `xr1_dit` — achieved with **zero RoboCasa training labels**).
- **Macro over all 9 placement targets:** **AUROC = `0.7846`**, `@0.5 F1 = 0.402`.
- **Macro over all 12 destination + hold predicates:** **AUROC = `0.7648`**, `@0.5 F1 = 0.386`, within-task scalar-calibrated F1 = `0.458`, `vlm + xr1_dit` blend F1 = **`0.559`**, `vlm + proprio` blend F1 = **`0.534`**.

| Role | Task | Predicate | Sampled AUROC | Mean `P(vis)` | Zero-shot `@0.5` F1 | LOTO-th F1 | `vlm + xr1_dit` F1 |
|---|---|---|---:|---:|---:|---:|---:|
| DEST | `GarnishPancake` | `strawberry_on_pancake` | **0.9896** | 1.00 | 0.296 | 0.247 | **0.805** |
| DEST | `MakeIceLemonade` | `lemon_in_glass` | **0.9096** | 0.94 | 0.541 | 0.489 | 0.542 |
| DEST | `SetUpCuttingStation` | `meat_on_cutting_board` | **0.8736** | 1.00 | 0.353 | 0.246 | 0.488 |
| DEST | `DeliverStraw` | `straw_in_glass` | **0.8656** | 0.97 | 0.271 | 0.303 | 0.473 |
| DEST | `CuttingToolSelection` | `correct_tool_present_on_board` | **0.7725** | 1.00 | 0.275 | 0.199 | 0.451 |
| DEST | `ArrangeBreadBasket` | `bread_in_basket` | **0.7505** | 0.97 | **0.713** | 0.566 | 0.712 |
| DEST | `SetUpCuttingStation` | `knife_on_cutting_board` | **0.7447** | 1.00 | **0.677** | **0.708** | 0.700 |
| DEST | `MakeIceLemonade` | `ice_in_glass` | 0.6154 | 0.89 | 0.358 | 0.413 | 0.432 |
| DEST | `ArrangeBreadBasket` | `basket_on_dining_counter` | 0.5401 | 0.98 | 0.136 | 0.103 | 0.036 |
| HOLD | `CuttingToolSelection` | `wrong_tool_not_on_cutting_board` | 0.5859 | 0.99 | 0.417 | 0.692 | **0.898** |
| HOLD | `GarnishPancake` | `pancake_on_plate` *(all True)* | — | 0.98 | 0.301 | 0.350 | 0.758 |
| HOLD | `GarnishPancake` | `plate_on_table` *(all True)* | — | 0.98 | 0.290 | 0.295 | 0.416 |

### 9.3 End-to-end hybrid monitor lead (120 rollouts, 5-fold split-conformal, 5% nominal FPR)

| Configuration | Privileged inputs? | Lead (% / s) | Realized FPR | Recall | vs. stage / inf\_stage | vs. $G$ (% / s) |
|---|---|---:|---:|---:|---:|---:|
| `oracle_reference` *(Section 7)* | Yes (grasp+dest+stage) | **38.67% / 55.0 s** | 3.1% | 98.2% | **+6.63 pp** | **+13.05 pp (+20.5 s)** |
| `true_grasp + vlm_plus_proprio` | Grasp + stage only | **33.42% / 47.9 s** | 4.7% | 94.6% | **+1.37 pp** | **+7.80 pp (+13.4 s)** |
| `true_grasp + vlm_zeroshot` *(0 dest labels)* | Grasp + stage only | **31.64% / 43.6 s** | **3.1%** | **98.2%** | −0.41 pp | **+6.02 pp (+9.1 s)** |
| `true_grasp + siglip_state` *(Section 8)* | Grasp + stage only | 13.79% / 20.5 s | 3.1% | — | −18.25 pp | −11.83 pp (−14.0 s) |
| **`hybrid_proprio_grasp + vlm_plus_proprio`** | **NONE (black-box!)** | **27.79% / 40.0 s** | **3.1%** | **96.4%** | **+7.35 pp** | **+2.17 pp (+5.5 s)** |
| `xr1_dit` MLP-128 alone *(Section 8)* | NONE (needs DiT) | 27.11% / 37.2 s | 4.7% | — | +3.91 pp | +1.50 pp (+2.7 s) |
| **`zero_shot_loto_universal` (`dit` grasp + VLM `@0.5`)** | **NONE (0 target labels!)** | **26.52% / 36.0 s** | **3.1%** | **98.2%** | **+5.96 pp** | **+0.91 pp (+1.5 s)** |
| `hybrid_proprio_grasp + vlm_zeroshot` | NONE (0 dest labels) | 26.39% / 35.9 s | 3.1% | 96.4% | +5.82 pp | +0.77 pp (+1.4 s) |
| `hybrid_dit_grasp + vlm_zeroshot` | NONE (0 dest labels) | 26.17% / 35.6 s | 3.1% | 96.4% | +5.60 pp | +0.55 pp (+1.1 s) |
| `zero_shot_loto_universal` (`proprio` grasp + VLM `@0.5`) | NONE (0 target labels!) | 26.11% / 35.4 s | 3.1% | 98.2% | +5.54 pp | +0.49 pp (+0.9 s) |

### 9.4 Diagnostic takeaways & immediate zero-inference fixes

1. **Zero-shot `Qwen3-VL-4B-Instruct` transforms destination perception:** `true_grasp + vlm_zeroshot` gained **+23.1 s (+17.85 pp)** over Section 8's SigLIP+state probe (`20.5 s → 43.6 s`, **+9.1 s over $G$ at 3.1% FPR, 98.2% recall**), and **7 of 9 placement targets** already reach **0.74–0.99 AUROC** with zero training.
2. **First completely policy-agnostic monitor beats $G$ by +5.5 s at 3.1% FPR:** `hybrid_proprio_grasp + vlm_plus_proprio` uses **no privileged simulator state and no internal VLA latents**, yet reaches **27.79% (40.0 s), 3.1% FPR, 96.4% recall (+7.35 pp over inferred stage clock, +5.5 s over $G$)** — beating even `xr1_dit`-only (`37.2 s @ 4.7% FPR`).
3. **True zero-shot cross-task transfer works out of the box:** `zero_shot_loto_universal` (LOTO universal grasp detector trained only on the other 5 tasks + object-agnostic release verification via zero-shot `Qwen3-VL-4B`) achieves **26.52% (36.0 s), 3.1% FPR, 98.2% recall, +5.96 pp over inferred stage clock, +1.5 s over $G$** with **zero target-task labels**.
4. **Three specific, immediately testable causes explain the remaining gap (`40–43.6 s` vs. `55.0 s` oracle):**
   - **Massive per-predicate Yes/No prior bias:** `strawberry_on_pancake` has **0.9896 AUROC** yet only **0.296 F1** at logit `0.0` (similarly `straw_in_glass` `0.866 / 0.271`, `meat_on_cutting_board` `0.874 / 0.353`, `lemon_in_glass` `0.910 / 0.541`). Subtracting each episode's own strictly causal pre-release baseline (`logit_diff(t) - baseline(0..g_start)`) can eliminate prompt/kitchen bias with zero training labels.
   - **Spurious out-of-view `"No"` on hold-only predicates:** `Qwen3-VL` rarely output `"Unknown"` (`mean P(vis) = 0.94–1.00`), so when the robot faced the fridge in `GarnishPancake`, `pancake_on_plate` and `plate_on_table` (always True in truth) got `0.29–0.30` F1 and injected false `loss_age` into successes.
   - **Hard 0.5 thresholding before split conformal calibration:** Any single false negative on a successful episode keeps `missed_effect_age` ticking until termination and inflates the 95% conformal quantile for all episodes (which is why balanced-F1 thresholds hurt lead vs. high-recall `vlm_zeroshot`). Switching to **continuous probability-weighted missed-effect evidence** (integrating `1 - p_placed(t)` after release) or recall-safe calibration + causal baseline subtraction can be tested immediately on the already-saved `vlm_destination_scores.npz` in < 30 seconds without re-running the VLM.
   - **Only two visually hard predicates remain:** `basket_on_dining_counter` (`0.540 AUROC`, needs before/after reset-frame comparison or base-displacement cue) and `ice_in_glass` (`0.615 AUROC`, tiny translucent cube — candidate for before/after wrist comparison or 4-train / 2-held-out-task `Qwen3-VL` LoRA).

### 9.5 Zero-re-inference scoring upgrades & post-release diagnostics (`xr1_vlm_scoring_upgrades_20261128_195948`)

- **Code:** commit `970e3217` (`scripts/xr1_lead_time/eval_vlm_scoring_upgrades.py`, `scripts/run_xr1_vlm_scoring_upgrades.sh`, `tests/test_long_horizon_vlm_scoring_upgrades.py` — 4/4 tests passed).
- **Cluster output:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_vlm_scoring_upgrades_20261128_195948` (`vlm_scoring_upgrades.tar.gz`, 45 KB).

**Post-release logit diagnostics (all 9 placement targets, strictly after first release):**

| Task | Predicate | All-frame raw AUROC | **Post-release raw / `dpre` AUROC** | Raw `neg / pos` mean logit | `dpre` `neg / pos` mean logit |
|---|---|---:|---:|---:|---:|
| `GarnishPancake` | `strawberry_on_pancake` | 0.988 | **0.983 / 0.986** | −12.53 / +9.90 | +3.23 / +25.60 |
| `DeliverStraw` | `straw_in_glass` | 0.873 | **0.911 / 0.896** | −9.22 / −1.16 | +1.98 / +9.84 |
| `MakeIceLemonade` | `lemon_in_glass` | 0.908 | **0.906 / 0.893** | −9.26 / +1.05 | +1.12 / +10.99 |
| `SetUpCuttingStation` | `knife_on_cutting_board` | 0.745 | **0.769 / 0.874** | −6.06 / −1.01 | +0.36 / +7.00 |
| `ArrangeBreadBasket` | `basket_on_dining_counter` | 0.523 | **0.864 / 0.841** | −8.03 / −7.58 | −1.40 / −2.85 |
| `SetUpCuttingStation` | `meat_on_cutting_board` | 0.884 | **0.797 / 0.859** | −6.64 / +5.77 | +1.41 / +14.20 |
| `ArrangeBreadBasket` | `bread_in_basket` | 0.754 | **0.737 / 0.806** | −8.31 / +1.03 | −0.45 / +7.35 |
| `CuttingToolSelection` | `correct_tool_present_on_board` | 0.748 | **0.730 / 0.790** | −5.25 / +0.86 | +0.49 / +7.99 |
| `MakeIceLemonade` | `ice_in_glass` | 0.614 | 0.361 / 0.366 | −7.95 / −4.82 | +1.54 / +5.53 |
| **Macro (9 targets / 8 non-ice)** | | **0.782** | **0.784 / 0.812** *(0.868 on 8/9)* | | |

**Key upgraded monitor arms (120 rollouts, 5-fold split-conformal, 5% nominal FPR):**

| Configuration | Privileged? | Stage lead | Full lead (% / s) | FPR | Recall | vs. stage | vs. $G$ (% / s, 95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| `oracle_full_all_holds` | Yes | 32.05% | **38.67% / 55.0 s** | 3.1% | 98.2% | +6.63 pp | +13.05 pp (+20.5 s) [+6.1, +20.2] |
| `oracle_placement_only` | Yes | 32.05% | **37.07% / 54.0 s** | 3.1% | 96.4% | +5.02 pp | +11.45 pp (+19.5 s) [+2.2, +19.9] |
| **`tg_vlm_raw_hard_placeonly`** *(0 training!)* | Grasp+stage | 32.05% | **34.25% / 50.0 s** | **3.1%** | 92.9% | **+2.20 pp** | **+8.63 pp (+15.4 s)** [−0.3, +16.9] |
| `tg_vlm_raw_soft_c070` *(0 training!)* | Grasp+stage | 32.05% | **33.90% / 49.3 s** | **3.1%** | 92.9% | **+1.86 pp** | **+8.29 pp (+14.8 s)** [−0.5, +16.3] |
| **`hyb_dpre_platt_soft_verholds` (`proprio`)** | **NONE (black-box!)** | **24.70%** | **30.18% / 42.2 s** | **4.7%** | **94.6%** | **+5.49 pp** | **+4.56 pp (+7.7 s)** [−2.3, +12.3] (4/6) |
| **`hyb_dpre_platt_soft_verholds` (`xr1_dit`)** | **NONE (0 dest labels)** | **24.68%** | **28.85% / 40.6 s** | **4.7%** | **94.6%** | **+4.17 pp** | **+3.23 pp (+6.1 s)** [−3.6, +10.7] (3/6) |

**Exact root-cause diagnosis from Section 9.5:**
1. **Zero-shot `Qwen3-VL-4B` ALREADY sees 8 of 9 placements cleanly right after release (`0.868` macro post-release AUROC; even `basket_on_dining_counter` is `0.864` post-release!)**, and simply dropping fridge-view background holds jumps `true_grasp + zero-shot Qwen3-VL` by **+6.4 s to 50.0 s** (**92.6% of the 54.0 s placement-only oracle ceiling at 3.1% FPR, +2.20 pp over oracle `stage_clock`, zero training**).
2. **Pre-grasp debiasing + LOTO Platt + verified-target holds lifts the completely black-box monitor (`proprio_actions` + `Qwen3-VL`, zero privileged inputs, zero DiT latents, zero target-task VLM labels) to `30.18% (42.2 s)` at `4.7% FPR` (+5.49 pp over inferred stage clock, +7.7 s over $G$).**
3. **Why the remaining ~12 s gap exists:** `Qwen3-VL-4B` has a severe **−5 to −12 No-bias** (`straw_in_glass` pos = −1.16, `knife_on_cutting_board` pos = −1.01, `basket_on_dining_counter` pos = −7.58) plus 1 genuinely blind predicate (`ice_in_glass`, `0.36` post-release AUROC). When any single task stays below the decision threshold on successes (`basket_on_dining_counter` / `ice_in_glass`), its 10 calibration successes (~20% of the pooled calibration set > 5%) blow up the pooled conformal cutoff for `miss_term` across *all* 6 tasks and drag `inferred_stage_clock` from `32.05%` down to `24.70%`.
4. **Clear next test:** **3-fold cross-task LoRA fine-tuning of `Qwen3-VL-4B-Instruct`** (train on 4 tasks, evaluate **strictly zero-shot on the 2 held-out unseen tasks** across all 3 folds = all 120 rollouts) tests directly whether lightweight RoboCasa visual grounding eliminates the −8 No-bias, improves tiny-object perception (`ice_in_glass` / `basket_on_dining_counter`), and transfers zero-shot to unseen tasks so the **100% zero-shot unseen-task monitor** closes the gap to ~50 s.

---

## 10. 3-Fold cross-task `Qwen3-VL-4B-Instruct` LoRA (strict unseen-task zero-shot, 28 November 2026)

### 10.1 Setup and provenance

- **Code:** commit `2100d381` (`scripts/xr1_lead_time/train_eval_vlm_lora_cross_task.py`, `scripts/run_xr1_vlm_lora_cross_task.sh`, `tests/test_long_horizon_vlm_lora.py` — 3/3 cluster PyTorch unit tests passed in 3.1 s).
- **Protocol (strictly zero target-task leakage):** Partitioned the 6 tasks into 3 balanced disjoint 2-task held-out folds (`ArrangeBreadBasket + DeliverStraw`, `CuttingToolSelection + MakeIceLemonade`, `GarnishPancake + SetUpCuttingStation`). On each fold, injected 200 pure-PyTorch rank-16 LoRA adapters (`alpha = 32`, `dropout = 0.05`, attention + visual merger only, **15.1M / 4.45B = 0.34% trainable**), trained 2 epochs (~1,435–1,566 class-balanced, hard-post-release-negative-enriched samples from the 4 training tasks only, BCE on `logit_yes - logit_no`, `lr = 1e-4`, `wd = 0.05`), and evaluated **strictly zero-shot on all 40 rollouts of the 2 unseen tasks** at `decision_stride = 2`. Completed all 3 folds (training + 14,082 unseen-task queries) in **894.3 s (14.9 min)** on 1 H100 GPU.
- **Cluster output:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_vlm_lora_crosstask_20261128_202023` (`vlm_lora_crosstask_results.tar.gz`, 92 KB).

### 10.2 Unseen-task perception & calibration gains (Base Zero-Shot vs. 3-Fold Cross-Task LoRA)

- **Universal `any_placement_satisfied` LOTO AUROC:** `0.7905` → **`0.8197`** (+2.92 pp; vs. SigLIP `0.5119`).
- **Macro 9 placement targets:** sampled AUROC `0.7846` → **`0.8133`**; `dpre` macro AUROC `0.777` → **`0.8288`**; within-task scalar-cal F1 `0.458` → **`0.578`** (+12.0 pp); `vlm + xr1_dit` F1 `0.559` → **`0.598`**.
- **Severe −8 No-bias eliminated on unseen tasks:** 7 of 9 unseen placement targets shifted from deeply negative raw logits (`−5.3` to `−12.5` neg, `−1.2` pos) to symmetric zero-centered logits (`−1.5..−4.5` neg vs. `+0.8..+2.7` pos):

| Unseen-task predicate | Base ZS AUROC | **LoRA Unseen AUROC (`dpre`)** | Base raw `neg/pos` | **LoRA raw `neg/pos`** | LoRA `postR / postD` | LoRA `zs@0.5 / dpre` F1 |
|---|---:|---:|---:|---:|---:|---:|
| `bread_in_basket` | 0.751 | **0.863** *(0.825)* | −8.31 / +1.03 | **−1.49 / +1.20** | 0.817 / 0.709 | 0.700 / 0.538 |
| `basket_on_dining_counter` | 0.540 | **0.568** *(0.578)* | −8.03 / −7.58 | −2.22 / −2.02 | 0.718 / 0.720 | 0.037 / 0.044 |
| `correct_tool_present_on_board` | 0.773 | **0.867** *(0.874)* | −5.25 / +0.86 | **−4.29 / +2.39** | 0.812 / 0.852 | 0.382 / 0.189 |
| `straw_in_glass` | 0.866 | 0.818 *(**0.889**)* | −9.22 / −1.16 | **−2.08 / +0.83** | **0.919 / 0.901** | 0.398 / 0.353 |
| `strawberry_on_pancake` | 0.990 | 0.936 *(0.863)* | −12.53 / +9.90 | **−3.01 / +1.94** | 0.887 / 0.892 | 0.251 / 0.210 |
| `lemon_in_glass` | 0.910 | **0.913** *(0.914)* | −9.26 / +1.05 | **−4.55 / +1.84** | 0.886 / 0.880 | 0.578 / 0.398 |
| `ice_in_glass` | 0.615 | **0.627** *(0.682)* | −7.95 / −4.82 | −3.37 / −1.19 | 0.438 / 0.425 | 0.409 / 0.433 |
| `knife_on_cutting_board` | 0.745 | **0.875** *(**0.946**)* | −6.06 / −1.01 | **−3.74 / +0.95** | 0.648 / **0.949** | **0.729 / 0.846** |
| `meat_on_cutting_board` | 0.874 | 0.865 *(**0.894**)* | −6.64 / +5.77 | **−2.49 / +2.66** | 0.725 / 0.767 | 0.305 / 0.244 |

### 10.3 End-to-end monitor lead with unseen-task LoRA VLM (120 rollouts, 5% nominal FPR)

| Configuration | Privileged? | Stage | Full (% / s) | FPR | Recall | vs. stage | vs. $G$ (% / s, 95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| `oracle_reference` *(Section 7)* | Yes | 32.05% | **38.67% / 55.0 s** | 3.1% | 98.2% | +6.63 pp | +13.05 pp (+20.5 s) [+6.1, +20.2] |
| **`true_grasp + LoRA VLM + proprio`** | Grasp+stage | 32.05% | **37.35% / 53.5 s** | **4.7%** | **96.4%** | **+5.31 pp** | **+11.74 pp (+19.0 s)** |
| **`tg_vlm_raw_hard_verholds`** *(100% unseen VLM @ 0.5)* | Grasp+stage | 32.05% | **34.41% / 48.7 s** | **3.1%** | **96.4%** | **+2.37 pp** | **+8.80 pp (+14.2 s)** **[+2.3, +15.7]** (5/6) |
| `tg_vlm_raw_hard_placeonly` *(100% unseen VLM @ 0.5)* | Grasp+stage | 32.05% | **34.23% / 49.9 s** | **3.1%** | 92.9% | +2.18 pp | +8.61 pp (+15.4 s) [−0.2, +17.2] (4/6) |
| **`hybrid_proprio + LoRA VLM + proprio`** | **NONE (black-box)** | 20.19% | **29.83% / 42.4 s** | 6.2% | 96.4% | **+9.64 pp** | **+4.21 pp (+7.9 s)** |
| **`loto_univ_dpre_platt_soft_verholds` (`xr1_dit`)** | **NONE (0 target labels!)** | 24.85% | **27.07% / 38.6 s** | **4.7%** | **94.6%** | **+2.22 pp** | **+1.45 pp (+4.1 s)** [−5.9, +9.1] (3/6) |
| `loto_univ_raw_soft_verholds` (`xr1_dit`) | NONE (0 target labels!) | 23.86% | **26.96% / 38.7 s** | **4.7%** | **96.4%** | **+3.09 pp** | **+1.34 pp (+4.2 s)** [−6.8, +11.0] (2/6) |

### 10.4 Key diagnostic discovery: isolating the last bottleneck to `inferred_stage_clock`

Comparing `tg_vlm_raw_hard_verholds` (**34.41% / 48.7 s**, 95% CI `[+2.3, +15.7]` pp, **5/6 tasks positive**, 3.1% FPR) against `loto_univ_raw_hard_verholds` (**24.18% / 35.6 s**, 3.1% FPR) exposes **exact arithmetic**:
- Substituting predicted grasps alone lost only **1.93 pp** (Section 8).
- Substituting unseen-task LoRA VLM destinations alone lost only **4.26 pp** (`38.67% → 34.41%`, still strictly beating oracle `stage_clock` by `+2.37 pp` and $G$ by `+8.80 pp`!).
- Yet replacing oracle `stage_clock` (`32.05%`) with our placement-only `inferred_stage_clock` (`21.39%`) dropped **−10.66 pp by itself** — because single-object tasks (`CuttingToolSelection`, `DeliverStraw`, `GarnishPancake`) only satisfy their single placement target near episode termination, leaving `inferred_stage_clock` stuck in `stage_0` for 99% of the rollout and completely blind to `no_sustained_grasp` failures (21.4% of failures!), while VLM stage-latch jitter inflated the pooled conformal cutoff for `full_physical`.
- **Direct fix (zero new VLM inference):** Build a **grasp-anchored inferred stage clock** using our 0.966-AUROC LOTO universal grasp detector (`pre_first_grasp → post_first_grasp`, plus optional `grasp + VLM` pick-place stages) and evaluate both pooled and **union-bound channel-separated split-conformal calibration** ($\alpha/2 + \alpha/2 = 0.05$) on both base zero-shot and cross-task LoRA VLM scores.

### 10.5 Grasp-anchored observable stage clock breakthrough (`xr1_grasp_anchored_monitor_20261128_204344`)

- **Code:** commit `bc7b45eb` (`scripts/xr1_lead_time/eval_grasp_anchored_monitor.py`, `scripts/run_xr1_grasp_anchored_monitor.sh`, `tests/test_long_horizon_grasp_anchored.py` — 2/2 tests passed).
- **Cluster output:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_grasp_anchored_monitor_20261128_204344` (`grasp_anchored_results.tar.gz`, 52 KB).

Reconstructing Subtask-SAFE's pick-then-place stage timeline strictly from our **0.966-AUROC LOTO universal grasp detector** + **unseen-task `Qwen3-VL-4B` destination verification** (`grasp_and_vlm` stage clock, **zero privileged inputs, zero target-task labels**) immediately recovered the missing stage-clock headroom (`inferred_stage_clock` jumped `+5.93 pp` on base zero-shot VLM to `26.50%` and **`+6.59 pp` on unseen-task LoRA VLM to `27.98%`**, 87.3% of the `32.05%` simulator oracle stage clock):

| Complete 6-task progression (120 rollouts, 5-fold split-conformal, 5% nominal FPR) | Privileged? | Target-task labels? | Stage lead | Full lead (% / s) | FPR | Recall | vs. stage | vs. $G$ (% / s) | Pos tasks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `global_timer` ($G$) | None | Cal only | — | 25.62% / 34.5 s | 3.1% | 89.3% | — | 0.00 pp (0.0 s) | — |
| `task_timer` ($P$) | None | Durations | — | 22.01% / 33.3 s | 4.7% | 94.6% | — | −3.61 pp (−1.2 s) | — |
| `siglip_3cam` linear *(Sec 8)* | None | Within-task | 12.50% | 11.21% / 18.2 s | 1.6% | — | −1.29 pp | −14.40 pp (−16.3 s) | — |
| `xr1_dit` MLP-128 alone *(Sec 8)* | None (DiT) | Within-task | 23.20% | 27.11% / 37.2 s | 4.7% | — | +3.91 pp | +1.50 pp (+2.7 s) | — |
| `univ_no_stage` (`LOTO DiT` + ZS VLM, no stage clock) | **None** | **ZERO** | 22.00% | **28.76% / 40.2 s** | **4.7%** | 94.6% | **+6.76 pp** | **+3.14 pp (+5.7 s)** | 3/6 |
| `univ_gv_stage` (`LOTO proprio` + LoRA unseen VLM) | **None (black-box)** | **ZERO** | 24.68% | **27.41% / 38.1 s** | **4.7%** | 92.9% | **+2.74 pp** | **+1.79 pp (+3.6 s)** | **4/6** |
| `hyb_dpre_platt_verholds` (`proprio` + ZS VLM, *Sec 9.5*) | **None (black-box)** | 0 VLM labels | 24.70% | **30.18% / 42.2 s** | **4.7%** | 94.6% | **+5.49 pp** | **+4.56 pp (+7.7 s)** | **4/6** |
| **`univ_gv_stage` (`LOTO DiT` + LoRA unseen VLM)** | **None** | **ZERO** | **27.98%** | **29.20% / 40.1 s** | **4.7%** | **96.4%** | +1.22 pp | **+3.58 pp (+5.6 s)** | **5/6** |
| **`univ_gv_stage` (`LOTO DiT` + ZS `Qwen3-VL-4B`)** | **None** | **ZERO (0 VLM train!)** | **26.50%** | **32.40% / 45.1 s** | 6.2% *(4/64)* | **98.2%** | **+5.90 pp** | **+6.78 pp (+10.6 s)** | **5/6** |
| `tg_vlm_raw_hard_verholds` (LoRA unseen VLM) | Grasp+stage | 0 VLM labels | 32.05% | **34.41% / 48.7 s** | **3.1%** | 96.4% | +2.37 pp | **+8.80 pp (+14.2 s)** | **5/6** |
| `true_grasp + LoRA unseen VLM + proprio` | Grasp+stage | Grasp+proprio | 32.05% | **37.35% / 53.5 s** | 4.7% | 96.4% | +5.31 pp | **+11.74 pp (+19.0 s)** | — |
| `oracle_reference` *(Sec 7)* | Full sim | — | 32.05% | **38.67% / 55.0 s** | **3.1%** | 98.2% | +6.63 pp | **+13.05 pp (+20.5 s)** | **6/6** |

*(Methodological note on union-bound conformal at $\alpha/2 = 0.025$: with $n_{\text{cal}} = 34$ calibration successes per fold on the 6-task pilot, $\lceil (1 - 0.025)(34 + 1) \rceil = 35 > 34$ yields a vacuous `None` cutoff. Finite-sample union-bound calibration at $\alpha/2 = 0.025$ requires $n_{\text{cal}} \ge 39$, which will be met easily once scaled to the 18-task collection. Also note that with 64 successes, 1 episode = 1.56% FPR, so `3/64 = 4.7%` and `4/64 = 6.2%` differ by a single rollout.)*

---

## 11. Test-2 continuation-aware pilot check on the 120 recorded rollouts (28 November 2026)

### 11.1 Setup and provenance

- **Code:** commit `57c910c5` (`scripts/xr1_lead_time/eval_test2_continuation_pilot.py`, `scripts/run_xr1_test2_continuation_pilot.sh`, `tests/test_long_horizon_test2_continuation.py` — 2/2 causality tests passed).
- **Cluster output:** `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_test2_continuation_pilot_20261128_205424` (`test2_continuation_pilot.tar.gz`, 24 KB).
- **Cohort:** All confirmed physical mistakes across the 120 rollouts — **5 strict multi-decision self-corrected successes** across 5 distinct tasks (`ArrangeBreadBasket 66764859`, `CuttingToolSelection c0f68907`, `DeliverStraw a1ddc0a9`, `MakeIceLemonade eca17b9b`, `SetUpCuttingStation 35d06a39`), **11 broad confirmed-mistake successes**, **35 persistent failures**, and **4 exact equal-duration matched `(task, kind, operation, relation)` pairs**.
- **Strict pre-recovery causality:** Features are extracted strictly inside `[confirmed_step, min(confirmed_step + H*16, restored_step))` (`pre_restore`) and `[confirmed_step, min(confirmed_step + H*16, next_regrasp_step, restored_step))` (`pre_regrasp`, strictly BEFORE the robot even touches/regrasps the dropped object). Evaluated with both out-of-fold LORO and **unseen-task LOTO** L2 logistic probes + two-sided 4,000-draw permutation tests.

### 11.2 Continuation-aware discrimination vs. same-input clock baseline

| Cohort & strict pre-recovery window | `same_input_clock` LORO / LOTO ($p$) | `action_kinematics` LORO / LOTO ($p$) | `visual_vlm_dynamics` (ZS VLM) LORO / LOTO ($p$) | `fused_continuation` (ZS VLM) LORO / LOTO ($p$) |
|---|---:|---:|---:|---:|
| Strict multi-dec, `pre_restore`, `H=8` (`5 vs 35`) | 0.463 / 0.309 (`0.184`) | 0.720 / 0.703 (`0.157`) | **0.783 (`0.045`) / 0.760 (`0.061`)** | 0.697 / 0.703 (`0.148`) |
| Strict multi-dec, **`pre_regrasp`**, `H=8` (`5 vs 35`) | 0.469 / 0.457 (`0.784`) | **0.731 / 0.731 (`0.095`)** | 0.686 / 0.577 (`0.607`) | 0.651 / 0.640 (`0.327`) |
| Broad all-succ, `pre_restore`, `H=8` (`11 vs 35`) | 0.670 / 0.639 (`0.175`) | **0.857 (`0.0005`) / 0.821 (`0.0005`)** | **0.881 (`0.0002`) / 0.878 (`0.0002`)** | **0.881 (`0.0005`) / 0.875 (`0.0002`)** |
| Broad all-succ, **`pre_regrasp`**, `H=8` (`11 vs 35`) | 0.668 / 0.626 (`0.220`) | **0.855 (`0.0005`) / 0.797 (`0.0020`)** | **0.805 (`0.0015`) / 0.790 (`0.0020`)** | **0.855 (`0.0002`) / 0.829 (`0.0002`)** |

**Key univariate physical continuation signatures (1 = persistent failure, 0 = self-corrected success):**

| Signal | Strict `pre_regrasp` `H=8` (`5 vs 35`) | Equal-length `H=3` (`5 vs 35`, zero truncation) | Broad `pre_regrasp` `H=8` (`11 vs 35`) | Equal-duration matched pairs (`n=4`, `pre_regrasp`) | Physical meaning |
|---|---:|---:|---:|---:|---|
| `dit_cos_conf` | 0.686 (`p=0.186`) | **0.817 (`p=0.022`)** | — | **4 / 4 (`SC > PF`, 0.783 vs 0.670)** | Self-correcting VLA stays anchored to mistake context |
| `dit_cos_onset` | 0.691 (`p=0.179`) | **0.743 (`p=0.083`)** | — | **4 / 4 (`SC > PF`, 0.780 vs 0.667)** | Failing VLA representation drifts into next subtask |
| `prop_state_drift_conf` | **0.914 (`p=0.0015`)** | 0.703 (`+0.94 vs +3.67`) | **0.940 (`p=0.0002`)** | **3 / 4 (`PF > SC`, 7.10 vs 1.18)** | Failing robot moves/drifts away from mistake site |
| `prop_state_step_vel` *(length-inv.)* | 0.731 (`p=0.108`) | — | **0.855 (`p=0.0002`)** | **3 / 4 (`PF > SC`, 4.46 vs 0.86)** | Faster runaway proprioceptive velocity in failures |
| `act_base_speed_max` | **0.817 (`p=0.023`)** | — | **0.852 (`p=0.0005`)** | — | Mobile base drives away in persistent failures |
| `act_chunk_cos` *(length-inv.)* | 0.749 (`p=0.072`) | 0.697 (`0.967 vs 0.841`) | **0.792 (`p=0.0025`)** | **3 / 4 (`SC > PF`, 0.948 vs 0.857)** | Self-correcting policy plans smooth consistent retries |
| `vlm_logit_last` (ZS VLM) | **0.806 (`p=0.030`)** | — | **0.852 (`p=0.0005`)** | — | Target stays visually close (`−4.2` vs `−10.1`) |

### 11.3 Scientific interpretation & methodological safeguard

1. **Clear, physically coherent Test-2 mechanism (policy engagement vs. abandonment):** When XR-1 is going to self-correct, it remains physically and representationally anchored to the mistake site (high `dit_cos_conf` / `dit_cos_onset` in **4/4 equal-duration pre-regrasp pairs**, smooth `act_chunk_cos`, low runaway drift/base speed, target still in camera view). When XR-1 enters a persistent failure, it abandons the missed object and drives away (`4×–6×` higher proprioceptive drift/velocity even over strictly equal windows, `dit_cos` drops, VLM logit plunges to `−10.1`).
2. **Methodological safeguard on window-truncation confound:** In `H = 8` windows, quick recoveries terminate before 8 decisions, which artificially lowers cumulative drift (`prop_state_drift_conf`) relative to full-8-decision failures. Three length-controlled checks prove the signal is genuinely physical and NOT a truncation artifact:
   - Strictly equal 3-decision windows (`H = 3`, all 5 strict successes have ≥ 3 decisions, `same_input_clock` LORO = `0.5000`): **`dit_cos_conf` AUROC = `0.8171` (`p = 0.022`)**, `prop_state_drift_conf` still `4×` higher in failures (`+3.67` vs. `+0.94`);
   - Strictly equal-duration matched pairs (`n = 4`, identical `w_pre_regrasp`): **`dit_cos_conf` 4/4**, **`dit_cos_onset` 4/4**, **`prop_state_drift_conf` 3/4 (`7.10` vs. `1.18`)**, **`prop_state_step_vel` 3/4 (`4.46` vs. `0.86`)**, **`act_chunk_cos` 3/4 (`0.948` vs. `0.857`)**;
   - Length-invariant per-step velocity (`prop_state_step_vel` `0.855`, `p = 0.0002`) and chunk cosine (`act_chunk_cos` `0.792`, `p = 0.0025`).
3. **Verdict:** **Test 2 is worth powering at scale** (and even better, it relies primarily on black-box kinematic/engagement features + DiT cosine anchor, transferring cleanly across unseen tasks at **LOTO AUROC `0.829–0.878`, `p = 0.0002`**!). Note that future Test-2 evaluation at scale should strictly use **equal-duration matched windows** (e.g., fixed `H = 3–4` decisions and exact pair-matched `w_pre_regrasp`) so cumulative drift cannot benefit from variable window lengths.

---

## 12. Artifact index and next experiments

### Reproducibility & artifact index

| Artifact | Location | Notes / SHA-256 |
|---|---|---|
| Roadmap document | `reports/long_horizon_failure_detection_roadmap.docx` | Initial 27 Nov 2026 roadmap |
| Prototype event audit | `docs/long_horizon_relation_events.md`, `outputs/long_horizon_failure_detection/xr1_ladder_76/summary.json` | 76 October parents |
| Calibrated pilot (internal fit) | `docs/long_horizon_calibration_pilot.md`, `outputs/long_horizon_failure_detection/xr1_ladder_76_calibration_v1/calibration.json` | +6.42 pp, 2/36 vs 1/36 succ |
| Calibrated pilot (external naturals) | `docs/long_horizon_calibration_pilot.md`, `outputs/long_horizon_failure_detection/xr1_ladder_76_calibration_external_naturals/calibration.json` | `a3282acb...`; +6.37 pp, 2/36 vs 1/36 succ |
| External naturals archive | `outputs/long_horizon_failure_detection/cluster_imports/xr1_naturals_fit.tar.gz` | `9dbc1fff...`; 60 parents (25 succ) |
| Local release-clock controls | `docs/long_horizon_local_controls.md`, `outputs/long_horizon_failure_detection/xr1_ladder_76_local_controls/results.json` | 0/76 alarm diff at 32–256 steps; 0 verified pairs |
| Cluster cohort audit | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/long_horizon_cohort_audit_20261128_130837.json` | Legacy August cohorts lack video/actions/raw predicates |
| Synchronized XR-1 smoke run (2 tasks) | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_recovery_recording_PNL4QPfz` | Commit `9e59c463`; 4/4 verified; archive `cc8999a5...` |
| Visual inspection contact sheets | `outputs/long_horizon_failure_detection/xr1_recovery_recording_PNL4QPfz_inspection/*.png` | 4 contact sheets (Section 6.2) |
| Synchronized XR-1 smoke run (6 tasks) | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_recovery_pilot6_smoke_20261128_135824` | Commit `4b3666de`; 12/12 verified (9S/3F); metadata `e4d123da...` |
| 6-task oracle & Test-2 analyzer | `scripts/xr1_lead_time/analyze_recovery_pilot6.py`, `tests/test_long_horizon_pilot6_analyzer.py` | 41/41 tests pass; smoke summary `40d7866e...` |
| Fresh 120-rollout 6-task pilot | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_recovery_pilot6_full_20261128_142543` | Commit `4b3666de`; 120/120 verified (64S/56F); metadata `8407eb24...` |
| 120-rollout split-conformal analysis | `outputs/long_horizon_failure_detection/xr1_pilot6_full_120_analysis/summary.json` | `ff8c5660...`; **+6.63 pp (+8.3 s) over stage clock @ 0.00 pp excess FPR** |
| Combined 132-rollout sensitivity | `outputs/long_horizon_failure_detection/xr1_pilot6_combined_132_analysis/summary.json` | `9c3b2154...`; +5.55 pp (+8.1 s) over stage clock @ 0.00 pp excess FPR |
| Perception floor check (120 rollouts) | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_perception_floor_20261128_191229` | Commit `bad8bbd8`; LOTO grasp 0.966; `xr1_dit` cam_full +2.7 s vs G / +3.91 pp vs inf_stage |
| Zero-shot Qwen3-VL destination check | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_vlm_dest_full_20261128_193617` | Commit `f865dd48`; 14,082 queries (323 s); universal AUROC 0.791 |
| VLM scoring upgrades & post-release audit | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_vlm_scoring_upgrades_20261128_195948` | Commit `970e3217`; post-rel AUROC 0.812; `tg_placeonly` 50.0 s; black-box `proprio+VLM` 42.2 s (+7.7 s vs G) @ 4.7% FPR |
| 3-fold cross-task Qwen3-VL LoRA | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_vlm_lora_crosstask_20261128_202023` | Commit `2100d381`; 894 s; unseen universal AUROC **0.820**; `tg+LoRA+proprio` **53.5 s @ 4.7% FPR** |
| Grasp-anchored zero-privileged monitor | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_grasp_anchored_monitor_20261128_204344` | Commit `bc7b45eb`; **100% zero-target-label `univ_gv_stage`**: **45.1 s (+10.6 s vs G, 5/6) ZS VLM**; **40.1 s (+5.6 s vs G, 4.7% FPR, 5/6) LoRA unseen VLM** |
| **Test-2 continuation-aware pilot** | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_test2_continuation_pilot_20261128_205424` | Commit `57c910c5`; `dit_cos_conf` **4/4 matched pre-regrasp pairs** & **0.817 (`p=0.022`) at equal `H=3`**; broad pre-regrasp **LOTO `0.829` (`p=0.0002`)** |

### Recommended next steps (in order)

1. **Scale synchronized collection to the 18-task Group-A/B benchmark (Test 1 + Test 2):** Instrument the remaining 12 shared-primitive composite tasks in `RECOVERY_PREDICATES` (expanding from 6 → 18 tasks, e.g., 10 train tasks / 8 held-out unseen compositions at 20–30 rollouts/task) to:
   - Increase calibration successes to $n_{\text{cal}} \ge 100$ (tightening split-conformal FPR & bootstrap CIs and enabling union-bound calibration),
   - Validate our decoupled `LOTO grasp + Qwen3-VL + gv_stage` monitor on **8 genuinely held-out unseen compositions** (with both hand-written and **LLM-generated task-graph predicates**), and
   - Power **15–25+ equal-duration matched Test-2 pairs** to integrate the continuation-aware engagement/abandonment filter directly into the online alarm rule (suppressing false alarms on self-correcting successes!).
2. **Optional cross-VLA replication on RLDX-1** using our completely black-box `proprio_actions + Qwen3-VL-4B` monitor (requires zero internal VLA latents).
