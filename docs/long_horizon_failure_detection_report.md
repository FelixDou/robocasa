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
4. **First live synchronized XR-1 recording and visual inspection (`xr1_recovery_recording_PNL4QPfz`; Section 6).** Four fresh passive rollouts (`2 × ArrangeBreadBasket`, `2 × CuttingToolSelection`) passed end-to-end verification (`PASS: four synchronized XR-1 recordings`, exact 1-to-1 decoded frame match: 4,350/4,350 and 1,200/1,200 at 20.0 fps). Visual frame inspection against `recovery_observations` uncovered **three critical simulator-predicate defects** in RoboCasa's existing helpers:
   - `OU.check_obj_grasped(..., threshold=0.035)` produced a **complete false negative** while XR-1 visibly grasped, lifted, and carried a wide bread loaf into the basket (`00072ff15e9a5bb3695e4bcd`, steps 700–851);
   - `OU.check_obj_grasped` uses `env.check_contact(gripper, obj) and gripper_closed` (any contact while closed, not opposing-finger grasp), producing a **false positive** on `wrong_tool_grasped` when the closed gripper holding the knife grazed the adjacent peeler (`cd28746d88039ebcffc5a123`, step 973) and 22 brief grazing pulses on `basket_grasped` (`00072ff15e9a5bb3695e4bcd`);
   - `is_open` stayed `False` throughout **3 of 4 episodes** (`1dd3b4ce`, `f8d2c686`, `cd28746d`) even though the cabinet/drawer was visually open wide enough for XR-1 to reach inside and manipulate objects.

**Current decision:** Do **not** scale up multi-task data collection or camera model training until the grasp-predicate defect (`check_obj_grasped`) is repaired/verified and an action-event / missed-grasp baseline is added alongside the stage clock.

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

- **Validation:** `RoboCasa SAFE dataset: VALID` (12 rollouts: **9 successes, 3 failures**; 42 usable segments: 39 successes, 3 failures, 0 errors), `PASS: 12 synchronized XR-1 recordings`, exact 1-to-1 decoded frame sync on every episode.
- **Per-rollout opposing-pad `_check_grasp` summary:**

| Task | Rollout ID | Outcome | Frames / Decs | Obs changes | Grasp onsets (`_check_grasp`) |
|---|---|---|---:|---:|---|
| `ArrangeBreadBasket` | `aa214332a8c17e2f576155da` | **succeeded** | 1,015 / 64 | 14 | `bread_grasped: 1`, `basket_grasped: 2` |
| `ArrangeBreadBasket` | `eaa2bfe17ae17ca418363127` | failed | 4,350 / 272 | 12 | `bread_grasped: 4`, `basket_grasped: 1` |
| `CuttingToolSelection` | `0991c30bf37e02796a17b0cf` | **succeeded** | 308 / 20 | 4 | `correct_tool_grasped: 1`, `wrong_tool_grasped: 0` |
| `CuttingToolSelection` | `5a4fdad1b1e1fc325653009f` | failed | 1,200 / 75 | 7 | `correct_tool_grasped: 1`, `wrong_tool_grasped: 1` |
| `DeliverStraw` | `25c61d7219b98c6503e93a52` | **succeeded** | 948 / 60 | 19 | `straw_grasped: 2` |
| `DeliverStraw` | `1c0979ccc457e27459793396` | **succeeded** | 1,009 / 64 | 8 | `straw_grasped: 1` |
| `GarnishPancake` | `147b5ad937f776e90ce6e5cc` | **succeeded** | 1,248 / 78 | 13 | `strawberry_grasped: 1` |
| `GarnishPancake` | `1d703e912d8c0752d4d48182` | failed | 2,700 / 169 | 11 | `strawberry_grasped: 1` |
| `MakeIceLemonade` | `d285494e886a902cf64b5a26` | **succeeded** | 2,034 / 128 | 44 | `lemon: 1`, `ice_cube1: 1`, `ice_cube2: 1` |
| `MakeIceLemonade` | `99438606c14f973d368b6bf4` | **succeeded** | 1,532 / 96 | 20 | `lemon: 1`, `ice_cube1: 0`, `ice_cube2: 1` |
| `SetUpCuttingStation` | `87bbbc908c954b2806152b6e` | **succeeded** | 1,673 / 105 | 15 | `knife_grasped: 2`, `meat_grasped: 1` |
| `SetUpCuttingStation` | `b1b33b2b23925a92fbd93609` | **succeeded** | 798 / 50 | 13 | `knife_grasped: 1`, `meat_grasped: 1` |

- **Key validation result:** The `_check_grasp` fix worked immediately on live XR-1:
  - `bread_grasped` now fires cleanly on wide bread loaves (`1` on success, `4` on failure — previously `0` throughout `00072ff1`);
  - `basket_grasped` dropped from 23 grazing pulses (`00072ff1`) to `1`–`2` real opposing-pad grasps;
  - **3 of 9 successful episodes** (`ArrangeBreadBasket aa214332`, `DeliverStraw 25c61d72`, `SetUpCuttingStation 87bbbc90`) recorded **multiple grasp onsets (`2`)** — strong candidate unassisted retries to inspect once `metadata_review.tar.gz` is synced locally.
- **Decision:** Gate passed — proceed immediately to the **full 6-task pilot** (`NUM_ROLLOUTS=20`/task = 120 rollouts, `seed = 2800007`, `video_frame_stride = 2`).

---

## 7. Artifact index and next experiments

### Reproducibility & artifact index

| Artifact | Location | Notes / SHA-256 |
|---|---|---|
| Roadmap document | `reports/long_horizon_failure_detection_roadmap.docx` | Initial 27 Nov 2026 roadmap |
| Prototype event audit | `docs/long_horizon_relation_events.md`, `outputs/long_horizon_failure_detection/xr1_ladder_76/summary.json` | 76 October parents |
| Calibrated pilot (internal fit) | `docs/long_horizon_calibration_pilot.md`, `outputs/long_horizon_failure_detection/xr1_ladder_76_calibration_v1/calibration.json` | +6.42 pp, 2/36 vs 1/36 succ |
| Calibrated pilot (external naturals) | `outputs/long_horizon_failure_detection/xr1_ladder_76_calibration_external_naturals/calibration.json` | `a3282acb...`; +6.37 pp, 2/36 vs 1/36 succ |
| External naturals archive | `outputs/long_horizon_failure_detection/cluster_imports/xr1_naturals_fit.tar.gz` | `9dbc1fff...`; 60 parents (25 succ) |
| Local release-clock controls | `docs/long_horizon_local_controls.md`, `outputs/long_horizon_failure_detection/xr1_ladder_76_local_controls/results.json` | 0/76 alarm diff at 32–256 steps; 0 verified pairs |
| Cluster cohort audit | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/long_horizon_cohort_audit_20261128_130837.json` | Legacy August cohorts lack video/actions/raw predicates |
| Synchronized XR-1 smoke run | `/gs/bs/tga-shinoda/felid/robocasa_rollouts/safe/xr1_recovery_recording_PNL4QPfz` | Commit `9e59c463`; 4/4 verified; archive `cc8999a5...` |
| Visual inspection contact sheets | `outputs/long_horizon_failure_detection/xr1_recovery_recording_PNL4QPfz_inspection/*.png` | 4 contact sheets (Section 6.2) |
| 6-task pilot launcher & verifier | `scripts/run_xr1_recovery_pilot_6tasks.sh`, `scripts/xr1_lead_time/verify_recovery_recording.py` | 34/34 unit tests pass; ready to push & run |

### Recommended next steps (in order)

1. **Commit & push the repaired `_grasped` + 6-task pilot code**, then run **`--smoke` (2 rollouts/task, seed `2700007`)** on the cluster to confirm opposing-pad `_check_grasp` fires cleanly on `bread_grasped` and eliminates grazing false positives across all 6 tasks.
2. **Run the full 6-task synchronized passive pilot (`NUM_ROLLOUTS=20`/task = 120 rollouts, seed `2800007`, stride 2, 2–3 GPUs)** to measure:
   - true rate of multi-decision unassisted self-corrections (including action-anchored missed-grasp retries) to decide whether Test 2 is sufficiently powered;
   - oracle lead hierarchy: `P` → `stage clock` → `grasp/release action clock` → `+ lost condition & missed effect` with repaired predicates;
   - **perception floor check:** evaluate a frozen vision-language model / SigLIP probe on 10–20 synchronized videos from the pilot (wrist + agentview) before building any custom training pipeline.
