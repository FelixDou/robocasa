"""Analyze synchronized 6-task XR-1 recovery pilot recordings.

Reads one or more shard directories produced by
``scripts/run_xr1_recovery_pilot_6tasks.sh`` (or a single collection directory
containing ``manifest.jsonl`` and ``subtasks/``) and computes:

1. Step-level physical events (missed placements, lost conditions, wrong-tool
   grasps, and unassisted self-corrections) with strictly causal online score
   trajectories at every XR-1 policy decision.
2. Self-correction prevalence and candidate Test-2 matched pairs
   (multi-decision self-corrected successes vs. persistent failures on the same
   task and physical relation).
3. A 6-arm episode-level calibrated oracle comparison at nominal 5% FPR:
   ``global_timer`` (G) -> ``task_timer`` (P) -> ``stage_clock`` ->
   ``clocks_release_timer`` (blind post-release action clock control) ->
   ``clocks_loss`` -> ``full_physical``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


CONTROL_HZ = 20.0
ARMS = (
    "global_timer",
    "task_timer",
    "stage_clock",
    "clocks_release_timer",
    "clocks_loss",
    "full_physical",
)

# Pre-registered 18-task shared-primitive Group-A/B benchmark split:
#   - 10 Train/Dev tasks (6 pilot tasks + 4 appliance/fixture tasks covering dishwasher,
#     toaster, sink, and microwave primitives)
#   - 8 Held-Out Unseen Composition Test-1 tasks (4 Group A + 4 Group B)
PILOT6_TASKS = (
    "ArrangeBreadBasket",
    "CuttingToolSelection",
    "DeliverStraw",
    "GarnishPancake",
    "MakeIceLemonade",
    "SetUpCuttingStation",
)
BENCHMARK18_TRAIN_TASKS = PILOT6_TASKS + (
    "LoadDishwasher",
    "GetToastedBread",
    "PreSoakPan",
    "SteamInMicrowave",
)
BENCHMARK18_HELDOUT_TASKS = (
    "BreadSelection",
    "PackIdenticalLunches",
    "StackBowlsCabinet",
    "WashFruitColander",
    "ArrangeTea",
    "KettleBoiling",
    "SearingMeat",
    "WaffleReheat",
)
BENCHMARK18_NEW12_TASKS = (
    "LoadDishwasher",
    "GetToastedBread",
    "PreSoakPan",
    "SteamInMicrowave",
    "BreadSelection",
    "PackIdenticalLunches",
    "StackBowlsCabinet",
    "WashFruitColander",
    "ArrangeTea",
    "KettleBoiling",
    "SearingMeat",
    "WaffleReheat",
)
BENCHMARK18_ALL_TASKS = BENCHMARK18_TRAIN_TASKS + BENCHMARK18_HELDOUT_TASKS

# Declarative per-task physical event specs over RECOVERY_PREDICATES.
# Each placement tuple is (operation_name, grasp_predicates, target_predicate).
TASK_SPECS = {
    "ArrangeBreadBasket": {
        "placements": (
            ("place_bread", ("bread_grasped",), "bread_in_basket"),
            ("place_basket", ("basket_grasped",), "basket_on_dining_counter"),
        ),
        "hold_conditions": ("bread_in_basket", "basket_on_dining_counter"),
        "wrong_grasps": (),
    },
    "CuttingToolSelection": {
        "placements": (
            ("place_correct_tool", ("correct_tool_grasped",), "correct_tool_present_on_board"),
        ),
        "hold_conditions": ("wrong_tool_not_on_cutting_board", "correct_tool_present_on_board"),
        "wrong_grasps": ("wrong_tool_grasped",),
    },
    "DeliverStraw": {
        "placements": (
            ("place_straw", ("straw_grasped",), "straw_in_glass"),
        ),
        "hold_conditions": ("straw_in_glass",),
        "wrong_grasps": (),
    },
    "GarnishPancake": {
        "placements": (
            ("place_strawberry", ("strawberry_grasped",), "strawberry_on_pancake"),
        ),
        "hold_conditions": ("pancake_on_plate", "plate_on_table", "strawberry_on_pancake"),
        "wrong_grasps": (),
    },
    "MakeIceLemonade": {
        "placements": (
            ("place_lemon", ("lemon_grasped",), "lemon_in_glass"),
            ("place_ice", ("ice_cube1_grasped", "ice_cube2_grasped"), "ice_in_glass"),
        ),
        "hold_conditions": ("lemon_in_glass", "ice_in_glass"),
        "wrong_grasps": (),
    },
    "SetUpCuttingStation": {
        "placements": (
            ("place_knife", ("knife_grasped",), "knife_on_cutting_board"),
            ("place_meat", ("meat_grasped",), "meat_on_cutting_board"),
        ),
        "hold_conditions": ("knife_on_cutting_board", "meat_on_cutting_board"),
        "wrong_grasps": (),
    },
    # --- 12 additional shared-primitive Group-A/B tasks ---
    "LoadDishwasher": {
        "placements": (
            ("place_cup", ("cup_grasped",), "cup_on_rack"),
            ("place_bowl", ("bowl_grasped",), "bowl_on_rack"),
        ),
        "hold_conditions": ("cup_on_rack", "bowl_on_rack"),
        "wrong_grasps": (),
    },
    "BreadSelection": {
        "placements": (
            ("place_croissant", ("croissant_grasped",), "croissant_on_cutting_board"),
            ("place_jam", ("jam_grasped",), "jam_on_cutting_board"),
        ),
        "hold_conditions": ("croissant_on_cutting_board", "jam_on_cutting_board"),
        "wrong_grasps": (),
    },
    "PackIdenticalLunches": {
        "placements": (
            ("pack_veg0", ("vegetable0_grasped",), "vegetable0_in_tupperware"),
            ("pack_veg1", ("vegetable1_grasped",), "vegetable1_in_tupperware"),
            ("pack_meat0", ("meat0_grasped",), "meat0_in_tupperware"),
            ("pack_meat1", ("meat1_grasped",), "meat1_in_tupperware"),
        ),
        "hold_conditions": (
            "vegetable0_in_tupperware", "vegetable1_in_tupperware",
            "meat0_in_tupperware", "meat1_in_tupperware",
        ),
        "wrong_grasps": (),
    },
    "StackBowlsCabinet": {
        "placements": (
            ("place_larger_bowl", ("larger_bowl_grasped",), "larger_bowl_in_cabinet"),
            ("place_smaller_bowl", ("smaller_bowl_grasped",), "smaller_bowl_in_cabinet"),
        ),
        "hold_conditions": ("larger_bowl_in_cabinet", "smaller_bowl_in_cabinet", "bowls_stacked"),
        "wrong_grasps": (),
    },
    "WashFruitColander": {
        "placements": (
            ("place_colander", ("colander_grasped",), "colander_in_sink"),
            ("place_fruit0", ("fruit0_grasped",), "fruit0_in_colander"),
            ("place_fruit1", ("fruit1_grasped",), "fruit1_in_colander"),
            ("place_fruit2", ("fruit2_grasped",), "fruit2_in_colander"),
        ),
        "hold_conditions": ("colander_in_sink", "fruit_in_colander"),
        "wrong_grasps": (),
    },
    "GetToastedBread": {
        "placements": (
            ("place_bread", ("bread_grasped",), "bread_on_plate"),
        ),
        "hold_conditions": ("bread_on_plate",),
        "wrong_grasps": (),
    },
    "KettleBoiling": {
        "placements": (
            ("place_kettle", ("kettle_grasped",), "kettle_on_burner"),
        ),
        "hold_conditions": ("kettle_on_burner",),
        "wrong_grasps": (),
    },
    "PreSoakPan": {
        "placements": (
            ("place_pan", ("pan_grasped",), "pan_in_sink"),
            ("place_sponge", ("sponge_grasped",), "sponge_in_sink"),
        ),
        "hold_conditions": ("pan_in_sink", "sponge_in_sink"),
        "wrong_grasps": (),
    },
    "SearingMeat": {
        "placements": (
            ("place_pan", ("pan_grasped",), "pan_on_target_burner"),
            ("place_meat", ("meat_grasped",), "meat_in_pan"),
        ),
        "hold_conditions": ("pan_on_target_burner", "meat_in_pan"),
        "wrong_grasps": (),
    },
    "ArrangeTea": {
        "placements": (
            ("place_kettle", ("kettle_grasped",), "kettle_on_tray"),
            ("place_mug", ("mug_grasped",), "mug_on_tray"),
        ),
        "hold_conditions": ("kettle_on_tray", "mug_on_tray"),
        "wrong_grasps": (),
    },
    "SteamInMicrowave": {
        "placements": (
            ("place_vegetable", ("vegetable_grasped",), "vegetable_in_bowl"),
            ("place_bowl", ("bowl_grasped",), "bowl_in_microwave"),
        ),
        "hold_conditions": ("vegetable_in_bowl", "bowl_in_microwave"),
        "wrong_grasps": (),
    },
    "WaffleReheat": {
        # Waffle starts inside the bowl at reset; robot grasps and places the bowl into the microwave.
        "placements": (
            ("place_bowl", ("bowl_grasped",), "bowl_in_microwave"),
        ),
        "hold_conditions": ("waffle_in_bowl", "bowl_in_microwave"),
        "wrong_grasps": (),
    },
}


@dataclass
class EpisodeData:
    rollout_id: str
    task: str
    seed: int
    success: bool
    horizon: int
    num_steps: int
    decision_steps: np.ndarray
    decision_stages: list[str]
    stage_entries: dict[str, int]
    stage_durations: dict[str, int]
    release_only_age: np.ndarray
    missed_effect_age: np.ndarray
    loss_age: np.ndarray
    placement_delays: list[int]
    attempts: list[dict]
    losses: list[dict]
    wrong_grasp_events: list[dict]


def expand_recovery_series(history: list[dict], num_steps: int, predicate: str) -> np.ndarray:
    """Expand sparse change-point ``recovery_observations`` into bool array[0..num_steps]."""
    arr = np.zeros(num_steps + 1, dtype=bool)
    for idx, item in enumerate(history):
        start = int(item["environment_step"])
        end = int(history[idx + 1]["environment_step"]) if idx + 1 < len(history) else num_steps + 1
        val = bool(item["values"].get(predicate, False))
        arr[start:end] = val
    return arr


def _sustained_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Return inclusive-start, exclusive-end ``(start, end)`` runs of True with length >= min_len."""
    runs = []
    start = None
    for step, val in enumerate(mask):
        if val and start is None:
            start = step
        elif not val and start is not None:
            if step - start >= min_len:
                runs.append((start, step))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask)))
    return runs


def _first_decision_at_or_after(decision_steps: np.ndarray, step: int, before: int | None = None) -> int | None:
    idx = int(np.searchsorted(decision_steps, step, side="left"))
    if idx < len(decision_steps):
        cand = int(decision_steps[idx])
        if before is None or cand < before:
            return cand
    return None


def _count_decisions_in_interval(decision_steps: np.ndarray, start_incl: int, end_excl: int) -> int:
    lo = int(np.searchsorted(decision_steps, start_incl, side="left"))
    hi = int(np.searchsorted(decision_steps, end_excl, side="left"))
    return max(0, hi - lo)


def extract_episode(
    row: dict,
    trace: dict,
    *,
    min_grasp_steps: int = 16,
    latch_steps: int = 16,
    loss_confirm_steps: int = 16,
    miss_confirm_steps: int = 32,
) -> EpisodeData:
    task = row["task_name"]
    if task not in TASK_SPECS:
        raise ValueError(f"Unsupported task for pilot6 analyzer: {task}")
    spec = TASK_SPECS[task]
    rollout_id = row["rollout_id"]
    seed = int(row["environment_seed"])
    success = not bool(row["failed"])
    horizon = int(row["rollout_horizon"])
    num_steps = int(row["num_environment_steps"])
    history = trace["recovery_observations"]
    inf_records = trace["inference_records"]
    decision_steps = np.array([int(r["environment_step"]) for r in inf_records], dtype=int)
    decision_stages = [str(r["subtask_id"]) for r in inf_records]

    # Stage entries and completed stage durations from Subtask-SAFE segments.
    stage_entries: dict[str, int] = {}
    stage_durations: dict[str, int] = {}
    for seg in trace.get("segments", []):
        sid = seg.get("subtask_id")
        if sid is None:
            continue
        start_s = seg.get("entry_environment_step")
        end_s = seg.get("end_environment_step")
        if start_s is not None and sid not in stage_entries:
            stage_entries[sid] = int(start_s)
        if success and bool(seg.get("completed")) and start_s is not None and end_s is not None:
            stage_durations[sid] = max(stage_durations.get(sid, 0), int(end_s) - int(start_s))
    for step, stg in zip(decision_steps, decision_stages):
        if stg not in stage_entries:
            stage_entries[stg] = int(step)

    pred_series = {
        name: expand_recovery_series(history, num_steps, name)
        for name in history[0]["values"]
    }

    release_only_age = np.full(len(decision_steps), -np.inf, dtype=float)
    missed_effect_age = np.full(len(decision_steps), -np.inf, dtype=float)
    loss_age = np.full(len(decision_steps), -np.inf, dtype=float)

    attempts: list[dict] = []
    placement_delays: list[int] = []

    for op_name, grasp_preds, target_pred in spec["placements"]:
        any_grasp = np.zeros(num_steps + 1, dtype=bool)
        for gp in grasp_preds:
            any_grasp |= pred_series[gp]
        target = pred_series[target_pred]
        grasp_runs = _sustained_runs(any_grasp, min_grasp_steps)

        for idx, (g_start, g_end) in enumerate(grasp_runs):
            # If the grasp lasted all the way through episode termination, no release occurred.
            if g_end > num_steps:
                continue
            release_step = g_end  # First step where grasp became False.
            # Next regrasp of any object for this operation (any sustained grasp after release).
            next_grasp_start = grasp_runs[idx + 1][0] if idx + 1 < len(grasp_runs) else None
            # Also track the first raw contact/grasp onset after release (for offline recovery tracking).
            raw_after = np.flatnonzero(any_grasp[release_step + 1 :])
            next_raw_grasp = int(release_step + 1 + raw_after[0]) if raw_after.size else None

            # Blind release-only clock runs from release_step until next sustained grasp or episode end.
            blind_end = next_grasp_start if next_grasp_start is not None else num_steps + 1
            active_blind = (decision_steps >= release_step) & (decision_steps < blind_end)
            release_only_age[active_blind] = np.maximum(
                release_only_age[active_blind], decision_steps[active_blind] - release_step
            )

            # Check target truth at release and afterward.
            if bool(target[release_step]):
                placement_delays.append(0)
                attempts.append(
                    dict(
                        rollout_id=rollout_id,
                        task=task,
                        success=success,
                        operation=op_name,
                        relation=target_pred,
                        grasp_start=int(g_start),
                        release_step=int(release_step),
                        disposition="target_at_release",
                        disposition_step=int(release_step),
                        delay_steps=0,
                        confirmed_miss_step=None,
                        next_regrasp_step=next_grasp_start,
                        restored_step=int(release_step),
                        decisions_before_recovery=0,
                        classification="clean_placement",
                    )
                )
                continue

            # Search forward from release_step + 1 for first target arrival before next sustained grasp.
            search_end = next_grasp_start if next_grasp_start is not None else num_steps + 1
            arr_rel = np.flatnonzero(target[release_step + 1 : search_end])
            if arr_rel.size:
                arrival_step = int(release_step + 1 + arr_rel[0])
                delay = arrival_step - release_step
                placement_delays.append(delay)
                active_miss = (decision_steps >= release_step) & (decision_steps < arrival_step)
                missed_effect_age[active_miss] = np.maximum(
                    missed_effect_age[active_miss], decision_steps[active_miss] - release_step
                )
                confirmed = _first_decision_at_or_after(
                    decision_steps, release_step + miss_confirm_steps, before=arrival_step
                )
                dec_window = (
                    _count_decisions_in_interval(decision_steps, confirmed, arrival_step)
                    if confirmed is not None
                    else 0
                )
                classification = (
                    "clean_placement"
                    if confirmed is None
                    else ("delayed_placement_success" if success else "delayed_placement_failed_later")
                )
                attempts.append(
                    dict(
                        rollout_id=rollout_id,
                        task=task,
                        success=success,
                        operation=op_name,
                        relation=target_pred,
                        grasp_start=int(g_start),
                        release_step=int(release_step),
                        disposition="target_observed",
                        disposition_step=arrival_step,
                        delay_steps=delay,
                        confirmed_miss_step=confirmed,
                        next_regrasp_step=next_grasp_start,
                        restored_step=arrival_step,
                        decisions_before_recovery=dec_window,
                        classification=classification,
                    )
                )
                continue

            # Target did not arrive before next_grasp_start or episode end.
            active_miss = (decision_steps >= release_step) & (decision_steps < search_end)
            missed_effect_age[active_miss] = np.maximum(
                missed_effect_age[active_miss], decision_steps[active_miss] - release_step
            )
            confirmed = _first_decision_at_or_after(
                decision_steps, release_step + miss_confirm_steps, before=search_end
            )

            # Offline follow-up: did target ever become True later in the episode?
            later_arr = np.flatnonzero(target[release_step + 1 :])
            restored_step = int(release_step + 1 + later_arr[0]) if later_arr.size else None

            if next_grasp_start is not None:
                disposition = "regrasp_attempt"
                disposition_step = int(next_grasp_start)
                recovery_marker = restored_step if restored_step is not None else disposition_step
                dec_window = (
                    _count_decisions_in_interval(decision_steps, confirmed, recovery_marker)
                    if confirmed is not None
                    else 0
                )
                if confirmed is None:
                    classification = "quick_regrasp_before_confirm"
                elif success and restored_step is not None:
                    classification = (
                        "self_corrected_multi_decision_success"
                        if dec_window >= 2
                        else "quick_self_corrected_success"
                    )
                elif restored_step is not None:
                    classification = "restored_then_failed"
                else:
                    classification = "retry_then_failed"
            else:
                disposition = "unresolved_at_end"
                disposition_step = num_steps
                dec_window = (
                    _count_decisions_in_interval(decision_steps, confirmed, num_steps)
                    if confirmed is not None
                    else 0
                )
                if confirmed is None:
                    classification = "censored"
                elif success:
                    classification = "unresolved_in_success"
                else:
                    classification = "persistent_failure"

            attempts.append(
                dict(
                    rollout_id=rollout_id,
                    task=task,
                    success=success,
                    operation=op_name,
                    relation=target_pred,
                    grasp_start=int(g_start),
                    release_step=int(release_step),
                    disposition=disposition,
                    disposition_step=disposition_step,
                    delay_steps=None,
                    confirmed_miss_step=confirmed,
                    next_regrasp_step=next_grasp_start if next_grasp_start is not None else next_raw_grasp,
                    restored_step=restored_step,
                    decisions_before_recovery=dec_window,
                    classification=classification,
                )
            )

    # Lost conditions: once latched (True from step 0 or continuously True >= latch_steps),
    # track every drop interval.
    losses: list[dict] = []
    for cond in spec["hold_conditions"]:
        series = pred_series[cond]
        latched = bool(series[0])
        true_run = 1 if latched else 0
        in_loss = False
        loss_start = None
        for step in range(1, num_steps + 1):
            val = bool(series[step])
            if not latched:
                if val:
                    true_run += 1
                    if true_run >= latch_steps:
                        latched = True
                else:
                    true_run = 0
                continue
            if not val and not in_loss:
                in_loss = True
                loss_start = step
            elif val and in_loss:
                loss_end = step
                active_loss = (decision_steps >= loss_start) & (decision_steps < loss_end)
                loss_age[active_loss] = np.maximum(loss_age[active_loss], decision_steps[active_loss] - loss_start)
                confirmed = _first_decision_at_or_after(
                    decision_steps, loss_start + loss_confirm_steps, before=loss_end
                )
                if confirmed is not None:
                    dec_window = _count_decisions_in_interval(decision_steps, confirmed, loss_end)
                    if success:
                        cls = (
                            "self_corrected_multi_decision_success"
                            if dec_window >= 2
                            else "quick_self_corrected_success"
                        )
                    else:
                        cls = "restored_then_failed"
                    losses.append(
                        dict(
                            rollout_id=rollout_id,
                            task=task,
                            success=success,
                            operation=f"hold_{cond}",
                            relation=cond,
                            onset_step=int(loss_start),
                            confirmed_step=confirmed,
                            restored_step=int(loss_end),
                            duration_steps=int(loss_end - loss_start),
                            decisions_before_recovery=dec_window,
                            classification=cls,
                        )
                    )
                in_loss = False
                loss_start = None

        if in_loss and loss_start is not None:
            active_loss = decision_steps >= loss_start
            loss_age[active_loss] = np.maximum(loss_age[active_loss], decision_steps[active_loss] - loss_start)
            confirmed = _first_decision_at_or_after(decision_steps, loss_start + loss_confirm_steps)
            if confirmed is not None:
                dec_window = _count_decisions_in_interval(decision_steps, confirmed, num_steps)
                losses.append(
                    dict(
                        rollout_id=rollout_id,
                        task=task,
                        success=success,
                        operation=f"hold_{cond}",
                        relation=cond,
                        onset_step=int(loss_start),
                        confirmed_step=confirmed,
                        restored_step=None,
                        duration_steps=int(num_steps - loss_start),
                        decisions_before_recovery=dec_window,
                        classification="unresolved_in_success" if success else "persistent_failure",
                    )
                )

    # Wrong-object grasps (diagnostic).
    wrong_grasp_events: list[dict] = []
    for wg in spec["wrong_grasps"]:
        for start, end in _sustained_runs(pred_series[wg], min_grasp_steps):
            wrong_grasp_events.append(
                dict(
                    rollout_id=rollout_id,
                    task=task,
                    success=success,
                    predicate=wg,
                    start_step=int(start),
                    end_step=int(min(end, num_steps)),
                    duration_steps=int(min(end, num_steps) - start),
                )
            )

    return EpisodeData(
        rollout_id=rollout_id,
        task=task,
        seed=seed,
        success=success,
        horizon=horizon,
        num_steps=num_steps,
        decision_steps=decision_steps,
        decision_stages=decision_stages,
        stage_entries=stage_entries,
        stage_durations=stage_durations,
        release_only_age=release_only_age,
        missed_effect_age=missed_effect_age,
        loss_age=loss_age,
        placement_delays=placement_delays,
        attempts=attempts,
        losses=losses,
        wrong_grasp_events=wrong_grasp_events,
    )


def _conformal_threshold(maxima: list[float], alpha: float = 0.05) -> float | None:
    """Episode-level finite-sample order statistic ceil((1 - alpha) * (n + 1))."""
    n = len(maxima)
    if n == 0:
        return None
    rank = math.ceil((1.0 - alpha) * (n + 1))
    if rank > n:
        return None
    return max(0.0, float(sorted(maxima)[rank - 1]))


def _alarm_step(decision_steps: np.ndarray, scores: np.ndarray, cutoff: float | None) -> int | None:
    if cutoff is None:
        return None
    hits = np.flatnonzero(scores > cutoff)
    return int(decision_steps[hits[0]]) if hits.size else None


def _hash_order(episodes: list[EpisodeData], salt: str) -> list[EpisodeData]:
    return sorted(
        episodes,
        key=lambda e: hashlib.sha256(f"pilot6-v1/{salt}/{e.rollout_id}".encode()).digest(),
    )


def _fit_baselines(
    train_successes: list[EpisodeData],
    q: float = 0.95,
    loss_ref: int = 16,
) -> dict:
    by_task_dur: dict[str, list[int]] = defaultdict(list)
    by_stage_dur: dict[tuple[str, str], list[int]] = defaultdict(list)
    all_delays: list[int] = []
    by_task_delays: dict[str, list[int]] = defaultdict(list)

    for ep in train_successes:
        by_task_dur[ep.task].append(ep.num_steps)
        for sid, dur in ep.stage_durations.items():
            by_stage_dur[(ep.task, sid)].append(dur)
        for d in ep.placement_delays:
            all_delays.append(d)
            by_task_delays[ep.task].append(d)

    def quantile_val(vals: list[int], qv: float) -> float:
        s = sorted(vals)
        idx = max(0, min(len(s) - 1, math.ceil(qv * len(s)) - 1))
        return float(s[idx])

    task_q = {t: quantile_val(v, q) for t, v in by_task_dur.items() if v}
    stage_q = {k: quantile_val(v, q) for k, v in by_stage_dur.items() if len(v) >= 2}
    pooled_grace = max(1.0, quantile_val(all_delays, 0.9)) if all_delays else 16.0
    task_grace = {
        t: max(1.0, quantile_val(v, 0.9)) if len(v) >= 3 else pooled_grace
        for t, v in by_task_delays.items()
    }
    return {
        "task_q": task_q,
        "stage_q": stage_q,
        "task_grace": task_grace,
        "pooled_grace": pooled_grace,
        "loss_ref": float(loss_ref),
    }


def _arm_scores(ep: EpisodeData, params: dict) -> dict[str, np.ndarray]:
    h = float(ep.horizon)
    t = ep.decision_steps.astype(float)
    global_timer = t / h

    t_ref = params["task_q"].get(ep.task, h)
    task_timer = (t - t_ref + 1.0) / h

    stage_excess = np.full(len(t), -np.inf, dtype=float)
    for idx, (step, sid) in enumerate(zip(ep.decision_steps, ep.decision_stages)):
        s_ref = params["stage_q"].get((ep.task, sid))
        entry = ep.stage_entries.get(sid)
        if s_ref is not None and entry is not None and step >= entry:
            stage_excess[idx] = (float(step - entry) - s_ref + 1.0) / h
    stage_clock = np.maximum(task_timer, stage_excess)

    grace = params["task_grace"].get(ep.task, params["pooled_grace"])
    release_term = (ep.release_only_age - grace + 1.0) / h
    miss_term = (ep.missed_effect_age - grace + 1.0) / h
    loss_term = (ep.loss_age - params["loss_ref"] + 1.0) / h

    clocks_release_timer = np.maximum(stage_clock, release_term)
    clocks_loss = np.maximum(stage_clock, loss_term)
    full_physical = np.maximum(clocks_loss, miss_term)

    return {
        "global_timer": global_timer,
        "task_timer": task_timer,
        "stage_clock": stage_clock,
        "clocks_release_timer": clocks_release_timer,
        "clocks_loss": clocks_loss,
        "full_physical": full_physical,
    }


def _wilson95(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4 * n)) / n) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _paired_bootstrap_ci(
    failures: list[EpisodeData],
    alarms_a: dict[str, int | None],
    alarms_b: dict[str, int | None],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    by_task: dict[str, list[tuple[float, float]]] = defaultdict(list)
    earlier = later = 0
    for ep in failures:
        a = alarms_a[ep.rollout_id]
        b = alarms_b[ep.rollout_id]
        la = (ep.horizon - a) / ep.horizon if a is not None else 0.0
        lb = (ep.horizon - b) / ep.horizon if b is not None else 0.0
        if la > lb + 1e-12:
            earlier += 1
        elif la < lb - 1e-12:
            later += 1
        by_task[ep.task].append((la - lb, (la - lb) * ep.horizon / CONTROL_HZ))

    tasks = sorted(by_task)
    if not tasks:
        return {"mean_gain_frac": 0.0, "mean_gain_sec": 0.0, "ci95_frac": [0.0, 0.0], "earlier": 0, "later": 0}

    task_means_frac = {t: float(np.mean([x[0] for x in by_task[t]])) for t in tasks}
    task_means_sec = {t: float(np.mean([x[1] for x in by_task[t]])) for t in tasks}
    macro_frac = float(np.mean(list(task_means_frac.values())))
    macro_sec = float(np.mean(list(task_means_sec.values())))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b_idx in range(n_boot):
        chosen_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        vals = []
        for t in chosen_tasks:
            arr = np.array([x[0] for x in by_task[t]], dtype=float)
            vals.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
        draws[b_idx] = float(np.mean(vals))

    return {
        "mean_gain_frac": macro_frac,
        "mean_gain_sec": macro_sec,
        "ci95_frac": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "by_task_frac": task_means_frac,
        "by_task_sec": task_means_sec,
        "positive_tasks": sum(v > 0 for v in task_means_frac.values()),
        "total_failure_tasks": len(tasks),
        "earlier": earlier,
        "later": later,
    }


def evaluate_cross_fitted(
    episodes: list[EpisodeData],
    *,
    num_folds: int = 5,
    alpha: float = 0.05,
    loss_ref: int = 16,
) -> dict:
    # Stratified assignment of complete episodes into folds.
    groups: dict[tuple[bool, str], list[EpisodeData]] = defaultdict(list)
    for ep in episodes:
        groups[(ep.success, ep.task)].append(ep)
    folds: list[list[EpisodeData]] = [[] for _ in range(num_folds)]
    offset = 0
    for key in sorted(groups):
        for ep in _hash_order(groups[key], "fold"):
            folds[offset % num_folds].append(ep)
            offset += 1

    oof_alarms: dict[str, dict[str, int | None]] = {arm: {} for arm in ARMS}
    fold_records = []

    for fold_idx, held in enumerate(folds):
        held_ids = {e.rollout_id for e in held}
        train_succ = _hash_order([e for e in episodes if e.success and e.rollout_id not in held_ids], f"cal/{fold_idx}")
        # Reserve 2/3 of training successes for conformal calibration (at least 19 if available),
        # and 1/3 for duration/grace fitting; if total < 25, fall back to descriptive mode.
        n_total = len(train_succ)
        if n_total >= 25:
            n_cal = max(19, math.ceil(n_total * 2 / 3))
            n_cal = min(n_cal, n_total - 3)
            cal_pool = train_succ[:n_cal]
            fit_pool = train_succ[n_cal:]
            mode = "split_conformal"
        elif n_total >= 4:
            # Small smoke cohort: use training successes for both fitting and empirical 95% cutoff
            # (flagged explicitly as underpowered/descriptive).
            cal_pool = train_succ
            fit_pool = train_succ
            mode = "small_sample_empirical"
        else:
            cal_pool = []
            fit_pool = train_succ
            mode = "insufficient_successes"

        params = _fit_baselines(fit_pool, q=1.0 - alpha, loss_ref=loss_ref)
        cutoffs: dict[str, float | None] = {}
        for arm in ARMS:
            maxima = [float(np.max(_arm_scores(ep, params)[arm])) for ep in cal_pool]
            if mode == "split_conformal":
                cutoffs[arm] = _conformal_threshold(maxima, alpha=alpha)
            elif maxima:
                s = sorted(maxima)
                cutoffs[arm] = max(0.0, float(s[min(len(s) - 1, math.ceil((1.0 - alpha) * len(s)) - 1)]))
            else:
                cutoffs[arm] = None

        for ep in held:
            sc = _arm_scores(ep, params)
            for arm in ARMS:
                oof_alarms[arm][ep.rollout_id] = _alarm_step(ep.decision_steps, sc[arm], cutoffs[arm])

        fold_records.append(
            {
                "fold": fold_idx,
                "mode": mode,
                "n_eval": len(held),
                "n_fit_successes": len(fit_pool),
                "n_cal_successes": len(cal_pool),
                "pooled_grace_steps": params["pooled_grace"],
                "cutoffs": cutoffs,
            }
        )

    successes = [e for e in episodes if e.success]
    failures = [e for e in episodes if not e.success]
    arm_summary = {}
    for arm in ARMS:
        fa = sum(oof_alarms[arm][e.rollout_id] is not None for e in successes)
        tp = sum(oof_alarms[arm][e.rollout_id] is not None for e in failures)
        by_task_lead_frac = defaultdict(list)
        by_task_lead_sec = defaultdict(list)
        for ep in failures:
            a = oof_alarms[arm][ep.rollout_id]
            lf = (ep.horizon - a) / ep.horizon if a is not None else 0.0
            ls = (ep.horizon - a) / CONTROL_HZ if a is not None else 0.0
            by_task_lead_frac[ep.task].append(lf)
            by_task_lead_sec[ep.task].append(ls)
        macro_frac = float(np.mean([np.mean(v) for v in by_task_lead_frac.values()])) if by_task_lead_frac else 0.0
        macro_sec = float(np.mean([np.mean(v) for v in by_task_lead_sec.values()])) if by_task_lead_sec else 0.0
        arm_summary[arm] = {
            "failures_warned": tp,
            "total_failures": len(failures),
            "failure_recall": round(tp / len(failures), 4) if failures else 0.0,
            "false_alarms": fa,
            "total_successes": len(successes),
            "realized_fpr": round(fa / len(successes), 4) if successes else 0.0,
            "fpr_wilson95": _wilson95(fa, len(successes)),
            "macro_lead_frac": macro_frac,
            "macro_lead_sec": macro_sec,
            "by_task_lead_frac": {t: float(np.mean(v)) for t, v in sorted(by_task_lead_frac.items())},
        }

    paired = {
        "stage_clock_minus_task_timer": _paired_bootstrap_ci(
            failures, oof_alarms["stage_clock"], oof_alarms["task_timer"]
        ),
        "clocks_release_timer_minus_stage_clock": _paired_bootstrap_ci(
            failures, oof_alarms["clocks_release_timer"], oof_alarms["stage_clock"]
        ),
        "full_physical_minus_stage_clock": _paired_bootstrap_ci(
            failures, oof_alarms["full_physical"], oof_alarms["stage_clock"]
        ),
        "full_physical_minus_clocks_release_timer": _paired_bootstrap_ci(
            failures, oof_alarms["full_physical"], oof_alarms["clocks_release_timer"]
        ),
        "full_physical_minus_global_timer": _paired_bootstrap_ci(
            failures, oof_alarms["full_physical"], oof_alarms["global_timer"]
        ),
    }

    # Check how many episodes had identical alarms between full_physical and clocks_release_timer.
    diff_vs_release = sum(
        oof_alarms["full_physical"][e.rollout_id] != oof_alarms["clocks_release_timer"][e.rollout_id]
        for e in episodes
    )

    return {
        "calibration_modes": sorted({f["mode"] for f in fold_records}),
        "arms": arm_summary,
        "paired": paired,
        "episodes_full_differs_from_release_timer": diff_vs_release,
        "folds": fold_records,
        "oof_alarms": oof_alarms,
    }


def build_test2_pairs(episodes: list[EpisodeData]) -> dict:
    """Inventory confirmed mistakes, self-correction prevalence, and candidate Test-2 pairs."""
    confirmed_events = []
    for ep in episodes:
        for att in ep.attempts:
            if att["confirmed_miss_step"] is not None:
                confirmed_events.append(
                    {
                        "kind": "missed_placement",
                        "rollout_id": ep.rollout_id,
                        "task": ep.task,
                        "success": ep.success,
                        "operation": att["operation"],
                        "relation": att["relation"],
                        "onset_step": att["release_step"],
                        "confirmed_step": att["confirmed_miss_step"],
                        "restored_step": att["restored_step"],
                        "decisions_before_recovery": att["decisions_before_recovery"],
                        "classification": att["classification"],
                    }
                )
        for loss in ep.losses:
            confirmed_events.append(
                {
                    "kind": "lost_condition",
                    "rollout_id": ep.rollout_id,
                    "task": ep.task,
                    "success": ep.success,
                    "operation": loss["operation"],
                    "relation": loss["relation"],
                    "onset_step": loss["onset_step"],
                    "confirmed_step": loss["confirmed_step"],
                    "restored_step": loss["restored_step"],
                    "decisions_before_recovery": loss["decisions_before_recovery"],
                    "classification": loss["classification"],
                }
            )

    # First confirmed mistake per rollout (avoid double counting within an episode).
    first_per_ep = {}
    for ev in sorted(confirmed_events, key=lambda x: (x["confirmed_step"], x["onset_step"])):
        if ev["rollout_id"] not in first_per_ep:
            first_per_ep[ev["rollout_id"]] = ev
    first_events = list(first_per_ep.values())

    class_counts = Counter(ev["classification"] for ev in first_events)
    self_corr = [ev for ev in first_events if ev["classification"] == "self_corrected_multi_decision_success"]
    pers_fail = [ev for ev in first_events if ev["classification"] in ("persistent_failure", "retry_then_failed")]

    # Greedy 1-to-1 matching on (task, kind, operation, relation) by closest onset_step.
    used_fail = set()
    pairs = []
    for sc in sorted(self_corr, key=lambda x: (x["task"], x["onset_step"])):
        cands = [
            pf
            for pf in pers_fail
            if pf["rollout_id"] not in used_fail
            and (pf["task"], pf["kind"], pf["operation"], pf["relation"])
            == (sc["task"], sc["kind"], sc["operation"], sc["relation"])
        ]
        if not cands:
            continue
        best = min(cands, key=lambda pf: abs(pf["onset_step"] - sc["onset_step"]))
        used_fail.add(best["rollout_id"])
        pairs.append(
            {
                "task": sc["task"],
                "kind": sc["kind"],
                "operation": sc["operation"],
                "relation": sc["relation"],
                "success_rollout_id": sc["rollout_id"],
                "failure_rollout_id": best["rollout_id"],
                "success_confirmed_step": sc["confirmed_step"],
                "success_restored_step": sc["restored_step"],
                "failure_confirmed_step": best["confirmed_step"],
                "onset_gap_steps": abs(best["onset_step"] - sc["onset_step"]),
                "matched_window_decisions": sc["decisions_before_recovery"],
            }
        )

    n_succ = sum(e.success for e in episodes)
    n_total = len(episodes)
    sc_rate_among_succ = len(self_corr) / n_succ if n_succ else 0.0
    pair_rate_per_rollout = len(pairs) / n_total if n_total else 0.0
    return {
        "total_confirmed_events": len(confirmed_events),
        "first_event_rollouts": len(first_events),
        "first_event_classifications": dict(sorted(class_counts.items())),
        "multi_decision_self_corrected_successes": len(self_corr),
        "self_correction_rate_among_successes": round(sc_rate_among_succ, 4),
        "candidate_persistent_failures": len(pers_fail),
        "matched_test2_pairs": len(pairs),
        "projected_rollouts_for_30_pairs": math.ceil(30 / pair_rate_per_rollout) if pair_rate_per_rollout > 0 else None,
        "self_corrected_cases": self_corr,
        "pairs": pairs,
    }


def load_episodes(
    roots: list[Path],
    *,
    min_grasp_steps: int = 16,
    latch_steps: int = 16,
    loss_confirm_steps: int = 16,
    miss_confirm_steps: int = 32,
    tasks_filter: set[str] | None = None,
) -> tuple[list[EpisodeData], dict[str, str]]:
    manifest_paths = []
    for r in roots:
        if (r / "manifest.jsonl").exists():
            manifest_paths.append(r / "manifest.jsonl")
        else:
            manifest_paths.extend(sorted(r.glob("shard*/manifest.jsonl")))
    if not manifest_paths:
        raise FileNotFoundError(f"No manifest.jsonl found under {[str(r) for r in roots]}")

    episodes: list[EpisodeData] = []
    seen_ids = set()
    hashes: dict[str, str] = {}
    for mp in sorted(manifest_paths):
        base = mp.parent
        hashes[str(mp)] = hashlib.sha256(mp.read_bytes()).hexdigest()
        for line in mp.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row["rollout_id"]
            if rid in seen_ids:
                raise ValueError(f"Duplicate rollout_id across shards: {rid}")
            if tasks_filter is not None and row["task_name"] not in tasks_filter:
                continue
            seen_ids.add(rid)
            trace = json.loads((base / row["subtask_trace_path"]).read_text())
            episodes.append(
                extract_episode(
                    row,
                    trace,
                    min_grasp_steps=min_grasp_steps,
                    latch_steps=latch_steps,
                    loss_confirm_steps=loss_confirm_steps,
                    miss_confirm_steps=miss_confirm_steps,
                )
            )
    return episodes, hashes


def run_analysis(
    roots: list[Path],
    output_dir: Path,
    *,
    min_grasp_steps: int = 16,
    latch_steps: int = 16,
    loss_confirm_steps: int = 16,
    miss_confirm_steps: int = 32,
    num_folds: int = 5,
    tasks_filter: set[str] | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")
    episodes, input_hashes = load_episodes(
        roots,
        min_grasp_steps=min_grasp_steps,
        latch_steps=latch_steps,
        loss_confirm_steps=loss_confirm_steps,
        miss_confirm_steps=miss_confirm_steps,
        tasks_filter=tasks_filter,
    )
    by_task_counts = {}
    for t in sorted({e.task for e in episodes}):
        sub = [e for e in episodes if e.task == t]
        s = sum(e.success for e in sub)
        by_task_counts[t] = {"total": len(sub), "successes": s, "failures": len(sub) - s}

    all_delays = [d for e in episodes for d in e.placement_delays]
    delay_summary = (
        {
            "count": len(all_delays),
            "median_steps": float(np.median(all_delays)),
            "p90_steps": float(np.percentile(all_delays, 90)),
            "max_steps": int(max(all_delays)),
        }
        if all_delays
        else {"count": 0}
    )

    test2 = build_test2_pairs(episodes)
    oracle = evaluate_cross_fitted(episodes, num_folds=num_folds, loss_ref=loss_confirm_steps)
    oof_alarms = oracle.pop("oof_alarms")

    summary = {
        "schema_version": 1,
        "config": {
            "min_grasp_steps": min_grasp_steps,
            "latch_steps": latch_steps,
            "loss_confirm_steps": loss_confirm_steps,
            "miss_confirm_steps": miss_confirm_steps,
            "num_folds": num_folds,
        },
        "inputs_sha256": input_hashes,
        "cohort": {
            "total_rollouts": len(episodes),
            "successes": sum(e.success for e in episodes),
            "failures": sum(not e.success for e in episodes),
            "by_task": by_task_counts,
            "clean_placement_delays": delay_summary,
            "wrong_grasp_events": sum(len(e.wrong_grasp_events) for e in episodes),
        },
        "test2_readiness": test2,
        "oracle_hierarchy": oracle,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "out"
        staged.mkdir()
        (staged / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (staged / "rollouts.csv").open("w", newline="") as f:
            cols = [
                "rollout_id", "task", "seed", "success", "horizon", "num_steps",
                "num_attempts", "num_losses", "num_wrong_grasps", *ARMS,
            ]
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for ep in episodes:
                row = {
                    "rollout_id": ep.rollout_id,
                    "task": ep.task,
                    "seed": ep.seed,
                    "success": ep.success,
                    "horizon": ep.horizon,
                    "num_steps": ep.num_steps,
                    "num_attempts": len(ep.attempts),
                    "num_losses": len(ep.losses),
                    "num_wrong_grasps": len(ep.wrong_grasp_events),
                }
                for arm in ARMS:
                    val = oof_alarms[arm][ep.rollout_id]
                    row[arm] = "" if val is None else val
                writer.writerow(row)
        with (staged / "events.jsonl").open("w") as f:
            for ep in episodes:
                for att in ep.attempts:
                    f.write(json.dumps({"event_type": "attempt", **att}) + "\n")
                for loss in ep.losses:
                    f.write(json.dumps({"event_type": "loss", **loss}) + "\n")
                for wg in ep.wrong_grasp_events:
                    f.write(json.dumps({"event_type": "wrong_grasp", **wg}) + "\n")
        staged.rename(output_dir)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True, help="Run root(s) or shard directories")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--min-grasp-steps", type=int, default=16)
    parser.add_argument("--latch-steps", type=int, default=16)
    parser.add_argument("--loss-confirm-steps", type=int, default=16)
    parser.add_argument("--miss-confirm-steps", type=int, default=32)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional subset of tasks to analyze")
    args = parser.parse_args(argv)

    res = run_analysis(
        args.run,
        args.output,
        min_grasp_steps=args.min_grasp_steps,
        latch_steps=args.latch_steps,
        loss_confirm_steps=args.loss_confirm_steps,
        miss_confirm_steps=args.miss_confirm_steps,
        num_folds=args.num_folds,
        tasks_filter=set(args.tasks) if args.tasks else None,
    )
    c = res["cohort"]
    t2 = res["test2_readiness"]
    print(f"Cohort: {c['total_rollouts']} rollouts ({c['successes']}S / {c['failures']}F) across {len(c['by_task'])} tasks")
    print(f"Clean placement delays: {c['clean_placement_delays']}")
    print(
        f"Test-2 readiness: {t2['multi_decision_self_corrected_successes']} multi-decision self-corrected successes, "
        f"{t2['candidate_persistent_failures']} persistent failures, {t2['matched_test2_pairs']} matched pairs"
    )
    print(f"Oracle hierarchy ({res['oracle_hierarchy']['calibration_modes']}):")
    for arm, info in res["oracle_hierarchy"]["arms"].items():
        print(
            f"  {arm:22s} fail={info['failures_warned']:3d}/{info['total_failures']:<3d} "
            f"({info['failure_recall']*100:5.1f}%)  succ={info['false_alarms']:2d}/{info['total_successes']:<3d} "
            f"({info['realized_fpr']*100:4.1f}%)  lead={info['macro_lead_frac']*100:5.2f}% ({info['macro_lead_sec']:5.1f}s)"
        )
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
