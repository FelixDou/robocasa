"""Evaluate zero-shot Qwen3-VL destination verification and decoupled hybrid
monitors (fast proprio/XR-1 DiT grasp trigger + Qwen3-VL spatial destination
verifier) on the 120-rollout synchronized XR-1 pilot.

Measures:
1. Per-predicate and per-task zero-shot Qwen3-VL destination accuracy on both
   exact evaluated frames and full causal ZOH timelines:
   - AUROC and Average Precision (threshold-free)
   - Strict zero-shot F1 at logit_diff >= 0 (p = 0.5, ZERO label fitting)
   - Zero-shot LOTO-thresholded F1 (1 global scalar threshold fit on other 5 tasks)
   - Within-task 5-fold scalar-thresholded F1
   - Visibility-gated causal belief (updates EMA weighted by 1 - P(Unknown))
   - LOTO universal ``any_placement_satisfied`` AUROC/F1 (apples-to-apples with Section 8)
2. Component substitution & end-to-end 5-fold split-conformal lead at 5% nominal FPR:
   - ``oracle_reference`` (true grasp + true dest + true stage)
   - ``true_grasp_vlm_dest_zeroshot`` / ``true_grasp_vlm_dest_loto`` / ``true_grasp_vlm_dest_visgated``
   - ``hybrid_oof_dit_grasp_vlm_dest_zeroshot`` (ZERO privileged inputs)
   - ``hybrid_oof_dit_grasp_vlm_dest_loto`` (ZERO target-task dest labels, ZERO privileged inputs)
   - ``hybrid_oof_dit_grasp_vlm_dest_visgated`` (visibility-gated causal belief, ZERO privileged inputs)
   - ``hybrid_loto_universal_zeroshot`` (TRUE ZERO-SHOT cross-task transfer: LOTO universal
     ``xr1_dit`` grasp detector trained only on other 5 tasks + object-agnostic release progress
     verified by zero-shot Qwen3-VL + inferred stage clock!)
   - ``hybrid_vlm_plus_dit_blend`` (2-weight 5-fold logistic blend of VLM logit + DiT OOF logit)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import analyze_recovery_pilot6 as pilot6


def visibility_gated_belief(
    probs: np.ndarray,
    prob_visible: np.ndarray,
    *,
    init_prob: float = 0.0,
    ema_alpha: float = 0.65,
    min_vis: float = 0.15,
) -> np.ndarray:
    """Strictly causal visibility-weighted belief filter.

    When the VLM marks a frame as visible (``prob_visible -> 1``), updates the
    belief toward ``probs[i]`` with rate ``ema_alpha``. When the relevant object
    and destination are out of view (``prob_visible < min_vis``), holds the last
    visually observed belief instead of falsely dropping to 0.
    """
    p = np.asarray(probs, dtype=np.float32)
    v = np.asarray(prob_visible, dtype=np.float32)
    out = np.empty(len(p), dtype=np.float32)
    state = float(init_prob)
    for i in range(len(p)):
        vi = float(v[i])
        if vi >= min_vis:
            w = float(ema_alpha * vi)
            state = w * float(p[i]) + (1.0 - w) * state
        out[i] = state
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.where(
        arr >= 0.0,
        1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0))),
        np.exp(np.clip(arr, -40.0, 40.0)) / (1.0 + np.exp(np.clip(arr, -40.0, 40.0))),
    ).astype(np.float32)


def _best_f1_logit_threshold(y_true: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    best_th, best_f1 = 0.0, -1.0
    for th in np.linspace(-4.0, 4.0, 81):
        pred = logits >= th
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum((~pred) & (y_true == 1)))
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    return best_th, float(best_f1)


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y = y_true.astype(int)
    p = y_pred.astype(int)
    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def build_object_agnostic_episode(
    ep: dict,
    any_grasp_dec: np.ndarray,
    target_bool_map: dict[str, np.ndarray],
    *,
    use_inferred_stage: bool = True,
    min_grasp_steps: int = 16,
    latch_steps: int = 16,
    loss_confirm_steps: int = 16,
    miss_confirm_steps: int = 32,
) -> pilot6.EpisodeData:
    """Build an EpisodeData using ONLY a universal task-agnostic grasp signal +
    predicted destination/hold predicates (zero task-specific grasp labels!).

    At every sustained universal grasp ``[g_start, release_step)``, tracks the set
    of placement targets that were unsatisfied at ``g_start``. If at least one of
    those targets becomes satisfied at or after release (before the next sustained
    grasp), placement progress is verified and the missed-effect clock stops;
    otherwise ``missed_effect_age`` advances causally from ``release_step``.
    """
    task = ep["task"]
    spec = pilot6.TASK_SPECS[task]
    n_steps = int(ep["num_steps"])
    dec_steps = np.asarray(ep["decision_steps"], dtype=int)

    def _expand_dec_bool(b_dec: np.ndarray) -> np.ndarray:
        arr = np.zeros(n_steps + 1, dtype=bool)
        for i, t in enumerate(dec_steps):
            end = int(dec_steps[i + 1]) if i + 1 < len(dec_steps) else n_steps + 1
            arr[int(t) : end] = bool(b_dec[i])
        return arr

    any_grasp = _expand_dec_bool(any_grasp_dec)
    ordered_targets = [tp for _, _, tp in spec["placements"]]
    all_target_keys = sorted(set(ordered_targets) | set(spec["hold_conditions"]))
    target_series = {k: _expand_dec_bool(target_bool_map[k]) for k in all_target_keys}

    # Stage timeline (strictly inferred from predicted placement targets if requested).
    if use_inferred_stage:
        latched = {tp: False for tp in ordered_targets}
        consec = {tp: 0 for tp in ordered_targets}
        inferred_stages = []
        stage_entries: dict[str, int] = {}
        stage_durations: dict[str, int] = {}
        cur_stage = None
        cur_start = 0
        for t in dec_steps:
            for tp in ordered_targets:
                if target_series[tp][int(t)]:
                    consec[tp] += 1
                    if consec[tp] >= 2:
                        latched[tp] = True
                else:
                    consec[tp] = 0
            sid = f"inferred_stage_{sum(latched.values())}"
            inferred_stages.append(sid)
            if sid != cur_stage:
                if cur_stage is not None and ep["success"]:
                    stage_durations[cur_stage] = max(stage_durations.get(cur_stage, 0), int(t) - cur_start)
                if sid not in stage_entries:
                    stage_entries[sid] = int(t)
                cur_stage = sid
                cur_start = int(t)
        if cur_stage is not None and ep["success"]:
            stage_durations[cur_stage] = max(stage_durations.get(cur_stage, 0), n_steps - cur_start)
        decision_stages = inferred_stages
    else:
        decision_stages = list(ep["decision_stages"])
        stage_entries = {}
        stage_durations = {}
        for seg in ep.get("segments", []):
            sid = seg.get("subtask_id")
            if sid is None:
                continue
            s0, s1 = seg.get("entry_environment_step"), seg.get("end_environment_step")
            if s0 is not None and sid not in stage_entries:
                stage_entries[sid] = int(s0)
            if ep["success"] and bool(seg.get("completed")) and s0 is not None and s1 is not None:
                stage_durations[sid] = max(stage_durations.get(sid, 0), int(s1) - int(s0))
        for t, sid in zip(dec_steps, decision_stages):
            if sid not in stage_entries:
                stage_entries[sid] = int(t)

    release_only_age = np.full(len(dec_steps), -np.inf, dtype=float)
    missed_effect_age = np.full(len(dec_steps), -np.inf, dtype=float)
    loss_age = np.full(len(dec_steps), -np.inf, dtype=float)
    placement_delays: list[int] = []

    grasp_runs = pilot6._sustained_runs(any_grasp, min_grasp_steps)
    for idx, (g_start, g_end) in enumerate(grasp_runs):
        if g_end > n_steps:
            continue
        release_step = g_end
        next_grasp = grasp_runs[idx + 1][0] if idx + 1 < len(grasp_runs) else None
        search_end = next_grasp if next_grasp is not None else n_steps + 1

        active_blind = (dec_steps >= release_step) & (dec_steps < search_end)
        release_only_age[active_blind] = np.maximum(
            release_only_age[active_blind], dec_steps[active_blind] - release_step
        )

        # Targets that were unsatisfied when this grasp began.
        unsat = [tp for tp in ordered_targets if not bool(target_series[tp][g_start])]
        if not unsat:
            continue
        # Progress achieved if ANY previously-unsatisfied target is satisfied at release.
        if any(bool(target_series[tp][release_step]) for tp in unsat):
            placement_delays.append(0)
            continue

        any_arrived = np.zeros(search_end - (release_step + 1), dtype=bool)
        for tp in unsat:
            any_arrived |= target_series[tp][release_step + 1 : search_end]
        arr_rel = np.flatnonzero(any_arrived)
        if arr_rel.size:
            arrival_step = int(release_step + 1 + arr_rel[0])
            placement_delays.append(arrival_step - release_step)
            active_miss = (dec_steps >= release_step) & (dec_steps < arrival_step)
            missed_effect_age[active_miss] = np.maximum(
                missed_effect_age[active_miss], dec_steps[active_miss] - release_step
            )
        else:
            active_miss = (dec_steps >= release_step) & (dec_steps < search_end)
            missed_effect_age[active_miss] = np.maximum(
                missed_effect_age[active_miss], dec_steps[active_miss] - release_step
            )

    # Lost conditions across hold_conditions (identical semantics to pilot6.extract_episode)
    for cond in spec["hold_conditions"]:
        series = target_series[cond]
        latched = bool(series[0])
        true_run = 1 if latched else 0
        in_loss = False
        loss_start = None
        for step in range(1, n_steps + 1):
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
                active_loss = (dec_steps >= loss_start) & (dec_steps < step)
                loss_age[active_loss] = np.maximum(loss_age[active_loss], dec_steps[active_loss] - loss_start)
                in_loss = False
                loss_start = None
        if in_loss and loss_start is not None:
            active_loss = dec_steps >= loss_start
            loss_age[active_loss] = np.maximum(loss_age[active_loss], dec_steps[active_loss] - loss_start)

    return pilot6.EpisodeData(
        rollout_id=ep["rollout_id"],
        task=task,
        seed=int(ep["seed"]),
        success=bool(ep["success"]),
        horizon=int(ep["horizon"]),
        num_steps=n_steps,
        decision_steps=dec_steps,
        decision_stages=decision_stages,
        stage_entries=stage_entries,
        stage_durations=stage_durations,
        release_only_age=release_only_age,
        missed_effect_age=missed_effect_age,
        loss_age=loss_age,
        placement_delays=placement_delays,
        attempts=[],
        losses=[],
        wrong_grasp_events=[],
    )


def _fit_grasp_and_dit_oof(
    arrays: dict[str, np.ndarray],
    episodes: list[dict],
    fold_of: dict[str, int],
    *,
    grasp_stream: str = "xr1_dit",
    hidden_dim: int = 128,
    epochs: int = 100,
    num_folds: int = 5,
    device,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
]:
    """Fit:
    1. Within-task 5-fold OOF task-specific grasp booleans from ``grasp_stream``.
    2. LOTO zero-shot universal ``any_object_grasped`` booleans from ``grasp_stream``
       (trained strictly on the OTHER 5 tasks!).
    3. Within-task 5-fold OOF destination probabilities from ``grasp_stream`` (for VLM+DiT blend).
    """
    from . import eval_perception_floor as pfloor

    X_all = pfloor.build_stream_matrix(arrays, episodes, grasp_stream)
    tasks_sorted = sorted({e["task"] for e in episodes})

    # 1. Within-task OOF task-specific grasps + destination probs
    oof_grasp_bool: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    oof_dest_prob: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for task in tasks_sorted:
        task_eps = [e for e in episodes if e["task"] == task]
        spec = pilot6.TASK_SPECS[task]
        grasp_keys = sorted({gp for _, gps, _ in spec["placements"] for gp in gps} | set(spec["wrong_grasps"]))
        dest_keys = sorted({tp for _, _, tp in spec["placements"]} | set(spec["hold_conditions"]))
        for pname in grasp_keys + dest_keys:
            is_grasp = pname in grasp_keys
            for f in range(num_folds):
                tr_eps = [e for e in task_eps if fold_of[e["rollout_id"]] != f]
                ev_eps = [e for e in task_eps if fold_of[e["rollout_id"]] == f]
                if not tr_eps or not ev_eps:
                    continue
                X_tr = np.concatenate([X_all[e["start_row"] : e["end_row"]] for e in tr_eps], axis=0)
                y_tr = np.concatenate([np.asarray(e["predicates_at_decisions"][pname], dtype=np.float32) for e in tr_eps])
                X_ev = np.concatenate([X_all[e["start_row"] : e["end_row"]] for e in ev_eps], axis=0)
                p_tr, p_ev = pfloor.fit_predict_probe(
                    X_tr, y_tr, X_ev, device=device, hidden_dim=hidden_dim, epochs=epochs, seed=f
                )
                th, _ = pfloor._best_f1_threshold(y_tr, p_tr)
                cur = 0
                for e in ev_eps:
                    ln = e["end_row"] - e["start_row"]
                    ep_p = p_ev[cur : cur + ln]
                    cur += ln
                    if is_grasp:
                        oof_grasp_bool[e["rollout_id"]][pname] = pfloor._causal_smooth_and_threshold(ep_p, th)
                    else:
                        oof_dest_prob[e["rollout_id"]][pname] = ep_p

    # 2. LOTO zero-shot universal grasp detector (strictly trained on other 5 tasks)
    univ_y = {}
    for ep in episodes:
        spec = pilot6.TASK_SPECS[ep["task"]]
        g_keys = [gp for _, gps, _ in spec["placements"] for gp in gps]
        g_any = np.zeros(ep["end_row"] - ep["start_row"], dtype=bool)
        for k in g_keys:
            g_any |= np.asarray(ep["predicates_at_decisions"][k], dtype=bool)
        univ_y[ep["rollout_id"]] = g_any.astype(np.float32)

    loto_any_grasp_bool: dict[str, np.ndarray] = {}
    for held_task in tasks_sorted:
        tr_eps = [e for e in episodes if e["task"] != held_task]
        ev_eps = [e for e in episodes if e["task"] == held_task]
        X_tr = np.concatenate([X_all[e["start_row"] : e["end_row"]] for e in tr_eps], axis=0)
        y_tr = np.concatenate([univ_y[e["rollout_id"]] for e in tr_eps])
        X_ev = np.concatenate([X_all[e["start_row"] : e["end_row"]] for e in ev_eps], axis=0)
        p_tr, p_ev = pfloor.fit_predict_probe(
            X_tr, y_tr, X_ev, device=device, hidden_dim=hidden_dim, epochs=epochs, seed=0
        )
        th, _ = pfloor._best_f1_threshold(y_tr, p_tr)
        cur = 0
        for e in ev_eps:
            ln = e["end_row"] - e["start_row"]
            ep_p = p_ev[cur : cur + ln]
            cur += ln
            loto_any_grasp_bool[e["rollout_id"]] = pfloor._causal_smooth_and_threshold(ep_p, th)

    return oof_grasp_bool, loto_any_grasp_bool, oof_dest_prob


def run_hybrid_evaluation(
    bundle_dir: Path,
    vlm_dir: Path,
    output_dir: Path,
    *,
    grasp_stream: str = "xr1_dit",
    grasp_hidden_dim: int = 128,
    grasp_epochs: int = 100,
    num_folds: int = 5,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")

    import torch
    from . import eval_perception_floor as pfloor

    b_index_path = bundle_dir / "perception_index.json"
    b_npz_path = bundle_dir / "perception_features.npz"
    v_index_path = vlm_dir / "vlm_destination_index.json"
    v_npz_path = vlm_dir / "vlm_destination_scores.npz"

    b_index = json.loads(b_index_path.read_text())
    b_arrays = dict(np.load(b_npz_path))
    v_index = json.loads(v_index_path.read_text())
    v_arrays = dict(np.load(v_npz_path))

    v_by_rid = {e["rollout_id"]: e for e in v_index["episodes"]}
    episodes = [e for e in b_index["episodes"] if e["rollout_id"] in v_by_rid]
    if not episodes:
        raise RuntimeError("No overlapping rollout_ids between perception bundle and VLM scores")

    fold_of = pfloor._episode_folds(episodes, num_folds=num_folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks_sorted = sorted({e["task"] for e in episodes})

    # 1. Fit fast grasp detectors (within-task OOF + zero-shot LOTO universal) & DiT dest probs
    oof_grasp_bool, loto_any_grasp_bool, oof_dit_dest_prob = _fit_grasp_and_dit_oof(
        b_arrays,
        episodes,
        fold_of,
        grasp_stream=grasp_stream,
        hidden_dim=grasp_hidden_dim,
        epochs=grasp_epochs,
        num_folds=num_folds,
        device=device,
    )

    # 2. Compute zero-shot VLM metrics (per predicate, per task, and LOTO universal)
    # Also construct four causal boolean target-predicate tracks per rollout:
    #   (a) vlm_zeroshot: logit_diff >= 0.0 (p >= 0.5, ZERO label fitting)
    #   (b) vlm_loto: global scalar logit threshold fit on other 5 tasks
    #   (c) vlm_visgated: visibility-gated causal belief >= LOTO probability threshold
    #   (d) vlm_within_cal: within-task 5-fold scalar logit threshold
    #   (e) vlm_plus_dit: within-task 5-fold 2-feature blend of [vlm_logit, dit_logit]
    vlm_tracks: dict[str, dict[str, dict[str, np.ndarray]]] = {
        k: defaultdict(dict)
        for k in ("vlm_zeroshot", "vlm_loto", "vlm_visgated", "vlm_within_cal", "vlm_plus_dit")
    }

    # Pre-collect sampled arrays per task for LOTO global bias threshold fitting
    task_sampled_y: dict[str, list[np.ndarray]] = defaultdict(list)
    task_sampled_ld: dict[str, list[np.ndarray]] = defaultdict(list)
    for ep in episodes:
        rid = ep["rollout_id"]
        task = ep["task"]
        s_pos = np.asarray(v_by_rid[rid]["sampled_positions"], dtype=int)
        for pname in v_by_rid[rid]["predicates"]:
            y_all = np.asarray(ep["predicates_at_decisions"][pname], dtype=int)
            ld_s = v_arrays[f"{rid}__{pname}__sampled_logit_diff"]
            task_sampled_y[task].append(y_all[s_pos])
            task_sampled_ld[task].append(ld_s)

    loto_bias_th_by_task: dict[str, float] = {}
    for held_task in tasks_sorted:
        other_y = np.concatenate([arr for t in tasks_sorted if t != held_task for arr in task_sampled_y[t]])
        other_ld = np.concatenate([arr for t in tasks_sorted if t != held_task for arr in task_sampled_ld[t]])
        th_loto, _ = _best_f1_logit_threshold(other_y, other_ld)
        loto_bias_th_by_task[held_task] = th_loto

    per_task_vlm_metrics: dict[str, dict] = {}
    macro_sampled_aurocs, macro_zoh_aurocs = [], []
    macro_zs_f1, macro_loto_f1, macro_vis_f1, macro_within_f1, macro_blend_f1 = [], [], [], [], []
    macro_dest_only_aurocs, macro_dest_only_zs_f1, macro_dest_only_loto_f1, macro_dest_only_vis_f1 = [], [], [], []

    for task in tasks_sorted:
        task_eps = [e for e in episodes if e["task"] == task]
        spec = pilot6.TASK_SPECS[task]
        placement_targets = {tp for _, _, tp in spec["placements"]}
        pnames = v_by_rid[task_eps[0]["rollout_id"]]["predicates"]
        loto_th = loto_bias_th_by_task[task]
        loto_prob_th = float(_sigmoid(np.array([loto_th]))[0])
        per_pred_out = {}

        for pname in pnames:
            # Initial prior for visibility-gated belief:
            # Known reset-True preconditions (hold-only predicates like wrong_tool_not_on_cutting_board,
            # pancake_on_plate, plate_on_table) start at 1.0 when not visible at reset; placement targets start at 0.0.
            init_p = 0.0 if pname in placement_targets else 1.0

            # Within-task 5-fold scalar threshold (strictly out-of-fold) +
            # leak-free equal-weight log-odds blend of (centered VLM logit + OOF DiT logit).
            within_th_by_ep = {}
            blend_bool_by_ep = {}
            for f in range(num_folds):
                tr_eps = [e for e in task_eps if fold_of[e["rollout_id"]] != f]
                ev_eps = [e for e in task_eps if fold_of[e["rollout_id"]] == f]
                if not tr_eps or not ev_eps:
                    continue
                y_tr = np.concatenate([np.asarray(e["predicates_at_decisions"][pname], dtype=int) for e in tr_eps])
                ld_tr = np.concatenate([v_arrays[f"{e['rollout_id']}__{pname}__zoh_logit_diff"] for e in tr_eps])
                th_w, _ = _best_f1_logit_threshold(y_tr, ld_tr)

                for e in ev_eps:
                    rid = e["rollout_id"]
                    within_th_by_ep[rid] = th_w
                    p_dit = np.clip(oof_dit_dest_prob[rid][pname], 1e-4, 1.0 - 1e-4)
                    dit_logit = np.log(p_dit / (1.0 - p_dit)).astype(np.float32)
                    vlm_centered = v_arrays[f"{rid}__{pname}__zoh_logit_diff"] - th_w
                    blend_prob = _sigmoid(0.5 * vlm_centered + 0.5 * dit_logit)
                    blend_bool_by_ep[rid] = pfloor._causal_smooth_and_threshold(blend_prob, 0.5)

            s_y_list, s_ld_list, s_pv_list = [], [], []
            z_y_list, z_ld_list, z_zs_list, z_loto_list, z_vis_list, z_within_list, z_blend_list = (
                [], [], [], [], [], [], []
            )
            for ep in task_eps:
                rid = ep["rollout_id"]
                s_pos = np.asarray(v_by_rid[rid]["sampled_positions"], dtype=int)
                y_all = np.asarray(ep["predicates_at_decisions"][pname], dtype=int)
                s_y_list.append(y_all[s_pos])
                s_ld_list.append(v_arrays[f"{rid}__{pname}__sampled_logit_diff"])
                s_pv_list.append(v_arrays[f"{rid}__{pname}__sampled_prob_visible"])

                z_ld = v_arrays[f"{rid}__{pname}__zoh_logit_diff"]
                z_py = v_arrays[f"{rid}__{pname}__zoh_prob_yes"]
                z_pv = v_arrays[f"{rid}__{pname}__zoh_prob_visible"]

                b_zs = pfloor._causal_smooth_and_threshold(z_py, 0.5)
                b_loto = pfloor._causal_smooth_and_threshold(_sigmoid(z_ld - loto_th), 0.5)
                vis_bel = visibility_gated_belief(z_py, z_pv, init_prob=init_p, ema_alpha=0.65, min_vis=0.15)
                b_vis = vis_bel >= loto_prob_th
                b_within = pfloor._causal_smooth_and_threshold(_sigmoid(z_ld - within_th_by_ep[rid]), 0.5)
                b_blend = blend_bool_by_ep[rid]

                vlm_tracks["vlm_zeroshot"][rid][pname] = b_zs
                vlm_tracks["vlm_loto"][rid][pname] = b_loto
                vlm_tracks["vlm_visgated"][rid][pname] = b_vis
                vlm_tracks["vlm_within_cal"][rid][pname] = b_within
                vlm_tracks["vlm_plus_dit"][rid][pname] = b_blend

                z_y_list.append(y_all)
                z_ld_list.append(z_ld)
                z_zs_list.append(b_zs)
                z_loto_list.append(b_loto)
                z_vis_list.append(b_vis)
                z_within_list.append(b_within)
                z_blend_list.append(b_blend)

            s_y = np.concatenate(s_y_list)
            s_ld = np.concatenate(s_ld_list)
            s_pv = np.concatenate(s_pv_list)
            z_y = np.concatenate(z_y_list)
            z_ld = np.concatenate(z_ld_list)

            s_auroc = pfloor._auroc(s_y, s_ld)
            z_auroc = pfloor._auroc(z_y, z_ld)
            s_ap = pfloor._average_precision(s_y, s_ld)
            m_zs = _binary_f1(z_y, np.concatenate(z_zs_list))
            m_loto = _binary_f1(z_y, np.concatenate(z_loto_list))
            m_vis = _binary_f1(z_y, np.concatenate(z_vis_list))
            m_within = _binary_f1(z_y, np.concatenate(z_within_list))
            m_blend = _binary_f1(z_y, np.concatenate(z_blend_list))

            entry = {
                "is_placement_target": pname in placement_targets,
                "n_sampled": int(len(s_y)),
                "pos_sampled": int(s_y.sum()),
                "sampled_auroc": round(s_auroc, 4) if s_auroc is not None else None,
                "sampled_ap": round(s_ap, 4) if s_ap is not None else None,
                "zoh_auroc": round(z_auroc, 4) if z_auroc is not None else None,
                "mean_prob_visible": round(float(s_pv.mean()), 4),
                "loto_logit_threshold": round(float(loto_th), 3),
                "zeroshot_0p5": m_zs,
                "loto_threshold": m_loto,
                "visgated_loto": m_vis,
                "within_5fold_cal": m_within,
                "vlm_plus_dit_blend": m_blend,
            }
            per_pred_out[pname] = entry

            if s_auroc is not None:
                macro_sampled_aurocs.append(s_auroc)
            if z_auroc is not None:
                macro_zoh_aurocs.append(z_auroc)
            macro_zs_f1.append(m_zs["f1"])
            macro_loto_f1.append(m_loto["f1"])
            macro_vis_f1.append(m_vis["f1"])
            macro_within_f1.append(m_within["f1"])
            macro_blend_f1.append(m_blend["f1"])
            if pname in placement_targets:
                if s_auroc is not None:
                    macro_dest_only_aurocs.append(s_auroc)
                macro_dest_only_zs_f1.append(m_zs["f1"])
                macro_dest_only_loto_f1.append(m_loto["f1"])
                macro_dest_only_vis_f1.append(m_vis["f1"])

        per_task_vlm_metrics[task] = per_pred_out

    # Universal LOTO ``any_placement_satisfied`` (max logit over task's placement targets)
    univ_by_task = {}
    univ_aurocs, univ_zs_f1s, univ_loto_f1s = [], [], []
    for task in tasks_sorted:
        task_eps = [e for e in episodes if e["task"] == task]
        ptargets = [tp for _, _, tp in pilot6.TASK_SPECS[task]["placements"]]
        y_list, ld_list = [], []
        for ep in task_eps:
            rid = ep["rollout_id"]
            y_any = np.zeros(ep["end_row"] - ep["start_row"], dtype=bool)
            lds = []
            for tp in ptargets:
                y_any |= np.asarray(ep["predicates_at_decisions"][tp], dtype=bool)
                lds.append(v_arrays[f"{rid}__{tp}__zoh_logit_diff"])
            y_list.append(y_any.astype(int))
            ld_list.append(np.max(np.stack(lds, axis=0), axis=0))
        y_c = np.concatenate(y_list)
        ld_c = np.concatenate(ld_list)
        auc = pfloor._auroc(y_c, ld_c)
        f1_zs = _binary_f1(y_c, ld_c >= 0.0)["f1"]
        f1_loto = _binary_f1(y_c, ld_c >= loto_bias_th_by_task[task])["f1"]
        univ_by_task[task] = {
            "auroc": round(auc, 4) if auc is not None else None,
            "zeroshot_0p5_f1": f1_zs,
            "loto_f1": f1_loto,
        }
        if auc is not None:
            univ_aurocs.append(auc)
        univ_zs_f1s.append(f1_zs)
        univ_loto_f1s.append(f1_loto)

    # 3. End-to-end substitution and fully-observed hybrid monitor evaluation (5% nominal FPR)
    def _eval_and_trim(ep_objs: list[pilot6.EpisodeData]) -> dict:
        ev = pilot6.evaluate_cross_fitted(ep_objs, num_folds=num_folds)
        ev.pop("oof_alarms", None)
        ev.pop("folds", None)
        return ev

    monitor_arms: dict[str, dict] = {}

    # 3a. Oracle reference
    oracle_eps = [
        pfloor._build_substituted_episode(ep, {}, replace_grasps=False, replace_targets=False, use_inferred_stage=False)
        for ep in episodes
    ]
    monitor_arms["oracle_reference"] = _eval_and_trim(oracle_eps)

    # 3b. Component substitution: true grasp + VLM dest (isolates VLM destination quality)
    for track in ("vlm_zeroshot", "vlm_loto", "vlm_visgated", "vlm_within_cal", "vlm_plus_dit"):
        eps_tg = [
            pfloor._build_substituted_episode(
                ep,
                vlm_tracks[track][ep["rollout_id"]],
                replace_grasps=False,
                replace_targets=True,
                use_inferred_stage=False,
            )
            for ep in episodes
        ]
        monitor_arms[f"true_grasp_{track}"] = _eval_and_trim(eps_tg)

    # 3c. Fully observed hybrid monitor: OOF DiT grasps + VLM dest + inferred stage (ZERO privileged inputs)
    for track in ("vlm_zeroshot", "vlm_loto", "vlm_visgated", "vlm_within_cal", "vlm_plus_dit"):
        eps_hyb = []
        for ep in episodes:
            rid = ep["rollout_id"]
            merged = dict(oof_grasp_bool[rid])
            merged.update(vlm_tracks[track][rid])
            eps_hyb.append(
                pfloor._build_substituted_episode(
                    ep,
                    merged,
                    replace_grasps=True,
                    replace_targets=True,
                    use_inferred_stage=True,
                )
            )
        monitor_arms[f"hybrid_dit_grasp_{track}"] = _eval_and_trim(eps_hyb)

    # 3d. TRUE ZERO-SHOT CROSS-TASK UNIVERSAL MONITOR:
    # LOTO universal xr1_dit grasp (trained strictly on other 5 tasks) + object-agnostic Qwen3-VL destination check!
    for track in ("vlm_zeroshot", "vlm_loto", "vlm_visgated"):
        eps_univ = [
            build_object_agnostic_episode(
                ep,
                loto_any_grasp_bool[ep["rollout_id"]],
                vlm_tracks[track][ep["rollout_id"]],
                use_inferred_stage=True,
            )
            for ep in episodes
        ]
        monitor_arms[f"zero_shot_loto_universal_{track}"] = _eval_and_trim(eps_univ)

    summary = {
        "schema_version": 1,
        "bundle_dir": str(bundle_dir),
        "vlm_dir": str(vlm_dir),
        "bundle_index_sha256": hashlib.sha256(b_index_path.read_bytes()).hexdigest(),
        "vlm_index_sha256": hashlib.sha256(v_index_path.read_bytes()).hexdigest(),
        "config": {
            "grasp_stream": grasp_stream,
            "grasp_hidden_dim": grasp_hidden_dim,
            "grasp_epochs": grasp_epochs,
            "num_folds": num_folds,
            "vlm_decision_stride": v_index.get("decision_stride"),
            "vlm_model_dir": v_index.get("model_dir"),
        },
        "vlm_destination_metrics": {
            "macro_all_sampled_auroc": round(float(np.mean(macro_sampled_aurocs)), 4) if macro_sampled_aurocs else None,
            "macro_all_zoh_auroc": round(float(np.mean(macro_zoh_aurocs)), 4) if macro_zoh_aurocs else None,
            "macro_placement_only_auroc": round(float(np.mean(macro_dest_only_aurocs)), 4) if macro_dest_only_aurocs else None,
            "macro_all_zeroshot_0p5_f1": round(float(np.mean(macro_zs_f1)), 4),
            "macro_all_loto_f1": round(float(np.mean(macro_loto_f1)), 4),
            "macro_all_visgated_f1": round(float(np.mean(macro_vis_f1)), 4),
            "macro_all_within_cal_f1": round(float(np.mean(macro_within_f1)), 4),
            "macro_all_vlm_plus_dit_f1": round(float(np.mean(macro_blend_f1)), 4),
            "macro_placement_only_zeroshot_0p5_f1": round(float(np.mean(macro_dest_only_zs_f1)), 4),
            "macro_placement_only_loto_f1": round(float(np.mean(macro_dest_only_loto_f1)), 4),
            "macro_placement_only_visgated_f1": round(float(np.mean(macro_dest_only_vis_f1)), 4),
            "by_task": per_task_vlm_metrics,
        },
        "universal_any_placement_loto": {
            "macro_auroc": round(float(np.mean(univ_aurocs)), 4) if univ_aurocs else None,
            "macro_zeroshot_0p5_f1": round(float(np.mean(univ_zs_f1s)), 4),
            "macro_loto_f1": round(float(np.mean(univ_loto_f1s)), 4),
            "by_task": univ_by_task,
        },
        "monitor_lead_5pct_fpr": monitor_arms,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "out"
        staged.mkdir()
        (staged / "vlm_hybrid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        staged.rename(output_dir)

    vm = summary["vlm_destination_metrics"]
    print("=== ZERO-SHOT QWEN3-VL DESTINATION VERIFICATION ===")
    print(
        f"  All 12 predicates:   sampled_AUROC={vm['macro_all_sampled_auroc']}  "
        f"zs@0.5_F1={vm['macro_all_zeroshot_0p5_f1']:.3f}  loto_F1={vm['macro_all_loto_f1']:.3f}  "
        f"visgated_F1={vm['macro_all_visgated_f1']:.3f}  within_F1={vm['macro_all_within_cal_f1']:.3f}  "
        f"vlm+dit_F1={vm['macro_all_vlm_plus_dit_f1']:.3f}"
    )
    print(
        f"  Placement targets:   sampled_AUROC={vm['macro_placement_only_auroc']}  "
        f"zs@0.5_F1={vm['macro_placement_only_zeroshot_0p5_f1']:.3f}  "
        f"loto_F1={vm['macro_placement_only_loto_f1']:.3f}  "
        f"visgated_F1={vm['macro_placement_only_visgated_f1']:.3f}"
    )
    up = summary["universal_any_placement_loto"]
    print(
        f"  Universal any_place: AUROC={up['macro_auroc']}  "
        f"zs@0.5_F1={up['macro_zeroshot_0p5_f1']:.3f}  loto_F1={up['macro_loto_f1']:.3f}"
    )
    print("\n=== PER-PREDICATE BREAKDOWN ===")
    for task, pmap in sorted(per_task_vlm_metrics.items()):
        for pname, m in pmap.items():
            tag = "DEST" if m["is_placement_target"] else "HOLD"
            print(
                f"  [{tag}] {task:20s} {pname:32s} AUROC={str(m['sampled_auroc']):>6s} "
                f"vis={m['mean_prob_visible']:.2f} zs_F1={m['zeroshot_0p5']['f1']:.3f} "
                f"loto_F1={m['loto_threshold']['f1']:.3f} vis_F1={m['visgated_loto']['f1']:.3f} "
                f"blend_F1={m['vlm_plus_dit_blend']['f1']:.3f}"
            )
    print("\n=== END-TO-END HYBRID MONITOR LEAD (5% NOMINAL FPR) ===")
    for name, ev in monitor_arms.items():
        fp = ev["arms"]["full_physical"]
        sc = ev["arms"]["stage_clock"]
        vs_sc = ev["paired"]["full_physical_minus_stage_clock"]
        vs_g = ev["paired"]["full_physical_minus_global_timer"]
        print(
            f"  {name:38s} full={fp['macro_lead_frac']*100:5.2f}% ({fp['macro_lead_sec']:4.1f}s) "
            f"FPR={fp['realized_fpr']*100:4.1f}% rec={fp['failure_recall']*100:4.1f}% | "
            f"vs_stage={vs_sc['mean_gain_frac']*100:+5.2f}pp | "
            f"vs_G={vs_g['mean_gain_frac']*100:+5.2f}pp ({vs_g['mean_gain_sec']:+4.1f}s)"
        )
    print("Wrote:", output_dir / "vlm_hybrid_summary.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--vlm-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grasp-stream", default="xr1_dit")
    parser.add_argument("--grasp-hidden-dim", type=int, default=128)
    parser.add_argument("--grasp-epochs", type=int, default=100)
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args(argv)
    run_hybrid_evaluation(
        args.bundle,
        args.vlm_scores,
        args.output,
        grasp_stream=args.grasp_stream,
        grasp_hidden_dim=args.grasp_hidden_dim,
        grasp_epochs=args.grasp_epochs,
        num_folds=args.num_folds,
    )


if __name__ == "__main__":
    main()
