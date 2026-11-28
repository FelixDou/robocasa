"""18-Task Group-A/B Test-1 Evaluation (10 Train/Dev -> 8 Held-Out Unseen Tasks).

Evaluates the complete decoupled physical-consequence monitor under strict
10-Train -> 8-Held-Out unseen-task transfer, including:
1. Universal primitive transfer (trained strictly on 10 Train tasks -> evaluated
   strictly zero-shot on 8 Held-Out tasks):
   - ``any_object_grasped`` (MLP-128 on chosen ``--grasp-stream``)
   - Baseline ``any_placement_satisfied`` probes (``siglip_3cam``, ``proprio_actions``, ``xr1_dit``)
   - Destination verification AUROC / post-release AUROC / @0.5 F1 for base zero-shot
     ``Qwen3-VL-4B-Instruct`` vs. 10-Train LoRA ``Qwen3-VL-4B-Instruct``.
2. Episode-level calibrated warning lead at 5% nominal FPR on ALL 8 Held-Out Test-1
   tasks (160 rollouts) comparing:
   - Oracle hierarchy (``global_timer``, ``task_timer``, ``stage_clock``,
     ``clocks_release_timer``, ``full_physical``)
   - ``grasp_stage_only`` (zero VLM)
   - ``univ_gv_stage`` + **Base Zero-Shot VLM** (hand prompts)
   - ``univ_gv_stage`` + **10-Train LoRA VLM** (hand prompts)
   - ``univ_gv_stage`` + **100% Automated LLM-Generated Task-Graph Zero-Shot VLM**
   - ``univ_gv_stage`` + **100% Automated LLM-Generated Task-Graph 10-Train LoRA VLM**
   - Component substitution (``true_grasp + VLM``) isolating visual verification ceiling
3. Two strict calibration protocols:
   - **Protocol A**: 5-fold split-conformal within the 8 Held-Out tasks (both standard
     pooled AND non-vacuous channel-separated union-bound ``alpha/2 + alpha/2 = 0.05``).
   - **Protocol B (True Zero-Shot Threshold Transfer)**: Calibrate cutoffs and
     task-agnostic normalized stage durations / grace strictly on the 10 Train tasks'
     successes (ZERO held-out successes used even for calibration!) and evaluate
     directly on all 160 Held-Out rollouts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import analyze_recovery_pilot6 as pilot6
from . import eval_vlm_scoring_upgrades as vupg


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(y), dtype=float)
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _binary_metrics(y_true: np.ndarray, pred_bool: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pred_bool, dtype=bool)
    tp = int(np.sum(p & (y == 1)))
    fp = int(np.sum(p & (y == 0)))
    fn = int(np.sum((~p) & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def build_arbitrary_target_monitor_episode(
    ep: dict,
    *,
    any_grasp_dec: np.ndarray,
    target_prob_map: dict[str, np.ndarray],
    target_bool_map: dict[str, np.ndarray],
    target_keys: list[str],
    stage_mode: str = "grasp_and_vlm",
    phys_mode: str = "hard_binary",
    hold_policy: str = "verified_targets_only",
    clear_threshold: float = 0.70,
    gamma: float = 1.0,
    min_grasp_steps: int = 16,
    latch_steps: int = 16,
    min_grasp_decisions: int = 1,
) -> pilot6.EpisodeData:
    """Strictly causal, 100% object-agnostic monitor over ANY list of visual target keys.

    Works identically whether ``target_keys`` are hand-specified placement predicates
    from ``TASK_SPECS`` OR arbitrary LLM-generated question IDs (``llm_goal_0..K``)
    produced solely from the observable ``task_instruction``.
    """
    task = ep["task"]
    n_steps = int(ep["num_steps"])
    dec_steps = np.asarray(ep["decision_steps"], dtype=int)
    n_dec = len(dec_steps)
    ordered_targets = [k for k in target_keys if k in target_prob_map and k in target_bool_map]

    def _expand_dec_bool(b_dec: np.ndarray) -> np.ndarray:
        arr = np.zeros(n_steps + 1, dtype=bool)
        for i, t in enumerate(dec_steps):
            end = int(dec_steps[i + 1]) if i + 1 < n_dec else n_steps + 1
            arr[int(t) : end] = bool(b_dec[i])
        return arr

    ema_prob_dec = {k: vupg.causal_ema(target_prob_map[k], alpha=0.65) for k in ordered_targets}
    target_bool_step = {k: _expand_dec_bool(target_bool_map[k]) for k in ordered_targets}
    any_g_arr = np.asarray(any_grasp_dec, dtype=bool)

    # 1. Observable causal stage clock
    if stage_mode == "none":
        decision_stages = ["single_stage"] * n_dec
        stage_entries: dict[str, int] = {"single_stage": 0}
        stage_durations: dict[str, int] = {}
    elif stage_mode in ("grasp_only", "grasp_and_vlm"):
        include_vlm = stage_mode == "grasp_and_vlm"
        latched = {tp: False for tp in ordered_targets}
        consec_t = {tp: 0 for tp in ordered_targets}
        consec_g = 0
        grasped_in_phase = False
        prev_n_placed = 0
        decision_stages = []
        stage_entries = {}
        stage_durations = {}
        cur_stage: str | None = None
        cur_start = 0

        for i, t in enumerate(dec_steps):
            if include_vlm:
                for tp in ordered_targets:
                    is_hi = (
                        bool(target_bool_map[tp][i])
                        if phys_mode == "hard_binary"
                        else (float(ema_prob_dec[tp][i]) >= clear_threshold)
                    )
                    if is_hi:
                        consec_t[tp] += 1
                        if consec_t[tp] >= 2:
                            latched[tp] = True
                    else:
                        consec_t[tp] = 0
            n_placed = sum(latched.values()) if include_vlm else 0
            if n_placed > prev_n_placed:
                grasped_in_phase = False
                consec_g = 0
                prev_n_placed = n_placed

            if bool(any_g_arr[i]):
                consec_g += 1
                if consec_g >= min_grasp_decisions:
                    grasped_in_phase = True
            else:
                consec_g = 0

            sid = f"gv_p{n_placed}_g{int(grasped_in_phase)}" if include_vlm else f"g_stage_{int(grasped_in_phase)}"
            decision_stages.append(sid)
            if sid != cur_stage:
                if cur_stage is not None and ep["success"]:
                    stage_durations[cur_stage] = max(stage_durations.get(cur_stage, 0), int(t) - cur_start)
                if sid not in stage_entries:
                    stage_entries[sid] = int(t)
                cur_stage = sid
                cur_start = int(t)
        if cur_stage is not None and ep["success"]:
            stage_durations[cur_stage] = max(stage_durations.get(cur_stage, 0), n_steps - cur_start)
    else:
        raise ValueError(f"Unknown stage_mode: {stage_mode}")

    # 2. Strictly causal release -> missed-effect accumulation over unsatisfied targets
    release_only_age = np.full(n_dec, -np.inf, dtype=float)
    missed_effect_age = np.full(n_dec, -np.inf, dtype=float)
    loss_age = np.full(n_dec, -np.inf, dtype=float)
    placement_delays: list[int] = []

    if ordered_targets:
        any_grasp_step = _expand_dec_bool(any_g_arr)
        runs = pilot6._sustained_runs(any_grasp_step, min_grasp_steps)
        for idx, (g_start, g_end) in enumerate(runs):
            if g_end > n_steps:
                continue
            next_g = runs[idx + 1][0] if idx + 1 < len(runs) else None
            search_end = next_g if next_g is not None else n_steps + 1
            g_dec_i = max(0, int(np.searchsorted(dec_steps, g_start, side="right")) - 1)
            if phys_mode == "hard_binary":
                unsat = [tp for tp in ordered_targets if not bool(target_bool_step[tp][g_start])]
            else:
                unsat = [tp for tp in ordered_targets if float(ema_prob_dec[tp][g_dec_i]) < clear_threshold]
            if not unsat:
                continue

            active_idx = np.flatnonzero((dec_steps >= g_end) & (dec_steps < search_end))
            if active_idx.size == 0:
                continue
            release_only_age[active_idx] = np.maximum(
                release_only_age[active_idx], dec_steps[active_idx] - g_end
            )

            if phys_mode == "hard_binary":
                if any(bool(target_bool_step[tp][g_end]) for tp in unsat):
                    placement_delays.append(0)
                else:
                    any_arr = np.zeros(search_end - (g_end + 1), dtype=bool)
                    for tp in unsat:
                        any_arr |= target_bool_step[tp][g_end + 1 : search_end]
                    rel_hits = np.flatnonzero(any_arr)
                    if rel_hits.size:
                        arr_step = int(g_end + 1 + rel_hits[0])
                        placement_delays.append(arr_step - g_end)
                        act = (dec_steps >= g_end) & (dec_steps < arr_step)
                        missed_effect_age[act] = np.maximum(missed_effect_age[act], dec_steps[act] - g_end)
                    else:
                        missed_effect_age[active_idx] = np.maximum(
                            missed_effect_age[active_idx], dec_steps[active_idx] - g_end
                        )
            else:
                best_p = 0.0
                cum_ev = 0.0
                prev_t = g_end
                cleared_recorded = False
                for d_i in active_idx:
                    t_cur = int(dec_steps[d_i])
                    p_now = max(float(ema_prob_dec[tp][d_i]) for tp in unsat)
                    if p_now > best_p:
                        best_p = p_now
                    if not cleared_recorded and best_p >= clear_threshold:
                        placement_delays.append(max(0, t_cur - g_end))
                        cleared_recorded = True
                    dt = max(0, t_cur - prev_t)
                    deficit = max(0.0, 1.0 - best_p / max(1e-3, clear_threshold))
                    if gamma != 1.0:
                        deficit = deficit ** gamma
                    cum_ev += float(dt) * deficit
                    if cum_ev > missed_effect_age[d_i]:
                        missed_effect_age[d_i] = cum_ev
                    prev_t = t_cur

    # 3. Strictly causal verified-target hold loss (never pre-latched at step 0)
    if hold_policy == "verified_targets_only":
        for cond in ordered_targets:
            series = target_bool_step[cond]
            latched_h = False
            true_run = 0
            in_loss = False
            loss_start = None
            for step in range(1, n_steps + 1):
                val = bool(series[step])
                if not latched_h:
                    if val:
                        true_run += 1
                        if true_run >= latch_steps:
                            latched_h = True
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
    elif hold_policy != "placement_only":
        raise ValueError(f"Unsupported hold_policy for arbitrary targets: {hold_policy}")

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


def _summarize_alarm_dict(episodes: list[pilot6.EpisodeData], alarms: dict[str, int | None]) -> dict:
    successes = [e for e in episodes if e.success]
    failures = [e for e in episodes if not e.success]
    fa = sum(alarms[e.rollout_id] is not None for e in successes)
    tp = sum(alarms[e.rollout_id] is not None for e in failures)
    by_t_f = defaultdict(list)
    by_t_s = defaultdict(list)
    for ep in failures:
        a = alarms[ep.rollout_id]
        by_t_f[ep.task].append((ep.horizon - a) / ep.horizon if a is not None else 0.0)
        by_t_s[ep.task].append((ep.horizon - a) / pilot6.CONTROL_HZ if a is not None else 0.0)
    by_task_summary = {}
    for t in sorted({e.task for e in episodes}):
        ts = [e for e in successes if e.task == t]
        tf = [e for e in failures if e.task == t]
        by_task_summary[t] = {
            "successes": len(ts),
            "false_alarms": sum(alarms[e.rollout_id] is not None for e in ts),
            "failures": len(tf),
            "failures_warned": sum(alarms[e.rollout_id] is not None for e in tf),
            "lead_frac": round(float(np.mean(by_t_f[t])), 4) if t in by_t_f else None,
            "lead_sec": round(float(np.mean(by_t_s[t])), 2) if t in by_t_s else None,
        }
    return {
        "failures_warned": tp,
        "total_failures": len(failures),
        "failure_recall": round(tp / len(failures), 4) if failures else 0.0,
        "false_alarms": fa,
        "total_successes": len(successes),
        "realized_fpr": round(fa / len(successes), 4) if successes else 0.0,
        "fpr_wilson95": pilot6._wilson95(fa, len(successes)),
        "macro_lead_frac": float(np.mean([np.mean(v) for v in by_t_f.values()])) if by_t_f else 0.0,
        "macro_lead_sec": float(np.mean([np.mean(v) for v in by_t_s.values()])) if by_t_s else 0.0,
        "by_task": by_task_summary,
    }


def evaluate_union_bound_split_conformal_adaptive(
    episodes: list[pilot6.EpisodeData],
    *,
    num_folds: int = 5,
    alpha: float = 0.05,
    alpha_stage_frac: float = 0.5,
    loss_ref: int = 16,
) -> dict:
    """Exact finite-sample union-bound split-conformal (``alpha/2 + alpha/2 = alpha``)
    ensuring ``n_cal >= ceil(1 / (alpha/2)) - 1 = 39`` whenever enough training successes exist.
    """
    groups: dict[tuple[bool, str], list[pilot6.EpisodeData]] = defaultdict(list)
    for ep in episodes:
        groups[(ep.success, ep.task)].append(ep)
    folds: list[list[pilot6.EpisodeData]] = [[] for _ in range(num_folds)]
    offset = 0
    for key in sorted(groups):
        for ep in pilot6._hash_order(groups[key], "fold"):
            folds[offset % num_folds].append(ep)
            offset += 1

    a_stg = alpha * alpha_stage_frac
    a_phys = alpha * (1.0 - alpha_stage_frac)
    min_n_cal_union = math.ceil(1.0 / min(a_stg, a_phys)) - 1  # 39 at alpha=0.05, 50/50 split

    oof_union: dict[str, int | None] = {}
    oof_stage: dict[str, int | None] = {}
    oof_global: dict[str, int | None] = {}
    oof_phys_only: dict[str, int | None] = {}
    cal_sizes = []

    for fold_idx, held in enumerate(folds):
        held_ids = {e.rollout_id for e in held}
        train_succ = pilot6._hash_order(
            [e for e in episodes if e.success and e.rollout_id not in held_ids],
            f"cal/{fold_idx}",
        )
        n_total = len(train_succ)
        if n_total >= min_n_cal_union + 8:
            n_cal = min(n_total - 8, max(min_n_cal_union, math.ceil(n_total * 2 / 3)))
        else:
            n_cal = min(max(1, n_total - 3), max(19, math.ceil(n_total * 2 / 3)))
        cal_pool = train_succ[:n_cal]
        fit_pool = train_succ[n_cal:]
        cal_sizes.append(len(cal_pool))
        params = pilot6._fit_baselines(fit_pool, q=1.0 - alpha, loss_ref=loss_ref)

        def _phys_score(ep: pilot6.EpisodeData) -> np.ndarray:
            h = float(ep.horizon)
            grace = params["task_grace"].get(ep.task, params["pooled_grace"])
            miss_term = (ep.missed_effect_age - grace + 1.0) / h
            loss_term = (ep.loss_age - params["loss_ref"] + 1.0) / h
            return np.maximum(miss_term, loss_term)

        g_max = [float(np.max(pilot6._arm_scores(ep, params)["global_timer"])) for ep in cal_pool]
        s_max = [float(np.max(pilot6._arm_scores(ep, params)["stage_clock"])) for ep in cal_pool]
        p_max = [float(np.max(_phys_score(ep))) for ep in cal_pool]

        tau_g = pilot6._conformal_threshold(g_max, alpha=alpha)
        tau_s_full = pilot6._conformal_threshold(s_max, alpha=alpha)
        tau_s_half = pilot6._conformal_threshold(s_max, alpha=a_stg)
        tau_p_half = pilot6._conformal_threshold(p_max, alpha=a_phys)
        tau_p_full = pilot6._conformal_threshold(p_max, alpha=alpha)

        for ep in held:
            sc = pilot6._arm_scores(ep, params)
            ps = _phys_score(ep)
            oof_global[ep.rollout_id] = pilot6._alarm_step(ep.decision_steps, sc["global_timer"], tau_g)
            oof_stage[ep.rollout_id] = pilot6._alarm_step(ep.decision_steps, sc["stage_clock"], tau_s_full)
            oof_phys_only[ep.rollout_id] = pilot6._alarm_step(ep.decision_steps, ps, tau_p_full)
            t_s = pilot6._alarm_step(ep.decision_steps, sc["stage_clock"], tau_s_half)
            t_p = pilot6._alarm_step(ep.decision_steps, ps, tau_p_half)
            cands = [x for x in (t_s, t_p) if x is not None]
            oof_union[ep.rollout_id] = min(cands) if cands else None

    failures = [e for e in episodes if not e.success]
    return {
        "cal_sizes": cal_sizes,
        "non_vacuous": all(c >= min_n_cal_union for c in cal_sizes),
        "global_timer_5pct": _summarize_alarm_dict(episodes, oof_global),
        "stage_clock_5pct": _summarize_alarm_dict(episodes, oof_stage),
        "phys_only_5pct": _summarize_alarm_dict(episodes, oof_phys_only),
        "union_monitor": _summarize_alarm_dict(episodes, oof_union),
        "union_minus_global_timer": pilot6._paired_bootstrap_ci(failures, oof_union, oof_global),
        "union_minus_stage_clock": pilot6._paired_bootstrap_ci(failures, oof_union, oof_stage),
    }


def evaluate_zero_shot_threshold_transfer(
    train_episodes: list[pilot6.EpisodeData],
    test_episodes: list[pilot6.EpisodeData],
    *,
    alpha: float = 0.05,
    loss_ref: int = 16,
) -> dict:
    """Protocol B: True Zero-Shot Threshold Transfer.

    Fits task-agnostic horizon-normalized stage durations & grace AND calibrates
    exact finite-sample conformal thresholds STRICTLY on ``train_episodes`` successes
    (ZERO test/held-out episodes touched!), then evaluates directly on ``test_episodes``.
    """
    train_tasks = {e.task for e in train_episodes}
    test_tasks = {e.task for e in test_episodes}
    if train_tasks & test_tasks:
        raise ValueError(f"Protocol B leakage: overlapping tasks {train_tasks & test_tasks}")

    train_succ = pilot6._hash_order([e for e in train_episodes if e.success], "protoB/cal")
    n_total = len(train_succ)
    if n_total < 25:
        raise ValueError(f"Need >= 25 train successes for Protocol B, got {n_total}")

    n_cal = min(n_total - 8, max(39, math.ceil(n_total * 2 / 3))) if n_total >= 47 else math.ceil(n_total * 2 / 3)
    cal_pool = train_succ[:n_cal]
    fit_pool = train_succ[n_cal:]

    # Fit task-agnostic normalized episode/stage quantiles and pooled grace strictly on fit_pool
    ep_fracs: list[float] = []
    by_sid_frac: dict[str, list[float]] = defaultdict(list)
    all_delays: list[int] = []
    for ep in fit_pool:
        h = float(ep.horizon)
        ep_fracs.append(float(ep.num_steps) / h)
        for sid, dur in ep.stage_durations.items():
            by_sid_frac[sid].append(float(dur) / h)
        all_delays.extend(ep.placement_delays)

    def _qval(vals: list[float], q: float) -> float:
        s = sorted(vals)
        idx = max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))
        return float(s[idx])

    global_frac_q = _qval(ep_fracs, 1.0 - alpha) if ep_fracs else 1.0
    stage_frac_q = {sid: _qval(v, 1.0 - alpha) for sid, v in by_sid_frac.items() if len(v) >= 3}
    pooled_grace = max(1.0, _qval([float(x) for x in all_delays], 0.90)) if all_delays else 16.0

    def _universal_scores(ep: pilot6.EpisodeData) -> dict[str, np.ndarray]:
        h = float(ep.horizon)
        t = ep.decision_steps.astype(float)
        g_raw = t / h
        # Zero-centered excess-fraction global clock (matches zero-centered stage & phys scales!)
        g_centered = g_raw - global_frac_q + 1.0 / h
        stg_ex = np.full(len(t), -np.inf, dtype=float)
        for i, (step, sid) in enumerate(zip(ep.decision_steps, ep.decision_stages)):
            sq = stage_frac_q.get(sid)
            entry = ep.stage_entries.get(sid)
            if sq is not None and entry is not None and step >= entry:
                stg_ex[i] = float(step - entry) / h - sq + 1.0 / h
        univ_stage = np.maximum(g_centered, stg_ex)
        miss_term = (ep.missed_effect_age - pooled_grace + 1.0) / h
        loss_term = (ep.loss_age - float(loss_ref) + 1.0) / h
        phys = np.maximum(miss_term, loss_term)
        return {
            "global_timer": g_raw,
            "universal_stage": univ_stage,
            "phys_only": phys,
            "global_plus_phys": np.maximum(g_centered, phys),
            "univ_stage_plus_phys": np.maximum(univ_stage, phys),
        }

    # Calibrate cutoffs strictly on cal_pool (train successes only!)
    arms_b = ("global_timer", "universal_stage", "phys_only", "global_plus_phys", "univ_stage_plus_phys")
    cutoffs: dict[str, float | None] = {}
    for arm in arms_b:
        maxima = [float(np.max(_universal_scores(ep)[arm])) for ep in cal_pool]
        cutoffs[arm] = pilot6._conformal_threshold(maxima, alpha=alpha)

    # Also union-bound (alpha/2 + alpha/2) between global/univ_stage and phys_only
    g_max = [float(np.max(_universal_scores(ep)["global_timer"])) for ep in cal_pool]
    us_max = [float(np.max(_universal_scores(ep)["universal_stage"])) for ep in cal_pool]
    p_max = [float(np.max(_universal_scores(ep)["phys_only"])) for ep in cal_pool]
    tau_g_h = pilot6._conformal_threshold(g_max, alpha=alpha * 0.5)
    tau_us_h = pilot6._conformal_threshold(us_max, alpha=alpha * 0.5)
    tau_p_h = pilot6._conformal_threshold(p_max, alpha=alpha * 0.5)

    alarms: dict[str, dict[str, int | None]] = {arm: {} for arm in arms_b}
    alarms["union_global_phys"] = {}
    alarms["union_univ_stage_phys"] = {}

    for ep in test_episodes:
        sc = _universal_scores(ep)
        for arm in arms_b:
            alarms[arm][ep.rollout_id] = pilot6._alarm_step(ep.decision_steps, sc[arm], cutoffs[arm])
        tg = pilot6._alarm_step(ep.decision_steps, sc["global_timer"], tau_g_h)
        tus = pilot6._alarm_step(ep.decision_steps, sc["universal_stage"], tau_us_h)
        tp = pilot6._alarm_step(ep.decision_steps, sc["phys_only"], tau_p_h)
        alarms["union_global_phys"][ep.rollout_id] = min([x for x in (tg, tp) if x is not None], default=None)
        alarms["union_univ_stage_phys"][ep.rollout_id] = min([x for x in (tus, tp) if x is not None], default=None)

    failures = [e for e in test_episodes if not e.success]
    arm_summaries = {k: _summarize_alarm_dict(test_episodes, v) for k, v in alarms.items()}
    paired = {
        f"{arm}_minus_global_timer": pilot6._paired_bootstrap_ci(failures, alarms[arm], alarms["global_timer"])
        for arm in alarms
        if arm != "global_timer"
    }
    return {
        "n_train_fit_successes": len(fit_pool),
        "n_train_cal_successes": len(cal_pool),
        "pooled_grace_steps": round(pooled_grace, 2),
        "stage_frac_q": {k: round(v, 4) for k, v in sorted(stage_frac_q.items())},
        "cutoffs": {k: (round(v, 5) if v is not None else None) for k, v in cutoffs.items()},
        "union_cutoffs": {
            "tau_g_half": round(tau_g_h, 5) if tau_g_h is not None else None,
            "tau_us_half": round(tau_us_h, 5) if tau_us_h is not None else None,
            "tau_p_half": round(tau_p_h, 5) if tau_p_h is not None else None,
        },
        "arms": arm_summaries,
        "paired": paired,
    }


def _load_vlm_bundle(vlm_dir: Path) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    idx = json.loads((vlm_dir / "vlm_destination_index.json").read_text())
    arrs = dict(np.load(vlm_dir / "vlm_destination_scores.npz"))
    return {e["rollout_id"]: e for e in idx["episodes"]}, arrs


def _build_vlm_tracks(
    episodes: list[dict],
    v_by_rid: dict[str, dict],
    v_arrays: dict[str, np.ndarray],
    any_grasp_bool: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, dict[str, dict[str, np.ndarray]]], dict[str, dict[str, np.ndarray]]]:
    """Build strictly zero-shot ``raw`` and ``dpre_zs`` probability & boolean tracks."""
    prob_tracks: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    bool_tracks: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    raw_ld: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for ep in episodes:
        rid = ep["rollout_id"]
        if rid not in v_by_rid:
            continue
        g_any = any_grasp_bool[rid]
        for pname in v_by_rid[rid]["predicates"]:
            ld = v_arrays[f"{rid}__{pname}__zoh_logit_diff"].astype(np.float32)
            raw_ld[rid][pname] = ld
            dp = vupg.causal_delta_pregrasp(ld, g_any)
            p_raw = vupg._sigmoid(ld)
            p_dp = vupg._sigmoid(dp - 1.0)
            prob_tracks["raw"][rid][pname] = p_raw
            bool_tracks["raw"][rid][pname] = vupg.causal_ema(p_raw) >= 0.5
            prob_tracks["dpre_zs"][rid][pname] = p_dp
            bool_tracks["dpre_zs"][rid][pname] = vupg.causal_ema(p_dp) >= 0.5
    return prob_tracks, bool_tracks, raw_ld


def _vlm_destination_diagnostics(
    held_episodes: list[dict],
    raw_ld: dict[str, dict[str, np.ndarray]],
    any_grasp_bool: dict[str, np.ndarray],
) -> dict:
    """Compute per-task and macro AUROC, post-release AUROC, and @0.5 F1 on held-out tasks."""
    by_task = {}
    raw_aucs, dpre_aucs, pr_aucs, f1s = [], [], [], []
    all_y_any, all_r_any = [], []
    for task in sorted({e["task"] for e in held_episodes}):
        teps = [e for e in held_episodes if e["task"] == task]
        spec = pilot6.TASK_SPECS[task]
        ptargets = [tp for _, _, tp in spec["placements"]]
        t_out = {}
        for tp in ptargets:
            yl, rl, dpl, pr_yl, pr_rl = [], [], [], [], []
            for ep in teps:
                rid = ep["rollout_id"]
                y = np.asarray(ep["predicates_at_decisions"][tp], dtype=int)
                r = raw_ld[rid][tp]
                dp = vupg.causal_delta_pregrasp(r, any_grasp_bool[rid])
                yl.append(y)
                rl.append(r)
                dpl.append(dp)
                g_preds = [gps for _, gps, tname in spec["placements"] if tname == tp][0]
                g_arr = np.zeros(len(y), dtype=bool)
                for gp in g_preds:
                    g_arr |= np.asarray(ep["predicates_at_decisions"][gp], dtype=bool)
                g_on = np.flatnonzero(g_arr)
                if g_on.size:
                    after = np.flatnonzero(~g_arr[g_on[0] :])
                    if after.size:
                        ri = int(g_on[0] + after[0])
                        pr_yl.append(y[ri:])
                        pr_rl.append(r[ri:])
            yc = np.concatenate(yl)
            rc = np.concatenate(rl)
            dpc = np.concatenate(dpl)
            au_r = _auroc(yc, rc)
            au_d = _auroc(yc, dpc)
            bm = _binary_metrics(yc, rc >= 0.0)
            pr_au = _auroc(np.concatenate(pr_yl), np.concatenate(pr_rl)) if pr_yl else None
            t_out[tp] = {
                "raw_auroc": round(au_r, 4) if au_r is not None else None,
                "dpre_auroc": round(au_d, 4) if au_d is not None else None,
                "post_release_raw_auroc": round(pr_au, 4) if pr_au is not None else None,
                "raw_zs_f1": bm["f1"],
                "raw_zs_prec": bm["precision"],
                "raw_zs_rec": bm["recall"],
                "mean_neg_logit": round(float(rc[yc == 0].mean()), 2) if (yc == 0).any() else None,
                "mean_pos_logit": round(float(rc[yc == 1].mean()), 2) if (yc == 1).any() else None,
            }
            if au_r is not None:
                raw_aucs.append(au_r)
            if au_d is not None:
                dpre_aucs.append(au_d)
            if pr_au is not None:
                pr_aucs.append(pr_au)
            f1s.append(bm["f1"])
        # Universal any_placement_satisfied score per episode = max over task placement targets
        for ep in teps:
            rid = ep["rollout_id"]
            y_any = np.zeros(ep["end_row"] - ep["start_row"], dtype=bool)
            r_stack = []
            for tp in ptargets:
                y_any |= np.asarray(ep["predicates_at_decisions"][tp], dtype=bool)
                r_stack.append(raw_ld[rid][tp])
            all_y_any.append(y_any.astype(int))
            all_r_any.append(np.max(np.stack(r_stack, axis=0), axis=0))
        by_task[task] = t_out

    return {
        "universal_any_placement_auroc": round(_auroc(np.concatenate(all_y_any), np.concatenate(all_r_any)) or 0.5, 4),
        "macro_raw_auroc": round(float(np.mean(raw_aucs)), 4) if raw_aucs else None,
        "macro_dpre_auroc": round(float(np.mean(dpre_aucs)), 4) if dpre_aucs else None,
        "macro_post_release_raw_auroc": round(float(np.mean(pr_aucs)), 4) if pr_aucs else None,
        "macro_raw_zs_f1": round(float(np.mean(f1s)), 4) if f1s else None,
        "by_task": by_task,
    }


def run_test1_benchmark18(
    bundle_dir: Path,
    zs_vlm_dir: Path,
    lora_vlm_dir: Path | None,
    llm_zs_vlm_dir: Path | None,
    llm_lora_vlm_dir: Path | None,
    output_dir: Path,
    *,
    grasp_stream: str = "proprio_actions",
    grasp_hidden_dim: int = 128,
    grasp_epochs: int = 100,
    num_folds: int = 5,
    train_tasks: tuple[str, ...] = pilot6.BENCHMARK18_TRAIN_TASKS,
    heldout_tasks: tuple[str, ...] = pilot6.BENCHMARK18_HELDOUT_TASKS,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")

    import torch
    from . import eval_perception_floor as pfloor

    b_index = json.loads((bundle_dir / "perception_index.json").read_text())
    b_arrays = dict(np.load(bundle_dir / "perception_features.npz"))
    all_episodes = b_index["episodes"]

    train_set = set(train_tasks)
    held_set = set(heldout_tasks)
    if train_set & held_set:
        raise ValueError(f"Train/heldout overlap: {train_set & held_set}")

    train_eps = [e for e in all_episodes if e["task"] in train_set]
    held_eps = [e for e in all_episodes if e["task"] in held_set]
    if not train_eps or not held_eps:
        raise RuntimeError(f"Missing episodes: train={len(train_eps)}, heldout={len(held_eps)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Test-1 Benchmark-18 ({grasp_stream} | {device}): "
        f"train={len(train_eps)} ({len({e['task'] for e in train_eps})} tasks), "
        f"heldout={len(held_eps)} ({len({e['task'] for e in held_eps})} tasks)"
    )

    # 1. Train universal any_object_grasped STRICTLY on 10 Train tasks -> test zero-shot on 8 Held-Out tasks
    #    Also compute LOTO universal grasp on the 10 Train tasks themselves for leak-free Protocol B calibration!
    X_grasp = pfloor.build_stream_matrix(b_arrays, all_episodes, grasp_stream)

    def _any_grasp_label(ep: dict) -> np.ndarray:
        n = ep["end_row"] - ep["start_row"]
        y = np.zeros(n, dtype=bool)
        for _, gpreds, _ in pilot6.TASK_SPECS[ep["task"]]["placements"]:
            for gp in gpreds:
                y |= np.asarray(ep["predicates_at_decisions"][gp], dtype=bool)
        return y.astype(np.float32)

    def _any_place_label(ep: dict) -> np.ndarray:
        n = ep["end_row"] - ep["start_row"]
        y = np.zeros(n, dtype=bool)
        for _, _, tp in pilot6.TASK_SPECS[ep["task"]]["placements"]:
            y |= np.asarray(ep["predicates_at_decisions"][tp], dtype=bool)
        return y.astype(np.float32)

    X_tr_g = np.concatenate([X_grasp[e["start_row"] : e["end_row"]] for e in train_eps], axis=0)
    y_tr_g = np.concatenate([_any_grasp_label(e) for e in train_eps])
    X_ho_g = np.concatenate([X_grasp[e["start_row"] : e["end_row"]] for e in held_eps], axis=0)
    y_ho_g = np.concatenate([_any_grasp_label(e) for e in held_eps])

    p_tr_g, p_ho_g = pfloor.fit_predict_probe(
        X_tr_g, y_tr_g, X_ho_g, device=device, hidden_dim=grasp_hidden_dim, epochs=grasp_epochs, seed=0
    )
    grasp_th, _ = pfloor._best_f1_threshold(y_tr_g, p_tr_g)

    any_grasp_prob: dict[str, np.ndarray] = {}
    any_grasp_bool: dict[str, np.ndarray] = {}
    cur = 0
    for ep in held_eps:
        ln = ep["end_row"] - ep["start_row"]
        pr = p_ho_g[cur : cur + ln]
        any_grasp_prob[ep["rollout_id"]] = pr
        any_grasp_bool[ep["rollout_id"]] = pfloor._causal_smooth_and_threshold(pr, grasp_th)
        cur += ln

    # Leak-free LOTO grasp on Train-10 (so Protocol B calibration on Train-10 uses out-of-task grasp predictions!)
    for tr_held_task in sorted(train_set):
        sub_tr = [e for e in train_eps if e["task"] != tr_held_task]
        sub_ev = [e for e in train_eps if e["task"] == tr_held_task]
        if not sub_ev:
            continue
        Xs_tr = np.concatenate([X_grasp[e["start_row"] : e["end_row"]] for e in sub_tr], axis=0)
        ys_tr = np.concatenate([_any_grasp_label(e) for e in sub_tr])
        Xs_ev = np.concatenate([X_grasp[e["start_row"] : e["end_row"]] for e in sub_ev], axis=0)
        ps_tr, ps_ev = pfloor.fit_predict_probe(
            Xs_tr, ys_tr, Xs_ev, device=device, hidden_dim=grasp_hidden_dim, epochs=grasp_epochs, seed=1
        )
        th_s, _ = pfloor._best_f1_threshold(ys_tr, ps_tr)
        c2 = 0
        for ep in sub_ev:
            ln = ep["end_row"] - ep["start_row"]
            pr = ps_ev[c2 : c2 + ln]
            any_grasp_prob[ep["rollout_id"]] = pr
            any_grasp_bool[ep["rollout_id"]] = pfloor._causal_smooth_and_threshold(pr, th_s)
            c2 += ln

    # Per-task zero-shot grasp transfer metrics on the 8 Held-Out tasks
    grasp_by_task = {}
    for task in sorted(held_set):
        teps = [e for e in held_eps if e["task"] == task]
        yt = np.concatenate([_any_grasp_label(e) for e in teps]).astype(int)
        pt = np.concatenate([any_grasp_prob[e["rollout_id"]] for e in teps])
        bt = np.concatenate([any_grasp_bool[e["rollout_id"]] for e in teps])
        grasp_by_task[task] = {
            "pos_rate": round(float(yt.mean()), 4),
            "auroc": round(_auroc(yt, pt), 4) if _auroc(yt, pt) is not None else None,
            **_binary_metrics(yt, bt),
        }
    grasp_transfer = {
        "stream": grasp_stream,
        "train10_threshold": round(float(grasp_th), 4),
        "heldout_pooled_auroc": round(_auroc(y_ho_g, p_ho_g) or 0.5, 4),
        "heldout_pooled_metrics": _binary_metrics(
            y_ho_g, np.concatenate([any_grasp_bool[e["rollout_id"]] for e in held_eps])
        ),
        "by_task": grasp_by_task,
    }

    # 2. Baseline non-VLM universal placement probes (10-Train -> 8-Held-Out)
    y_tr_p = np.concatenate([_any_place_label(e) for e in train_eps])
    y_ho_p = np.concatenate([_any_place_label(e) for e in held_eps])
    baseline_placement_transfer = {}
    for b_stream in ("proprio_actions", "xr1_dit", "siglip_3cam"):
        if b_stream == "siglip_3cam" and "siglip_3cam" not in b_arrays:
            continue
        Xb = pfloor.build_stream_matrix(b_arrays, all_episodes, b_stream)
        Xb_tr = np.concatenate([Xb[e["start_row"] : e["end_row"]] for e in train_eps], axis=0)
        Xb_ho = np.concatenate([Xb[e["start_row"] : e["end_row"]] for e in held_eps], axis=0)
        _, pb_ho = pfloor.fit_predict_probe(
            Xb_tr, y_tr_p, Xb_ho, device=device, hidden_dim=grasp_hidden_dim, epochs=grasp_epochs, seed=0
        )
        baseline_placement_transfer[b_stream] = {
            "heldout_universal_any_placement_auroc": round(_auroc(y_ho_p, pb_ho) or 0.5, 4),
        }

    # 3. Load VLM score bundles (Base Zero-Shot, 10-Train LoRA, LLM-Graph ZS, LLM-Graph LoRA)
    vlm_sources: dict[str, tuple[dict[str, dict], dict[str, np.ndarray]]] = {}
    vlm_sources["zeroshot_hand"] = _load_vlm_bundle(zs_vlm_dir)
    if lora_vlm_dir is not None and (lora_vlm_dir / "vlm_destination_index.json").exists():
        vlm_sources["lora10_hand"] = _load_vlm_bundle(lora_vlm_dir)
    if llm_zs_vlm_dir is not None and (llm_zs_vlm_dir / "vlm_destination_index.json").exists():
        vlm_sources["llm_graph_zs"] = _load_vlm_bundle(llm_zs_vlm_dir)
    if llm_lora_vlm_dir is not None and (llm_lora_vlm_dir / "vlm_destination_index.json").exists():
        vlm_sources["llm_graph_lora10"] = _load_vlm_bundle(llm_lora_vlm_dir)

    vlm_tracks = {}
    vlm_diag = {}
    for src_name, (v_by_rid, v_arrs) in vlm_sources.items():
        pt, bt, rld = _build_vlm_tracks(all_episodes, v_by_rid, v_arrs, any_grasp_bool)
        vlm_tracks[src_name] = (v_by_rid, pt, bt)
        if src_name in ("zeroshot_hand", "lora10_hand"):
            vlm_diag[src_name] = _vlm_destination_diagnostics(held_eps, rld, any_grasp_bool)

    # 4. Evaluate Episode-Level Monitor Ladder on the 8 Held-Out Tasks (Protocol A: 5-fold pooled + union-bound)
    def _eval_trim(ep_objs: list[pilot6.EpisodeData]) -> dict:
        ev = pilot6.evaluate_cross_fitted(ep_objs, num_folds=num_folds)
        ev.pop("oof_alarms", None)
        ev.pop("folds", None)
        return ev

    oracle_held = [
        pfloor._build_substituted_episode(ep, {}, replace_grasps=False, replace_targets=False, use_inferred_stage=False)
        for ep in held_eps
    ]
    protocol_a_pooled: dict[str, dict] = {"oracle_reference": _eval_trim(oracle_held)}
    protocol_a_union: dict[str, dict] = {
        "oracle_reference": evaluate_union_bound_split_conformal_adaptive(oracle_held, num_folds=num_folds)
    }

    # Zero-VLM grasp-stage-only baseline
    grasp_only_held = [
        build_arbitrary_target_monitor_episode(
            ep,
            any_grasp_dec=any_grasp_bool[ep["rollout_id"]],
            target_prob_map={},
            target_bool_map={},
            target_keys=[],
            stage_mode="grasp_only",
            phys_mode="hard_binary",
            hold_policy="placement_only",
        )
        for ep in held_eps
    ]
    protocol_a_pooled["univ_grasp_stage_only_no_vlm"] = _eval_trim(grasp_only_held)

    def _target_keys_for_source(src_name: str, ep: dict, v_by_rid: dict[str, dict]) -> list[str]:
        if src_name.startswith("llm_graph"):
            return list(v_by_rid[ep["rollout_id"]]["predicates"])
        return [tp for _, _, tp in pilot6.TASK_SPECS[ep["task"]]["placements"]]

    # Build Protocol A monitors for every available VLM source
    held_monitor_eps: dict[str, list[pilot6.EpisodeData]] = {"univ_grasp_stage_only_no_vlm": grasp_only_held}
    for src_name, (v_by_rid, pt, bt) in vlm_tracks.items():
        # Verify all held-out episodes are present
        if not all(e["rollout_id"] in v_by_rid for e in held_eps):
            continue
        variants = [
            (f"{src_name}__gv_raw_hard_verholds", "grasp_and_vlm", "raw", "hard_binary", "verified_targets_only"),
            (f"{src_name}__gv_raw_soft_verholds", "grasp_and_vlm", "raw", "soft_evidence", "verified_targets_only"),
            (f"{src_name}__gv_dpre_soft_verholds", "grasp_and_vlm", "dpre_zs", "soft_evidence", "verified_targets_only"),
            (f"{src_name}__grasp_raw_hard_verholds", "grasp_only", "raw", "hard_binary", "verified_targets_only"),
            (f"{src_name}__nostage_raw_hard_verholds", "none", "raw", "hard_binary", "verified_targets_only"),
        ]
        for vname, stg_m, trk, phys_m, hpol in variants:
            eps_v = [
                build_arbitrary_target_monitor_episode(
                    ep,
                    any_grasp_dec=any_grasp_bool[ep["rollout_id"]],
                    target_prob_map=pt[trk][ep["rollout_id"]],
                    target_bool_map=bt[trk][ep["rollout_id"]],
                    target_keys=_target_keys_for_source(src_name, ep, v_by_rid),
                    stage_mode=stg_m,
                    phys_mode=phys_m,
                    hold_policy=hpol,
                )
                for ep in held_eps
            ]
            held_monitor_eps[vname] = eps_v
            protocol_a_pooled[vname] = _eval_trim(eps_v)
            if stg_m in ("grasp_only", "grasp_and_vlm"):
                protocol_a_union[vname] = evaluate_union_bound_split_conformal_adaptive(eps_v, num_folds=num_folds)

        # Component substitution: TRUE grasp + VLM destination (hand-prompted sources only)
        if src_name in ("zeroshot_hand", "lora10_hand"):
            tg_eps = [
                vupg.build_soft_monitor_episode(
                    ep,
                    grasp_bool_map=None,
                    any_grasp_dec=None,
                    target_prob_map=pt["raw"][ep["rollout_id"]],
                    target_bool_map=bt["raw"][ep["rollout_id"]],
                    mode="hard_binary",
                    hold_policy="verified_targets_only",
                    use_inferred_stage=False,
                    use_universal_grasp=False,
                )
                for ep in held_eps
            ]
            protocol_a_pooled[f"{src_name}__true_grasp_raw_hard_verholds"] = _eval_trim(tg_eps)

    # 5. Protocol B: True Zero-Shot Threshold Transfer (calibrate strictly on 10 Train tasks!)
    protocol_b: dict[str, dict] = {}
    zs_v_by_rid, zs_pt, zs_bt = vlm_tracks["zeroshot_hand"]
    has_train10_zs = all(e["rollout_id"] in zs_v_by_rid for e in train_eps)
    if has_train10_zs:
        # Build Train-10 episodes for each (stage_mode, track, phys_mode, hold_policy) using
        # strictly LOTO Train-10 grasp + zero-shot VLM (never trained on Train-10!).
        train_cache: dict[tuple[str, str, str, str], list[pilot6.EpisodeData]] = {}
        for stg_m, trk, phys_m, hpol in (
            ("grasp_and_vlm", "raw", "hard_binary", "verified_targets_only"),
            ("grasp_and_vlm", "raw", "soft_evidence", "verified_targets_only"),
            ("grasp_and_vlm", "dpre_zs", "soft_evidence", "verified_targets_only"),
            ("grasp_only", "raw", "hard_binary", "verified_targets_only"),
        ):
            train_cache[(stg_m, trk, phys_m, hpol)] = [
                build_arbitrary_target_monitor_episode(
                    ep,
                    any_grasp_dec=any_grasp_bool[ep["rollout_id"]],
                    target_prob_map=zs_pt[trk][ep["rollout_id"]],
                    target_bool_map=zs_bt[trk][ep["rollout_id"]],
                    target_keys=[tp for _, _, tp in pilot6.TASK_SPECS[ep["task"]]["placements"]],
                    stage_mode=stg_m,
                    phys_mode=phys_m,
                    hold_policy=hpol,
                )
                for ep in train_eps
            ]
        # Oracle Protocol B reference
        oracle_train = [
            pfloor._build_substituted_episode(ep, {}, replace_grasps=False, replace_targets=False, use_inferred_stage=False)
            for ep in train_eps
        ]
        protocol_b["oracle_reference"] = evaluate_zero_shot_threshold_transfer(oracle_train, oracle_held)
        # Grasp-stage-only Protocol B
        grasp_only_train = [
            build_arbitrary_target_monitor_episode(
                ep,
                any_grasp_dec=any_grasp_bool[ep["rollout_id"]],
                target_prob_map={},
                target_bool_map={},
                target_keys=[],
                stage_mode="grasp_only",
                phys_mode="hard_binary",
                hold_policy="placement_only",
            )
            for ep in train_eps
        ]
        protocol_b["univ_grasp_stage_only_no_vlm"] = evaluate_zero_shot_threshold_transfer(
            grasp_only_train, grasp_only_held
        )
        # Transfer Train-10 calibrated thresholds directly to every Held-Out-8 VLM monitor!
        for src_name in vlm_tracks:
            for suffix, key_cfg in (
                ("gv_raw_hard_verholds", ("grasp_and_vlm", "raw", "hard_binary", "verified_targets_only")),
                ("gv_raw_soft_verholds", ("grasp_and_vlm", "raw", "soft_evidence", "verified_targets_only")),
                ("gv_dpre_soft_verholds", ("grasp_and_vlm", "dpre_zs", "soft_evidence", "verified_targets_only")),
                ("grasp_raw_hard_verholds", ("grasp_only", "raw", "hard_binary", "verified_targets_only")),
            ):
                vname = f"{src_name}__{suffix}"
                if vname in held_monitor_eps:
                    protocol_b[vname] = evaluate_zero_shot_threshold_transfer(
                        train_cache[key_cfg], held_monitor_eps[vname]
                    )

    summary = {
        "schema_version": 1,
        "grasp_stream": grasp_stream,
        "train_tasks": list(train_tasks),
        "heldout_tasks": list(heldout_tasks),
        "num_train_rollouts": len(train_eps),
        "num_heldout_rollouts": len(held_eps),
        "grasp_transfer_10to8": grasp_transfer,
        "baseline_placement_transfer_10to8": baseline_placement_transfer,
        "vlm_destination_diagnostics_heldout8": vlm_diag,
        "protocol_a_5fold_pooled_5pct_fpr": protocol_a_pooled,
        "protocol_a_5fold_union_5pct_fpr": protocol_a_union,
        "protocol_b_zero_shot_transfer_10to8": protocol_b,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "out"
        staged.mkdir()
        (staged / "test1_benchmark18_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        staged.rename(output_dir)

    # Concise summary stdout
    print(f"\n=== 10-TRAIN -> 8-HELDOUT ZERO-SHOT PERCEPTION ({grasp_stream}) ===")
    print(
        f"  Universal any_object_grasped: AUROC={grasp_transfer['heldout_pooled_auroc']:.4f} "
        f"F1={grasp_transfer['heldout_pooled_metrics']['f1']:.4f} (th={grasp_transfer['train10_threshold']:.3f})"
    )
    for bs, bm in baseline_placement_transfer.items():
        print(f"  Baseline {bs:22s} any_placement AUROC={bm['heldout_universal_any_placement_auroc']:.4f}")
    for vn, vd in vlm_diag.items():
        print(
            f"  VLM {vn:27s} any_placement AUROC={vd['universal_any_placement_auroc']:.4f} | "
            f"macro_raw={vd['macro_raw_auroc']} post_rel={vd['macro_post_release_raw_auroc']} @0.5_F1={vd['macro_raw_zs_f1']}"
        )

    print(f"\n=== PROTOCOL A: 8 HELD-OUT TASKS (5-FOLD SPLIT-CONFORMAL, 5% FPR, {grasp_stream}) ===")
    for name, ev in protocol_a_pooled.items():
        fp = ev["arms"]["full_physical"]
        sc = ev["arms"]["stage_clock"]
        vs_g = ev["paired"]["full_physical_minus_global_timer"]
        ci = vs_g["ci95_frac"]
        print(
            f"  {name:44s} stg={sc['macro_lead_frac']*100:5.2f}% | "
            f"full={fp['macro_lead_frac']*100:5.2f}% ({fp['macro_lead_sec']:4.1f}s) "
            f"FPR={fp['realized_fpr']*100:4.1f}% rec={fp['failure_recall']*100:4.1f}% | "
            f"vs_G={vs_g['mean_gain_frac']*100:+5.2f}pp ({vs_g['mean_gain_sec']:+5.1f}s) "
            f"[{ci[0]*100:+5.1f},{ci[1]*100:+5.1f}] pos={vs_g['positive_tasks']}/{vs_g['total_failure_tasks']}"
        )

    print(f"\n=== PROTOCOL A UNION-BOUND (alpha/2 + alpha/2 = 5%, {grasp_stream}) ===")
    for name, u in protocol_a_union.items():
        um = u["union_monitor"]
        vs_g = u["union_minus_global_timer"]
        ci = vs_g["ci95_frac"]
        print(
            f"  {name:44s} nonvac={str(u['non_vacuous']):5s} | "
            f"union={um['macro_lead_frac']*100:5.2f}% ({um['macro_lead_sec']:4.1f}s) "
            f"FPR={um['realized_fpr']*100:4.1f}% rec={um['failure_recall']*100:4.1f}% | "
            f"vs_G={vs_g['mean_gain_frac']*100:+5.2f}pp ({vs_g['mean_gain_sec']:+5.1f}s) "
            f"[{ci[0]*100:+5.1f},{ci[1]*100:+5.1f}] pos={vs_g['positive_tasks']}/{vs_g['total_failure_tasks']}"
        )

    if protocol_b:
        print(f"\n=== PROTOCOL B: TRUE ZERO-SHOT THRESHOLD TRANSFER (10-TRAIN CAL -> 8-HELDOUT TEST, {grasp_stream}) ===")
        for name, pb in protocol_b.items():
            arm_k = "universal_stage" if name == "univ_grasp_stage_only_no_vlm" else "univ_stage_plus_phys"
            am = pb["arms"][arm_k]
            gm = pb["arms"]["global_timer"]
            vg = pb["paired"][f"{arm_k}_minus_global_timer"]
            ci = vg["ci95_frac"]
            print(
                f"  {name:44s} G={gm['macro_lead_frac']*100:5.2f}% ({gm['realized_fpr']*100:3.1f}%) | "
                f"{arm_k:20s}={am['macro_lead_frac']*100:5.2f}% ({am['macro_lead_sec']:4.1f}s) "
                f"FPR={am['realized_fpr']*100:4.1f}% rec={am['failure_recall']*100:4.1f}% | "
                f"vs_G={vg['mean_gain_frac']*100:+5.2f}pp ({vg['mean_gain_sec']:+5.1f}s) "
                f"[{ci[0]*100:+5.1f},{ci[1]*100:+5.1f}] pos={vg['positive_tasks']}/{vg['total_failure_tasks']}"
            )

    print("Wrote:", output_dir / "test1_benchmark18_summary.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--zs-vlm", type=Path, required=True)
    parser.add_argument("--lora-vlm", type=Path, default=None)
    parser.add_argument("--llm-zs-vlm", type=Path, default=None)
    parser.add_argument("--llm-lora-vlm", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grasp-stream", default="proprio_actions")
    parser.add_argument("--grasp-hidden-dim", type=int, default=128)
    parser.add_argument("--grasp-epochs", type=int, default=100)
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args(argv)
    run_test1_benchmark18(
        args.bundle,
        args.zs_vlm,
        args.lora_vlm,
        args.llm_zs_vlm,
        args.llm_lora_vlm,
        args.output,
        grasp_stream=args.grasp_stream,
        grasp_hidden_dim=args.grasp_hidden_dim,
        grasp_epochs=args.grasp_epochs,
        num_folds=args.num_folds,
    )


if __name__ == "__main__":
    main()
