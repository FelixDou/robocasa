"""Powered 18-Task (360-Rollout) Equal-Duration Test-2 Continuation Evaluation.

Tests whether observable post-mistake continuation signals separate
**self-correcting successes** from **persistent failures** STRICTLY PRIOR TO
RECOVERY (and strictly prior to regrasp) under bias-free equal-duration windows:

1. **Strictly equal-duration fixed-horizon cohorts (`H = 3`, `H = 4` decisions):**
   Filters ONLY to rollouts with >= ``H`` decisions strictly inside
   ``[confirmed_step, stop_step)`` and slices EXACTLY the first ``H`` decisions,
   eliminating ALL truncation-length confound (``ctrl_win_decisions == H`` for
   every single episode!). Evaluates:
   - ``same_input_clock`` (control)
   - ``compact_4feat_signature`` (pre-registered length-invariant hypothesis:
     ``[dit_cos_conf, dit_cos_onset, prop_state_step_vel, act_chunk_cos]``)
   - ``action_kinematics``, ``dit_dynamics``, ``visual_vlm_dynamics``,
     ``fused_continuation``, and ``xr1_dit_raw_1024d``
   under LORO, 18-task LOTO, and strict **10-Train -> 8-Held-Out** transfer, with
   two-sided permutation p-values and bootstrap 95% CIs.
2. **Exact pair-matched equal-duration evaluation (`pre_restore` & `pre_regrasp`):**
   Every matched ``(task, kind, operation, relation)`` pair is scored on identical
   window length ``W`` with exact two-sided binomial sign-test p-values and
   leave-one-task-out paired win rates.
3. **Online Continuation-Modulated Alarm Integration (Test 1 + Test 2):**
   Tests whether modulating physical monitor scores by the strictly LOTO-predicted
   continuation persistence probability improves calibrated 5% FPR failure lead
   across all 18 tasks by suppressing self-correcting success excursions.
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
from . import eval_test2_continuation_pilot as t2p


COMPACT_4FEAT = [
    "dit_cos_conf",
    "dit_cos_onset",
    "prop_state_step_vel",
    "act_chunk_cos",
]

FEATURE_GROUPS = {
    "same_input_clock": ["ctrl_age_frac", "ctrl_t_frac"],
    "compact_4feat_signature": list(COMPACT_4FEAT),
    "action_kinematics": [
        "act_eef_speed_mean", "act_eef_speed_max", "act_eef_rot_mean",
        "act_base_speed_mean", "act_base_speed_max", "act_grip_mean",
        "act_grip_activity", "act_grip_range", "act_chunk_std_mean",
        "act_chunk_delta_mean", "act_chunk_step_diff", "act_chunk_cos",
        "prop_state_drift_onset", "prop_state_drift_conf", "prop_state_step_vel",
    ],
    "dit_dynamics": [
        "dit_drift_onset", "dit_drift_conf", "dit_cos_onset",
        "dit_cos_conf", "dit_step_vel", "dit_step_cos",
    ],
    "visual_vlm_dynamics": [
        "vis_wrist_drift_conf", "vis_wrist_step_vel",
        "vis_cam3_drift_conf", "vis_cam3_step_vel",
        "vlm_logit_mean", "vlm_logit_last", "vlm_logit_max",
        "vlm_logit_delta", "vlm_logit_delta_onset",
    ],
}
FEATURE_GROUPS["fused_continuation"] = (
    FEATURE_GROUPS["action_kinematics"]
    + FEATURE_GROUPS["dit_dynamics"]
    + FEATURE_GROUPS["visual_vlm_dynamics"]
)


def _exact_binomial_two_sided_p(k_hi: int, n: int) -> float | None:
    if n <= 0:
        return None
    k_lo = min(k_hi, n - k_hi)
    tail = sum(math.comb(n, i) for i in range(0, k_lo + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def _bootstrap_auroc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        au = t2p._auroc(y[idx], s[idx])
        if au is not None:
            draws.append(au)
    if not draws:
        return None
    return [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]


def _fit_ridge_logistic_train_test(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    *,
    l2: float = 2.0,
) -> np.ndarray:
    yt = np.asarray(y_tr, dtype=np.float64)
    if len(yt) == 0 or yt.sum() == 0 or yt.sum() == len(yt):
        return np.full(len(X_ev), float(yt.mean()) if len(yt) else 0.5, dtype=np.float64)
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xt = (X_tr - mu) / sd
    Xe = (X_ev - mu) / sd
    n_pos = float(yt.sum())
    n_neg = float(len(yt) - n_pos)
    cw = np.where(yt > 0.5, len(yt) / (2.0 * n_pos), len(yt) / (2.0 * n_neg))
    d = Xt.shape[1]
    reg = np.eye(d + 1, dtype=np.float64) * l2
    reg[-1, -1] = 1e-6
    Xt_aug = np.concatenate([Xt, np.ones((len(Xt), 1))], axis=1)
    Xe_aug = np.concatenate([Xe, np.ones((len(Xe), 1))], axis=1)
    theta = np.zeros(d + 1, dtype=np.float64)
    for _ in range(25):
        z = np.clip(Xt_aug @ theta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xt_aug.T @ (cw * (p - yt)) / len(yt) + reg @ theta
        W = cw * p * (1.0 - p) / len(yt)
        H = (Xt_aug.T * W) @ Xt_aug + reg
        step = np.linalg.solve(H, grad)
        theta -= step
        if float(np.linalg.norm(step)) < 1e-6:
            break
    z_ev = np.clip(Xe_aug @ theta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_ev))


def extract_strict_equal_duration_cohort(
    events: list[dict],
    meta_by_rid: dict[str, dict],
    arrays: dict[str, np.ndarray],
    vlm_arrays: dict[str, np.ndarray] | None,
    *,
    exact_decisions: int,
    pre_regrasp_only: bool = False,
) -> list[dict]:
    """Extract strictly equal-duration continuation feature rows.

    ONLY retains events having AT LEAST ``exact_decisions`` policy decisions strictly
    inside ``[confirmed_step, stop_step)``, and slices EXACTLY the first
    ``exact_decisions`` decisions so every retained episode has
    ``ctrl_win_decisions == exact_decisions`` (zero truncation-length confound!).
    """
    rows = []
    for ev in events:
        epm = meta_by_rid[ev["rollout_id"]]
        dec_steps = np.asarray(epm["decision_steps"], dtype=int)
        stop_step = int(ev["num_steps"]) + 1
        if ev["restored_step"] is not None:
            stop_step = min(stop_step, int(ev["restored_step"]))
        if pre_regrasp_only and ev["next_regrasp_step"] is not None and ev["next_regrasp_step"] > ev["confirmed_step"]:
            stop_step = min(stop_step, int(ev["next_regrasp_step"]))

        avail = np.flatnonzero((dec_steps >= ev["confirmed_step"]) & (dec_steps < stop_step))
        if avail.size < exact_decisions:
            continue
        last_pos = int(avail[exact_decisions - 1])
        # Exclusive end = step just after the exact_decisions-th decision
        win_end = int(dec_steps[last_pos]) + 1
        fdict = t2p.extract_window_continuation_features(
            epm,
            arrays,
            vlm_arrays,
            relation=ev["relation"],
            onset_step=ev["onset_step"],
            confirmed_step=ev["confirmed_step"],
            window_end_step=win_end,
        )
        if fdict is None:
            continue
        assert int(round(fdict["ctrl_win_decisions"])) == exact_decisions
        s0, e0 = int(epm["start_row"]), int(epm["end_row"])
        dit_mean = arrays["dit_last"][s0:e0][avail[:exact_decisions]].astype(np.float32).mean(axis=0)
        rows.append(
            {
                "rollout_id": ev["rollout_id"],
                "task": ev["task"],
                "y": 0 if ev["success"] else 1,  # 1 = persistent failure
                "features": fdict,
                "dit_raw_mean": dit_mean,
            }
        )
    return rows


def evaluate_online_continuation_modulation(
    ep_objs: list[pilot6.EpisodeData],
    meta_by_rid: dict[str, dict],
    arrays: dict[str, np.ndarray],
    vlm_arrays: dict[str, np.ndarray] | None,
    first_by_rid: dict[str, dict],
    *,
    feature_cols: list[str] = COMPACT_4FEAT,
    lambdas: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35),
    num_folds: int = 5,
) -> dict:
    """Test whether modulating physical monitor scores with a strictly LOTO-trained
    continuation persistence probability reduces false alarms on self-correcting
    successes and improves 5% FPR failure warning lead across all 18 tasks.
    """
    # Fit strictly LOTO continuation classifier on H=3 equal-duration pre-restore windows
    sc_all = [
        ev for ev in first_by_rid.values()
        if ev["success"] and ev["classification"] in (
            "self_corrected_multi_decision_success",
            "quick_self_corrected_success",
            "delayed_placement_success",
        )
    ]
    pf_all = [
        ev for ev in first_by_rid.values()
        if ev["classification"] in ("persistent_failure", "retry_then_failed")
    ]
    rows_h3 = extract_strict_equal_duration_cohort(
        sc_all + pf_all, meta_by_rid, arrays, vlm_arrays, exact_decisions=3, pre_regrasp_only=False
    )
    if len(rows_h3) < 10:
        return {"skipped": True, "reason": f"insufficient equal-duration H=3 events ({len(rows_h3)})"}

    # Fit one LOTO continuation model per task -> score every decision where missed_effect_age >= 0
    X_all = np.array([[r["features"][c] for c in feature_cols] for r in rows_h3], dtype=np.float64)
    y_all = np.array([r["y"] for r in rows_h3], dtype=int)
    t_all = [r["task"] for r in rows_h3]

    # Precompute causal rolling-3-decision persistence probability p_persist(t) at every decision
    # where missed_effect_age >= 0 or loss_age >= 0, strictly using OTHER tasks' continuation weights!
    p_persist_by_rid: dict[str, np.ndarray] = {}
    for ep in ep_objs:
        epm = meta_by_rid[ep.rollout_id]
        dec_steps = ep.decision_steps
        n_dec = len(dec_steps)
        tr_mask = np.array([tk != ep.task for tk in t_all], dtype=bool)
        # Collect active physical-event decision windows strictly causally
        p_vec = np.full(n_dec, 0.5, dtype=np.float64)
        active_idx = np.flatnonzero((ep.missed_effect_age >= 0) | (ep.loss_age >= 0))
        if active_idx.size > 0 and tr_mask.any():
            ev_rows = []
            ev_pos = []
            for d_i in active_idx:
                t_cur = int(dec_steps[d_i])
                age = max(
                    float(ep.missed_effect_age[d_i]) if np.isfinite(ep.missed_effect_age[d_i]) else 0.0,
                    float(ep.loss_age[d_i]) if np.isfinite(ep.loss_age[d_i]) else 0.0,
                )
                onset_est = max(0, int(round(t_cur - age)))
                # Strictly causal window of up to 3 decisions ending at t_cur (inclusive)
                win_start = int(dec_steps[max(0, int(d_i) - 2)])
                conf_est = max(onset_est, win_start)
                fd = t2p.extract_window_continuation_features(
                    epm,
                    arrays,
                    vlm_arrays,
                    relation="",
                    onset_step=onset_est,
                    confirmed_step=conf_est,
                    window_end_step=t_cur + 1,
                )
                if fd is not None:
                    ev_rows.append([fd[c] for c in feature_cols])
                    ev_pos.append(int(d_i))
            if ev_rows:
                preds = _fit_ridge_logistic_train_test(
                    X_all[tr_mask], y_all[tr_mask], np.asarray(ev_rows, dtype=np.float64), l2=2.0
                )
                for d_i, pr in zip(ev_pos, preds):
                    p_vec[d_i] = float(pr)
        p_persist_by_rid[ep.rollout_id] = p_vec

    # Evaluate unmodified full_physical vs. continuation-modulated full_physical at exact 5% FPR
    base_ev = pilot6.evaluate_cross_fitted(ep_objs, num_folds=num_folds)
    base_fp = base_ev["arms"]["full_physical"]
    base_alarms = base_ev["oof_alarms"]["full_physical"]
    g_alarms = base_ev["oof_alarms"]["global_timer"]
    failures = [e for e in ep_objs if not e.success]
    successes = [e for e in ep_objs if e.success]

    mod_results = {}
    for lam in lambdas:
        # Modulate missed_effect_age and loss_age strictly causally by (p_persist - 0.5) * lam * horizon
        mod_eps = []
        for ep in ep_objs:
            pp = p_persist_by_rid[ep.rollout_id]
            shift = (pp - 0.5) * float(lam) * float(ep.horizon)
            m_age = np.where(np.isfinite(ep.missed_effect_age), ep.missed_effect_age + shift, -np.inf)
            l_age = np.where(np.isfinite(ep.loss_age), ep.loss_age + shift, -np.inf)
            mod_eps.append(
                pilot6.EpisodeData(
                    rollout_id=ep.rollout_id,
                    task=ep.task,
                    seed=ep.seed,
                    success=ep.success,
                    horizon=ep.horizon,
                    num_steps=ep.num_steps,
                    decision_steps=ep.decision_steps,
                    decision_stages=ep.decision_stages,
                    stage_entries=ep.stage_entries,
                    stage_durations=ep.stage_durations,
                    release_only_age=ep.release_only_age,
                    missed_effect_age=m_age,
                    loss_age=l_age,
                    placement_delays=ep.placement_delays,
                    attempts=[],
                    losses=[],
                    wrong_grasp_events=[],
                )
            )
        ev_m = pilot6.evaluate_cross_fitted(mod_eps, num_folds=num_folds)
        m_alarms = ev_m["oof_alarms"]["full_physical"]
        mod_results[f"lambda_{lam:.2f}"] = {
            "full_physical": ev_m["arms"]["full_physical"],
            "vs_unmodulated_full_physical": pilot6._paired_bootstrap_ci(failures, m_alarms, base_alarms),
            "vs_global_timer": pilot6._paired_bootstrap_ci(failures, m_alarms, g_alarms),
        }

    return {
        "feature_cols": list(feature_cols),
        "n_train_h3_events": len(rows_h3),
        "baseline_full_physical": base_fp,
        "modulated_variants": mod_results,
    }


def run_test2_benchmark18(
    bundle_dir: Path,
    vlm_dir: Path | None,
    output_dir: Path,
    *,
    train_tasks: tuple[str, ...] = pilot6.BENCHMARK18_TRAIN_TASKS,
    heldout_tasks: tuple[str, ...] = pilot6.BENCHMARK18_HELDOUT_TASKS,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")

    b_index = json.loads((bundle_dir / "perception_index.json").read_text())
    b_arrays = dict(np.load(bundle_dir / "perception_features.npz"))
    v_arrays = dict(np.load(vlm_dir / "vlm_destination_scores.npz")) if (vlm_dir is not None and (vlm_dir / "vlm_destination_scores.npz").exists()) else None

    episodes_meta = b_index["episodes"]
    meta_by_rid = {e["rollout_id"]: e for e in episodes_meta}
    ep_objs, t2_summary = t2p._build_episode_events(episodes_meta)
    first_by_rid = t2p._event_with_regrasp_info(ep_objs)

    train_set = set(train_tasks)
    held_set = set(heldout_tasks)

    sc_multi = [ev for ev in first_by_rid.values() if ev["classification"] == "self_corrected_multi_decision_success"]
    sc_all = [
        ev for ev in first_by_rid.values()
        if ev["success"] and ev["classification"] in (
            "self_corrected_multi_decision_success",
            "quick_self_corrected_success",
            "delayed_placement_success",
        )
    ]
    pers_fail = [
        ev for ev in first_by_rid.values()
        if ev["classification"] in ("persistent_failure", "retry_then_failed")
    ]

    all_feat_names = sorted({f for cols in FEATURE_GROUPS.values() for f in cols})

    def _eval_equal_cohort(sc_list: list[dict], pf_list: list[dict], exact_h: int, pre_rg: bool) -> dict:
        rows = extract_strict_equal_duration_cohort(
            sc_list + pf_list, meta_by_rid, b_arrays, v_arrays,
            exact_decisions=exact_h, pre_regrasp_only=pre_rg,
        )
        y_arr = np.array([r["y"] for r in rows], dtype=int)
        rids = [r["rollout_id"] for r in rows]
        tasks = [r["task"] for r in rows]
        n_sc = int((y_arr == 0).sum())
        n_pf = int((y_arr == 1).sum())
        if n_sc == 0 or n_pf == 0:
            return {"n_self_corr": n_sc, "n_pers_fail": n_pf, "exact_decisions": exact_h, "pre_regrasp_only": pre_rg, "skipped": True}

        # Univariate signals
        univariate = {}
        for fn in all_feat_names:
            vals = np.array([r["features"][fn] for r in rows], dtype=float)
            au = t2p._auroc(y_arr, vals)
            pv = t2p._perm_pvalue_two_sided(y_arr, vals, n_perm=5000, seed=0)
            univariate[fn] = {
                "auroc_fail_higher": round(au, 4) if au is not None else None,
                "abs_separation": round(abs(au - 0.5) + 0.5, 4) if au is not None else None,
                "direction": "higher_in_failure" if (au is not None and au >= 0.5) else "higher_in_self_corr",
                "perm_p_two_sided": round(pv, 4) if pv is not None else None,
                "ci95_auroc": _bootstrap_auroc_ci(y_arr, vals, n_boot=2000, seed=0),
                "mean_self_corr": round(float(vals[y_arr == 0].mean()), 4),
                "mean_pers_fail": round(float(vals[y_arr == 1].mean()), 4),
            }

        # Group classifiers: LORO, 18-task LOTO, and strict 10-Train -> 8-Held-Out transfer
        tr_mask = np.array([tk in train_set for tk in tasks], dtype=bool)
        ho_mask = np.array([tk in held_set for tk in tasks], dtype=bool)
        can_10to8 = (
            tr_mask.any() and ho_mask.any()
            and (y_arr[tr_mask] == 0).any() and (y_arr[tr_mask] == 1).any()
            and (y_arr[ho_mask] == 0).any() and (y_arr[ho_mask] == 1).any()
        )

        group_cv = {}
        for gname, cols in FEATURE_GROUPS.items():
            X = np.array([[r["features"][c] for c in cols] for r in rows], dtype=np.float64)
            oof_loro = t2p._fit_ridge_logistic_oof(X, y_arr, rids, l2=2.0)
            oof_loto = t2p._fit_ridge_logistic_oof(X, y_arr, tasks, l2=2.0)
            rec = {
                "loro_auroc": round(t2p._auroc(y_arr, oof_loro) or 0.5, 4),
                "loro_ci95": _bootstrap_auroc_ci(y_arr, oof_loro, seed=1),
                "loro_perm_p": round(t2p._perm_pvalue_two_sided(y_arr, oof_loro, n_perm=5000, seed=1) or 1.0, 4),
                "loto_auroc": round(t2p._auroc(y_arr, oof_loto) or 0.5, 4),
                "loto_ci95": _bootstrap_auroc_ci(y_arr, oof_loto, seed=2),
                "loto_perm_p": round(t2p._perm_pvalue_two_sided(y_arr, oof_loto, n_perm=5000, seed=2) or 1.0, 4),
            }
            if can_10to8:
                p_ho = _fit_ridge_logistic_train_test(X[tr_mask], y_arr[tr_mask], X[ho_mask], l2=2.0)
                rec["train10_to_heldout8_auroc"] = round(t2p._auroc(y_arr[ho_mask], p_ho) or 0.5, 4)
                rec["train10_to_heldout8_perm_p"] = round(
                    t2p._perm_pvalue_two_sided(y_arr[ho_mask], p_ho, n_perm=5000, seed=5) or 1.0, 4
                )
                rec["train10_to_heldout8_counts"] = {
                    "heldout_sc": int((y_arr[ho_mask] == 0).sum()),
                    "heldout_pf": int((y_arr[ho_mask] == 1).sum()),
                }
            group_cv[gname] = rec

        # Raw 1024-d XR-1 DiT mean vector
        X_dr = np.stack([r["dit_raw_mean"] for r in rows], axis=0).astype(np.float64)
        oof_dr_loro = t2p._fit_ridge_logistic_oof(X_dr, y_arr, rids, l2=50.0)
        oof_dr_loto = t2p._fit_ridge_logistic_oof(X_dr, y_arr, tasks, l2=50.0)
        dr_rec = {
            "loro_auroc": round(t2p._auroc(y_arr, oof_dr_loro) or 0.5, 4),
            "loro_ci95": _bootstrap_auroc_ci(y_arr, oof_dr_loro, seed=3),
            "loro_perm_p": round(t2p._perm_pvalue_two_sided(y_arr, oof_dr_loro, n_perm=5000, seed=3) or 1.0, 4),
            "loto_auroc": round(t2p._auroc(y_arr, oof_dr_loto) or 0.5, 4),
            "loto_ci95": _bootstrap_auroc_ci(y_arr, oof_dr_loto, seed=4),
            "loto_perm_p": round(t2p._perm_pvalue_two_sided(y_arr, oof_dr_loto, n_perm=5000, seed=4) or 1.0, 4),
        }
        if can_10to8:
            p_ho_dr = _fit_ridge_logistic_train_test(X_dr[tr_mask], y_arr[tr_mask], X_dr[ho_mask], l2=50.0)
            dr_rec["train10_to_heldout8_auroc"] = round(t2p._auroc(y_arr[ho_mask], p_ho_dr) or 0.5, 4)
            dr_rec["train10_to_heldout8_perm_p"] = round(
                t2p._perm_pvalue_two_sided(y_arr[ho_mask], p_ho_dr, n_perm=5000, seed=6) or 1.0, 4
            )
        group_cv["xr1_dit_raw_1024d"] = dr_rec

        return {
            "n_self_corr": n_sc,
            "n_pers_fail": n_pf,
            "n_tasks_represented": len(set(tasks)),
            "exact_decisions": exact_h,
            "pre_regrasp_only": pre_rg,
            "univariate": univariate,
            "group_cv": group_cv,
        }

    cohorts = {
        "strict_multi_equal_h3_pre_restore": _eval_equal_cohort(sc_multi, pers_fail, exact_h=3, pre_rg=False),
        "strict_multi_equal_h4_pre_restore": _eval_equal_cohort(sc_multi, pers_fail, exact_h=4, pre_rg=False),
        "strict_multi_equal_h3_pre_regrasp": _eval_equal_cohort(sc_multi, pers_fail, exact_h=3, pre_rg=True),
        "broad_all_equal_h3_pre_restore": _eval_equal_cohort(sc_all, pers_fail, exact_h=3, pre_rg=False),
        "broad_all_equal_h4_pre_restore": _eval_equal_cohort(sc_all, pers_fail, exact_h=4, pre_rg=False),
        "broad_all_equal_h3_pre_regrasp": _eval_equal_cohort(sc_all, pers_fail, exact_h=3, pre_rg=True),
    }

    # Exact equal-duration matched pairs across ALL 18 tasks
    matched_records = []
    for pair in t2_summary["pairs"]:
        sc_ev = first_by_rid[pair["success_rollout_id"]]
        pf_ev = first_by_rid[pair["failure_rollout_id"]]
        w_res = int(sc_ev["restored_step"] - sc_ev["confirmed_step"])
        f_sc_res = t2p.extract_window_continuation_features(
            meta_by_rid[sc_ev["rollout_id"]], b_arrays, v_arrays,
            relation=sc_ev["relation"], onset_step=sc_ev["onset_step"],
            confirmed_step=sc_ev["confirmed_step"], window_end_step=sc_ev["restored_step"],
        )
        f_pf_res = t2p.extract_window_continuation_features(
            meta_by_rid[pf_ev["rollout_id"]], b_arrays, v_arrays,
            relation=pf_ev["relation"], onset_step=pf_ev["onset_step"],
            confirmed_step=pf_ev["confirmed_step"],
            window_end_step=min(pf_ev["num_steps"] + 1, pf_ev["confirmed_step"] + w_res),
        )
        sc_end_rg = sc_ev["restored_step"]
        if sc_ev["next_regrasp_step"] is not None and sc_ev["next_regrasp_step"] > sc_ev["confirmed_step"]:
            sc_end_rg = min(sc_end_rg, int(sc_ev["next_regrasp_step"]))
        w_rg = int(sc_end_rg - sc_ev["confirmed_step"])
        f_sc_rg = t2p.extract_window_continuation_features(
            meta_by_rid[sc_ev["rollout_id"]], b_arrays, v_arrays,
            relation=sc_ev["relation"], onset_step=sc_ev["onset_step"],
            confirmed_step=sc_ev["confirmed_step"], window_end_step=sc_end_rg,
        )
        f_pf_rg = t2p.extract_window_continuation_features(
            meta_by_rid[pf_ev["rollout_id"]], b_arrays, v_arrays,
            relation=pf_ev["relation"], onset_step=pf_ev["onset_step"],
            confirmed_step=pf_ev["confirmed_step"],
            window_end_step=min(pf_ev["num_steps"] + 1, pf_ev["confirmed_step"] + w_rg),
        )
        matched_records.append(
            {
                "pair": pair,
                "w_restore_steps": w_res,
                "w_pre_regrasp_steps": w_rg,
                "pre_restore_success": f_sc_res,
                "pre_restore_failure": f_pf_res,
                "pre_regrasp_success": f_sc_rg,
                "pre_regrasp_failure": f_pf_rg,
            }
        )

    matched_summary = {}
    for wkind, sk, fk in (
        ("pre_restore", "pre_restore_success", "pre_restore_failure"),
        ("pre_regrasp", "pre_regrasp_success", "pre_regrasp_failure"),
    ):
        valid = [p for p in matched_records if p[sk] is not None and p[fk] is not None]
        # Univariate win rates + exact binomial sign-test p-values
        uni_wins = {}
        for fn in all_feat_names:
            sc_gt = sum(p[sk][fn] > p[fk][fn] for p in valid)
            pf_gt = sum(p[fk][fn] > p[sk][fn] for p in valid)
            n_non_tie = sc_gt + pf_gt
            hi = max(sc_gt, pf_gt)
            uni_wins[fn] = {
                "n_pairs": len(valid),
                "self_corr_higher": sc_gt,
                "pers_fail_higher": pf_gt,
                "sign_test_p": round(_exact_binomial_two_sided_p(hi, n_non_tie), 4) if n_non_tie > 0 else None,
                "mean_sc": round(float(np.mean([p[sk][fn] for p in valid])), 4) if valid else None,
                "mean_pf": round(float(np.mean([p[fk][fn] for p in valid])), 4) if valid else None,
            }
        # Out-of-fold paired classifier win rate (LOTO across tasks)
        paired_clf = {}
        if len(valid) >= 4:
            p_y = np.array([0] * len(valid) + [1] * len(valid), dtype=int)
            p_tasks = [p["pair"]["task"] for p in valid] * 2
            p_pair_ids = [f"pair_{i}" for i in range(len(valid))] * 2
            for gname, cols in FEATURE_GROUPS.items():
                Xp = np.array(
                    [[p[sk][c] for c in cols] for p in valid] + [[p[fk][c] for c in cols] for p in valid],
                    dtype=np.float64,
                )
                oof_lopo = t2p._fit_ridge_logistic_oof(Xp, p_y, p_pair_ids, l2=2.0)
                oof_loto = t2p._fit_ridge_logistic_oof(Xp, p_y, p_tasks, l2=2.0)
                m = len(valid)
                lopo_wins = int(np.sum(oof_lopo[m:] > oof_lopo[:m]))
                loto_wins = int(np.sum(oof_loto[m:] > oof_loto[:m]))
                paired_clf[gname] = {
                    "n_pairs": m,
                    "lopo_pair_wins": lopo_wins,
                    "lopo_sign_p": round(_exact_binomial_two_sided_p(lopo_wins, m) or 1.0, 4),
                    "lopo_auroc": round(t2p._auroc(p_y, oof_lopo) or 0.5, 4),
                    "loto_pair_wins": loto_wins,
                    "loto_sign_p": round(_exact_binomial_two_sided_p(loto_wins, m) or 1.0, 4),
                    "loto_auroc": round(t2p._auroc(p_y, oof_loto) or 0.5, 4),
                }
        matched_summary[wkind] = {
            "n_valid_pairs": len(valid),
            "univariate_wins": uni_wins,
            "paired_classifiers": paired_clf,
        }

    # Online Continuation-Modulated Alarm Integration (Test 1 + Test 2)
    online_mod = evaluate_online_continuation_modulation(
        ep_objs, meta_by_rid, b_arrays, v_arrays, first_by_rid, feature_cols=COMPACT_4FEAT
    )

    by_task_counts = defaultdict(lambda: {"sc_multi": 0, "sc_broad": 0, "pers_fail": 0, "pairs": 0})
    for ev in sc_multi:
        by_task_counts[ev["task"]]["sc_multi"] += 1
    for ev in sc_all:
        by_task_counts[ev["task"]]["sc_broad"] += 1
    for ev in pers_fail:
        by_task_counts[ev["task"]]["pers_fail"] += 1
    for p in matched_records:
        by_task_counts[p["pair"]["task"]]["pairs"] += 1

    summary = {
        "schema_version": 1,
        "bundle_dir": str(bundle_dir),
        "vlm_dir": str(vlm_dir) if vlm_dir is not None else None,
        "num_rollouts": len(episodes_meta),
        "num_tasks": len({e["task"] for e in episodes_meta}),
        "cohort_counts": {
            "strict_multi_decision_self_corrected": len(sc_multi),
            "broad_confirmed_mistake_successes": len(sc_all),
            "persistent_failures": len(pers_fail),
            "matched_pairs": len(matched_records),
            "by_task": dict(sorted(by_task_counts.items())),
        },
        "equal_duration_cohorts": cohorts,
        "matched_pairs_summary": matched_summary,
        "online_continuation_modulation": online_mod,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "out"
        staged.mkdir()
        (staged / "test2_benchmark18_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        staged.rename(output_dir)

    print(f"=== POWERED 18-TASK TEST-2 EQUAL-DURATION EVALUATION ({len(episodes_meta)} ROLLOUTS) ===")
    cc = summary["cohort_counts"]
    print(
        f"Cohort: strict_multi_SC={cc['strict_multi_decision_self_corrected']}, "
        f"broad_SC={cc['broad_confirmed_mistake_successes']}, "
        f"pers_fail={cc['persistent_failures']}, matched_pairs={cc['matched_pairs']}"
    )
    for cname, cd in cohorts.items():
        if cd.get("skipped"):
            print(f"\n--- {cname}: SKIPPED (SC={cd['n_self_corr']}, PF={cd['n_pers_fail']}) ---")
            continue
        print(
            f"\n--- {cname} (SC={cd['n_self_corr']}, PF={cd['n_pers_fail']}, "
            f"exact_H={cd['exact_decisions']} dec, pre_rg={cd['pre_regrasp_only']}, tasks={cd['n_tasks_represented']}) ---"
        )
        for gname, gm in cd["group_cv"].items():
            t10 = ""
            if "train10_to_heldout8_auroc" in gm:
                t10 = f" | 10->8={gm['train10_to_heldout8_auroc']:.4f} (p={gm['train10_to_heldout8_perm_p']:.4f})"
            print(
                f"  {gname:24s} LORO={gm['loro_auroc']:.4f} (p={gm['loro_perm_p']:.4f}) | "
                f"LOTO={gm['loto_auroc']:.4f} (p={gm['loto_perm_p']:.4f}){t10}"
            )

    for wkind in ("pre_restore", "pre_regrasp"):
        ms = matched_summary[wkind]
        print(f"\n=== EXACT EQUAL-DURATION MATCHED PAIRS ({wkind}, n={ms['n_valid_pairs']}) ===")
        for gname, pm in ms["paired_classifiers"].items():
            print(
                f"  {gname:24s} LOPO={pm['lopo_pair_wins']}/{pm['n_pairs']} (p={pm['lopo_sign_p']:.4f}, AUC={pm['lopo_auroc']:.4f}) | "
                f"LOTO={pm['loto_pair_wins']}/{pm['n_pairs']} (p={pm['loto_sign_p']:.4f}, AUC={pm['loto_auroc']:.4f})"
            )

    if not online_mod.get("skipped"):
        bf = online_mod["baseline_full_physical"]
        print(
            f"\n=== ONLINE CONTINUATION-MODULATED ALARM INTEGRATION (18-TASK 5% FPR) ===\n"
            f"  baseline_full_physical    lead={bf['macro_lead_frac']*100:5.2f}% ({bf['macro_lead_sec']:4.1f}s) "
            f"FPR={bf['realized_fpr']*100:4.1f}% rec={bf['failure_recall']*100:4.1f}%"
        )
        for lk, mv in online_mod["modulated_variants"].items():
            mf = mv["full_physical"]
            vg = mv["vs_unmodulated_full_physical"]
            print(
                f"  {lk:25s} lead={mf['macro_lead_frac']*100:5.2f}% ({mf['macro_lead_sec']:4.1f}s) "
                f"FPR={mf['realized_fpr']*100:4.1f}% rec={mf['failure_recall']*100:4.1f}% | "
                f"vs_base={vg['mean_gain_frac']*100:+5.2f}pp ({vg['mean_gain_sec']:+5.1f}s)"
            )

    print("Wrote:", output_dir / "test2_benchmark18_summary.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--vlm-scores", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_test2_benchmark18(args.bundle, args.vlm_scores, args.output)


if __name__ == "__main__":
    main()
