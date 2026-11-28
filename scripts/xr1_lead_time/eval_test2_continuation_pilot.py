"""Test-2 Continuation-Aware Pilot Check on the 120-rollout synchronized XR-1 cohort.

Tests whether observable post-mistake continuation signals (action-chunk intent,
executed kinematic motion, proprioceptive drift, XR-1 DiT latent stagnation,
wrist-camera visual novelty, and Qwen3-VL logit trend) separate **self-correcting
successes** from **persistent failures** STRICTLY PRIOR TO RECOVERY (and strictly
prior to regrasp) when mistake age / condition-checker inputs are held equal.

Evaluates:
1. Exact matched-pair Test-2 windows (4 matched `(task, kind, operation, relation)`
   pairs + all 5 multi-decision self-corrected successes vs. 35 persistent failures,
   and broad 11-success vs. 35-failure confirmed-mistake cohort).
2. Two strict causal pre-recovery windows:
   - ``pre_restore``: ``[confirmed_step, restored_step)`` (matched duration ``W`` on failures)
   - ``pre_regrasp``: ``[confirmed_step, min(next_regrasp_step, restored_step))``
     (strictly BEFORE the robot even touches/regrasps the object!)
3. Zero-training interpretable physical/representation continuation signatures
   (with exact univariate AUROC, matched-pair win rates, and exact Mann-Whitney U
   permutation p-values — immune to high-dimensional overfitting at small n).
4. Strictly out-of-fold LORO and **unseen-task LOTO** linear continuation probes
   comparing same-input age baseline vs. action dynamics vs. DiT dynamics vs.
   visual/VLM dynamics vs. fused continuation.
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


def _perm_pvalue_two_sided(y_true: np.ndarray, y_score: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float | None:
    obs = _auroc(y_true, y_score)
    if obs is None:
        return None
    obs_dist = abs(obs - 0.5)
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    ge = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        au = _auroc(yp, s)
        if au is not None and abs(au - 0.5) >= obs_dist - 1e-12:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def extract_window_continuation_features(
    ep_meta: dict,
    arrays: dict[str, np.ndarray],
    vlm_arrays: dict[str, np.ndarray] | None,
    *,
    relation: str,
    onset_step: int,
    confirmed_step: int,
    window_end_step: int,
) -> dict[str, float] | None:
    """Compute strictly causal continuation features inside ``[confirmed_step, window_end_step)``.

    Uses ONLY observations at decision steps ``t`` with ``confirmed_step <= t < window_end_step``
    (plus causal anchor at ``onset_step`` <= ``confirmed_step``). Never reads any frame,
    action, or latent at or after ``window_end_step``.
    """
    dec_steps = np.asarray(ep_meta["decision_steps"], dtype=int)
    s0, e0 = int(ep_meta["start_row"]), int(ep_meta["end_row"])
    rid = ep_meta["rollout_id"]

    win_local = np.flatnonzero((dec_steps >= confirmed_step) & (dec_steps < window_end_step))
    if win_local.size == 0:
        return None
    # Anchor decision at or immediately before onset_step (strictly causal <= onset_step)
    onset_pos = max(0, int(np.searchsorted(dec_steps, onset_step, side="right")) - 1)
    conf_pos = int(win_local[0])

    # Slice episode arrays locally (0 .. n_dec-1)
    ex = arrays["executed_actions"][s0:e0].astype(np.float32)       # (D, 18)
    ch = arrays["action_chunks"][s0:e0].astype(np.float32)          # (D, 3*A)
    st = arrays["state_history"][s0:e0].astype(np.float32)          # (D, D_st)
    dit = arrays["dit_last"][s0:e0].astype(np.float32)              # (D, 1024)
    has_vis = "siglip_3cam" in arrays
    vis_w = arrays["siglip_3cam"][s0:e0, 2, :].astype(np.float32) if has_vis else None
    vis_all = arrays["siglip_3cam"][s0:e0].astype(np.float32).reshape(e0 - s0, -1) if has_vis else None

    ex_w = ex[win_local]
    ch_w = ch[win_local]
    st_w = st[win_local]
    dit_w = dit[win_local]

    # 1. Kinematic / action-chunk dynamics
    # executed_actions layout (extract_perception_features.py):
    #   0:grip_mean 1:grip_min 2:grip_max 3:grip_last 4:grip_delta
    #   5..7:eef_pos_mean 8:eef_pos_norm 9..11:eef_rot_mean 12:eef_rot_norm
    #   13..16:base_mean 17:base_norm
    eef_speed_mean = float(ex_w[:, 8].mean())
    eef_speed_max = float(ex_w[:, 8].max())
    eef_rot_mean = float(ex_w[:, 12].mean())
    base_speed_mean = float(ex_w[:, 17].mean())
    base_speed_max = float(ex_w[:, 17].max())
    grip_mean = float(ex_w[:, 0].mean())
    grip_activity = float(np.abs(ex_w[:, 4]).mean())
    grip_range = float(ex_w[:, 2].max() - ex_w[:, 1].min())

    a_dim = ch.shape[1] // 3
    ch_mean = ch_w[:, :a_dim]
    ch_std = ch_w[:, a_dim : 2 * a_dim]
    ch_delta = ch_w[:, 2 * a_dim :]
    chunk_std_mean = float(np.linalg.norm(ch_std, axis=1).mean())
    chunk_delta_mean = float(np.linalg.norm(ch_delta, axis=1).mean())
    if len(ch_mean) > 1:
        chunk_step_diff = float(np.linalg.norm(ch_mean[1:] - ch_mean[:-1], axis=1).mean())
        chunk_cos = float(np.mean([_cosine_sim(ch_mean[i], ch_mean[i - 1]) for i in range(1, len(ch_mean))]))
    else:
        chunk_step_diff = 0.0
        chunk_cos = 1.0

    # Proprioceptive state movement & drift from onset
    st_drift_from_onset = float(np.linalg.norm(st_w[-1] - st[onset_pos]))
    st_drift_from_conf = float(np.linalg.norm(st_w[-1] - st[conf_pos]))
    st_step_vel = float(np.linalg.norm(st_w[1:] - st_w[:-1], axis=1).mean()) if len(st_w) > 1 else 0.0

    # 2. XR-1 DiT latent dynamics (attractor stagnation vs. active recovery transition)
    dit_drift_onset = float(np.linalg.norm(dit_w[-1] - dit[onset_pos]))
    dit_drift_conf = float(np.linalg.norm(dit_w[-1] - dit[conf_pos]))
    dit_cos_onset = _cosine_sim(dit_w[-1], dit[onset_pos])
    dit_cos_conf = _cosine_sim(dit_w[-1], dit[conf_pos])
    if len(dit_w) > 1:
        dit_step_vel = float(np.linalg.norm(dit_w[1:] - dit_w[:-1], axis=1).mean())
        dit_step_cos = float(np.mean([_cosine_sim(dit_w[i], dit_w[i - 1]) for i in range(1, len(dit_w))]))
    else:
        dit_step_vel = 0.0
        dit_step_cos = 1.0

    # 3. Visual novelty (SigLIP wrist & 3-cam) + VLM destination logit trend
    if has_vis and vis_w is not None and vis_all is not None:
        vis_w_cur = vis_w[win_local]
        vis_a_cur = vis_all[win_local]
        wrist_drift_conf = float(np.linalg.norm(vis_w_cur[-1] - vis_w[conf_pos]))
        wrist_step_vel = float(np.linalg.norm(vis_w_cur[1:] - vis_w_cur[:-1], axis=1).mean()) if len(vis_w_cur) > 1 else 0.0
        cam3_drift_conf = float(np.linalg.norm(vis_a_cur[-1] - vis_all[conf_pos]))
        cam3_step_vel = float(np.linalg.norm(vis_a_cur[1:] - vis_a_cur[:-1], axis=1).mean()) if len(vis_a_cur) > 1 else 0.0
    else:
        wrist_drift_conf = wrist_step_vel = cam3_drift_conf = cam3_step_vel = 0.0

    vlm_key = f"{rid}__{relation}__zoh_logit_diff"
    if vlm_arrays is not None and vlm_key in vlm_arrays:
        v_ld = vlm_arrays[vlm_key].astype(np.float32)
        v_w = v_ld[win_local]
        vlm_mean = float(v_w.mean())
        vlm_last = float(v_w[-1])
        vlm_max = float(v_w.max())
        vlm_delta = float(v_w[-1] - v_w[0])
        vlm_delta_onset = float(v_w[-1] - v_ld[onset_pos])
    else:
        vlm_mean = vlm_last = vlm_max = vlm_delta = vlm_delta_onset = 0.0

    # 4. Same-input clock/age baseline features (control)
    age_end_frac = float((int(dec_steps[win_local[-1]]) - onset_step) / max(1, int(ep_meta["horizon"])))
    t_frac = float(int(dec_steps[win_local[-1]]) / max(1, int(ep_meta["horizon"])))

    return {
        # Control (same-input / time)
        "ctrl_age_frac": age_end_frac,
        "ctrl_t_frac": t_frac,
        "ctrl_win_decisions": float(len(win_local)),
        # Action / kinematic dynamics
        "act_eef_speed_mean": eef_speed_mean,
        "act_eef_speed_max": eef_speed_max,
        "act_eef_rot_mean": eef_rot_mean,
        "act_base_speed_mean": base_speed_mean,
        "act_base_speed_max": base_speed_max,
        "act_grip_mean": grip_mean,
        "act_grip_activity": grip_activity,
        "act_grip_range": grip_range,
        "act_chunk_std_mean": chunk_std_mean,
        "act_chunk_delta_mean": chunk_delta_mean,
        "act_chunk_step_diff": chunk_step_diff,
        "act_chunk_cos": chunk_cos,
        "prop_state_drift_onset": st_drift_from_onset,
        "prop_state_drift_conf": st_drift_from_conf,
        "prop_state_step_vel": st_step_vel,
        # DiT internal latent dynamics
        "dit_drift_onset": dit_drift_onset,
        "dit_drift_conf": dit_drift_conf,
        "dit_cos_onset": dit_cos_onset,
        "dit_cos_conf": dit_cos_conf,
        "dit_step_vel": dit_step_vel,
        "dit_step_cos": dit_step_cos,
        # Visual / VLM dynamics
        "vis_wrist_drift_conf": wrist_drift_conf,
        "vis_wrist_step_vel": wrist_step_vel,
        "vis_cam3_drift_conf": cam3_drift_conf,
        "vis_cam3_step_vel": cam3_step_vel,
        "vlm_logit_mean": vlm_mean,
        "vlm_logit_last": vlm_last,
        "vlm_logit_max": vlm_max,
        "vlm_logit_delta": vlm_delta,
        "vlm_logit_delta_onset": vlm_delta_onset,
    }


def _fit_ridge_logistic_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    *,
    l2: float = 1.0,
) -> np.ndarray:
    """Simple, deterministic, strictly out-of-fold L2-regularized logistic regression
    (pure NumPy Newton-IRLS, zero PyTorch dependency, fast & exact for small n).
    """
    n, d = X.shape
    oof = np.full(n, 0.5, dtype=np.float64)
    unique_g = sorted(set(groups))
    for g in unique_g:
        ev_mask = np.array([grp == g for grp in groups], dtype=bool)
        tr_mask = ~ev_mask
        y_tr = y[tr_mask].astype(np.float64)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            oof[ev_mask] = float(y_tr.mean()) if len(y_tr) else 0.5
            continue
        X_tr = X[tr_mask].astype(np.float64)
        X_ev = X[ev_mask].astype(np.float64)
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Xt = (X_tr - mu) / sd
        Xe = (X_ev - mu) / sd

        # Class-balanced weights
        n_pos = float(y_tr.sum())
        n_neg = float(len(y_tr) - n_pos)
        cw = np.where(y_tr > 0.5, len(y_tr) / (2.0 * n_pos), len(y_tr) / (2.0 * n_neg))

        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        reg = np.eye(d + 1, dtype=np.float64) * l2
        reg[-1, -1] = 1e-6  # do not shrink intercept
        Xt_aug = np.concatenate([Xt, np.ones((len(Xt), 1))], axis=1)
        Xe_aug = np.concatenate([Xe, np.ones((len(Xe), 1))], axis=1)
        theta = np.zeros(d + 1, dtype=np.float64)
        for _ in range(25):
            z = np.clip(Xt_aug @ theta, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xt_aug.T @ (cw * (p - y_tr)) / len(y_tr) + reg @ theta
            W = cw * p * (1.0 - p) / len(y_tr)
            H = (Xt_aug.T * W) @ Xt_aug + reg
            step = np.linalg.solve(H, grad)
            theta -= step
            if float(np.linalg.norm(step)) < 1e-6:
                break
        z_ev = np.clip(Xe_aug @ theta, -30.0, 30.0)
        oof[ev_mask] = 1.0 / (1.0 + np.exp(-z_ev))
    return oof


def _build_episode_events(episodes_meta: list[dict]) -> tuple[list[pilot6.EpisodeData], dict]:
    ep_objs = []
    for ep in episodes_meta:
        dec_steps = ep["decision_steps"]
        row = {
            "rollout_id": ep["rollout_id"],
            "task_name": ep["task"],
            "environment_seed": ep["seed"],
            "failed": not ep["success"],
            "rollout_horizon": ep["horizon"],
            "num_environment_steps": ep["num_steps"],
        }
        trace = {
            "task_name": ep["task"],
            "inference_records": [
                {"environment_step": int(t), "inference_index": i, "subtask_id": stg}
                for i, (t, stg) in enumerate(zip(dec_steps, ep["decision_stages"]))
            ],
            "recovery_observations": ep["recovery_observations"],
            "segments": ep["segments"],
        }
        ep_objs.append(pilot6.extract_episode(row, trace))
    t2 = pilot6.build_test2_pairs(ep_objs)
    return ep_objs, t2


def _event_with_regrasp_info(ep_objs: list[pilot6.EpisodeData]) -> dict[str, dict]:
    """Return first confirmed mistake per rollout enriched with ``next_regrasp_step``."""
    all_evs = []
    for ep in ep_objs:
        for att in ep.attempts:
            if att["confirmed_miss_step"] is not None:
                all_evs.append(
                    {
                        "kind": "missed_placement",
                        "rollout_id": ep.rollout_id,
                        "task": ep.task,
                        "success": ep.success,
                        "operation": att["operation"],
                        "relation": att["relation"],
                        "onset_step": att["release_step"],
                        "confirmed_step": att["confirmed_miss_step"],
                        "next_regrasp_step": att["next_regrasp_step"],
                        "restored_step": att["restored_step"],
                        "decisions_before_recovery": att["decisions_before_recovery"],
                        "classification": att["classification"],
                        "num_steps": ep.num_steps,
                    }
                )
        for loss in ep.losses:
            all_evs.append(
                {
                    "kind": "lost_condition",
                    "rollout_id": ep.rollout_id,
                    "task": ep.task,
                    "success": ep.success,
                    "operation": loss["operation"],
                    "relation": loss["relation"],
                    "onset_step": loss["onset_step"],
                    "confirmed_step": loss["confirmed_step"],
                    "next_regrasp_step": None,
                    "restored_step": loss["restored_step"],
                    "decisions_before_recovery": loss["decisions_before_recovery"],
                    "classification": loss["classification"],
                    "num_steps": ep.num_steps,
                }
            )
    first_per_ep = {}
    for ev in sorted(all_evs, key=lambda x: (x["confirmed_step"], x["onset_step"])):
        if ev["rollout_id"] not in first_per_ep:
            first_per_ep[ev["rollout_id"]] = ev
    return first_per_ep


def run_test2_pilot(
    bundle_dir: Path,
    vlm_dir: Path | None,
    output_dir: Path,
    *,
    fixed_horizon_decisions: int = 8,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")

    b_index = json.loads((bundle_dir / "perception_index.json").read_text())
    b_arrays = dict(np.load(bundle_dir / "perception_features.npz"))
    v_arrays = dict(np.load(vlm_dir / "vlm_destination_scores.npz")) if vlm_dir is not None else None

    episodes_meta = b_index["episodes"]
    meta_by_rid = {e["rollout_id"]: e for e in episodes_meta}
    ep_objs, t2_summary = _build_episode_events(episodes_meta)
    first_by_rid = _event_with_regrasp_info(ep_objs)

    # Cohort definitions (label y=1 means PERSISTENT FAILURE; y=0 means SELF-CORRECTED SUCCESS)
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

    # 1. Exact matched-pair evaluation on BOTH pre_restore and pre_regrasp equal-duration windows!
    matched_pair_records = []
    for pair in t2_summary["pairs"]:
        sc_ev = first_by_rid[pair["success_rollout_id"]]
        pf_ev = first_by_rid[pair["failure_rollout_id"]]

        # Pre-restore matched window: W_steps = restored_step - confirmed_step on success
        w_restore = int(sc_ev["restored_step"] - sc_ev["confirmed_step"])
        f_sc_res = extract_window_continuation_features(
            meta_by_rid[sc_ev["rollout_id"]], b_arrays, v_arrays,
            relation=sc_ev["relation"],
            onset_step=sc_ev["onset_step"],
            confirmed_step=sc_ev["confirmed_step"],
            window_end_step=sc_ev["restored_step"],
        )
        f_pf_res = extract_window_continuation_features(
            meta_by_rid[pf_ev["rollout_id"]], b_arrays, v_arrays,
            relation=pf_ev["relation"],
            onset_step=pf_ev["onset_step"],
            confirmed_step=pf_ev["confirmed_step"],
            window_end_step=min(pf_ev["num_steps"] + 1, pf_ev["confirmed_step"] + w_restore),
        )

        # Strict pre-regrasp window: stop BEFORE next_regrasp_step (if any) AND restored_step!
        sc_end_pre_rg = sc_ev["restored_step"]
        if sc_ev["next_regrasp_step"] is not None and sc_ev["next_regrasp_step"] > sc_ev["confirmed_step"]:
            sc_end_pre_rg = min(sc_end_pre_rg, int(sc_ev["next_regrasp_step"]))
        w_pre_rg = int(sc_end_pre_rg - sc_ev["confirmed_step"])
        f_sc_rg = extract_window_continuation_features(
            meta_by_rid[sc_ev["rollout_id"]], b_arrays, v_arrays,
            relation=sc_ev["relation"],
            onset_step=sc_ev["onset_step"],
            confirmed_step=sc_ev["confirmed_step"],
            window_end_step=sc_end_pre_rg,
        )
        f_pf_rg = extract_window_continuation_features(
            meta_by_rid[pf_ev["rollout_id"]], b_arrays, v_arrays,
            relation=pf_ev["relation"],
            onset_step=pf_ev["onset_step"],
            confirmed_step=pf_ev["confirmed_step"],
            window_end_step=min(pf_ev["num_steps"] + 1, pf_ev["confirmed_step"] + w_pre_rg),
        )
        matched_pair_records.append(
            {
                "pair": pair,
                "w_restore_steps": w_restore,
                "w_pre_regrasp_steps": w_pre_rg,
                "pre_restore_success": f_sc_res,
                "pre_restore_failure": f_pf_res,
                "pre_regrasp_success": f_sc_rg,
                "pre_regrasp_failure": f_pf_rg,
            }
        )

    # 2. Fixed-horizon post-confirmation cohort evaluation (strictly capped at min(fixed_horizon, pre_restore))
    #    Evaluates BOTH strict multi-decision (n=5 vs 35) AND broad recovery (n=11 vs 35)
    #    at fixed early horizons H in (3, 6, 8) decisions (2.4s, 4.8s, 6.4s after confirmation!).
    feature_groups = {
        "same_input_clock": ["ctrl_age_frac", "ctrl_t_frac"],
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
    feature_groups["fused_continuation"] = (
        feature_groups["action_kinematics"]
        + feature_groups["dit_dynamics"]
        + feature_groups["visual_vlm_dynamics"]
    )
    all_feat_names = sorted({f for cols in feature_groups.values() for f in cols})

    def _eval_cohort(sc_list: list[dict], pf_list: list[dict], horizon_decs: int, pre_regrasp_only: bool) -> dict:
        rows_feat, rows_y, rows_rid, rows_task = [], [], [], []
        for ev in sc_list + pf_list:
            max_end = ev["confirmed_step"] + horizon_decs * 16
            if ev["restored_step"] is not None:
                max_end = min(max_end, int(ev["restored_step"]))
            if pre_regrasp_only and ev["next_regrasp_step"] is not None and ev["next_regrasp_step"] > ev["confirmed_step"]:
                max_end = min(max_end, int(ev["next_regrasp_step"]))
            max_end = min(max_end, int(ev["num_steps"]) + 1)
            fdict = extract_window_continuation_features(
                meta_by_rid[ev["rollout_id"]],
                b_arrays,
                v_arrays,
                relation=ev["relation"],
                onset_step=ev["onset_step"],
                confirmed_step=ev["confirmed_step"],
                window_end_step=max_end,
            )
            if fdict is None:
                continue
            rows_feat.append(fdict)
            rows_y.append(0 if ev["success"] else 1)  # 1 = persistent failure
            rows_rid.append(ev["rollout_id"])
            rows_task.append(ev["task"])

        y_arr = np.asarray(rows_y, dtype=int)
        # Univariate AUROC + permutation p-value + mean(self_corr) vs mean(pers_fail)
        univariate = {}
        for fn in all_feat_names:
            vals = np.array([r[fn] for r in rows_feat], dtype=float)
            au = _auroc(y_arr, vals)
            pval = _perm_pvalue_two_sided(y_arr, vals, n_perm=4000, seed=0)
            univariate[fn] = {
                "auroc_fail_higher": round(au, 4) if au is not None else None,
                "abs_separation": round(abs(au - 0.5) + 0.5, 4) if au is not None else None,
                "direction": "higher_in_failure" if (au is not None and au >= 0.5) else "higher_in_self_corr",
                "perm_p_two_sided": round(pval, 4) if pval is not None else None,
                "mean_self_corr": round(float(vals[y_arr == 0].mean()), 4) if (y_arr == 0).any() else None,
                "mean_pers_fail": round(float(vals[y_arr == 1].mean()), 4) if (y_arr == 1).any() else None,
            }

        # LORO and strict unseen-task LOTO logistic continuation classifiers
        group_cv = {}
        for gname, cols in feature_groups.items():
            X = np.array([[r[c] for c in cols] for r in rows_feat], dtype=np.float64)
            oof_loro = _fit_ridge_logistic_oof(X, y_arr, rows_rid, l2=2.0)
            oof_loto = _fit_ridge_logistic_oof(X, y_arr, rows_task, l2=2.0)
            group_cv[gname] = {
                "loro_auroc": round(_auroc(y_arr, oof_loro) or 0.5, 4),
                "loro_perm_p": round(_perm_pvalue_two_sided(y_arr, oof_loro, n_perm=4000, seed=1) or 1.0, 4),
                "loto_auroc": round(_auroc(y_arr, oof_loto) or 0.5, 4),
                "loto_perm_p": round(_perm_pvalue_two_sided(y_arr, oof_loto, n_perm=4000, seed=2) or 1.0, 4),
            }

        # Also evaluate raw high-dim XR-1 DiT mean vector inside the pre-recovery window (strong L2!)
        X_dit_raw = []
        for rid, ev in zip(rows_rid, [first_by_rid[r] for r in rows_rid]):
            epm = meta_by_rid[rid]
            s0, e0 = int(epm["start_row"]), int(epm["end_row"])
            ds = np.asarray(epm["decision_steps"], dtype=int)
            max_end = ev["confirmed_step"] + horizon_decs * 16
            if ev["restored_step"] is not None:
                max_end = min(max_end, int(ev["restored_step"]))
            if pre_regrasp_only and ev["next_regrasp_step"] is not None and ev["next_regrasp_step"] > ev["confirmed_step"]:
                max_end = min(max_end, int(ev["next_regrasp_step"]))
            wl = np.flatnonzero((ds >= ev["confirmed_step"]) & (ds < min(max_end, int(ev["num_steps"]) + 1)))
            dit_ep = b_arrays["dit_last"][s0:e0].astype(np.float32)
            X_dit_raw.append(dit_ep[wl].mean(axis=0))
        X_dr = np.stack(X_dit_raw, axis=0)
        oof_dr_loro = _fit_ridge_logistic_oof(X_dr, y_arr, rows_rid, l2=50.0)
        oof_dr_loto = _fit_ridge_logistic_oof(X_dr, y_arr, rows_task, l2=50.0)
        group_cv["xr1_dit_raw_1024d"] = {
            "loro_auroc": round(_auroc(y_arr, oof_dr_loro) or 0.5, 4),
            "loro_perm_p": round(_perm_pvalue_two_sided(y_arr, oof_dr_loro, n_perm=4000, seed=3) or 1.0, 4),
            "loto_auroc": round(_auroc(y_arr, oof_dr_loto) or 0.5, 4),
            "loto_perm_p": round(_perm_pvalue_two_sided(y_arr, oof_dr_loto, n_perm=4000, seed=4) or 1.0, 4),
        }

        return {
            "n_self_corr": int((y_arr == 0).sum()),
            "n_pers_fail": int((y_arr == 1).sum()),
            "horizon_decisions": horizon_decs,
            "pre_regrasp_only": pre_regrasp_only,
            "univariate": univariate,
            "group_cv": group_cv,
        }

    cohorts = {
        "strict_multi_pre_restore_h8": _eval_cohort(sc_multi, pers_fail, horizon_decs=fixed_horizon_decisions, pre_regrasp_only=False),
        "strict_multi_pre_regrasp_h8": _eval_cohort(sc_multi, pers_fail, horizon_decs=fixed_horizon_decisions, pre_regrasp_only=True),
        "strict_multi_pre_restore_h3": _eval_cohort(sc_multi, pers_fail, horizon_decs=3, pre_regrasp_only=False),
        "broad_all_succ_pre_restore_h8": _eval_cohort(sc_all, pers_fail, horizon_decs=fixed_horizon_decisions, pre_regrasp_only=False),
        "broad_all_succ_pre_regrasp_h8": _eval_cohort(sc_all, pers_fail, horizon_decs=fixed_horizon_decisions, pre_regrasp_only=True),
    }

    # Matched-pair win rates across all interpretable univariate features (strict equal-duration windows!)
    pair_feature_wins = {}
    for win_kind, skey, fkey in (
        ("pre_restore", "pre_restore_success", "pre_restore_failure"),
        ("pre_regrasp", "pre_regrasp_success", "pre_regrasp_failure"),
    ):
        valid_pairs = [p for p in matched_pair_records if p[skey] is not None and p[fkey] is not None]
        wins = {}
        for fn in all_feat_names:
            sc_gt = sum(p[skey][fn] > p[fkey][fn] for p in valid_pairs)
            pf_gt = sum(p[fkey][fn] > p[skey][fn] for p in valid_pairs)
            wins[fn] = {
                "n_pairs": len(valid_pairs),
                "self_corr_higher": sc_gt,
                "pers_fail_higher": pf_gt,
                "mean_sc": round(float(np.mean([p[skey][fn] for p in valid_pairs])), 4) if valid_pairs else None,
                "mean_pf": round(float(np.mean([p[fkey][fn] for p in valid_pairs])), 4) if valid_pairs else None,
            }
        pair_feature_wins[win_kind] = wins

    summary = {
        "schema_version": 1,
        "bundle_dir": str(bundle_dir),
        "vlm_dir": str(vlm_dir) if vlm_dir is not None else None,
        "cohort_counts": {
            "strict_multi_decision_self_corrected": len(sc_multi),
            "broad_confirmed_mistake_successes": len(sc_all),
            "persistent_failures": len(pers_fail),
            "matched_pairs": len(matched_pair_records),
        },
        "matched_pairs_detail": matched_pair_records,
        "matched_pair_feature_wins": pair_feature_wins,
        "cohort_evaluations": cohorts,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "out"
        staged.mkdir()
        (staged / "test2_continuation_pilot.json").write_text(json.dumps(summary, indent=2) + "\n")
        staged.rename(output_dir)

    print("=== TEST-2 CONTINUATION-AWARE PILOT CHECK (120 ROLLOUTS) ===")
    print(f"Cohort: {summary['cohort_counts']}")

    for cname, cdata in cohorts.items():
        print(
            f"\n--- {cname} (self_corr={cdata['n_self_corr']}, pers_fail={cdata['n_pers_fail']}, "
            f"H={cdata['horizon_decisions']} dec, pre_regrasp={cdata['pre_regrasp_only']}) ---"
        )
        print("  Cross-validated continuation probes (1=persistent failure, 0=self-corrected success):")
        for gname, gm in cdata["group_cv"].items():
            print(
                f"    {gname:24s} LORO_AUROC={gm['loro_auroc']:.4f} (p={gm['loro_perm_p']:.4f}) | "
                f"LOTO_AUROC={gm['loto_auroc']:.4f} (p={gm['loto_perm_p']:.4f})"
            )
        ranked = sorted(
            [(fn, m) for fn, m in cdata["univariate"].items() if m["abs_separation"] is not None],
            key=lambda x: (-x[1]["abs_separation"], x[1]["perm_p_two_sided"]),
        )
        print("  Top 8 univariate continuation signals:")
        for fn, m in ranked[:8]:
            print(
                f"    {fn:24s} sep={m['abs_separation']:.4f} (p={m['perm_p_two_sided']:.4f}, {m['direction']:19s}) | "
                f"SC={m['mean_self_corr']:+.4f} vs PF={m['mean_pers_fail']:+.4f}"
            )

    print("\n=== EXACT EQUAL-DURATION MATCHED PAIRS (n=4, SAME TASK+RELATION) ===")
    for rec in matched_pair_records:
        p = rec["pair"]
        print(
            f"  {p['task']:20s} {p['relation']:30s} | SC={p['success_rollout_id'][:8]} vs PF={p['failure_rollout_id'][:8]} | "
            f"w_restore={rec['w_restore_steps']}st ({p['matched_window_decisions']}dec), w_pre_regrasp={rec['w_pre_regrasp_steps']}st"
        )
        for tag, sk, fk in (("pre_restore", "pre_restore_success", "pre_restore_failure"), ("pre_regrasp", "pre_regrasp_success", "pre_regrasp_failure")):
            s_f, f_f = rec[sk], rec[fk]
            if s_f is None or f_f is None:
                continue
            print(
                f"    [{tag:11s}] eef_spd={s_f['act_eef_speed_mean']:.3f}/{f_f['act_eef_speed_mean']:.3f} "
                f"base_spd={s_f['act_base_speed_mean']:.3f}/{f_f['act_base_speed_mean']:.3f} "
                f"grip_rng={s_f['act_grip_range']:.2f}/{f_f['act_grip_range']:.2f} "
                f"dit_vel={s_f['dit_step_vel']:.2f}/{f_f['dit_step_vel']:.2f} "
                f"dit_cos={s_f['dit_cos_conf']:.3f}/{f_f['dit_cos_conf']:.3f} "
                f"wrist_vel={s_f['vis_wrist_step_vel']:.2f}/{f_f['vis_wrist_step_vel']:.2f} "
                f"vlm_dl={s_f['vlm_logit_delta']:+.2f}/{f_f['vlm_logit_delta']:+.2f}"
            )

    print("\n  Features with 4/4 (or 3/4) directional agreement on matched pre_regrasp pairs:")
    for fn, w in sorted(pair_feature_wins["pre_regrasp"].items()):
        n_p = w["n_pairs"]
        hi = max(w["self_corr_higher"], w["pers_fail_higher"])
        if n_p > 0 and hi >= max(3, n_p - 1):
            who = "SC>PF" if w["self_corr_higher"] >= w["pers_fail_higher"] else "PF>SC"
            print(f"    {fn:24s} {hi}/{n_p} ({who}) | mean SC={w['mean_sc']:+.4f} vs PF={w['mean_pf']:+.4f}")

    print("Wrote:", output_dir / "test2_continuation_pilot.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--vlm-scores", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-decisions", type=int, default=8)
    args = parser.parse_args(argv)
    run_test2_pilot(
        args.bundle,
        args.vlm_scores,
        args.output,
        fixed_horizon_decisions=args.horizon_decisions,
    )


if __name__ == "__main__":
    main()
