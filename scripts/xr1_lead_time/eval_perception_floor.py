"""Perception-floor probe and end-to-end camera-predicted alarm evaluator.

Reads the bundle created by ``extract_perception_features.py`` and runs:
1. Within-task 5-fold cross-fitted relation probing (strictly split by complete
   episode) across six observation streams:
   ``proprio_actions``, ``xr1_dit``, ``siglip_wrist``, ``siglip_3cam``,
   ``siglip_3cam_plus_state``, and ``all_fused``.
2. Leave-one-task-out (LOTO) zero-shot transfer on universal shared primitives
   (``any_object_grasped`` and ``any_placement_satisfied``).
3. End-to-end substitution of out-of-fold camera-predicted relations into the
   exact 6-arm split-conformal oracle pipeline (nominal 5% FPR), plus component
   substitution (true grasp + pred destination vs. pred grasp + true destination)
   to pinpoint where any lost warning lead comes from.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn

from . import analyze_recovery_pilot6 as pilot6


STREAMS = (
    "proprio_actions",
    "xr1_dit",
    "siglip_wrist",
    "siglip_3cam",
    "siglip_3cam_plus_state",
    "all_fused",
)


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = y_true.astype(int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y), dtype=float)
    s_sorted = y_score[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos_ranks = float(ranks[y == 1].sum())
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = y_true.astype(int)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    y_ord = y[order]
    tp = np.cumsum(y_ord)
    prec = tp / np.arange(1, len(y_ord) + 1)
    return float(prec[y_ord == 1].sum() / n_pos)


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_th, best_f1 = 0.5, -1.0
    for th in np.linspace(0.15, 0.85, 29):
        pred = y_prob >= th
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum((~pred) & (y_true == 1)))
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    return best_th, float(best_f1)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y = y_true.astype(int)
    pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "auroc": _auroc(y, y_prob),
        "ap": _average_precision(y, y_prob),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "threshold": round(float(threshold), 3),
    }


def _causal_temporal_augment(X_ep: np.ndarray) -> np.ndarray:
    """Append causal 1-decision delta ``x_t - x_{t-1}`` (zero at t=0) without future leakage."""
    deltas = np.zeros_like(X_ep)
    if len(X_ep) > 1:
        deltas[1:] = X_ep[1:] - X_ep[:-1]
    return np.concatenate([X_ep, deltas], axis=1)


def build_stream_matrix(arrays: dict[str, np.ndarray], episodes: list[dict], stream: str) -> np.ndarray:
    """Assemble strictly causal per-decision feature matrix for ``stream``."""
    state = arrays["state_history"].astype(np.float32)
    chunks = arrays["action_chunks"].astype(np.float32)
    exec_act = arrays["executed_actions"].astype(np.float32)
    proprio = np.concatenate([state, chunks, exec_act], axis=1)

    dit = np.concatenate(
        [arrays["dit_last"].astype(np.float32), arrays["dit_mean"].astype(np.float32)], axis=1
    )

    has_vis = "siglip_3cam" in arrays
    if stream == "proprio_actions":
        raw = proprio
    elif stream == "xr1_dit":
        raw = dit
    elif stream == "siglip_wrist":
        if not has_vis:
            raise ValueError("siglip_3cam not present in bundle")
        raw = arrays["siglip_3cam"][:, 2, :].astype(np.float32)
    elif stream == "siglip_3cam":
        if not has_vis:
            raise ValueError("siglip_3cam not present in bundle")
        vis = arrays["siglip_3cam"].astype(np.float32)
        raw = vis.reshape(len(vis), -1)
    elif stream == "siglip_3cam_plus_state":
        if not has_vis:
            raise ValueError("siglip_3cam not present in bundle")
        vis = arrays["siglip_3cam"].astype(np.float32).reshape(len(state), -1)
        raw = np.concatenate([vis, proprio], axis=1)
    elif stream == "all_fused":
        if not has_vis:
            raise ValueError("siglip_3cam not present in bundle")
        vis = arrays["siglip_3cam"].astype(np.float32).reshape(len(state), -1)
        raw = np.concatenate([vis, proprio, dit], axis=1)
    else:
        raise ValueError(f"Unknown stream: {stream}")

    # Add causal 1-step delta within each episode independently (never across episode boundary).
    out = np.empty((len(raw), raw.shape[1] * 2), dtype=np.float32)
    for ep in episodes:
        s, e = ep["start_row"], ep["end_row"]
        out[s:e] = _causal_temporal_augment(raw[s:e])
    return out


def fit_predict_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    device: torch.device,
    hidden_dim: int = 0,
    epochs: int = 80,
    lr: float = 3e-3,
    weight_decay: float = 1e-2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an L2-regularized linear (hidden_dim=0) or 1-hidden-layer MLP binary probe.

    Standardizes using training-fold statistics only. Returns
    ``(train_probs, eval_probs)``.
    """
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos == 0 or n_neg == 0:
        const = float(n_pos > 0)
        return np.full(len(y_train), const, dtype=np.float32), np.full(len(X_eval), const, dtype=np.float32)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    Xt = torch.from_numpy(((X_train - mean) / std).astype(np.float32)).to(device)
    Xe = torch.from_numpy(((X_eval - mean) / std).astype(np.float32)).to(device)
    yt = torch.from_numpy(y_train.astype(np.float32)).to(device)

    torch.manual_seed(seed)
    d_in = Xt.shape[1]
    if hidden_dim > 0:
        net: nn.Module = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 1),
        ).to(device)
    else:
        net = nn.Linear(d_in, 1).to(device)

    pos_weight = torch.tensor([min(10.0, max(0.2, n_neg / max(1, n_pos)))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = net(Xt).squeeze(-1)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(net(Xt).squeeze(-1)).cpu().numpy().astype(np.float32)
        p_ev = torch.sigmoid(net(Xe).squeeze(-1)).cpu().numpy().astype(np.float32)
    del Xt, Xe, yt, net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return p_tr, p_ev


def _episode_folds(episodes: list[dict], num_folds: int = 5) -> dict[str, int]:
    groups = defaultdict(list)
    for ep in episodes:
        groups[(ep["success"], ep["task"])].append(ep)
    fold_of = {}
    offset = 0
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda e: hashlib.sha256(f"pilot6-v1/fold/{e['rollout_id']}".encode()).digest(),
        )
        for ep in ordered:
            fold_of[ep["rollout_id"]] = offset % num_folds
            offset += 1
    return fold_of


def _causal_smooth_and_threshold(probs: np.ndarray, threshold: float, ema_alpha: float = 0.65) -> np.ndarray:
    """Strictly causal exponential smoothing followed by thresholding."""
    sm = np.empty_like(probs, dtype=np.float32)
    val = float(probs[0])
    sm[0] = val
    for i in range(1, len(probs)):
        val = ema_alpha * float(probs[i]) + (1.0 - ema_alpha) * val
        sm[i] = val
    return sm >= threshold


def _transition_latency(true_bool: np.ndarray, pred_bool: np.ndarray, decision_steps: np.ndarray) -> list[int]:
    """Steps from first true 0->1 onset to first subsequent pred 1 (when both occur)."""
    t_on = np.flatnonzero(true_bool)
    p_on = np.flatnonzero(pred_bool)
    if t_on.size == 0 or p_on.size == 0:
        return []
    first_t = int(decision_steps[t_on[0]])
    after = p_on[decision_steps[p_on] >= first_t]
    if after.size == 0:
        return []
    return [int(decision_steps[after[0]] - first_t)]


def _build_substituted_episode(
    ep: dict,
    pred_bool_map: dict[str, np.ndarray],
    *,
    replace_grasps: bool,
    replace_targets: bool,
    use_inferred_stage: bool,
) -> pilot6.EpisodeData:
    """Substitute causal decision-level predicted relations into EpisodeData.

    When ``replace_grasps`` / ``replace_targets`` is True, the corresponding
    predicates are stepped causally at policy decisions (zero-order hold between
    decisions) using ONLY out-of-fold predictions; otherwise simulator truth is
    retained (allowing exact component-substitution diagnostics).
    """
    task = ep["task"]
    spec = pilot6.TASK_SPECS[task]
    n_steps = int(ep["num_steps"])
    dec_steps = np.asarray(ep["decision_steps"], dtype=int)

    grasp_preds = {gp for _, gps, _ in spec["placements"] for gp in gps} | set(spec["wrong_grasps"])
    target_preds = {tp for _, _, tp in spec["placements"]} | set(spec["hold_conditions"])

    # Reconstruct sparse recovery_observations with selected predicates replaced by
    # causal decision-level predictions (held constant between decisions).
    orig_hist = ep["recovery_observations"]
    pred_names = sorted(orig_hist[0]["values"].keys())
    series = {}
    for pname in pred_names:
        should_replace = (pname in grasp_preds and replace_grasps) or (
            pname in target_preds and replace_targets
        )
        if should_replace and pname in pred_bool_map:
            arr = np.zeros(n_steps + 1, dtype=bool)
            p_dec = pred_bool_map[pname]
            for i, t in enumerate(dec_steps):
                end = int(dec_steps[i + 1]) if i + 1 < len(dec_steps) else n_steps + 1
                arr[int(t) : end] = bool(p_dec[i])
            series[pname] = arr
        else:
            series[pname] = pilot6.expand_recovery_series(orig_hist, n_steps, pname)

    new_hist = []
    prev_vals = None
    for step in range(n_steps + 1):
        vals = {p: bool(series[p][step]) for p in pred_names}
        if step == 0 or vals != prev_vals or step == n_steps:
            new_hist.append({"environment_step": step, "values": vals})
            prev_vals = vals

    # Optionally build an inferred stage timeline strictly from predicted placement targets.
    # Require two consecutive positive decisions to latch a stage transition (suppresses 1-step spikes).
    if use_inferred_stage:
        ordered_targets = [tp for _, _, tp in spec["placements"]]
        latched = {tp: False for tp in ordered_targets}
        consec = {tp: 0 for tp in ordered_targets}
        inferred_stages = []
        inferred_segments = []
        cur_stage = None
        cur_start = 0
        for i, t in enumerate(dec_steps):
            for tp in ordered_targets:
                if series[tp][int(t)]:
                    consec[tp] += 1
                    if consec[tp] >= 2:
                        latched[tp] = True
                else:
                    consec[tp] = 0
            stage_idx = sum(latched.values())
            sid = f"inferred_stage_{stage_idx}"
            inferred_stages.append(sid)
            if sid != cur_stage:
                if cur_stage is not None:
                    inferred_segments.append(
                        {
                            "subtask_id": cur_stage,
                            "entry_environment_step": cur_start,
                            "end_environment_step": int(t),
                            "completed": True,
                        }
                    )
                cur_stage = sid
                cur_start = int(t)
        if cur_stage is not None:
            inferred_segments.append(
                {
                    "subtask_id": cur_stage,
                    "entry_environment_step": cur_start,
                    "end_environment_step": n_steps,
                    "completed": bool(ep["success"]),
                }
            )
        inf_records = [
            {"environment_step": int(t), "inference_index": i, "subtask_id": stg}
            for i, (t, stg) in enumerate(zip(dec_steps, inferred_stages))
        ]
        segments = inferred_segments
    else:
        inf_records = [
            {"environment_step": int(t), "inference_index": i, "subtask_id": stg}
            for i, (t, stg) in enumerate(zip(dec_steps, ep["decision_stages"]))
        ]
        segments = ep["segments"]

    row = {
        "rollout_id": ep["rollout_id"],
        "task_name": task,
        "environment_seed": ep["seed"],
        "failed": not ep["success"],
        "rollout_horizon": ep["horizon"],
        "num_environment_steps": n_steps,
    }
    trace = {
        "task_name": task,
        "inference_records": inf_records,
        "recovery_observations": new_hist,
        "segments": segments,
    }
    return pilot6.extract_episode(row, trace, min_grasp_steps=16, latch_steps=16, loss_confirm_steps=16, miss_confirm_steps=32)


def run_evaluation(
    bundle_dir: Path,
    output_dir: Path,
    *,
    streams: list[str] | None = None,
    hidden_dim: int = 0,
    epochs: int = 80,
    num_folds: int = 5,
    primary_stream: str = "siglip_3cam_plus_state",
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")

    index_path = bundle_dir / "perception_index.json"
    npz_path = bundle_dir / "perception_features.npz"
    index = json.loads(index_path.read_text())
    arrays = dict(np.load(npz_path))
    episodes = index["episodes"]
    fold_of = _episode_folds(episodes, num_folds=num_folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if streams is None:
        streams = list(STREAMS) if "siglip_3cam" in arrays else ["proprio_actions", "xr1_dit"]

    # Map tasks -> predicates
    task_predicates: dict[str, list[str]] = {}
    for ep in episodes:
        t = ep["task"]
        if t not in task_predicates:
            task_predicates[t] = sorted(ep["predicates_at_decisions"].keys())

    # 1. Within-task 5-fold cross-fitted relation probing per stream
    within_task_results: dict[str, dict] = {}
    oof_pred_bool_by_stream: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for stream in streams:
        X_all = build_stream_matrix(arrays, episodes, stream)
        per_task_out = {}
        oof_bool_ep: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        all_aurocs, all_f1s, all_grasp_f1s, all_dest_f1s = [], [], [], []

        for task, pnames in sorted(task_predicates.items()):
            task_eps = [e for e in episodes if e["task"] == task]
            grasp_set = {gp for _, gps, _ in pilot6.TASK_SPECS[task]["placements"] for gp in gps}
            dest_set = {tp for _, _, tp in pilot6.TASK_SPECS[task]["placements"]}
            per_pred = {}
            for pname in pnames:
                oof_probs = {}
                oof_thresholds = {}
                latencies = []
                for f in range(num_folds):
                    tr_eps = [e for e in task_eps if fold_of[e["rollout_id"]] != f]
                    ev_eps = [e for e in task_eps if fold_of[e["rollout_id"]] == f]
                    if not tr_eps or not ev_eps:
                        continue
                    X_tr = np.concatenate([X_all[e["start_row"] : e["end_row"]] for e in tr_eps], axis=0)
                    y_tr = np.concatenate([np.asarray(e["predicates_at_decisions"][pname], dtype=np.float32) for e in tr_eps])
                    X_ev = np.concatenate([X_all[e["start_row"] : e["end_row"]] for e in ev_eps], axis=0)
                    p_tr, p_ev = fit_predict_probe(
                        X_tr, y_tr, X_ev, device=device, hidden_dim=hidden_dim, epochs=epochs, seed=f
                    )
                    th, _ = _best_f1_threshold(y_tr, p_tr)
                    cur = 0
                    for e in ev_eps:
                        ln = e["end_row"] - e["start_row"]
                        ep_prob = p_ev[cur : cur + ln]
                        cur += ln
                        oof_probs[e["rollout_id"]] = ep_prob
                        oof_thresholds[e["rollout_id"]] = th
                        ep_bool = _causal_smooth_and_threshold(ep_prob, th)
                        oof_bool_ep[e["rollout_id"]][pname] = ep_bool
                        latencies.extend(
                            _transition_latency(
                                np.asarray(e["predicates_at_decisions"][pname], dtype=bool),
                                ep_bool,
                                np.asarray(e["decision_steps"], dtype=int),
                            )
                        )

                y_cat = np.concatenate([np.asarray(e["predicates_at_decisions"][pname], dtype=int) for e in task_eps])
                p_cat = np.concatenate([oof_probs[e["rollout_id"]] for e in task_eps])
                b_cat = np.concatenate([oof_bool_ep[e["rollout_id"]][pname].astype(int) for e in task_eps])
                m = _metrics(y_cat, p_cat, float(np.median(list(oof_thresholds.values()))))
                # Also record exact smoothed-OOF F1
                tp = int(np.sum((b_cat == 1) & (y_cat == 1)))
                fp = int(np.sum((b_cat == 1) & (y_cat == 0)))
                fn = int(np.sum((b_cat == 0) & (y_cat == 1)))
                m["smoothed_oof_f1"] = round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 0.0
                m["median_onset_latency_steps"] = float(np.median(latencies)) if latencies else None
                per_pred[pname] = m
                if m["auroc"] is not None:
                    all_aurocs.append(m["auroc"])
                all_f1s.append(m["smoothed_oof_f1"])
                if pname in grasp_set:
                    all_grasp_f1s.append(m["smoothed_oof_f1"])
                if pname in dest_set:
                    all_dest_f1s.append(m["smoothed_oof_f1"])
            per_task_out[task] = per_pred

        within_task_results[stream] = {
            "macro_auroc": round(float(np.mean(all_aurocs)), 4) if all_aurocs else None,
            "macro_f1": round(float(np.mean(all_f1s)), 4) if all_f1s else 0.0,
            "macro_grasp_f1": round(float(np.mean(all_grasp_f1s)), 4) if all_grasp_f1s else 0.0,
            "macro_destination_f1": round(float(np.mean(all_dest_f1s)), 4) if all_dest_f1s else 0.0,
            "by_task": per_task_out,
        }
        oof_pred_bool_by_stream[stream] = oof_bool_ep

    # 2. Leave-one-task-out (LOTO) transfer on universal shared primitives
    loto_results: dict[str, dict] = {}
    universal_labels = {"any_object_grasped": {}, "any_placement_satisfied": {}}
    for ep in episodes:
        spec = pilot6.TASK_SPECS[ep["task"]]
        grasp_keys = [gp for _, gps, _ in spec["placements"] for gp in gps]
        dest_keys = [tp for _, _, tp in spec["placements"]]
        n = ep["end_row"] - ep["start_row"]
        g_any = np.zeros(n, dtype=bool)
        for k in grasp_keys:
            g_any |= np.asarray(ep["predicates_at_decisions"][k], dtype=bool)
        d_any = np.zeros(n, dtype=bool)
        for k in dest_keys:
            d_any |= np.asarray(ep["predicates_at_decisions"][k], dtype=bool)
        universal_labels["any_object_grasped"][ep["rollout_id"]] = g_any.astype(np.float32)
        universal_labels["any_placement_satisfied"][ep["rollout_id"]] = d_any.astype(np.float32)

    tasks_sorted = sorted(task_predicates)
    for stream in streams:
        X_all = build_stream_matrix(arrays, episodes, stream)
        stream_loto = {}
        for prim, ep_y in universal_labels.items():
            per_held = {}
            aurocs, f1s = [], []
            for held_task in tasks_sorted:
                tr_eps = [e for e in episodes if e["task"] != held_task]
                ev_eps = [e for e in episodes if e["task"] == held_task]
                X_tr = np.concatenate([X_all[e["start_row"] : e["end_row"]] for e in tr_eps], axis=0)
                y_tr = np.concatenate([ep_y[e["rollout_id"]] for e in tr_eps])
                X_ev = np.concatenate([X_all[e["start_row"] : e["end_row"]] for e in ev_eps], axis=0)
                y_ev = np.concatenate([ep_y[e["rollout_id"]] for e in ev_eps])
                p_tr, p_ev = fit_predict_probe(
                    X_tr, y_tr, X_ev, device=device, hidden_dim=hidden_dim, epochs=epochs, seed=0
                )
                th, _ = _best_f1_threshold(y_tr, p_tr)
                m = _metrics(y_ev, p_ev, th)
                per_held[held_task] = m
                if m["auroc"] is not None:
                    aurocs.append(m["auroc"])
                f1s.append(m["f1"])
            stream_loto[prim] = {
                "macro_loto_auroc": round(float(np.mean(aurocs)), 4) if aurocs else None,
                "macro_loto_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
                "by_held_out_task": per_held,
            }
        loto_results[stream] = stream_loto

    # 3. End-to-end substitution into the 6-arm split-conformal oracle pipeline
    if primary_stream not in oof_pred_bool_by_stream:
        primary_stream = streams[-1]

    substitutions = {}
    sub_configs = [
        ("oracle_reference", False, False, False),
        ("true_grasp_pred_dest", False, True, False),
        ("pred_grasp_true_dest", True, False, False),
        ("pred_all_true_stage", True, True, False),
        ("fully_observed_camera_monitor", True, True, True),
    ]
    for name, rep_g, rep_t, inf_stg in sub_configs:
        sub_eps = [
            _build_substituted_episode(
                ep,
                oof_pred_bool_by_stream[primary_stream][ep["rollout_id"]],
                replace_grasps=rep_g,
                replace_targets=rep_t,
                use_inferred_stage=inf_stg,
            )
            for ep in episodes
        ]
        ev = pilot6.evaluate_cross_fitted(sub_eps, num_folds=num_folds)
        ev.pop("oof_alarms", None)
        ev.pop("folds", None)
        substitutions[name] = ev

    # Also evaluate fully observed camera monitor for every stream!
    fully_observed_by_stream = {}
    for stream in streams:
        sub_eps = [
            _build_substituted_episode(
                ep,
                oof_pred_bool_by_stream[stream][ep["rollout_id"]],
                replace_grasps=True,
                replace_targets=True,
                use_inferred_stage=True,
            )
            for ep in episodes
        ]
        ev = pilot6.evaluate_cross_fitted(sub_eps, num_folds=num_folds)
        fully_observed_by_stream[stream] = {
            "global_timer": ev["arms"]["global_timer"],
            "task_timer": ev["arms"]["task_timer"],
            "inferred_stage_clock": ev["arms"]["stage_clock"],
            "camera_full_physical": ev["arms"]["full_physical"],
            "full_minus_global_timer": ev["paired"]["full_physical_minus_global_timer"],
            "full_minus_inferred_stage_clock": ev["paired"]["full_physical_minus_stage_clock"],
            "inferred_stage_minus_task_timer": ev["paired"]["stage_clock_minus_task_timer"],
        }

    summary = {
        "schema_version": 1,
        "bundle_dir": str(bundle_dir),
        "bundle_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "config": {
            "streams": streams,
            "primary_stream": primary_stream,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "num_folds": num_folds,
        },
        "within_task_probes": within_task_results,
        "loto_universal_probes": loto_results,
        "component_substitution": substitutions,
        "fully_observed_by_stream": fully_observed_by_stream,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "out"
        staged.mkdir()
        (staged / "perception_floor_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        staged.rename(output_dir)

    print("=== WITHIN-TASK 5-FOLD RELATION PROBING ===")
    for st in streams:
        r = within_task_results[st]
        print(
            f"  {st:24s} macro_AUROC={r['macro_auroc']}  macro_F1={r['macro_f1']:.3f}  "
            f"grasp_F1={r['macro_grasp_f1']:.3f}  dest_F1={r['macro_destination_f1']:.3f}"
        )
    print("\n=== LOTO ZERO-SHOT UNIVERSAL PRIMITIVE TRANSFER ===")
    for st in streams:
        g = loto_results[st]["any_object_grasped"]
        d = loto_results[st]["any_placement_satisfied"]
        print(
            f"  {st:24s} grasp_AUROC={g['macro_loto_auroc']} F1={g['macro_loto_f1']:.3f} | "
            f"place_AUROC={d['macro_loto_auroc']} F1={d['macro_loto_f1']:.3f}"
        )
    print(f"\n=== COMPONENT SUBSTITUTION (stream={primary_stream}, 5% nominal FPR) ===")
    for cname, ev in substitutions.items():
        fp = ev["arms"]["full_physical"]
        sc = ev["arms"]["stage_clock"]
        gt = ev["arms"]["global_timer"]
        g_vs_sc = ev["paired"]["full_physical_minus_stage_clock"]
        g_vs_g = ev["paired"]["full_physical_minus_global_timer"]
        print(
            f"  {cname:30s} full={fp['macro_lead_frac']*100:5.2f}% ({fp['macro_lead_sec']:4.1f}s) "
            f"FPR={fp['realized_fpr']*100:4.1f}% | vs_stage={g_vs_sc['mean_gain_frac']*100:+5.2f}pp "
            f"| vs_G={g_vs_g['mean_gain_frac']*100:+5.2f}pp ({g_vs_g['mean_gain_sec']:+4.1f}s)"
        )
    print("\n=== FULLY OBSERVED MONITOR BY STREAM (NO PRIVILEGED INPUTS) ===")
    for st in streams:
        info = fully_observed_by_stream[st]
        fp = info["camera_full_physical"]
        istg = info["inferred_stage_clock"]
        vs_g = info["full_minus_global_timer"]
        vs_stg = info["full_minus_inferred_stage_clock"]
        print(
            f"  {st:24s} inf_stage={istg['macro_lead_frac']*100:5.2f}% ({istg['macro_lead_sec']:4.1f}s, FPR={istg['realized_fpr']*100:4.1f}%) | "
            f"cam_full={fp['macro_lead_frac']*100:5.2f}% ({fp['macro_lead_sec']:4.1f}s, FPR={fp['realized_fpr']*100:4.1f}%) | "
            f"vs_G={vs_g['mean_gain_frac']*100:+5.2f}pp ({vs_g['mean_gain_sec']:+4.1f}s) | vs_inf_stage={vs_stg['mean_gain_frac']*100:+5.2f}pp"
        )
    print("Wrote:", output_dir / "perception_floor_summary.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--streams", nargs="+")
    parser.add_argument("--hidden-dim", type=int, default=0, help="0=linear probe, >0=1-hidden-layer MLP")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--primary-stream", default="siglip_3cam_plus_state")
    args = parser.parse_args(argv)
    run_evaluation(
        args.bundle,
        args.output,
        streams=args.streams,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        num_folds=args.num_folds,
        primary_stream=args.primary_stream,
    )


if __name__ == "__main__":
    main()
