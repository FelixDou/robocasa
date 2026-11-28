"""Zero-re-inference scoring upgrades on cached Qwen3-VL-4B-Instruct logits.

Reads the already-extracted ``perception_features.npz`` bundle and
``vlm_destination_scores.npz`` logits and evaluates three strictly causal,
leak-free algorithmic upgrades to the decoupled grasp + VLM destination monitor:

1. **Per-episode causal baseline subtraction (zero-shot debiasing):**
   Subtracts each episode's own causal initial / pre-grasp logit baseline
   (``delta_t0 = logit(t) - logit(0)`` and ``delta_pregrasp = logit(t) - median(logit[0..g_start])``)
   to remove kitchen- and prompt-specific Yes/No prior bias with zero target-task labels.
   Also reports **post-release AUROC** (evaluating visual discrimination specifically
   at the post-release decision window where the monitor queries the VLM).
2. **Placement-only & verified-hold gating:**
   Separates placement targets from unseen reset-True background hold preconditions
   (e.g., ``pancake_on_plate``, ``plate_on_table`` while facing the fridge) so
   out-of-view background queries do not inject spurious ``loss_age`` into successes.
3. **Continuous soft missed-effect evidence (``E_miss``):**
   Replaces hard 0.5 binary thresholding before split-conformal calibration with
   strictly causal cumulative miss evidence
   ``E_miss(t) = sum_{tau in [t_rel, t]} dt * max(0, 1 - m_tau / clear_th)``,
   where ``m_tau = max_{t_rel <= s <= tau} ema(p_s)`` tracks the best visually
   confirmed placement confidence since release (freezing accumulation cleanly
   once placed, even after the robot drives away to the next subtask, while
   remaining smooth on uncertain successes).
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
from . import eval_vlm_hybrid_monitor as vhyb


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.where(
        arr >= 0.0,
        1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0))),
        np.exp(np.clip(arr, -40.0, 40.0)) / (1.0 + np.exp(np.clip(arr, -40.0, 40.0))),
    ).astype(np.float32)


def causal_delta_t0(logits: np.ndarray) -> np.ndarray:
    """Strictly causal reset-frame baseline subtraction ``logit(t) - logit(0)``."""
    arr = np.asarray(logits, dtype=np.float32)
    if len(arr) == 0:
        return arr.copy()
    return (arr - arr[0]).astype(np.float32)


def causal_delta_pregrasp(logits: np.ndarray, any_grasp_dec: np.ndarray) -> np.ndarray:
    """Strictly causal pre-grasp baseline subtraction.

    At every decision ``k`` prior to the first detected grasp onset ``k0``,
    uses the running causal median of ``logits[0..k]``; once the robot starts
    its first grasp at ``k0``, freezes the baseline at ``median(logits[0..k0])``
    so post-placement frames never contaminate the negative baseline.
    """
    arr = np.asarray(logits, dtype=np.float32)
    g = np.asarray(any_grasp_dec, dtype=bool)
    n = len(arr)
    if n == 0:
        return arr.copy()
    g_on = np.flatnonzero(g)
    first_g = int(g_on[0]) if g_on.size else n - 1
    out = np.empty(n, dtype=np.float32)
    frozen_base = float(np.median(arr[: first_g + 1]))
    for k in range(n):
        base = float(np.median(arr[: k + 1])) if k < first_g else frozen_base
        out[k] = arr[k] - base
    return out


def causal_ema(probs: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float32)
    if len(arr) == 0:
        return arr.copy()
    out = np.empty(len(arr), dtype=np.float32)
    val = float(arr[0])
    out[0] = val
    for i in range(1, len(arr)):
        val = alpha * float(arr[i]) + (1.0 - alpha) * val
        out[i] = val
    return out


def _fit_1d_platt(logits: np.ndarray, y: np.ndarray, *, l2: float = 1e-2) -> tuple[float, float]:
    """Fit convex 2-parameter calibration ``sigmoid(a * x + b)`` (a >= 0.05) via Newton/GD."""
    x = np.asarray(logits, dtype=np.float64)
    yt = np.asarray(y, dtype=np.float64)
    n_pos = float(yt.sum())
    n_neg = float(len(yt) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 1.0, 0.0
    # Balanced class weights so rare positive placements are not ignored
    w_pos = min(8.0, max(1.0, n_neg / n_pos))
    weights = np.where(yt > 0.5, w_pos, 1.0)
    a, b = 1.0, 0.0
    for _ in range(25):
        z = np.clip(a * x + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        r = weights * (p - yt)
        w2 = weights * p * (1.0 - p)
        g_a = float(np.sum(r * x)) / len(x) + l2 * (a - 1.0)
        g_b = float(np.sum(r)) / len(x) + l2 * b
        h_aa = float(np.sum(w2 * x * x)) / len(x) + l2 + 1e-6
        h_ab = float(np.sum(w2 * x)) / len(x)
        h_bb = float(np.sum(w2)) / len(x) + l2 + 1e-6
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (-h_ab * g_a + h_aa * g_b) / det
        a = max(0.05, min(10.0, a - da))
        b = max(-15.0, min(15.0, b - db))
        if abs(da) + abs(db) < 1e-6:
            break
    return float(a), float(b)


def _best_recall_safe_threshold(
    y_true: np.ndarray,
    logits: np.ndarray,
    *,
    min_recall: float = 0.85,
) -> float:
    """Pick highest threshold achieving at least ``min_recall`` on positive labels
    (maximizing precision / F1 subject to protecting successful episodes from FN cliffs).
    """
    y = np.asarray(y_true, dtype=int)
    pos_mask = y == 1
    if int(pos_mask.sum()) == 0:
        return 0.0
    best_th = -4.0
    best_prec = -1.0
    for th in np.linspace(-4.0, 4.0, 81):
        pred = logits >= th
        rec = float(np.mean(pred[pos_mask]))
        if rec + 1e-12 >= min_recall:
            tp = int(np.sum(pred & pos_mask))
            fp = int(np.sum(pred & (~pos_mask)))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            if prec > best_prec:
                best_prec = prec
                best_th = float(th)
    return best_th


def build_soft_monitor_episode(
    ep: dict,
    *,
    grasp_bool_map: dict[str, np.ndarray] | None,
    any_grasp_dec: np.ndarray | None,
    target_prob_map: dict[str, np.ndarray],
    target_bool_map: dict[str, np.ndarray],
    mode: str = "soft_evidence",
    hold_policy: str = "placement_only",
    use_inferred_stage: bool = True,
    use_universal_grasp: bool = False,
    clear_threshold: float = 0.70,
    gamma: float = 1.0,
    min_grasp_steps: int = 16,
    latch_steps: int = 16,
) -> pilot6.EpisodeData:
    """Construct strictly causal EpisodeData supporting:

    - ``mode="hard_binary"``: standard binary threshold missed-effect age.
    - ``mode="soft_evidence"``: continuous probability-weighted missed-effect evidence
      ``E_miss(t) = sum_{tau in [t_rel, t]} dt * max(0, 1 - m_tau / clear_th)^gamma``
      where ``m_tau = max_{t_rel <= s <= tau} ema(p_s)`` tracks best-seen post-release
      placement confidence (resetting on next grasp).
    - ``hold_policy``:
      - ``"placement_only"``: ignore hold-condition loss_age completely.
      - ``"verified_targets_only"``: track loss_age ONLY on placement targets after latching
        (never on unseen reset-True background preconditions like ``pancake_on_plate``).
      - ``"all_holds"``: track loss_age on all ``spec["hold_conditions"]``.
    - ``use_universal_grasp``: if True, uses only ``any_grasp_dec`` (object-agnostic
      progress tracking over targets unsatisfied at grasp onset); if False, uses
      per-operation ``grasp_bool_map`` (or true simulator grasps when ``grasp_bool_map is None``).
    """
    task = ep["task"]
    spec = pilot6.TASK_SPECS[task]
    n_steps = int(ep["num_steps"])
    dec_steps = np.asarray(ep["decision_steps"], dtype=int)
    n_dec = len(dec_steps)
    ordered_targets = [tp for _, _, tp in spec["placements"]]

    def _expand_dec_bool(b_dec: np.ndarray) -> np.ndarray:
        arr = np.zeros(n_steps + 1, dtype=bool)
        for i, t in enumerate(dec_steps):
            end = int(dec_steps[i + 1]) if i + 1 < n_dec else n_steps + 1
            arr[int(t) : end] = bool(b_dec[i])
        return arr

    # Pre-smooth target probabilities strictly causally at decision resolution
    ema_prob_dec = {k: causal_ema(v, alpha=0.65) for k, v in target_prob_map.items()}
    target_bool_step = {k: _expand_dec_bool(v) for k, v in target_bool_map.items()}

    # Stage timeline
    if use_inferred_stage:
        latched = {tp: False for tp in ordered_targets}
        consec = {tp: 0 for tp in ordered_targets}
        decision_stages = []
        stage_entries: dict[str, int] = {}
        stage_durations: dict[str, int] = {}
        cur_stage = None
        cur_start = 0
        for i, t in enumerate(dec_steps):
            for tp in ordered_targets:
                # High-confidence latch: either bool True or EMA prob >= clear_threshold for 2 consecutive decisions
                is_hi = bool(target_bool_map[tp][i]) if mode == "hard_binary" else (float(ema_prob_dec[tp][i]) >= clear_threshold)
                if is_hi:
                    consec[tp] += 1
                    if consec[tp] >= 2:
                        latched[tp] = True
                else:
                    consec[tp] = 0
            sid = f"inferred_stage_{sum(latched.values())}"
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

    release_only_age = np.full(n_dec, -np.inf, dtype=float)
    missed_effect_age = np.full(n_dec, -np.inf, dtype=float)
    loss_age = np.full(n_dec, -np.inf, dtype=float)
    placement_delays: list[int] = []

    def _accumulate_release_window(
        release_step: int,
        search_end: int,
        candidate_targets: list[str],
    ) -> None:
        active_idx = np.flatnonzero((dec_steps >= release_step) & (dec_steps < search_end))
        if active_idx.size == 0:
            return
        release_only_age[active_idx] = np.maximum(
            release_only_age[active_idx], dec_steps[active_idx] - release_step
        )

        if mode == "hard_binary":
            if any(bool(target_bool_step[tp][release_step]) for tp in candidate_targets):
                placement_delays.append(0)
                return
            any_arr = np.zeros(search_end - (release_step + 1), dtype=bool)
            for tp in candidate_targets:
                any_arr |= target_bool_step[tp][release_step + 1 : search_end]
            rel_hits = np.flatnonzero(any_arr)
            if rel_hits.size:
                arr_step = int(release_step + 1 + rel_hits[0])
                placement_delays.append(arr_step - release_step)
                act = (dec_steps >= release_step) & (dec_steps < arr_step)
                missed_effect_age[act] = np.maximum(missed_effect_age[act], dec_steps[act] - release_step)
            else:
                missed_effect_age[active_idx] = np.maximum(
                    missed_effect_age[active_idx], dec_steps[active_idx] - release_step
                )
            return

        # Soft continuous evidence: strictly causal best-seen EMA confidence since release.
        best_p = 0.0
        cum_ev = 0.0
        prev_t = release_step
        cleared_recorded = False
        for d_i in active_idx:
            t_cur = int(dec_steps[d_i])
            p_now = max(float(ema_prob_dec[tp][d_i]) for tp in candidate_targets)
            if p_now > best_p:
                best_p = p_now
            if not cleared_recorded and best_p >= clear_threshold:
                placement_delays.append(max(0, t_cur - release_step))
                cleared_recorded = True
            dt = max(0, t_cur - prev_t)
            deficit = max(0.0, 1.0 - best_p / max(1e-3, clear_threshold))
            if gamma != 1.0:
                deficit = deficit ** gamma
            cum_ev += float(dt) * deficit
            if cum_ev > missed_effect_age[d_i]:
                missed_effect_age[d_i] = cum_ev
            prev_t = t_cur

    if use_universal_grasp:
        assert any_grasp_dec is not None
        any_grasp_step = _expand_dec_bool(any_grasp_dec)
        runs = pilot6._sustained_runs(any_grasp_step, min_grasp_steps)
        for idx, (g_start, g_end) in enumerate(runs):
            if g_end > n_steps:
                continue
            next_g = runs[idx + 1][0] if idx + 1 < len(runs) else None
            search_end = next_g if next_g is not None else n_steps + 1
            # Targets unsatisfied at grasp start (causally determined at g_start!)
            g_dec_i = max(0, int(np.searchsorted(dec_steps, g_start, side="right")) - 1)
            if mode == "hard_binary":
                unsat = [tp for tp in ordered_targets if not bool(target_bool_step[tp][g_start])]
            else:
                unsat = [tp for tp in ordered_targets if float(ema_prob_dec[tp][g_dec_i]) < clear_threshold]
            if not unsat:
                continue
            _accumulate_release_window(g_end, search_end, unsat)
    else:
        orig_hist = ep["recovery_observations"]
        for _, grasp_preds, target_pred in spec["placements"]:
            any_g = np.zeros(n_steps + 1, dtype=bool)
            for gp in grasp_preds:
                if grasp_bool_map is None:
                    any_g |= pilot6.expand_recovery_series(orig_hist, n_steps, gp)
                else:
                    any_g |= _expand_dec_bool(grasp_bool_map[gp])
            runs = pilot6._sustained_runs(any_g, min_grasp_steps)
            for idx, (g_start, g_end) in enumerate(runs):
                if g_end > n_steps:
                    continue
                next_g = runs[idx + 1][0] if idx + 1 < len(runs) else None
                search_end = next_g if next_g is not None else n_steps + 1
                _accumulate_release_window(g_end, search_end, [target_pred])

    # Hold conditions
    if hold_policy == "placement_only":
        holds_to_check: tuple[str, ...] = ()
    elif hold_policy == "verified_targets_only":
        holds_to_check = tuple(tp for tp in ordered_targets if tp in spec["hold_conditions"])
    elif hold_policy == "all_holds":
        holds_to_check = tuple(spec["hold_conditions"])
    else:
        raise ValueError(f"Unknown hold_policy: {hold_policy}")

    for cond in holds_to_check:
        series = target_bool_step[cond]
        # For verified_targets_only, do NOT start pre-latched at step 0; require visual confirmation first.
        latched = bool(series[0]) if hold_policy == "all_holds" else False
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


def run_scoring_upgrades(
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
        raise RuntimeError("No overlapping rollout_ids")

    fold_of = pfloor._episode_folds(episodes, num_folds=num_folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks_sorted = sorted({e["task"] for e in episodes})

    # Fit fast grasp detectors (OOF task-specific + LOTO zero-shot universal) + OOF state dest probs
    oof_grasp_bool, loto_any_grasp_bool, oof_state_dest_prob = vhyb._fit_grasp_and_dit_oof(
        b_arrays,
        episodes,
        fold_of,
        grasp_stream=grasp_stream,
        hidden_dim=grasp_hidden_dim,
        epochs=grasp_epochs,
        num_folds=num_folds,
        device=device,
    )

    # Also fit LOTO universal any_placement_satisfied from grasp_stream (for leak-free LOTO blend!)
    X_all = pfloor.build_stream_matrix(b_arrays, episodes, grasp_stream)
    loto_state_any_place_prob: dict[str, np.ndarray] = {}
    univ_dest_y = {}
    for ep in episodes:
        ptargets = [tp for _, _, tp in pilot6.TASK_SPECS[ep["task"]]["placements"]]
        d_any = np.zeros(ep["end_row"] - ep["start_row"], dtype=bool)
        for tp in ptargets:
            d_any |= np.asarray(ep["predicates_at_decisions"][tp], dtype=bool)
        univ_dest_y[ep["rollout_id"]] = d_any.astype(np.float32)
    for held_task in tasks_sorted:
        tr_eps = [e for e in episodes if e["task"] != held_task]
        ev_eps = [e for e in episodes if e["task"] == held_task]
        X_tr = np.concatenate([X_all[e["start_row"] : e["end_row"]] for e in tr_eps], axis=0)
        y_tr = np.concatenate([univ_dest_y[e["rollout_id"]] for e in tr_eps])
        X_ev = np.concatenate([X_all[e["start_row"] : e["end_row"]] for e in ev_eps], axis=0)
        _, p_ev = pfloor.fit_predict_probe(
            X_tr, y_tr, X_ev, device=device, hidden_dim=grasp_hidden_dim, epochs=grasp_epochs, seed=0
        )
        cur = 0
        for e in ev_eps:
            ln = e["end_row"] - e["start_row"]
            loto_state_any_place_prob[e["rollout_id"]] = p_ev[cur : cur + ln]
            cur += ln

    # 1. Precompute raw, delta_t0, and delta_pregrasp logits per rollout & predicate,
    #    plus diagnostic stats (all-frame AUROC, post-release AUROC, neg/pos logit means!).
    raw_ld: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    dt0_ld: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    dpre_ld: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    for ep in episodes:
        rid = ep["rollout_id"]
        g_any = loto_any_grasp_bool[rid]
        for pname in v_by_rid[rid]["predicates"]:
            ld = v_arrays[f"{rid}__{pname}__zoh_logit_diff"].astype(np.float32)
            raw_ld[rid][pname] = ld
            dt0_ld[rid][pname] = causal_delta_t0(ld)
            dpre_ld[rid][pname] = causal_delta_pregrasp(ld, g_any)

    # Fit LOTO Platt calibration (2 scalars a, b) on raw and delta_pregrasp strictly on other 5 placement targets
    loto_platt_raw: dict[str, tuple[float, float]] = {}
    loto_platt_dpre: dict[str, tuple[float, float]] = {}
    loto_rec90_raw: dict[str, float] = {}
    loto_rec90_dpre: dict[str, float] = {}

    for held_task in tasks_sorted:
        tr_y, tr_raw, tr_dpre = [], [], []
        for ep in episodes:
            if ep["task"] == held_task:
                continue
            rid = ep["rollout_id"]
            ptargets = {tp for _, _, tp in pilot6.TASK_SPECS[ep["task"]]["placements"]}
            for pname in ptargets:
                tr_y.append(np.asarray(ep["predicates_at_decisions"][pname], dtype=int))
                tr_raw.append(raw_ld[rid][pname])
                tr_dpre.append(dpre_ld[rid][pname])
        yc = np.concatenate(tr_y)
        rc = np.concatenate(tr_raw)
        dc = np.concatenate(tr_dpre)
        loto_platt_raw[held_task] = _fit_1d_platt(rc, yc)
        loto_platt_dpre[held_task] = _fit_1d_platt(dc, yc)
        loto_rec90_raw[held_task] = _best_recall_safe_threshold(yc, rc, min_recall=0.90)
        loto_rec90_dpre[held_task] = _best_recall_safe_threshold(yc, dc, min_recall=0.90)

    # Diagnostic per-predicate table (raw vs delta_t0 vs delta_pregrasp + post-release AUROC!)
    pred_diag: dict[str, dict] = {}
    raw_aurocs, dt0_aurocs, dpre_aurocs, post_rel_raw_aurocs, post_rel_dpre_aurocs = [], [], [], [], []
    raw_f1s, dt0_f1s, dpre_f1s = [], [], []

    for task in tasks_sorted:
        task_eps = [e for e in episodes if e["task"] == task]
        spec = pilot6.TASK_SPECS[task]
        ptargets = [tp for _, _, tp in spec["placements"]]
        p_out = {}
        a_d, b_d = loto_platt_dpre[task]
        for tp in ptargets:
            y_all_l, r_l, dt0_l, dpre_l = [], [], [], []
            pr_y_l, pr_r_l, pr_dpre_l = [], [], []
            for ep in task_eps:
                rid = ep["rollout_id"]
                y = np.asarray(ep["predicates_at_decisions"][tp], dtype=int)
                r = raw_ld[rid][tp]
                d0 = dt0_ld[rid][tp]
                dp = dpre_ld[rid][tp]
                y_all_l.append(y)
                r_l.append(r)
                dt0_l.append(d0)
                dpre_l.append(dp)
                # Post-release mask: all decisions strictly at/after the first true release of this operation
                g_preds = [gps for _, gps, tname in spec["placements"] if tname == tp][0]
                g_arr = np.zeros(len(y), dtype=bool)
                for gp in g_preds:
                    g_arr |= np.asarray(ep["predicates_at_decisions"][gp], dtype=bool)
                g_on = np.flatnonzero(g_arr)
                if g_on.size:
                    # First decision after first grasp where grasp becomes False
                    after_g = np.flatnonzero(~g_arr[g_on[0] :])
                    if after_g.size:
                        rel_idx = int(g_on[0] + after_g[0])
                        pr_y_l.append(y[rel_idx:])
                        pr_r_l.append(r[rel_idx:])
                        pr_dpre_l.append(dp[rel_idx:])
            yc = np.concatenate(y_all_l)
            rc = np.concatenate(r_l)
            d0c = np.concatenate(dt0_l)
            dpc = np.concatenate(dpre_l)
            a_raw = pfloor._auroc(yc, rc)
            a_dt0 = pfloor._auroc(yc, d0c)
            a_dpre = pfloor._auroc(yc, dpc)
            f_raw = vhyb._binary_f1(yc, rc >= 0.0)["f1"]
            f_dt0 = vhyb._binary_f1(yc, d0c >= 1.0)["f1"]
            f_dpre = vhyb._binary_f1(yc, _sigmoid(a_d * dpc + b_d) >= 0.5)["f1"]
            if pr_y_l:
                pr_yc = np.concatenate(pr_y_l)
                pr_rc = np.concatenate(pr_r_l)
                pr_dpc = np.concatenate(pr_dpre_l)
                pr_auc_r = pfloor._auroc(pr_yc, pr_rc)
                pr_auc_d = pfloor._auroc(pr_yc, pr_dpc)
            else:
                pr_auc_r = pr_auc_d = None

            neg_m = yc == 0
            pos_m = yc == 1
            p_out[tp] = {
                "raw_auroc": round(a_raw, 4) if a_raw is not None else None,
                "delta_t0_auroc": round(a_dt0, 4) if a_dt0 is not None else None,
                "delta_pregrasp_auroc": round(a_dpre, 4) if a_dpre is not None else None,
                "post_release_raw_auroc": round(pr_auc_r, 4) if pr_auc_r is not None else None,
                "post_release_dpre_auroc": round(pr_auc_d, 4) if pr_auc_d is not None else None,
                "raw_zs_f1": f_raw,
                "delta_t0_gt1_f1": f_dt0,
                "dpre_loto_platt_f1": f_dpre,
                "mean_neg_logit": round(float(rc[neg_m].mean()), 2) if neg_m.any() else None,
                "mean_pos_logit": round(float(rc[pos_m].mean()), 2) if pos_m.any() else None,
                "mean_neg_dpre": round(float(dpc[neg_m].mean()), 2) if neg_m.any() else None,
                "mean_pos_dpre": round(float(dpc[pos_m].mean()), 2) if pos_m.any() else None,
            }
            if a_raw is not None:
                raw_aurocs.append(a_raw)
            if a_dt0 is not None:
                dt0_aurocs.append(a_dt0)
            if a_dpre is not None:
                dpre_aurocs.append(a_dpre)
            if pr_auc_r is not None:
                post_rel_raw_aurocs.append(pr_auc_r)
            if pr_auc_d is not None:
                post_rel_dpre_aurocs.append(pr_auc_d)
            raw_f1s.append(f_raw)
            dt0_f1s.append(f_dt0)
            dpre_f1s.append(f_dpre)
        pred_diag[task] = p_out

    # 2. Construct probability & boolean tracks for each VLM scoring variant
    prob_tracks: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    bool_tracks: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))

    for ep in episodes:
        rid = ep["rollout_id"]
        task = ep["task"]
        a_r, b_r = loto_platt_raw[task]
        a_d, b_d = loto_platt_dpre[task]
        th_r90 = loto_rec90_raw[task]
        th_d90 = loto_rec90_dpre[task]
        for pname in v_by_rid[rid]["predicates"]:
            r = raw_ld[rid][pname]
            d0 = dt0_ld[rid][pname]
            dp = dpre_ld[rid][pname]

            p_raw = _sigmoid(r)
            p_raw_platt = _sigmoid(a_r * r + b_r)
            p_dpre_platt = _sigmoid(a_d * dp + b_d)
            # Zero-shot delta_pregrasp probability centered at +1.0 logit margin (zero training!)
            p_dpre_zs = _sigmoid(dp - 1.0)
            # Blend of raw VLM + OOF state probe (within-task)
            p_st = np.clip(oof_state_dest_prob[rid][pname], 1e-4, 1.0 - 1e-4)
            st_logit = np.log(p_st / (1.0 - p_st)).astype(np.float32)
            p_blend_oof = _sigmoid(0.5 * (a_r * r + b_r) + 0.5 * st_logit)
            # Zero-shot LOTO blend: LOTO-Platt VLM + LOTO universal state placement prob
            p_lst = np.clip(loto_state_any_place_prob[rid], 1e-4, 1.0 - 1e-4)
            lst_logit = np.log(p_lst / (1.0 - p_lst)).astype(np.float32)
            p_blend_loto = _sigmoid(0.5 * (a_d * dp + b_d) + 0.5 * lst_logit)

            prob_tracks["vlm_raw"][rid][pname] = p_raw
            bool_tracks["vlm_raw"][rid][pname] = causal_ema(p_raw) >= 0.5

            prob_tracks["vlm_raw_rec90"][rid][pname] = p_raw_platt
            bool_tracks["vlm_raw_rec90"][rid][pname] = causal_ema(_sigmoid(r - th_r90)) >= 0.5

            prob_tracks["vlm_raw_platt"][rid][pname] = p_raw_platt
            bool_tracks["vlm_raw_platt"][rid][pname] = causal_ema(p_raw_platt) >= 0.5

            prob_tracks["vlm_dpre_zs"][rid][pname] = p_dpre_zs
            bool_tracks["vlm_dpre_zs"][rid][pname] = causal_ema(p_dpre_zs) >= 0.5

            prob_tracks["vlm_dpre_platt"][rid][pname] = p_dpre_platt
            bool_tracks["vlm_dpre_platt"][rid][pname] = causal_ema(p_dpre_platt) >= 0.5

            prob_tracks["vlm_dpre_rec90"][rid][pname] = p_dpre_platt
            bool_tracks["vlm_dpre_rec90"][rid][pname] = causal_ema(_sigmoid(dp - th_d90)) >= 0.5

            prob_tracks["vlm_plus_state_oof"][rid][pname] = p_blend_oof
            bool_tracks["vlm_plus_state_oof"][rid][pname] = causal_ema(p_blend_oof) >= 0.5

            prob_tracks["vlm_plus_state_loto"][rid][pname] = p_blend_loto
            bool_tracks["vlm_plus_state_loto"][rid][pname] = causal_ema(p_blend_loto) >= 0.5

    # 3. Evaluate comprehensive ablation ladder at 5% nominal FPR (exact 5-fold split-conformal)
    def _eval_trim(ep_objs: list[pilot6.EpisodeData]) -> dict:
        ev = pilot6.evaluate_cross_fitted(ep_objs, num_folds=num_folds)
        ev.pop("oof_alarms", None)
        ev.pop("folds", None)
        return ev

    arms_out: dict[str, dict] = {}

    # 3a. Oracles (full vs placement_only ceiling)
    oracle_full = [
        pfloor._build_substituted_episode(ep, {}, replace_grasps=False, replace_targets=False, use_inferred_stage=False)
        for ep in episodes
    ]
    arms_out["oracle_full_all_holds"] = _eval_trim(oracle_full)

    # Exact true targets/grasps but placement_only (no loss_age)
    oracle_po = []
    for ep in episodes:
        t_bool = {
            p: np.asarray(ep["predicates_at_decisions"][p], dtype=bool)
            for p in v_by_rid[ep["rollout_id"]]["predicates"]
        }
        t_prob = {p: arr.astype(np.float32) for p, arr in t_bool.items()}
        oracle_po.append(
            build_soft_monitor_episode(
                ep,
                grasp_bool_map=None,
                any_grasp_dec=None,
                target_prob_map=t_prob,
                target_bool_map=t_bool,
                mode="hard_binary",
                hold_policy="placement_only",
                use_inferred_stage=False,
                use_universal_grasp=False,
            )
        )
    arms_out["oracle_placement_only"] = _eval_trim(oracle_po)

    # 3b. Component substitution: TRUE grasp + upgraded VLM destination (isolates destination scoring!)
    tg_configs = [
        ("tg_vlm_raw_hard_allholds", "vlm_raw", "hard_binary", "all_holds", 0.70, 1.0),
        ("tg_vlm_raw_hard_placeonly", "vlm_raw", "hard_binary", "placement_only", 0.70, 1.0),
        ("tg_vlm_raw_hard_verholds", "vlm_raw", "hard_binary", "verified_targets_only", 0.70, 1.0),
        ("tg_vlm_raw_rec90_placeonly", "vlm_raw_rec90", "hard_binary", "placement_only", 0.70, 1.0),
        ("tg_vlm_dpre_zs_placeonly", "vlm_dpre_zs", "hard_binary", "placement_only", 0.70, 1.0),
        ("tg_vlm_dpre_platt_placeonly", "vlm_dpre_platt", "hard_binary", "placement_only", 0.70, 1.0),
        ("tg_vlm_raw_soft_c070", "vlm_raw", "soft_evidence", "placement_only", 0.70, 1.0),
        ("tg_vlm_raw_soft_verholds", "vlm_raw", "soft_evidence", "verified_targets_only", 0.70, 1.0),
        ("tg_vlm_raw_platt_soft_c070", "vlm_raw_platt", "soft_evidence", "placement_only", 0.70, 1.0),
        ("tg_vlm_dpre_zs_soft_c070", "vlm_dpre_zs", "soft_evidence", "placement_only", 0.70, 1.0),
        ("tg_vlm_dpre_platt_soft_c070", "vlm_dpre_platt", "soft_evidence", "placement_only", 0.70, 1.0),
        ("tg_vlm_dpre_platt_soft_c055_g2", "vlm_dpre_platt", "soft_evidence", "placement_only", 0.55, 2.0),
        ("tg_vlm_plus_state_oof_soft", "vlm_plus_state_oof", "soft_evidence", "placement_only", 0.70, 1.0),
    ]
    for name, track, mode, hpol, cth, gam in tg_configs:
        eps_c = [
            build_soft_monitor_episode(
                ep,
                grasp_bool_map=None,
                any_grasp_dec=None,
                target_prob_map=prob_tracks[track][ep["rollout_id"]],
                target_bool_map=bool_tracks[track][ep["rollout_id"]],
                mode=mode,
                hold_policy=hpol,
                use_inferred_stage=False,
                use_universal_grasp=False,
                clear_threshold=cth,
                gamma=gam,
            )
            for ep in episodes
        ]
        arms_out[name] = _eval_trim(eps_c)

    # 3c. FULLY OBSERVED (ZERO PRIVILEGED INPUTS): OOF grasp + VLM dest + inferred stage
    hyb_configs = [
        ("hyb_raw_hard_allholds", "vlm_raw", "hard_binary", "all_holds", False, 0.70, 1.0),
        ("hyb_raw_hard_placeonly", "vlm_raw", "hard_binary", "placement_only", False, 0.70, 1.0),
        ("hyb_raw_hard_verholds", "vlm_raw", "hard_binary", "verified_targets_only", False, 0.70, 1.0),
        ("hyb_raw_soft_c070", "vlm_raw", "soft_evidence", "placement_only", False, 0.70, 1.0),
        ("hyb_raw_soft_verholds", "vlm_raw", "soft_evidence", "verified_targets_only", False, 0.70, 1.0),
        ("hyb_raw_platt_soft_c070", "vlm_raw_platt", "soft_evidence", "placement_only", False, 0.70, 1.0),
        ("hyb_dpre_zs_soft_c070", "vlm_dpre_zs", "soft_evidence", "placement_only", False, 0.70, 1.0),
        ("hyb_dpre_platt_soft_c070", "vlm_dpre_platt", "soft_evidence", "placement_only", False, 0.70, 1.0),
        ("hyb_dpre_platt_soft_verholds", "vlm_dpre_platt", "soft_evidence", "verified_targets_only", False, 0.70, 1.0),
        ("hyb_dpre_platt_soft_c055_g2", "vlm_dpre_platt", "soft_evidence", "placement_only", False, 0.55, 2.0),
        ("hyb_vlm_plus_state_oof_soft", "vlm_plus_state_oof", "soft_evidence", "placement_only", False, 0.70, 1.0),
        # TRUE ZERO-SHOT CROSS-TASK UNIVERSAL (LOTO universal grasp + zero target-task dest labels!)
        ("loto_univ_raw_hard_placeonly", "vlm_raw", "hard_binary", "placement_only", True, 0.70, 1.0),
        ("loto_univ_raw_hard_verholds", "vlm_raw", "hard_binary", "verified_targets_only", True, 0.70, 1.0),
        ("loto_univ_raw_soft_c070", "vlm_raw", "soft_evidence", "placement_only", True, 0.70, 1.0),
        ("loto_univ_raw_soft_verholds", "vlm_raw", "soft_evidence", "verified_targets_only", True, 0.70, 1.0),
        ("loto_univ_raw_platt_soft_c070", "vlm_raw_platt", "soft_evidence", "placement_only", True, 0.70, 1.0),
        ("loto_univ_dpre_zs_soft_c070", "vlm_dpre_zs", "soft_evidence", "placement_only", True, 0.70, 1.0),
        ("loto_univ_dpre_platt_soft_c070", "vlm_dpre_platt", "soft_evidence", "placement_only", True, 0.70, 1.0),
        ("loto_univ_dpre_platt_soft_verholds", "vlm_dpre_platt", "soft_evidence", "verified_targets_only", True, 0.70, 1.0),
        ("loto_univ_dpre_platt_soft_c055_g2", "vlm_dpre_platt", "soft_evidence", "placement_only", True, 0.55, 2.0),
        ("loto_univ_vlm_plus_state_loto_soft", "vlm_plus_state_loto", "soft_evidence", "placement_only", True, 0.70, 1.0),
    ]
    for name, track, mode, hpol, univ_g, cth, gam in hyb_configs:
        eps_c = [
            build_soft_monitor_episode(
                ep,
                grasp_bool_map=None if univ_g else oof_grasp_bool[ep["rollout_id"]],
                any_grasp_dec=loto_any_grasp_bool[ep["rollout_id"]] if univ_g else None,
                target_prob_map=prob_tracks[track][ep["rollout_id"]],
                target_bool_map=bool_tracks[track][ep["rollout_id"]],
                mode=mode,
                hold_policy=hpol,
                use_inferred_stage=True,
                use_universal_grasp=univ_g,
                clear_threshold=cth,
                gamma=gam,
            )
            for ep in episodes
        ]
        arms_out[name] = _eval_trim(eps_c)

    summary = {
        "schema_version": 1,
        "bundle_dir": str(bundle_dir),
        "vlm_dir": str(vlm_dir),
        "grasp_stream": grasp_stream,
        "loto_platt_raw": {k: [round(v[0], 4), round(v[1], 4)] for k, v in loto_platt_raw.items()},
        "loto_platt_dpre": {k: [round(v[0], 4), round(v[1], 4)] for k, v in loto_platt_dpre.items()},
        "placement_diagnostics": {
            "macro_raw_auroc": round(float(np.mean(raw_aurocs)), 4),
            "macro_delta_t0_auroc": round(float(np.mean(dt0_aurocs)), 4),
            "macro_delta_pregrasp_auroc": round(float(np.mean(dpre_aurocs)), 4),
            "macro_post_release_raw_auroc": round(float(np.mean(post_rel_raw_aurocs)), 4) if post_rel_raw_aurocs else None,
            "macro_post_release_dpre_auroc": round(float(np.mean(post_rel_dpre_aurocs)), 4) if post_rel_dpre_aurocs else None,
            "macro_raw_zs_f1": round(float(np.mean(raw_f1s)), 4),
            "macro_delta_t0_gt1_f1": round(float(np.mean(dt0_f1s)), 4),
            "macro_dpre_loto_platt_f1": round(float(np.mean(dpre_f1s)), 4),
            "by_task": pred_diag,
        },
        "arms_5pct_fpr": arms_out,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "out"
        staged.mkdir()
        (staged / "vlm_scoring_upgrades_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        staged.rename(output_dir)

    pd = summary["placement_diagnostics"]
    print(f"=== PLACEMENT LOGIT DIAGNOSTICS (grasp_stream={grasp_stream}) ===")
    print(
        f"  Macro 9 targets: raw_AUROC={pd['macro_raw_auroc']}  dt0_AUROC={pd['macro_delta_t0_auroc']}  "
        f"dpre_AUROC={pd['macro_delta_pregrasp_auroc']} | post_rel_raw={pd['macro_post_release_raw_auroc']}  "
        f"post_rel_dpre={pd['macro_post_release_dpre_auroc']} | "
        f"raw_F1={pd['macro_raw_zs_f1']:.3f}  dt0_F1={pd['macro_delta_t0_gt1_f1']:.3f}  dpre_F1={pd['macro_dpre_loto_platt_f1']:.3f}"
    )
    for task, pmap in sorted(pred_diag.items()):
        for tp, m in pmap.items():
            print(
                f"  {task:20s} {tp:28s} raw={m['raw_auroc']:.3f} dpre={m['delta_pregrasp_auroc']:.3f} "
                f"postR={str(m['post_release_raw_auroc']):>6s} postD={str(m['post_release_dpre_auroc']):>6s} | "
                f"neg/pos={m['mean_neg_logit']:+5.2f}/{m['mean_pos_logit']:+5.2f} "
                f"dpre={m['mean_neg_dpre']:+5.2f}/{m['mean_pos_dpre']:+5.2f} | "
                f"F1 raw={m['raw_zs_f1']:.3f} dpre={m['dpre_loto_platt_f1']:.3f}"
            )
    print(f"\n=== SCORING UPGRADE ABLATION LADDER (grasp_stream={grasp_stream}, 5% NOMINAL FPR) ===")
    for name, ev in arms_out.items():
        fp = ev["arms"]["full_physical"]
        sc = ev["arms"]["stage_clock"]
        vs_sc = ev["paired"]["full_physical_minus_stage_clock"]
        vs_g = ev["paired"]["full_physical_minus_global_timer"]
        ci = vs_g["ci95_frac"]
        print(
            f"  {name:35s} stg={sc['macro_lead_frac']*100:5.2f}% | "
            f"full={fp['macro_lead_frac']*100:5.2f}% ({fp['macro_lead_sec']:4.1f}s) "
            f"FPR={fp['realized_fpr']*100:4.1f}% rec={fp['failure_recall']*100:4.1f}% | "
            f"vs_stg={vs_sc['mean_gain_frac']*100:+5.2f}pp | "
            f"vs_G={vs_g['mean_gain_frac']*100:+5.2f}pp ({vs_g['mean_gain_sec']:+5.1f}s) "
            f"[{ci[0]*100:+5.1f},{ci[1]*100:+5.1f}] pos={vs_g['positive_tasks']}/{vs_g['total_failure_tasks']}"
        )
    print("Wrote:", output_dir / "vlm_scoring_upgrades_summary.json")
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
    run_scoring_upgrades(
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
