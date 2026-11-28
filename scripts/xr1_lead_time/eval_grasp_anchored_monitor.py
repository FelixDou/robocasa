"""Evaluate grasp-anchored inferred stage clocks and fully decoupled zero-shot
unseen-task monitors (LOTO universal grasp + unseen-task VLM destination
verification) with zero privileged inputs.

Diagnoses and fixes the inferred-stage-clock bottleneck discovered in Sections 9–10:
- Previously, ``inferred_stage_clock`` only latched on placement targets, so on
  single-object tasks (half of the 6 tasks) it stayed in ``stage_0`` for 99% of
  the episode and was completely blind to ``no_sustained_grasp`` failures (21.4%
  of all failures) while noisy VLM stage latches inflated the conformal cutoff.
- Here we build strictly causal, zero-privileged-input stage clocks anchored to
  our 0.966-AUROC LOTO universal grasp detector:
  1. ``grasp_stage``: ``pre_first_grasp`` -> ``post_first_grasp`` (and subsequent
     sustained-grasp epochs), cleanly catching approach/open/grasp stalls without
     any VLM stage noise.
  2. ``grasp_and_vlm_stage``: combines LOTO universal grasp onset with verified
     VLM placement completion (reconstructing Subtask-SAFE's pick-then-place
     stages from observable streams alone).
- Compares every combination on the exact 5-fold split-conformal 5% FPR protocol
  (both standard pooled calibration and union-bound channel-separated calibration
  ``alpha/2 + alpha/2 = 0.05`` so one noisy channel never drowns out another):
  - ``global_timer`` (G) and ``task_timer`` (P)
  - ``grasp_stage_only`` (ZERO VLM)
  - ``task_timer_plus_vlm_phys`` (NO stage clock; isolates pure VLM physical gain)
  - ``grasp_stage_plus_vlm_phys`` (clean LOTO grasp stage + unseen-task VLM miss/loss)
  - ``grasp_and_vlm_stage_plus_vlm_phys``
  - ``union_bound_grasp_stage_plus_vlm_phys`` (exact finite-sample <=5% FPR)
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
from . import eval_vlm_hybrid_monitor as vhyb
from . import eval_vlm_scoring_upgrades as vupg


def build_grasp_anchored_stages(
    ep: dict,
    any_grasp_dec: np.ndarray,
    target_bool_map: dict[str, np.ndarray] | None,
    *,
    include_vlm_stages: bool = False,
    min_grasp_decisions: int = 1,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Build strictly causal observable stage IDs, entries, and completed durations.

    - ``include_vlm_stages=False`` (``grasp_stage``):
      Transitions causally from ``g_stage_0`` (before first sustained grasp) to
      ``g_stage_1`` (after first sustained grasp onset). Zero VLM noise!
    - ``include_vlm_stages=True`` (``grasp_and_vlm_stage``):
      Tracks ``(num_confirmed_placements, has_grasped_in_current_placement_stage)``
      strictly causally, reconstructing Subtask-SAFE's pick-then-place stage
      progression on both single-object and multi-object tasks.
    """
    dec_steps = np.asarray(ep["decision_steps"], dtype=int)
    n_dec = len(dec_steps)
    n_steps = int(ep["num_steps"])
    ordered_targets = [tp for _, _, tp in pilot6.TASK_SPECS[ep["task"]]["placements"]]

    latched = {tp: False for tp in ordered_targets}
    consec_t = {tp: 0 for tp in ordered_targets}
    consec_g = 0
    grasped_in_phase = False
    prev_n_placed = 0

    stages: list[str] = []
    stage_entries: dict[str, int] = {}
    stage_durations: dict[str, int] = {}
    cur_stage: str | None = None
    cur_start = 0

    for i, t in enumerate(dec_steps):
        if include_vlm_stages and target_bool_map is not None:
            for tp in ordered_targets:
                if bool(target_bool_map[tp][i]):
                    consec_t[tp] += 1
                    if consec_t[tp] >= 2:
                        latched[tp] = True
                else:
                    consec_t[tp] = 0
        n_placed = sum(latched.values()) if include_vlm_stages else 0
        if n_placed > prev_n_placed:
            # New placement stage began; reset grasp-in-phase flag
            grasped_in_phase = False
            consec_g = 0
            prev_n_placed = n_placed

        if bool(any_grasp_dec[i]):
            consec_g += 1
            if consec_g >= min_grasp_decisions:
                grasped_in_phase = True
        else:
            consec_g = 0

        if include_vlm_stages:
            sid = f"gv_p{n_placed}_g{int(grasped_in_phase)}"
        else:
            sid = f"g_stage_{int(grasped_in_phase)}"

        stages.append(sid)
        if sid != cur_stage:
            if cur_stage is not None and ep["success"]:
                stage_durations[cur_stage] = max(stage_durations.get(cur_stage, 0), int(t) - cur_start)
            if sid not in stage_entries:
                stage_entries[sid] = int(t)
            cur_stage = sid
            cur_start = int(t)

    if cur_stage is not None and ep["success"]:
        stage_durations[cur_stage] = max(stage_durations.get(cur_stage, 0), n_steps - cur_start)

    return stages, stage_entries, stage_durations


def build_monitor_episode(
    ep: dict,
    *,
    any_grasp_dec: np.ndarray,
    grasp_bool_map: dict[str, np.ndarray] | None,
    target_prob_map: dict[str, np.ndarray],
    target_bool_map: dict[str, np.ndarray],
    stage_mode: str,
    phys_mode: str = "hard_binary",
    hold_policy: str = "verified_targets_only",
    use_universal_grasp: bool = True,
    clear_threshold: float = 0.70,
) -> pilot6.EpisodeData:
    """Build EpisodeData with chosen observable stage clock + VLM physical signal.

    ``stage_mode`` in ``{"none", "grasp_only", "grasp_and_vlm", "oracle_stage"}``.
    """
    base = vupg.build_soft_monitor_episode(
        ep,
        grasp_bool_map=None if use_universal_grasp else grasp_bool_map,
        any_grasp_dec=any_grasp_dec if use_universal_grasp else None,
        target_prob_map=target_prob_map,
        target_bool_map=target_bool_map,
        mode=phys_mode,
        hold_policy=hold_policy,
        use_inferred_stage=False,
        use_universal_grasp=use_universal_grasp,
        clear_threshold=clear_threshold,
    )
    if stage_mode == "oracle_stage":
        return base
    if stage_mode == "none":
        # Single constant stage so stage_clock collapses cleanly to task_timer (P)
        n_dec = len(base.decision_steps)
        base.decision_stages = ["single_stage"] * n_dec
        base.stage_entries = {"single_stage": 0}
        base.stage_durations = {}
        return base
    if stage_mode == "grasp_only":
        stgs, entries, durs = build_grasp_anchored_stages(ep, any_grasp_dec, None, include_vlm_stages=False)
    elif stage_mode == "grasp_and_vlm":
        stgs, entries, durs = build_grasp_anchored_stages(
            ep, any_grasp_dec, target_bool_map, include_vlm_stages=True
        )
    else:
        raise ValueError(f"Unknown stage_mode: {stage_mode}")

    base.decision_stages = stgs
    base.stage_entries = entries
    base.stage_durations = durs
    return base


def evaluate_union_bound_split_conformal(
    episodes: list[pilot6.EpisodeData],
    *,
    num_folds: int = 5,
    alpha: float = 0.05,
    alpha_stage_frac: float = 0.5,
    loss_ref: int = 16,
) -> dict:
    """Exact finite-sample split-conformal monitor separating the stage-clock
    channel (budget ``alpha * alpha_stage_frac``) from the VLM physical channel
    (budget ``alpha * (1 - alpha_stage_frac)``).

    By the union bound, firing when ``stage_clock > tau_stage OR phys > tau_phys``
    maintains nominal episode-level FPR <= ``alpha`` while ensuring noise in one
    channel NEVER inflates the cutoff of the other channel.
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
    oof_union: dict[str, int | None] = {}
    oof_stage: dict[str, int | None] = {}
    oof_global: dict[str, int | None] = {}
    oof_phys_only: dict[str, int | None] = {}

    for fold_idx, held in enumerate(folds):
        held_ids = {e.rollout_id for e in held}
        train_succ = pilot6._hash_order(
            [e for e in episodes if e.success and e.rollout_id not in held_ids],
            f"cal/{fold_idx}",
        )
        n_total = len(train_succ)
        n_cal = min(n_total - 3, max(19, math.ceil(n_total * 2 / 3)))
        cal_pool = train_succ[:n_cal]
        fit_pool = train_succ[n_cal:]
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

    successes = [e for e in episodes if e.success]
    failures = [e for e in episodes if not e.success]

    def _summarize(alarms: dict[str, int | None]) -> dict:
        fa = sum(alarms[e.rollout_id] is not None for e in successes)
        tp = sum(alarms[e.rollout_id] is not None for e in failures)
        by_t_f = defaultdict(list)
        by_t_s = defaultdict(list)
        for ep in failures:
            a = alarms[ep.rollout_id]
            by_t_f[ep.task].append((ep.horizon - a) / ep.horizon if a is not None else 0.0)
            by_t_s[ep.task].append((ep.horizon - a) / pilot6.CONTROL_HZ if a is not None else 0.0)
        return {
            "failures_warned": tp,
            "total_failures": len(failures),
            "failure_recall": round(tp / len(failures), 4) if failures else 0.0,
            "false_alarms": fa,
            "total_successes": len(successes),
            "realized_fpr": round(fa / len(successes), 4) if successes else 0.0,
            "macro_lead_frac": float(np.mean([np.mean(v) for v in by_t_f.values()])),
            "macro_lead_sec": float(np.mean([np.mean(v) for v in by_t_s.values()])),
        }

    return {
        "stage_clock_5pct": _summarize(oof_stage),
        "phys_only_5pct": _summarize(oof_phys_only),
        "union_monitor": _summarize(oof_union),
        "union_minus_global_timer": pilot6._paired_bootstrap_ci(failures, oof_union, oof_global),
        "union_minus_stage_clock": pilot6._paired_bootstrap_ci(failures, oof_union, oof_stage),
    }


def run_grasp_anchored_eval(
    bundle_dir: Path,
    vlm_dir: Path,
    output_dir: Path,
    *,
    label: str = "vlm",
    grasp_stream: str = "proprio_actions",
    grasp_hidden_dim: int = 128,
    grasp_epochs: int = 100,
    num_folds: int = 5,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")

    import torch
    from . import eval_perception_floor as pfloor

    b_index = json.loads((bundle_dir / "perception_index.json").read_text())
    b_arrays = dict(np.load(bundle_dir / "perception_features.npz"))
    v_index = json.loads((vlm_dir / "vlm_destination_index.json").read_text())
    v_arrays = dict(np.load(vlm_dir / "vlm_destination_scores.npz"))

    v_by_rid = {e["rollout_id"]: e for e in v_index["episodes"]}
    episodes = [e for e in b_index["episodes"] if e["rollout_id"] in v_by_rid]
    fold_of = pfloor._episode_folds(episodes, num_folds=num_folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks_sorted = sorted({e["task"] for e in episodes})

    oof_grasp_bool, loto_any_grasp_bool, _ = vhyb._fit_grasp_and_dit_oof(
        b_arrays,
        episodes,
        fold_of,
        grasp_stream=grasp_stream,
        hidden_dim=grasp_hidden_dim,
        epochs=grasp_epochs,
        num_folds=num_folds,
        device=device,
    )

    # Build raw and pre-grasp-debiased LOTO-Platt probability & boolean tracks
    raw_ld: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    dpre_ld: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for ep in episodes:
        rid = ep["rollout_id"]
        g_any = loto_any_grasp_bool[rid]
        for pname in v_by_rid[rid]["predicates"]:
            ld = v_arrays[f"{rid}__{pname}__zoh_logit_diff"].astype(np.float32)
            raw_ld[rid][pname] = ld
            dpre_ld[rid][pname] = vupg.causal_delta_pregrasp(ld, g_any)

    loto_platt_dpre = {}
    for held_task in tasks_sorted:
        tr_y, tr_d = [], []
        for ep in episodes:
            if ep["task"] == held_task:
                continue
            rid = ep["rollout_id"]
            for tp in {t for _, _, t in pilot6.TASK_SPECS[ep["task"]]["placements"]}:
                tr_y.append(np.asarray(ep["predicates_at_decisions"][tp], dtype=int))
                tr_d.append(dpre_ld[rid][tp])
        loto_platt_dpre[held_task] = vupg._fit_1d_platt(np.concatenate(tr_d), np.concatenate(tr_y))

    prob_tracks: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    bool_tracks: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for ep in episodes:
        rid = ep["rollout_id"]
        a_d, b_d = loto_platt_dpre[ep["task"]]
        for pname in v_by_rid[rid]["predicates"]:
            p_raw = vupg._sigmoid(raw_ld[rid][pname])
            p_dp = vupg._sigmoid(a_d * dpre_ld[rid][pname] + b_d)
            prob_tracks["raw"][rid][pname] = p_raw
            bool_tracks["raw"][rid][pname] = vupg.causal_ema(p_raw) >= 0.5
            prob_tracks["dpre_platt"][rid][pname] = p_dp
            bool_tracks["dpre_platt"][rid][pname] = vupg.causal_ema(p_dp) >= 0.5

    def _eval_trim(ep_objs):
        ev = pilot6.evaluate_cross_fitted(ep_objs, num_folds=num_folds)
        ev.pop("oof_alarms", None)
        ev.pop("folds", None)
        return ev

    # Oracle reference
    oracle_eps = [
        pfloor._build_substituted_episode(ep, {}, replace_grasps=False, replace_targets=False, use_inferred_stage=False)
        for ep in episodes
    ]
    results_pooled: dict[str, dict] = {"oracle_reference": _eval_trim(oracle_eps)}
    results_union: dict[str, dict] = {}

    configs = [
        # (name, stage_mode, track, phys_mode, hold_policy, universal_grasp)
        ("univ_not_vlm_grasp_stage_only", "grasp_only", "raw", "hard_binary", "placement_only", True),
        ("univ_no_stage_vlm_raw_verholds", "none", "raw", "hard_binary", "verified_targets_only", True),
        ("univ_no_stage_vlm_raw_placeonly", "none", "raw", "hard_binary", "placement_only", True),
        ("univ_grasp_stage_vlm_raw_placeonly", "grasp_only", "raw", "hard_binary", "placement_only", True),
        ("univ_grasp_stage_vlm_raw_verholds", "grasp_only", "raw", "hard_binary", "verified_targets_only", True),
        ("univ_grasp_stage_vlm_raw_soft_verholds", "grasp_only", "raw", "soft_evidence", "verified_targets_only", True),
        ("univ_grasp_stage_vlm_dpre_soft_verholds", "grasp_only", "dpre_platt", "soft_evidence", "verified_targets_only", True),
        ("univ_gv_stage_vlm_raw_hard_verholds", "grasp_and_vlm", "raw", "hard_binary", "verified_targets_only", True),
        ("univ_gv_stage_vlm_raw_soft_verholds", "grasp_and_vlm", "raw", "soft_evidence", "verified_targets_only", True),
        ("univ_gv_stage_vlm_dpre_soft_verholds", "grasp_and_vlm", "dpre_platt", "soft_evidence", "verified_targets_only", True),
        ("oofg_grasp_stage_vlm_raw_verholds", "grasp_only", "raw", "hard_binary", "verified_targets_only", False),
        ("oofg_gv_stage_vlm_raw_hard_verholds", "grasp_and_vlm", "raw", "hard_binary", "verified_targets_only", False),
        ("oofg_gv_stage_vlm_dpre_soft_verholds", "grasp_and_vlm", "dpre_platt", "soft_evidence", "verified_targets_only", False),
    ]

    for name, stg_m, track, phys_m, hpol, univ_g in configs:
        eps_c = [
            build_monitor_episode(
                ep,
                any_grasp_dec=loto_any_grasp_bool[ep["rollout_id"]],
                grasp_bool_map=oof_grasp_bool[ep["rollout_id"]],
                target_prob_map=prob_tracks[track][ep["rollout_id"]],
                target_bool_map=bool_tracks[track][ep["rollout_id"]],
                stage_mode=stg_m,
                phys_mode=phys_m,
                hold_policy=hpol,
                use_universal_grasp=univ_g,
            )
            for ep in episodes
        ]
        if name == "univ_not_vlm_grasp_stage_only":
            # Evaluate stage_clock only (zero out physical channels)
            for e in eps_c:
                e.missed_effect_age[:] = -np.inf
                e.loss_age[:] = -np.inf
        results_pooled[name] = _eval_trim(eps_c)
        if stg_m in ("grasp_only", "grasp_and_vlm") and name != "univ_not_vlm_grasp_stage_only":
            results_union[name] = evaluate_union_bound_split_conformal(eps_c, num_folds=num_folds)

    summary = {
        "schema_version": 1,
        "label": label,
        "grasp_stream": grasp_stream,
        "bundle_dir": str(bundle_dir),
        "vlm_dir": str(vlm_dir),
        "pooled_5pct_fpr": results_pooled,
        "union_bound_5pct_fpr": results_union,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "out"
        staged.mkdir()
        (staged / "grasp_anchored_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        staged.rename(output_dir)

    print(f"=== GRASP-ANCHORED ZERO-PRIVILEGED MONITOR ({label} | grasp={grasp_stream}) ===")
    for name, ev in results_pooled.items():
        fp = ev["arms"]["full_physical"]
        sc = ev["arms"]["stage_clock"]
        vs_sc = ev["paired"]["full_physical_minus_stage_clock"]
        vs_g = ev["paired"]["full_physical_minus_global_timer"]
        ci = vs_g["ci95_frac"]
        print(
            f"  {name:40s} stg={sc['macro_lead_frac']*100:5.2f}% ({sc['realized_fpr']*100:3.1f}%) | "
            f"full={fp['macro_lead_frac']*100:5.2f}% ({fp['macro_lead_sec']:4.1f}s) "
            f"FPR={fp['realized_fpr']*100:4.1f}% rec={fp['failure_recall']*100:4.1f}% | "
            f"vs_stg={vs_sc['mean_gain_frac']*100:+5.2f}pp | "
            f"vs_G={vs_g['mean_gain_frac']*100:+5.2f}pp ({vs_g['mean_gain_sec']:+5.1f}s) "
            f"[{ci[0]*100:+5.1f},{ci[1]*100:+5.1f}] pos={vs_g['positive_tasks']}/{vs_g['total_failure_tasks']}"
        )
    print(f"\n=== UNION-BOUND CHANNEL-SEPARATED CONFORMAL (alpha/2 + alpha/2 = 5%, {label}) ===")
    for name, u in results_union.items():
        um = u["union_monitor"]
        po = u["phys_only_5pct"]
        vs_g = u["union_minus_global_timer"]
        vs_s = u["union_minus_stage_clock"]
        ci = vs_g["ci95_frac"]
        print(
            f"  {name:40s} phys5%={po['macro_lead_frac']*100:5.2f}% ({po['realized_fpr']*100:3.1f}%) | "
            f"union={um['macro_lead_frac']*100:5.2f}% ({um['macro_lead_sec']:4.1f}s) "
            f"FPR={um['realized_fpr']*100:4.1f}% rec={um['failure_recall']*100:4.1f}% | "
            f"vs_stg={vs_s['mean_gain_frac']*100:+5.2f}pp | "
            f"vs_G={vs_g['mean_gain_frac']*100:+5.2f}pp ({vs_g['mean_gain_sec']:+5.1f}s) "
            f"[{ci[0]*100:+5.1f},{ci[1]*100:+5.1f}] pos={vs_g['positive_tasks']}/{vs_g['total_failure_tasks']}"
        )
    print("Wrote:", output_dir / "grasp_anchored_summary.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--vlm-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="vlm")
    parser.add_argument("--grasp-stream", default="proprio_actions")
    parser.add_argument("--grasp-hidden-dim", type=int, default=128)
    parser.add_argument("--grasp-epochs", type=int, default=100)
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args(argv)
    run_grasp_anchored_eval(
        args.bundle,
        args.vlm_scores,
        args.output,
        label=args.label,
        grasp_stream=args.grasp_stream,
        grasp_hidden_dim=args.grasp_hidden_dim,
        grasp_epochs=args.grasp_epochs,
        num_folds=args.num_folds,
    )


if __name__ == "__main__":
    main()
