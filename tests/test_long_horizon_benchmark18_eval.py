"""Unit tests for the 18-task Test-1, LLM-task-graph, and equal-duration Test-2 evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.xr1_lead_time import analyze_recovery_pilot6 as pilot6
from scripts.xr1_lead_time import eval_test1_benchmark18 as t1b
from scripts.xr1_lead_time import eval_test2_benchmark18 as t2b
from scripts.xr1_lead_time import extract_vlm_destinations as evlm
from scripts.xr1_lead_time import generate_llm_task_graphs as llm_g


def _synthetic_ep_dict(
    rid: str,
    task: str,
    *,
    success: bool,
    n_dec: int = 20,
    stride_steps: int = 16,
    horizon: int = 400,
    grasp_range: tuple[int, int] = (3, 7),
    place_on_dec: int | None = 8,
    pred_name: str = "llm_goal_0",
) -> tuple[dict, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    dec_steps = (np.arange(n_dec, dtype=int) * stride_steps).tolist()
    n_steps = dec_steps[-1] + stride_steps
    any_g = np.zeros(n_dec, dtype=bool)
    any_g[grasp_range[0] : grasp_range[1]] = True
    p_arr = np.full(n_dec, 0.1, dtype=np.float32)
    b_arr = np.zeros(n_dec, dtype=bool)
    if place_on_dec is not None:
        p_arr[place_on_dec:] = 0.92
        b_arr[place_on_dec:] = True
    ep = {
        "rollout_id": rid,
        "task": task,
        "seed": 0,
        "success": success,
        "horizon": horizon,
        "num_steps": n_steps,
        "decision_steps": dec_steps,
        "decision_stages": ["s0"] * n_dec,
        "segments": [],
        "recovery_observations": [],
    }
    return ep, any_g, {pred_name: p_arr}, {pred_name: b_arr}


class Benchmark18EvalTests(unittest.TestCase):
    def test_waffle_reheat_spec_matches_reset_geometry(self):
        spec = pilot6.TASK_SPECS["WaffleReheat"]
        self.assertEqual(len(spec["placements"]), 1)
        op, grasps, target = spec["placements"][0]
        self.assertEqual(op, "place_bowl")
        self.assertEqual(grasps, ("bowl_grasped",))
        self.assertEqual(target, "bowl_in_microwave")
        self.assertEqual(spec["hold_conditions"], ("waffle_in_bowl", "bowl_in_microwave"))
        for p in ("waffle_in_bowl", "bowl_in_microwave"):
            self.assertIn(p, evlm.PREDICATE_PROMPTS["WaffleReheat"])

    def test_parse_llm_task_graph_response_handles_fences_dupes_cap_and_fallback(self):
        fenced = """```json
        [
          {"id": "Pan In Sink", "question": "Is the pan resting in the sink"},
          {"id": "pan_in_sink", "question": "Is the second pan resting in the sink?"},
          {"id": "sponge_in_sink", "question": "Is the sponge resting in the sink?"},
          {"id": "extra_3", "question": "Is object three placed at its destination?"},
          {"id": "extra_4", "question": "Is object four placed at its destination?"}
        ]
        ```"""
        parsed = llm_g.parse_llm_task_graph_response(fenced, "Place the pan and sponge in the sink.")
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed[0]["id"], "pan_in_sink")
        self.assertTrue(parsed[0]["question"].endswith("?"))
        self.assertEqual(len({x["id"] for x in parsed}), 4)

        fb = llm_g.parse_llm_task_graph_response("not valid json at all", "Put the mug on the tray.")
        self.assertEqual(len(fb), 1)
        self.assertEqual(fb[0]["id"], "llm_goal_0")
        self.assertIn("Put the mug on the tray.", fb[0]["question"])

    def test_arbitrary_target_monitor_supports_llm_ids_and_is_strictly_causal(self):
        ep, any_g, p_map, b_map = _synthetic_ep_dict(
            "r_miss", "BreadSelection", success=False, place_on_dec=None, pred_name="llm_goal_croissant"
        )
        ep_ok, any_g_ok, p_map_ok, b_map_ok = _synthetic_ep_dict(
            "r_ok", "BreadSelection", success=True, place_on_dec=7, pred_name="llm_goal_croissant"
        )

        # Clean placement clears missed_effect_age immediately on release (step 7*16=112)
        mon_ok = t1b.build_arbitrary_target_monitor_episode(
            ep_ok,
            any_grasp_dec=any_g_ok,
            target_prob_map=p_map_ok,
            target_bool_map=b_map_ok,
            target_keys=["llm_goal_croissant"],
            stage_mode="grasp_and_vlm",
            phys_mode="hard_binary",
            hold_policy="verified_targets_only",
        )
        self.assertTrue(np.all(~np.isfinite(mon_ok.missed_effect_age)))
        self.assertIn("gv_p0_g0", mon_ok.decision_stages)
        self.assertIn("gv_p0_g1", mon_ok.decision_stages)
        self.assertIn("gv_p1_g0", mon_ok.decision_stages)

        # Missed placement accumulates missed_effect_age strictly after release
        mon_miss = t1b.build_arbitrary_target_monitor_episode(
            ep,
            any_grasp_dec=any_g,
            target_prob_map=p_map,
            target_bool_map=b_map,
            target_keys=["llm_goal_croissant"],
            stage_mode="grasp_and_vlm",
            phys_mode="hard_binary",
            hold_policy="verified_targets_only",
        )
        self.assertTrue(np.all(~np.isfinite(mon_miss.missed_effect_age[:7])))
        self.assertGreater(mon_miss.missed_effect_age[-1], 100.0)

        # Strict causality: flipping future target probability at decision >= 12 must not change prefix [0..11]
        p_mut = {"llm_goal_croissant": p_map["llm_goal_croissant"].copy()}
        b_mut = {"llm_goal_croissant": b_map["llm_goal_croissant"].copy()}
        p_mut["llm_goal_croissant"][12:] = 0.99
        b_mut["llm_goal_croissant"][12:] = True
        mon_mut = t1b.build_arbitrary_target_monitor_episode(
            ep,
            any_grasp_dec=any_g,
            target_prob_map=p_mut,
            target_bool_map=b_mut,
            target_keys=["llm_goal_croissant"],
            stage_mode="grasp_and_vlm",
            phys_mode="hard_binary",
            hold_policy="verified_targets_only",
        )
        np.testing.assert_allclose(mon_miss.missed_effect_age[:12], mon_mut.missed_effect_age[:12])
        self.assertEqual(mon_miss.decision_stages[:12], mon_mut.decision_stages[:12])

    def test_protocol_b_zero_shot_transfer_and_adaptive_union_bound(self):
        # Build 60 clean train successes across 2 train tasks + 20 test successes & 10 test failures on 2 unseen tasks
        train_eps = []
        for i in range(60):
            task = "LoadDishwasher" if i < 30 else "PreSoakPan"
            ep, any_g, pm, bm = _synthetic_ep_dict(
                f"tr_{i}", task, success=True, n_dec=16 + (i % 4), horizon=400, grasp_range=(3, 7), place_on_dec=7
            )
            train_eps.append(
                t1b.build_arbitrary_target_monitor_episode(
                    ep, any_grasp_dec=any_g, target_prob_map=pm, target_bool_map=bm, target_keys=["llm_goal_0"]
                )
            )
        test_eps = []
        for i in range(20):
            task = "BreadSelection" if i < 10 else "ArrangeTea"
            ep, any_g, pm, bm = _synthetic_ep_dict(
                f"te_s_{i}", task, success=True, n_dec=16 + (i % 3), horizon=400, grasp_range=(3, 7), place_on_dec=7
            )
            test_eps.append(
                t1b.build_arbitrary_target_monitor_episode(
                    ep, any_grasp_dec=any_g, target_prob_map=pm, target_bool_map=bm, target_keys=["llm_goal_0"]
                )
            )
        for i in range(10):
            task = "BreadSelection" if i < 5 else "ArrangeTea"
            ep, any_g, pm, bm = _synthetic_ep_dict(
                f"te_f_{i}", task, success=False, n_dec=25, horizon=400, grasp_range=(2, 5), place_on_dec=None
            )
            test_eps.append(
                t1b.build_arbitrary_target_monitor_episode(
                    ep, any_grasp_dec=any_g, target_prob_map=pm, target_bool_map=bm, target_keys=["llm_goal_0"]
                )
            )

        # Rejects task leakage
        with self.assertRaises(ValueError):
            t1b.evaluate_zero_shot_threshold_transfer(train_eps, train_eps[:5])

        pb = t1b.evaluate_zero_shot_threshold_transfer(train_eps, test_eps)
        self.assertGreaterEqual(pb["n_train_cal_successes"], 39)
        self.assertEqual(pb["arms"]["univ_stage_plus_phys"]["realized_fpr"], 0.0)
        self.assertGreater(
            pb["arms"]["univ_stage_plus_phys"]["macro_lead_frac"],
            pb["arms"]["global_timer"]["macro_lead_frac"],
        )

        # Adaptive union bound on >= 60 successes across folds is strictly non-vacuous
        u = t1b.evaluate_union_bound_split_conformal_adaptive(train_eps + test_eps[-10:], num_folds=5)
        self.assertTrue(u["non_vacuous"])
        self.assertGreater(u["union_monitor"]["macro_lead_frac"], u["global_timer_5pct"]["macro_lead_frac"])

    def test_strict_equal_duration_test2_excludes_short_and_is_causal(self):
        # 2 episodes: one short (only 2 decisions before restore), one long (5 decisions before restore)
        dec_steps = (np.arange(12, dtype=int) * 16).tolist()
        meta_by_rid = {
            "short_ep": {"rollout_id": "short_ep", "task": "T1", "horizon": 200, "start_row": 0, "end_row": 12, "decision_steps": dec_steps},
            "long_ep": {"rollout_id": "long_ep", "task": "T1", "horizon": 200, "start_row": 12, "end_row": 24, "decision_steps": dec_steps},
        }
        rng = np.random.default_rng(7)
        arrays = {
            "executed_actions": rng.normal(size=(24, 18)).astype(np.float32),
            "action_chunks": rng.normal(size=(24, 36)).astype(np.float32),
            "state_history": rng.normal(size=(24, 16)).astype(np.float32),
            "dit_last": rng.normal(size=(24, 32)).astype(np.float32),
            "siglip_3cam": rng.normal(size=(24, 3, 16)).astype(np.float32),
        }
        events = [
            {
                "rollout_id": "short_ep", "task": "T1", "success": True, "relation": "r",
                "onset_step": 16, "confirmed_step": 32, "next_regrasp_step": None,
                "restored_step": 64, "num_steps": 192,  # decisions in [32, 64) are {32, 48} -> only 2!
            },
            {
                "rollout_id": "long_ep", "task": "T1", "success": False, "relation": "r",
                "onset_step": 16, "confirmed_step": 32, "next_regrasp_step": None,
                "restored_step": None, "num_steps": 192,  # >= 3 decisions!
            },
        ]
        rows = t2b.extract_strict_equal_duration_cohort(events, meta_by_rid, arrays, None, exact_decisions=3)
        self.assertEqual([r["rollout_id"] for r in rows], ["long_ep"])
        self.assertEqual(rows[0]["features"]["ctrl_win_decisions"], 3.0)

        # Mutate observations strictly AFTER the 3rd window decision (indices >= 5 inside long_ep -> global >= 17)
        arrays_mut = {k: v.copy() for k, v in arrays.items()}
        for k in arrays_mut:
            arrays_mut[k][17:] = 999.0
        rows_mut = t2b.extract_strict_equal_duration_cohort(events, meta_by_rid, arrays_mut, None, exact_decisions=3)
        self.assertEqual(rows[0]["features"], rows_mut[0]["features"])

    def test_tasks_filter_in_load_episodes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "subtasks").mkdir()
            rows = []
            for rid, task in (("r1", "DeliverStraw"), ("r2", "WaffleReheat")):
                vals = (
                    {"straw_grasped": 0, "straw_in_glass": 0}
                    if task == "DeliverStraw"
                    else {"bowl_grasped": 0, "waffle_in_bowl": 1, "bowl_in_microwave": 0}
                )
                trace = {
                    "task_name": task,
                    "inference_records": [{"environment_step": 0, "inference_index": 0, "subtask_id": "s0"}],
                    "recovery_observations": [{"environment_step": 0, "values": vals}],
                    "segments": [],
                }
                tp = f"subtasks/{rid}.json"
                (root / tp).write_text(json.dumps(trace))
                rows.append({
                    "rollout_id": rid, "task_name": task, "environment_seed": 0,
                    "failed": False, "rollout_horizon": 200, "num_environment_steps": 16,
                    "subtask_trace_path": tp,
                })
            (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            eps, _ = pilot6.load_episodes([root], tasks_filter={"WaffleReheat"})
            self.assertEqual([e.rollout_id for e in eps], ["r2"])


if __name__ == "__main__":
    unittest.main()
