"""Unit tests for the 6-task XR-1 recovery pilot analyzer."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.xr1_lead_time import analyze_recovery_pilot6 as pilot6


def _make_trace(task, num_steps, observations, stages=None, replan=16):
    decisions = list(range(0, num_steps, replan))
    if stages is None:
        stages = ["stage_0"] * len(decisions)
    inf_records = [
        {"environment_step": s, "inference_index": i, "subtask_id": stg}
        for i, (s, stg) in enumerate(zip(decisions, stages))
    ]
    return {
        "task_name": task,
        "num_environment_states": num_steps + 1,
        "inference_records": inf_records,
        "recovery_observations": observations,
        "segments": [],
        "transitions": [],
    }


def _make_row(task, rid, seed, failed, horizon, num_steps):
    return {
        "task_name": task,
        "rollout_id": rid,
        "environment_seed": seed,
        "failed": failed,
        "rollout_horizon": horizon,
        "num_environment_steps": num_steps,
    }


class Pilot6AnalyzerTests(unittest.TestCase):
    def test_ice_cube_miss_and_self_correction_is_detected_and_paired(self):
        # Success episode: lemon placed cleanly, ice_cube1 grasped 100..130 then dropped,
        # ice_cube2 grasped 300..360 then placed at 362 -> multi-decision self-correction!
        base = {
            "fridge_open": True, "lemon_grasped": False, "ice_cube1_grasped": False,
            "ice_cube2_grasped": False, "lemon_in_glass": True, "ice_in_glass": False,
            "gripper_released": False,
        }
        succ_obs = [
            {"environment_step": 0, "values": dict(base)},
            {"environment_step": 100, "values": dict(base, ice_cube1_grasped=True)},
            {"environment_step": 130, "values": dict(base)},
            {"environment_step": 300, "values": dict(base, ice_cube2_grasped=True)},
            {"environment_step": 360, "values": dict(base)},
            {"environment_step": 362, "values": dict(base, ice_in_glass=True)},
            {"environment_step": 400, "values": dict(base, ice_in_glass=True, gripper_released=True)},
        ]
        ep_succ = pilot6.extract_episode(
            _make_row("MakeIceLemonade", "succ_ice", 1, False, 1000, 400),
            _make_trace("MakeIceLemonade", 400, succ_obs),
        )
        # Failure episode: ice_cube1 grasped 120..150 then dropped and never placed.
        fail_obs = [
            {"environment_step": 0, "values": dict(base)},
            {"environment_step": 120, "values": dict(base, ice_cube1_grasped=True)},
            {"environment_step": 150, "values": dict(base)},
            {"environment_step": 1000, "values": dict(base)},
        ]
        ep_fail = pilot6.extract_episode(
            _make_row("MakeIceLemonade", "fail_ice", 2, True, 1000, 1000),
            _make_trace("MakeIceLemonade", 1000, fail_obs),
        )
        t2 = pilot6.build_test2_pairs([ep_succ, ep_fail])
        self.assertEqual(t2["multi_decision_self_corrected_successes"], 1)
        self.assertEqual(t2["candidate_persistent_failures"], 1)
        self.assertEqual(t2["matched_test2_pairs"], 1)
        pair = t2["pairs"][0]
        self.assertEqual((pair["task"], pair["operation"], pair["relation"]), ("MakeIceLemonade", "place_ice", "ice_in_glass"))
        self.assertGreaterEqual(pair["matched_window_decisions"], 10)

    def test_release_only_clock_ignores_destination_arrival_while_missed_effect_stops(self):
        base = {"drawer_open": True, "straw_grasped": False, "straw_in_glass": False, "gripper_released": False}
        obs = [
            {"environment_step": 0, "values": dict(base)},
            {"environment_step": 16, "values": dict(base, straw_grasped=True)},
            {"environment_step": 48, "values": dict(base)},
            {"environment_step": 52, "values": dict(base, straw_in_glass=True)},
            {"environment_step": 160, "values": dict(base, straw_in_glass=True, gripper_released=True)},
        ]
        ep = pilot6.extract_episode(
            _make_row("DeliverStraw", "clean_straw", 3, False, 500, 160),
            _make_trace("DeliverStraw", 160, obs),
        )
        # At decision step 64 (after straw_in_glass became True at 52):
        idx_64 = int(np.where(ep.decision_steps == 64)[0][0])
        self.assertEqual(ep.release_only_age[idx_64], 16.0)
        self.assertEqual(ep.missed_effect_age[idx_64], -np.inf)
        self.assertEqual(ep.placement_delays, [4])

    def test_short_release_touch_below_min_grasp_steps_is_ignored(self):
        base = {"drawer_open": True, "straw_grasped": False, "straw_in_glass": False, "gripper_released": False}
        obs = [
            {"environment_step": 0, "values": dict(base)},
            {"environment_step": 20, "values": dict(base, straw_grasped=True)},
            {"environment_step": 80, "values": dict(base, straw_in_glass=True)},
            {"environment_step": 82, "values": dict(base, straw_grasped=True, straw_in_glass=True)},
            {"environment_step": 85, "values": dict(base, straw_in_glass=True)},
            {"environment_step": 120, "values": dict(base, straw_in_glass=True, gripper_released=True)},
        ]
        ep = pilot6.extract_episode(
            _make_row("DeliverStraw", "touch_straw", 4, False, 500, 120),
            _make_trace("DeliverStraw", 120, obs),
            min_grasp_steps=16,
        )
        self.assertEqual(len(ep.attempts), 1)
        self.assertEqual(ep.attempts[0]["disposition"], "target_at_release")

    def test_wrong_tool_placed_on_board_triggers_initial_invariant_loss(self):
        base = {
            "correct_tool_grasped": False, "correct_tool_present_on_board": False,
            "wrong_tool_grasped": False, "wrong_tool_not_on_cutting_board": True,
            "correct_tool_on_cutting_board": False,
        }
        obs = [
            {"environment_step": 0, "values": dict(base)},
            {"environment_step": 40, "values": dict(base, wrong_tool_grasped=True)},
            {"environment_step": 80, "values": dict(base, wrong_tool_not_on_cutting_board=False)},
            {"environment_step": 400, "values": dict(base, wrong_tool_not_on_cutting_board=False)},
        ]
        ep = pilot6.extract_episode(
            _make_row("CuttingToolSelection", "wrong_tool", 5, True, 400, 400),
            _make_trace("CuttingToolSelection", 400, obs),
        )
        self.assertEqual(len(ep.wrong_grasp_events), 1)
        self.assertEqual(len(ep.losses), 1)
        self.assertEqual(ep.losses[0]["relation"], "wrong_tool_not_on_cutting_board")
        self.assertEqual(ep.losses[0]["onset_step"], 80)
        self.assertEqual(ep.losses[0]["confirmed_step"], 96)

    def test_strict_causality_future_recovery_does_not_change_prefix_scores(self):
        base = {
            "fridge_open": True, "strawberry_grasped": False, "pancake_on_plate": True,
            "plate_on_table": True, "strawberry_on_pancake": False, "gripper_released": False,
        }
        obs_fail = [
            {"environment_step": 0, "values": dict(base)},
            {"environment_step": 32, "values": dict(base, strawberry_grasped=True)},
            {"environment_step": 64, "values": dict(base)},
            {"environment_step": 300, "values": dict(base)},
        ]
        obs_recover = [
            {"environment_step": 0, "values": dict(base)},
            {"environment_step": 32, "values": dict(base, strawberry_grasped=True)},
            {"environment_step": 64, "values": dict(base)},
            {"environment_step": 180, "values": dict(base, strawberry_on_pancake=True)},
            {"environment_step": 200, "values": dict(base, strawberry_on_pancake=True, gripper_released=True)},
        ]
        ep_a = pilot6.extract_episode(
            _make_row("GarnishPancake", "a", 1, True, 300, 300),
            _make_trace("GarnishPancake", 300, obs_fail),
        )
        ep_b = pilot6.extract_episode(
            _make_row("GarnishPancake", "b", 2, False, 300, 200),
            _make_trace("GarnishPancake", 200, obs_recover),
        )
        # Up through step 176 (before recovery at 180), online ages must be identical.
        mask_a = ep_a.decision_steps <= 176
        mask_b = ep_b.decision_steps <= 176
        np.testing.assert_array_equal(ep_a.missed_effect_age[mask_a], ep_b.missed_effect_age[mask_b])
        np.testing.assert_array_equal(ep_a.release_only_age[mask_a], ep_b.release_only_age[mask_b])
        np.testing.assert_array_equal(ep_a.loss_age[mask_a], ep_b.loss_age[mask_b])

    def test_stage_durations_parsed_from_subtask_safe_segment_schema(self):
        base = {"drawer_open": True, "straw_grasped": False, "straw_in_glass": True, "gripper_released": True}
        obs = [{"environment_step": 0, "values": dict(base)}, {"environment_step": 200, "values": dict(base)}]
        tr = _make_trace("DeliverStraw", 200, obs, stages=["OpenDrawer_1"] * 5 + ["PickPlaceDrawerToGlass_2_place"] * 8)
        tr["segments"] = [
            {"subtask_id": "OpenDrawer_1", "entry_environment_step": 0, "end_environment_step": 72, "completed": True},
            {"subtask_id": "PickPlaceDrawerToGlass_2_place", "entry_environment_step": 72, "end_environment_step": 200, "completed": True},
        ]
        ep = pilot6.extract_episode(_make_row("DeliverStraw", "seg1", 1, False, 500, 200), tr)
        self.assertEqual(ep.stage_entries["PickPlaceDrawerToGlass_2_place"], 72)
        self.assertEqual(ep.stage_durations["OpenDrawer_1"], 72)
        self.assertEqual(ep.stage_durations["PickPlaceDrawerToGlass_2_place"], 128)

    def test_end_to_end_run_analysis_writes_outputs_and_rejects_duplicates(self):
        base = {"drawer_open": True, "straw_grasped": False, "straw_in_glass": False, "gripper_released": False}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "shard0"
            (root / "subtasks/DeliverStraw").mkdir(parents=True)
            rows = []
            for i in range(6):
                rid = f"r{i}"
                failed = i >= 4
                n = 200 if not failed else 400
                obs = [
                    {"environment_step": 0, "values": dict(base)},
                    {"environment_step": 20, "values": dict(base, straw_grasped=True)},
                    {"environment_step": 60, "values": dict(base)},
                ]
                if not failed:
                    obs.append({"environment_step": 62, "values": dict(base, straw_in_glass=True)})
                    obs.append({"environment_step": n, "values": dict(base, straw_in_glass=True, gripper_released=True)})
                else:
                    obs.append({"environment_step": n, "values": dict(base)})
                tpath = f"subtasks/DeliverStraw/{rid}.json"
                (root / tpath).write_text(json.dumps(_make_trace("DeliverStraw", n, obs)))
                row = _make_row("DeliverStraw", rid, 100 + i, failed, 400, n)
                row["subtask_trace_path"] = tpath
                rows.append(row)
            (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            out = Path(tmp) / "analysis"
            summary = pilot6.run_analysis([Path(tmp)], out, num_folds=2)
            self.assertEqual(summary["cohort"]["total_rollouts"], 6)
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "rollouts.csv").exists())
            self.assertTrue((out / "events.jsonl").exists())
            with self.assertRaises(FileExistsError):
                pilot6.run_analysis([Path(tmp)], out, num_folds=2)


if __name__ == "__main__":
    unittest.main()
