"""Unit tests for zero-shot Qwen3-VL destination scoring, visibility-gated
causal belief filtering, object-agnostic universal grasp+destination tracking,
and end-to-end hybrid monitor evaluation.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None


class FakeScorer:
    """Deterministic fake VLM scorer driven by RGB mean of wrist view."""

    def score_batch(self, items, batch_size=16):
        ly, ln, lu = [], [], []
        for (_l, _r, wrist_im), question in items:
            arr = np.asarray(wrist_im, dtype=np.float32)
            # Red channel > 150 encodes positive predicate, Blue > 150 encodes Unknown (out of view)
            r_mean = float(arr[:, :, 0].mean())
            b_mean = float(arr[:, :, 2].mean())
            if b_mean > 150.0:
                ly.append(-2.0)
                ln.append(-2.0)
                lu.append(4.0)
            elif r_mean > 150.0:
                ly.append(3.0)
                ln.append(-3.0)
                lu.append(-4.0)
            else:
                ly.append(-3.0)
                ln.append(3.0)
                lu.append(-4.0)
        return (
            np.asarray(ly, dtype=np.float32),
            np.asarray(ln, dtype=np.float32),
            np.asarray(lu, dtype=np.float32),
        )


class VLMVerifierTests(unittest.TestCase):
    @unittest.skipIf(cv2 is None, "cv2 required")
    def test_all_pilot6_destination_and_hold_predicates_have_prompts(self):
        from scripts.xr1_lead_time import analyze_recovery_pilot6 as pilot6
        from scripts.xr1_lead_time.extract_vlm_destinations import (
            PREDICATE_PROMPTS,
            format_predicate_question,
            task_destination_predicates,
        )

        for task in pilot6.TASK_SPECS:
            preds = task_destination_predicates(task)
            self.assertTrue(preds)
            for p in preds:
                self.assertIn(p, PREDICATE_PROMPTS[task])
                q, inv = format_predicate_question(task, p, "Place appropriate tool for avocado skin.")
                self.assertIn("Yes, No, or Unknown", q)
                if p == "wrong_tool_not_on_cutting_board":
                    self.assertTrue(inv)
                    self.assertIn("avocado", q)
                else:
                    self.assertFalse(inv)

    @unittest.skipIf(cv2 is None, "cv2 required")
    def test_causal_zoh_never_leaks_future_vlm_scores(self):
        from scripts.xr1_lead_time.extract_vlm_destinations import (
            expand_causal_zoh,
            sampled_decision_positions,
        )

        pos = sampled_decision_positions(7, decision_stride=3)
        self.assertEqual(pos.tolist(), [0, 3, 6])
        base = expand_causal_zoh(7, pos, np.array([-2.0, 1.0, 4.0]))
        mut = expand_causal_zoh(7, pos, np.array([-2.0, 99.0, 4.0]))
        # Decisions 0, 1, 2 must only see pos=0 (-2.0), never pos=3!
        np.testing.assert_allclose(base[:3], [-2.0, -2.0, -2.0])
        np.testing.assert_allclose(mut[:3], [-2.0, -2.0, -2.0])
        np.testing.assert_allclose(base[3:6], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(float(base[6]), 4.0)

    @unittest.skipIf(cv2 is None, "cv2 required")
    def test_ternary_scores_inversion_and_visibility_gate(self):
        from scripts.xr1_lead_time.eval_vlm_hybrid_monitor import visibility_gated_belief
        from scripts.xr1_lead_time.extract_vlm_destinations import compute_ternary_scores

        sc = compute_ternary_scores(
            np.array([3.0, -3.0, -2.0]),
            np.array([-3.0, 3.0, -2.0]),
            np.array([-4.0, -4.0, 5.0]),
            invert=True,
        )
        # Inverted: first frame had high "yes" for wrong tool on board -> low prob for "not on board"
        self.assertLess(float(sc["prob_yes"][0]), 0.01)
        self.assertGreater(float(sc["prob_yes"][1]), 0.99)
        self.assertLess(float(sc["prob_visible"][2]), 0.05)

        # Visibility-gated belief must hold previous True belief when camera turns away (vis=0.02)
        probs = np.array([0.05, 0.95, 0.95, 0.05, 0.05, 0.90], dtype=np.float32)
        vis = np.array([0.98, 0.98, 0.98, 0.02, 0.02, 0.98], dtype=np.float32)
        bel = visibility_gated_belief(probs, vis, init_prob=0.0, ema_alpha=0.7, min_vis=0.15)
        self.assertGreater(float(bel[2]), 0.85)
        # During out-of-view frames 3 and 4, belief stays latched instead of dropping to 0.05!
        self.assertAlmostEqual(float(bel[3]), float(bel[2]), places=5)
        self.assertAlmostEqual(float(bel[4]), float(bel[2]), places=5)

    def test_object_agnostic_universal_tracker_multi_object(self):
        from scripts.xr1_lead_time.eval_vlm_hybrid_monitor import build_object_agnostic_episode

        # Multi-object ArrangeBreadBasket:
        # Grasp 1 (bread, steps 16..48) -> bread_in_basket becomes True at step 48 -> CLEAN!
        # Grasp 2 (basket, steps 80..112) -> basket_on_dining_counter stays False -> MISSED PLACEMENT!
        dec_steps = list(range(0, 240, 16))  # 15 decisions
        d = len(dec_steps)
        any_grasp = np.zeros(d, dtype=bool)
        any_grasp[1:3] = True   # steps 16..48 -> release at 48
        any_grasp[5:7] = True   # steps 80..112 -> release at 112
        bread_in = np.zeros(d, dtype=bool)
        bread_in[3:] = True     # arrives right at release 48
        basket_on = np.zeros(d, dtype=bool)  # never arrives!

        ep = {
            "rollout_id": "abb_test",
            "task": "ArrangeBreadBasket",
            "seed": 1,
            "success": False,
            "horizon": 320,
            "num_steps": 240,
            "decision_steps": dec_steps,
            "decision_stages": ["s0"] * d,
            "segments": [],
        }
        ed = build_object_agnostic_episode(
            ep,
            any_grasp,
            {"bread_in_basket": bread_in, "basket_on_dining_counter": basket_on},
            use_inferred_stage=True,
        )
        # After release 1 (step 48), bread_in_basket arrived immediately -> missed_effect_age must be -inf at steps 48..96!
        self.assertTrue(np.all(np.isneginf(ed.missed_effect_age[:7])))
        # After release 2 (step 112), basket never arrived -> missed_effect_age ticks cleanly from step 112!
        self.assertEqual(float(ed.missed_effect_age[7]), 0.0)
        self.assertEqual(float(ed.missed_effect_age[9]), 32.0)
        self.assertEqual(ed.placement_delays, [0])

    @unittest.skipIf(cv2 is None, "cv2 required")
    def test_extract_vlm_scores_checkpoint_resume_and_npz(self):
        from scripts.xr1_lead_time.extract_vlm_destinations import extract_vlm_scores

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            (root / "subtasks").mkdir(parents=True)
            (root / "videos").mkdir(parents=True)

            # Write a tiny 10-frame 768x256 synthetic obs_grid video
            vid_rel = "videos/ep1.mp4"
            vw = cv2.VideoWriter(
                str(root / vid_rel),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (768, 256),
            )
            for f in range(10):
                frame = np.zeros((256, 768, 3), dtype=np.uint8)
                # Frames >= 5 turn wrist view red (BGR channel 2 = 255 -> RGB channel 0 = 255)
                if f >= 5:
                    frame[:, 512:768, 2] = 255
                vw.write(frame)
            vw.release()

            trace_rel = "subtasks/ep1.json"
            trace = {
                "task_name": "DeliverStraw",
                "inference_records": [{"environment_step": s, "subtask_id": "s0"} for s in (0, 16, 32, 48, 64)],
                "video_frame_environment_steps": [1, 8, 15, 24, 31, 40, 47, 56, 63, 64],
                "recovery_observations": [
                    {"environment_step": 0, "values": {"drawer_open": True, "straw_grasped": False, "straw_in_glass": False, "gripper_released": True}},
                    {"environment_step": 48, "values": {"drawer_open": True, "straw_grasped": False, "straw_in_glass": True, "gripper_released": True}},
                    {"environment_step": 64, "values": {"drawer_open": True, "straw_grasped": False, "straw_in_glass": True, "gripper_released": True}},
                ],
                "segments": [],
            }
            (root / trace_rel).write_text(json.dumps(trace))
            manifest_row = {
                "rollout_id": "ep1",
                "task_name": "DeliverStraw",
                "task_instruction": "Place straw in glass.",
                "environment_seed": 7,
                "failed": False,
                "rollout_horizon": 200,
                "num_environment_steps": 64,
                "video_path": vid_rel,
                "subtask_trace_path": trace_rel,
            }
            (root / "manifest.jsonl").write_text(json.dumps(manifest_row) + "\n")

            out = Path(tmp) / "vlm_out"
            idx1 = extract_vlm_scores([root], out, Path("/unused"), decision_stride=2, scorer=FakeScorer())
            self.assertEqual(idx1["computed_rollouts"], 1)
            self.assertEqual(idx1["resumed_rollouts"], 0)
            arrs = np.load(out / "vlm_destination_scores.npz")
            zoh_p = arrs["ep1__straw_in_glass__zoh_prob_yes"]
            self.assertEqual(len(zoh_p), 5)
            self.assertLess(float(zoh_p[0]), 0.1)
            self.assertGreater(float(zoh_p[4]), 0.9)

            # Second call resumes cleanly with 0 recomputed rollouts!
            idx2 = extract_vlm_scores([root], out, Path("/unused"), decision_stride=2, scorer=FakeScorer())
            self.assertEqual(idx2["computed_rollouts"], 0)
            self.assertEqual(idx2["resumed_rollouts"], 1)

    @unittest.skipIf(torch is None or cv2 is None, "torch and cv2 required")
    def test_end_to_end_hybrid_evaluator_on_synthetic_data(self):
        from scripts.xr1_lead_time.eval_vlm_hybrid_monitor import run_hybrid_evaluation

        rng = np.random.default_rng(11)
        episodes = []
        vlm_eps = []
        vlm_arrs = {}
        all_dit, all_state, all_chunk, all_exec = [], [], [], []
        offset = 0
        for task in ("DeliverStraw", "GarnishPancake"):
            for i in range(6):
                succ = i < 4
                n_steps = 160 if succ else 320
                dec_steps = list(range(0, n_steps, 16))
                d = len(dec_steps)
                rid = f"{task}_{i}"
                g_key = "straw_grasped" if task == "DeliverStraw" else "strawberry_grasped"
                t_key = "straw_in_glass" if task == "DeliverStraw" else "strawberry_on_pancake"
                g_bool = [2 <= k <= 4 for k in range(d)]
                t_bool = [k >= 5 for k in range(d)] if succ else [False] * d

                dit = rng.normal(0, 0.2, size=(d, 8)).astype(np.float16)
                for k in range(d):
                    if g_bool[k]:
                        dit[k, 0] += 3.0
                    if t_bool[k]:
                        dit[k, 1] += 2.0
                all_dit.append(dit)
                all_state.append(rng.normal(0, 0.2, size=(d, 8)).astype(np.float32))
                all_chunk.append(rng.normal(0, 0.2, size=(d, 6)).astype(np.float32))
                all_exec.append(rng.normal(0, 0.2, size=(d, 18)).astype(np.float32))

                if task == "DeliverStraw":
                    pnames = ["drawer_open", "gripper_released", "straw_grasped", "straw_in_glass"]
                    pat = {"drawer_open": [True] * d, "gripper_released": [not g for g in g_bool], "straw_grasped": g_bool, "straw_in_glass": t_bool}
                    vlm_preds = ["straw_in_glass"]
                else:
                    pnames = ["fridge_open", "gripper_released", "pancake_on_plate", "plate_on_table", "strawberry_grasped", "strawberry_on_pancake"]
                    pat = {"fridge_open": [True] * d, "gripper_released": [not g for g in g_bool], "pancake_on_plate": [True] * d, "plate_on_table": [True] * d, "strawberry_grasped": g_bool, "strawberry_on_pancake": t_bool}
                    vlm_preds = ["strawberry_on_pancake", "pancake_on_plate", "plate_on_table"]

                hist = [{"environment_step": int(t), "values": {p: bool(pat[p][k]) for p in pnames}} for k, t in enumerate(dec_steps)]
                hist.append({"environment_step": n_steps, "values": dict(hist[-1]["values"])})
                episodes.append(
                    {
                        "rollout_id": rid, "task": task, "seed": i, "success": succ,
                        "horizon": 320, "num_steps": n_steps, "replan_steps": 16,
                        "start_row": offset, "end_row": offset + d,
                        "decision_steps": dec_steps, "decision_stages": ["s0"] * d,
                        "predicates_at_decisions": pat, "segments": [], "recovery_observations": hist,
                    }
                )
                offset += d

                s_pos = list(range(0, d, 2))
                if s_pos[-1] != d - 1:
                    s_pos.append(d - 1)
                vlm_eps.append({"rollout_id": rid, "task": task, "sampled_positions": s_pos, "predicates": vlm_preds})
                for vp in vlm_preds:
                    y_arr = np.asarray(pat[vp], dtype=np.float32)
                    ld = np.where(y_arr > 0.5, 3.0, -3.0).astype(np.float32)
                    py = np.where(y_arr > 0.5, 0.95, 0.05).astype(np.float32)
                    pv = np.full(d, 0.95, dtype=np.float32)
                    vlm_arrs[f"{rid}__{vp}__sampled_logit_diff"] = ld[s_pos]
                    vlm_arrs[f"{rid}__{vp}__sampled_prob_yes"] = py[s_pos]
                    vlm_arrs[f"{rid}__{vp}__sampled_prob_visible"] = pv[s_pos]
                    vlm_arrs[f"{rid}__{vp}__zoh_logit_diff"] = ld
                    vlm_arrs[f"{rid}__{vp}__zoh_prob_yes"] = py
                    vlm_arrs[f"{rid}__{vp}__zoh_prob_visible"] = pv

        with tempfile.TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "bundle"
            vdir = Path(tmp) / "vlm"
            bdir.mkdir()
            vdir.mkdir()
            np.savez(
                bdir / "perception_features.npz",
                dit_last=np.concatenate(all_dit, axis=0),
                dit_mean=np.concatenate(all_dit, axis=0),
                state_history=np.concatenate(all_state, axis=0),
                action_chunks=np.concatenate(all_chunk, axis=0),
                executed_actions=np.concatenate(all_exec, axis=0),
            )
            (bdir / "perception_index.json").write_text(json.dumps({"episodes": episodes}))
            np.savez(vdir / "vlm_destination_scores.npz", **vlm_arrs)
            (vdir / "vlm_destination_index.json").write_text(json.dumps({"decision_stride": 2, "model_dir": "fake", "episodes": vlm_eps}))

            out = Path(tmp) / "hybrid_out"
            res = run_hybrid_evaluation(bdir, vdir, out, grasp_stream="xr1_dit", grasp_hidden_dim=0, grasp_epochs=20, num_folds=2)
            self.assertTrue((out / "vlm_hybrid_summary.json").exists())
            self.assertGreater(res["vlm_destination_metrics"]["macro_all_zeroshot_0p5_f1"], 0.95)
            self.assertIn("zero_shot_loto_universal_vlm_zeroshot", res["monitor_lead_5pct_fpr"])


if __name__ == "__main__":
    unittest.main()
