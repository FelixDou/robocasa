"""Unit tests for strictly causal perception feature extraction and floor probing."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None


class PerceptionFloorPureTests(unittest.TestCase):
    @unittest.skipIf(cv2 is None, "cv2 required")
    def test_causal_frame_indices_never_leak_future_frames(self):
        from scripts.xr1_lead_time.extract_perception_features import causal_frame_indices

        # Stride-2 post-action video frames: steps 1, 3, 5, ..., 31, 32
        vsteps = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 32]
        decs = np.array([0, 16, 32])
        idxs = causal_frame_indices(decs, vsteps)
        # t=0 -> index 0 (reset proxy step 1); t=16 -> step 15 (idx 7, <= 16); t=32 -> step 32 (idx 16)
        self.assertEqual(idxs.tolist(), [0, 7, 16])
        for t, i in zip(decs[1:], idxs[1:]):
            self.assertLessEqual(vsteps[i], t)

    @unittest.skipIf(cv2 is None, "cv2 required")
    def test_executed_action_summary_is_strictly_causal(self):
        from scripts.xr1_lead_time.extract_perception_features import summarize_executed_actions

        T = 48
        base_actions = {
            "action_gripper_close": np.zeros((T, 1), dtype=np.float32),
            "action_end_effector_position": np.zeros((T, 3), dtype=np.float32),
            "action_end_effector_rotation": np.zeros((T, 3), dtype=np.float32),
            "action_base_motion": np.zeros((T, 4), dtype=np.float32),
        }
        mod_actions = {k: v.copy() for k, v in base_actions.items()}
        # Mutate ONLY steps >= 16 (must NOT affect summary at t=0 or t=16!)
        mod_actions["action_gripper_close"][16:] = 1.0
        mod_actions["action_end_effector_position"][16:] = 5.0
        decs = np.array([0, 16, 32])
        s0 = summarize_executed_actions(base_actions, decs, replan_steps=16)
        s1 = summarize_executed_actions(mod_actions, decs, replan_steps=16)
        np.testing.assert_array_equal(s0[:2], s1[:2])
        self.assertGreater(s1[2, 0], 0.9)

    @unittest.skipIf(cv2 is None, "cv2 required")
    def test_obs_grid_split_and_preprocess_siglip(self):
        from scripts.xr1_lead_time.extract_perception_features import preprocess_siglip_rgb, split_obs_grid_3cam

        grid = np.zeros((256, 768, 3), dtype=np.uint8)
        grid[:, 0:256, 2] = 255    # left = red in BGR -> R=255
        grid[:, 256:512, 1] = 255  # right = green
        grid[:, 512:768, 0] = 255  # wrist = blue
        l, r, w = split_obs_grid_3cam(grid)
        self.assertEqual((l.shape, r.shape, w.shape), ((256, 256, 3), (256, 256, 3), (256, 256, 3)))
        lp = preprocess_siglip_rgb(l, 224)
        self.assertEqual(lp.shape, (3, 224, 224))
        self.assertAlmostEqual(float(lp[0].mean()), 1.0, places=4)
        self.assertAlmostEqual(float(lp[1].mean()), -1.0, places=4)

    @unittest.skipIf(torch is None or cv2 is None, "torch and cv2 required")
    def test_end_to_end_probe_and_substitution_on_synthetic_bundle(self):
        from scripts.xr1_lead_time.eval_perception_floor import run_evaluation

        rng = np.random.default_rng(7)
        episodes = []
        all_vis, all_dit, all_state, all_chunk, all_exec = [], [], [], [], []
        offset = 0
        for task in ("DeliverStraw", "GarnishPancake"):
            for i in range(6):
                succ = i < 4
                n_steps = 160 if succ else 320
                dec_steps = list(range(0, n_steps, 16))
                d = len(dec_steps)
                g_key = "straw_grasped" if task == "DeliverStraw" else "strawberry_grasped"
                t_key = "straw_in_glass" if task == "DeliverStraw" else "strawberry_on_pancake"
                g_bool = [2 <= k <= 4 for k in range(d)]
                t_bool = [k >= 5 for k in range(d)] if succ else [False] * d

                # Plant separable signal in siglip wrist view + state
                vis = rng.normal(0, 0.2, size=(d, 3, 16)).astype(np.float16)
                for k in range(d):
                    if g_bool[k]:
                        vis[k, 2, 0] += 3.0
                    if t_bool[k]:
                        vis[k, 2, 1] += 3.0
                dit = rng.normal(0, 0.2, size=(d, 8)).astype(np.float16)
                state = rng.normal(0, 0.2, size=(d, 8)).astype(np.float32)
                chunk = rng.normal(0, 0.2, size=(d, 6)).astype(np.float32)
                ex = rng.normal(0, 0.2, size=(d, 18)).astype(np.float32)

                all_vis.append(vis)
                all_dit.append(dit)
                all_state.append(state)
                all_chunk.append(chunk)
                all_exec.append(ex)

                if task == "DeliverStraw":
                    pnames = ["drawer_open", "gripper_released", "straw_grasped", "straw_in_glass"]
                    pat = {
                        "drawer_open": [True] * d,
                        "gripper_released": [not g for g in g_bool],
                        "straw_grasped": g_bool,
                        "straw_in_glass": t_bool,
                    }
                else:
                    pnames = [
                        "fridge_open", "gripper_released", "pancake_on_plate",
                        "plate_on_table", "strawberry_grasped", "strawberry_on_pancake",
                    ]
                    pat = {
                        "fridge_open": [True] * d,
                        "gripper_released": [not g for g in g_bool],
                        "pancake_on_plate": [True] * d,
                        "plate_on_table": [True] * d,
                        "strawberry_grasped": g_bool,
                        "strawberry_on_pancake": t_bool,
                    }
                hist = [
                    {"environment_step": int(t), "values": {p: bool(pat[p][k]) for p in pnames}}
                    for k, t in enumerate(dec_steps)
                ]
                hist.append({"environment_step": n_steps, "values": dict(hist[-1]["values"])})

                episodes.append(
                    {
                        "rollout_id": f"{task}_{i}",
                        "task": task,
                        "seed": 100 + i,
                        "success": succ,
                        "horizon": 320,
                        "num_steps": n_steps,
                        "replan_steps": 16,
                        "start_row": offset,
                        "end_row": offset + d,
                        "decision_steps": dec_steps,
                        "decision_stages": ["stage_0"] * d,
                        "predicates_at_decisions": pat,
                        "segments": [],
                        "recovery_observations": hist,
                    }
                )
                offset += d

        with tempfile.TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "bundle"
            bdir.mkdir()
            np.savez(
                bdir / "perception_features.npz",
                siglip_3cam=np.concatenate(all_vis, axis=0),
                dit_last=np.concatenate(all_dit, axis=0),
                dit_mean=np.concatenate(all_dit, axis=0),
                state_history=np.concatenate(all_state, axis=0),
                action_chunks=np.concatenate(all_chunk, axis=0),
                executed_actions=np.concatenate(all_exec, axis=0),
            )
            (bdir / "perception_index.json").write_text(json.dumps({"episodes": episodes}))
            out = Path(tmp) / "eval_out"
            res = run_evaluation(
                bdir,
                out,
                streams=["proprio_actions", "siglip_wrist", "siglip_3cam_plus_state"],
                hidden_dim=0,
                epochs=40,
                num_folds=2,
                primary_stream="siglip_wrist",
            )
            self.assertTrue((out / "perception_floor_summary.json").exists())
            self.assertGreater(res["within_task_probes"]["siglip_wrist"]["macro_auroc"], 0.85)
            self.assertIn("fully_observed_camera_monitor", res["component_substitution"])


if __name__ == "__main__":
    unittest.main()
