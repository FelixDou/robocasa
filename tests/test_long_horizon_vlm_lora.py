"""Unit tests for 3-fold cross-task LoRA partitioning, balanced training sampling
(strictly excluding held-out tasks), and pure-PyTorch LoRA injection/reset.
"""

import unittest

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    import cv2
except ImportError:
    cv2 = None


class CrossTaskLoRATests(unittest.TestCase):
    @unittest.skipIf(cv2 is None, "cv2 required")
    def test_cross_task_folds_are_disjoint_and_cover_all_six_tasks(self):
        from scripts.xr1_lead_time import analyze_recovery_pilot6 as pilot6
        from scripts.xr1_lead_time.train_eval_vlm_lora_cross_task import (
            CROSS_TASK_FOLDS,
            validate_cross_task_folds,
        )

        mapping = validate_cross_task_folds(CROSS_TASK_FOLDS)
        self.assertEqual(set(mapping.keys()), set(pilot6.TASK_SPECS.keys()))
        self.assertEqual(len(CROSS_TASK_FOLDS), 3)
        for f in CROSS_TASK_FOLDS:
            self.assertEqual(len(f), 2)

    @unittest.skipIf(cv2 is None, "cv2 required")
    def test_balanced_sampler_never_includes_held_out_tasks_and_respects_invert(self):
        from scripts.xr1_lead_time.train_eval_vlm_lora_cross_task import (
            select_balanced_training_samples,
        )

        d = 12
        recs = [
            {
                "rollout_id": "cts_1",
                "task": "CuttingToolSelection",
                "task_instruction": "Place tool for avocado skin.",
                "num_decisions": d,
                "decision_steps": np.arange(d) * 16,
                "frame_indices": np.arange(d),
                "predicates_at_decisions": {
                    "correct_tool_grasped": [k in (2, 3) for k in range(d)],
                    "wrong_tool_grasped": [False] * d,
                    "correct_tool_present_on_board": [k >= 5 for k in range(d)],
                    "wrong_tool_not_on_cutting_board": [k != 4 for k in range(d)],
                },
            },
            {
                "rollout_id": "ds_held",
                "task": "DeliverStraw",
                "task_instruction": "Place straw in glass.",
                "num_decisions": d,
                "decision_steps": np.arange(d) * 16,
                "frame_indices": np.arange(d),
                "predicates_at_decisions": {
                    "straw_grasped": [k in (2, 3) for k in range(d)],
                    "straw_in_glass": [k >= 5 for k in range(d)],
                },
            },
        ]

        samples = select_balanced_training_samples(recs, train_tasks={"CuttingToolSelection"}, seed=0)
        self.assertTrue(samples)
        # Strictly zero samples from held-out DeliverStraw!
        self.assertTrue(all(s["task"] == "CuttingToolSelection" for s in samples))
        # Check inverted yes_label on wrong_tool_not_on_cutting_board:
        # when wrong_tool_not_on_cutting_board is False (pred_label==0), positive question "is WRONG tool on board?" has yes_label=1.0!
        wt_neg = [s for s in samples if s["predicate"] == "wrong_tool_not_on_cutting_board" and s["pred_label"] == 0]
        self.assertTrue(wt_neg)
        self.assertTrue(all(s["yes_label"] == 1.0 for s in wt_neg))
        wt_pos = [s for s in samples if s["predicate"] == "wrong_tool_not_on_cutting_board" and s["pred_label"] == 1]
        self.assertTrue(wt_pos)
        self.assertTrue(all(s["yes_label"] == 0.0 for s in wt_pos))

    @unittest.skipIf(torch is None or cv2 is None, "torch and cv2 required")
    def test_pure_pytorch_lora_injection_zero_init_grad_and_clean_reset(self):
        from scripts.xr1_lead_time.train_eval_vlm_lora_cross_task import inject_lora_adapters

        class TinyVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = nn.Module()
                self.visual.proj = nn.Linear(8, 8)
                self.q_proj = nn.Linear(8, 8)
                self.lm_head = nn.Linear(8, 4)

            def forward(self, x):
                return self.lm_head(self.q_proj(self.visual.proj(x)))

        torch.manual_seed(0)
        m = TinyVLM()
        x = torch.randn(3, 8)
        with torch.no_grad():
            y_before = m(x).clone()

        adapters = inject_lora_adapters(m, rank=4, alpha=8.0, dropout=0.0, include_vision=True)
        # Should wrap visual.proj and q_proj, and MUST NOT wrap lm_head!
        self.assertEqual(len(adapters), 2)
        self.assertFalse(m.lm_head.weight.requires_grad)

        # Zero-init B must preserve exact forward output before training!
        with torch.no_grad():
            y_init = m(x)
        torch.testing.assert_close(y_init, y_before)

        # One gradient step updates ONLY LoRA weights (base weights remain frozen!)
        w_base_before = m.q_proj.base.weight.detach().clone()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-1)
        loss = m(x).pow(2).mean()
        loss.backward()
        opt.step()
        torch.testing.assert_close(m.q_proj.base.weight, w_base_before)
        with torch.no_grad():
            y_after = m(x)
        self.assertGreater(float((y_after - y_before).abs().max()), 1e-4)

        # Resetting LoRA between cross-task folds restores bit-exact zero-init output!
        for a in adapters:
            a.reset_lora()
        with torch.no_grad():
            y_reset = m(x)
        torch.testing.assert_close(y_reset, y_before)


if __name__ == "__main__":
    unittest.main()
