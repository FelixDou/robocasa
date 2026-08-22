import unittest

from tests.safe_import_helper import install_lightweight_robocasa_packages


install_lightweight_robocasa_packages()

from robocasa.recovery.recovery_rollout import (  # noqa: E402
    begin_policy_recovery,
    end_policy_recovery,
    pop_policy_candidate_selection,
)


class HookPolicy:
    def __init__(self):
        self.begin_kwargs = None
        self.ended = False
        self.record = {"selected_index": 2}

    def begin_recovery(self, **kwargs):
        self.begin_kwargs = kwargs
        return {"enabled": True}

    def pop_candidate_selection_record(self):
        record = self.record
        self.record = None
        return record

    def end_recovery(self):
        self.ended = True


class TestSafeRecoveryPolicyHooks(unittest.TestCase):
    def test_hooks_are_optional(self):
        policy = object()
        self.assertIsNone(
            begin_policy_recovery(
                policy,
                task_name="Task",
                target_subtask="stage",
                instruction="do it",
            )
        )
        self.assertIsNone(pop_policy_candidate_selection(policy))
        end_policy_recovery(policy)

    def test_hooks_forward_context_and_records(self):
        policy = HookPolicy()
        activation = begin_policy_recovery(
            policy,
            task_name="CloseBlenderLid",
            target_subtask="close_lid",
            instruction="close the blender lid",
        )
        self.assertEqual(activation, {"enabled": True})
        self.assertEqual(
            policy.begin_kwargs,
            {
                "task_name": "CloseBlenderLid",
                "target_subtask": "close_lid",
                "instruction": "close the blender lid",
            },
        )
        self.assertEqual(pop_policy_candidate_selection(policy), {"selected_index": 2})
        self.assertIsNone(pop_policy_candidate_selection(policy))
        end_policy_recovery(policy)
        self.assertTrue(policy.ended)


if __name__ == "__main__":
    unittest.main()
