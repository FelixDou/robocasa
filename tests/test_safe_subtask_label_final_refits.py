import json
from pathlib import Path
import pickle
import tempfile
import unittest

from robocasa.recovery.safe.evaluate_subtask_label_final_refits import (
    evaluate_final_refits,
)


def catalog_record(parent, failed):
    return {
        "rollout_id": f"{parent}:Stage",
        "parent_rollout_id": parent,
        "parent_task_name": "Composite",
        "parent_rollout_failed": failed,
        "task_name": "Composite::Stage",
        "subtask_id": "Stage",
        "episode_success": int(not failed),
        "model_infer_times": 2,
        "inference_environment_steps": [0, 1],
        "subtask_safe_segment": {
            "inference_start_index": 0,
            "inference_end_index_exclusive": 2,
            "num_policy_inferences": 2,
            "entry_environment_step": 0,
            "end_environment_step": 2,
        },
    }


class TestSubtaskLabelFinalRefits(unittest.TestCase):
    def test_terminal_and_subtask_refits_share_outer_subtask_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_root = root / "catalog"
            env_root = catalog_root / "env_records"
            env_root.mkdir(parents=True)
            catalog = [
                catalog_record("train-S", False),
                catalog_record("train-F", True),
                catalog_record("test-S", False),
                catalog_record("test-F", True),
            ]
            for index, record in enumerate(catalog):
                with (env_root / f"{index:02d}.pkl").open("wb") as stream:
                    pickle.dump(record, stream)

            cv_root = root / "cv"
            stage_catalog = {
                "selected_stages": ["Composite::Stage"],
                "validation_parent_ids_by_fold": {"0": [], "1": [], "2": []},
            }
            for treatment in ("terminal", "subtask"):
                path = cv_root / treatment
                path.mkdir(parents=True)
                (path / "cv_plan_indep.json").write_text(
                    json.dumps({"common_subtask_stage_catalog": stage_catalog})
                )

            final_root = root / "final"
            for treatment in ("terminal", "subtask"):
                run = final_root / treatment / "indep_seed0"
                run.mkdir(parents=True)
                rows = []
                for split, parents in (
                    ("train", ("train-S", "train-F")),
                    ("test", ("test-S", "test-F")),
                ):
                    for parent in parents:
                        failed = parent.endswith("F")
                        if treatment == "terminal":
                            row = {
                                "rollout_id": parent,
                                "parent_rollout_id": None,
                                "subtask_safe_segment": None,
                            }
                        else:
                            item = next(
                                value
                                for value in catalog
                                if value["parent_rollout_id"] == parent
                            )
                            row = {
                                "rollout_id": item["rollout_id"],
                                "parent_rollout_id": parent,
                                "subtask_safe_segment": item["subtask_safe_segment"],
                            }
                        rows.append(
                            {
                                **row,
                                "split": split,
                                "scores": [0.8, 0.9] if failed else [0.1, 0.2],
                            }
                        )
                (run / "scores.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                (run / "metrics.json").write_text("{}\n")

            result = evaluate_final_refits(
                final_root,
                cv_root,
                catalog_root,
                root / "analysis",
                models=("indep",),
                seeds=(0,),
                prefixes=(1, 2),
                primary_prefixes=(1, 2),
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_refits"], 2)
            self.assertEqual(result["outer_test_parents"], 2)
            self.assertEqual(
                result["aggregate"]["terminal/indep"][
                    "primary_task_stage_macro_roc_auc"
                ]["mean"],
                1.0,
            )
            self.assertTrue(
                (root / "analysis" / "outer_subtask_label_results.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
