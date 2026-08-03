import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe.collect_atomic_rollouts import (
    _assert_resume_compatible,
    build_parser,
    prepare_plan,
    run_collection,
)
from robocasa.recovery.safe.audit_subtask_safe_dataset import (
    audit_subtask_datasets,
)
from robocasa.recovery.safe.dataset import load_manifest
from robocasa.recovery.safe.export_to_official_safe import export_to_official_safe
from robocasa.recovery.safe.merge_atomic_datasets import merge_atomic_datasets
from robocasa.recovery.safe.validate_atomic_dataset import validate_atomic_dataset


TASK = "TurnOnSinkFaucet"


class FakeEnv:
    def __init__(self, seed):
        self.seed = seed
        self.steps = 0
        self.reset_calls = 0
        self.closed = False

    def reset(self):
        self.reset_calls += 1
        self.steps = 0
        return {"annotation.human.task_description": "turn on the sink faucet"}, {}

    def step(self, action):
        self.steps += 1
        success = self.seed % 2 == 0 and self.steps >= 2
        return {}, 0.0, success, False, {"success": success}

    def close(self):
        self.closed = True

    def get_subtask_progress(self):
        success = self.seed % 2 == 0 and self.steps >= 2
        return {
            "required_predicates": ["mock_subtask"],
            "predicates": {
                "mock_subtask": {
                    "value": success,
                    "stage": "subtask",
                }
            },
            "task_success": success,
        }


class FakePolicy:
    def __init__(self, env, replan_steps=2, **kwargs):
        self.env = env
        self.replan_steps = int(replan_steps)
        self.reset()

    def reset(self):
        self.env_step = 0
        self.inference_index = 0
        self.pending = None

    def __call__(self, obs, instruction=None):
        if self.env_step % self.replan_steps == 0:
            value = self.env.seed + self.inference_index + 1
            self.pending = {
                "env_step": self.env_step,
                "environment_step": self.env_step,
                "inference_index": self.inference_index,
                "features": np.full((2, 4, 8), value, dtype=np.float32),
                "actions": np.full((4, 12), value, dtype=np.float32),
                "metadata": {
                    "schema_version": 1,
                    "feature_layer": "action_expert_suffix_pre_action_out_proj",
                    "feature_dtype": "float32",
                    "feature_aggregation": "raw",
                    "policy_name": "mock-pi0",
                    "policy_checkpoint": "mock-checkpoint",
                    "action_horizon": 4,
                    "flow_steps": 2,
                },
            }
            self.inference_index += 1
        self.env_step += 1
        return np.full(12, self.env.seed, dtype=np.float32)

    def pop_inference_record(self):
        record, self.pending = self.pending, None
        return record


class FakeRLDXPolicy(FakePolicy):
    def __call__(self, obs, instruction=None):
        action = super().__call__(obs, instruction=instruction)
        if self.pending is not None:
            self.pending["metadata"].update(
                {
                    "model_family": "rldx1",
                    "feature_layer": (
                        "action_model_msat_action_suffix_pre_action_decoder"
                    ),
                    "policy_name": "mock-rldx1",
                    "policy_checkpoint": "mock-rldx-checkpoint",
                }
            )
        return action


def fake_runtime(tracker=None, policy_cls=FakePolicy):
    def make_env(task, interface, split, seed, render):
        env = FakeEnv(seed)
        if tracker is not None:
            tracker.setdefault("envs", []).append(env)
        return env

    return {
        "load_factory": lambda spec: policy_cls,
        "parse_policy_args": lambda values: {},
        "make_env": make_env,
        "open_video_writer": lambda path, fps: None,
        "call_factory": lambda factory, env, args: factory(env, **args),
        "append_frame": lambda *args, **kwargs: None,
        "success_fn": lambda info, reward, env: info.get("success", False),
        "step_fn": lambda env, action: env.step(action),
    }


def collection_args(
    output_dir, *, num_rollouts=2, resume=False, success_quota=None, failure_quota=None
):
    values = [
        "--output-dir",
        str(output_dir),
        "--tasks",
        TASK,
        "--num-rollouts",
        str(num_rollouts),
        "--policy-name",
        "mock-pi0",
        "--checkpoint",
        "mock-checkpoint",
        "--replan-steps",
        "2",
        "--horizon",
        "3",
    ]
    if resume:
        values.append("--resume")
    if success_quota is not None:
        values.extend(("--success-quota", str(success_quota)))
    if failure_quota is not None:
        values.extend(("--failure-quota", str(failure_quota)))
    return build_parser().parse_args(values)


class TestSafeAtomicCollection(unittest.TestCase):
    def test_atomic_task_and_numeric_option_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp)
            args.tasks = ["NotARegisteredAtomicTask"]
            with self.assertRaisesRegex(ValueError, "not registered"):
                prepare_plan(args)
            args = collection_args(tmp)
            args.success_quota = -1
            with self.assertRaisesRegex(ValueError, "non-negative"):
                prepare_plan(args)
            args = collection_args(tmp)
            args.max_errors = 0
            with self.assertRaisesRegex(ValueError, "--max-errors must be positive"):
                prepare_plan(args)

    def test_dry_run_is_simulator_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp)
            args.dry_run = True
            result = run_collection(args, runtime=None)
            self.assertTrue(result["dry_run"])
            self.assertEqual(len(result["attempts"]), 2)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_environment_split_is_provenance_and_rollout_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_args = collection_args(tmp)
            test_plan = prepare_plan(test_args)
            pretrain_args = collection_args(tmp)
            pretrain_args.split = "pretrain"
            pretrain_plan = prepare_plan(pretrain_args)

            self.assertEqual(test_plan["config"]["split"], "test")
            self.assertEqual(test_plan["identity"]["split"], "test")
            self.assertNotEqual(
                test_plan["attempts"][0]["rollout_id"],
                pretrain_plan["attempts"][0]["rollout_id"],
            )
            with self.assertRaisesRegex(ValueError, "split"):
                _assert_resume_compatible(test_plan["config"], pretrain_plan["config"])

    def test_resume_accepts_atomic_config_from_before_task_scope_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp)
            current = prepare_plan(args)["config"]
            previous = dict(current)
            previous.pop("task_types")
            previous.pop("task_scope")
            previous.pop("dataset_type")

            _assert_resume_compatible(previous, current)

    def test_resume_can_extend_but_not_shrink_official_reset_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            initial_args = collection_args(tmp, num_rollouts=2)
            initial_args.seed_protocol = "official_rldx"
            initial_args.model_family = "rldx1"
            initial = prepare_plan(initial_args)["config"]

            extended_args = collection_args(tmp, num_rollouts=4)
            extended_args.seed_protocol = "official_rldx"
            extended_args.model_family = "rldx1"
            extended = prepare_plan(extended_args)["config"]

            _assert_resume_compatible(initial, extended)
            with self.assertRaisesRegex(ValueError, "environment_reset_indices"):
                _assert_resume_compatible(extended, initial)

    def test_official_openpi_seed_protocol_reuses_one_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp, num_rollouts=3)
            args.seed = 7
            args.seed_protocol = "official_openpi"
            args.video_frame_stride = 2
            plan = prepare_plan(args)

            self.assertEqual(plan["config"]["seeds"], [7])
            self.assertEqual(plan["config"]["environment_reset_indices"], [0, 1, 2])
            self.assertEqual(
                [attempt["environment_seed"] for attempt in plan["attempts"]],
                [7, 7, 7],
            )
            self.assertEqual(
                [attempt["environment_reset_index"] for attempt in plan["attempts"]],
                [0, 1, 2],
            )

            tracker = {}
            result = run_collection(args, runtime=fake_runtime(tracker))
            records = load_manifest(tmp)
            self.assertEqual(result["counts"]["valid_rollouts"], 3)
            self.assertEqual(len(tracker["envs"]), 1)
            self.assertEqual(tracker["envs"][0].reset_calls, 3)
            self.assertTrue(tracker["envs"][0].closed)
            self.assertEqual([record.environment_seed for record in records], [7, 7, 7])
            self.assertEqual(
                [record.environment_reset_index for record in records], [0, 1, 2]
            )
            self.assertTrue(
                all(record.seed_protocol == "official_openpi" for record in records)
            )
            self.assertTrue(all(record.video_frame_stride == 2 for record in records))
            validation = validate_atomic_dataset(tmp)
            self.assertTrue(validation["valid"], validation["errors"])

    def test_rldx_collection_records_model_family_and_repeated_reset_protocol(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as exported,
        ):
            args = collection_args(tmp, num_rollouts=2)
            args.seed = 7
            args.seed_protocol = "official_rldx"
            args.model_family = "rldx1"
            args.policy_module = "robocasa.recovery.rldx_zmq_policy:make_policy"
            args.policy_name = "mock-rldx1"
            args.checkpoint = "mock-rldx-checkpoint"
            tracker = {}

            result = run_collection(
                args,
                runtime=fake_runtime(tracker, policy_cls=FakeRLDXPolicy),
            )
            records = load_manifest(tmp)

            self.assertEqual(result["counts"]["valid_rollouts"], 2)
            self.assertEqual(len(tracker["envs"]), 1)
            self.assertTrue(all(record.model_family == "rldx1" for record in records))
            self.assertTrue(
                all(record.seed_protocol == "official_rldx" for record in records)
            )
            self.assertTrue(
                all(record.openpi_repository_commit is None for record in records)
            )
            self.assertTrue(all(record.rldx_repository_commit for record in records))
            validation = validate_atomic_dataset(tmp)
            self.assertTrue(validation["valid"], validation["errors"])
            report = export_to_official_safe(tmp, exported)
            self.assertEqual(
                report["format"],
                "official_safe_rldx1_env_records_policy_records",
            )
            self.assertEqual(report["model_families"], ["rldx1"])
            policy_path = next((Path(exported) / "policy_records").glob("*meta.pkl"))
            with policy_path.open("rb") as stream:
                policy_record = pickle.load(stream)
            self.assertEqual(policy_record["model_family"], "rldx1")
            self.assertEqual(policy_record["pre_velocity"].shape, (2, 4, 8))

    def test_model_family_rejects_other_policy_seed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp)
            args.model_family = "rldx1"
            args.seed_protocol = "official_openpi"
            with self.assertRaisesRegex(ValueError, "incompatible"):
                prepare_plan(args)

            args = collection_args(tmp)
            args.model_family = "pi0"
            args.seed_protocol = "official_rldx"
            with self.assertRaisesRegex(ValueError, "incompatible"):
                prepare_plan(args)

    def test_official_protocol_rejects_seed_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp)
            args.seed_protocol = "official_openpi"
            args.seed_end = 9
            with self.assertRaisesRegex(ValueError, "--seed-end is incompatible"):
                prepare_plan(args)

    def test_official_task_horizons_are_resolved_per_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp)
            args.tasks = ["OpenCabinet", "TurnOnElectricKettle", TASK]
            args.horizon = None
            plan = prepare_plan(args)

            self.assertEqual(
                plan["config"]["task_horizons"],
                {
                    "OpenCabinet": 1050,
                    "TurnOnElectricKettle": 450,
                    TASK: 600,
                },
            )
            self.assertEqual(
                plan["config"]["horizon_source"],
                "robocasa_dataset_registry",
            )
            self.assertEqual(
                [attempt["rollout_horizon"] for attempt in plan["attempts"]],
                [1050, 1050, 450, 450, 600, 600],
            )

    def test_registered_composite_horizons_and_mixed_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp, num_rollouts=1)
            args.tasks = [
                TASK,
                "LoadDishwasher",
                "PreSoakPan",
                "ScrubCuttingBoard",
                "StackBowlsCabinet",
                "WashLettuce",
            ]
            args.horizon = None
            plan = prepare_plan(args)

            self.assertEqual(
                plan["config"]["task_horizons"],
                {
                    TASK: 600,
                    "LoadDishwasher": 1800,
                    "PreSoakPan": 2400,
                    "ScrubCuttingBoard": 1200,
                    "StackBowlsCabinet": 2100,
                    "WashLettuce": 1650,
                },
            )
            self.assertEqual(plan["config"]["task_scope"], "mixed")
            self.assertEqual(
                plan["config"]["dataset_type"],
                "robocasa_mixed_safe_rollouts",
            )
            self.assertEqual(plan["config"]["task_types"][TASK], "atomic")
            self.assertEqual(
                plan["config"]["task_types"]["LoadDishwasher"],
                "composite",
            )

    def test_registered_composite_collection_validates_without_escape_hatch(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as exported,
        ):
            args = collection_args(tmp)
            args.tasks = ["PreSoakPan"]
            result = run_collection(args, runtime=fake_runtime())

            self.assertEqual(result["dataset_type"], "robocasa_composite_safe_rollouts")
            self.assertEqual(result["config"]["task_scope"], "composite")
            validation = validate_atomic_dataset(tmp)
            self.assertTrue(validation["valid"], validation["errors"])
            report = export_to_official_safe(tmp, exported)
            self.assertEqual(report["task_types"], {"PreSoakPan": "composite"})
            env_path = next((Path(exported) / "env_records").glob("*.pkl"))
            with env_path.open("rb") as stream:
                env_record = pickle.load(stream)
            self.assertEqual(env_record["task_suite_name"], "robocasa_composite")
            self.assertEqual(env_record["task_type"], "composite")

    def test_unregistered_task_requires_explicit_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp)
            args.tasks = ["CustomAtomicTask"]
            args.allow_unregistered_atomic_tasks = True
            args.horizon = None
            with self.assertRaisesRegex(ValueError, "--horizon is required"):
                prepare_plan(args)

    def test_success_failure_schema_validation_and_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_collection(collection_args(tmp), runtime=fake_runtime())
            self.assertFalse(result["partial"])
            self.assertEqual(result["counts"]["valid_rollouts"], 2)
            self.assertEqual(result["counts"]["successes"], 1)
            self.assertEqual(result["counts"]["failures"], 1)
            records = load_manifest(tmp)
            self.assertEqual([record.failed for record in records], [False, True])
            self.assertEqual(records[0].termination_reason, "success")
            self.assertEqual(records[1].termination_reason, "timeout")
            self.assertTrue(
                all(record.environment_split == "test" for record in records)
            )
            self.assertEqual(records[0].inference_env_steps, [0])
            self.assertEqual(records[1].inference_env_steps, [0, 2])
            for record in records:
                self.assertTrue((Path(tmp) / record.action_path).is_file())
                with np.load(
                    Path(tmp) / record.tensor_path, allow_pickle=False
                ) as payload:
                    self.assertIn("policy_action_chunks", payload.files)
                    self.assertEqual(str(payload["rollout_id"]), record.rollout_id)
            validation = validate_atomic_dataset(tmp)
            self.assertTrue(validation["valid"], validation["errors"])
            self.assertTrue(validation["official_safe_loader_compatible"])

    def test_subtask_safe_artifacts_and_labels_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp)
            args.record_subtask_trace = True
            result = run_collection(args, runtime=fake_runtime())
            records = load_manifest(tmp)
            validation = validate_atomic_dataset(tmp)

            self.assertFalse(result["partial"])
            self.assertTrue(validation["valid"], validation["errors"])
            self.assertEqual(
                validation["subtask_safe"],
                {
                    "recorded_rollouts": 2,
                    "segments": 2,
                    "usable_segments": 2,
                    "successful_segments": 1,
                    "failed_segments": 1,
                    "labeled_without_inference": 0,
                    "excluded_completed_subtasks": 0,
                    "excluded_bypassed_subtasks": 0,
                },
            )
            for record in records:
                self.assertTrue(record.subtask_recording_requested)
                self.assertTrue(record.subtask_recording_available)
                path = Path(tmp) / record.subtask_trace_path
                self.assertTrue(path.is_file())
                payload = json.loads(path.read_text())
                self.assertEqual(payload["rollout_id"], record.rollout_id)
                self.assertEqual(
                    payload["segments"][0]["failure_label"],
                    int(record.failed),
                )

    def test_subtask_audit_can_explicitly_inspect_partial_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(
                tmp,
                num_rollouts=2,
                success_quota=2,
                failure_quota=2,
            )
            args.record_subtask_trace = True
            result = run_collection(args, runtime=fake_runtime())
            self.assertTrue(result["partial"])

            with self.assertRaisesRegex(ValueError, "partial/incomplete"):
                audit_subtask_datasets(
                    [tmp],
                    task_type_filter="atomic",
                )
            audit = audit_subtask_datasets(
                [tmp],
                task_type_filter="atomic",
                allow_partial=True,
                target_successes=1,
                target_failures=1,
            )

            self.assertEqual(audit["counts"]["rollouts"], 2)
            self.assertEqual(audit["counts"]["pairs_reaching_target"], 1)
            self.assertTrue(audit["sources"][0]["partial"])

    def test_resume_does_not_duplicate_manifest_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_collection(collection_args(tmp, num_rollouts=1), runtime=fake_runtime())
            run_collection(
                collection_args(tmp, num_rollouts=1, resume=True),
                runtime=fake_runtime(),
            )
            self.assertEqual(len(load_manifest(tmp)), 1)
            lines = (Path(tmp) / "manifest.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)

    def test_class_quotas_stop_remaining_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_collection(
                collection_args(tmp, num_rollouts=6, success_quota=1, failure_quota=1),
                runtime=fake_runtime(),
            )
            self.assertEqual(result["counts"]["valid_rollouts"], 2)
            skipped = [
                json.loads(line)
                for line in (Path(tmp) / "skipped.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(skipped), 4)
            self.assertTrue(
                all(event["status"] == "skipped_quota_reached" for event in skipped)
            )

    def test_retain_only_quota_discards_majority_class_excess(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(
                tmp,
                num_rollouts=6,
                success_quota=1,
                failure_quota=2,
            )
            args.retain_only_quota = True
            result = run_collection(args, runtime=fake_runtime())

            self.assertFalse(result["partial"])
            self.assertEqual(result["counts"]["valid_rollouts"], 3)
            self.assertEqual(result["counts"]["successes"], 1)
            self.assertEqual(result["counts"]["failures"], 2)
            self.assertTrue(result["per_task"][TASK]["quota_reached"])
            skipped = [
                json.loads(line)
                for line in (Path(tmp) / "skipped.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [event["status"] for event in skipped],
                [
                    "skipped_class_quota_reached",
                    "skipped_quota_reached",
                    "skipped_quota_reached",
                ],
            )
            self.assertTrue(validate_atomic_dataset(tmp)["valid"])

    def test_unmet_quota_marks_dataset_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(
                tmp,
                num_rollouts=2,
                success_quota=2,
                failure_quota=2,
            )
            result = run_collection(args, runtime=fake_runtime())
            self.assertTrue(result["partial"])
            validation = validate_atomic_dataset(tmp)
            self.assertFalse(validation["valid"])
            self.assertTrue(any("partial" in error for error in validation["errors"]))

    def test_max_errors_stops_runaway_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp, num_rollouts=6)
            args.max_errors = 2
            runtime = fake_runtime()

            def fail_factory(factory, env, values):
                raise ConnectionError("mock server unavailable")

            runtime["call_factory"] = fail_factory
            with self.assertRaisesRegex(RuntimeError, "--max-errors=2"):
                run_collection(args, runtime=runtime)

            errors = [
                json.loads(line)
                for line in (Path(tmp) / "errors.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(errors), 2)

    def test_merge_atomic_dataset_shards_with_hardlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "source_a"
            source_b = root / "source_b"
            output = root / "merged"

            args_a = collection_args(source_a, success_quota=1, failure_quota=1)
            args_b = collection_args(
                source_b,
                num_rollouts=3,
                success_quota=2,
                failure_quota=1,
            )
            args_b.seed = 2
            args_b.robocasa_commit = "different-collector-commit"
            run_collection(args_a, runtime=fake_runtime())
            run_collection(args_b, runtime=fake_runtime())
            source_b_summary_path = source_b / "summary.json"
            source_b_summary = json.loads(source_b_summary_path.read_text())
            source_b_summary["config"]["record_videos"] = True
            source_b_summary_path.write_text(json.dumps(source_b_summary))

            result = merge_atomic_datasets([source_a, source_b], output)
            summary = result["summary"]
            self.assertEqual(summary["counts"]["valid_rollouts"], 5)
            self.assertEqual(summary["counts"]["successes"], 3)
            self.assertEqual(summary["counts"]["failures"], 2)
            self.assertEqual(set(summary["per_task"]), {TASK})
            self.assertEqual(summary["config"]["base_environment_seeds"], [0, 2])
            self.assertEqual(len(summary["config"]["robocasa_commits"]), 2)
            self.assertIsNone(summary["config"]["robocasa_commit"])
            self.assertIsNone(summary["config"]["record_videos"])
            self.assertIsNone(summary["config"]["success_quota"])
            self.assertTrue(result["validation"]["valid"])
            for record in load_manifest(output):
                source = source_a if record.environment_seed < 2 else source_b
                self.assertEqual(
                    (source / record.tensor_path).stat().st_ino,
                    (output / record.tensor_path).stat().st_ino,
                )

    def test_merge_atomic_and_composite_shards_records_mixed_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic = root / "atomic"
            composite = root / "composite"
            output = root / "mixed"

            atomic_args = collection_args(atomic, num_rollouts=1)
            composite_args = collection_args(composite, num_rollouts=1)
            composite_args.tasks = ["PreSoakPan"]
            run_collection(atomic_args, runtime=fake_runtime())
            run_collection(composite_args, runtime=fake_runtime())

            result = merge_atomic_datasets([atomic, composite], output)
            config = result["summary"]["config"]
            self.assertEqual(config["task_scope"], "mixed")
            self.assertEqual(config["dataset_type"], "robocasa_mixed_safe_rollouts")
            self.assertEqual(
                config["task_types"],
                {TASK: "atomic", "PreSoakPan": "composite"},
            )
            self.assertTrue(result["validation"]["valid"])

    def test_resume_quarantines_orphan_then_recollects(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collection_args(tmp, num_rollouts=1, resume=True)
            rollout_id = prepare_plan(args)["attempts"][0]["rollout_id"]
            orphan = Path(tmp) / "rollouts" / f"{rollout_id}.npz"
            orphan.parent.mkdir(parents=True)
            orphan.write_bytes(b"interrupted")
            action_temp = Path(tmp) / "actions" / TASK / f"{rollout_id}.tmp.npz"
            action_temp.parent.mkdir(parents=True)
            action_temp.write_bytes(b"interrupted")
            run_collection(args, runtime=fake_runtime())
            quarantined = list(
                (Path(tmp) / "incomplete" / rollout_id).glob("feature--*")
            )
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(
                len(
                    list((Path(tmp) / "incomplete" / rollout_id).glob("action_temp--*"))
                ),
                1,
            )
            self.assertEqual(len(load_manifest(tmp)), 1)

    def test_validator_detects_missing_requested_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_collection(collection_args(tmp, num_rollouts=1), runtime=fake_runtime())
            record = load_manifest(tmp)[0]
            (Path(tmp) / record.action_path).unlink()
            validation = validate_atomic_dataset(tmp)
            self.assertFalse(validation["valid"])
            self.assertTrue(
                any("action artifact" in error for error in validation["errors"])
            )

    def test_validator_detects_success_failure_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_collection(collection_args(tmp, num_rollouts=1), runtime=fake_runtime())
            manifest = Path(tmp) / "manifest.jsonl"
            payload = json.loads(manifest.read_text())
            payload["success"] = not payload["success"]
            manifest.write_text(json.dumps(payload) + "\n")
            validation = validate_atomic_dataset(tmp)
            self.assertFalse(validation["valid"])
            self.assertTrue(
                any("success disagrees" in error for error in validation["errors"])
            )

    def test_deterministic_official_export_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as exported:
            run_collection(collection_args(tmp), runtime=fake_runtime())
            dry_run = export_to_official_safe(tmp, exported, dry_run=True)
            self.assertTrue(dry_run["dry_run"])
            report = export_to_official_safe(tmp, exported)
            self.assertEqual(report["num_rollouts"], 2)
            self.assertEqual(report["num_policy_records"], 3)
            env_paths = sorted((Path(exported) / "env_records").glob("*.pkl"))
            policy_paths = sorted((Path(exported) / "policy_records").glob("*meta.pkl"))
            self.assertEqual((len(env_paths), len(policy_paths)), (2, 3))
            policy_step = 0
            labels = []
            for env_path in env_paths:
                with env_path.open("rb") as stream:
                    env_record = pickle.load(stream)
                labels.append(env_record["episode_success"])
                for _ in range(env_record["model_infer_times"]):
                    with policy_paths[policy_step].open("rb") as stream:
                        policy_record = pickle.load(stream)
                    self.assertEqual(policy_record["pre_velocity"].shape, (2, 4, 8))
                    self.assertEqual(policy_record["actions"].shape, (4, 12))
                    policy_step += 1
            self.assertEqual(sorted(labels), [0, 1])
            resumed = export_to_official_safe(tmp, exported, resume=True)
            self.assertTrue(resumed["complete"])
            self.assertEqual(
                len(list((Path(exported) / "policy_records").glob("*meta.pkl"))), 3
            )

    def test_official_export_can_select_exact_per_task_class_balance(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as exported:
            run_collection(
                collection_args(tmp, num_rollouts=6),
                runtime=fake_runtime(),
            )
            report = export_to_official_safe(
                tmp,
                exported,
                successes_per_task=2,
                failures_per_task=2,
                selection_seed=7,
            )
            self.assertEqual(report["num_rollouts"], 4)
            self.assertEqual(
                report["selection"],
                {
                    "mode": "per_task_class_balance",
                    "seed": 7,
                    "successes_per_task": 2,
                    "failures_per_task": 2,
                    "source_num_rollouts": 6,
                    "selected_num_rollouts": 4,
                    "per_task": {
                        TASK: {
                            "source_successes": 3,
                            "source_failures": 3,
                            "selected_successes": 2,
                            "selected_failures": 2,
                        }
                    },
                },
            )
            labels = []
            for path in sorted((Path(exported) / "env_records").glob("*.pkl")):
                with path.open("rb") as stream:
                    labels.append(pickle.load(stream)["episode_success"])
            self.assertEqual(sorted(labels), [0, 0, 1, 1])

            with self.assertRaisesRegex(ValueError, "different rollout selection"):
                export_to_official_safe(
                    tmp,
                    exported,
                    resume=True,
                    successes_per_task=1,
                    failures_per_task=1,
                    selection_seed=7,
                )


if __name__ == "__main__":
    unittest.main()
