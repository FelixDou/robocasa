"""Collect successful and naturally failed π0 rollouts with raw SAFE latents."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .dataset import save_rollout
from .schema import SafeRolloutMetadata
from .subtask_safe import build_subtask_safe_record


def collect_single_rollout(
    policy,
    env,
    *,
    horizon,
    instruction=None,
    step_fn=None,
    success_fn=None,
    video_writer=None,
    frame_fn=None,
    video_frame_stride=1,
    require_safe_features=True,
    record_subtask_trace=False,
    subtask_eval_fn=None,
):
    """Simulator-light collection core used by the live CLI and mocked tests."""
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    reset_info = (
        reset_result[1]
        if (
            isinstance(reset_result, tuple)
            and len(reset_result) > 1
            and isinstance(reset_result[1], dict)
        )
        else {}
    )
    if hasattr(policy, "reset"):
        policy.reset()
    if instruction is None and isinstance(obs, dict):
        instruction = obs.get("annotation.human.task_description")
    records = []
    success = False
    termination_reason = "timeout"
    previous_frame = None
    actions = []
    num_env_steps = 0
    step_fn = step_fn or (lambda environment, action: environment.step(action))
    video_frame_stride = int(video_frame_stride)
    if video_frame_stride < 1:
        raise ValueError("video_frame_stride must be positive")
    num_video_frames = 0
    subtask_evals = []
    if record_subtask_trace:
        if subtask_eval_fn is None:
            from robocasa.recovery.subtask_eval import get_subtask_eval

            subtask_eval_fn = get_subtask_eval
        subtask_evals.append(
            reset_info.get("subtask_eval")
            if reset_info.get("subtask_eval") is not None
            else subtask_eval_fn(env)
        )
    for step_index in range(horizon):
        action = policy(obs, instruction=instruction)
        actions.append(action)
        pop_record = getattr(policy, "pop_inference_record", None)
        record = pop_record() if pop_record is not None else None
        if record is not None:
            records.append(record)
        result = step_fn(env, action)
        num_env_steps += 1
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        if record_subtask_trace:
            subtask_evals.append(
                info.get("subtask_eval")
                if isinstance(info, dict)
                and info.get("subtask_eval") is not None
                else subtask_eval_fn(env)
            )
        success = bool(success_fn(info, reward, env) if success_fn else (info or {}).get("success", False))
        if (
            video_writer is not None
            and frame_fn is not None
            and (
                step_index % video_frame_stride == 0
                or step_index == horizon - 1
                or done
                or success
            )
        ):
            previous_frame = frame_fn(env, video_writer, obs, previous_frame)
            num_video_frames += 1
        if success:
            termination_reason = "success"
            break
        if done:
            termination_reason = "environment_done"
            break
    if require_safe_features and not records:
        raise RuntimeError("Rollout produced no SAFE policy-inference records")
    expected = list(range(len(records)))
    if [record["inference_index"] for record in records] != expected:
        raise RuntimeError("Inference indices are not contiguous within the rollout")
    inference_steps = [
        int(record.get("environment_step", record.get("env_step")))
        for record in records
    ]
    if inference_steps != sorted(set(inference_steps)):
        raise RuntimeError("SAFE inference environment steps must be strictly increasing")
    metadata = records[0]["metadata"] if records else {}
    for record in records[1:]:
        candidate = record["metadata"]
        for key in (
            "schema_version",
            "model_family",
            "feature_layer",
            "feature_dtype",
            "feature_aggregation",
            "policy_name",
            "policy_checkpoint",
            "action_horizon",
        ):
            if candidate.get(key) != metadata.get(key):
                raise RuntimeError(f"SAFE feature metadata changed within rollout: {key}")
    features = (
        np.stack([record["features"] for record in records]).astype(np.float32)
        if records
        else None
    )
    policy_action_chunks = (
        np.stack([record["actions"] for record in records]).astype(np.float32)
        if records and all("actions" in record for record in records)
        else None
    )
    subtask_safe_record = (
        build_subtask_safe_record(
            subtask_evals,
            inference_steps,
            rollout_failed=not success,
        )
        if record_subtask_trace
        else None
    )
    return {
        "success": success,
        "num_env_steps": num_env_steps,
        "instruction": instruction or "",
        "termination_reason": termination_reason,
        "features": features,
        "inference_env_steps": inference_steps,
        "feature_metadata": metadata,
        "policy_action_chunks": policy_action_chunks,
        "actions": actions,
        "num_video_frames": num_video_frames,
        "subtask_safe_record": subtask_safe_record,
    }


def _runtime():
    from robocasa.recovery.evaluate_recovery_benchmark import (
        call_factory,
        load_factory,
        make_env,
        open_video_writer,
        parse_policy_args,
    )
    from robocasa.recovery.recovery_rollout import (
        _append_video_frame_from_env,
        _is_task_success,
        _step_env,
    )

    return (
        call_factory,
        load_factory,
        make_env,
        open_video_writer,
        parse_policy_args,
        _append_video_frame_from_env,
        _is_task_success,
        _step_env,
    )


def run_collection(args):
    (
        call_factory,
        load_factory,
        make_env,
        open_video_writer,
        parse_policy_args,
        append_frame,
        success_fn,
        step_fn,
    ) = _runtime()
    factory = load_factory(args.policy_module)
    policy_args = parse_policy_args(args.policy_arg)
    policy_args["collect_safe_features"] = True
    for task_name in args.envs:
        successes = 0
        failures = 0
        for rollout_index in range(args.num_rollouts):
            seed = args.seed + rollout_index
            env = make_env(task_name, args.env_interface, args.split, seed, args.record_videos)
            video_path = (
                args.output_dir / "videos" / task_name / f"seed_{seed:06d}.mp4"
                if args.record_videos
                else None
            )
            video_writer = open_video_writer(video_path, args.video_fps)
            try:
                policy = call_factory(factory, env, policy_args)

                def frame_fn(environment, writer, obs, previous):
                    return append_frame(
                        environment,
                        writer,
                        camera_name=args.video_camera_name,
                        height=args.video_height,
                        width=args.video_width,
                        obs=obs,
                        previous_frame=previous,
                    )

                rollout = collect_single_rollout(
                    policy,
                    env,
                    horizon=args.horizon,
                    step_fn=step_fn,
                    success_fn=success_fn,
                    video_writer=video_writer,
                    frame_fn=frame_fn,
                )
            finally:
                if video_writer is not None:
                    video_writer.close()
                env.close()
            feature_meta = rollout["feature_metadata"]
            stable_id = hashlib.sha256(
                f"{task_name}:{seed}:{args.policy_id}:{args.checkpoint}".encode()
            ).hexdigest()[:20]
            metadata = SafeRolloutMetadata(
                rollout_id=stable_id,
                task_name=task_name,
                task_instruction=rollout["instruction"],
                environment_seed=seed,
                policy_id=args.policy_id,
                checkpoint=args.checkpoint,
                failed=not rollout["success"],
                num_env_steps=rollout["num_env_steps"],
                inference_env_steps=rollout["inference_env_steps"],
                valid_sequence_length=len(rollout["inference_env_steps"]),
                action_horizon=int(feature_meta["action_horizon"]),
                replan_steps=int(policy.replan_steps),
                feature_layer=feature_meta["feature_layer"],
                feature_aggregation="raw",
                flow_steps=int(feature_meta["flow_steps"]),
                termination_reason=rollout["termination_reason"],
                timeout_horizon=args.horizon,
                video_path=str(video_path) if video_path is not None else None,
            )
            save_rollout(args.output_dir, metadata, rollout["features"])
            successes += int(rollout["success"])
            failures += int(not rollout["success"])
            if (
                successes >= args.min_successes_per_task
                and failures >= args.min_failures_per_task
            ):
                break


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-module", default="robocasa.recovery.openpi_websocket_policy:make_policy")
    parser.add_argument("--policy-arg", action="append", default=[])
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--env-interface", choices=["gym", "robosuite"], default="gym")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--record-videos", action="store_true")
    parser.add_argument("--video-camera-name", default="robot0_agentview_center")
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--video-width", type=int, default=768)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--min-successes-per-task", type=int, default=0)
    parser.add_argument("--min-failures-per-task", type=int, default=0)
    return parser


def main(argv=None):
    run_collection(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
