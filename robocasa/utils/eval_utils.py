from robocasa.utils.dataset_registry import (
    ATOMIC_TASK_DATASETS,
    COMPOSITE_TASK_DATASETS,
)
from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import (
    get_env_metadata_from_dataset,
)
from robosuite.controllers import load_composite_controller_config
import os
import robosuite
import imageio
import numpy as np
from tqdm import tqdm
from termcolor import colored
from robocasa.recovery.subtask_eval import get_subtask_eval, summarize_subtask_rollout


def create_eval_env(
    env_name,
    # robosuite-related configs
    robots="PandaMobile",
    controllers="OSC_POSE",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    seed=None,
    # robocasa-related configs
    obj_instance_split="target",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
):
    controller_configs = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env


def run_random_rollouts(env, num_rollouts, num_steps, video_path=None):
    video_writer = None
    if video_path is not None:
        video_writer = imageio.get_writer(video_path, fps=20)

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        for step_i in range(num_steps):
            # sample and execute random action
            action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
            obs, _, _, _ = env.step(action)

            if video_writer is not None:
                video_img = env.sim.render(
                    height=512, width=512, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(video_img)

            if env._check_success():
                num_success_rollouts += 1
                break

    if video_writer is not None:
        video_writer.close()
        print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))

    info["num_success_rollouts"] = num_success_rollouts

    return info


def run_random_subtask_rollouts(
    env,
    num_rollouts,
    num_steps,
    video_path=None,
    stuck_patience=10,
    include_subtask_trace=True,
):
    """
    Run random rollouts while collecting optional subtask-level progress.

    This is a parallel evaluator to ``run_random_rollouts``. It reports the same
    official sparse success count, plus named predicate progress when the task
    implements ``get_subtask_progress()``. The subtask trace evaluates named
    predicates at every step and derives progress, stuck-subtask, failed
    precondition, wrong-object / wrong-receptacle, and no-progress signals.
    """
    video_writer = None
    if video_path is not None:
        video_writer = imageio.get_writer(video_path, fps=20)

    info = {"rollouts": []}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        rollout_subtask_evals = [get_subtask_eval(env)]

        success = False
        steps_taken = 0
        for step_i in range(num_steps):
            steps_taken = step_i + 1
            action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
            obs, _, _, _ = env.step(action)

            if video_writer is not None:
                video_img = env.sim.render(
                    height=512, width=512, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(video_img)

            rollout_subtask_evals.append(get_subtask_eval(env))

            if env._check_success():
                num_success_rollouts += 1
                success = True
                break

        rollout_summary = summarize_subtask_rollout(
            rollout_subtask_evals,
            stuck_patience=stuck_patience,
            include_trace=include_subtask_trace,
        )
        rollout_summary["success"] = success
        rollout_summary["num_steps"] = steps_taken
        info["rollouts"].append(rollout_summary)

    if video_writer is not None:
        video_writer.close()
        print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))

    info["num_success_rollouts"] = num_success_rollouts

    return info


if __name__ == "__main__":
    # select random task to run rollouts for
    env_name = np.random.choice(
        list(ATOMIC_TASK_DATASETS) + list(COMPOSITE_TASK_DATASETS)
    )
    env = create_eval_env(env_name=env_name)
    info = run_random_rollouts(
        env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
    )
