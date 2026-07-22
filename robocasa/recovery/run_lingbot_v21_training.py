"""Run LingBot training with its LeRobot v2.1 decoding branch enabled.

LingBot-VLA-v2 documents direct support for LeRobot v2.1 and v3.0.  Its
``base_dataset`` module currently chooses the v3 video layout whenever the
modern ``lerobot.datasets`` import path exists.  LeRobot 0.3.3 uses that import
path but has ``CODEBASE_VERSION = "v2.1"``, so the import-path heuristic selects
the wrong video layout.

This launcher is intentionally narrow: it requires LeRobot 0.3.3/v2.1, switches
LingBot's already-implemented layout flag to v2, then executes the unmodified
upstream training script.  It does not modify either source checkout.
"""

from __future__ import annotations

import argparse
import functools
from importlib.metadata import version
import os
from pathlib import Path
import runpy
import sys


def install_lerobot_v21_import_compat() -> None:
    """Expose the v0.3.3 constants module at the path LingBot imports.

    LeRobot 0.3.3 keeps ``HF_LEROBOT_HOME`` in ``lerobot.constants`` while
    LingBot tries only the older ``lerobot.common.constants`` and newer
    ``lerobot.utils.constants`` paths.  Registering the latter as an alias is
    sufficient; no package files are edited.
    """
    import lerobot.constants as constants

    sys.modules.setdefault("lerobot.utils.constants", constants)


def enable_lerobot_v21_layout() -> tuple[str, str]:
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION

    install_lerobot_v21_import_compat()
    from lingbotvla.data.vla_data import base_dataset
    from lingbotvla.data.vla_data import video_utils

    package_version = version("lerobot")
    codebase_version = str(CODEBASE_VERSION)
    if package_version != "0.3.3" or codebase_version != "v2.1":
        raise RuntimeError(
            "RoboCasa training requires the dedicated LeRobot v2.1 loader: "
            f"expected lerobot=0.3.3/CODEBASE_VERSION=v2.1, got "
            f"lerobot={package_version}/CODEBASE_VERSION={codebase_version}."
        )

    base_dataset.LEROBOT_DATASET_API = "v2"

    # LingBot's dataset constructors currently default unconditionally to
    # TorchCodec. The released environment can import torchcodec even when its
    # FFmpeg shared libraries are unavailable on the host. PyAV is an official
    # LingBot dependency and its existing decoder handles the v2.1 per-episode
    # videos without requiring system FFmpeg libraries.
    def decode_video_frames_pyav(video_path, timestamps, tolerance_s, backend=None):
        return video_utils.decode_video_frames(
            video_path,
            timestamps,
            tolerance_s,
            backend="pyav",
        )

    base_dataset.decode_video_frames = decode_video_frames_pyav
    return package_version, codebase_version


def combine_action_padding_mask(joint_mask, action_is_pad):
    """Remove temporally padded action targets from LingBot V2's loss mask.

    LingBot V2 already receives LeRobot's ``action_is_pad`` tensor, but the
    released forward method only applies the per-joint mask. Consequently,
    repeated episode-tail actions contribute to the flow-matching objective.
    The effective mask must be true only for both a real action dimension and
    a real timestep.
    """
    if joint_mask is None or action_is_pad is None:
        return joint_mask
    if joint_mask.ndim != 3:
        raise ValueError(
            "Expected LingBot joint_mask with shape (batch, time, action_dim), "
            f"got {tuple(joint_mask.shape)}"
        )
    if action_is_pad.ndim != 2:
        raise ValueError(
            "Expected LingBot action_is_pad with shape (batch, time), "
            f"got {tuple(action_is_pad.shape)}"
        )
    if joint_mask.shape[:2] != action_is_pad.shape:
        raise ValueError(
            "LingBot action mask shapes disagree: "
            f"joint_mask={tuple(joint_mask.shape)}, "
            f"action_is_pad={tuple(action_is_pad.shape)}"
        )
    import torch

    valid_timesteps = ~action_is_pad.to(dtype=torch.bool)
    return joint_mask.to(dtype=torch.bool) & valid_timesteps.unsqueeze(-1)


def install_lingbot_v2_action_padding_mask() -> None:
    """Patch the pinned LingBot V2 policy to mask padded action timesteps."""
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
        LingbotVlaV2Policy,
    )

    original_forward = LingbotVlaV2Policy.forward
    if getattr(original_forward, "_robocasa_masks_action_padding", False):
        return

    @functools.wraps(original_forward)
    def forward_with_action_padding(self, *args, **kwargs):
        # The training script calls the policy with ``model(**micro_batch)``.
        # Keep the wrapper narrow and fail clearly if the pinned call contract
        # changes instead of silently training without the temporal mask.
        if "joint_mask" not in kwargs or "action_is_pad" not in kwargs:
            raise RuntimeError(
                "LingBot V2 training must pass joint_mask and action_is_pad as "
                "keyword arguments for RoboCasa temporal padding masking."
            )
        kwargs["joint_mask"] = combine_action_padding_mask(
            kwargs["joint_mask"], kwargs["action_is_pad"]
        )
        return original_forward(self, *args, **kwargs)

    forward_with_action_padding._robocasa_masks_action_padding = True
    LingbotVlaV2Policy.forward = forward_with_action_padding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-script", type=Path, required=True)
    return parser


def main() -> None:
    args, train_args = build_parser().parse_known_args()
    train_script = args.train_script.expanduser().resolve()
    if not train_script.is_file():
        raise FileNotFoundError(f"LingBot training script does not exist: {train_script}")

    package_version, codebase_version = enable_lerobot_v21_layout()
    install_lingbot_v2_action_padding_mask()
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "LingBot RoboCasa compatibility: "
            f"lerobot={package_version}, layout={codebase_version}, "
            "LINGBOT_DATASET_API=v2, action_padding_mask=enabled",
            flush=True,
        )

    sys.argv = [str(train_script), *train_args]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
