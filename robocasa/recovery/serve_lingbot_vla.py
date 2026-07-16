"""Launch LingBot-VLA-v2 with an explicit RoboCasa normalization file.

Run this script with the LingBot conda environment.  Imports are deliberately
deferred so ``--help`` also works from the lighter RoboCasa environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Serve a LingBot-VLA-v2 checkpoint for zero-shot RoboCasa evaluation."
    )
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--robot-norm-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9330)
    parser.add_argument("--use-length", type=int, default=8)
    parser.add_argument("--use-compile", type=str_to_bool, default=False)
    parser.add_argument("--use-bf16", type=str_to_bool, default=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    lingbot_repo = args.lingbot_repo.resolve()
    model_path = args.model_path.resolve()
    robot_norm_path = args.robot_norm_path.resolve()
    for path, label in (
        (lingbot_repo, "LingBot repository"),
        (model_path, "model path"),
        (robot_norm_path, "normalization file"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    sys.path.insert(0, str(lingbot_repo))
    os.chdir(lingbot_repo)
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from deploy.websocket_policy_server import WebsocketPolicyServer

    model = LingbotVLAv2Server(
        path_to_pi_model=str(model_path),
        robot_norm_path=str(robot_norm_path),
        use_length=int(args.use_length),
        chunk_ret=True,
        use_bf16=bool(args.use_bf16),
        use_fp32=not bool(args.use_bf16),
        use_compile=bool(args.use_compile),
    )
    WebsocketPolicyServer(model, port=int(args.port)).serve_forever()


if __name__ == "__main__":
    main()
