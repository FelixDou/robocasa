"""Create a 4-GPU LingBot expert-only config for RoboCasa human300.

The released LingBot-VLA-v2 checkpoint is native-depth.  We therefore preserve
the upstream RoboTwin architecture and dual-query distillation section, while
overriding only dataset paths, checkpoint paths, and training controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def configure(base: dict, args: argparse.Namespace) -> dict:
    model = base["model"]
    data = base["data"]
    train = base["train"]

    model["model_path"] = str(args.model_path)
    model["tokenizer_path"] = str(args.tokenizer_path)
    model["post_training"] = True

    data.update(
        {
            "datasets_type": "vla",
            "data_name": "multi",
            "train_path": str(args.train_manifest),
            "robot_config_root": str(args.robot_config_root),
            "joints": [
                {"end.position": 14},
                {"effector.position": 2},
                {"waist.position": 4},
                {"base.position": 3},
            ],
            "cameras": [
                "camera_top",
                "camera_wrist_left",
                "camera_wrist_right",
            ],
            "prompt_type": "global",
            "norm_type": [
                {"end.position": "meanstd"},
                {"effector.position": "meanstd"},
                {"waist.position": "meanstd"},
                {"base.position": "meanstd"},
            ],
            "norm_stats_file": str(args.norm_stats_file),
            "num_workers": args.num_workers,
            "use_future_image": True,
        }
    )

    train.update(
        {
            "output_dir": str(args.output_dir),
            "train_expert_only": True,
            "freeze_vision_encoder": True,
            "use_wandb": False,
            "micro_batch_size": 1,
            "global_batch_size": args.global_batch_size,
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "lr": args.learning_rate,
            "enable_resume": True,
        }
    )

    align = train.get("align_params")
    if not isinstance(align, dict):
        raise KeyError(
            "Base config must contain train.align_params for the native-depth checkpoint"
        )
    align["depth"]["moge_path"] = str(args.moge_path)
    align["depth"]["morgbd_path"] = str(args.depth_path)
    align["video"]["ckpt_path"] = str(args.dino_checkpoint)
    align["video"]["config_path"] = str(args.dino_config)
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--robot-config-root", type=Path, required=True)
    parser.add_argument("--norm-stats-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--moge-path", type=Path, required=True)
    parser.add_argument("--depth-path", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-config", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to generate the training config") from exc

    base = yaml.safe_load(args.base_config.read_text())
    config = configure(base, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"config={args.output}")
    print("train_expert_only=true")
    print(f"global_batch_size={args.global_batch_size}")
    print(f"max_steps={args.max_steps}")


if __name__ == "__main__":
    main()
