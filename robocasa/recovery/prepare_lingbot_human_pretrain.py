"""Inventory RoboCasa human pretraining data and write a LingBot multi-dataset list.

The official RoboCasa ``pretrain_human300`` configuration is represented by
``PRETRAINING_TASKS["pretrain300"]``.  LingBot's multi-dataset loader expects
one line per local LeRobot dataset::

    <robot-config-name> <absolute-path-to-lerobot-dataset>

This utility deliberately refuses to produce a complete manifest when any
registered task is absent or structurally incomplete unless
``--allow-incomplete`` is passed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetInventory:
    task: str
    path: str
    exists: bool
    info_exists: bool
    parquet_file_count: int

    @property
    def complete(self) -> bool:
        return self.exists and self.info_exists and self.parquet_file_count > 0


def inspect_dataset(task: str, path: Path) -> DatasetInventory:
    path = path.expanduser().resolve()
    exists = path.is_dir()
    info_exists = (path / "meta" / "info.json").is_file() if exists else False
    parquet_count = len(list(path.glob("data/*/*.parquet"))) if exists else 0
    return DatasetInventory(
        task=task,
        path=str(path),
        exists=exists,
        info_exists=info_exists,
        parquet_file_count=parquet_count,
    )


def write_outputs(
    inventories: list[DatasetInventory],
    manifest_path: Path,
    report_path: Path,
    robot_config_name: str,
    task_set: str,
    dataset_base: Path,
    allow_incomplete: bool,
) -> None:
    complete = [item for item in inventories if item.complete]
    incomplete = [item for item in inventories if not item.complete]
    report = {
        "dataset_config": f"pretrain_human{task_set.removeprefix('pretrain')}",
        "task_set": task_set,
        "source": "human",
        "dataset_base": str(dataset_base.expanduser().resolve()),
        "robot_config_name": robot_config_name,
        "expected_task_count": len(inventories),
        "complete_task_count": len(complete),
        "incomplete_task_count": len(incomplete),
        "parquet_file_count": sum(item.parquet_file_count for item in complete),
        "complete": not incomplete,
        "incomplete_tasks": [item.task for item in incomplete],
        "datasets": [asdict(item) | {"complete": item.complete} for item in inventories],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    if incomplete and not allow_incomplete:
        raise RuntimeError(
            f"{len(incomplete)}/{len(inventories)} datasets are incomplete; "
            f"see {report_path}. The LingBot manifest was not written."
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(f"{robot_config_name} {item.path}\n" for item in complete)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the LingBot manifest for RoboCasa human pretraining."
    )
    parser.add_argument("--dataset-base", type=Path, required=True)
    parser.add_argument("--task-set", default="pretrain300")
    parser.add_argument("--robot-config-name", default="robocasa_lerobot")
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    import robocasa.macros as macros
    from robocasa.utils.dataset_registry import PRETRAINING_TASKS
    from robocasa.utils.dataset_registry_utils import get_ds_meta

    if args.task_set not in PRETRAINING_TASKS:
        raise KeyError(
            f"Unknown task set {args.task_set!r}; choices: {sorted(PRETRAINING_TASKS)}"
        )

    macros.DATASET_BASE_PATH = str(args.dataset_base.expanduser().resolve())
    tasks = list(PRETRAINING_TASKS[args.task_set])
    if len(tasks) != len(set(tasks)):
        raise RuntimeError(f"Task set {args.task_set} contains duplicate task names")

    inventories = []
    for task in tasks:
        meta = get_ds_meta(task=task, split="pretrain", source="human")
        if meta is None:
            inventories.append(
                inspect_dataset(task, args.dataset_base / "__unregistered__" / task)
            )
        else:
            inventories.append(inspect_dataset(task, Path(meta["path"])))

    try:
        write_outputs(
            inventories=inventories,
            manifest_path=args.output_manifest,
            report_path=args.output_report,
            robot_config_name=args.robot_config_name,
            task_set=args.task_set,
            dataset_base=args.dataset_base,
            allow_incomplete=args.allow_incomplete,
        )
    finally:
        complete = sum(item.complete for item in inventories)
        print(f"expected_tasks={len(inventories)}")
        print(f"complete_tasks={complete}")
        print(f"incomplete_tasks={len(inventories) - complete}")
        print(f"parquet_files={sum(item.parquet_file_count for item in inventories)}")
        print(f"report={args.output_report}")
        if complete == len(inventories) or args.allow_incomplete:
            print(f"manifest={args.output_manifest}")


if __name__ == "__main__":
    main()
