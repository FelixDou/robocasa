"""Simulator-free discovery and validation of registered RoboCasa atomic tasks."""

from __future__ import annotations

import ast
from pathlib import Path


def dataset_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "utils" / "dataset_registry.py"


def registered_atomic_tasks(path: str | Path | None = None) -> set[str]:
    """Parse `ATOMIC_TASK_DATASETS` without importing RoboSuite or RoboCasa."""
    path = Path(path) if path is not None else dataset_registry_path()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "ATOMIC_TASK_DATASETS" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            break
        tasks = {keyword.arg for keyword in node.value.keywords if keyword.arg is not None}
        if tasks:
            return tasks
    raise RuntimeError(f"Could not parse ATOMIC_TASK_DATASETS from {path}")


def validate_atomic_tasks(tasks, *, allow_unregistered=False, registry_path=None):
    tasks = list(tasks)
    if not tasks:
        raise ValueError("At least one atomic task is required")
    if len(tasks) != len(set(tasks)):
        raise ValueError("Atomic task list contains duplicates")
    if allow_unregistered:
        return tasks
    registered = registered_atomic_tasks(registry_path)
    unknown = sorted(set(tasks) - registered)
    if unknown:
        raise ValueError(
            "Tasks are not registered RoboCasa atomic tasks: " + ", ".join(unknown)
        )
    return tasks
