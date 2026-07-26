"""Simulator-free discovery of registered RoboCasa SAFE collection tasks."""

from __future__ import annotations

import ast
from pathlib import Path


def dataset_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "utils" / "dataset_registry.py"


def _registry_call(registry_name: str, path: str | Path | None = None) -> ast.Call:
    path = Path(path) if path is not None else dataset_registry_path()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == registry_name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call):
            break
        return node.value
    raise RuntimeError(f"Could not parse {registry_name} from {path}")


def _registered_tasks(registry_name: str, path=None) -> set[str]:
    registry = _registry_call(registry_name, path)
    tasks = {
        keyword.arg for keyword in registry.keywords if keyword.arg is not None
    }
    if not tasks:
        raise RuntimeError(f"{registry_name} contains no named tasks")
    return tasks


def _registered_horizons(registry_name: str, path=None) -> dict[str, int]:
    registry = _registry_call(registry_name, path)
    source_path = Path(path) if path is not None else dataset_registry_path()
    horizons = {}
    for task_keyword in registry.keywords:
        if task_keyword.arg is None or not isinstance(task_keyword.value, ast.Call):
            continue
        for config_keyword in task_keyword.value.keywords:
            if config_keyword.arg != "horizon":
                continue
            value = ast.literal_eval(config_keyword.value)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RuntimeError(
                    f"Invalid horizon for {task_keyword.arg} in "
                    f"{source_path}: {value!r}"
                )
            horizons[task_keyword.arg] = value
            break
    if not horizons:
        raise RuntimeError(f"Could not parse {registry_name} horizons from {source_path}")
    return horizons


def registered_atomic_tasks(path: str | Path | None = None) -> set[str]:
    """Parse `ATOMIC_TASK_DATASETS` without importing RoboSuite or RoboCasa."""
    return _registered_tasks("ATOMIC_TASK_DATASETS", path)


def registered_composite_tasks(path: str | Path | None = None) -> set[str]:
    """Parse `COMPOSITE_TASK_DATASETS` without importing RoboSuite or RoboCasa."""
    return _registered_tasks("COMPOSITE_TASK_DATASETS", path)


def registered_safe_tasks(path: str | Path | None = None) -> set[str]:
    """Return every registered atomic or composite RoboCasa task."""
    return registered_atomic_tasks(path) | registered_composite_tasks(path)


def registered_atomic_task_horizons(
    path: str | Path | None = None,
) -> dict[str, int]:
    """Parse official atomic horizons without importing RoboCasa."""
    return _registered_horizons("ATOMIC_TASK_DATASETS", path)


def registered_composite_task_horizons(
    path: str | Path | None = None,
) -> dict[str, int]:
    """Parse official composite horizons without importing RoboCasa."""
    return _registered_horizons("COMPOSITE_TASK_DATASETS", path)


def registered_safe_task_horizons(
    path: str | Path | None = None,
) -> dict[str, int]:
    """Return official horizons for all registered collection tasks."""
    return {
        **registered_atomic_task_horizons(path),
        **registered_composite_task_horizons(path),
    }


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


def validate_safe_tasks(tasks, *, allow_unregistered=False, registry_path=None):
    """Validate unique atomic or composite task names for SAFE collection."""
    tasks = list(tasks)
    if not tasks:
        raise ValueError("At least one task is required")
    if len(tasks) != len(set(tasks)):
        raise ValueError("Task list contains duplicates")
    if allow_unregistered:
        return tasks
    unknown = sorted(set(tasks) - registered_safe_tasks(registry_path))
    if unknown:
        raise ValueError(
            "Tasks are not registered RoboCasa atomic or composite tasks: "
            + ", ".join(unknown)
        )
    return tasks
