"""Recovery utilities with lazy imports for simulator-independent tooling."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "build_subtask_trace": ("robocasa.recovery.subtask_eval", "build_subtask_trace"),
    "get_subtask_eval": ("robocasa.recovery.subtask_eval", "get_subtask_eval"),
    "infer_stuck_subtask": ("robocasa.recovery.subtask_eval", "infer_stuck_subtask"),
    "summarize_subtask_rollout": (
        "robocasa.recovery.subtask_eval",
        "summarize_subtask_rollout",
    ),
    "get_eval_composite_subtask_predicates": (
        "robocasa.recovery.eval_composite_predicates",
        "get_eval_composite_subtask_predicates",
    ),
    "RecoveryConfig": ("robocasa.recovery.recovery_rollout", "RecoveryConfig"),
    "RecoveryMode": ("robocasa.recovery.recovery_rollout", "RecoveryMode"),
    "apply_recovery_mode": (
        "robocasa.recovery.recovery_rollout",
        "apply_recovery_mode",
    ),
    "run_recovery_after_failed_rollout": (
        "robocasa.recovery.recovery_rollout",
        "run_recovery_after_failed_rollout",
    ),
    "run_dataset_creation": (
        "robocasa.recovery.create_recovery_failure_dataset",
        "run_dataset_creation",
    ),
    "FullSnapshot": ("robocasa.recovery.full_snapshot", "FullSnapshot"),
    "capture_full_snapshot": (
        "robocasa.recovery.full_snapshot",
        "capture_full_snapshot",
    ),
    "restore_full_snapshot": (
        "robocasa.recovery.full_snapshot",
        "restore_full_snapshot",
    ),
    "BranchSpec": ("robocasa.recovery.counterfactual_branch", "BranchSpec"),
    "run_counterfactual_branch": (
        "robocasa.recovery.counterfactual_branch",
        "run_counterfactual_branch",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
