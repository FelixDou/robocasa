"""Subtask-level evaluation helpers for recovery-oriented RoboCasa runs."""


DEFAULT_STUCK_PATIENCE = 10


def get_subtask_eval(env):
    """
    Return the subtask progress payload for an environment when available.

    This helper intentionally treats subtask evaluation as an optional layer so
    existing binary task success evaluators can continue to run unchanged.
    """
    if hasattr(env, "get_subtask_progress"):
        return env.get_subtask_progress()

    inner_env = getattr(env, "env", None)
    if inner_env is not None and hasattr(inner_env, "get_subtask_progress"):
        return inner_env.get_subtask_progress()

    return None


def _predicate_value(subtask_eval, predicate_name):
    predicate = subtask_eval.get("predicates", {}).get(predicate_name, {})
    return bool(predicate.get("value", False))


def _current_subtask_estimate(subtask_eval, completed_required=None):
    completed_required = set(completed_required or [])
    for name in subtask_eval.get("required_predicates", []):
        if name not in completed_required:
            return name
    return None


def _failed_preconditions(subtask_eval):
    failed = []
    for name, predicate in subtask_eval.get("predicates", {}).items():
        if predicate.get("stage") == "precondition" and not predicate.get(
            "value", False
        ):
            failed.append(name)
    return failed


def _failure_modes(final_subtask_eval, max_subtask_progress, failed_required_names=None):
    modes = []
    failed_names = set(
        failed_required_names
        if failed_required_names is not None
        else final_subtask_eval.get("failed_required_predicates", [])
    )
    failed_preconditions = _failed_preconditions(final_subtask_eval)

    if failed_preconditions:
        modes.append("failed_precondition")

    if max_subtask_progress == 0.0 and failed_names:
        modes.append("no_progress")

    if any("grasped" in name for name in failed_names):
        modes.append("wrong_object")

    if any(
        token in name
        for name in failed_names
        for token in ("on_target", "in_target", "in_glass", "in_cup", "in_bowl")
    ):
        modes.append("wrong_receptacle")

    return sorted(set(modes))


def _trace_failure_modes(trace_entry, ever_completed):
    modes = []
    failed_preconditions = trace_entry.get("failed_preconditions", [])
    current_subtask = trace_entry.get("current_subtask_estimate")

    if failed_preconditions:
        modes.append("failed_precondition")

    if not ever_completed and current_subtask is not None:
        modes.append("no_progress")

    if current_subtask is not None and any(
        name.endswith("_grasped") for name in ever_completed
    ):
        modes.append("wrong_object")

    if current_subtask is not None and any(
        token in current_subtask
        for token in ("on_target", "in_target", "in_glass", "in_cup", "in_bowl")
    ):
        modes.append("wrong_receptacle")

    return sorted(set(modes))


def build_subtask_trace(subtask_evals):
    """Build stepwise progress diagnostics from subtask-evaluation payloads."""
    trace = []
    previous_required_values = {}
    completed_ever = set()
    ordered_required_names = []
    ordered_completed = []
    ordered_index = 0

    for step_i, subtask_eval in enumerate(subtask_evals):
        if not subtask_eval:
            trace.append(
                {
                    "step": step_i,
                    "subtask_eval_available": False,
                    "subtask_progress": 0.0,
                    "completed_subtask_estimate": [],
                    "current_subtask_estimate": None,
                    "newly_completed_predicates": [],
                    "ordered_completed_subtasks": [],
                    "ordered_current_subtask": None,
                    "ordered_newly_completed_subtasks": [],
                    "ordered_subtask_progress": 0.0,
                    "regressed_predicates": [],
                    "failed_preconditions": [],
                }
            )
            continue

        required_names = subtask_eval.get("required_predicates", [])
        if not ordered_required_names:
            ordered_required_names = list(required_names)
        required_values = {
            name: _predicate_value(subtask_eval, name) for name in required_names
        }
        completed = {name for name, value in required_values.items() if value}
        ordered_newly_completed = []
        while ordered_index < len(ordered_required_names):
            current_name = ordered_required_names[ordered_index]
            if not required_values.get(current_name, False):
                break
            ordered_completed.append(current_name)
            ordered_newly_completed.append(current_name)
            ordered_index += 1

        if subtask_eval.get("task_success", False):
            while ordered_index < len(ordered_required_names):
                current_name = ordered_required_names[ordered_index]
                ordered_completed.append(current_name)
                ordered_newly_completed.append(current_name)
                ordered_index += 1

        ordered_current_subtask = (
            ordered_required_names[ordered_index]
            if ordered_index < len(ordered_required_names)
            else None
        )
        ordered_subtask_progress = (
            float(len(ordered_completed) / len(ordered_required_names))
            if ordered_required_names
            else float(subtask_eval.get("task_success", False))
        )
        regressed = {
            name
            for name, value in required_values.items()
            if previous_required_values.get(name, False) and not value
        }
        completed_ever.update(completed)
        for name, predicate in subtask_eval.get("predicates", {}).items():
            if predicate.get("value", False):
                completed_ever.add(name)

        trace_entry = {
            "step": step_i,
            "subtask_eval_available": True,
            "subtask_progress": ordered_subtask_progress,
            "instantaneous_subtask_progress": subtask_eval.get("subtask_progress", 0.0),
            "completed_subtask_estimate": list(ordered_completed),
            "instantaneous_completed_subtask_estimate": sorted(completed),
            "current_subtask_estimate": ordered_current_subtask,
            "instantaneous_current_subtask_estimate": _current_subtask_estimate(
                subtask_eval, completed_required=completed
            ),
            "newly_completed_predicates": list(ordered_newly_completed),
            "ordered_completed_subtasks": list(ordered_completed),
            "ordered_current_subtask": ordered_current_subtask,
            "ordered_newly_completed_subtasks": list(ordered_newly_completed),
            "ordered_subtask_progress": ordered_subtask_progress,
            "regressed_predicates": sorted(regressed),
            "failed_preconditions": _failed_preconditions(subtask_eval),
        }
        trace_entry["failure_modes"] = _trace_failure_modes(trace_entry, completed_ever)
        trace.append(trace_entry)

        previous_required_values = required_values

    return trace


def infer_stuck_subtask(trace, stuck_patience=DEFAULT_STUCK_PATIENCE):
    """Infer the active subtask if progress has not changed recently."""
    valid_trace = [entry for entry in trace if entry.get("subtask_eval_available")]
    if not valid_trace:
        return None

    latest = valid_trace[-1]
    current_subtask = latest.get("current_subtask_estimate")
    if current_subtask is None:
        return None

    recent_trace = valid_trace[-stuck_patience:]
    made_progress_recently = any(
        entry.get("newly_completed_predicates") for entry in recent_trace
    )
    if made_progress_recently:
        return None

    return current_subtask


def summarize_subtask_rollout(
    subtask_evals, stuck_patience=DEFAULT_STUCK_PATIENCE, include_trace=True
):
    """Summarize a sequence of subtask-evaluation payloads from one rollout."""
    valid_evals = [subtask_eval for subtask_eval in subtask_evals if subtask_eval]
    if not valid_evals:
        return {
            "final_subtask_eval": None,
            "subtask_trace": build_subtask_trace(subtask_evals)
            if include_trace
            else [],
            "max_subtask_progress": 0.0,
            "completed_predicates_ever": [],
            "completed_subtask_estimate": [],
            "current_subtask_estimate": None,
            "stuck_subtask": None,
            "failed_preconditions": [],
            "failed_preconditions_ever": [],
            "failure_modes": ["subtask_eval_unavailable"],
            "failed_required_predicates_final": [],
            "task_success": False,
        }

    trace = build_subtask_trace(subtask_evals)
    completed_predicates_ever = set()
    max_subtask_progress = 0.0
    for entry in trace:
        if entry.get("subtask_eval_available"):
            max_subtask_progress = max(
                max_subtask_progress, entry.get("ordered_subtask_progress", 0.0)
            )
    for subtask_eval in valid_evals:
        for name, predicate in subtask_eval.get("predicates", {}).items():
            if predicate.get("value", False):
                completed_predicates_ever.add(name)

    final_subtask_eval = valid_evals[-1]
    final_trace = next(
        entry for entry in reversed(trace) if entry.get("subtask_eval_available")
    )
    required_names = final_subtask_eval.get("required_predicates", [])
    task_success = final_subtask_eval.get("task_success", False)
    ordered_completed = final_trace.get("ordered_completed_subtasks", [])
    if task_success:
        ordered_completed = list(required_names)
        max_subtask_progress = 1.0
    failed_required_ordered = (
        []
        if task_success
        else [name for name in required_names if name not in set(ordered_completed)]
    )
    failed_preconditions = _failed_preconditions(final_subtask_eval)
    failed_preconditions_ever = sorted(
        {name for entry in trace for name in entry.get("failed_preconditions", [])}
    )
    if task_success:
        failure_modes = set()
    else:
        failure_modes = set(
            _failure_modes(
                final_subtask_eval,
                max_subtask_progress,
                failed_required_names=failed_required_ordered,
            )
        )
        for entry in trace:
            failure_modes.update(entry.get("failure_modes", []))
    return {
        "final_subtask_eval": final_subtask_eval,
        "subtask_trace": trace if include_trace else [],
        "max_subtask_progress": max_subtask_progress,
        "completed_predicates_ever": sorted(completed_predicates_ever),
        "completed_subtask_estimate": ordered_completed,
        "current_subtask_estimate": (
            None if task_success else final_trace["ordered_current_subtask"]
        ),
        "ordered_completed_required_subtasks": ordered_completed,
        "ordered_current_subtask": (
            None if task_success else final_trace["ordered_current_subtask"]
        ),
        "failed_required_subtasks_ordered": failed_required_ordered,
        "stuck_subtask": (
            None
            if task_success
            else infer_stuck_subtask(trace, stuck_patience=stuck_patience)
        ),
        "failed_preconditions": failed_preconditions,
        "failed_preconditions_ever": failed_preconditions_ever,
        "failure_modes": sorted(failure_modes),
        "failed_required_predicates_final": failed_required_ordered,
        "task_success": task_success,
    }
