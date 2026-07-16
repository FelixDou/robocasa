"""Load an exported RoboCasa dataset with the pinned official SAFE pi0 loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


OFFICIAL_SAFE_COMMIT = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"


def parse_selector(value: str):
    """Match Hydra's float-or-string selector values used by official SAFE."""
    try:
        selector = float(value)
    except ValueError:
        return value
    if not 0.0 <= selector <= 1.0:
        raise ValueError(f"Relative selector must be in [0, 1], got {value}")
    return selector


def make_official_config(export_dir, horizon_selector, diffusion_selector):
    """Construct only the config fields read by failure_prob.data.pizero."""
    return SimpleNamespace(
        dataset=SimpleNamespace(
            data_path=str(Path(export_dir).resolve()),
            data_path_unseen=None,
            feat_name="pre_velocity",
            horizon_idx_rel=parse_selector(horizon_selector),
            diff_idx_rel=parse_selector(diffusion_selector),
        ),
        train=SimpleNamespace(log_precomputed=False, log_precomputed_only=False),
    )


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def validate_loaded_rollouts(
    rollouts,
    *,
    expected_rollouts=None,
    expected_successes=None,
    expected_failures=None,
    expected_tasks=(),
    task_id_to_name=None,
):
    errors = []
    task_counts = {}
    feature_shapes = set()
    action_shapes = set()
    successes = 0
    task_id_to_name = task_id_to_name or {}

    for index, rollout in enumerate(rollouts):
        task_id = int(rollout.task_id)
        task = task_id_to_name.get(task_id, str(rollout.task_description))
        task_counts.setdefault(task, {"successes": 0, "failures": 0})
        success = bool(rollout.episode_success)
        successes += int(success)
        task_counts[task]["successes" if success else "failures"] += 1

        features = _to_numpy(rollout.hidden_states)
        actions = _to_numpy(rollout.action_vectors)
        if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
            errors.append(
                f"rollout {index}: invalid loaded feature shape {features.shape}"
            )
        elif not np.isfinite(features).all():
            errors.append(f"rollout {index}: loaded features contain non-finite values")
        if actions.ndim != 2 or actions.shape[0] != features.shape[0]:
            errors.append(
                f"rollout {index}: loaded action shape {actions.shape} is incompatible "
                f"with feature shape {features.shape}"
            )
        elif not np.isfinite(actions).all():
            errors.append(f"rollout {index}: loaded actions contain non-finite values")
        feature_shapes.add(tuple(int(x) for x in features.shape[1:]))
        action_shapes.add(tuple(int(x) for x in actions.shape[1:]))

    failures = len(rollouts) - successes
    checks = (
        (expected_rollouts, len(rollouts), "rollouts"),
        (expected_successes, successes, "successes"),
        (expected_failures, failures, "failures"),
    )
    for expected, actual, label in checks:
        if expected is not None and actual != expected:
            errors.append(f"expected {expected} {label}, loaded {actual}")

    if expected_tasks:
        expected = set(expected_tasks)
        actual = set(task_counts)
        if actual != expected:
            errors.append(
                f"task mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )

    return {
        "valid": not errors,
        "official_safe_loader_compatible": not errors,
        "num_rollouts": len(rollouts),
        "successes": successes,
        "failures": failures,
        "task_counts": dict(sorted(task_counts.items())),
        "loaded_feature_dimensions": [list(x) for x in sorted(feature_shapes)],
        "loaded_action_dimensions": [list(x) for x in sorted(action_shapes)],
        "errors": errors,
    }


def validate_official_export(
    export_dir,
    *,
    safe_repo=None,
    horizon_selector="0.0",
    diffusion_selector="0.0",
    expected_rollouts=None,
    expected_successes=None,
    expected_failures=None,
    expected_tasks=(),
    expected_safe_commit=OFFICIAL_SAFE_COMMIT,
):
    export_dir = Path(export_dir).resolve()
    safe_commit = None
    if safe_repo is not None:
        safe_repo = Path(safe_repo).resolve()
        if not (safe_repo / "failure_prob" / "data" / "pizero.py").is_file():
            raise ValueError(f"Not an official SAFE checkout: {safe_repo}")
        if expected_safe_commit is not None:
            try:
                safe_commit = subprocess.run(
                    ["git", "-C", str(safe_repo), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(
                    f"Cannot identify SAFE checkout commit: {safe_repo}"
                ) from error
            if safe_commit != expected_safe_commit:
                raise ValueError(
                    f"SAFE checkout is at {safe_commit}, "
                    f"expected {expected_safe_commit}"
                )
        sys.path.insert(0, str(safe_repo))
    try:
        from failure_prob.data.pizero import load_rollouts_from_root
    except ImportError as error:
        raise RuntimeError(
            "Cannot import the official SAFE pi0 loader. Activate the SAFE environment "
            "and install the pinned SAFE checkout, or pass --safe-repo."
        ) from error

    cfg = make_official_config(
        export_dir,
        horizon_selector=horizon_selector,
        diffusion_selector=diffusion_selector,
    )
    rollouts = load_rollouts_from_root(export_dir, cfg)
    report_path = export_dir / "conversion_report.json"
    task_id_to_name = {}
    if report_path.is_file():
        conversion = json.loads(report_path.read_text())
        task_id_to_name = {
            int(task_id): task_name
            for task_name, task_id in conversion.get("task_ids", {}).items()
        }
    report = validate_loaded_rollouts(
        rollouts,
        expected_rollouts=expected_rollouts,
        expected_successes=expected_successes,
        expected_failures=expected_failures,
        expected_tasks=expected_tasks,
        task_id_to_name=task_id_to_name,
    )
    report.update(
        {
            "export_dir": str(export_dir),
            "safe_repo": str(safe_repo) if safe_repo is not None else None,
            "safe_repository_commit": safe_commit,
            "horizon_selector": horizon_selector,
            "diffusion_selector": diffusion_selector,
            "official_loader_dataset_fields": {
                "dim_features": getattr(cfg.dataset, "dim_features", None),
                "dim_action": getattr(cfg.dataset, "dim_action", None),
                "pred_horizon": getattr(cfg.dataset, "pred_horizon", None),
                "exec_horizon": getattr(cfg.dataset, "exec_horizon", None),
            },
        }
    )
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--safe-repo")
    parser.add_argument("--expected-safe-commit", default=OFFICIAL_SAFE_COMMIT)
    parser.add_argument("--horizon-selector", default="0.0")
    parser.add_argument("--diffusion-selector", default="0.0")
    parser.add_argument("--expected-rollouts", type=int)
    parser.add_argument("--expected-successes", type=int)
    parser.add_argument("--expected-failures", type=int)
    parser.add_argument("--expected-task", action="append", default=[])
    parser.add_argument("--json-output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = validate_official_export(
            args.export_dir,
            safe_repo=args.safe_repo,
            horizon_selector=args.horizon_selector,
            diffusion_selector=args.diffusion_selector,
            expected_rollouts=args.expected_rollouts,
            expected_successes=args.expected_successes,
            expected_failures=args.expected_failures,
            expected_tasks=args.expected_task,
            expected_safe_commit=args.expected_safe_commit,
        )
    except (AssertionError, IndexError, KeyError, RuntimeError, ValueError) as error:
        report = {
            "valid": False,
            "official_safe_loader_compatible": False,
            "export_dir": str(Path(args.export_dir).resolve()),
            "errors": [f"{type(error).__name__}: {error}"],
        }
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
