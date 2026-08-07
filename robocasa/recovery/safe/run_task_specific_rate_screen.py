"""Train one rollout-level SAFE model per task and rate regime."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REGIMES = ("matched_weighted", "natural_weighted")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def slugify(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def run_screen(args):
    plan_path = Path(args.experiment_plan).resolve()
    plan = json.loads(plan_path.read_text())
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_tasks = args.tasks or list(plan["tasks"])
    unknown_tasks = sorted(set(selected_tasks) - set(plan["tasks"]))
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    selected_regimes = args.regimes or list(REGIMES)
    unknown_regimes = sorted(set(selected_regimes) - set(plan["regimes"]))
    if unknown_regimes:
        raise ValueError(f"Unknown regimes: {unknown_regimes}")
    selected_subset_seeds = args.subset_seeds or [
        int(value) for value in plan["subset_seeds"]
    ]
    manifests = {
        (item["task_name"], item["regime"], int(item["subset_seed"])): item["path"]
        for item in plan["manifests"]
    }
    runs = []
    for task_name in selected_tasks:
        for regime in selected_regimes:
            for subset_seed in selected_subset_seeds:
                manifest = manifests.get((task_name, regime, int(subset_seed)))
                if manifest is None:
                    raise ValueError(
                        f"No manifest for {task_name}/{regime}/seed-{subset_seed}"
                    )
                for model_seed in args.model_seeds:
                    run_slug = (
                        f"{args.model}__{regime}__subset-{subset_seed}"
                        f"__seed-{model_seed}"
                    )
                    runs.append(
                        {
                            "task_name": task_name,
                            "task_type": plan["task_types"][task_name],
                            "model": args.model,
                            "regime": regime,
                            "subset_seed": int(subset_seed),
                            "model_seed": int(model_seed),
                            "selection_manifest": manifest,
                            "class_weighting": plan["regimes"][regime][
                                "class_weighting"
                            ],
                            "slug": run_slug,
                        }
                    )
    shard_key = hashlib.sha256(
        json.dumps(
            {"model": args.model, "tasks": selected_tasks}, sort_keys=True
        ).encode()
    ).hexdigest()[:10]
    screen_plan = {
        "schema_version": 1,
        "protocol": plan["protocol"],
        "experiment_plan": str(plan_path),
        "model": args.model,
        "tasks": selected_tasks,
        "regimes": selected_regimes,
        "subset_seeds": selected_subset_seeds,
        "model_seeds": args.model_seeds,
        "selection_summary": str(Path(args.selection_summary).resolve()),
        "num_runs": len(runs),
        "runs": runs,
    }
    write_json(
        output_root / f"task_screen_plan_{args.model}_{shard_key}.json",
        screen_plan,
    )
    events_path = output_root / f"task_screen_events_{args.model}_{shard_key}.jsonl"
    environment = dict(os.environ)
    environment.update(
        {
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "WANDB_ENABLED": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    completed = 0
    for index, run in enumerate(runs, start=1):
        run_root = output_root / slugify(run["task_name"]) / run["slug"]
        metrics_path = run_root / "metrics.json"
        failure_path = run_root / "failure.json"
        if args.resume and metrics_path.is_file():
            completed += 1
            print(
                f"[{index}/{len(runs)}] complete: {run['task_name']} / {run['slug']}",
                flush=True,
            )
            continue
        if args.resume and failure_path.is_file() and not args.retry_errors:
            print(
                f"[{index}/{len(runs)}] known failure: {run['task_name']} / "
                f"{run['slug']}",
                flush=True,
            )
            continue
        run_root.mkdir(parents=True, exist_ok=True)
        write_json(
            run_root / "task_screen_run.json",
            {"schema_version": 1, "experiment_plan": str(plan_path), **run},
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "robocasa.recovery.safe.train_seen_tasks",
            "--export-dir",
            str(Path(args.export_dir).resolve()),
            "--safe-repo",
            str(Path(args.safe_repo).resolve()),
            "--output-dir",
            str(run_root),
            "--model",
            args.model,
            "--seed",
            str(run["model_seed"]),
            "--split-seed",
            str(args.split_seed),
            "--selection-manifest",
            str(run["selection_manifest"]),
            "--class-weighting",
            run["class_weighting"],
            "--task-type",
            "all",
            "--tasks",
            run["task_name"],
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--selection-summary",
            str(Path(args.selection_summary).resolve()),
            "--resume",
        ]
        started = utc_now()
        with (run_root / "train.log").open("a") as stream:
            stream.write("COMMAND: " + " ".join(command) + "\n")
            stream.flush()
            result = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
        event = {
            "schema_version": 1,
            "task_name": run["task_name"],
            "slug": run["slug"],
            "returncode": result.returncode,
            "started_at": started,
            "completed_at": utc_now(),
        }
        with events_path.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        if result.returncode:
            write_json(
                failure_path,
                {**event, "status": "error", "train_log": str(run_root / "train.log")},
            )
            print(
                f"[{index}/{len(runs)}] ERROR {run['task_name']} / {run['slug']} "
                f"(exit {result.returncode})",
                flush=True,
            )
            if args.fail_fast:
                raise RuntimeError(f"Training failed: {run['task_name']}/{run['slug']}")
        else:
            failure_path.unlink(missing_ok=True)
            completed += 1
            print(
                f"[{index}/{len(runs)}] complete: {run['task_name']} / {run['slug']}",
                flush=True,
            )
    return {"completed": completed, "planned": len(runs)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-plan", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--safe-repo", required=True)
    parser.add_argument("--selection-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("indep", "lstm"), default="indep")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--regimes", nargs="+", choices=REGIMES)
    parser.add_argument("--subset-seeds", nargs="+", type=int)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run_screen(args)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
