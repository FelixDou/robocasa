"""Run and resume the pinned official SAFE pi0 MLP/LSTM hyperparameter grid."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

from .evaluate_official_safe import OFFICIAL_SAFE_COMMIT, SELECTION_METRIC, write_json


OFFICIAL_SELECTORS = ("0.0", "1.0", "concat-2")
OFFICIAL_LEARNING_RATES = ("1e-5", "3e-5", "1e-4", "3e-4", "1e-3")
OFFICIAL_REGULARIZATION = ("1e-3", "1e-2", "1e-1")
OFFICIAL_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class GridRun:
    model: str
    horizon_selector: str
    diffusion_selector: str
    learning_rate: str
    lambda_reg: str
    seed: int

    @property
    def slug(self):
        def clean(value):
            return str(value).replace(".", "p").replace("-", "m")

        return "__".join(
            (
                self.model,
                f"h-{clean(self.horizon_selector)}",
                f"d-{clean(self.diffusion_selector)}",
                f"lr-{clean(self.learning_rate)}",
                f"reg-{clean(self.lambda_reg)}",
                f"seed-{self.seed}",
            )
        )


def generate_grid(
    models=("indep", "lstm"),
    horizon_selectors=OFFICIAL_SELECTORS,
    diffusion_selectors=OFFICIAL_SELECTORS,
    learning_rates=OFFICIAL_LEARNING_RATES,
    regularization=OFFICIAL_REGULARIZATION,
    seeds=OFFICIAL_SEEDS,
):
    return [
        GridRun(*values)
        for values in itertools.product(
            models,
            horizon_selectors,
            diffusion_selectors,
            learning_rates,
            regularization,
            seeds,
        )
    ]


def train_command(run, args, run_root):
    artifacts = run_root / "artifacts"
    return [
        sys.executable,
        "-u",
        "-m",
        "failure_prob.train",
        "dataset=pizero",
        f"dataset.data_path={Path(args.export_dir).resolve()}",
        f"dataset.horizon_idx_rel={run.horizon_selector}",
        f"dataset.diff_idx_rel={run.diffusion_selector}",
        f"model={run.model}",
        f"model.lr={run.learning_rate}",
        f"model.lambda_reg={run.lambda_reg}",
        f"model.n_epochs={args.epochs}",
        f"train.seed={run.seed}",
        "train.eval_save_logs=false",
        "train.eval_save_ckpt=true",
        f"train.logs_save_path={artifacts}",
        f"train.wandb_dir={run_root / 'wandb'}",
        f"hydra.run.dir={run_root / 'hydra'}",
    ]


def evaluation_command(run, args, run_root):
    artifacts = run_root / "artifacts"
    return [
        sys.executable,
        "-u",
        "-m",
        "robocasa.recovery.safe.evaluate_official_safe",
        "--export-dir",
        str(Path(args.export_dir).resolve()),
        "--safe-repo",
        str(Path(args.safe_repo).resolve()),
        "--checkpoint",
        str(artifacts / "model_final.ckpt"),
        "--config",
        str(artifacts / "config.yaml"),
        "--output-dir",
        str(run_root / "evaluation"),
        "--device",
        args.device,
    ]


def _run_logged(command, *, cwd, env, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        stream.write("COMMAND: " + " ".join(str(x) for x in command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def _append_event(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def validate_export_gate(export_dir):
    export_dir = Path(export_dir).resolve()
    validation_path = export_dir / "official_loader_validation.json"
    if not validation_path.is_file():
        raise ValueError(
            f"Missing official loader gate: {validation_path}. "
            "Run validate_official_export before the grid."
        )
    validation = json.loads(validation_path.read_text())
    if not validation.get("valid") or not validation.get(
        "official_safe_loader_compatible"
    ):
        raise ValueError(f"Official loader validation did not pass: {validation_path}")
    if int(validation.get("num_rollouts", -1)) != 100:
        raise ValueError("Official grid requires the validated 100-rollout balanced export")
    if (int(validation.get("successes", -1)), int(validation.get("failures", -1))) != (
        50,
        50,
    ):
        raise ValueError("Official grid requires the validated 50-success/50-failure export")
    return validation_path


def run_grid(args):
    safe_repo = Path(args.safe_repo).resolve()
    robocasa_repo = Path(args.robocasa_repo).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    validation_path = validate_export_gate(args.export_dir)
    commit = subprocess.run(
        ["git", "-C", str(safe_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_SAFE_COMMIT:
        raise ValueError(f"SAFE checkout is at {commit}, expected {OFFICIAL_SAFE_COMMIT}")

    complete_grid = generate_grid(
        models=args.models,
        horizon_selectors=args.horizon_selectors,
        diffusion_selectors=args.diffusion_selectors,
        learning_rates=args.learning_rates,
        regularization=args.regularization,
        seeds=args.seeds,
    )
    assigned = [
        run
        for index, run in enumerate(complete_grid)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_runs is not None:
        assigned = assigned[: args.max_runs]
    manifest = {
        "schema_version": 1,
        "official_safe_commit": OFFICIAL_SAFE_COMMIT,
        "official_full_grid_size": 810,
        "configured_grid_size": len(complete_grid),
        "assigned_runs": len(assigned),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "epochs": args.epochs,
        "export_dir": str(Path(args.export_dir).resolve()),
        "official_loader_validation": str(validation_path),
        "selection_metric": SELECTION_METRIC,
        "runs": [asdict(run) | {"slug": run.slug} for run in assigned],
    }
    write_json(output_root / f"grid_plan_shard_{args.shard_index:03d}.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest

    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "WANDB_ENABLED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    (
                        str(robocasa_repo),
                        str(safe_repo),
                        env.get("PYTHONPATH"),
                    ),
                )
            ),
        }
    )
    events_path = output_root / f"grid_events_shard_{args.shard_index:03d}.jsonl"
    for position, run in enumerate(assigned, start=1):
        run_root = output_root / run.slug
        checkpoint = run_root / "artifacts" / "model_final.ckpt"
        config = run_root / "artifacts" / "config.yaml"
        metrics = run_root / "evaluation" / "metrics.json"
        failure = run_root / "failure.json"
        if args.resume and checkpoint.is_file() and config.is_file() and metrics.is_file():
            print(f"[{position}/{len(assigned)}] already complete: {run.slug}")
            continue
        if args.resume and failure.is_file() and not args.retry_errors:
            print(
                f"[{position}/{len(assigned)}] known invalid run (use --retry-errors): "
                f"{run.slug}"
            )
            continue
        run_root.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        _append_event(
            events_path,
            {"status": "started", "run": asdict(run), "slug": run.slug, "at": started},
        )
        print(f"[{position}/{len(assigned)}] training {run.slug}", flush=True)
        stage = "training"
        try:
            if not (args.resume and checkpoint.is_file() and config.is_file()):
                _run_logged(
                    train_command(run, args, run_root),
                    cwd=safe_repo,
                    env=env,
                    log_path=run_root / "train.log",
                )
            stage = "evaluation"
            _run_logged(
                evaluation_command(run, args, run_root),
                cwd=robocasa_repo,
                env=env,
                log_path=run_root / "evaluate.log",
            )
        except subprocess.CalledProcessError as error:
            failure_record = {
                "schema_version": 1,
                "status": "error",
                "stage": stage,
                "run": asdict(run),
                "slug": run.slug,
                "returncode": error.returncode,
                "train_log": str(run_root / "train.log"),
                "evaluate_log": str(run_root / "evaluate.log"),
                "at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(failure, failure_record)
            _append_event(
                events_path,
                failure_record,
            )
            print(
                f"[{position}/{len(assigned)}] ERROR during {stage}: {run.slug}; "
                f"continuing (see {failure})",
                flush=True,
            )
            if args.fail_fast:
                raise
            continue
        failure.unlink(missing_ok=True)
        _append_event(
            events_path,
            {
                "status": "complete",
                "run": asdict(run),
                "slug": run.slug,
                "at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--safe-repo", required=True)
    parser.add_argument("--robocasa-repo", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--models", nargs="+", choices=["indep", "lstm"], default=["indep", "lstm"])
    parser.add_argument("--horizon-selectors", nargs="+", default=list(OFFICIAL_SELECTORS))
    parser.add_argument("--diffusion-selectors", nargs="+", default=list(OFFICIAL_SELECTORS))
    parser.add_argument("--learning-rates", nargs="+", default=list(OFFICIAL_LEARNING_RATES))
    parser.add_argument("--regularization", nargs="+", default=list(OFFICIAL_REGULARIZATION))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(OFFICIAL_SEEDS))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry runs with a saved failure.json; otherwise --resume skips them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the shard at the first failed run instead of recording and continuing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    run_grid(args)


if __name__ == "__main__":
    main()
