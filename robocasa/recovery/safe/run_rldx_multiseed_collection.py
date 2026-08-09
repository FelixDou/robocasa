"""Run deterministic multi-seed Subtask-SAFE collection on parallel RLDX servers.

The launcher treats one group of parallel servers and clients as an atomic
block. Every block starts fresh explicitly seeded RLDX servers, assigns one
task to each server, validates every completed shard, and stops the servers
before advancing. A cyclic Latin schedule rotates tasks across GPUs between
blocks. Interrupted blocks are never rollout-resumed because skipped rollouts
cannot replay the stateful RLDX sampling RNG; they must be quarantined and
replayed from their declared seeds.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

from .collect_atomic_rollouts import (
    RLDX1_BENCHMARK_COMMIT,
    SUMMARY_NAME,
    atomic_write_json,
    current_robocasa_commit,
)
from .merge_atomic_datasets import merge_atomic_datasets
from .validate_atomic_dataset import validate_atomic_dataset


PROTOCOL = "rldx_observation_context_seeded_block_collection_v1"
PLAN_SCHEMA_VERSION = 1
DEFAULT_TASKS = (
    "LoadDishwasher",
    "PreSoakPan",
    "ScrubCuttingBoard",
    "WashLettuce",
)
DEFAULT_GPUS = (0, 1, 2, 3)
DEFAULT_PORTS = (20100, 20101, 20102, 20103)
DEFAULT_MODEL_PATH = "RLWRLD/RLDX-1-FT-RC365"
DEFAULT_POLICY_NAME = "RLDX-1-FT-RC365"
DEFAULT_FEATURE_MODE = "action_observation_context"
PLAN_NAME = "allocation_plan.json"
STATUS_NAME = "status.json"
BLOCK_STATUS_NAME = "block_status.json"

SEEDED_SERVER_BOOTSTRAP = """\
import os
import random
import runpy
import numpy as np
import torch
seed = int(os.environ["RLDX_SERVER_RNG_SEED"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
runpy.run_path("rldx/eval/run_rldx_server.py", run_name="__main__")
"""


class TerminationRequested(RuntimeError):
    """Raised when the launcher receives SIGINT or SIGTERM."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _require_unique(name: str, values: list[Any]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique values")


def rollout_counts_by_block(total_per_task: int, num_blocks: int) -> list[int]:
    """Allocate an exact per-task target over deterministic complete blocks."""
    total_per_task = int(total_per_task)
    num_blocks = int(num_blocks)
    if total_per_task < 1:
        raise ValueError("total_rollouts_per_task must be positive")
    if num_blocks < 1:
        raise ValueError("num_blocks must be positive")
    if total_per_task < num_blocks:
        raise ValueError(
            "total_rollouts_per_task must be at least num_blocks so every block "
            "contains one or more rollouts per task"
        )
    quotient, remainder = divmod(total_per_task, num_blocks)
    return [quotient + int(index < remainder) for index in range(num_blocks)]


def build_collection_plan(
    *,
    output_root: str | Path,
    log_root: str | Path,
    rldx_repo: str | Path,
    tasks: list[str] | tuple[str, ...] = DEFAULT_TASKS,
    gpus: list[int] | tuple[int, ...] = DEFAULT_GPUS,
    ports: list[int] | tuple[int, ...] = DEFAULT_PORTS,
    total_rollouts_per_task: int = 250,
    num_blocks: int = 12,
    environment_seed_start: int = 32,
    server_seed_start: int = 1000,
    model_path: str = DEFAULT_MODEL_PATH,
    policy_name: str = DEFAULT_POLICY_NAME,
    safe_feature_mode: str = DEFAULT_FEATURE_MODE,
    split: str = "target",
    replan_steps: int = 8,
    record_videos: bool = False,
    video_frame_stride: int = 4,
    robocasa_commit: str,
    rldx_commit: str,
    sim_python: str,
    uv_executable: str = "uv",
) -> dict[str, Any]:
    tasks = [str(task) for task in tasks]
    gpus = [int(gpu) for gpu in gpus]
    ports = [int(port) for port in ports]
    if not tasks:
        raise ValueError("At least one task is required")
    if not (len(tasks) == len(gpus) == len(ports)):
        raise ValueError("tasks, gpus, and ports must have the same length")
    _require_unique("tasks", tasks)
    _require_unique("gpus", gpus)
    _require_unique("ports", ports)
    if any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU indices must be non-negative")
    if any(port < 1 or port > 65535 for port in ports):
        raise ValueError("Ports must be between 1 and 65535")
    if safe_feature_mode != DEFAULT_FEATURE_MODE:
        raise ValueError(
            "This Subtask-SAFE launcher requires action_observation_context"
        )
    if int(environment_seed_start) < 0 or int(server_seed_start) < 0:
        raise ValueError("Seed starts must be non-negative")
    if int(replan_steps) < 1 or int(video_frame_stride) < 1:
        raise ValueError("replan_steps and video_frame_stride must be positive")

    output_root = Path(output_root).resolve()
    log_root = Path(log_root).resolve()
    rldx_repo = Path(rldx_repo).resolve()
    block_counts = rollout_counts_by_block(total_rollouts_per_task, num_blocks)
    num_lanes = len(tasks)
    if int(num_blocks) % num_lanes:
        raise ValueError(
            "num_blocks must be divisible by the number of lanes so every task "
            "runs equally often on every GPU"
        )
    blocks = []
    server_seeds = set()
    environment_seeds = []
    task_gpu_blocks: dict[str, Counter] = defaultdict(Counter)
    task_gpu_rollouts: dict[str, Counter] = defaultdict(Counter)

    for block_index, num_rollouts in enumerate(block_counts):
        environment_seed = int(environment_seed_start) + block_index
        environment_seeds.append(environment_seed)
        block_dir = output_root / "blocks" / f"block_{block_index:03d}"
        block_log_dir = log_root / f"block_{block_index:03d}"
        lanes = []
        for lane_index, (gpu, port) in enumerate(zip(gpus, ports)):
            task = tasks[(lane_index + block_index) % num_lanes]
            server_seed = int(server_seed_start) + block_index * num_lanes + lane_index
            if server_seed in server_seeds:
                raise AssertionError("Generated duplicate RLDX server RNG seed")
            server_seeds.add(server_seed)
            task_gpu_blocks[task][gpu] += 1
            task_gpu_rollouts[task][gpu] += num_rollouts
            lane_name = f"lane_{lane_index}_gpu{gpu}_{task}"
            lanes.append(
                {
                    "lane_index": lane_index,
                    "gpu": gpu,
                    "port": port,
                    "task": task,
                    "task_position": 0,
                    "environment_seed": environment_seed,
                    "server_rng_seed": server_seed,
                    "num_rollouts": num_rollouts,
                    "output_dir": str(block_dir / lane_name),
                    "server_log": str(
                        block_log_dir / f"server_gpu{gpu}_port{port}.log"
                    ),
                    "client_log": str(block_log_dir / f"client_gpu{gpu}_{task}.log"),
                }
            )
        blocks.append(
            {
                "block_index": block_index,
                "environment_seed": environment_seed,
                "num_rollouts_per_task": num_rollouts,
                "block_dir": str(block_dir),
                "block_log_dir": str(block_log_dir),
                "lanes": lanes,
            }
        )

    per_task_rollouts = {
        task: sum(
            lane["num_rollouts"]
            for block in blocks
            for lane in block["lanes"]
            if lane["task"] == task
        )
        for task in tasks
    }
    if set(per_task_rollouts.values()) != {int(total_rollouts_per_task)}:
        raise AssertionError("Cyclic schedule failed the exact per-task target")
    per_task_gpu_rollouts = {
        task: list(counts.values()) for task, counts in task_gpu_rollouts.items()
    }
    if any(max(counts) - min(counts) > 1 for counts in per_task_gpu_rollouts.values()):
        raise AssertionError("Cyclic schedule failed task-to-GPU rollout balance")

    core = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "output_root": str(output_root),
        "log_root": str(log_root),
        "rldx_repo": str(rldx_repo),
        "merged_output_dir": str(output_root / "merged_raw"),
        "tasks": tasks,
        "gpus": gpus,
        "ports": ports,
        "num_lanes": num_lanes,
        "num_blocks": int(num_blocks),
        "total_rollouts_per_task": int(total_rollouts_per_task),
        "total_rollouts": int(total_rollouts_per_task) * len(tasks),
        "rollouts_per_block": block_counts,
        "environment_seed_start": int(environment_seed_start),
        "environment_seeds": environment_seeds,
        "server_seed_start": int(server_seed_start),
        "server_rng_seeds": sorted(server_seeds),
        "model_path": str(model_path),
        "policy_name": str(policy_name),
        "safe_feature_mode": safe_feature_mode,
        "split": str(split),
        "replan_steps": int(replan_steps),
        "record_videos": bool(record_videos),
        "video_frame_stride": int(video_frame_stride),
        "robocasa_commit": str(robocasa_commit),
        "rldx_commit": str(rldx_commit),
        "sim_python": str(sim_python),
        "uv_executable": str(uv_executable),
        "schedule": {
            "kind": "cyclic_latin",
            "one_task_per_server": True,
            "within_server_task_order_confounded": False,
            "natural_outcomes_no_class_quotas": True,
            "task_gpu_blocks": {
                task: {str(gpu): count for gpu, count in sorted(counts.items())}
                for task, counts in sorted(task_gpu_blocks.items())
            },
            "task_gpu_rollouts": {
                task: {str(gpu): count for gpu, count in sorted(counts.items())}
                for task, counts in sorted(task_gpu_rollouts.items())
            },
        },
        "per_task_rollouts": per_task_rollouts,
        "blocks": blocks,
    }
    return {**core, "plan_fingerprint": _fingerprint(core)}


def collection_provenance(
    plan: dict[str, Any], block: dict[str, Any], lane: dict[str, Any]
) -> dict[str, Any]:
    return {
        "protocol": plan["protocol"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "block_index": block["block_index"],
        "lane_index": lane["lane_index"],
        "gpu": lane["gpu"],
        "port": lane["port"],
        "task": lane["task"],
        "task_position": lane["task_position"],
        "environment_seed": lane["environment_seed"],
        "server_rng_seed": lane["server_rng_seed"],
        "num_rollouts": lane["num_rollouts"],
    }


def server_command(plan: dict[str, Any], lane: dict[str, Any]) -> list[str]:
    return [
        plan["uv_executable"],
        "run",
        "python",
        "-u",
        "-c",
        SEEDED_SERVER_BOOTSTRAP,
        "--model-path",
        plan["model_path"],
        "--embodiment-tag",
        "GENERAL_EMBODIMENT",
        "--host",
        "127.0.0.1",
        "--port",
        str(lane["port"]),
        "--use-sim-policy-wrapper",
    ]


def server_environment(plan: dict[str, Any], lane: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(lane["gpu"]),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": str(lane["server_rng_seed"]),
            "RLDX_SERVER_RNG_SEED": str(lane["server_rng_seed"]),
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "WANDB_ENABLED": "0",
        }
    )
    return environment


def collector_command(
    plan: dict[str, Any], block: dict[str, Any], lane: dict[str, Any]
) -> list[str]:
    provenance = collection_provenance(plan, block, lane)
    policy_config = {
        "embodiment_tag": "GENERAL_EMBODIMENT",
        "safe_feature_mode": plan["safe_feature_mode"],
        "collection_provenance": provenance,
    }
    session_id = (
        f"rldx-safe-{plan['plan_fingerprint'][:10]}-"
        f"b{block['block_index']:03d}-l{lane['lane_index']}"
    )
    command = [
        plan["sim_python"],
        "-u",
        "-m",
        "robocasa.recovery.safe.collect_atomic_rollouts",
        "--output-dir",
        lane["output_dir"],
        "--tasks",
        lane["task"],
        "--num-rollouts",
        str(lane["num_rollouts"]),
        "--seed",
        str(lane["environment_seed"]),
        "--seed-protocol",
        "official_rldx",
        "--policy-module",
        "robocasa.recovery.rldx_zmq_policy:make_policy",
        "--policy-arg",
        f"session_id={session_id}",
        "--model-family",
        "rldx1",
        "--safe-feature-mode",
        plan["safe_feature_mode"],
        "--policy-name",
        plan["policy_name"],
        "--checkpoint",
        plan["model_path"],
        "--policy-config",
        json.dumps(policy_config, sort_keys=True, separators=(",", ":")),
        "--host",
        "127.0.0.1",
        "--port",
        str(lane["port"]),
        "--split",
        plan["split"],
        "--replan-steps",
        str(plan["replan_steps"]),
        "--record-safe-features",
        "--record-actions",
        "--record-subtask-trace",
        "--max-errors",
        "1",
        "--no-continue-on-error",
        "--video-frame-stride",
        str(plan["video_frame_stride"]),
        "--rldx-repository-commit",
        plan["rldx_commit"],
        "--robocasa-commit",
        plan["robocasa_commit"],
    ]
    command.append("--record-videos" if plan["record_videos"] else "--no-record-videos")
    return command


def collector_environment(lane: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(lane["gpu"]),
            "MUJOCO_EGL_DEVICE_ID": str(lane["gpu"]),
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "WANDB_ENABLED": "0",
        }
    )
    return environment


def write_or_verify_plan(
    plan: dict[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    root = Path(plan["output_root"])
    path = root / PLAN_NAME
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise ValueError(
                "Existing allocation plan disagrees with the requested protocol"
            )
        if not resume:
            raise FileExistsError(
                f"Collection plan already exists at {path}; pass --resume"
            )
        return existing
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Output root is non-empty but has no {PLAN_NAME}: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, plan)
    return plan


def _status_snapshot(
    plan: dict[str, Any], status: str, *, error: str | None = None
) -> dict[str, Any]:
    completed = []
    failed = []
    for block in plan["blocks"]:
        path = Path(block["block_dir"]) / BLOCK_STATUS_NAME
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete":
            completed.append(block["block_index"])
        elif payload.get("status") == "failed":
            failed.append(block["block_index"])
    pending = sorted(set(range(plan["num_blocks"])) - set(completed))
    value = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol": plan["protocol"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "status": status,
        "updated_at": utc_now(),
        "completed_blocks": completed,
        "failed_blocks": failed,
        "pending_blocks": pending,
        "expected_rollouts": plan["total_rollouts"],
        "merged_output_dir": plan["merged_output_dir"],
    }
    if error is not None:
        value["error"] = str(error)
    return value


def write_status(
    plan: dict[str, Any], status: str, *, error: str | None = None
) -> dict[str, Any]:
    value = _status_snapshot(plan, status, error=error)
    atomic_write_json(Path(plan["output_root"]) / STATUS_NAME, value)
    return value


def _git_head(path: str | Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _tcp_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.25):
            return True
    except OSError:
        return False


def _tail(path: str | Path, lines: int = 40) -> str:
    path = Path(path)
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def preflight(plan: dict[str, Any], *, min_free_gb: float = 100.0) -> dict[str, Any]:
    output_root = Path(plan["output_root"])
    rldx_repo = Path(plan["rldx_repo"])
    if str(output_root).startswith("/gs/fs/"):
        raise ValueError("Collection outputs must not be written under /gs/fs")
    output_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_root).free / (1024**3)
    if free_gb < float(min_free_gb):
        raise RuntimeError(
            f"Only {free_gb:.1f} GiB free at {output_root}; "
            f"--min-free-gb requires {float(min_free_gb):.1f}"
        )
    actual_rldx_commit = _git_head(rldx_repo)
    if actual_rldx_commit != plan["rldx_commit"]:
        raise RuntimeError(
            "RLDX repository commit changed after planning: "
            f"{actual_rldx_commit} != {plan['rldx_commit']}"
        )
    if actual_rldx_commit != RLDX1_BENCHMARK_COMMIT:
        raise RuntimeError(
            f"RLDX must be pinned to {RLDX1_BENCHMARK_COMMIT}, got "
            f"{actual_rldx_commit}"
        )
    marker_checks = {
        rldx_repo / "rldx/model/core/rldx.py": "safe_observation_context",
        rldx_repo / "rldx/policy/policy_runtime.py": "safe_feature_mode",
        rldx_repo / "rldx/policy/step_request.py": "safe_feature_mode",
    }
    for path, marker in marker_checks.items():
        if not path.is_file() or marker not in path.read_text(errors="replace"):
            raise RuntimeError(
                f"RLDX observation-context patch marker {marker!r} is missing from {path}"
            )
    subprocess.run(["git", "diff", "--check"], cwd=rldx_repo, check=True, text=True)
    if shutil.which(plan["uv_executable"]) is None:
        raise RuntimeError(f"uv executable is unavailable: {plan['uv_executable']}")
    sim_python = Path(plan["sim_python"])
    if not sim_python.is_file() or not os.access(sim_python, os.X_OK):
        raise RuntimeError(f"Simulator Python is not executable: {sim_python}")
    for port in plan["ports"]:
        if _tcp_ready(port):
            raise RuntimeError(
                f"Port {port} is already listening; stop the existing server before launch"
            )
    gpu_query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    )
    available_gpus = {
        int(line.strip()) for line in gpu_query.splitlines() if line.strip()
    }
    missing_gpus = sorted(set(plan["gpus"]) - available_gpus)
    if missing_gpus:
        raise RuntimeError(f"Requested GPUs are unavailable: {missing_gpus}")
    return {
        "free_gb": free_gb,
        "rldx_commit": actual_rldx_commit,
        "robocasa_commit": plan["robocasa_commit"],
        "available_gpus": sorted(available_gpus),
        "ports_free": list(plan["ports"]),
    }


def _start_process(
    command: list[str],
    *,
    cwd: str | Path,
    environment: dict[str, str],
    log_path: str | Path,
) -> dict[str, Any]:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", buffering=1)
    stream.write(f"\n[{utc_now()}] COMMAND: {' '.join(command)}\n")
    stream.flush()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        stream.close()
        raise
    return {"process": process, "stream": stream, "log_path": str(log_path)}


def _stop_processes(processes: list[dict[str, Any]], timeout: float = 30.0) -> None:
    for item in processes:
        process = item["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + float(timeout)
    for item in processes:
        process = item["process"]
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        finally:
            item["stream"].close()


def _wait_for_servers(
    processes: list[dict[str, Any]],
    lanes: list[dict[str, Any]],
    timeout: float,
) -> None:
    deadline = time.monotonic() + float(timeout)
    pending = set(range(len(processes)))
    while pending:
        for index in list(pending):
            process = processes[index]["process"]
            if process.poll() is not None:
                log_path = processes[index]["log_path"]
                raise RuntimeError(
                    f"RLDX server exited before readiness on port "
                    f"{lanes[index]['port']} (exit={process.returncode})\n"
                    + _tail(log_path)
                )
            if _tcp_ready(lanes[index]["port"]):
                pending.remove(index)
        if not pending:
            return
        if time.monotonic() >= deadline:
            ports = [lanes[index]["port"] for index in sorted(pending)]
            raise TimeoutError(f"Timed out waiting for RLDX server ports {ports}")
        time.sleep(2.0)


def _wait_for_clients(processes: list[dict[str, Any]]) -> None:
    pending = set(range(len(processes)))
    while pending:
        failures = []
        for index in list(pending):
            returncode = processes[index]["process"].poll()
            if returncode is None:
                continue
            pending.remove(index)
            if returncode != 0:
                failures.append(index)
        if failures:
            details = []
            for index in failures:
                details.append(
                    f"client {index} exit={processes[index]['process'].returncode}\n"
                    + _tail(processes[index]["log_path"])
                )
            raise RuntimeError(
                "One or more collection clients failed:\n" + "\n".join(details)
            )
        if pending:
            time.sleep(1.0)


def annotate_source_summary(
    plan: dict[str, Any], block: dict[str, Any], lane: dict[str, Any]
) -> dict[str, Any]:
    path = Path(lane["output_dir"]) / SUMMARY_NAME
    summary = json.loads(path.read_text())
    provenance = collection_provenance(plan, block, lane)
    policy_provenance = (
        summary["config"].get("policy_config", {}).get("collection_provenance")
    )
    if policy_provenance != provenance:
        raise ValueError(
            f"Collector provenance disagrees with allocation plan for {lane['output_dir']}"
        )
    summary["config"]["orchestration"] = provenance
    atomic_write_json(path, summary)
    return summary


def _block_complete(plan: dict[str, Any], block: dict[str, Any]) -> bool:
    path = Path(block["block_dir"]) / BLOCK_STATUS_NAME
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "complete"
        or payload.get("plan_fingerprint") != plan["plan_fingerprint"]
    ):
        return False
    for lane in block["lanes"]:
        summary_path = Path(lane["output_dir"]) / SUMMARY_NAME
        if not summary_path.is_file():
            return False
        summary = json.loads(summary_path.read_text())
        if summary.get("config", {}).get("orchestration") != collection_provenance(
            plan, block, lane
        ):
            return False
        validation = validate_atomic_dataset(lane["output_dir"])
        if (
            not validation["valid"]
            or validation["num_rollouts"] != lane["num_rollouts"]
        ):
            return False
    return True


def quarantine_incomplete_block(
    plan: dict[str, Any], block: dict[str, Any]
) -> Path | None:
    source = Path(block["block_dir"])
    if not source.exists():
        return None
    destination_root = Path(plan["output_root"]) / "incomplete_blocks"
    destination_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = destination_root / f"{source.name}__{stamp}"
    suffix = 1
    while destination.exists():
        destination = destination_root / f"{source.name}__{stamp}_{suffix}"
        suffix += 1
    shutil.move(str(source), str(destination))
    return destination


def run_block(
    plan: dict[str, Any],
    block: dict[str, Any],
    *,
    server_start_timeout: float = 1200.0,
    process_stop_timeout: float = 30.0,
    restart_incomplete: bool = False,
) -> dict[str, Any]:
    if _block_complete(plan, block):
        return json.loads((Path(block["block_dir"]) / BLOCK_STATUS_NAME).read_text())
    block_dir = Path(block["block_dir"])
    if block_dir.exists():
        if not restart_incomplete:
            raise RuntimeError(
                f"Block {block['block_index']} is incomplete; pass "
                "--restart-incomplete-blocks to quarantine and replay it"
            )
        quarantine_incomplete_block(plan, block)
    block_dir.mkdir(parents=True, exist_ok=True)
    block_status_path = block_dir / BLOCK_STATUS_NAME
    base_status = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol": plan["protocol"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "block_index": block["block_index"],
        "environment_seed": block["environment_seed"],
        "started_at": utc_now(),
        "status": "running",
        "lanes": block["lanes"],
    }
    atomic_write_json(block_status_path, base_status)

    servers: list[dict[str, Any]] = []
    clients: list[dict[str, Any]] = []
    try:
        for lane in block["lanes"]:
            if _tcp_ready(lane["port"]):
                raise RuntimeError(
                    f"Port {lane['port']} became occupied before block launch"
                )
            servers.append(
                _start_process(
                    server_command(plan, lane),
                    cwd=plan["rldx_repo"],
                    environment=server_environment(plan, lane),
                    log_path=lane["server_log"],
                )
            )
        _wait_for_servers(servers, block["lanes"], server_start_timeout)

        for lane in block["lanes"]:
            clients.append(
                _start_process(
                    collector_command(plan, block, lane),
                    cwd=Path(__file__).resolve().parents[3],
                    environment=collector_environment(lane),
                    log_path=lane["client_log"],
                )
            )
        _wait_for_clients(clients)
        _stop_processes(clients, timeout=process_stop_timeout)
        clients = []
        _stop_processes(servers, timeout=process_stop_timeout)
        servers = []

        validations = []
        for lane in block["lanes"]:
            annotate_source_summary(plan, block, lane)
            validation = validate_atomic_dataset(lane["output_dir"])
            if not validation["valid"]:
                raise RuntimeError(
                    f"Block {block['block_index']} lane {lane['lane_index']} failed "
                    "dataset validation: " + "; ".join(validation["errors"])
                )
            if validation["num_rollouts"] != lane["num_rollouts"]:
                raise RuntimeError(
                    f"Block {block['block_index']} lane {lane['lane_index']} has "
                    f"{validation['num_rollouts']} rollouts, expected "
                    f"{lane['num_rollouts']}"
                )
            validations.append(
                {
                    "lane_index": lane["lane_index"],
                    "dataset_dir": validation["dataset_dir"],
                    "rollouts": validation["num_rollouts"],
                    "successes": validation["num_successes"],
                    "failures": validation["num_failures"],
                    "warnings": validation["warnings"],
                }
            )
        completed = {
            **base_status,
            "status": "complete",
            "completed_at": utc_now(),
            "validation": validations,
        }
        atomic_write_json(block_status_path, completed)
        return completed
    except BaseException as error:
        _stop_processes(clients, timeout=process_stop_timeout)
        _stop_processes(servers, timeout=process_stop_timeout)
        failed = {
            **base_status,
            "status": "failed",
            "failed_at": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(block_status_path, failed)
        raise


def all_source_dirs(plan: dict[str, Any]) -> list[str]:
    return [lane["output_dir"] for block in plan["blocks"] for lane in block["lanes"]]


def all_blocks_complete(plan: dict[str, Any]) -> bool:
    return all(_block_complete(plan, block) for block in plan["blocks"])


def merge_completed_plan(plan: dict[str, Any], *, copy: bool = False) -> dict[str, Any]:
    if not all_blocks_complete(plan):
        raise RuntimeError("All planned blocks must complete before merge")
    merged = Path(plan["merged_output_dir"])
    if merged.exists() and any(merged.iterdir()):
        validation = validate_atomic_dataset(merged)
        if validation["valid"] and validation["num_rollouts"] == plan["total_rollouts"]:
            return {"validation": validation, "existing": True}
        raise FileExistsError(f"Merged output already exists but is invalid: {merged}")
    result = merge_atomic_datasets(all_source_dirs(plan), merged, copy=copy)
    summary_path = merged / SUMMARY_NAME
    summary = json.loads(summary_path.read_text())
    source_allocations = [
        collection_provenance(plan, block, lane)
        for block in plan["blocks"]
        for lane in block["lanes"]
    ]
    merged_policy_provenance = [
        {key: value for key, value in provenance.items() if key != "source_dataset"}
        for provenance in summary["config"].get("source_policy_provenance", [])
    ]
    if merged_policy_provenance != source_allocations:
        raise RuntimeError(
            "Merged source policy provenance disagrees with the allocation plan"
        )
    summary["config"]["orchestration"] = {
        "protocol": plan["protocol"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "allocation_manifest": str(merged / PLAN_NAME),
        "num_blocks": plan["num_blocks"],
        "gpus": plan["gpus"],
        "ports": plan["ports"],
        "environment_seeds": plan["environment_seeds"],
        "server_rng_seeds": plan["server_rng_seeds"],
        "source_allocations": source_allocations,
    }
    atomic_write_json(summary_path, summary)
    atomic_write_json(merged / PLAN_NAME, plan)
    validation = validate_atomic_dataset(merged)
    if not validation["valid"]:
        raise RuntimeError(
            "Merged orchestrated dataset failed validation: "
            + "; ".join(validation["errors"])
        )
    if validation["num_rollouts"] != plan["total_rollouts"]:
        raise RuntimeError(
            f"Merged dataset has {validation['num_rollouts']} rollouts, expected "
            f"{plan['total_rollouts']}"
        )
    result["validation"] = validation
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--rldx-repo", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--gpus", nargs="+", type=int, default=list(DEFAULT_GPUS))
    parser.add_argument("--ports", nargs="+", type=int, default=list(DEFAULT_PORTS))
    parser.add_argument("--total-rollouts-per-task", type=int, default=250)
    parser.add_argument("--num-blocks", type=int, default=12)
    parser.add_argument("--environment-seed-start", type=int, default=32)
    parser.add_argument("--server-seed-start", type=int, default=1000)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument(
        "--safe-feature-mode",
        choices=(DEFAULT_FEATURE_MODE,),
        default=DEFAULT_FEATURE_MODE,
    )
    parser.add_argument("--split", default="target")
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument(
        "--record-videos", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--video-frame-stride", type=int, default=4)
    parser.add_argument("--sim-python", default=sys.executable)
    parser.add_argument("--uv-executable", default="uv")
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    parser.add_argument("--server-start-timeout", type=float, default=1200.0)
    parser.add_argument("--process-stop-timeout", type=float, default=30.0)
    parser.add_argument("--block-indices", nargs="+", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-incomplete-blocks", action="store_true")
    parser.add_argument("--action-parity-verified", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--copy-on-merge", action="store_true")
    return parser


def _install_signal_handlers():
    previous = {}

    def handler(signum, _frame):
        raise TerminationRequested(f"Received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, handler)
    return previous


def _restore_signal_handlers(previous) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    rldx_commit = _git_head(args.rldx_repo)
    plan = build_collection_plan(
        output_root=args.output_root,
        log_root=args.log_root,
        rldx_repo=args.rldx_repo,
        tasks=args.tasks,
        gpus=args.gpus,
        ports=args.ports,
        total_rollouts_per_task=args.total_rollouts_per_task,
        num_blocks=args.num_blocks,
        environment_seed_start=args.environment_seed_start,
        server_seed_start=args.server_seed_start,
        model_path=args.model_path,
        policy_name=args.policy_name,
        safe_feature_mode=args.safe_feature_mode,
        split=args.split,
        replan_steps=args.replan_steps,
        record_videos=args.record_videos,
        video_frame_stride=args.video_frame_stride,
        robocasa_commit=current_robocasa_commit(),
        rldx_commit=rldx_commit,
        sim_python=args.sim_python,
        uv_executable=args.uv_executable,
    )
    plan = write_or_verify_plan(plan, resume=args.resume)
    if not (Path(plan["output_root"]) / STATUS_NAME).exists():
        write_status(plan, "planned")
    print(
        json.dumps(
            {
                "protocol": plan["protocol"],
                "plan_fingerprint": plan["plan_fingerprint"],
                "output_root": plan["output_root"],
                "total_rollouts": plan["total_rollouts"],
                "num_blocks": plan["num_blocks"],
                "rollouts_per_block": plan["rollouts_per_block"],
                "gpus": plan["gpus"],
                "ports": plan["ports"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.plan_only:
        return
    if not args.action_parity_verified:
        raise SystemExit(
            "error: execution requires --action-parity-verified after the clean "
            "action/context parity smoke"
        )

    indices = (
        list(range(plan["num_blocks"]))
        if args.block_indices is None
        else list(args.block_indices)
    )
    if len(indices) != len(set(indices)) or any(
        index < 0 or index >= plan["num_blocks"] for index in indices
    ):
        raise SystemExit(
            "error: --block-indices contains duplicates or invalid indices"
        )

    previous_handlers = _install_signal_handlers()
    try:
        preflight_report = preflight(plan, min_free_gb=args.min_free_gb)
        status = write_status(plan, "running")
        status["preflight"] = preflight_report
        atomic_write_json(Path(plan["output_root"]) / STATUS_NAME, status)
        for index in indices:
            run_block(
                plan,
                plan["blocks"][index],
                server_start_timeout=args.server_start_timeout,
                process_stop_timeout=args.process_stop_timeout,
                restart_incomplete=args.restart_incomplete_blocks,
            )
            write_status(plan, "running")
        if all_blocks_complete(plan) and not args.no_merge:
            result = merge_completed_plan(plan, copy=args.copy_on_merge)
            final = write_status(plan, "complete")
            final["validation"] = {
                "dataset_dir": result["validation"]["dataset_dir"],
                "rollouts": result["validation"]["num_rollouts"],
                "successes": result["validation"]["num_successes"],
                "failures": result["validation"]["num_failures"],
            }
            atomic_write_json(Path(plan["output_root"]) / STATUS_NAME, final)
            print(json.dumps(final, indent=2, sort_keys=True), flush=True)
        else:
            partial = write_status(plan, "partial")
            print(json.dumps(partial, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write_status(plan, "failed", error=str(error))
        raise
    finally:
        _restore_signal_handlers(previous_handlers)


if __name__ == "__main__":
    main()
