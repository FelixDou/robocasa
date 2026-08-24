"""Complete simulator-policy snapshots for causal recovery branches.

The historical recovery helpers can rewind the RoboCasa environment, but a
counterfactual policy branch is identified only when the policy queues/cache,
semantic tracker, and random-number state are restored as well.  This module
defines that stronger, versioned contract.

Snapshot files use pickle because RoboCasa environment states and RNG payloads
contain NumPy arrays and tuples that should not be coerced through JSON.  They
must only be loaded from trusted experiment directories.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import os
from pathlib import Path
import pickle
import random
import struct
from typing import Any, Mapping

import numpy as np


FULL_SNAPSHOT_SCHEMA_VERSION = 2
FULL_SNAPSHOT_PROTOCOL = "robocasa_complete_simulator_policy_snapshot"

# MuJoCo's generalized position and velocity are not a complete integration
# state. In particular, the constraint solver warm-starts from the preceding
# acceleration, and controls, applied forces, mocap bodies, actuator activation,
# and simulation time can all affect the next mj_step. RoboCasa's reset_to()
# path restores the flattened environment state but does not guarantee that
# these data fields survive the reset/forward cycle.
SIMULATOR_INTEGRATION_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "eq_active",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "qacc_warmstart",
)

# These fields must be installed before mj_forward so derived kinematics and
# actuator forces correspond to the restored state. qacc_warmstart is restored
# afterwards because it is solver history rather than a derived quantity.
SIMULATOR_PRE_FORWARD_FIELDS = tuple(
    name for name in SIMULATOR_INTEGRATION_FIELDS if name != "qacc_warmstart"
)
SIMULATOR_POST_FORWARD_FIELDS = ("qacc_warmstart",)

# These values affect termination, observation construction, or task-language
# state but are not part of MuJoCo's flattened qpos/qvel state.  They are
# intentionally explicit: copying arbitrary wrapper attributes would capture
# sockets, renderers, and other non-restorable objects.
ENVIRONMENT_CONTROL_ATTRIBUTES = (
    "_elapsed_steps",
    "timestep",
    "cur_time",
    "done",
    "override_task_description",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_digest(digest, value: Any) -> None:
    """Hash nested scientific payloads without depending on dict insertion order."""
    if value is None:
        digest.update(b"none")
        return
    if isinstance(value, bool):
        digest.update(b"bool:1" if value else b"bool:0")
        return
    if isinstance(value, (int, np.integer)):
        digest.update(b"int:")
        digest.update(str(int(value)).encode())
        return
    if isinstance(value, (float, np.floating)):
        digest.update(b"float:")
        digest.update(struct.pack("!d", float(value)))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"str:")
        digest.update(struct.pack("!Q", len(encoded)))
        digest.update(encoded)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
        digest.update(b"bytes:")
        digest.update(struct.pack("!Q", len(encoded)))
        digest.update(encoded)
        return
    if isinstance(value, Path):
        _update_digest(digest, str(value))
        return

    # Torch is optional in simulator-independent tests.  Hash tensor metadata
    # explicitly and then its exact storage bytes.  NumPy cannot represent
    # every torch dtype (notably bfloat16), so unsupported dtypes are viewed as
    # uint16 without numerically converting them.  This preserves every bit
    # used by the policy request and keeps repeat comparisons exact.
    if value.__class__.__module__.startswith("torch") and hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        dtype = str(tensor.dtype)
        digest.update(b"torch_tensor:")
        _update_digest(digest, dtype)
        _update_digest(digest, tuple(tensor.shape))
        try:
            array = tensor.numpy()
        except TypeError:
            if dtype != "torch.bfloat16":
                raise
            import torch

            array = tensor.view(torch.uint16).numpy()
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray:")
        _update_digest(digest, str(array.dtype))
        _update_digest(digest, tuple(array.shape))
        digest.update(array.tobytes(order="C"))
        return
    if is_dataclass(value):
        digest.update(f"dataclass:{value.__class__.__qualname__}:".encode())
        for field in fields(value):
            _update_digest(digest, field.name)
            _update_digest(digest, getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping:")
        ordered = sorted(value.items(), key=lambda item: stable_digest(item[0]))
        for key, item in ordered:
            _update_digest(digest, key)
            _update_digest(digest, item)
        return
    if isinstance(value, (list, tuple, deque)):
        digest.update(f"sequence:{value.__class__.__name__}:".encode())
        for item in value:
            _update_digest(digest, item)
        return
    if isinstance(value, (set, frozenset)):
        digest.update(b"set:")
        for item in sorted(value, key=stable_digest):
            _update_digest(digest, item)
        return

    # Environment state dictionaries occasionally contain small simulator
    # value objects. Pickle preserves their full content and fails loudly for
    # objects that cannot safely be snapshotted.
    digest.update(
        f"pickle:{value.__class__.__module__}.{value.__class__.__qualname__}:".encode()
    )
    digest.update(pickle.dumps(value, protocol=5))


def stable_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _capture_global_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
    }
    try:
        import torch
    except ImportError:
        state["torch_available"] = False
        return state

    state["torch_available"] = True
    state["torch_cpu"] = torch.get_rng_state().cpu().numpy().copy()
    cuda_available = bool(torch.cuda.is_available())
    state["torch_cuda_available"] = cuda_available
    if cuda_available:
        state["torch_cuda"] = [
            item.cpu().numpy().copy() for item in torch.cuda.get_rng_state_all()
        ]
    return state


def _restore_global_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_legacy"])
    if not state.get("torch_available"):
        return
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Snapshot requires torch RNG restoration") from error
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8))
    if state.get("torch_cuda_available"):
        if not torch.cuda.is_available():
            raise RuntimeError("Snapshot requires CUDA RNG restoration")
        saved = state.get("torch_cuda", [])
        if len(saved) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from snapshot: "
                f"snapshot={len(saved)} current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8) for item in saved]
        )


def _environment_nodes(env):
    queue = [("env", env)]
    visited = set()
    while queue:
        path, candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        yield path, candidate
        for attribute in ("env", "_env", "unwrapped"):
            try:
                nested = getattr(candidate, attribute, None)
            except (AttributeError, RuntimeError):
                nested = None
            if nested is not None and id(nested) not in visited:
                queue.append((f"{path}.{attribute}", nested))


def _rng_payload(rng):
    if isinstance(rng, np.random.Generator):
        return {"kind": "numpy_generator", "state": deepcopy(rng.bit_generator.state)}
    if isinstance(rng, np.random.RandomState):
        return {"kind": "numpy_random_state", "state": deepcopy(rng.get_state())}
    if isinstance(rng, random.Random):
        return {"kind": "python_random", "state": deepcopy(rng.getstate())}
    if rng.__class__.__module__.startswith("torch") and hasattr(rng, "get_state"):
        value = rng.get_state()
        if hasattr(value, "cpu"):
            value = value.cpu().numpy().copy()
        return {"kind": "torch_generator", "state": value}
    return None


def _capture_environment_rng_state(env) -> list[dict]:
    records = []
    seen_rng = set()
    for path, candidate in _environment_nodes(env):
        for attribute in (
            "np_random",
            "_np_random",
            "rng",
            "_rng",
            "random_state",
            "_random_state",
        ):
            try:
                rng = getattr(candidate, attribute, None)
            except (AttributeError, RuntimeError):
                continue
            if rng is None or id(rng) in seen_rng:
                continue
            payload = _rng_payload(rng)
            if payload is None:
                continue
            seen_rng.add(id(rng))
            records.append({"object_path": path, "attribute": attribute, **payload})
    return records


def _capture_environment_control_state(env) -> list[dict]:
    records = []
    for path, candidate in _environment_nodes(env):
        for attribute in ENVIRONMENT_CONTROL_ATTRIBUTES:
            try:
                value = getattr(candidate, attribute)
            except (AttributeError, RuntimeError):
                continue
            if callable(value):
                continue
            records.append(
                {
                    "object_path": path,
                    "attribute": attribute,
                    "value": deepcopy(value),
                }
            )
    return records


def _node_by_path(env, path: str):
    if path == "env":
        return env
    candidate = env
    for attribute in path.split(".")[1:]:
        candidate = getattr(candidate, attribute)
    return candidate


def _restore_environment_rng_state(env, records) -> None:
    for record in records:
        candidate = _node_by_path(env, record["object_path"])
        rng = getattr(candidate, record["attribute"])
        kind = record["kind"]
        state = deepcopy(record["state"])
        if kind == "numpy_generator":
            rng.bit_generator.state = state
        elif kind == "numpy_random_state":
            rng.set_state(state)
        elif kind == "python_random":
            rng.setstate(state)
        elif kind == "torch_generator":
            import torch

            rng.set_state(torch.as_tensor(state, dtype=torch.uint8))
        else:
            raise ValueError(f"Unsupported environment RNG kind: {kind}")


def _restore_environment_control_state(env, records) -> None:
    for record in records:
        candidate = _node_by_path(env, record["object_path"])
        setattr(candidate, record["attribute"], deepcopy(record["value"]))


def _tracker_state(tracker):
    if tracker is None:
        return None
    getter = getattr(tracker, "get_state", None)
    if not callable(getter):
        raise TypeError("Stage tracker must expose get_state()")
    return deepcopy(getter())


def _capture_simulator_integration_state(env) -> dict:
    """Capture every available MuJoCo field that can affect the next step."""
    from robocasa.recovery.recovery_rollout import _get_sim

    sim = _get_sim(env)
    data = getattr(sim, "data", None) if sim is not None else None
    if data is None:
        return {}
    state = {}
    for name in SIMULATOR_INTEGRATION_FIELDS:
        try:
            value = getattr(data, name)
        except (AttributeError, RuntimeError):
            continue
        if callable(value):
            continue
        if isinstance(value, np.ndarray) or hasattr(value, "shape"):
            state[name] = np.asarray(value).copy()
        else:
            state[name] = deepcopy(value)
    return state


def _assign_simulator_data_field(data, name: str, saved) -> None:
    current = getattr(data, name)
    saved_array = np.asarray(saved)
    current_shape = getattr(current, "shape", None)
    if current_shape is not None and tuple(current_shape) != ():
        if tuple(current_shape) != tuple(saved_array.shape):
            raise ValueError(
                f"MuJoCo data field {name} changed shape: "
                f"snapshot={saved_array.shape} current={current_shape}"
            )
        current[...] = saved_array
        return
    value = saved_array.item() if saved_array.shape == () else deepcopy(saved)
    setattr(data, name, value)


def restore_simulator_integration_state(env, state: Mapping[str, Any]) -> None:
    """Restore MuJoCo step inputs around one forward-dynamics refresh."""
    from robocasa.recovery.recovery_rollout import _get_sim

    sim = _get_sim(env)
    data = getattr(sim, "data", None) if sim is not None else None
    if not state:
        return
    if data is None:
        raise RuntimeError(
            "Snapshot contains MuJoCo integration state but env has none"
        )

    available = set(state)
    for name in SIMULATOR_PRE_FORWARD_FIELDS:
        if name in available:
            _assign_simulator_data_field(data, name, state[name])
    sim.forward()
    for name in SIMULATOR_POST_FORWARD_FIELDS:
        if name in available:
            _assign_simulator_data_field(data, name, state[name])


def environment_fingerprint(env) -> dict:
    """Return the environment fields that must match across paired branches."""
    from robocasa.recovery.recovery_rollout import _capture_state

    return {
        "simulator_state": {
            "environment_state": deepcopy(_capture_state(env)),
            "integration_state": _capture_simulator_integration_state(env),
        },
        "control_state": _capture_environment_control_state(env),
        "rng_state": _capture_environment_rng_state(env),
    }


@dataclass
class FullSnapshot:
    schema_version: int
    protocol: str
    snapshot_id: str
    parent_id: str
    task_name: str
    trigger_name: str
    environment_step: int
    captured_at: str
    environment_state: Any
    simulator_integration_state: Any
    policy_state: Any
    observation: Any
    subtask_eval: Any
    tracker_state: Any
    global_rng_state: Any
    environment_rng_state: Any
    environment_control_state: Any
    metadata: dict
    payload_sha256: str

    def payload(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in {"snapshot_id", "payload_sha256"}
        }

    def validate(self) -> None:
        if self.schema_version != FULL_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported FullSnapshot schema {self.schema_version}; "
                f"expected {FULL_SNAPSHOT_SCHEMA_VERSION}"
            )
        if self.protocol != FULL_SNAPSHOT_PROTOCOL:
            raise ValueError(f"Unexpected snapshot protocol: {self.protocol}")
        expected = stable_digest(self.payload())
        if expected != self.payload_sha256:
            raise ValueError(
                "FullSnapshot payload checksum mismatch: "
                f"expected={self.payload_sha256} actual={expected}"
            )
        expected_id = hashlib.sha256(
            f"{self.parent_id}:{self.trigger_name}:{self.environment_step}:"
            f"{self.payload_sha256}".encode()
        ).hexdigest()[:24]
        if self.snapshot_id != expected_id:
            raise ValueError(
                f"FullSnapshot ID mismatch: expected={expected_id} "
                f"actual={self.snapshot_id}"
            )


def capture_full_snapshot(
    env,
    policy,
    *,
    parent_id: str,
    task_name: str,
    trigger_name: str,
    environment_step: int,
    observation,
    subtask_eval=None,
    tracker=None,
    metadata=None,
) -> FullSnapshot:
    """Capture a complete branch point and fail if policy state is unavailable."""
    from robocasa.recovery.recovery_rollout import _capture_state
    from robocasa.recovery.subtask_eval import get_subtask_eval

    getter = getattr(policy, "get_state", None)
    if not callable(getter):
        raise TypeError("Policy must expose get_state() for complete snapshots")
    policy_state = deepcopy(getter())
    if not policy_state.get("at_inference_boundary", False):
        raise ValueError(
            "Full snapshots must be captured at a genuine policy inference boundary"
        )
    if subtask_eval is None:
        subtask_eval = get_subtask_eval(env)
    provisional = FullSnapshot(
        schema_version=FULL_SNAPSHOT_SCHEMA_VERSION,
        protocol=FULL_SNAPSHOT_PROTOCOL,
        snapshot_id="",
        parent_id=str(parent_id),
        task_name=str(task_name),
        trigger_name=str(trigger_name),
        environment_step=int(environment_step),
        captured_at=utc_now(),
        environment_state=deepcopy(_capture_state(env)),
        simulator_integration_state=_capture_simulator_integration_state(env),
        policy_state=policy_state,
        observation=deepcopy(observation),
        subtask_eval=deepcopy(subtask_eval),
        tracker_state=_tracker_state(tracker),
        global_rng_state=_capture_global_rng_state(),
        environment_rng_state=_capture_environment_rng_state(env),
        environment_control_state=_capture_environment_control_state(env),
        metadata=deepcopy(metadata or {}),
        payload_sha256="",
    )
    provisional.payload_sha256 = stable_digest(provisional.payload())
    provisional.snapshot_id = hashlib.sha256(
        f"{provisional.parent_id}:{provisional.trigger_name}:"
        f"{provisional.environment_step}:{provisional.payload_sha256}".encode()
    ).hexdigest()[:24]
    provisional.validate()
    return provisional


def _array(value):
    if value.__class__.__module__.startswith("torch") and hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compare_structures(expected, actual, *, atol=0.0, rtol=0.0, path="root"):
    """Return detailed mismatches while allowing declared numeric tolerances."""
    mismatches = []
    if isinstance(expected, np.ndarray) or (
        expected.__class__.__module__.startswith("torch")
        and hasattr(expected, "detach")
    ):
        try:
            left, right = _array(expected), _array(actual)
        except Exception:
            return [{"path": path, "reason": "array_type"}]
        if left.shape != right.shape or left.dtype != right.dtype:
            return [
                {
                    "path": path,
                    "reason": "array_schema",
                    "expected_shape": list(left.shape),
                    "actual_shape": list(right.shape),
                    "expected_dtype": str(left.dtype),
                    "actual_dtype": str(right.dtype),
                }
            ]
        if np.issubdtype(left.dtype, np.number):
            equal = np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True)
            if not equal:
                delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                mismatches.append(
                    {
                        "path": path,
                        "reason": "array_value",
                        "max_abs_error": float(np.nanmax(delta)) if delta.size else 0.0,
                    }
                )
        elif not np.array_equal(left, right):
            mismatches.append({"path": path, "reason": "array_value"})
        return mismatches
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [{"path": path, "reason": "mapping_type"}]
        if set(expected) != set(actual):
            mismatches.append(
                {
                    "path": path,
                    "reason": "mapping_keys",
                    "missing": sorted(str(key) for key in set(expected) - set(actual)),
                    "extra": sorted(str(key) for key in set(actual) - set(expected)),
                }
            )
        for key in set(expected) & set(actual):
            mismatches.extend(
                compare_structures(
                    expected[key],
                    actual[key],
                    atol=atol,
                    rtol=rtol,
                    path=f"{path}.{key}",
                )
            )
        return mismatches
    if isinstance(expected, (list, tuple, deque)):
        if not isinstance(actual, (list, tuple, deque)) or len(expected) != len(actual):
            return [
                {
                    "path": path,
                    "reason": "sequence_schema",
                    "expected_length": len(expected),
                    "actual_length": len(actual)
                    if hasattr(actual, "__len__")
                    else None,
                }
            ]
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatches.extend(
                compare_structures(
                    left,
                    right,
                    atol=atol,
                    rtol=rtol,
                    path=f"{path}[{index}]",
                )
            )
        return mismatches
    if isinstance(expected, (float, np.floating)):
        if not np.isclose(
            float(expected), float(actual), atol=atol, rtol=rtol, equal_nan=True
        ):
            mismatches.append(
                {
                    "path": path,
                    "reason": "numeric_value",
                    "expected": float(expected),
                    "actual": float(actual),
                }
            )
        return mismatches
    if expected != actual:
        mismatches.append(
            {
                "path": path,
                "reason": "value",
                "expected": repr(expected),
                "actual": repr(actual),
            }
        )
    return mismatches


def restore_full_snapshot(
    snapshot: FullSnapshot,
    env,
    policy,
    *,
    tracker=None,
    atol=1e-8,
    rtol=1e-8,
    verify_observation=True,
) -> dict:
    """Restore all declared state and return an exact round-trip audit."""
    from robocasa.recovery.recovery_rollout import (
        _capture_state,
        _get_obs_after_state_change,
        _reset_to_state,
    )

    snapshot.validate()
    setter = getattr(policy, "set_state", None)
    if not callable(setter):
        raise TypeError("Policy must expose set_state() for complete snapshots")
    _reset_to_state(env, deepcopy(snapshot.environment_state))
    restore_simulator_integration_state(env, snapshot.simulator_integration_state)
    _restore_environment_control_state(env, snapshot.environment_control_state)
    setter(deepcopy(snapshot.policy_state))
    if snapshot.tracker_state is not None:
        if tracker is None or not callable(getattr(tracker, "set_state", None)):
            raise TypeError(
                "Snapshot contains tracker state but no set_state() tracker"
            )
        tracker.set_state(deepcopy(snapshot.tracker_state))

    # reset_to() and policy reconstruction may consume randomness. Restore RNG
    # last so continuation begins from the captured schedule.
    _restore_environment_rng_state(env, snapshot.environment_rng_state)
    _restore_global_rng_state(snapshot.global_rng_state)

    current_environment_state = _capture_state(env)
    current_simulator_integration_state = _capture_simulator_integration_state(env)
    current_policy_state = policy.get_state()
    environment_mismatches = compare_structures(
        snapshot.environment_state,
        current_environment_state,
        atol=atol,
        rtol=rtol,
        path="environment",
    )
    simulator_integration_mismatches = compare_structures(
        snapshot.simulator_integration_state,
        current_simulator_integration_state,
        atol=0.0,
        rtol=0.0,
        path="simulator_integration",
    )
    policy_mismatches = compare_structures(
        snapshot.policy_state,
        current_policy_state,
        atol=0.0,
        rtol=0.0,
        path="policy",
    )
    observation_mismatches = []
    if verify_observation:
        current_observation = _get_obs_after_state_change(
            env, fallback_obs=snapshot.observation
        )
        observation_mismatches = compare_structures(
            snapshot.observation,
            current_observation,
            atol=atol,
            rtol=rtol,
            path="observation",
        )
    control_mismatches = compare_structures(
        snapshot.environment_control_state,
        _capture_environment_control_state(env),
        atol=0.0,
        rtol=0.0,
        path="environment_control",
    )
    global_rng_mismatches = compare_structures(
        snapshot.global_rng_state,
        _capture_global_rng_state(),
        atol=0.0,
        rtol=0.0,
        path="global_rng",
    )
    environment_rng_mismatches = compare_structures(
        snapshot.environment_rng_state,
        _capture_environment_rng_state(env),
        atol=0.0,
        rtol=0.0,
        path="environment_rng",
    )
    return {
        "snapshot_id": snapshot.snapshot_id,
        "environment_exact": not environment_mismatches,
        "simulator_integration_exact": not simulator_integration_mismatches,
        "policy_exact": not policy_mismatches,
        "observation_exact": not observation_mismatches,
        "environment_control_exact": not control_mismatches,
        "global_rng_exact": not global_rng_mismatches,
        "environment_rng_exact": not environment_rng_mismatches,
        "environment_mismatches": environment_mismatches,
        "simulator_integration_mismatches": simulator_integration_mismatches,
        "policy_mismatches": policy_mismatches,
        "observation_mismatches": observation_mismatches,
        "environment_control_mismatches": control_mismatches,
        "global_rng_mismatches": global_rng_mismatches,
        "environment_rng_mismatches": environment_rng_mismatches,
        "valid": not (
            environment_mismatches
            or simulator_integration_mismatches
            or policy_mismatches
            or observation_mismatches
            or control_mismatches
            or global_rng_mismatches
            or environment_rng_mismatches
        ),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def save_full_snapshot(snapshot: FullSnapshot, path: str | Path) -> Path:
    snapshot.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    envelope = {
        "schema_version": FULL_SNAPSHOT_SCHEMA_VERSION,
        "protocol": FULL_SNAPSHOT_PROTOCOL,
        "snapshot": snapshot,
        "snapshot_sha256": stable_digest(snapshot),
    }
    with gzip.open(temporary, "wb") as stream:
        pickle.dump(envelope, stream, protocol=5)
    os.replace(temporary, path)
    return path


def load_full_snapshot(path: str | Path) -> FullSnapshot:
    with gzip.open(Path(path), "rb") as stream:
        envelope = pickle.load(stream)  # noqa: S301 - trusted experiment artifact
    if envelope.get("schema_version") != FULL_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Unsupported FullSnapshot file schema")
    snapshot = envelope.get("snapshot")
    if not isinstance(snapshot, FullSnapshot):
        raise ValueError("Snapshot file does not contain FullSnapshot")
    if stable_digest(snapshot) != envelope.get("snapshot_sha256"):
        raise ValueError("Snapshot file checksum mismatch")
    snapshot.validate()
    return snapshot


__all__ = [
    "FULL_SNAPSHOT_PROTOCOL",
    "FULL_SNAPSHOT_SCHEMA_VERSION",
    "FullSnapshot",
    "capture_full_snapshot",
    "compare_structures",
    "environment_fingerprint",
    "load_full_snapshot",
    "restore_full_snapshot",
    "restore_simulator_integration_state",
    "save_full_snapshot",
    "stable_digest",
]
