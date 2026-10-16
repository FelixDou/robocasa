"""Observation sampler state for new references, without changing frozen snapshots.

Forced reads are measurements: Observable.update advances its timer even when
force=True. Keep the live sensor functions bound to the new simulator and copy
only their declared sampling state and the shared dependency cache.
"""

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache, partial
import hashlib
import inspect

import numpy as np

from robocasa.recovery.full_snapshot import (
    _environment_nodes, _node_by_path, compare_structures,
)

SCHEMA = "xr1_observation_runtime_v1"
MUTABLE_FIELDS = (
    "_time_since_last_sample", "_sampled", "_current_delay", "_current_observed_value",
)
CALLBACK_FIELDS = ("_sensor", "_corrupter", "_filter", "_delayer")


def plain(value):
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return deepcopy(value)
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        return value.copy()
    if isinstance(value, Mapping):
        return {plain(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(plain(v) for v in value)
    raise TypeError(f"Unsupported observation runtime value: {type(value)}")


@lru_cache(maxsize=1024)
def source_sha256(obj):
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError) as exc:
        raise ValueError(f"Observation implementation source unavailable: {obj}") from exc
    return hashlib.sha256(source.encode()).hexdigest()


def source_identity(obj):
    # Cache code objects/classes, never sensor closures retaining an old env.
    return {"module": obj.__module__, "qualname": obj.__qualname__,
            "source_sha256": source_sha256(obj.__code__ if inspect.isfunction(obj) else obj)}


def callback_identity(callback, *, sensor=False):
    if isinstance(callback, partial):
        return {"partial": callback_identity(callback.func, sensor=sensor),
                "args": plain(callback.args), "keywords": plain(callback.keywords)}
    if not inspect.isfunction(callback):
        # A custom callable object may hide temporal filter/corruption state.
        # No copying closures or guessing that such objects are stateless.
        raise ValueError(f"Unsupported observation callback: {type(callback)}")
    config = {"defaults": plain(callback.__defaults__), "kwdefaults": plain(callback.__kwdefaults__),
              "attributes": plain(vars(callback))}
    if not sensor:
        closure = {}
        for name, cell in zip(callback.__code__.co_freevars, callback.__closure__ or ()):
            value = cell.cell_contents
            if value is not None and not isinstance(value, (str, bool, int, float, np.generic)):
                raise ValueError(f"Unsupported stateful observation callback closure: {name}")
            closure[name] = plain(value)
        config["closure_configuration"] = closure
    return {**source_identity(callback), **config}


def owned_observation_nodes(env):
    result = []
    for path, node in _environment_nodes(env):
        try:
            namespace = object.__getattribute__(node, "__dict__")
        except AttributeError:
            continue
        if "_observables" in namespace or "_obs_cache" in namespace:
            if not isinstance(namespace.get("_observables"), Mapping) or not isinstance(namespace.get("_obs_cache"), Mapping):
                raise ValueError(f"Incomplete observable/cache owner: {path}")
            result.append((path, node))
    if not result:
        raise ValueError("No owned RoboSuite observables/cache found; cannot certify observation runtime")
    return result


def capture_observation_runtime(env):
    owners = []
    for path, node in owned_observation_nodes(env):
        records = []
        for name, observable in node._observables.items():
            attributes = vars(observable)
            if any(key not in attributes for key in (*MUTABLE_FIELDS, *CALLBACK_FIELDS)):
                raise ValueError(f"Unsupported observable sampling schema: {path}.{name}")
            records.append({"name": name, "implementation": source_identity(type(observable)),
                "callbacks": {key: callback_identity(attributes[key], sensor=key == "_sensor") for key in CALLBACK_FIELDS},
                "configuration": {key: plain(value) for key, value in attributes.items()
                                  if key not in (*MUTABLE_FIELDS, *CALLBACK_FIELDS)},
                "sampling_state": {key: plain(attributes[key]) for key in MUTABLE_FIELDS}})
        owners.append({"object_path": path, "observables": records, "obs_cache": plain(node._obs_cache)})
    return {"schema": SCHEMA, "owners": owners}


def restore_observation_runtime(env, saved):
    current = capture_observation_runtime(env)
    def topology(state):
        if state.get("schema") != SCHEMA:
            raise ValueError("Unsupported observation runtime schema")
        return [{"object_path": owner["object_path"], "observables": [
            {k: record[k] for k in ("name", "implementation", "callbacks", "configuration")}
            for record in owner["observables"]]} for owner in state["owners"]]
    differences = compare_structures(topology(saved), topology(current))
    if differences:
        raise ValueError(f"Observable topology/configuration changed: {differences[:3]}")
    for owner in saved["owners"]:
        node = _node_by_path(env, owner["object_path"])
        for record in owner["observables"]:
            observable = node._observables[record["name"]]
            if set(record["sampling_state"]) != set(MUTABLE_FIELDS):
                raise ValueError("Incomplete saved observable sampling state")
            for key, value in record["sampling_state"].items():
                setattr(observable, key, plain(value))
        node._obs_cache.clear()
        node._obs_cache.update(plain(owner["obs_cache"]))
    differences = compare_structures(saved, capture_observation_runtime(env))
    if differences:
        raise ValueError(f"Observation runtime restore differs: {differences[:3]}")
