"""E2's collection: each new RLDX-1 scene is attempted twice, from the same initial state (docs/rldx_e2_plan.md).

The standard collector (robocasa.recovery.safe.collect_atomic_rollouts) runs unchanged, with the
official_rldx protocol (one environment per task, repeated resets) and 2 x scenes rollouts per task. Only
the environment's reset() is replaced, per task environment:
  - reset 2j (first attempt of scene j): a normal reset; the full simulator state, model XML and episode
    metadata right after it are kept in memory;
  - reset 2j+1 (retry of scene j): instead of building a new scene, the kept state is restored through the
    environment's reset_to (the RoboCasa playback path: soft reset, reload the scene's XML, set the state),
    and its observation is returned. The policy is reset by the collector as for any rollout.
Both attempts run to ceil(1.25 H) control steps (--horizon-scale 1.25) with SAFE features recorded, so every
reset rule of the analysis can be applied afterwards: under rule c, scene j succeeds if the first attempt
succeeds before c's alarm, or, once c alarms, if the retry succeeds within the budget left. The retry is the
same for every rule (one shared retry per scene), as the plan's "seed shared by all monitors" requires.

Each reset appends a line to <output-dir>/pairs.jsonl: the reset index, role, hashes of the state and the
model XML, and for a retry the restore residual and how far its first observation is from the first
attempt's (images: mean absolute difference per camera; other arrays: maximum absolute difference).

An error stops the shard: the reset sequence must not be interrupted, so --resume and --continue-on-error are
refused. Rerun the unfinished tasks into a new output directory (docs/rldx_e2_plan.md, runbook step 6).

Usage: the collector's own arguments, e.g.
  python -u scripts/rldx_e2/collect_e2.py --output-dir DIR --tasks T1 T2 $(cat collector_flags_e2.txt) \\
      --host 127.0.0.1 --port 20150
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PAIRS_NAME = "pairs.jsonl"


def _sha(data):
    if isinstance(data, str):
        data = data.encode()
    elif isinstance(data, np.ndarray):
        data = np.ascontiguousarray(data).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def env_method(env, name):
    """First callable `name` along the wrapper chain (gymnasium wrappers do not forward everything)."""
    seen, node = set(), env
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        method = getattr(node, name, None)
        if callable(method):
            return method
        node = getattr(node, "env", None)
    return None


def obs_distance(first, other):
    """Images (uint8, 3-D): mean |diff| per key; other numeric arrays: max |diff|; strings: equality."""
    out = dict(image={}, state={}, text_equal=True, missing=[])
    for key, a in first.items():
        if key not in other:
            out["missing"].append(key)
            continue
        b = other[key]
        if isinstance(a, str) or isinstance(b, str):
            out["text_equal"] &= a == b
            continue
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.issubdtype(a.dtype, np.number):
            if a.shape != b.shape:
                out["missing"].append(f"{key}:shape")
            continue
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        if a.dtype == np.uint8 and a.ndim >= 3:
            out["image"][key] = float(d.mean())
        else:
            out["state"][key] = float(d.max()) if d.size else 0.0
    out["image_max"] = max(out["image"].values(), default=0.0)
    out["state_max"] = max(out["state"].values(), default=0.0)
    return out


def first_instruction(saved):
    """The instruction stored with the first attempt's state (ep_meta["lang"]), or None."""
    meta = saved.get("ep_meta")
    if not meta:
        return None
    try:
        lang = json.loads(meta).get("lang") if isinstance(meta, str) else dict(meta).get("lang")
    except (ValueError, TypeError, AttributeError):
        return None
    return lang or None


def _flat_state(env):
    get_state = env_method(env, "get_state")
    state = get_state()
    return state, np.asarray(state["states"], dtype=np.float64)


class PairedReset:
    """Replaces env.reset: even resets build a scene and keep its state, odd resets restore it."""

    def __init__(self, env, task_name, log_path, clock=time.time):
        self.env, self.task, self.log_path, self.clock = env, task_name, Path(log_path), clock
        self.base_reset = env.reset
        self.k = 0
        self.saved = None
        self.first_obs = None
        if env_method(env, "reset_to") is None or env_method(env, "get_state") is None:
            raise RuntimeError("E2 needs an environment with get_state() and reset_to()")

    def _log(self, row):
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def __call__(self, *args, **kwargs):
        k = self.k
        row = dict(task_name=self.task, reset_index=k, scene=k // 2, created_at=self.clock())
        if k % 2 == 0:
            result = self.base_reset(*args, **kwargs)
            obs = result[0] if isinstance(result, tuple) else result
            state, flat = _flat_state(self.env)
            self.saved = {key: state[key] for key in ("states", "model", "ep_meta") if key in state}
            self.saved["states"] = np.array(flat, copy=True)
            self.first_obs = {key: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                              for key, v in obs.items()} if isinstance(obs, dict) else None
            row.update(role="first", state_sha=_sha(flat), model_sha=_sha(state.get("model", "")),
                       ep_meta_sha=_sha(state.get("ep_meta", "") or ""))
        else:
            if self.saved is None:
                raise RuntimeError("E2: a retry reset came before any first-attempt reset")
            obs = env_method(self.env, "reset_to")(dict(self.saved))
            # reset_to reloads the model, and some tasks draw their instruction again while doing so
            # (OpenDrawer: "Open the {side} drawer.", the side drawn when the robot is placed). The
            # robot pose and the target come back from the saved state, so the first attempt's own
            # instruction is the right one: give it to the retry for its whole length. The wrapper's
            # next ordinary reset clears the override.
            lang = first_instruction(self.saved)
            setter = env_method(self.env, "set_task_description")
            if lang and setter is not None:
                setter(lang)
                row["instruction_restored"] = True
            else:
                row["instruction_restored"] = False
            getter = env_method(self.env, "get_current_observation")
            if getter is not None and (row["instruction_restored"] or not isinstance(obs, dict)):
                obs = getter()
            _, flat = _flat_state(self.env)
            residual = (float(np.max(np.abs(flat - self.saved["states"])))
                        if flat.shape == self.saved["states"].shape else float("inf"))
            dist = obs_distance(self.first_obs, obs) if isinstance(obs, dict) and self.first_obs else None
            row.update(role="retry", restored_sha=_sha(self.saved["states"]), state_sha=_sha(flat),
                       model_sha=_sha(self.saved.get("model", "")), restore_residual=residual,
                       first_obs_distance=dist)
            info = {"success": False}
            subtask = env_method(self.env, "_get_subtask_eval")
            if subtask is not None:
                try:
                    value = subtask()
                    if value is not None:
                        info["subtask_eval"] = value
                except Exception:
                    pass
            result = (obs, info)
        self._log(row)
        self.k += 1
        return result


def install(env, task_name, log_path):
    env.reset = PairedReset(env, task_name, log_path)
    return env


def paired_runtime(output_dir, runtime=None):
    """The collector's runtime with make_env wrapped so every task environment gets PairedReset."""
    from robocasa.recovery.safe import collect_atomic_rollouts as cac

    runtime = dict(runtime or cac._runtime())
    make_env = runtime["make_env"]
    log_path = Path(output_dir) / PAIRS_NAME

    def make_paired_env(task_name, *args, **kwargs):
        return install(make_env(task_name, *args, **kwargs), task_name, log_path)

    runtime["make_env"] = make_paired_env
    return runtime


def main(argv=None):
    from robocasa.recovery.safe import collect_atomic_rollouts as cac

    args = cac.build_parser().parse_args(argv)
    problems = []
    if args.resume:
        problems.append("--resume would replay resets and break the scene pairs; rerun the task into a new directory")
    if args.continue_on_error:
        problems.append("--continue-on-error rebuilds the environment after an error and breaks the scene pairs")
    if args.seed_protocol != "official_rldx":
        problems.append(f"--seed-protocol {args.seed_protocol}: E2 uses official_rldx")
    if args.num_rollouts % 2:
        problems.append("--num-rollouts must be even (two attempts per scene)")
    if any(getattr(args, k, None) is not None for k in ("success_quota", "failure_quota", "attempt_limits")):
        problems.append("E2 keeps every attempt: no quotas or attempt limits")
    if problems:
        raise SystemExit("error: " + "; ".join(problems))
    out = Path(args.output_dir)
    if (out / PAIRS_NAME).exists():
        raise SystemExit(f"error: {out / PAIRS_NAME} exists; E2 writes each shard into a new directory")
    out.mkdir(parents=True, exist_ok=True)
    try:
        return cac.run_collection(args, runtime=paired_runtime(out))
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
