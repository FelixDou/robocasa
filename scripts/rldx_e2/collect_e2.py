"""E2's collection: each new RLDX-1 scene is attempted twice, from the same initial state (docs/rldx_e2_plan.md).

The standard collector (robocasa.recovery.safe.collect_atomic_rollouts) runs unchanged, with the
official_rldx protocol (one environment per task, repeated resets) and 2 x scenes rollouts per task. Only
the environment's reset() is replaced, per task environment:
  - reset 2j (first attempt of scene j): a normal reset; the full simulator state, model XML and episode
    metadata right after it are kept in memory;
  - reset 2j+1 (retry of scene j), mode "restore" (default since 2026-11-10): the kept state is restored
    through reset_to (RoboCasa's replay path), then every plain value of the task, the robots and their
    controllers (numbers, strings, arrays) that differs from the first attempt's is set back, the stored
    episode metadata is cleared (RoboCasa would otherwise reuse it for the next scene), and the retry gets
    the first attempt's instruction. Mode "restore_legacy" is E2's first collection (none of this; its
    scenes repeated and its retries started from default controller goals).
  - Mode "rebuild" (tried 2026-11-10, not repeatable in RoboCasa): the random generators the
    scene is built from (the environment's rng, numpy's global state and Python's random) are wound back
    to their state before reset 2j, and the scene is built again by the same normal reset. Unseeded
    np.random.default_rng() calls during a build (RoboCasa's fallbacks) are seeded from the scene's own
    generator state (seeded_fallbacks), so they too repeat. Every task choice,
    target and controller value is then the same as the first attempt's, and the state hash and model XML
    must match exactly (check_e2). Every other numpy generator reachable from the environment or held at
    module level in RoboCasa / robosuite / gym is wound back too (find_generators). Mode "restore" (E2's first collection) restored the kept state through
    reset_to, which reloads the model: the restore check showed that this redraws task attributes (OpenDrawer's
    side, NavigateKitchen's target_pos) and leaves the arm controllers' goals at their default pose.
    The policy is reset by the collector as for any rollout.
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
import contextlib
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PAIRS_NAME = "pairs.jsonl"
DEFAULT_MODE = "restore"
MODES = ("restore", "restore_legacy", "rebuild")
PLAIN_EXCLUDE = {"_ep_meta", "override_task_description"}


def task_env(env):
    """The innermost environment with a simulator (the RoboCasa task), under any wrappers."""
    seen, node, found = set(), env, None
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if hasattr(node, "__dict__") and "sim" in vars(node):
            found = node
        node = vars(node).get("env") if hasattr(node, "__dict__") else None
    return found


def _plain(v, depth=0):
    if v is None or isinstance(v, (bool, int, float, str, np.generic)):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype.kind in "biuf" and v.size <= 100000
    if depth < 3 and isinstance(v, (list, tuple)):
        return all(_plain(x, depth + 1) for x in v)
    if depth < 3 and isinstance(v, dict):
        return all(isinstance(k, (str, int)) and _plain(x, depth + 1) for k, x in v.items())
    return False


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(np.asarray(a), np.asarray(b))
        except Exception:
            return False
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


def plain_holders(env):
    """The task environment, each robot, its composite controller and each part controller."""
    raw = task_env(env)
    out = [] if raw is None else [raw]
    for robot in (getattr(raw, "robots", None) or []) if raw is not None else []:
        out.append(robot)
        cc = getattr(robot, "composite_controller", None)
        if cc is not None:
            out.append(cc)
            out.extend((getattr(cc, "part_controllers", None) or {}).values())
    return out


def snapshot_plain(env):
    """Plain values (numbers, strings, arrays and containers of them) of the task, robots and controllers."""
    return [(obj, {k: copy.deepcopy(v) for k, v in vars(obj).items() if k not in PLAIN_EXCLUDE and _plain(v)})
            for obj in plain_holders(env)]


def copy_back(saved):
    """Set every plain value that differs from the first attempt's back to it. Returns the names restored."""
    restored = []
    for obj, snap in saved:
        cur = vars(obj)
        for k, v in snap.items():
            if k in cur and not _same(cur[k], v):
                setattr(obj, k, copy.deepcopy(v))
                restored.append(f"{type(obj).__name__}.{k}")
    return restored


def clear_ep_meta(env):
    """reset_to stores the scene's metadata on the task, and RoboCasa reuses it for the next build (layout,
    fixtures, objects, robot start): clear it so the next first attempt is a new scene."""
    raw = task_env(env)
    if raw is None or not hasattr(raw, "_ep_meta"):
        return False
    unset = getattr(raw, "unset_ep_meta", None)
    if callable(unset):
        unset()
    else:
        raw._ep_meta = {}
    return True


def env_rng(env):
    """The numpy Generator the scene is built from (RoboCasa's Kitchen.rng), searched down the wrapper chain."""
    seen, node = set(), env
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        rng = vars(node).get("rng") if hasattr(node, "__dict__") else None
        if isinstance(rng, np.random.Generator):
            return node, rng
        node = vars(node).get("env") if hasattr(node, "__dict__") else None
    return None, None


GEN_TYPES = (np.random.Generator, np.random.RandomState)
SCAN_LIMIT = 300000
SCAN_MODULES = ("robocasa", "robosuite", "gymnasium", "gym")


def find_generators(roots, limit=SCAN_LIMIT):
    """Every numpy Generator / RandomState reachable from `roots` through attributes, dicts, lists and
    tuples (arrays, strings and simulator handles are not entered), plus module-level ones in RoboCasa,
    robosuite and gym modules. Returns {id: object}."""
    found, seen, stack, n = {}, set(), list(roots), 0
    for name, mod in list(sys.modules.items()):
        if mod is not None and name.split(".")[0] in SCAN_MODULES:
            for v in list(vars(mod).values()):
                if isinstance(v, GEN_TYPES):
                    found[id(v)] = v
    while stack and n < limit:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        n += 1
        if isinstance(obj, GEN_TYPES):
            found[id(obj)] = obj
            continue
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, np.ndarray, np.generic, type)):
            continue
        if type(obj).__module__.split(".")[0] in ("mujoco", "ctypes"):
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif hasattr(obj, "__dict__"):
            try:
                stack.extend(vars(obj).values())
            except TypeError:
                pass
    return found


def _gen_state(g):
    return copy.deepcopy(g.bit_generator.state) if isinstance(g, np.random.Generator) else g.get_state()


def capture_rng(env):
    owner, rng = env_rng(env)
    if rng is None:
        raise RuntimeError("E2 rebuild mode: no numpy Generator 'rng' under the environment wrappers")
    state = copy.deepcopy(rng.bit_generator.state)
    entropy = int.from_bytes(hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).digest()[:16], "big")
    gens = find_generators([env])
    return dict(owner=owner, env=state, numpy=np.random.get_state(), python=random.getstate(), fallback=entropy,
                generators=[(g, _gen_state(g)) for g in gens.values() if g is not rng])


@contextlib.contextmanager
def seeded_fallbacks(entropy):
    """While a scene is built, an unseeded np.random.default_rng() (RoboCasa's fallback when no rng is passed:
    fixtures, object sampling, placement samplers) is seeded from a sequence derived from the scene's own
    generator state, so the first build and its rebuild draw the same values and different scenes differ."""
    original = np.random.default_rng
    seq = np.random.SeedSequence(entropy)

    def default_rng(seed=None):
        if seed is None:
            return original(seq.spawn(1)[0])
        return original(seed)
    np.random.default_rng = default_rng
    try:
        yield
    finally:
        np.random.default_rng = original


def restore_rng(saved):
    """Wind every generator back. The env's rng object is replaced only if the env made a new one."""
    owner = saved["owner"]
    rng = vars(owner).get("rng")
    rng.bit_generator.state = copy.deepcopy(saved["env"])
    np.random.set_state(saved["numpy"])
    random.setstate(saved["python"])
    for g, st in saved.get("generators", []):          # every other generator that existed before the build
        if isinstance(g, np.random.Generator):
            g.bit_generator.state = copy.deepcopy(st)
        else:
            g.set_state(st)


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

    def __init__(self, env, task_name, log_path, clock=time.time, mode=DEFAULT_MODE):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.env, self.task, self.log_path, self.clock, self.mode = env, task_name, Path(log_path), clock, mode
        self.base_reset = env.reset
        self.k = 0
        self.saved = None
        self.first_obs = None
        self.rng_states = None
        if env_method(env, "get_state") is None or (mode != "rebuild" and env_method(env, "reset_to") is None):
            raise RuntimeError("E2 needs an environment with get_state() (and reset_to() in restore mode)")
        if mode == "rebuild":
            env_rng(env)[1] or capture_rng(env)          # fail at install, not at the first retry

    def rebuild(self, *args, **kwargs):
        """The first attempt's scene, built again from the wound-back generators."""
        restore_rng(self.rng_states)
        with seeded_fallbacks(self.rng_states["fallback"]):
            return self.base_reset(*args, **kwargs)

    def _log(self, row):
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def __call__(self, *args, **kwargs):
        k = self.k
        row = dict(task_name=self.task, reset_index=k, scene=k // 2, created_at=self.clock(), mode=self.mode)
        if k % 2 == 0:
            if self.mode == "rebuild":
                self.rng_states = capture_rng(self.env)
                with seeded_fallbacks(self.rng_states["fallback"]):
                    result = self.base_reset(*args, **kwargs)
            else:
                result = self.base_reset(*args, **kwargs)
            obs = result[0] if isinstance(result, tuple) else result
            self.plain = snapshot_plain(self.env) if self.mode == "restore" else None
            state, flat = _flat_state(self.env)
            self.saved = {key: state[key] for key in ("states", "model", "ep_meta") if key in state}
            self.saved["states"] = np.array(flat, copy=True)
            self.first_obs = {key: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                              for key, v in obs.items()} if isinstance(obs, dict) else None
            row.update(role="first", state_sha=_sha(flat), model_sha=_sha(state.get("model", "")),
                       ep_meta_sha=_sha(state.get("ep_meta", "") or ""))
        elif self.mode == "rebuild":
            if self.saved is None or self.rng_states is None:
                raise RuntimeError("E2: a retry reset came before any first-attempt reset")
            result = self.rebuild(*args, **kwargs)
            obs = result[0] if isinstance(result, tuple) else result
            state, flat = _flat_state(self.env)
            residual = (float(np.max(np.abs(flat - self.saved["states"])))
                        if flat.shape == self.saved["states"].shape else float("inf"))
            dist = obs_distance(self.first_obs, obs) if isinstance(obs, dict) and self.first_obs else None
            # restored_sha = the state this retry was meant to reproduce; state_sha / model_sha = what it is.
            row.update(role="retry", restored_sha=_sha(self.saved["states"]), state_sha=_sha(flat),
                       model_sha=_sha(state.get("model", "")), restore_residual=residual, first_obs_distance=dist,
                       instruction_restored=False)
        else:
            if self.saved is None:
                raise RuntimeError("E2: a retry reset came before any first-attempt reset")
            obs = env_method(self.env, "reset_to")(dict(self.saved))
            if self.mode == "restore":
                # reset_to rebuilds the Python side of the task before loading the saved XML and state:
                # controller goals stay at the default pose and values drawn during the build (targets,
                # sides, anchors) can differ. Put the first attempt's plain values back, and clear the
                # stored metadata so the next scene is built fresh.
                names = copy_back(self.plain or [])
                row.update(values_restored=len(names), values_restored_names=sorted(set(names))[:40],
                           ep_meta_cleared=clear_ep_meta(self.env))
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


def install(env, task_name, log_path, mode=DEFAULT_MODE):
    env.reset = PairedReset(env, task_name, log_path, mode=mode)
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
