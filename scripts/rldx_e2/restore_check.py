"""Is E2's restored scene the same task as its first attempt? (docs/rldx_e2_plan.md, "Restore check").

The E2 follow-up found full-budget retries succeeding far less often than first attempts from the same
state (0.392 against 0.508), and never on some tasks after a successful first attempt (StackBowlsCabinet
0 of 33). This script needs no policy. For each task it builds environments exactly as E2's collector
does (make_env with the plan's split, seed and render setting; reset() without a seed; collect_e2's
PairedReset installed), and for each scene:
  1. first attempt: reset (builds the scene, saves its state), record the task's Python-side attributes
     and ep_meta, then apply a fixed sequence of random actions, recording the simulator state, the
     success flag and _check_success() at every step;
  2. retry: reset (restores the saved state, as in E2), record the same attributes and ep_meta, apply the
     same actions and record the same quantities.
Reported per scene: the attributes and ep_meta entries that differ after the restore, the instruction of
each attempt, the first step at which the simulator states diverge (tolerance STATE_TOL) and the largest
difference, and any step at which the success checks differ. A task is SAME only if nothing differs.

Usage (CPU; RoboCasa installed):
  python -u restore_check.py --plan <E2>/plan.json --output <E2>/restore_check.json [--tasks T ...]
      [--scenes 2] [--steps 60] [--render]
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import collect_e2 as ce  # noqa: E402

PLAN = "docs/rldx_e2_plan.md (Restore check)"
STATE_TOL = 1e-6
ACTION_SCALE = 0.3
ACTION_SEED = 20261109
VOLATILE = {"timestep", "cur_time", "done", "_obs_cache", "rng", "np_random"}


def raw_env(env):
    """The innermost environment with a simulator: the RoboCasa task itself, not a wrapper that forwards
    sim or get_ep_meta (the first version of this check read the gym wrapper)."""
    seen, node, found = set(), env, None
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if "sim" in vars(node) or hasattr(type(node), "sim") or hasattr(node, "_check_success"):
            if hasattr(node, "sim"):
                found = node
        node = vars(node).get("env") if hasattr(node, "__dict__") else None
    if found is None:
        raise RuntimeError("no robosuite environment under the wrappers")
    return found


def summarize(v, depth=0):
    """A comparable summary of one attribute, or None to skip it."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return round(v, 6) if math.isfinite(v) else str(v)
    if isinstance(v, np.generic):
        return summarize(v.item(), depth)
    if isinstance(v, np.ndarray):
        return np.round(v.astype(float), 6).tolist() if v.size <= 16 and np.issubdtype(v.dtype, np.number) else None
    name = getattr(v, "name", None)
    if isinstance(name, str) and not isinstance(v, type):
        return f"<{type(v).__name__} {name}>"
    if depth < 2 and isinstance(v, (list, tuple)) and len(v) <= 16:
        out = [summarize(x, depth + 1) for x in v]
        return out if all(x is not None or y is None for x, y in zip(out, v)) else None
    if depth < 2 and isinstance(v, dict) and len(v) <= 64:
        return {str(k): summarize(x, depth + 1) for k, x in v.items()}
    return None


def attributes(raw):
    out = {}
    for k, v in vars(raw).items():
        if k in VOLATILE:
            continue
        s = summarize(v)
        if s is not None:
            out[k] = s
    return out


def diff(a, b, path="", out=None, limit=200):
    """Every leaf that differs, by path (dicts by key, equal-length lists by index)."""
    out = {} if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}" if path else str(k), out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out[path] = [a, b]
    return out


def controllers(raw):
    """Summaries of each robot's composite controller and part controllers (goals, references)."""
    out = {}
    for i, robot in enumerate(getattr(raw, "robots", []) or []):
        out[f"robot{i}"] = {k: s for k, v in vars(robot).items() if (s := summarize(v)) is not None}
        cc = getattr(robot, "composite_controller", None)
        if cc is None:
            continue
        out[f"robot{i}.composite"] = {k: s for k, v in vars(cc).items() if (s := summarize(v)) is not None}
        for name, c in (getattr(cc, "part_controllers", None) or {}).items():
            out[f"robot{i}.{name}"] = {k: s for k, v in vars(c).items() if (s := summarize(v)) is not None}
    return out


def ep_meta(env, raw):
    try:
        return json.loads(json.dumps(raw.get_ep_meta(), default=str))
    except Exception as exc:  # report, do not hide
        return {"<error>": str(exc)}


def instruction(obs):
    return obs.get("annotation.human.task_description") if isinstance(obs, dict) else None


def sample(space, rng):
    """One action: Box parts uniform within ACTION_SCALE of their range, Discrete parts 0, Dict parts by key
    (RoboCasa's gym wrapper uses a Dict of gripper, end-effector, base-motion and control-mode parts)."""
    parts = getattr(space, "spaces", None)
    if isinstance(parts, dict):
        return {k: sample(v, rng) for k, v in sorted(parts.items())}
    if hasattr(space, "low") and hasattr(space, "high"):
        low = np.where(np.isfinite(space.low), space.low, -1.0)
        high = np.where(np.isfinite(space.high), space.high, 1.0)
        mid, half = (low + high) / 2, (high - low) / 2
        return (mid + ACTION_SCALE * half * rng.uniform(-1, 1, size=space.shape)).astype(np.float32)
    if hasattr(space, "n"):
        return 0
    raise ValueError(f"unsupported action space {type(space).__name__}")


def copy_action(a):
    return {k: copy_action(v) for k, v in a.items()} if isinstance(a, dict) else (a.copy() if isinstance(a, np.ndarray) else a)


def actions_for(env, steps, seed=ACTION_SEED):
    rng = np.random.default_rng(seed)
    return [sample(env.action_space, rng) for _ in range(steps)]


def run_attempt(env, raw, actions, reset=None):
    out = (reset or env.reset)()
    obs = out[0] if isinstance(out, tuple) else out
    getter = ce.env_method(env, "get_state")
    model = (getter() or {}).get("model", "") if getter is not None else ""
    snap = dict(instruction=instruction(obs), attributes=attributes(raw), ep_meta=ep_meta(env, raw),
                controllers=controllers(raw), model_sha=ce._sha(model or ""),
                states=[np.asarray(raw.sim.get_state().flatten(), float)], success=[], check=[])
    for a in actions:
        step = env.step(copy_action(a))
        info = step[-1] if isinstance(step, tuple) and isinstance(step[-1], dict) else {}
        snap["states"].append(np.asarray(raw.sim.get_state().flatten(), float))
        snap["success"].append(bool(info.get("success", False)))
        try:
            snap["check"].append(bool(raw._check_success()))
        except Exception as exc:
            snap["check"].append(f"error: {exc}")
    return snap


def divergence(states_a, states_b):
    diverge, worst = None, 0.0
    for i, (a, b) in enumerate(zip(states_a, states_b)):
        d = float(np.max(np.abs(a - b))) if a.shape == b.shape else math.inf
        worst = max(worst, d)
        if diverge is None and d > STATE_TOL:
            diverge = i
    return diverge, worst


def compare(first, retry, retry2=None):
    diverge, worst = divergence(first["states"], retry["states"])
    ctrl = diff(first["controllers"], retry["controllers"])
    attr = diff(first["attributes"], retry["attributes"])
    meta = diff(first["ep_meta"], retry["ep_meta"])
    succ = [i for i, (x, y) in enumerate(zip(first["success"], retry["success"])) if x != y]
    check = [i for i, (x, y) in enumerate(zip(first["check"], retry["check"])) if x != y]
    same = not attr and not meta and not ctrl and diverge is None and not succ and not check and \
        first["instruction"] == retry["instruction"] and first["model_sha"] == retry["model_sha"]
    control = None
    if retry2 is not None:
        d2, w2 = divergence(retry["states"], retry2["states"])
        control = dict(retry_vs_retry2_diverges_at_step=d2, max_state_diff=w2,
                       controller_diffs=diff(retry["controllers"], retry2["controllers"]))
    return dict(same=same, instruction=[first["instruction"], retry["instruction"]], attribute_diffs=attr,
                ep_meta_diffs=meta, controller_diffs=ctrl, determinism_control=control,
                model_same=first["model_sha"] == retry["model_sha"],
                state_diverges_at_step=diverge, max_state_diff=worst,
                success_differs_at=succ[:10], check_differs_at=check[:10],
                success_steps=[sum(first["success"]), sum(retry["success"])],
                check_steps=[sum(x is True for x in first["check"]), sum(x is True for x in retry["check"])])


def check_task(env, task, scenes, steps, log_dir, mode=ce.DEFAULT_MODE):
    """env: a fresh environment for `task`; PairedReset (in `mode`) is installed here."""
    raw = raw_env(env)
    env = ce.install(env, task, Path(log_dir) / f"{task}_pairs.jsonl", mode=mode)
    acts = actions_for(env, steps)
    paired = env.reset
    out = []
    for j in range(scenes):
        first = run_attempt(env, raw, acts)
        retry = run_attempt(env, raw, acts)

        def restore_again():                        # the same retry once more, outside the pair count
            if paired.mode == "rebuild":
                return paired.rebuild()
            obs = ce.env_method(env, "reset_to")(dict(paired.saved))
            lang = ce.first_instruction(paired.saved)
            setter = ce.env_method(env, "set_task_description")
            if lang and setter is not None:
                setter(lang)
                getter = ce.env_method(env, "get_current_observation")
                obs = getter() if getter is not None else obs
            return obs
        retry2 = run_attempt(env, raw, acts, reset=restore_again)
        out.append(dict(scene=j, **compare(first, retry, retry2)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--scenes", type=int, default=2)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, help="environment seed (default: the plan's E2 seed)")
    p.add_argument("--render", action="store_true", help="build with rendering, as E2 did (needs EGL)")
    p.add_argument("--mode", choices=ce.MODES, default=ce.DEFAULT_MODE,
                   help="retry mode to check: rebuild (default) or restore (E2's first collection)")
    a = p.parse_args(argv)
    if Path(a.output).exists():
        raise SystemExit(f"error: {a.output} exists")
    plan = json.loads(Path(a.plan).read_text())
    split = plan["settings"].get("split", "target")
    seed = a.seed if a.seed is not None else int(plan["seed"])
    tasks = a.tasks or sorted(plan["tasks"])
    from robocasa.recovery.evaluate_recovery_benchmark import make_env
    results, n_same = {}, 0
    with tempfile.TemporaryDirectory() as log_dir:
        for t in tasks:
            env = make_env(t, "gym", split, seed, bool(a.render))
            try:
                rows = check_task(env, t, a.scenes, a.steps, log_dir, mode=a.mode)
            except Exception as exc:
                rows = [dict(error=f"{type(exc).__name__}: {exc}")]
            finally:
                env.close()
            results[t] = rows
            ok = all(r.get("same") for r in rows)
            n_same += ok
            print(f"{t:28s} {'SAME' if ok else 'DIFFERENT'}")
            for r in rows:
                if "error" in r:
                    print(f"    error: {r['error']}")
                    continue
                if r["same"]:
                    continue
                print(f"    scene {r['scene']}: state diverges at step {r['state_diverges_at_step']} "
                      f"(max {r['max_state_diff']:.3g}); success steps {r['success_steps']}, "
                      f"_check_success steps {r['check_steps']}")
                if not r["model_same"]:
                    print("      the scene model XML differs")
                if r["instruction"][0] != r["instruction"][1]:
                    print(f"      instruction: {r['instruction']}")
                c = r["determinism_control"] or {}
                print(f"      control (restore vs restore): diverges at step {c.get('retry_vs_retry2_diverges_at_step')} "
                      f"(max {c.get('max_state_diff', float('nan')):.3g}); controller diffs {len(c.get('controller_diffs') or {})}")
                for label, key, n in (("ctrl", "controller_diffs", 14), ("attr", "attribute_diffs", 14),
                                      ("ep_meta", "ep_meta_diffs", 10)):
                    items = list(r[key].items())
                    for k, (x, y) in items[:n]:
                        print(f"      {label} {k}: {str(x)[:60]} -> {str(y)[:60]}")
                    if len(items) > n:
                        print(f"      {label}: {len(items) - n} more")
    out = dict(plan=PLAN, mode=a.mode, split=split, seed=seed, scenes=a.scenes, steps=a.steps, render=bool(a.render),
               action_scale=ACTION_SCALE, action_seed=ACTION_SEED, state_tol=STATE_TOL, tasks=results)
    Path(a.output).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"{n_same} of {len(tasks)} tasks SAME; wrote {a.output}")
    print("RESTORE CHECK DONE")


if __name__ == "__main__":
    main()
