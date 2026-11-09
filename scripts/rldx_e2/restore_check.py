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
    """The robosuite environment under the gym wrappers (the object with sim and get_ep_meta)."""
    seen, node = set(), env
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if hasattr(node, "sim") and hasattr(node, "get_ep_meta"):
            return node
        node = getattr(node, "env", None)
    raise RuntimeError("no robosuite environment under the wrappers")


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


def diff(a, b, prefix=""):
    """Keys whose summarized values differ (one level into dicts)."""
    out = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k, "<missing>"), b.get(k, "<missing>")
        if x == y:
            continue
        if isinstance(x, dict) and isinstance(y, dict) and not prefix:
            for kk, vv in diff(x, y, prefix=k).items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = [x, y]
    return out


def ep_meta(env, raw):
    try:
        return json.loads(json.dumps(raw.get_ep_meta(), default=str))
    except Exception as exc:  # report, do not hide
        return {"<error>": str(exc)}


def instruction(obs):
    return obs.get("annotation.human.task_description") if isinstance(obs, dict) else None


def actions_for(env, steps, seed=ACTION_SEED):
    space = env.action_space
    low = np.where(np.isfinite(space.low), space.low, -1.0)
    high = np.where(np.isfinite(space.high), space.high, 1.0)
    mid, half = (low + high) / 2, (high - low) / 2
    rng = np.random.default_rng(seed)
    return [mid + ACTION_SCALE * half * rng.uniform(-1, 1, size=space.shape) for _ in range(steps)]


def run_attempt(env, raw, actions):
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    snap = dict(instruction=instruction(obs), attributes=attributes(raw), ep_meta=ep_meta(env, raw),
                states=[np.asarray(raw.sim.get_state().flatten(), float)], success=[], check=[])
    for a in actions:
        step = env.step(a)
        info = step[-1] if isinstance(step, tuple) and isinstance(step[-1], dict) else {}
        snap["states"].append(np.asarray(raw.sim.get_state().flatten(), float))
        snap["success"].append(bool(info.get("success", False)))
        try:
            snap["check"].append(bool(raw._check_success()))
        except Exception as exc:
            snap["check"].append(f"error: {exc}")
    return snap


def compare(first, retry):
    states_a, states_b = first["states"], retry["states"]
    diverge, worst = None, 0.0
    for i, (a, b) in enumerate(zip(states_a, states_b)):
        d = float(np.max(np.abs(a - b))) if a.shape == b.shape else math.inf
        worst = max(worst, d)
        if diverge is None and d > STATE_TOL:
            diverge = i
    attr = diff(first["attributes"], retry["attributes"])
    meta = diff(first["ep_meta"], retry["ep_meta"])
    succ = [i for i, (x, y) in enumerate(zip(first["success"], retry["success"])) if x != y]
    check = [i for i, (x, y) in enumerate(zip(first["check"], retry["check"])) if x != y]
    same = not attr and not meta and diverge is None and not succ and not check and \
        first["instruction"] == retry["instruction"]
    return dict(same=same, instruction=[first["instruction"], retry["instruction"]], attribute_diffs=attr,
                ep_meta_diffs=meta, state_diverges_at_step=diverge, max_state_diff=worst,
                success_differs_at=succ[:10], check_differs_at=check[:10],
                success_steps=[sum(first["success"]), sum(retry["success"])],
                check_steps=[sum(x is True for x in first["check"]), sum(x is True for x in retry["check"])])


def check_task(env, task, scenes, steps, log_dir):
    """env: a fresh environment for `task`; PairedReset is installed here."""
    raw = raw_env(env)
    env = ce.install(env, task, Path(log_dir) / f"{task}_pairs.jsonl")
    acts = actions_for(env, steps)
    out = []
    for j in range(scenes):
        first = run_attempt(env, raw, acts)
        retry = run_attempt(env, raw, acts)
        out.append(dict(scene=j, **compare(first, retry)))
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
                rows = check_task(env, t, a.scenes, a.steps, log_dir)
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
                if r["instruction"][0] != r["instruction"][1]:
                    print(f"      instruction: {r['instruction']}")
                for k, (x, y) in list(r["attribute_diffs"].items())[:12]:
                    print(f"      attr {k}: {str(x)[:70]} -> {str(y)[:70]}")
                for k, (x, y) in list(r["ep_meta_diffs"].items())[:8]:
                    print(f"      ep_meta {k}: {str(x)[:70]} -> {str(y)[:70]}")
    out = dict(plan=PLAN, split=split, seed=seed, scenes=a.scenes, steps=a.steps, render=bool(a.render),
               action_scale=ACTION_SCALE, action_seed=ACTION_SEED, state_tol=STATE_TOL, tasks=results)
    Path(a.output).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"{n_same} of {len(tasks)} tasks SAME; wrote {a.output}")
    print("RESTORE CHECK DONE")


if __name__ == "__main__":
    main()
