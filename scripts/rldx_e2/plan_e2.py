"""Plan E2, closed-loop reset and retry on new RLDX-1 scenes (plan: docs/rldx_e2_plan.md).

Everything E2's analysis uses is fixed here, before any E2 rollout:
  - the collector settings: E1's (plan.json of E1, itself copied from the recorded collection), with a new
    base seed (and reserved seeds for rerunning a stopped shard's tasks), 2 x SCENES rollouts per task (a first attempt and a retry per scene) and --horizon-scale 1.25;
  - the reset rules: P and the fused monitors with E1's margins (lambda-hat set on E1's 400 new calibration
    successes, paths from the freeze manifest), and the success-tuned time-only rule R_opt with the per-task
    cutoffs the reset simulation chose on the recorded training rollouts at budget 1.25 H (r = 0.245);
  - the primary contrast (SAFE-LSTM + P against P), the latency charge, the bootstrap and its seed.
It stops if the inputs disagree (tasks, H_k, paths), or a seed was used by any RLDX-1 collection found under
--seed-roots, by E1, or by a smoke run.

Writes into --output: plan.json, collector_flags_e2.txt, tasks_e2_shard<k>.txt, shards.tsv.

Usage:
  python -I plan_e2.py --e1-plan <E1 root>/plan.json --e1-result <E1 root>/e1_result.json \\
      --freeze-manifest <e2_freeze_manifest.json> --reset-sim <reset.json> --seed-roots <dir> ... --output <E2 root>
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import importlib.util

REPO = Path(__file__).resolve().parents[2]
PLAN_DOC = "docs/rldx_e2_plan.md"
DEFAULT_SEED = 1401
SMOKE_SEED = 1102
RERUN_SEEDS = 9                     # a stopped shard's tasks rerun with seed + 10 i (i = 1..9): new scenes, new IDs
E1_SMOKE_SEED = 1101
SCENES = 50
BUDGET_SCALE = 1.25
RETRY_RATE = "0.245"                 # the reset simulation's retry rate whose R_opt cutoffs E2 uses
PORT_BASE = 20150
LATENCY_STEPS = 1                    # control steps charged per learned-monitor reset (50 ms at 20 Hz)
LATENCY_SENSITIVITY_STEPS = 8        # one full inference (0.4 s), reported alongside
BOOTSTRAP = 2000
SEED = 20261106                      # bootstrap seed, fixed with the plan
PRIMARY = ("safe_lstm_plus_P", "P")
CONTRASTS = [("safe_lstm_plus_P", "P"), ("P", "none"), ("safe_lstm_plus_P", "none"), ("R_opt", "P"),
             ("safe_lstm_plus_P", "R_opt"), ("hide_and_seek_plus_P", "P")]
DESCRIPTIVE = ("hide_and_seek_plus_P",)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions(e1_result, freeze, reset_sim, tasks):
    """The reset rules, every number fixed: margins from E1, paths from the freeze, cutoffs from the sim."""
    det = e1_result["detectors"]
    problems = []
    for key, m in (("safe", "safe_plus_P"), ("hide_and_seek", "hide_and_seek_plus_P")):
        fz = freeze["detectors"][m]["path"]
        e1p = det[key]["path"]
        if (float(fz["u"]), float(fz["s"])) != (float(e1p["u"]), float(e1p["s"])):
            problems.append(f"{key}: E1's path {e1p} differs from the freeze manifest's {fz}")
    per_task = reset_sim["runs"][RETRY_RATE]["per_task"]
    if sorted(per_task) != sorted(tasks):
        problems.append(f"reset simulation tasks {sorted(per_task)} != E2 tasks {sorted(tasks)}")
    cut = {t: float(per_task[t]["budgets"][str(BUDGET_SCALE)]["a_opt"]) for t in tasks if t in per_task}
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems))
    p_lam = float(e1_result["P"]["path"]["lam"])
    return dict(
        none=dict(kind="none"),
        P=dict(kind="fused", lam=p_lam, u=math.inf, s=0.0, latency_steps=0),
        safe_lstm_plus_P=dict(kind="fused", detector="safe", lam=float(det["safe"]["path"]["lam"]),
                              u=float(det["safe"]["path"]["u"]), s=float(det["safe"]["path"]["s"]),
                              latency_steps=LATENCY_STEPS),
        hide_and_seek_plus_P=dict(kind="fused", detector="hide_and_seek",
                                  lam=float(det["hide_and_seek"]["path"]["lam"]),
                                  u=float(det["hide_and_seek"]["path"]["u"]),
                                  s=float(det["hide_and_seek"]["path"]["s"]), latency_steps=LATENCY_STEPS,
                                  descriptive=True),
        R_opt=dict(kind="time", cutoffs=cut, latency_steps=0),
    )


def build(e1_plan, e1_result, freeze, reset_sim, seed, used, scenes=SCENES, shards=4, smoke_seed=SMOKE_SEED):
    settings = dict(e1_plan["settings"])
    tasks = sorted(e1_plan["tasks"])
    problems = []
    if sorted(freeze["tasks"]) != tasks:
        problems.append("freeze manifest tasks differ from E1's")
    for t in tasks:
        v = e1_plan["tasks"][t]
        if int(freeze["tasks"][t]["H_k"]) != int(v["H_inferences"]) or \
                int(v["H_inferences"]) != math.ceil(v["H"] / settings["replan_steps"]):
            problems.append(f"{t}: H_k disagrees between E1's plan and the freeze manifest")
    taken = {int(s): w for s, w in used.items()}
    for s in list(e1_plan["seeds"].values()) + [E1_SMOKE_SEED, int(settings["seed"])]:
        taken.setdefault(int(s), ["E1 plan / recorded collection"])
    seeds = [seed + 10 * i for i in range(RERUN_SEEDS + 1)]
    for name, s in [("E2", x) for x in seeds] + [("smoke", smoke_seed)]:
        if int(s) in taken:
            problems.append(f"{name} seed {s} was used before: {taken[int(s)][:3]}")
    if smoke_seed in seeds:
        problems.append("the smoke seed collides with an E2 seed")
    if e1_result["counts"]["test"]["successes"] < 1 or not e1_result["detectors"]["safe"]["H3"]:
        problems.append("E1 did not confirm SAFE-LSTM + P; E2's primary monitor would be unsupported")
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems))
    conds = conditions(e1_result, freeze, reset_sim, tasks)
    plan_e1 = _load("_e2_plan_e1", REPO / "scripts" / "rldx_e1" / "plan_e1.py")
    out_tasks, cost = {}, {}
    for t in tasks:
        v = e1_plan["tasks"][t]
        budget = int(math.ceil(round(v["H"] * BUDGET_SCALE, 6)))
        # Each scene: a first attempt and a retry, each ending at success or at the budget.
        per_attempt = v["p"] * v["mean_success_length"] + (1 - v["p"]) * budget
        cost[t] = 2 * scenes * per_attempt
        out_tasks[t] = dict(type=v["type"], H=v["H"], H_k=int(v["H_inferences"]),
                            rho_k=float(freeze["tasks"][t]["rho_k"]), budget_steps=budget, p=v["p"],
                            expected_env_steps=round(cost[t]))
    shard_tasks, loads = plan_e1.balance(tasks, cost, shards)
    shard_rows = [dict(set="e2", shard=k, gpu=k, port=PORT_BASE + k, tasks=s, expected_env_steps=round(loads[k]))
                  for k, s in enumerate(shard_tasks)]
    flags = plan_e1.pilot.flags({**settings, "seed": seed, "num_rollouts": 2 * scenes,
                                 "horizon_scale": BUDGET_SCALE}) + ["--no-continue-on-error"]
    return dict(plan=PLAN_DOC, source=e1_plan["source"], seed=seed, rerun_seeds=seeds[1:], smoke_seed=smoke_seed, scenes_per_task=scenes,
                rollouts_per_task=2 * scenes, budget_scale=BUDGET_SCALE, settings=settings, tasks=out_tasks,
                conditions=conds, primary=list(PRIMARY), contrasts=[list(c) for c in CONTRASTS],
                descriptive=list(DESCRIPTIVE), latency_sensitivity_steps=LATENCY_SENSITIVITY_STEPS,
                bootstrap=BOOTSTRAP, level=0.95, analysis_seed=SEED, retry_rate_for_cutoffs=float(RETRY_RATE),
                shards=shard_rows, used_seeds=sorted(taken),
                cost=dict(rollouts=2 * scenes * len(tasks), env_steps=round(sum(cost.values())))), flags


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--e1-plan", required=True)
    p.add_argument("--e1-result", required=True)
    p.add_argument("--freeze-manifest", required=True)
    p.add_argument("--reset-sim", required=True, help="reset.json of the reset simulation (R_opt cutoffs)")
    p.add_argument("--seed-roots", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--scenes", type=int, default=SCENES)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if (out / "plan.json").exists():
        raise SystemExit(f"error: {out} already has a plan")
    plan_e1 = _load("_e2_plan_e1_main", REPO / "scripts" / "rldx_e1" / "plan_e1.py")
    used = plan_e1.used_seeds(a.seed_roots)
    inputs = {k: getattr(a, k) for k in ("e1_plan", "e1_result", "freeze_manifest", "reset_sim")}
    loaded = {k: json.loads(Path(v).read_text()) for k, v in inputs.items()}
    plan, flags = build(loaded["e1_plan"], loaded["e1_result"], loaded["freeze_manifest"], loaded["reset_sim"],
                        a.seed, used, a.scenes, a.shards)
    plan["inputs"] = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in inputs.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "collector_flags_e2.txt").write_text("\n".join(flags) + "\n")
    for r in plan["shards"]:
        (out / f"tasks_e2_shard{r['shard']}.txt").write_text("\n".join(r["tasks"]) + "\n")
    (out / "shards.tsv").write_text("".join(f"e2\t{r['shard']}\t{r['gpu']}\t{r['port']}\n" for r in plan["shards"]))
    (out / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True, default=str) + "\n")
    c = plan["conditions"]
    print(f"E2 seed {a.seed} (smoke {plan['smoke_seed']}), {a.scenes} scenes x 20 tasks, two attempts each, "
          f"budget {BUDGET_SCALE} H | used RLDX-1 seeds: {plan['used_seeds']}")
    print(f"P lam {c['P']['lam']}; SAFE-LSTM + P lam {c['safe_lstm_plus_P']['lam']} u {c['safe_lstm_plus_P']['u']} "
          f"s {c['safe_lstm_plus_P']['s']}; H&S + P (descriptive) lam {c['hide_and_seek_plus_P']['lam']}; "
          f"R_opt cutoffs {sorted(set(c['R_opt']['cutoffs'].values()))}")
    for r in plan["shards"]:
        print(f"shard{r['shard']} GPU {r['gpu']} port {r['port']} ~{r['expected_env_steps']} steps: {r['tasks']}")
    print(f"cost: {plan['cost']['rollouts']} rollouts, ~{plan['cost']['env_steps']} env steps")
    print("PLAN OK")


if __name__ == "__main__":
    main()
