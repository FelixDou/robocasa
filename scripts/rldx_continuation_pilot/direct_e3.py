"""Direct paired estimate of E3's contrast on the RLDX-1 pilot scenes (plan: docs/rldx_e3_pilot_plan.md).

Exploratory. Each same-scene pair holds two independent attempts at one scene: the pilot rollout
(run to ceil(1.25 H_k)) and the original rollout (run to H_k). With P's alarm at the fraction
f_k = min(1, rho_k + lambda) of the time limit (P refitted exactly as in the E3 sizing, and
checked against its RLDX-1 lead of 0.2711):
  - continuation: the pilot rollout succeeds by 1.25 H_k;
  - reset at P's alarm: the pilot rollout succeeds before the alarm, or, once alarmed, the
    original attempt (the retry) succeeds within the remaining budget 1.25 H_k - alarm. A retry
    still running at its own alarm would be reset again; no third attempt exists, so it counts
    as a failure (the share is reported);
  - no monitor (descriptive): the pilot rollout succeeds by H_k, or the retry succeeds within
    the 0.25 H_k left after the time limit.
Pairs come from analyze_pilot.py's scene check; if that gate failed, the estimate is void.

Usage:
  python -I direct_e3.py --pilot-json pilot.json --pilot-collection <dir> --selection selection.json \
      --rldx-calibration-dir <SAFE20 calibration/lstm> --output direct.json [--bootstrap 2000]
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED_P_LEAD = 0.2711          # RLDX-1 P lead in the E3 sizing runs (stage2_sizing.md)
FRAME_TOL = 10.0


def p_fractions(cal_dir):
    sys.path.insert(0, str(HERE.parent / "xr1_lead_time"))
    import confirm_lead as cl
    import stage2_sizing as s2
    rows, train, tasks, rk = s2.rldx_rows(cal_dir, "lstm")
    cal_s = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "evaluation"]
    tm, _ = cl.fit_fast(cl.Cohort(cal_s + [r for r in train if r["failed"]]), rk)
    eP, _ = cl.evaluate_fast(cl.Cohort(test), rk, tm)
    if abs(eP["mean_lead"] - EXPECTED_P_LEAD) > 1e-4:
        sys.exit(f"STOP: P's RLDX-1 lead {eP['mean_lead']} does not reproduce {EXPECTED_P_LEAD}")
    return {t: min(1.0, float(rk[t]) + float(tm["lam"])) for t in tasks}, float(tm["lam"]), eP["mean_lead"]


def arms(pilot_success, Lp, orig_success, Lo, H, f):
    """Outcomes of one same-scene pair under continuation, reset at P's alarm, and no monitor."""
    limit = int(math.ceil(round(1.25 * H, 6)))
    alarm = int(math.ceil(f * H - 1e-9))
    cont = pilot_success
    third = False
    if pilot_success and Lp <= alarm:
        reset = True
    else:
        left = limit - alarm
        window = min(left, alarm)            # a second alarm resets the retry again
        reset = orig_success and Lo <= window
        third = orig_success and window < Lo <= left
    none = (pilot_success and Lp <= H) or (orig_success and Lo <= limit - H)
    return cont, reset, none, third


def bootstrap(per, tasks, fn, reps, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        vals.append(fn(rng.choice(tasks, len(tasks))))
    return np.round(np.percentile(vals, [2.5, 97.5]), 4).tolist()


def run(args, fractions=None):
    out = Path(args.output)
    if out.exists():
        sys.exit(f"error: output exists: {out}")
    pilot = json.loads(Path(args.pilot_json).read_text())
    if not pilot["gates"]["scene"]["passed"]:
        out.write_text(json.dumps(dict(status="void: scene gate failed"), indent=1) + "\n")
        print("VOID: the scene gate failed, so the pairs are not same-scene retries"); return
    sel = json.loads(Path(args.selection).read_text())
    if fractions is None:
        fractions, lam, p_lead = p_fractions(args.rldx_calibration_dir)
    else:
        lam, p_lead = None, None
    if set(fractions) != set(sel["tasks"]):
        sys.exit(f"STOP: P tasks differ from the pilot tasks: {sorted(set(fractions) ^ set(sel['tasks']))}")
    per = defaultdict(lambda: defaultdict(int))
    for shard in sorted(Path(args.pilot_collection).glob("shard*")):
        if not (shard / "manifest.jsonl").is_file():
            continue
        for line in (shard / "manifest.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line); t, i = r["task_name"], str(int(r["environment_reset_index"]))
            d = pilot["scene_pairs"].get(f"{t}/{i}")
            if d is None or d > FRAME_TOL:
                continue
            o = sel["tasks"][t]["scenes"][i]
            cont, reset, none, third = arms(not r["failed"], int(r["num_env_steps"]), not o["failed"], o["L"],
                                            sel["tasks"][t]["H"], fractions[t])
            x = per[t]
            x["pairs"] += 1; x["cont"] += cont; x["reset"] += reset; x["none"] += none; x["third"] += third
            x["reset_only"] += reset and not cont; x["cont_only"] += cont and not reset
    tasks = sorted(per)
    types = {t: sel["tasks"][t]["type"] for t in tasks}

    def summary(ts, reps):
        n = sum(per[t]["pairs"] for t in ts)
        rate = {k: sum(per[t][k] for t in ts) / n for k in ("cont", "reset", "none")}
        macro = {k: float(np.mean([per[t][k] / per[t]["pairs"] for t in ts])) for k in ("cont", "reset", "none")}
        diff = lambda draw: float(np.mean([(per[t]["reset"] - per[t]["cont"]) / per[t]["pairs"] for t in draw]))
        b, c = sum(per[t]["reset_only"] for t in ts), sum(per[t]["cont_only"] for t in ts)
        return dict(tasks=len(ts), pairs=n,
                    pooled={k: round(v, 4) for k, v in rate.items()},
                    task_macro={k: round(v, 4) for k, v in macro.items()},
                    reset_minus_cont=dict(task_macro=round(macro["reset"] - macro["cont"], 4),
                                          cluster95=bootstrap(per, ts, diff, reps) if len(ts) > 1 else None,
                                          discordant=dict(reset_only=b, cont_only=c), mcnemar_exact_p=mcnemar(b, c)),
                    third_attempt_needed=sum(per[t]["third"] for t in ts))

    res = dict(plan="docs/rldx_e3_pilot_plan.md", status="exploratory", P=dict(lam=lam, lead=p_lead,
               alarm_fraction={t: round(fractions[t], 4) for t in tasks}),
               all=summary(tasks, args.bootstrap),
               by_type={k: summary([t for t in tasks if types[t] == k], args.bootstrap)
                        for k in sorted(set(types.values()))},
               per_task={t: dict(type=types[t], alarm_fraction=round(fractions[t], 4), **dict(per[t])) for t in tasks})
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    for name, s in [("all", res["all"])] + list(res["by_type"].items()):
        d = s["reset_minus_cont"]
        print(f"{name:9s} pairs {s['pairs']:4d} | task-macro: continuation {s['task_macro']['cont']:.3f}, reset {s['task_macro']['reset']:.3f}, "
              f"no monitor {s['task_macro']['none']:.3f} | reset - cont {d['task_macro']:+.4f} {d['cluster95']} "
              f"| discordant {d['discordant']} McNemar p {d['mcnemar_exact_p']} | third attempt {s['third_attempt_needed']}")


def mcnemar(b, c):
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, p), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot-json", required=True); p.add_argument("--pilot-collection", required=True)
    p.add_argument("--selection", required=True); p.add_argument("--rldx-calibration-dir", required=True)
    p.add_argument("--output", required=True); p.add_argument("--bootstrap", type=int, default=2000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
