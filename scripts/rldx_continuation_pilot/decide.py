"""Apply the pre-stated E3 rule after the RLDX-1 pilot (plan: docs/rldx_e3_pilot_plan.md).

Inputs: the pilot analysis (analyze_pilot.py) and two E3 sizing runs on RLDX-1
(`stage2_sizing.py e3`), one at the pilot's retry rate and one at its lower bound, each run with
the pilot's continuation values (decision interval and point) added to the default c grid.

Rule, on RLDX-1 over all 20 tasks, with P as the alarm (reset at P's alarm vs continuation to
1.25 H_k, paired by seed):
  - the direction is decided if reset - continuation has the same nonzero sign at both ends of
    c's decision interval, under both retry rates;
  - n = the largest pairs-needed (McNemar, two-sided 0.05, power 0.8) at those four points,
    rounded up to a multiple of 20 tasks, and at least 400 (20 per task);
  - if the direction is decided and n <= 1,000: REGISTER reset vs continuation as a two-sided
    superiority test with n pairs, and state the expected direction;
  - otherwise: ESTIMATION. E3 reports both arms and their paired difference with 95% CIs at
    1,000 pairs (50 per task), with no superiority claim.
  - cross-check (if --direct is given): a REGISTER verdict is downgraded to ESTIMATION when the
    direct paired estimate's 95% task-cluster interval lies entirely on the other side of zero
    from the expected direction.

Usage:
  python -I decide.py --pilot pilot.json --e3 e3_r.json --e3-low e3_rlow.json --output decision.json
"""
import argparse, json, math, sys
from pathlib import Path

CAP, FLOOR, TASKS, Z = 1000, 400, 20, 1.959964


def rows(e3, policy="rldx1", stratum="all"):
    cont = e3["policies"][policy]["by_type"][stratum]["continuation"]
    return {float(c): v["P"] for c, v in cont.items()}


def at(table, c):
    for key, value in table.items():
        if abs(key - c) < 1e-9:
            return value
    raise SystemExit(f"STOP: c = {c} is not in the E3 run; rerun it with the pilot's continuation values")


def crossover(table):
    """c* by linear interpolation where reset - continuation changes sign."""
    pts = sorted(table.items())
    for (c0, a), (c1, b) in zip(pts, pts[1:]):
        if a["delta"] == 0:
            return c0
        if a["delta"] * b["delta"] < 0:
            return round(c0 + (c1 - c0) * a["delta"] / (a["delta"] - b["delta"]), 4)
    return None


def num(x):
    return math.inf if x in (None, "Infinity", "inf") else float(x)


def decide(pilot, e3_runs, direct=None):
    c_lo, c_hat, c_hi = pilot["sizing_inputs"]["continuation"]
    if None in (c_lo, c_hat, c_hi):
        raise SystemExit("STOP: the pilot has no continuation estimate")
    per_run, signs, needed = {}, set(), []
    for name, e3 in e3_runs.items():
        t = rows(e3)
        ends = {c: at(t, c) for c in (c_lo, c_hat, c_hi)}
        per_run[name] = dict(retry_success=e3.get("retry_success"), crossover_c=crossover(t),
                             points={str(c): dict(reset=v["reset"], cont=v["cont"], delta=v["delta"], pairs=num(v["paired"]),
                                                  p10=v["p10"], p01=v["p01"]) for c, v in ends.items()})
        for c in (c_lo, c_hi):
            d = ends[c]["delta"]
            signs.add(0 if d == 0 else (1 if d > 0 else -1))
            needed.append(num(ends[c]["paired"]))
    decided = len(signs) == 1 and 0 not in signs
    n_max = max(needed)
    n_reg = None if not math.isfinite(n_max) else max(FLOOR, TASKS * math.ceil(n_max / TASKS))
    if decided and n_reg is not None and n_reg <= CAP:
        direction = "reset better" if signs == {1} else "continuation better"
        verdict = dict(decision="REGISTER", test="reset at P's alarm vs continuation to 1.25 H_k, two-sided McNemar, alpha 0.05",
                       pairs=n_reg, pairs_per_task=n_reg // TASKS, expected_direction=direction)
    else:
        main = per_run["retry_success"]["points"][str(c_hat)]
        psi, d = main["p10"] + main["p01"], main["delta"]
        hw = Z * math.sqrt(max(psi - d * d, 0.0) / CAP)
        reason = ("direction not decided across c's interval and the retry rates" if not decided
                  else f"needs {n_reg if n_reg is not None else 'infinitely many'} pairs, above {CAP}")
        verdict = dict(decision="ESTIMATION", pairs=CAP, pairs_per_task=CAP // TASKS, reason=reason,
                       expected_difference=d, expected_ci95_half_width=round(hw, 4))
    if direct is not None and direct.get("all"):
        ci = direct["all"]["reset_minus_cont"]["cluster95"]
        verdict["direct_estimate"] = dict(task_macro=direct["all"]["reset_minus_cont"]["task_macro"], cluster95=ci)
        if verdict["decision"] == "REGISTER" and ci and None not in ci:
            against = (ci[1] < 0) if verdict["expected_direction"] == "reset better" else (ci[0] > 0)
            if against:
                main = per_run["retry_success"]["points"][str(c_hat)]
                hw = Z * math.sqrt(max(main["p10"] + main["p01"] - main["delta"] ** 2, 0.0) / CAP)
                verdict = dict(decision="ESTIMATION", pairs=CAP, pairs_per_task=CAP // TASKS,
                               reason="the direct paired estimate contradicts the expected direction",
                               expected_difference=main["delta"], expected_ci95_half_width=round(hw, 4),
                               direct_estimate=verdict["direct_estimate"])
    return dict(plan="docs/rldx_e3_pilot_plan.md", continuation=dict(lo=c_lo, point=c_hat, hi=c_hi),
                runs=per_run, direction_decided=decided, pairs_needed_max=n_max, **verdict)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", required=True); p.add_argument("--e3", required=True); p.add_argument("--e3-low", required=True)
    p.add_argument("--direct", help="direct_e3.py output (cross-check)")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        sys.exit(f"error: output exists: {out}")
    pilot = json.loads(Path(a.pilot).read_text())
    runs = {"retry_success": json.loads(Path(a.e3).read_text()), "retry_success_low": json.loads(Path(a.e3_low).read_text())}
    for name, key in (("retry_success", "retry_success"), ("retry_success_low", "retry_success_low")):
        want = pilot["sizing_inputs"][key]
        if abs(float(runs[name]["retry_success"]) - float(want)) > 1e-9:
            sys.exit(f"STOP: {name} run used r = {runs[name]['retry_success']}, the pilot says {want}")
    res = decide(pilot, runs, json.loads(Path(a.direct).read_text()) if a.direct else None)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    for name, v in res["runs"].items():
        print(f"r = {v['retry_success']}: c* = {v['crossover_c']}")
        for c, x in v["points"].items():
            print(f"   c = {c}: continuation {x['cont']:.4f}, reset {x['reset']:.4f}, reset - cont {x['delta']:+.4f}, pairs {x['pairs']}")
    print(f"DECISION: {res['decision']} | " + json.dumps({k: v for k, v in res.items() if k not in ('runs', 'plan', 'continuation')}))


if __name__ == "__main__":
    main()
