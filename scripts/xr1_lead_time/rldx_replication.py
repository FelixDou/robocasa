"""RLDX-1 offline replication of the lead-over-timer analysis (plan: docs/rldx_lead_replication_plan.md).

Exploratory, second policy, no new rollouts. Reads the per-seed normalized SAFE trajectories
of the RLDX-1 SAFE20 run and its calibration manifest, then compares SAFE + P with the
per-task quantile timer P, using the same fit as E1 (confirm_lead.fit_fast).

Usage:
  python3 -I rldx_replication.py --calibration-dir <root>/calibration/lstm --model lstm --output OUT.json
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl
import lead_time
import quantile_timer as qt

SEEDS = (0, 1, 2)
Q, ALPHA, NULL_DRAWS, NULL_SEED = 0.95, 0.05, 20, 1


def load_seed(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["rollout_id"] in rows:
                raise ValueError(f"Duplicate rollout {r['rollout_id']} in {path}")
            rows[r["rollout_id"]] = r
    return rows


def assemble(cal_dir, model):
    """Seed-mean of each seed's running max of the full normalized trajectory (C2 checks)."""
    seeds = {s: load_seed(Path(cal_dir) / f"{model}_seed{s}" / "normalized_scores.jsonl") for s in SEEDS}
    ref = seeds[SEEDS[0]]
    for s, rows in seeds.items():
        if set(rows) != set(ref):
            raise ValueError(f"Seed {s} covers different rollouts")
        for rid, r in rows.items():
            o = ref[rid]
            if (r["task_name"], bool(r["failed"]), r["split"], len(r["full_scores"])) != \
               (o["task_name"], bool(o["failed"]), o["split"], len(o["full_scores"])):
                raise ValueError(f"Seed {s} disagrees with seed {SEEDS[0]} on {rid}")
    out = []
    for rid, o in sorted(ref.items()):
        safe = np.mean([np.maximum.accumulate(np.asarray(seeds[s][rid]["full_scores"], float)) for s in SEEDS], axis=0)
        out.append(dict(rollout_id=rid, task_name=o["task_name"], task_type=o.get("task_type"), failed=bool(o["failed"]),
                        split=o["split"], L=len(safe), safe=safe, environment_seed=o.get("environment_seed"),
                        environment_reset_index=o.get("environment_reset_index"),
                        timeout=o.get("online_task_timeout_inferences")))
    return out


def horizons(rows):
    H = defaultdict(int)
    for r in rows:
        if r["split"] == "train" and r["failed"]:
            H[r["task_name"]] = max(H[r["task_name"]], r["L"])
    return dict(H)


def finish(rows, H):
    for r in rows:
        r["horizon"] = H[r["task_name"]]
        r["timer"] = np.arange(1, r["L"] + 1) / r["horizon"]
        r["risk"] = np.zeros(r["L"])        # no time-risk curve here; unused by the detectors below
    return rows


def failure_check(rows, label):
    F = [r for r in rows if r["failed"]]
    frac = np.array([r["L"] / r["horizon"] for r in F])
    return dict(split=label, failures=len(F), at_limit=round(float(np.mean(frac == 1.0)), 4) if len(F) else None,
                at_least_0p95=round(float(np.mean(frac >= 0.95)), 4) if len(F) else None,
                quantiles_L_over_H=np.round(np.quantile(frac, [0, 0.1, 0.25, 0.5, 1]), 4).tolist() if len(F) else None,
                longer_than_H=int(np.sum(frac > 1.0)),
                early_ending=[dict(rollout_id=r["rollout_id"], task=r["task_name"], L=r["L"], H=r["horizon"])
                              for r in F if r["L"] < 0.95 * r["horizon"]][:50])


def dev_fractions(train, tasks):
    fr = defaultdict(list)
    for r in train:
        if not r["failed"]:
            fr[r["task_name"]].append(min(1.0, r["L"] / r["horizon"]))
    return {t: fr.get(t, []) for t in tasks}


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    rows = assemble(args.calibration_dir, args.model)
    H = horizons(rows)
    tasks = sorted({r["task_name"] for r in rows})
    missing = [t for t in tasks if t not in H]
    if missing:
        raise ValueError(f"Tasks without training failures: {missing}")
    finish(rows, H)
    train = [r for r in rows if r["split"] == "train"]
    cal_s = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "evaluation"]
    if any(r["failed"] for r in cal_s):
        raise ValueError("Calibration split contains failures")
    counts = defaultdict(lambda: dict(train_s=0, train_f=0, cal_s=0, test_s=0, test_f=0))
    for r in rows:
        k = {"train": "train", "calibration": "cal", "evaluation": "test"}.get(r["split"])
        if k:
            counts[r["task_name"]][f"{k}_{'f' if r['failed'] else 's'}"] += 1
    timeouts = sorted({(r["task_name"], r["timeout"]) for r in rows if r["timeout"] is not None})
    checks = dict(
        C1=[failure_check(train, "train"), failure_check(test, "test")],
        C2=dict(tasks=len(tasks), per_task=dict(counts), splits={s: sum(r["split"] == s for r in rows) for s in sorted({r["split"] for r in rows})},
                horizon_vs_recorded_timeout=[dict(task=t, H=H[t], timeout=to) for t, to in timeouts]),
    )
    early = checks["C1"][1]
    checks["early_ending_flag"] = bool(early["failures"] and 1 - early["at_least_0p95"] > 0.10)
    # Calibration cohort: held-out successes (FPR margin) + training failures (path choice only).
    cal = cal_s + [r for r in train if r["failed"]]
    rk = qt.rho(dev_fractions(train, tasks), Q)
    calC, teC = cl.Cohort(cal), cl.Cohort(test)
    tm, fu = cl.fit_fast(calC, rk)
    eP, lP = cl.evaluate_fast(teC, rk, tm); eF, lF = cl.evaluate_fast(teC, rk, fu)
    Fte = [r for r in test if r["failed"]]
    d = lF - lP
    res = dict(P=dict(**cl.det_json(tm), **eP), safe_plus_P=dict(**cl.det_json(fu), **eF),
               gain=round(float(d.mean()), 4), gain_test_only=qt.boot(d, Fte), by_type={})
    for s in sorted({r["task_type"] for r in test}):
        m = np.array([r["task_type"] == s for r in Fte])
        sub = teC.take(np.array([r["task_type"] == s for r in test]))
        eP_s, _ = cl.evaluate_fast(sub, rk, tm); eF_s, _ = cl.evaluate_fast(sub, rk, fu)
        res["by_type"][str(s)] = dict(P_lead=eP_s["mean_lead"], P_fpr=eP_s["fpr"], fused_lead=eF_s["mean_lead"],
                                     fused_fpr=eF_s["fpr"], gain=round(float(d[m].mean()), 4),
                                     gain_test_only=qt.boot(d[m], [r for r, k in zip(Fte, m) if k]))
    sec = {}
    for name, keys in (("global_timer", ["timer"]), ("safe_plus_global_timer", ["safe", "timer"])):
        th = lead_time.search(cal, keys); e = lead_time.ev(test, th)
        sec[name] = dict(thresholds={k: float(v) for k, v in th.items()}, test=lead_time.strip(e),
                         fpr_wilson95=cl.wilson(e["fp"], sum(not r["failed"] for r in test)),
                         gain_vs_P=qt.boot(e["lead_vec"] - lP, Fte))
    rng = np.random.default_rng(NULL_SEED); null = []
    for _ in range(NULL_DRAWS):
        for r in cal + test:
            r["noise"] = np.maximum.accumulate(rng.standard_normal(r["L"]))
        _, f = cl.fit_fast(cl.Cohort(cal, "noise"), rk)
        e, l = cl.evaluate_fast(cl.Cohort(test, "noise"), rk, f)
        null.append(dict(gain=round(float(np.mean(l - lP)), 4), fpr=e["fpr"]))
    out = dict(plan="docs/rldx_lead_replication_plan.md", status="exploratory (second policy, offline)",
               model=args.model, calibration_dir=str(Path(args.calibration_dir).resolve()), q=Q, alpha=ALPHA,
               counts=dict(train=len(train), calibration_successes=len(cal_s), test=len(test),
                           test_failures=len(Fte), path_choice_failures=sum(r["failed"] for r in cal)),
               checks=checks, primary=res, secondary=sec,
               null_noise=dict(draws=NULL_DRAWS, mean_gain=round(float(np.mean([n["gain"] for n in null])), 4),
                               max_gain=max(n["gain"] for n in null), per_draw=null))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    for c in checks["C1"]:
        print(f"C1 {c['split']:5s} failures {c['failures']}: at limit {c['at_limit']}  >=0.95H {c['at_least_0p95']}  "
              f"L/H quantiles {c['quantiles_L_over_H']}  L>H {c['longer_than_H']}")
    print("early-ending flag:", checks["early_ending_flag"], "| tasks", len(tasks), "| splits", checks["C2"]["splits"])
    print(f"P          lead {eP['mean_lead']:.4f}  FPR {eP['fpr']} {eP['fpr_wilson95']}  recall {eP['recall']}")
    print(f"SAFE + P   lead {eF['mean_lead']:.4f}  FPR {eF['fpr']} {eF['fpr_wilson95']}  recall {eF['recall']}  "
          f"gain {res['gain']:+.4f} CI {res['gain_test_only']['ci95']}")
    for k, v in sec.items():
        print(f"{k:22s} lead {v['test']['mean_lead']}  FPR {v['test']['fpr']}  recall {v['test']['recall']}  vs P {v['gain_vs_P']}")
    for k, v in res["by_type"].items():
        print(f"{k:10s} P {v['P_lead']} ({v['P_fpr']})  SAFE+P {v['fused_lead']} ({v['fused_fpr']})  gain {v['gain']:+.4f} CI {v['gain_test_only']['ci95']}")
    print(f"noise + P: mean gain {out['null_noise']['mean_gain']:+.4f}, max {out['null_noise']['max_gain']:+.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration-dir", required=True)
    p.add_argument("--model", required=True, choices=("lstm", "indep"))
    p.add_argument("--output", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
