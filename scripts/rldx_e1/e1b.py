"""E1b, the second test of H3 on new RLDX-1 rollouts (plan: docs/rldx_e1b_plan.md).

E1 tested SAFE-LSTM + P and Hide-and-Seek + P with paths fixed on the validation set (60 calibration
successes). SAFE-LSTM passed; Hide-and-Seek did not, but with its path re-chosen on E1's new
calibration rollouts it gained 0.094 with 23 false alarms (R15). E1b keeps everything of E1 (models,
H_k, rho_k, sets of 20 + 20 per task, the rule (a)-(d), 2,000 bootstrap replicates, 39 + 39 null
fusions) except two things, both fixed here before any E1b rollout exists:
  - each detector's path (u, s) is chosen with confirm_lead.fit_fast on all 1,600 E1 rollouts
    (calibration and test sets: successes set each candidate's margin, failures choose); only the
    margin is set on E1b's new calibration successes, so Proposition 1 applies;
  - each null fusion's path is chosen on null scores of the same 1,600 rollouts, and the resampled
    null draws from E1's normalized SAFE-LSTM values (E1 used the validation set for both).

Commands:
  freeze   choose and write the paths (e1b_freeze.json). With --design-check, also an out-of-sample
           preview on E1's data alone: paths chosen on E1's calibration set, the rule applied to E1's
           test set. The preview is not a result and decides nothing.
  analyze  E1's analysis with the frozen E1b paths, after three gates:
             R   E1's gate on the recorded rollouts (models, H_k, rho_k unchanged);
             F   the E1 score files are the ones the freeze read, and the paths recomputed from them
                 equal the frozen ones;
             S   E1b's sets were planned with E1b's seeds and are disjoint from E1's.

Usage:
  python -I e1b.py freeze --freeze-manifest E1_FREEZE --e1-scores-dir E1/scores [--design-check] --output F.json
  python -I e1b.py analyze --freeze-manifest E1_FREEZE --e1b-freeze F.json --calibration-dir <SAFE20>/calibration/lstm \\
      --hs-scores-dir <freeze>/scores_run2 --e1-scores-dir E1/scores --e1b-scores-dir E1B/scores \\
      --plan E1B/plan.json --output OUT.json
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "xr1_lead_time"))
import confirm_lead as cl  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ae = _load("_e1b_analyze_e1", HERE / "analyze_e1.py")

PLAN = "docs/rldx_e1b_plan.md"
SEED = 20261112                 # fixed with the plan; streams as E1: 1 bootstrap, 2 nulls
SEEDS = ("0", "1", "2")
SETS = ("calibration", "test")
DEFAULT_SEEDS = {"calibration": 1601, "test": 1701}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _default(x):
    return x.tolist() if isinstance(x, np.ndarray) else float(x)


def e1_files(scores_dir):
    d = Path(scores_dir)
    names = [f"{s}_{f}_predictions.jsonl" for s in SETS for f in ae.FILES.values()] + ["safe_lstm_scoring.json"]
    return {n: sha256(d / n) for n in names}


def e1_data(scores_dir, man):
    """E1's calibration and test rows (both detectors' trajectories), with H_k and task types from the manifest."""
    rk, H, _ = ae.frozen(man)
    types = {t: v["type"] for t, v in man["tasks"].items()}
    out = {}
    for name in SETS:
        rows = ae.e1_rows(scores_dir, name, H)
        for r in rows:
            r["task_type"] = types[r["task_name"]]
        out[name] = rows
    return rk, H, out


def choose(rows, rk):
    """Each detector's fused path, chosen with fit_fast on rows (successes set the margin, failures choose)."""
    out = {}
    for k in ae.DETS:
        _, fu = cl.fit_fast(cl.Cohort(rows, k), rk)
        out[k] = dict(u=fu["u"], s=fu["s"], lam_at_fit=round(fu["lam"], 4))
    return out


def seed_pools(scores_dir, rows):
    """Per task, per seed: E1's normalized SAFE-LSTM values of the given rollouts (as e2_null_rules.seed_pools_from)."""
    d = Path(scores_dir)
    norms = json.loads((d / "safe_lstm_scoring.json").read_text())["normalizations"]
    want = {str(r["rollout_id"]) for r in rows}
    per = {s: defaultdict(list) for s in SEEDS}
    for name in SETS:
        for r in jsonl(d / f"{name}_safe_lstm_predictions.jsonl"):
            rid = str(r["rollout_id"])
            if rid not in want:
                continue
            for s in SEEDS:
                n = norms[s][r["task_name"]]
                per[s][r["task_name"]].append((np.asarray(r["per_seed_raw"][s], float) - n["location"]) / n["scale"])
    tasks = sorted({r["task_name"] for r in rows})
    if any(not per[s][t] for s in SEEDS for t in tasks):
        raise SystemExit("STOP: a task has no SAFE-LSTM values for the resampled null")
    return {t: [np.concatenate(per[s][t]) for s in SEEDS] for t in tasks}


def summary(res, boot, dec, max_fa):
    lines = [f"P: false alarms {res['P']['test']['fp']}, lead {res['P']['test']['mean_lead']} | max false alarms {max_fa}"]
    for k in ae.DETS:
        r, b, d = res[k], boot[k], dec[k]
        lines.append(f"{k:14s} FA {r['test']['fp']:3d} | gain {r['gain']:+.4f} CI [{b['gain_ci'][0]:+.4f}, "
                     f"{b['gain_ci'][1]:+.4f}] | matched {r['matched_gain']:+.4f} CI [{b['matched_ci'][0]:+.4f}, "
                     f"{b['matched_ci'][1]:+.4f}] | null p {d['null_p']} | a {d['a_gain_ci_above_zero']} b "
                     f"{d['b_matched_ci_above_zero']} c {d['c_false_alarms_within_bound']} d {d['d_beats_null_fusions']} "
                     f"-> H3 {'HOLDS' if d['H3'] else 'does not hold'}")
    return lines


# ---- freeze ----------------------------------------------------------------------------------

def freeze(a):
    out = Path(a.output)
    if out.exists():
        raise SystemExit(f"error: output exists: {out}")
    man = json.loads(Path(a.freeze_manifest).read_text())
    rk, H, data = e1_data(a.e1_scores_dir, man)
    rows = data["calibration"] + data["test"]
    counts = {n: dict(successes=sum(not r["failed"] for r in v), failures=sum(r["failed"] for r in v))
              for n, v in data.items()}
    paths = choose(rows, rk)
    for k, p in paths.items():
        print(f"{k:14s} path chosen on {len(rows)} E1 rollouts: u {p['u']}, s {p['s']} (margin at fit {p['lam_at_fit']}); "
              f"E1 froze u {man['detectors'][ae.DETS[k]]['path']['u']}, s {man['detectors'][ae.DETS[k]]['path']['s']}",
              flush=True)
    result = dict(plan=PLAN, status="frozen before any E1b rollout",
                  e1_freeze=dict(path=str(Path(a.freeze_manifest).resolve()), sha256=sha256(a.freeze_manifest)),
                  e1_scores=dict(path=str(Path(a.e1_scores_dir).resolve()), files=e1_files(a.e1_scores_dir)),
                  e1_counts=counts, chosen_on="E1 calibration and test sets (fit_fast; successes set the margin, failures choose)",
                  paths=paths, null_paths_and_pools="E1 calibration and test sets", analysis_seed=SEED,
                  e1b_seeds=dict(DEFAULT_SEEDS))
    if a.design_check:
        cal, te = data["calibration"], data["test"]
        pc = choose(cal, rk)
        res, boot, refs, max_fa, dec = ae.analyze(cal, te, rk, pc, cl.Cohort(cal, "safe"),
                                                  seed_pools(a.e1_scores_dir, cal), B=a.replicates, seed=SEED)
        lines = summary(res, boot, dec, max_fa)
        print("Design check (not a result): paths chosen on E1's calibration set, rule applied to E1's test set")
        for line in lines:
            print("  " + line)
        result["design_check"] = dict(
            note="out-of-sample preview on E1's data; not a result and not part of the decision",
            paths=pc, replicates=a.replicates, lines=lines,
            detectors={k: dict(gain=res[k]["gain"], matched_gain=res[k]["matched_gain"], fp=res[k]["test"]["fp"],
                               **boot[k], **dec[k]) for k in ae.DETS})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=_default) + "\n")
    print(f"wrote {out} (sha256 {sha256(out)})")
    print("FREEZE OK")
    return 0


# ---- analyze ---------------------------------------------------------------------------------

def analyze(a):
    out_path = Path(a.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    man = json.loads(Path(a.freeze_manifest).read_text())
    fz = json.loads(Path(a.e1b_freeze).read_text())
    plan = json.loads(Path(a.plan).read_text())
    problems, checks = [], {}

    # R: E1's gate on the recorded rollouts.
    rows, H0, tasks, train, cal_s, test = ae.recorded_rows(a.calibration_dir, a.hs_scores_dir)
    pr, gate = ae.gate_recorded(man, H0, tasks, train, cal_s, test)
    checks["R_recorded"] = dict(gate=gate, ok=not pr)
    problems += pr

    # F: the freeze's inputs and paths.
    if fz["e1_freeze"]["sha256"] != sha256(a.freeze_manifest):
        problems.append("F: the E1 freeze manifest differs from the one the E1b freeze read")
    files = e1_files(a.e1_scores_dir)
    if files != fz["e1_scores"]["files"]:
        problems.append("F: E1's score files differ from the ones the E1b freeze read")
    rk, H, data = e1_data(a.e1_scores_dir, man)
    e1_rows_all = data["calibration"] + data["test"]
    again = choose(e1_rows_all, rk)
    same = {k: (again[k]["u"], again[k]["s"], again[k]["lam_at_fit"]) == (fz["paths"][k]["u"], fz["paths"][k]["s"],
                                                                          fz["paths"][k]["lam_at_fit"]) for k in ae.DETS}
    if not all(same.values()):
        problems.append(f"F: paths recomputed from E1 differ from the frozen ones {same}")
    checks["F_freeze"] = dict(files_equal=files == fz["e1_scores"]["files"], paths_equal=same)

    # S: E1b's sets.
    if plan["seeds"] != fz["e1b_seeds"]:
        problems.append(f"S: plan seeds {plan['seeds']} != frozen E1b seeds {fz['e1b_seeds']}")
    types = {t: v["type"] for t, v in man["tasks"].items()}
    new = {}
    for name in SETS:
        new[name] = ae.e1_rows(a.e1b_scores_dir, name, H)
        for r in new[name]:
            r["task_type"] = types[r["task_name"]]
    overlap = {str(r["rollout_id"]) for v in new.values() for r in v} & {str(r["rollout_id"]) for r in e1_rows_all}
    if overlap:
        problems.append(f"S: {len(overlap)} E1b rollouts are E1 rollouts")
    checks["S_sets"] = dict(seeds=plan["seeds"], overlap_with_e1=len(overlap),
                            counts={n: ae.limit_check(v, n) for n, v in new.items()})
    for k, v in checks.items():
        print(f"gate {k}: {v}", flush=True)
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems))

    paths = {k: dict(u=fz["paths"][k]["u"], s=fz["paths"][k]["s"]) for k in ae.DETS}
    devC = cl.Cohort(e1_rows_all, "safe")
    pools = seed_pools(a.e1_scores_dir, e1_rows_all)
    res, boot, refs, max_fa, dec = ae.analyze(new["calibration"], new["test"], rk, paths, devC, pools,
                                              B=a.replicates, seed=SEED)
    out = dict(
        plan=PLAN, status="E1b (second test of H3; rule and paths fixed before collection)", seed=SEED,
        replicates=a.replicates, level=ae.LEVEL, k_null=ae.K_NULL, mc_alpha=ae.MC_ALPHA, frozen_paths=paths,
        e1b_freeze=dict(path=str(Path(a.e1b_freeze).resolve()), sha256=sha256(a.e1b_freeze)), e1b_seeds=plan["seeds"],
        checks=checks, max_false_alarms=max_fa,
        P={k: v for k, v in res["P"].items() if k != "lead"},
        detectors={k: dict({kk: vv for kk, vv in res[k].items() if kk != "lead"}, **boot[k], **dec[k]) for k in ae.DETS},
        null_references=refs, by_type=ae.by_type(new["test"], res, rk), seconds=ae.seconds(new["test"], res))
    out_path.write_text(json.dumps(out, indent=1, default=_default) + "\n")
    for line in summary(res, boot, dec, max_fa):
        print(line)
    print("by type:", out["by_type"], "| seconds:", out["seconds"])
    print("ANALYSIS DONE")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--freeze-manifest", required=True, help="E1's e2_freeze_manifest.json")
    f.add_argument("--e1-scores-dir", required=True)
    f.add_argument("--design-check", action="store_true")
    f.add_argument("--replicates", type=int, default=ae.B_DEFAULT)
    f.add_argument("--output", required=True)
    z = sub.add_parser("analyze")
    for name in ("--freeze-manifest", "--e1b-freeze", "--calibration-dir", "--hs-scores-dir", "--e1-scores-dir",
                 "--e1b-scores-dir", "--plan", "--output"):
        z.add_argument(name, required=True)
    z.add_argument("--replicates", type=int, default=ae.B_DEFAULT)
    a = p.parse_args(argv)
    return {"freeze": freeze, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
