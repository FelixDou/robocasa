"""Per-seed RLDX-1 rows for RSS Table II (score + P, mean and std over 3 seeds).

Exploratory, no new rollouts. The protocol is report sections 15.15 and 15.22
(rldx_replication.py, rldx_hide_and_seek.py): the same rows, splits, H_k, P (q = 0.95) and
fused fit (confirm_lead.fit_fast), with held-out calibration successes setting the FPR and
training failures choosing the path. Only the scores change: each seed's score is used alone
instead of the seed average.

Scores:
  safe_<model>       SAFE head <model> (lstm, indep), seed s: the running maximum of seed s's
                     normalized trajectory (the seed average is their mean, as in 15.15).
  score_cohorts      every detector in each --scores-dir (Hide-and-Seek and its controls,
                     OOD baselines): seed s's normalized running maximum,
                     (running max of per_seed_raw[s] - location) / scale (the seed average
                     is their mean, as stored in safe_scores).

For each score, the seed average and each seed are fused with P and evaluated on the test set:
FPR, mean lead l, gain over P and TWA = (l + 1 - FPR) / 2. Gates, before any mean is printed:
  - the stored seed-averaged trajectories are reproduced (max diff <= 1e-9);
  - P and every seed-averaged score + P reproduce the stored results exactly
    (--reference-hs: rldx_hide_and_seek.json for safe_lstm, Hide-and-Seek and controls;
     --reference-safe MODEL=rldx_lead_MODEL.json for the SAFE heads).
Scores without a stored result (e.g. new OOD baselines) are reported without that gate.

Usage:
  python3 -I rldx_per_seed.py --safe-dir lstm=<SAFE20>/calibration/lstm indep=<SAFE20>/calibration/indep \\
      --inputs-dir IN --scores-dir HS_SCORES OOD_SCORES --reference-hs rldx_hide_and_seek.json \\
      --reference-safe lstm=rldx_lead_lstm.json indep=rldx_lead_indep.json --output OUT.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_hide_and_seek as chs  # noqa: E402
import confirm_lead as cl  # noqa: E402
import quantile_timer as qt  # noqa: E402
import rldx_hide_and_seek as hs  # noqa: E402
import rldx_replication as rr  # noqa: E402

KEYS = ("fpr", "lead", "gain", "twa")


def attach_safe(rows, model, cal_dir):
    """Per-seed running maxima of one SAFE head; checks rollouts, labels and lengths."""
    seeds = {s: rr.load_seed(Path(cal_dir) / f"{model}_seed{s}" / "normalized_scores.jsonl") for s in rr.SEEDS}
    for s, data in seeds.items():
        if set(data) != {r["rollout_id"] for r in rows}:
            raise ValueError(f"{model} seed {s} covers different rollouts")
    for r in rows:
        per = {}
        for s in rr.SEEDS:
            o = seeds[s][r["rollout_id"]]
            if (o["task_name"], bool(o["failed"]), o["split"], len(o["full_scores"])) != \
               (r["task_name"], r["failed"], r["split"], r["L"]):
                raise ValueError(f"{model} seed {s} disagrees with the 15.15 rows on {r['rollout_id']}")
            per[s] = np.maximum.accumulate(np.asarray(o["full_scores"], float))
        r[f"safe_{model}"] = np.mean([per[s] for s in rr.SEEDS], axis=0)
        for s in rr.SEEDS:
            r[f"safe_{model}@{s}"] = per[s]
    return [f"safe_{model}"]


def attach_cohorts(groups, scores_dir):
    """score_cohorts detectors: per-seed normalized running maxima and their stored average."""
    scores_dir = Path(scores_dir)
    norms = json.loads((scores_dir / "normalizations.json").read_text())
    files = {name: chs.load_scores(scores_dir / f"{name}_predictions.jsonl") for name in groups}
    names = sorted(set.intersection(*(set(f) for f in files.values())))
    if not names:
        raise ValueError(f"{scores_dir}: no detector present in every cohort file")
    worst = 0.0
    for group, rows in groups.items():
        for r in rows:
            for name in names:
                row = files[group][name].get(str(r["rollout_id"]))
                if row is None:
                    raise ValueError(f"{scores_dir}/{group}: no {name} trajectory for {r['rollout_id']}")
                if bool(row["failed"]) != r["failed"] or row["task_name"] != r["task_name"]:
                    raise ValueError(f"{scores_dir}/{group}: label or task differs for {r['rollout_id']} ({name})")
                stored = np.asarray(row["safe_scores"], float)
                if len(stored) != r["L"]:
                    raise ValueError(f"{scores_dir}/{group}: {name} has {len(stored)} inferences for {r['rollout_id']}")
                per = {}
                for s in sorted(row["per_seed_raw"], key=int):
                    n = norms[name][s][r["task_name"]]
                    per[int(s)] = (np.maximum.accumulate(np.asarray(row["per_seed_raw"][s], float)) - n["location"]) / n["scale"]
                if sorted(per) != list(rr.SEEDS):
                    raise ValueError(f"{name}: seeds {sorted(per)} instead of {list(rr.SEEDS)}")
                mean = np.mean([per[s] for s in rr.SEEDS], axis=0)
                worst = max(worst, float(np.max(np.abs(mean - stored))))
                r[name] = stored
                for s in rr.SEEDS:
                    r[f"{name}@{s}"] = per[s]
    if worst > 1e-9:
        raise ValueError(f"{scores_dir}: seed-averaged trajectories differ from the stored ones (max {worst})")
    print(f"{scores_dir}: {names}, stored seed averages reproduced (max diff {worst:.2g})", flush=True)
    return names


def fused(cal, test, key, rk, lP):
    _, fu = cl.fit_fast(cl.Cohort(cal, key), rk)
    e, lF = cl.evaluate_fast(cl.Cohort(test, key), rk, fu)
    fpr, lead = float(e["fpr"]), float(e["mean_lead"])
    return {"fpr": fpr, "lead": lead, "gain": round(float(np.mean(lF - lP)), 4), "twa": (lead + 1.0 - fpr) / 2.0}


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    safe_dirs = dict(item.split("=", 1) for item in args.safe_dir)
    if "lstm" not in safe_dirs:
        raise SystemExit("error: --safe-dir needs lstm=<dir> (it defines the 15.15 rows)")
    rows, H, tasks = hs.load_rows(safe_dirs["lstm"])
    inputs = Path(args.inputs_dir)
    bundle = json.loads((inputs / "rldx_bundle.json").read_text())
    if bundle["task_horizons"] != {t: int(H[t]) for t in tasks}:
        raise ValueError("The scoring bundle's horizons differ from H_k recomputed now")
    train = [r for r in rows if r["split"] == "train"]
    cal_s = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "evaluation"]
    for name, part in (("train", train), ("calibration", cal_s), ("test", test)):
        if sorted(r["rollout_id"] for r in part) != json.loads((inputs / f"{name}_ids.json").read_text()):
            raise ValueError(f"{name} IDs differ from the prepared ID list")
    train_f = [r for r in train if r["failed"]]
    cal = cal_s + train_f

    detectors = []
    for model, cal_dir in safe_dirs.items():
        detectors += attach_safe(rows, model, cal_dir)
    if any(np.max(np.abs(r["safe_lstm"] - r["safe"])) > 0 for r in rows):
        raise ValueError("safe_lstm seed average differs from the 15.15 rows")
    for scores_dir in args.scores_dir:
        detectors += attach_cohorts({"calibration": cal_s, "train": train_f, "test": test}, scores_dir)
    if len(detectors) != len(set(detectors)):
        raise ValueError(f"A detector appears twice: {detectors}")

    rk = qt.rho(rr.dev_fractions(train, tasks), rr.Q)
    tm, _ = cl.fit_fast(cl.Cohort(cal, "safe"), rk)
    eP, lP = cl.evaluate_fast(cl.Cohort(test, "safe"), rk, tm)
    P = {"lead": float(eP["mean_lead"]), "fpr": float(eP["fpr"])}
    print(f"P alone: lead {P['lead']:.4f} FPR {P['fpr']:.4f} | train {len(train)}, calibration successes "
          f"{len(cal_s)}, path-choice failures {len(train_f)}, test {len(test)}")

    results = {}
    for key in detectors:
        results[key] = {"seed_average": fused(cal, test, key, rk, lP),
                        "per_seed": {s: fused(cal, test, f"{key}@{s}", rk, lP) for s in rr.SEEDS}}

    # Gates against the stored 15.15 / 15.22 results (exact, at their stored precision).
    refs = {}
    if args.reference_hs:
        ref = json.loads(Path(args.reference_hs).read_text())
        refs["P"] = (ref["P"]["mean_lead"], ref["P"]["fpr"])
        for name, d in ref["detectors"].items():
            refs["safe_lstm" if name == "safe" else name] = (d["test"]["mean_lead"], d["test"]["fpr"], d["gain"])
    for item in args.reference_safe or []:
        model, path = item.split("=", 1)
        prim = json.loads(Path(path).read_text())["primary"]
        refs.setdefault("P", (prim["P"]["mean_lead"], prim["P"]["fpr"]))
        if (prim["P"]["mean_lead"], prim["P"]["fpr"]) != refs["P"]:
            raise ValueError(f"{path}: P differs from the other reference")
        refs[f"safe_{model}"] = (prim["safe_plus_P"]["mean_lead"], prim["safe_plus_P"]["fpr"], prim["gain"])
    ok = True
    if "P" in refs:
        match = (P["lead"], P["fpr"]) == tuple(refs["P"])
        ok &= match
        print(f"reference P: stored {refs['P']} now {(P['lead'], P['fpr'])} -> {'OK' if match else 'MISMATCH'}")
    for key in detectors:
        if key not in refs:
            print(f"reference {key} + P: no stored result (new score), not gated")
            continue
        a = results[key]["seed_average"]
        now = (a["lead"], a["fpr"], a["gain"])
        match = now == tuple(refs[key])
        ok &= match
        print(f"reference {key} + P seed average: stored lead/FPR/gain {refs[key]} now {now} -> "
              f"{'OK' if match else 'MISMATCH'}")
    if not ok:
        raise SystemExit("STOP: the seed averages do not reproduce the stored RLDX-1 results")

    summary = {}
    for key in detectors:
        per = results[key]["per_seed"]
        summary[key] = {k: {"mean": float(np.mean([per[s][k] for s in rr.SEEDS])),
                            "std": float(np.std([per[s][k] for s in rr.SEEDS], ddof=1))} for k in KEYS}
        for s in rr.SEEDS:
            m = per[s]
            print(f"  {key} + P seed {s}: fpr {m['fpr']:.3f} lead {m['lead']:.3f} gain {m['gain']:+.3f} twa {m['twa']:.3f}")
        a = results[key]["seed_average"]
        print(f"{key} + P " + " ".join(f"{k} {summary[key][k]['mean']:.3f}+-{summary[key][k]['std']:.3f}" for k in KEYS)
              + f" | seed average: lead {a['lead']:.3f} gain {a['gain']:+.3f} fpr {a['fpr']:.3f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "status": "exploratory (second policy, opened cohorts)", "P": P,
        "counts": {"train": len(train), "calibration_successes": len(cal_s), "path_choice_failures": len(train_f),
                   "test": len(test)},
        "inputs": {"safe_dirs": safe_dirs, "scores_dirs": args.scores_dir, "reference_hs": args.reference_hs,
                   "reference_safe": args.reference_safe},
        "results": {k: {"seed_average": v["seed_average"], "per_seed": {str(s): m for s, m in v["per_seed"].items()}}
                    for k, v in results.items()},
        "summary": summary, "std_ddof": 1}, indent=1, default=float) + "\n")
    print("REFERENCE OK" if refs else "no reference given")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--safe-dir", nargs="+", required=True, help="MODEL=<SAFE20>/calibration/MODEL")
    p.add_argument("--inputs-dir", required=True, help="rldx_hide_and_seek.py prepare output")
    p.add_argument("--scores-dir", nargs="*", default=[], help="score_cohorts output directories")
    p.add_argument("--reference-hs", help="rldx_hide_and_seek.json (report 15.22)")
    p.add_argument("--reference-safe", nargs="*", help="MODEL=rldx_lead_MODEL.json (report 15.15)")
    p.add_argument("--output", required=True)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
