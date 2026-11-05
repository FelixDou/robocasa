"""Per-seed Table II rows for the four OOD baselines fused with P (mean and std over model seeds).

Table II's OOD + P rows (report Section 15.14, dose_response.py) fuse the seed-0 baseline
models with the per-task timer P on the held-out E1 cohorts (calibration and test, 10 + 10 per
task each). Here each model seed's trajectories, attached with
``run_offline_detector_baselines score-baselines`` from that seed's calibration, are fused and
evaluated exactly as there (dose_response.fused_gain: confirm_lead.fit_fast and evaluate_fast,
q = 0.95):
  FPR, mean lead l of the fused detector, its gain over P (mean per-failure difference) and
  TWA = (l + 1 - FPR) / 2.
P alone must be identical for every seed (it reads no baseline). Seed 0 must reproduce the
original Section 15.14 output given with --reference dose_response.json exactly, at its stored
4-decimal precision (P lead and FPR, each fused FPR, mean lead and gain), and the Table II row
given with --expect NAME=GAIN,FPR to within half a unit of the table's last digit (the table
rounds the stored values, e.g. FPR 0.0605 -> 0.061). Otherwise the script stops before any mean.

Usage:
  python3 -I ood_per_seed_fused.py --runtime-bundle runtime_bundle.json \\
      --seed 0 CAL0.jsonl TEST0.jsonl --seed 1 CAL1.jsonl TEST1.jsonl --seed 2 CAL2.jsonl TEST2.jsonl \\
      --reference dose_response.json \\
      --expect mahalanobis=0.034,0.055 knn_cosine=0.021,0.058 pca_kmeans=0.022,0.061 \\
               rnd=0.020,0.058 --output OUT.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl  # noqa: E402
import dose_response as dr  # noqa: E402
import quantile_timer as qt  # noqa: E402

KEYS = ("fpr", "lead", "gain", "twa")


def seed_metrics(cal_path, test_path, rk, baselines=None):
    """Fused-with-P metrics of every baseline for one seed's cohort files."""
    _, cal = cl.primary(cal_path, "calibration")
    _, te = cl.primary(test_path, "test")
    names = sorted(cal[0]["baseline_scores"])
    if sorted(te[0]["baseline_scores"]) != names:
        raise ValueError("Calibration and test carry different baselines")
    names = [n for n in names if baselines is None or n in baselines]
    dr.attach(cal, names)
    dr.attach(te, names)
    out, p_alone = {}, None
    for name in names:
        res, d, _ = dr.fused_gain(cal, te, name, rk)
        P = (float(res["P"]["mean_lead"]), float(res["P"]["fpr"]))
        if p_alone is not None and P != p_alone:
            raise ValueError(f"P alone differs between detectors: {P} vs {p_alone}")
        p_alone = P
        fpr, lead = float(res["fused"]["fpr"]), float(res["fused"]["mean_lead"])
        out[name] = {"fpr": fpr, "lead": lead, "gain": float(np.mean(d)),
                     "twa": (lead + 1.0 - fpr) / 2.0}
    return {"P": {"lead": p_alone[0], "fpr": p_alone[1]}, "detectors": out,
            "counts": {"calibration": len(cal), "test": len(te)}}


def check_reference(seed0, reference):
    """Seed 0 against the original dose_response.json, at its stored precision."""
    ok = True
    for name, got in seed0["detectors"].items():
        pooled = reference["detectors"][name]["pooled"]
        pairs = {
            "P lead": (round(seed0["P"]["lead"], 4), pooled["P"]["mean_lead"]),
            "P FPR": (round(seed0["P"]["fpr"], 4), pooled["P"]["fpr"]),
            "fused FPR": (round(got["fpr"], 4), pooled["fused"]["fpr"]),
            "fused lead": (round(got["lead"], 4), pooled["fused"]["mean_lead"]),
            "gain": (round(got["gain"], 4), pooled["gain"]),
        }
        bad = {k: v for k, v in pairs.items() if v[0] != v[1]}
        ok &= not bad
        print(f"original {name} + P: gain {pooled['gain']:.4f} FPR {pooled['fused']['fpr']:.4f} "
              f"lead {pooled['fused']['mean_lead']:.4f} vs seed 0 {round(got['gain'], 4):.4f} "
              f"{round(got['fpr'], 4):.4f} {round(got['lead'], 4):.4f} -> {'OK' if not bad else f'MISMATCH {bad}'}")
    if not ok:
        raise SystemExit("STOP: seed 0 does not reproduce the original Section 15.14 output")


def summarize(per_seed, expect, reference=None):
    """Gate on seed 0 and P, then mean and std (ddof 1) over seeds."""
    seeds = sorted(per_seed)
    if 0 not in per_seed or len(seeds) < 2:
        raise SystemExit("STOP: need seed 0 and at least one other seed")
    first = per_seed[seeds[0]]
    for s in seeds[1:]:
        if per_seed[s]["P"] != first["P"] or per_seed[s]["counts"] != first["counts"]:
            raise SystemExit(f"STOP: seed {s} has a different P or cohort ({per_seed[s]['P']}, {per_seed[s]['counts']})")
    print(f"P alone: lead {first['P']['lead']:.4f} FPR {first['P']['fpr']:.4f} (all seeds); cohorts {first['counts']}")
    if reference is not None:
        check_reference(per_seed[0], reference)
    ok = True
    for name, (gain, fpr) in expect.items():
        got = per_seed[0]["detectors"][name]
        match = abs(got["gain"] - gain) <= 5e-4 + 1e-9 and abs(got["fpr"] - fpr) <= 5e-4 + 1e-9
        ok &= match
        print(f"reference {name} + P seed 0: gain {got['gain']:.4f} FPR {got['fpr']:.4f} "
              f"vs table {gain:.3f} {fpr:.3f} (within 0.0005) -> {'OK' if match else 'MISMATCH'}")
    if not ok:
        raise SystemExit("STOP: seed 0 does not reproduce Table II")
    summary = {}
    for name in first["detectors"]:
        rows = [per_seed[s]["detectors"][name] for s in seeds]
        summary[name] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                             "std": float(np.std([r[k] for r in rows], ddof=1))} for k in KEYS}
        for s, r in zip(seeds, rows):
            print(f"  {name} + P seed {s}: fpr {r['fpr']:.3f} lead {r['lead']:.3f} "
                  f"gain {r['gain']:+.3f} twa {r['twa']:.3f}")
        print(f"{name} + P " + " ".join(
            f"{k} {summary[name][k]['mean']:.3f}+-{summary[name][k]['std']:.3f}" for k in KEYS))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime-bundle", required=True)
    parser.add_argument("--seed", nargs=3, action="append", required=True,
                        metavar=("SEED", "CALIBRATION", "TEST"))
    parser.add_argument("--reference", help="original dose_response.json (Section 15.14) for seed 0")
    parser.add_argument("--expect", nargs="*", default=[], help="NAME=GAIN,FPR for seed 0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise SystemExit(f"error: output exists: {args.output}")
    bundle = json.load(open(args.runtime_bundle))
    rk = qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)
    expect = {}
    for item in args.expect:
        name, values = item.split("=", 1)
        gain, fpr = (float(v) for v in values.split(","))
        expect[name] = (gain, fpr)
    per_seed = {}
    for seed, cal_path, test_path in args.seed:
        per_seed[int(seed)] = seed_metrics(cal_path, test_path, rk)
        print(f"seed {seed}: {cal_path} | {test_path}", flush=True)
    reference = json.load(open(args.reference)) if args.reference else None
    summary = summarize(per_seed, expect, reference)
    Path(args.output).write_text(json.dumps(
        {"inputs": {s: [c, t] for s, c, t in args.seed}, "per_seed": {str(s): v for s, v in per_seed.items()},
         "summary": summary, "std_ddof": 1}, indent=1, sort_keys=True) + "\n")
    print("REFERENCE OK" if expect or reference else "no reference given")


if __name__ == "__main__":
    main()
