"""Per-seed Table I rows for the four OOD baselines (mean and std over model seeds).

Each input is the ``evaluate/analysis.json`` of one run of
``robocasa.recovery.safe.run_offline_detector_baselines`` (Section 15.10 pipeline). Seed 0 is
the original run; seeds 1 and 2 refit the same configurations with ``develop
--fixed-selection ... --model-seed S`` and are calibrated and evaluated the same way.

From each run's ``<baseline>_only`` row (primary balanced cohort, 380 + 380):
  bACC = balanced_accuracy = (TPR + 1 - FPR) / 2,
  DT   = missed_failure_adjusted_detection_fraction (miss counted as 1),
  TWA  = (1 - DT + 1 - FPR) / 2   (FIPER / Hide-and-Seek, miss = 0).
The seed-0 run must reproduce the current Table I row given with --expect, otherwise the script
stops before printing any mean.

Usage:
  python3 -I ood_per_seed_summary.py --analysis 0=A0.json 1=A1.json 2=A2.json \\
      --expect mahalanobis=0.666,0.068 knn_cosine=0.582,0.055 \\
               pca_kmeans=0.605,0.063 rnd=0.629,0.068 --output OUT.json
"""
import argparse
import json
import sys

import numpy as np

BASELINES = ("mahalanobis", "knn_cosine", "pca_kmeans", "rnd")


def row_metrics(row):
    tpr = float(row["true_positive_rate"])
    fpr = float(row["false_positive_rate"])
    dt = float(row["missed_failure_adjusted_detection_fraction"])
    return {
        "tpr": tpr,
        "fpr": fpr,
        "bacc": float(row["balanced_accuracy"]),
        "dt": dt,
        "twa": (1.0 - dt + 1.0 - fpr) / 2.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", nargs="+", required=True, help="SEED=analysis.json")
    parser.add_argument("--expect", nargs="*", default=[], help="NAME=TPR,FPR for seed 0")
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    runs = {}
    for item in args.analysis:
        seed, path = item.split("=", 1)
        data = json.load(open(path))
        if data.get("status") is not None and not data.get("frozen_results_reproduced", False):
            sys.exit(f"STOP: seed {seed} run did not reproduce the frozen results ({path})")
        counts = data["counts"]["primary_balanced"]
        runs[int(seed)] = {"path": path, "counts": counts, "overall": data["overall"]}
        print(f"seed {seed}: {path} primary cohort {counts}")
    if 0 not in runs or len(runs) < 2:
        sys.exit("STOP: need seed 0 and at least one other seed")
    if len({json.dumps(run["counts"], sort_keys=True) for run in runs.values()}) != 1:
        sys.exit("STOP: the runs evaluated different cohorts")

    per_seed = {
        name: {seed: row_metrics(run["overall"][f"{name}_only"]) for seed, run in sorted(runs.items())}
        for name in args.baselines
    }

    ok = True
    for item in args.expect:
        name, values = item.split("=", 1)
        tpr, fpr = (float(value) for value in values.split(","))
        got = per_seed[name][0]
        match = round(got["tpr"], 3) == tpr and round(got["fpr"], 3) == fpr
        ok &= match
        print(f"reference {name} seed 0: tpr {got['tpr']:.3f} FPR {got['fpr']:.3f} "
              f"vs table {tpr:.3f} {fpr:.3f} -> {'OK' if match else 'MISMATCH'}")
    if not ok:
        sys.exit("STOP: seed 0 does not reproduce Table I")

    summary = {}
    for name in args.baselines:
        seeds = per_seed[name]
        summary[name] = {
            key: {
                "mean": float(np.mean([m[key] for m in seeds.values()])),
                "std": float(np.std([m[key] for m in seeds.values()], ddof=1)),
            }
            for key in ("tpr", "fpr", "bacc", "twa", "dt")
        }
        for seed, m in seeds.items():
            print(f"  {name} seed {seed}: tpr {m['tpr']:.3f} fpr {m['fpr']:.3f} "
                  f"bacc {m['bacc']:.3f} twa {m['twa']:.3f} dt {m['dt']:.3f}")
        print(f"{name} " + " ".join(
            f"{key} {summary[name][key]['mean']:.3f}+-{summary[name][key]['std']:.3f}"
            for key in ("tpr", "fpr", "bacc", "twa", "dt")))
    with open(args.output, "w") as handle:
        json.dump({"runs": {str(s): r["path"] for s, r in runs.items()},
                   "per_seed": {n: {str(s): m for s, m in v.items()} for n, v in per_seed.items()},
                   "summary": summary, "std_ddof": 1}, handle, indent=2, sort_keys=True)
    print("REFERENCE OK" if args.expect else "no reference given")


if __name__ == "__main__":
    main()
