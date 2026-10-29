"""Data for the per-task figure (Fig. 2) and the false-alarm sweep.

Reads review_analyses.json (run xr1_review_analyses_20261028_155557, branch claude/review-analyses,
reports/safe_robocasa365_research_plan_20260823/evidence/xr1_review_analyses_20261028_155557/) and
writes two CSV files next to the figures:

- figures/per_task_gains.csv: gain in mean lead over P per task (alpha = 0.05, held-out test set),
  for Hide-and-Seek + P and SAFE + P, sorted by the Hide-and-Seek gain.
- figures/matched_sweep.csv: gain over P loosened on the test set to the fused detector's false
  alarms, for each calibration target alpha.

Usage: python3 -I analysis/per_task_figure.py <review_analyses.json>
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")


def main(path):
    with open(path) as f:
        data = json.load(f)

    hs = data["R3"]["hide_and_seek"]["per_task"]
    safe = data["R3"]["safe"]["per_task"]
    assert set(hs) == set(safe) and len(hs) == 38
    tasks = sorted(hs, key=lambda k: (-hs[k]["gain"], k))
    with open(os.path.join(FIG, "per_task_gains.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "task", "type", "hs", "hs_atomic", "hs_composite", "safe"])
        for i, k in enumerate(tasks, 1):
            g = hs[k]["gain"]
            atomic = hs[k]["type"] == "atomic"
            w.writerow([i, k, hs[k]["type"], g,
                        g if atomic else "nan", "nan" if atomic else g, safe[k]["gain"]])

    r1 = data["R1"]
    names = ["hide_and_seek", "safe", "mahalanobis", "hide_and_seek_zero",
             "hide_and_seek_first", "hide_and_seek_task_shuffle"]
    with open(os.path.join(FIG, "matched_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha"] + names)
        for i, a in enumerate(r1["alphas"]):
            row = [a]
            for n in names:
                e = r1["detectors"][n][i]
                assert e["alpha"] == a
                row.append(e["matched_P"]["gain"])
            w.writerow(row)

    for n in ("hide_and_seek", "safe"):
        s = data["R3"][n]["summary"]
        print(n, "pooled", s["pooled_gain"], "by type", s["by_type"],
              "+/0/-", s["positive"], s["zero"], s["negative"])


if __name__ == "__main__":
    main(sys.argv[1])
