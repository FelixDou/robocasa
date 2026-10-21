"""Analyse point-of-no-return labels against detector alarms (plan: docs/no_return_labeling_plan.md).

Usage:
  python3 -I no_return_analysis.py --labels no_return_labels.json --key key/key.json --output OUT.json \
      [--second-labels other_labeler.json] [--allow-incomplete]
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

BOOT, SEED = 2000, 0
LANDMARKS = (0.25, 0.50, 0.75)


def complete(l):
    return bool(l) and (l.get("t_u_env_step") is not None or l.get("recoverable_until_timeout")) \
        and bool(l.get("category")) and bool(l.get("confidence"))


def rows_from(labels, key):
    rows = []
    for vid, k in sorted(key["videos"].items()):
        l = labels.get(vid, {})
        if not complete(l):
            continue
        s = int(k["steps_per_inference"]); H = int(k["horizon_inferences"]) * s; L = int(k["length_inferences"]) * s
        rec = bool(l.get("recoverable_until_timeout"))
        tu = L if rec else int(l["t_u_env_step"])
        te = l.get("t_e_env_step")
        row = dict(id=vid, task_type=k["task_type"], category=l["category"], confidence=l["confidence"],
                   recoverable=rec, H=H, L=L, t_u=tu, t_e=None if te is None else int(te), detectors={})
        for name, field in (("P", "alarm_P_inference"), ("safe_plus_P", "alarm_safe_plus_P_inference")):
            a = k.get(field)
            step = L if a is None else (float(a) - 1) * s      # alarm at inference a happens at env step s*(a-1)
            step = min(step, L)
            usable = (L - step) / H if rec else (H - max(step, tu)) / H
            row["detectors"][name] = dict(alarm=step, wasted=None if rec else (step - tu) / H,
                                          preemptive=None if rec else step <= tu, usable=max(0.0, usable))
        rows.append(row)
    return rows


def median_ci(x):
    x = np.asarray(x, float)
    if not len(x):
        return None
    rng = np.random.default_rng(SEED)
    b = [np.median(rng.choice(x, len(x))) for _ in range(BOOT)]
    return dict(median=round(float(np.median(x)), 4), ci95=np.round(np.percentile(b, [2.5, 97.5]), 4).tolist(),
                q25_q75=np.round(np.percentile(x, [25, 75]), 4).tolist())


def summarize(rows):
    if not rows:
        return dict(n=0)
    fu = [r["t_u"] / r["H"] for r in rows]
    decided = [r for r in rows if not r["recoverable"]]
    out = dict(n=len(rows), recoverable_until_timeout=round(float(np.mean([r["recoverable"] for r in rows])), 4),
               t_u_over_H=median_ci(fu),
               decided_by={str(f): round(float(np.mean([(not r["recoverable"]) and r["t_u"] / r["H"] <= f for r in rows])), 4)
                           for f in LANDMARKS},
               t_e_over_H=median_ci([r["t_e"] / r["H"] for r in rows if r["t_e"] is not None]),
               gap_t_u_minus_t_e_over_H=median_ci([(r["t_u"] - r["t_e"]) / r["H"] for r in decided if r["t_e"] is not None]),
               categories=dict(Counter(r["category"] for r in rows)), detectors={})
    for name in ("P", "safe_plus_P"):
        d = [r["detectors"][name] for r in rows]
        dd = [r["detectors"][name] for r in decided]
        out["detectors"][name] = dict(
            wasted_after_t_u_over_H=median_ci([v["wasted"] for v in dd]),
            mean_wasted_after_t_u_over_H=round(float(np.mean([v["wasted"] for v in dd])), 4) if dd else None,
            preemptive_fraction=round(float(np.mean([v["preemptive"] for v in dd])), 4) if dd else None,
            mean_usable_lead_over_H=round(float(np.mean([v["usable"] for v in d])), 4))
    if decided:
        diff = [r["detectors"]["safe_plus_P"]["wasted"] - r["detectors"]["P"]["wasted"] for r in decided]
        out["safe_plus_P_minus_P_wasted"] = dict(mean=round(float(np.mean(diff)), 4),
                                                  fraction_earlier=round(float(np.mean([v < -1e-12 for v in diff])), 4))
    return out


def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    labels = json.load(open(args.labels))["labels"]
    key = json.load(open(args.key))
    missing = [v for v in key["videos"] if not complete(labels.get(v, {}))]
    if missing and not args.allow_incomplete:
        raise SystemExit(f"error: {len(missing)} videos are not fully labeled: {missing[:10]}")
    rows = rows_from(labels, key)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["task_type"]].append(r)
    out = dict(plan="docs/no_return_labeling_plan.md", status="exploratory", n_labeled=len(rows), n_missing=len(missing),
               overall=summarize(rows), by_type={t: summarize(v) for t, v in sorted(by_type.items())},
               high_confidence=summarize([r for r in rows if r["confidence"] == "high"]))
    if args.second_labels:
        other = rows_from(json.load(open(args.second_labels))["labels"], key)
        mine = {r["id"]: r for r in rows}
        pairs = [(mine[o["id"]]["t_u"] / o["H"], o["t_u"] / o["H"]) for o in other if o["id"] in mine]
        out["inter_rater"] = dict(videos=len(pairs),
                                  median_abs_diff_t_u_over_H=round(float(np.median([abs(a - b) for a, b in pairs])), 4) if pairs else None,
                                  same_recoverable_call=round(float(np.mean([mine[o["id"]]["recoverable"] == o["recoverable"]
                                                                              for o in other if o["id"] in mine])), 4) if pairs else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    for name, block in [("overall", out["overall"])] + [(t, b) for t, b in out["by_type"].items()]:
        if not block.get("n"):
            continue
        tu = block["t_u_over_H"]
        print(f"{name:10s} n={block['n']:2d}  median t_u/H {tu['median']} {tu['ci95']}  decided by 25/50/75%: "
              f"{block['decided_by']}  recoverable {block['recoverable_until_timeout']}")
        for det, v in block["detectors"].items():
            w = v["wasted_after_t_u_over_H"]
            print(f"   {det:12s} wasted after t_u: median {w['median'] if w else None}  mean {v['mean_wasted_after_t_u_over_H']}  "
                  f"pre-emptive {v['preemptive_fraction']}  usable lead {v['mean_usable_lead_over_H']}")
    print("categories:", out["overall"]["categories"])
    if "inter_rater" in out:
        print("inter-rater:", out["inter_rater"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True); p.add_argument("--key", required=True); p.add_argument("--output", required=True)
    p.add_argument("--second-labels"); p.add_argument("--allow-incomplete", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
