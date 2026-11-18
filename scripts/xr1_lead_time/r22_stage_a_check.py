"""R22 stage A gate (docs/r22_fixed_paths_plan.md): rescored fold directories against their stored runs.

Every stage A job writes a fold directory in the cross-task format that adds the validation cohorts
(the August sets) to an earlier run. The gate (A0):
  E1 cohorts   every stored (detector, rollout) of lead_calibration and confirmatory_test is present
               in the new directory with the same task, label, held-out flag and per-seed raw
               trajectory (max |diff| <= 1e-12), for every detector the new directory scores;
  norms        normalizations.json equals the stored one for those detectors (<= 1e-12);
  validation   every listed validation rollout is scored for every detector and seed, with finite
               values, and its task and label equal the labels file's.

Usage:
  python3 -I r22_stage_a_check.py --pair xt/0 NEW STORED [--pair ...] \\
      --validation august_calibration IDS LABELS --validation august_test IDS LABELS --output OUT.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

E1 = ("lead_calibration", "confirmatory_test")
TOL = 1e-12


def jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_ids(path):
    text = Path(path).read_text()
    if str(path).endswith(".jsonl"):
        return [str(json.loads(l)["rollout_id"]) for l in text.splitlines() if l.strip()]
    value = json.loads(text)
    if isinstance(value, dict):
        for key in ("rollout_ids", "evaluation_rollout_ids", "test"):
            if key in value:
                value = value[key]
                break
    return [str(v) for v in value]


def rows_by(path):
    out = {}
    for r in jsonl(path):
        out[(r["detector"], str(r["rollout_id"]))] = r
    return out


def max_diff(a, b):
    worst = 0.0
    if set(a) != set(b):
        return math.inf
    for s in a:
        x, y = np.asarray(a[s], float), np.asarray(b[s], float)
        worst = max(worst, math.inf if x.shape != y.shape else float(np.max(np.abs(x - y))) if x.size else 0.0)
    return worst


def norm_diff(a, b):
    """Max |diff| between two nested normalization dicts (inf if their keys differ)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return math.inf
        return max([norm_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    return 0.0 if a == b else math.inf


def check_pair(new, stored, validation):
    new, stored = Path(new), Path(stored)
    out, problems = dict(new=str(new.resolve()), stored=str(stored.resolve())), []
    new_dets = set()
    for cohort in E1:
        now = rows_by(new / f"{cohort}_predictions.jsonl")
        new_dets |= {d for d, _ in now}
    for cohort in E1:
        now = rows_by(new / f"{cohort}_predictions.jsonl")
        ref = rows_by(stored / f"{cohort}_predictions.jsonl")
        ref = {k: v for k, v in ref.items() if k[0] in new_dets}
        missing = [k for k in ref if k not in now]
        worst, meta_bad = 0.0, 0
        for k, r in ref.items():
            if k not in now:
                continue
            n = now[k]
            if (n["task_name"], bool(n["failed"]), bool(n.get("held_out"))) != (r["task_name"], bool(r["failed"]), bool(r.get("held_out"))):
                meta_bad += 1
            worst = max(worst, max_diff(n["per_seed_raw"], r["per_seed_raw"]))
        ok = bool(ref) and not missing and meta_bad == 0 and worst <= TOL
        out[cohort] = dict(compared=len(ref), missing=len(missing), label_mismatch=meta_bad, max_abs_diff=worst, ok=ok)
        if not ok:
            problems.append(f"{cohort}: {out[cohort]}")
    n_norm = json.loads((new / "normalizations.json").read_text())
    s_norm = json.loads((stored / "normalizations.json").read_text())
    nd = max([norm_diff(n_norm[d], s_norm.get(d, {})) for d in new_dets] or [math.inf])
    out["normalizations"] = dict(max_abs_diff=nd, ok=nd <= TOL)
    if not out["normalizations"]["ok"]:
        problems.append(f"normalizations differ by {nd}")
    for cohort, (ids, labels) in validation.items():
        path = new / f"{cohort}_predictions.jsonl"
        if not path.exists():
            out[cohort] = dict(ok=False, reason="no file")
            problems.append(f"{cohort}: not scored")
            continue
        now = rows_by(path)
        seeds = None
        bad = dict(missing=0, nonfinite=0, label_mismatch=0, seed_mismatch=0)
        for d in sorted(new_dets):
            for rid in ids:
                r = now.get((d, rid))
                if r is None:
                    bad["missing"] += 1
                    continue
                if seeds is None:
                    seeds = sorted(r["per_seed_raw"])
                if sorted(r["per_seed_raw"]) != seeds:
                    bad["seed_mismatch"] += 1
                if not all(np.all(np.isfinite(np.asarray(v, float))) and len(v) for v in r["per_seed_raw"].values()):
                    bad["nonfinite"] += 1
                lab = labels.get(rid)
                if lab is None or (r["task_name"], bool(r["failed"])) != lab:
                    bad["label_mismatch"] += 1
        ok = not any(bad.values())
        out[cohort] = dict(rollouts=len(ids), detectors=sorted(new_dets), seeds=seeds, **bad, ok=ok)
        if not ok:
            problems.append(f"{cohort}: {bad}")
    out["ok"] = not problems
    return out, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", nargs=3, action="append", required=True, metavar=("NAME", "NEW", "STORED"))
    ap.add_argument("--validation", nargs=3, action="append", default=[], metavar=("COHORT", "IDS", "LABELS"))
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    validation = {}
    for cohort, ids_path, labels_path in a.validation:
        labels = {str(r["rollout_id"]): (r["task_name"], bool(r["failed"])) for r in jsonl(labels_path)}
        validation[cohort] = (load_ids(ids_path), labels)
    result, failed = {}, []
    for name, new, stored in a.pair:
        res, problems = check_pair(new, stored, validation)
        result[name] = res
        status = "OK" if res["ok"] else "FAILED: " + "; ".join(problems)
        print(f"{name}: " + ", ".join(f"{c} {res[c]['compared']} max {res[c]['max_abs_diff']:.1e}" for c in E1)
              + f", norms {res['normalizations']['max_abs_diff']:.1e}; "
              + ", ".join(f"{c} {res[c].get('rollouts', 0)}" for c in validation) + f" -> {status}", flush=True)
        if not res["ok"]:
            failed.append(name)
    Path(a.output).write_text(json.dumps(dict(plan="docs/r22_fixed_paths_plan.md (R22 stage A)", pairs=result,
                                              failed=failed), indent=1, default=float) + "\n")
    print("STAGE A OK" if not failed else f"STAGE A FAILED: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
