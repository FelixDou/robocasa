"""ActProbe and Hide-and-Seek against the per-task timer P and fused with elapsed time.

Exploratory follow-up to the published-baseline run (docs/xr1_baselines.md). It reuses
two existing protocols on the same August cohorts, so the new rows sit next to SAFE:

  A. Lead over P (physical_signals.analyse, report section 15.16): each score fused with
     the per-task 95th-percentile timer P on the nested (u, s) path, chosen on
     calibration failures, conformal margin on calibration successes (alpha = 0.05).
     Reports the raw lead gain over P, the gain over P loosened to the same number of
     test false alarms (matched FPR), and a noise null fitted the same way.
  B. Fused with the frozen global time risk (fused_safe_time of
     prospective_matched_fpr, report sections 15.10-15.11): the score OR the time risk,
     both eligible for the whole rollout, thresholds chosen jointly on calibration at
     the bundle's target FPR. Paired against time-only and against fused SAFE with the
     frozen task-then-rollout bootstrap (2000 replicates, seed 0).

Cohorts: calibration = primary balanced cohort (3 + 3 per task, 228 rollouts); test =
August prospective cohort (10 + 10 per task, 760 rollouts), already opened. Nothing here
is confirmatory; confirmatory seeds are refused by physical_signals.load_predictions.

Inputs: the baseline run directory must contain calibration_predictions.jsonl and
evaluation_predictions.jsonl (run_baselines writes both since this script was added),
plus the frozen August SAFE predictions and runtime bundle. Before reporting, the script
reproduces the P and SAFE + P numbers of section 15.16 and fused SAFE of section 15.10,
and the baseline-only rows of the baseline run's results.json; it stops if any differ.

Usage (TSUBAME, CPU only):
  python -I -u published_baselines_lead.py --baseline-run RUN_DIR [RUN_DIR ...] \
      --calibration calibration_predictions.jsonl --test event_predictions.jsonl \
      --bundle runtime_bundle.json --out OUT.json [--workers 8] [--noise-draws 100] \
      [--compare-run EARLIER_RUN_DIR] [--contrast A B ...]

Controls (added 2026-10-22): hide_and_seek_zero (all-zero inputs: elapsed time only)
and hide_and_seek_task_shuffle (same-task, same-step features of other training
rollouts) come from a separate run. Pass both runs and contrast them with
hide_and_seek, e.g. --contrast hide_and_seek hide_and_seek_zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import physical_signals as ps  # noqa: E402
from robocasa.recovery.safe.prospective_matched_fpr import (  # noqa: E402
    event_metrics,
    select_single_threshold,
    validate_stage_windows,
)
from robocasa.recovery.safe.run_offline_detector_baselines import (  # noqa: E402
    _run_jobs,
    detector_outcomes,
    paired_comparison,
    task_rollout_bootstrap,
)

# Numbers this script must reproduce on the August cohorts before it reports anything.
REFERENCE = {
    "timer_mean_lead": 0.1952,     # report 15.16: P alone, lead 0.1952 at 17/380 false alarms
    "timer_fp": 17,
    "safe_gain_over_timer": 0.0069,  # report 15.16: SAFE + P gain +0.0069
    "fused_safe_tpr": 0.955,       # report 15.10: fused SAFE + time 0.955 / 0.058
    "fused_safe_fpr": 0.058,
}
BOOT_SAMPLES, BOOT_SEED = 2000, 0
LANDMARK_KEYS = ("0.1", "0.25", "0.5")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_baseline_predictions(path):
    """detector -> rollout_id -> record, from a run_baselines predictions file."""
    out = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if ps.is_confirmatory(record.get("environment_seed")):
            raise SystemExit(f"{path} contains a confirmatory seed; refusing to read it")
        name, rid = record["detector"], str(record["rollout_id"])
        if rid in out[name]:
            raise ValueError(f"{path}: duplicate rollout {rid} for {name}")
        out[name][rid] = record
    return dict(out)


def attach(rows, predictions, names, label):
    """Put each detector's trajectory on the SAFE rows, checking alignment."""
    worst_time = 0.0
    for r in rows:
        rid = str(r["rollout_id"])
        r.setdefault("baseline_scores", {})
        for name in names:
            if rid not in predictions[name]:
                raise ValueError(f"{label}: rollout {rid} has no {name} trajectory")
            b = predictions[name][rid]
            if bool(b["failed"]) != r["failed"] or b["task_name"] != r["task_name"]:
                raise ValueError(f"{label}: label or task differs for {rid} ({name})")
            v = np.asarray(b["safe_scores"], float)
            if len(v) != r["L"] or not np.all(np.isfinite(v)):
                raise ValueError(f"{label}: {name} trajectory for {rid} has length {len(v)}, SAFE has {r['L']}")
            t = np.asarray(b["time_scores"], float)
            worst_time = max(worst_time, float(np.max(np.abs(t - np.asarray(r["time_scores"], float)))))
            r[name] = v
            r["baseline_scores"][name] = v.tolist()
    if worst_time > 1e-9:
        raise ValueError(f"{label}: baseline time scores differ from the frozen ones (max {worst_time})")
    return worst_time


def check(name, value, expected, tol, failures):
    ok = value is not None and abs(float(value) - float(expected)) <= tol
    if not ok:
        failures.append(f"{name}: got {value}, expected {expected} (tol {tol})")
    return dict(value=value, expected=expected, ok=ok)


# --------------------------------------------------------------------------- part B
def spec(score):
    return {"name": "time_only" if score is None else f"fused_{score}_time",
            "base": "time_only" if score is None else "fused_safe_time",
            "score": "safe" if score is None else score}


def views(rows, s):
    out = []
    for r in rows:
        score = r["safe_scores"] if s["score"] == "safe" else r["baseline_scores"][s["score"]]
        out.append(dict(rollout_id=r["rollout_id"], task_name=r["task_name"], failed=r["failed"],
                        horizon=r["horizon"], source_inferences=r["L"],
                        safe_scores=list(map(float, score)), time_scores=list(map(float, r["time_scores"]))))
    return out


def metric_row(m):
    return dict(tpr=round(m["true_positive_rate"], 4), fpr=round(m["false_positive_rate"], 4),
                tp=m["confusion"]["tp"], fp=m["confusion"]["fp"],
                tpr_wilson95=[round(x, 4) for x in m["tpr_wilson_95"]],
                fpr_wilson95=[round(x, 4) for x in m["fpr_wilson_95"]],
                miss_adjusted_detection=round(m["missed_failure_adjusted_detection_fraction"], 4),
                recall_by={k: round(m["failure_recall_by_landmark"][k], 4) for k in LANDMARK_KEYS
                           if k in m["failure_recall_by_landmark"]},
                roc_auc=round(m["roc_auc"], 4))


def fused_with_time(cal, te, names, target_fpr, safe_end, time_start, workers, contrasts=()):
    specs = [spec(None), spec("safe")] + [spec(n) for n in names]
    jobs = [(s["name"], s["base"], views(cal, s), target_fpr, safe_end, time_start) for s in specs]
    selections = {name: sel for name, sel, _ in _run_jobs(jobs, workers)}
    out = dict(target_fpr=target_fpr, detectors={}, paired={})
    outcomes = {}
    for s in specs:
        th = selections[s["name"]]["thresholds"]
        m = event_metrics(views(te, s), s["base"], th, safe_end=safe_end, time_start=time_start)
        out["detectors"][s["name"]] = dict(thresholds=th, calibration=metric_row(selections[s["name"]]["validation_metrics"]),
                                           test=metric_row(m))
        outcomes[s["name"]] = detector_outcomes(te, s, th, safe_end, time_start)
    pairs = [("fused_safe_time", "time_only")]
    for n in names:
        pairs += [(f"fused_{n}_time", "time_only"), (f"fused_{n}_time", "fused_safe_time")]
    pairs += [(f"fused_{a}_time", f"fused_{b}_time") for a, b in contrasts
              if (f"fused_{a}_time", f"fused_{b}_time") not in pairs]
    boot = task_rollout_bootstrap(te, outcomes, pairs, BOOT_SAMPLES, BOOT_SEED)
    for a, b in pairs:
        out["paired"][f"{a}_vs_{b}"] = dict(counts=paired_comparison(te, outcomes[a], outcomes[b]),
                                            bootstrap=boot[(a, b)])
    return out


# --------------------------------------------------------------------------- contrasts
def lead_contrasts(cal, te, rk, contrasts):
    """Per-failure lead of score A + P minus score B + P, each fitted as in part A.

    ``raw`` compares the two fused detectors as fitted (they may spend different
    false-alarm budgets). ``matched`` compares their gains over P loosened to each
    detector's own test false-alarm count, which removes the budget difference.
    """
    F_te = [r for r in te if r["failed"]]
    fits = {}
    for key in sorted({k for pair in contrasts for k in pair}):
        f = ps.fit_fused(cal, key, rk)
        e, lv = ps.evaluate(te, key, rk, f["lam"], f["u"], f["s"])
        _, lm = ps.matched_timer(te, rk, e["fp"])
        fits[key] = dict(fp=e["fp"], lead=lv, matched=lv - lm)
    out = {}
    for a, b in contrasts:
        out[f"{a}_minus_{b}"] = dict(
            fp=[fits[a]["fp"], fits[b]["fp"]],
            raw=ps.boot(fits[a]["lead"] - fits[b]["lead"], F_te),
            matched=ps.boot(fits[a]["matched"] - fits[b]["matched"], F_te))
    return out


# --------------------------------------------------------------------------- main
def compare_runs(runs, earlier, names):
    """Largest trajectory difference between these runs and an earlier one (same IDs)."""
    now = {}
    for run in runs:
        now.update(load_baseline_predictions(Path(run) / "evaluation_predictions.jsonl"))
    before = load_baseline_predictions(Path(earlier) / "evaluation_predictions.jsonl")
    out = {}
    for n in names:
        common = sorted(set(now.get(n, {})) & set(before.get(n, {})))
        diff = max((float(np.max(np.abs(np.asarray(now[n][i]["safe_scores"], float) -
                                         np.asarray(before[n][i]["safe_scores"], float))))
                    for i in common), default=None)
        out[n] = dict(rollouts_compared=len(common), max_abs_diff=diff)
    return out


def run(args):
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    run_dirs = [Path(d) for d in args.baseline_run]
    files = dict(calibration=Path(args.calibration), test=Path(args.test), bundle=Path(args.bundle))
    for i, d in enumerate(run_dirs):
        files[f"run{i}_calibration"] = d / "calibration_predictions.jsonl"
        files[f"run{i}_evaluation"] = d / "evaluation_predictions.jsonl"
        files[f"run{i}_results"] = d / "results.json"
    for label, path in files.items():
        if not path.is_file():
            raise SystemExit(f"error: missing {label}: {path}")
    inputs = {k: dict(path=str(v.resolve()), sha256=sha256(v)) for k, v in files.items()}

    bundle = json.loads(files["bundle"].read_text())
    target_fpr = float(bundle["target_fpr"])
    safe_end, time_start = validate_stage_windows(bundle["safe_eligibility_end_fraction"],
                                                  bundle["time_eligibility_start_fraction"])
    cal = ps.select_balanced(ps.load_predictions(files["calibration"]), 3)
    te = ps.select_balanced(ps.load_predictions(files["test"]), 10)
    primary = set(map(str, bundle["primary_calibration_rollout_ids"]))
    if {str(r["rollout_id"]) for r in cal} != primary:
        raise ValueError("Calibration selection differs from the bundle's primary calibration IDs")

    # Merge the runs; a detector name may come from one run only.
    pred_cal, pred_te, results, source = {}, {}, {}, {}
    for i, d in enumerate(run_dirs):
        run_cal = load_baseline_predictions(files[f"run{i}_calibration"])
        run_te = load_baseline_predictions(files[f"run{i}_evaluation"])
        run_results = json.loads(files[f"run{i}_results"].read_text())
        for n in run_cal:
            if n in source:
                raise ValueError(f"Detector {n} appears in two runs: {source[n]} and {d}")
            if n not in run_te:
                raise ValueError(f"Detector {n} has calibration but no evaluation trajectories in {d}")
            source[n] = str(d.resolve())
            pred_cal[n], pred_te[n] = run_cal[n], run_te[n]
            results[n] = run_results.get("detectors", {}).get(n)
    names = sorted(pred_cal)
    if args.detectors:
        missing = sorted(set(args.detectors) - set(names))
        if missing:
            raise ValueError(f"Requested detectors not in the runs: {missing}")
        names = [n for n in names if n in args.detectors]
    contrasts = [tuple(pair) for pair in (args.contrast or [])]
    for pair in contrasts:
        unknown = [k for k in pair if k != "safe" and k not in names]
        if unknown:
            raise ValueError(f"Contrast {pair} names unknown detectors {unknown}")
    attach(cal, pred_cal, names, "calibration")
    attach(te, pred_te, names, "test")
    print(f"cohorts: calibration {len(cal)}, test {len(te)}; detectors {names}", flush=True)

    failures = []
    checks = {}
    # The baseline-only rows of each run's results.json must come back from these trajectories.
    for n in names:
        sel = select_single_threshold(views(cal, spec(n)), "safe_only", target_fpr=target_fpr,
                                      safe_end=safe_end, time_start=time_start)
        m = event_metrics(views(te, spec(n)), "safe_only", sel["thresholds"],
                          safe_end=safe_end, time_start=time_start)
        if results[n] is None:
            raise ValueError(f"{n} is missing from results.json in {source[n]}")
        ref = results[n]["evaluation"]["baseline_only"]
        checks[f"{n}_baseline_only_tpr"] = check(f"{n} baseline-only TPR", m["true_positive_rate"],
                                                 ref["true_positive_rate"], 1e-9, failures)
        checks[f"{n}_baseline_only_fpr"] = check(f"{n} baseline-only FPR", m["false_positive_rate"],
                                                 ref["false_positive_rate"], 1e-9, failures)

    # A. Lead over P.
    rk = ps.rho(bundle, ps.PRIMARY_Q)
    lead = ps.analyse(cal, te, rk, tuple(names), args.noise_draws, args.noise_seed)
    checks["timer_mean_lead"] = check("P lead", lead["timer"]["mean_lead"], REFERENCE["timer_mean_lead"], 1e-4, failures)
    checks["timer_fp"] = check("P false alarms", lead["timer"]["fp"], REFERENCE["timer_fp"], 0, failures)
    checks["safe_gain_over_timer"] = check("SAFE + P gain", lead["safe_plus_timer"]["gain_over_timer"]["mean"],
                                           REFERENCE["safe_gain_over_timer"], 1e-4, failures)
    if contrasts:
        lead["contrasts"] = lead_contrasts(cal, te, rk, contrasts)

    # B. Fused with the global time risk.
    fused = fused_with_time(cal, te, names, target_fpr, safe_end, time_start, args.workers,
                            [pair for pair in contrasts if "safe" not in pair])
    fs = fused["detectors"]["fused_safe_time"]["test"]
    checks["fused_safe_tpr"] = check("fused SAFE TPR", fs["tpr"], REFERENCE["fused_safe_tpr"], 6e-4, failures)
    checks["fused_safe_fpr"] = check("fused SAFE FPR", fs["fpr"], REFERENCE["fused_safe_fpr"], 6e-4, failures)

    out = dict(status="exploratory (August cohorts, already opened)",
               note="scripts/xr1_lead_time/published_baselines_lead.py", inputs=inputs,
               counts=dict(calibration=len(cal), test=len(te)), detectors=names, detector_runs=source,
               contrasts=[list(c) for c in contrasts], checks=checks,
               reproduction_failures=failures, lead_over_P=lead, fused_with_global_time=fused)
    if args.compare_run:
        out["compare_run"] = dict(path=str(Path(args.compare_run).resolve()),
                                  evaluation=compare_runs(run_dirs, args.compare_run, names))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print_summary(out)
    if failures and not args.allow_reference_mismatch:
        print("REPRODUCTION FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("REPRODUCTION OK" if not failures else "REPRODUCTION MISMATCH (allowed by flag)")
    return 0


def print_summary(out):
    lead = out["lead_over_P"]
    print("\n=== A. lead over the per-task timer P (fraction of H_k), August test 380/380 ===")
    t = lead["timer"]
    print(f"P alone: fp {t['fp']}/{t['n_success']}  lead {t['mean_lead']:.4f}")
    rows = [("safe", lead["safe_plus_timer"], lead["safe_plus_timer"]["gain_over_timer"],
             lead["safe_plus_timer"]["vs_matched_fpr_timer"])]
    rows += [(k, v["test"], v["gain_over_timer"], v["vs_matched_fpr_timer"])
             for k, v in lead["all_features_descriptive"].items()]
    nn = lead.get("noise_null", {}).get("per_feature_vs_single_null", {})
    print(f"{'score + P':16s} {'fp':>4s} {'gain':>8s} {'95% CI':>18s} {'matched':>8s} {'95% CI':>18s} {'p_noise':>8s}")
    for k, e, g, m in rows:
        p = nn.get(k, {}).get("p_matched_gain", "")
        print(f"{k:16s} {e['fp']:4d} {g['mean']:+8.4f} {str(g['ci95']):>18s} "
              f"{m['gain']['mean']:+8.4f} {str(m['gain']['ci95']):>18s} {p!s:>8s}")
    for pair, c in lead.get("contrasts", {}).items():
        print(f"contrast {pair} (fp {c['fp'][0]} vs {c['fp'][1]}): raw {c['raw']['mean']:+.4f} {c['raw']['ci95']}"
              f"  matched {c['matched']['mean']:+.4f} {c['matched']['ci95']}")
    if "noise_null" in lead:
        s = lead["noise_null"]["single"]
        print(f"noise null (single, {lead['noise_null']['draws']} draws): gain mean {s['gain_mean']:+.4f} "
              f"(max {s['gain_max']:+.4f}), matched mean {s['matched_gain_mean']:+.4f} (max {s['matched_gain_max']:+.4f})")

    fused = out["fused_with_global_time"]
    print(f"\n=== B. score OR global time risk, thresholds at FPR <= {fused['target_fpr']} on calibration ===")
    print(f"{'detector':26s} {'TPR':>6s} {'FPR':>6s} {'FP':>4s} {'miss-adj':>8s} {'rec@25%':>8s}")
    for name, d in fused["detectors"].items():
        m = d["test"]
        print(f"{name:26s} {m['tpr']:6.4f} {m['fpr']:6.4f} {m['fp']:4d} {m['miss_adjusted_detection']:8.4f} "
              f"{m['recall_by'].get('0.25', float('nan')):8.4f}")
    print(f"{'pair':48s} {'dTPR [95% CI]':>28s} {'d miss-adj [95% CI]':>30s}")
    for pair, d in fused["paired"].items():
        b = d["bootstrap"]
        fmt = lambda x: f"{x['mean']:+.4f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
        print(f"{pair:48s} {fmt(b['tpr_delta']):>28s} {fmt(b['adjusted_detection_delta']):>30s}")
    if "compare_run" in out:
        print("\nrerun vs earlier run (evaluation trajectories):", json.dumps(out["compare_run"]["evaluation"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-run", required=True, nargs="+",
                    help="run_baselines output directories (detector names must not repeat)")
    ap.add_argument("--calibration", required=True, help="frozen August calibration_predictions.jsonl")
    ap.add_argument("--test", required=True, help="frozen August prospective event_predictions.jsonl")
    ap.add_argument("--bundle", required=True, help="frozen runtime_bundle.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detectors", nargs="+", help="subset of the run's detectors (default: all)")
    ap.add_argument("--contrast", nargs=2, action="append", metavar=("A", "B"),
                    help="also report lead(A + P) minus lead(B + P) with bootstrap CIs; 'safe' is allowed")
    ap.add_argument("--noise-draws", type=int, default=100)
    ap.add_argument("--noise-seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1, help="processes for the joint threshold search")
    ap.add_argument("--compare-run", help="earlier run_baselines directory to diff trajectories against")
    ap.add_argument("--allow-reference-mismatch", action="store_true",
                    help="write results even if the section 15.10/15.16 numbers are not reproduced")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
