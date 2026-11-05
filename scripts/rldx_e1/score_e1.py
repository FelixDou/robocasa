"""Score E1's new RLDX-1 rollouts with the frozen monitors (plan: docs/rldx_e1_plan.md).

Nothing is trained, chosen or calibrated here, and no metric is computed. Each mode first
reproduces the frozen monitor's recorded scores exactly, and writes no E1 score unless it does.

  hs    Hide-and-Seek, the three frozen models of the freeze manifest (R10 part B).
        Checks: model files and content hashes, the freeze run's normalizations.json and
        scoring.json hashes, the bundle and ID lists; then the reloaded models must give the
        freeze run's per-seed raw scores on every recorded calibration and test rollout
        (max |diff| <= 1e-12). E1 rows: the normalized seed-averaged running maximum with the
        freeze run's per-task normalization (score_cohorts' formula).
  safe  SAFE-LSTM, the three frozen final runs. Inputs are score_seen_checkpoint_external outputs
        (--group all) for the recorded export and for each E1 set's export.
        Checks: each final run's tree digest equals the freeze manifest's; each score directory was
        made with that run's checkpoint; the per-task normalization recomputed from the run's own
        training scores (calibrate_seen_tasks.fit_task_normalization) equals the recorded
        normalized_scores.jsonl's; the re-scored recorded rollouts equal their recorded raw scores.
        E1 rows: the seed mean of each seed's normalized running maximum (report 15.15's formula).

Outputs in --output-dir: <set>_<detector>_predictions.jsonl (rows: detector, rollout_id, task_name,
failed, horizon, safe_scores, per_seed_raw), and <detector>_scoring.json (inputs, hashes, checks).

Usage:
  python -I score_e1.py hs --freeze-manifest M.json --freeze-scores-dir <freeze>/scores_run2 \\
      --recorded-dataset RLDX_RAW --output-dir OUT calibration=<shard dir> ... test=<shard dir> ...
  python -I score_e1.py safe --freeze-manifest M.json --calibration-dir <SAFE20>/calibration/lstm \\
      --recorded-scores 0 DIR --recorded-scores 1 DIR --recorded-scores 2 DIR \\
      --e1-scores calibration 0 DIR ... --e1-scores test 2 DIR --output-dir OUT
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "xr1_lead_time"))

SEEDS = ("0", "1", "2")
HS, SAFE = "hide_and_seek", "safe_lstm"
EXACT = 1e-12


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        for r in rows:
            stream.write(json.dumps(r, default=_default) + "\n")


def _default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    raise TypeError(type(x))


def parse_sets(items):
    """['calibration=/a', 'test=/b', ...] -> {'calibration': ['/a'], 'test': ['/b']}."""
    out = defaultdict(list)
    for s in items:
        if "=" not in s:
            raise SystemExit(f"error: {s!r} is not <set>=<dir>")
        name, d = s.split("=", 1)
        out[name].append(d)
    if sorted(out) != ["calibration", "test"]:
        raise SystemExit(f"error: give shards of exactly the calibration and test sets, got {sorted(out)}")
    return dict(out)


def max_gap(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")


def normalized_mean(per_seed_raw, norms, task):
    """Seed mean of each seed's normalized running maximum."""
    return np.mean([(np.maximum.accumulate(np.asarray(per_seed_raw[s], float)) - norms[s][task]["location"])
                    / norms[s][task]["scale"] for s in SEEDS], axis=0)


def records(rows, per_seed_raw, norms, horizons, detector):
    out = []
    for r in rows:
        rid, t = r["rollout_id"], r["task_name"]
        raw = {s: np.asarray(per_seed_raw[s][rid], float) for s in SEEDS}
        if len({len(v) for v in raw.values()}) != 1:
            raise ValueError(f"{rid}: seeds give trajectories of different lengths")
        out.append(dict(detector=detector, rollout_id=rid, task_name=t, failed=bool(r["failed"]),
                        horizon=int(horizons[t]), safe_scores=normalized_mean(raw, norms, t),
                        per_seed_raw=raw))
    return out


# ---- Hide-and-Seek ---------------------------------------------------------------------------

def hs_reference_gap(score_fn, rows, reference):
    """Largest |reloaded - freeze-run| raw score over seeds and rows; reference: rid -> per_seed_raw."""
    worst, missing = 0.0, []
    for r in rows:
        ref = reference.get(r["rollout_id"])
        if ref is None:
            missing.append(r["rollout_id"])
            continue
        for s in SEEDS:
            worst = max(worst, max_gap(score_fn(s, r), ref[s]))
    return worst, missing


def run_hs(a):
    from robocasa.recovery.safe.baselines.detectors import load_detector
    from robocasa.recovery.safe.baselines.run_baselines import index_datasets, load_baseline_rollout, load_rollout_ids

    out = Path(a.output_dir)
    if (out / f"{HS}_scoring.json").exists():
        raise SystemExit(f"error: {out} already has Hide-and-Seek scores")
    sets = parse_sets(a.shards)
    man = json.loads(Path(a.freeze_manifest).read_text())
    problems = []
    files = man["files"]
    for key in ("hs_rldx_bundle", "hs_calibration_ids", "hs_test_ids"):
        if sha256(files[key]["path"]) != files[key]["sha256"]:
            problems.append(f"{key} changed since the freeze")
    sd = Path(a.freeze_scores_dir)
    fr = man["checks"]["freeze_run"]
    if sha256(sd / "scoring.json") != fr["scoring_json_sha256"]:
        problems.append("freeze run scoring.json changed")
    if sha256(sd / "normalizations.json") != fr["normalizations_sha256"]:
        problems.append("freeze run normalizations.json changed")
    scoring = json.loads((sd / "scoring.json").read_text())
    agg = scoring["feature_aggregation"]
    models = {}
    for s in SEEDS:
        m = man["detectors"]["hide_and_seek_plus_P"]["models"][s]
        if sha256(m["path"]) != m["sha256"]:
            problems.append(f"model seed {s} file changed")
        det, content = load_detector(m["path"])
        if content != m["content_sha256"]:
            problems.append(f"model seed {s} content changed")
        models[s] = det
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems))
    norms = json.loads((sd / "normalizations.json").read_text())[HS]
    bundle = json.loads(Path(files["hs_rldx_bundle"]["path"]).read_text())
    horizons = {t: int(v) for t, v in bundle["task_horizons"].items()}
    frozen_H = {t: int(v["H_k"]) for t, v in man["tasks"].items()}
    if horizons != frozen_H:
        raise SystemExit("STOP: the bundle's horizons differ from the freeze manifest's H_k")

    dirs = [a.recorded_dataset] + [d for v in sets.values() for d in v]
    index = index_datasets(dirs)
    cache = {}

    def load(rid):
        if rid not in cache:
            cache[rid] = load_baseline_rollout(*index[rid], agg)
        return cache[rid]

    def score(s, r):
        return np.asarray(models[s].score(r), np.float64)

    checks = {}
    for cohort, key in (("calibration", "hs_calibration_ids"), ("test", "hs_test_ids")):
        ids = load_rollout_ids(files[key]["path"])
        ref = {}
        for row in jsonl(sd / f"{cohort}_predictions.jsonl"):
            if row["detector"] == HS:
                ref[str(row["rollout_id"])] = {s: row["per_seed_raw"][s] for s in SEEDS}
        rows = [load(i) for i in ids]
        gap, missing = hs_reference_gap(score, rows, ref)
        ok = not missing and len(ref) == len(ids) and gap <= EXACT
        checks[f"recorded_{cohort}"] = dict(rollouts=len(ids), max_abs_diff=gap, missing=missing[:5], ok=ok)
        print(f"reproduce recorded {cohort}: {len(ids)} rollouts, max |diff| {gap:.3g} -> {'OK' if ok else 'MISMATCH'}",
              flush=True)
    if not all(c["ok"] for c in checks.values()):
        raise SystemExit("STOP: the reloaded models do not reproduce the freeze run; no E1 score written")

    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, ds in sets.items():
        sub = index_datasets(ds)
        rows = [load(rid) for rid in sorted(sub)]
        unknown = sorted({r["task_name"] for r in rows} - set(horizons))
        if unknown:
            raise SystemExit(f"STOP: {name} has tasks without a frozen horizon: {unknown}")
        raw = {s: {} for s in SEEDS}
        for r in rows:
            for s in SEEDS:
                v = score(s, r)
                if v.shape != (len(r["features"]),) or not np.all(np.isfinite(v)):
                    raise SystemExit(f"STOP: invalid Hide-and-Seek scores for {r['rollout_id']}")
                raw[s][r["rollout_id"]] = v
        write_jsonl(out / f"{name}_{HS}_predictions.jsonl", records(rows, raw, norms, horizons, HS))
        counts[name] = len(rows)
        print(f"E1 {name}: {len(rows)} rollouts scored", flush=True)
    meta = dict(protocol="e1_frozen_scoring_no_metrics", plan="docs/rldx_e1_plan.md", detector=HS,
                freeze_manifest=dict(path=str(Path(a.freeze_manifest).resolve()), sha256=sha256(a.freeze_manifest)),
                freeze_scores_dir=str(sd.resolve()), feature_aggregation=agg, recorded_dataset=str(Path(a.recorded_dataset).resolve()),
                shards={k: [str(Path(d).resolve()) for d in v] for k, v in sets.items()}, counts=counts, checks=checks)
    (out / f"{HS}_scoring.json").write_text(json.dumps(meta, indent=1, default=_default) + "\n")
    print("SCORING OK")


# ---- SAFE-LSTM -------------------------------------------------------------------------------

def safe_normalizations(training_rows):
    """Per-task normalization of one final run, from its own training scores (SAFE's formula)."""
    from robocasa.recovery.safe.calibrate_seen_tasks import fit_task_normalization
    return fit_task_normalization([r for r in training_rows if r.get("split") == "train"])


def check_safe_recorded(norms, recorded_by_seed, rescored_by_seed):
    """Normalization vs the recorded normalized_scores rows, and re-scored raw vs recorded raw.

    recorded_by_seed[s]: rid -> normalized_scores.jsonl row; rescored_by_seed[s]: rid -> scores.jsonl row.
    Only the recorded calibration and evaluation rows are compared for the raw scores."""
    norm_gap, raw_gap, missing, n = 0.0, 0.0, [], 0
    for s in SEEDS:
        for rid, r in recorded_by_seed[s].items():
            mine = norms[s][r["task_name"]]
            norm_gap = max(norm_gap, abs(mine["location"] - float(r["normalization_location"])),
                           abs(mine["scale"] - float(r["normalization_scale"])))
            if r.get("split") not in ("calibration", "evaluation"):
                continue
            x = rescored_by_seed[s].get(rid)
            if x is None:
                missing.append(rid)
                continue
            raw_gap = max(raw_gap, max_gap(x["scores"], r["full_raw_scores"]))
            n += 1
    return norm_gap, raw_gap, missing, n


def run_safe(a):
    import e2_freeze_manifest as efm

    out = Path(a.output_dir)
    if (out / f"{SAFE}_scoring.json").exists():
        raise SystemExit(f"error: {out} already has SAFE-LSTM scores")
    man = json.loads(Path(a.freeze_manifest).read_text())
    runs = {s: Path(man["trees"][f"safe_lstm_seed{s}"]["path"]) for s in SEEDS}
    problems = []
    for s, run in runs.items():
        if efm.tree_hash(run)["digest"] != man["trees"][f"safe_lstm_seed{s}"]["digest"]:
            problems.append(f"final run seed {s} changed since the freeze")
    if efm.tree_hash(a.calibration_dir)["digest"] != man["trees"]["safe_lstm_calibration"]["digest"]:
        problems.append("the recorded calibration directory changed since the freeze")

    def scores_dir(d, s):
        d = Path(d)
        prov = json.loads((d / "provenance.json").read_text())
        if Path(prov["training_run_dir"]).resolve() != runs[s].resolve():
            problems.append(f"{d}: scored with {prov['training_run_dir']}, not seed {s}'s final run")
        if prov["checkpoint_sha256"] != sha256(runs[s] / "model_final.ckpt"):
            problems.append(f"{d}: checkpoint hash differs")
        if prov.get("group") != "all":
            problems.append(f"{d}: group {prov.get('group')}, not all")
        return {str(r["rollout_id"]): r for r in jsonl(d / "scores.jsonl")}, prov

    rec_dirs = {str(s): d for s, d in a.recorded_scores}
    e1_dirs = defaultdict(dict)
    for name, s, d in a.e1_scores:
        e1_dirs[name][str(s)] = d
    if sorted(rec_dirs) != list(SEEDS) or sorted(e1_dirs) != ["calibration", "test"] or \
            any(sorted(v) != list(SEEDS) for v in e1_dirs.values()):
        raise SystemExit("error: give --recorded-scores for seeds 0-2 and --e1-scores for both sets and seeds 0-2")
    rescored = {s: scores_dir(rec_dirs[s], s)[0] for s in SEEDS}
    e1 = {name: {s: scores_dir(v[s], s) for s in SEEDS} for name, v in e1_dirs.items()}
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems))
    norms = {s: safe_normalizations(jsonl(runs[s] / "scores.jsonl")) for s in SEEDS}
    recorded = {s: {str(r["rollout_id"]): r for r in jsonl(Path(a.calibration_dir) / f"lstm_seed{s}" / "normalized_scores.jsonl")}
                for s in SEEDS}
    norm_gap, raw_gap, missing, n = check_safe_recorded(norms, recorded, rescored)
    tol = float(a.tolerance)
    checks = dict(normalization=dict(max_abs_diff=norm_gap, ok=norm_gap <= EXACT),
                  recorded_raw=dict(rows=n, missing=missing[:5], max_abs_diff=raw_gap, tolerance=tol,
                                    ok=not missing and raw_gap <= tol))
    for k, v in checks.items():
        print(f"reproduce {k}: max |diff| {v['max_abs_diff']:.3g} -> {'OK' if v['ok'] else 'MISMATCH'}", flush=True)
    if not all(v["ok"] for v in checks.values()):
        raise SystemExit("STOP: the frozen SAFE-LSTM runs do not reproduce the recorded scores; no E1 score written")

    horizons = {t: int(v["H_k"]) for t, v in man["tasks"].items()}
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, by_seed in e1.items():
        ids = sorted(by_seed["0"][0])
        if any(sorted(by_seed[s][0]) != ids for s in SEEDS):
            raise SystemExit(f"STOP: {name}: seeds scored different rollouts")
        rows = []
        for rid in ids:
            r0 = by_seed["0"][0][rid]
            if any((by_seed[s][0][rid]["task_name"], bool(by_seed[s][0][rid]["failed"])) != (r0["task_name"], bool(r0["failed"]))
                   for s in SEEDS):
                raise SystemExit(f"STOP: {name}: seeds disagree on {rid}")
            if r0["task_name"] not in horizons:
                raise SystemExit(f"STOP: {name}: task {r0['task_name']} has no frozen horizon")
            rows.append(dict(rollout_id=rid, task_name=r0["task_name"], failed=bool(r0["failed"])))
        raw = {s: {rid: by_seed[s][0][rid]["scores"] for rid in ids} for s in SEEDS}
        write_jsonl(out / f"{name}_{SAFE}_predictions.jsonl", records(rows, raw, norms, horizons, SAFE))
        counts[name] = len(rows)
        print(f"E1 {name}: {len(rows)} rollouts scored", flush=True)
    meta = dict(protocol="e1_frozen_scoring_no_metrics", plan="docs/rldx_e1_plan.md", detector=SAFE,
                freeze_manifest=dict(path=str(Path(a.freeze_manifest).resolve()), sha256=sha256(a.freeze_manifest)),
                final_runs={s: str(r) for s, r in runs.items()},
                recorded_scores=rec_dirs, e1_scores={k: dict(v) for k, v in e1_dirs.items()},
                e1_exports={k: sorted({v[s][1]["export_dir"] for s in SEEDS}) for k, v in e1.items()},
                normalizations=norms, counts=counts, checks=checks)
    (out / f"{SAFE}_scoring.json").write_text(json.dumps(meta, indent=1, default=_default) + "\n")
    print("SCORING OK")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    h = sub.add_parser("hs")
    h.add_argument("--freeze-manifest", required=True)
    h.add_argument("--freeze-scores-dir", required=True)
    h.add_argument("--recorded-dataset", required=True)
    h.add_argument("--output-dir", required=True)
    h.add_argument("shards", nargs="+", help="<set>=<E1 collector shard dir>")
    s = sub.add_parser("safe")
    s.add_argument("--freeze-manifest", required=True)
    s.add_argument("--calibration-dir", required=True, help="recorded <SAFE20>/calibration/lstm")
    s.add_argument("--recorded-scores", nargs=2, action="append", required=True, metavar=("SEED", "DIR"))
    s.add_argument("--e1-scores", nargs=3, action="append", required=True, metavar=("SET", "SEED", "DIR"))
    s.add_argument("--tolerance", type=float, default=EXACT,
                   help="largest |re-scored - recorded| raw SAFE score accepted (default exact, 1e-12)")
    s.add_argument("--output-dir", required=True)
    a = p.parse_args(argv)
    (run_hs if a.mode == "hs" else run_safe)(a)


if __name__ == "__main__":
    main()
