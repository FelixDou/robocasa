"""Frozen Hide-and-Seek + P alarm for recovery experiments (Phase A).

Plan: docs/recovery_harvest_finetune_plan.md. The alarm is the registered
Hide-and-Seek + P of docs/xr1_confirmatory_amendment_2_hide_and_seek.md (report
section 15.21). Nothing is refitted: the detector seeds are retrained with the
registered settings and must reproduce every Amendment 2 alarm on a reference
cohort, the per-task normalizations are the stored Amendment 2 values, and the
timer P and the fused path (u, s, lam) are recomputed with the registered code
on the registered calibration cohort and must match its stored result.

Commands (CPU only, SAFE environment with PyTorch and scikit-learn):

``export``
    Reads the Amendment 2 scoring run (``scoring.json``: dataset dirs, runtime
    bundle, training IDs, seeds, epochs), retrains the Hide-and-Seek seeds as
    ``score_cohorts`` did (or copies them from an earlier export with
    ``--reuse-detectors``), saves and reloads them. The reloaded detectors must
    give the same alarm as the stored trajectory on every reference rollout,
    with raw scores within 1e-5 and recomputed normalizations within 1e-6
    (retraining on another CPU can change float32 scores by a few ulps).
    Whether the reproduction was bit-exact is recorded too. Refits P
    and the fused path on the Amendment 2 calibration cohort and requires them
    to equal ``confirm_hide_and_seek.json``. Measures how storing features in
    float16 (as the ladder v2 collector did) moves alarms on the reference
    cohort. Writes a spec directory.

``score-ladder``
    Scores recoverability-ladder parents from their recorded features and
    writes one alarm row per parent.

Alarm rule (``scripts/xr1_lead_time/quantile_timer.alarm``): with R_t the
seed-averaged, per-task normalized running maximum of the Hide-and-Seek score at
1-based inference t, H the task horizon in inferences and rho_k the q = 0.95
quantile of development success durations,

    timer = ceil(min(1, rho_k + lam) * H - eps)
    score = first t with R_t >= u + s * lam
    alarm = min(timer, score), raised if alarm <= L (rollout length in inferences).

The alarm is known at the policy's ``alarm``-th inference. In a ladder parent,
snapshots are taken at every inference, so that inference's snapshot is the one
whose step equals the step of feature row ``alarm - 1``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np


DETECTOR = "hide_and_seek"
SPEC_NAME = "alarm_spec.json"
SPEC_KIND = "hs_p_alarm_spec"
SPEC_VERSION = 1
EPS = 1e-9
PRIMARY_Q = 0.95
EXACT_TOLERANCE = 1e-12
# Retraining on another node can change float32 scores by a few ulps (BLAS and
# randomized-PCA kernels differ by CPU). The export accepts that only if every
# reference alarm is unchanged; the exact flags are recorded either way.
RAW_TOLERANCE = 1e-5
COMBINED_TOLERANCE = 1e-4
NORM_TOLERANCE = 1e-6
PLAN = "docs/recovery_harvest_finetune_plan.md"
AMENDMENT = "docs/xr1_confirmatory_amendment_2_hide_and_seek.md"
LEAD_TIME_DIR = Path(__file__).resolve().parents[2] / "scripts" / "xr1_lead_time"


# ---- small utilities ------------------------------------------------------------------------

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _finite_or_none(value):
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _inf_if_none(value):
    return math.inf if value is None else float(value)


# ---- the alarm rule (numpy only) ------------------------------------------------------------

def first_crossing(values, tau):
    """1-based index of the first running-max value >= tau (lead_time.first); inf if none."""
    if tau is None or not math.isfinite(float(tau)):
        return math.inf
    running = np.maximum.accumulate(np.asarray(values, dtype=np.float64))
    index = int(np.searchsorted(running, float(tau), side="left"))
    return math.inf if index >= len(running) else index + 1


def timer_index(horizon, rho, lam):
    return math.ceil(min(1.0, float(rho) + float(lam)) * float(horizon) - EPS)


def alarm_index(scores, horizon, rho, path):
    """``quantile_timer.alarm`` for one rollout. ``path``: {lam, u, s}; u None or inf = timer only."""
    lam = float(path["lam"])
    u = _inf_if_none(path.get("u"))
    s = float(path.get("s") or 0.0)
    timer = timer_index(horizon, rho, lam)
    score = math.inf if u == math.inf else first_crossing(scores, u + s * lam)
    return {"alarm": min(timer, score), "timer": timer, "score": score}


def alarm_component(timer, score):
    if score < timer:
        return "score"
    if timer < score:
        return "timer"
    return "both"


def combined_score(raw_by_seed, normalizations, task):
    """Seed average of per-task normalized running maxima (run_baselines.build_records)."""
    values = []
    for seed in sorted(raw_by_seed, key=int):
        stats = normalizations[str(seed)][task]
        running = np.maximum.accumulate(np.asarray(raw_by_seed[seed], dtype=np.float64))
        values.append((running - float(stats["location"])) / float(stats["scale"]))
    return np.mean(values, axis=0)


class AlarmSpec:
    """Frozen alarm parameters; detectors load on demand (needs PyTorch)."""

    def __init__(self, payload, directory=None):
        if payload.get("kind") != SPEC_KIND:
            raise ValueError("Not a Hide-and-Seek + P alarm spec")
        self.payload = payload
        self.directory = None if directory is None else Path(directory)
        self._detectors = None

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        return cls(json.loads((directory / SPEC_NAME).read_text()), directory)

    @property
    def tasks(self):
        return self.payload["tasks"]

    @property
    def seeds(self):
        return [int(seed) for seed in self.payload["seeds"]]

    @property
    def path(self):
        return self.payload["fused"]

    def task_entry(self, task):
        if task not in self.tasks:
            raise ValueError(f"Task {task} is not among the frozen alarm tasks")
        return self.tasks[task]

    def alarm(self, task, scores):
        """Alarm of one rollout from its combined score trajectory (length L)."""
        entry = self.task_entry(task)
        result = alarm_index(scores, entry["horizon_inferences"], entry["rho"], self.path)
        length = len(scores)
        result["fired"] = result["alarm"] <= length
        result["component"] = alarm_component(result["timer"], result["score"]) if result["fired"] else None
        return result

    def detectors(self, device="cpu"):
        if self._detectors is None:
            if self.directory is None:
                raise ValueError("Spec has no directory with saved detectors")
            self._detectors = {}
            for seed in self.seeds:
                path = self.directory / self.payload["detector_files"][str(seed)]
                if sha256(path) != self.payload["detector_sha256"][str(seed)]:
                    raise ValueError(f"Saved detector {path} differs from the spec's checksum")
                self._detectors[seed] = load_detector(path, device=device)
        return self._detectors

    def raw_scores(self, task, features, device="cpu"):
        rollout = {"rollout_id": "spec", "task_name": task, "failed": False,
                   "features": np.asarray(features, dtype=np.float32)}
        out = {}
        for seed, detector in self.detectors(device).items():
            values = np.asarray(detector.score(rollout), dtype=np.float64)
            if values.shape != (len(features),) or not np.all(np.isfinite(values)):
                raise ValueError(f"Detector seed {seed} produced invalid scores")
            out[seed] = values
        return out

    def score(self, task, features, device="cpu"):
        return combined_score(self.raw_scores(task, features, device), self.payload["normalizations"], task)


# ---- detector persistence (PyTorch) ----------------------------------------------------------

def save_detector(detector, path):
    import torch

    model = detector.model_
    payload = {
        "name": detector.name,
        "config": detector.config(),
        "input_dim": int(model.lstm.input_size),
        "hidden_dim": int(model.lstm.hidden_size),
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "mean": np.asarray(detector.mean_),
        "std": np.asarray(detector.std_),
        "pca": detector.pca_,
    }
    torch.save(payload, path)


def load_detector(path, device="cpu"):
    import torch

    from robocasa.recovery.safe.baselines.detectors import HideAndSeekDetector, _lstm_scorer

    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 1.13 has no weights_only
        payload = torch.load(path, map_location=device)
    if payload["name"] != DETECTOR:
        raise ValueError(f"{path} holds {payload['name']}, not {DETECTOR}")
    config = dict(payload["config"])
    detector = HideAndSeekDetector(
        seed=config["seed"], margin=config["margin"], intra_weight=config["intra_weight"],
        pca_components=config["pca_components"], epochs=config["epochs"],
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        hidden_dim=config["hidden_dim"], batch_size=config["batch_size"], device=device,
    )
    detector.pca_ = payload["pca"]
    detector.mean_ = np.asarray(payload["mean"])
    detector.std_ = np.asarray(payload["std"])
    model = _lstm_scorer(torch, payload["input_dim"], payload["hidden_dim"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    detector.model_ = model
    return detector


# ---- export -----------------------------------------------------------------------------------

def _lead_time_modules():
    if str(LEAD_TIME_DIR) not in sys.path:
        sys.path.insert(0, str(LEAD_TIME_DIR))
    import confirm_lead  # noqa: E402
    import quantile_timer  # noqa: E402

    return confirm_lead, quantile_timer


def _hs_rows(path):
    rows = {}
    for row in read_jsonl(path):
        if row.get("detector") != DETECTOR:
            continue
        rid = str(row["rollout_id"])
        if rid in rows:
            raise ValueError(f"{path}: duplicate {DETECTOR} row for {rid}")
        rows[rid] = row
    if not rows:
        raise ValueError(f"{path} has no {DETECTOR} rows")
    return rows


def refit_paths(e1_calibration, hs_calibration_scores, bundle):
    """P and Hide-and-Seek + P on the registered calibration cohort (confirm_hide_and_seek.run)."""
    cl, qt = _lead_time_modules()
    _, cal = cl.primary(e1_calibration, "calibration")
    scores = _hs_rows(hs_calibration_scores)
    for r in cal:
        row = scores.get(str(r["rollout_id"]))
        if row is None:
            raise ValueError(f"No {DETECTOR} calibration trajectory for {r['rollout_id']}")
        if bool(row["failed"]) != r["failed"] or row["task_name"] != r["task_name"]:
            raise ValueError(f"Label or task differs for {r['rollout_id']}")
        values = np.asarray(row["safe_scores"], dtype=np.float64)
        if len(values) != r["L"]:
            raise ValueError(f"{r['rollout_id']}: {len(values)} inferences, SAFE row has {r['L']}")
        r[DETECTOR] = values
    rho = qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)
    timer, fused = cl.fit_fast(cl.Cohort(cal, DETECTOR), rho)
    return rho, timer, fused, len(cal), cl.det_json


def _path_json(path):
    return {"lam": float(path["lam"]), "u": _finite_or_none(path["u"]), "s": float(path["s"])}


def _same_rounded(stored, refit, det_json):
    got = det_json(refit)
    want = {"lam": stored["lam"], "u": stored["u"], "s": stored["s"]}
    if want["u"] is None:
        want["u"] = math.inf
    return {key: float(got[key]) for key in got} == {key: float(want[key]) for key in want}, got, want


def export(args):
    started = time.time()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"error: output directory is not empty: {output}")
    from robocasa.recovery.safe.analyze_time_safe_hybrid import fit_safe_normalization
    from robocasa.recovery.safe.baselines.detectors import build_detector
    from robocasa.recovery.safe.baselines.run_baselines import (
        index_datasets,
        load_baseline_rollout,
        load_rollout_ids,
    )

    scores_dir = Path(args.amendment_scores_dir)
    scoring = json.loads((scores_dir / "scoring.json").read_text())
    if DETECTOR not in scoring.get("detectors", {}):
        raise ValueError(f"{scores_dir}/scoring.json did not score {DETECTOR}")
    inputs = scoring["inputs"]
    bundle_path = Path(inputs["runtime_bundle"]["path"])
    train_ids_path = Path(inputs["train_ids"]["path"])
    for label, path, want in (("runtime bundle", bundle_path, inputs["runtime_bundle"]["sha256"]),
                              ("training IDs", train_ids_path, inputs["train_ids"]["sha256"])):
        if sha256(path) != want:
            raise ValueError(f"The {label} {path} changed since the Amendment 2 scoring run")
    bundle = json.loads(bundle_path.read_text())
    horizons = {task: int(value) for task, value in bundle["task_horizons"].items()}
    seeds = [int(seed) for seed in scoring["seeds"]]
    aggregation = scoring["feature_aggregation"]
    epochs = int(scoring["epochs"])
    stored_norms = json.loads((scores_dir / "normalizations.json").read_text())[DETECTOR]
    stored_configs = scoring["detectors"][DETECTOR]["configs"]

    train_ids = load_rollout_ids(train_ids_path, split=args.train_split)
    if len(train_ids) != int(scoring["counts"]["train"]):
        raise ValueError(f"{len(train_ids)} training IDs, the scoring run used {scoring['counts']['train']}")
    reference_path = scores_dir / f"{args.reference_cohort}_predictions.jsonl"
    reference = _hs_rows(reference_path)
    index = index_datasets(scoring["dataset_dirs"])
    missing = [rid for rid in (*train_ids, *reference) if rid not in index]
    if missing:
        raise ValueError(f"{len(missing)} rollouts missing from the scoring run's datasets, e.g. {missing[:3]}")
    load = lambda ids: [load_baseline_rollout(*index[rid], aggregation) for rid in ids]
    train = load(train_ids)
    ref_rollouts = load(sorted(reference))
    print(f"train {len(train)}; reference {args.reference_cohort} {len(ref_rollouts)}", flush=True)

    checks = {"compute": _compute_environment()}
    detector_dir = output / "detectors"
    detector_dir.mkdir(parents=True)
    files, digests, configs, train_scores = {}, {}, {}, {}
    reuse = Path(args.reuse_detectors) if getattr(args, "reuse_detectors", None) else None
    for seed in seeds:
        path = detector_dir / f"seed{seed}.pt"
        if reuse is None:
            detector = build_detector(DETECTOR, seed=seed, device="cpu", epochs=epochs)
            detector.fit(train)
            save_detector(detector, path)
            verb = "trained"
        else:
            shutil.copyfile(reuse / f"seed{seed}.pt", path)
            verb = f"reused from {reuse}"
        reloaded = load_detector(path)
        configs[str(seed)] = reloaded.config()
        if configs[str(seed)] != stored_configs[str(seed)]:
            raise ValueError(f"Seed {seed} config differs from the scoring run: {configs[str(seed)]}")
        files[str(seed)] = str(path.relative_to(output))
        digests[str(seed)] = sha256(path)
        train_scores[seed] = {r["rollout_id"]: np.asarray(reloaded.score(r), np.float64) for r in train}
        print(f"seed {seed} {verb}, saved and reloaded", flush=True)
    checks["detectors_reused_from"] = None if reuse is None else str(reuse.resolve())

    # The spec keeps the Amendment 2 normalizations: the fused path was fitted on
    # trajectories normalized with them. Recomputing them checks the detectors.
    normalizations = {str(seed): stored_norms[str(seed)] for seed in seeds}
    recomputed = {
        str(seed): fit_safe_normalization(
            [{"task_name": r["task_name"], "scores": train_scores[seed][r["rollout_id"]]} for r in train], horizons)
        for seed in seeds
    }
    worst = 0.0
    for seed in seeds:
        for task, stats in stored_norms[str(seed)].items():
            mine = recomputed[str(seed)][task]
            worst = max(worst, abs(mine["location"] - stats["location"]), abs(mine["scale"] - stats["scale"]))
    checks["normalizations_max_abs_diff"] = worst
    checks["normalizations_exact"] = worst <= EXACT_TOLERANCE
    if worst > NORM_TOLERANCE:
        raise ValueError(f"Normalizations differ from the scoring run (max diff {worst} > {NORM_TOLERANCE})")

    spec_payload = {
        "kind": SPEC_KIND,
        "version": SPEC_VERSION,
        "plan": PLAN,
        "amendment": AMENDMENT,
        "detector": DETECTOR,
        "seeds": seeds,
        "feature_aggregation": aggregation,
        "detector_configs": configs,
        "detector_files": files,
        "detector_sha256": digests,
        "normalizations": normalizations,
        "q": PRIMARY_Q,
        "eps": EPS,
    }

    # Paths, refitted with the registered code and compared with the stored result.
    rho, timer, fused, n_cal, det_json = refit_paths(args.e1_calibration, args.hs_calibration_scores, bundle)
    stored = json.loads(Path(args.hs_result).read_text())
    ok_p, got_p, want_p = _same_rounded(stored["P"], timer, det_json)
    ok_f, got_f, want_f = _same_rounded(stored["detectors"][DETECTOR]["path"], fused, det_json)
    checks["P_matches_amendment2"] = dict(ok=ok_p, refit=got_p, stored=want_p)
    checks["fused_matches_amendment2"] = dict(ok=ok_f, refit=got_f, stored=want_f)
    checks["calibration_rollouts"] = n_cal
    if not (ok_p and ok_f):
        raise ValueError(f"Refitted paths differ from {args.hs_result}: {checks}")
    spec_payload["P"] = _path_json(timer)
    spec_payload["fused"] = _path_json(fused)
    spec_payload["tasks"] = {
        task: {"horizon_inferences": horizons[task], "rho": float(rho[task])} for task in sorted(horizons)
    }
    spec = AlarmSpec(spec_payload, output)

    # Reference trajectories from the reloaded detectors, and float16 sensitivity.
    worst_raw = worst_norm = worst_fp16 = 0.0
    changed, shifts, reference_alarms = 0, [], Counter()
    alarm_mismatches = []
    for r in ref_rollouts:
        row = reference[r["rollout_id"]]
        raw = spec.raw_scores(r["task_name"], r["features"])
        for seed in seeds:
            want = np.asarray(row["per_seed_raw"][str(seed)], dtype=np.float64)
            if want.shape != raw[seed].shape:
                raise ValueError(f"{r['rollout_id']}: reference length differs for seed {seed}")
            worst_raw = max(worst_raw, float(np.max(np.abs(raw[seed] - want))))
        combined = combined_score(raw, normalizations, r["task_name"])
        stored_combined = np.asarray(row["safe_scores"], float)
        worst_norm = max(worst_norm, float(np.max(np.abs(combined - stored_combined))))
        stored_alarm = spec.alarm(r["task_name"], stored_combined)
        new_alarm = spec.alarm(r["task_name"], combined)
        if (stored_alarm["fired"], stored_alarm["alarm"]) != (new_alarm["fired"], new_alarm["alarm"]):
            alarm_mismatches.append(r["rollout_id"])
        half = r["features"].astype(np.float16).astype(np.float32)
        combined16 = spec.score(r["task_name"], half)
        worst_fp16 = max(worst_fp16, float(np.max(np.abs(combined16 - combined))))
        a32, a16 = spec.alarm(r["task_name"], combined), spec.alarm(r["task_name"], combined16)
        reference_alarms[(bool(r["failed"]), a32["fired"])] += 1
        if (a32["fired"], a32["alarm"]) != (a16["fired"], a16["alarm"]):
            changed += 1
            if a32["fired"] and a16["fired"]:
                shifts.append(int(a16["alarm"] - a32["alarm"]))
    checks["reference_cohort"] = args.reference_cohort
    checks["reference_rollouts"] = len(ref_rollouts)
    checks["reference_raw_max_abs_diff"] = worst_raw
    checks["reference_combined_max_abs_diff"] = worst_norm
    checks["reference_exact"] = worst_raw <= EXACT_TOLERANCE
    checks["reference_alarm_mismatches"] = len(alarm_mismatches)
    if alarm_mismatches or worst_raw > RAW_TOLERANCE or worst_norm > COMBINED_TOLERANCE:
        raise ValueError(
            f"Reloaded detectors do not reproduce {reference_path}: raw {worst_raw}, combined {worst_norm}, "
            f"{len(alarm_mismatches)} alarms differ (e.g. {alarm_mismatches[:3]})")
    checks["float16"] = {
        "combined_max_abs_diff": worst_fp16,
        "rollouts_with_changed_alarm": changed,
        "alarm_shift_inferences": dict(Counter(shifts)),
    }
    checks["reference_alarms"] = {
        f"{'failure' if failed else 'success'}_{'alarm' if fired else 'silent'}": count
        for (failed, fired), count in sorted(reference_alarms.items())
    }
    spec_payload["checks"] = checks
    spec_payload["inputs"] = {
        "amendment_scoring": {"path": str((scores_dir / "scoring.json").resolve()),
                              "sha256": sha256(scores_dir / "scoring.json")},
        "amendment_normalizations": {"path": str((scores_dir / "normalizations.json").resolve()),
                                     "sha256": sha256(scores_dir / "normalizations.json")},
        "reference_predictions": {"path": str(reference_path.resolve()), "sha256": sha256(reference_path)},
        "runtime_bundle": {"path": str(bundle_path), "sha256": inputs["runtime_bundle"]["sha256"]},
        "train_ids": {"path": str(train_ids_path), "sha256": inputs["train_ids"]["sha256"]},
        "e1_calibration": {"path": str(Path(args.e1_calibration).resolve()), "sha256": sha256(args.e1_calibration)},
        "hs_calibration_scores": {"path": str(Path(args.hs_calibration_scores).resolve()),
                                  "sha256": sha256(args.hs_calibration_scores)},
        "hs_result": {"path": str(Path(args.hs_result).resolve()), "sha256": sha256(args.hs_result)},
    }
    spec_payload["versions"] = _versions()
    spec_payload["code_commit"] = code_commit()
    spec_payload["elapsed_seconds"] = time.time() - started
    write_json(output / SPEC_NAME, spec_payload)
    print(json.dumps(checks, indent=1, default=_json_default))
    print(f"EXPORT OK: {output / SPEC_NAME}")
    return 0


def _compute_environment():
    out = {"machine": platform.machine(), "node": platform.node(), "cpu": None, "torch_threads": None}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                out["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        import torch

        out["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    return out


def _versions():
    out = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("torch", "sklearn"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


# ---- score ladder parents ---------------------------------------------------------------------

def _parents(run_dir):
    rows = read_jsonl(Path(run_dir) / "parents.jsonl")
    ids = [row["parent_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{run_dir}: duplicate parent IDs in parents.jsonl")
    return rows


def reduce_features(raw, aggregation):
    from robocasa.recovery.safe.dataset import aggregate_features

    return aggregate_features(np.asarray(raw), aggregation).copy()


def parent_alarm(parent, combined, feature_steps, spec, feature_dtype):
    """One alarm row: the alarm against the parent's snapshots and the ladder's anchors."""
    task = parent["task_name"]
    entry = spec.task_entry(task)
    length = len(combined)
    result = spec.alarm(task, combined)
    steps = list(parent["snapshot_steps"])
    inference_steps = steps[:-1]  # the last snapshot is the terminal state
    row = {
        "parent_id": parent["parent_id"],
        "attempt_uid": parent["attempt_uid"],
        "task_name": task,
        "environment_seed": parent.get("environment_seed"),
        "success": bool(parent["success"]),
        "ladder_role": parent.get("role"),
        "num_inferences": int(length),
        "horizon_inferences": int(entry["horizon_inferences"]),
        "horizon_steps": int(parent["horizon"]),
        "length_matches_horizon": bool(parent["success"]) or length == int(entry["horizon_inferences"]),
        "feature_steps_match_snapshots": [int(s) for s in feature_steps] == [int(s) for s in inference_steps],
        "feature_dtype": feature_dtype,
        "alarm_fired": bool(result["fired"]),
        "alarm_inference": None if not result["fired"] else int(result["alarm"]),
        "timer_inference": _finite_or_none(result["timer"]),
        "score_inference": _finite_or_none(result["score"]),
        "component": result["component"],
        "max_score": float(np.max(combined)) if length else None,
        "alarm_snapshot_index": None,
        "alarm_step": None,
        "alarm_fraction": None,
    }
    if result["fired"]:
        a = int(result["alarm"])
        step = int(feature_steps[a - 1])
        matches = [i for i, s in enumerate(inference_steps) if int(s) == step]
        if len(matches) != 1:
            raise ValueError(f"{parent['parent_id']}: alarm step {step} matches {len(matches)} snapshots")
        row.update(
            alarm_snapshot_index=int(matches[0]),
            alarm_step=step,
            alarm_fraction=step / float(parent["horizon"]),
            score_at_alarm=float(combined[a - 1]),
        )
        anchors = parent.get("anchors") or {}
        if "detection" in anchors:
            row["ladder_detection_index"] = int(anchors["detection"])
            row["ladder_detection_step"] = int(parent["anchor_steps"]["detection"])
            row["alarm_minus_ladder_detection"] = int(matches[0]) - int(anchors["detection"])
    return row


def score_ladder(args, spec=None, scorer=None):
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"error: output exists: {output}")
    spec = spec or AlarmSpec.load(args.spec_dir)
    scorer = scorer or (lambda task, features: spec.score(task, features, args.device))
    aggregation = spec.payload["feature_aggregation"]
    rows, skipped = [], []
    seen = set()
    for run_dir in args.run_dir:
        run_dir = Path(run_dir)
        for parent in _parents(run_dir):
            pid = parent["parent_id"]
            if pid in seen:
                raise ValueError(f"Parent {pid} appears in two run dirs")
            seen.add(pid)
            if parent["task_name"] not in spec.tasks:
                skipped.append({"parent_id": pid, "reason": "task_not_frozen"})
                continue
            if not parent.get("features_path"):
                skipped.append({"parent_id": pid, "reason": "no_features"})
                continue
            with np.load(run_dir / parent["features_path"]) as data:
                raw = np.asarray(data["features"])
                feature_steps = np.asarray(data["feature_steps"], dtype=np.int64)
            if raw.ndim != 4 or len(raw) != len(feature_steps):
                raise ValueError(f"{pid}: features {raw.shape} vs {len(feature_steps)} steps")
            features = reduce_features(raw, aggregation)
            combined = scorer(parent["task_name"], features)
            row = parent_alarm(parent, combined, feature_steps, spec, str(raw.dtype))
            row["run_dir"] = str(run_dir.resolve())
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=_json_default) + "\n")
    summary = summarize_alarms(rows)
    summary.update(
        spec=str(Path(args.spec_dir).resolve()) if getattr(args, "spec_dir", None) else None,
        spec_sha256=sha256(Path(args.spec_dir) / SPEC_NAME) if getattr(args, "spec_dir", None) else None,
        run_dirs=[str(Path(d).resolve()) for d in args.run_dir],
        skipped=skipped,
        code_commit=code_commit(),
    )
    write_json(output.with_name(output.stem + "_summary.json"), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped"}, indent=1, default=_json_default))
    return summary


def summarize_alarms(rows):
    by_task = defaultdict(lambda: Counter())
    fractions = defaultdict(list)
    relative = defaultdict(list)
    for r in rows:
        counts = by_task[r["task_name"]]
        outcome = "success" if r["success"] else "failure"
        counts["successes" if r["success"] else "failures"] += 1
        if r["alarm_fired"]:
            counts[f"{outcome}_alarms"] += 1
            counts[f"{outcome}_alarm_{r['component']}"] += 1
            if not r["success"]:
                fractions[r["task_name"]].append(r["alarm_fraction"])
                if r.get("alarm_minus_ladder_detection") is not None:
                    relative[r["task_name"]].append(r["alarm_minus_ladder_detection"])
    out = {"parents": len(rows), "by_task": {}, "checks": {
        "length_mismatches": sum(not r["length_matches_horizon"] for r in rows),
        "feature_snapshot_mismatches": sum(not r["feature_steps_match_snapshots"] for r in rows),
        "feature_dtypes": dict(Counter(r["feature_dtype"] for r in rows)),
    }}
    for task, counts in sorted(by_task.items()):
        entry = dict(counts)
        if fractions[task]:
            entry["failure_alarm_fraction_median"] = float(np.median(fractions[task]))
        if relative[task]:
            rel = np.asarray(relative[task])
            entry["alarm_vs_ladder_detection_inferences_median"] = float(np.median(rel))
            entry["alarm_before_ladder_detection"] = int(np.sum(rel < 0))
            entry["alarm_after_ladder_detection"] = int(np.sum(rel > 0))
        out["by_task"][task] = entry
    return out


# ---- CLI --------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="retrain, verify and save the frozen alarm")
    e.add_argument("--amendment-scores-dir", required=True,
                   help="Amendment 2 score_cohorts output (scoring.json, normalizations.json, *_predictions.jsonl)")
    e.add_argument("--reference-cohort", default="confirmatory_test",
                   help="cohort whose <name>_predictions.jsonl the reloaded detectors must reproduce")
    e.add_argument("--e1-calibration", required=True,
                   help="E1 lead-calibration calibration_predictions.jsonl (SAFE rows, as confirm_hide_and_seek run)")
    e.add_argument("--hs-calibration-scores", required=True,
                   help="Amendment 2 lead_calibration_predictions.jsonl")
    e.add_argument("--hs-result", required=True, help="Amendment 2 confirm_hide_and_seek.json")
    e.add_argument("--train-split", default="train")
    e.add_argument("--reuse-detectors",
                   help="detectors/ dir of an earlier export: copy its seed*.pt instead of retraining")
    e.add_argument("--output-dir", required=True)
    s = sub.add_parser("score-ladder", help="alarm rows for recoverability-ladder parents")
    s.add_argument("--spec-dir", required=True)
    s.add_argument("--run-dir", nargs="+", required=True, help="ladder shard dirs (parents.jsonl, features/)")
    s.add_argument("--output", required=True, help="alarms.jsonl (must not exist)")
    s.add_argument("--device", default="cpu")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "export":
        return export(args)
    score_ladder(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
