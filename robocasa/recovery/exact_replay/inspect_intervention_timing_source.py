"""Inspect saved source images and replay histories without simulation or inference."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from robocasa.recovery.exact_replay.diagnose_intervention_timing_observations import (
    camera_statistics, load_state, observation_difference,
)
from robocasa.recovery.exact_replay.diagnose_intervention_timing_replay import PROTOCOL as REPLAY_PROTOCOL, first_divergence
from robocasa.recovery.full_snapshot import stable_digest
from robocasa.recovery.exact_replay.intervention_timing import validate_registration
from robocasa.recovery.exact_replay.phase3b_training_horizon import (
    atomic_write_json, load_registered_phase2_snapshot, read_jsonl, sha256_file,
)
from robocasa.recovery.exact_replay.run_intervention_timing import reference_audit


PROTOCOL = "xr1_offline_historical_source_inspection_v1"


def validated_diagnostic(root):
    root = root.resolve()
    report = json.loads((root/"report.json").read_text())
    if report.get("protocol") != REPLAY_PROTOCOL or report.get("status") != "diagnostic_complete":
        raise ValueError("Require a completed nominal replay diagnostic")
    required = {"registration.json"}
    for i in range(2):
        required.update(f"attempt_{i}/{name}" for name in (
            "initial_diagnostic_state.pkl.gz", "endpoint_diagnostic_state.pkl.gz",
            "nominal_progress.jsonl", "source_endpoint_audit.json", "requests.jsonl",
            "observation_window/trace.jsonl", "observation_window/all_observations.pkl.gz"))
    if not required.issubset(report["artifact_hashes"]):
        raise ValueError("Require registered initial/endpoint states, traces and all raw observations")
    # Validate all files before loading any diagnostic pickle.
    for name, expected in report["artifact_hashes"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != expected:
            raise ValueError(f"Diagnostic artifact changed or escaped its root: {name}")
    registration = json.loads((root/"registration.json").read_text())
    if registration.get("protocol") != REPLAY_PROTOCOL or registration["parent"]["parent_id"] != report["parent_id"]:
        raise ValueError("Diagnostic report and registration identities differ")
    return {"root": root, "registration": registration, "report": report,
            "initial": [load_state(root/f"attempt_{i}/initial_diagnostic_state.pkl.gz") for i in range(2)],
            "endpoint": [load_state(root/f"attempt_{i}/endpoint_diagnostic_state.pkl.gz") for i in range(2)]}


def observation_trace_comparison(left, right):
    a, b = ([r["environment_step"] for r in rows] for rows in (left, right))
    first, keys = None, {}
    if a != b:
        return {"step_support_exact": False, "first_observation_difference_step": None,
                "first_difference_by_observation_key": {}, "rows": [len(left), len(right)]}
    for x, y in zip(left, right):
        if first is None and x["observation_sha256"] != y["observation_sha256"]:
            first = x["environment_step"]
        for key in set(x["observation_key_sha256"]) | set(y["observation_key_sha256"]):
            if x["observation_key_sha256"].get(key) != y["observation_key_sha256"].get(key):
                keys.setdefault(key, x["environment_step"])
    return {"step_support_exact": True, "first_observation_difference_step": first,
            "first_difference_by_observation_key": keys, "rows": [len(left), len(right)]}


def image_statistics(observation):
    stats = camera_statistics(observation)
    for key, row in stats.items():
        rgb = np.asarray(observation[key], dtype=np.float64)
        row["mean_absolute_red_green_gap"] = round(float(np.abs(rgb[..., 0]-rgb[..., 1]).mean()), 2)
        row["equal_rgb_pixel_percent"] = round(100*float(
            ((rgb[..., 0] == rgb[..., 1]) & (rgb[..., 1] == rgb[..., 2])).mean()), 2)
    return stats


def request_trace_comparison(left, right):
    fields = ("request_index", "sampling_seed", "request_sha256", "response_sha256")
    first = next(({"row": i, "fields": [k for k in fields if a.get(k) != b.get(k)]}
                  for i, (a, b) in enumerate(zip(left, right))
                  if any(a.get(k) != b.get(k) for k in fields)), None)
    if first is None and len(left) != len(right):
        first = {"row": min(len(left), len(right)), "fields": ["trace_length"]}
    return {"rows": [len(left), len(right)], "first_difference": first,
            "same_request_different_response_rows": [i for i, (a, b) in enumerate(zip(left, right))
                if a["request_sha256"] == b["request_sha256"] and a["response_sha256"] != b["response_sha256"]]}


def latest_policy_records(state):
    inference = state.get("latest_inference_record") or {}
    return {"policy_environment_step": state.get("environment_step"),
            "ordinary_inference_index": state.get("ordinary_inference_index"),
            "sampling_config": state.get("sampling_config"),
            "latest_requests": state.get("latest_request_records", []),
            "latest_inference": {key: inference.get(key) for key in (
                "environment_step", "inference_index", "sampling_seed", "request_sha256")},
            "latest_actions_sha256": stable_digest(inference.get("actions")),
            "latest_features_sha256": stable_digest(inference.get("features"))}


def history_statistics(state):
    # Queue positions are saved facts. Do not invent historical frame timestamps
    # or reconstruct a past encoded request from an endpoint's later queues.
    return {key: [{"queue_index": i, "frame_sha256": stable_digest(frame),
                   "shape": list(np.asarray(frame).shape), "dtype": str(np.asarray(frame).dtype),
                   "statistics": image_statistics({key: frame}).get(key)}
                  for i, frame in enumerate(frames)]
            for key, frames in state.get("image_queues", {}).items()}


def write_image_panel(path, rows):
    cells = [[(label, np.asarray(frame)) for label, frame in row
              if np.asarray(frame).ndim == 3 and np.asarray(frame).shape[-1] == 3
              and np.asarray(frame).dtype == np.uint8] for row in rows]
    if not any(cells):
        return None
    width = max(280, max(a.shape[1]+8 for row in cells for _, a in row))
    height = max(a.shape[0]+42 for row in cells for _, a in row)
    panel = Image.new("RGB", (width*max(map(len, cells)), height*len(cells)), "white")
    draw = ImageDraw.Draw(panel)
    for y, row in enumerate(cells):
        for x, (label, frame) in enumerate(row):
            draw.text((x*width+4, y*height+4), label, fill="black")
            panel.paste(Image.fromarray(frame), (x*width+4, y*height+32))
    panel.save(path)
    return path.name


def validate_raw_observations(diagnostic):
    result = []
    for i, endpoint in enumerate(diagnostic["endpoint"]):
        folder = diagnostic["root"]/f"attempt_{i}/observation_window"
        trace = read_jsonl(folder/"trace.jsonl")
        raw = load_state(folder/"all_observations.pkl.gz")
        expected = list(range(endpoint["environment_step"]+1))
        if [r["environment_step"] for r in trace] != expected or set(raw) != set(expected):
            raise ValueError("Raw observations must cover every replay step exactly once")
        if any(stable_digest(raw[r["environment_step"]]) != r["observation_sha256"] for r in trace):
            raise ValueError("Raw observation archive differs from its trace")
        if any({k: stable_digest(v) for k, v in raw[r["environment_step"]].items()}
               != r["observation_key_sha256"] for r in trace):
            raise ValueError("Raw observation keys differ from their trace")
        if stable_digest(raw[0]) != stable_digest(diagnostic["initial"][i]["observation"]):
            raise ValueError("Raw initial observation differs from saved initial state")
        if stable_digest(raw[expected[-1]]) != stable_digest(endpoint["observation"]):
            raise ValueError("Raw endpoint observation differs from saved endpoint")
        result.append(len(raw))
    return result


def run(args):
    primary = validated_diagnostic(args.diagnostic_dir)
    comparison = validated_diagnostic(args.comparison_diagnostic_dir) if args.comparison_diagnostic_dir else None
    registration = primary["registration"]
    timing_root = Path(registration["timing_run_dir"]).resolve()
    timing = json.loads((timing_root/"registration.json").read_text())
    validate_registration(timing)
    parent = registration["parent"]
    if registration["timing_registration_sha256"] != timing["registration_sha256"] or parent not in timing["parents"]:
        raise ValueError("Diagnostic is not bound to this frozen timing parent")
    if comparison and (comparison["registration"]["parent"] != parent or
                       comparison["registration"]["timing_registration_sha256"] != timing["registration_sha256"] or
                       Path(comparison["registration"]["timing_run_dir"]).resolve() != timing_root):
        raise ValueError("Comparison diagnostic must use the same frozen parent and timing registration")
    sid = parent["snapshot_id"]
    source, integrity = load_registered_phase2_snapshot(
        Path(timing["phase2_run_dir"])/"snapshots"/f"{sid}.pkl.gz",
        expected_file_sha256=timing["phase2_registration"]["phase2_snapshot_file_sha256"][sid],
        expected_snapshot_id=sid, expected_parent_id=parent["parent_id"], expected_task_name=parent["task_name"])
    raw_counts = validate_raw_observations(primary)
    if comparison:
        validate_raw_observations(comparison)
    root = args.output_dir.resolve()
    protected = [primary["root"], timing_root, Path(timing["phase2_run_dir"]).resolve(),
                 Path(timing["operator_run_dir"]).resolve()]
    if comparison:
        protected.append(comparison["root"])
    if root.is_relative_to(Path("/gs/fs")) or any(
            root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError("Inspection output must be separate from all frozen inputs and outside /gs/fs")
    observations = [source.observation, *[e["observation"] for e in primary["endpoint"]]]
    states = [source.policy_state, *[e["policy_state"] for e in primary["endpoint"]]]
    labels = ["historical source", "replay attempt 0", "replay attempt 1"]
    cameras = sorted(k for k in source.observation if k.startswith("video."))
    report = {"protocol": PROTOCOL, "parent": parent, "environment_step": source.environment_step,
              "source_integrity": integrity, "raw_observation_counts": raw_counts,
              "diagnostic_dir": str(primary["root"]), "timing_run_dir": str(timing_root),
              "diagnostic_report_sha256": sha256_file(primary["root"]/"report.json"),
              "timing_registration_file_sha256": sha256_file(timing_root/"registration.json"),
              "code_sha256": sha256_file(Path(__file__)),
              "device_assignment": primary["report"].get("device_assignment"),
              "source_endpoint_audits": [reference_audit(source, SimpleNamespace(**e)) for e in primary["endpoint"]],
              "endpoint_camera_statistics": {label: image_statistics(obs) for label, obs in zip(labels, observations)},
              "endpoint_observation_differences": [observation_difference(source.observation, e["observation"])
                                                    for e in primary["endpoint"]],
              "latest_policy_records": {label: latest_policy_records(s) for label, s in zip(labels, states)},
              "saved_history_statistics": {label: history_statistics(s) for label, s in zip(labels, states)},
              "comparison_with_cohort": None, "scientific_trajectories_collected": 0,
              "simulation_steps": 0, "policy_requests": 0, "full_timing_study_complete": False,
              "frozen_eligibility_changed": False,
              "scope": "saved endpoint and retained image queues only; no historical first-divergence localization, request reconstruction, or camera-health certification"}
    if comparison:
        report["comparison_with_cohort"] = {"diagnostic_dir": str(comparison["root"]),
            "report_sha256": sha256_file(comparison["root"]/"report.json"),
            "device_assignment": comparison["report"].get("device_assignment"),
            "attempts": [{"initial_audit": reference_audit(SimpleNamespace(**a), SimpleNamespace(**b)),
                "endpoint_audit": reference_audit(SimpleNamespace(**x), SimpleNamespace(**y)),
                "first_progress_divergence": first_divergence(
                    read_jsonl(comparison["root"]/f"attempt_{i}/nominal_progress.jsonl"),
                    read_jsonl(primary["root"]/f"attempt_{i}/nominal_progress.jsonl")),
                "request_comparison": request_trace_comparison(
                    read_jsonl(comparison["root"]/f"attempt_{i}/requests.jsonl"),
                    read_jsonl(primary["root"]/f"attempt_{i}/requests.jsonl")),
                "observation_comparison": observation_trace_comparison(
                    read_jsonl(comparison["root"]/f"attempt_{i}/observation_window/trace.jsonl"),
                    read_jsonl(primary["root"]/f"attempt_{i}/observation_window/trace.jsonl"))}
                for i, (a, b, x, y) in enumerate(zip(comparison["initial"], primary["initial"],
                                                     comparison["endpoint"], primary["endpoint"]))]}
    root.mkdir(parents=True, exist_ok=False)
    images = []
    name = write_image_panel(root/"endpoint_cameras.png", [
        [(f"{label}\n{key}", obs[key]) for key in cameras if key in obs]
        for label, obs in zip(labels, observations)])
    if name:
        images.append(name)
    for camera_index, key in enumerate(cameras):
        rows = []
        for label, state in zip(labels, states):
            frames = state.get("image_queues", {}).get(key, [])
            indices = sorted(set(np.linspace(0, len(frames)-1, min(8, len(frames)), dtype=int))) if frames else []
            rows.append([(f"{label}\n{key} queue index {i}", frames[i]) for i in indices])
        name = write_image_panel(root/f"history_camera_{camera_index}.png", rows)
        if name:
            images.append(name)
    report["image_artifact_hashes"] = {name: sha256_file(root/name) for name in images}
    atomic_write_json(root/"report.json", report)
    summary = {"parent_id": parent["parent_id"], "environment_step": source.environment_step,
               "source_integrity_accepted": integrity["accepted"], "raw_observation_counts": raw_counts,
               "endpoint_camera_statistics": {k: report["endpoint_camera_statistics"][k] for k in labels[:2]},
               "latest_policy_records": {k: report["latest_policy_records"][k] for k in labels[:2]},
               "cohort_comparison": [{"initial_exact": r["initial_audit"]["all_exact"],
                    "endpoint_exact": r["endpoint_audit"]["all_exact"],
                    "first_progress_divergence": r["first_progress_divergence"],
                    "requests": r["request_comparison"],
                    "observations": r["observation_comparison"]}
                   for r in report["comparison_with_cohort"]["attempts"]] if comparison else None,
               "images": [str(root/name) for name in images], "report": str(root/"report.json"),
               "scope": report["scope"]}
    print(json.dumps(summary, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--comparison-diagnostic-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
