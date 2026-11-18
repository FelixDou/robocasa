"""Score one SAFE export with several frozen checkpoints, loading the export once.

`score_seen_checkpoint_external` loads the whole export for every checkpoint. Loading reads every
policy record from disk and dominates the run time (about 3 s per rollout on the cluster), so 60
scorings of a 1,643-rollout export take about two days. SAFE's loader
(`failure_prob.data.pizero.load_rollouts_from_root`) reads only `cfg.dataset.feat_name`,
`horizon_idx_rel` and `diff_idx_rel` (and the hand-crafted-metric flags, which scoring turns off),
and it never changes a rollout's features afterwards. This command therefore loads the export once
per feature selection and scores each checkpoint in turn, with the steps and outputs of the
single-checkpoint command: the same cutoffs, records and provenance.

With `--verify-existing`, a job whose output directory already holds a completed scoring is scored
again and compared with it instead of written. The first such job runs before any output is
written and the others run last, so the comparison covers both the loading and the reuse of the
loaded export.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .score_seen_checkpoint_external import _write_json, assign_cutoffs, sha256_file

LOADER_KEYS = ("feat_name", "horizon_idx_rel", "diff_idx_rel")
LOADER_SETS = ("dim_features", "dim_action", "pred_horizon", "exec_horizon")
ARTIFACTS = ("model_final.ckpt", "config.yaml", "metrics.json", "split_manifest.json")


def loader_key(dataset_cfg):
    """The configuration values SAFE's loader reads; checkpoints sharing them share one load."""
    return json.dumps([dataset_cfg[k] for k in LOADER_KEYS], default=str)


def job_state(output):
    output = Path(output)
    if not output.exists() or not any(output.iterdir()):
        return "write"
    if (output / "provenance.json").is_file() and (output / "scores.jsonl").is_file():
        return "verify"
    return "partial"


def plan_jobs(jobs, verify_existing):
    """Each job's action, and the order: one verification first, then the writes, then the rest."""
    outputs = [str(Path(o).resolve()) for _, o in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Two jobs share an output directory")
    states = [job_state(o) for _, o in jobs]
    for (_, out), state in zip(jobs, states):
        if state == "partial":
            raise ValueError(f"Output directory holds an incomplete scoring: {out}")
        if state == "verify" and not verify_existing:
            raise FileExistsError(f"Output directory is not empty: {out}")
    verify = [i for i, s in enumerate(states) if s == "verify"]
    write = [i for i, s in enumerate(states) if s == "write"]
    return verify[:1] + write + verify[1:], states


def build_records(rollouts, trajectories, identity, catalog, *, group, model, seed, outside, nondefault):
    """The single-checkpoint command's score records."""
    records = []
    for index, (rollout, score) in enumerate(zip(rollouts, trajectories)):
        env_path, env = identity[id(rollout)]
        manifest = env.get("robocasa_manifest_record", {})
        task_name = catalog["task_names"][int(rollout.task_id)]
        records.append(
            {
                "rollout_id": str(env["rollout_id"]),
                "group": str(group),
                "task_id": int(rollout.task_id),
                "task_name": task_name,
                "task_type": catalog["task_types"][task_name],
                "failed": not bool(int(rollout.episode_success)),
                "model": str(model),
                "seed": seed,
                "scores": np.asarray(score, dtype=np.float64).tolist(),
                "num_inferences": len(score),
                "task_min_step": int(rollout.task_min_step),
                "environment_seed": env.get("environment_seed"),
                "environment_reset_index": env.get("environment_reset_index"),
                "seed_protocol": env.get("seed_protocol"),
                "video_path": str(env_path.with_suffix(".mp4")),
                "video_frame_stride": int(env.get("video_frame_stride", 1)),
                "inference_environment_steps": manifest.get(
                    "inference_environment_steps",
                    list(range(0, len(score) * int(env["replan_steps"]), int(env["replan_steps"]))),
                ),
            }
        )
        if nondefault:
            records[-1]["training_task"] = not outside[index]
    ids = [row["rollout_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("External scoring produced duplicate rollout IDs")
    return records


def build_provenance(*, group, export_dir, training_run, checkpoint, config_path, safe_repo, commit, model, seed,
                     actual_tasks, expected_tasks, allow_task_subset, records, unseen_tasks, cutoff_rule,
                     rollout_tasks, outside):
    """The single-checkpoint command's provenance."""
    provenance = {
        "schema_version": 1,
        "protocol": "frozen_seen_safe_external_scoring",
        "group": str(group),
        "export_dir": str(export_dir),
        "training_run_dir": str(training_run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "safe_repository": str(Path(safe_repo).resolve()),
        "safe_repository_commit": commit,
        "model": str(model),
        "seed": seed,
        "task_names": sorted(actual_tasks),
        "training_task_names": sorted(expected_tasks),
        "task_subset_allowed": bool(allow_task_subset),
        "rollouts": len(records),
        "successes": sum(not row["failed"] for row in records),
        "failures": sum(row["failed"] for row in records),
        "checkpoint_updated": False,
        "thresholds_fitted": False,
    }
    if unseen_tasks != "refuse" or cutoff_rule != "training":
        provenance.update(
            unseen_tasks=str(unseen_tasks),
            cutoff_rule=str(cutoff_rule),
            unseen_task_names=sorted({t for t, o in zip(rollout_tasks, outside) if o}),
        )
    return provenance


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def compare_existing(output, records, provenance, tolerance=0.0):
    """Compare a fresh scoring with the completed one in `output`: every field equal, scores within tolerance."""
    output = Path(output)
    old = [json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines() if line.strip()]
    old_prov = json.loads((output / "provenance.json").read_text())
    new, new_prov = [_plain(r) for r in records], _plain(provenance)
    problems = [f"provenance {k}" for k in sorted(set(old_prov) | set(new_prov)) if old_prov.get(k) != new_prov.get(k)]
    if [r["rollout_id"] for r in old] != [r["rollout_id"] for r in new]:
        problems.append("rollout order")
    worst = 0.0
    for a, b in zip(old, new):
        fields = sorted((set(a) | set(b)) - {"scores"})
        bad = [k for k in fields if a.get(k) != b.get(k)]
        if bad:
            problems.append(f"{a.get('rollout_id')}: {', '.join(bad)}")
        if len(a["scores"]) != len(b["scores"]):
            problems.append(f"{a.get('rollout_id')}: score length")
        elif a["scores"]:
            worst = max(worst, float(np.max(np.abs(np.subtract(a["scores"], b["scores"])))))
    if worst > tolerance:
        problems.append(f"max |score difference| {worst:.3g} > {tolerance:.3g}")
    return dict(ok=not problems, max_abs_diff=worst, problems=problems[:8])


class SafeBackend:
    """The official SAFE code and robocasa's export helpers, imported on first use."""

    def __init__(self, safe_repo):
        from .train_seen_tasks import (
            OFFICIAL_SAFE_COMMIT,
            load_env_records,
            task_catalog,
            validate_alignment,
            verify_safe_repo,
        )

        verify_safe_repo(safe_repo)
        if str(Path(safe_repo).resolve()) not in sys.path:
            sys.path.insert(0, str(Path(safe_repo).resolve()))
        import torch
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader
        from failure_prob.data.pizero import load_rollouts_from_root
        from failure_prob.data.utils import RolloutDataset
        from failure_prob.model import get_model
        from failure_prob.utils.random import seed_everything
        from failure_prob.utils.routines import model_forward_dataloader

        self.commit = OFFICIAL_SAFE_COMMIT
        self.env_records, self.catalog, self.align = load_env_records, task_catalog, validate_alignment
        self.torch, self.OmegaConf, self.DataLoader = torch, OmegaConf, DataLoader
        self.load_rollouts, self.RolloutDataset, self.get_model = load_rollouts_from_root, RolloutDataset, get_model
        self.seed, self.forward = seed_everything, model_forward_dataloader

    def config(self, path, export_dir, device):
        if str(device).startswith("cuda") and not self.torch.cuda.is_available():
            raise ValueError(f"Requested {device}, but CUDA is unavailable")
        cfg = self.OmegaConf.load(path)
        cfg.dataset.data_path_prefix = ""
        cfg.dataset.data_path = str(export_dir)
        cfg.dataset.data_path_unseen = None
        cfg.dataset.load_to_cuda = str(device).startswith("cuda")
        cfg.train.log_precomputed = False
        cfg.train.log_precomputed_only = False
        return cfg

    def load(self, export_dir, cfg):
        self.seed(0)
        return self.load_rollouts(export_dir, cfg)

    def to_device(self, rollouts, cfg, device):
        return [r.to(device) for r in rollouts] if cfg.dataset.load_to_cuda else rollouts

    def score(self, cfg, rollouts, checkpoint, device):
        torch = self.torch
        self.seed(0)
        dataset = self.RolloutDataset(cfg, rollouts)
        loader = self.DataLoader(dataset, batch_size=dataset.cfg.model.batch_size, shuffle=False, num_workers=0)
        model = self.get_model(cfg, int(rollouts[0].hidden_states.shape[-1]))
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        with torch.no_grad():
            scores, masks, _ = self.forward(model, loader)
        scores = scores.detach().cpu().numpy()
        lengths = masks.sum(dim=-1).detach().cpu().numpy().astype(int)
        out = [scores[i, :n] for i, n in enumerate(lengths)]
        del dataset, loader, model, state, masks
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return out


def score_export_batch(*, export_dir, safe_repo, jobs, group, device="cuda", allow_task_subset=False,
                       unseen_tasks="refuse", cutoff_rule="training", verify_existing=False, tolerance=0.0,
                       backend=None, log=None):
    log = log or (lambda m: print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {m}", flush=True))
    if not jobs:
        raise ValueError("No jobs")
    export_dir = Path(export_dir).resolve()
    order, states = plan_jobs(jobs, verify_existing)
    runs = []
    for run_dir, out in jobs:
        training_run = Path(run_dir).resolve()
        for name in ARTIFACTS:
            if not (training_run / name).is_file():
                raise ValueError(f"Frozen training artifact is missing: {training_run / name}")
        runs.append((training_run, Path(out).resolve()))
    backend = backend or SafeBackend(safe_repo)
    cfgs = [backend.config(run / "config.yaml", export_dir, device) for run, _ in runs]
    for i in order:
        if states[i] == "write":
            runs[i][1].mkdir(parents=True, exist_ok=True)
    env_records = backend.env_records(export_dir)
    catalog = backend.catalog(export_dir)
    nondefault = unseen_tasks != "refuse" or cutoff_rule != "training"
    groups = {}
    for i in order:
        groups.setdefault(loader_key(cfgs[i].dataset), []).append(i)
    results = [None] * len(jobs)
    done = 0
    for key, members in groups.items():
        first = cfgs[members[0]]
        start = time.time()
        rollouts = backend.load(export_dir, first)
        log(f"loaded {len(rollouts)} rollouts ({key}) in {time.time() - start:.0f} s for {len(members)} checkpoints")
        identity = backend.align(rollouts, env_records)
        shapes = {k: first.dataset[k] for k in LOADER_SETS}
        rollout_tasks = [catalog["task_names"][int(r.task_id)] for r in rollouts]
        lengths = [len(r.hidden_states) for r in rollouts]
        actual_tasks = set(rollout_tasks)
        if not actual_tasks:
            raise ValueError("External export contains no scored tasks")
        meta = {}
        for i in members:
            run = runs[i][0]
            metrics = json.loads((run / "metrics.json").read_text())
            expected = set(json.loads((run / "split_manifest.json").read_text())["task_names"])
            unexpected, missing = sorted(actual_tasks - expected), sorted(expected - actual_tasks)
            if (unexpected and unseen_tasks != "score") or (missing and not allow_task_subset):
                raise ValueError(f"{run}: external export task coverage differs from training: "
                                 f"missing={missing}, unexpected={unexpected}")
            cutoffs = metrics.get("task_min_steps", {})
            if set(cutoffs) != expected:
                raise ValueError(f"{run}: frozen training metrics lack exact task cutoff coverage")
            meta[i] = (expected, *assign_cutoffs(rollout_tasks, lengths, cutoffs, expected,
                                                 unseen_tasks=unseen_tasks, cutoff_rule=cutoff_rule))
        rollouts = backend.to_device(rollouts, first, device)
        for i in members:
            (run, out), cfg = runs[i], cfgs[i]
            for k, v in shapes.items():
                cfg.dataset[k] = v
            expected, task_cutoffs, outside = meta[i]
            for rollout, cutoff in zip(rollouts, task_cutoffs):
                rollout.task_min_step = cutoff
            trajectories = backend.score(cfg, rollouts, run / "model_final.ckpt", device)
            if any(not len(s) or not np.all(np.isfinite(s)) for s in trajectories):
                raise RuntimeError(f"{run}: frozen checkpoint produced an invalid score trajectory")
            seed = int(cfg.train.seed)
            records = build_records(rollouts, trajectories, identity, catalog, group=group, model=cfg.model.name,
                                    seed=seed, outside=outside, nondefault=nondefault)
            provenance = build_provenance(
                group=group, export_dir=export_dir, training_run=run, checkpoint=run / "model_final.ckpt",
                config_path=run / "config.yaml", safe_repo=safe_repo, commit=backend.commit, model=cfg.model.name,
                seed=seed, actual_tasks=actual_tasks, expected_tasks=expected, allow_task_subset=allow_task_subset,
                records=records, unseen_tasks=unseen_tasks, cutoff_rule=cutoff_rule, rollout_tasks=rollout_tasks,
                outside=outside)
            done += 1
            if states[i] == "verify":
                check = compare_existing(out, records, provenance, tolerance)
                results[i] = dict(action="verify", output=str(out), **check)
                log(f"[{done}/{len(jobs)}] verified {out}: max |diff| {check['max_abs_diff']:.3g} -> "
                    + ("OK" if check["ok"] else "MISMATCH " + "; ".join(check["problems"])))
                if not check["ok"]:
                    raise RuntimeError(f"VERIFY MISMATCH at {out}")
            else:
                with (out / "scores.jsonl").open("w") as stream:
                    for record in records:
                        stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
                _write_json(out / "provenance.json", provenance)
                results[i] = dict(action="write", output=str(out), rollouts=len(records))
                log(f"[{done}/{len(jobs)}] wrote {out} ({len(records)} rollouts)")
        del rollouts, identity
    return results


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--safe-repo", required=True)
    parser.add_argument("--group", choices=("calibration", "prospective_test", "confirmatory_test", "lead_calibration", "all"),
                        required=True)
    parser.add_argument("--job", nargs=2, action="append", required=True, metavar=("TRAINING_RUN_DIR", "OUTPUT_DIR"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-task-subset", action="store_true")
    parser.add_argument("--unseen-tasks", choices=("refuse", "score"), default="refuse")
    parser.add_argument("--cutoff-rule", choices=("training", "rollout_length"), default="training")
    parser.add_argument("--verify-existing", action="store_true",
                        help="rescore jobs whose output holds a completed scoring and compare instead of writing")
    parser.add_argument("--tolerance", type=float, default=0.0, help="largest score difference a verification accepts")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    kwargs = vars(args)
    kwargs["jobs"] = [tuple(j) for j in kwargs.pop("job")]
    try:
        results = score_export_batch(**kwargs)
    except (FileExistsError, ValueError, RuntimeError) as error:
        print(f"BATCH FAILED: {error}", flush=True)
        raise SystemExit(1) from error
    written = sum(r["action"] == "write" for r in results)
    verified = [r for r in results if r["action"] == "verify"]
    worst = max((r["max_abs_diff"] for r in verified), default=0.0)
    print(f"BATCH OK: {written} written, {len(verified)} verified (max |diff| {worst:.3g})", flush=True)


if __name__ == "__main__":
    main()
