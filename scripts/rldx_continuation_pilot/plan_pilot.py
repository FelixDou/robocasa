"""Plan the RLDX-1 continuation and same-scene retry pilot (plan: docs/rldx_e3_pilot_plan.md).

Reads the raw manifest of the RLDX-1 SAFE20 natural-rate collection (the source dataset of
`official_safe_supported20_natural50`) and the summary configs of the collection it came from.
Checks that every task has one unbroken reset sequence, that the settings agree across rollouts
and summaries, and that the current registry horizons equal the original ones (the pilot runs
to ceil(1.25 x registry horizon)). Writes:
  - selection.json: per-task horizons, success rates and the original outcome, length and video
    of every scene (task, reset index); the collector settings; the expected cost;
  - collector_flags.txt: the common collector arguments, one per line (read with mapfile);
  - tasks_shard{0,1}.txt and attempt_limits_shard{0,1}.json (limits only where a task's
    sequence is shorter than --num-rollouts), shards balanced by expected env steps.
Reads outcomes only to describe the cohort; the pilot's endpoints are not computed here.

Usage (from outside the data directory):
  python -I plan_pilot.py <raw manifest dir> --collection-root <SAFE35 root> --output <pilot root>
"""
import argparse, importlib.util, json, math, sys
from collections import defaultdict
from pathlib import Path

HORIZON_SCALE = 1.25
DEFAULT_POLICY_MODULE = "robocasa.recovery.rldx_zmq_policy:make_policy"
REPO = Path(__file__).resolve().parents[2]


def atomic_tasks_module():
    spec = importlib.util.spec_from_file_location(
        "_atomic_tasks", REPO / "robocasa" / "recovery" / "safe" / "atomic_tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def scaled(h, scale=HORIZON_SCALE):
    return int(math.ceil(round(h * scale, 6)))


def one(values, name):
    values = {json.dumps(v, sort_keys=True) for v in values}
    if len(values) != 1:
        raise SystemExit(f"STOP: {name} is not unique across rollouts: {sorted(values)[:4]}")
    return json.loads(values.pop())


def summaries(root, rollout_ids):
    """Summary configs of the collector shards that hold any of the rollouts.

    Merged or filtered datasets also write a summary.json, without a base seed; they are
    skipped. A shard is used only if its own manifest contains some of the rollouts, and the
    shards used must hold all of them."""
    found, covered = [], set()
    for path in sorted(Path(root).rglob("summary.json")):
        try:
            config = json.loads(path.read_text()).get("config") or {}
        except (OSError, ValueError):
            continue
        if config.get("model_family") != "rldx1" or config.get("base_environment_seed") is None:
            continue
        manifest = path.parent / "manifest.jsonl"
        ids = {r["rollout_id"] for r in read_jsonl(manifest)} if manifest.is_file() else set()
        if ids & rollout_ids:
            found.append((path, config)); covered |= ids & rollout_ids
    missing = rollout_ids - covered
    if found and missing:
        raise SystemExit(f"STOP: {len(missing)} rollouts are in no collector shard manifest under the root")
    return found


def first_frame_shape(path):
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(path))
    try:
        return list(reader.get_data(0).shape)
    finally:
        reader.close()


def build(rows, source, configs, registry, atomic, frame_shape_fn=None):
    """registry: task -> current registry horizon; atomic: the registered atomic tasks."""
    frame_shape_fn = frame_shape_fn or first_frame_shape
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_name"]].append(r)
    tasks = sorted(by_task)
    settings = dict(
        seed=one([r["environment_seed"] for r in rows], "environment_seed"),
        seed_protocol=one([r["seed_protocol"] for r in rows], "seed_protocol"),
        split=one([r["environment_split"] for r in rows], "environment_split"),
        replan_steps=one([r["replan_steps"] for r in rows], "replan_steps"),
        policy_name=one([r["policy_id"] for r in rows], "policy_id"),
        checkpoint=one([r["checkpoint"] for r in rows], "checkpoint"),
        policy_config=one([r.get("policy_config") or {} for r in rows], "policy_config"),
        model_family=one([r.get("model_family") for r in rows], "model_family"),
        video_frame_stride=one([r.get("video_frame_stride") for r in rows], "video_frame_stride"),
        horizon_scale=HORIZON_SCALE)
    if not configs:
        raise SystemExit("STOP: no RLDX-1 summary.json found under --collection-root")
    # Older manifest rows may lack these fields; the shard summaries record them.
    for key in ("model_family", "video_frame_stride"):
        if settings[key] is None:
            settings[key] = one([c.get(key) for _, c in configs], f"summary config {key}")
        if settings[key] is None:
            raise SystemExit(f"STOP: {key} is recorded neither in the manifest nor in the summaries")
    if settings["seed_protocol"] != "official_rldx" or settings["model_family"] != "rldx1":
        raise SystemExit("STOP: expected an RLDX-1 collection under the official_rldx seed protocol")
    for key in ("policy_module", "record_actions", "record_safe_features", "record_subtask_trace", "record_videos"):
        settings[key] = one([c.get(key) for _, c in configs], f"summary config {key}")
    for key, value in (("base_environment_seed", settings["seed"]), ("split", settings["split"]),
                       ("replan_steps", settings["replan_steps"]), ("policy_checkpoint", settings["checkpoint"])):
        if any(c.get(key) != value for _, c in configs):
            raise SystemExit(f"STOP: summary config {key} disagrees with the manifest")
    if any(c.get("success_quota") is not None or c.get("failure_quota") is not None or c.get("retain_only_quota")
           for _, c in configs):
        raise SystemExit("STOP: the original collection used quotas; its outcome rates are not natural")
    types = {t: "atomic" if t in atomic else "composite" for t in tasks}
    out_tasks, problems = {}, []
    for _, c in configs:
        for t, recorded in (c.get("task_types") or {}).items():
            if t in types and recorded != types[t]:
                problems.append(f"{t}: summary type {recorded} != registry type {types[t]}")
    for t in tasks:
        rs = sorted(by_task[t], key=lambda r: int(r["environment_reset_index"]))
        idx = [int(r["environment_reset_index"]) for r in rs]
        if len(set(idx)) != len(idx):
            problems.append(f"{t}: duplicate reset indices")
        H = one([r["rollout_horizon"] for r in rs], f"{t} rollout_horizon")
        if registry.get(t) != H:
            problems.append(f"{t}: registry horizon {registry.get(t)} != original {H}")
        scenes = {}
        for r in rs:
            video = r.get("video_path")
            scenes[str(int(r["environment_reset_index"]))] = dict(
                failed=bool(r["failed"]), L=int(r["num_env_steps"]),
                video=str((Path(source) / video).resolve()) if video else None)
        n = len(rs); s = sum(not r["failed"] for r in rs)
        out_tasks[t] = dict(type=types.get(t), H=H, H_scaled=scaled(H), n=n, successes=s, failures=n - s,
                            p=round(s / n, 4), max_reset_index=max(idx), scenes=scenes)
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems))
    videos = [sc["video"] for v in out_tasks.values() for sc in v["scenes"].values()]
    present = [p for p in videos if p and Path(p).is_file()]
    settings["original_videos_present"] = len(present)
    settings["original_videos_total"] = len(videos)
    if present:
        shape = frame_shape_fn(present[0])
        settings["video_height"], settings["video_width"] = int(shape[0]), int(shape[1])
    settings["num_rollouts"] = max(v["max_reset_index"] for v in out_tasks.values()) + 1
    return tasks, out_tasks, settings


def expected_steps(v):
    """Env steps if the pilot reproduced the original outcomes: successes keep their length,
    failures run to the scaled horizon."""
    return sum(sc["L"] if not sc["failed"] else v["H_scaled"] for sc in v["scenes"].values())


def shards(tasks, out_tasks):
    a, b = [], []
    for t in sorted(tasks, key=lambda t: -expected_steps(out_tasks[t])):
        load_a = sum(expected_steps(out_tasks[x]) for x in a)
        load_b = sum(expected_steps(out_tasks[x]) for x in b)
        (a if load_a <= load_b else b).append(t)
    return sorted(a), sorted(b)


def flags(settings):
    f = ["--num-rollouts", str(settings["num_rollouts"]), "--seed", str(settings["seed"]),
         "--seed-protocol", settings["seed_protocol"], "--split", settings["split"],
         "--replan-steps", str(settings["replan_steps"]),
         "--policy-module", settings["policy_module"] or DEFAULT_POLICY_MODULE,
         "--model-family", settings["model_family"], "--policy-name", settings["policy_name"],
         "--checkpoint", settings["checkpoint"],
         "--policy-config", json.dumps(settings["policy_config"], sort_keys=True, separators=(",", ":")),
         "--horizon-scale", str(settings["horizon_scale"]),
         "--record-safe-features",
         "--record-actions" if settings["record_actions"] else "--no-record-actions",
         "--no-record-subtask-trace", "--record-videos",
         "--video-frame-stride", str(settings["video_frame_stride"])]
    if "video_height" in settings:
        f += ["--video-height", str(settings["video_height"]), "--video-width", str(settings["video_width"])]
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="raw manifest dir (source_dataset of the official export)")
    p.add_argument("--collection-root", required=True, help="original collection root (summary.json files)")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if (out / "selection.json").exists():
        raise SystemExit(f"error: {out} already has a selection")
    rows = read_jsonl(Path(a.source) / "manifest.jsonl")
    if not rows:
        raise SystemExit("STOP: empty manifest")
    at = atomic_tasks_module()
    configs = summaries(a.collection_root, {r["rollout_id"] for r in rows})
    tasks, out_tasks, settings = build(rows, a.source, configs, at.registered_safe_task_horizons(),
                                       at.registered_atomic_tasks())
    s0, s1 = shards(tasks, out_tasks)
    out.mkdir(parents=True, exist_ok=True)
    (out / "collector_flags.txt").write_text("\n".join(flags(settings)) + "\n")
    for i, s in enumerate((s0, s1)):
        (out / f"tasks_shard{i}.txt").write_text("\n".join(s) + "\n")
        limits = {t: out_tasks[t]["max_reset_index"] + 1 for t in s
                  if out_tasks[t]["max_reset_index"] + 1 < settings["num_rollouts"]}
        (out / f"attempt_limits_shard{i}.json").write_text(json.dumps(limits, indent=1, sort_keys=True) + "\n")
    orig_steps = sum(sc["L"] for v in out_tasks.values() for sc in v["scenes"].values())
    pilot_steps = sum(expected_steps(v) for v in out_tasks.values())
    cost = dict(rollouts=sum(v["n"] for v in out_tasks.values()), original_env_steps=orig_steps,
                expected_env_steps=pilot_steps, ratio=round(pilot_steps / orig_steps, 3),
                expected_inferences=int(math.ceil(pilot_steps / settings["replan_steps"])))
    sel = dict(plan="docs/rldx_e3_pilot_plan.md", source=str(Path(a.source).resolve()),
               summaries=[str(path) for path, _ in configs], settings=settings, tasks=out_tasks,
               shards=[s0, s1], cost=cost,
               types={k: sum(out_tasks[t]["type"] == k for t in tasks) for k in ("atomic", "composite")})
    (out / "selection.json").write_text(json.dumps(sel, indent=1, sort_keys=True) + "\n")
    for t in tasks:
        v = out_tasks[t]
        print(f"{t:36s} {v['type']:9s} H={v['H']:5d} -> {v['H_scaled']:5d}  n={v['n']:3d}  p={v['p']:.2f}")
    print(f"settings: seed {settings['seed']}, {settings['seed_protocol']}, split {settings['split']}, "
          f"replan {settings['replan_steps']}, num_rollouts {settings['num_rollouts']}, "
          f"videos {settings['original_videos_present']}/{settings['original_videos_total']}, "
          f"frame {settings.get('video_height')}x{settings.get('video_width')}")
    print(f"cost: {cost['rollouts']} rollouts, ~{cost['expected_inferences']} inferences, "
          f"{cost['ratio']}x the original env steps | types {sel['types']}")
    print(f"shard0 {s0}\nshard1 {s1}")
    if settings["original_videos_present"] < 0.95 * settings["original_videos_total"]:
        print("WARNING: fewer than 95% of original videos exist; the scene gate cannot pass, so r will be void "
              "(c is unaffected). See the plan.")
    print("PLAN OK")


if __name__ == "__main__":
    main()
