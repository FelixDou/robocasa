"""Sample and bundle XR-1 failure videos for point-of-no-return labeling.

Plan: docs/no_return_labeling_plan.md. Draws 2 failures per atomic task and 1 per composite
task from the primary confirmatory test cohort (seed 0), copies their videos under anonymous
IDs in random order into <output>/bundle/, and writes the labeler-facing manifest there.
Detector alarm times (P and SAFE + P fitted exactly as in E1) go to <output>/key/key.json,
which must not be opened until labeling is finished.

Usage:
  python3 -I no_return_sample.py --calibration CAL.jsonl --test TEST.jsonl \
      --runtime-bundle runtime_bundle.json --collection-root <confirmatory_collection> --output OUT
"""
import argparse, json, math, random, shutil, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl
import quantile_timer as qt

PER_TYPE = {"atomic": 2, "composite": 1}
SEED = 0
DEFAULT_FPS = 20


def manifest_rows(collection_root):
    rows = {}
    for shard in sorted(Path(collection_root).glob("shard*")):
        path = shard / "manifest.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_shard"] = str(shard)
                rows[r["rollout_id"]] = r
    if not rows:
        raise ValueError(f"No shard manifests under {collection_root}")
    return rows


def alarm_indices(cal, test, bundle):
    rk = qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)
    calC, teC = cl.Cohort(cal), cl.Cohort(test)
    tm, fu = cl.fit_fast(calC, rk)
    rho = cl.rho_rows(teC, rk)
    aP = cl.alarm_vec(teC, rho, tm["lam"], tm["u"], tm["s"])
    aF = cl.alarm_vec(teC, rho, fu["lam"], fu["u"], fu["s"])
    return {r["rollout_id"]: (float(p), float(f)) for r, p, f in zip(test, aP, aF)}, dict(P=cl.det_json(tm), safe_plus_P=cl.det_json(fu))


def run(args):
    out = Path(args.output)
    if (out / "bundle").exists() or (out / "key").exists():
        raise SystemExit(f"error: {out} already holds a bundle or key")
    bundle = json.load(open(args.runtime_bundle))
    _, cal = cl.primary(args.calibration, "calibration")
    _, test = cl.primary(args.test, "test")
    alarms, fits = alarm_indices(cal, test, bundle)
    meta = manifest_rows(args.collection_root)
    failures = defaultdict(list)
    for r in test:
        if r["failed"]:
            failures[r["task_name"]].append(r)
    rng = random.Random(SEED)
    chosen = []
    for task in sorted(failures):
        rows = sorted(failures[task], key=lambda r: r["rollout_id"])
        kind = rows[0].get("task_type")
        k = PER_TYPE.get(kind)
        if k is None:
            raise ValueError(f"Unknown task type {kind!r} for {task}")
        chosen += rng.sample(rows, k)
    rng.shuffle(chosen)
    (out / "bundle" / "videos").mkdir(parents=True)
    (out / "key").mkdir(parents=True)
    items, key = [], {}
    for i, r in enumerate(chosen, 1):
        m = meta.get(r["rollout_id"])
        if m is None or not m.get("video_path"):
            raise ValueError(f"No video recorded for {r['rollout_id']}")
        src = Path(m["_shard"]) / m["video_path"]
        if not src.is_file():
            raise ValueError(f"Missing video file {src}")
        vid = f"v{i:02d}"
        shutil.copy2(src, out / "bundle" / "videos" / f"{vid}{src.suffix}")
        steps = int(m.get("replan_steps") or 16)
        items.append(dict(id=vid, file=f"videos/{vid}{src.suffix}", task_name=r["task_name"],
                          instruction=m.get("task_instruction"), fps=int(m.get("video_fps") or DEFAULT_FPS),
                          video_frame_stride=int(m.get("video_frame_stride") or 1), steps_per_inference=steps,
                          horizon_inferences=int(r["horizon"]), length_inferences=int(r["L"]),
                          num_env_steps=m.get("num_env_steps"), timeout_env_steps=m.get("timeout_horizon")))
        p, f = alarms[r["rollout_id"]]
        key[vid] = dict(rollout_id=r["rollout_id"], task_name=r["task_name"], task_type=r.get("task_type"),
                        horizon_inferences=int(r["horizon"]), length_inferences=int(r["L"]), steps_per_inference=steps,
                        alarm_P_inference=None if math.isinf(p) else p, alarm_safe_plus_P_inference=None if math.isinf(f) else f)
    (out / "bundle" / "manifest.json").write_text(json.dumps(dict(
        plan="docs/no_return_labeling_plan.md", sample_seed=SEED, videos=items), indent=1) + "\n")
    (out / "key" / "key.json").write_text(json.dumps(dict(
        plan="docs/no_return_labeling_plan.md", note="Do not open before labeling is finished.",
        fits=fits, videos=key), indent=1, default=float) + "\n")
    counts = defaultdict(int)
    for v in key.values():
        counts[v["task_type"]] += 1
    print(f"bundled {len(items)} videos ({dict(counts)}) into {out / 'bundle'}; key in {out / 'key'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--calibration", "--test", "--runtime-bundle", "--collection-root", "--output"):
        p.add_argument(name, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
