"""Build the second-labeler bundle for the point-of-no-return reliability check.

Plan: docs/no_return_labeling_plan.md (optional reliability check). Picks 6 atomic and
6 composite videos from the first bundle (seed 1), keeps their anonymous IDs so labels can be
joined, and writes a self-contained folder: the videos, a manifest, the labeling page and a
short guide. It reads only the task type from the key; no alarm times and no first-labeler
labels are copied.

Usage:
  python3 -I no_return_subset.py --bundle <out>/bundle --key <out>/key/key.json \
      --labeler-page scripts/xr1_lead_time/no_return_labeler.html --output <out>/second_labeler_bundle
"""
import argparse, json, random, shutil
from pathlib import Path

PER_TYPE = {"atomic": 6, "composite": 6}
SEED = 1
BUNDLE_ID = "second_labeler_v1"

GUIDE = """# Point-of-no-return labeling: guide for the second labeler

You will label 12 short videos of a robot that FAILED a kitchen task in simulation: it did not
finish before the time limit. Label each video on your own; please do not discuss specific
videos with the first labeler until you have exported your labels.

Open `no_return_labeler.html` (Chrome or Safari), click **Choose bundle folder**, select this
folder, and type your initials in **Labeler**. Labels save in the browser as you go.

For each video mark:

- **t_e, first visible error (key E):** the first moment the robot does something that deviates
  from a successful execution (wrong grasp, slip, collision, heading for the wrong object), even
  if it could still recover.
- **t_u, point of no return (key U):** the earliest moment after which, in your judgment, the
  robot can no longer complete the task before the time limit without outside help (object
  dropped out of reach or off the counter, object knocked over or wedged so it cannot be
  re-grasped, an irreversible wrong action, arm or gripper stuck for good).
- **Recoverable until timeout:** tick this instead of t_u if the robot stays in a state from
  which it could still succeed until the end (still trying, slow, hesitating near the goal).
- **Category** and your **confidence** in t_u.

Controls: Space play/pause, Left/Right one frame, Shift+Left/Right one second, [ and ] change
speed (2-4x is fine for watching; slow down to place marks).

When all 12 show a check mark, click **Export labels** and send the downloaded
`no_return_labels.json` back. It takes about 30 minutes.
"""


def run(args):
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"error: output exists: {out}")
    bundle = Path(args.bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    types = {vid: v["task_type"] for vid, v in json.loads(Path(args.key).read_text())["videos"].items()}
    rng = random.Random(SEED)
    chosen = []
    for kind, k in PER_TYPE.items():
        pool = sorted((v for v in manifest["videos"] if types[v["id"]] == kind), key=lambda v: v["id"])
        if len(pool) < k:
            raise ValueError(f"Only {len(pool)} {kind} videos in the bundle")
        chosen += rng.sample(pool, k)
    chosen.sort(key=lambda v: v["id"])
    (out / "videos").mkdir(parents=True)
    for v in chosen:
        shutil.copy2(bundle / v["file"], out / v["file"])
    (out / "manifest.json").write_text(json.dumps(dict(manifest, bundle_id=BUNDLE_ID, subset_seed=SEED, videos=chosen), indent=1) + "\n")
    shutil.copy2(args.labeler_page, out / "no_return_labeler.html")
    (out / "LABELING_GUIDE.md").write_text(GUIDE)
    print(f"second-labeler bundle: {len(chosen)} videos ({', '.join(v['id'] for v in chosen)}) in {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--bundle", "--key", "--labeler-page", "--output"):
        p.add_argument(name, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
