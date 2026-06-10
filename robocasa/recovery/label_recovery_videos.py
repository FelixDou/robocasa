"""Rename recovery rollout videos using labels from results.json.

This is useful for older runs created before video filenames included outcome
labels. It scans result records, finds matching ``rollout_XXXX_seed_Y.mp4``
files under a video root, and renames them with the same convention used by
``evaluate_recovery_benchmark.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa.recovery.evaluate_recovery_benchmark import (
    make_labeled_video_path,
    rename_video_with_result,
)


def iter_result_records(results_root):
    for path in sorted(results_root.glob("*/results.json")):
        data = json.loads(path.read_text())
        for mode, payload in data.get("modes", {}).items():
            for record in payload.get("rollouts", []):
                yield mode, record


def candidate_video_paths(video_root, mode, record):
    task = str(record.get("task", "")).replace("/", "_")
    rollout_i = record.get("rollout_index")
    seed = record.get("seed")
    if rollout_i is None or seed is None:
        return
    basename = f"rollout_{int(rollout_i):04d}_seed_{seed}.mp4"

    # Common layouts:
    #   videos/<mode>/<task>/rollout_...
    #   videos/<mode>/<mode>/<task>/rollout_...
    yield video_root / mode / task / basename
    yield video_root / mode / mode / task / basename
    yield from video_root.glob(f"**/{task}/{basename}")


def label_videos(results_root, video_root, dry_run=False):
    renamed = 0
    missing = 0
    already_labeled = 0
    for mode, record in iter_result_records(results_root):
        found = None
        for candidate in candidate_video_paths(video_root, mode, record):
            if candidate.exists():
                found = candidate
                break
        if found is None:
            missing += 1
            continue

        labeled = make_labeled_video_path(found, record)
        if labeled == found or found.name.startswith(("HIGH_LEVEL_", "RECOVERY_", "ERROR")):
            already_labeled += 1
            continue
        if dry_run:
            print(f"{found} -> {labeled}")
        else:
            renamed_path = rename_video_with_result(found, record)
            print(f"{found} -> {renamed_path}")
        renamed += 1
    return renamed, missing, already_labeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Run root containing <mode>/results.json files.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Video root. Defaults to <results-root>/videos.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    video_root = args.video_root or args.results_root / "videos"
    renamed, missing, already_labeled = label_videos(
        args.results_root,
        video_root,
        dry_run=args.dry_run,
    )
    print(
        f"renamed={renamed} missing={missing} already_labeled={already_labeled}"
    )


if __name__ == "__main__":
    main()
