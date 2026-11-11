"""R1 expert segments: human-demo segments for the stages the alarm states are stuck on.

Plan: docs/recovery_harvest_finetune_plan.md, step R1 of "Redesign after Gate A".
Option 1 there is segment replay from the RoboCasa v1.0 composite human demos.
This module answers its two open questions:

* do the demos carry replayable simulator states? (yes: the LeRobot folders
  ship ``extras/episode_*/{states.npz, ep_meta.json, model.xml.gz}``);
* for each stage an R0 class-2 state is stuck on, are there enough clean demo
  segments that complete that stage?

``index`` (simulator, CPU, no policy) replays each demo frame by frame through
``reset_to`` exactly as upstream's subtask labeling pipeline does, and records
per frame:

- upstream's per-frame labels from its registered labeler for the task
  (``robocasa.utils.subtask_labeling``: subtask index, atomic-task name,
  pick/place/... stage, instruction) and the labeler's own sanity checks;
- our runtime predicates (``get_subtask_eval``), i.e. the same stage
  definitions the ladder, the alarm branches and the R0 audit use;
- every task object's pose (distractors excluded) and the gripper position.

Upstream's labels need the whole demo and assume it succeeds, so they are used
here only on successful expert demos, never on policy rollouts. Our predicates
stay the stage definition. The two are joined per stage: the frame where a
stage's predicate first becomes (and stays) true picks the upstream segment
that completes it, and that segment's subtask (its pick, place and any
navigation between them) is the expert segment for the stage. The ladder's
ordered window is recorded too. It can be empty when the demonstrator did the
stages in another order (ArrangeTea's mug before the kettle).

``coverage`` (no simulator) joins the index with the R0 audit and writes, per
stuck (task, stage), the number of usable segments, how consistently the stage
maps to one upstream subtask and object, and the R1 decision with the
thresholds frozen below. It also writes ``stage_segments.jsonl``, the lookup
table R2 draws source segments from.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np

from robocasa.recovery import recoverability_ladder as ladder


PILOT_TASKS = ("ArrangeBreadBasket", "ArrangeTea", "BreadSelection", "CuttingToolSelection", "GarnishPancake")
DISTRACTOR_PREFIXES = ("distr", "dstr")
DONE_STAGE = "done"
ALL_LATCHED = "all_latched"
NAVIGATE = "NavigateKitchen"

# Frozen with this code (11 November 2026, before any R1 output exists).
DEFAULT_MAX_EPISODES = 100  # first N demos of each task's 500 (target split)
STABLE_FRAMES = 5           # a stage counts as reached once its predicate holds this long
BOUNDARY_TOLERANCE = 10     # frames: a predicate turning true this soon after a boundary completes the earlier segment
MIN_MOVED_DISTANCE = 0.03   # m: below this no task object moved (fixture subtask: open, close)
R1_MIN_SEGMENTS = 10        # usable segments per stuck stage (R2 draws N = 10 attempts from distinct segments)
R1_MIN_CONSISTENCY = 0.8    # share of usable segments that agree on the upstream subtask and moved object

INDEX_GLOB = "index*.jsonl"
ERRORS_GLOB = "errors*.jsonl"
MANIFEST_NAME = "index_manifest.json"
EPISODES_DIR = "episodes"


# ---- dataset access -----------------------------------------------------------------------------

def lerobot_dir(task_name, dataset_base, overrides=None):
    overrides = overrides or {}
    if task_name in overrides:
        return Path(overrides[task_name])
    from robocasa.utils.dataset_registry import COMPOSITE_TASK_DATASETS

    rel = COMPOSITE_TASK_DATASETS[task_name]["target"]["human_path"]
    return Path(dataset_base) / rel / "lerobot"


def episode_indices(lerobot):
    extras = Path(lerobot) / "extras"
    return sorted(int(p.name.split("_")[1]) for p in extras.glob("episode_*") if p.is_dir())


def load_episode(lerobot, episode):
    ep_dir = Path(lerobot) / "extras" / f"episode_{int(episode):06d}"
    states = np.load(ep_dir / "states.npz")["states"]
    ep_meta = json.loads((ep_dir / "ep_meta.json").read_text())
    with gzip.open(ep_dir / "model.xml.gz", "rt") as stream:
        model_xml = stream.read()
    return states, ep_meta, model_xml


def parquet_annotation(lerobot, episode):
    """(row count, published per-frame ``subtask_idx`` or None); (None, None) without pyarrow or file."""
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None, None
    paths = sorted((Path(lerobot) / "data").rglob(f"episode_{int(episode):06d}.parquet"))
    if not paths:
        return None, None
    table = pq.read_table(paths[0])
    idx = np.asarray(table.column("subtask_idx").to_numpy()) if "subtask_idx" in table.column_names else None
    return int(table.num_rows), idx


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- pure helpers (simulator-free, tested) ------------------------------------------------------

def segments_from_labels(subtask_idx, names, stages, langs):
    """Contiguous runs of the per-frame segment index -> [{index, name, stage, lang, start, end}]."""
    idx = np.asarray(subtask_idx)
    out = []
    if idx.size == 0:
        return out
    starts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
    ends = np.concatenate([starts[1:], [idx.size]])
    for start, end in zip(starts, ends):
        out.append({"index": int(idx[start]), "name": str(names[start]), "stage": str(stages[start]),
                    "lang": str(langs[start]), "start": int(start), "end": int(end)})
    return out


def first_stable_true(values, stable=STABLE_FRAMES):
    """First frame from which ``values`` stays true for ``stable`` frames (or to the end); None if never."""
    values = np.asarray(values, dtype=bool)
    run = 0
    for t, value in enumerate(values):
        run = run + 1 if value else 0
        if run >= stable:
            return t - stable + 1
    if run:  # true through the last frame, for fewer than ``stable`` frames
        return len(values) - run
    return None


def segment_at(segments, frame):
    for k, seg in enumerate(segments):
        if seg["start"] <= frame < seg["end"]:
            return k
    return len(segments) - 1 if segments else None


def match_segment(segments, frame, tolerance=BOUNDARY_TOLERANCE):
    """Index of the upstream segment that completes a stage whose predicate turned true at ``frame``.

    Usually the segment containing ``frame`` (a placement settles before the
    release that ends the place). The previous segment is used when ``frame``
    falls in the terminal ``done`` segment, or just after a boundary: within
    ``tolerance`` frames and the first quarter of the segment (a grasp predicate
    turning true right after the pick ends at grasp onset).
    """
    k = segment_at(segments, frame)
    if k is None or k == 0:
        return k
    seg = segments[k]
    offset = frame - seg["start"]
    if seg["stage"] == DONE_STAGE or offset <= min(tolerance, 0.25 * (seg["end"] - seg["start"])):
        k -= 1
    return k


def subtask_group(segments, k):
    """[first, k]: segment k with the earlier segments of its subtask.

    Walks back over segments with the same atomic-task name and over navigation
    between them, and stops at that subtask's pick (a group holds one pick at most).
    """
    name = segments[k]["name"]
    j = k
    while j > 0 and not (segments[j]["stage"] == "pick" and segments[j]["name"] == name):
        prev = segments[j - 1]
        if prev["name"] == name or (prev["name"] == NAVIGATE and name != NAVIGATE):
            j -= 1
        else:
            break
    return j, k


def moved_object(object_names, object_pos, start, end, min_distance=MIN_MOVED_DISTANCE):
    """Task object with the largest displacement over frames [start, end]; None if none moved."""
    if not len(object_names):
        return None, 0.0
    end = min(int(end), object_pos.shape[0] - 1)
    disp = np.linalg.norm(object_pos[end] - object_pos[int(start)], axis=-1)
    i = int(np.argmax(disp))
    return (str(object_names[i]) if disp[i] >= min_distance else None), float(disp[i])


def stage_events(predicate_names, predicate_values, segments, object_names, object_pos):
    """Per predicate: when it is first reached and which upstream subtask completes it."""
    events = {}
    for j, name in enumerate(predicate_names):
        column = predicate_values[:, j]
        event = {"true_at_start": bool(column[0]), "first_true": None, "segment": None}
        if column[0]:
            events[name] = event
            continue
        frame = first_stable_true(column)
        event["first_true"] = frame
        if frame is not None and segments:
            k = match_segment(segments, frame)
            first, last = subtask_group(segments, k)
            obj, dist = moved_object(object_names, object_pos, segments[first]["start"], segments[last]["end"] - 1)
            event["segment"] = {
                "index": segments[k]["index"], "name": segments[k]["name"], "stage": segments[k]["stage"],
                "lang": segments[k]["lang"], "start": segments[first]["start"], "end": segments[last]["end"],
                "stages": [[s["name"], s["stage"]] for s in segments[first:last + 1]],
                "langs": [s["lang"] for s in segments[first:last + 1]],
                "moved_object": obj, "moved_distance": round(dist, 4),
            }
        events[name] = event
    return events


def ordered_windows(required, predicate_names, predicate_values, task_success=None):
    """Ladder-ordered stage windows {stage: (entry_frame, latch_frame or None)} (``OrderedProgress`` rules)."""
    column = {name: i for i, name in enumerate(predicate_names)}
    missing = [name for name in required if name not in column]
    if missing:
        raise ValueError(f"required stages missing from the recorded predicates: {missing}")
    tracker = ladder.OrderedProgress(required)
    windows = {}
    entry = 0
    for t in range(predicate_values.shape[0]):
        payload = {"predicates": {name: {"value": bool(predicate_values[t, i])} for name, i in column.items()},
                   "task_success": bool(task_success[t]) if task_success is not None else False}
        for name in tracker.update(payload):
            windows[name] = (entry, t)
            entry = t
    for name in required:
        windows.setdefault(name, (entry if name == tracker.current else None, None))
    return windows


def instruction_template(langs, object_lang):
    """Upstream instructions with each task object's display name replaced by ``{object_key}``."""
    names = sorted(((str(v), k) for k, v in (object_lang or {}).items() if v), key=lambda kv: -len(kv[0]))
    out = []
    for lang in langs:
        for text, key in names:
            lang = lang.replace(text, "{" + key + "}")
        out.append(lang)
    return out


def summarize_episode(task_name, episode, *, labels, predicate_names, predicate_values, task_success,
                      object_names, object_pos, issues, review, parquet_rows=None, parquet_idx=None,
                      object_lang=None):
    segments = segments_from_labels(labels["subtask_idx"], labels["subtask_name"], labels["subtask_stage"],
                                    labels["subtask_lang"])
    n = int(predicate_values.shape[0])
    agreement = None
    if parquet_idx is not None and len(parquet_idx) == n:
        agreement = float(np.mean(np.asarray(parquet_idx) == np.asarray(labels["subtask_idx"])))
    return {
        "task_name": task_name,
        "episode": int(episode),
        "frames": n,
        "parquet_rows": parquet_rows,
        "rows_match": None if parquet_rows is None else bool(parquet_rows == n),
        "published_label_agreement": agreement,
        "success_at_end": bool(task_success[-1]) if n else False,
        "labeler_issues": list(issues),
        "labeler_review": list(review),
        "clean": bool(n and task_success[-1] and not issues),
        "segments": segments,
        "objects": [str(o) for o in object_names],
        "object_lang": dict(object_lang or {}),
        "stage_events": stage_events(predicate_names, predicate_values, segments, object_names, object_pos),
        "created_at": ladder.utc_now(),
    }


# ---- simulator replay ---------------------------------------------------------------------------

def build_env(env_args):
    """Robosuite env from the dataset's env_args, as upstream builds it (no renderer, no controller)."""
    import robosuite
    import robocasa  # noqa: F401  (registers the kitchen envs)

    kwargs = dict(env_args.get("env_kwargs", env_args))
    kwargs.update(env_name=env_args["env_name"], has_renderer=False, has_offscreen_renderer=False,
                  use_camera_obs=False, translucent_robot=False)
    kwargs.pop("controller_configs", None)
    if kwargs.get("obj_instance_split") == "train":
        kwargs["obj_instance_split"] = "pretrain"
    return robosuite.make(**kwargs)


def kitchen_frame(env, object_names):
    from robocasa.recovery.subtask_eval import get_subtask_eval

    payload = get_subtask_eval(env) or {}
    predicates = {name: bool((p or {}).get("value", False)) for name, p in (payload.get("predicates") or {}).items()}
    data = env.sim.data
    pos = np.array([data.body_xpos[env.obj_body_id[n]] for n in object_names], dtype=np.float32).reshape(-1, 3)
    quat = np.array([data.body_xquat[env.obj_body_id[n]] for n in object_names], dtype=np.float32).reshape(-1, 4)
    eef = np.asarray(data.site_xpos[env.robots[0].eef_site_id["right"]], dtype=np.float32)
    return {"predicates": predicates, "task_success": bool(payload.get("task_success", False)),
            "required": list(payload.get("required_predicates", [])), "pos": pos, "quat": quat, "eef": eef}


def replay_episode(env, labeler, states, ep_meta, model_xml, *, reset_fn, frame_fn, objects_fn):
    """Replay one demo. Returns (LabelResult, issues, object_names, frames)."""
    reset_fn(env, {"model": model_xml, "states": states[0], "ep_meta": json.dumps(ep_meta)})
    labeler.setup(env, ep_meta)
    object_names = objects_fn(env)
    history, frames = [], []
    for t in range(states.shape[0]):
        reset_fn(env, {"states": states[t]})
        history.append(labeler.per_step(env, t))
        frames.append(frame_fn(env, object_names))
    result = labeler.resolve(history)
    issues = labeler.validate(result)
    return result, issues, object_names, frames


def frames_to_arrays(frames):
    names = sorted(frames[0]["predicates"])
    values = np.array([[f["predicates"].get(n, False) for n in names] for f in frames], dtype=bool)
    success = np.array([f["task_success"] for f in frames], dtype=bool)
    pos = np.stack([f["pos"] for f in frames]) if frames else np.zeros((0, 0, 3), np.float32)
    quat = np.stack([f["quat"] for f in frames]) if frames else np.zeros((0, 0, 4), np.float32)
    eef = np.stack([f["eef"] for f in frames]) if frames else np.zeros((0, 3), np.float32)
    return names, values, success, pos, quat, eef


def save_episode_arrays(path, *, labels, predicate_names, predicate_values, task_success, required,
                        object_names, object_pos, object_quat, eef_pos):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = {}
    for key in ("subtask_name", "subtask_stage", "subtask_lang"):
        vocab, inverse = np.unique(np.asarray(labels[key]).astype(str), return_inverse=True)
        tables[f"{key}_vocab"] = vocab
        tables[f"{key}_code"] = inverse.astype(np.int16)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp, subtask_idx=np.asarray(labels["subtask_idx"], dtype=np.int32), **tables,
        predicate_names=np.asarray(predicate_names), predicate_values=predicate_values,
        task_success=task_success, required_env=np.asarray(required), object_names=np.asarray(object_names),
        object_pos=object_pos, object_quat=object_quat, eef_pos=eef_pos)
    tmp.replace(path)


def _jobs(tasks, lerobots, max_episodes):
    jobs = []
    for task in tasks:
        eps = episode_indices(lerobots[task])
        if max_episodes:
            eps = eps[: int(max_episodes)]
        jobs.extend((task, ep) for ep in eps)
    return jobs


def _done(output_dir):
    done = set()
    for path in Path(output_dir).glob(INDEX_GLOB):
        done.update((r["task_name"], int(r["episode"])) for r in ladder.read_jsonl(path))
    return done


def run_index(output_dir, *, tasks=PILOT_TASKS, dataset_base=None, lerobot_overrides=None,
              max_episodes=DEFAULT_MAX_EPISODES, shard_index=0, num_shards=1, resume=False, max_errors=10,
              log=print):
    from robocasa.scripts.dataset_scripts.label_subtasks import reset_to
    from robocasa.utils.subtask_labeling import get_labeler

    output_dir = Path(output_dir)
    lerobots = {t: lerobot_dir(t, dataset_base, lerobot_overrides) for t in tasks}
    for task, path in lerobots.items():
        if not (path / "extras" / "dataset_meta.json").exists():
            raise FileNotFoundError(f"{task}: no LeRobot extras at {path} (download the target human split)")
    manifest_path = output_dir / MANIFEST_NAME
    if output_dir.exists() and any(output_dir.glob(INDEX_GLOB)) and not resume:
        raise FileExistsError(f"{output_dir} already has an index; pass --resume")
    if not manifest_path.exists():
        import robocasa.utils.subtask_labeling as labeling

        labeler_files = {}
        for task in tasks:
            module = sys.modules[type(get_labeler(task)).__module__]
            labeler_files[task] = {"file": str(Path(module.__file__).relative_to(Path(labeling.__file__).parents[3])),
                                   "sha256": file_sha256(module.__file__)}
        ladder.atomic_write_json(manifest_path, {
            "kind": "r1_expert_segment_index", "git_commit": ladder.current_git_commit(),
            "tasks": list(tasks), "lerobot": {t: str(p) for t, p in lerobots.items()},
            "max_episodes": max_episodes, "num_shards": num_shards, "labelers": labeler_files,
            "thresholds": {"stable_frames": STABLE_FRAMES, "boundary_tolerance": BOUNDARY_TOLERANCE,
                           "min_moved_distance": MIN_MOVED_DISTANCE},
            "created_at": ladder.utc_now()})
    index_path = output_dir / f"index.shard{shard_index}.jsonl"
    errors_path = output_dir / f"errors.shard{shard_index}.jsonl"
    done = _done(output_dir)
    jobs = [j for i, j in enumerate(_jobs(tasks, lerobots, max_episodes)) if i % num_shards == shard_index]
    todo = [j for j in jobs if j not in done]
    log(f"shard {shard_index}/{num_shards}: {len(jobs)} demos, {len(jobs) - len(todo)} done, {len(todo)} to go")
    env, env_task, errors = None, None, 0
    for n, (task, episode) in enumerate(todo, 1):
        try:
            if env_task != task:
                if env is not None:
                    env.close()
                meta = json.loads((lerobots[task] / "extras" / "dataset_meta.json").read_text())
                env, env_task = build_env(meta["env_args"]), task
            labeler = get_labeler(task)
            if labeler is None:
                raise KeyError(f"no upstream labeler registered for {task}")
            states, ep_meta, model_xml = load_episode(lerobots[task], episode)
            result, issues, object_names, frames = replay_episode(
                env, labeler, states, ep_meta, model_xml, reset_fn=reset_to, frame_fn=kitchen_frame,
                objects_fn=lambda e: [o for o in e.objects if not str(o).startswith(DISTRACTOR_PREFIXES)])
            names, values, success, pos, quat, eef = frames_to_arrays(frames)
            labels = {"subtask_idx": result.subtask_idx, "subtask_name": result.subtask_name,
                      "subtask_stage": result.subtask_stage, "subtask_lang": result.subtask_lang}
            save_episode_arrays(output_dir / EPISODES_DIR / task / f"episode_{episode:06d}.npz", labels=labels,
                                predicate_names=names, predicate_values=values, task_success=success,
                                required=frames[0]["required"], object_names=object_names, object_pos=pos,
                                object_quat=quat, eef_pos=eef)
            object_lang = {}
            for name in object_names:
                try:
                    object_lang[name] = str(env.get_obj_lang(name))
                except Exception:
                    object_lang[name] = None
            rows, published = parquet_annotation(lerobots[task], episode)
            row = summarize_episode(task, episode, labels=labels, predicate_names=names, predicate_values=values,
                                    task_success=success, object_names=object_names, object_pos=pos,
                                    issues=issues, review=result.review, parquet_rows=rows, parquet_idx=published,
                                    object_lang=object_lang)
            row["required_env"] = list(frames[0]["required"])
            ladder.append_jsonl(index_path, row)
            log(f"[{n}/{len(todo)}] {task} ep {episode}: {row['frames']} frames, "
                f"success={row['success_at_end']} issues={len(issues)}")
        except Exception as exc:  # recorded, the run continues up to max_errors
            errors += 1
            ladder.append_jsonl(errors_path, {"task_name": task, "episode": int(episode), "error": repr(exc),
                                              "traceback": traceback.format_exc(), "created_at": ladder.utc_now()})
            log(f"ERROR {task} ep {episode}: {exc!r}")
            if errors >= max_errors:
                raise RuntimeError(f"stopping after {errors} errors (see {errors_path})") from exc
    if env is not None:
        env.close()
    log(f"INDEX SHARD {shard_index} DONE ({errors} errors)")


# ---- coverage and the R1 decision ---------------------------------------------------------------

def load_index(index_dir):
    rows = []
    for path in sorted(Path(index_dir).glob(INDEX_GLOB)):
        rows.extend(ladder.read_jsonl(path))
    seen = Counter((r["task_name"], r["episode"]) for r in rows)
    dup = sorted(k for k, n in seen.items() if n > 1)
    if dup:
        raise ValueError(f"duplicate index rows: {dup[:5]}")
    errors = []
    for path in sorted(Path(index_dir).glob(ERRORS_GLOB)):
        errors.extend(ladder.read_jsonl(path))
    return rows, errors


def load_r0(audit_dir, allow_provisional=False):
    """Audited states with their class and the ladder's required list per task."""
    audit_dir = Path(audit_dir)
    states = ladder.read_jsonl(audit_dir / "states.jsonl")
    if not states:
        raise ValueError(f"No R0 audit states in {audit_dir}")
    final = {r["parent_id"]: r["final_class"] for r in ladder.read_jsonl(audit_dir / "r0_states.jsonl")}
    if not final and not allow_provisional:
        raise ValueError("R0 decide has not run (no r0_states.jsonl); run it or pass --allow-provisional")
    required = {}
    out = []
    for row in states:
        req = list(row["required_subtasks"])
        if required.setdefault(row["task_name"], req) != req:
            raise ValueError(f"{row['task_name']}: required subtasks differ between audited parents")
        cls = final.get(row["parent_id"], row["provisional_class"] if allow_provisional else None)
        if cls is None:
            raise ValueError(f"{row['parent_id']} has no final class in r0_states.jsonl")
        stage, source = row["stage_at_alarm"], "alarm"
        if stage == ALL_LATCHED:
            # Amendment 1: every stage had latched, so the stage to redo is the one undone at the alarm.
            undone = [n for n in (row.get("regressed_at_alarm") or []) if n in req]
            stage, source = (undone[0], "undone") if undone else (ALL_LATCHED, "none")
        out.append({"parent_id": row["parent_id"], "task_name": row["task_name"], "stage": stage,
                    "stage_source": source, "class": cls, "class_source": "final" if final else "provisional"})
    return out, required


def _modal_share(values):
    values = [v for v in values]
    if not values:
        return None, 0.0
    value, n = Counter(values).most_common(1)[0]
    return value, n / len(values)


def target_object(segment, object_lang):
    """(object key, source): the first task object the segment's instructions name, else the most-moved one."""
    for text in instruction_template(segment.get("langs") or [segment["lang"]], object_lang):
        match = re.search(r"\{([^{}]+)\}", text)
        if match:
            return match.group(1), "instruction"
    return segment.get("moved_object"), "displacement"


def stage_coverage(rows, task_name, stage, ordered=None, *, min_segments=R1_MIN_SEGMENTS,
                   min_consistency=R1_MIN_CONSISTENCY):
    task_rows = [r for r in rows if r["task_name"] == task_name]
    events = [(r, r["stage_events"].get(stage)) for r in task_rows]
    reached = [(r, e) for r, e in events if e and e["segment"]]
    usable = [(r, e) for r, e in reached if r["clean"]]
    keys = [(e["segment"]["name"], e["segment"]["stage"]) for _, e in usable]
    modal_key, key_share = _modal_share(keys)
    objects = [e["segment"]["moved_object"] for _, e in usable]
    modal_object, object_share = _modal_share(objects)
    consistent = [(r, e) for r, e in usable if (e["segment"]["name"], e["segment"]["stage"]) == modal_key
                  and e["segment"]["moved_object"] == modal_object]
    ok_original = len(consistent) >= min_segments and min(key_share, object_share) >= min_consistency
    # Amendment 1 (11 November 2026, after the first coverage run): the object is the one the
    # instructions name, and each object that is a target in at least (1 - min_consistency) of the
    # segments needs min_segments of its own. Carried contents (bread in the basket) and scene-dependent
    # targets (knife or peeler) broke the single most-moved-object rule.
    on_key = [(r, e) for r, e in usable if (e["segment"]["name"], e["segment"]["stage"]) == modal_key]
    targets = [target_object(e["segment"], r.get("object_lang")) for r, e in on_key]
    counts = Counter(t for t, _ in targets)
    floor = (1.0 - min_consistency) * len(on_key)
    groups = {t: n for t, n in counts.items() if n >= floor and n > 0}
    covered = sum(groups.values()) / len(on_key) if on_key else 0.0
    ok = bool(groups) and key_share >= min_consistency and covered >= min_consistency \
        and all(n >= min_segments for n in groups.values())
    consistent = [(r, e) for (r, e), (t, _) in zip(on_key, targets) if t in groups]
    lengths = [e["segment"]["end"] - e["segment"]["start"] for _, e in consistent]
    templates = [tuple(instruction_template(e["segment"].get("langs") or [e["segment"]["lang"]], r.get("object_lang")))
                 for r, e in consistent]
    modal_template, template_share = _modal_share(templates)
    out = {
        "task_name": task_name, "stage": stage, "demos": len(task_rows),
        "true_at_start": sum(bool(e and e["true_at_start"]) for _, e in events),
        "reached": len(reached), "usable": len(usable), "consistent": len(consistent),
        "upstream_segment": list(modal_key) if modal_key else None, "segment_share": round(key_share, 3),
        "moved_object": modal_object, "object_share": round(object_share, 3),
        "median_segment_frames": float(np.median(lengths)) if lengths else None,
        "instruction_template": list(modal_template) if modal_template else None,
        "instruction_share": round(template_share, 3),
        "target_objects": {str(t): n for t, n in sorted(groups.items(), key=lambda kv: -kv[1])},
        "target_coverage": round(covered, 3),
        "target_from_instruction": round(sum(src == "instruction" for _, src in targets) / len(targets), 3)
        if targets else 0.0,
        "segment_replay_ok": bool(ok),
        "segment_replay_ok_original_rule": bool(ok_original),
    }
    if ordered is not None:
        spans = [o for o in ordered if o is not None]
        out["ordered_window_empty"] = sum(1 for s, e in spans if e is not None and e == s)
        out["ordered_window_median_frames"] = float(np.median([e - s for s, e in spans if e is not None])) \
            if any(e is not None for _, e in spans) else None
    return out


def _ordered_by_stage(index_dir, rows, required):
    """{(task, stage): [window or None per clean demo]} from the saved arrays, with the ladder's lists."""
    out = defaultdict(list)
    for row in rows:
        task = row["task_name"]
        if task not in required or not row["clean"]:
            continue
        path = Path(index_dir) / EPISODES_DIR / task / f"episode_{int(row['episode']):06d}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            names = [str(n) for n in data["predicate_names"]]
            windows = ordered_windows(required[task], names, data["predicate_values"], data["task_success"])
        for stage in required[task]:
            out[(task, stage)].append(windows.get(stage) if windows.get(stage, (None, None))[1] is not None else None)
    return out


def coverage(index_dir, audit_dir, output_dir, *, allow_provisional=False, min_segments=R1_MIN_SEGMENTS,
             min_consistency=R1_MIN_CONSISTENCY):
    output_dir = Path(output_dir)
    rows, errors = load_index(index_dir)
    if not rows:
        raise ValueError(f"No index rows in {index_dir}")
    states, required = load_r0(audit_dir, allow_provisional)
    ordered = _ordered_by_stage(index_dir, rows, required)
    class2 = [s for s in states if s["class"] == "recoverable"]
    unstaged = [s for s in class2 if s["stage"] == ALL_LATCHED]
    stuck = Counter((s["task_name"], s["stage"]) for s in class2 if s["stage"] != ALL_LATCHED)
    tables = []
    for task in sorted(required):
        for stage in required[task]:
            cov = stage_coverage(rows, task, stage, ordered.get((task, stage)), min_segments=min_segments,
                                 min_consistency=min_consistency)
            cov["class2_states"] = stuck.get((task, stage), 0)
            tables.append(cov)
    by_key = {(c["task_name"], c["stage"]): c for c in tables}
    unknown = sorted(k for k in stuck if k not in by_key)
    needed = [by_key[k] for k in sorted(stuck) if k in by_key]
    failing = [f"{c['task_name']}:{c['stage']}" for c in needed if not c["segment_replay_ok"]]
    failing += [f"{t}:{s} (no stage in the ladder list)" for t, s in unknown]
    failing_original = [f"{c['task_name']}:{c['stage']}" for c in needed if not c["segment_replay_ok_original_rule"]]
    if failing:
        decision = "segment_replay_incomplete_check_scripted_expert"
    elif unstaged:
        decision = "segment_replay_except_unstaged_states"
    else:
        decision = "segment_replay"
    health = {}
    for task in sorted({r["task_name"] for r in rows}):
        task_rows = [r for r in rows if r["task_name"] == task]
        agreement = [r["published_label_agreement"] for r in task_rows if r["published_label_agreement"] is not None]
        health[task] = {
            "demos": len(task_rows), "clean": sum(r["clean"] for r in task_rows),
            "success_at_end": sum(r["success_at_end"] for r in task_rows),
            "labeler_flagged": sum(bool(r["labeler_issues"]) for r in task_rows),
            "rows_mismatch": sum(r["rows_match"] is False for r in task_rows),
            "published_label_agreement_mean": round(float(np.mean(agreement)), 4) if agreement else None,
            "errors": sum(e["task_name"] == task for e in errors),
        }
    result = {
        "kind": "r1_coverage", "index_dir": str(index_dir), "audit_dir": str(audit_dir),
        "class_source": states[0]["class_source"], "class2_states": len(class2),
        "class2_states_by_stage_source": dict(Counter(s["stage_source"] for s in class2)),
        "unstaged_states": [{"parent_id": s["parent_id"], "task_name": s["task_name"]} for s in unstaged],
        "decision_original_rule": "segment_replay" if not failing_original and not unknown
        else "segment_replay_incomplete_check_scripted_expert",
        "stuck_stages_failing_original_rule": failing_original,
        "thresholds": {"min_segments": min_segments, "min_consistency": min_consistency},
        "decision": decision,
        "stuck_stages_failing": failing, "health": health, "stages": tables, "created_at": ladder.utc_now(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    ladder.atomic_write_json(output_dir / "r1_coverage.json", result)
    with (output_dir / "stage_segments.jsonl").open("w") as stream:
        for row in sorted(rows, key=lambda r: (r["task_name"], r["episode"])):
            if not row["clean"]:
                continue
            for stage, event in sorted(row["stage_events"].items()):
                if event["segment"]:
                    target, source = target_object(event["segment"], row.get("object_lang"))
                    stream.write(json.dumps({"task_name": row["task_name"], "episode": row["episode"],
                                             "stage": stage, **event["segment"], "target_object": target,
                                             "target_source": source}) + "\n")
    write_coverage_md(result, output_dir / "r1_coverage.md")
    return result


def write_coverage_md(result, path):
    lines = ["# R1: expert segments from the human demos", "",
             f"Decision: **{result['decision']}** (class-2 states: {result['class2_states']}, "
             f"classes from {result['class_source']} R0 classes).", "",
             f"Thresholds: at least {result['thresholds']['min_segments']} consistent clean segments per stuck "
             f"stage, and at least {result['thresholds']['min_consistency']:.0%} agreement on the upstream "
             "subtask and on the moved object.", ""]
    lines += ["Amendment 1 (11 November 2026, after the first run): the object of a segment is the one its "
              "instructions name, every object that is a target in at least 20% of the segments needs its own "
              "segments, and an `all_latched` state is assigned the stage undone at its alarm. Under the original "
              f"rule: **{result['decision_original_rule']}** "
              f"(failing: {', '.join(result['stuck_stages_failing_original_rule']) or 'none'}).", ""]
    if result["stuck_stages_failing"]:
        lines += ["Stuck stages without enough segments: " + ", ".join(result["stuck_stages_failing"]), ""]
    if result["unstaged_states"]:
        lines += ["Class-2 states with every stage latched and nothing undone (no stage to replay): "
                  + ", ".join(f"{s['task_name']} `{s['parent_id']}`" for s in result["unstaged_states"]), ""]
    lines += ["## Demo health", "", "| Task | Demos | Clean | Success | Labeler flagged | Row mismatch | "
              "Agreement with published labels | Errors |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task, h in result["health"].items():
        lines.append(f"| {task} | {h['demos']} | {h['clean']} | {h['success_at_end']} | {h['labeler_flagged']} | "
                     f"{h['rows_mismatch']} | {h['published_label_agreement_mean']} | {h['errors']} |")
    lines += ["", "## Stages", "", "| Task | Stage | Class-2 states | Usable | Consistent | Upstream segment | "
              "Share | Target objects | Target coverage | Median frames | Ordered window empty | OK | OK (original) |",
              "|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|---|---|"]
    for c in result["stages"]:
        seg = "/".join(c["upstream_segment"]) if c["upstream_segment"] else "-"
        lines.append(f"| {c['task_name']} | `{c['stage']}` | {c['class2_states']} | {c['usable']} | "
                     f"{c['consistent']} | {seg} | {c['segment_share']} | "
                     f"{', '.join(f'{k}: {n}' for k, n in c['target_objects'].items()) or '-'} | "
                     f"{c['target_coverage']} | {c['median_segment_frames']} | {c.get('ordered_window_empty')} | "
                     f"{'yes' if c['segment_replay_ok'] else 'no'} | "
                     f"{'yes' if c['segment_replay_ok_original_rule'] else 'no'} |")
    lines += ["", "## Upstream instructions per stage", "",
              "Modal instruction sequence of the consistent segments (object names as `{object_key}`).", "",
              "| Task | Stage | Instructions | Share |", "|---|---|---|---:|"]
    for c in result["stages"]:
        text = " → ".join(f"\"{t}\"" for t in (c.get("instruction_template") or [])) or "-"
        lines.append(f"| {c['task_name']} | `{c['stage']}` | {text} | {c.get('instruction_share')} |")
    Path(path).write_text("\n".join(lines) + "\n")


# ---- CLI ----------------------------------------------------------------------------------------

def _overrides(items):
    out = {}
    for item in items or []:
        task, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--lerobot expects TASK=PATH, got {item!r}")
        out[task] = path
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("index", help="replay human demos (simulator, CPU)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--dataset-base", type=Path, help="RoboCasa dataset root (holds v1.0/target/composite)")
    p.add_argument("--lerobot", nargs="*", default=[], help="TASK=PATH overrides of the LeRobot folder")
    p.add_argument("--tasks", nargs="*", default=list(PILOT_TASKS))
    p.add_argument("--max-episodes", type=int, default=DEFAULT_MAX_EPISODES)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-errors", type=int, default=10)
    p = sub.add_parser("coverage", help="join with the R0 audit and decide R1 (no simulator)")
    p.add_argument("--index-dir", required=True, type=Path)
    p.add_argument("--audit-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--allow-provisional", action="store_true", help="use provisional R0 classes if decide has not run")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "index":
        if args.dataset_base is None and not args.lerobot:
            raise SystemExit("pass --dataset-base or --lerobot TASK=PATH")
        run_index(args.output_dir, tasks=args.tasks, dataset_base=args.dataset_base,
                  lerobot_overrides=_overrides(args.lerobot), max_episodes=args.max_episodes,
                  shard_index=args.shard_index, num_shards=args.num_shards, resume=args.resume,
                  max_errors=args.max_errors)
    else:
        result = coverage(args.index_dir, args.audit_dir, args.output_dir, allow_provisional=args.allow_provisional)
        print(json.dumps({k: result[k] for k in ("decision", "class2_states", "stuck_stages_failing")}, indent=2))


if __name__ == "__main__":
    main()
