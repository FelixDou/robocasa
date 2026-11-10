"""Simulator-free tests for R1: human-demo expert segments joined with the R0 audit."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()
from robocasa.recovery import expert_segments as es
from robocasa.recovery import recoverability_ladder as ladder


REQUIRED = ["kettle_on_tray", "mug_on_tray", "cabinet_closed", "gripper_released"]
PREDICATES = sorted(REQUIRED)
OBJECTS = ["obj", "obj2"]  # mug, kettle


def tea_demo(mug_first=False, close=True):
    """ArrangeTea-like demo: two pick/place subtasks, a close, then done (60 frames)."""
    first, second = (("PickPlaceCabinetToCounter", "mug", 0), ("PickPlaceCounterToCounter", "kettle", 1)) \
        if mug_first else (("PickPlaceCounterToCounter", "kettle", 1), ("PickPlaceCabinetToCounter", "mug", 0))
    plan = [(10, first[0], "pick"), (25, first[0], "place"), (30, second[0], "pick"), (45, second[0], "place"),
            (55, "CloseCabinet", "execute"), (60, "done", "done")]
    T = 60
    idx, names, stages, langs = (np.zeros(T, np.int32), np.empty(T, object), np.empty(T, object),
                                 np.empty(T, object))
    start = 0
    for i, (end, name, stage) in enumerate(plan):
        idx[start:end], names[start:end], stages[start:end], langs[start:end] = i, name, stage, f"{stage} {name}"
        start = end
    pos = np.zeros((T, 2, 3), np.float32)
    pos[:, 0, 0] = np.clip((np.arange(T) - (0 if mug_first else 25)) / 20, 0, 1) * 0.5  # mug moves
    pos[:, 1, 0] = np.clip((np.arange(T) - (25 if mug_first else 0)) / 20, 0, 1) * 0.5  # kettle moves
    values = np.zeros((T, len(PREDICATES)), bool)
    col = {n: i for i, n in enumerate(PREDICATES)}
    values[(40 if mug_first else 22):, col["kettle_on_tray"]] = True
    values[(22 if mug_first else 40):, col["mug_on_tray"]] = True
    if close:
        values[54:, col["cabinet_closed"]] = True
    values[57:, col["gripper_released"]] = True
    success = np.zeros(T, bool)
    success[57:] = close
    labels = {"subtask_idx": idx, "subtask_name": names, "subtask_stage": stages, "subtask_lang": langs}
    return labels, values, success, pos


def write_demo(index_dir, episode, *, mug_first=False, close=True, issues=()):
    labels, values, success, pos = tea_demo(mug_first, close)
    es.save_episode_arrays(Path(index_dir) / es.EPISODES_DIR / "ArrangeTea" / f"episode_{episode:06d}.npz",
                           labels=labels, predicate_names=PREDICATES, predicate_values=values,
                           task_success=success, required=REQUIRED, object_names=OBJECTS, object_pos=pos,
                           object_quat=np.zeros(pos.shape[:2] + (4,), np.float32),
                           eef_pos=np.zeros((pos.shape[0], 3), np.float32))
    row = es.summarize_episode("ArrangeTea", episode, labels=labels, predicate_names=PREDICATES,
                               predicate_values=values, task_success=success, object_names=OBJECTS,
                               object_pos=pos, issues=list(issues), review=[], parquet_rows=60,
                               parquet_idx=labels["subtask_idx"])
    ladder.append_jsonl(Path(index_dir) / "index.shard0.jsonl", row)
    return row


def write_audit(audit_dir, stuck, decided=True):
    audit_dir = Path(audit_dir)
    for i, (stage, cls) in enumerate(stuck):
        row = {"parent_id": f"p{i}", "task_name": "ArrangeTea", "required_subtasks": REQUIRED,
               "stage_at_alarm": stage, "provisional_class": cls}
        ladder.append_jsonl(audit_dir / "states.jsonl", row)
        if decided:
            ladder.append_jsonl(audit_dir / "r0_states.jsonl", {"parent_id": f"p{i}", "final_class": cls})


class HelperTest(unittest.TestCase):
    def test_segments_from_labels(self):
        labels = tea_demo()[0]
        segs = es.segments_from_labels(labels["subtask_idx"], labels["subtask_name"], labels["subtask_stage"],
                                       labels["subtask_lang"])
        self.assertEqual([(s["start"], s["end"]) for s in segs], [(0, 10), (10, 25), (25, 30), (30, 45),
                                                                    (45, 55), (55, 60)])
        self.assertEqual(segs[1]["stage"], "place")

    def test_first_stable_true(self):
        self.assertEqual(es.first_stable_true([0, 1, 0, 1, 1, 1, 1, 1, 0], stable=5), 3)
        self.assertIsNone(es.first_stable_true([0, 1, 1, 0, 0], stable=5))
        self.assertEqual(es.first_stable_true([0, 0, 0, 1, 1], stable=5), 3)  # true to the end
        self.assertIsNone(es.first_stable_true([], stable=5))

    def test_match_segment_and_group(self):
        segs = [{"index": i, "name": n, "stage": s, "lang": "", "start": a, "end": b} for i, (n, s, a, b) in
                enumerate([("OpenFridge", "execute", 0, 10), ("PickPlaceFridgeToCounter", "pick", 10, 20),
                           ("NavigateKitchen", "navigate", 20, 40), ("PickPlaceFridgeToCounter", "place", 40, 50),
                           ("done", "done", 50, 60)])]
        self.assertEqual(es.match_segment(segs, 45), 3)          # inside the place
        self.assertEqual(es.match_segment(segs, 22, tolerance=5), 1)  # just after the grasp boundary
        self.assertEqual(es.match_segment(segs, 55), 3)          # in done -> the place before it
        self.assertEqual(es.subtask_group(segs, 3), (1, 3))       # pick, navigation, place
        self.assertEqual(es.subtask_group(segs, 0), (0, 0))       # fixture subtask alone
        jam = [{"index": i, "name": n, "stage": s, "lang": "", "start": i * 10, "end": i * 10 + 10}
               for i, (n, s) in enumerate([("NavigateKitchen", "navigate"), ("PickPlaceCabinetToCounter", "pick"),
                                           ("NavigateKitchen", "navigate"), ("PickPlaceCabinetToCounter", "place")])]
        self.assertEqual(es.subtask_group(jam, 3), (1, 3))        # stops at the pick, not the drive before it

    def test_moved_object(self):
        pos = tea_demo()[3]
        self.assertEqual(es.moved_object(OBJECTS, pos, 0, 24)[0], "obj2")
        self.assertEqual(es.moved_object(OBJECTS, pos, 25, 44)[0], "obj")
        self.assertIsNone(es.moved_object(OBJECTS, pos, 50, 59)[0])

    def test_stage_events_follow_the_object_not_the_order(self):
        for mug_first in (False, True):
            labels, values, success, pos = tea_demo(mug_first)
            row = es.summarize_episode("ArrangeTea", 0, labels=labels, predicate_names=PREDICATES,
                                       predicate_values=values, task_success=success, object_names=OBJECTS,
                                       object_pos=pos, issues=[], review=[])
            mug = row["stage_events"]["mug_on_tray"]["segment"]
            self.assertEqual((mug["name"], mug["stage"], mug["moved_object"]),
                             ("PickPlaceCabinetToCounter", "place", "obj"))
            self.assertEqual(mug["stages"], [["PickPlaceCabinetToCounter", "pick"],
                                             ["PickPlaceCabinetToCounter", "place"]])
            close = row["stage_events"]["cabinet_closed"]["segment"]
            self.assertEqual((close["name"], close["moved_object"]), ("CloseCabinet", None))
            self.assertTrue(row["clean"])

    def test_ordered_window_is_empty_when_the_mug_goes_first(self):
        _, values, success, _ = tea_demo(mug_first=True)
        windows = es.ordered_windows(REQUIRED, PREDICATES, values, success)
        self.assertEqual(windows["kettle_on_tray"], (0, 40))
        self.assertEqual(windows["mug_on_tray"], (40, 40))
        _, values, success, _ = tea_demo(mug_first=False)
        self.assertEqual(es.ordered_windows(REQUIRED, PREDICATES, values, success)["mug_on_tray"], (22, 40))
        with self.assertRaises(ValueError):
            es.ordered_windows(REQUIRED + ["unknown"], PREDICATES, values, success)

    def test_replay_episode_calls_the_labeler_per_frame(self):
        class Env:
            state = None

        class Labeler:
            def setup(self, env, ep_meta):
                self.meta = ep_meta

            def per_step(self, env, t):
                return {"t": t, "state": env.state}

            def resolve(self, history):
                self.history = history
                return "result"

            def validate(self, result):
                return []

        def reset_fn(env, state):
            env.state = float(state["states"][0])
            if "model" in state:
                env.model = state["model"]

        labeler = Labeler()
        states = np.arange(4, dtype=float)[:, None]
        result, issues, objects, frames = es.replay_episode(
            Env(), labeler, states, {"a": 1}, "<xml/>", reset_fn=reset_fn,
            frame_fn=lambda env, names: {"state": env.state}, objects_fn=lambda env: ["obj"])
        self.assertEqual(result, "result")
        self.assertEqual([h["state"] for h in labeler.history], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual([f["state"] for f in frames], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(objects, ["obj"])


class CoverageTest(unittest.TestCase):
    def test_segment_replay_when_enough_consistent_demos(self):
        with tempfile.TemporaryDirectory() as tmp:
            index, audit, out = Path(tmp, "index"), Path(tmp, "audit"), Path(tmp, "r1")
            for ep in range(12):
                write_demo(index, ep, mug_first=ep % 4 == 0)
            write_demo(index, 12, close=False)                                     # failed demo: not usable
            write_demo(index, 13, issues=["expected 2 pick stages, found 1"])      # labeler flag: not usable
            write_audit(audit, [("mug_on_tray", "recoverable"), ("mug_on_tray", "recoverable"),
                                ("cabinet_closed", "recoverable"), ("kettle_on_tray", "recovered_by_resampling")])
            result = es.coverage(index, audit, out)
            self.assertEqual(result["decision"], "segment_replay")
            self.assertEqual(result["class2_states"], 3)
            stages = {c["stage"]: c for c in result["stages"]}
            self.assertEqual(stages["mug_on_tray"]["usable"], 12)
            self.assertEqual(stages["mug_on_tray"]["class2_states"], 2)
            self.assertEqual(stages["mug_on_tray"]["ordered_window_empty"], 3)
            self.assertEqual(stages["mug_on_tray"]["moved_object"], "obj")
            health = result["health"]["ArrangeTea"]
            self.assertEqual((health["demos"], health["clean"], health["labeler_flagged"]), (14, 12, 1))
            self.assertEqual(health["published_label_agreement_mean"], 1.0)
            lines = (out / "stage_segments.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 12 * 4)
            self.assertTrue((out / "r1_coverage.md").exists())

    def test_too_few_segments_fails_the_stuck_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            index, audit = Path(tmp, "index"), Path(tmp, "audit")
            for ep in range(5):
                write_demo(index, ep)
            write_audit(audit, [("mug_on_tray", "recoverable"), ("all_latched", "recoverable")])
            result = es.coverage(index, audit, Path(tmp, "r1"))
            self.assertEqual(result["decision"], "segment_replay_incomplete_check_scripted_expert")
            self.assertIn("ArrangeTea:mug_on_tray", result["stuck_stages_failing"])
            self.assertTrue(any(f.startswith("ArrangeTea:all_latched") for f in result["stuck_stages_failing"]))

    def test_requires_decided_classes_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            index, audit = Path(tmp, "index"), Path(tmp, "audit")
            write_demo(index, 0)
            write_audit(audit, [("mug_on_tray", "recoverable")], decided=False)
            with self.assertRaises(ValueError):
                es.coverage(index, audit, Path(tmp, "r1"))
            result = es.coverage(index, audit, Path(tmp, "r1"), allow_provisional=True)
            self.assertEqual(result["class_source"], "provisional")

    def test_duplicate_index_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp, "index")
            write_demo(index, 0)
            row = ladder.read_jsonl(index / "index.shard0.jsonl")[0]
            ladder.append_jsonl(index / "index.shard1.jsonl", row)
            with self.assertRaises(ValueError):
                es.load_index(index)


if __name__ == "__main__":
    unittest.main()
