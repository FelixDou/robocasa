"""Simulator-free tests for R2: source matching, the executor on a kinematic stand-in, and the gates."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()
from robocasa.recovery import expert_recovery as er  # noqa: E402
from robocasa.recovery import expert_segments as es  # noqa: E402
from tests.test_expert_segments import write_demo  # noqa: E402

R_DOWN = np.diag([1.0, -1.0, -1.0])     # gripper pointing down
PLACE_HEIGHT = 0.05


def world_pose(x, y, z, yaw=0.0):
    return er.pose([x, y, z], er.rot_z(yaw))


def quat_xyzw(R):
    """Rotation matrix -> xyzw quaternion (test helper)."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    q = np.zeros(4)
    q[i] = math.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) / 2
    q[3] = (R[k, j] - R[j, k]) / (4 * q[i])
    q[j] = (R[j, i] + R[i, j]) / (4 * q[i])
    q[k] = (R[k, i] + R[i, k]) / (4 * q[i])
    return q


TRUE_GAINS = er.Gains(
    arm_pos=er.rot_z(math.radians(10)) @ np.diag([0.03, 0.025, 0.02]),
    arm_rot=np.diag([0.2, 0.2, 0.25]),
    base=np.diag([0.03, 0.025, 0.08]),
)


class FakeWorld:
    """Kinematic stand-in for ``SimWorld``: actions move the arm and base exactly by the gains."""

    def __init__(self, *, mobile=(0.0, 0.0, 0.0), eef_rel=(0.5, 0.0, 0.9), objects=None, gains=TRUE_GAINS,
                 predicate=None, goals=None):
        self.gains = gains
        self.mobile = np.array(mobile, dtype=float)
        self.eef = self.base_T() @ er.pose(eef_rel, R_DOWN)
        self.objects = {k: v.copy() for k, v in (objects or {}).items()}
        self.attached, self.gripper = None, 0.0
        self.predicate = predicate or (lambda w: {})
        self.goals = goals or {}
        self.steps = 0

    def base_T(self):
        return world_pose(self.mobile[0], self.mobile[1], 0.3, self.mobile[2])

    def observe(self):
        return {"base": self.base_T(), "eef": self.eef.copy(), "mobile": tuple(float(v) for v in self.mobile),
                "objects": {k: v.copy() for k, v in self.objects.items()}}

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=float), -1, 1)
        self.steps += 1
        if a[11] >= 0.5:  # base mode: the arm rides along
            rel = er.inverse(self.base_T()) @ self.eef
            d = self.gains.base @ a[7:10]
            self.mobile[:2] += er.rot_z(self.mobile[2])[:2, :2] @ d[:2]
            self.mobile[2] = er.wrap(self.mobile[2] + d[2])
            self.eef = self.base_T() @ rel
        else:
            Rb = self.base_T()[:3, :3]
            self.eef[:3, 3] += Rb @ (self.gains.arm_pos @ a[0:3])
            self.eef[:3, :3] = Rb @ er.rot_from_vec(self.gains.arm_rot @ a[3:6]) @ Rb.T @ self.eef[:3, :3]
        self.gripper = a[6]
        if self.gripper >= 0.5 and self.attached is None:
            for name, T in self.objects.items():
                if name in ("container", "board") or np.linalg.norm(T[:3, 3] - self.eef[:3, 3]) >= 0.02:
                    continue
                self.attached = (name, er.inverse(self.eef) @ T)
                break
        if self.gripper < 0.5:
            self.attached = None
        if self.attached is not None:
            name, rel = self.attached
            self.objects[name] = self.eef @ rel
        predicates = self.predicate(self)
        return {"predicates": predicates, "task_success": bool(predicates.get("task_success"))}

    def navigation_goal(self, reference):
        return self.goals.get(tuple(reference))

    def holding(self):
        return self.attached is not None


def mug_on_tray(world):
    mug, tray = world.objects["obj"][:3, 3], world.objects["container"][:3, 3]
    on = world.attached is None and np.linalg.norm(mug - (tray + [0, 0, PLACE_HEIGHT])) < 0.03
    return {"mug_on_tray": bool(on)}


def demo_source(obj_T, tray_T, base_T, *, episode=0, predicate="mug_on_tray", target="obj"):
    """A recorded pick (frames 0-19) and place (20-43) in the demo's own scene."""
    yaw = er.yaw_of(obj_T[:3, :3])
    R = er.rot_z(yaw) @ R_DOWN
    p_obj, p_tray = obj_T[:3, 3], tray_T[:3, 3] + [0, 0, PLACE_HEIGHT]
    above_obj, above_tray = p_obj + [0, 0, 0.15], p_tray + [0, 0, 0.15]
    path = ([above_obj + (p_obj - above_obj) * s for s in np.linspace(0, 1, 10)]
            + [p_obj + (above_obj - p_obj) * s for s in np.linspace(0, 1, 10)]
            + [above_obj + (above_tray - above_obj) * s for s in np.linspace(0, 1, 12)]
            + [above_tray + (p_tray - above_tray) * s for s in np.linspace(0, 1, 6)]
            + [p_tray] * 6)
    gripper = np.array([0.0] * 10 + [1.0] * 28 + [0.0] * 6)
    n = len(path)
    eef = np.stack([er.pose(p, R) for p in path])
    return er.Source(task_name="ArrangeTea", episode=episode, predicate=predicate, target=target, start=0, end=n,
                     steps=[er.SourceStep("PickPlaceCabinetToCounter", er.PICK, 0, 20),
                            er.SourceStep("PickPlaceCabinetToCounter", er.PLACE, 20, n)],
                     eef=eef, base=np.stack([base_T] * n), gripper=gripper,
                     objects={"obj": np.stack([obj_T] * n), "container": np.stack([tray_T] * n)})


class FakeLibrary:
    def __init__(self, sources):
        self._sources = sources

    def sources(self, task_name, predicate, target=None):
        return [s for s in self._sources if s.task_name == task_name and s.predicate == predicate
                and (target is None or s.target == target)]


def facts_of(world, held=(), held_any=None, predicates=None, correct_tool=None):
    obs = world.observe()
    return {**obs, "held_task": list(held), "held_any": bool(held) if held_any is None else held_any,
            "held_bodies": [], "predicates": predicates or {}, "correct_tool": correct_tool}


class PoseMathTest(unittest.TestCase):
    def test_rotvec_round_trip_including_half_turns(self):
        rng = np.random.default_rng(0)
        for v in list(rng.normal(size=(20, 3))) + [np.array([math.pi, 0, 0]), np.array([0, -math.pi, 0]),
                                                   np.array([1, 1, 0]) / math.sqrt(2) * math.pi]:
            R = er.rot_from_vec(v)
            np.testing.assert_allclose(er.rot_from_vec(er.rotvec(R)), R, atol=1e-6)

    def test_quaternion_conventions(self):
        R = er.rot_z(math.pi / 2)
        np.testing.assert_allclose(er.quat_xyzw_to_mat(quat_xyzw(R)), R, atol=1e-9)
        x, y, z, w = quat_xyzw(R)
        np.testing.assert_allclose(er.quat_wxyz_to_mat([w, x, y, z]), R, atol=1e-9)

    def test_yaw_frame_drops_tilt(self):
        T = er.pose([1, 2, 3], er.rot_z(0.7) @ er.rot_from_vec([0.3, 0, 0]))
        F = er.yaw_frame(T)
        self.assertAlmostEqual(er.yaw_of(F[:3, :3]), er.yaw_of(T[:3, :3]))
        np.testing.assert_allclose(F[:3, 2], [0, 0, 1], atol=1e-12)

    def test_interpolation_respects_the_step_limits(self):
        a, b = world_pose(0, 0, 0), er.pose([0.1, 0, 0], er.rot_z(0.4))
        path = er.interpolate(a, b)
        self.assertEqual(len(path), 10)
        np.testing.assert_allclose(path[-1], b, atol=1e-9)
        for p, q in zip([a] + path[:-1], path):
            d, r = er.pose_error(p, q)
            self.assertLessEqual(d, er.APPROACH_POS_STEP + 1e-9)
            self.assertLessEqual(r, er.APPROACH_ROT_STEP + 1e-9)

    def test_gain_estimate_recovers_the_map(self):
        rng = np.random.default_rng(1)
        samples = [(a, TRUE_GAINS.arm_pos @ a) for a in rng.uniform(-0.5, 0.5, size=(30, 3))]
        M, r2 = er.estimate_gains(samples)
        np.testing.assert_allclose(M, TRUE_GAINS.arm_pos, atol=1e-9)
        self.assertGreater(r2, 0.999)


class SourceTest(unittest.TestCase):
    def test_build_source_composes_base_and_relative_poses(self):
        T = 30
        named = {
            "base_position": np.tile([1.0, 2.0, 0.3], (T, 1)),
            "base_rotation": np.tile(quat_xyzw(er.rot_z(math.pi / 2)), (T, 1)),
            "end_effector_position_relative": np.tile([0.5, 0.0, 0.6], (T, 1)),
            "end_effector_rotation_relative": np.tile([0, 0, 0, 1.0], (T, 1)),
            "gripper_qpos": np.zeros((T, 2)),
        }
        actions = np.zeros((T, 12))
        actions[15:, 6] = 1.0
        row = {"task_name": "ArrangeTea", "episode": 4, "object_lang": {"obj": "mug"},
               "segments": [{"index": 0, "name": "PickPlace", "stage": "pick", "lang": "pick up the mug",
                             "start": 5, "end": 15},
                            {"index": 1, "name": "NavigateKitchen", "stage": "navigate", "lang": "go", "start": 15,
                             "end": 20},
                            {"index": 2, "name": "PickPlace", "stage": "place", "lang": "place the mug on the tray",
                             "start": 20, "end": 28}],
               "stage_events": {"mug_on_tray": {"segment": {"start": 5, "end": 28, "lang": "place the mug",
                                                            "langs": ["pick up the mug", "place the mug on the tray"],
                                                            "moved_object": "obj"}}}}
        pos = np.zeros((T, 1, 3))
        quat = np.tile([1.0, 0, 0, 0], (T, 1, 1))
        src = er.build_source(row, "mug_on_tray", named=named, actions=actions, object_names=["obj"],
                              object_pos=pos, object_quat=quat)
        self.assertEqual(src.target, "obj")
        self.assertEqual([s.kind for s in src.steps], [er.PICK, er.NAVIGATE, er.PLACE])
        self.assertEqual((src.start, src.end, len(src.gripper)), (5, 28, 23))
        np.testing.assert_allclose(src.eef[0][:3, 3], [1.0, 2.5, 0.9], atol=1e-9)   # base x-axis is world y
        self.assertEqual(src.gripper[src.at(15)], 1.0)
        with self.assertRaises(IndexError):
            src.at(28)
        # A predicate settling after the segment end: the source runs on to first_true + STABLE_FRAMES.
        for first_true, end in ((22, 28), (24, 24 + es.STABLE_FRAMES), (27, T)):
            row["stage_events"]["mug_on_tray"]["first_true"] = first_true
            src = er.build_source(row, "mug_on_tray", named=named, actions=actions, object_names=["obj"],
                                  object_pos=pos, object_quat=quat)
            self.assertEqual((src.end, src.segment_end, src.steps[-1].end, len(src.gripper)), (end, 28, end, end - 5))

    def test_library_reads_the_index_and_filters_by_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ep in range(3):
                write_demo(tmp, ep, mug_first=ep == 0)
            write_demo(tmp, 3, close=False)   # not clean

            def reader(task, episode):
                named = {k: np.zeros((60, 4 if "rotation" in k else (2 if k == "gripper_qpos" else 3)))
                         for k in ("base_position", "base_rotation", "end_effector_position_relative",
                                   "end_effector_rotation_relative", "gripper_qpos")}
                named["base_rotation"][:, 3] = named["end_effector_rotation_relative"][:, 3] = 1.0
                return named, np.zeros((60, 12))

            lib = er.SourceLibrary(tmp, {"ArrangeTea": tmp}, reader=reader)
            mugs = lib.sources("ArrangeTea", "mug_on_tray")
            self.assertEqual([s.episode for s in mugs], [0, 1, 2])
            self.assertEqual({s.target for s in mugs}, {"obj"})
            self.assertEqual(lib.sources("ArrangeTea", "mug_on_tray", target="obj2"), [])
            self.assertIs(lib.sources("ArrangeTea", "mug_on_tray")[0], mugs[0])   # cached

    def test_source_table_rows_keep_the_predicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            index, audit, out = Path(tmp, "index"), Path(tmp, "audit"), Path(tmp, "r1")
            for ep in range(2):
                write_demo(index, ep)
            ladder_rows = [{"parent_id": "p0", "task_name": "ArrangeTea",
                            "required_subtasks": ["kettle_on_tray", "mug_on_tray", "cabinet_closed",
                                                  "gripper_released"],
                            "stage_at_alarm": "mug_on_tray", "provisional_class": "recoverable"}]
            audit.mkdir()
            (audit / "states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ladder_rows))
            (audit / "r0_states.jsonl").write_text(json.dumps({"parent_id": "p0", "final_class": "recoverable"}) + "\n")
            es.coverage(index, audit, out, min_segments=1)
            rows = [json.loads(line) for line in (out / "stage_segments.jsonl").read_text().splitlines()]
        self.assertEqual({r["stage"] for r in rows},
                         {"kettle_on_tray", "mug_on_tray", "cabinet_closed", "gripper_released"})
        self.assertTrue({r["upstream_stage"] for r in rows} <= {"pick", "place", "execute", "done"})


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.base = world_pose(0, 0, 0.3)
        self.near = [demo_source(world_pose(0.6, 0.1 * i, 0.9), world_pose(0.6, -0.4, 0.9), self.base, episode=i)
                     for i in range(3)]
        self.lib = FakeLibrary(self.near)

    def world(self, obj=(0.6, 0.0, 0.9), tray=(0.6, -0.4, 0.9), mobile=(0.0, 0.0, 0.0)):
        return FakeWorld(mobile=mobile, objects={"obj": world_pose(*obj), "container": world_pose(*tray)},
                         predicate=mug_on_tray)

    def test_pick_then_place_ranked_by_relative_pose(self):
        programs = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(self.world(obj=(0.6, 0.19, 0.9))),
                                    self.lib, n=4)
        self.assertEqual([p["source"][1] for p in programs], [2, 1, 0, 2])   # closest first, then cycled
        self.assertEqual([ph["kind"] for ph in programs[0]["phases"]], [er.PICK, er.PLACE])
        self.assertEqual(programs[0]["phases"][0]["reference"], ["object", "obj"])
        self.assertEqual(programs[0]["phases"][1]["reference"], ["object", "container"])

    def test_held_target_starts_at_the_place_and_other_objects_are_released(self):
        held = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(self.world(), held=["obj"]), self.lib, n=1)
        self.assertEqual([ph["phase"] for ph in held[0]["phases"]], ["track"])
        self.assertEqual(held[0]["phases"][0]["kind"], er.PLACE)
        ham = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(self.world(), held_any=True), self.lib, n=1)
        self.assertEqual(ham[0]["phases"][0], {"phase": "release"})

    def test_far_targets_are_driven_to_first(self):
        programs = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(self.world(mobile=(-2.0, 1.0, 1.0))),
                                    self.lib, n=1)
        self.assertGreater(programs[0]["rank_distance"], er.NAV_TRIGGER)
        self.assertEqual(programs[0]["phases"][0], {"phase": "navigate", "reference": ["object", "obj"],
                                                    "if_needed": False})

    def test_release_stage_and_fixture_stage(self):
        rel = er.plan_attempts("ArrangeTea", "gripper_released", facts_of(self.world()), self.lib, n=2)
        self.assertEqual([p["phases"] for p in rel], [[{"phase": "release"}]] * 2)
        src = self.near[0]
        phases = er.stage_phases(src, er.STAGE_SPECS[("ArrangeTea", "cabinet_closed")], None, 0)
        self.assertEqual(phases[0], {"phase": "navigate", "reference": ["fixture", "cab"], "if_needed": True})
        self.assertTrue(all(ph["reference"] == ["door", "cab"] for ph in phases[1:]))
        own = er.stage_phases(src, er.STAGE_SPECS[("ArrangeTea", "cabinet_closed")], None, 0,
                              navigate_to_fixture=False)
        self.assertEqual([ph["phase"] for ph in own], ["track", "track"])

    def test_wrong_tool_on_the_board_is_set_aside_first(self):
        base = world_pose(0, 0, 0.3)
        tools = [demo_source(world_pose(0.6, 0, 0.9), world_pose(0.6, -0.3, 0.9), base, episode=i,
                             predicate="correct_tool_on_cutting_board", target=t) for i, t in enumerate(("knife", "peeler"))]
        for s in tools:
            s.task_name = "CuttingToolSelection"
            s.objects = {s.target: s.objects["obj"], "food_container": s.objects["container"]}
        lib = FakeLibrary(tools)
        world = FakeWorld(objects={"knife": world_pose(0.6, 0, 0.9), "peeler": world_pose(0.6, 0.2, 0.9),
                                   "food_container": world_pose(0.6, -0.3, 0.9)})
        facts = facts_of(world, predicates={"wrong_tool_not_on_cutting_board": False}, correct_tool="knife")
        program = er.plan_attempts("CuttingToolSelection", "correct_tool_on_cutting_board", facts, lib, n=1)[0]
        self.assertEqual(program["target"], "knife")
        self.assertTrue(program["wrong_tool_on_board"])
        kinds = [(ph["phase"], ph.get("kind"), (ph.get("reference") or [None, None])[1]) for ph in program["phases"]]
        self.assertEqual(kinds, [("track", er.PICK, "peeler"), ("set_aside", None, None),
                                 ("track", er.PICK, "knife"), ("track", er.PLACE, "food_container")])
        held = er.plan_attempts("CuttingToolSelection", "correct_tool_on_cutting_board",
                                facts_of(world, held=["knife"], predicates={"wrong_tool_not_on_cutting_board": False},
                                         correct_tool="knife"), lib, n=1)[0]
        self.assertEqual([ph["phase"] for ph in held["phases"]], ["track", "release", "track", "set_aside"])


class ExecutorTest(unittest.TestCase):
    def run_program(self, world, program, sources, predicate="mug_on_tray", cap=10 ** 5):
        run = er.ExpertRun(world, TRUE_GAINS, {s.key: s for s in sources}, predicate, cap=cap, budget=10 ** 5)
        return run.run(program["phases"])

    def test_pick_and_place_transfer_to_another_scene(self):
        demo = demo_source(world_pose(0.55, 0.05, 0.9, yaw=0.0), world_pose(0.6, -0.4, 0.9), world_pose(0, 0, 0.3))
        world = FakeWorld(mobile=(1.0, 2.0, 0.6), eef_rel=(0.4, 0.2, 0.8),
                          objects={"obj": er.yaw_frame(er.pose([0, 0, 0], np.eye(3))) @ world_pose(0, 0, 0)},
                          predicate=mug_on_tray)
        base = world.base_T()
        world.objects = {"obj": base @ world_pose(0.6, 0.15, 0.6, yaw=1.0), "container": base @ world_pose(0.5, -0.3, 0.6)}
        program = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(world), FakeLibrary([demo]), n=1)[0]
        result = self.run_program(world, program, [demo])
        self.assertTrue(result["stage_completed"], result)
        self.assertEqual(result["stop"], "stage")
        self.assertEqual([p["kind"] for p in result["phases"]], [er.PICK, er.PLACE])
        self.assertEqual(result["phases"][1]["stopped"], "stage")            # stopped inside the place step
        self.assertTrue(result["phases"][0]["align"]["reached"], result["phases"][0]["align"])

    def test_navigation_then_pick_and_place(self):
        demo = demo_source(world_pose(0.6, 0.0, 0.9), world_pose(0.6, -0.4, 0.9), world_pose(0, 0, 0.3))
        obj, tray = world_pose(3.6, 1.0, 0.9), world_pose(3.6, 0.6, 0.9)
        world = FakeWorld(mobile=(0.0, 0.0, 2.5), objects={"obj": obj, "container": tray}, predicate=mug_on_tray,
                          goals={("object", "obj"): (3.0, 1.0, 0.0)})
        program = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(world), FakeLibrary([demo]), n=1)[0]
        self.assertEqual(program["phases"][0]["phase"], "navigate")
        result = self.run_program(world, program, [demo])
        self.assertTrue(result["phases"][0]["reached"], result["phases"][0])
        self.assertGreater(result["navigation_steps"], 0)
        self.assertTrue(result["stage_completed"], result)

    def test_the_cap_stops_the_expert(self):
        demo = demo_source(world_pose(0.6, 0.0, 0.9), world_pose(0.6, -0.4, 0.9), world_pose(0, 0, 0.3))
        world = FakeWorld(objects={"obj": world_pose(0.6, 0.0, 0.9), "container": world_pose(0.6, -0.4, 0.9)},
                          predicate=mug_on_tray)
        program = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(world), FakeLibrary([demo]), n=1)[0]
        result = self.run_program(world, program, [demo], cap=15)
        self.assertEqual((result["stop"], result["stage_completed"], result["expert_steps"]), ("cap", False, 15))
        self.assertEqual(er.program_cap(program, {demo.key: demo}), math.ceil(44 * 1.5))

    def test_release_and_set_aside(self):
        world = FakeWorld(objects={"peeler": world_pose(0.5, 0.0, 1.2), "board": world_pose(0.5, -0.1, 0.9)})
        world.objects["peeler"] = world.eef.copy()
        world.step(er.action_vector(gripper=er.GRIPPER_CLOSE))
        self.assertEqual(world.attached[0], "peeler")
        run = er.ExpertRun(world, TRUE_GAINS, {}, None, cap=10 ** 5, budget=10 ** 5)
        before = world.eef[:3, 3].copy()
        result = run.run([{"phase": "set_aside", "object": "peeler", "board": "board"}])
        self.assertEqual(result["stop"], "program_end")
        self.assertIsNone(world.attached)
        moved = world.objects["peeler"][:3, 3] - before
        self.assertAlmostEqual(moved[1], er.SET_ASIDE_SIDE, delta=0.03)      # away from the board (+y here)
        self.assertAlmostEqual(moved[2], er.SET_ASIDE_LIFT - er.SET_ASIDE_LOWER, delta=0.03)

    def test_base_motion_inside_a_step_is_replayed_on_the_base(self):
        """The demonstrator drives 0.3 m toward the mug during the pick; the scene is the demo moved rigidly."""
        G = world_pose(2.0, 1.0, 0.0, yaw=0.8)
        base0 = world_pose(0, 0, 0.3)
        drive = [world_pose(0.3 * s, 0, 0.3) for s in np.linspace(0, 1, 15)]
        mug, tray = world_pose(0.9, 0.0, 0.9), world_pose(0.9, -0.4, 0.9)
        rest = demo_source(mug, tray, drive[-1])
        rel0 = er.inverse(drive[-1]) @ rest.eef[0]
        n = 15 + len(rest.gripper)
        src = er.Source(task_name="ArrangeTea", episode=7, predicate="mug_on_tray", target="obj", start=0, end=n,
                        steps=[er.SourceStep("PickPlace", er.PICK, 0, 35), er.SourceStep("PickPlace", er.PLACE, 35, n)],
                        eef=np.concatenate([np.stack([b @ rel0 for b in drive]), rest.eef]),
                        base=np.concatenate([np.stack(drive), rest.base]),
                        gripper=np.concatenate([np.zeros(15), rest.gripper]),
                        objects={"obj": np.stack([mug] * n), "container": np.stack([tray] * n)},
                        mode=np.concatenate([np.ones(15), np.zeros(len(rest.gripper))]))
        b = G @ base0
        world = FakeWorld(mobile=(b[0, 3], b[1, 3], er.yaw_of(b[:3, :3])), predicate=mug_on_tray)
        world.eef = G @ src.eef[0]
        world.objects = {"obj": G @ mug, "container": G @ tray}
        program = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(world), FakeLibrary([src]), n=1)[0]
        self.assertEqual(program["phases"][0]["phase"], "track")       # relative pose matches: no navigation
        result = self.run_program(world, program, [src])
        self.assertTrue(result["stage_completed"], result)
        self.assertGreaterEqual(result["phases"][0]["base_mode_steps"], 14)
        np.testing.assert_allclose(world.base_T()[:2, 3], (G @ drive[-1])[:2, 3], atol=0.03)

    def test_slide_to_faces_the_object(self):
        goal = er.slide_to((0.0, 0.0, 0.3), fixture_xy=(1.0, 0.0), target_xy=(1.0, 0.7))
        np.testing.assert_allclose(goal, (0.0, 0.7, 0.3), atol=1e-9)    # same standoff, shifted along the front

    def test_torso_follows_only_with_a_measured_gain(self):
        g = er.Gains(TRUE_GAINS.arm_pos, TRUE_GAINS.arm_rot, TRUE_GAINS.base, torso=0.01, torso_mode=er.ARM_MODE)
        self.assertEqual(g.torso_command(0.05, er.ARM_MODE), 1.0)
        self.assertAlmostEqual(g.torso_command(-0.015, er.ARM_MODE), -1.0)        # clipped
        self.assertAlmostEqual(er.Gains(g.arm_pos, g.arm_rot, g.base, torso=0.02).torso_command(0.015, er.ARM_MODE), 0.75)
        self.assertEqual(g.torso_command(0.005, er.ARM_MODE), 0.0)          # within tolerance
        self.assertEqual(g.torso_command(0.05, er.BASE_MODE), 0.0)          # other mode
        self.assertEqual(TRUE_GAINS.torso_command(0.05, er.ARM_MODE), 0.0)   # no torso found
        self.assertEqual(er.Gains.from_json(g.to_json()).torso, 0.01)

    def test_navigation_converges_with_commands_in_the_reset_heading(self):
        goal = (2.0, -1.5, -2.8)

        def drive(compensate):
            world = StaleControllerWorld(mobile=(0.0, 0.0, 2.5))
            world.base_ref_yaw = 0.0      # the episode reset faced +x; the base has since turned 2.5 rad
            for _ in range(er.NAVIGATE_MAX_STEPS):
                obs = world.observe()
                if er.navigation_done(obs["mobile"], goal, er.NAV_FINE_POS, math.cos(er.NAV_FINE_YAW)):
                    break
                frame = obs["command_frame_yaw"] if compensate else 0.0
                world.step(er.base_command(obs["mobile"], goal, TRUE_GAINS, frame_yaw=frame))
            return er.navigation_done(world.observe()["mobile"], goal)

        self.assertTrue(drive(True))
        self.assertFalse(drive(False))

    def test_navigation_keeps_a_held_object(self):
        world = FakeWorld(objects={"obj": world_pose(0.5, 0.0, 1.2), "container": world_pose(2.0, 1.0, 0.9)},
                          goals={("object", "container"): (1.5, 1.0, 0.0)})
        world.objects["obj"] = world.eef.copy()
        world.step(er.action_vector(gripper=er.GRIPPER_CLOSE))
        self.assertEqual(world.attached[0], "obj")
        run = er.ExpertRun(world, TRUE_GAINS, {}, None, cap=10 ** 5, budget=10 ** 5)
        result = run.run([{"phase": "navigate", "reference": ["object", "container"], "if_needed": False}])
        self.assertTrue(result["phases"][0]["reached"], result)
        self.assertIsNotNone(world.attached)                 # still held after driving
        self.assertLess(np.linalg.norm(world.objects["obj"][:3, 3] - world.eef[:3, 3]), 1e-9)

    def test_navigation_controller_converges(self):
        world = FakeWorld(mobile=(0.0, 0.0, 0.0))
        goal = (2.0, -1.5, -2.8)
        for _ in range(er.NAVIGATE_MAX_STEPS):
            if er.navigation_done(world.observe()["mobile"], goal, er.NAV_FINE_POS, math.cos(er.NAV_FINE_YAW)):
                break
            world.step(er.base_command(world.observe()["mobile"], goal, TRUE_GAINS))
        self.assertTrue(er.navigation_done(world.observe()["mobile"], goal))


class DiagnosticsTest(unittest.TestCase):
    def test_the_tail_catches_a_predicate_that_turns_true_just_after_the_program(self):
        for turns_true, expected in ((3, "stage"), (er.TAIL_STEPS - er.HOLD_STEPS + 2, "program_end")):
            world = FakeWorld(predicate=lambda w, k=turns_true: {"late": w.steps >= k})
            result = er.ExpertRun(world, TRUE_GAINS, {}, "late", cap=10 ** 5, budget=10 ** 5).run([])
            self.assertEqual(result["stop"], expected, turns_true)
        self.assertEqual(result["expert_steps"], er.TAIL_STEPS)

    def test_tracking_reports_the_lag_the_close_and_the_grasp(self):
        demo = demo_source(world_pose(0.55, 0.05, 0.9), world_pose(0.6, -0.4, 0.9), world_pose(0, 0, 0.3))
        world = FakeWorld(objects={"obj": world_pose(0.55, 0.05, 0.9), "container": world_pose(0.6, -0.4, 0.9)},
                          predicate=mug_on_tray)
        program = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(world), FakeLibrary([demo]), n=1)[0]
        result = ExecutorTest.run_program(None, world, program, [demo])
        pick = result["phases"][0]
        self.assertLess(pick["pos_lag_max"], er.TRACK_TOL)
        self.assertEqual([c["frame"] for c in pick["closes"]], [10])
        self.assertLess(pick["closes"][0]["pos"], 0.02)
        self.assertTrue(pick["holding_after"])

    def test_a_grasp_check_returning_the_bodies_is_read_as_a_bool(self):
        for out, gripper in (((False, []), er.GRIPPER_OPEN), ((True, ["mug"]), er.GRIPPER_CLOSE)):
            world = FakeWorld()
            world.holding = lambda out=out: out
            self.assertEqual(er.ExpertRun(world, TRUE_GAINS, {}, None, cap=1, budget=1).gripper, gripper)
            self.assertEqual(er.world_holding(world), out[0])

    def test_action_replay_sends_the_demo_actions_then_holds_still(self):
        n = 10
        actions = np.zeros((n, 12))
        actions[:, 0] = 1.0
        actions[:, 6] = -1.0
        src = er.Source(task_name="ArrangeTea", episode=0, predicate="moved", target=None, start=0, end=n,
                        steps=[], eef=np.stack([np.eye(4)] * n), base=np.stack([np.eye(4)] * n),
                        gripper=actions[:, 6].copy(), actions=actions)
        world = FakeWorld()
        x0 = world.eef[0, 3]
        moved = lambda w: {"moved": w.eef[0, 3] - x0 >= n * TRUE_GAINS.arm_pos[0, 0] - 1e-9}  # noqa: E731
        world.predicate = moved
        out = er.replay_actions(world, src, "moved")
        self.assertEqual((out["stage_completed"], out["stop"], out["steps"]), (True, "stage", n + er.HOLD_STEPS - 1))
        self.assertAlmostEqual(world.eef[0, 3] - x0, n * TRUE_GAINS.arm_pos[0, 0])
        world = FakeWorld(predicate=lambda w: {})
        out = er.replay_actions(world, src, "moved")
        self.assertEqual((out["stage_completed"], out["stop"], out["steps"]), (False, "end", n + er.TAIL_STEPS))


class SelfReplayLoopTest(unittest.TestCase):
    """The self-replay loop end to end, with the simulator parts replaced by the kinematic world."""

    def test_executor_control_and_videos_run_without_errors(self):
        from unittest import mock

        obj, tray, base = world_pose(0.55, 0.05, 0.9), world_pose(0.6, -0.4, 0.9), world_pose(0, 0, 0.3)
        demos = []
        for episode in (3, 4):
            src = demo_source(obj, tray, base, episode=episode)
            src.actions = np.zeros((src.end - src.start, 12))
            src.first_true = src.end - 2
            demos.append(src)

        class Library:
            lerobots = {"ArrangeTea": Path("unused")}

            def __init__(self, *args, **kwargs):
                pass

            def sources(self, task, predicate, target=None):
                return demos if (task, predicate) == ("ArrangeTea", "mug_on_tray") else []

            def named_state(self, task, episode, frame):
                return {}

        class World(FakeWorld):
            def __init__(self, env, recorder=None, *, reset=True):
                super().__init__(objects={"obj": obj.copy(), "container": tray.copy()}, predicate=mug_on_tray)
                self.obs, self.recorder = {}, recorder

            def facts(self, task):
                return facts_of(self)

        class Recorder:
            made = []

            def __init__(self, folder):
                self.folder = Path(folder)
                Recorder.made.append(self.folder.name)

            def finish(self, meta):
                self.folder.mkdir(parents=True)

        env = mock.MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(er, "load_gains", return_value=TRUE_GAINS), \
                mock.patch.object(er, "SourceLibrary", Library), \
                mock.patch.object(er, "_lerobots", return_value={}), \
                mock.patch.object(er, "_make_env", return_value=env), \
                mock.patch.object(er.es, "load_episode", return_value=(np.zeros((50, 3)), {}, "<mujoco/>")), \
                mock.patch.object(er, "SimWorld", World), \
                mock.patch.object(er, "AttemptRecorder", Recorder), \
                mock.patch.object(er, "convention_residual", return_value=0.0), \
                mock.patch.object(er.ladder, "_close"):
            lines = []
            er.run_self_replay(Path(tmp), Path(tmp) / "out", Path(tmp) / "gains.json", tasks=["ArrangeTea"],
                               stages=["ArrangeTea:mug_on_tray"], action_replay=True, videos=1, log=lines.append)
            rows = er.ladder.read_jsonl(Path(tmp) / "out" / "self_replay.jsonl")
        self.assertEqual([r["episode"] for r in rows], [3, 4])
        for r in rows:
            self.assertNotIn("error", r, r.get("error"))
            self.assertIn(r["stop"], ("stage", "cap", "program_end"))
            self.assertIn(r["action_replay"]["stop"], ("stage", "end"))
            self.assertIn("holding_after", r["phases"][0])
            self.assertEqual(r["first_true_after_end"], -2)
        self.assertEqual(Recorder.made, ["episode_000003", "episode_000003_actions"])   # first demo only
        self.assertIn("recording", rows[0])
        self.assertNotIn("recording", rows[1])


class BlockedWorld(FakeWorld):
    """The base stops where its collision box would overlap an island's footprint.

    FakeWorld turns the base about its own origin, so the box's turning axis is the origin; the
    box sits behind it, as on the Omron base.
    """

    SHAPE = er.bp.BaseShape(half=(0.35, 0.25), centre=(-0.20, 0.0), pivot=(0.0, 0.0))

    def __init__(self, island, planner=True, **kwargs):
        super().__init__(**kwargs)
        self.island, self.blocked = island, 0
        if planner:
            self.obstacles = lambda: [("table", self.island)]
            self.base_shape = lambda: self.SHAPE

    def step(self, a):
        mobile, eef = self.mobile.copy(), self.eef.copy()
        info = super().step(a)
        if er.bp.polygons_overlap(self.SHAPE.corners(self.mobile), self.island):
            self.mobile, self.eef = mobile, eef
            self.blocked += 1
        return info


class PlannedNavigationTest(unittest.TestCase):
    ISLAND = er.bp.footprint(np.array([[x, y, z] for x in (1.5, 2.5) for y in (-1.5, 1.5) for z in (0, 0.9)]))

    def drive(self, planner):
        world = BlockedWorld(self.ISLAND, planner=planner, mobile=(0.0, 0.0, 0.0),
                             goals={("fixture", "table"): (4.0, 0.0, math.pi)})
        run = er.ExpertRun(world, TRUE_GAINS, {}, None, cap=10 ** 5, budget=10 ** 5)
        return world, run.run([{"phase": "navigate", "reference": ["fixture", "table"], "if_needed": False}])

    def test_a_straight_line_pins_against_the_island_and_the_planned_path_goes_around(self):
        world, result = self.drive(planner=False)
        nav = result["phases"][0]
        self.assertFalse(nav["reached"])
        self.assertEqual(nav["plan"], "straight")
        self.assertGreater(world.blocked, 100)
        world, result = self.drive(planner=True)
        nav = result["phases"][0]
        self.assertTrue(nav["reached"], {k: v for k, v in nav.items() if k != "trace_every_10"})
        self.assertEqual(nav["plan"], "planned")
        self.assertEqual(world.blocked, 0)
        self.assertGreater(nav["planned_length"], nav["straight_length"])
        self.assertLessEqual(nav["steps"], nav["max_steps"])

    def test_a_footprint_around_the_start_is_dropped(self):
        room = er.bp.footprint(np.array([[x, y, z] for x in (-5, 5) for y in (-5, 5) for z in (0, 3)]))
        follower, log = er.plan_navigation([("room", room), ("table", self.ISLAND)], (0, 0, 0), (4, 0, 0))
        self.assertEqual((log["plan"], log["dropped_obstacles"], log["obstacles"]), ("planned", ["room"], 1))
        follower, log = er.plan_navigation([("room", room)], (0, 0, 0), (4, 0, 0))
        self.assertIsNone(follower)
        self.assertEqual(log["plan"], "straight")


class MisreportedFrameWorld(FakeWorld):
    """Body-frame base commands (as FakeWorld) but a frame model claiming the reset heading."""

    def observe(self):
        obs = super().observe()
        obs["command_frame_yaw"] = float(self.mobile[2])
        return obs


class BaseFrameEstimatorTest(unittest.TestCase):
    def drive(self, gain):
        world = MisreportedFrameWorld(mobile=(0.0, 0.0, 0.0), goals={("fixture", "table"): (2.0, -1.5, -math.pi)})
        run = er.ExpertRun(world, TRUE_GAINS, {}, None, cap=10 ** 5, budget=10 ** 5)
        run.frame = er.BaseFrameEstimator(gain=gain)
        return run.run([{"phase": "navigate", "reference": ["fixture", "table"], "if_needed": False}])["phases"][0]

    def test_a_wrong_frame_model_is_corrected_while_driving(self):
        nav = self.drive(er.FRAME_GAIN)
        self.assertTrue(nav["reached"], {k: v for k, v in nav.items() if k != "trace_every_10"})
        self.assertLess(abs(er.wrap(nav["frame_offset"] - math.pi)), 0.1)   # the full heading change, measured
        self.assertFalse(self.drive(0.0)["reached"])                       # the model alone drives off

    def test_a_right_model_stays_put(self):
        world = FakeWorld(mobile=(0.0, 0.0, 0.0), goals={("fixture", "table"): (2.0, -1.5, -math.pi)})
        run = er.ExpertRun(world, TRUE_GAINS, {}, None, cap=10 ** 5, budget=10 ** 5)
        nav = run.run([{"phase": "navigate", "reference": ["fixture", "table"], "if_needed": False}])["phases"][0]
        self.assertTrue(nav["reached"])
        self.assertLess(abs(nav["frame_offset"]), 1e-6)

    def test_blocked_or_still_steps_are_not_measured(self):
        est = er.BaseFrameEstimator()
        obs = {"mobile": (0.0, 0.0, 0.5), "command_frame_yaw": 0.5}
        a = er.action_vector(base=[1.0, 0.0, 0.0], mode=er.BASE_MODE)
        self.assertIsNone(est.update(obs, a, (0.0, 0.0, 0.5), TRUE_GAINS))
        self.assertIsNone(est.update(obs, a, (0.03, 0.0, 0.5), TRUE_GAINS, touching=True))
        self.assertAlmostEqual(est.update(obs, a, (0.0, 0.03, 0.5), TRUE_GAINS), math.pi / 2)


class TaskSuccessTest(unittest.TestCase):
    def test_a_last_stage_succeeding_with_the_task_counts(self):
        for holds, completed in ((True, True), (False, False)):
            world = FakeWorld(predicate=lambda w, h=holds: {"cabinet_closed": h and w.steps >= 2,
                                                            "task_success": w.steps >= 2})
            result = er.ExpertRun(world, TRUE_GAINS, {}, "cabinet_closed", cap=10 ** 5, budget=10 ** 5).run([])
            self.assertEqual((result["stop"], result["stage_completed"]), ("task_success", completed))


class FixturePlaceTest(unittest.TestCase):
    """A place on a fixture lands where the base stands: the drive picks a free part of the top."""

    # A 2.4 x 0.8 m island with a 0.5 x 0.4 m sink at its middle; the robot faces it from y < 0.
    SURFACE = er.bp.convex_hull([[-1.2, 0.0], [1.2, 0.0], [1.2, 0.8], [-1.2, 0.8]])
    SINK = er.bp.convex_hull([[-0.25, 0.1], [0.25, 0.1], [0.25, 0.5], [-0.25, 0.5]])
    GOAL = (0.0, -0.2, math.pi / 2)   # RoboCasa's centred placement pose
    OFFSET = (0.5, 0.0)               # the demo set the basket down 0.5 m ahead of its base

    def choose(self, **kw):
        return er.choose_place_goal(self.GOAL, self.OFFSET, self.SURFACE, **kw)

    def test_a_free_centre_is_kept(self):
        goal, log = self.choose()
        self.assertEqual(goal, self.GOAL)
        self.assertEqual(log["place_shift"], 0.0)
        np.testing.assert_allclose(log["place_point"], [0.0, 0.3], atol=1e-9)

    def test_the_place_moves_off_the_sink_by_the_least_slide(self):
        goal, log = self.choose(others=[("sink", self.SINK)])
        point = np.array(log["place_point"])
        self.assertGreaterEqual(er.bp.polygon_distance(point[None], self.SINK)[0], er.PLACE_FIXTURE_MARGIN)
        self.assertLessEqual(abs(log["place_shift"]), 0.25 + er.PLACE_FIXTURE_MARGIN + er.PLACE_SLIDE_STEP)
        self.assertAlmostEqual(goal[1], self.GOAL[1])          # slid along the front, still facing it
        self.assertEqual(goal[2], self.GOAL[2])
        self.assertIn("sink", log["place_rejected"])

    def test_objects_and_blocked_base_positions_push_it_to_the_other_side(self):
        # The robot faces +y, so its left is -x. Block the left: an object on the top, a fixture by the base.
        left_object = [("bowl", np.array([-0.45, 0.3]))]
        goal, log = self.choose(others=[("sink", self.SINK)], objects=left_object)
        self.assertGreater(log["place_point"][0], 0.25)
        stool = er.bp.convex_hull([[0.2, -0.9], [0.9, -0.9], [0.9, -0.5], [0.2, -0.5]])
        goal, log = self.choose(others=[("sink", self.SINK)], obstacles=[("stool", stool)])
        self.assertLess(log["place_point"][0], -0.25)
        self.assertFalse(er.bp.polygons_overlap(er.bp.BaseShape().corners(goal), stool))

    def test_no_free_spot_keeps_the_centred_goal_and_says_so(self):
        goal, log = er.choose_place_goal(self.GOAL, (2.0, 0.0), self.SURFACE)   # always past the far edge
        self.assertEqual(goal, self.GOAL)
        self.assertIsNone(log["place_shift"])
        self.assertGreater(log["place_rejected"]["edge"], 10)

    def test_the_offset_is_where_the_demo_left_the_object(self):
        base = world_pose(1.0, 2.0, 0.3, math.pi / 2)
        src = demo_source(world_pose(1.0, 2.6, 0.9), world_pose(0.7, 2.5, 0.9), base)
        src.objects["obj"][-1] = world_pose(0.7, 2.5, 0.9 + PLACE_HEIGHT)
        np.testing.assert_allclose(er.place_offset(src, 1, "obj"), (0.5, 0.3), atol=1e-9)
        np.testing.assert_allclose(er.place_offset(src, 1), (0.5, 0.3), atol=1e-9)   # the hand, there too

    def place_source(self):
        src = demo_source(world_pose(0.6, 0, 0.9), world_pose(0.6, -0.3, 0.9), world_pose(0, 0, 0.3),
                          predicate="basket_on_dining_counter", target="basket")
        src.task_name = "ArrangeBreadBasket"
        src.objects["basket"] = src.objects.pop("obj")
        src.steps = [src.steps[0], er.SourceStep("NavigateKitchen", er.NAVIGATE, 20, 20), src.steps[1]]
        return src

    def test_the_navigation_before_a_fixture_place_names_the_place_step(self):
        src = self.place_source()
        spec = er.STAGE_SPECS[("ArrangeBreadBasket", "basket_on_dining_counter")]
        phases = er.stage_phases(src, spec, "basket", 0, navigate_to_fixture=False)
        nav = phases[1]
        self.assertEqual(nav["reference"], ["fixture", "dining_table"])
        self.assertEqual(nav["place"], {"source": list(src.key), "step": 2, "object": "basket"})
        pick_nav = er.stage_phases(src, er.STAGE_SPECS[("ArrangeBreadBasket", "bread_in_basket")], "bread", 0)
        self.assertTrue(all("place" not in ph for ph in pick_nav))   # an object reference: no change

    def test_the_executor_drives_to_the_world_chosen_place_goal(self):
        src = self.place_source()
        calls = []

        class PlaceWorld(FakeWorld):
            def place_goal(self, reference, offset, carried=None):
                calls.append((reference, offset, carried))
                return (0.5, 0.0, 0.0), {"place_shift": 0.5}

        world = PlaceWorld(mobile=(0.0, 0.0, 0.0), goals={("fixture", "dining_table"): (0.0, 0.0, 0.0)})
        run = er.ExpertRun(world, TRUE_GAINS, {src.key: src}, None, cap=10 ** 5, budget=10 ** 5)
        place = {"source": list(src.key), "step": 2, "object": "basket"}
        result = run.run([{"phase": "navigate", "reference": ["fixture", "dining_table"], "if_needed": True,
                           "place": place}])
        nav = result["phases"][0]
        self.assertEqual(calls[0][0], ("fixture", "dining_table"))
        np.testing.assert_allclose(calls[0][1], er.place_offset(src, 2, "basket"))
        self.assertEqual(calls[0][2], "basket")
        self.assertTrue(nav["reached"])
        self.assertEqual(nav["place_shift"], 0.5)
        np.testing.assert_allclose(world.mobile[:2], [0.5, 0.0], atol=0.05)


class ShortArmWorld(FakeWorld):
    """The arm reaches only ``share`` of each commanded motion."""

    def __init__(self, share=0.7, **kw):
        super().__init__(**kw)
        self.share = share

    def step(self, a):
        a = np.asarray(a, dtype=float).copy()
        if a[11] < 0.5:
            a[0:6] *= self.share
        return super().step(a)


class FeedforwardTrackingTest(unittest.TestCase):
    """Tracking sends the demo's own arm actions (carried to here) plus a share of the error."""

    def recorded_demo(self, n=40):
        world = FakeWorld(mobile=(0.0, 0.0, 0.0))
        t = np.linspace(0, 1, n)
        actions = np.zeros((n, 12))
        actions[:, 0] = 0.3 * np.sin(3 * t)
        actions[:, 1] = 0.4 * np.cos(2 * t)
        actions[:, 2] = -0.2
        actions[:, 3] = 0.1 * np.sin(4 * t)
        actions[:, 5] = 0.15
        actions[:, 6] = (t > 0.5).astype(float)
        actions[:, 11] = -1.0
        eef, base = [], []
        for a in actions:
            eef.append(world.eef.copy())
            base.append(world.base_T())
            world.step(a)
        return er.Source(task_name="ArrangeTea", episode=0, predicate="moved", target=None, start=0, end=n,
                         steps=[er.SourceStep("PickPlace", er.PICK, 0, n)], eef=np.stack(eef), base=np.stack(base),
                         gripper=actions[:, 6].copy(), actions=actions)

    def run_track(self, world, src):
        sent = []
        step = world.step
        world.step = lambda a: (sent.append(np.asarray(a, dtype=float).copy()), step(a))[1]
        run = er.ExpertRun(world, TRUE_GAINS, {src.key: src}, None, cap=10 ** 5, budget=10 ** 5)
        out = run.run([{"phase": "track", "source": list(src.key), "step": 0, "kind": er.PICK,
                        "reference": ["base", None]}])
        return np.array(sent), out["phases"][0]

    def test_self_replay_sends_the_demo_actions(self):
        src = self.recorded_demo()
        sent, phase = self.run_track(FakeWorld(mobile=(0.0, 0.0, 0.0)), src)
        self.assertEqual(len(sent), len(src.actions) + er.TAIL_STEPS)       # then the tail holds still
        np.testing.assert_allclose(sent[:len(src.actions), :7], src.actions[:, :7], atol=1e-9)
        self.assertEqual((phase["catchup_steps"], phase["stalled_frames"]), (0, 0))

    def test_another_base_pose_replays_the_motion_in_its_own_frame(self):
        src = self.recorded_demo()
        world = FakeWorld(mobile=(1.0, 2.0, 0.7))
        sent, phase = self.run_track(world, src)
        self.assertLess(phase["pos_lag_max"], 1e-9)
        self.assertLess(phase["rot_err_max"], 1e-9)
        n = len(src.actions)
        np.testing.assert_allclose(np.linalg.norm(sent[:n, 0:3], axis=1), np.linalg.norm(src.actions[:, 0:3], axis=1),
                                   rtol=0.25)   # same motion, other axes: the gains are anisotropic

    def test_a_lagging_arm_is_corrected_then_caught_up_without_resending_the_demo_action(self):
        src = self.recorded_demo()
        n = len(src.actions)
        for share, catches_up in ((0.7, False), (0.3, True)):
            sent, phase = self.run_track(ShortArmWorld(share=share, mobile=(0.0, 0.0, 0.0)), src)
            self.assertEqual(phase["catchup_steps"] > 0, catches_up, share)
            self.assertLessEqual(phase["pos_lag_max"], er.TRACK_TOL)   # logged to 4 decimals
            self.assertLess(np.abs(np.diff(sent[:n, 0:3], axis=0)).max(), 1.0, share)   # no full reversals


class GateTest(unittest.TestCase):
    def test_gate_r_counts_states_with_a_full_success(self):
        states = [{"parent_id": f"p{i}", "task_name": "ArrangeTea", "stage": "mug_on_tray"} for i in range(4)]
        attempts = ([{"parent_id": "p0", "k": k, "stage_completed": True, "success": k == 3} for k in range(10)]
                    + [{"parent_id": "p1", "k": k, "stage_completed": True, "success": False} for k in range(10)]
                    + [{"parent_id": "p2", "k": k, "stage_completed": k < 5, "success": k < 2} for k in range(10)])
        result = er.gate_r(attempts, states)
        self.assertEqual((result["states_with_success"], result["share"], result["passed"]), (2, 0.5, True))
        self.assertEqual(result["expert_rate"], round(25 / 30, 4))
        self.assertEqual(result["full_rate"], round(3 / 30, 4))
        self.assertEqual(result["states"][3]["attempts"], 0)    # never run: counts as no success
        self.assertFalse(er.gate_r(attempts[:20], states)["passed"])

    def test_executor_check_summaries(self):
        rows = [{"task_name": "ArrangeTea", "predicate": "mug_on_tray", "stage_completed": i < 18,
                 "convention_residual": 1e-5} for i in range(20)]
        rows += [{"task_name": "ArrangeTea", "predicate": "kettle_on_tray", "stage_completed": True}] * 5
        s = er.self_replay_summary(rows, expected=[("ArrangeTea", "mug_on_tray")])
        self.assertEqual(s["failing"], ["ArrangeTea:kettle_on_tray"])           # too few demos
        every = er.self_replay_summary(rows)                                     # every ArrangeTea stage
        self.assertEqual(every["failing"], ["ArrangeTea:cabinet_closed", "ArrangeTea:kettle_on_tray"])
        self.assertEqual(next(x for x in every["stages"] if x["predicate"] == "cabinet_closed")["demos"], 0)
        for share, passed in ((1.0, True), (0.97, False)):          # true at every demo's start: nothing to replay
            marked = rows + [{"task_name": "ArrangeTea", "predicate": "cabinet_closed", "episode": None,
                              "no_sources": True, "stage_completed": False, "true_at_start_share": share}]
            row = next(x for x in er.self_replay_summary(marked)["stages"] if x["predicate"] == "cabinet_closed")
            self.assertEqual(row["passed"], passed, share)
        index = [{"task_name": "T", "clean": True, "stage_events": {"p": {"true_at_start": i < 3}}} for i in range(4)]
        index.append({"task_name": "T", "clean": False, "stage_events": {"p": {"true_at_start": False}}})
        self.assertEqual(er.true_at_start_share(index, "T", "p"), 0.75)
        self.assertTrue(next(x for x in s["stages"] if x["predicate"] == "mug_on_tray")["passed"])
        runs = [{"task_name": "ArrangeTea", "stages": [{"predicate": "kettle_on_tray", "stage_completed": True},
                                                       {"predicate": "mug_on_tray", "stage_completed": i < 6}]}
                for i in range(10)]
        t = er.transfer_summary(runs)
        self.assertEqual(t["failing"], ["ArrangeTea:mug_on_tray"])               # 6 of 10 < 0.7
        self.assertEqual(er.transfer_stages("ArrangeTea", ["kettle_on_tray", "mug_on_tray", "cabinet_closed",
                                                           "gripper_released"]),
                         ["kettle_on_tray", "mug_on_tray", "cabinet_closed"])
        self.assertEqual(er.transfer_stages("CuttingToolSelection", ["drawer_open", "correct_tool_on_cutting_board",
                                                                     "wrong_tool_not_on_cutting_board"]),
                         ["correct_tool_on_cutting_board"])         # the reset opens the drawer
        self.assertNotIn(("CuttingToolSelection", "drawer_open"), er.self_replay_stages(["CuttingToolSelection"]))

    def test_cli_parser(self):
        args = er.build_parser().parse_args(["gate-r", "--index-dir", "i", "--gains", "g.json", "--output-dir", "o",
                                             "--source-runs", "a", "b", "--alarms", "al.jsonl", "--audit-dir", "r0",
                                             "--dataset-base", "d"])
        self.assertEqual(args.attempts, er.ATTEMPTS_PER_STATE)
        self.assertEqual(len(args.source_runs), 2)

    def test_load_gains_refuses_a_failed_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            path.write_text(json.dumps({"ok": False, "gains": TRUE_GAINS.to_json()}))
            with self.assertRaises(ValueError):
                er.load_gains(path)
            path.write_text(json.dumps({"ok": True, "gains": TRUE_GAINS.to_json()}))
            np.testing.assert_allclose(er.load_gains(path).arm_pos, TRUE_GAINS.arm_pos)


class ArmFollowWorld(FakeWorld):
    """In base mode the hand is left at its world pose, then closes ``catch_up`` of its base-frame offset per step."""

    def __init__(self, catch_up, **kw):
        super().__init__(**kw)
        self.catch_up = catch_up
        self.hold = er.inverse(self.base_T()) @ self.eef   # the arm controller's goal, in the base frame

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=float), -1, 1)
        if a[11] < 0.5:
            out = super().step(a)
            self.hold = er.inverse(self.base_T()) @ self.eef
            return out
        self.steps += 1
        world_eef = self.eef.copy()
        d = self.gains.base @ a[7:10]
        self.mobile[:2] += er.rot_z(self.mobile[2])[:2, :2] @ d[:2]
        self.mobile[2] = er.wrap(self.mobile[2] + d[2])
        rel = er.inverse(self.base_T()) @ world_eef
        rel[:3, 3] += self.catch_up * (self.hold[:3, 3] - rel[:3, 3])
        rel[:3, :3] = self.hold[:3, :3]
        self.eef = self.base_T() @ rel
        return {"predicates": {}, "task_success": False}


class SaggingWorld(ArmFollowWorld):
    """ArmFollowWorld whose hand sinks after the restore, as an arm controller holding its achieved pose would.

    ``settle`` < 1: the sag shrinks geometrically and levels off at ``depth``; ``settle`` = 1: it sinks by
    ``depth`` every step, without end.
    """

    def __init__(self, depth, settle, catch_up=0.25, **kw):
        super().__init__(catch_up, **kw)
        self.depth, self.settle, self.t = depth, settle, 0

    def step(self, a):
        out = super().step(a)
        dz = self.depth if self.settle == 1 else self.depth * (1 - self.settle) * self.settle ** self.t
        self.t += 1
        self.eef[2, 3] -= dz
        self.hold[2, 3] -= dz
        return out


class ProbeTest(unittest.TestCase):
    def probe(self, world_cls, **kw):
        world = {}

        def reset():
            world["w"] = world_cls(**kw)

        reset()
        proxy = type("Proxy", (), {"observe": lambda _s: world["w"].observe(),
                                   "step": lambda _s, a: world["w"].step(a)})()
        return er.probe_world(proxy, reset)

    def test_recovers_the_gains_of_a_rigid_arm(self):
        result = self.probe(FakeWorld)
        self.assertTrue(result["ok"])
        self.assertEqual(set(result["checks"]), {"fit", "conditioning", "arm_moves_base", "base_moves_arm",
                                                 "drift", "base_frame"})
        gains = er.Gains.from_json(result["gains"])
        np.testing.assert_allclose(gains.arm_pos, TRUE_GAINS.arm_pos, atol=1e-9)
        np.testing.assert_allclose(gains.base, TRUE_GAINS.base, atol=1e-9)
        self.assertLess(result["mode_leaks"]["base_moves_arm"], 1e-9)
        self.assertEqual(len(result["base_runs"]), 3 * len(er.PROBE_AMPLITUDES) * 2)

    def test_an_arm_that_lags_but_returns_passes(self):
        result = self.probe(ArmFollowWorld, catch_up=0.25)
        leaks = result["mode_leaks"]
        self.assertGreater(leaks["base_moves_arm"], er.ARM_HOLD_TOL)   # what failed the first probe
        self.assertLess(leaks["base_moves_arm_settled"], er.ARM_HOLD_TOL)
        self.assertTrue(result["ok"])

    def test_an_arm_that_holds_its_world_pose_fails(self):
        result = self.probe(ArmFollowWorld, catch_up=0.0)
        run = max(result["base_runs"], key=lambda r: r["travel"])
        self.assertGreater(run["settled"], 0.5 * run["travel"])
        self.assertFalse(result["ok"])

    def test_a_sag_that_levels_off_is_compared_against_the_zero_motion_run(self):
        # The second probe: base runs and the zero-motion run both ended about 7 mm off.
        result = self.probe(SaggingWorld, depth=0.0073, settle=0.85)
        drift = result["zero_action_drift"]
        self.assertGreater(drift["base"]["max"], er.ARM_HOLD_TOL)
        self.assertLess(drift["base"]["max"], er.DRIFT_TOL)
        self.assertEqual(len(drift["base"]["every_10_steps"]), er.DRIFT_STEPS // 10)
        self.assertLess(result["mode_leaks"]["base_moves_arm_settled"], er.ARM_HOLD_TOL)
        self.assertTrue(all(r["offset"] > er.ARM_HOLD_TOL for r in result["base_runs"]))
        self.assertTrue(result["ok"])

    def test_base_commands_in_the_reset_heading_pass_when_the_world_reports_that_heading(self):
        # StaleControllerWorld takes base commands in the frame of its heading at construction.
        result = self.probe(StaleControllerWorld)
        turn = result["base_frame_after_turn"]
        self.assertEqual(turn["start_unblocked"], len(er.TURN_DIRECTIONS))
        for row in turn["turns"]:
            self.assertAlmostEqual(row["median_heading_change"], -row["median_turned"], places=6)
            self.assertAlmostEqual(row["median_model_residual"], 0.0, places=6)
        self.assertTrue(result["checks"]["base_frame"])    # (this fake world ignores rotation commands)
        np.testing.assert_allclose(er.Gains.from_json(result["gains"]).base, TRUE_GAINS.base, atol=1e-9)

    def test_base_commands_in_an_unreported_frame_fail(self):
        class Unreported(StaleControllerWorld):    # the third probe's finding, without the frame reading
            def observe(self):
                obs = super().observe()
                del obs["command_frame_yaw"]
                return obs

        result = self.probe(Unreported)
        self.assertFalse(result["checks"]["base_frame"])
        self.assertFalse(result["ok"])

    def test_a_rigid_world_keeps_its_base_frame_after_turning(self):
        turn = self.probe(FakeWorld)["base_frame_after_turn"]
        for row in turn["turns"]:
            self.assertEqual(row["unblocked"], len(er.TURN_DIRECTIONS))
            self.assertAlmostEqual(row["median_heading_change"], 0.0, places=6)

    def test_a_sag_that_keeps_growing_fails(self):
        result = self.probe(SaggingWorld, depth=0.0003, settle=1)   # 9 cm over a navigation
        drift = result["zero_action_drift"]
        self.assertGreater(drift["arm"]["final"], er.DRIFT_TOL)
        self.assertLess(result["mode_leaks"]["base_moves_arm_settled"], er.ARM_HOLD_TOL)
        self.assertFalse(result["ok"])


class StaleControllerWorld(FakeWorld):
    """A world whose controllers keep state across ``restore()``, which only resets the simulator.

    The arm controller tracks a goal in the base frame (moving half the way each step) that arm
    commands shift and a restore leaves alone, as robosuite's arm does in base mode; the base takes
    commands in the frame of its heading at construction (the episode reset), which nothing resets.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rel = (er.inverse(self.base_T()) @ self.eef)[:3, 3].copy()
        self.goal, self.base_ref_yaw = self.rel.copy(), self.mobile[2]
        self.saved = (self.mobile.copy(), self.rel.copy())

    def restore(self):
        self.mobile, self.rel = self.saved[0].copy(), self.saved[1].copy()
        self._place()

    def reset_controllers(self):
        self.goal = self.rel.copy()

    def observe(self):
        obs = super().observe()
        obs["command_frame_yaw"] = er.wrap(self.mobile[2] - self.base_ref_yaw)
        return obs

    def _place(self):
        self.eef = self.base_T() @ er.pose(self.rel, R_DOWN)

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=float), -1, 1)
        self.steps += 1
        if a[11] >= 0.5:
            d = self.gains.base @ a[7:10]
            self.mobile[:2] += er.rot_z(self.base_ref_yaw)[:2, :2] @ d[:2]
            self.mobile[2] = er.wrap(self.mobile[2] + d[2])
        else:
            self.goal = self.goal + self.gains.arm_pos @ a[0:3]
        self.rel = self.rel + 0.5 * (self.goal - self.rel)
        self._place()
        return {"predicates": {}, "task_success": False}


def rigid_proxy():
    """A FakeWorld rebuilt on every restore (nothing carries over), behind a fixed handle."""
    world = {}

    def restore():
        world["w"] = FakeWorld()

    restore()
    proxy = type("Proxy", (), {"observe": lambda _s: world["w"].observe(),
                               "step": lambda _s, a: world["w"].step(a)})()
    return proxy, restore


class RestoreProbeTest(unittest.TestCase):
    def setUp(self):
        self.world = StaleControllerWorld()

    def trials(self):
        w = self.world

        def reset():
            w.restore()
            w.reset_controllers()

        rows = er.restore_trials(w, {"plain": w.restore, "reset": reset})
        return {(r["perturbation"], r["variant"], r["hold_mode"]): r for r in rows}

    def test_a_stale_arm_goal_shows_after_an_arm_move_and_the_reset_removes_it(self):
        rows = self.trials()
        self.assertEqual(len(rows), len(er.RESTORE_PERTURBATIONS) * 2 * 2)
        self.assertLess(rows[("none", "plain", "arm")]["final"], 1e-9)
        stale = rows[("arm_x", "plain", "arm")]
        self.assertGreater(stale["final"], 0.05)       # pulled back toward the previous run's goal
        self.assertLess(rows[("arm_x", "reset", "arm")]["final"], 1e-9)
        self.assertLess(rows[("arm_x", "reset", "base")]["final"], 1e-9)

    def test_base_commands_in_a_frame_fixed_at_reset_turn_against_the_base(self):
        out = er.base_frame_after_turn(self.world, self.world.restore)
        self.assertEqual([round(r["angle"], 3) for r in out["turns"]], list(er.TURN_ANGLES))
        for row in out["turns"]:
            self.assertGreater(abs(row["median_turned"]), 0.45)
            self.assertAlmostEqual(row["median_heading_change"], -row["median_turned"], places=6)
            self.assertFalse(any(d["blocked"] for d in row["directions"].values()))

    def test_a_rigid_world_shows_nothing(self):
        proxy, restore = rigid_proxy()
        rows = er.restore_trials(proxy, {"plain": restore})
        self.assertTrue(all(r["final"] < 1e-9 for r in rows))
        out = er.base_frame_after_turn(proxy, restore)
        for row in out["turns"]:
            self.assertAlmostEqual(row["median_heading_change"], 0.0, places=6)

    def test_a_blocked_command_is_left_out_of_the_median(self):
        real_step = FakeWorld.step

        class Wall(FakeWorld):
            def step(self, a):       # after any turn, a wall stops motion to the left
                if a[11] >= 0.5 and a[8] > 0 and abs(self.mobile[2]) > 0.1:
                    a = np.array(a, dtype=float)
                    a[8] = 0.0
                return real_step(self, a)

        world = {}

        def restore_wall():
            world["w"] = Wall()

        restore_wall()
        wall = type("Proxy", (), {"observe": lambda _s: world["w"].observe(),
                                  "step": lambda _s, a: world["w"].step(a)})()
        out = er.base_frame_after_turn(wall, restore_wall)
        for row in out["turns"]:
            self.assertTrue(row["directions"]["left"]["blocked"])
            self.assertAlmostEqual(row["median_heading_change"], 0.0, places=6)


class ContactWorld(FakeWorld):
    """A wall touches the base once it has turned by more than 0.1 rad; motion itself is unaffected."""

    def base_in_contact(self):
        return abs(self.mobile[2]) > 0.1


class ContactTest(unittest.TestCase):
    def test_contact_marks_a_command_blocked_even_when_it_moves_freely(self):
        world = {}

        def restore():
            world["w"] = ContactWorld()

        restore()
        proxy = type("Proxy", (), {"observe": lambda _s: world["w"].observe(),
                                   "step": lambda _s, a: world["w"].step(a),
                                   "base_in_contact": lambda _s: world["w"].base_in_contact()})()
        out = er.base_frame_after_turn(proxy, restore)
        self.assertEqual(out["start_unblocked"], len(er.TURN_DIRECTIONS))
        for row in out["turns"]:
            self.assertEqual(row["unblocked"], 0)
            self.assertIsNone(row["median_heading_change"])
            self.assertTrue(all(d["contact"] for d in row["directions"].values()))


class FakePart:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)
        self.updated = False

    def update(self, force=False):
        self.updated = force


class ResetControllersTest(unittest.TestCase):
    def test_goals_follow_the_current_state_in_each_controller_s_own_frame(self):
        R0 = er.rot_z(0.7)
        arm = FakePart(input_ref_frame="base", origin_pos=np.array([1.0, 2.0, 0.7]), origin_ori=R0,
                       ref_pos=np.array([1.3, 2.2, 1.2]), ref_ori_mat=R0 @ R_DOWN,
                       goal_pos=np.array([9.0, 9.0, 9.0]), goal_ori=np.eye(3))
        torso = FakePart(goal_qpos=np.array([0.2]), joint_pos=np.array([0.05]))
        init = er.rot_z(1.2)
        base = FakePart(goal_qvel=np.array([0.5, 0.0, 0.1]), init_ori=init, init_pos=np.zeros(3))
        gripper = FakePart(goal_qvel=np.array([1.0, -1.0]))
        cc = type("CC", (), {"arms": ["right"], "update_state": lambda _s: None,
                             "part_controllers": {"right": arm, "torso": torso, "base": base,
                                                  "right_gripper": gripper}})()
        kitchen = type("K", (), {"robots": [type("R", (), {"composite_controller": cc})()]})()
        original = er._kitchen
        er._kitchen = lambda env: kitchen
        try:
            done = er.reset_controllers(None)
        finally:
            er._kitchen = original
        self.assertEqual(set(done), {"right", "torso", "base"})
        np.testing.assert_allclose(arm.origin_pos + R0 @ arm.goal_pos, arm.ref_pos)   # base frame
        np.testing.assert_allclose(R0 @ arm.goal_ori, arm.ref_ori_mat)
        np.testing.assert_allclose(torso.goal_qpos, [0.05])
        np.testing.assert_allclose(base.goal_qvel, 0.0)
        self.assertIs(base.init_ori, init)                  # the heading reference is kept
        np.testing.assert_allclose(gripper.goal_qvel, [1.0, -1.0])
        self.assertTrue(arm.updated and torso.updated and base.updated)


class NpyWriter:
    """Stand-in for an imageio video writer: the frames, exactly, as a .npy file at the .mp4 path."""

    def __init__(self, path, fps):
        self.path, self.frames = Path(path), []

    def append_data(self, frame):
        self.frames.append(np.asarray(frame).copy())

    def close(self):
        with self.path.open("wb") as stream:
            np.save(stream, np.stack(self.frames))


def read_npy(path):
    with Path(path).open("rb") as stream:
        return np.load(stream)


def fake_obs(t, rng):
    from robocasa.recovery.xiaomi_robotics_1_policy import CAMERA_KEYS

    obs = {key: rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8) for key in CAMERA_KEYS}
    obs.update({"state.base_position": np.array([t, 0.0, 0.3]), "state.base_rotation": np.array([0, 0, 0, 1.0]),
                "state.end_effector_position_relative": np.array([0.5, 0.0, t / 10]),
                "state.end_effector_rotation_relative": np.array([1.0, 0, 0, 0]),
                "state.gripper_qpos": np.array([0.04, -0.04])})
    return obs


class RecorderTest(unittest.TestCase):
    def test_executed_actions_use_the_demo_convention(self):
        vec = er.executed_action(er.action_vector(arm_pos=[0.1, 0, 0], gripper=er.GRIPPER_CLOSE, mode=er.ARM_MODE))
        self.assertEqual((vec[6], vec[11], vec[0]), (1.0, -1.0, 0.1))
        policy = {"action.end_effector_position": [0, 0, 0.2], "action.end_effector_rotation": [0, 0, 0],
                  "action.gripper_close": [0.3], "action.base_motion": [0, 0, 0, 0], "action.control_mode": [0.9]}
        vec = er.executed_action(policy)
        self.assertEqual((vec[2], vec[6], vec[11]), (0.2, -1.0, 1.0))
        with self.assertRaises(ValueError):
            er.executed_action(np.zeros(7))

    def record(self, folder, n_expert, n_policy, seed=0):
        rng = np.random.default_rng(seed)
        rec = er.AttemptRecorder(folder, writer_factory=NpyWriter)
        observations, actions = [], []
        for t in range(n_expert + n_policy):
            obs, action = fake_obs(t, rng), rng.uniform(-1, 1, size=12)
            rec.add(obs, action, "expert" if t < n_expert else "policy")
            observations.append(obs)
            actions.append(er.executed_action(action))
        return rec, observations, np.stack(actions)

    def test_recorder_writes_and_renames(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "recordings" / "p0" / "k00"
            rec, _, _ = self.record(folder, 3, 2)
            self.assertTrue(rec.partial.exists() and not folder.exists())
            self.assertEqual(rec.finish({"instruction": "make tea"}), 5)
            meta = json.loads((folder / er.ATTEMPT_META).read_text())
            self.assertEqual((meta["frames"], meta["frames_by_phase"]), (5, {"expert": 3, "policy": 2}))
            with self.assertRaises(FileExistsError):
                er.AttemptRecorder(folder, writer_factory=NpyWriter)
            other = Path(tmp) / "recordings" / "p0" / "k01"
            rec, _, _ = self.record(other, 1, 0)
            rec.abort()
            self.assertFalse(rec.partial.exists() or other.exists())

    def test_the_continuation_records_what_the_policy_saw(self):
        rng = np.random.default_rng(3)

        class Env:
            def __init__(self):
                self.t = 0

            def step(self, action):
                self.t += 1
                return fake_obs(self.t, rng), float(self.t == 4), False, False, {"success": self.t == 4}

        class Policy:
            def __init__(self):
                self.seen = []

            def __call__(self, obs, instruction=None):
                self.seen.append(obs["state.base_position"][0])
                return {"action.end_effector_position": np.zeros(3), "action.end_effector_rotation": np.zeros(3),
                        "action.gripper_close": np.array([1.0]), "action.base_motion": np.zeros(4),
                        "action.control_mode": np.array([0.0])}

        class World:
            obs = fake_obs(0, rng)

        with tempfile.TemporaryDirectory() as tmp:
            rec = er.AttemptRecorder(Path(tmp) / "a", writer_factory=NpyWriter)
            policy = Policy()
            out = er.continue_with_policy(Env(), policy, World(), instruction="go", budget_left=10, seed=1,
                                          required=[], recorder=rec)
            self.assertEqual(out, {"policy_steps": 4, "success": True})
            self.assertEqual(len(rec), 4)
            self.assertEqual(rec.phases, [1] * 4)
            self.assertEqual([s[0] for s in rec.states["base_position"]], policy.seen)   # frame t = obs before action t
            self.assertEqual(rec.actions[0][6], 1.0)

    def test_export_round_trips_through_the_step0_reader(self):
        from robocasa.recovery import xr1_samples as xs
        from robocasa.recovery.xiaomi_robotics_1_policy import CAMERA_KEYS

        with tempfile.TemporaryDirectory() as tmp:
            gate = Path(tmp) / "gate_r" / "shard0"
            recorded = {}
            rows = []
            for k, success in enumerate((True, False, True)):
                rel = Path("recordings") / "p0" / f"k{k:02d}"
                rec, observations, actions = self.record(gate / rel, 4, 3 + k, seed=k)
                rec.finish({"instruction": "arrange the tea", "parent_id": "p0", "k": k})
                recorded[k] = (observations, actions)
                rows.append({"parent_id": "p0", "task_name": "ArrangeTea", "k": k, "success": success,
                             "recording": str(rel), "stage": "mug_on_tray"})
            rows.append({"parent_id": "p1", "task_name": "ArrangeTea", "k": 0, "success": True,
                         "error": "Traceback ..."})
            (gate / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = Path(tmp) / "export"
            summary = er.export_lerobot(gate.parent, out)
            self.assertEqual(summary["ArrangeTea"]["episodes"], 2)            # successes only, no errors
            root = out / "ArrangeTea" / "lerobot"
            self.assertEqual(xs.lerobot_episodes(root), [0, 1])
            for ep, k in ((0, 0), (1, 2)):
                observations, actions = recorded[k]
                episode = xs.read_lerobot_episode(root, ep, "ArrangeTea", decode=read_npy)
                self.assertEqual(episode.instruction, "arrange the tea")
                np.testing.assert_allclose(episode.actions, actions, atol=1e-6)
                for key in xs.STATE_KEYS:
                    np.testing.assert_allclose(episode.named[key], np.stack([o[f"state.{key}"] for o in observations]),
                                               atol=1e-6)
                for key in CAMERA_KEYS:
                    np.testing.assert_array_equal(episode.frames[key], np.stack([o[key] for o in observations]))
                named, acts = xs.read_lerobot_arrays(root, ep)
                np.testing.assert_allclose(acts, actions, atol=1e-6)
                sample = xs.sample_arrays(episode, 0, chunk=4)
                self.assertEqual(sample[2].shape, (1, 4, 60))
            import pyarrow.parquet as pq

            phase = pq.read_table(root / "data/chunk-000/episode_000001.parquet").column("annotation.r2.phase")
            self.assertEqual(phase.to_pylist(), [0] * 4 + [1] * 5)
            self.assertEqual(len(er.recorded_attempts(gate.parent, include="all")), 3)
            with self.assertRaises(FileExistsError):
                er.export_lerobot(gate.parent, out)


if __name__ == "__main__":
    unittest.main()


class DoorWorld(FakeWorld):
    """FakeWorld with one measured door on the fixture ``cab``."""

    def __init__(self, door, **kw):
        super().__init__(**kw)
        self.door = door

    def doors(self, fixture):
        return [self.door] if fixture == "cab" else []


def door_demo(door_frame, base_T, *, episode=0):
    """A demo that reaches the handle (frames 0-14), closes, and pulls 0.2 m out (15-29)."""
    R = door_frame[:3, :3] @ er.rot_z(math.pi) @ R_DOWN       # facing the door
    handle = door_frame[:3, 3]
    out = door_frame[:3, 0]
    start = handle + 0.25 * out + np.array([0, 0, 0.1])
    path = ([start + (handle - start) * s for s in np.linspace(0, 1, 15)]
            + [handle + 0.2 * out * s for s in np.linspace(0, 1, 15)])
    n = len(path)
    door = {"role": "door_hinge", "type": "hinge", "frame": door_frame, "q": 0.0, "open_sign": 1.0,
            "range": [0.0, 1.6], "side": 1, "radius": 0.4, "height": float(handle[2])}
    return er.Source(task_name="ArrangeTea", episode=episode, predicate="cabinet_closed", target=None, start=0,
                     end=n, steps=[er.SourceStep("OpenCabinet", er.EXECUTE, 0, n)],
                     eef=np.stack([er.pose(p, R) for p in path]), base=np.stack([base_T] * n),
                     gripper=np.array([0.0] * 15 + [1.0] * 15), door=door)


class AlignmentTest(unittest.TestCase):
    def test_align_goal_puts_the_base_where_the_source_stood(self):
        src_base = world_pose(0.2, -0.1, 0.3, yaw=0.4)
        transform = world_pose(1.0, 2.0, 0.0, yaw=0.7)
        world = FakeWorld(mobile=(-1.0, 0.5, 2.0))
        obs = world.observe()
        goal = er.align_goal(transform, src_base, obs["base"], obs["mobile"])
        world.mobile = np.array(goal)
        np.testing.assert_allclose(er.yaw_frame(world.base_T()), er.yaw_frame(transform @ src_base), atol=1e-9)

    def test_door_reference_is_read_at_the_closed_door_on_either_hinge_side(self):
        a1, a2 = np.array([0.0, 0.0, 0.0]), np.array([0.8, 0.0, 0.0])
        h1 = a1 + er.rot_z(0.5) @ np.array([0.4, 0.0, 0.9])           # left door, opened 0.5 rad
        left = er.door_reference("left_door", "hinge", h1, a1, [0, 0, 1], 0.5, (0.0, 1.6))
        right = er.door_reference("right_door", "hinge", [0.4, 0.0, 0.9], a2, [0, 0, 1], 0.0, (-1.6, 0.0))
        for door in (left, right):
            np.testing.assert_allclose(door["frame"][:3, 3], [0.4, 0.0, 0.9], atol=1e-9)
            self.assertAlmostEqual(er.yaw_of(door["frame"][:3, :3]), math.pi / 2)   # out of the front (+y)
            self.assertAlmostEqual(door["radius"], 0.4)
        self.assertEqual((left["side"], right["side"]), (1, -1))
        drawer = er.door_reference("slidejoint", "slide", [1.0, 0.1, 0.5], [9, 9, 9], [0, -1, 0], -0.1, (-0.3, 0.0))
        np.testing.assert_allclose(drawer["frame"][:3, 3], [1.0, 0.0, 0.5], atol=1e-9)
        self.assertAlmostEqual(er.yaw_of(drawer["frame"][:3, :3]), math.pi / 2)
        self.assertEqual(drawer["side"], 0)
        back = er.door_from_record(json.loads(json.dumps(er.door_record(left))))
        np.testing.assert_allclose(back["frame"], left["frame"], atol=1e-6)

    def test_doors_are_matched_by_type_side_size_and_role(self):
        src = er.door_reference("left_door", "hinge", [0.4, 0, 0.9], [0, 0, 0], [0, 0, 1], 0.0, (0.0, 1.6))
        mirror = er.door_reference("right_door", "hinge", [0.4, 0, 0.9], [0.8, 0, 0], [0, 0, 1], 0.0, (-1.6, 0.0))
        wider = er.door_reference("door", "hinge", [0.5, 0, 0.9], [0, 0, 0], [0, 0, 1], 0.0, (0.0, 1.6))
        door, d = er.match_door(src, [mirror, wider])
        self.assertIs(door, wider)
        self.assertAlmostEqual(d, 0.1 + er.DOOR_ROLE_MISMATCH)
        self.assertEqual(er.match_door(None, [wider])[0], None)
        world = FakeWorld()
        facts = {**facts_of(world), "doors": {"cab": [wider]}}
        near = door_demo(wider["frame"], world_pose(0, 0, 0.3), episode=1)
        far = door_demo(mirror["frame"], world_pose(0, 0, 0.3), episode=2)
        near.door, far.door = wider, mirror
        ranked = er.rank_sources([far, near], facts, lambda s: 0, lambda st: ("door", "cab"))
        self.assertEqual([s.episode for _, s in ranked], [1, 2])

    def test_a_door_step_replays_relative_to_the_door_after_aligning_the_base(self):
        demo_door = er.pose([0.7, 0.0, 0.8], er.rot_z(math.pi))      # handle ahead, front facing the robot
        demo = door_demo(demo_door, world_pose(0.0, 0.0, 0.3))
        here_frame = er.pose([3.0, 2.0, 0.8], er.rot_z(math.pi + 0.6))
        here = {**demo.door, "frame": here_frame}
        world = DoorWorld(here, mobile=(2.5, 1.0, -0.4))
        run = er.ExpertRun(world, TRUE_GAINS, {demo.key: demo}, None, cap=10 ** 5, budget=10 ** 5)
        out = run.run([{"phase": "track", "source": list(demo.key), "step": 0, "kind": er.EXECUTE,
                        "reference": ["door", "cab"]}])
        phase = out["phases"][0]
        self.assertEqual(phase["door"], "door_hinge")
        transform = here_frame @ er.inverse(demo_door)
        want = er.yaw_frame(transform @ demo.base[0])
        np.testing.assert_allclose(er.yaw_frame(world.base_T())[:2, 3], want[:2, 3], atol=er.NAV_FINE_POS)
        np.testing.assert_allclose(world.eef[:3, 3], (transform @ demo.eef[-1])[:3, 3], atol=er.TRACK_TOL)
        self.assertLess(phase["closes"][0]["pos"], er.TRACK_TOL)        # closes on the handle here
        no_door = DoorWorld(None, mobile=(2.5, 1.0, -0.4))
        no_door.doors = lambda fixture: []
        run = er.ExpertRun(no_door, TRUE_GAINS, {demo.key: demo}, None, cap=10 ** 5, budget=10 ** 5)
        phase = run.run([{"phase": "track", "source": list(demo.key), "step": 0, "kind": er.EXECUTE,
                          "reference": ["door", "cab"]}])["phases"][0]
        self.assertEqual(phase["door_fallback"], "base")                # base-relative, no alignment
        self.assertNotIn("align", phase)

    def test_the_door_a_restored_source_opens_is_the_joint_that_moved(self):
        class Restored:
            def __init__(self):
                self.q = {"left_door": 0.0, "right_door": 0.0}

            def doors(self, fixture):
                return [er.door_reference(r, "hinge", [0.4, 0, 0.9], [0, 0, 0], [0, 0, 1], q, (0.0, 1.6))
                        for r, q in self.q.items()]

        world = Restored()
        door, info = er.source_door(world, "cab", lambda: world.q.update(right_door=1.2))
        self.assertEqual((door["role"], door["q"], info["moved"]), ("right_door", 0.0, 1.2))

    def test_a_blocked_aligned_goal_moves_to_the_nearest_free_pose_or_is_refused(self):
        counter = np.array([[1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [1.0, 1.0]])
        goal = (0.95, 0.0, 0.0)           # the box reaches 0.15 m ahead of the origin: 10 cm into the counter
        _, log = er.plan_navigation([("counter", counter)], (-1.5, 0.0, 0.0), goal, free_goal=True)
        self.assertIn("free_goal", log)
        self.assertLessEqual(log["goal_shift"], er.ALIGN_SHIFT_MAX + 1e-9)
        self.assertLess(log["free_goal"][0], goal[0])                   # backed off the counter
        self.assertEqual(log["plan"], "planned")
        room = np.array([[0.0, -2.0], [2.0, -2.0], [2.0, 2.0], [0.0, 2.0]])
        _, log = er.plan_navigation([("island", room)], (-1.5, 0.0, 0.0), (1.0, 0.0, 0.0), free_goal=True)
        self.assertTrue(log["goal_blocked"])
        _, log = er.plan_navigation([("island", room)], (-1.5, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.assertEqual(log["dropped_obstacles"], ["island"])           # a plain drive still drops it

    def test_a_stop_inside_a_step_logs_how_far_it_got(self):
        demo = demo_source(world_pose(0.6, 0.0, 0.9), world_pose(0.6, -0.4, 0.9), world_pose(0, 0, 0.3))
        world = FakeWorld(objects={"obj": world_pose(0.6, 0.0, 0.9), "container": world_pose(0.6, -0.4, 0.9)},
                          predicate=mug_on_tray)
        program = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(world), FakeLibrary([demo]), n=1)[0]
        run = er.ExpertRun(world, TRUE_GAINS, {demo.key: demo}, "mug_on_tray", cap=30, budget=10 ** 5)
        out = run.run(program["phases"])
        last = out["phases"][-1]
        self.assertEqual((out["stop"], last["stopped"], last["kind"]), ("cap", "cap", er.PICK))
        self.assertGreater(last["frame_reached"], 0)
        self.assertLess(last["frame_reached"], last["frames"])
        self.assertIn("error_at_stop", last)
        self.assertIsInstance(last["holding_after"], bool)


def swing_door(anchor, radius, yaw=0.0, height=0.8, role="door"):
    """A left-hinged door (vertical axis) whose closed handle sits ``radius`` from ``anchor``."""
    a = np.array([anchor[0], anchor[1], height])
    handle = a + er.rot_z(yaw) @ np.array([radius, 0.0, 0.0])
    return er.door_reference(role, "hinge", handle, a, [0, 0, 1], 0.0, (0.0, 1.6))


HAND_ON_HANDLE = er.pose([0.0, 0.0, 0.0], er.rot_z(math.pi) @ R_DOWN)


def swing_demo(door, base_T, *, q_open=1.0, episode=0):
    """Reach the handle (frames 0-9), close, and swing the door to ``q_open`` (10-29)."""
    start = er.door_handle_frame(door, 0.0) @ er.pose([0.15, 0.0, 0.05], np.eye(3)) @ HAND_ON_HANDLE
    goal = er.door_handle_frame(door, 0.0) @ HAND_ON_HANDLE
    reach = [er.pose(start[:3, 3] + (goal[:3, 3] - start[:3, 3]) * s, goal[:3, :3]) for s in np.linspace(0, 1, 10)]
    qs = [0.0] * 10 + list(np.linspace(0.0, q_open, 20))
    swing = [er.door_handle_frame(door, q) @ HAND_ON_HANDLE for q in qs[10:]]
    n = len(qs)
    src_door = {**door, "q_track": qs, "q_track_start": 0}
    return er.Source(task_name="ArrangeTea", episode=episode, predicate="cabinet_closed", target=None, start=0,
                     end=n, steps=[er.SourceStep("OpenCabinet", er.EXECUTE, 0, n)],
                     eef=np.stack(reach + swing), base=np.stack([base_T] * n),
                     gripper=np.array([0.0] * 10 + [1.0] * 20), door=src_door)


class DoorAngleReplayTest(unittest.TestCase):
    def test_the_handle_frame_follows_the_door(self):
        door = swing_door((0.0, 0.0), 0.4)
        moved = er.door_handle_frame(door, 0.7)
        np.testing.assert_allclose(moved[:3, 3], [0.4 * math.cos(0.7), 0.4 * math.sin(0.7), 0.8], atol=1e-9)
        opened = er.door_reference("door", "hinge", moved[:3, 3], door["anchor"], [0, 0, 1], 0.7, (0.0, 1.6))
        np.testing.assert_allclose(opened["frame"], door["frame"], atol=1e-9)   # read back closed
        drawer = er.door_reference("slide", "slide", [1.0, 0.0, 0.5], [9, 9, 9], [0, -1, 0], 0.0, (0.0, 0.4))
        np.testing.assert_allclose(er.door_handle_frame(drawer, 0.3)[:3, 3], [1.0, -0.3, 0.5], atol=1e-9)

    def test_a_hand_on_the_source_handle_lands_on_this_handle_at_the_same_angle(self):
        src = swing_door((0.0, 0.0), 0.4)
        here = swing_door((2.0, 1.0), 0.55, yaw=0.5)
        qs = np.linspace(0.0, 1.2, 7)
        Ts, q_here = er.door_track_transforms(src, here, qs, 0.0)
        np.testing.assert_allclose(q_here, qs)
        for T, q in zip(Ts, qs):
            hand = T @ er.door_handle_frame(src, q) @ HAND_ON_HANDLE
            np.testing.assert_allclose(hand, er.door_handle_frame(here, q) @ HAND_ON_HANDLE, atol=1e-9)
            radius = np.hypot(*(hand[:2, 3] - np.asarray(here["anchor"])[:2]))
            self.assertAlmostEqual(radius, 0.55)
        _, q_here = er.door_track_transforms(src, here, qs, 0.9)   # a door already partly open here
        np.testing.assert_allclose(q_here, np.clip(0.9 + qs, 0.0, 1.6))

    def test_the_executor_swings_the_hand_on_this_doors_radius(self):
        src = swing_door((0.0, 0.0), 0.4)
        demo = swing_demo(src, world_pose(0.2, -0.9, 0.3, yaw=math.pi / 2))
        here = swing_door((2.0, 1.0), 0.55, yaw=0.5)
        world = DoorWorld(here, mobile=(2.0, 0.0, 1.0))
        run = er.ExpertRun(world, TRUE_GAINS, {demo.key: demo}, None, cap=10 ** 5, budget=10 ** 5)
        phase = run.run([{"phase": "track", "source": list(demo.key), "step": 0, "kind": er.EXECUTE,
                          "reference": ["door", "cab"]}])["phases"][0]
        self.assertTrue(phase["door_track"])
        self.assertEqual(phase["door_q"], [0.0, 1.0])
        # [frame, wanted, actual] every DOOR_SAMPLE_EVERY frames and at the end; this world has no joint
        self.assertEqual([s[0] for s in phase["door_samples"]], [0, 25, 29])
        self.assertEqual(phase["door_samples"][-1][1:], [1.0, None])
        want = er.door_handle_frame(here, 1.0) @ HAND_ON_HANDLE
        np.testing.assert_allclose(world.eef[:3, 3], want[:3, 3], atol=er.TRACK_TOL)
        rigid = {**demo.door, "q_track": None}
        demo.door = rigid
        world = DoorWorld(here, mobile=(2.0, 0.0, 1.0))
        run = er.ExpertRun(world, TRUE_GAINS, {demo.key: demo}, None, cap=10 ** 5, budget=10 ** 5)
        phase = run.run([{"phase": "track", "source": list(demo.key), "step": 0, "kind": er.EXECUTE,
                          "reference": ["door", "cab"]}])["phases"][0]
        self.assertFalse(phase["door_track"])                              # the rigid arc, off this circle
        self.assertGreater(np.linalg.norm(world.eef[:3, 3] - want[:3, 3]), 0.05)

    def test_a_stopped_door_step_says_where_the_door_was_and_what_the_robot_touched(self):
        src = swing_door((0.0, 0.0), 0.4)
        demo = swing_demo(src, world_pose(0.2, -0.9, 0.3, yaw=math.pi / 2))
        here = {**swing_door((2.0, 1.0), 0.55, yaw=0.5), "qpos_adr": 7}
        world = DoorWorld(here, mobile=(2.0, 0.0, 1.0))
        world.joint_q = lambda adr: 0.25 if adr == 7 else float("nan")
        world.robot_contacts = lambda: {"gripper0_right_leftfinger>cab_door": 3}
        run = er.ExpertRun(world, TRUE_GAINS, {demo.key: demo}, None, cap=60, budget=10 ** 5)
        out = run.run([{"phase": "track", "source": list(demo.key), "step": 0, "kind": er.EXECUTE,
                        "reference": ["door", "cab"]}])
        phase = out["phases"][-1]
        self.assertEqual((out["stop"], phase["stopped"]), ("cap", "cap"))
        frame, wanted, actual = phase["door_samples"][-1]
        self.assertEqual(frame, phase["frame_reached"])
        self.assertEqual(actual, 0.25)
        self.assertEqual(phase["contacts_at_stop"], {"gripper0_right_leftfinger>cab_door": 3})

    def test_the_joint_track_is_read_from_the_states_and_checked_against_the_restore(self):
        n = 12
        q = np.linspace(0.0, 1.1, n)
        states = np.column_stack([np.arange(n) * 0.05, np.full(n, 0.3), q, np.zeros((n, 3))])   # time, qpos, qvel
        src = er.Source(task_name="T", episode=0, predicate="p", target=None, start=2, end=10, steps=[],
                        eef=np.zeros((8, 4, 4)), base=np.zeros((8, 4, 4)), gripper=np.zeros(8))
        door = {"q": float(q[3]), "qpos_adr": 1}
        out = er.source_q_track(states, door, src, 3, 9, round(float(q[9]), 4))
        self.assertEqual(out["q_track_column"], 2)
        np.testing.assert_allclose(out["q_track"], q[2:10], atol=1e-5)
        self.assertEqual(out["q_track_start"], 2)
        bad = er.source_q_track(states, {"q": 0.7, "qpos_adr": 1}, src, 3, 9, float(q[9]))
        self.assertIsNone(bad["q_track"])


GRASP_AXES = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))   # FakeWorld: z out of the hand, fingers on y


def front_door(anchor, radius, *, hinge="left", joint="door", height=0.8):
    """A door in a fixture whose front faces +y: hinged at ``anchor``, its handle ``radius`` to the
    right (left hinge) or left (right hinge), opening toward +y. The handle bar is vertical, 30 cm
    long, 4 cm in front of the door."""
    a = np.array([anchor[0], anchor[1], height])
    sign = 1.0 if hinge == "left" else -1.0
    handle = a + np.array([sign * radius, 0.0, 0.0])
    door = er.door_reference(joint, "hinge", handle, a, [0.0, 0.0, sign], 0.0, (0.0, 1.6))
    door.update(joint=joint, qpos_adr=joint, width=radius + 0.05, bar={"centre": list(handle + [0.0, 0.04, 0.0]),
                                                 "axis": [0.0, 0.0, 1.0], "half_length": 0.15})
    return door


class DoorPullWorld(FakeWorld):
    """FakeWorld with doors that follow a hand closed on their handle bar (pads within 2 cm), and
    let go when it opens or drifts more than 5 cm from the grasp point. ``predicate``: all open.
    With ``shut``, a hand whose path crosses the leaf of an open door it does not hold shuts it."""

    def __init__(self, doors, *, grip=True, knock=None, shut=False, **kw):
        super().__init__(predicate=self.all_open, **kw)
        self.door_list = doors
        self.q = {d["qpos_adr"]: 0.0 for d in doors}
        self.held_door, self.grip = None, grip
        self.knock = dict(knock or {})   # {door gripped: door it swings back to 0.3 rad, once}
        self.shut, self.shut_doors, self.last_hand = shut, [], None

    @staticmethod
    def crosses(p0, p1, q0, q1):
        """Whether the horizontal segments p0-p1 and q0-q1 cross."""
        def turn(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (turn(p0, p1, q0) * turn(p0, p1, q1) < 0) and (turn(q0, q1, p0) * turn(q0, q1, p1) < 0)

    def shut_crossed(self, hand):
        held = self.held_door[0]["qpos_adr"] if self.held_door is not None else None
        for d in self.door_list:
            adr = d["qpos_adr"]
            if adr == held or abs(self.q[adr]) < 0.1:
                continue
            a = np.asarray(d["anchor"])
            tip = er.door_handle_frame(d, self.q[adr])[:3, 3] - a
            tip = a + tip / np.linalg.norm(tip[:2]) * d["width"]
            if self.crosses(self.last_hand[:2], hand[:2], a[:2], tip[:2]):
                self.q[adr] = 0.0
                self.shut_doors.append(adr)

    @staticmethod
    def all_open(world):
        return {"fridge_open": all(er.door_open_fraction(d, world.q[d["qpos_adr"]]) >= er.DOOR_OPEN_DONE
                                   for d in world.door_list)}

    def doors(self, fixture, predicate_joints=False):
        return [{**d, "q": self.q[d["qpos_adr"]]} for d in self.door_list] if fixture == "fridge" else []

    def joint_q(self, adr):
        return self.q[adr]

    def gripper_axes(self):
        return tuple(np.array(v) for v in GRASP_AXES)

    def step(self, a):
        info = super().step(a)
        hand = self.eef[:3, 3].copy()
        if self.shut and self.last_hand is not None:
            self.shut_crossed(hand)
        self.last_hand = hand
        if self.gripper < 0.5:
            self.held_door = None
        elif self.held_door is None and self.grip:
            for d in self.door_list:
                centre, axis, half = er.door_bar(d, self.q[d["qpos_adr"]])
                t = float(np.clip((hand - centre) @ axis, -half, half))
                if np.linalg.norm(centre + t * axis - hand) < 0.02:
                    local = er.inverse(er.door_motion(d, self.q[d["qpos_adr"]])) @ np.append(hand, 1.0)
                    self.held_door = (d, local[:3])
                    other = self.knock.pop(d["qpos_adr"], None)
                    if other is not None:
                        self.q[other] = 0.3
                    break
        if self.held_door is not None:
            d, local = self.held_door
            anchor, u = np.asarray(d["anchor"]), np.asarray(d["axis"])
            ang = lambda v: math.atan2(v[1], v[0])  # noqa: E731
            lo, hi = d["range"]
            q = float(np.clip(u[2] * (ang(hand - anchor) - ang(local - anchor)), lo, hi))
            q = q - 2 * math.pi * round((q - self.q[d["qpos_adr"]]) / (2 * math.pi))
            point = (er.door_motion(d, q) @ np.append(local, 1.0))[:3]
            if np.linalg.norm(point - hand) > 0.05:
                self.held_door = None
            else:
                self.q[d["qpos_adr"]] = q
        predicates = self.predicate(self)
        return {**info, "predicates": predicates}


def door_open_demo(door, base_T, *, episode=0):
    """A source that reaches the handle (frames 0-9), closes, and swings its door to 1.5 (10-29)."""
    hand = er.pose([0.0, 0.0, 0.0], er.rot_z(math.pi) @ R_DOWN)
    start = er.door_handle_frame(door, 0.0) @ er.pose([0.15, 0.0, 0.05], np.eye(3)) @ hand
    goal = er.door_handle_frame(door, 0.0) @ hand
    reach = [er.pose(start[:3, 3] + (goal[:3, 3] - start[:3, 3]) * s, goal[:3, :3]) for s in np.linspace(0, 1, 10)]
    qs = [0.0] * 10 + list(np.linspace(0.0, 1.5, 20))
    swing = [er.door_handle_frame(door, q) @ hand for q in qs[10:]]
    n = len(qs)
    return er.Source(task_name="GarnishPancake", episode=episode, predicate="fridge_open", target=None, start=0,
                     end=n, steps=[er.SourceStep("OpenFridge", er.EXECUTE, 0, n)],
                     eef=np.stack(reach + swing), base=np.stack([base_T] * n),
                     gripper=np.array([0.0] * 10 + [1.0] * 20), door={**door, "q_track": qs, "q_track_start": 0})


# The source stands 0.65 m in front of its left door's handle and 0.2 m away from the hinge, facing it.
SOURCE_DOOR = front_door((0.0, 0.0), 0.4)
SOURCE_BASE = world_pose(0.6, 0.65, 0.3, yaw=-math.pi / 2)


class ScriptedDoorTest(unittest.TestCase):
    def test_open_fraction_reads_like_robocasa_for_either_opening_direction(self):
        door = SOURCE_DOOR
        self.assertAlmostEqual(er.door_open_fraction(door, 0.8), 0.5)
        self.assertAlmostEqual(er.door_goal_q(door, 0.9), 1.44)
        neg = {**door, "range": [-1.6, 0.0], "open_sign": -1.0}
        self.assertAlmostEqual(er.door_open_fraction(neg, -1.44), 0.9)
        self.assertAlmostEqual(er.door_goal_q(neg, 0.9), -1.44)

    def test_the_hand_comes_in_against_the_doors_normal_across_its_bar(self):
        door = SOURCE_DOOR
        grasp = er.handle_grasp(door, 0.0, *GRASP_AXES, R_DOWN, 1.5)
        R = grasp[:3, :3]
        np.testing.assert_allclose(R @ [0, 0, 1], [0, -1, 0], atol=1e-9)        # approach: into the door
        self.assertAlmostEqual(float((R @ [0, 1, 0])[2]), 0.0)                  # fingers close horizontally
        np.testing.assert_allclose(grasp[:3, 3], [0.4, 0.04, 0.8 + 0.15 - er.DOOR_BAR_MARGIN], atol=1e-9)
        opened = er.handle_grasp(door, 0.5, *GRASP_AXES, R_DOWN, 0.8)
        centre, _, _ = er.door_bar(door, 0.5)
        np.testing.assert_allclose(opened[:3, 3], centre, atol=1e-9)             # the bar moved with the door
        normal = er.door_handle_frame(door, 0.5)[:3, 0]
        np.testing.assert_allclose(opened[:3, :3] @ [0, 0, 1], -normal, atol=1e-9)

    def test_the_stance_is_mirrored_for_a_door_hinged_on_the_other_side(self):
        rel = er.inverse(er.yaw_frame(er.door_handle_frame(SOURCE_DOOR, 0.0))) @ er.yaw_frame(SOURCE_BASE)
        rel = (rel[0, 3], rel[1, 3], er.yaw_of(rel[:3, :3]))
        same = er.door_stance(SOURCE_DOOR, 0.0, rel, False)
        np.testing.assert_allclose(same[:2, 3], [0.6, 0.65], atol=1e-9)
        right = front_door((1.2, 0.0), 0.4, hinge="right")
        self.assertEqual((SOURCE_DOOR["side"], right["side"]), (1, -1))
        mirrored = er.door_stance(right, 0.0, rel, True)
        np.testing.assert_allclose(mirrored[:2, 3], [0.6, 0.65], atol=1e-9)   # away from this door's hinge too
        self.assertAlmostEqual(abs(er.wrap(er.yaw_of(mirrored[:3, :3]) + math.pi / 2)), 0.0)

    def test_the_pull_steps_on_from_the_door_when_it_runs_ahead(self):
        self.assertEqual(er.next_pull_q(0.2, 0.1, 1.4, 0.04), (0.24000000000000002, False))   # trailing: from the pull
        self.assertEqual(er.next_pull_q(0.2, 0.6, 1.4, 0.04), (0.64, False))                 # ahead: from the door
        self.assertEqual(er.next_pull_q(1.38, 1.38, 1.4, 0.04), (1.4, True))
        self.assertEqual(er.next_pull_q(-0.2, -0.5, -1.4, 0.04), (-0.54, False))              # opening toward negative

    def test_a_door_that_swung_back_is_reopened_on_a_second_pass(self):
        left = front_door((0.0, 0.0), 0.4, joint="left")
        right = front_door((1.0, 0.0), 0.4, hinge="right", joint="right")
        world, out = self.run_doors([left, right], knock={"right": "left"})
        self.assertTrue(out["stage_completed"], json.dumps(out["phases"], default=str)[:3000])
        doors = out["phases"][-1]["doors"]
        self.assertEqual([(d["joint"], d["pass"]) for d in doors], [("left", 0), ("right", 0), ("left", 1)])
        self.assertGreater(doors[1]["cycles"][0]["fractions"]["left"], er.DOOR_OPEN_DONE)   # open before the knock

    def run_doors(self, doors, *, grip=True, knock=None, shut=False, start=(0.5, 0.9, -math.pi / 2)):
        demo = door_open_demo(SOURCE_DOOR, SOURCE_BASE)
        world = DoorPullWorld(doors, grip=grip, knock=knock, shut=shut, mobile=start, eef_rel=(0.45, 0.0, 0.5))
        run = er.ExpertRun(world, TRUE_GAINS, {demo.key: demo}, "fridge_open", cap=10 ** 6, budget=10 ** 7,
                           scripted_door=True)
        out = run.run([{"phase": "track", "source": list(demo.key), "step": 0, "kind": er.EXECUTE,
                        "reference": ["door", "fridge"]}])
        return world, out

    def test_the_route_around_an_open_door_passes_beyond_its_free_edge(self):
        door = SOURCE_DOOR                                   # width 0.45, opens toward +y
        self.assertIsNone(er.door_edge_route(door, 0.2, [0.0, 0.5, 0.8]))
        outer, inner, side = er.door_edge_route(door, math.pi / 2, [-0.14, 0.4, 0.9])
        edge = 0.45 + er.DOOR_EDGE_CLEAR
        np.testing.assert_allclose(outer, [-er.DOOR_EDGE_SIDE, edge, 0.9], atol=1e-9)   # the handle's side
        np.testing.assert_allclose(inner, [er.DOOR_EDGE_SIDE, edge, 0.9], atol=1e-9)
        self.assertEqual(side, 1)
        self.assertEqual(er.door_edge_route(door, math.pi / 2, [0.3, 0.2, 0.9])[2], -1)
        self.assertEqual(er.door_edge_route(door, math.pi / 2, [0.0, 0.7, 0.9])[2], 0)
        right = front_door((1.0, 0.0), 0.4, hinge="right")
        outer, inner, _ = er.door_edge_route(right, math.pi / 2, [1.0, 0.4, 0.8])
        np.testing.assert_allclose(outer[:2], [1.0 + er.DOOR_EDGE_SIDE, edge], atol=1e-9)

    def test_the_pads_offset_from_the_bar_is_split_into_depth_across_and_along(self):
        door = front_door((0.0, 0.0), 0.4)                    # bar at (0.4, 0.04, 0.8), vertical, out is +y
        eef = np.eye(4)
        eef[:3, 3] = [0.4 + 0.02, 0.04 + 0.01, 0.8 + 0.05]
        np.testing.assert_allclose(er.pad_bar_offset(door, 0.0, eef, (0.0, 0.0, 0.0)), [0.01, -0.02, 0.05], atol=1e-9)
        np.testing.assert_allclose(er.pad_bar_offset(door, 0.0, eef, (0.0, -0.03, 0.0)), [-0.02, -0.02, 0.05], atol=1e-9)

    def test_each_pull_logs_the_grip_after_closing_and_on_its_first_steps(self):
        world, out = self.run_doors([front_door((0.0, 0.0), 0.4)])
        self.assertTrue(out["stage_completed"])
        cyc = out["phases"][0]["doors"][0]["cycles"][0]
        self.assertGreaterEqual(len(cyc["grip"]), 2)
        self.assertLessEqual(len(cyc["grip"]), er.DOOR_GRIP_TRACE + 1)
        q_wanted, q, offset = cyc["grip"][0]
        self.assertLess(max(abs(v) for v in offset[:2]), 0.03)   # closed on the bar

    def test_the_hand_goes_around_the_first_open_door_to_the_second(self):
        left = front_door((0.0, 0.0), 0.4, joint="left")
        right = front_door((1.0, 0.0), 0.4, hinge="right", joint="right")
        world, out = self.run_doors([left, right], shut=True)
        self.assertEqual(world.shut_doors, [])
        self.assertTrue(out["stage_completed"], json.dumps(out["phases"], default=str)[:3000])
        doors = out["phases"][0]["doors"]
        self.assertEqual(doors[0].get("around_edge"), "out")

    def test_without_the_route_the_hand_shuts_the_first_door(self):
        left = front_door((0.0, 0.0), 0.4, joint="left")
        right = front_door((1.0, 0.0), 0.4, hinge="right", joint="right")
        with mock.patch.object(er, "DOOR_EDGE_MIN", 10.0), mock.patch.object(er, "DOOR_STANCE_GAIN", math.inf):
            world, out = self.run_doors([left, right], shut=True)
        self.assertNotEqual(world.shut_doors, [])

    def test_a_stance_the_arc_leaves_is_replaced_by_one_that_covers_it(self):
        door = SOURCE_DOOR
        hand = er.door_bar(door, 0.0)[0]
        close = world_pose(0.4, 0.45, 0.3, yaw=-math.pi / 2)   # in front of the handle: the door swings into it
        goal = er.door_goal_q(door, er.DOOR_OPEN_GOAL)
        before = er.door_arc_reach(door, 0.0, goal, hand, close, er.DOOR_STANCE_MARGIN)
        self.assertLess(before, goal / 2)
        where, log = er.door_pull_stance(door, 0.0, goal, hand, close)
        self.assertTrue(log["planned"], log)
        self.assertAlmostEqual(log["planned_reach"], round(goal, 3))
        self.assertAlmostEqual(er.door_arc_reach(door, 0.0, goal, hand, where, er.DOOR_STANCE_MARGIN), goal, places=6)
        self.assertGreater(where[1, 3], 0.3)                    # in front of the fixture
        kept, log = er.door_pull_stance(door, 0.0, goal, hand, where)   # a stance that covers it is kept
        self.assertFalse(log["planned"])
        self.assertIs(kept, where)

    def test_a_door_of_another_radius_is_pulled_open_on_its_own_arc(self):
        door = front_door((0.1, 0.0), 0.55, joint="fridge_door")
        world, out = self.run_doors([door])
        self.assertTrue(out["stage_completed"], json.dumps(out, default=str)[:3000])
        self.assertEqual(out["stop"], "stage")
        self.assertGreaterEqual(er.door_open_fraction(door, world.q["fridge_door"]), er.DOOR_OPEN_DONE)

    def test_both_doors_of_a_french_door_fridge_are_opened(self):
        left = front_door((0.0, 0.0), 0.4, joint="left")
        right = front_door((1.0, 0.0), 0.4, hinge="right", joint="right")
        world, out = self.run_doors([left, right], start=(0.5, 0.9, -math.pi / 2))
        self.assertTrue(out["stage_completed"], json.dumps(out["phases"], default=str)[:3000])
        for d in (left, right):
            self.assertGreaterEqual(er.door_open_fraction(d, world.q[d["qpos_adr"]]), er.DOOR_OPEN_DONE)

    def test_a_door_that_does_not_follow_ends_each_cycle_as_a_slip(self):
        door = front_door((0.1, 0.0), 0.55, joint="fridge_door")
        world, out = self.run_doors([door], grip=False)
        self.assertFalse(out["stage_completed"])
        phase = out["phases"][0]
        self.assertTrue(phase["door_script"])
        cycles = phase["doors"][0]["cycles"]
        self.assertEqual(len(cycles), er.DOOR_CYCLES)
        self.assertEqual({c["stop"] for c in cycles}, {"slip"})
        self.assertEqual(world.q["fridge_door"], 0.0)


class GraspWorld(FakeWorld):
    """FakeWorld whose objects have bounding boxes (centre, half sizes in the object frame)."""

    def __init__(self, boxes, **kw):
        super().__init__(**kw)
        self.boxes = boxes

    def object_box(self, name):
        if name not in self.boxes:
            return None
        centre, half = self.boxes[name]
        return self.objects[name], np.asarray(centre, dtype=float), np.asarray(half, dtype=float)

    def gripper_axes(self):
        return tuple(np.array(v) for v in GRASP_AXES)


def carried(demo, name="obj"):
    """The demo with its object moving in the hand between the close and the release."""
    closed = demo.gripper >= 0.5
    k0 = int(np.argmax(closed))
    rel = er.inverse(demo.eef[k0]) @ demo.objects[name][k0]
    poses, last = [], demo.objects[name][0]
    for k in range(len(demo.gripper)):
        if closed[k]:
            last = demo.eef[k] @ rel
        poses.append(last)
    demo.objects[name] = np.stack(poses)
    return demo


class ScriptedGraspTest(unittest.TestCase):
    def test_the_grasp_closes_across_the_narrow_side_below_the_top(self):
        T = world_pose(1.0, 2.0, 0.9, yaw=0.3)
        corners = er.box_corners(T, [0, 0, 0], [0.06, 0.025, 0.04])
        plan = er.grasp_plan(corners, *GRASP_AXES, R_DOWN)
        self.assertAlmostEqual(plan["width"], 0.05)
        R = plan["grasp"][:3, :3]
        np.testing.assert_allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-9)             # approach: straight down
        g = er.rot_z(0.3) @ np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(abs(float((R @ [0, 1, 0]) @ g)), 1.0)                 # fingers across the 5 cm
        np.testing.assert_allclose(plan["grasp"][:3, 3], [1.0, 2.0, 0.94 - er.GRASP_DEPTH], atol=1e-9)
        np.testing.assert_allclose(plan["above"][:3, 3], plan["grasp"][:3, 3] + [0, 0, er.GRASP_ABOVE], atol=1e-9)
        flat = er.grasp_plan(er.box_corners(T, [0, 0, 0], [0.03, 0.02, 0.005]), *GRASP_AXES, R_DOWN)
        self.assertAlmostEqual(flat["pad_height"], 0.895 + er.GRASP_CLEAR)           # a thin object: above its bottom
        wide = er.grasp_plan(er.box_corners(T, [0, 0, 0], [0.06, 0.05, 0.04]), *GRASP_AXES, R_DOWN)
        self.assertEqual(wide, {"too_wide": 0.1})

    def test_object_path_targets_carry_the_object_then_repeat_the_hands_retreat(self):
        demo = carried(demo_source(world_pose(0.55, 0.05, 0.9), world_pose(0.6, -0.4, 0.9), world_pose(0, 0, 0.3)))
        frames = range(20, len(demo.gripper))
        T = world_pose(1.0, 2.0, 0.0, yaw=0.8)
        G = er.pose([0.03, 0.0, 0.05], er.rot_z(0.6) @ R_DOWN)     # held unlike the demo: offset and turned
        objs = [demo.objects["obj"][f] for f in frames]
        eefs = [demo.eef[f] for f in frames]
        grip = [float(demo.gripper[f]) for f in frames]
        targets = er.object_path_targets(T, objs, eefs, grip, G)
        opened = next(k for k in range(1, len(grip)) if grip[k] < 0.5 <= grip[k - 1])
        for k in range(opened):
            np.testing.assert_allclose(targets[k] @ er.inverse(G), T @ objs[k], atol=1e-9)   # object on the path
        for k in range(opened, len(grip)):
            np.testing.assert_allclose(er.inverse(targets[opened]) @ targets[k],
                                       er.inverse(eefs[opened]) @ eefs[k], atol=1e-9)

    def scene(self, boxes, ceiling=None):
        demo = carried(demo_source(world_pose(0.55, 0.05, 0.9), world_pose(0.6, -0.4, 0.9), world_pose(0, 0, 0.3)))
        world = GraspWorld(boxes, mobile=(1.0, 2.0, 0.6), eef_rel=(0.4, 0.2, 0.8), predicate=mug_on_tray)
        if ceiling is not None:
            world.ceiling = ceiling
        base = world.base_T()
        world.objects = {"obj": base @ world_pose(0.6, 0.15, 0.6, yaw=1.0),
                         "container": base @ world_pose(0.5, -0.3, 0.6)}
        program = er.plan_attempts("ArrangeTea", "mug_on_tray", facts_of(world), FakeLibrary([demo]), n=1)[0]
        run = er.ExpertRun(world, TRUE_GAINS, {demo.key: demo}, "mug_on_tray", cap=10 ** 5, budget=10 ** 5,
                           scripted_grasp=True)
        return world, run.run(program["phases"])

    def test_a_scripted_pick_then_the_object_follows_the_source_objects_path(self):
        # The box centre sits 1.5 cm off the object's origin: the hand holds it there, not as the demo did.
        world, out = self.scene({"obj": ([0.015, 0.0, 0.0], [0.02, 0.03, 0.03])})
        self.assertTrue(out["stage_completed"], out)
        pick, place = out["phases"]
        self.assertEqual((pick["grasp"], pick["holding_after"]), ("scripted", True))
        self.assertAlmostEqual(pick["width"], 0.04)
        self.assertEqual(place["place"], "object_path")
        tray = world.objects["container"][:3, 3] + [0, 0, PLACE_HEIGHT]
        self.assertLess(np.linalg.norm(world.objects["obj"][:3, 3] - tray), er.TRACK_TOL)

    def test_the_ceiling_rays_start_at_the_fingers_and_the_hand_edges(self):
        plan = {"centre": [1.0, 2.0], "pad_height": 0.9, "axis_yaw": 0.0, "width": 0.04}
        pts = np.array(er.ceiling_points(plan))
        np.testing.assert_allclose(pts[:, 2], 0.9)
        np.testing.assert_allclose(pts[:3, :2], [[1.0, 2.0], [1.0 + 0.02 + er.GRASP_FINGER, 2.0],
                                                 [1.0 - 0.02 - er.GRASP_FINGER, 2.0]], atol=1e-12)
        np.testing.assert_allclose(pts[3:, :2], [[1.0, 2.0 + er.GRASP_HAND_HALF], [1.0, 2.0 - er.GRASP_HAND_HALF]],
                                   atol=1e-12)

    def test_something_over_the_object_keeps_the_replayed_pick(self):
        asked = []

        def ceiling(name, points, height):
            asked.append((name, len(points), height))
            return ("shelf_main", 0.12)

        _, out = self.scene({"obj": ([0.015, 0.0, 0.0], [0.02, 0.03, 0.03])}, ceiling=ceiling)
        self.assertEqual(asked, [("obj", 5, er.GRASP_CEILING)])
        self.assertTrue(out["stage_completed"], out)
        self.assertEqual(out["phases"][0]["grasp"], "replay")
        self.assertEqual(out["phases"][0]["grasp_skipped"], {"blocked": "shelf_main", "clearance": 0.12, "width": 0.04})
        _, clear = self.scene({"obj": ([0.015, 0.0, 0.0], [0.02, 0.03, 0.03])}, ceiling=lambda *a: None)
        self.assertEqual(clear["phases"][0]["grasp"], "scripted")

    def test_an_object_too_wide_for_the_gripper_keeps_the_replayed_pick(self):
        _, out = self.scene({"obj": ([0.0, 0.0, 0.0], [0.06, 0.06, 0.03])})
        self.assertTrue(out["stage_completed"], out)
        self.assertEqual(out["phases"][0]["grasp"], "replay")
        self.assertEqual(out["phases"][0]["grasp_skipped"], {"too_wide": 0.12})
