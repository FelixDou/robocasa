"""Simulator-free tests for R2: source matching, the executor on a kinematic stand-in, and the gates."""

import json
import math
from pathlib import Path
import tempfile
import unittest

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
        self.assertTrue(all(ph["reference"] == list(er.BASE_REF) for ph in phases[1:]))
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
        self.assertEqual([p["kind"] for p in result["phases"]], [er.PICK])   # stopped inside the place step

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

    def test_navigation_controller_converges(self):
        world = FakeWorld(mobile=(0.0, 0.0, 0.0))
        goal = (2.0, -1.5, -2.8)
        for _ in range(er.NAVIGATE_MAX_STEPS):
            if er.navigation_done(world.observe()["mobile"], goal, er.NAV_FINE_POS, math.cos(er.NAV_FINE_YAW)):
                break
            world.step(er.base_command(world.observe()["mobile"], goal, TRUE_GAINS))
        self.assertTrue(er.navigation_done(world.observe()["mobile"], goal))


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
        s = er.self_replay_summary(rows)
        self.assertEqual(s["failing"], ["ArrangeTea:kettle_on_tray"])           # too few demos
        self.assertTrue(next(x for x in s["stages"] if x["predicate"] == "mug_on_tray")["passed"])
        runs = [{"task_name": "ArrangeTea", "stages": [{"predicate": "kettle_on_tray", "stage_completed": True},
                                                       {"predicate": "mug_on_tray", "stage_completed": i < 6}]}
                for i in range(10)]
        t = er.transfer_summary(runs)
        self.assertEqual(t["failing"], ["ArrangeTea:mug_on_tray"])               # 6 of 10 < 0.7
        self.assertEqual(er.transfer_stages("ArrangeTea", ["kettle_on_tray", "mug_on_tray", "cabinet_closed",
                                                           "gripper_released"]),
                         ["kettle_on_tray", "mug_on_tray", "cabinet_closed"])

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


if __name__ == "__main__":
    unittest.main()
