"""R2: expert recoveries at the alarm states (plan step R2 of "Redesign after Gate A").

Plan: docs/recovery_harvest_finetune_plan.md, "R2 design". From each class-2
alarm state, an executor replays a human-demo segment for the stage the state
is stuck on (R1 source: segment replay), then XR-1 continues under the task
instruction. The parts:

* **R2a source matching** (``plan_attempt``, no simulator). The facts it reads
  (what the gripper holds, object and base poses) come from the restored alarm
  snapshot, because the R0 audit stores neither orientations nor the base pose.
* **R2b executor** (``run_program``). A program of phases:
  ``release`` (open, lift, back off), ``navigate`` (scripted, to
  ``compute_robot_base_placement_pose``), ``track`` (a source step's
  end-effector poses, moved into the alarm scene through the step's reference
  frame, with the source's gripper commands) and ``set_aside`` (the wrong
  cutting tool off the board). It talks to the world through a small adapter,
  so the state machine is tested here on a kinematic stand-in.
* **Checks and Gate R** (``self_replay_summary``, ``transfer_summary``,
  ``gate_r``, no simulator).

Actions are the environment's 12-d vector (end-effector position delta,
rotation delta, gripper, base motion, control mode), the space XR-1 acts in.
The arm and base gains (action -> motion per step) are measured by
``probe-controls`` instead of read from robosuite internals, and every
pose the executor uses is in the observation's own convention (the state keys
the policy and the LeRobot demos share).

Commands: ``probe-controls``, ``self-replay``, ``transfer``, ``gate-r``
(simulator) and ``analyze`` (no simulator).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from robocasa.recovery import expert_segments as es
from robocasa.recovery import recoverability_ladder as ladder


# ---- frozen with this code (26 November 2026, before any R2 output) ----------------------------

ATTEMPTS_PER_STATE = 10       # Gate R: attempts per class-2 state, from distinct sources where available
GATE_R_MIN_SHARE = 0.5        # class-2 states with >= 1 full-task success among their attempts
BUDGET_FRACTION = 0.5         # fixed budget from the alarm, expert time included (as Phase A)
HOLD_STEPS = 5                # the expert stops once the stage predicate holds this long
SEGMENT_SLACK = 0.5           # expert cap: executed source frames x (1 + slack), navigation apart
NAVIGATE_MAX_STEPS = 300
NAV_POS_TOL, NAV_COS_TOL = 0.20, 0.98        # NavigateKitchen's success test
NAV_FINE_POS, NAV_FINE_YAW = 0.05, 0.05      # m, rad: the controller drives on to this before stopping
NAV_STEP_POS, NAV_STEP_YAW = 0.02, 0.05      # per-step displacement limits (0.4 m/s, 1 rad/s at 20 Hz)
NAV_TRIGGER = 0.25            # m: navigate before the first step if no source has its reference this close
RANK_YAW_WEIGHT = 0.1         # m per rad in the source ranking distance
APPROACH_POS_STEP, APPROACH_ROT_STEP = 0.01, 0.05   # m, rad per step toward a step's first pose
SETTLE_TOL, SETTLE_STEPS = 0.01, 20
TRACK_TOL = 0.02              # m: a source frame is held until the end effector is this close
TRACK_YAW_TOL = 0.05          # rad: ... and, in the source's base-mode frames, the base this close in heading
TORSO_TOL = 0.01              # m: the torso follows the source's base height beyond this
TORSO_MIN_GAIN = 1e-4         # m per step per unit action: below this the probe found no torso motion
RELEASE_OPEN_STEPS, RELEASE_LIFT, RELEASE_BACKOFF = 10, 0.10, 0.10
SET_ASIDE_LIFT, SET_ASIDE_SIDE, SET_ASIDE_LOWER = 0.15, 0.30, 0.10
SELF_REPLAY_DEMOS, SELF_REPLAY_MIN = 20, 0.9
TRANSFER_RESETS, TRANSFER_MIN = 20, 0.7
TRANSFER_BASE_SEED = 710007   # fresh pretrain-split seeds 710007-710026; Phase A/B and Step 0 do not use them
CONVENTION_TOL = 1e-3         # self-replay: observation vs LeRobot state at the segment start
PROBE_AMPLITUDES = (0.25, 0.5)
PROBE_STEPS = 6
PROBE_MIN_R2, PROBE_MAX_COND = 0.9, 20.0
ARM_MOVES_BASE_TOL = 1e-3     # m per step: arm-mode commands must not move the base
ARM_HOLD_TOL = 5e-3           # m: hand offset in the base frame after a base run and SETTLE_STEPS of zero motion

GRIPPER_OPEN, GRIPPER_CLOSE = 0.0, 1.0       # the gym wrapper closes at >= 0.5
ARM_MODE, BASE_MODE = 0.0, 1.0               # ... and switches to base motion at >= 0.5
NAVIGATE, PICK, PLACE, EXECUTE = "navigate", "pick", "place", "execute"
BASE_REF = ("base", None)

# Stage -> what the executor moves and relative to what. Object keys are the task's object
# config names (the same in every kitchen); fixtures are kitchen env attributes. The correct
# and the wrong cutting tool depend on the food, so they are resolved from the scene.
STAGE_SPECS = {
    ("ArrangeBreadBasket", "cabinet_open"): {"kind": "fixture", "fixture": "cab"},
    ("ArrangeBreadBasket", "bread_in_basket"): {"kind": "pick_place", "object": "bread", "reference": ("object", "basket")},
    ("ArrangeBreadBasket", "basket_on_dining_counter"): {"kind": "pick_place", "object": "basket",
                                                         "reference": ("fixture", "dining_table")},
    ("ArrangeTea", "kettle_on_tray"): {"kind": "pick_place", "object": "obj2", "reference": ("object", "container")},
    ("ArrangeTea", "mug_on_tray"): {"kind": "pick_place", "object": "obj", "reference": ("object", "container")},
    ("ArrangeTea", "cabinet_closed"): {"kind": "fixture", "fixture": "cab"},
    ("BreadSelection", "croissant_on_cutting_board"): {"kind": "pick_place", "object": "croissant",
                                                       "reference": ("object", "cutting_board")},
    ("BreadSelection", "jam_on_cutting_board"): {"kind": "pick_place", "object": "jam",
                                                 "reference": ("object", "cutting_board")},
    ("CuttingToolSelection", "drawer_open"): {"kind": "fixture", "fixture": "drawer"},
    ("CuttingToolSelection", "correct_tool_on_cutting_board"): {"kind": "pick_place", "object": "@correct_tool",
                                                                "reference": ("object", "food_container")},
    ("CuttingToolSelection", "wrong_tool_not_on_cutting_board"): {"kind": "set_aside", "object": "@wrong_tool",
                                                                  "reference": ("object", "food_container")},
    ("GarnishPancake", "fridge_open"): {"kind": "fixture", "fixture": "fridge"},
    ("GarnishPancake", "pancake_on_plate"): {"kind": "pick_place", "object": "pancake",
                                             "reference": ("object", "pancake_container")},
    ("GarnishPancake", "plate_on_table"): {"kind": "pick_place", "object": "pancake_container",
                                           "reference": ("fixture", "dining_counter")},
    ("GarnishPancake", "strawberry_on_pancake"): {"kind": "pick_place", "object": "strawberry",
                                                  "reference": ("object", "pancake")},
}
CUTTING_TOOLS = ("knife", "peeler")
WRONG_TOOL_STAGE = ("CuttingToolSelection", "wrong_tool_not_on_cutting_board")
CORRECT_TOOL_STAGE = ("CuttingToolSelection", "correct_tool_on_cutting_board")


# ---- pose math ----------------------------------------------------------------------------------

def quat_xyzw_to_mat(q):
    x, y, z, w = np.asarray(q, dtype=float).reshape(4) / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, dtype=float).reshape(4)
    return quat_xyzw_to_mat([x, y, z, w])


def pose(position, rotation):
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotation, dtype=float)
    out[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return out


def inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def wrap(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def yaw_of(R):
    R = np.asarray(R)
    return float(math.atan2(R[1, 0], R[0, 0]))


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_frame(T):
    """The pose's position with only its heading: reference frames ignore tilt."""
    return pose(T[:3, 3], rot_z(yaw_of(T[:3, :3])))


def rotvec(R):
    R = np.asarray(R, dtype=float)
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = math.acos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:  # near pi, (R + I) / 2 = k k^T: the axis is its largest column
        B = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(B)))
        axis = B[:, i] / math.sqrt(max(B[i, i], 1e-12))
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    return axis * angle


def rot_from_vec(v):
    v = np.asarray(v, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def pose_error(current, target):
    """(position error in m, rotation error in rad) between two world poses."""
    return (float(np.linalg.norm(target[:3, 3] - current[:3, 3])),
            float(np.linalg.norm(rotvec(target[:3, :3] @ current[:3, :3].T))))


def interpolate(start, end, pos_step=APPROACH_POS_STEP, rot_step=APPROACH_ROT_STEP):
    """Poses from ``start`` (excluded) to ``end`` (included) within the per-step limits."""
    dist, angle = pose_error(start, end)
    n = max(1, int(math.ceil(max(dist / pos_step, angle / rot_step))))
    delta = rotvec(end[:3, :3] @ start[:3, :3].T)
    out = []
    for i in range(1, n + 1):
        s = i / n
        out.append(pose(start[:3, 3] + s * (end[:3, 3] - start[:3, 3]), rot_from_vec(s * delta) @ start[:3, :3]))
    return out


# ---- control (gains from probe-controls) --------------------------------------------------------

@dataclass
class Gains:
    """Motion per step per unit action: arm in the robot-base frame, base in the mobile-base frame.

    ``torso`` is the robot-base height change per unit torso action (action dim 10) in ``torso_mode``
    (``ARM_MODE`` or ``BASE_MODE``); 0 when the probe found none, and the torso is then left alone.
    """

    arm_pos: np.ndarray
    arm_rot: np.ndarray
    base: np.ndarray
    torso: float = 0.0
    torso_mode: float = ARM_MODE

    @classmethod
    def from_json(cls, data):
        mats = (np.asarray(data[k], dtype=float).reshape(3, 3) for k in ("arm_pos", "arm_rot", "base"))
        return cls(*mats, torso=float(data.get("torso", 0.0)), torso_mode=float(data.get("torso_mode", ARM_MODE)))

    def to_json(self):
        return {"arm_pos": self.arm_pos.tolist(), "arm_rot": self.arm_rot.tolist(), "base": self.base.tolist(),
                "torso": float(self.torso), "torso_mode": float(self.torso_mode)}

    def torso_command(self, dz, mode):
        """Torso action toward a base-height error ``dz`` in the given control mode (0 if unusable)."""
        if abs(self.torso) < TORSO_MIN_GAIN or mode != self.torso_mode or abs(dz) <= TORSO_TOL:
            return 0.0
        return float(np.clip(dz / self.torso, -1, 1))


def action_vector(*, arm_pos=None, arm_rot=None, gripper=GRIPPER_OPEN, base=None, mode=ARM_MODE):
    out = np.zeros(12)
    out[0:3] = 0 if arm_pos is None else arm_pos
    out[3:6] = 0 if arm_rot is None else arm_rot
    out[6] = gripper
    out[7:10] = 0 if base is None else base
    out[11] = mode
    return out


def arm_command(base_T, eef_T, target_T, gains, gripper):
    """Arm action that moves the end effector toward ``target_T`` (all world poses)."""
    Rb = base_T[:3, :3]
    dp = Rb.T @ (target_T[:3, 3] - eef_T[:3, 3])
    drot = rotvec((Rb.T @ target_T[:3, :3]) @ (Rb.T @ eef_T[:3, :3]).T)
    a_pos = np.clip(np.linalg.solve(gains.arm_pos, dp), -1, 1)
    a_rot = np.clip(np.linalg.solve(gains.arm_rot, drot), -1, 1)
    return action_vector(arm_pos=a_pos, arm_rot=a_rot, gripper=gripper, mode=ARM_MODE)


def base_command(mobile, goal, gains, gripper=GRIPPER_OPEN, torso=0.0):
    """Base action toward ``goal`` = (x, y, yaw) from ``mobile`` = (x, y, yaw), holonomic P-control."""
    x, y, yaw = mobile
    d_world = np.array([goal[0] - x, goal[1] - y])
    d_body = rot_z(-yaw)[:2, :2] @ d_world
    norm = float(np.linalg.norm(d_body))
    if norm > NAV_STEP_POS:
        d_body = d_body * (NAV_STEP_POS / norm)
    d_yaw = float(np.clip(wrap(goal[2] - yaw), -NAV_STEP_YAW, NAV_STEP_YAW))
    a = np.clip(np.linalg.solve(gains.base, np.array([d_body[0], d_body[1], d_yaw])), -1, 1)
    out = action_vector(gripper=gripper, base=a, mode=BASE_MODE)
    out[10] = torso
    return out


def navigation_done(mobile, goal, pos_tol=NAV_POS_TOL, cos_tol=NAV_COS_TOL):
    return bool(math.hypot(goal[0] - mobile[0], goal[1] - mobile[1]) <= pos_tol
                and math.cos(goal[2] - mobile[2]) >= cos_tol)


def estimate_gains(samples):
    """Least-squares 3x3 map from unit actions to measured per-step motion: samples = [(a, d)]."""
    A = np.array([s[0] for s in samples], dtype=float)
    D = np.array([s[1] for s in samples], dtype=float)
    M, *_ = np.linalg.lstsq(A, D, rcond=None)
    pred = A @ M
    ss_res = float(((D - pred) ** 2).sum())
    ss_tot = float(((D - D.mean(0)) ** 2).sum()) or 1e-12
    return M.T, 1.0 - ss_res / ss_tot


# ---- sources (demo segments) --------------------------------------------------------------------

@dataclass
class SourceStep:
    name: str
    kind: str
    start: int   # demo frames, end exclusive
    end: int
    lang: str = ""


@dataclass
class Source:
    """One demo segment that completes ``predicate``: per-frame world poses from ``start`` to ``end``."""

    task_name: str
    episode: int
    predicate: str
    target: str | None
    start: int
    end: int
    steps: list
    eef: np.ndarray            # (N, 4, 4) world
    base: np.ndarray           # (N, 4, 4) world, robot base
    gripper: np.ndarray        # (N,)
    objects: dict = field(default_factory=dict)   # name -> (N, 4, 4) world
    mode: np.ndarray | None = None                # (N,) the demo's control mode (>= 0.5: base motion)

    def base_mode(self, frame):
        return self.mode is not None and float(self.mode[self.at(frame)]) >= 0.5

    def at(self, frame):
        i = int(frame) - self.start
        if not 0 <= i < len(self.gripper):
            raise IndexError(f"frame {frame} outside source {self.task_name}:{self.episode} [{self.start}, {self.end})")
        return i

    @property
    def key(self):
        return (self.task_name, int(self.episode), self.predicate)


def step_kind(segment):
    if segment["name"] == es.NAVIGATE or segment["stage"] == NAVIGATE:
        return NAVIGATE
    if segment["stage"] in (PICK, PLACE):
        return segment["stage"]
    return EXECUTE


def source_steps(row, event_segment):
    lo, hi = int(event_segment["start"]), int(event_segment["end"])
    return [SourceStep(s["name"], step_kind(s), int(s["start"]), int(s["end"]), s.get("lang", ""))
            for s in row["segments"] if s["start"] >= lo and s["end"] <= hi]


def world_poses(named, frames):
    """Robot-base and end-effector world poses from the observation state (LeRobot convention, xyzw)."""
    base = np.stack([pose(named["base_position"][t], quat_xyzw_to_mat(named["base_rotation"][t])) for t in frames])
    rel = np.stack([pose(named["end_effector_position_relative"][t],
                         quat_xyzw_to_mat(named["end_effector_rotation_relative"][t])) for t in frames])
    return base, base @ rel


def build_source(row, predicate, *, named, actions, object_names, object_pos, object_quat):
    """A ``Source`` from an index row, its LeRobot states/actions and the index's object poses (wxyz)."""
    event = row["stage_events"][predicate]
    seg = event["segment"]
    target, _ = es.target_object(seg, row.get("object_lang"))
    frames = np.arange(int(seg["start"]), int(seg["end"]))
    base, eef = world_poses(named, frames)
    objects = {str(name): np.stack([pose(object_pos[t, j], quat_wxyz_to_mat(object_quat[t, j])) for t in frames])
               for j, name in enumerate(object_names)}
    return Source(task_name=row["task_name"], episode=int(row["episode"]), predicate=predicate, target=target,
                  start=int(seg["start"]), end=int(seg["end"]), steps=source_steps(row, seg), eef=eef, base=base,
                  gripper=np.asarray(actions[frames, 6], dtype=float), objects=objects,
                  mode=np.asarray(actions[frames, 11], dtype=float))


def usable_rows(rows, task_name, predicate):
    """Clean index rows whose demo completes ``predicate`` with a segment and matching frame counts."""
    out = []
    for row in rows:
        if row["task_name"] != task_name or not row["clean"] or row.get("rows_match") is False:
            continue
        event = (row.get("stage_events") or {}).get(predicate)
        if event and event.get("segment"):
            out.append(row)
    return sorted(out, key=lambda r: int(r["episode"]))


class SourceLibrary:
    """Lazily built ``Source`` objects from the R1 index and the LeRobot demos (simulator-free)."""

    def __init__(self, index_dir, lerobots, *, reader=None):
        self.index_dir = Path(index_dir)
        self.rows, _ = es.load_index(index_dir)
        self.lerobots = {k: Path(v) for k, v in lerobots.items()}
        self._reader = reader
        self._cache = {}
        self._episodes = {}

    def _episode(self, task, episode):
        key = (task, int(episode))
        if key not in self._episodes:
            if self._reader is not None:
                named, actions = self._reader(task, episode)
            else:
                from robocasa.recovery.xr1_samples import read_lerobot_arrays

                named, actions = read_lerobot_arrays(self.lerobots[task], episode)
            path = self.index_dir / es.EPISODES_DIR / task / f"episode_{int(episode):06d}.npz"
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in ("object_names", "object_pos", "object_quat")}
            if len(actions) != arrays["object_pos"].shape[0]:
                raise ValueError(f"{task} episode {episode}: {len(actions)} LeRobot rows, "
                                 f"{arrays['object_pos'].shape[0]} replayed frames")
            self._episodes[key] = (named, actions, arrays)
        return self._episodes[key]

    def sources(self, task_name, predicate, target=None):
        out = []
        for row in usable_rows(self.rows, task_name, predicate):
            seg = row["stage_events"][predicate]["segment"]
            if target is not None and es.target_object(seg, row.get("object_lang"))[0] != target:
                continue
            key = (task_name, int(row["episode"]), predicate)
            if key not in self._cache:
                named, actions, arrays = self._episode(task_name, row["episode"])
                self._cache[key] = build_source(row, predicate, named=named, actions=actions,
                                                object_names=[str(n) for n in arrays["object_names"]],
                                                object_pos=arrays["object_pos"], object_quat=arrays["object_quat"])
            out.append(self._cache[key])
        return out

    def named_state(self, task_name, episode, frame):
        named, _, _ = self._episode(task_name, episode)
        return {k: np.asarray(v[int(frame)]) for k, v in named.items()}


# ---- R2a: what to run at a state (simulator-free) -----------------------------------------------

def stage_spec(task_name, predicate):
    if ladder.is_release_predicate(predicate):
        return {"kind": "release"}
    spec = STAGE_SPECS.get((task_name, predicate))
    if spec is None:
        raise KeyError(f"no R2 stage spec for {task_name}:{predicate}")
    return spec


def resolve_object(key, facts):
    if key == "@correct_tool":
        return facts["correct_tool"]
    if key == "@wrong_tool":
        return next(t for t in CUTTING_TOOLS if t != facts["correct_tool"])
    return key


def step_reference(kind, spec, target):
    """Frame a step's poses are replayed in: the target for a pick, the receptacle for a place, else the base."""
    if kind == PICK and target is not None:
        return ("object", target)
    if kind == PLACE and spec.get("reference", (None,))[0] == "object":
        return tuple(spec["reference"])
    return BASE_REF


def navigation_reference(kind, spec, target):
    """What a navigate phase drives to: the fixture holding the step's object, or the stage's fixture."""
    if spec["kind"] == "fixture":
        return ("fixture", spec["fixture"])
    if kind == PICK:
        return ("object", target)
    return tuple(spec["reference"])


def relative_pose(base_T, obj_T):
    """Object pose in the robot-base frame as (position, yaw)."""
    rel = inverse(base_T) @ obj_T
    return rel[:3, 3], yaw_of(rel[:3, :3])


def rank_distance(src_rel, alarm_rel, yaw_weight=RANK_YAW_WEIGHT):
    (p1, y1), (p2, y2) = src_rel, alarm_rel
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)) + yaw_weight * abs(wrap(y1 - y2)))


def first_step_index(source, held):
    """0, or the step after the pick when the target is already held."""
    if not held:
        return 0
    picks = [i for i, s in enumerate(source.steps) if s.kind == PICK]
    return picks[-1] + 1 if picks else 0


def rank_sources(sources, facts, start_of, reference_of):
    """Sources ordered by how close their first step's reference sits to the robot, versus the alarm state."""
    scored = []
    for src in sources:
        i0 = start_of(src)
        manip = [s for s in src.steps[i0:] if s.kind != NAVIGATE]
        ref = reference_of(manip[0]) if manip else BASE_REF
        if ref[0] != "object" or ref[1] not in src.objects or ref[1] not in facts["objects"]:
            scored.append((float("inf"), src))
            continue
        f = src.at(manip[0].start)
        d = rank_distance(relative_pose(src.base[f], src.objects[ref[1]][f]),
                          relative_pose(facts["base"], facts["objects"][ref[1]]))
        scored.append((d, src))
    scored.sort(key=lambda x: (x[0], x[1].episode))
    return scored


def plan_attempts(task_name, predicate, facts, library, n=ATTEMPTS_PER_STATE):
    """R2a: ``n`` programs for one state, from distinct sources where available (cycled otherwise)."""
    spec = stage_spec(task_name, predicate)
    if spec["kind"] == "release":
        return [{"k": k, "source": None, "rank_distance": None, "phases": [{"phase": "release"}]} for k in range(n)]
    if spec["kind"] == "set_aside":
        wrong = resolve_object(spec["object"], facts)
        pick_sources = library.sources(*CORRECT_TOOL_STAGE, target=wrong)
        if not pick_sources:
            raise ValueError(f"no pick segments of the {wrong} for the set-aside")
        prefix = [{"phase": "release"}] if facts["held_any"] and wrong not in facts["held_task"] else []
        ranked = rank_sources(pick_sources, facts, lambda s: 0, lambda st: ("object", wrong))
        programs = []
        for k in range(n):
            dist, src = ranked[k % len(ranked)]
            programs.append({"k": k, "source": list(src.key), "rank_distance": dist,
                             "phases": prefix + set_aside_phases(src, wrong, spec, facts)})
        return programs
    target = resolve_object(spec.get("object"), facts) if spec.get("object") else None
    held = target is not None and target in facts["held_task"]
    sources = library.sources(task_name, predicate, target=target)
    if not sources:
        raise ValueError(f"no sources for {task_name}:{predicate} target {target}")
    prefix = []
    if facts["held_any"] and not held:
        prefix.append({"phase": "release"})
    wrong_on_board = ((task_name, predicate) == CORRECT_TOOL_STAGE
                      and facts["predicates"].get(WRONG_TOOL_STAGE[1]) is False)
    start_of = lambda s: first_step_index(s, held)  # noqa: E731
    ranked = rank_sources(sources, facts, start_of, lambda st: step_reference(st.kind, spec, target))
    programs = []
    for k in range(n):
        dist, src = ranked[k % len(ranked)]
        phases = list(prefix)
        if wrong_on_board and not held:
            wrong = resolve_object("@wrong_tool", facts)
            tool_sources = library.sources(*CORRECT_TOOL_STAGE, target=wrong)
            if tool_sources:
                _, tool_src = rank_sources(tool_sources, facts, lambda s: 0, lambda st: ("object", wrong))[0]
                phases += set_aside_phases(tool_src, wrong, STAGE_SPECS[WRONG_TOOL_STAGE], facts)
        phases += stage_phases(src, spec, target, start_of(src), navigate_first=dist > NAV_TRIGGER)
        if wrong_on_board and held:
            wrong = resolve_object("@wrong_tool", facts)
            tool_sources = library.sources(*CORRECT_TOOL_STAGE, target=wrong)
            if tool_sources:
                _, tool_src = rank_sources(tool_sources, facts, lambda s: 0, lambda st: ("object", wrong))[0]
                phases += [{"phase": "release"}] + set_aside_phases(tool_src, wrong, STAGE_SPECS[WRONG_TOOL_STAGE],
                                                                     facts)
        programs.append({"k": k, "source": list(src.key), "rank_distance": dist, "target": target, "held": held,
                         "wrong_tool_on_board": bool(wrong_on_board), "phases": phases})
    return programs


def stage_phases(source, spec, target, first, *, navigate_first=False, navigate_to_fixture=True):
    """Phases for a source's steps from index ``first``: scripted navigation, tracked manipulation.

    ``navigate_to_fixture=False`` (self-replay) keeps the demo's own base pose for a fixture stage.
    """
    phases = []
    if spec["kind"] == "fixture" and navigate_to_fixture:
        phases.append({"phase": "navigate", "reference": ["fixture", spec["fixture"]], "if_needed": True})
    elif navigate_first:
        manip = [s for s in source.steps[first:] if s.kind != NAVIGATE]
        kind = manip[0].kind if manip else PICK
        phases.append({"phase": "navigate", "reference": list(navigation_reference(kind, spec, target)),
                       "if_needed": False})
    for i, step in enumerate(source.steps[first:], start=first):
        if step.kind == NAVIGATE:
            nxt = next((s for s in source.steps[i + 1:] if s.kind != NAVIGATE), None)
            kind = nxt.kind if nxt else PLACE
            phases.append({"phase": "navigate", "reference": list(navigation_reference(kind, spec, target)),
                           "if_needed": True})
        else:
            phases.append({"phase": "track", "source": list(source.key), "step": i, "kind": step.kind,
                           "reference": list(step_reference(step.kind, spec, target))})
    return phases


def set_aside_phases(source, tool, spec, facts):
    """Pick the wrong tool with a pick segment of that tool, then put it beside the board."""
    first_pick = next((i for i, s in enumerate(source.steps) if s.kind == PICK), None)
    if first_pick is None:
        raise ValueError(f"source {source.key} has no pick step")
    phases = []
    if tool not in facts["held_task"]:
        phases.append({"phase": "track", "source": list(source.key), "step": first_pick, "kind": PICK,
                       "reference": ["object", tool]})
    phases.append({"phase": "set_aside", "object": tool, "board": spec["reference"][1]})
    return phases


def program_cap(program, library_sources):
    """Expert step cap: executed source frames x (1 + slack); navigation is capped per phase.

    Scripted release and set-aside phases count as 60 frames each.
    """
    frames = 0
    for ph in program["phases"]:
        if ph["phase"] == "track":
            src = library_sources[tuple(ph["source"])]
            step = src.steps[ph["step"]]
            frames += step.end - step.start
        elif ph["phase"] in ("release", "set_aside"):
            frames += 60
    return int(math.ceil(frames * (1 + SEGMENT_SLACK)))


# ---- R2b: the executor (world adapter: real simulator or a test stand-in) -----------------------

def toward_base(obs):
    """Horizontal unit vector from the end effector back toward the robot base."""
    d = obs["base"][:3, 3] - obs["eef"][:3, 3]
    d[2] = 0.0
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-6 else -obs["base"][:3, 0] * np.array([1.0, 1.0, 0.0])


class StageWatch:
    """True once ``predicate`` has held for ``HOLD_STEPS`` consecutive steps."""

    def __init__(self, predicate, hold=HOLD_STEPS):
        self.predicate, self.hold, self.run = predicate, int(hold), 0

    def update(self, predicates):
        self.run = self.run + 1 if predicates.get(self.predicate) else 0
        return self.run >= self.hold


class ExpertStop(Exception):
    pass


class ExpertRun:
    """Runs a program's phases on ``world``; stops at the stage, the step cap or the budget."""

    def __init__(self, world, gains, sources, predicate, *, cap, budget):
        self.world, self.gains, self.sources = world, gains, sources
        self.watch = StageWatch(predicate) if predicate else None
        self.cap, self.budget = int(cap), int(budget)
        self.steps, self.nav_steps, self.completed, self.success = 0, 0, False, False
        self.log = []

    def act(self, action, navigating=False):
        if self.steps >= self.budget or (not navigating and self.steps - self.nav_steps >= self.cap):
            raise ExpertStop("cap")
        info = self.world.step(action)
        self.steps += 1
        if info.get("done"):
            raise ExpertStop("env_done")
        if navigating:
            self.nav_steps += 1
        self.success = bool(info.get("task_success"))
        if self.watch is not None and self.watch.update(info.get("predicates") or {}):
            self.completed = True
            raise ExpertStop("stage")
        if self.success:
            raise ExpertStop("task_success")
        return info

    def move_to(self, target, gripper, *, settle=True):
        obs = self.world.observe()
        for waypoint in interpolate(obs["eef"], target):
            obs = self.world.observe()
            self.act(arm_command(obs["base"], obs["eef"], waypoint, self.gains, gripper))
        if settle:
            for _ in range(SETTLE_STEPS):
                obs = self.world.observe()
                if pose_error(obs["eef"], target)[0] < SETTLE_TOL:
                    break
                self.act(arm_command(obs["base"], obs["eef"], target, self.gains, gripper))

    def hold_gripper(self, gripper, steps):
        for _ in range(int(steps)):
            obs = self.world.observe()
            self.act(arm_command(obs["base"], obs["eef"], obs["eef"], self.gains, gripper))

    def release(self):
        self.hold_gripper(GRIPPER_OPEN, RELEASE_OPEN_STEPS)
        obs = self.world.observe()
        up = obs["eef"].copy()
        up[2, 3] += RELEASE_LIFT
        self.move_to(up, GRIPPER_OPEN)
        obs = self.world.observe()
        back = obs["eef"].copy()
        back[:3, 3] += toward_base(obs) * RELEASE_BACKOFF
        self.move_to(back, GRIPPER_OPEN)

    def navigate(self, reference, if_needed):
        goal = self.world.navigation_goal(tuple(reference))
        if goal is None:
            return {"phase": "navigate", "skipped": "no_goal"}
        mobile = self.world.observe()["mobile"]
        if if_needed and navigation_done(mobile, goal):
            return {"phase": "navigate", "skipped": "already_there"}
        start = self.steps
        for _ in range(NAVIGATE_MAX_STEPS):
            mobile = self.world.observe()["mobile"]
            if navigation_done(mobile, goal, NAV_FINE_POS, math.cos(NAV_FINE_YAW)):
                break
            self.act(base_command(mobile, goal, self.gains), navigating=True)
        mobile = self.world.observe()["mobile"]
        return {"phase": "navigate", "steps": self.steps - start, "reached": navigation_done(mobile, goal),
                "goal": [float(v) for v in goal]}

    def track(self, source, step_index, reference):
        step = source.steps[step_index]
        i0 = source.at(step.start)
        obs = self.world.observe()
        if reference[0] == "object":
            src_ref = yaw_frame(source.objects[reference[1]][i0])
            here_ref = yaw_frame(obs["objects"][reference[1]])
        else:
            src_ref, here_ref = yaw_frame(source.base[i0]), yaw_frame(obs["base"])
        transform = here_ref @ inverse(src_ref)
        frames = range(step.start, step.end)
        targets = [transform @ source.eef[source.at(f)] for f in frames]
        bases = [transform @ source.base[source.at(f)] for f in frames]
        grippers = [float(source.gripper[source.at(f)]) for f in frames]
        driving = [source.base_mode(f) for f in frames]
        start, base_frames = self.steps, 0
        i = 0
        while i < len(targets):
            obs = self.world.observe()
            dz = bases[i][2, 3] - obs["base"][2, 3]
            if driving[i]:
                # The demonstrator moved the base here: move the robot base by the same planar offset.
                d = bases[i][:3, 3] - obs["base"][:3, 3]
                d_yaw = wrap(yaw_of(bases[i][:3, :3]) - yaw_of(obs["base"][:3, :3]))
                m = obs["mobile"]
                goal = (m[0] + d[0], m[1] + d[1], m[2] + d_yaw)
                self.act(base_command(m, goal, self.gains, grippers[i],
                                      torso=self.gains.torso_command(dz, BASE_MODE)))
                base_frames += 1
                obs = self.world.observe()
                d = bases[i][:3, 3] - obs["base"][:3, 3]
                d_yaw = wrap(yaw_of(bases[i][:3, :3]) - yaw_of(obs["base"][:3, :3]))
                if math.hypot(d[0], d[1]) < TRACK_TOL and abs(d_yaw) < TRACK_YAW_TOL:
                    i += 1
                continue
            if (i == 0 or driving[i - 1]) and pose_error(obs["eef"], targets[i])[0] >= TRACK_TOL:
                self.move_to(targets[i], grippers[i])   # approach the first arm frame of a run
                obs = self.world.observe()
            action = arm_command(obs["base"], obs["eef"], targets[i], self.gains, grippers[i])
            action[10] = self.gains.torso_command(dz, ARM_MODE)
            self.act(action)
            obs = self.world.observe()
            if pose_error(obs["eef"], targets[i])[0] < TRACK_TOL:
                i += 1
        return {"phase": "track", "kind": step.kind, "steps": self.steps - start, "frames": len(targets),
                "base_mode_steps": base_frames}

    def set_aside(self, tool, board):
        obs = self.world.observe()
        up = obs["eef"].copy()
        up[2, 3] += SET_ASIDE_LIFT
        self.move_to(up, GRIPPER_CLOSE)
        obs = self.world.observe()
        back = toward_base(obs)
        lateral = np.array([-back[1], back[0], 0.0])   # horizontal, perpendicular to the reach
        away = np.sign(float(lateral @ (obs["eef"][:3, 3] - obs["objects"][board][:3, 3]))) or 1.0
        side = obs["eef"].copy()
        side[:3, 3] += away * lateral * SET_ASIDE_SIDE
        self.move_to(side, GRIPPER_CLOSE)
        down = side.copy()
        down[2, 3] -= SET_ASIDE_LOWER
        self.move_to(down, GRIPPER_CLOSE)
        self.release()
        return {"phase": "set_aside", "object": tool}

    def run(self, phases):
        stop = None
        try:
            for ph in phases:
                if ph["phase"] == "release":
                    self.release()
                    self.log.append({"phase": "release"})
                elif ph["phase"] == "navigate":
                    self.log.append(self.navigate(ph["reference"], ph.get("if_needed", True)))
                elif ph["phase"] == "track":
                    self.log.append(self.track(self.sources[tuple(ph["source"])], ph["step"], ph["reference"]))
                elif ph["phase"] == "set_aside":
                    self.log.append(self.set_aside(ph["object"], ph["board"]))
                else:
                    raise ValueError(f"unknown phase {ph['phase']}")
            stop = "program_end"
        except ExpertStop as exc:
            stop = str(exc)
        return {"stop": stop, "expert_steps": self.steps, "navigation_steps": self.nav_steps,
                "stage_completed": self.completed, "task_success": self.success, "phases": self.log}


# ---- checks and Gate R (simulator-free) ---------------------------------------------------------

def self_replay_summary(records, *, min_demos=SELF_REPLAY_DEMOS, threshold=SELF_REPLAY_MIN):
    by = defaultdict(list)
    for r in records:
        by[(r["task_name"], r["predicate"])].append(r)
    stages, failing = [], []
    for (task, pred), rows in sorted(by.items()):
        done = sum(bool(r["stage_completed"]) for r in rows)
        conv = [r["convention_residual"] for r in rows if r.get("convention_residual") is not None]
        row = {"task_name": task, "predicate": pred, "demos": len(rows), "completed": done,
               "rate": round(done / len(rows), 3), "enough_demos": len(rows) >= min_demos,
               "max_convention_residual": max(conv) if conv else None,
               "errors": sum(bool(r.get("error")) for r in rows)}
        row["passed"] = row["enough_demos"] and row["rate"] >= threshold
        stages.append(row)
        if not row["passed"]:
            failing.append(f"{task}:{pred}")
    return {"kind": "r2_self_replay", "threshold": threshold, "min_demos": min_demos, "stages": stages,
            "failing": failing, "passed": not failing and bool(stages)}


def transfer_summary(records, *, threshold=TRANSFER_MIN):
    """Per stage: completion among the runs that reached it (stages run in the ladder's order)."""
    by = defaultdict(lambda: [0, 0])
    for r in records:
        for st in r["stages"]:
            by[(r["task_name"], st["predicate"])][0] += 1
            by[(r["task_name"], st["predicate"])][1] += int(bool(st["stage_completed"]))
    stages, failing = [], []
    for (task, pred), (reached, done) in sorted(by.items()):
        rate = done / reached if reached else 0.0
        row = {"task_name": task, "predicate": pred, "reached": reached, "completed": done, "rate": round(rate, 3),
               "passed": reached > 0 and rate >= threshold}
        stages.append(row)
        if not row["passed"]:
            failing.append(f"{task}:{pred}")
    return {"kind": "r2_pretrain_transfer", "threshold": threshold, "runs": len(records), "stages": stages,
            "failing": failing, "passed": not failing and bool(stages)}


def gate_r(attempts, states, *, threshold=GATE_R_MIN_SHARE):
    """Share of class-2 states with >= 1 attempt ending in full-task success, plus the two attempt rates."""
    by = defaultdict(list)
    for a in attempts:
        by[a["parent_id"]].append(a)
    per_state = []
    for s in states:
        rows = by.get(s["parent_id"], [])
        per_state.append({"parent_id": s["parent_id"], "task_name": s["task_name"], "stage": s["stage"],
                          "attempts": len(rows), "expert_completed": sum(bool(r["stage_completed"]) for r in rows),
                          "successes": sum(bool(r["success"]) for r in rows),
                          "errors": sum(bool(r.get("error")) for r in rows)})
    n = len(per_state)
    hit = sum(p["successes"] > 0 for p in per_state)
    total = sum(p["attempts"] for p in per_state)
    by_task = defaultdict(lambda: [0, 0])
    for p in per_state:
        by_task[p["task_name"]][0] += 1
        by_task[p["task_name"]][1] += int(p["successes"] > 0)
    share = hit / n if n else 0.0
    return {
        "kind": "r2_gate_r", "threshold": threshold, "class2_states": n, "states_with_success": hit,
        "share": round(share, 4), "passed": bool(n and share >= threshold),
        "attempts": total,
        "expert_rate": round(sum(p["expert_completed"] for p in per_state) / total, 4) if total else None,
        "full_rate": round(sum(p["successes"] for p in per_state) / total, 4) if total else None,
        "by_task": {t: {"states": v[0], "with_success": v[1]} for t, v in sorted(by_task.items())},
        "states": per_state,
    }


# ---- recording attempts as training data (simulator-free) --------------------------------------

RECORD_FPS = 20               # the environment's control rate, as the LeRobot demos
RECORD_QUALITY = 8            # imageio/ffmpeg quality (0-10)
PHASES = ("expert", "policy")
ENV_ACTION_KEYS = ("action.end_effector_position", "action.end_effector_rotation", "action.gripper_close",
                   "action.base_motion", "action.control_mode")
BINARY_DIMS = (6, 11)         # gripper and control mode: stored as -1/1, the demos' raw robosuite convention
MODALITY_PATH = Path(__file__).resolve().parents[1] / "models/assets/groot_dataset_assets/PandaOmron_modality.json"
RECORDING_NAME = "frames.npz"
ATTEMPT_META = "attempt.json"


def executed_action(action):
    """12-d environment-order action from a policy dict or vector, binary dims as the env executes them."""
    if isinstance(action, dict):
        vec = np.concatenate([np.asarray(action[k], dtype=float).reshape(-1) for k in ENV_ACTION_KEYS])
    else:
        vec = np.asarray(action, dtype=float).reshape(-1).copy()
    if vec.shape != (12,):
        raise ValueError(f"expected a 12-d action, got {vec.shape}")
    for i in BINARY_DIMS:
        vec[i] = 1.0 if vec[i] >= 0.5 else -1.0
    return vec


def _imageio_writer(path, fps):
    import imageio

    return imageio.get_writer(str(path), fps=fps, quality=RECORD_QUALITY)


class AttemptRecorder:
    """Writes one attempt as XR-1 saw it: per-camera videos, named states, executed actions, phases.

    Frame t is the observation before action t. The folder is written as ``<name>.partial`` and
    renamed on ``finish``; a partial folder left by a crash is replaced.
    """

    def __init__(self, folder, *, writer_factory=_imageio_writer, fps=RECORD_FPS):
        import shutil

        from robocasa.recovery.xr1_samples import STATE_KEYS
        from robocasa.recovery.xiaomi_robotics_1_policy import CAMERA_KEYS

        self.folder = Path(folder)
        self.partial = self.folder.with_name(self.folder.name + ".partial")
        if self.folder.exists():
            raise FileExistsError(f"{self.folder} is already recorded")
        if self.partial.exists():
            shutil.rmtree(self.partial)
        self.partial.mkdir(parents=True)
        self.camera_keys, self.state_keys = tuple(CAMERA_KEYS), tuple(STATE_KEYS)
        self.writers = {k: writer_factory(self.partial / f"{k}.mp4", fps) for k in self.camera_keys}
        self.states = {k: [] for k in self.state_keys}
        self.actions, self.phases = [], []
        self.suspect_frames = 0

    def add(self, obs, action, phase):
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        for key in self.camera_keys:
            frame = np.ascontiguousarray(np.asarray(obs[key])[..., :3], dtype=np.uint8)
            self.writers[key].append_data(frame)
        for key in self.state_keys:
            self.states[key].append(np.asarray(obs[f"state.{key}"], dtype=np.float32).reshape(-1))
        self.actions.append(executed_action(action).astype(np.float32))
        self.phases.append(PHASES.index(phase))

    def __len__(self):
        return len(self.actions)

    def _close(self):
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception:
                pass

    def finish(self, meta):
        self._close()
        n = len(self.actions)
        np.savez_compressed(self.partial / RECORDING_NAME, actions=np.stack(self.actions) if n else np.zeros((0, 12)),
                            phase=np.asarray(self.phases, dtype=np.int8),
                            **{f"state.{k}": np.stack(v) if n else np.zeros((0, 0)) for k, v in self.states.items()})
        counts = Counter(PHASES[i] for i in self.phases)
        ladder.atomic_write_json(self.partial / ATTEMPT_META, {
            **meta, "frames": n, "frames_by_phase": {p: counts.get(p, 0) for p in PHASES},
            "camera_keys": list(self.camera_keys), "fps": RECORD_FPS, "binary_dims_convention": "-1/1",
            "created_at": ladder.utc_now()})
        self.partial.rename(self.folder)
        return n

    def abort(self):
        import shutil

        self._close()
        shutil.rmtree(self.partial, ignore_errors=True)


def recorded_attempts(gate_root, *, include="success"):
    """(attempt record, recording folder) pairs from every shard's attempts.jsonl."""
    out = []
    for path in sorted(Path(gate_root).glob("**/attempts.jsonl")):
        for row in ladder.read_jsonl(path):
            if not row.get("recording") or row.get("error"):
                continue
            if include == "success" and not row.get("success"):
                continue
            folder = path.parent / row["recording"]
            if (folder / ATTEMPT_META).exists():
                out.append((row, folder))
    return sorted(out, key=lambda x: (x[0]["task_name"], x[0]["parent_id"], int(x[0]["k"])))


def export_lerobot(gate_root, output_dir, *, include="success", copy=True, log=print):
    """Recorded attempts -> one LeRobot folder per task, readable by the Step 0 sample builder.

    Columns: ``observation.state`` and ``action`` in the demos' layout (meta/modality.json is the
    PandaOmron one the demos ship), the instruction index, frame/episode indices, and
    ``annotation.r2.phase`` (0 expert, 1 policy) so training can filter frames.
    """
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from robocasa.recovery import xr1_samples as xs

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty")
    modality = json.loads(MODALITY_PATH.read_text())
    layout = xs.lerobot_layout(modality)
    state_dim = max(hi for _, hi in layout["state"].values())
    by_task = defaultdict(list)
    for row, folder in recorded_attempts(gate_root, include=include):
        by_task[row["task_name"]].append((row, folder))
    info_paths = {"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    summary = {}
    for task, items in sorted(by_task.items()):
        root = output_dir / task / "lerobot"
        (root / "meta").mkdir(parents=True)
        tasks, episodes, index = {}, [], 0
        for ep, (row, folder) in enumerate(items):
            meta = json.loads((folder / ATTEMPT_META).read_text())
            with np.load(folder / RECORDING_NAME, allow_pickle=False) as data:
                actions = data["actions"]
                phase = data["phase"]
                named = {k: data[f"state.{k}"] for k in xs.STATE_KEYS}
            n = len(actions)
            if n != meta["frames"] or any(len(v) != n for v in named.values()):
                raise ValueError(f"{folder}: frame counts differ")
            states = np.zeros((n, state_dim), dtype=np.float32)
            for key in xs.STATE_KEYS:
                states[:, slice(*layout["state"][key])] = named[key]
            lerobot_actions = np.zeros((n, 12), dtype=np.float32)
            start = 0
            for key, width in xs.ENV_ACTION_ORDER:
                lerobot_actions[:, slice(*layout["action"][key])] = actions[:, start:start + width]
                start += width
            instruction = meta["instruction"]
            task_index = tasks.setdefault(instruction, len(tasks))
            chunk = ep // 1000
            data_path = root / info_paths["data_path"].format(episode_chunk=chunk, episode_index=ep)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({
                "observation.state": [list(map(float, r)) for r in states],
                "action": [list(map(float, r)) for r in lerobot_actions],
                "annotation.human.task_description": [task_index] * n,
                "task_index": [task_index] * n,
                "annotation.r2.phase": phase.astype(np.int64).tolist(),
                "timestamp": (np.arange(n) / RECORD_FPS).tolist(),
                "frame_index": list(range(n)),
                "episode_index": [ep] * n,
                "index": list(range(index, index + n)),
            }), data_path)
            index += n
            for key in meta["camera_keys"]:
                dst = root / info_paths["video_path"].format(episode_chunk=chunk, video_key=layout["video"][key],
                                                             episode_index=ep)
                dst.parent.mkdir(parents=True, exist_ok=True)
                (shutil.copy2 if copy else os_link)(folder / f"{key}.mp4", dst)
            episodes.append({"episode_index": ep, "tasks": [instruction], "length": n,
                             "r2": {k: row.get(k) for k in ("parent_id", "k", "stage", "source", "success",
                                                             "stage_completed", "expert_steps", "policy_steps")},
                             "frames_by_phase": meta["frames_by_phase"]})
        (root / "meta" / "modality.json").write_text(json.dumps(modality, indent=2))
        (root / "meta" / "tasks.jsonl").write_text(
            "".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in sorted(tasks.items(), key=lambda x: x[1])))
        (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
        ladder.atomic_write_json(root / "meta" / "info.json", {
            "fps": RECORD_FPS, "chunks_size": 1000, **info_paths, "total_episodes": len(episodes),
            "total_frames": index, "source": "r2_gate_r_recordings", "include": include})
        summary[task] = {"episodes": len(episodes), "frames": index, "folder": str(root)}
        log(f"[export] {task}: {len(episodes)} episodes, {index} frames -> {root}")
    ladder.atomic_write_json(output_dir / "export.json", {
        "kind": "r2_lerobot_export", "gate_root": str(Path(gate_root).resolve()), "include": include,
        "tasks": summary, "git_commit": ladder.current_git_commit(), "created_at": ladder.utc_now()})
    return summary


def os_link(src, dst):
    import os

    os.link(src, dst)


# ---- simulator side -----------------------------------------------------------------------------

def _kitchen(env):
    from robocasa.recovery.failure_audit import _robocasa_env

    return _robocasa_env(env)


def gripper_holds_any(kitchen):
    """Both finger pads touch the same non-robot body (R0's ``holding`` lists task objects only)."""
    sim = kitchen.sim
    gripper = kitchen.robots[0].gripper
    gripper = gripper["right"] if isinstance(gripper, dict) else gripper
    pads = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
    pad_ids = [{sim.model.geom_name2id(g) for g in side} for side in pads]
    robot_prefixes = ("robot0_", "gripper0_", "mobilebase0_")
    touched = [set(), set()]
    for c in sim.data.contact[: sim.data.ncon]:
        for mine, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            for side, ids in enumerate(pad_ids):
                if mine in ids:
                    body = sim.model.body_id2name(sim.model.geom_bodyid[other]) or ""
                    if not body.startswith(robot_prefixes):
                        touched[side].add(body)
    common = sorted(touched[0] & touched[1])
    return bool(common), common


def fixture_holding(kitchen, position):
    """The fixture an object is in or on.

    First the smallest fixture whose box contains the point (fridge, cabinet, drawer, microwave),
    else the smallest ground fixture under it (counter, stove, housing cabinet, fridge), the same
    kinds ``compute_robot_base_placement_pose`` stands the robot in front of. None if neither.
    """
    from robocasa.models.fixtures import Counter, Fridge, HousingCabinet, Stove, Stovetop
    from robocasa.utils import object_utils as OU

    ground = (Counter, Stove, Stovetop, HousingCabinet, Fridge)
    for only_2d in (False, True):
        best = []
        for fxtr in kitchen.fixtures.values():
            name = str(getattr(fxtr, "name", "")).lower()
            if any(word in name for word in ("wall", "floor", "accessor")):
                continue
            if only_2d and not isinstance(fxtr, ground):
                continue
            try:
                inside = OU.point_in_fixture(point=position, fixture=fxtr, only_2d=only_2d)
            except Exception:
                continue
            if inside:
                size = np.prod(np.asarray(getattr(fxtr, "size", [1, 1, 1]), dtype=float))
                best.append((float(size), fxtr))
        if best:
            return min(best, key=lambda x: x[0])[1]
    return None


def slide_to(goal, fixture_xy, target_xy):
    """Move a placement goal sideways (parallel to the fixture front) until it faces ``target_xy``.

    ``compute_robot_base_placement_pose`` without a reference object centres the robot on the
    fixture; with one it uses the object's initial placement, not where it is now.
    """
    x, y, yaw = goal
    facing = np.asarray(fixture_xy, dtype=float) - np.array([x, y])
    n = float(np.linalg.norm(facing))
    if n < 1e-6:
        return goal
    facing /= n
    lateral = np.array([-facing[1], facing[0]])
    t = float(lateral @ (np.asarray(target_xy, dtype=float) - np.array([x, y])))
    return (x + t * lateral[0], y + t * lateral[1], yaw)


class SimWorld:
    """World adapter on a RoboCasa gym env: poses in the observation's convention, 12-d actions."""

    def __init__(self, env, recorder=None):
        self.env = env
        self.kitchen = _kitchen(env)
        self.obs = ladder.live_observation(env)
        self.info = {}
        self.recorder = recorder

    def observe(self):
        o = self.obs
        base = pose(o["state.base_position"], quat_xyzw_to_mat(o["state.base_rotation"]))
        eef = base @ pose(o["state.end_effector_position_relative"],
                          quat_xyzw_to_mat(o["state.end_effector_rotation_relative"]))
        sim = self.kitchen.sim
        mid = sim.model.body_name2id("mobilebase0_base")
        R = np.asarray(sim.data.body_xmat[mid]).reshape(3, 3)
        mobile = (float(sim.data.body_xpos[mid][0]), float(sim.data.body_xpos[mid][1]), yaw_of(R))
        objects = {}
        for name in self.kitchen.objects:
            bid = self.kitchen.obj_body_id[name]
            objects[str(name)] = pose(sim.data.body_xpos[bid], quat_wxyz_to_mat(sim.data.body_xquat[bid]))
        return {"base": base, "eef": eef, "mobile": mobile, "objects": objects}

    def step(self, action):
        from robocasa.recovery.openpi_websocket_policy import _convert_action
        from robocasa.recovery.recovery_rollout import _is_task_success, _step_env
        from robocasa.recovery.subtask_eval import get_subtask_eval

        if self.recorder is not None:
            self.recorder.add(self.obs, action, "expert")
        self.obs, reward, done, info = _step_env(self.env, _convert_action(action))
        payload = (info or {}).get("subtask_eval") or get_subtask_eval(self.env) or {}
        predicates = {k: bool((v or {}).get("value")) for k, v in (payload.get("predicates") or {}).items()}
        self.info = {"predicates": predicates, "task_success": bool(_is_task_success(info, reward=reward, env=self.env)),
                     "done": bool(done), "subtask_eval": payload}
        return self.info

    def predicates(self):
        from robocasa.recovery.subtask_eval import get_subtask_eval

        payload = get_subtask_eval(self.env) or {}
        return {k: bool((v or {}).get("value")) for k, v in (payload.get("predicates") or {}).items()}

    def navigation_goal(self, reference):
        from robocasa.utils.env_utils import compute_robot_base_placement_pose

        kind, name = reference
        target = None
        if kind == "fixture":
            fxtr = getattr(self.kitchen, name, None)
        else:
            bid = self.kitchen.obj_body_id[name]
            target = np.asarray(self.kitchen.sim.data.body_xpos[bid], dtype=float)
            fxtr = fixture_holding(self.kitchen, target)
        if fxtr is None:
            return None
        pos, ori = compute_robot_base_placement_pose(self.kitchen, fxtr)
        goal = (float(pos[0]), float(pos[1]), float(ori[2]))
        if target is not None:
            goal = slide_to(goal, np.asarray(fxtr.pos, dtype=float)[:2], target[:2])
        return goal

    def facts(self, task_name):
        from robocasa.utils import object_utils as OU

        obs = self.observe()
        task_objects = [n for n in obs["objects"] if not n.startswith(es.DISTRACTOR_PREFIXES)]
        held_task = []
        for name in task_objects:
            try:
                if OU.check_obj_grasped(self.kitchen, name):
                    held_task.append(name)
            except Exception:
                pass
        held_any, bodies = gripper_holds_any(self.kitchen)
        facts = {**obs, "held_task": held_task, "held_any": bool(held_any or held_task), "held_bodies": bodies,
                 "predicates": self.predicates(), "correct_tool": None}
        if task_name == "CuttingToolSelection":
            facts["correct_tool"] = self.kitchen._CUTTING_MAP[self.kitchen.food]
        return facts


def facts_record(facts):
    """JSON-safe summary of the facts R2a used."""
    rel = {name: relative_pose(facts["base"], T) for name, T in facts["objects"].items()}
    return {"held_task": facts["held_task"], "held_any": facts["held_any"], "held_bodies": facts["held_bodies"],
            "correct_tool": facts["correct_tool"], "predicates": facts["predicates"],
            "mobile": [round(v, 4) for v in facts["mobile"]],
            "objects_in_base_frame": {k: [round(float(x), 4) for x in p] + [round(y, 4)] for k, (p, y) in rel.items()}}


def load_gains(path):
    data = json.loads(Path(path).read_text())
    if not data.get("ok"):
        raise ValueError(f"{path}: the control probe did not pass (ok=false); inspect it before running R2")
    return Gains.from_json(data["gains"])


def _lerobots(tasks, dataset_base, overrides):
    return {t: es.lerobot_dir(t, dataset_base, overrides) for t in tasks}


def _sources_of(programs, library, task_name):
    keys = {tuple(ph["source"]) for p in programs for ph in p["phases"] if ph["phase"] == "track"}
    out = {}
    for task, episode, predicate in keys:
        for src in library.sources(task, predicate):
            if src.episode == episode:
                out[src.key] = src
    return out


def _rel_position(obs):
    """End-effector position in the robot base frame."""
    return (inverse(obs["base"]) @ obs["eef"])[:3, 3]


def probe_world(world, reset):
    """The probe's measurements on ``world``; ``reset()`` puts it back in the probe's start state.

    Base-mode runs drive for PROBE_STEPS, then send zero motion for SETTLE_STEPS. The arm may lag
    while the base accelerates (``base_moves_arm``, per step, reported only); what the executor needs
    is that the hand returns to its pose in the base frame once the base stops (``settled``). An arm
    that held its world pose instead would end offset by about the base's travel.
    """
    arm_pos, arm_rot, base = [], [], []
    leaks = {"arm_moves_base": 0.0, "base_moves_arm": 0.0, "base_moves_arm_settled": 0.0,
             "zero_action_drift": 0.0}
    base_runs = []
    for dim in range(9):
        for amp in PROBE_AMPLITUDES:
            for sign in (1.0, -1.0):
                reset()
                start_obs = world.observe()
                a = np.zeros(12)
                if dim < 6:
                    a[dim] = sign * amp
                    a[11] = ARM_MODE
                else:
                    a[7 + dim - 6] = sign * amp
                    a[11] = BASE_MODE
                peak = 0.0
                for t in range(PROBE_STEPS):
                    before = world.observe()
                    world.step(a)
                    after = world.observe()
                    if t == 0:
                        continue  # the first step includes the controller's reaction to the restore
                    Rb = before["base"][:3, :3]
                    if dim < 6:
                        dp = Rb.T @ (after["eef"][:3, 3] - before["eef"][:3, 3])
                        drot = rotvec((Rb.T @ after["eef"][:3, :3]) @ (Rb.T @ before["eef"][:3, :3]).T)
                        unit = np.zeros(3)
                        unit[dim % 3] = sign * amp
                        (arm_pos if dim < 3 else arm_rot).append((unit, dp if dim < 3 else drot))
                        moved = math.hypot(after["mobile"][0] - before["mobile"][0],
                                           after["mobile"][1] - before["mobile"][1])
                        leaks["arm_moves_base"] = max(leaks["arm_moves_base"], moved)
                    else:
                        m0, m1 = before["mobile"], after["mobile"]
                        d = rot_z(-m0[2])[:2, :2] @ np.array([m1[0] - m0[0], m1[1] - m0[1]])
                        unit = np.zeros(3)
                        unit[dim - 6] = sign * amp
                        base.append((unit, np.array([d[0], d[1], wrap(m1[2] - m0[2])])))
                        peak = max(peak, float(np.linalg.norm(_rel_position(after) - _rel_position(before))))
                if dim < 6:
                    continue
                moved_obs = world.observe()
                hold = np.zeros(12)
                hold[11] = BASE_MODE
                for _ in range(SETTLE_STEPS):
                    world.step(hold)
                end_obs = world.observe()
                settled = float(np.linalg.norm(_rel_position(end_obs) - _rel_position(start_obs)))
                m0, m1 = start_obs["mobile"], moved_obs["mobile"]
                base_runs.append({"dim": dim - 6, "amplitude": amp, "sign": sign,
                                  "travel": math.hypot(m1[0] - m0[0], m1[1] - m0[1]),
                                  "turn": abs(wrap(m1[2] - m0[2])), "peak_step_lag": peak, "settled": settled})
                leaks["base_moves_arm"] = max(leaks["base_moves_arm"], peak)
                leaks["base_moves_arm_settled"] = max(leaks["base_moves_arm_settled"], settled)
    # Control: zero motion in base mode for as long as a base run, to separate the restore's own effect.
    reset()
    start_obs = world.observe()
    hold = np.zeros(12)
    hold[11] = BASE_MODE
    for _ in range(PROBE_STEPS + SETTLE_STEPS):
        world.step(hold)
    leaks["zero_action_drift"] = float(np.linalg.norm(_rel_position(world.observe()) - _rel_position(start_obs)))
    torso = {}
    for mode in (ARM_MODE, BASE_MODE):
        rates = []
        for sign in (1.0, -1.0):
            reset()
            a = np.zeros(12)
            a[10], a[11] = sign * PROBE_AMPLITUDES[-1], mode
            for t in range(PROBE_STEPS):
                before = world.observe()
                world.step(a)
                if t:
                    rates.append((world.observe()["base"][2, 3] - before["base"][2, 3]) / (sign * PROBE_AMPLITUDES[-1]))
        torso[mode] = float(np.mean(rates))
    torso_mode = max(torso, key=lambda m: abs(torso[m]))
    reset()
    M_pos, r2_pos = estimate_gains(arm_pos)
    M_rot, r2_rot = estimate_gains(arm_rot)
    M_base, r2_base = estimate_gains(base)
    torso_gain = torso[torso_mode] if abs(torso[torso_mode]) >= TORSO_MIN_GAIN else 0.0
    gains = Gains(M_pos, M_rot, M_base, torso=torso_gain, torso_mode=torso_mode)
    conds = {k: float(np.linalg.cond(m)) for k, m in (("arm_pos", M_pos), ("arm_rot", M_rot), ("base", M_base))}
    ok = min(r2_pos, r2_rot, r2_base) >= PROBE_MIN_R2 and max(conds.values()) < PROBE_MAX_COND \
        and leaks["arm_moves_base"] < ARM_MOVES_BASE_TOL and leaks["base_moves_arm_settled"] < ARM_HOLD_TOL \
        and leaks["zero_action_drift"] < ARM_HOLD_TOL
    return {"kind": "r2_control_probe", "gains": gains.to_json(),
            "r2": {"arm_pos": r2_pos, "arm_rot": r2_rot, "base": r2_base}, "condition_numbers": conds,
            "mode_leaks": leaks, "base_runs": base_runs,
            "torso_by_mode": {"arm": torso[ARM_MODE], "base": torso[BASE_MODE]},
            "amplitudes": list(PROBE_AMPLITUDES), "steps": PROBE_STEPS, "settle_steps": SETTLE_STEPS,
            "ok": bool(ok)}


def run_probe(env, output, *, log=print):
    """Measure the per-step motion each unit action produces (arm in base frame, base in mobile frame)."""
    world = SimWorld(env)
    snap = ladder.capture_snapshot(env)

    def reset():
        ladder.restore_world(env, snap)
        world.obs = ladder.live_observation(env)

    result = probe_world(world, reset)
    result.update(git_commit=ladder.current_git_commit(), created_at=ladder.utc_now())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    ladder.atomic_write_json(output, result)
    runs = result["base_runs"]
    summary = {k: result[k] for k in ("ok", "r2", "condition_numbers", "mode_leaks", "torso_by_mode")}
    summary["largest_base_travel"] = max(r["travel"] for r in runs)
    summary["largest_base_turn"] = max(r["turn"] for r in runs)
    log(json.dumps(summary, indent=2))
    return result


def _make_env(task, split, seed, env_interface="gym"):
    from robocasa.recovery.evaluate_recovery_benchmark import make_env

    return make_env(task, env_interface=env_interface, split=split, seed=int(seed), enable_render=True)


def convention_residual(world_obs, named):
    """Max abs difference between the live observation state and the demo's LeRobot state."""
    keys = {"base_position": "state.base_position", "end_effector_position_relative":
            "state.end_effector_position_relative"}
    diffs = [float(np.max(np.abs(np.asarray(world_obs[v]).reshape(-1) - np.asarray(named[k]).reshape(-1))))
             for k, v in keys.items()]
    for k, v in (("base_rotation", "state.base_rotation"),
                 ("end_effector_rotation_relative", "state.end_effector_rotation_relative")):
        q1, q2 = np.asarray(world_obs[v]).reshape(-1), np.asarray(named[k]).reshape(-1)
        diffs.append(float(min(np.max(np.abs(q1 - q2)), np.max(np.abs(q1 + q2)))))
    return max(diffs)


def run_self_replay(index_dir, output_dir, gains_path, *, tasks=es.PILOT_TASKS, dataset_base=None,
                    lerobot_overrides=None, demos=SELF_REPLAY_DEMOS, resume=False, log=print):
    """Each stage's own demos from their own state at the segment start (target kitchens)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gains = load_gains(gains_path)
    library = SourceLibrary(index_dir, _lerobots(tasks, dataset_base, lerobot_overrides))
    out_path = output_dir / "self_replay.jsonl"
    done = {(r["task_name"], r["predicate"], r["episode"]) for r in ladder.read_jsonl(out_path)} if resume else set()
    if out_path.exists() and not resume:
        raise FileExistsError(f"{out_path} exists; pass --resume")
    for task in tasks:
        env = None
        try:
            for (t, predicate), spec in STAGE_SPECS.items():
                if t != task or spec["kind"] == "set_aside":
                    continue
                for src in library.sources(task, predicate)[: int(demos)]:
                    if (task, predicate, src.episode) in done:
                        continue
                    record = {"task_name": task, "predicate": predicate, "episode": src.episode,
                              "target": src.target, "created_at": ladder.utc_now()}
                    try:
                        if env is None:
                            env = _make_env(task, "target", 0)
                            env.reset()   # gymnasium refuses step() before the outer reset()
                        states, ep_meta, model_xml = es.load_episode(library.lerobots[task], src.episode)
                        ladder._env_method(env, "reset_to")({"model": model_xml, "states": states[src.start],
                                                             "ep_meta": json.dumps(ep_meta)})
                        world = SimWorld(env)
                        record["convention_residual"] = convention_residual(
                            world.obs, library.named_state(task, src.episode, src.start))
                        if record["convention_residual"] > CONVENTION_TOL:
                            record["error"] = "convention_mismatch"
                        else:
                            facts = world.facts(task)
                            target = src.target
                            phases = stage_phases(src, spec, target, 0, navigate_to_fixture=False)
                            program = {"phases": phases}
                            sources = {src.key: src}
                            run = ExpertRun(world, gains, sources, predicate, cap=program_cap(program, sources),
                                            budget=10 ** 6)
                            record.update(run.run(phases))
                            record["facts"] = facts_record(facts)
                    except Exception:
                        record["error"] = traceback.format_exc()
                        record.setdefault("stage_completed", False)
                        ladder._close(env, None)
                        env = None
                    record.setdefault("stage_completed", False)
                    ladder.append_jsonl(out_path, record)
                    log(f"[self-replay] {task}:{predicate} ep {src.episode} completed={record['stage_completed']}"
                        f"{' ERROR ' + record['error'].splitlines()[-1] if record.get('error') else ''}")
        finally:
            ladder._close(env, None)
    summary = self_replay_summary(ladder.read_jsonl(out_path))
    ladder.atomic_write_json(output_dir / "self_replay_summary.json", summary)
    log(json.dumps({k: summary[k] for k in ("passed", "failing")}, indent=2))
    return summary


def transfer_stages(task_name, required):
    return [p for p in required if (task_name, p) in STAGE_SPECS and (task_name, p) != WRONG_TOOL_STAGE]


def run_transfer(index_dir, output_dir, gains_path, *, tasks=es.PILOT_TASKS, dataset_base=None,
                 lerobot_overrides=None, resets=TRANSFER_RESETS, base_seed=TRANSFER_BASE_SEED, split="pretrain",
                 resume=False, log=print):
    """Every stage in order from fresh pretrain-split resets, sources from the target demos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gains = load_gains(gains_path)
    library = SourceLibrary(index_dir, _lerobots(tasks, dataset_base, lerobot_overrides))
    out_path = output_dir / "transfer.jsonl"
    if out_path.exists() and not resume:
        raise FileExistsError(f"{out_path} exists; pass --resume")
    done = {(r["task_name"], int(r["environment_seed"])) for r in ladder.read_jsonl(out_path)}
    for task in tasks:
        env = None
        try:
            for i in range(int(resets)):
                seed = int(base_seed) + i
                if (task, seed) in done:
                    continue
                record = {"task_name": task, "environment_seed": seed, "stages": [], "created_at": ladder.utc_now()}
                try:
                    if env is None:
                        env = _make_env(task, split, base_seed)
                    env.reset(seed=seed)
                    world = SimWorld(env)
                    from robocasa.recovery.subtask_eval import get_subtask_eval

                    required = list((get_subtask_eval(env) or {}).get("required_predicates") or [])
                    for k, predicate in enumerate(transfer_stages(task, required)):
                        facts = world.facts(task)
                        if facts["predicates"].get(predicate):
                            record["stages"].append({"predicate": predicate, "stage_completed": True,
                                                     "already_true": True})
                            continue
                        programs = plan_attempts(task, predicate, facts, library, n=i + 1)
                        program = programs[i]
                        sources = _sources_of([program], library, task)
                        run = ExpertRun(world, gains, sources, predicate, cap=program_cap(program, sources),
                                        budget=10 ** 6)
                        result = run.run(program["phases"])
                        if result["stage_completed"] and world.facts(task)["held_any"]:
                            ExpertRun(world, gains, sources, None, cap=10 ** 6, budget=10 ** 6).run([{"phase": "release"}])
                        record["stages"].append({"predicate": predicate, "source": program["source"],
                                                 "rank_distance": program["rank_distance"], **result})
                        if not result["stage_completed"]:
                            break
                except Exception:
                    record["error"] = traceback.format_exc()
                    ladder._close(env, None)
                    env = None
                ladder.append_jsonl(out_path, record)
                log(f"[transfer] {task} seed={seed} " + " ".join(
                    f"{s['predicate']}={'ok' if s['stage_completed'] else 'no'}" for s in record["stages"])
                    + (" ERROR" if record.get("error") else ""))
        finally:
            ladder._close(env, None)
    summary = transfer_summary(ladder.read_jsonl(out_path))
    ladder.atomic_write_json(output_dir / "transfer_summary.json", summary)
    log(json.dumps({k: summary[k] for k in ("passed", "failing")}, indent=2))
    return summary


def continue_with_policy(env, policy, world, *, instruction, budget_left, seed, required, recorder=None):
    """XR-1 from the expert's end state: task instruction, context reset, until success or the budget."""
    from robocasa.recovery.recovery_rollout import _is_task_success, _step_env, call_policy, set_observation_instruction

    ladder.reset_policy(policy)
    ladder.seed_policy(policy, seed)
    obs = world.obs
    success, steps = False, 0
    for step in range(max(0, int(budget_left))):
        action = call_policy(policy, set_observation_instruction(obs, instruction), instruction=instruction)
        if recorder is not None:
            recorder.add(obs, action, "policy")
        obs, reward, done, info = _step_env(env, action)
        steps = step + 1
        if _is_task_success(info, reward=reward, env=env):
            success = True
            break
        if done:
            break
    world.obs = obs
    return {"policy_steps": steps, "success": success}


def run_gate_r(source_runs, alarms_path, audit_dir, index_dir, output_dir, gains_path, *, dataset_base=None,
               lerobot_overrides=None, attempts=ATTEMPTS_PER_STATE, host=None, port=None, scene_tolerance=1e-6,
               policy_args=None, shard_index=0, num_shards=1, record=True, resume=False, max_errors=20, log=print):
    from robocasa.recovery import alarm_branching as ab

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gains = load_gains(gains_path)
    states, _ = es.load_r0(audit_dir)
    class2 = [s for s in states if s["class"] == "recoverable"]
    if any(s["stage"] == es.ALL_LATCHED for s in class2):
        raise ValueError("class-2 states without a stage; rerun R1 coverage with Amendment 2")
    source_config, parents = ab.load_sources(source_runs)
    config = ab.branch_config(source_config, k_branches=attempts, policy_args=policy_args,
                              budget_mode="fixed_fraction", budget_fraction=BUDGET_FRACTION)
    alarms = {r["parent_id"]: r for r in ladder.read_jsonl(alarms_path)}
    horizons = {s["task_name"]: int(parents[s["parent_id"]][1]["horizon"]) for s in class2}
    tasks = sorted(ladder.shard_tasks(sorted(horizons), horizons, int(num_shards), int(shard_index)))
    class2 = [s for s in class2 if s["task_name"] in tasks]
    library = SourceLibrary(index_dir, _lerobots(tasks, dataset_base, lerobot_overrides))
    runtime = ladder.Runtime(config, host=host, port=port)
    out_path = output_dir / "attempts.jsonl"
    if out_path.exists() and not resume:
        raise FileExistsError(f"{out_path} exists; pass --resume")
    plan_path = output_dir / "gate_r_plan.json"
    plan = {
        "kind": "r2_gate_r_plan", "plan_doc": "docs/recovery_harvest_finetune_plan.md",
        "source_runs": [str(Path(d).resolve()) for d in source_runs], "alarms": str(Path(alarms_path).resolve()),
        "audit_dir": str(Path(audit_dir).resolve()), "index_dir": str(Path(index_dir).resolve()),
        "gains": json.loads(Path(gains_path).read_text())["gains"], "attempts": int(attempts),
        "budget_fraction": BUDGET_FRACTION, "threshold": GATE_R_MIN_SHARE, "record": bool(record),
        "states": [s["parent_id"] for s in class2], "git_commit": ladder.current_git_commit(),
        "created_at": ladder.utc_now()}
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        frozen = ("source_runs", "alarms", "audit_dir", "index_dir", "gains", "attempts", "states", "record")
        diffs = sorted(k for k in frozen if previous.get(k) != plan[k])
        if diffs:
            raise ValueError(f"resume differs from the registered Gate R plan in {diffs}")
    else:
        ladder.atomic_write_json(plan_path, plan)
    done = {(r["parent_id"], int(r["k"])) for r in ladder.read_jsonl(out_path)}
    errors = 0
    base_seed, protocol = int(config["base_seed"]), config["seed_protocol"]
    for task in tasks:
        wanted = {}
        for s in class2:
            if s["task_name"] != task:
                continue
            run_dir, parent = parents[s["parent_id"]]
            wanted[int(parent["environment_seed"])] = (s, run_dir, parent)
        env = policy = None
        try:
            for seed in range(base_seed, max(wanted) + 1):
                if protocol == "fresh_env":
                    if seed not in wanted:
                        continue
                    ladder._close(env, policy)
                    env = runtime.make_env(task, seed)
                    policy = runtime.make_policy(env)
                    env.reset()
                else:
                    if env is None:
                        env = runtime.make_env(task, base_seed)
                        policy = runtime.make_policy(env)
                    env.reset(seed=seed)
                if seed not in wanted:
                    continue
                state, run_dir, parent = wanted[seed]
                pid = state["parent_id"]
                todo = [k for k in range(int(attempts)) if (pid, k) not in done]
                if not todo:
                    continue
                rollout = ab.load_rollout(run_dir, parent)
                residual = ladder.state_residual(env, rollout["snapshots"][0]["states"])
                if residual > scene_tolerance:
                    log(f"[gate-r] {task} {pid}: scene mismatch ({residual:.3g}); skipped")
                    continue
                alarm = alarms[pid]
                snap = rollout["snapshots"][int(alarm["alarm_snapshot_index"])]
                horizon = int(parent["horizon"])
                budget = ladder.branch_budget(horizon, snap["step"], config, task)
                instruction = rollout["instruction"] or ladder.env_instruction(env, None)
                programs = None
                for k in todo:
                    record = {"parent_id": pid, "task_name": task, "stage": state["stage"],
                              "stage_source": state["stage_source"], "k": k, "environment_seed": seed,
                              "alarm_step": int(snap["step"]), "budget_steps": int(budget),
                              "created_at": ladder.utc_now()}
                    started = time.time()
                    recorder = None
                    try:
                        ladder.restore_world(env, snap)
                        ladder._refresh_sim_visuals(env, num_forwards=3)
                        if record:
                            rel = Path("recordings") / pid / f"k{k:02d}"
                            if (output_dir / rel).exists():   # recorded, but the run stopped before logging it
                                import shutil

                                shutil.rmtree(output_dir / rel)
                            recorder = AttemptRecorder(output_dir / rel)
                        world = SimWorld(env, recorder=recorder)
                        facts = world.facts(task)
                        if programs is None:
                            programs = plan_attempts(task, state["stage"], facts, library, n=int(attempts))
                        program = programs[k]
                        sources = _sources_of([program], library, task)
                        run = ExpertRun(world, gains, sources, state["stage"], cap=program_cap(program, sources),
                                        budget=budget)
                        result = run.run(program["phases"])
                        record.update({"source": program["source"], "rank_distance": program["rank_distance"],
                                       "facts": facts_record(facts), **result})
                        if result["task_success"]:
                            record.update({"policy_steps": 0, "success": True})
                        else:
                            world.recorder = None   # the continuation records its own frames
                            record.update(continue_with_policy(
                                env, policy, world, instruction=instruction, budget_left=budget - result["expert_steps"],
                                seed=ladder.stable_seed(pid, "r2", k), required=rollout["required_subtasks"],
                                recorder=recorder))
                        record["steps_used"] = record["expert_steps"] + record["policy_steps"]
                        if recorder is not None:
                            record["recorded_frames"] = recorder.finish({
                                "parent_id": pid, "task_name": task, "stage": state["stage"], "k": k,
                                "instruction": instruction, "source": record.get("source"),
                                "success": record["success"], "stage_completed": record["stage_completed"],
                                "expert_steps": record["expert_steps"], "policy_steps": record["policy_steps"],
                                "git_commit": ladder.current_git_commit()})
                            record["recording"] = str(rel)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        errors += 1
                        if recorder is not None:
                            recorder.abort()
                        record.update({"error": traceback.format_exc(), "success": False,
                                       "stage_completed": False})
                    record["wall_seconds"] = time.time() - started
                    ladder.append_jsonl(out_path, record)
                    log(f"[gate-r] {task} {pid} k={k} expert={record.get('stage_completed')} "
                        f"success={record.get('success')}{' ERROR' if record.get('error') else ''}")
                    if errors >= max_errors:
                        raise RuntimeError(f"stopping after {errors} errors (see {out_path})")
        finally:
            ladder._close(env, policy)
    result = gate_r(ladder.read_jsonl(out_path), class2)
    ladder.atomic_write_json(output_dir / "gate_r.json", result)
    write_gate_r_md(result, output_dir / "gate_r.md")
    log(json.dumps({k: result[k] for k in ("passed", "share", "expert_rate", "full_rate")}, indent=2))
    return result


def write_gate_r_md(result, path):
    lines = ["# R2: Gate R", "",
             f"**{'Passed' if result['passed'] else 'Not passed'}**: {result['states_with_success']} of "
             f"{result['class2_states']} class-2 states have a full-task success among their attempts "
             f"(share {result['share']}, threshold {result['threshold']}).", "",
             f"Attempts: {result['attempts']}. Expert rate (stuck stage completed): {result['expert_rate']}. "
             f"Full rate (task success): {result['full_rate']}.", "",
             "| State | Task | Stage | Attempts | Expert completed | Successes | Errors |",
             "|---|---|---|---:|---:|---:|---:|"]
    for s in result["states"]:
        lines.append(f"| `{s['parent_id']}` | {s['task_name']} | `{s['stage']}` | {s['attempts']} | "
                     f"{s['expert_completed']} | {s['successes']} | {s['errors']} |")
    Path(path).write_text("\n".join(lines) + "\n")


# ---- CLI ----------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p):
        p.add_argument("--index-dir", required=True, type=Path, help="R1 expert_segments index")
        p.add_argument("--dataset-base", type=Path, help="RoboCasa dataset root (holds v1.0/target/composite)")
        p.add_argument("--lerobot", nargs="*", default=[], help="TASK=PATH overrides of the LeRobot folder")
        p.add_argument("--gains", required=True, type=Path, help="probe-controls output")
        p.add_argument("--output-dir", required=True, type=Path)
        p.add_argument("--resume", action="store_true")

    p = sub.add_parser("probe-controls", help="measure action -> motion gains (simulator)")
    p.add_argument("--task", default="ArrangeTea")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--seed", type=int, default=TRANSFER_BASE_SEED)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("self-replay", help="executor check 1: own segment from own state (simulator)")
    data_args(p)
    p.add_argument("--tasks", nargs="*", default=list(es.PILOT_TASKS))
    p.add_argument("--demos", type=int, default=SELF_REPLAY_DEMOS)
    p = sub.add_parser("transfer", help="executor check 2: fresh pretrain-split resets (simulator)")
    data_args(p)
    p.add_argument("--tasks", nargs="*", default=list(es.PILOT_TASKS))
    p.add_argument("--resets", type=int, default=TRANSFER_RESETS)
    p = sub.add_parser("gate-r", help="expert + XR-1 at the class-2 alarm states (simulator + policy server)")
    data_args(p)
    p.add_argument("--source-runs", nargs="+", required=True, type=Path)
    p.add_argument("--alarms", required=True, type=Path, help="hs_p_alarm score-ladder rows")
    p.add_argument("--audit-dir", required=True, type=Path, help="R0 audit (decide has run)")
    p.add_argument("--attempts", type=int, default=ATTEMPTS_PER_STATE)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--policy-arg", action="append", default=[], help="key=value passed to the policy factory")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--no-record", action="store_true", help="outcomes only, no videos or frames")
    p = sub.add_parser("export-lerobot", help="recorded Gate R attempts -> LeRobot folders per task (no simulator)")
    p.add_argument("--gate-r", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--include", choices=("success", "all"), default="success")
    p.add_argument("--link", action="store_true", help="hard-link the videos instead of copying")
    p = sub.add_parser("analyze", help="recompute the summaries from the records (no simulator)")
    p.add_argument("--self-replay", type=Path)
    p.add_argument("--transfer", type=Path)
    p.add_argument("--gate-r", type=Path)
    p.add_argument("--audit-dir", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "probe-controls":
        env = _make_env(args.task, args.split, args.seed)
        env.reset(seed=args.seed)
        try:
            run_probe(env, args.output)
        finally:
            ladder._close(env, None)
        return
    if args.command == "export-lerobot":
        print(json.dumps(export_lerobot(args.gate_r, args.output_dir, include=args.include, copy=not args.link),
                         indent=2))
        return
    if args.command == "analyze":
        out = {}
        if args.self_replay:
            out["self_replay"] = self_replay_summary(ladder.read_jsonl(Path(args.self_replay) / "self_replay.jsonl"))
        if args.transfer:
            out["transfer"] = transfer_summary(ladder.read_jsonl(Path(args.transfer) / "transfer.jsonl"))
        if args.gate_r:
            if not args.audit_dir:
                raise SystemExit("--gate-r needs --audit-dir")
            states, _ = es.load_r0(args.audit_dir)
            attempts = []
            for path in sorted(Path(args.gate_r).glob("**/attempts.jsonl")):
                attempts.extend(ladder.read_jsonl(path))
            out["gate_r"] = gate_r(attempts, [s for s in states if s["class"] == "recoverable"])
        print(json.dumps(out, indent=2, default=ladder.json_default))
        return
    common = dict(dataset_base=args.dataset_base, lerobot_overrides=es._overrides(args.lerobot), resume=args.resume)
    if args.dataset_base is None and not args.lerobot:
        raise SystemExit("pass --dataset-base or --lerobot TASK=PATH")
    if args.command == "self-replay":
        run_self_replay(args.index_dir, args.output_dir, args.gains, tasks=args.tasks, demos=args.demos, **common)
    elif args.command == "transfer":
        run_transfer(args.index_dir, args.output_dir, args.gains, tasks=args.tasks, resets=args.resets, **common)
    else:
        run_gate_r(args.source_runs, args.alarms, args.audit_dir, args.index_dir, args.output_dir, args.gains,
                   attempts=args.attempts, host=args.host, port=args.port,
                   policy_args=ladder._parse_policy_args(args.policy_arg), shard_index=args.shard_index,
                   num_shards=args.num_shards, record=not args.no_record, **common)


if __name__ == "__main__":
    main()
