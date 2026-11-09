"""Tests for E2's paired reset, plan, collection check and analysis (docs/rldx_e2_plan.md)."""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ce = _load("_t_collect_e2", "scripts/rldx_e2/collect_e2.py")
pe = _load("_t_plan_e2", "scripts/rldx_e2/plan_e2.py")
ck = _load("_t_check_e2", "scripts/rldx_e2/check_e2.py")
an = _load("_t_analyze_e2", "scripts/rldx_e2/analyze_e2.py")
fu = _load("_t_followup_e2", "scripts/rldx_e2/followup_e2.py")
rc = _load("_t_restore_check", "scripts/rldx_e2/restore_check.py")


class FakeGymEnv:
    """A scene is a state vector; reset builds a new one, step moves it."""

    def __init__(self):
        self.n_resets = 0
        self.x = None

    def reset(self, **kwargs):
        self.n_resets += 1
        self.x = np.full(3, float(self.n_resets))
        return self._obs(), {"success": False}

    def _obs(self):
        return {"image": np.full((4, 4, 3), int(self.x[0]) % 256, np.uint8), "state": self.x.copy(),
                "annotation.human.task_description": "do it"}

    def step(self, action):
        self.x = self.x + 1.0
        return self._obs(), 0.0, False, False, {"success": False}

    def get_state(self):
        return {"states": self.x.copy(), "model": f"<xml scene={int(self.x[0])}/>", "ep_meta": "{}"}

    def reset_to(self, state):
        self.x = np.asarray(state["states"], float).copy()
        return self._obs()


class DrawerGymEnv(FakeGymEnv):
    """Like OpenDrawer under robocasa's gym wrapper: the instruction's side is drawn again whenever
    the model is (re)loaded, reset_to included; the wrapper can override the instruction."""

    def __init__(self, sides):
        super().__init__()
        self.sides, self.draws, self.override = list(sides), 0, None

    def _draw(self):
        self.side = self.sides[self.draws % len(self.sides)]
        self.draws += 1

    def reset(self, **kwargs):
        self.override = None                      # clear_task_description, as the wrapper's reset does
        self._draw()
        return super().reset(**kwargs)

    def _obs(self):
        obs = super()._obs()
        obs["annotation.human.task_description"] = self.override or f"Open the {self.side} drawer."
        return obs

    def get_state(self):
        st = super().get_state()
        st["ep_meta"] = json.dumps({"lang": f"Open the {self.side} drawer."})
        return st

    def reset_to(self, state):
        self._draw()                              # the model reload places the robot again
        return super().reset_to(state)

    def set_task_description(self, text):
        self.override = text

    def get_current_observation(self):
        return self._obs()


class Outer:
    """A gymnasium-like wrapper that does not forward reset_to."""

    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class PairedResetTest(unittest.TestCase):
    def test_alternates_and_restores(self):
        with tempfile.TemporaryDirectory() as d:
            env = ce.install(Outer(FakeGymEnv()), "T", Path(d) / "pairs.jsonl")
            o0, _ = env.reset()
            env.step(None); env.step(None)
            o1, info = env.reset()
            np.testing.assert_array_equal(o0["state"], o1["state"])
            self.assertFalse(info["success"])
            o2, _ = env.reset()
            self.assertFalse(np.array_equal(o2["state"], o0["state"]))
            env.step(None)
            env.reset()
            log = [json.loads(x) for x in (Path(d) / "pairs.jsonl").read_text().splitlines()]
            self.assertEqual([x["role"] for x in log], ["first", "retry", "first", "retry"])
            self.assertEqual(log[1]["restored_sha"], log[0]["state_sha"])
            self.assertEqual(log[1]["restore_residual"], 0.0)
            self.assertEqual(log[1]["first_obs_distance"]["image_max"], 0.0)
            self.assertTrue(log[1]["first_obs_distance"]["text_equal"])
            self.assertIsNone(ck.pair_problem(log[0], log[1]))
            self.assertIsNone(ck.pair_problem(log[2], log[3]))
            self.assertIsNotNone(ck.pair_problem(log[0], log[3]))      # restored another scene's state

    def test_retry_keeps_the_first_attempts_instruction(self):
        """OpenDrawer on the cluster: reset_to drew the other side, so the retry's instruction differed."""
        with tempfile.TemporaryDirectory() as d:
            env = ce.install(Outer(DrawerGymEnv(["left", "right", "right", "left"])), "OpenDrawer",
                             Path(d) / "pairs.jsonl")
            o0, _ = env.reset()
            self.assertEqual(o0["annotation.human.task_description"], "Open the left drawer.")
            o1, _ = env.reset()                    # reset_to draws "right"; the override restores "left"
            self.assertEqual(o1["annotation.human.task_description"], "Open the left drawer.")
            o, *_ = env.step(None)
            self.assertEqual(o["annotation.human.task_description"], "Open the left drawer.")   # every step
            o2, _ = env.reset()                    # a new scene: the override is cleared
            self.assertEqual(o2["annotation.human.task_description"], "Open the right drawer.")
            log = [json.loads(x) for x in (Path(d) / "pairs.jsonl").read_text().splitlines()]
            self.assertTrue(log[1]["instruction_restored"])
            self.assertIsNone(ck.pair_problem(log[0], log[1]))

    def test_without_the_override_the_check_catches_it(self):
        class NoOverride(DrawerGymEnv):
            set_task_description = None
        with tempfile.TemporaryDirectory() as d:
            env = ce.install(Outer(NoOverride(["left", "right"])), "OpenDrawer", Path(d) / "pairs.jsonl")
            env.reset(); env.reset()
            log = [json.loads(x) for x in (Path(d) / "pairs.jsonl").read_text().splitlines()]
            self.assertFalse(log[1]["instruction_restored"])
            self.assertEqual(ck.pair_problem(log[0], log[1]), "instruction differs")

    def test_breakdown_marks_instruction_reruns(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "OpenDrawer", Path(d) / "OpenCabinet"
            a.mkdir(); b.mkdir()
            first = dict(role="first", task_name="OpenDrawer", state_sha="s", model_sha="m")
            good = dict(role="retry", task_name="OpenDrawer", restored_sha="s", model_sha="m", restore_residual=0.0,
                        first_obs_distance=dict(missing=[], text_equal=True, image_max=0.0, state_max=0.0))
            bad = json.loads(json.dumps(good)); bad["first_obs_distance"]["text_equal"] = False
            (a / "pairs.jsonl").write_text("\n".join(json.dumps(x) for x in (first, good, first, bad)) + "\n")
            f2, g2 = dict(first, task_name="OpenCabinet"), dict(good, task_name="OpenCabinet")
            (b / "pairs.jsonl").write_text("\n".join(json.dumps(x) for x in (f2, g2)) + "\n")
            lines, rerun = ck.breakdown([("e2", a), ("e2", b)])
            self.assertEqual(rerun, ["OpenDrawer"])
            self.assertIn("RERUN OpenDrawer", lines)
            self.assertIn("total excluded: 1 of 3 pairs", lines)

    def test_pair_problem_tolerances(self):
        first = dict(role="first", state_sha="a", model_sha="m")
        good = dict(role="retry", restored_sha="a", model_sha="m", restore_residual=0.0,
                    first_obs_distance=dict(missing=[], text_equal=True, image_max=0.5, state_max=0.0))
        self.assertIsNone(ck.pair_problem(first, good))
        for change in (dict(restore_residual=1e-3), dict(model_sha="x"),
                       dict(first_obs_distance=dict(missing=[], text_equal=True, image_max=9.0, state_max=0.0)),
                       dict(first_obs_distance=dict(missing=[], text_equal=False, image_max=0.0, state_max=0.0))):
            self.assertIsNotNone(ck.pair_problem(first, {**good, **change}), change)

    def test_main_refuses_resume_and_quotas(self):
        base = ["--output-dir", "/nonexistent/x", "--tasks", "T", "--seed-protocol", "official_rldx",
                "--num-rollouts", "4", "--no-continue-on-error", "--record-safe-features", "--policy-name", "p",
                "--checkpoint", "c"]
        for extra, why in ((["--resume"], "--resume"), (["--continue-on-error"], "--continue-on-error"),
                           (["--success-quota", "2"], "quotas")):
            with self.assertRaises(SystemExit) as cm:
                ce.main(base + extra)
            self.assertIn(why, str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            ce.main([x if x != "4" else "3" for x in base])
        self.assertIn("even", str(cm.exception))


def e1_inputs(tasks=("A", "B")):
    settings = dict(seed=20, seed_protocol="official_rldx", split="target", replan_steps=8, policy_module=None,
                    model_family="rldx1", policy_name="rldx1", checkpoint="ck", policy_config={},
                    record_actions=False, video_frame_stride=2)
    e1_plan = dict(source="/rec", settings=settings, seeds=dict(calibration=1201, test=1301),
                   tasks={t: dict(type="atomic" if t == "A" else "composite", H=400, H_inferences=50, p=0.5,
                                  mean_success_length=200.0) for t in tasks})
    e1_result = dict(P=dict(path=dict(lam=0.0739, u=math.inf, s=0.0)), counts=dict(test=dict(successes=400)),
                     detectors=dict(safe=dict(path=dict(lam=0.0739, u=7.9314, s=0.0), H3=True),
                                    hide_and_seek=dict(path=dict(lam=0.0956, u=-0.17, s=20.0), H3=False)))
    freeze = dict(tasks={t: dict(H_k=50, rho_k=0.6) for t in tasks},
                  detectors=dict(safe_plus_P=dict(path=dict(u=7.9314, s=0.0)),
                                 hide_and_seek_plus_P=dict(path=dict(u=-0.17, s=20.0))))
    reset_sim = dict(runs={"0.245": dict(per_task={t: dict(budgets={"1.25": dict(a_opt=0.4)}) for t in tasks})})
    return e1_plan, e1_result, freeze, reset_sim


class PlanTest(unittest.TestCase):
    def test_build(self):
        plan, flags = pe.build(*e1_inputs(), seed=1401, used={20: ["x"], 7: ["y"]}, shards=2)
        self.assertEqual(plan["tasks"]["A"]["budget_steps"], 500)
        self.assertEqual(plan["conditions"]["P"]["lam"], 0.0739)
        self.assertEqual(plan["conditions"]["safe_lstm_plus_P"]["latency_steps"], pe.LATENCY_STEPS)
        self.assertEqual(plan["conditions"]["R_opt"]["cutoffs"], {"A": 0.4, "B": 0.4})
        self.assertEqual(plan["primary"], ["safe_lstm_plus_P", "P"])
        self.assertIn("--no-continue-on-error", flags)
        self.assertEqual(flags[flags.index("--horizon-scale") + 1], "1.25")
        self.assertEqual(flags[flags.index("--num-rollouts") + 1], "100")
        self.assertEqual(plan["rerun_seeds"][:2], [1411, 1421])
        self.assertEqual(len(plan["shards"]), 2)

    def test_stops(self):
        e1_plan, e1_result, freeze, reset_sim = e1_inputs()
        for kwargs in (dict(seed=1301), dict(seed=1201), dict(seed=1291), dict(seed=1401, used={1431: ["z"]}),
                       dict(seed=1401, smoke_seed=1101)):
            used = kwargs.pop("used", {})
            with self.assertRaises(SystemExit, msg=str(kwargs)):
                pe.build(e1_plan, e1_result, freeze, reset_sim, used=used, **kwargs)
        bad = json.loads(json.dumps(freeze))
        bad["detectors"]["safe_plus_P"]["path"]["u"] = 1.0
        with self.assertRaises(SystemExit):
            pe.build(e1_plan, e1_result, bad, reset_sim, seed=1401, used={})
        e1_bad = json.loads(json.dumps(e1_result).replace("Infinity", "1e308"))
        e1_bad["detectors"]["safe"]["H3"] = False
        with self.assertRaises(SystemExit):
            pe.build(e1_plan, e1_bad, freeze, reset_sim, seed=1401, used={})


def scene(t="A", s0=False, n0=500, s1=False, n1=500, scores=None, n_inf=None, B=500):
    n_inf = n_inf if n_inf is not None else math.ceil(n0 / 8)
    return dict(task_name=t, type="atomic", scene=0, first_id="a", retry_id="r", s0=s0, n0=n0, s1=s1, n1=n1,
                inf_steps=[8 * i for i in range(n_inf)], B=B, H=50, rho=0.6,
                safe=np.asarray(scores if scores is not None else np.zeros(n_inf), float))


class OutcomeTest(unittest.TestCase):
    P = dict(kind="fused", lam=0.1, u=math.inf, s=0.0, latency_steps=0)        # alarms at ceil(0.7 x 50) = 35
    S = dict(kind="fused", detector="safe", lam=0.1, u=1.0, s=0.0, latency_steps=1)

    def test_failure_reset_and_retry(self):
        # P resets at inference 35, control step 272; the retry has 500 - 272 = 228 steps left.
        sc = [scene(s1=True, n1=228), scene(s1=True, n1=229)]
        won, reset, unnec, lost = an.outcomes(sc, self.P)
        self.assertEqual(won.tolist(), [True, False])
        self.assertEqual(reset.tolist(), [True, True])
        self.assertEqual(unnec.tolist(), [False, False])

    def test_earlier_score_alarm_and_latency(self):
        scores = np.zeros(63); scores[9] = 2.0                                   # alarm at inference 10, step 72
        sc = [scene(s1=True, n1=427, scores=scores)]
        self.assertTrue(an.outcomes(sc, self.S)[0][0])                           # 72 + 1 + 427 = 500
        self.assertFalse(an.outcomes(sc, self.S, latency=2)[0][0])
        self.assertFalse(an.outcomes(sc, self.P)[0][0])                          # P: 272 + 427 > 500

    def test_success_before_and_after_alarm(self):
        before = scene(s0=True, n0=272, n_inf=34)                                 # 34 inferences < alarm 35
        after = scene(s0=True, n0=273, n_inf=35, s1=False)                        # alarm at its 35th inference
        won, reset, unnec, lost = an.outcomes([before, after], self.P)
        self.assertEqual(won.tolist(), [True, False])
        self.assertEqual(unnec.tolist(), [False, True])
        self.assertEqual(lost.tolist(), [False, True])
        none = an.outcomes([before, after], dict(kind="none"))[0]
        self.assertEqual(none.tolist(), [True, True])

    def test_time_rule(self):
        sc = [scene(s1=True, n1=340)]                                             # cutoff 0.4: inference 20, step 152
        self.assertTrue(an.outcomes(sc, dict(kind="time", cutoffs={"A": 0.4}))[0][0])
        self.assertFalse(an.outcomes(sc, dict(kind="time", cutoffs={"A": 0.4}, latency_steps=9))[0][0])

    def test_analyze_and_bootstrap(self):
        rng = np.random.default_rng(0)
        sc = []
        for t in ("A", "B"):
            for j in range(20):
                s0 = bool(rng.random() < 0.5)
                sc.append(dict(scene(t=t, s0=s0, n0=200 if s0 else 500, n_inf=25 if s0 else 63,
                                     s1=bool(rng.random() < 0.3), n1=100), type="atomic" if t == "A" else "composite"))
        plan, _ = pe.build(*e1_inputs(), seed=1401, used={}, shards=2)
        plan["conditions"]["P"]["lam"] = 0.1
        plan["bootstrap"] = 200
        out = an.analyze(plan, sc, has_hs=False)
        self.assertEqual(out["scenes"], 40)
        c = out["contrasts"]["safe_lstm_plus_P_minus_P"]
        self.assertTrue(c["primary"])
        self.assertLessEqual(c["ci"][0], c["delta"])
        self.assertLessEqual(c["delta"], c["ci"][1])
        self.assertNotIn("hide_and_seek_plus_P_minus_P", out["contrasts"])
        self.assertEqual(set(out["by_type"]), {"atomic", "composite"})


class CheckTest(unittest.TestCase):
    def _dir(self, root, seed, task, n, pairs_ok=True, error=False):
        root.mkdir(parents=True)
        (root / "summary.json").write_text(json.dumps(dict(config=dict(base_environment_seed=seed))))
        rows, log = [], []
        for k in range(n):
            rows.append(dict(rollout_id=f"{seed}-{task}-{k}", task_name=task, environment_seed=seed,
                             environment_reset_index=k, rollout_horizon=500, failed=bool(k % 3), seed_protocol="official_rldx",
                             environment_split="target", replan_steps=8, policy_id="rldx1", checkpoint="ck",
                             policy_config={}, model_family="rldx1", video_frame_stride=2, feature_layer="L",
                             tensor_path="x.npz"))
            if k % 2 == 0:
                log.append(dict(task_name=task, reset_index=k, role="first", state_sha=f"s{k}", model_sha="m"))
            else:
                log.append(dict(task_name=task, reset_index=k, role="retry", restored_sha=f"s{k - 1}" if pairs_ok else "zz",
                                model_sha="m", restore_residual=0.0,
                                first_obs_distance=dict(missing=[], text_equal=True, image_max=0.0, state_max=0.0)))
        (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (root / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
        if error:
            (root / "errors.jsonl").write_text(json.dumps(dict(task_name=task)) + "\n")

    def test_complete_partial_and_seeds(self):
        plan, _ = pe.build(*e1_inputs(), seed=1401, used={}, shards=2, scenes=2)
        rec = [dict(rollout_id="rec", task_name=t, feature_layer="L", tensor_path="r.npz") for t in ("A", "B")]
        tensor = lambda root, row: dict(shape=[3, 2, 4], dtype="float32", finite=True, nonzero=True, steps=[0, 8, 16])
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._dir(d / "s0", 1401, "A", 4)
            self._dir(d / "s1", 1401, "B", 3, error=True)                         # stopped: B partial
            self._dir(d / "s2", 1411, "B", 4)                                     # rerun with a reserved seed
            dirs = [("e2", str(d / x)) for x in ("s0", "s1", "s2")]
            problems, out = ck.check(plan, dirs, rec, tensor_fn=tensor)
            self.assertEqual(problems, [])                                        # one task per directory, same seed
            self.assertEqual(out["tasks"], {"A": str(d / "s0"), "B": str(d / "s2")})
            self._dir(d / "s4", 1401, "A", 2)                                     # A re-run with the same seed
            problems, _ = ck.check(plan, dirs + [("e2", str(d / "s4"))], rec, tensor_fn=tensor)
            self.assertTrue(any("more than one directory" in x for x in problems))
            self.assertEqual(len(out["pairs"]), 4)
            import shutil
            shutil.rmtree(d / "s1")
            self._dir(d / "s1", 1421, "B", 3, error=True)                         # stopped run with its own seed
            problems, out = ck.check(plan, dirs, rec, tensor_fn=tensor)
            self.assertEqual(problems, [])
            self._dir(d / "s3", 1431, "A", 4, pairs_ok=False)                     # A complete twice, bad pairs
            problems, out = ck.check(plan, dirs + [("e2", str(d / "s3"))], rec, tensor_fn=tensor)
            self.assertTrue(any("complete in 2 directories" in x for x in problems))
            self.assertTrue(any("fail the restore checks" in x for x in problems))


if __name__ == "__main__":
    unittest.main()


class FollowupTest(unittest.TestCase):
    """followup_e2.py: the reset ceiling, scene dependence and stopping at the alarm."""

    def plan(self):
        plan, _ = pe.build(*e1_inputs(), seed=1401, used={}, shards=2)
        plan["conditions"]["P"]["lam"] = 0.1                                    # P alarms at inference 35, step 272
        plan["conditions"]["safe_lstm_plus_P"]["lam"] = 0.1
        plan["bootstrap"] = 200
        return plan

    def scenes(self, n=20, dependent=False, seed=0):
        rng = np.random.default_rng(seed)
        sc = []
        for t in ("A", "B"):
            for j in range(n):
                s0 = bool(rng.random() < 0.5)
                s1 = s0 if dependent else bool(rng.random() < 0.5)
                sc.append(dict(scene(t=t, s0=s0, n0=200 if s0 else 500, n_inf=25 if s0 else 63, s1=s1, n1=100),
                               type="atomic" if t == "A" else "composite"))
        return sc

    def test_oracle_resets_only_failures(self):
        plan = self.plan()
        sc = [scene(s0=True, n0=200, n_inf=25),                    # success: never reset by the oracle
              scene(s1=True, n1=228),                              # failure, retry fits after P's alarm (272)
              scene(s1=True, n1=300),                              # fits only if reset at once
              scene(s1=False)]
        self.assertEqual(fu.oracle(sc, plan["conditions"]["P"]).tolist(), [True, True, False, False])
        self.assertEqual(fu.oracle(sc).tolist(), [True, True, True, False])
        out = fu.f1(plan, self.scenes())
        for k in ("oracle@P", "oracle@SAFE", "oracle@0"):
            self.assertGreaterEqual(out[f"{k}_minus_none"]["delta"], 0.0)
            self.assertGreaterEqual(out[f"{k}_minus_P"]["delta"], 0.0)
        self.assertGreaterEqual(out["success"]["oracle@0"]["rate"], out["success"]["oracle@P"]["rate"])

    def test_odds_ratio_separates_hard_scenes(self):
        plan = self.plan()
        indep = fu.f2(plan, self.scenes(n=200, dependent=False, seed=1))
        dep = fu.f2(plan, self.scenes(n=200, dependent=True, seed=1))
        self.assertLess(abs(math.log(indep["odds_ratio"])), 0.5)
        self.assertGreater(dep["odds_ratio"], 50)
        self.assertEqual(dep["all"]["retry_after_failure"], 0.0)
        self.assertLessEqual(indep["odds_ratio_ci"][0], indep["odds_ratio"])
        self.assertEqual(set(dep["by_type"]), {"atomic", "composite"})

    def test_stopping_saves_time_on_failures_and_stops_late_successes(self):
        plan = self.plan()
        before = scene(s0=True, n0=272, n_inf=34)                  # succeeds before P's alarm
        after = scene(s0=True, n0=273, n_inf=35)                   # P alarms at its 35th inference: stopped
        fail = scene()                                             # alarmed at step 272 of 500
        saved, stopped = fu.stop_values([before, after, fail], plan["conditions"]["P"])
        self.assertEqual(stopped.tolist(), [False, True, False])
        self.assertAlmostEqual(saved[2], (500 - 272) / 20.0)
        out = fu.f3(plan, self.scenes())
        self.assertIn("minus_P", out["rules"]["safe_lstm_plus_P"])
        m = out["rules"]["safe_lstm_plus_P"]["minus_P"]
        self.assertLessEqual(m["seconds_saved_per_scene_ci"][0], m["seconds_saved_per_scene"] + 1e-9)

    def test_f0_reproduces_published_rates(self):
        plan = self.plan()
        sc = self.scenes()
        result = an.analyze(plan, sc, has_hs=False)
        self.assertEqual(fu.reproduce(plan, sc, result, has_hs=False), [])
        result["rates"]["P"]["success"] += 0.01
        self.assertTrue(fu.reproduce(plan, sc, result, has_hs=False))


class _Sim:
    def __init__(self, raw):
        self.raw = raw

    def get_state(self):
        return self

    def flatten(self):
        return self.raw.x.copy()


class _Raw:
    """A robosuite-like task: a state vector, a target chosen at build time, success when x[0] > target."""

    def __init__(self):
        self.sim, self.x, self.target, self.cabinet = _Sim(self), np.zeros(3), 0.5, None

    def get_ep_meta(self):
        return {"lang": "Put it away.", "fixture_refs": {"cab": self.cabinet}}

    def _check_success(self):
        return bool(self.x[0] > self.target)


class _Space:
    shape, low, high = (3,), -np.ones(3), np.ones(3)


class _Gym:
    """Gym-wrapper-like: reset builds a scene; reset_to restores the state and, if `redraw`, picks the
    cabinet (and the success target) again, as a model reload can."""

    action_space = _Space()

    def __init__(self, redraw=False, drift=False):
        self.env, self.redraw, self.drift, self.n = _Raw(), redraw, drift, 0
        self.override = None

    def _obs(self):
        return {"state": self.env.x.copy(), "annotation.human.task_description": self.override or "Put it away."}

    def reset(self, **kwargs):
        self.n += 1
        self.override = None
        self.env.x = np.full(3, 0.1 * self.n)
        self.env.cabinet, self.env.target = f"cab{self.n}", 0.2
        return self._obs(), {}

    def step(self, a):
        self.env.x = self.env.x + 0.01 * np.asarray(a) + (1e-3 if self.drift else 0.0)
        return self._obs(), 0.0, False, False, {"success": self.env._check_success()}

    def get_state(self):
        return {"states": self.env.x.copy(), "model": "<xml/>", "ep_meta": json.dumps(self.env.get_ep_meta())}

    def reset_to(self, state):
        self.env.x = np.asarray(state["states"], float).copy()
        if self.redraw:
            self.env.cabinet, self.env.target = "cab_other", 9.0
        return self._obs()

    def set_task_description(self, t):
        self.override = t

    def get_current_observation(self):
        return self._obs()

    def close(self):
        pass


class RestoreCheckTest(unittest.TestCase):
    def run_check(self, env):
        with tempfile.TemporaryDirectory() as d:
            return rc.check_task(env, "T", scenes=2, steps=20, log_dir=d)

    def test_faithful_restore_is_same(self):
        rows = self.run_check(_Gym())
        self.assertTrue(all(r["same"] for r in rows), rows)
        self.assertIsNone(rows[0]["state_diverges_at_step"])

    def test_redrawn_target_is_caught(self):
        rows = self.run_check(_Gym(redraw=True))
        r = rows[0]
        self.assertFalse(r["same"])
        self.assertEqual(r["attribute_diffs"]["cabinet"], ["cab1", "cab_other"])
        self.assertEqual(r["attribute_diffs"]["target"], [0.2, 9.0])
        self.assertIn("fixture_refs.cab", r["ep_meta_diffs"])
        self.assertTrue(r["check_differs_at"] or r["check_steps"][0] == r["check_steps"][1])

    def test_actions_are_fixed_and_bounded(self):
        a1, a2 = rc.actions_for(_Gym(), 5), rc.actions_for(_Gym(), 5)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a1, a2)))
        self.assertLessEqual(max(float(np.max(np.abs(x))) for x in a1), rc.ACTION_SCALE)

    def test_dict_action_space(self):
        class Box:
            def __init__(self, n):
                self.shape, self.low, self.high = (n,), -np.ones(n), np.ones(n)

        class Discrete:
            n = 2

        class Dict:
            spaces = {"action.gripper_close": Box(1), "action.base_motion": Box(4), "action.mode": Discrete()}

        class G(_Gym):
            action_space = Dict()

            def step(self, a):
                self.last = a
                a = np.concatenate([a["action.gripper_close"], a["action.base_motion"][:2]])
                return super().step(a)
        acts = rc.actions_for(G(), 3)
        self.assertEqual(sorted(acts[0]), ["action.base_motion", "action.gripper_close", "action.mode"])
        self.assertEqual(acts[0]["action.mode"], 0)
        self.assertLessEqual(float(np.max(np.abs(acts[0]["action.base_motion"]))), rc.ACTION_SCALE)
        rows = self.run_check(G())
        self.assertTrue(all(r["same"] for r in rows), rows)

    def test_stale_controller_goal_is_caught_and_control_is_clean(self):
        """A controller goal left from the rebuilt scene: the controller differs and the states diverge,
        while two restores of the same state agree with each other."""
        class Ctrl:
            def __init__(self):
                self.goal = np.zeros(3)

        class Robot:
            def __init__(self):
                self.composite_controller = type("CC", (), {})()
                self.composite_controller.part_controllers = {"right": Ctrl()}

        class G(_Gym):
            def __init__(self):
                super().__init__()
                self.env.robots = [Robot()]

            def _ctrl(self):
                return self.env.robots[0].composite_controller.part_controllers["right"]

            def reset(self, **kw):
                out = super().reset(**kw)
                self._ctrl().goal = self.env.x.copy()
                return out

            def reset_to(self, state):
                obs = super().reset_to(state)
                self._ctrl().goal = np.full(3, 7.0)          # not restored: left from a rebuild
                return obs

            def step(self, a):
                self.env.x = self.env.x + 0.01 * (self._ctrl().goal - self.env.x)
                return super().step(np.zeros(3))
        r = self.run_check(G())[0]
        self.assertFalse(r["same"])
        self.assertTrue(any(k.startswith("robot0.right.goal") for k in r["controller_diffs"]), r["controller_diffs"])
        self.assertEqual(r["state_diverges_at_step"], 1)
        self.assertIsNone(r["determinism_control"]["retry_vs_retry2_diverges_at_step"])
        self.assertEqual(r["determinism_control"]["controller_diffs"], {})

    def test_summarize_and_diff(self):
        class Fx:
            name = "counter_1"
        self.assertEqual(rc.summarize(Fx()), "<Fx counter_1>")
        self.assertIsNone(rc.summarize(object()))
        self.assertEqual(rc.diff({"a": 1, "d": {"x": 1}}, {"a": 1, "d": {"x": 2}}), {"d.x": [1, 2]})
        self.assertEqual(rc.diff({"c": [{"n": "a", "p": 1}]}, {"c": [{"n": "a", "p": 2}]}), {"c[0].p": [1, 2]})
