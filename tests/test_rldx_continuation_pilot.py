import argparse
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "rldx_continuation_pilot"


def load(name):
    spec = importlib.util.spec_from_file_location(f"rldx_pilot_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan, analyze, decide = load("plan_pilot"), load("analyze_pilot"), load("decide")

TASKS = {"TurnOnSinkFaucet": 600, "PreSoakPan": 2400}   # registry horizons; atomic, composite
ATOMIC = {"TurnOnSinkFaucet"}
N = 10


def raw_rows(source):
    rows = []
    for t, H in TASKS.items():
        for i in range(N):
            failed = i % 2 == 0
            rows.append(dict(task_name=t, environment_reset_index=i, environment_seed=7, seed_protocol="official_rldx",
                             environment_split="target", replan_steps=8, policy_id="RLDX-1-FT-RC365",
                             checkpoint="RLWRLD/RLDX-1-FT-RC365", policy_config={"embodiment_tag": "GENERAL_EMBODIMENT"},
                             model_family="rldx1", video_frame_stride=2, rollout_horizon=H,
                             num_env_steps=H if failed else H // 2, failed=failed,
                             video_path=f"videos/{t}_{i}.mp4", rollout_id=f"{t}-{i}"))
            (Path(source) / "videos").mkdir(parents=True, exist_ok=True)
            (Path(source) / "videos" / f"{t}_{i}.mp4").write_bytes(b"x")
    return rows


def config(**over):
    c = dict(model_family="rldx1", policy_module="robocasa.recovery.rldx_zmq_policy:make_policy", base_environment_seed=7,
             split="target", replan_steps=8, policy_checkpoint="RLWRLD/RLDX-1-FT-RC365", record_actions=True,
             record_safe_features=True, record_subtask_trace=False, record_videos=True, success_quota=None,
             failure_quota=None, retain_only_quota=False, tasks=list(TASKS),
             task_types={"TurnOnSinkFaucet": "atomic", "PreSoakPan": "composite"})
    c.update(over)
    return c


def build(tmp, configs=None, rows=None):
    rows = rows if rows is not None else raw_rows(tmp)
    return plan.build(rows, tmp, [(Path(tmp) / "summary.json", c) for c in (configs or [config()])], dict(TASKS), ATOMIC,
                      frame_shape_fn=lambda path: [256, 384, 3])


class TestPlan(unittest.TestCase):
    def test_build_and_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, out, settings = build(tmp)
            self.assertEqual(tasks, sorted(TASKS))
            self.assertEqual(out["TurnOnSinkFaucet"]["H_scaled"], 750)
            self.assertEqual(out["PreSoakPan"]["H_scaled"], 3000)
            self.assertEqual(out["PreSoakPan"]["type"], "composite")
            self.assertEqual(out["TurnOnSinkFaucet"]["p"], 0.5)
            self.assertEqual(settings["num_rollouts"], N)
            self.assertEqual((settings["video_height"], settings["video_width"]), (256, 384))
            f = plan.flags(settings)
            self.assertEqual(f[f.index("--horizon-scale") + 1], "1.25")
            self.assertEqual(f[f.index("--seed-protocol") + 1], "official_rldx")
            self.assertEqual(json.loads(f[f.index("--policy-config") + 1]), {"embodiment_tag": "GENERAL_EMBODIMENT"})
            self.assertIn("--record-videos", f)
            s0, s1 = plan.shards(tasks, out)
            self.assertEqual(sorted(s0 + s1), tasks)

    def test_main_writes_plan_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"; source.mkdir()
            rows = raw_rows(source)
            (source / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            shard = Path(tmp) / "collection" / "shard0"; shard.mkdir(parents=True)
            (shard / "summary.json").write_text(json.dumps(dict(config=config())))
            (Path(tmp) / "collection" / "merged_summary.json").write_text("{}")
            out = Path(tmp) / "pilot"
            old_argv, old_fn = sys.argv, plan.first_frame_shape
            sys.argv = ["plan_pilot", str(source), "--collection-root", str(Path(tmp) / "collection"), "--output", str(out)]
            plan.first_frame_shape = lambda path: [256, 384, 3]
            try:
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    plan.main()
            finally:
                sys.argv, plan.first_frame_shape = old_argv, old_fn
            self.assertIn("PLAN OK", buf.getvalue())
            sel = json.loads((out / "selection.json").read_text())
            self.assertEqual(sel["cost"]["rollouts"], 2 * N)
            self.assertEqual(sel["settings"]["original_videos_present"], 2 * N)
            flags = (out / "collector_flags.txt").read_text().splitlines()
            self.assertEqual(flags[flags.index("--num-rollouts") + 1], str(N))
            shards = [(out / f"tasks_shard{i}.txt").read_text().split() for i in (0, 1)]
            self.assertEqual(sorted(shards[0] + shards[1]), sorted(TASKS))
            self.assertEqual(json.loads((out / "attempt_limits_shard0.json").read_text()), {})
            with self.assertRaises(SystemExit):     # never overwrite a plan
                sys.argv = ["plan_pilot", str(source), "--collection-root", str(Path(tmp) / "collection"), "--output", str(out)]
                try:
                    plan.main()
                finally:
                    sys.argv = old_argv

    def test_fields_missing_from_old_rows_come_from_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = raw_rows(tmp)
            for r in rows:
                r.pop("video_frame_stride"); r.pop("model_family")
            _, _, settings = build(tmp, configs=[config(video_frame_stride=2)], rows=rows)
            self.assertEqual((settings["video_frame_stride"], settings["model_family"]), (2, "rldx1"))
            with self.assertRaises(SystemExit):
                build(tmp, configs=[config()], rows=rows)   # stride recorded nowhere

    def test_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):        # registry horizon changed since the collection
                plan.build(raw_rows(tmp), tmp, [(None, config())], {"TurnOnSinkFaucet": 500, "PreSoakPan": 2400}, ATOMIC,
                           frame_shape_fn=lambda p: [1, 1, 3])
            with self.assertRaises(SystemExit):        # quota collection: not natural rates
                build(tmp, configs=[config(success_quota=10)])
            with self.assertRaises(SystemExit):        # settings disagree across shards
                build(tmp, configs=[config(), config(policy_module="other:make_policy")])
            rows = raw_rows(tmp); rows[0]["environment_seed"] = 8
            with self.assertRaises(SystemExit):        # more than one seed
                build(tmp, rows=rows)
            rows = raw_rows(tmp); rows[1]["environment_reset_index"] = 0
            with self.assertRaises(SystemExit):        # duplicate reset index
                build(tmp, rows=rows)


def write_pilot(root, sel, outcome, *, horizon_scale=1.25, drop=(), extra=None, frame_names=None):
    """outcome(task, i) -> (failed, L). Writes one shard with a manifest, summary and video files."""
    shard = Path(root) / "shard0"
    (shard / "videos").mkdir(parents=True, exist_ok=True)
    rows = []
    for t, v in sel["tasks"].items():
        for i in v["scenes"]:
            if (t, i) in drop:
                continue
            failed, L = outcome(t, int(i))
            name = f"videos/new_{t}_{i}.mp4"
            (shard / name).write_bytes(b"x")
            rows.append(dict(task_name=t, environment_reset_index=int(i), environment_seed=7, seed_protocol="official_rldx",
                             environment_split="target", replan_steps=8, policy_id="RLDX-1-FT-RC365",
                             checkpoint="RLWRLD/RLDX-1-FT-RC365", rollout_horizon=v["H_scaled"], num_env_steps=L,
                             failed=failed, video_path=name))
    rows += extra or []
    (shard / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (shard / "summary.json").write_text(json.dumps(dict(config=dict(horizon_scale=horizon_scale))))
    return shard


def selection(tmp):
    tasks, out, settings = build(tmp)
    return dict(settings=settings, tasks=out)


def frames_same(path):
    return np.zeros((4, 4, 3))


def frames_differ_for(bad):
    def fn(path):
        name = Path(path).name
        return np.full((4, 4, 3), 50.0) if name.startswith("new_") and any(name == f"new_{t}_{i}.mp4" for t, i in bad) \
            else np.zeros((4, 4, 3))
    return fn


def outcome_fn(H):
    # Original failures (even i): i = 0, 4, 8 succeed late (L = 1.1 H); i = 2, 6 fail at 1.25 H.
    # Original successes (odd i): i = 1, 3 succeed by H; i = 5, 7, 9 fail at 1.25 H.
    def f(t, i):
        h = H[t]
        if i % 2 == 0:
            return (False, int(1.1 * h)) if i % 4 == 0 else (True, int(math.ceil(1.25 * h)))
        return (False, h // 2) if i in (1, 3) else (True, int(math.ceil(1.25 * h)))
    return f


def run_analyze(tmp, sel, frame_fn, check_sel=None, **kw):
    """Writes the pilot from `sel`; the analysis reads `check_sel` (default: the same)."""
    (Path(tmp) / "sel.json").write_text(json.dumps(check_sel or sel))
    pilot = Path(tmp) / "pilot"; write_pilot(pilot, sel, outcome_fn(TASKS), **kw)
    args = argparse.Namespace(pilot=str(pilot), selection=str(Path(tmp) / "sel.json"),
                              output=str(Path(tmp) / f"out_{len(list(Path(tmp).glob('out_*')))}.json"), bootstrap=200)
    with contextlib.redirect_stdout(io.StringIO()):
        analyze.run(args, frame_fn=frame_fn)
    return json.loads(Path(args.output).read_text())


class TestAnalyze(unittest.TestCase):
    def test_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            sel = selection(tmp)
            res = run_analyze(tmp, sel, frames_same)
            self.assertTrue(res["gates"]["scene"]["passed"])
            # Per task: 8 rollouts without success by H (3 late + 5 at the limit) -> c = 3/8.
            self.assertEqual((res["c"]["late_successes"], res["c"]["no_success_by_H"]), (6, 16))
            self.assertAlmostEqual(res["c"]["rate"], 0.375)
            self.assertEqual(res["c"]["by_original_outcome"]["originally failed"]["late"], 6)
            # Retries of originally failed scenes never succeed by H; of successful ones 2 of 5 do.
            self.assertEqual((res["r_fail"]["successes"], res["r_fail"]["n"]), (0, 10))
            self.assertEqual((res["r_succ"]["successes"], res["r_succ"]["n"]), (4, 10))
            lo, hi = res["c"]["decision95"]
            self.assertLessEqual(lo, min(res["c"]["wilson95"][0], res["c"]["cluster95"][0]))
            si = res["sizing_inputs"]
            self.assertEqual(si["retry_success"], 0.0)
            self.assertEqual(si["continuation"][1], 0.375)

    def test_scene_gate_failure_voids_r_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            sel = selection(tmp)
            bad = [("TurnOnSinkFaucet", "0"), ("TurnOnSinkFaucet", "1")]   # 2 of 20 = 90% same scene
            res = run_analyze(tmp, sel, frames_differ_for(bad))
            self.assertFalse(res["gates"]["scene"]["passed"])
            self.assertFalse(res["r_fail"]["used"])
            self.assertEqual(res["sizing_inputs"]["retry_success"], analyze.FALLBACK_R)
            self.assertAlmostEqual(res["c"]["rate"], 0.375)               # c does not need pairing
            self.assertEqual(res["r_fail"]["n"] + res["r_succ"]["n"], 18)  # mismatched pairs excluded

    def test_identity_gate_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            sel = selection(tmp)
            for kw in (dict(horizon_scale=1.0),
                       dict(drop={("PreSoakPan", str(i)) for i in range(3)}),   # task coverage 70%
                       dict(extra=[dict(task_name="PreSoakPan", environment_reset_index=99)])):
                with self.assertRaises(SystemExit), tempfile.TemporaryDirectory() as sub:
                    run_analyze(sub, sel, frames_same, **kw)

    def test_wrong_horizon_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            sel = selection(tmp)
            check = json.loads(json.dumps(sel))
            check["tasks"]["PreSoakPan"]["H_scaled"] = 2400   # pilot ran to 3000, not the planned horizon
            with self.assertRaises(SystemExit):
                run_analyze(tmp, sel, frames_same, check_sel=check)


def e3_run(r, slope, c_star, cs):
    """Synthetic E3 output: reset - continuation = slope * (c_star - c); pairs ~ 1 / delta^2."""
    cont = {}
    for c in cs:
        d = round(slope * (c_star - c), 6)
        pairs = math.inf if d == 0 else math.ceil(1.2 / d ** 2)
        cont[str(float(c))] = {"P": dict(reset=0.6, cont=round(0.6 - d, 4), delta=d, paired=pairs, unpaired=pairs * 4,
                                         p10=0.1 + max(d, 0), p01=0.1 + max(-d, 0))}
    return dict(retry_success=str(r), policies={"rldx1": {"by_type": {"all": {"continuation": cont}}}})


GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.27, 0.4)


class TestDecide(unittest.TestCase):
    def pilot(self, lo, hat, hi, r=0.3, r_low=0.22):
        return dict(sizing_inputs=dict(continuation=[lo, hat, hi], retry_success=r, retry_success_low=r_low))

    def runs(self, cs, c_star=0.10, c_star_low=0.09):
        return {"retry_success": e3_run(0.3, 0.44, c_star, GRID + cs), "retry_success_low": e3_run(0.22, 0.44, c_star_low, GRID + cs)}

    def test_register_when_interval_excludes_crossover(self):
        cs = (0.21, 0.25, 0.30)
        res = decide.decide(self.pilot(*cs), self.runs(cs))
        self.assertEqual(res["decision"], "REGISTER")
        self.assertEqual(res["expected_direction"], "continuation better")
        self.assertEqual(res["pairs"] % 20, 0)
        self.assertGreaterEqual(res["pairs"], decide.FLOOR)
        self.assertAlmostEqual(res["runs"]["retry_success"]["crossover_c"], 0.10, places=3)

    def test_estimation_when_interval_straddles_crossover(self):
        cs = (0.06, 0.09, 0.13)
        res = decide.decide(self.pilot(*cs), self.runs(cs))
        self.assertEqual(res["decision"], "ESTIMATION")
        self.assertEqual(res["pairs"], decide.CAP)
        self.assertGreater(res["expected_ci95_half_width"], 0)

    def test_estimation_when_too_many_pairs(self):
        cs = (0.13, 0.14, 0.15)    # decided (all above c*), but delta at 0.13 is small
        res = decide.decide(self.pilot(*cs), self.runs(cs))
        self.assertTrue(res["direction_decided"])
        self.assertEqual(res["decision"], "ESTIMATION")

    def test_missing_c_stops(self):
        with self.assertRaises(SystemExit):
            decide.decide(self.pilot(0.01, 0.02, 0.03), self.runs(()))


direct = load("direct_e3")


class TestDirect(unittest.TestCase):
    def test_arms(self):
        # H = 100: limit 125. f = 0.8: alarm at 80, 45 steps left after it.
        self.assertEqual(direct.arms(True, 50, False, 100, 100, 0.8), (True, True, True, False))    # done before the alarm
        self.assertEqual(direct.arms(True, 110, True, 40, 100, 0.8), (True, True, False, False))    # late success; retry wins
        self.assertEqual(direct.arms(False, 125, True, 60, 100, 0.8), (False, False, False, False))  # retry too slow
        self.assertEqual(direct.arms(False, 125, True, 20, 100, 0.8), (False, True, True, False))   # retry fits 25 too
        # f = 0.3: alarm at 30, 95 left, but the retry's own alarm comes at 30.
        self.assertEqual(direct.arms(False, 125, True, 50, 100, 0.3), (False, False, False, True))

    def test_run_and_void(self):
        with tempfile.TemporaryDirectory() as tmp:
            sel = selection(tmp)
            res = run_analyze(tmp, sel, frames_same)
            (Path(tmp) / "pilot.json").write_text(json.dumps(res))
            args = argparse.Namespace(pilot_json=str(Path(tmp) / "pilot.json"), pilot_collection=str(Path(tmp) / "pilot"),
                                      selection=str(Path(tmp) / "sel.json"), rldx_calibration_dir=None,
                                      output=str(Path(tmp) / "direct.json"), bootstrap=200)
            with contextlib.redirect_stdout(io.StringIO()):
                direct.run(args, fractions={t: 0.8 for t in TASKS})
            out = json.loads(Path(args.output).read_text())
            self.assertEqual(out["all"]["pairs"], 2 * N)
            # Continuation wins 3 late successes per task + 2 early ones = 5/10. Reset: the 2 early
            # successes, plus retries of alarmed scenes whose original attempt fits in 0.45 H
            # (originally successful scenes, L = 0.5 H, do not fit) -> 2/10.
            self.assertAlmostEqual(out["all"]["task_macro"]["cont"], 0.5)
            self.assertAlmostEqual(out["all"]["task_macro"]["reset"], 0.2)
            self.assertAlmostEqual(out["all"]["reset_minus_cont"]["task_macro"], -0.3)
            self.assertEqual(out["all"]["reset_minus_cont"]["discordant"], dict(reset_only=0, cont_only=6))
            self.assertEqual(direct.mcnemar(0, 6), round(2 / 64, 4))
            with self.assertRaises(SystemExit):     # P tasks must equal the pilot tasks
                args.output = str(Path(tmp) / "direct2.json")
                direct.run(args, fractions={"TurnOnSinkFaucet": 0.8})
            res["gates"]["scene"]["passed"] = False
            (Path(tmp) / "pilot.json").write_text(json.dumps(res))
            args.output = str(Path(tmp) / "direct3.json")
            with contextlib.redirect_stdout(io.StringIO()):
                direct.run(args, fractions={t: 0.8 for t in TASKS})
            self.assertTrue(json.loads(Path(args.output).read_text())["status"].startswith("void"))

    def test_decide_downgrades_on_contradiction(self):
        cs = (0.21, 0.25, 0.30)
        t = TestDecide()
        pilot, runs = t.pilot(*cs), t.runs(cs)
        agree = dict(all=dict(reset_minus_cont=dict(task_macro=-0.05, cluster95=[-0.09, -0.01])))
        against = dict(all=dict(reset_minus_cont=dict(task_macro=0.05, cluster95=[0.01, 0.09])))
        self.assertEqual(decide.decide(pilot, runs, agree)["decision"], "REGISTER")
        res = decide.decide(pilot, runs, against)
        self.assertEqual(res["decision"], "ESTIMATION")
        self.assertIn("contradicts", res["reason"])


if __name__ == "__main__":
    unittest.main()
