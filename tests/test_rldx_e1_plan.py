import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "rldx_e1"


def load(name):
    spec = importlib.util.spec_from_file_location(f"rldx_e1_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan_e1, gate = load("plan_e1"), load("check_e1_collection")

TASKS = {"TurnOnSinkFaucet": 600, "PreSoakPan": 2400, "CloseFridge": 900}   # registry horizons
ATOMIC = {"TurnOnSinkFaucet", "CloseFridge"}
N = 10
SEEDS = {"calibration": 1201, "test": 1301}


def raw_rows(source, seed=20):
    rows = []
    for t, H in TASKS.items():
        for i in range(N):
            failed = i % 3 == 0
            rows.append(dict(task_name=t, environment_reset_index=i, environment_seed=seed, seed_protocol="official_rldx",
                             environment_split="target", replan_steps=8, policy_id="RLDX-1-FT-RC365",
                             checkpoint="RLWRLD/RLDX-1-FT-RC365", policy_config={"embodiment_tag": "GENERAL_EMBODIMENT"},
                             model_family="rldx1", video_frame_stride=2, rollout_horizon=H,
                             num_env_steps=H if failed else H // 2, failed=failed,
                             feature_layer="action_model_msat_action_suffix_pre_action_decoder",
                             tensor_path=f"rollouts/{t}-{i}.npz",
                             video_path=f"videos/{t}_{i}.mp4", rollout_id=f"{t}-{i}"))
            (Path(source) / "videos").mkdir(parents=True, exist_ok=True)
            (Path(source) / "videos" / f"{t}_{i}.mp4").write_bytes(b"x")
    return rows


def config(**over):
    c = dict(model_family="rldx1", policy_module="robocasa.recovery.rldx_zmq_policy:make_policy", base_environment_seed=20,
             split="target", replan_steps=8, policy_checkpoint="RLWRLD/RLDX-1-FT-RC365", record_actions=True,
             record_safe_features=True, record_subtask_trace=False, record_videos=True, success_quota=None,
             failure_quota=None, retain_only_quota=False, tasks=list(TASKS),
             task_types={t: ("atomic" if t in ATOMIC else "composite") for t in TASKS})
    c.update(over)
    return c


def freeze(**over):
    tasks = {t: {"type": "atomic" if t in ATOMIC else "composite", "H_k": -(-H // 8), "rho_k": 0.6} for t, H in TASKS.items()}
    tasks.update(over)
    return {"tasks": tasks}


def build(tmp, frz=None, seeds=None, used=None, k=2):
    rows = raw_rows(tmp)
    return plan_e1.build(rows, tmp, [(Path(tmp) / "summary.json", config())], dict(TASKS), ATOMIC,
                         frz or freeze(), seeds or dict(SEEDS), used or {}, k,
                         frame_shape_fn=lambda path: [256, 384, 3])


class PlanTest(unittest.TestCase):
    def test_settings_copied_and_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, plan_tasks, settings, shards, cost = build(tmp)
            self.assertEqual(settings["seed"], 20)
            self.assertNotIn("horizon_scale", settings)
            f = plan_e1.flags(settings, 1201)
            self.assertNotIn("--horizon-scale", f)
            self.assertEqual(f[f.index("--seed") + 1], "1201")
            self.assertEqual(f[f.index("--num-rollouts") + 1], str(plan_e1.ATTEMPT_CAP))
            self.assertEqual(f[f.index("--success-quota") + 1], "20")
            self.assertEqual(f[f.index("--failure-quota") + 1], "20")
            self.assertIn("--retain-only-quota", f)
            self.assertEqual(f[f.index("--replan-steps") + 1], "8")
            self.assertEqual(f[f.index("--seed-protocol") + 1], "official_rldx")
            self.assertEqual(f[f.index("--video-height") + 1], "256")
            self.assertEqual(plan_tasks["PreSoakPan"]["H_inferences"], 300)

    def test_shards_cover_every_task_once_per_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, _, _, shards, _ = build(tmp)
            for name in plan_e1.SETS:
                got = sorted(t for s in shards if s["set"] == name for t in s["tasks"])
                self.assertEqual(got, sorted(TASKS))
            self.assertEqual([s["gpu"] for s in shards], [0, 1, 2, 3])
            self.assertEqual([s["port"] for s in shards], [20150, 20151, 20152, 20153])

    def test_stops_on_reused_or_equal_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                build(tmp, seeds={"calibration": 20, "test": 1301})
            with self.assertRaises(SystemExit):
                build(tmp, seeds={"calibration": 1201, "test": 1201})
            with self.assertRaises(SystemExit):
                build(tmp, used={1301: ["x/summary.json"]})

    def test_stops_on_freeze_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                build(tmp, frz=freeze(PreSoakPan={"type": "composite", "H_k": 299, "rho_k": 0.5}))
            frz = freeze()
            frz["tasks"]["OpenDrawer"] = {"type": "atomic", "H_k": 50, "rho_k": 0.5}
            with self.assertRaises(SystemExit):
                build(tmp, frz=frz)

    def test_expected_attempts(self):
        mean, p95, over = plan_e1.expected_attempts(0.5)
        self.assertTrue(40 < mean < 55 and p95 >= mean and over == 0.0)
        mean, _, _ = plan_e1.expected_attempts(0.9)
        self.assertTrue(180 < mean < 225)

    def test_used_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, cfg in (("a", config(base_environment_seed=7)), ("b", config(base_environment_seed=None)),
                              ("c", config(model_family="pi0", base_environment_seed=99))):
                d = Path(tmp) / name
                d.mkdir()
                (d / "summary.json").write_text(json.dumps({"config": cfg}))
            self.assertEqual(sorted(plan_e1.used_seeds([tmp])), [7])


def new_rows(name, seed, tasks=TASKS, s=20, f=20):
    rows = []
    for t, H in tasks.items():
        for i in range(s + f):
            rows.append(dict(task_name=t, environment_reset_index=i, environment_seed=seed, seed_protocol="official_rldx",
                             environment_split="target", replan_steps=8, policy_id="RLDX-1-FT-RC365",
                             checkpoint="RLWRLD/RLDX-1-FT-RC365", policy_config={"embodiment_tag": "GENERAL_EMBODIMENT"},
                             model_family="rldx1", video_frame_stride=2, rollout_horizon=H, failed=i >= s,
                             feature_layer="action_model_msat_action_suffix_pre_action_decoder",
                             tensor_path=f"rollouts/{name}-{t}-{i}.npz", rollout_id=f"{name}-{t}-{i}"))
    return rows


def write_shard(root, rows, errors=()):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if errors:
        (root / "errors.jsonl").write_text("".join(json.dumps(e) + "\n" for e in errors))


def fake_tensor(shape=(3, 10, 16, 2048)):
    return lambda root, row: dict(shape=list(shape), dtype="float32", finite=True, nonzero=True, steps=[0, 8, 16])


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        _, plan_tasks, settings, _, _ = build(t)
        self.plan = dict(settings=settings, seeds=dict(SEEDS), tasks=plan_tasks, quota=20)
        self.recorded = raw_rows(t)
        self.cal, self.test = Path(t) / "cal", Path(t) / "test"
        write_shard(self.cal, new_rows("c", 1201))
        write_shard(self.test, new_rows("t", 1301))

    def tearDown(self):
        self.tmp.cleanup()

    def run_gate(self, shards=None, smoke=False, tensor=None):
        shards = shards or [("calibration", str(self.cal)), ("test", str(self.test))]
        return gate.check(self.plan, shards, self.recorded, self.tmp.name, smoke=smoke, tensor_fn=tensor or fake_tensor())

    def test_ok(self):
        problems, counts = self.run_gate()
        self.assertEqual(problems, [])
        self.assertEqual(counts[("test", "PreSoakPan")]["failures"], 20)

    def test_wrong_seed_and_duplicates(self):
        write_shard(self.test, new_rows("c", 1201))            # same IDs and the calibration seed
        problems, _ = self.run_gate()
        self.assertTrue(any("does not belong to test" in p for p in problems))
        self.assertTrue(any("more than once" in p for p in problems))

    def test_quota_and_errors(self):
        write_shard(self.cal, new_rows("c", 1201, s=20, f=19), errors=[{"rollout_id": "x"}])
        problems, _ = self.run_gate()
        self.assertTrue(any("19 failures" in p for p in problems))
        self.assertTrue(any("collection errors" in p for p in problems))
        problems, _ = self.run_gate(shards=[("calibration", str(self.cal))], smoke=True)
        self.assertEqual(problems, [])

    def test_feature_contract(self):
        calls = {"n": 0}

        def tensor(root, row):
            calls["n"] += 1
            wide = str(root) != self.tmp.name
            return dict(shape=[3, 10, 16, 1024 if wide else 2048], dtype="float32", finite=True, nonzero=True,
                        steps=[0, 8, 16])
        problems, _ = self.run_gate(tensor=tensor)
        self.assertTrue(any("vs recorded" in p for p in problems))

    def test_smoke_set(self):
        smoke = Path(self.tmp.name) / "smoke"
        write_shard(smoke, new_rows("s", 1101, tasks={"TurnOnSinkFaucet": 600}, s=1, f=1))
        self.plan["seeds"] = {**self.plan["seeds"], "smoke": 1101}
        problems, _ = self.run_gate(shards=[("smoke", str(smoke))], smoke=True)
        self.assertEqual(problems, [])

    def test_setting_mismatch(self):
        rows = new_rows("c", 1201)
        rows[0]["replan_steps"] = 16
        write_shard(self.cal, rows)
        problems, _ = self.run_gate()
        self.assertTrue(any("replan_steps 16" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
