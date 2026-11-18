"""The batch scorer loads an export once and writes exactly what the single-checkpoint scorer writes.

The official SAFE code, torch and omegaconf are replaced by small stand-ins, so both scorers run here
end to end through their real code paths and their files are compared byte for byte.
"""
import contextlib
import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe import score_seen_checkpoint_batch as batch  # noqa: E402
from robocasa.recovery.safe import score_seen_checkpoint_external as single  # noqa: E402
from robocasa.recovery.safe import train_seen_tasks  # noqa: E402

TASKS = {"TaskA": 0, "TaskB": 1, "TaskC": 2}


class Cfg(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key, value):
        self[key] = value


def to_cfg(value):
    return Cfg({k: to_cfg(v) for k, v in value.items()}) if isinstance(value, dict) else value


class T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def sum(self, dim):
        return T(self.a.sum(axis=dim))


class Rollout(SimpleNamespace):
    def to(self, device):
        return self


LOADS = []


def fake_load(root, cfg):
    LOADS.append(cfg.dataset.feat_name)
    out = []
    for i, (_, env) in enumerate(train_seen_tasks.load_env_records(root)):
        rng = np.random.default_rng(i)
        h = rng.normal(size=(env["model_infer_times"], 3)) * (2.0 if cfg.dataset.feat_name == "b" else 1.0)
        out.append(Rollout(hidden_states=h, task_id=env["task_id"], episode_success=env["episode_success"],
                           task_min_step=None))
    cfg.dataset.dim_features, cfg.dataset.dim_action = 3, 7
    cfg.dataset.pred_horizon, cfg.dataset.exec_horizon = 50, 5
    return out


def fake_forward(model, loader):
    cfg, rollouts = loader.dataset.cfg, loader.dataset.rollouts
    # The checkpoint's configuration must carry the loader's shapes, as in a single-checkpoint run.
    assert cfg.dataset.dim_features == 3 and cfg.dataset.exec_horizon == 5
    n = max(len(r.hidden_states) for r in rollouts)
    scores, masks = np.zeros((len(rollouts), n)), np.zeros((len(rollouts), n))
    for i, r in enumerate(rollouts):
        s = np.cumsum(r.hidden_states[:, 0]) * (1 + cfg.train.seed) / 7 + (0.5 if cfg.model.name == "lstm" else 0.0)
        scores[i, : len(s)], masks[i, : len(s)] = s, 1
    return T(scores), T(masks), None


def stub_modules():
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch.load = lambda path, map_location=None, weights_only=None: {}
    torch.no_grad = contextlib.nullcontext
    tu, tud = types.ModuleType("torch.utils"), types.ModuleType("torch.utils.data")
    tud.DataLoader = lambda dataset, batch_size, shuffle, num_workers: SimpleNamespace(dataset=dataset)
    torch.utils, tu.data = tu, tud
    om = types.ModuleType("omegaconf")
    om.OmegaConf = SimpleNamespace(load=lambda path: to_cfg(json.loads(Path(path).read_text())))
    fp = {n: types.ModuleType(n) for n in ("failure_prob", "failure_prob.data", "failure_prob.data.pizero",
                                           "failure_prob.data.utils", "failure_prob.model", "failure_prob.utils",
                                           "failure_prob.utils.random", "failure_prob.utils.routines")}
    fp["failure_prob.data.pizero"].load_rollouts_from_root = fake_load
    fp["failure_prob.data.utils"].RolloutDataset = lambda cfg, rollouts: SimpleNamespace(cfg=cfg, rollouts=rollouts)
    fp["failure_prob.model"].get_model = lambda cfg, dim: SimpleNamespace(
        load_state_dict=lambda state, strict: None, to=lambda device: None, eval=lambda: None)
    fp["failure_prob.utils.random"].seed_everything = lambda seed: None
    fp["failure_prob.utils.routines"].model_forward_dataloader = fake_forward
    return {"torch": torch, "torch.utils": tu, "torch.utils.data": tud, "omegaconf": om, **fp}


class BatchScorerTest(unittest.TestCase):
    def setUp(self):
        LOADS.clear()
        self.tmp = Path(tempfile.mkdtemp())
        self.export = self.tmp / "export"
        (self.export / "env_records").mkdir(parents=True)
        (self.export / "conversion_report.json").write_text(json.dumps(
            dict(task_ids=TASKS, task_types={"TaskA": "atomic", "TaskB": "atomic", "TaskC": "composite"})))
        for i in range(12):
            task = list(TASKS)[i % 3]
            env = dict(rollout_id=f"r{i:02d}", task_id=TASKS[task], episode_success=int(i % 4 != 0), replan_steps=5,
                       model_infer_times=6 + i % 5, environment_seed=100 + i, video_frame_stride=1)
            with open(self.export / "env_records" / f"{i}.pkl", "wb") as f:
                pickle.dump(env, f)
        self.runs = []
        for k, (name, seed, feat, tasks) in enumerate([("mlp", 0, "a", ["TaskA", "TaskB"]), ("mlp", 1, "a", ["TaskA", "TaskB", "TaskC"]),
                                                       ("lstm", 2, "a", ["TaskA", "TaskC"]), ("mlp", 3, "b", ["TaskB"])]):
            run = self.tmp / f"run{k}"
            run.mkdir()
            (run / "config.yaml").write_text(json.dumps(dict(dataset=dict(feat_name=feat, horizon_idx_rel=0.0, diff_idx_rel=1.0),
                                                             train=dict(seed=seed), model=dict(name=name, batch_size=4))))
            (run / "metrics.json").write_text(json.dumps(dict(task_min_steps={t: 5 for t in tasks})))
            (run / "split_manifest.json").write_text(json.dumps(dict(task_names=tasks)))
            (run / "model_final.ckpt").write_bytes(bytes([k]) * 64)
            self.runs.append(run)
        self.patches = [mock.patch.dict(sys.modules, stub_modules()),
                        mock.patch.object(train_seen_tasks, "verify_safe_repo", lambda repo: Path(repo))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def run_single(self, run, out):
        return single.score_external_export(export_dir=self.export, safe_repo=self.tmp, training_run_dir=run, output_dir=out,
                                            group="calibration", device="cpu", unseen_tasks="score")

    def run_batch(self, jobs, **kw):
        logs = []
        res = batch.score_export_batch(export_dir=self.export, safe_repo=self.tmp, jobs=jobs, group="calibration",
                                       device="cpu", unseen_tasks="score", log=logs.append, **kw)
        return res, logs

    def same_files(self, a, b):
        for name in ("scores.jsonl", "provenance.json"):
            self.assertEqual((Path(a) / name).read_bytes(), (Path(b) / name).read_bytes(), name)

    def test_batch_writes_what_the_single_scorer_writes_with_one_load_per_feature_selection(self):
        for k, run in enumerate(self.runs):
            self.run_single(run, self.tmp / "single" / str(k))
        LOADS.clear()
        jobs = [(run, self.tmp / "batch" / str(k)) for k, run in enumerate(self.runs)]
        res, logs = self.run_batch(jobs)
        self.assertEqual(LOADS, ["a", "b"])
        self.assertEqual([r["action"] for r in res], ["write"] * 4)
        for k in range(4):
            self.same_files(self.tmp / "single" / str(k), self.tmp / "batch" / str(k))
        rows = [json.loads(line) for line in (self.tmp / "batch" / "2" / "scores.jsonl").read_text().splitlines()]
        self.assertEqual({r["task_name"]: r["training_task"] for r in rows}, {"TaskA": True, "TaskB": False, "TaskC": True})

    def test_verify_existing_checks_first_and_last_and_writes_the_rest(self):
        outs = [self.tmp / "out" / str(k) for k in range(3)]
        self.run_single(self.runs[0], outs[0])
        self.run_single(self.runs[2], outs[2])
        jobs = [(self.runs[k], outs[k]) for k in range(3)]
        with self.assertRaises(FileExistsError):
            self.run_batch(jobs)
        res, logs = self.run_batch(jobs, verify_existing=True)
        self.assertEqual([r["action"] for r in res], ["verify", "write", "verify"])
        self.assertTrue(all(r["ok"] for r in res if r["action"] == "verify"))
        self.assertEqual([m.split("] ")[1].split()[0] for m in logs if m.startswith("[")], ["verified", "wrote", "verified"])
        self.run_single(self.runs[1], self.tmp / "ref1")
        self.same_files(self.tmp / "ref1", outs[1])

    def test_a_changed_stored_score_or_field_is_a_mismatch_and_partial_output_stops(self):
        out = self.tmp / "out0"
        self.run_single(self.runs[0], out)
        path = out / "scores.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[3]["scores"][1] += 1e-9
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        with self.assertRaises(RuntimeError):
            self.run_batch([(self.runs[0], out)], verify_existing=True)
        res, _ = self.run_batch([(self.runs[0], out)], verify_existing=True, tolerance=1e-6)
        self.assertTrue(res[0]["ok"])
        self.assertGreater(res[0]["max_abs_diff"], 0)
        rows[3]["task_min_step"] += 1
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        with self.assertRaises(RuntimeError):
            self.run_batch([(self.runs[0], out)], verify_existing=True, tolerance=1e-6)
        partial = self.tmp / "partial"
        partial.mkdir()
        (partial / "scores.jsonl").write_text("")
        with self.assertRaises(ValueError):
            self.run_batch([(self.runs[1], partial)], verify_existing=True)
        with self.assertRaises(ValueError):
            self.run_batch([(self.runs[1], self.tmp / "x"), (self.runs[2], self.tmp / "x")])

    def test_cli(self):
        out = self.tmp / "cli"
        argv = ["--export-dir", str(self.export), "--safe-repo", str(self.tmp), "--group", "calibration", "--device", "cpu",
                "--unseen-tasks", "score", "--job", str(self.runs[0]), str(out / "0"), "--job", str(self.runs[1]), str(out / "1")]
        with mock.patch("builtins.print") as printed:
            batch.main(argv)
        self.assertIn("BATCH OK: 2 written, 0 verified", printed.call_args_list[-1].args[0])
        with mock.patch("builtins.print") as printed:
            batch.main(argv + ["--verify-existing"])
        self.assertIn("BATCH OK: 0 written, 2 verified (max |diff| 0)", printed.call_args_list[-1].args[0])


if __name__ == "__main__":
    unittest.main()
