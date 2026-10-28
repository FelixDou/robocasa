import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parents[0] / "scripts" / "xr1_lead_time"
sys.path.insert(0, str(SCRIPTS))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fx = load("ct_fixture", TESTS / "test_xr1_cross_task_lead.py")
cts = load("cross_task_safe_analysis", SCRIPTS / "cross_task_safe.py")
import confirm_lead as cl
import quantile_timer as qt
from robocasa.recovery.safe import cross_task_safe as mod
from robocasa.recovery.safe.analyze_time_safe_hybrid import fit_safe_normalization
from robocasa.recovery.safe.score_seen_checkpoint_external import assign_cutoffs

COHORTS = (("lead_calibration", "cal.jsonl"), ("confirmatory_test", "test.jsonl"))


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pipeline(root, boot=30):
    """Synthetic E1 cohorts, Hide-and-Seek fold dirs, SAFE training runs and scores, assembled SAFE folds."""
    spec, held, hs_dirs = fx.fixture(root)
    tasks = sorted(spec["tasks"])
    bundle = json.loads((root / "bundle.json").read_text())
    H = {t: int(v) for t, v in bundle["task_horizons"].items()}
    rng = np.random.default_rng(3)
    runs = {}
    for F in fx.FOLDS:
        seen = [t for t in tasks if t not in held[F]]
        for s in fx.SEEDS:
            d = root / "train" / f"fold_{F}" / f"indep_seed{s}"
            d.mkdir(parents=True)
            rows = [dict(rollout_id=f"tr-{t}-{i}", task_name=t, split="train",
                         scores=(rng.standard_normal(12) + 0.1 * int(s)).tolist()) for t in seen for i in range(6)]
            write_jsonl(d / "scores.jsonl", rows)
            (d / "metrics.json").write_text(json.dumps(dict(seed=int(s), selection_manifest=str(root / f"fold_{F}.json"))))
            (d / "split_manifest.json").write_text(json.dumps(dict(task_names=seen)))
            (d / "model_final.ckpt").write_bytes(f"{F}-{s}".encode())
            runs[(F, s)] = d
    # The runtime bundle's SAFE normalization is fold all's per-task normalization.
    bundle["task_normalizations"] = {s: fit_safe_normalization(
        [dict(task_name=r["task_name"], scores=np.asarray(r["scores"])) for r in jsonl(runs[("all", s)] / "scores.jsonl")],
        H) for s in fx.SEEDS}
    (root / "bundle.json").write_text(json.dumps(bundle))
    # E1 rows carry SAFE's raw per-seed scores; safe_scores is their normalized seed average.
    for _, name in COHORTS:
        rows = jsonl(root / name)
        for r in rows:
            n = r["source_inferences"]
            ramp = 1.2 * np.linspace(0, 1, n) if r["failed"] else 0.0
            r["per_seed_raw_safe"] = {s: (rng.standard_normal(n) + ramp).tolist() for s in fx.SEEDS}
            raw = {s: np.asarray(v) for s, v in r["per_seed_raw_safe"].items()}
            r["safe_scores"] = fx.ct.normalized(raw, {s: bundle["task_normalizations"][s][r["task_name"]]
                                                       for s in fx.SEEDS}).tolist()
        write_jsonl(root / name, rows)
    # Scores of both cohorts by every fold model (fold all = the frozen E1 trajectories).
    score_dirs = {}
    for F in fx.FOLDS:
        for cohort, name in COHORTS:
            for s in fx.SEEDS:
                d = root / "score" / f"fold_{F}" / cohort / f"seed{s}"
                d.mkdir(parents=True)
                rows = []
                for r in jsonl(root / name):
                    if F == "all":
                        v = r["per_seed_raw_safe"][s]
                    else:
                        n = r["source_inferences"]
                        v = (rng.standard_normal(n) + (0.8 * np.linspace(0, 1, n) if r["failed"] else 0.0)).tolist()
                    rows.append(dict(rollout_id=r["rollout_id"], task_name=r["task_name"], failed=r["failed"], scores=v,
                                     training_task=r["task_name"] not in held[F]))
                write_jsonl(d / "scores.jsonl", rows)
                (d / "provenance.json").write_text(json.dumps(dict(
                    training_run_dir=str(runs[(F, s)]), checkpoint_sha256=sha(runs[(F, s)] / "model_final.ckpt"),
                    group=cohort, unseen_tasks="score", cutoff_rule="training", unseen_task_names=sorted(held[F]),
                    task_names=tasks, export_dir=f"/export/{cohort}")))
                score_dirs[(F, cohort, s)] = d
    safe_dirs = {}
    for F in fx.FOLDS:
        argv = ["assemble", "--fold", F, "--folds", str(root / "folds.json"), "--runtime-bundle", str(root / "bundle.json"),
                "--output-dir", str(root / "safe" / f"fold_{F}")]
        for s in fx.SEEDS:
            argv += ["--training-run", s, str(runs[(F, s)])]
            for cohort, _ in COHORTS:
                argv += ["--scores", cohort, s, str(score_dirs[(F, cohort, s)])]
        if F == "all":
            for cohort, name in COHORTS:
                argv += ["--reference-predictions", cohort, str(root / name)]
        else:
            argv += ["--fold-manifest", str(root / f"fold_{F}.json")]
        with contextlib.redirect_stdout(io.StringIO()):
            assert mod.main(argv) == 0, F
        safe_dirs[F] = root / "safe" / f"fold_{F}"
    # E1's registered result on these cohorts.
    _, cal = cl.primary(root / "cal.jsonl", "calibration")
    _, te = cl.primary(root / "test.jsonl", "test")
    rk = qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)
    tm, fu = cl.fit_fast(cl.Cohort(cal), rk)
    eP, lP = cl.evaluate_fast(cl.Cohort(te), rk, tm)
    eF, lF = cl.evaluate_fast(cl.Cohort(te), rk, fu)
    (root / "e1.json").write_text(json.dumps({"by_q": {str(cl.PRIMARY_Q): dict(
        P=eP, safe_plus_P=eF, gain=round(float(np.mean(lF - lP)), 4))}}, default=float))
    # The cross-task (Hide-and-Seek) result on the same folds.
    with contextlib.redirect_stdout(io.StringIO()):
        assert fx.ct.main(fx.run_args(root, hs_dirs, out="xt.json", boot=boot)) == 0
    return spec, held, hs_dirs, safe_dirs, runs, score_dirs


def analysis_args(root, hs_dirs, safe_dirs, boot=30):
    return ["run", "--calibration", str(root / "cal.jsonl"), "--test", str(root / "test.jsonl"),
            "--safe-fold-dirs", *[str(safe_dirs[F]) for F in fx.FOLDS],
            "--hs-fold-dirs", *[str(hs_dirs[F]) for F in fx.FOLDS], "--runtime-bundle", str(root / "bundle.json"),
            "--e1-result", str(root / "e1.json"), "--cross-task-result", str(root / "xt.json"),
            "--folds", str(root / "folds.json"), "--output", str(root / "safe_out.json"), "--bootstrap", str(boot),
            "--null-draws", "2", "--few-draws", "2"]


class CutoffTests(unittest.TestCase):
    def test_cutoffs_default_refuse_and_unseen_options(self):
        cut, outside = assign_cutoffs(["A", "B"], [10, 4], {"A": 6, "B": 8}, {"A", "B"})
        self.assertEqual((cut, outside), ([6, 4], [False, False]))
        with self.assertRaises(ValueError):
            assign_cutoffs(["A", "C"], [10, 9], {"A": 6}, {"A"})
        cut, outside = assign_cutoffs(["A", "C"], [10, 9], {"A": 6}, {"A"}, unseen_tasks="score")
        self.assertEqual((cut, outside), ([6, 9], [False, True]))
        cut, _ = assign_cutoffs(["A", "C"], [10, 9], {"A": 6}, {"A"}, unseen_tasks="score", cutoff_rule="rollout_length")
        self.assertEqual(cut, [10, 9])
        with self.assertRaises(ValueError):
            assign_cutoffs(["A"], [3], {"A": 2}, {"A"}, cutoff_rule="other")


class ModuleTests(unittest.TestCase):
    def test_manifests_drop_exactly_the_held_out_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = fx.ct.make_folds({t: "atomic" if t in fx.ATOMIC else "composite" for t in fx.TASKS})
            (root / "folds.json").write_text(json.dumps(spec))
            train = [f"tr-{t}-{i}" for t in fx.TASKS for i in range(4)]
            test = [f"te-{t}-{i}" for t in fx.TASKS for i in range(2)]
            (root / "seen.json").write_text(json.dumps(dict(train=train, test=test, included_tasks=list(fx.TASKS),
                                                            split_seed=0, split_unit="rollout")))
            write_jsonl(root / "frozen.jsonl", [dict(rollout_id=i, task_name=i.split("-")[1], split="train") for i in train]
                        + [dict(rollout_id=i, task_name=i.split("-")[1], split="test") for i in test])
            argv = ["manifests", "--selection-manifest", str(root / "seen.json"), "--folds", str(root / "folds.json"),
                    "--frozen-scores", str(root / "frozen.jsonl"), "--output-dir", str(root / "m")]
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                self.assertEqual(mod.main(argv), 0)
            self.assertIn("MANIFESTS OK", buffer.getvalue())
            for f in spec["folds"]:
                m = json.loads((root / "m" / f"fold_{f['fold']}.json").read_text())
                self.assertFalse({i.split("-")[1] for i in m["train"] + m["test"]} & set(f["held_out"]))
                self.assertEqual(len(m["train"]), 4 * (len(fx.TASKS) - len(f["held_out"])))
                self.assertEqual(set(m["dropped_train_by_task"].values()), {4})
            # A manifest that is not the frozen refit's split is refused.
            write_jsonl(root / "frozen.jsonl", [dict(rollout_id=i, task_name=i.split("-")[1], split="train") for i in train[1:]])
            with self.assertRaises(ValueError):
                mod.main(argv[:-1] + [str(root / "m2")])

    def test_compare_reports_identical_and_different_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, v in (("a", [1.0, 2.0]), ("b", [1.0, 2.0]), ("c", [1.0, 2.5])):
                (root / name).mkdir()
                write_jsonl(root / name / "scores.jsonl", [dict(rollout_id="r", scores=v)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mod.main(["compare", str(root / "a"), str(root / "b")]), 0)
                self.assertEqual(mod.main(["compare", str(root / "a"), str(root / "c")]), 1)


class PipelineTests(unittest.TestCase):
    def test_assembly_reproduces_e1_and_the_analysis_reproduces_the_cross_task_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, held, hs_dirs, safe_dirs, _, _ = pipeline(root)
            meta = json.loads((safe_dirs["all"] / "scoring.json").read_text())
            self.assertTrue(meta["checks"]["bundle_normalization"]["ok"])
            self.assertTrue(all(v["ok"] for v in meta["checks"]["reference_predictions"].values()))
            row = jsonl(safe_dirs["0"] / "confirmatory_test_predictions.jsonl")[0]
            self.assertEqual(row["held_out"], row["task_name"] in held["0"])
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                self.assertEqual(cts.main(analysis_args(root, hs_dirs, safe_dirs)), 0)
            self.assertIn("CROSS-TASK SAFE OK", buffer.getvalue())
            out = json.loads((root / "safe_out.json").read_text())
            for k in ("S2_e1_reproduction", "S3_hs_zero_shot", "S3_hs_decision_ci"):
                self.assertTrue(out["checks"][k]["ok"], k)
            self.assertEqual(set(out["U1_S"]), {"ci_lower_above_0", "point_gain_at_least_margin", "fused_fpr_at_most_0p075",
                                                "fused_fpr_wilson_upper_at_most_0p10", "supported"})
            self.assertEqual(set(out["U5_S_few_shot"]), {"1", "3", "5", "all"})
            self.assertIn("hs_earlier", out["U_HS"])

    def test_assembly_refuses_a_score_from_another_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, _, runs, score_dirs = pipeline(root)
            d = score_dirs[("1", "confirmatory_test", "0")]
            prov = json.loads((d / "provenance.json").read_text())
            prov["checkpoint_sha256"] = "0" * 64
            (d / "provenance.json").write_text(json.dumps(prov))
            argv = ["assemble", "--fold", "1", "--folds", str(root / "folds.json"), "--runtime-bundle", str(root / "bundle.json"),
                    "--output-dir", str(root / "again")]
            for s in fx.SEEDS:
                argv += ["--training-run", s, str(runs[("1", s)])]
                for cohort, _ in COHORTS:
                    argv += ["--scores", cohort, s, str(score_dirs[("1", cohort, s)])]
            with self.assertRaises(ValueError):
                mod.main(argv)

    def test_wrong_e1_reference_blocks_the_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, hs_dirs, safe_dirs, _, _ = pipeline(root)
            e1 = json.loads((root / "e1.json").read_text())
            e1["by_q"]["0.95"]["gain"] += 0.01
            (root / "e1.json").write_text(json.dumps(e1))
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                self.assertEqual(cts.main(analysis_args(root, hs_dirs, safe_dirs)), 1)
            self.assertIn("REPRODUCTION FAILED", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
