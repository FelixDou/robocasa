import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robocasa.recovery.safe.prospective_matched_fpr import event_metrics, select_single_threshold
from robocasa.recovery.safe.run_prospective_matched_fpr import _select_per_task_class

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "xr1_lead_time"
TASKS, HORIZON = ("TaskA", "TaskB", "TaskC", "TaskD"), 20


def load():
    spec = importlib.util.spec_from_file_location("published_baselines_lead",
                                                  SCRIPTS / "published_baselines_lead.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohort(rng, label, base, per_task):
    rows = []
    for ti, task in enumerate(TASKS):
        for i in range(2 * per_task):
            failed = i % 2 == 1
            length = HORIZON if failed else int(rng.integers(6, HORIZON - 1))
            shift = 0.8 if failed and ti % 2 else 0.0
            rows.append({
                "rollout_id": f"{label}-{task}-{i}", "task_name": task, "failed": failed,
                "horizon": HORIZON, "source_inferences": length,
                "environment_seed": base + 100 * ti + i, "environment_reset_index": 0,
                "safe_scores": np.maximum.accumulate(rng.standard_normal(length) + shift).tolist(),
                "time_scores": (np.arange(1, length + 1) / HORIZON).tolist(),
            })
    return rows


def write_fixture(root, *, truncate=False):
    rng = np.random.default_rng(3)
    cal, test = cohort(rng, "cal", 1000, 4), cohort(rng, "test", 5000, 10)
    primary, _ = _select_per_task_class(cal, 3)
    active = [30] + [30 - min(10, max(0, i - 8)) for i in range(1, HORIZON)]
    bundle = {
        "target_fpr": 0.2, "safe_eligibility_end_fraction": 0.25, "time_eligibility_start_fraction": 0.5,
        "primary_calibration_rollout_ids": [r["rollout_id"] for r in primary],
        "time_risk_curves": {t: {"active_counts": active} for t in TASKS},
    }
    (root / "cal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cal))
    (root / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test))
    (root / "bundle.json").write_text(json.dumps(bundle))
    run = root / "run"
    run.mkdir()
    results = {"detectors": {}}
    names = ("safe_copy", "noise")
    with (run / "calibration_predictions.jsonl").open("w") as fc, \
            (run / "evaluation_predictions.jsonl").open("w") as fe:
        for name in names:
            def record(r):
                score = (r["safe_scores"] if name == "safe_copy"
                         else np.maximum.accumulate(rng.standard_normal(len(r["safe_scores"]))).tolist())
                if truncate and name == "noise":
                    score = score[:-1]
                return {"detector": name, **{k: r[k] for k in ("rollout_id", "task_name", "failed", "horizon",
                                                               "source_inferences", "time_scores",
                                                               "environment_seed", "environment_reset_index")},
                        "safe_scores": score}
            cr, er = [record(r) for r in primary], [record(r) for r in test]
            fc.write("".join(json.dumps(x) + "\n" for x in cr))
            fe.write("".join(json.dumps(x) + "\n" for x in er))
            if truncate:
                continue
            sel = select_single_threshold(cr, "safe_only", target_fpr=0.2)
            balanced, _ = _select_per_task_class(er, 10)
            m = event_metrics(balanced, "safe_only", sel["thresholds"])
            results["detectors"][name] = {"evaluation": {"baseline_only": {
                "true_positive_rate": m["true_positive_rate"], "false_positive_rate": m["false_positive_rate"]}}}
    (run / "results.json").write_text(json.dumps(results))
    return run


def split_run(root):
    """Write run_a (safe_copy) and run_b (noise) from the combined fixture run."""
    results = json.loads((root / "run" / "results.json").read_text())
    for name, target in (("safe_copy", "run_a"), ("noise", "run_b")):
        out = root / target
        out.mkdir()
        for filename in ("calibration_predictions.jsonl", "evaluation_predictions.jsonl"):
            lines = [line for line in (root / "run" / filename).read_text().splitlines()
                     if json.loads(line)["detector"] == name]
            (out / filename).write_text("\n".join(lines) + "\n")
        (out / "results.json").write_text(json.dumps({"detectors": {name: results["detectors"][name]}}))


def argv(root, *extra):
    return ["--baseline-run", str(root / "run"), "--calibration", str(root / "cal.jsonl"),
            "--test", str(root / "test.jsonl"), "--bundle", str(root / "bundle.json"),
            "--out", str(root / "out.json"), "--noise-draws", "2", *extra]


class PublishedBaselinesLeadTests(unittest.TestCase):
    def test_copy_of_safe_reproduces_safe_in_both_protocols(self):
        module = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                code = module.main(argv(root, "--allow-reference-mismatch"))
            self.assertEqual(code, 0)
            out = json.loads((root / "out.json").read_text())
            self.assertEqual(out["detectors"], ["noise", "safe_copy"])
            self.assertEqual(out["counts"], {"calibration": 24, "test": 80})
            for key in ("noise_baseline_only_tpr", "safe_copy_baseline_only_fpr"):
                self.assertTrue(out["checks"][key]["ok"], key)

            lead = out["lead_over_P"]
            copy = lead["all_features_descriptive"]["safe_copy"]
            self.assertEqual(copy["gain_over_timer"], lead["safe_plus_timer"]["gain_over_timer"])
            self.assertEqual(copy["vs_matched_fpr_timer"], lead["safe_plus_timer"]["vs_matched_fpr_timer"])
            self.assertIn("noise_null", lead)

            fused = out["fused_with_global_time"]
            self.assertEqual(fused["detectors"]["fused_safe_copy_time"]["test"],
                             fused["detectors"]["fused_safe_time"]["test"])
            same = fused["paired"]["fused_safe_copy_time_vs_fused_safe_time"]
            self.assertEqual(same["counts"]["earlier"] + same["counts"]["later"], 0)
            self.assertEqual(same["bootstrap"]["tpr_delta"]["ci95"], [0.0, 0.0])
            self.assertIn("fused_noise_time_vs_time_only", fused["paired"])

    def test_runs_merge_and_contrasts_are_paired(self):
        module = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            split_run(root)
            args = argv(root, "--allow-reference-mismatch", "--contrast", "safe_copy", "safe",
                        "--contrast", "noise", "safe_copy")
            args[args.index(str(root / "run"))] = str(root / "run_a")
            args.insert(args.index(str(root / "run_a")) + 1, str(root / "run_b"))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(args), 0)
            out = json.loads((root / "out.json").read_text())
            self.assertEqual(out["detectors"], ["noise", "safe_copy"])
            self.assertEqual(set(out["detector_runs"].values()),
                             {str((root / "run_a").resolve()), str((root / "run_b").resolve())})
            same = out["lead_over_P"]["contrasts"]["safe_copy_minus_safe"]
            self.assertEqual(same["raw"], {"mean": 0.0, "ci95": [0.0, 0.0]})
            self.assertEqual(same["matched"], {"mean": 0.0, "ci95": [0.0, 0.0]})
            lead = out["lead_over_P"]
            noise = lead["contrasts"]["noise_minus_safe_copy"]
            expected = (lead["all_features_descriptive"]["noise"]["gain_over_timer"]["mean"]
                        - lead["all_features_descriptive"]["safe_copy"]["gain_over_timer"]["mean"])
            self.assertAlmostEqual(noise["raw"]["mean"], expected, places=3)
            self.assertIn("fused_noise_time_vs_fused_safe_copy_time", out["fused_with_global_time"]["paired"])

    def test_detector_in_two_runs_is_rejected(self):
        module = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            split_run(root)
            args = argv(root, "--allow-reference-mismatch")
            args.insert(args.index(str(root / "run")) + 1, str(root / "run_a"))
            with self.assertRaises(ValueError):
                module.main(args)

    def test_reference_mismatch_fails_but_keeps_output(self):
        module = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                code = module.main(argv(root))
            self.assertEqual(code, 1)
            out = json.loads((root / "out.json").read_text())
            self.assertTrue(out["reproduction_failures"])

    def test_misaligned_trajectory_is_rejected(self):
        module = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, truncate=True)
            with self.assertRaises(ValueError):
                module.main(argv(root, "--allow-reference-mismatch"))

    def test_existing_output_is_not_overwritten(self):
        module = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            (root / "out.json").write_text("{}")
            with self.assertRaises(SystemExit):
                module.main(argv(root))


if __name__ == "__main__":
    unittest.main()
