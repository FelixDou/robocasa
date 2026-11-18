"""R22 stage A gate (scripts/xr1_lead_time/r22_stage_a_check.py) on synthetic fold directories."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load

chk = load("r22_stage_a_check")
SEEDS = ("0", "1", "2")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def fold_dir(d, rng, dets, cohorts, perturb=None):
    d.mkdir(parents=True)
    for cohort, rows in cohorts.items():
        out = []
        for det in dets:
            for r in rows:
                raw = {s: (np.arange(r["n"]) * 0.1 + int(s) + (0.5 if r["failed"] else 0)).tolist() for s in SEEDS}
                if perturb and perturb == (cohort, det, r["rollout_id"]):
                    raw["1"][0] += 1e-9
                out.append(dict(detector=det, rollout_id=r["rollout_id"], task_name=r["task_name"], failed=r["failed"],
                                held_out=r["task_name"] == "TaskA", per_seed_raw=raw))
        write_jsonl(d / f"{cohort}_predictions.jsonl", out)
    (d / "normalizations.json").write_text(json.dumps({det: {s: {"pooled": {"location": 0.1, "scale": 1.0}}
                                                             for s in SEEDS} for det in dets}))


class StageACheckTest(unittest.TestCase):
    def test_gate(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e1 = {c: [dict(rollout_id=f"{c}-{t}-{i}", task_name=t, failed=bool(i % 2), n=6) for t in ("TaskA", "TaskB")
                      for i in range(4)] for c in chk.E1}
            aug = [dict(rollout_id=f"aug-{t}-{i}", task_name=t, failed=bool(i % 2), n=5) for t in ("TaskA", "TaskB")
                   for i in range(6)]
            write_jsonl(root / "labels.jsonl", [dict(rollout_id=r["rollout_id"], task_name=r["task_name"], failed=r["failed"])
                                                for r in aug])
            (root / "ids.json").write_text(json.dumps([r["rollout_id"] for r in aug[:8]]))
            fold_dir(root / "stored", rng, ["hs", "knn"], e1)
            fold_dir(root / "new", rng, ["hs"], {**e1, "august_calibration": aug})
            args = ["--validation", "august_calibration", str(root / "ids.json"), str(root / "labels.jsonl")]
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                status = chk.main(["--pair", "xt/0", str(root / "new"), str(root / "stored"), *args,
                                   "--output", str(root / "out.json")])
            self.assertEqual(status, 0, buf.getvalue())
            self.assertIn("STAGE A OK", buf.getvalue())
            out = json.loads((root / "out.json").read_text())["pairs"]["xt/0"]
            self.assertEqual(out["lead_calibration"]["compared"], 8)          # only the new run's detector
            self.assertEqual(out["august_calibration"]["rollouts"], 8)
            # A changed E1 trajectory, a missing validation rollout or a wrong label fails the gate.
            fold_dir(root / "new2", rng, ["hs"], {**e1, "august_calibration": aug},
                     perturb=("confirmatory_test", "hs", e1["confirmatory_test"][0]["rollout_id"]))
            fold_dir(root / "new3", rng, ["hs"], {**e1, "august_calibration": aug[:5]})
            for bad in ("new2", "new3"):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(chk.main(["--pair", bad, str(root / bad), str(root / "stored"), *args,
                                               "--output", str(root / f"{bad}.json")]), 1)
            labels = [dict(rollout_id=r["rollout_id"], task_name=r["task_name"], failed=not r["failed"]) for r in aug]
            write_jsonl(root / "labels_bad.jsonl", labels)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(chk.main(["--pair", "x", str(root / "new"), str(root / "stored"), "--validation",
                                           "august_calibration", str(root / "ids.json"), str(root / "labels_bad.jsonl"),
                                           "--output", str(root / "o.json")]), 1)


if __name__ == "__main__":
    unittest.main()
