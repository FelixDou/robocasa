"""R27e (scripts/xr1_lead_time/r27e_table1_bands.py): Table I's own-band rows on R27's 30/70 assignments."""
import copy
import math
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load
from tests.test_r21_native_bands import HELD, SEEDS, TASKS, rows

r27e = load("r27e_table1_bands")
r26, r27d, r27, r21 = r27e.r26, r27e.r27d, r27e.r27, r27e.r21
SAFE = "safe_mlp"
TIMER = dict(seen="P", unseen="G_abs")


class R27eTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(527)
        self.cal, self.te = rows(rng, "cal"), rows(rng, "te")
        for r in self.cal + self.te:                 # a second head: a time ramp per seed
            for s in SEEDS:
                r[f"z~#{s}@seen"] = r["z~@seen"] + 0.05 * int(s)
                for F in HELD:
                    r[f"z~#{s}@{F}"] = r[f"z~@{F}"] + 0.05 * int(s)
        self.rk = {t: 0.5 for t in TASKS}
        every = set(TASKS)
        self.cal_tasks, self.test_tasks = {F: every - HELD[F] for F in HELD}, dict(HELD)
        at = dict(seen=lambda F: "seen", unseen=lambda F: F)
        self.keys = {h: {side: {f"seed_{s}": (lambda F, p=p, s=s, at=at[side]: f"{p}~#{s}@{at(F)}") for s in SEEDS}
                         for side in at} for h, p in ((r27.FULL, "h"), (SAFE, "z"))}

    def references(self, splits=2):
        """R26's saved rows (R21's evaluate at the fixed split, recounted) and R27d's saved mean curves."""
        ds = r27e.r27c.designs(self.cal, self.te, self.cal_tasks, self.test_tasks)
        rows26 = {}
        for h, by_side in self.keys.items():
            rows26[f"{h}_own_band"] = {}
            for side, by_seed in by_side.items():
                rows26[f"{h}_own_band"][side] = {
                    lab: r26.from_counts(r21.evaluate(ds[side], k, (TIMER[side],), self.rk)[TIMER[side]]["standalone"])
                    for lab, k in by_seed.items()}
        saved = []
        for j in range(splits):
            dj = r27e.r27c.designs(self.cal, self.te, self.cal_tasks, self.test_tasks,
                                   split=lambda rs, j=j: r27.split_30_70(rs, r27.SPLIT_SEED + j))
            saved.append(dict(mean_curves=r27d.all_curves(dj, self.keys)[1]))
        return dict(rows=rows26), dict(splits=saved)

    def go(self, ref26, ref27d, splits=2):
        return r27e.run(self.cal, self.te, self.cal_tasks, self.test_tasks, self.keys, ref26, ref27d, splits=splits)

    def test_standalone_is_r21s(self):
        d = r21.Design(self.cal, self.te, self.cal_tasks, self.test_tasks)
        key = self.keys[r27.FULL]["unseen"]["seed_1"]
        mine = r27e.standalone(d, key)
        theirs = r21.evaluate(d, key, ("G_abs",), self.rk)["G_abs"]["standalone"]
        for k in ("fpr", "fp", "tpr", "mean_lead", "alarms_at_limit", "n_failure", "n_success"):
            self.assertEqual(mine[k], theirs[k], k)

    def test_run_gates_and_summary(self):
        ref26, ref27d = self.references()
        res, problems = self.go(ref26, ref27d)
        self.assertEqual(problems, [])
        self.assertEqual(res["checks"]["E0_fixed_split_rows_equal_r26"]["compared"], 2 * 2 * 3)
        self.assertEqual(res["checks"]["E1_means_equal_r27d_curves"]["compared"], 2 * 2 * 2)
        cell = res["summary"][SAFE]["seen"]["mean_lead"]
        means = [row["rows"][SAFE]["seen"]["mean_std_over_seeds"]["mean_lead"]["mean"] for row in res["splits"]]
        self.assertAlmostEqual(cell["mean"], float(np.median(means)))
        self.assertEqual(set(cell["by_seed"]), set(r26.SEED_LABELS))
        self.assertIn("bacc_before_limit", res["summary"][r27.FULL]["unseen"])
        # Gates fail on perturbed or missing references.
        bad26 = copy.deepcopy(ref26)
        bad26["rows"][f"{SAFE}_own_band"]["unseen"]["seed_2"]["bacc_before_limit"] += 1e-9
        self.assertEqual(self.go(bad26, ref27d)[1], ["E0"])
        del bad26["rows"][f"{SAFE}_own_band"]["seen"]["seed_0"]
        self.assertEqual(self.go(bad26, ref27d)[1], ["E0"])
        bad27d = copy.deepcopy(ref27d)
        point = next(q for q in bad27d["splits"][1]["mean_curves"][r27.FULL]["seen"] if abs(q["alpha"] - 0.05) < 1e-9)
        point["mean_lead"] += 1e-9
        self.assertEqual(self.go(ref26, bad27d)[1], ["E1"])
        self.assertEqual(self.go(ref26, dict(splits=ref27d["splits"][:1]))[1], ["E1"])

    def test_summarize_takes_medians(self):
        def row(v):
            seed = {k: v for k in r26.KEYS}
            summary = {k: dict(mean=v, std=v / 10) for k in r26.KEYS}
            return dict(rows=dict(x=dict(seen={**{lab: seed for lab in r26.SEED_LABELS}, "mean_std_over_seeds": summary})))
        s = r27e.summarize([row(v) for v in (0.1, 0.3, 0.2)])["x"]["seen"]
        self.assertAlmostEqual(s["fpr"]["mean"], 0.2)
        self.assertAlmostEqual(s["fpr"]["std"], 0.02)
        self.assertEqual((s["fpr"]["mean_min"], s["fpr"]["mean_max"]), (0.1, 0.3))
        self.assertEqual(s["splits"], 3)
        self.assertTrue(math.isclose(s["tpr"]["by_seed"]["seed_0"], 0.2))


if __name__ == "__main__":
    unittest.main()
