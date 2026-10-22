import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "xr1_lead_time"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TASKS, H = ["TaskA", "TaskB", "TaskC", "TaskD"], 30


def held_rows(seed=0, hs_shift=1.5, copy=False):
    """Held-out pool: 8 successes and 5 failures per task; failures score higher late."""
    rng = np.random.default_rng(seed)
    rows = []
    for ti, task in enumerate(TASKS):
        for i in range(13):
            failed = i >= 8
            L = H if failed else int(rng.integers(8, H - 3))
            ramp = np.linspace(0, 1, L)
            safe = np.maximum.accumulate(rng.standard_normal(L) + (0.8 * ramp if failed else 0))
            hs = safe.copy() if copy else np.maximum.accumulate(rng.standard_normal(L) + (hs_shift * ramp if failed else 0))
            rows.append(dict(rollout_id=f"{task}-{i}", task_name=task, failed=failed, L=L, horizon=H,
                             safe=safe, hide_and_seek=hs))
    return rows


RK = {t: 0.7 for t in TASKS}


class E2CoprimarySizingTests(unittest.TestCase):
    def setUp(self):
        self.m = load("e2_coprimary_sizing")
        self.s2 = sys.modules["stage2_sizing"]

    def test_binomial_rule_matches_paper_and_brute_force(self):
        self.assertEqual(self.m.binom_max_false_alarms(400), 39)
        for n in (40, 100, 400):
            pmf = [math.comb(n, k) * 0.075 ** k * 0.925 ** (n - k) for k in range(n + 1)]
            tails = [sum(pmf[k:]) for k in range(n + 1)]
            self.assertEqual(self.m.binom_max_false_alarms(n), max(k for k in range(n + 1) if tails[k] >= 0.05))

    def test_safe_half_reproduces_stage2_sizing_exactly(self):
        rows = held_rows()
        pools = self.m.build_pools(rows[:20], rows[20:], TASKS)
        sims = self.m.simulate_pair(pools, RK, (3, 5), 30, 1, chunk=7)
        ref = self.s2.simulate(pools["safe"], pools["safe"], RK, (), (3, 5), 30, 1, (), 0, 0)
        for n in (3, 5):
            for mode in ("within_task", "with_tasks"):
                mine = self.m.summary(sims[(n, mode)], "safe")
                theirs = ref["plug_in"][str(n)][mode]
                self.assertEqual((mine["mean_gain"], mine["sd"], mine["mean_fused_fpr"], mine["fpr_pass_a1"]),
                                 (theirs["mean_gain"], theirs["sd"], theirs["mean_fused_fpr"], theirs["fpr_criteria_pass"]),
                                 (n, mode))

    def test_identical_scores_give_identical_power(self):
        rows = held_rows(copy=True)
        pools = self.m.build_pools(rows[:20], rows[20:], TASKS)
        sims = self.m.simulate_pair(pools, RK, (4,), 20, 1)
        for r in sims[(4, "within_task")]:
            self.assertEqual(r["safe"], r["hide_and_seek"])
        tables = self.m.power_tables(sims, (4,))
        a = tables["power"]["safe"]["plug_in"]["0.975"]["4"]
        b = tables["power"]["hide_and_seek"]["plug_in"]["0.975"]["4"]
        self.assertEqual(a, b)
        j = tables["joint"]["safe 0.034 / hs 0.052"]["0.975"]["4"]
        self.assertLessEqual(j["both"], min(j["safe"], j["hide_and_seek"]))
        self.assertGreaterEqual(j["at_least_one"], max(j["safe"], j["hide_and_seek"]))

    def test_wider_level_never_raises_power(self):
        rows = held_rows(seed=3)
        pools = self.m.build_pools(rows[:20], rows[20:], TASKS)
        tables = self.m.power_tables(self.m.simulate_pair(pools, RK, (4,), 40, 1), (4,))
        for key in ("safe", "hide_and_seek"):
            for g, by_l in tables["power"][key].items():
                p95, p975 = by_l["0.95"]["4"], by_l["0.975"]["4"]
                self.assertLessEqual(p975["ci"], p95["ci"])
                self.assertGreaterEqual(p975["hw_within"], p95["hw_within"])

    def test_pools_share_draw_indices(self):
        rows = held_rows()
        pools = self.m.build_pools(rows[:20], rows[20:], TASKS)
        self.assertEqual(set(pools), {"safe", "hide_and_seek"})
        for t in TASKS:
            for a, b in zip(pools["safe"].idx[t], pools["hide_and_seek"].idx[t]):
                np.testing.assert_array_equal(a, b)
        i = int(pools["safe"].idx["TaskA"][1][0])
        np.testing.assert_array_equal(pools["hide_and_seek"].C.R[i][:H], np.maximum.accumulate(rows[i]["hide_and_seek"]))

if __name__ == "__main__":
    unittest.main()
