"""R22 (scripts/xr1_lead_time/r22_fixed_paths.py) on synthetic folds: fixed, adaptive and frozen paths."""
import math
import unittest

import numpy as np

from tests.test_rldx_hide_and_seek import load

r22, hk, cl, rx = (load(n) for n in ("r22_fixed_paths", "hk_free", "confirm_lead", "r19_extras"))

TASKS = ("TaskA", "TaskB", "TaskC", "TaskD")
HELD = {"0": {"TaskA"}, "1": {"TaskB"}, "2": {"TaskC"}, "3": {"TaskD"}}
H = {"TaskA": 30, "TaskB": 40, "TaskC": 50, "TaskD": 60}
SEEDS = ("0", "1", "2")
NORMS = {s: {t: dict(location=0.05 * int(s), scale=1.0 + 0.1 * i) for i, t in enumerate(TASKS)} for s in SEEDS}


def rows(rng, name, n_s=10, n_f=10):
    out = []
    for t in TASKS:
        for i in range(n_s + n_f):
            failed = i >= n_s
            L = H[t] if failed else int(rng.integers(8, H[t] - 3))
            base = np.cumsum(rng.normal(0.06 if failed else 0.0, 0.3, L))
            per_raw = {s: (base + rng.normal(0, 0.2, L)).tolist() for s in SEEDS}
            r = dict(rollout_id=f"{name}-{t}-{i:02d}", task_name=t, failed=failed, L=L, horizon=H[t],
                     per_seed_raw_safe=per_raw, raw={"hs": {s: np.asarray(v) for s, v in per_raw.items()}})
            per = {s: (np.maximum.accumulate(np.asarray(per_raw[s])) - NORMS[s][t]["location"]) / NORMS[s][t]["scale"]
                   for s in SEEDS}
            r["safe"] = np.mean([per[s] for s in SEEDS], axis=0)
            for F in list(HELD) + ["all"]:
                r[f"h@{F}"] = r["safe"] + 0.01 * (0 if F == "all" else int(F))
            out.append(r)
    return out


class R22Test(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(22)
        self.cal, self.te, self.val = rows(rng, "cal"), rows(rng, "te"), rows(rng, "val")
        self.rk = {t: 0.5 for t in TASKS}

    def seen(self, rows_, key="h@all"):
        return cl.Cohort(rows_, key)

    def unseen_pairs(self):
        return {F: (cl.Cohort([r for r in self.cal if r["task_name"] not in HELD[F]], f"h@{F}"),
                    cl.Cohort([r for r in self.te if r["task_name"] in HELD[F]], f"h@{F}")) for F in HELD}

    def test_adaptive_is_hk_free_and_fixed_on_calibration_is_adaptive(self):
        pairs = self.unseen_pairs()
        for t in ("G_abs", "G"):
            mine = r22.run_folds(pairs, t, self.rk)
            ref = hk.run_folds(pairs, t, None, self.rk)
            self.assertEqual(r22.compare(mine, ref), 0.0)
            # Choosing on the calibration cohort itself and certifying there is the adaptive path.
            same = r22.run_folds(pairs, t, self.rk, val={F: c for F, (c, _) in pairs.items()})
            self.assertEqual(r22.compare(same, ref), 0.0)
        k = r22.k0({"G": {"seed_average": mine}}, {"G": {"seed_average": ref}}, ("G",), ["seed_average"])
        self.assertTrue(k["ok"])
        bad = dict(ref, gain=ref["gain"] + 1e-9)
        self.assertFalse(r22.k0({"G": {"seed_average": mine}}, {"G": {"seed_average": bad}}, ("G",), ["seed_average"])["ok"])

    def test_fixed_paths_are_chosen_on_validation_and_certified_on_calibration(self):
        pairs = {"all": (self.seen(self.cal), self.seen(self.te))}
        fixed = r22.run_folds(pairs, "P", self.rk, val={"all": self.seen(self.val)}, keep=True)
        vu, rv = hk.prepared(self.seen(self.val), "P", "all", None, self.rk)
        _, chosen = hk.fit(vu, rv)
        self.assertEqual(fixed["_chosen"]["all"], dict(u=chosen["u"], s=chosen["s"]))
        cu, rc = hk.prepared(self.seen(self.cal), "P", "all", None, self.rk)
        tm, fu = r22.certify(cu, rc, chosen)
        self.assertEqual(fixed["paths"]["all"]["fused"], cl.det_json(fu))
        # The timer's margin is the adaptive run's: only the path's choice changes.
        adaptive = r22.run_folds(pairs, "P", self.rk)
        self.assertEqual(fixed["paths"]["all"]["timer"], adaptive["paths"]["all"]["timer"])
        # A frozen path reproduces the fixed run.
        frozen = r22.run_folds(pairs, "P", self.rk, frozen=fixed["_chosen"])
        self.assertEqual(r22.compare(frozen, fixed), 0.0)
        self.assertEqual(len(fixed["_arrays"]["fused"]), len(fixed["_arrays"]["task"]))
        self.assertNotIn("_arrays", r22.strip(fixed))
        self.assertNotIn("_chosen", r22.strip(fixed))

    def test_ensemble_row_per_type_bootstrap_nulls_budget_teaser(self):
        pairs = {"all": (self.seen(self.cal), self.seen(self.te))}
        Fte = [r for r in self.te if r["failed"]]
        row, run_ = r22.ensemble_row(pairs, {"all": self.seen(self.val)}, "P", self.rk, Fte)
        self.assertEqual(len(row["gain_ci95"]), 2)
        self.assertIn("ci95", row["matched"])
        types = {t: ("atomic" if t in ("TaskA", "TaskB") else "composite") for t in TASKS}
        pt = r22.per_type(run_, types, 0.8)
        self.assertIn("atomic", pt["seconds"])
        design = rx.UnseenDesign(self.cal, self.te, HELD, sorted(HELD))
        val_of = lambda n: {F: cl.Cohort([r for r in self.val if r["task_name"] not in HELD[F]], f"h@{F}") for F in HELD}
        boot, point = r22.boot_unseen(design, {"h": lambda F: f"h@{F}", "h2": lambda F: f"h@{F}"}, val_of,
                                      ("G_abs",), self.rk, B=3, contrasts=(("h", "h2"),))
        self.assertEqual(boot["samples"], 3)
        self.assertEqual(len(boot["timers"]["G_abs"]["h"]["matched_ci95"]), 2)
        self.assertIn("h_minus_h2", boot["timers"]["G_abs"])
        hs_norms = {"hs": NORMS}
        out = r22.nulls(self.cal, self.te, self.val, self.rk, NORMS, hs_norms, "hs",
                        {"safe": (row["matched"]["gain"], "safe_resampled")}, B=3)
        self.assertEqual(set(out["nulls"]), {"noise", "safe_resampled", "hs_resampled"})
        self.assertIn("rank_p", out["safe"]["noise"])
        dev = {t: list(np.linspace(0.2, 0.6, 12)) for t in TASKS}
        b = r22.budget_timer(self.seen(self.val), self.seen(self.cal), self.seen(self.te), dev, self.rk, run_)
        self.assertIsNotNone(b)
        self.assertIn("fused_over_it_matched", b)
        r = self.te[-1]
        chosen = run_["_chosen"]["all"]
        tz = r22.teaser(r, r["h@all"], self.rk[r["task_name"]], r["horizon"], chosen, 0.2, 0.1)
        self.assertEqual(tz["P_alarm"], math.ceil(min(r["horizon"], 0.6 * r["horizon"]) - hk.EPS))

    def test_boot_levels_and_frozen_row(self):
        """R23: boot_levels' 95% interval is qt.boot's, and frozen_row on the validation-chosen path equals
        ensemble_row's point estimates and intervals."""
        pairs = {"all": (self.seen(self.cal), self.seen(self.te))}
        Fte = [r for r in self.te if r["failed"]]
        row, run_ = r22.ensemble_row(pairs, {"all": self.seen(self.val)}, "P", self.rk, Fte)
        frow, _ = r22.frozen_row(pairs, run_["_chosen"], "P", self.rk, Fte)
        for k in ("fp", "gain", "mean_lead"):
            self.assertEqual(frow[k], row[k])
        self.assertEqual(frow["matched"]["gain"], row["matched"]["gain"])
        self.assertEqual(frow["gain_ci"]["ci95"], row["gain_ci95"])
        self.assertEqual(frow["matched"]["ci"]["ci95"], row["matched"]["ci95"])
        lo95, hi95 = frow["matched"]["ci"]["ci95"]
        lo, hi = frow["matched"]["ci"]["ci97.5"]
        self.assertLessEqual(lo, lo95)
        self.assertGreaterEqual(hi, hi95)
        with self.assertRaises(ValueError):
            r22.frozen_row(pairs, run_["_chosen"], "P", self.rk, Fte[:-1])

    def test_k8(self):
        pairs = {"all": (self.seen(self.cal), self.seen(self.te))}
        row, _ = r22.ensemble_row(pairs, {"all": self.seen(self.val)}, "P", self.rk, [r for r in self.te if r["failed"]])
        ch = row["chosen"]["all"]
        r8 = {"rows": {"safe": dict(path=dict(u=None if ch["u"] is None else round(ch["u"], 4), s=ch["s"]),
                                    gain=round(row["gain"], 4))}}
        self.assertTrue(r22.k8({"safe_mlp": row}, r8, {"safe": "safe_mlp"})["ok"])
        r8["rows"]["safe"]["gain"] += 0.01
        self.assertFalse(r22.k8({"safe_mlp": row}, r8, {"safe": "safe_mlp"})["ok"])


if __name__ == "__main__":
    unittest.main()
