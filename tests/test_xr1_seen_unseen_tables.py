"""seen_unseen_tables.py on synthetic cohorts and fold directories (R6 of the review-analyses plan)."""
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


sx = load("ct_safe_fixture", TESTS / "test_xr1_cross_task_safe.py")
fx = sx.fx
su = load("seen_unseen_tables", SCRIPTS / "seen_unseen_tables.py")
import confirm_lead as cl  # noqa: E402
import ood_per_seed_fused as opf  # noqa: E402
import per_seed_detection as psd  # noqa: E402
import quantile_timer as qt  # noqa: E402

SEEDS = fx.SEEDS
COHORTS = sx.COHORTS
OOD = su.OOD
FAMILY_STRENGTH = {"actprobe": 0.6, "mahalanobis": 0.9, "knn_cosine": 0.7, "pca_kmeans": 0.5, "rnd": 0.4}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw(rng, r, strength):
    n = r["source_inferences"]
    ramp = strength * np.linspace(0, 1, n) if r["failed"] else 0.0
    return {s: (rng.standard_normal(n) + ramp).tolist() for s in SEEDS}


def write_family(root, label, F, held, rng, detectors, extra_meta=None):
    """A fold directory in the cross-task format for synthetic detectors."""
    d = root / label / f"fold_{F}"
    d.mkdir(parents=True)
    norms = {det: {s: {"pooled": {"location": 0.1 * int(s), "scale": 1.2},
                       "per_task": {t: {"location": 0.05 * (i % 3), "scale": 1.0 + 0.1 * (i % 2)}
                                    for i, t in enumerate(fx.TASKS) if t not in held}}
                   for s in SEEDS} for det in detectors}
    for cohort, name in COHORTS:
        with (d / f"{cohort}_predictions.jsonl").open("w") as stream:
            for det in detectors:
                for r in sx.jsonl(root / name):
                    stream.write(json.dumps({"detector": det, "rollout_id": r["rollout_id"], "task_name": r["task_name"],
                                             "failed": r["failed"], "fold": F, "held_out": r["task_name"] in held,
                                             "per_seed_raw": raw(rng, r, FAMILY_STRENGTH[det])}) + "\n")
    meta = {"fold": F, "held_out_tasks": sorted(held), "training_rollouts_by_task": {t: 30 for t in fx.TASKS if t not in held},
            "folds_sha256": sha(root / "folds.json"), "seeds": [0, 1, 2], "detectors": list(detectors), "code_commit": "test"}
    meta.update(extra_meta or {})
    (d / "scoring.json").write_text(json.dumps(meta))
    (d / "normalizations.json").write_text(json.dumps(norms))
    return d, norms


def per_task_traj(rows_raw, norms):
    return {s: (np.maximum.accumulate(np.asarray(v, float)) - norms[s]["location"]) / norms[s]["scale"]
            for s, v in rows_raw.items()}


def build(root, ood_source_ok=True, actprobe_reference_ok=True, perturb_run_c=False):
    spec, held, hs_dirs, safe_dirs, _, _ = sx.pipeline(root)
    with contextlib.redirect_stdout(io.StringIO()):
        assert sx.cts.main(sx.analysis_args(root, hs_dirs, safe_dirs)) == 0
    bundle = json.loads((root / "bundle.json").read_text())
    rng = np.random.default_rng(11)
    _, cal = cl.primary(root / "cal.jsonl", "calibration")
    _, te = cl.primary(root / "test.jsonl", "test")
    rows = cal + te

    # ActProbe: folds 0-3 and fold all, with fold all's B1 reference and normalization.
    act_dirs = {}
    for F in fx.FOLDS:
        extra = None
        if F == "all":
            extra = {"reference_checks": {f"august_{c}/actprobe": {"ok": actprobe_reference_ok, "max_abs_diff": 0.0}
                                          for c in ("calibration", "test")}}
        act_dirs[F], norms = write_family(root, "actprobe", F, held[F], rng, ["actprobe"], extra)
        if F == "all":
            b1 = {"actprobe": {s: norms["actprobe"][s]["per_task"] for s in SEEDS}}
    (root / "b1_norms.json").write_text(json.dumps(b1))

    # OOD pooled refits per fold, from a frozen source bundle.
    (root / "ood_source.json").write_text(json.dumps({"selected": {}}))
    source_sha = sha(root / "ood_source.json") if ood_source_ok else "0" * 64
    ood_dirs = {F: write_family(root, "ood", F, held[F], rng, list(OOD),
                                {"variant": "pooled", "fixed_selection": {"bundle_sha256": source_sha}})[0]
                for F in ("0", "1", "2", "3")}

    # Amendment 2 scores (seen Hide-and-Seek family and the shuffle control), per-task normalized.
    xt_all = {}
    for cohort, _ in COHORTS:
        for line in (hs_dirs["all"] / f"{cohort}_predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            xt_all[(row["detector"], row["rollout_id"])] = row["per_seed_raw"]
    amend2_norms = {det: {s: {t: {"location": 0.02 * i, "scale": 1.0 + 0.05 * i} for i, t in enumerate(fx.TASKS)}
                          for s in SEEDS} for det in (*su.HS_FAMILY, su.SHUFFLE)}
    for cohort, name in COHORTS:
        out = []
        for r in sx.jsonl(root / name):
            for det in (*su.HS_FAMILY, su.SHUFFLE):
                per_raw = xt_all[(det, r["rollout_id"])] if det != su.SHUFFLE else raw(rng, r, 0.2)
                per = per_task_traj(per_raw, {s: amend2_norms[det][s][r["task_name"]] for s in SEEDS})
                out.append({"detector": det, "rollout_id": r["rollout_id"], "task_name": r["task_name"],
                            "failed": r["failed"], "per_seed_raw": per_raw,
                            "safe_scores": np.mean([per[s] for s in SEEDS], axis=0).tolist()})
        sx.write_jsonl(root / f"amend2_{cohort}.jsonl", out)
    (root / "amend2_norms_seen.json").write_text(json.dumps(amend2_norms))

    # OOD seen per seed: E1 rows plus baseline_scores (already normalized, monotone).
    ood_seen = []
    for s in SEEDS:
        paths = []
        for cohort, name in COHORTS:
            out = []
            for r in sx.jsonl(root / name):
                r = dict(r)
                r["baseline_scores"] = {
                    det: np.maximum.accumulate(np.asarray(raw(rng, r, FAMILY_STRENGTH[det])[s])).tolist() for det in OOD}
                out.append(r)
            path = root / f"ood_seen_{s}_{cohort}.jsonl"
            sx.write_jsonl(path, out)
            paths.append(path)
        ood_seen.append((s, *paths))

    # Stored per-seed runs that T2 must reproduce.
    rk = qt.rho(qt.dev_fractions(bundle), cl.PRIMARY_Q)
    need = {str(r["rollout_id"]) for r in rows}
    seen = {su.SAFE: su.seen_safe(rows, bundle)}
    for det in (*su.HS_FAMILY, su.SHUFFLE):
        seen[det] = su.seen_amend2([root / f"amend2_{c}.jsonl" for c, _ in COHORTS], root / "amend2_norms_seen.json",
                                   det, need)
    run_c = {"detectors": {det: psd.run_fused(seen[det], seen[det], cal, te, bundle) for det in seen}}
    if perturb_run_c:
        run_c["detectors"][su.HS]["seed_1"]["gain"] += 1e-6
    (root / "run_c.json").write_text(json.dumps(run_c, default=float))
    per_seed = {s: opf.seed_metrics(c, t, rk) for s, c, t in ood_seen}
    (root / "ood_fused.json").write_text(json.dumps({"per_seed": per_seed}, default=float))

    args = ["run", "--calibration", str(root / "cal.jsonl"), "--test", str(root / "test.jsonl"),
            "--runtime-bundle", str(root / "bundle.json"), "--folds", str(root / "folds.json"),
            "--safe-fold-dirs", *[str(safe_dirs[F]) for F in ("0", "1", "2", "3")],
            "--xt-fold-dirs", *[str(hs_dirs[F]) for F in ("0", "1", "2", "3")],
            "--actprobe-fold-dirs", *[str(act_dirs[F]) for F in fx.FOLDS],
            "--ood-fold-dirs", *[str(ood_dirs[F]) for F in ("0", "1", "2", "3")],
            "--ood-source-bundle", str(root / "ood_source.json"),
            "--amend2-scores", str(root / "amend2_lead_calibration.jsonl"), str(root / "amend2_confirmatory_test.jsonl"),
            "--amend2-normalizations", str(root / "amend2_norms_seen.json"),
            "--run-c", str(root / "run_c.json"), "--ood-fused", str(root / "ood_fused.json"),
            "--cross-task-result", str(root / "xt.json"), "--cross-task-safe-result", str(root / "safe_out.json"),
            "--actprobe-reference-normalizations", str(root / "b1_norms.json"),
            "--output", str(root / "tables.json")]
    for s, c, t in ood_seen:
        args += ["--ood-seen", s, str(c), str(t)]
    return args, held, cal, te


class HelperTests(unittest.TestCase):
    def test_pooled_outcomes_equal_the_per_seed_detection_metrics(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(40):
            failed = i % 2 == 1
            L = 20 if failed else int(rng.integers(5, 19))
            rows.append(dict(score=np.maximum.accumulate(rng.standard_normal(L) + (0.05 * np.arange(L) if failed else 0)),
                             L=L, horizon=20, failed=failed))
        tau = psd.threshold(rows)
        want = psd.evaluate(rows, tau)
        got = su.standalone_metrics(su.pool_outcomes([su.outcomes(rows[:15], tau), su.outcomes(rows[15:], tau)]))
        for k in ("tpr", "fpr", "bacc", "twa", "dt", "mean_lead"):
            self.assertAlmostEqual(got[k], want[k], places=12)


class RunTests(unittest.TestCase):
    def test_tables_report_every_row_and_reproduce_the_published_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, held, cal, te = build(root)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(su.main(args), 0)
            self.assertIn("SEEN/UNSEEN TABLES OK", buf.getvalue())
            out = json.loads((root / "tables.json").read_text())
            self.assertEqual(out["reproduction_failures"], [])
            self.assertTrue(all(v["ok"] for v in out["checks"]["T3_unseen_seed_average"].values()))
            self.assertTrue(all(v <= 1e-12 for v in out["checks"]["T2_seen_fused_max_abs_diff"].values()))
            for name in su.TABLE1:
                self.assertIn(name, out["table1"])
                self.assertIsNotNone(out["table1"][name]["seen"])
            for name in ("time_risk", "per_task_timer"):
                self.assertIsNone(out["table1"][name]["unseen"])
            for name in (*OOD, su.SAFE, su.ACTPROBE, su.HS):
                u = out["table1"][name]["unseen"]
                # Every test task is held out exactly once: 8 tasks x 10 successes and 10 failures.
                self.assertEqual(u["seed_average"]["n_success"], 10 * len(fx.TASKS))
                self.assertEqual(u["seed_average"]["n_failure"], 10 * len(fx.TASKS))
                self.assertIn("std", u["mean_std_over_seeds"]["bacc"])
            self.assertIsNone(out["table2"][su.SHUFFLE]["unseen"])
            # Seen P standalone is E1's P alone.
            e1 = json.loads((root / "e1.json").read_text())["by_q"][str(cl.PRIMARY_Q)]["P"]
            p = out["table1"]["per_task_timer"]["seen"]["seed_average"]
            self.assertEqual(round(p["fpr"], 4), e1["fpr"])
            self.assertEqual(round(p["mean_lead"], 4), e1["mean_lead"])
            # Unseen standalone pools per-fold thresholds: recompute SAFE seed 0 by hand.
            folds = sorted(F for F in held if F != "all")
            safe_dirs = {F: root / "safe" / f"fold_{F}" for F in folds}
            fp = n = 0
            hits = []
            for F in folds:
                norms = json.loads((safe_dirs[F] / "normalizations.json").read_text())["safe"]
                traj = {}
                for cohort, _ in COHORTS:
                    for line in (safe_dirs[F] / f"{cohort}_predictions.jsonl").read_text().splitlines():
                        row = json.loads(line)
                        st = norms["0"]["pooled"]
                        traj[row["rollout_id"]] = {"0": (np.maximum.accumulate(np.asarray(row["per_seed_raw"]["0"]))
                                                         - st["location"]) / st["scale"]}
                c = [r for r in cal if r["task_name"] not in held[F]]
                e = [r for r in te if r["task_name"] in held[F]]
                tau = psd.threshold(su.scored(c, traj, "0"))
                o = su.outcomes(su.scored(e, traj, "0"), tau)
                fp, n, hits = fp + o["fp"], n + o["n"], hits + o["hits"]
            got = out["table1"][su.SAFE]["unseen"]["seed_0"]
            self.assertAlmostEqual(got["fpr"], fp / n, places=12)
            self.assertAlmostEqual(got["tpr"], float(np.mean(hits)), places=12)

    def test_r11_rows_and_unchanged_r6_rows(self):
        """R11: ActProbe joins Table II; SAFE-LSTM and ActProbe-T rows come from optional fold dirs; T6
        requires every row of the earlier output to be reproduced exactly."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, held, cal, te = build(root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(su.main(args), 0)
            first = json.loads((root / "tables.json").read_text())
            self.assertIsNotNone(first["table2"][su.ACTPROBE]["seen"])
            self.assertIsNotNone(first["table2"][su.ACTPROBE]["unseen"])
            rng = np.random.default_rng(21)
            FAMILY_STRENGTH.update({su.SAFE_LSTM: 0.8, su.ACTPROBE_T: 0.9})
            dirs = {name: [str(write_family(root, name, F, held[F], rng, [name])[0]) for F in fx.FOLDS]
                    for name in (su.SAFE_LSTM, su.ACTPROBE_T)}
            # The R6 output had no ActProbe Table II row; T6 checks the rows it has.
            r6 = json.loads(json.dumps(first))
            r6["table2"].pop(su.ACTPROBE)
            (root / "r6.json").write_text(json.dumps(r6))
            out_arg = args.index("--output") + 1
            argv = list(args)
            argv[out_arg] = str(root / "tables_r11.json")
            argv += ["--safe-lstm-fold-dirs", *dirs[su.SAFE_LSTM], "--actprobe-t-fold-dirs", *dirs[su.ACTPROBE_T],
                     "--r6-result", str(root / "r6.json")]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(su.main(argv), 0, buf.getvalue())
            out = json.loads((root / "tables_r11.json").read_text())
            t6 = out["checks"]["T6_r6_rows_unchanged"]
            self.assertTrue(t6["ok"])
            self.assertGreater(t6["values"], 100)
            for name in (su.SAFE_LSTM, su.ACTPROBE_T):
                for table in ("table1", "table2"):
                    self.assertIsNotNone(out[table][name]["seen"])
                    self.assertIsNotNone(out[table][name]["unseen"])
                self.assertEqual(out["table1"][name]["unseen"]["seed_average"]["n_success"], 10 * len(fx.TASKS))
            # A changed R6 value blocks the tables.
            bad = json.loads(json.dumps(r6))
            bad["table1"][su.SAFE]["seen"]["seed_0"]["bacc"] += 1e-6
            (root / "r6_bad.json").write_text(json.dumps(bad))
            argv[out_arg] = str(root / "tables_bad.json")
            argv[-1] = str(root / "r6_bad.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(su.main(argv), 1)
            self.assertIn("T6", buf.getvalue())

    def test_prerequisite_failures_block_the_tables(self):
        cases = (dict(perturb_run_c=True), dict(actprobe_reference_ok=False))
        for case in cases:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, _, _, _ = build(root, **case)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    self.assertEqual(su.main(args), 1, case)
                self.assertIn("REPRODUCTION FAILED", buf.getvalue())
                out = json.loads((root / "tables.json").read_text())
                self.assertNotIn("table1", out)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, _, _ = build(root, ood_source_ok=False)
            with self.assertRaisesRegex(ValueError, "T5"):
                su.main(args)


if __name__ == "__main__":
    unittest.main()
