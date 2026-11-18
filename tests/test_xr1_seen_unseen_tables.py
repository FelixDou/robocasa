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


    def test_r18_vectorized_threshold_equals_per_seed_detection(self):
        """R18: lead_fpr_curves.calibration_table/threshold_at are psd.evaluate/psd.threshold at any alpha."""
        import lead_fpr_curves as lfc
        rng = np.random.default_rng(3)
        rows = []
        for i in range(60):
            failed = i % 3 == 0
            L = 25 if failed else int(rng.integers(5, 24))
            rows.append(dict(score=rng.standard_normal(L) + (0.04 * np.arange(L) if failed else 0), L=L, horizon=25,
                             failed=failed, task_name="T"))
        taus, fpr, lead = lfc.calibration_table(rows)
        for j in range(0, len(taus), 7):
            e = psd.evaluate(rows, taus[j])
            self.assertEqual((fpr[j], lead[j]), (e["fpr"], e["mean_lead"]))
        self.assertEqual(lfc.threshold_at((taus, fpr, lead), 0.05), psd.threshold(rows))
        keep = psd.ALPHA
        try:
            for alpha in (0.0, 0.02, 0.1, 0.3):
                psd.ALPHA = alpha
                self.assertEqual(lfc.threshold_at((taus, fpr, lead), alpha), psd.threshold(rows))
        finally:
            psd.ALPHA = keep
        # Timers are scores: E alarms at one inference count, P at rho_k + margin of H_k.
        e_rows = lfc.timer_rows(rows, "E")
        self.assertEqual(psd.first(e_rows[0]["score"], 7.0), 7)
        p_rows = lfc.timer_rows(rows, "P", {"T": 0.4})
        # Thresholds are score values, so the boundary is computed as the score is (15 / 25 - 0.4).
        self.assertEqual(psd.first(p_rows[0]["score"], 15 / 25 - 0.4), 15)
        self.assertEqual(psd.first(p_rows[0]["score"], np.nextafter(15 / 25 - 0.4, 1.0)), 16)


class RunTests(unittest.TestCase):
    def test_r18_curves_reproduce_table_one(self):
        """R18: --curves adds calibrated lead-FPR curves; at alpha = 0.05 they equal Table I (C1), and the
        tables do not change."""
        import lead_fpr_curves as lfc
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, held, cal, te = build(root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(su.main(args), 0)
            base = json.loads((root / "tables.json").read_text())
            (root / "tables.json").unlink()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = su.main(args + ["--curves"])
            self.assertEqual(status, 0, buf.getvalue())
            self.assertIn("R18 CURVES OK", buf.getvalue())
            out = json.loads((root / "tables.json").read_text())
            self.assertEqual(out["table1"], base["table1"])
            self.assertEqual(out["table2"], base["table2"])
            r = out["r18"]
            self.assertEqual(out["r18_failures"], [])
            c1 = r["checks"]["C1_alpha_005_equals_table1"]
            self.assertTrue(set(OOD) | {su.SAFE, su.HS, "global_timer"} <= set(c1))
            self.assertTrue(all(v <= 1e-12 for v in c1.values()))
            for side in ("seen", "unseen"):
                for name in (*OOD, su.SAFE, *su.HS_FAMILY):
                    by = r[side][name]
                    self.assertEqual(len(by["seed_average"]), len(lfc.ALPHAS))
                    self.assertEqual(len(by["mean_std_over_seeds"]), len(lfc.ALPHAS))
                    n = by["seed_average"][0]
                    self.assertEqual((n["n_success"], n["n_failure"]), (10 * len(fx.TASKS), 10 * len(fx.TASKS)))
            self.assertEqual(set(r["timers"]["seen"]), {"G", "E", "P"})
            self.assertEqual(set(r["timers"]["unseen"]), {"G", "E"})
            # A larger target never lowers the calibration FPR budget: the timer's test lead does not
            # decrease from alpha = 0 to alpha = 0.3.
            e = r["timers"]["unseen"]["E"]
            self.assertLessEqual(e[0]["mean_lead"], e[-1]["mean_lead"])

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

    def test_r14_timers_without_hk(self):
        """R14: --r14-durations adds every timer, reproduces unseen + G (K1) and seen + P (K5), and
        leaves the published tables unchanged."""
        import hashlib
        import hk_free as hk
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, held, cal, te = build(root)
            tasks = sorted({r["task_name"] for r in cal})
            ins = {t: f"Do the {t} job." for t in tasks}
            med = {t: 10.0 + 3 * i for i, t in enumerate(tasks)}
            prompts = hk.build_prompts(ins, held, med)
            prompts["example_median_seconds"] = med
            answer = {f"Do the {t} job.\"\n\nHow many": str(8 + 5 * i) for i, t in enumerate(tasks)}
            zero, few, raw, fails = hk.answer_all(prompts, lambda m: next((v for k, v in answer.items()
                                                                         if k in m[1]["content"]), "?"))
            folds_path = args[args.index("--folds") + 1]
            dur = dict(prompt_sha256=hk.prompts_sha256(),
                       folds_sha256=hashlib.sha256(Path(folds_path).read_bytes()).hexdigest(),
                       zero=zero, few=few, answers=len(raw), parse_failures=fails)
            (root / "dur.json").write_text(json.dumps(dur))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(su.main(args), 0)
            base = json.loads((root / "tables.json").read_text())
            (root / "tables.json").unlink()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = su.main(args + ["--r14-durations", str(root / "dur.json")])
            self.assertEqual(status, 0, buf.getvalue())
            self.assertIn("R14 OK", buf.getvalue())
            out = json.loads((root / "tables.json").read_text())
            self.assertEqual(out["table2"], base["table2"])
            self.assertEqual(out["table1"], base["table1"])
            r = out["r14"]
            self.assertTrue(all(v <= 1e-12 for v in r["checks"]["K1_G_reproduces_unseen_plus_G"].values()))
            for name, by in r["unseen"].items():
                self.assertEqual(set(by), set(hk.TIMERS))
                self.assertEqual(by["G_abs"]["seed_average"]["n_success"], 10 * len(fx.TASKS))
            for name in (su.SAFE, su.HS):
                self.assertEqual(set(r["seen"][name]), set(hk.SEEN_TIMERS))
                self.assertLessEqual(r["checks"]["K5_P_reproduces_seen_plus_P"][name]["max_abs_diff"], 1e-12)

    def test_r19_offline_analyses(self):
        """R19 on XR-1: unseen native bands, the bootstrap (B0 against R14), leads and saturation."""
        import hashlib
        import hk_free as hk
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, held, cal, te = build(root)
            tasks = sorted({r["task_name"] for r in cal})
            ins = {t: f"Do the {t} job." for t in tasks}
            med = {t: 10.0 + 3 * i for i, t in enumerate(tasks)}
            prompts = hk.build_prompts(ins, held, med)
            prompts["example_median_seconds"] = med
            answer = {f"Do the {t} job.\"\n\nHow many": str(8 + 5 * i) for i, t in enumerate(tasks)}
            zero, few, raw, fails = hk.answer_all(prompts, lambda m: next((v for k, v in answer.items()
                                                                         if k in m[1]["content"]), "?"))
            folds_path = args[args.index("--folds") + 1]
            dur = dict(prompt_sha256=hk.prompts_sha256(),
                       folds_sha256=hashlib.sha256(Path(folds_path).read_bytes()).hexdigest(),
                       zero=zero, few=few, answers=len(raw), parse_failures=fails)
            (root / "dur.json").write_text(json.dumps(dur))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(su.main(args + ["--r14-durations", str(root / "dur.json")]), 0)
            (root / "tables.json").rename(root / "r14.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = su.main(args + ["--r19", "--r19-reference", str(root / "r14.json"), "--r19-boot", "3"])
            self.assertEqual(status, 0, buf.getvalue())
            self.assertIn("R19 OK", buf.getvalue())
            r = json.loads((root / "tables.json").read_text())["r19"]
            self.assertTrue(r["checks"]["B0_point_equals_r14"]["ok"])
            self.assertTrue(set(r["native_bands"]) >= {su.SAFE, su.HS})
            for b in r["native_bands"].values():
                self.assertEqual(set(b["unseen"]), {"G_abs", "G"})
            self.assertIn(f"{su.HS}_minus_{su.SAFE}", r["bootstrap_unseen"]["timers"]["G_abs"])
            self.assertEqual(set(r["timers_at_limit"]), {"P_seen", "E_unseen", "G_unseen"})
            self.assertIn("seen", r["leads"][su.HS])

    def test_r20_timer_g_row(self):
        """R20 on XR-1: G against P (seen) and E (unseen); gate T0 against R14's timers_alone."""
        import hashlib
        import hk_free as hk
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, held, cal, te = build(root)
            tasks = sorted({r["task_name"] for r in cal})
            ins = {t: f"Do the {t} job." for t in tasks}
            med = {t: 10.0 + 3 * i for i, t in enumerate(tasks)}
            prompts = hk.build_prompts(ins, held, med)
            prompts["example_median_seconds"] = med
            answer = {f"Do the {t} job.\"\n\nHow many": str(8 + 5 * i) for i, t in enumerate(tasks)}
            zero, few, raw, fails = hk.answer_all(prompts, lambda m: next((v for k, v in answer.items()
                                                                         if k in m[1]["content"]), "?"))
            folds_path = args[args.index("--folds") + 1]
            dur = dict(prompt_sha256=hk.prompts_sha256(),
                       folds_sha256=hashlib.sha256(Path(folds_path).read_bytes()).hexdigest(),
                       zero=zero, few=few, answers=len(raw), parse_failures=fails)
            (root / "dur.json").write_text(json.dumps(dur))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(su.main(args + ["--r14-durations", str(root / "dur.json")]), 0)
            (root / "tables.json").rename(root / "r14.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = su.main(args + ["--r20", "--r20-reference", str(root / "r14.json")])
            self.assertEqual(status, 0, buf.getvalue())
            self.assertIn("R20 OK", buf.getvalue())
            out = json.loads((root / "tables.json").read_text())
            r = out["r20"]
            self.assertEqual(out["r20_failures"], [])
            self.assertEqual(r["checks"]["T0_timers_equal_reference"]["compared"], 4)
            self.assertEqual((r["seen"]["comparator"], r["unseen"]["comparator"]), ("P", "G_abs"))
            # G alone on unseen tasks is Table II's global timer.
            g = out["table2"]["global_timer"]["unseen"]
            self.assertEqual((r["unseen"]["fp"], r["unseen"]["mean_lead"]), (g["fp"], g["mean_lead"]))
            # A wrong reference stops R20.
            ref = json.loads((root / "r14.json").read_text())
            ref["r14"]["timers_alone"]["seen"]["P"]["mean_lead"] += 1e-9
            (root / "r14_bad.json").write_text(json.dumps(ref))
            (root / "tables.json").unlink()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = su.main(args + ["--r20", "--r20-reference", str(root / "r14_bad.json")])
            self.assertEqual(status, 1)
            self.assertIn("R20 FAILED", buf.getvalue())

    def test_r21_original_rules(self):
        """R21 on XR-1: SAFE's band for SAFE-MLP and Hide-and-Seek, seen (+ P) and unseen (+ E, + G); gates R0,
        N0, B0, C0, N1 (against R19's output) and N2 (against an R12-shaped reference)."""
        import hashlib
        import hk_free as hk
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, held, cal, te = build(root)
            tasks = sorted({r["task_name"] for r in cal})
            ins = {t: f"Do the {t} job." for t in tasks}
            med = {t: 10.0 + 3 * i for i, t in enumerate(tasks)}
            prompts = hk.build_prompts(ins, held, med)
            prompts["example_median_seconds"] = med
            answer = {f"Do the {t} job.\"\n\nHow many": str(8 + 5 * i) for i, t in enumerate(tasks)}
            zero, few, raw_, fails = hk.answer_all(prompts, lambda m: next((v for k, v in answer.items()
                                                                          if k in m[1]["content"]), "?"))
            folds_path = args[args.index("--folds") + 1]
            dur = dict(prompt_sha256=hk.prompts_sha256(),
                       folds_sha256=hashlib.sha256(Path(folds_path).read_bytes()).hexdigest(),
                       zero=zero, few=few, answers=len(raw_), parse_failures=fails)
            (root / "dur.json").write_text(json.dumps(dur))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(su.main(args + ["--r14-durations", str(root / "dur.json")]), 0)
            (root / "tables.json").rename(root / "r14.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(su.main(args + ["--r19", "--r19-reference", str(root / "r14.json"), "--r19-boot", "2"]), 0)
            (root / "tables.json").rename(root / "r19.json")
            base = json.loads((root / "r19.json").read_text())
            # First run: a placeholder R12 reference, so N2 fails and nothing else does.
            (root / "list_b_wrong.json").write_text(json.dumps({"B2": {k: dict(
                standalone=dict(fpr=0.0, fp=0, mean_lead=0.0), fused_bonferroni=dict(fpr=0.0, fp=0, mean_lead=0.0, gain=0.0))
                for k in ("safe", "hide_and_seek")}}))
            r21_args = ["--r21", "--r21-reference", str(root / "r19.json"), "--r21-boot", "2"]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = su.main(args + r21_args + ["--r21-r12-reference", str(root / "list_b_wrong.json")])
            self.assertEqual(status, 1)
            self.assertIn("R21 FAILED (no R21 result is used): N2", buf.getvalue())
            out = json.loads((root / "tables.json").read_text())
            self.assertEqual(out["r21_failures"], ["N2"])
            r = out["r21"]
            for k in ("R0_raw_running_max_equals_score", "N0_timer_equals_hk_fit", "B0_equals_r19_native_band",
                      "C0_curve_at_alpha_equals_standalone", "N1_ensemble_equals_r19"):
                self.assertTrue(r["checks"][k]["ok"], (k, r["checks"][k]))
            self.assertEqual(r["checks"]["N1_ensemble_equals_r19"]["compared"], 4)      # 2 heads x (E, G)
            self.assertEqual(out["table1"], base["table1"])
            self.assertEqual(out["table2"], base["table2"])
            # Second run: an R12 reference at 4 decimals of the seen ensemble passes N2.
            ref = {}
            for theirs, mine in (("safe", su.SAFE), ("hide_and_seek", su.HS)):
                e = r["heads"][mine]["seen"]["P"]["seed_average"]
                ref[theirs] = dict(standalone={k: (round(e["standalone"][k], 4) if k != "fp" else e["standalone"][k])
                                               for k in ("fpr", "fp", "mean_lead", "tpr")},
                                   fused_bonferroni={k: (round(e["fused"][k], 4) if k != "fp" else e["fused"][k])
                                                     for k in ("fpr", "fp", "mean_lead", "tpr", "gain")})
            (root / "list_b.json").write_text(json.dumps({"B2": ref}))
            (root / "tables.json").unlink()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = su.main(args + r21_args + ["--r21-r12-reference", str(root / "list_b.json")])
            self.assertEqual(status, 0, buf.getvalue())
            self.assertIn("R21 OK", buf.getvalue())
            r = json.loads((root / "tables.json").read_text())["r21"]
            self.assertTrue(r["checks"]["N2_seen_ensemble_equals_r12_b2"]["ok"])
            self.assertEqual(set(r["heads"]), {su.SAFE, su.HS})
            self.assertEqual(set(r["heads"][su.HS]["unseen"]), {"G_abs", "G"})
            self.assertIn("mean_std_over_seeds", r["heads"][su.HS]["seen"]["P"])
            self.assertIn(f"{su.HS}_minus_{su.SAFE}", r["bootstrap"]["unseen"]["timers"]["G_abs"])
            self.assertEqual(len(r["curves"][su.HS]["unseen"]["mean_std_over_seeds"]), len(r["curve_alphas"]))
            # R21b after R21 (its own bootstrap skipped): variant `full` replays the R21 output's bootstrap (D0).
            (root / "tables.json").rename(root / "r21.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = su.main(args + r21_args + ["--r21-r12-reference", str(root / "list_b.json"), "--r21-boot", "0",
                                                    "--r21b", "--r21b-reference", str(root / "r21.json"), "--r21b-boot", "2"])
            self.assertEqual(status, 0, buf.getvalue())
            self.assertIn("R21b OK", buf.getvalue())
            out = json.loads((root / "tables.json").read_text())
            self.assertEqual(out["r21b_failures"], [])
            self.assertTrue(out["r21b"]["checks"]["D0_full_equals_r21_bootstrap"]["ok"])
            self.assertTrue((root / out["r21b"]["replicates_file"]).exists())
            with np.load(root / out["r21b"]["replicates_file"]) as z:
                self.assertEqual(len(z[f"unseen__test__G_abs__{su.HS}__fused_matched"]), 2)

    def test_r22_fixed_paths(self):
        """R22 on XR-1: validation rows (R8's selection), stage A fold directories with the August cohorts,
        Table II and the ensembles with fixed paths; gates V0, V1, K0 (against R14), K8 and N0."""
        import hashlib
        import hk_free as hk
        import lead_time
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, held, cal, te = build(root)
            rng = np.random.default_rng(22)
            bundle = json.loads((root / "bundle.json").read_text())
            tn = bundle["task_normalizations"]
            # R14's output (K0's reference).
            tasks = sorted({r["task_name"] for r in cal})
            ins = {t: f"Do the {t} job." for t in tasks}
            med = {t: 10.0 + 3 * i for i, t in enumerate(tasks)}
            prompts = hk.build_prompts(ins, held, med)
            prompts["example_median_seconds"] = med
            answer = {f"Do the {t} job.\"\n\nHow many": str(8 + 5 * i) for i, t in enumerate(tasks)}
            zero, few, raw_, fails = hk.answer_all(prompts, lambda m: next((v for k, v in answer.items()
                                                                          if k in m[1]["content"]), "?"))
            folds_path = args[args.index("--folds") + 1]
            dur = dict(prompt_sha256=hk.prompts_sha256(),
                       folds_sha256=hashlib.sha256(Path(folds_path).read_bytes()).hexdigest(),
                       zero=zero, few=few, answers=len(raw_), parse_failures=fails)
            (root / "dur.json").write_text(json.dumps(dur))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(su.main(args + ["--r14-durations", str(root / "dur.json")]), 0)
            (root / "tables.json").rename(root / "r14.json")
            # Validation (August) rows: copies of the test rows with new IDs, seeds and SAFE scores.
            val_files = {}
            for name, per in (("valc", 4), ("valt", 10)):
                out = []
                for r in sx.jsonl(root / "test.jsonl"):
                    i = int(r["rollout_id"].rsplit("-", 1)[1])
                    if i >= per:
                        continue
                    n = r["source_inferences"]
                    raw_safe = {s: (rng.standard_normal(n) + (0.8 * np.linspace(0, 1, n) if r["failed"] else 0.0)).tolist()
                                for s in SEEDS}
                    norm = {s: (np.maximum.accumulate(np.asarray(raw_safe[s])) - tn[s][r["task_name"]]["location"])
                            / tn[s][r["task_name"]]["scale"] for s in SEEDS}
                    out.append(dict(r, rollout_id=f"{name}-{r['task_name']}-{i}-{int(r['failed'])}",
                                    environment_seed=r["environment_seed"] + 100000, per_seed_raw_safe=raw_safe,
                                    safe_scores=np.mean([norm[s] for s in SEEDS], axis=0).tolist()))
                sx.write_jsonl(root / f"{name}.jsonl", out)
                val_files[name] = root / f"{name}.jsonl"
            sel = {c: lead_time.sel(lead_time.load(val_files[n]), q) for c, n, q in
                   (("august_calibration", "valc", 3), ("august_test", "valt", 10))}
            for c, rows in sel.items():
                (root / f"{c}_ids.json").write_text(json.dumps([r["rollout_id"] for r in rows]))
            val_rows = sel["august_calibration"] + sel["august_test"]
            # Amendment 2's August files (Hide-and-Seek family and shuffle) and the OOD per-seed outputs.
            a2n = json.loads((root / "amend2_norms_seen.json").read_text())
            for c, rows in sel.items():
                out = []
                for r in rows:
                    for det in (*su.HS_FAMILY, su.SHUFFLE):
                        per_raw = raw(rng, r, 0.5)
                        per = per_task_traj(per_raw, {s: a2n[det][s][r["task_name"]] for s in SEEDS})
                        out.append(dict(detector=det, rollout_id=r["rollout_id"], task_name=r["task_name"],
                                        failed=r["failed"], per_seed_raw=per_raw,
                                        safe_scores=np.mean([per[s] for s in SEEDS], axis=0).tolist()))
                sx.write_jsonl(root / f"a2_{c}.jsonl", out)
            ood_val = []
            for s in SEEDS:
                paths = []
                for c, rows in sel.items():
                    out = [dict(rollout_id=r["rollout_id"], task_name=r["task_name"], failed=r["failed"],
                                baseline_scores={d: np.maximum.accumulate(rng.standard_normal(r["L"])).tolist() for d in OOD})
                           for r in rows]
                    sx.write_jsonl(root / f"ood_val_{s}_{c}.jsonl", out)
                    paths.append(root / f"ood_val_{s}_{c}.jsonl")
                ood_val.append((s, *paths))
            # Stage A: each family's fold directories with the August cohorts added.
            fam_dirs = {"xt": args[args.index("--xt-fold-dirs") + 1:][:4], "ood": args[args.index("--ood-fold-dirs") + 1:][:4],
                        "safe": args[args.index("--safe-fold-dirs") + 1:][:4]}
            for family, dirs in fam_dirs.items():
                for F, d in zip(("0", "1", "2", "3"), dirs):
                    d, new = Path(d), root / "r22a" / family / f"fold_{F}"
                    new.mkdir(parents=True)
                    for f in d.iterdir():
                        (new / f.name).write_bytes(f.read_bytes())
                    dets = list(json.loads((d / "normalizations.json").read_text()))
                    for c, rows in sel.items():
                        sx.write_jsonl(new / f"{c}_predictions.jsonl", [
                            dict(detector=det, rollout_id=r["rollout_id"], task_name=r["task_name"], failed=r["failed"],
                                 fold=F, held_out=r["task_name"] in held[F],
                                 per_seed_raw={s: rng.standard_normal(r["L"]).tolist() for s in SEEDS})
                            for det in dets for r in rows])
            r22 = ["--r22", "--r22-scores-dir", str(root / "r22a"), "--r22-val-calibration", str(val_files["valc"]),
                   "--r22-val-test", str(val_files["valt"]),
                   "--r22-val-ids", str(root / "august_calibration_ids.json"), str(root / "august_test_ids.json"),
                   "--r22-val-hs-scores", str(root / "a2_august_calibration.jsonl"), str(root / "a2_august_test.jsonl"),
                   "--r22-reference", str(root / "r14.json"), "--r22-boot", "2", "--r22-null-draws", "2"]
            for s, c, t in ood_val:
                r22 += ["--r22-ood-val", s, str(c), str(t)]
            (root / "r8.json").write_text(json.dumps({"rows": {}}))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = su.main(args + r22 + ["--r22-r8", str(root / "r8.json")])
            self.assertEqual(status, 1, buf.getvalue())                     # K8 against an empty R8
            out = json.loads((root / "tables.json").read_text())
            self.assertEqual(out["r22_failures"], ["K8"])
            r = out["r22"]
            for k in ("V0_disjoint", "K0_adaptive_reproduces_r14", "N0_timer_margins_unchanged"):
                self.assertTrue(r["checks"][k]["ok"], (k, r["checks"][k]))
            self.assertTrue(all(v["ok"] for v in r["checks"]["V1_selection_equals_stage_a_ids"].values()))
            self.assertEqual(r["validation"]["counts"], {c: len(v) for c, v in sel.items()})
            rows8 = {k: dict(path=dict(u=None if e["chosen"]["all"]["u"] is None else round(e["chosen"]["all"]["u"], 4),
                                       s=e["chosen"]["all"]["s"]), gain=round(e["gain"], 4))
                     for k, e in (("safe", r["ensembles_seen"][su.SAFE]), ("hide_and_seek", r["ensembles_seen"][su.HS]))}
            (root / "r8.json").write_text(json.dumps({"rows": rows8}))
            (root / "tables.json").unlink()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = su.main(args + r22 + ["--r22-r8", str(root / "r8.json")])
            self.assertEqual(status, 0, buf.getvalue())
            self.assertIn("R22 OK", buf.getvalue())
            r = json.loads((root / "tables.json").read_text())["r22"]
            self.assertIn(su.HS, r["table2"]["seen"])
            self.assertEqual(set(r["table2"]["unseen"][su.HS]), {"G_abs", "G"})
            self.assertIn("ci95", r["ensembles_seen"][su.HS]["matched"])
            self.assertEqual(set(r["nulls"]["nulls"]), {"noise", "safe_resampled", f"{su.HS}_resampled"})
            self.assertIn(f"{su.HS}_minus_{su.SAFE}", r["unseen_bootstrap"]["timers"]["G_abs"])
            self.assertIsNotNone(r["budget_timer"])

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
