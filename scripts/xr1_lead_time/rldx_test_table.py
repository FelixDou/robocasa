"""R15: RLDX-1's Table II rows on its test set (plan: docs/rldx_test_table_plan.md). Exploratory.

The paper gives both policies the same three sets: training, validation and test. On RLDX-1 the
recorded held-out rollouts (100 + 100, with 60 calibration successes) are the validation set, and
E1's new collection (calibration and test sets of 20 successes and 20 failures per task) is the
test set. R7 (rldx_unseen.py) computed Table II's RLDX-1 rows on the validation set; this script
computes the same rows, with the same models, on the test set.

Protocol, as on XR-1: per detector and training seed, the score is OR-fused with P (seen tasks) or
G (unseen tasks, per fold, fitted on the seen tasks); the calibration successes set the margin and
the calibration failures choose the fused path (confirm_lead.fit_fast). R7 had no calibration
failures and chose the path on in-sample training failures; that variant is also reported
("train") for comparison. Rows: rldx_unseen.DETECTORS, the two SAFE heads, and the same-task
shuffle control (seen tasks only).

Commands:
  ids        the test-set ID lists, from E1's scores (the rows of Table III).
  safe-fold  one SAFE head's fold directory on the E1 sets: R7's assembled fold directory (training
             cohort and normalization, copied) plus the fold's models' scores of E1's two exports.
  run        the table, after these gates:
    G0  the code path: with R7's fold directories and R7's protocol, every recorded row of
        rldx_unseen_safe.json (seen and unseen) and the shuffle row of rldx_per_seed.json are
        reproduced exactly;
    G1  the models: every new cross-task fold directory reproduces R7's trajectories of the
        recorded calibration, test and training rollouts (max |diff| <= 1e-12) and its
        normalizations (both at 1e-12); every SAFE fold directory copies R7's training cohort and normalization
        byte for byte; the shuffle directory reproduces the earlier trajectories;
    G2  E1: H_k and rho_k equal the freeze manifest's; P alone on the test set equals E1's P; the
        seed-averaged SAFE-LSTM and Hide-and-Seek trajectories on E1's sets equal E1's.

R17 (docs/r17_elapsed_time_plan.md), with run --elapsed-time: every row of the paper's Table II
(path chosen on calibration failures) is also fitted against R14's timers (hk_free.py):
  unseen tasks  G, as a gate (K1: equal to the table's unseen + G rows, 1e-12), and G_abs, the
                elapsed-time timer, which is not given H_k;
  seen tasks    P, as a gate (K5: equal to the table's seen + P rows at their stored 4 decimals),
                and P_mult, P without H_k.
With --reference-table, the table itself must equal an earlier run's (K0), so the R17 rows belong to
the published Table II.

R19 (docs/r19_triage_b_plan.md), with run --r19: the draft v3 mock review's offline analyses
(r19_extras.py): native functional bands (seen + P, unseen + E and + G), a full-pipeline bootstrap of
the unseen-task gains (gate B0: its point estimates equal R17's), the timers' alarms at the limit,
lead in seconds, per-task gains and the zero-input model's saturation.

R18 (docs/r18_lead_fpr_curves_plan.md), with run --curves: calibrated lead-FPR curves on the test
set (lead_fpr_curves.py) of every monitor alone (seen and unseen tasks; the shuffle on seen tasks only)
and of the timers G, E and P. The same code reproduces XR-1's Table I at alpha = 0.05 (gate C1 in
seen_unseen_tables.py); RLDX-1 has no standalone table to compare with.

Usage:
  python -I rldx_test_table.py ids --inputs-dir RLDXHS/inputs --e1-scores-dir E1/scores --output-dir IDS
  python -I rldx_test_table.py safe-fold --r7-fold-dir R7S/folds/lstm/fold_0 --folds FOLDS \\
      --cohort calibration IDS/calibration_ids.json --cohort test IDS/test_ids.json \\
      --scores calibration 0 DIR ... --scores test 2 DIR [--e1-reference calibration E1/scores/... ] --output-dir OUT
  python -I rldx_test_table.py run --safe-dir <SAFE20>/calibration/lstm --inputs-dir RLDXHS/inputs \\
      --folds FOLDS --ids-dir IDS --e1-scores-dir E1/scores --e1-result E1/e1_result.json --freeze-manifest M.json \\
      --fold-dirs NEW/fold_{0,1,2,3,all} --r7-fold-dirs R7/folds/fold_{0,1,2,3,all} \\
      --fold-dirs NEWS/lstm/fold_{...} --r7-fold-dirs R7S/folds/lstm/fold_{...} (and indep) \\
      --shuffle-dir SHUFFLE --rldxhs-scores-dir RLDXHS/scores --r7-table rldx_unseen_safe.json \\
      --reference-per-seed rldx_per_seed.json --output OUT.json
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parents[1]))

import confirm_lead as cl  # noqa: E402
import cross_task_lead as ctl  # noqa: E402
import quantile_timer as qt  # noqa: E402
import rldx_replication as rr  # noqa: E402
import rldx_unseen as ru  # noqa: E402

PLAN = "docs/rldx_test_table_plan.md (R15)"
SEEDS, ALL = ru.SEEDS, ru.ALL
SETS = ("calibration", "test")
SHUFFLE = "hide_and_seek_task_shuffle"
MODES = ("calibration", "train")     # which failures choose the fused path
EXACT = 1e-12


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def empty_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"error: output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def gap(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")


# ---- ids -------------------------------------------------------------------------------------

def ids_cmd(a):
    inputs = Path(a.inputs_dir)
    bundle = json.loads((inputs / "rldx_bundle.json").read_text())
    recorded = {n: set(json.loads((inputs / f"{n}_ids.json").read_text())) for n in ("train", "calibration", "test")}
    seen_ids, summary = set(), {}
    lists = {}
    for name in SETS:
        path = Path(a.e1_scores_dir) / f"{name}_safe_lstm_predictions.jsonl"
        rows = jsonl(path)
        ids = [str(r["rollout_id"]) for r in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"STOP: {name}: duplicate rollout IDs")
        clash = sorted(set(ids) & set().union(*recorded.values()))
        if clash:
            raise SystemExit(f"STOP: {name} shares {len(clash)} rollouts with the recorded sets, e.g. {clash[:3]}")
        if set(ids) & seen_ids:
            raise SystemExit("STOP: the calibration and test sets share rollouts")
        seen_ids |= set(ids)
        unknown = sorted({r["task_name"] for r in rows} - set(bundle["task_horizons"]))
        if unknown:
            raise SystemExit(f"STOP: {name} has tasks outside the bundle: {unknown}")
        per_task = {}
        for r in rows:
            c = per_task.setdefault(r["task_name"], [0, 0])
            c[bool(r["failed"])] += 1
        lists[name] = sorted(ids)
        summary[name] = dict(source=str(path.resolve()), sha256=sha256(path), rollouts=len(ids),
                             successes=sum(c[0] for c in per_task.values()), failures=sum(c[1] for c in per_task.values()),
                             per_task={t: dict(successes=c[0], failures=c[1]) for t, c in sorted(per_task.items())})
    out = empty_dir(a.output_dir)
    for name, ids in lists.items():
        (out / f"{name}_ids.json").write_text(json.dumps(ids) + "\n")
    write_json(out / "ids_summary.json", dict(plan=PLAN, sets=summary))
    for name, s in summary.items():
        counts = {(v["successes"], v["failures"]) for v in s["per_task"].values()}
        print(f"{name}: {s['rollouts']} rollouts ({s['successes']} successes, {s['failures']} failures), "
              f"{len(s['per_task'])} tasks, per-task (successes, failures): {sorted(counts)}")
    print("IDS OK")
    return 0


# ---- safe-fold -------------------------------------------------------------------------------

def safe_fold_cmd(a):
    r7 = Path(a.r7_fold_dir)
    meta7 = json.loads((r7 / "scoring.json").read_text())
    F, head = str(meta7["fold"]), meta7["head"]
    det = f"safe_{head}"
    if [str(s) for s in meta7["seeds"]] != list(SEEDS):
        raise SystemExit(f"STOP: {r7}: seeds {meta7['seeds']}")
    bad = sorted(k for k, v in {**meta7.get("checks", {}), **meta7.get("reference_checks", {})}.items()
                 if isinstance(v, dict) and not v.get("ok", True))
    if bad:
        raise SystemExit(f"STOP: {r7} recorded failed checks {bad}")
    if meta7["folds_sha256"] != sha256(a.folds):
        raise SystemExit(f"STOP: {r7} was assembled with another fold file")
    tasks, held = ctl.fold_sets(json.loads(Path(a.folds).read_text()))
    ids = {}
    for name, path in a.cohort:
        if name not in SETS or name in ids:
            raise SystemExit(f"error: --cohort must name each of {SETS} once")
        ids[name] = [str(x) for x in json.loads(Path(path).read_text())]
    if sorted(ids) != sorted(SETS):
        raise SystemExit(f"error: --cohort must name each of {SETS} once")
    runs = {s: Path(p).resolve() for s, p in meta7["training_runs"].items()}
    ck = meta7["training_checkpoint_sha256"]
    problems = []
    for s in SEEDS:
        if sha256(runs[s] / "model_final.ckpt") != ck[s]:
            problems.append(f"seed {s}: the training run's checkpoint changed since R7")
    scores, prov_out = {n: {} for n in SETS}, {n: {} for n in SETS}
    for name, s, d in a.scores:
        d, s = Path(d).resolve(), str(s)
        if name not in SETS or s not in SEEDS or s in scores[name]:
            raise SystemExit(f"error: --scores {name} {s} is unknown or given twice")
        prov = json.loads((d / "provenance.json").read_text())
        if Path(prov["training_run_dir"]).resolve() != runs[s]:
            problems.append(f"{d}: scored with {prov['training_run_dir']}, not R7's seed {s} run")
        if prov["checkpoint_sha256"] != ck[s]:
            problems.append(f"{d}: checkpoint hash differs from R7's")
        if prov.get("group") != "all" or prov.get("unseen_tasks") != "score" or prov.get("cutoff_rule") != "training":
            problems.append(f"{d}: not --group all --unseen-tasks score with the training cutoff rule")
        if set(prov.get("unseen_task_names", [])) != held[F] & set(prov.get("task_names", [])):
            problems.append(f"{d}: tasks scored as unseen differ from fold {F}'s held-out tasks")
        scores[name][s] = {str(r["rollout_id"]): r for r in jsonl(d / "scores.jsonl")}
        prov_out[name][s] = dict(path=str(d), export_dir=prov["export_dir"], checkpoint_sha256=prov["checkpoint_sha256"])
    for name in SETS:
        if sorted(scores[name]) != list(SEEDS):
            raise SystemExit(f"error: --scores {name} needs seeds 0, 1, 2")
        if len({p["export_dir"] for p in prov_out[name].values()}) != 1:
            problems.append(f"{name}: seeds scored different exports")
        for s in SEEDS:
            if set(scores[name][s]) != set(ids[name]):
                problems.append(f"{name} seed {s}: scored rollouts differ from the ID list")
    if prov_out["calibration"]["0"]["export_dir"] == prov_out["test"]["0"]["export_dir"]:
        problems.append("calibration and test were scored from the same export")
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems))

    out = empty_dir(a.output_dir)
    counts = {}
    for name in SETS:
        n = 0
        with (out / f"{name}_predictions.jsonl").open("w") as stream:
            for rid in sorted(ids[name]):
                rows = {s: scores[name][s][rid] for s in SEEDS}
                r0 = rows["0"]
                if any(r["task_name"] != r0["task_name"] or bool(r["failed"]) != bool(r0["failed"])
                       or len(r["scores"]) != len(r0["scores"]) for r in rows.values()):
                    raise SystemExit(f"STOP: {name}: seeds disagree on {rid}")
                is_held = r0["task_name"] in held[F]
                if any(bool(r.get("training_task")) == is_held for r in rows.values()):
                    raise SystemExit(f"STOP: {name}: training-task flag of {rid} contradicts fold {F}")
                stream.write(json.dumps(dict(detector=det, rollout_id=rid, task_name=r0["task_name"],
                                             failed=bool(r0["failed"]), fold=F, held_out=is_held,
                                             source_inferences=len(r0["scores"]),
                                             per_seed_raw={s: rows[s]["scores"] for s in SEEDS})) + "\n")
                n += 1
        counts[name] = n
    copied = {}
    for f in ("train_predictions.jsonl", "normalizations.json"):
        shutil.copyfile(r7 / f, out / f)
        copied[f] = dict(r7=sha256(r7 / f), here=sha256(out / f))
        if copied[f]["r7"] != copied[f]["here"]:
            raise SystemExit(f"STOP: copying {f} changed it")

    reference = {}
    for name, path in a.e1_reference or []:
        if F != ALL or head != "lstm":
            raise SystemExit("error: --e1-reference applies to fold all of the LSTM head (E1's frozen models)")
        ref = {str(r["rollout_id"]): r["per_seed_raw"] for r in jsonl(path)}
        worst, missing = 0.0, 0
        for rid in ids[name]:
            if rid not in ref:
                missing += 1
                continue
            for s in SEEDS:
                worst = max(worst, gap(scores[name][s][rid]["scores"], ref[rid][s]))
        reference[f"{name}/{det}/e1"] = dict(reference=str(Path(path).resolve()), max_abs_diff=worst, missing=missing,
                                             ok=missing == 0 and len(ref) == len(ids[name]) and worst <= EXACT)
        print(f"reference {name} vs E1: max |diff| {worst:.3g}, missing {missing} -> "
              f"{'OK' if reference[f'{name}/{det}/e1']['ok'] else 'MISMATCH'}", flush=True)
    meta = dict(protocol="r15_safe_fold_e1_sets_no_metrics", plan=PLAN, fold=F, held_out_tasks=sorted(held[F]),
                training_rollouts_by_task=meta7["training_rollouts_by_task"], folds_sha256=sha256(a.folds),
                detectors=[det], head=head, seeds=[0, 1, 2], score_train=True, r7_fold_dir=str(r7.resolve()),
                copied_from_r7=copied, training_runs={s: str(r) for s, r in runs.items()}, training_checkpoint_sha256=ck,
                scores=prov_out, counts=counts, checks={}, reference_checks=reference)
    write_json(out / "scoring.json", meta)
    ok = all(v["ok"] for v in reference.values())
    print(f"fold {F} {det}: " + ", ".join(f"{c} {n}" for c, n in counts.items()) + " (training cohort and normalization "
          "copied from R7)")
    print("SAFE FOLD OK" if ok else "REFERENCE MISMATCH: do not analyze this fold")
    return 0 if ok else 1


# ---- rows ------------------------------------------------------------------------------------

def e1_rows(scores_dir, name, H):
    """E1 rows (Table III's): IDs, labels and lengths from E1's SAFE-LSTM file, checked against Hide-and-Seek's."""
    files = {d: {str(r["rollout_id"]): r for r in jsonl(Path(scores_dir) / f"{name}_{d}_predictions.jsonl")}
             for d in ("safe_lstm", "hide_and_seek")}
    if sorted(files["safe_lstm"]) != sorted(files["hide_and_seek"]):
        raise SystemExit(f"STOP: E1 {name}: the two detectors scored different rollouts")
    rows = []
    for rid in sorted(files["safe_lstm"]):
        a, b = files["safe_lstm"][rid], files["hide_and_seek"][rid]
        if (a["task_name"], bool(a["failed"])) != (b["task_name"], bool(b["failed"])) or \
                len(a["safe_scores"]) != len(b["safe_scores"]):
            raise SystemExit(f"STOP: E1 {name}: detectors disagree on {rid}")
        if a["task_name"] not in H:
            raise SystemExit(f"STOP: E1 {name}: task {a['task_name']} has no H_k")
        rows.append(dict(rollout_id=rid, task_name=a["task_name"], failed=bool(a["failed"]), L=len(a["safe_scores"]),
                         split=name, e1_safe_lstm=np.asarray(a["safe_scores"], float),
                         e1_hide_and_seek=np.asarray(b["safe_scores"], float)))
    return rr.finish(rows, H)


def raw_file(path, wanted=None):
    """detector -> rollout -> seed -> raw trajectory (any predictions file with per_seed_raw)."""
    out = {}
    with open(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r["rollout_id"])
            if wanted is not None and rid not in wanted:
                continue
            out.setdefault(r["detector"], {})[rid] = {s: np.asarray(v, float) for s, v in r["per_seed_raw"].items()}
    return out


def same_raw(a, b):
    """Largest |a - b| over detectors, rollouts and seeds; inf if the sets differ."""
    if set(a) != set(b):
        return float("inf")
    worst = 0.0
    for d in a:
        if set(a[d]) != set(b[d]):
            return float("inf")
        for rid, seeds in a[d].items():
            if set(seeds) != set(b[d][rid]):
                return float("inf")
            for s in seeds:
                worst = max(worst, gap(seeds[s], b[d][rid][s]))
    return worst


def tree_gap(a, b):
    """Largest |a - b| over the numbers of two JSON trees of the same shape; inf if the shapes differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max((tree_gap(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


def check_new_meta(F, meta, tasks, held, folds_sha, detectors):
    if str(meta["fold"]) != F or set(meta["held_out_tasks"]) != held[F]:
        raise SystemExit(f"STOP: fold {F}: fold or held-out tasks differ from the fold file")
    if set(meta["training_rollouts_by_task"]) != set(tasks) - held[F]:
        raise SystemExit(f"STOP: fold {F}: training tasks are not exactly the seen tasks")
    if meta["folds_sha256"] != folds_sha:
        raise SystemExit(f"STOP: fold {F}: scored with a different fold file")
    if [str(s) for s in meta["seeds"]] != list(SEEDS) or not meta.get("score_train"):
        raise SystemExit(f"STOP: fold {F}: seeds are not 0-2 or --score-train is missing")
    if list(meta["detectors"]) != list(detectors):
        raise SystemExit(f"STOP: fold {F}: detectors {meta['detectors']} differ from the family's {detectors}")
    bad = sorted(k for k, v in {**(meta.get("checks") or {}), **(meta.get("reference_checks") or {})}.items()
                 if isinstance(v, dict) and not v.get("ok", True))
    if bad:
        raise SystemExit(f"STOP: fold {F}: the directory recorded failed checks {bad}")


def attach_shuffle(parts, scores_dir, cohort_files):
    """Seen-only rows of a score_cohorts directory: per-seed normalized running maxima (per task)."""
    scores_dir = Path(scores_dir)
    norms = json.loads((scores_dir / "normalizations.json").read_text())[SHUFFLE]
    worst = 0.0
    for cohort, rows in parts.items():
        wanted = {str(r["rollout_id"]) for r in rows}
        raw, stored = {}, {}
        with open(scores_dir / cohort_files[cohort]) as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = str(row["rollout_id"])
                if row["detector"] == SHUFFLE and rid in wanted:
                    raw[rid] = {s: np.asarray(v, float) for s, v in row["per_seed_raw"].items()}
                    stored[rid] = np.asarray(row["safe_scores"], float)
        for r in rows:
            if cohort == "train" and not r["failed"]:
                continue
            x = raw.get(str(r["rollout_id"]))
            if x is None or sorted(x) != list(SEEDS) or any(len(v) != r["L"] for v in x.values()):
                raise SystemExit(f"STOP: {scores_dir}: no valid {SHUFFLE} trajectory for {r['rollout_id']} ({cohort})")
            per = {s: ru.normalized(x[s], norms[s][r["task_name"]]) for s in SEEDS}
            for s in SEEDS:
                r[f"{SHUFFLE}#{s}@all/task"] = per[s]
            # The seed average is score_cohorts' stored one, as rldx_per_seed.py used it.
            worst = max(worst, gap(np.mean([per[s] for s in SEEDS], axis=0), stored[str(r["rollout_id"])]))
            r[f"{SHUFFLE}@all/task"] = stored[str(r["rollout_id"])]
    if worst > 1e-9:
        raise SystemExit(f"STOP: {scores_dir}: stored seed averages differ from the per-seed ones (max {worst})")


# ---- fits ------------------------------------------------------------------------------------

def cal_rows(parts, mode, keep=lambda r: True):
    """The cohort fit_fast reads: calibration successes set the margin, `mode` failures choose the path."""
    cal = [r for r in parts["calibration"] if keep(r)]
    if mode == "calibration":
        return cal
    if mode == "train":
        return [r for r in cal if not r["failed"]] + [r for r in parts["train"] if r["failed"] and keep(r)]
    raise ValueError(mode)


def seen_plus_P(parts, key, rk, mode):
    """rldx_unseen.seen_plus_P with the path-choice failures of `mode` (train: identical to R7)."""
    tm, fu = cl.fit_fast(cl.Cohort(cal_rows(parts, mode), key), rk)
    eP, lP = cl.evaluate_fast(cl.Cohort(parts["test"], key), rk, tm)
    e, lF = cl.evaluate_fast(cl.Cohort(parts["test"], key), rk, fu)
    fpr, lead = float(e["fpr"]), float(e["mean_lead"])
    return dict(fpr=fpr, mean_lead=lead, gain=round(float(np.mean(lF - lP)), 4), twa=(lead + 1.0 - fpr) / 2.0,
                fp=e["fp"], P=dict(lead=float(eP["mean_lead"]), fpr=float(eP["fpr"])), path=cl.det_json(fu))


def unseen_plus_G(parts, held, folds, key_of, rk0, mode):
    """rldx_unseen.unseen_plus_G with the path-choice failures of `mode` (train: identical to R7)."""
    pt, pf, paths = [], [], {}
    for F in folds:
        cal = cal_rows(parts, mode, keep=lambda r, F=F: r["task_name"] not in held[F])
        test = [r for r in parts["test"] if r["task_name"] in held[F]]
        key = key_of(F)
        tm, fu = cl.fit_fast(cl.Cohort(cal, key), rk0)
        e = cl.Cohort(test, key)
        pt.append(ctl.raw_eval(e, rk0, tm))
        pf.append(ctl.raw_eval(e, rk0, fu))
        paths[F] = dict(timer=cl.det_json(tm), fused=cl.det_json(fu), cal_successes=sum(not r["failed"] for r in cal),
                        cal_failures=sum(r["failed"] for r in cal), test=len(test))
    tG, fD = ctl.pool(pt), ctl.pool(pf)
    fpr, lead = fD["fp"] / fD["n"], float(fD["lead"].mean())
    gfpr, glead = tG["fp"] / tG["n"], float(tG["lead"].mean())
    return dict(fpr=fpr, mean_lead=lead, gain=ctl.gain(fD, tG), twa=(lead + 1.0 - fpr) / 2.0, fp=fD["fp"],
                n_success=fD["n"], n_failure=int(len(fD["lead"])),
                timer=dict(fpr=gfpr, mean_lead=glead, fp=tG["fp"], twa=(glead + 1.0 - gfpr) / 2.0), paths=paths)


def timer_seen(parts, key, r, mode, lead_P=None):
    """A timer alone on the seen tasks (P with rho_k, G with rho = 0)."""
    tm, _ = cl.fit_fast(cl.Cohort(cal_rows(parts, mode), key), r)
    e, _ = cl.evaluate_fast(cl.Cohort(parts["test"], key), r, tm)
    fpr, lead = float(e["fpr"]), float(e["mean_lead"])
    out = dict(fpr=fpr, mean_lead=lead, fp=e["fp"], twa=(lead + 1.0 - fpr) / 2.0, lam=cl.det_json(tm)["lam"])
    if lead_P is not None:
        out["gain_over_P"] = round(lead - lead_P, 4)
    return out


def table(parts, held, folds, detectors, rk, rk0, mode, seen_only=()):
    out = {}
    for d in detectors:
        row = {}
        seen = {"seed_average": seen_plus_P(parts, f"{d}@all/task", rk, mode)}
        for s in SEEDS:
            seen[f"seed_{s}"] = seen_plus_P(parts, f"{d}#{s}@all/task", rk, mode)
        seen["mean_std_over_seeds"] = ru.mean_std([seen[f"seed_{s}"] for s in SEEDS])
        row["seen"] = seen
        if d not in seen_only:
            un = {"seed_average": unseen_plus_G(parts, held, folds, lambda F, d=d: f"{d}@{F}", rk0, mode)}
            for s in SEEDS:
                un[f"seed_{s}"] = unseen_plus_G(parts, held, folds, lambda F, d=d, s=s: f"{d}#{s}@{F}", rk0, mode)
            un["mean_std_over_seeds"] = ru.mean_std([un[f"seed_{s}"] for s in SEEDS])
            row["unseen"] = un
        out[d] = row
    return out


def strip(x):
    """JSON round trip, so stored and recomputed results compare as the file stores them."""
    return json.loads(json.dumps(x, default=float))


# ---- R17: the elapsed-time timer as the unseen-task comparator ---------------------------------

R17_PLAN = "docs/r17_elapsed_time_plan.md (R17)"
R17_UNSEEN_TIMERS = ("G", "G_abs")
R17_SEEN_TIMERS = ("P", "P_mult")


def elapsed_time_block(parts, held, folds, detectors, rk, rows):
    """R17: the rows of table2.calibration fitted against R14's timers (hk_free.block).
    rows: that table, for gates K1 (G = unseen + G, 1e-12) and K5 (P = seen + P at 4 decimals)."""
    import hk_free as hk
    labels = ["seed_average"] + [f"seed_{s}" for s in SEEDS]
    tag = lambda lab: "" if lab == "seed_average" else "#" + lab.split("_", 1)[1]
    out_u, out_s, k1, k5 = {}, {}, {}, {}
    for d in detectors:
        def pairs_u(lab, d=d):
            out = {}
            for F in folds:
                cal = cal_rows(parts, "calibration", keep=lambda r, F=F: r["task_name"] not in held[F])
                test = [r for r in parts["test"] if r["task_name"] in held[F]]
                key = f"{d}{tag(lab)}@{F}"
                out[F] = (cl.Cohort(cal, key), cl.Cohort(test, key))
            return out

        def pairs_s(lab, d=d):
            key = f"{d}{tag(lab)}@all/task"
            return {ALL: (cl.Cohort(cal_rows(parts, "calibration"), key), cl.Cohort(parts["test"], key))}
        out_s[d] = hk.block(pairs_s, labels, R17_SEEN_TIMERS, None, rk)
        k5[d] = all(round(out_s[d]["P"][lab][k], 4) == round(rows[d]["seen"][lab][k], 4)
                    for lab in labels for k in ("fpr", "mean_lead", "gain"))
        if "unseen" in rows[d]:
            out_u[d] = hk.block(pairs_u, labels, R17_UNSEEN_TIMERS, None, rk)
            k1[d] = max(abs(out_u[d]["G"][lab][k] - rows[d]["unseen"][lab][k])
                        for lab in labels for k in ("fpr", "mean_lead", "gain"))
    problems = [f"K1: {d}: G here differs from unseen + G by {v}" for d, v in k1.items() if not v <= EXACT]
    problems += [f"K5: {d}: P here differs from seen + P at 4 decimals" for d, ok in k5.items() if not ok]
    first = next(d for d in detectors if d in out_u)
    result = dict(plan=R17_PLAN, path_choice="calibration failures (the paper's Table II)",
                  checks=dict(K1_G_reproduces_unseen_plus_G=k1, K5_P_reproduces_seen_plus_P=k5),
                  timers_alone=dict(unseen={t: out_u[first][t]["seed_average"]["timer"] for t in R17_UNSEEN_TIMERS},
                                    seen={t: out_s[first][t]["seed_average"]["timer"] for t in R17_SEEN_TIMERS}),
                  unseen=out_u, seen=out_s)
    return result, problems


R18_PLAN = "docs/r18_lead_fpr_curves_plan.md (R18)"


def rldx_curves(parts, held, folds, detectors, rk):
    """R18 on RLDX-1: the same curves as seen_unseen_tables.r18_xr1, from the test-set rows' scores."""
    import lead_fpr_curves as lfc
    tag = lambda lab: "" if lab == "seed_average" else "#" + lab.split("_", 1)[1]
    labels = ["seed_average"] + [f"seed_{s}" for s in SEEDS]
    rows = lambda rs, key: [dict(score=np.asarray(r[key], float), L=r["L"], horizon=r["horizon"], failed=bool(r["failed"]),
                                 task_name=r["task_name"]) for r in rs]
    cal, te = parts["calibration"], parts["test"]
    out = dict(plan=R18_PLAN, alphas=list(lfc.ALPHAS), seen={}, unseen={}, timers=dict(seen={}, unseen={}))
    for d in detectors:
        sc = {lab: lfc.curve([(rows(cal, f"{d}{tag(lab)}@all/task"), rows(te, f"{d}{tag(lab)}@all/task"))])
              for lab in labels}
        sc["mean_std_over_seeds"] = lfc.mean_std([sc[f"seed_{s}"] for s in SEEDS])
        out["seen"][d] = sc
        if d == SHUFFLE:
            continue
        uc = {lab: lfc.curve([(rows([r for r in cal if r["task_name"] not in held[F]], f"{d}{tag(lab)}@{F}"),
                               rows([r for r in te if r["task_name"] in held[F]], f"{d}{tag(lab)}@{F}")) for F in folds])
              for lab in labels}
        uc["mean_std_over_seeds"] = lfc.mean_std([uc[f"seed_{s}"] for s in SEEDS])
        out["unseen"][d] = uc
    for kind in lfc.TIMER_KINDS:
        out["timers"]["seen"][kind] = lfc.curve([(lfc.timer_rows(cal, kind, rk), lfc.timer_rows(te, kind, rk))])
    for kind in ("G", "E"):
        out["timers"]["unseen"][kind] = lfc.curve(
            [(lfc.timer_rows([r for r in cal if r["task_name"] not in held[F]], kind),
              lfc.timer_rows([r for r in te if r["task_name"] in held[F]], kind)) for F in folds])
    return out


def r19_rldx(parts, held, folds, rk, types, reference, B):
    """R19 on RLDX-1 (r19_extras). reference: R17's output, for gate B0."""
    import r19_extras as rx
    spi = 0.4                                   # 8 control steps per inference at 20 Hz
    cal, te = parts["calibration"], parts["test"]
    unseen_parts = {F: ([r for r in cal if r["task_name"] not in held[F]], [r for r in te if r["task_name"] in held[F]])
                    for F in folds}
    heads = ("safe_indep", "safe_lstm", "hide_and_seek")
    bands = {}
    for d in heads:
        bands[d] = dict(seen=rx.native_band({ALL: (cal, te)}, lambda F, d=d: f"{d}~@all/task",
                                            lambda F, d=d: f"{d}@all/task", "P", rk),
                        unseen={t: rx.native_band(unseen_parts, lambda F, d=d: f"{d}~@{F}", lambda F, d=d: f"{d}@{F}", t, rk)
                                for t in ("G_abs", "G")})
    design = rx.UnseenDesign(cal, te, held, folds)
    key_ofs = {d: (lambda F, d=d: f"{d}@{F}") for d in heads + ("hide_and_seek_zero",)}
    boot, point = rx.boot_unseen(design, key_ofs, ("G_abs", "G"), rk, B=B,
                                 contrasts=(("hide_and_seek", "safe_indep"), ("hide_and_seek", "hide_and_seek_zero")))
    ref = {t: {d: reference["r17"]["unseen"][d][t]["seed_average"] for d in key_ofs} for t in ("G_abs", "G")}
    b0 = rx.check_point(point, ref)
    detail, limits = {}, {}
    for d in heads:
        seen_pairs = {ALL: (cl.Cohort(cal, f"{d}@all/task"), cl.Cohort(te, f"{d}@all/task"))}
        un_pairs = design.pairs(lambda F, d=d: f"{d}@{F}")
        ds, du = rx.leads_detail(seen_pairs, "P", rk), rx.leads_detail(un_pairs, "G_abs", rk)
        detail[d] = dict(seen=dict(seconds=rx.seconds(ds, types, spi), per_task=rx.per_task(ds)),
                         unseen=dict(seconds=rx.seconds(du, types, spi), per_task=rx.per_task(du)))
        if d == "hide_and_seek":
            limits = dict(P_seen=rx.timer_at_limit(ds), E_unseen=rx.timer_at_limit(du),
                          G_unseen=rx.timer_at_limit(rx.leads_detail(un_pairs, "G", rk)))
    sat = {f"{d}{tag}": rx.saturation(cal, f"{d}{tag}@all/task")
           for d in ("hide_and_seek_zero", "hide_and_seek") for tag in ("", "#0", "#1", "#2")}
    band_ok = all(b["seen"]["checks"]["ok"] and all(u["checks"]["ok"] for u in b["unseen"].values()) for b in bands.values())
    checks = dict(B0_point_equals_r17=b0, N0_band_timer_equals_hk_fit=band_ok)
    problems = [k for k, v in (("B0", b0["ok"]), ("N0", band_ok)) if not v]
    return dict(plan=rx.PLAN, checks=checks, native_bands=bands, bootstrap_unseen=boot, leads=detail,
                timers_at_limit=limits, zero_input_saturation=sat), problems


def print_r19(r):
    import r19_extras as rx
    rx.print_summary(r)


def print_r17(r):
    print("\nR17 timers alone on the test set (FPR, FP, lead, TWA)")
    for side, timers in (("unseen", R17_UNSEEN_TIMERS), ("seen", R17_SEEN_TIMERS)):
        for t in timers:
            v = r["timers_alone"][side][t]
            print(f"  {side:6s} {t:7s} {v['fpr']:.3f} {v['fp']:3d} {v['mean_lead']:.3f} {v['twa']:.3f}")
    print("R17 unseen tasks against the elapsed-time timer G_abs (single models: FPR, gain+-std, TWA+-std, "
          "matched gain+-std | three-seed ensemble: gain [matched]; G's gain for comparison)")
    for d, by in r["unseen"].items():
        m, e, g = by["G_abs"]["mean_std_over_seeds"], by["G_abs"]["seed_average"], by["G"]["seed_average"]
        mg = m.get("matched_gain", dict(mean=float("nan"), std=float("nan")))
        em = e["matched"]["gain"] if e["matched"] else float("nan")
        gm = g["matched"]["gain"] if g["matched"] else float("nan")
        print(f"  {d:26s} {m['fpr']['mean']:.3f} {m['gain']['mean']:+.3f}+-{m['gain']['std']:.3f} "
              f"{m['twa']['mean']:.3f}+-{m['twa']['std']:.3f} {mg['mean']:+.3f}+-{mg['std']:.3f} | "
              f"{e['gain']:+.3f} [{em:+.3f}] | G {g['gain']:+.3f} [{gm:+.3f}]")
    print("R17 seen tasks: single-model gain over P and over P without H_k (P_mult), mean+-std")
    for d, by in r["seen"].items():
        p, q = by["P"]["mean_std_over_seeds"]["gain"], by["P_mult"]["mean_std_over_seeds"]["gain"]
        print(f"  {d:26s} P {p['mean']:+.3f}+-{p['std']:.3f}  P_mult {q['mean']:+.3f}+-{q['std']:.3f}")


# ---- run -------------------------------------------------------------------------------------

def load_family(new_dirs, r7_dirs, parts_new, parts_rec, tasks, held, folds_sha):
    """One family: new fold directories attached to the E1 rows, R7's to the recorded rows; G1 checks."""
    wanted_new = {c: {str(r["rollout_id"]) for r in parts_new[c]} for c in ru.COHORTS}
    wanted_rec = {c: {str(r["rollout_id"]) for r in parts_rec[c]} for c in ru.COHORTS}
    by_fold, dets, g1 = {}, None, {}
    for path in r7_dirs:
        meta = json.loads((Path(path) / "scoring.json").read_text())
        by_fold.setdefault(str(meta["fold"]), {})["r7"] = Path(path)
    for path in new_dirs:
        meta = json.loads((Path(path) / "scoring.json").read_text())
        by_fold.setdefault(str(meta["fold"]), {})["new"] = Path(path)
    if set(by_fold) != set(held) or any(set(v) != {"r7", "new"} for v in by_fold.values()):
        raise SystemExit(f"STOP: a family needs one new and one R7 directory for each fold {sorted(held)}")
    for F, d in sorted(by_fold.items()):
        meta7, norms7, scores7 = ru.load_fold_dir(d["r7"], wanted_rec)
        meta, norms, scores = ru.load_fold_dir(d["new"], wanted_new)
        if dets is None:
            dets = list(meta7["detectors"])
        check_new_meta(F, meta, tasks, held, folds_sha, dets)
        if list(meta7["detectors"]) != dets:
            raise SystemExit(f"STOP: R7 fold {F} scored {meta7['detectors']}")
        ru.attach(parts_rec, F, norms7, scores7, held, dets)
        ru.attach(parts_new, F, norms, scores, held, dets)
        if "r7_fold_dir" in meta:          # a SAFE head: R7's training cohort and normalization, copied
            same_dir = Path(meta["r7_fold_dir"]).resolve() == d["r7"].resolve()
            hashes = all(sha256(d["r7"] / f) == v["r7"] == v["here"] == sha256(d["new"] / f)
                         for f, v in meta["copied_from_r7"].items())
            g1[F] = dict(kind="safe_copy", same_r7_dir=same_dir, hashes_equal=hashes, ok=same_dir and hashes)
        else:                              # cross-task detectors, retrained: R7's trajectories and normalizations
            worst = 0.0
            for mine, theirs in (("rec_calibration", "calibration"), ("rec_test", "test"), ("train", "train")):
                worst = max(worst, same_raw(raw_file(d["new"] / f"{mine}_predictions.jsonl"),
                                            raw_file(d["r7"] / f"{theirs}_predictions.jsonl")))
            # Normalizations follow from the training trajectories, so they get the same tolerance
            # (amended 2026-11-12 after the first run; the plan's G1 section says why).
            mine = json.loads((d["new"] / "normalizations.json").read_text())
            by_det = {k: tree_gap(mine.get(k), norms7.get(k)) for k in sorted(set(mine) | set(norms7))}
            nd = max(by_det.values())
            g1[F] = dict(kind="retrained", max_abs_diff=worst, normalizations_max_abs_diff=nd,
                         normalizations_by_detector={k: v for k, v in by_det.items() if v > 0},
                         ok=worst <= EXACT and nd <= EXACT)
        print(f"  fold {F} {'+'.join(dets)}: {'OK' if g1[F]['ok'] else 'MISMATCH'} {g1[F]}", flush=True)
    return dets, g1


R22_TOL_E1B = 1e-9


def r22_rldx(a, parts, parts_rec, held, folds, detectors, rk):
    """R22 on RLDX-1 (r22_fixed_paths): Table II with every path fixed on the recorded validation rollouts
    (R7's calibration successes and test rollouts), the unseen ensembles' bootstrap, and gates K0 (the adaptive
    choice reproduces R17), K3 (H3's frozen path reproduces E1b's registered result) and N0."""
    import r19_extras as rx
    import r22_fixed_paths as r22
    ref17 = json.loads(Path(a.r22_reference).read_text())["r17"]
    e1b = json.loads(Path(a.e1_result).read_text())
    labels = list(r22.LABELS)
    tag = lambda lab: "" if lab == "seed_average" else "#" + lab.split("_", 1)[1]
    val_rows = parts_rec["calibration"] + parts_rec["test"]
    checks = {}
    shared = {str(r["rollout_id"]) for r in val_rows} & {str(r["rollout_id"]) for c in SETS for r in parts[c]}
    checks["V0_disjoint"] = dict(shared_ids=len(shared), validation=len(val_rows), ok=not shared)

    def seen_pairs(d):
        return lambda lab: {ALL: (cl.Cohort(cal_rows(parts, "calibration"), f"{d}{tag(lab)}@all/task"),
                                  cl.Cohort(parts["test"], f"{d}{tag(lab)}@all/task"))}

    def seen_val(d):
        return lambda lab: {ALL: cl.Cohort(val_rows, f"{d}{tag(lab)}@all/task")}

    def unseen_pairs(d):
        def f(lab):
            out = {}
            for F in folds:
                cal = cal_rows(parts, "calibration", keep=lambda r, F=F: r["task_name"] not in held[F])
                test = [r for r in parts["test"] if r["task_name"] in held[F]]
                out[F] = (cl.Cohort(cal, f"{d}{tag(lab)}@{F}"), cl.Cohort(test, f"{d}{tag(lab)}@{F}"))
            return out
        return f

    def unseen_val(d):
        return lambda lab: {F: cl.Cohort([r for r in val_rows if r["task_name"] not in held[F]], f"{d}{tag(lab)}@{F}")
                            for F in folds}

    table2, k0, n0 = dict(seen={}, unseen={}), {}, {}
    for d in detectors + [SHUFFLE]:
        fixed = r22.block(seen_pairs(d), seen_val(d), labels, ("P",), rk)
        adaptive = r22.block(seen_pairs(d), None, labels, ("P",), rk)
        ref = (ref17["seen"].get(d) or {}).get("P")
        k0[f"seen/{d}"] = r22.k0(adaptive, {"P": ref}, ("P",), labels) if ref else dict(ok=True, note="no R17 row")
        n0[f"seen/{d}"] = all(fixed["P"][lab]["paths"][ALL]["timer"] == adaptive["P"][lab]["paths"][ALL]["timer"] for lab in labels)
        table2["seen"][d] = r22.strip(fixed)
        if d == SHUFFLE:
            continue
        fixed = r22.block(unseen_pairs(d), unseen_val(d), labels, ("G_abs", "G"), rk)
        adaptive = r22.block(unseen_pairs(d), None, labels, ("G_abs", "G"), rk)
        ref = {t: (ref17["unseen"].get(d) or {}).get(t) for t in ("G_abs", "G")}
        k0[f"unseen/{d}"] = r22.k0(adaptive, ref, ("G_abs", "G"), labels)
        n0[f"unseen/{d}"] = all(fixed[t][lab]["paths"][F]["timer"] == adaptive[t][lab]["paths"][F]["timer"]
                                for t in ("G_abs", "G") for lab in labels for F in fixed[t][lab]["paths"])
        table2["unseen"][d] = r22.strip(fixed)
    checks["K0_adaptive_reproduces_r17"] = dict(by=k0, ok=all(v["ok"] for v in k0.values()))
    checks["N0_timer_margins_unchanged"] = dict(by=n0, ok=all(n0.values()))
    # K3: H3's registered rule, frozen path from E1b, margin on the new calibration successes.
    hs = "hide_and_seek"
    fp_ = e1b["frozen_paths"][hs]
    h3 = r22.run_folds(seen_pairs(hs)("seed_average"), "P", rk, frozen={ALL: dict(u=float(fp_["u"]), s=float(fp_["s"]))})
    reg = e1b["results"][hs]
    d3 = max(abs(h3["gain"] - reg["gain"]), 0.0 if h3["fp"] == reg["test"]["fp"] else float("inf"))
    checks["K3_h3_reproduces_e1b"] = dict(fp=h3["fp"], e1b_fp=reg["test"]["fp"], gain=h3["gain"], e1b_gain=reg["gain"],
                                          max_abs_diff=d3, ok=d3 <= R22_TOL_E1B,
                                          matched_information=dict(r22=(h3["matched"] or {}).get("gain"),
                                                                   e1b=reg.get("matched_gain")))
    # Unseen ensembles: the full-pipeline bootstrap with the paths frozen.
    design = rx.UnseenDesign(parts["calibration"], parts["test"], held, folds)
    names = ["hide_and_seek", "safe_indep", "safe_lstm"]
    boot, _ = r22.boot_unseen(design, {n: (lambda F, n=n: f"{n}@{F}") for n in names},
                              lambda n: unseen_val(n)("seed_average"), ("G_abs", "G"), rk, B=a.r22_boot,
                              contrasts=(("hide_and_seek", "safe_indep"),))
    problems = [k.split("_", 1)[0] for k, v in checks.items() if not v["ok"]]
    return dict(plan=r22.PLAN, checks=checks, validation=dict(rows=len(val_rows)), table2=table2,
                h3=r22.strip(h3), unseen_bootstrap=r22.strip(boot),
                inputs=dict(r17=dict(path=str(Path(a.r22_reference).resolve()), sha256=sha256(a.r22_reference)),
                            e1b=dict(path=str(Path(a.e1_result).resolve()), sha256=sha256(a.e1_result)))), problems


R21_HEADS = ("safe_indep", "safe_lstm", "hide_and_seek")


def r21_rldx(a, parts, held, folds, tasks, rk):
    """R21 on RLDX-1 (r21_native_bands): each head's band on seen tasks against P and on unseen tasks against
    E (G_abs) and G, per seed and as an ensemble, with curves and the bootstrap. Gate N1 against R19."""
    import r21_native_bands as r21
    ref19 = json.loads(Path(a.r21_reference).read_text()) if a.r21_reference else None
    if ref19 is None or "native_bands" not in (ref19.get("r19") or {}):
        raise SystemExit("error: --r21 needs --r21-reference, R19's RLDX-1 output (rldx_e1b_r19.json)")
    cal, te = parts["calibration"], parts["test"]
    r0 = {}
    for d in R21_HEADS:
        for s in SEEDS:
            r0[f"{d}/seen/seed_{s}"] = r21.running_max_gap(cal + te, f"{d}~#{s}@all/task", f"{d}#{s}@all/task")
            for F in folds:
                r0[f"{d}/unseen/{F}/seed_{s}"] = r21.running_max_gap(cal + te, f"{d}~#{s}@{F}", f"{d}#{s}@{F}")
    every = set(tasks)
    sides = dict(seen=dict(design=r21.Design(cal, te, {ALL: every}, {ALL: every}), timers=("P",)),
                 unseen=dict(design=r21.Design(cal, te, {F: every - held[F] for F in folds}, {F: held[F] for F in folds}),
                             timers=("G_abs", "G")))

    def keys(d, side):
        at = (lambda F: "all/task") if side == "seen" else (lambda F: F)
        out = {r21.ENSEMBLE: lambda F, d=d, at=at: f"{d}~@{at(F)}"}
        for s in SEEDS:
            out[f"seed_{s}"] = lambda F, d=d, s=s, at=at: f"{d}~#{s}@{at(F)}"
        return out

    spec = {d: {side: keys(d, side) for side in sides} for d in R21_HEADS}
    res, problems = r21.run(sides, spec, rk, B=a.r21_boot, contrasts=(("hide_and_seek", "safe_indep"),), r0=r0,
                            progress=lambda m: print(m, flush=True))
    n1 = r21.check_r19(res, ref19["r19"]["native_bands"])
    res["checks"]["N1_ensemble_equals_r19"] = n1
    if not n1["ok"]:
        problems.append("N1")
    res["inputs"] = dict(r19=dict(path=str(Path(a.r21_reference).resolve()), sha256=sha256(a.r21_reference)))
    diag, diag_problems = None, []
    if a.r21b:
        if not a.r21b_reference:
            raise SystemExit("error: --r21b needs --r21b-reference, R21's RLDX-1 output (rldx_e1b_r21.json)")
        diag, diag_problems = r21.run_diagnostic(a, sides, spec, rk, a.output, a.r21b_reference)
    return res, problems, diag, diag_problems


def run(a):
    if a.r21:
        ru.KEEP_RAW_SEEDS = R21_HEADS           # R21's single-model bands read each seed's raw score
    if a.r19 or a.r21:
        ru.KEEP_RAW = True                      # native bands read the normalized raw score
    out_path = Path(a.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    parts_rec, H, tasks, bundle = ru.load_rows(a.safe_dir, a.inputs_dir)
    spec = json.loads(Path(a.folds).read_text())
    if spec.get("seed") != ru.FOLD_SEED or spec.get("k") != ru.K or spec.get("tasks") != bundle["task_types"]:
        raise SystemExit("STOP: the fold file is not R7's")
    fold_tasks, held = ctl.fold_sets(spec)
    if fold_tasks != set(tasks):
        raise SystemExit("STOP: the fold file's tasks differ from the RLDX-1 tasks")
    folds = sorted(F for F in held if F != ALL)
    folds_sha = sha256(a.folds)
    failures, checks = [], {}

    # G2a: H_k and rho_k are E1's.
    train = parts_rec["train"]
    rk = qt.rho(rr.dev_fractions(train, tasks), rr.Q)
    rk0 = {t: 0.0 for t in tasks}
    man = json.loads(Path(a.freeze_manifest).read_text())
    h_ok = {t: int(v["H_k"]) for t, v in man["tasks"].items()} == {t: int(H[t]) for t in tasks}
    r_gap = max(abs(rk[t] - float(man["tasks"][t]["rho_k"])) for t in tasks) if set(man["tasks"]) == set(tasks) else 1.0
    checks["G2a_Hk_rho_k"] = dict(H_equal=h_ok, rho_max_abs_diff=r_gap, ok=h_ok and r_gap <= EXACT)

    parts_new = {"train": [dict(r) for r in train]}     # own row dicts: G0 must not see the new scores
    ids = {n: json.loads((Path(a.ids_dir) / f"{n}_ids.json").read_text()) for n in SETS}
    for name in SETS:
        parts_new[name] = e1_rows(a.e1_scores_dir, name, H)
        if [r["rollout_id"] for r in parts_new[name]] != ids[name]:
            raise SystemExit(f"STOP: E1 {name} rows differ from the ID list")
    if len(a.fold_dirs) != len(a.r7_fold_dirs):
        raise SystemExit("error: give one --r7-fold-dirs per --fold-dirs, in the same order")

    print("G1: models (new directories against R7's)")
    detectors, g1 = [], {}
    for new_dirs, r7_dirs in zip(a.fold_dirs, a.r7_fold_dirs):
        dets, g = load_family(new_dirs, r7_dirs, parts_new, parts_rec, tasks, held, folds_sha)
        if set(dets) & set(detectors):
            raise SystemExit(f"STOP: two families score {sorted(set(dets) & set(detectors))}")
        detectors += dets
        g1["+".join(dets)] = g
    missing = [d for d in ru.DETECTORS + ("safe_lstm", "safe_indep") if d not in detectors]
    if missing:
        raise SystemExit(f"STOP: no family scores {missing}")
    detectors = list(ru.DETECTORS) + [d for d in detectors if d not in ru.DETECTORS]

    sd = Path(a.shuffle_dir)
    smeta = json.loads((sd / "scoring.json").read_text())
    ref_ok = {k: v.get("ok") for k, v in smeta.get("reference_checks", {}).items()}
    attach_shuffle(parts_new, sd, {c: f"{c}_predictions.jsonl" for c in ru.COHORTS})
    attach_shuffle(parts_rec, sd, {"calibration": "rec_calibration_predictions.jsonl", "test": "rec_test_predictions.jsonl",
                                   "train": "train_predictions.jsonl"})
    hs_train = raw_file(Path(a.rldxhs_scores_dir) / "train_predictions.jsonl").get(SHUFFLE, {})
    mine_train = raw_file(sd / "train_predictions.jsonl").get(SHUFFLE, {})
    train_gap = same_raw({SHUFFLE: mine_train}, {SHUFFLE: hs_train})
    need = {f"rec_calibration/{SHUFFLE}", f"rec_test/{SHUFFLE}"}
    g1["shuffle"] = dict(reference_checks=ref_ok, train_max_abs_diff=train_gap,
                         ok=need <= set(ref_ok) and all(ref_ok.values()) and train_gap <= EXACT)
    print(f"  shuffle: {'OK' if g1['shuffle']['ok'] else 'MISMATCH'} {g1['shuffle']}")
    checks["G1_models"] = dict(families=g1, ok=all(v["ok"] for k, fam in g1.items()
                                                   for v in ([fam] if k == "shuffle" else fam.values())))

    # G0: the code path reproduces R7's published validation rows with R7's protocol.
    print("G0: recorded rows with R7's protocol")
    r7 = json.loads(Path(a.r7_table).read_text())["table2"]
    rec = table(parts_rec, held, folds, detectors, rk, rk0, "train")
    diff = [d for d in detectors if strip(rec[d]) != strip({k: r7[d][k] for k in ("seen", "unseen")})]
    ps = json.loads(Path(a.reference_per_seed).read_text())["results"][SHUFFLE]
    sh = table(parts_rec, held, folds, [SHUFFLE], rk, rk0, "train", seen_only=(SHUFFLE,))[SHUFFLE]["seen"]
    sh_ok = all((sh[lab]["mean_lead"], sh[lab]["fpr"], sh[lab]["gain"]) == (want["lead"], want["fpr"], want["gain"])
                for lab, want in [("seed_average", ps["seed_average"])] + [(f"seed_{s}", ps["per_seed"][s]) for s in SEEDS])
    g_ok = strip(r7["global_timer"]["unseen"]) == strip(rec[detectors[0]]["unseen"]["seed_average"]["timer"])
    checks["G0_code_path"] = dict(rows_differing=diff, shuffle_ok=sh_ok, G_unseen_ok=g_ok,
                                  ok=not diff and sh_ok and g_ok)
    print(f"  {len(detectors) - len(diff)} of {len(detectors)} rows equal rldx_unseen_safe.json; shuffle "
          f"{'OK' if sh_ok else 'MISMATCH'}; G {'OK' if g_ok else 'MISMATCH'}")

    # G2b-c: E1's P and trajectories.
    e1 = json.loads(Path(a.e1_result).read_text())
    P_new = timer_seen(parts_new, f"{detectors[0]}@all/task", rk, "calibration")
    p_ok = (P_new["fpr"], P_new["mean_lead"]) == (e1["P"]["test"]["fpr"], e1["P"]["test"]["mean_lead"])
    traj = {}
    for d in ("safe_lstm", "hide_and_seek"):
        traj[d] = max(gap(r[f"{d}@all/task"], r[f"e1_{d}"]) for c in SETS for r in parts_new[c])
    checks["G2b_P_equals_E1"] = dict(now=(P_new["fpr"], P_new["mean_lead"]),
                                     e1=(e1["P"]["test"]["fpr"], e1["P"]["test"]["mean_lead"]), ok=p_ok)
    checks["G2c_trajectories_equal_E1"] = dict(max_abs_diff=traj, ok=all(v <= 1e-9 for v in traj.values()))
    for k in ("G2a_Hk_rho_k", "G2b_P_equals_E1", "G2c_trajectories_equal_E1"):
        print(f"{k}: {'OK' if checks[k]['ok'] else 'MISMATCH'} {checks[k]}")
    failures = [k for k, v in checks.items() if not v["ok"]]

    result = dict(plan=PLAN, status="exploratory (Table II on RLDX-1's test set)", checks=checks, failures=failures,
                  counts={n: dict(rollouts=len(parts_new[n]), successes=sum(not r["failed"] for r in parts_new[n]),
                                  failures=sum(r["failed"] for r in parts_new[n])) for n in SETS},
                  inputs=dict(folds=dict(path=str(Path(a.folds).resolve()), sha256=folds_sha),
                              e1_result=dict(path=str(Path(a.e1_result).resolve()), sha256=sha256(a.e1_result)),
                              r7_table=dict(path=str(Path(a.r7_table).resolve()), sha256=sha256(a.r7_table)),
                              ids={n: sha256(Path(a.ids_dir) / f"{n}_ids.json") for n in SETS}))
    if failures:
        print("GATES FAILED: " + ", ".join(failures) + ". No table is reported.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_path, result)
        return 1

    result["table2"] = {}
    for mode in MODES:
        key0 = f"{detectors[0]}@all/task"
        P = timer_seen(parts_new, key0, rk, mode)
        G = timer_seen(parts_new, key0, rk0, mode, lead_P=P["mean_lead"])
        rows = table(parts_new, held, folds, detectors + [SHUFFLE], rk, rk0, mode, seen_only=(SHUFFLE,))
        G_unseen = rows[detectors[0]]["unseen"]["seed_average"]["timer"]
        result["table2"][mode] = dict(timer_P=dict(seen=P), timer_G=dict(seen=G, unseen=G_unseen), **rows)
        print(f"\nTable II, RLDX-1 test set (path chosen on {mode} failures): FPR, gain, TWA (mean+-std over seeds)")
        print(f"  Timer G   seen FPR {G['fpr']:.3f} gain {G['gain_over_P']:+.3f} TWA {G['twa']:.3f} | "
              f"unseen FPR {G_unseen['fpr']:.3f} TWA {G_unseen['twa']:.3f}")
        print(f"  Timer P   seen FPR {P['fpr']:.3f} TWA {P['twa']:.3f} | lead P {P['mean_lead']:.3f}, "
              f"G {G['mean_lead']:.3f}/{G_unseen['mean_lead']:.3f} (seen/unseen)")
        for d, row in rows.items():
            cell = lambda m: f"{m['fpr']['mean']:.3f} {m['gain']['mean']:+.3f}+-{m['gain']['std']:.3f} " \
                             f"{m['twa']['mean']:.3f}+-{m['twa']['std']:.3f}"
            u = cell(row["unseen"]["mean_std_over_seeds"]) if "unseen" in row else "--"
            print(f"  {d:26s} {cell(row['seen']['mean_std_over_seeds'])} | {u}")
    print("\nRLDX-1 TEST TABLE OK")

    status = 0
    if a.reference_table:
        ref = json.loads(Path(a.reference_table).read_text())
        same = json.dumps(strip(ref.get("table2")), sort_keys=True) == json.dumps(strip(result["table2"]), sort_keys=True)
        result["checks"]["R17_K0_table_equals_reference"] = dict(
            reference=str(Path(a.reference_table).resolve()), sha256=sha256(a.reference_table), ok=same)
        print(f"R17 K0: Table II {'equals' if same else 'DIFFERS FROM'} {a.reference_table}")
        if not same:
            result["r17_failures"] = ["K0: Table II differs from the reference table"]
            status = 1
    if a.elapsed_time and status == 0:
        r17, problems = elapsed_time_block(parts_new, held, folds, detectors + [SHUFFLE], rk,
                                           result["table2"]["calibration"])
        result["r17"], result["r17_failures"] = r17, problems
        if problems:
            print("R17 FAILED (no R17 result is used):\n  " + "\n  ".join(problems))
            status = 1
        else:
            print("R17 K1: G reproduces every unseen + G row; K5: P reproduces every seen + P row")
            print_r17(r17)
            print("\nR17 ELAPSED-TIME OK")
    elif a.elapsed_time:
        print("R17 FAILED: no R17 rows without K0")
    if a.r19 and status == 0:
        reference = json.loads(Path(a.r19_reference).read_text()) if a.r19_reference else None
        if reference is None or "r17" not in reference:
            raise SystemExit("error: --r19 needs --r19-reference, R17's output (it holds the r17 block)")
        r19, problems = r19_rldx(parts_new, held, folds, rk, bundle["task_types"], reference, a.r19_boot)
        result["r19"], result["r19_failures"] = r19, problems
        if problems:
            print("R19 FAILED (no R19 result is used): " + ", ".join(problems) + f" {r19['checks']}")
            status = 1
        else:
            print(f"R19 B0: point estimates equal R17's (max |diff| {r19['checks']['B0_point_equals_r17']['max_abs_diff']}); "
                  "N0: band timers equal hk_free.fit's")
            print_r19(r19)
            print("\nR19 OK")
    if a.r20 and status == 0:
        import r20_timer_g as r20m
        reference = json.loads(Path(a.r20_reference).read_text()) if a.r20_reference else None
        if reference is None or "timers_alone" not in (reference.get("r17") or {}):
            raise SystemExit("error: --r20 needs --r20-reference, R17's output (it holds r17.timers_alone)")
        rounded = (((reference.get("table2") or {}).get("calibration") or {}).get("timer_G") or {}).get("seen")
        r20, problems = r20m.run(parts_new["calibration"], parts_new["test"], held, folds, rk,
                                 reference["r17"]["timers_alone"], rounded)
        result["r20"], result["r20_failures"] = r20, problems
        print()
        r20m.print_summary(r20, "RLDX-1")
        if problems:
            print("R20 FAILED (no R20 result is used): " + ", ".join(problems))
            status = 1
        else:
            print("\nR20 OK")
    elif a.r20:
        print("R20 FAILED: no R20 rows after a failed gate")
    if a.r21b and not a.r21:
        raise SystemExit("error: --r21b runs after R21: give --r21 and its reference too")
    if a.r21 and status == 0:
        import r21_native_bands as r21m
        r21, problems, r21b, problems_b = r21_rldx(a, parts_new, held, folds, tasks, rk)
        result["r21"], result["r21_failures"] = r21, problems
        print()
        r21m.print_summary(r21, "RLDX-1")
        if problems:
            print("R21 FAILED (no R21 result is used): " + ", ".join(problems))
            status = 1
        else:
            print("\nR21 OK")
        if a.r21b:
            result["r21b"], result["r21b_failures"] = r21b, problems_b
            print()
            r21m.print_diagnostic(r21b, "RLDX-1")
            if problems_b or problems:
                print("R21b FAILED (no R21b result is used): " + ", ".join(problems_b + problems))
                status = 1
            else:
                print("\nR21b OK")
    elif a.r21:
        print("R21 FAILED: no R21 result after a failed gate")
    if a.r22 and status == 0:
        import r22_fixed_paths as r22m
        r22, problems = r22_rldx(a, parts_new, parts_rec, held, folds, detectors, rk)
        result["r22"], result["r22_failures"] = r22, problems
        print()
        r22m.print_summary(r22, "RLDX-1")
        if problems:
            print("R22 FAILED (no R22 result is used): " + ", ".join(problems))
            status = 1
        else:
            print("\nR22 OK")
    elif a.r22:
        print("R22 FAILED: no R22 result after a failed gate")
    if a.curves and status == 0:
        import seen_unseen_tables as sut
        result["r18"] = rldx_curves(parts_new, held, folds, detectors + [SHUFFLE], rk)
        sut.print_r18(result["r18"])
        print("\nR18 CURVES OK")
    elif a.curves:
        print("R18 FAILED: no curves after a failed gate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, result)
    return status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ids")
    i.add_argument("--inputs-dir", required=True)
    i.add_argument("--e1-scores-dir", required=True)
    i.add_argument("--output-dir", required=True)
    f = sub.add_parser("safe-fold")
    f.add_argument("--r7-fold-dir", required=True)
    f.add_argument("--folds", required=True)
    f.add_argument("--cohort", nargs=2, action="append", required=True, metavar=("NAME", "IDS"))
    f.add_argument("--scores", nargs=3, action="append", required=True, metavar=("SET", "SEED", "DIR"))
    f.add_argument("--e1-reference", nargs=2, action="append", metavar=("SET", "PREDICTIONS"))
    f.add_argument("--output-dir", required=True)
    r = sub.add_parser("run")
    for name in ("--safe-dir", "--inputs-dir", "--folds", "--ids-dir", "--e1-scores-dir", "--e1-result",
                 "--freeze-manifest", "--shuffle-dir", "--rldxhs-scores-dir", "--r7-table", "--reference-per-seed",
                 "--output"):
        r.add_argument(name, required=True)
    r.add_argument("--fold-dirs", nargs="+", action="append", required=True)
    r.add_argument("--r7-fold-dirs", nargs="+", action="append", required=True)
    r.add_argument("--elapsed-time", action="store_true", help="R17: also fit every row against R14's timers")
    r.add_argument("--curves", action="store_true", help="R18: calibrated lead-FPR curves (lead_fpr_curves.py)")
    r.add_argument("--r19", action="store_true", help="R19: the mock review's offline analyses (r19_extras.py)")
    r.add_argument("--r19-reference", help="R19 gate B0: R17's output (rldx_e1b_r17.json)")
    r.add_argument("--r19-boot", type=int, default=1000, help="R19 bootstrap replicates")
    r.add_argument("--r20", action="store_true", help="R20: timer G as a Table II row (r20_timer_g.py)")
    r.add_argument("--r20-reference", help="R20 gate T0: R17's output (rldx_e1b_r17.json)")
    r.add_argument("--r21", action="store_true", help="R21: the original alarm rules (r21_native_bands.py)")
    r.add_argument("--r21-reference", help="R21 gate N1: R19's output (rldx_e1b_r19.json)")
    r.add_argument("--r21-boot", type=int, default=1000, help="R21 bootstrap replicates")
    r.add_argument("--r21b", action="store_true", help="R21b: the band CI diagnostic, after R21")
    r.add_argument("--r21b-reference", help="R21b gate D0: R21's output (rldx_e1b_r21.json)")
    r.add_argument("--r21b-boot", type=int, default=1000, help="R21b replicates per variant")
    r.add_argument("--r22", action="store_true", help="R22: every fused path fixed on the validation set")
    r.add_argument("--r22-reference", help="R22 gate K0: R17's output (rldx_e1b_r17.json)")
    r.add_argument("--r22-boot", type=int, default=1000, help="R22 unseen bootstrap replicates")
    r.add_argument("--reference-table", help="R17 K0: an earlier run's output whose table2 this run must equal")
    a = p.parse_args(argv)
    return {"ids": ids_cmd, "safe-fold": safe_fold_cmd, "run": run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
