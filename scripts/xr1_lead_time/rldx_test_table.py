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
        normalizations; every SAFE fold directory copies R7's training cohort and normalization
        byte for byte; the shuffle directory reproduces the earlier trajectories;
    G2  E1: H_k and rho_k equal the freeze manifest's; P alone on the test set equals E1's P; the
        seed-averaged SAFE-LSTM and Hide-and-Seek trajectories on E1's sets equal E1's.

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
            norm_same = json.loads((d["new"] / "normalizations.json").read_text()) == norms7
            g1[F] = dict(kind="retrained", max_abs_diff=worst, normalizations_equal=norm_same,
                         ok=worst <= EXACT and norm_same)
        print(f"  fold {F} {'+'.join(dets)}: {'OK' if g1[F]['ok'] else 'MISMATCH'} {g1[F]}", flush=True)
    return dets, g1


def run(a):
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, result)
    print("\nRLDX-1 TEST TABLE OK")
    return 0


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
    a = p.parse_args(argv)
    return {"ids": ids_cmd, "safe-fold": safe_fold_cmd, "run": run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
