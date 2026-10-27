"""Cross-task (unseen-task) lead analysis, pre-specified in docs/xr1_cross_task_unseen_plan.md.

Exploratory: both E1 cohorts were opened by earlier analyses. The rules in the plan were
committed before any detector was trained on a subset of the XR-1 tasks.

Folds. The 38 tasks are split into 4 folds stratified by task type (cross_task_folds.json).
For fold f, robocasa.recovery.safe.baselines.score_cross_task trains every detector without
fold f's tasks. Here, for each fold, the timers and the fused path are fitted on the
lead-calibration rollouts of the SEEN tasks only and applied to the confirmatory-test rollouts
of the HELD-OUT tasks (Hide-and-Seek's unseen protocol also calibrates on seen tasks). Every
task is held out once, so the pooled unseen-task test set is the whole primary test cohort
(380/380). Fold "all" (no task held out) gives the seen reference.

Timers, each with Proposition 1's conformal margin on calibration successes (alpha = 0.05):
  G    global timer: P with rho_k = 0 for every task, i.e. one fraction of H_k for all tasks
  P_n  rho_k = 0.95 quantile of n development successes of task k drawn without replacement,
       for seen and held-out tasks alike (n = 1, 3, 5)
  P    rho_k from all development successes of task k (the registered P)
Monitor scores use the fold's pooled normalization (one location and scale over all seen-task
training maxima): a zero-shot monitor has no statistics for a task it has never seen.

Prerequisite: fold "all" with per-task normalization must reproduce the Amendment 2 result
(P, Hide-and-Seek + P and its two controls + P) exactly. Otherwise nothing else is computed.

Usage:
  python3 -I cross_task_lead.py folds --predictions E1_OUT/calibration_predictions.jsonl [--write FILE]
  python3 -I cross_task_lead.py run --calibration E1_OUT/calibration_predictions.jsonl \\
      --test E1_OUT/test_predictions.jsonl --fold-dirs F0 F1 F2 F3 FALL \\
      --runtime-bundle runtime_bundle.json --amend2-result AMEND2_ROOT/confirm_hide_and_seek.json \\
      --amend2-normalizations AMEND2_ROOT/scores/normalizations.json --output OUT.json
"""
import argparse, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl
import lead_time
import quantile_timer as qt

PLAN = "docs/xr1_cross_task_unseen_plan.md"
FOLDS_FILE = here / "cross_task_folds.json"
FOLD_SEED, K = 20261027, 4
TYPES = ("atomic", "composite")
PRIMARY, ZERO, FIRST, KNN = "hide_and_seek", "hide_and_seek_zero", "hide_and_seek_first", "cosine_knn"
DETECTORS = (PRIMARY, ZERO, FIRST, KNN)
AMEND2_DETECTORS = (PRIMARY, ZERO, FIRST)
NS = (1, 3, 5)
FEW_DRAWS, FEW_SEED = 20, 2
CAL_COHORT, TEST_COHORT = "lead_calibration", "confirmatory_test"
ALL = "all"
LAM_GRID = np.arange(0.0, 1.0001, 0.0025)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- folds -----------------------------------------------------------------------------------

def make_folds(task_types, k=K, seed=FOLD_SEED):
    """Stratified folds: within each task type, tasks sorted by name, permuted, dealt in turn."""
    other = sorted(t for t, v in task_types.items() if v not in TYPES)
    if other:
        raise ValueError(f"Tasks without an atomic/composite type: {other}")
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for tt in TYPES:
        for i, t in enumerate(rng.permutation(sorted(t for t, v in task_types.items() if v == tt))):
            folds[i % k].append(str(t))
    return dict(plan=PLAN, seed=seed, k=k,
                rule="per task type (atomic, then composite): sort names, numpy default_rng(seed).permutation, "
                     "deal round-robin into folds 0..k-1",
                tasks=dict(sorted(task_types.items())),
                folds=[dict(fold=i, held_out=sorted(f)) for i, f in enumerate(folds)])


def task_types_from(path):
    types = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            t, tt = r["task_name"], r.get("task_type")
            if types.setdefault(t, tt) != tt:
                raise ValueError(f"Task {t} has two task types")
    return types


def fold_sets(spec):
    tasks = set(spec["tasks"])
    listed = [t for f in spec["folds"] for t in f["held_out"]]
    if sorted(listed) != sorted(tasks) or len(listed) != len(set(listed)):
        raise ValueError("Folds do not hold out every task exactly once")
    held = {str(f["fold"]): set(f["held_out"]) for f in spec["folds"]}
    held[ALL] = set()
    return tasks, held


def folds_cmd(args):
    spec = make_folds(task_types_from(args.predictions))
    if args.write:
        Path(args.write).write_text(json.dumps(spec, indent=1) + "\n")
        print(f"wrote {args.write}: " + ", ".join(f"fold {f['fold']} {len(f['held_out'])} tasks" for f in spec["folds"]))
        return 0
    committed = json.loads(Path(args.folds).read_text())
    if committed != json.loads(json.dumps(spec)):
        print("FOLDS DIFFER from the committed assignment; stop.")
        return 1
    for f in spec["folds"]:
        types = [spec["tasks"][t] for t in f["held_out"]]
        print(f"fold {f['fold']}: {types.count('atomic')} atomic + {types.count('composite')} composite")
    print("FOLDS OK")
    return 0


# ---- fold scores -----------------------------------------------------------------------------

def load_fold_dir(path, wanted):
    """scoring.json, normalizations.json and the per-seed raw trajectories of wanted rollouts."""
    path = Path(path)
    meta = json.loads((path / "scoring.json").read_text())
    norms = json.loads((path / "normalizations.json").read_text())
    scores = {}
    for cohort in (CAL_COHORT, TEST_COHORT):
        out = defaultdict(dict)
        with open(path / f"{cohort}_predictions.jsonl") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = str(row["rollout_id"])
                if rid not in wanted[cohort]:
                    continue
                if rid in out[row["detector"]]:
                    raise ValueError(f"{path}: duplicate {row['detector']} row for {rid}")
                out[row["detector"]][rid] = dict(task_name=row["task_name"], failed=bool(row["failed"]),
                                                 held_out=bool(row["held_out"]),
                                                 raw={s: np.asarray(v, float) for s, v in row["per_seed_raw"].items()})
        scores[cohort] = dict(out)
    return meta, norms, scores


def normalized(raw, stats):
    """Seed-averaged normalized running maximum, as run_baselines.build_records computes it."""
    return np.mean([(np.maximum.accumulate(raw[s]) - stats[s]["location"]) / stats[s]["scale"] for s in sorted(raw)],
                   axis=0)


def check_meta(F, meta, tasks, held, folds_sha, first_meta):
    if str(meta["fold"]) != F:
        raise ValueError(f"Fold directory for {F} says fold {meta['fold']}")
    if set(meta["held_out_tasks"]) != held[F]:
        raise ValueError(f"Fold {F}: held-out tasks differ from the fold file")
    trained = set(meta["training_rollouts_by_task"])
    if trained & held[F] or trained != tasks - held[F]:
        raise ValueError(f"Fold {F}: training tasks are not exactly the seen tasks")
    if meta["folds_sha256"] != folds_sha:
        raise ValueError(f"Fold {F}: scored with a different fold file")
    for key in ("seeds", "epochs", "feature_aggregation", "detectors", "train_ids_sha256"):
        if meta[key] != first_meta[key]:
            raise ValueError(f"Fold {F}: {key} differs from the other folds")
    missing = [d for d in DETECTORS if d not in meta["detectors"]]
    if missing:
        raise ValueError(f"Fold {F}: detectors {missing} were not scored")


def check_reference(meta):
    """Fold all must have reproduced the Amendment 2 raw trajectories (score_cross_task --reference-raw)."""
    checks = meta.get("reference_checks") or {}
    need = {f"{c}/{d}" for c in (CAL_COHORT, TEST_COHORT) for d in AMEND2_DETECTORS}
    bad = sorted(k for k in need if not checks.get(k, {}).get("ok"))
    if bad:
        raise ValueError(f"Fold all did not reproduce the Amendment 2 raw trajectories for {bad}")
    return {k: checks[k]["max_abs_diff"] for k in sorted(need)}


def attach(rows, folds_data, cohort, held):
    for r in rows:
        r["raw"] = {}
    for F, (_, norms, scores) in folds_data.items():
        for d in DETECTORS:
            if d not in scores[cohort] or d not in norms:
                raise ValueError(f"Fold {F}: no {d} scores for {cohort}")
            seeds = sorted(norms[d])
            pooled = {s: norms[d][s]["pooled"] for s in seeds}
            for r in rows:
                rid = str(r["rollout_id"])
                row = scores[cohort][d].get(rid)
                if row is None:
                    raise ValueError(f"Fold {F}: no {d} trajectory for {rid} ({cohort})")
                if row["failed"] != r["failed"] or row["task_name"] != r["task_name"]:
                    raise ValueError(f"Fold {F}: label or task differs for {rid} ({d})")
                if row["held_out"] != (r["task_name"] in held[F]):
                    raise ValueError(f"Fold {F}: held-out flag wrong for {rid}")
                if sorted(row["raw"]) != seeds:
                    raise ValueError(f"Fold {F}: seeds of {rid} differ from the normalization")
                if any(len(v) != r["L"] or not np.all(np.isfinite(v)) for v in row["raw"].values()):
                    raise ValueError(f"Fold {F}: {d} trajectory of {rid} has the wrong length or non-finite values")
                r["raw"][(F, d)] = row["raw"]
                r[f"{d}@{F}"] = normalized(row["raw"], pooled)
                if F == ALL:
                    r[f"{d}@all/task"] = normalized(row["raw"], {s: norms[d][s]["per_task"][r["task_name"]]
                                                                 for s in seeds})


# ---- pooled fold evaluation ------------------------------------------------------------------

class Design:
    """Both cohorts as arrays, per score key, with each fold's seen and held-out indices."""

    def __init__(self, cal, te, held, keys, dev):
        self.cal, self.te, self.held = cal, te, held
        self.folds = sorted(F for F in held if F != ALL)
        self.tasks = sorted({r["task_name"] for r in te})
        self.C = {}
        self.add(keys)
        ct = np.array([r["task_name"] for r in cal]); tt = np.array([r["task_name"] for r in te])
        self.cal_idx = {F: np.flatnonzero(~np.isin(ct, sorted(held[F]))) for F in held}
        self.te_idx = {F: np.flatnonzero(np.isin(tt, sorted(held[F]))) for F in self.folds}
        self.te_idx[ALL] = np.arange(len(te))
        base = self.C[keys[0]]
        self.sc, self.st = cl.strata(base[0]), cl.strata(base[1])
        self.dev = dev
        self.rk0 = {t: 0.0 for t in dev}
        self.rkP = qt.rho(dev, cl.PRIMARY_Q)

    def add(self, keys):
        for k in keys:
            self.C[k] = (cl.Cohort(self.cal, k), cl.Cohort(self.te, k))

    def pooled_failures(self, te_sel=None):
        """Test failures in pooled order (folds in order, held-out rows only)."""
        idx = np.concatenate([(self.te_idx if te_sel is None else te_sel)[F] for F in self.folds])
        return [self.te[i] for i in idx if self.te[i]["failed"]]


def raw_eval(e, rk, det):
    a = cl.alarm_vec(e, cl.rho_rows(e, rk), det["lam"], det["u"], det["s"])
    hit, F = a <= e.L, e.failed
    return dict(fp=int(hit[~F].sum()), n=int((~F).sum()), lead=np.where(hit, (e.L - a) / e.H, 0.0)[F],
                hit=hit[F], frac=np.where(hit, a / e.H, 1.0)[F], final=int(np.sum(a[F] == e.L[F])))


def pool(parts):
    cat = lambda k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)
    return dict(fp=sum(p["fp"] for p in parts), n=sum(p["n"] for p in parts), lead=cat("lead"),
                hit=cat("hit").astype(bool), frac=cat("frac"), final=sum(p["final"] for p in parts))


def summary(p):
    return dict(fpr=round(p["fp"] / p["n"], 4), fp=p["fp"], fpr_wilson95=cl.wilson(p["fp"], p["n"]),
                recall=round(float(p["hit"].mean()), 4), mean_lead=round(float(p["lead"].mean()), 4),
                by50=round(float(np.mean(p["frac"] <= 0.5)), 4), by75=round(float(np.mean(p["frac"] <= 0.75)), 4),
                alarms_at_final_inference=p["final"])


def unseen(D, key_of, rk, cal_sel=None, te_sel=None, C=None):
    """Per fold: fit on seen-task calibration rows, evaluate on held-out test rows, pool."""
    C = D.C if C is None else C
    pt, pf, per_fold = [], [], {}
    for F in D.folds:
        ti = D.te_idx[F] if te_sel is None else te_sel[F]
        if len(ti) == 0:
            continue
        ci = D.cal_idx[F] if cal_sel is None else cal_sel[F]
        key = key_of(F)
        c, e = C[key][0].take(ci), C[key][1].take(ti)
        tm, fu = cl.fit_fast(c, rk)
        a, b = raw_eval(e, rk, tm), raw_eval(e, rk, fu)
        pt.append(a); pf.append(b); per_fold[F] = (tm, fu, a, b)
    return pool(pt), pool(pf), per_fold


def seen(D, key, rk, cal_sel=None, te_sel=None):
    """Fold all: every task in calibration and test."""
    ci = D.cal_idx[ALL] if cal_sel is None else cal_sel[ALL]
    ti = D.te_idx[ALL] if te_sel is None else te_sel[ALL]
    c, e = D.C[key][0].take(ci), D.C[key][1].take(ti)
    tm, fu = cl.fit_fast(c, rk)
    return raw_eval(e, rk, tm), raw_eval(e, rk, fu), (tm, fu)


def rho_n(dev, n, rng):
    """rho_k from n development successes of each task, drawn without replacement."""
    out = {}
    for t in sorted(dev):
        v = dev[t]
        out[t] = float(np.quantile(rng.choice(v, min(n, len(v)), replace=False), cl.PRIMARY_Q)) if v else 1.0
    return out


def gain(f, t):
    return float(np.mean(f["lead"] - t["lead"]))


# ---- one bootstrap replicate -----------------------------------------------------------------

def draw(D, rng):
    """Tasks drawn once for both cohorts and all folds; rollouts within task and class."""
    occ = []
    for t in rng.choice(D.tasks, len(D.tasks)):
        ci = np.concatenate([rng.choice(D.sc[t][k], len(D.sc[t][k])) for k in (0, 1)])
        ti = np.concatenate([rng.choice(D.st[t][k], len(D.st[t][k])) for k in (0, 1)])
        occ.append((t, ci, ti))
    empty = np.zeros(0, int)
    cal_sel = {F: np.concatenate([c for t, c, _ in occ if t not in D.held[F]] or [empty]) for F in D.held}
    te_sel = {F: np.concatenate([x for t, _, x in occ if t in D.held[F]] or [empty]) for F in D.folds}
    te_sel[ALL] = np.concatenate([x for _, _, x in occ])
    return cal_sel, te_sel


def statistics(D, cal_sel=None, te_sel=None, rk_few=None):
    """Every quantity that gets a decision-bootstrap CI, on one (re)sample."""
    out, leads = {}, {}
    for d in DETECTORS:
        t, f, _ = unseen(D, lambda F, d=d: f"{d}@{F}", D.rk0, cal_sel, te_sel)
        leads[d] = (t, f)
        out[f"zero_shot_gain/{d}"] = gain(f, t)
    lF = {d: leads[d][1]["lead"] for d in DETECTORS}
    out[f"contrast/{PRIMARY}_minus_{ZERO}"] = float(np.mean(lF[PRIMARY] - lF[ZERO]))
    out[f"contrast/{PRIMARY}_minus_{FIRST}"] = float(np.mean(lF[PRIMARY] - lF[FIRST]))
    out[f"contrast/{PRIMARY}_minus_{KNN}"] = float(np.mean(lF[PRIMARY] - lF[KNN]))
    tG = leads[PRIMARY][0]
    st, sf, _ = seen(D, f"{PRIMARY}@all", D.rk0, cal_sel, te_sel)
    out["seen_gain_over_G"] = gain(sf, st)
    out["transfer_loss"] = out[f"zero_shot_gain/{PRIMARY}"] - out["seen_gain_over_G"]
    st, sf, _ = seen(D, f"{PRIMARY}@all/task", D.rkP, cal_sel, te_sel)
    out["seen_per_task_gain_over_P"] = gain(sf, st)
    rks = dict(rk_few or {}); rks["all"] = D.rkP
    for n, rk in rks.items():
        t, f, _ = unseen(D, lambda F: f"{PRIMARY}@{F}", rk, cal_sel, te_sel)
        out[f"few_shot_gain/{n}"] = gain(f, t)
        out[f"timer_catch_up/{n}"] = float(np.mean(t["lead"] - tG["lead"]))
        out[f"fused_vs_zero_shot/{n}"] = float(np.mean(f["lead"] - leads[PRIMARY][1]["lead"]))
    out["few_shot_all_minus_seen_per_task"] = out["few_shot_gain/all"] - out["seen_per_task_gain_over_P"]
    return out


def bootstrap(D, B, seed):
    rng = np.random.default_rng(seed)
    reps = defaultdict(list)
    for b in range(B):
        cal_sel, te_sel = draw(D, rng)
        few = {str(n): rho_n(D.dev, n, rng) for n in NS}
        for k, v in statistics(D, cal_sel, te_sel, few).items():
            reps[k].append(v)
        if (b + 1) % 200 == 0:
            print(f"bootstrap {b + 1}/{B}", flush=True)
    return {k: dict(ci95=np.round(np.percentile(v, [2.5, 97.5]), 4).tolist(), sd=round(float(np.std(v)), 4))
            for k, v in reps.items()}


# ---- nulls -----------------------------------------------------------------------------------

def resample(rows, rng, D, F, norms):
    """Within-task resampled fold-F Hide-and-Seek, pooled-normalized (as confirm_hide_and_seek)."""
    seeds = sorted(norms)
    pools = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for s in seeds:
            pools[r["task_name"]][s].append(r["raw"][(F, PRIMARY)][s])
    pools = {t: {s: np.concatenate(v) for s, v in d.items()} for t, d in pools.items()}
    for r in rows:
        raw = {s: rng.choice(pools[r["task_name"]][s], size=r["L"], replace=True) for s in seeds}
        r[f"{PRIMARY}_resampled@{F}"] = normalized(raw, {s: norms[s]["pooled"] for s in seeds})


def nulls(D, fold_norms, draws, tG):
    rng = np.random.default_rng(cl.NULL_SEED)
    res = {"noise": [], f"{PRIMARY}_resampled": []}
    for _ in range(draws):
        for rows in (D.cal, D.te):
            for r in rows:
                r["noise"] = np.maximum.accumulate(rng.standard_normal(r["L"]))
        for F in D.folds:
            resample(D.cal, rng, D, F, fold_norms[F][PRIMARY]); resample(D.te, rng, D, F, fold_norms[F][PRIMARY])
        keys = ["noise"] + [f"{PRIMARY}_resampled@{F}" for F in D.folds]
        C = {k: (cl.Cohort(D.cal, k), cl.Cohort(D.te, k)) for k in keys}
        for name, key_of in (("noise", lambda F: "noise"), (f"{PRIMARY}_resampled", lambda F: f"{PRIMARY}_resampled@{F}")):
            _, f, _ = unseen(D, key_of, D.rk0, C=C)
            res[name].append(dict(gain=round(gain(f, tG), 4), fpr=round(f["fp"] / f["n"], 4),
                                  mean_lead=round(float(f["lead"].mean()), 4)))
    return {k: dict(draws=len(v), mean_gain=round(float(np.mean([d["gain"] for d in v])), 4),
                    max_gain=max(d["gain"] for d in v), mean_fpr=round(float(np.mean([d["fpr"] for d in v])), 4),
                    per_draw=v) for k, v in res.items()}


# ---- prerequisite ----------------------------------------------------------------------------

def reproduce_amend2(D, amend2, amend2_norms, fold_norms_all):
    """Fold all, per-task normalization: must equal the registered Amendment 2 analysis."""
    worst = 0.0
    for d in AMEND2_DETECTORS:
        for s, by_task in amend2_norms[d].items():
            for t, st in by_task.items():
                mine = fold_norms_all[d][s]["per_task"][t]
                worst = max(worst, abs(mine["location"] - st["location"]), abs(mine["scale"] - st["scale"]))
    got, want = {}, {}
    c, e = D.C[f"{PRIMARY}@all/task"]
    tm, _ = cl.fit_fast(c, D.rkP)
    eP, lP = cl.evaluate_fast(e, D.rkP, tm)
    got["P"], want["P"] = [eP["mean_lead"], eP["fp"]], [amend2["P"]["mean_lead"], amend2["P"]["fp"]]
    for d in AMEND2_DETECTORS:
        c, e = D.C[f"{d}@all/task"]
        _, fu = cl.fit_fast(c, D.rkP)
        eF, lF = cl.evaluate_fast(e, D.rkP, fu)
        a = amend2["detectors"][d]
        got[d] = [eF["mean_lead"], eF["fp"], round(float(np.mean(lF - lP)), 4)]
        want[d] = [a["test"]["mean_lead"], a["test"]["fp"], a["gain"]]
    ok = worst <= 1e-12 and got == want
    return dict(normalization_max_abs_diff=worst, got=got, want=want, ok=ok)


# ---- run -------------------------------------------------------------------------------------

def run(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    spec = json.loads(Path(args.folds).read_text())
    tasks, held = fold_sets(spec)
    bundle = json.loads(Path(args.runtime_bundle).read_text())
    dev = qt.dev_fractions(bundle)
    cal_all, cal = cl.primary(args.calibration, "calibration")
    te_all, te = cl.primary(args.test, "test")
    if {r["rollout_id"] for r in cal_all} & {r["rollout_id"] for r in te_all} or cl.identities(cal_all) & cl.identities(te_all):
        raise ValueError("Calibration and test cohorts overlap")
    if tasks != set(bundle["task_horizons"]) or {r["task_name"] for r in cal} != tasks or {r["task_name"] for r in te} != tasks:
        raise ValueError("Folds, bundle and cohorts do not cover the same tasks")
    if any(spec["tasks"][r["task_name"]] != r.get("task_type") for r in (*cal, *te)):
        raise ValueError("Task types in the cohorts differ from the fold file")
    if json.loads(json.dumps(make_folds({r["task_name"]: r["task_type"] for r in cal}))) != spec:
        raise ValueError("The fold file is not the registered assignment for these tasks")

    wanted = {CAL_COHORT: {str(r["rollout_id"]) for r in cal}, TEST_COHORT: {str(r["rollout_id"]) for r in te}}
    folds_sha = sha256(args.folds)
    folds_data, fold_paths, first_meta = {}, {}, None
    for path in args.fold_dirs:
        meta, norms, scores = load_fold_dir(path, wanted)
        F = str(meta["fold"])
        if F in folds_data:
            raise ValueError(f"Two fold directories for fold {F}")
        first_meta = first_meta or meta
        check_meta(F, meta, tasks, held, folds_sha, first_meta)
        folds_data[F], fold_paths[F] = (meta, norms, scores), path
        print(f"fold {F}: {len(held[F])} held-out tasks, {sum(meta['training_rollouts_by_task'].values())} "
              f"training rollouts", flush=True)
    if set(folds_data) != set(held):
        raise ValueError(f"Need fold directories for {sorted(held)}, got {sorted(folds_data)}")
    checks = dict(raw_reference_max_abs_diff=check_reference(folds_data[ALL][0]),
                  test_failures_end_at_H=all(r["L"] == r["horizon"] for r in te if r["failed"]))
    attach(cal, folds_data, CAL_COHORT, held)
    attach(te, folds_data, TEST_COHORT, held)
    fold_norms = {F: v[1] for F, v in folds_data.items()}
    keys = [f"{d}@{F}" for F in sorted(held) for d in DETECTORS] + [f"{d}@all/task" for d in DETECTORS]
    D = Design(cal, te, held, keys, dev)

    inputs = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in
              (("calibration", args.calibration), ("test", args.test), ("runtime_bundle", args.runtime_bundle),
               ("folds", args.folds), ("amend2_result", args.amend2_result),
               ("amend2_normalizations", args.amend2_normalizations))}
    inputs["fold_dirs"] = {F: dict(path=str(Path(p).resolve()), code_commit=folds_data[F][0].get("code_commit"))
                           for F, p in sorted(fold_paths.items())}
    base = dict(plan=PLAN, status="exploratory, pre-specified (opened cohorts)", inputs=inputs,
                counts=dict(calibration=len(cal), test=len(te), tasks=len(tasks),
                            held_out_per_fold={F: len(held[F]) for F in D.folds}),
                folds={F: sorted(held[F]) for F in D.folds})

    # Prerequisite.
    rep = reproduce_amend2(D, json.loads(Path(args.amend2_result).read_text()),
                           json.loads(Path(args.amend2_normalizations).read_text()), fold_norms[ALL])
    checks["amend2_reproduction"] = rep
    print(f"Amendment 2 reproduction: got {rep['got']} want {rep['want']} "
          f"(normalization diff {rep['normalization_max_abs_diff']})", flush=True)
    if not rep["ok"]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dict(base, checks=checks, reproduction_failed=True), indent=1, default=float) + "\n")
        print("REPRODUCTION FAILED: fold all does not reproduce Amendment 2. No result is reported.")
        return 1

    # Point estimates.
    point = statistics(D)
    rows, per_fold = {}, {}
    tG = None
    for d in DETECTORS:
        t, f, pf = unseen(D, lambda F, d=d: f"{d}@{F}", D.rk0)
        tG = t
        rows[d] = dict(test=summary(f), gain=round(gain(f, t), 4),
                       paths={F: cl.det_json(v[1]) for F, v in pf.items()})
        for F, (tm, fu, a, b) in pf.items():
            per_fold.setdefault(F, dict(held_out=len(held[F]), G=dict(delta=round(tm["lam"], 4), **summary(a))))
            per_fold[F][d] = dict(**summary(b), gain=round(gain(b, a), 4))
    G = summary(tG)
    Fte = D.pooled_failures()
    t, f, _ = unseen(D, lambda F: f"{PRIMARY}@{F}", D.rk0)
    rows[PRIMARY]["gain_test_only_bootstrap"] = qt.boot(f["lead"] - t["lead"], Fte)

    st, sf, (stm, sfu) = seen(D, f"{PRIMARY}@all", D.rk0)
    seen_ref = dict(G=summary(st), hs_plus_G=dict(path=cl.det_json(sfu), **summary(sf)), gain=round(gain(sf, st), 4))
    pt, pfu, _ = seen(D, f"{PRIMARY}@all", D.rkP)
    seen_ref["P"] = summary(pt); seen_ref["hs_pooled_plus_P"] = summary(pfu); seen_ref["hs_pooled_gain_over_P"] = round(gain(pfu, pt), 4)
    pt, pfu, _ = seen(D, f"{PRIMARY}@all/task", D.rkP)
    seen_ref["hs_per_task_plus_P"] = summary(pfu); seen_ref["hs_per_task_gain_over_P"] = round(gain(pfu, pt), 4)

    # Few-shot point estimates: mean over draws of which n successes are known.
    rng = np.random.default_rng(FEW_SEED)
    few = {}
    for n in NS:
        acc = defaultdict(list)
        for _ in range(args.few_draws):
            rk = rho_n(dev, n, rng)
            t, f, _ = unseen(D, lambda F: f"{PRIMARY}@{F}", rk)
            for k, v in (("P_n_lead", t["lead"].mean()), ("P_n_fpr", t["fp"] / t["n"]), ("fused_lead", f["lead"].mean()),
                         ("fused_fpr", f["fp"] / f["n"]), ("gain", gain(f, t)), ("timer_catch_up", np.mean(t["lead"] - tG["lead"]))):
                acc[k].append(float(v))
        few[str(n)] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
        few[str(n)]["draws"] = args.few_draws
    t, f, _ = unseen(D, lambda F: f"{PRIMARY}@{F}", D.rkP)
    few["all"] = dict(P_n_lead=round(float(t["lead"].mean()), 4), P_n_fpr=round(t["fp"] / t["n"], 4),
                      fused_lead=round(float(f["lead"].mean()), 4), fused_fpr=round(f["fp"] / f["n"], 4),
                      gain=round(gain(f, t), 4), timer_catch_up=round(float(np.mean(t["lead"] - tG["lead"])), 4))

    print(f"running the decision bootstrap ({args.bootstrap} replicates)", flush=True)
    boot = bootstrap(D, args.bootstrap, cl.BOOT_SEED)
    ci = lambda k: boot[k]["ci95"]

    # U1 (primary).
    pr = rows[PRIMARY]
    u1 = dict(ci_lower_above_0=ci(f"zero_shot_gain/{PRIMARY}")[0] > 0, point_gain_at_least_margin=pr["gain"] >= cl.MARGIN,
              fused_fpr_at_most_0p075=pr["test"]["fpr"] <= cl.FPR_MAX,
              fused_fpr_wilson_upper_at_most_0p10=pr["test"]["fpr_wilson95"][1] <= cl.FPR_WILSON_MAX)
    u1["supported"] = all(u1.values())
    null = nulls(D, fold_norms, args.null_draws, tG)
    null_ok = all(v["mean_gain"] < cl.MARGIN for v in null.values())

    def contrast(name):
        return dict(point=round(point[name], 4), decision_ci95=ci(name))
    u2 = contrast(f"contrast/{PRIMARY}_minus_{ZERO}"); u2["beyond_elapsed_time"] = u2["decision_ci95"][0] > 0
    u3 = contrast(f"contrast/{PRIMARY}_minus_{FIRST}"); u3["develops_during_rollout"] = u3["decision_ci95"][0] > 0
    u4 = dict(zero_shot_gain=pr["gain"], seen_gain=seen_ref["gain"], **contrast("transfer_loss"))
    u4["transfer_loss_detected"] = u4["decision_ci95"][1] < 0
    u5 = {n: dict(few[n], gain_decision_ci95=ci(f"few_shot_gain/{n}"), timer_catch_up_ci95=ci(f"timer_catch_up/{n}"),
                  fused_vs_zero_shot_ci95=ci(f"fused_vs_zero_shot/{n}")) for n in [*map(str, NS), "all"]}
    below = [n for n in [*map(str, NS), "all"] if u5[n]["gain"] < cl.MARGIN]
    u5_reading = dict(smallest_n_with_point_gain_below_margin=below[0] if below else "not reached",
                      all_minus_seen_per_task=contrast("few_shot_all_minus_seen_per_task"))
    u6 = dict(zero_shot_gain=rows[KNN]["gain"], decision_ci95=ci(f"zero_shot_gain/{KNN}"), test=rows[KNN]["test"],
              hs_minus_knn=contrast(f"contrast/{PRIMARY}_minus_{KNN}"))
    u7 = dict(G=dict(fpr=G["fpr"], wilson=G["fpr_wilson95"]), hs_plus_G=dict(fpr=pr["test"]["fpr"], wilson=pr["test"]["fpr_wilson95"]),
              per_fold={F: dict(G=v["G"]["fpr"], hs_plus_G=v[PRIMARY]["fpr"]) for F, v in per_fold.items()})
    u7["calibration_transfers"] = G["fpr_wilson95"][1] <= cl.FPR_WILSON_MAX and pr["test"]["fpr_wilson95"][1] <= cl.FPR_WILSON_MAX

    # Descriptive.
    desc = dict(by_task_type={}, per_fold=per_fold)
    types = np.array([r["task_type"] for r in Fte])
    t, f, _ = unseen(D, lambda F: f"{PRIMARY}@{F}", D.rk0)
    for tt in TYPES:
        m = types == tt
        desc["by_task_type"][tt] = dict(failures=int(m.sum()), G_lead=round(float(t["lead"][m].mean()), 4),
                                        hs_plus_G_lead=round(float(f["lead"][m].mean()), 4),
                                        gain=round(float(np.mean((f["lead"] - t["lead"])[m])), 4))
    best = None
    for lam in LAM_GRID:
        parts = [raw_eval(D.C[f"{PRIMARY}@{F}"][1].take(D.te_idx[F]), D.rk0, dict(lam=float(lam), u=math.inf, s=0.0))
                 for F in D.folds]
        p = pool(parts)
        if p["fp"] / p["n"] <= pr["test"]["fpr"] + 1e-12 and (best is None or p["lead"].mean() > best[1]["lead"].mean()):
            best = (float(lam), p)
    desc["G_tuned_on_test_to_hs_fpr"] = dict(delta=best[0], test=summary(best[1]), hs_gain=qt.boot(f["lead"] - best[1]["lead"], Fte))

    out = dict(base, checks=checks, G=G, detectors=rows, seen_reference=seen_ref,
               bootstrap=dict(samples=args.bootstrap, seed=cl.BOOT_SEED, intervals=boot),
               U1_zero_shot=u1, U1_null_sanity=dict(nulls=null, interpretable=null_ok),
               U2_learned_timer=u2, U3_difficulty_prior=u3, U4_transfer_loss=u4,
               U5_few_shot=dict(by_n=u5, reading=u5_reading), U6_success_only_knn=u6, U7_fpr_transfer=u7,
               descriptive=desc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")

    print(f"G {G['mean_lead']} (FPR {G['fpr']})")
    for d in DETECTORS:
        print(f"{d:22s} + G lead {rows[d]['test']['mean_lead']} (FPR {rows[d]['test']['fpr']})  gain {rows[d]['gain']:+.4f} "
              f"CI {ci(f'zero_shot_gain/{d}')}")
    print(f"seen reference: Hide-and-Seek(all) + G gain {seen_ref['gain']:+.4f}; per-task + P gain "
          f"{seen_ref['hs_per_task_gain_over_P']:+.4f}")
    for n in [*map(str, NS), "all"]:
        print(f"few-shot n={n:>3}: P_n {u5[n]['P_n_lead']}  H&S + P_n {u5[n]['fused_lead']}  gain {u5[n]['gain']:+.4f} "
              f"CI {u5[n]['gain_decision_ci95']}")
    print(json.dumps(dict(U1=u1, U1_nulls={k: (v["mean_gain"], v["max_gain"]) for k, v in null.items()},
                          U2=u2, U3=u3, U4=u4, U5=u5_reading, U6=dict(gain=u6["zero_shot_gain"], ci=u6["decision_ci95"]),
                          U7=dict(G=u7["G"], hs_plus_G=u7["hs_plus_G"], transfers=u7["calibration_transfers"])),
                     indent=1, default=float))
    print("U1:", "SUPPORTED" if u1["supported"] else "NOT SUPPORTED",
          "| nulls interpretable:", null_ok, "| U2 beyond elapsed time:", u2["beyond_elapsed_time"],
          "| U3 develops during rollout:", u3["develops_during_rollout"],
          "| U4 transfer loss detected:", u4["transfer_loss_detected"],
          "| U7 calibration transfers:", u7["calibration_transfers"])
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("folds")
    f.add_argument("--predictions", required=True)
    f.add_argument("--folds", default=str(FOLDS_FILE))
    f.add_argument("--write")
    r = sub.add_parser("run")
    for name in ("--calibration", "--test", "--runtime-bundle", "--amend2-result", "--amend2-normalizations", "--output"):
        r.add_argument(name, required=True)
    r.add_argument("--fold-dirs", nargs="+", required=True)
    r.add_argument("--folds", default=str(FOLDS_FILE))
    r.add_argument("--bootstrap", type=int, default=cl.BOOT_SAMPLES)
    r.add_argument("--null-draws", type=int, default=cl.NULL_DRAWS)
    r.add_argument("--few-draws", type=int, default=FEW_DRAWS)
    a = p.parse_args(argv)
    return folds_cmd(a) if a.cmd == "folds" else run(a)


if __name__ == "__main__":
    sys.exit(main())
