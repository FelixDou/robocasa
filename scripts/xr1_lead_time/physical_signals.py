"""Physical (action-based) failure signals versus the per-task 95th-percentile timer.

Question from the 2026-10-20 progress meeting: do physical signals, such as whether the
robot is moving or stalling, or whether its action deltas look like those of successful
rollouts, add warning lead over a good per-task timer?

Data. The review_data copies (calibration/event prediction JSONL and runtime_bundle.json)
hold only SAFE and time scores. The executed actions are on TSUBAME: every collection made
with collect_atomic_rollouts (record_actions defaults to True) writes
<collection>/manifest.jsonl, whose records point to actions/<task>/<rollout_id>.npz (one
action dict per 20 Hz control step) and to rollouts/*.npz (policy_action_chunks, one
predicted chunk per inference). Proprioception and end-effector poses were NOT logged, so
every signal here is built from COMMANDED motion (OSC end-effector deltas, gripper, base),
not measured motion.

Protocol (same as rss2027_stage1/analysis/quantile_timer.py, alpha = 0.05):
  * development successes (runtime_bundle development_rollout_ids): per-task feature
    normalisation and the "success action statistics" reference only;
  * calibration primary cohort (3 successes + 3 failures per task): the feature and the
    fused path (u, s) are chosen on calibration FAILURES; the conformal margin lam_hat on
    calibration SUCCESSES (Proposition 1);
  * August prospective test (10 + 10 per task): exploratory evaluation, the same cohort
    the 0.195 timer and the +0.007 SAFE gain were measured on.
The confirmatory collection (environment seeds in [1300007, 1325007)) is never read: any
record with a seed in that range is dropped before its action file is opened.

Controls (added after the first run, where every signal "gained" while raising test FPR):
  * matched FPR: each fused detector vs the timer alone with its margin loosened to the
    same number of test false alarms (vs_matched_fpr_timer; post hoc, descriptive);
  * noise null: the same fit with each signal replaced by i.i.d. N(0, 1) scores
    (--noise-draws, default 20, seed 1, as in rss2027_stage1/analysis/extra.py). The chosen
    signal is compared with the best-on-calibration of len(FEATURES) noise scores per draw,
    each single signal with one noise score per draw. p = share of draws >= the observed gain.

Usage (on TSUBAME, CPU only):
  python -u physical_signals.py inventory --roots ROOT [ROOT ...] \
      --calibration CAL.jsonl --test TEST.jsonl --bundle runtime_bundle.json
  python -u physical_signals.py evaluate  --roots ROOT [ROOT ...] \
      --calibration CAL.jsonl --test TEST.jsonl --bundle runtime_bundle.json --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ALPHA = 0.05
PRIMARY_Q = 0.95
EPS = 1e-9
CONFIRMATORY_SEEDS = (1300007, 1325007)
SKIP_DIRS = {"rollouts", "actions", "env_records", "policy_records", "videos", "frames",
             "tensors", "logs", "__pycache__", ".git"}
NO_ALARM = -1e9


# --------------------------------------------------------------------------- loading
def is_confirmatory(seed):
    return seed is not None and CONFIRMATORY_SEEDS[0] <= int(seed) < CONFIRMATORY_SEEDS[1]


def load_predictions(path):
    out = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if is_confirmatory(r.get("environment_seed")):
            raise SystemExit(f"{path} contains a confirmatory seed; refusing to read it")
        r["failed"] = bool(r["failed"])
        r["safe"] = np.asarray(r["safe_scores"], float)
        r["L"] = len(r["safe"])
        out.append(r)
    return out


def select_balanced(records, q):
    """Same cohort rule as lead_time.py: first q successes and q failures per task."""
    g = defaultdict(list)
    for r in records:
        g[(r["task_name"], r["failed"])].append(r)
    out = []
    for t in sorted({r["task_name"] for r in records}):
        for f in (False, True):
            v = sorted(g[(t, f)], key=lambda r: (
                r.get("environment_seed") if r.get("environment_seed") is not None else -1,
                r.get("environment_reset_index") if r.get("environment_reset_index") is not None else -1,
                str(r["rollout_id"])))
            out += v[:q]
    return out


def find_manifests(roots):
    for root in roots:
        root = Path(root)
        if "confirmatory" in str(root).lower():
            print(f"skip root (confirmatory): {root}", file=sys.stderr)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and "confirmatory" not in d.lower()]
            if "manifest.jsonl" in filenames:
                yield Path(dirpath)


def index_manifests(roots, wanted):
    """rollout_id -> dict(dataset_dir, record) for ids in `wanted` that have action data."""
    found, n_manifests = {}, 0
    for d in find_manifests(roots):
        n_manifests += 1
        try:
            lines = open(d / "manifest.jsonl").read().splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(rec.get("rollout_id"))
            if rid not in wanted or rid in found:
                continue
            if is_confirmatory(rec.get("environment_seed")):
                continue
            a = rec.get("action_path"); t = rec.get("tensor_path")
            a_ok = bool(a) and (d / a).is_file()
            t_ok = bool(t) and (d / t).is_file()
            if a_ok or t_ok:
                found[rid] = dict(dir=d, record=rec, action=(d / a) if a_ok else None,
                                  tensor=(d / t) if t_ok else None)
    return found, n_manifests


def _split_vector(v):
    """Raw XR-1 / OpenPI 12-d vector -> named parts (see openpi_websocket_policy._convert_action)."""
    v = v.reshape(len(v), -1)
    return dict(pos=v[:, 0:3], rot=v[:, 3:6], grip=v[:, 6:7], base=v[:, 7:11])


def load_actions(entry):
    """Executed per-step actions as dict of [T, d] arrays."""
    if entry["action"] is not None:
        with np.load(entry["action"], allow_pickle=False) as z:
            keys = {k: z[k] for k in z.files if z[k].dtype != object and z[k].ndim >= 1}
        parts = {}
        for name, pat in (("pos", "end_effector_position"), ("rot", "end_effector_rotation"),
                          ("grip", "gripper"), ("base", "base_motion")):
            k = [k for k in keys if pat in k]
            if k:
                arr = np.asarray(keys[k[0]], float)
                parts[name] = arr.reshape(len(arr), -1)
        if {"pos", "rot"} <= set(parts):
            return parts, "executed_actions"
        if "action" in keys and keys["action"].reshape(len(keys["action"]), -1).shape[1] >= 11:
            return _split_vector(np.asarray(keys["action"], float)), "executed_actions"
    if entry["tensor"] is not None:
        with np.load(entry["tensor"], allow_pickle=False) as z:
            if "policy_action_chunks" in z.files:
                ch = np.asarray(z["policy_action_chunks"], float)
                rep = int(entry["record"].get("replan_steps") or 16)
                return _split_vector(ch[:, :rep, :].reshape(-1, ch.shape[-1])), "chunk_prefix"
    raise ValueError("no usable action data")


def load_chunks(entry):
    if entry["tensor"] is None:
        return None
    with np.load(entry["tensor"], allow_pickle=False) as z:
        return np.asarray(z["policy_action_chunks"], float) if "policy_action_chunks" in z.files else None


# --------------------------------------------------------------------------- features
def window_stats(parts, steps, n_inf, chunks=None, replan=16):
    """Per inference j (0-based, at env step s_j) summarise the actions EXECUTED in
    [s_{j-1}, s_j), i.e. only what was observable when inference j ran (causal).
    Row j = alarm index j + 1. Row 0 has no history and is NaN."""
    pos, rot = parts["pos"], parts["rot"]
    grip = parts.get("grip"); base = parts.get("base")
    T = len(pos)
    speed = np.linalg.norm(pos, axis=1) + np.linalg.norm(rot, axis=1)
    arm = np.concatenate([pos, rot], axis=1)
    dlt = np.r_[0.0, np.linalg.norm(np.diff(arm, axis=0), axis=1)]
    gsign = np.sign(grip[:, 0]) if grip is not None else np.zeros(T)
    toggles = np.r_[0.0, (np.abs(np.diff(gsign)) > 0).astype(float)]
    bspeed = np.linalg.norm(base, axis=1) if base is not None else np.zeros(T)
    W = dict(speed=np.full(n_inf, np.nan), delta=np.full(n_inf, np.nan), straight=np.full(n_inf, np.nan),
             toggles_cum=np.zeros(n_inf), base=np.full(n_inf, np.nan), mean_arm=np.full((n_inf, 6), np.nan),
             chunk_disagree=np.full(n_inf, np.nan))
    for j in range(1, n_inf):
        a, b = int(steps[j - 1]), min(int(steps[j]), T)
        if b <= a:
            continue
        W["speed"][j] = speed[a:b].mean()
        W["delta"][j] = dlt[a + 1:b].mean() if b - a > 1 else 0.0
        path = np.linalg.norm(pos[a:b], axis=1).sum()
        W["straight"][j] = np.linalg.norm(pos[a:b].sum(0)) / path if path > 1e-9 else 1.0
        W["toggles_cum"][j] = toggles[:b].sum()
        W["base"][j] = bspeed[a:b].mean()
        W["mean_arm"][j] = arm[a:b].mean(0)
        if chunks is not None and j < len(chunks) and chunks.shape[1] >= 2 * replan:
            # what inference j now predicts for [s_j, s_j+replan) vs what j-1 predicted for it
            prev = chunks[j - 1, replan:2 * replan, :6]; cur = chunks[j, :replan, :6]
            W["chunk_disagree"][j] = np.linalg.norm(cur - prev, axis=1).mean()
    return W


def ema(x, alpha=0.5):
    out = np.full_like(x, np.nan); m = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = m; continue
        m = v if np.isnan(m) else alpha * v + (1 - alpha) * m
        out[i] = m
    return out


class Reference:
    """Per-task statistics of development SUCCESSES (never calibration or test)."""

    def __init__(self, dev_windows, H):
        self.stats = {}
        for task, ws in dev_windows.items():
            sp = np.concatenate([w["speed"] for w in ws]); sp = sp[np.isfinite(sp)]
            de = np.concatenate([w["delta"] for w in ws]); de = de[np.isfinite(de)]
            cd = np.concatenate([w["chunk_disagree"] for w in ws]); cd = cd[np.isfinite(cd)]
            ba = np.concatenate([w["base"] for w in ws]); ba = ba[np.isfinite(ba)]
            tog = np.array([np.nanmax(w["toggles_cum"]) for w in ws])
            # success action statistics by normalised-time bin (10 bins)
            bins = defaultdict(list)
            for w in ws:
                n = len(w["speed"])
                for j in range(1, n):
                    if np.all(np.isfinite(w["mean_arm"][j])):
                        bins[min(9, int(10 * j / H[task]))].append(w["mean_arm"][j])
            allv = np.concatenate([np.asarray(v) for v in bins.values()]) if bins else np.zeros((1, 6))
            gm, gs = allv.mean(0), allv.std(0) + 1e-6
            bm = {b: (np.mean(v, 0), np.std(v, 0) + gs * 0.25) if len(v) >= 5 else (gm, gs) for b, v in bins.items()}
            self.stats[task] = dict(
                eps=float(np.quantile(sp, 0.10)) if len(sp) else 0.0,
                sp=(sp.mean(), sp.std() + 1e-9) if len(sp) else (0.0, 1.0),
                de=(de.mean(), de.std() + 1e-9) if len(de) else (0.0, 1.0),
                cd=(cd.mean(), cd.std() + 1e-9) if len(cd) else (0.0, 1.0),
                ba=(ba.mean(), ba.std() + 1e-9) if len(ba) else (0.0, 1.0),
                tog=(tog.mean(), tog.std() + 0.5) if len(tog) else (0.0, 1.0),
                bins=bm, glob=(gm, gs), n=len(ws))


FEATURES = ("stall_run", "stall_total", "low_motion", "action_delta", "dither",
            "gripper_toggles", "success_distance", "chunk_disagreement", "base_motion")


def scores(W, task, H, ref):
    """Per-inference scores (higher = more failure-like). NaN rows -> NO_ALARM."""
    st = ref.stats[task]; n = len(W["speed"])
    z = lambda x, mu_sd: (x - mu_sd[0]) / mu_sd[1]
    still = np.where(np.isfinite(W["speed"]), W["speed"] < st["eps"], False)
    run = np.zeros(n); tot = np.zeros(n)
    for j in range(1, n):
        run[j] = run[j - 1] + 1 if still[j] else 0
        tot[j] = tot[j - 1] + still[j]
    dist = np.full(n, np.nan)
    for j in range(1, n):
        if np.all(np.isfinite(W["mean_arm"][j])):
            m, s = st["bins"].get(min(9, int(10 * j / H)), st["glob"])
            dist[j] = float(np.sqrt(np.mean(((W["mean_arm"][j] - m) / s) ** 2)))
    out = dict(
        stall_run=run / H,                                   # current stall length, fraction of H_k
        stall_total=tot / H,                                 # cumulative stalled time, fraction of H_k
        low_motion=-z(ema(W["speed"]), st["sp"]),            # slow commanded motion (smoothed)
        action_delta=z(ema(W["delta"]), st["de"]),           # step-to-step action change (jerk)
        dither=ema(1.0 - W["straight"]),                     # back-and-forth commanded motion
        gripper_toggles=z(W["toggles_cum"], st["tog"]),      # excess open/close cycles
        success_distance=ema(dist),                          # distance from success actions at this phase
        chunk_disagreement=z(ema(W["chunk_disagree"]), st["cd"]),
        base_motion=z(ema(W["base"]), st["ba"]),
    )
    for k, v in out.items():
        v = np.asarray(v, float).copy(); v[~np.isfinite(v)] = NO_ALARM; out[k] = v
    return out


# --------------------------------------------------------------------------- protocol
def first(x, tau):
    m = np.maximum.accumulate(x); i = np.searchsorted(m, tau, side="left")
    return math.inf if i >= len(m) else i + 1


def dev_success_fractions(bundle, task):
    ac = bundle["time_risk_curves"][task]["active_counts"]; H = len(ac); fr = []
    for i in range(1, H):
        fr += [i / H] * (ac[i - 1] - ac[i])
    return fr


def rho(bundle, q):
    return {t: float(np.quantile(v, q)) if v else 1.0
            for t in bundle["time_risk_curves"] for v in [dev_success_fractions(bundle, t)]}


def timer_idx(r, rk, delta):
    return math.ceil(min(1.0, rk[r["task_name"]] + delta) * r["horizon"] - EPS)


def V(r, key, rk, u, s):
    if r["L"] >= r["horizon"]:
        return math.inf
    v_t = r["L"] / r["horizon"] - rk[r["task_name"]]
    M = float(np.max(r[key])) if key else -math.inf
    if u == math.inf:
        v_s = -math.inf
    elif s == 0:
        v_s = -math.inf if u > M else math.inf
    else:
        v_s = (M - u) / s
    return max(v_t, v_s, 0.0) + EPS


def conformal_lam(S, key, rk, u, s):
    vs = sorted(V(r, key, rk, u, s) for r in S); n = len(vs); k = math.ceil((n + 1) * (1 - ALPHA))
    return math.inf if k > n else vs[k - 1]


def alarm(r, key, rk, lam, u, s):
    a = timer_idx(r, rk, lam)
    if u != math.inf:
        a = min(a, first(r[key], u + s * lam))
    return a


def evaluate(rows, key, rk, lam, u, s):
    S = [r for r in rows if not r["failed"]]; F = [r for r in rows if r["failed"]]
    fp = sum(alarm(r, key, rk, lam, u, s) <= r["L"] for r in S)
    lv = np.array([(r["L"] - a) / r["horizon"] if (a := alarm(r, key, rk, lam, u, s)) <= r["L"] else 0.0 for r in F])
    early = sum(alarm(r, key, rk, lam, u, s) < timer_idx(r, rk, lam) for r in F)
    return dict(fpr=round(fp / len(S), 4), fp=int(fp), n_success=len(S), mean_lead=round(float(lv.mean()), 4),
                failures_alarmed_by_signal_before_timer=int(early)), lv


def fit_fused(calrows, key, rk):
    S = [r for r in calrows if not r["failed"]]; F = [r for r in calrows if r["failed"]]
    t_lam = conformal_lam(S, None, rk, math.inf, 0.0)
    best = (float(np.mean([(r["L"] - a) / r["horizon"] if (a := alarm(r, None, rk, t_lam, math.inf, 0)) <= r["L"] else 0.0
                           for r in F])), math.inf, 0.0, t_lam)
    fm = np.array([float(np.max(r[key])) for r in F])
    fm = fm[fm > NO_ALARM / 2]
    if len(fm) == 0:
        return dict(u=math.inf, s=0.0, lam=t_lam, cal_fail_lead=best[0])
    for u in sorted(set(np.round(np.quantile(fm, np.linspace(0, 1, 21)), 6).tolist())):
        for s in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
            lam = conformal_lam(S, key, rk, u, s)
            if not math.isfinite(lam):
                continue
            v = float(np.mean([(r["L"] - a) / r["horizon"] if (a := alarm(r, key, rk, lam, u, s)) <= r["L"] else 0.0
                               for r in F]))
            if v > best[0] + 1e-12:
                best = (v, u, s, lam)
    return dict(u=best[1], s=best[2], lam=best[3], cal_fail_lead=round(best[0], 4))


def boot(d, F, B=2000, seed=0):
    rng = np.random.default_rng(seed); bi = defaultdict(list)
    for i, r in enumerate(F):
        bi[r["task_name"]].append(i)
    tasks = sorted(bi); bs = []
    for _ in range(B):
        ii = np.concatenate([rng.choice(bi[t], len(bi[t])) for t in rng.choice(tasks, len(tasks))]); bs.append(d[ii].mean())
    return dict(mean=round(float(d.mean()), 4), ci95=np.round(np.percentile(bs, [2.5, 97.5]), 4).tolist())


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean()
    return round(float(gt + 0.5 * eq), 4)


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["inventory", "evaluate"])
    ap.add_argument("--roots", nargs="+", required=True, help="directories searched for manifest.jsonl")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out")
    ap.add_argument("--max-dev-per-task", type=int, default=60)
    ap.add_argument("--noise-draws", type=int, default=20, help="noise-null draws (0 = skip)")
    ap.add_argument("--noise-seed", type=int, default=1)
    args = ap.parse_args(argv)

    bundle = json.load(open(args.bundle))
    cal = select_balanced(load_predictions(args.calibration), 3)
    te = select_balanced(load_predictions(args.test), 10)
    dev_ids = set(map(str, bundle["development_rollout_ids"]))
    wanted = dev_ids | {str(r["rollout_id"]) for r in cal + te}
    print(f"cohorts: dev ids {len(dev_ids)}, calibration {len(cal)}, test {len(te)}", flush=True)
    found, nm = index_manifests(args.roots, wanted)
    print(f"manifests scanned: {nm}; rollouts with action data: {len(found)}", flush=True)

    cov = {}
    for name, ids in (("development", dev_ids), ("calibration", {str(r['rollout_id']) for r in cal}),
                      ("test", {str(r['rollout_id']) for r in te})):
        hit = [i for i in ids if i in found]
        cov[name] = dict(n=len(ids), with_actions=len(hit),
                         action_files=sum(found[i]["action"] is not None for i in hit),
                         tensor_files=sum(found[i]["tensor"] is not None for i in hit))
    dev_succ = [i for i in dev_ids if i in found and not found[i]["record"].get("failed", True)]
    cov["development"]["successes_with_actions"] = len(dev_succ)
    sample = next(iter(found.values()), None)
    if sample is not None:
        if sample["action"] is not None:
            with np.load(sample["action"], allow_pickle=False) as z:
                cov["example_action_file"] = dict(path=str(sample["action"]),
                                                  keys={k: list(z[k].shape) for k in z.files})
        if sample["tensor"] is not None:
            with np.load(sample["tensor"], allow_pickle=False) as z:
                cov["example_tensor_file"] = dict(path=str(sample["tensor"]),
                                                  keys={k: list(z[k].shape) for k in z.files})
    print(json.dumps(cov, indent=1), flush=True)
    if args.mode == "inventory":
        return 0

    for name in ("calibration", "test"):
        if cov[name]["with_actions"] < cov[name]["n"]:
            raise SystemExit(f"{name}: only {cov[name]['with_actions']}/{cov[name]['n']} rollouts have action data")

    H = {t: len(c["active_counts"]) for t, c in bundle["time_risk_curves"].items()}
    tasks = sorted({r["task_name"] for r in cal + te})

    def windows_for(entry, steps=None):
        parts, src = load_actions(entry)
        rec = entry["record"]
        steps = steps if steps is not None else rec.get("inference_env_steps")
        rep = int(rec.get("replan_steps") or 16)
        return window_stats(parts, steps, len(steps), load_chunks(entry), rep), src

    # Reference statistics from development successes only.
    by_task = defaultdict(list)
    for i in sorted(dev_succ):
        by_task[found[i]["record"]["task_name"]].append(i)
    dev_windows, sources = {}, defaultdict(int)
    for t in tasks:
        ids = by_task.get(t, [])[: args.max_dev_per_task]
        if len(ids) < 5:
            raise SystemExit(f"development successes with actions for {t}: {len(ids)} (< 5)")
        dev_windows[t] = []
        for i in ids:
            w, src = windows_for(found[i]); dev_windows[t].append(w); sources[src] += 1
        print(f"dev reference {t}: {len(ids)} successes", flush=True)
    ref = Reference(dev_windows, H)

    for r in cal + te:
        w, src = windows_for(found[str(r["rollout_id"])], r["inference_environment_steps"])
        sources[src] += 1
        for k, v in scores(w, r["task_name"], H[r["task_name"]], ref).items():
            r[k] = v[: r["L"]]

    rk = rho(bundle, PRIMARY_Q)
    out = dict(alpha=ALPHA, q=PRIMARY_Q, coverage=cov, action_sources=dict(sources),
               note="test = August prospective cohort (exploratory, already opened); confirmatory seeds never read")
    out.update(analyse(cal, te, rk, FEATURES, args.noise_draws, args.noise_seed))

    txt = json.dumps(out, indent=1, default=float)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt)
    print_summary(out)
    return 0


def matched_timer(te, rk, fp_target):
    """Timer alone with its margin loosened (smallest delta) so that it raises at most
    fp_target false alarms on the TEST successes. Post hoc and descriptive: it answers
    "how much lead would the timer get if it spent the same false-alarm budget"."""
    vs = sorted((V(r, None, rk, math.inf, 0.0) for r in te if not r["failed"]), reverse=True)
    # A success is alarmed iff lam < V_i, so fp(lam) = #{V_i > lam}.
    lam = 0.0 if fp_target >= len(vs) else vs[fp_target]
    e, lv = evaluate(te, None, rk, lam, math.inf, 0.0)
    return dict(delta=round(lam, 4), fp=e["fp"], fpr=e["fpr"], mean_lead=e["mean_lead"]), lv


def gain_vs_matched(te, rk, e, lv, F_te):
    mt, lm = matched_timer(te, rk, e["fp"])
    return dict(target_fp=e["fp"], timer=mt, gain=boot(lv - lm, F_te))


def noise_null(cal, te, rk, lP, F_te, n_features, draws, seed):
    """Same fit as the physical signals, with each signal replaced by i.i.d. N(0, 1) per
    inference (as in rss2027_stage1/analysis/extra.py). 'single' = one noise score;
    'selected' = n_features noise scores, the best picked on calibration failures, which
    mirrors how the physical signal was chosen."""
    rng = np.random.default_rng(seed)
    names = [f"_noise{i}" for i in range(n_features)]
    single, selected = [], []
    for d in range(draws):
        for r in cal + te:
            for nm in names:
                r[nm] = rng.standard_normal(r["L"])
        res = {}
        for nm in names:
            f = fit_fused(cal, nm, rk)
            e, lv = evaluate(te, nm, rk, f["lam"], f["u"], f["s"])
            mt, lm = matched_timer(te, rk, e["fp"])
            res[nm] = dict(cal=f["cal_fail_lead"], fp=e["fp"], fpr=e["fpr"],
                           gain=float((lv - lP).mean()), matched_gain=float((lv - lm).mean()))
        best = max(names, key=lambda nm: res[nm]["cal"])
        single.append(res[names[0]]); selected.append(res[best])
        print(f"noise draw {d + 1}/{draws}: single gain {res[names[0]]['gain']:+.4f} "
              f"(fp {res[names[0]]['fp']}), selected gain {res[best]['gain']:+.4f} (fp {res[best]['fp']})",
              flush=True)
    for r in cal + te:
        for nm in names:
            r.pop(nm, None)

    def summ(rows):
        g = np.array([x["gain"] for x in rows]); mg = np.array([x["matched_gain"] for x in rows])
        fp = np.array([x["fp"] for x in rows])
        return dict(gain_mean=round(float(g.mean()), 4), gain_min=round(float(g.min()), 4),
                    gain_max=round(float(g.max()), 4), matched_gain_mean=round(float(mg.mean()), 4),
                    matched_gain_max=round(float(mg.max()), 4), fp_mean=round(float(fp.mean()), 2),
                    fpr_mean=round(float(fp.mean()) / sum(not r["failed"] for r in te), 4),
                    gains=[round(float(x), 4) for x in g], matched_gains=[round(float(x), 4) for x in mg],
                    fps=fp.tolist())
    return dict(draws=draws, seed=seed, n_features_per_draw=n_features,
                single=summ(single), selected=summ(selected))


def p_upper(null, x):
    """One-sided permutation-style p: share of null draws at least as large as x."""
    null = np.asarray(null)
    return round(float((1 + np.sum(null >= x - 1e-12)) / (len(null) + 1)), 4)


def analyse(cal, te, rk, features, noise_draws=20, noise_seed=1):
    """Everything after the per-inference scores are attached to the rows (rows carry
    task_name, failed, horizon, L, safe and one array per feature)."""
    out = {}
    # Dev+cal-only diagnostics: AUROC (failure vs success) of each score at fixed fractions of H_k.
    diag = {}
    for k in features:
        d = {}
        for frac in (0.25, 0.5, 0.75):
            pos, neg = [], []
            for r in cal:
                j = min(r["L"], max(1, int(frac * r["horizon"])))
                if j > r["L"] - 1 and not r["failed"]:
                    continue                                    # success already finished
                v = float(np.max(r[k][:j]))
                (pos if r["failed"] else neg).append(v)
            d[f"cal_auroc_at_{frac}"] = auroc(pos, neg)
        diag[k] = d
    out["calibration_diagnostics"] = diag

    # Timer alone (P_0.95 with conformal margin), and each feature fused with it.
    Scal = [r for r in cal if not r["failed"]]
    t_lam = conformal_lam(Scal, None, rk, math.inf, 0.0)
    eP, lP = evaluate(te, None, rk, t_lam, math.inf, 0.0)
    out["timer"] = dict(delta=round(t_lam, 4), **eP)
    F_te = [r for r in te if r["failed"]]

    # SAFE + timer, for the side-by-side reference (should reproduce quantile_timer.json).
    fs = fit_fused(cal, "safe", rk)
    eS, lS = evaluate(te, "safe", rk, fs["lam"], fs["u"], fs["s"])
    out["safe_plus_timer"] = dict(path=fs, **eS, gain_over_timer=boot(lS - lP, F_te))

    out["safe_plus_timer"]["vs_matched_fpr_timer"] = gain_vs_matched(te, rk, eS, lS, F_te)

    per = {}
    for k in features:
        f = fit_fused(cal, k, rk)
        e, lv = evaluate(te, k, rk, f["lam"], f["u"], f["s"])
        per[k] = dict(path=f, test=e, gain_over_timer=boot(lv - lP, F_te),
                      vs_matched_fpr_timer=gain_vs_matched(te, rk, e, lv, F_te), _lv=lv)
    chosen = max(features, key=lambda k: per[k]["path"]["cal_fail_lead"])
    out["chosen_on_calibration_failures"] = chosen
    out["chosen_gain_over_timer"] = per[chosen]["gain_over_timer"]
    out["chosen_test"] = per[chosen]["test"]
    out["chosen_vs_matched_fpr_timer"] = per[chosen]["vs_matched_fpr_timer"]
    out["all_features_descriptive"] = {k: {kk: vv for kk, vv in v.items() if kk != "_lv"} for k, v in per.items()}
    out["chosen_minus_safe"] = boot(per[chosen]["_lv"] - lS, F_te)

    # Controls. (1) matched FPR: above, each detector vs the timer at the same test false-alarm
    # count. (2) noise null: does an uninformative score fused the same way gain as much?
    if noise_draws > 0:
        nn = noise_null(cal, te, rk, lP, F_te, len(features), noise_draws, noise_seed)
        cg = per[chosen]["gain_over_timer"]["mean"]
        cm = per[chosen]["vs_matched_fpr_timer"]["gain"]["mean"]
        nn["chosen_vs_selected_null"] = dict(
            gain=cg, p_gain=p_upper(nn["selected"]["gains"], cg),
            matched_gain=cm, p_matched_gain=p_upper(nn["selected"]["matched_gains"], cm))
        nn["per_feature_vs_single_null"] = {
            k: dict(gain=per[k]["gain_over_timer"]["mean"],
                    p_gain=p_upper(nn["single"]["gains"], per[k]["gain_over_timer"]["mean"]),
                    matched_gain=per[k]["vs_matched_fpr_timer"]["gain"]["mean"],
                    p_matched_gain=p_upper(nn["single"]["matched_gains"],
                                           per[k]["vs_matched_fpr_timer"]["gain"]["mean"]))
            for k in features}
        out["noise_null"] = nn
    return out


def print_summary(out):
    t = out["timer"]
    print("\n=== summary (lead = fraction of H_k; gain = mean lead minus timer's) ===")
    print(f"timer alone: fp {t['fp']}/{t['n_success']}  lead {t['mean_lead']:.4f}")
    rows = [("safe", out["safe_plus_timer"], out["safe_plus_timer"]["gain_over_timer"],
             out["safe_plus_timer"]["vs_matched_fpr_timer"])]
    rows += [(k, v["test"], v["gain_over_timer"], v["vs_matched_fpr_timer"])
             for k, v in out["all_features_descriptive"].items()]
    nn = out.get("noise_null", {}).get("per_feature_vs_single_null", {})
    print(f"{'signal':20s} {'fp':>4s} {'gain':>8s} {'95% CI':>18s} {'matched':>8s} {'95% CI':>18s} {'p_noise':>8s}")
    for k, e, g, m in rows:
        p = nn.get(k, {}).get("p_matched_gain", "")
        print(f"{k:20s} {e['fp']:4d} {g['mean']:+8.4f} {str(g['ci95']):>18s} "
              f"{m['gain']['mean']:+8.4f} {str(m['gain']['ci95']):>18s} {p!s:>8s}")
    if "noise_null" in out:
        n = out["noise_null"]
        for kind in ("single", "selected"):
            s = n[kind]
            print(f"noise null ({kind}, {n['draws']} draws): gain mean {s['gain_mean']:+.4f} "
                  f"[min {s['gain_min']:+.4f}, max {s['gain_max']:+.4f}], matched gain mean "
                  f"{s['matched_gain_mean']:+.4f} (max {s['matched_gain_max']:+.4f}), fp mean {s['fp_mean']}")
        c = n["chosen_vs_selected_null"]
        print(f"chosen ({out['chosen_on_calibration_failures']}) vs selected noise null: gain {c['gain']:+.4f} "
              f"p={c['p_gain']}, matched gain {c['matched_gain']:+.4f} p={c['p_matched_gain']}")


if __name__ == "__main__":
    sys.exit(main())
