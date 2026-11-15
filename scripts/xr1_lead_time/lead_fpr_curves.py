"""R18 (docs/r18_lead_fpr_curves_plan.md): calibrated lead-FPR curves of standalone monitors and timers.

Every curve point is a threshold set on calibration rollouts at a target alpha and evaluated on the
test set, so no threshold sees the test set. The threshold rule is per_seed_detection.threshold
(Table I's): the best calibration lead among thresholds with calibration FPR <= alpha; at alpha =
0.05 the point is Table I's (gate C1 in seen_unseen_tables.py). The alarm is the first inference
whose running maximum reaches the threshold (per_seed_detection.first).

A timer is a score with the same rule:
  G   t / H_k            one fraction of each task's limit
  E   t                  one number of inferences for every task (not told H_k)
  P   t / H_k - rho_k    the per-task timer (seen tasks only)
Unseen tasks: per fold, the threshold comes from the fold's seen tasks' calibration rollouts and the
test outcomes of its held-out tasks are pooled over folds, as in Table I.
"""
import math

import numpy as np

import per_seed_detection as psd

ALPHAS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14,
          0.16, 0.18, 0.20, 0.25, 0.30)
TIMER_KINDS = ("G", "E", "P")


def calibration_table(cal):
    """psd.evaluate at every candidate threshold of psd.threshold, vectorized over thresholds.
    Returns (taus in psd's order, fpr per tau, mean lead per tau)."""
    mx = sorted({float(np.max(r["score"])) for r in cal}, reverse=True)
    taus = np.array([np.nextafter(mx[0], np.inf)] + mx)
    S = [r for r in cal if not r["failed"]]
    F = [r for r in cal if r["failed"]]
    fp = np.zeros(len(taus), int)
    for r in S:
        m = np.maximum.accumulate(np.asarray(r["score"], float))
        fp += np.searchsorted(m, taus, side="left") < len(m)
    leads = np.zeros((len(taus), len(F)))          # row j: the leads at taus[j], in F's order
    for i, r in enumerate(F):
        m = np.maximum.accumulate(np.asarray(r["score"], float))
        k = np.searchsorted(m, taus, side="left")
        hit = k < len(m)
        leads[:, i] = np.where(hit, (r["L"] - (k + 1)) / r["horizon"], 0.0)
    mean_lead = np.array([float(np.mean(leads[j])) for j in range(len(taus))])
    return taus, fp / len(S), mean_lead


def threshold_at(table, alpha):
    """psd.threshold with ALPHA = alpha, from a calibration_table."""
    taus, fpr, lead = table
    best = None
    for j in range(len(taus)):
        if fpr[j] <= alpha + 1e-12:
            key = (float(lead[j]), float(taus[j]))
            if best is None or key > best[0]:
                best = (key, float(taus[j]))
    return best[1]


def outcomes(rows, tau):
    fp, n, hits, leads = 0, 0, [], []
    for r in rows:
        a = psd.first(r["score"], tau)
        if r["failed"]:
            hit = math.isfinite(a)
            hits.append(hit)
            leads.append((r["L"] - a) / r["horizon"] if hit else 0.0)
        else:
            n += 1
            fp += int(math.isfinite(a))
    return dict(fp=fp, n=n, hits=hits, leads=leads)


def curve(parts, alphas=ALPHAS):
    """parts: [(calibration rows, test rows)], one per fold (one for seen tasks). One point per alpha."""
    tables = [calibration_table(c) for c, _ in parts]
    out = []
    for alpha in alphas:
        taus = [threshold_at(t, alpha) for t in tables]
        os_ = [outcomes(e, tau) for (_, e), tau in zip(parts, taus)]
        fp, n = sum(o["fp"] for o in os_), sum(o["n"] for o in os_)
        hits = [h for o in os_ for h in o["hits"]]
        leads = [v for o in os_ for v in o["leads"]]
        out.append(dict(alpha=alpha, fpr=fp / n, mean_lead=float(np.mean(leads)), tpr=float(np.mean(hits)),
                        fp=fp, n_success=n, n_failure=len(hits), thresholds=taus))
    return out


def mean_std(curves):
    """Per alpha, mean and std (ddof 1) over seeds of test FPR and mean lead."""
    out = []
    for points in zip(*curves):
        if len({p["alpha"] for p in points}) != 1:
            raise ValueError("curves with different alpha grids")
        row = dict(alpha=points[0]["alpha"])
        for k in ("fpr", "mean_lead", "tpr"):
            v = [p[k] for p in points]
            row[k] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        out.append(row)
    return out


def timer_rows(rows, kind, rk=None):
    """Score rows of a timer for the given rollout rows (task_name, L, horizon, failed)."""
    out = []
    for r in rows:
        t = np.arange(1, int(r["L"]) + 1, dtype=float)
        if kind == "G":
            s = t / r["horizon"]
        elif kind == "E":
            s = t
        elif kind == "P":
            s = t / r["horizon"] - rk[r["task_name"]]
        else:
            raise ValueError(kind)
        out.append(dict(score=s, L=r["L"], horizon=r["horizon"], failed=bool(r["failed"]), task_name=r["task_name"]))
    return out


def at_alpha(points, alpha):
    for p in points:
        if abs(p["alpha"] - alpha) < 1e-12:
            return p
    raise KeyError(alpha)
