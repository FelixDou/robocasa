"""R26 (docs/r26_alarms_at_limit_plan.md): Table I with an alarm at a failure's final inference counted as a miss.

Table I (R14's table1 and r14.timers_alone; the own bands from R21) counts any alarm on a failure as a detection,
with lead 0 when it comes at the final inference. A timer's alarm is capped there (hk_free.timer_idx:
ceil(min(H_k, ...))), so for timer E it adds detections that need H_k and carry no warning. R26 recounts every
row with such an alarm as a miss:

  tpr_before_limit   failures alarmed before their final inference / failures
  bacc_before_limit  (tpr_before_limit + 1 - FPR) / 2

Thresholds, false alarms, FPR and mean lead are unchanged: only failures are recounted, and an alarm at the final
inference already has lead 0. A failure's alarm is at its final inference exactly when it is a hit with lead 0,
since lead = (L - a) / H_k with a <= L, and a failure runs to L = H_k; so the recount needs only each failure's
hit and lead, which every Table I code path already returns.

Gates (seen_unseen_tables.r26_xr1): R0, each recounted row's ordinary values equal this run's table1; R1, they
equal R14's table1 and r14.timers_alone (Table I's source); R2, the timers' false alarms, recall, mean lead and
alarms at the limit equal R21's timer_alone (P on seen tasks, E on unseen tasks). The own-band rows come from
R21's saved results (from_counts), which already count alarms at the limit.
"""
import math

import numpy as np

PLAN = "docs/r26_alarms_at_limit_plan.md (R26)"
TOL = 1e-12
ORDINARY = ("tpr", "fpr", "bacc", "mean_lead", "fp")
KEYS = ORDINARY + ("alarms_at_limit", "tpr_before_limit", "bacc_before_limit")
SEED_LABELS = ("seed_0", "seed_1", "seed_2")


def recount(hits, leads, fp, n):
    """hits, leads: one per failure (a miss has lead 0); fp: successes with an alarm; n: successes."""
    hits, leads = np.asarray(hits, bool), np.asarray(leads, float)
    if hits.shape != leads.shape or hits.size == 0 or n <= 0:
        raise ValueError("recount needs one hit and one lead per failure, and successes")
    if np.any(leads[~hits] != 0.0) or np.any(leads < 0.0):
        raise ValueError("a missed failure must have lead 0, and no lead may be negative")
    at_limit = int(np.sum(hits & (leads == 0.0)))
    tpr, before, fpr = float(hits.mean()), float(np.mean(hits & (leads > 0.0))), fp / n
    return dict(tpr=tpr, fpr=fpr, bacc=(tpr + 1 - fpr) / 2, mean_lead=float(leads.mean()), fp=int(fp),
                n_success=int(n), n_failure=int(hits.size), alarms_at_limit=at_limit,
                tpr_before_limit=before, bacc_before_limit=(before + 1 - fpr) / 2)


def from_counts(o):
    """The same recount from a saved result that carries tpr, fpr, fp, mean_lead, n_failure and
    alarms_at_limit (r19_extras.outcome, as R21 saved it)."""
    before = o["tpr"] - o["alarms_at_limit"] / o["n_failure"]
    out = {k: o[k] for k in ORDINARY}
    out.update(n_success=o.get("n_success"), n_failure=o["n_failure"], alarms_at_limit=int(o["alarms_at_limit"]),
               tpr_before_limit=float(before), bacc_before_limit=float((before + 1 - o["fpr"]) / 2))
    return out


def with_seed_summary(by_label):
    """Adds mean_std_over_seeds (mean and std, ddof 1) when the three seeds are present."""
    seeds = [by_label[s] for s in SEED_LABELS if s in by_label]
    if len(seeds) > 1:
        by_label["mean_std_over_seeds"] = {
            k: dict(mean=float(np.mean([p[k] for p in seeds])), std=float(np.std([p[k] for p in seeds], ddof=1)))
            for k in KEYS}
    return by_label


def worst_diff(mine, ref, keys=ORDINARY):
    """max |mine[k] - ref[k]| over keys (inf if a key is missing on either side)."""
    worst = 0.0
    for k in keys:
        if k not in mine or k not in ref or mine[k] is None or ref[k] is None:
            return math.inf
        worst = max(worst, abs(float(mine[k]) - float(ref[k])))
    return worst


def compare(rows, reference):
    """{row/side/label: max |diff|} of the ordinary values, for every label present in both."""
    out = {}
    for name, sides in rows.items():
        for side, labels in (sides or {}).items():
            ref_side = ((reference.get(name) or {}).get(side)) if reference else None
            if not labels or ref_side is None:
                continue
            for lab, mine in labels.items():
                if lab == "mean_std_over_seeds" or lab not in ref_side:
                    continue
                out[f"{name}/{side}/{lab}"] = worst_diff(mine, ref_side[lab])
    return out


def bands_from_r21(r21, heads, sides=(("seen", "P"), ("unseen", "G_abs"))):
    """Own-band rows of Table I from R21's saved results: {head: {side: {label: recount}}}."""
    out = {}
    for h in heads:
        out[h] = {}
        for side, timer in sides:
            block = r21["heads"][h][side][timer]
            out[h][side] = with_seed_summary({lab: from_counts(block[lab]["standalone"])
                                              for lab in ("seed_average",) + SEED_LABELS if lab in block})
    return out


def gate(name, diffs, tol=TOL):
    worst = max(diffs.values()) if diffs else math.inf
    return dict(max_abs_diff=worst, compared=len(diffs), ok=bool(diffs) and worst <= tol,
                failing=sorted(k for k, v in diffs.items() if not v <= tol))


def cell(side_block, key):
    """'mean +- std' over seeds, or the deterministic value."""
    if side_block is None:
        return "--"
    if "mean_std_over_seeds" in side_block:
        m = side_block["mean_std_over_seeds"][key]
        return f"{m['mean']:.3f}+-{m['std']:.3f}"
    return f"{side_block['seed_average'][key]:.3f}"


def print_summary(res, label="XR-1"):
    print(f"R26 {label}: Table I with an alarm at a failure's final inference counted as a miss")
    print(f"  {'row':22s} {'side':6s} {'bACC (as printed)':>18s} {'bACC (limit = miss)':>20s} "
          f"{'recall before limit':>20s} {'at limit':>9s}")
    for name, sides in res["rows"].items():
        for side, block in (sides or {}).items():
            if block is None:
                continue
            at = (block["mean_std_over_seeds"]["alarms_at_limit"]["mean"] if "mean_std_over_seeds" in block
                  else block["seed_average"]["alarms_at_limit"])
            print(f"  {name:22s} {side:6s} {cell(block, 'bacc'):>18s} {cell(block, 'bacc_before_limit'):>20s} "
                  f"{cell(block, 'tpr_before_limit'):>20s} {at:9.1f}")
    for k, v in res["checks"].items():
        print(f"  {k}: {'ok' if v['ok'] else 'FAILED'} (max |diff| {v['max_abs_diff']:.3g}, {v['compared']} compared)"
              + ("" if v["ok"] else f"; failing: {v['failing'][:6]}"))
    print("R26 OK" if res["ok"] else "R26 FAILED")
