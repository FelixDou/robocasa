"""E2 follow-up: three offline analyses of the closed-loop scenes (docs/rldx_e2_plan.md, "Follow-up").

Written after E2's result (2026-11-09); descriptive. Inputs are E2's own files; no rollout is new.
Every rule's alarm comes from analyze_e2 (confirm_lead.alarm_vec with the plan's frozen paths), and
gate F0 requires E2's published success rates to be reproduced first.

F1. Ceiling of reset-and-retry. A perfect detector resets exactly the first attempts that would fail
    within the budget B, and never a success:
      oracle@P     at P's alarm (P's timing; latency 0)
      oracle@SAFE  at SAFE-LSTM + P's alarm (its timing; 1 control step of latency)
      oracle@0     at the first inference (the retry gets the whole budget)
    A scene succeeds if its first attempt does, or if it is reset and the shared retry succeeds within
    the budget left. Reported against no reset and against P, with 95% CIs (scenes resampled within task).
F2. Is failure a property of the scene? Per task and type: first-attempt success, full-budget retry
    success after a failed and after a successful first attempt, and the Mantel-Haenszel odds ratio
    (strata = tasks) of retry success given first-attempt success, with a within-task bootstrap CI.
    OR = 1: attempts independent given the task; OR >> 1: some scenes are hard for this policy.
F3. Stopping at the alarm instead of restarting (e.g. handing over). For P, SAFE-LSTM + P and
    Hide-and-Seek + P: robot time saved on failed first attempts, (B - t_a) control steps at 20 Hz,
    and successes stopped (alarm before the success). Differences against P with 95% CIs.

Usage:
  python3 -I followup_e2.py --plan <E2>/plan.json --pairs <E2>/pairs.json --scores-dir <E2>/scores \\
      --result <E2>/e2_result.json --output <E2>/e2_followup.json
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import analyze_e2 as an  # noqa: E402

PLAN = "docs/rldx_e2_plan.md (Follow-up)"
CONTROL_HZ = 20.0
F3_RULES = ("P", "safe_lstm_plus_P", "hide_and_seek_plus_P")


def first_ok(s):
    return bool(s["s0"] and s["n0"] <= s["B"])


def retry_ok(s, steps_left=None):
    left = s["B"] if steps_left is None else steps_left
    return bool(s["s1"] and s["n1"] <= left)


# ---- F1 --------------------------------------------------------------------------------------

def oracle(scenes, cond=None, latency=0):
    """Success when exactly the failing first attempts are reset (at cond's alarm, or at once)."""
    a = an.alarms(scenes, cond) if cond is not None else np.ones(len(scenes))
    won = np.zeros(len(scenes), bool)
    for i, s in enumerate(scenes):
        if first_ok(s):
            won[i] = True
        elif np.isfinite(a[i]):
            t_a = 0 if cond is None else s["inf_steps"][int(a[i]) - 1]
            won[i] = retry_ok(s, s["B"] - t_a - latency)
    return won


def f1(plan, scenes):
    conds = plan["conditions"]
    cols = dict(none=an.outcomes(scenes, conds["none"])[0], P=an.outcomes(scenes, conds["P"])[0],
                **{"oracle@P": oracle(scenes, conds["P"]),
                   "oracle@SAFE": oracle(scenes, conds["safe_lstm_plus_P"], conds["safe_lstm_plus_P"]["latency_steps"]),
                   "oracle@0": oracle(scenes)})
    names = list(cols)
    W = np.column_stack([cols[k] for k in names]).astype(float)
    means, ci, _ = an.bootstrap(W, an.strata(scenes), plan["bootstrap"], plan["analysis_seed"], plan["level"])
    j = {k: i for i, k in enumerate(names)}
    out = dict(success={k: dict(rate=round(float(W[:, j[k]].mean()), 4), ci=[round(x, 4) for x in ci(means[:, j[k]])])
                        for k in names})
    for k in ("oracle@P", "oracle@SAFE", "oracle@0"):
        for ref in ("none", "P"):
            d = W[:, j[k]] - W[:, j[ref]]
            out[f"{k}_minus_{ref}"] = dict(delta=round(float(d.mean()), 4),
                                           ci=[round(x, 4) for x in ci(means[:, j[k]] - means[:, j[ref]])])
    fails = [s for s in scenes if not first_ok(s)]
    out["failed_first_attempts"] = len(fails)
    out["their_full_budget_retry_success"] = round(float(np.mean([retry_ok(s) for s in fails])), 4) if fails else None
    return out


# ---- F2 --------------------------------------------------------------------------------------

def mh_odds_ratio(scenes):
    """Mantel-Haenszel OR over tasks of retry success given first-attempt success (+0.5 if a stratum has a 0 cell)."""
    num = den = 0.0
    by = defaultdict(lambda: np.zeros(4))
    for s in scenes:
        a, b = first_ok(s), retry_ok(s)
        by[s["task_name"]][(0 if a else 2) + (0 if b else 1)] += 1      # [ok,ok] [ok,fail] [fail,ok] [fail,fail]
    for c in by.values():
        n = c.sum()
        if n == 0:
            continue
        a, b, cc, d = c + (0.5 if (c == 0).any() else 0.0)
        n = a + b + cc + d
        num += a * d / n
        den += b * cc / n
    return num / den if den > 0 else math.inf


def f2(plan, scenes):
    def summary(sub):
        f = [s for s in sub if not first_ok(s)]
        ok = [s for s in sub if first_ok(s)]
        return dict(scenes=len(sub), first_success=round(float(np.mean([first_ok(s) for s in sub])), 4),
                    retry_success=round(float(np.mean([retry_ok(s) for s in sub])), 4),
                    retry_after_failure=round(float(np.mean([retry_ok(s) for s in f])), 4) if f else None,
                    retry_after_success=round(float(np.mean([retry_ok(s) for s in ok])), 4) if ok else None,
                    both_failed=sum(1 for s in sub if not first_ok(s) and not retry_ok(s)))
    by_task = {t: summary([s for s in scenes if s["task_name"] == t]) for t in sorted({s["task_name"] for s in scenes})}
    by_type = {ty: summary([s for s in scenes if s["type"] == ty]) for ty in sorted({s["type"] for s in scenes})}
    # Expected retry success after a failure if attempts were independent given the task.
    fails = [s for s in scenes if not first_ok(s)]
    expected = float(np.mean([by_task[s["task_name"]]["retry_success"] for s in fails])) if fails else None
    rng = np.random.default_rng(plan["analysis_seed"])
    groups = an.strata(scenes)
    ors = []
    for _ in range(plan["bootstrap"]):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        ors.append(mh_odds_ratio([scenes[i] for i in idx]))
    q = 100 * (1 - plan["level"]) / 2
    finite = np.array([x for x in ors if math.isfinite(x)])
    return dict(all=summary(scenes), by_type=by_type, by_task=by_task,
                retry_after_failure_if_independent=round(expected, 4) if expected is not None else None,
                odds_ratio=round(mh_odds_ratio(scenes), 3),
                odds_ratio_ci=[round(float(np.percentile(finite, q)), 3), round(float(np.percentile(finite, 100 - q)), 3)]
                if len(finite) else None)


# ---- F3 --------------------------------------------------------------------------------------

def stop_values(scenes, cond):
    """Per scene: seconds saved (failed first attempts only) and whether a success was stopped."""
    a = an.alarms(scenes, cond)
    saved, stopped = np.zeros(len(scenes)), np.zeros(len(scenes), bool)
    for i, s in enumerate(scenes):
        if not np.isfinite(a[i]):
            continue
        t_a = s["inf_steps"][int(a[i]) - 1]
        if first_ok(s):
            stopped[i] = True                       # alarms() only returns alarms within the attempt
        else:
            saved[i] = (s["B"] - t_a) / CONTROL_HZ
    return saved, stopped


def f3(plan, scenes):
    conds = plan["conditions"]
    rules = [r for r in F3_RULES if r in conds and (r != "hide_and_seek_plus_P" or "hide_and_seek" in scenes[0])]
    fail = np.array([not first_ok(s) for s in scenes])
    ok = ~fail
    budget_s = np.array([s["B"] / CONTROL_HZ for s in scenes])
    vals = {r: stop_values(scenes, conds[r]) for r in rules}
    W = np.column_stack([vals[r][0] for r in rules] + [vals[r][1].astype(float) for r in rules])
    means, ci, _ = an.bootstrap(W, an.strata(scenes), plan["bootstrap"], plan["analysis_seed"], plan["level"])
    k = len(rules)
    out = dict(failed_first_attempts=int(fail.sum()), successful_first_attempts=int(ok.sum()),
               mean_failure_length_s=round(float(budget_s[fail].mean()), 2) if fail.any() else None, rules={})
    for i, r in enumerate(rules):
        saved, stopped = vals[r]
        out["rules"][r] = dict(
            seconds_saved_per_failure=round(float(saved[fail].mean()), 2) if fail.any() else None,
            share_of_failure_time_saved=round(float((saved[fail] / budget_s[fail]).mean()), 4) if fail.any() else None,
            failures_stopped=int((saved[fail] > 0).sum()),
            successes_stopped=int(stopped.sum()),
            share_of_successes_stopped=round(float(stopped[ok].mean()), 4) if ok.any() else None)
        if r != "P" and "P" in rules:
            p = rules.index("P")
            # Per-scene means (scenes in the denominator), so the bootstrap CI matches the point value.
            d_saved = means[:, i] - means[:, p]
            d_stop = means[:, k + i] - means[:, k + p]
            out["rules"][r]["minus_P"] = dict(
                seconds_saved_per_scene=round(float((saved - vals["P"][0]).mean()), 3),
                seconds_saved_per_scene_ci=[round(x, 3) for x in ci(d_saved)],
                successes_stopped_per_scene=round(float((stopped.astype(float) - vals["P"][1]).mean()), 4),
                successes_stopped_per_scene_ci=[round(x, 4) for x in ci(d_stop)])
    return out


# ---- main ------------------------------------------------------------------------------------

def reproduce(plan, scenes, result, has_hs):
    """F0: E2's published success rates, recomputed here."""
    problems = []
    for k, v in result["rates"].items():
        if k == "hide_and_seek_plus_P" and not has_hs:
            continue
        got = round(float(an.outcomes(scenes, plan["conditions"][k])[0].mean()), 4)
        if got != v["success"]:
            problems.append(f"F0: {k} success {got} != published {v['success']}")
    if len(scenes) != result["scenes"]:
        problems.append(f"F0: {len(scenes)} scenes != published {result['scenes']}")
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--result", required=True, help="e2_result.json (gate F0)")
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    if Path(a.output).exists():
        raise SystemExit(f"error: {a.output} exists")
    plan = json.loads(Path(a.plan).read_text())
    pairs = json.loads(Path(a.pairs).read_text())
    result = json.loads(Path(a.result).read_text())
    dirs = sorted({x["dir"] for x in pairs["pairs"]})
    manifests = {r["rollout_id"]: r for d in dirs for r in an.jsonl(Path(d) / "manifest.jsonl")}
    scores = {}
    for k, f in an.SCORE_FILES.items():
        path = Path(a.scores_dir) / f
        if path.is_file():
            scores[k] = {str(r["rollout_id"]): r for r in an.jsonl(path)}
    scenes, problems = an.load_scenes(plan, pairs, manifests, scores)
    has_hs = "hide_and_seek" in scores
    problems += reproduce(plan, scenes, result, has_hs) if not problems else []
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems[:20]))
    out = dict(plan=PLAN, status="descriptive, after E2's result", scenes=len(scenes), F0="E2 success rates reproduced",
               F1=f1(plan, scenes), F2=f2(plan, scenes), F3=f3(plan, scenes))
    Path(a.output).write_text(json.dumps(out, indent=1) + "\n")
    print("F0 OK: E2's success rates reproduced")
    s = out["F1"]["success"]
    print("F1 success: " + ", ".join(f"{k} {v['rate']:.3f} {v['ci']}" for k, v in s.items()))
    for k in ("oracle@P", "oracle@SAFE", "oracle@0"):
        x, y = out["F1"][f"{k}_minus_none"], out["F1"][f"{k}_minus_P"]
        print(f"  {k:12s} vs none {x['delta']:+.3f} {x['ci']} | vs P {y['delta']:+.3f} {y['ci']}")
    print(f"  failed first attempts {out['F1']['failed_first_attempts']}, their full-budget retry success "
          f"{out['F1']['their_full_budget_retry_success']}")
    f = out["F2"]
    print(f"F2 all {f['all']}")
    print(f"  retry after failure if independent given task: {f['retry_after_failure_if_independent']}; "
          f"MH odds ratio {f['odds_ratio']} {f['odds_ratio_ci']}")
    for ty, v in f["by_type"].items():
        print(f"  {ty}: {v}")
    for t, v in f["by_task"].items():
        print(f"  {t:28s} first {v['first_success']:.2f} retry|fail {v['retry_after_failure']} "
              f"retry|ok {v['retry_after_success']} both failed {v['both_failed']}")
    g = out["F3"]
    print(f"F3 failed {g['failed_first_attempts']} (mean length {g['mean_failure_length_s']} s), "
          f"successful {g['successful_first_attempts']}")
    for r, v in g["rules"].items():
        print(f"  {r:22s} {v}")
    print("FOLLOW-UP DONE")


if __name__ == "__main__":
    main()
