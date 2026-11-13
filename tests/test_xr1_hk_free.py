"""R14 (hk_free.py): timers without H_k. Gate K2 and the prompt and parsing rules."""
import json
import math
import sys
from pathlib import Path

import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xr1_lead_time"))
import confirm_lead as cl  # noqa: E402
import cross_task_lead as ctl  # noqa: E402
import hk_free as hk  # noqa: E402


def cohort(seed=0, tasks=("A", "B", "C"), H=None, per=30):
    rng = np.random.default_rng(seed)
    H = H or {"A": 30, "B": 80, "C": 200}
    rows = []
    for t in tasks:
        for i in range(per):
            failed = i % 3 == 0
            L = H[t] if failed else int(rng.integers(max(2, H[t] // 4), H[t]))
            drift = 0.02 if failed else 0.0
            score = np.cumsum(rng.normal(drift, 0.05, L))
            rows.append(dict(task_name=t, failed=failed, horizon=H[t], L=L, score=score))
    return cl.Cohort(rows, "score"), rows


def test_k2_unit_H_equals_confirm_lead():
    c, _ = cohort(0)
    e, _ = cohort(1)
    for rk in ({"A": 0.0, "B": 0.0, "C": 0.0}, {"A": 0.6, "B": 0.4, "C": 0.7}):
        tm0, fu0 = cl.fit_fast(c, rk)
        cu, rc = hk.with_unit(c, c.H), cl.rho_rows(c, rk)
        tm1, fu1 = hk.fit(cu, rc)
        assert tm0 == tm1 and fu0 == fu1
        eu, re_ = hk.with_unit(e, e.H), cl.rho_rows(e, rk)
        for det in (tm0, fu0):
            a, b = ctl.raw_eval(e, rk, det), hk.raw_eval(eu, re_, det)
            assert a["fp"] == b["fp"] and np.array_equal(a["lead"], b["lead"]) and a["final"] == b["final"]


def test_k2_run_folds_G_equals_cross_task_unseen():
    c, _ = cohort(2)
    e, _ = cohort(3)
    pairs = {"0": (c.take(np.flatnonzero(c.task != "A")), e.take(np.flatnonzero(e.task == "A"))),
             "1": (c.take(np.flatnonzero(c.task == "A")), e.take(np.flatnonzero(e.task != "A")))}
    got = hk.run_folds(pairs, "G", None, None)
    rk0 = {"A": 0.0, "B": 0.0, "C": 0.0}
    pt, pf = [], []
    for F in sorted(pairs):
        tm, fu = cl.fit_fast(pairs[F][0], rk0)
        pt.append(ctl.raw_eval(pairs[F][1], rk0, tm)); pf.append(ctl.raw_eval(pairs[F][1], rk0, fu))
    tG, fD = ctl.pool(pt), ctl.pool(pf)
    assert got["fp"] == fD["fp"] and got["gain"] == ctl.gain(fD, tG)
    assert got["mean_lead"] == float(fD["lead"].mean())


def test_k2_absolute_timer_is_an_elapsed_time_conformal_cutoff():
    c, _ = cohort(4)
    cu = hk.with_unit(c, np.ones(len(c.L)))
    tm, _ = hk.fit(cu, np.zeros(len(c.L)))
    S = np.sort(c.L[~c.failed])
    n = len(S)
    k = math.ceil((n + 1) * (1 - cl.ALPHA))
    cutoff = S[k - 1] + 1                       # first inference after the k-th shortest success
    a = hk.timer_idx(cu, np.zeros(len(c.L)), tm["lam"])
    assert np.all(a[c.L >= cutoff] == np.minimum(cutoff, c.H[c.L >= cutoff]))
    fp = int(np.sum(a[~c.failed] <= c.L[~c.failed]))
    assert fp == n - k                           # successes longer than the k-th alarm, ties excluded
    assert fp / n <= cl.ALPHA + 1 / n


def test_k5_P_mult_and_P_agree_at_zero_margin():
    c, _ = cohort(5)
    rk = {"A": 0.55, "B": 0.37, "C": 0.81}
    Dp, rp = hk.unit_and_rho(c, "P", None, None, rk)
    Dm, rm = hk.unit_and_rho(c, "P_mult", None, None, rk)
    a = hk.timer_idx(hk.with_unit(c, Dp), rp, 0.0)
    b = hk.timer_idx(hk.with_unit(c, Dm), rm, 0.0)
    assert np.array_equal(a, b)


def test_scaled_unit_gives_the_same_alarms():
    """Any constant factor in D is absorbed by lam: the LLM's seconds need no conversion."""
    c, _ = cohort(6)
    D = np.array([{"A": 20.0, "B": 50.0, "C": 140.0}[t] for t in c.task])
    t1, f1 = hk.fit(hk.with_unit(c, D), np.zeros(len(D)))
    t2, f2 = hk.fit(hk.with_unit(c, 0.4 * D), np.zeros(len(D)))
    a1 = hk.alarm(hk.with_unit(c, D), np.zeros(len(D)), f1["lam"], f1["u"], f1["s"])
    a2 = hk.alarm(hk.with_unit(c, 0.4 * D), np.zeros(len(D)), f2["lam"], f2["u"], f2["s"])
    assert np.array_equal(a1, a2)


def test_matched_timer_is_the_smallest_margin_within_the_target():
    e, _ = cohort(7)
    eu, r = hk.with_unit(e, e.H), np.zeros(len(e.L))
    for target in (0, 3, 10):
        m, lam = hk.matched_timer([(eu, r)], target)
        assert m["fp"] <= target
        v = np.sort(hk.V(hk.sub(eu, ~e.failed), r[~e.failed], math.inf, 0.0))
        smaller = v[v < lam]
        if len(smaller):                         # the next smaller candidate margin exceeds the target
            looser = hk.raw_eval(eu, r, dict(u=math.inf, s=0.0, lam=float(smaller[-1])))
            assert looser["fp"] > target


def test_prompts_leave_the_task_out_and_never_name_the_benchmark():
    ins = {"A": "Open the fridge.", "B": "Pick the cup and place it in the sink.", "C": "Turn on the stove."}
    held = {"0": {"A"}, "1": {"B", "C"}, "all": set()}
    med = {"A": 12.0, "B": 30.0, "C": 9.0}
    p = hk.build_prompts(ins, held, med)
    assert p["examples"]["0"]["A"] == ["B", "C"]          # held out: every seen task
    assert p["examples"]["0"]["B"] == ["C"]               # seen: the others, never itself
    assert p["examples"]["all"]["C"] == ["A", "B"]
    for F, by in p["few"].items():
        for t, text in by.items():
            assert f'"{ins[t]}": ' not in text and ins[t] in text
    allp = [p["system"], p["retry"], *p["zero"].values(), *(x for v in p["few"].values() for x in v.values())]
    assert not any("robocasa" in x.lower() for x in allp)


def test_parse_seconds():
    assert hk.parse_seconds("45") == 45.0
    assert hk.parse_seconds("About 12.5 seconds.") == 12.5
    assert hk.parse_seconds("<think>maybe 3</think> 60") == 60.0
    assert hk.parse_seconds("unknown") is None
    assert hk.parse_seconds("0") is None


def test_answer_all_retries_then_falls_back_and_counts():
    ins = {"A": "Open the fridge.", "B": "Close the drawer.", "C": "Turn on the stove."}
    held = {"0": {"A"}, "all": set()}
    med = {"A": 12.0, "B": 30.0, "C": 9.0}
    p = hk.build_prompts(ins, held, med)
    p["example_median_seconds"] = med
    calls = []

    def gen(msgs):
        calls.append(msgs)
        text = msgs[1]["content"]
        if '"Close the drawer."\n\nHow many' in text:
            return "no idea" if len(msgs) == 2 else "still no"   # B: never parsed
        if "Turn on the stove" in text and len(msgs) == 2:
            return "hmm"                                         # C: parsed on the retry
        return "20"
    zero, few, raw, fails = hk.answer_all(p, gen)
    assert zero == {"A": 20.0, "B": 20.0, "C": 20.0}             # B falls back to the median of the others
    assert fails["zero"] == dict(prompts=3, failed=1)
    assert few["0"]["B"] == 9.0                                  # median of B's examples in fold 0 (only C)
    assert fails["few"]["failed"] == 2
    assert sum(r["fallback"] for r in raw) == 3


def test_check_durations_contract():
    tasks, held = {"A", "B"}, {"0": {"A"}, "1": {"B"}, "all": set()}
    good = dict(prompt_sha256=hk.prompts_sha256(), folds_sha256="x", zero={"A": 10.0, "B": 20.0},
                few={F: {"A": 10.0, "B": 20.0} for F in held}, answers=8,
                parse_failures={"zero": dict(prompts=2, failed=0), "few": dict(prompts=6, failed=1)})
    problems, ok = hk.check_durations(good, tasks, held, "x")
    assert problems == [] and ok == {"zero": True, "few": False}
    bad = json.loads(json.dumps(good)); bad["prompt_sha256"]["system.txt"] = "0"
    assert any("K4" in p for p in hk.check_durations(bad, tasks, held, "x")[0])
    bad = json.loads(json.dumps(good)); del bad["few"]["1"]["B"]
    assert hk.check_durations(bad, tasks, held, "x")[0]
    assert hk.check_durations(good, tasks, held, "y")[0]


def test_accuracy_ranks_and_ratios():
    held = {"0": {"A"}, "1": {"B", "C"}}
    dur = dict(zero={"A": 10.0, "B": 20.0, "C": 40.0}, few={"0": {"A": 5.0}, "1": {"B": 20.0, "C": 80.0}})
    a = hk.accuracy(dur, ["A", "B", "C"], held, ["0", "1"], {"A": 10.0, "B": 20.0, "C": 40.0})
    assert abs(a["zero"]["spearman"] - 1.0) < 1e-12 and a["zero"]["median_abs_log_ratio"] == 0.0
    assert a["few_held_out"]["worst"]["task"] in ("A", "C")


class HkFreeTests(unittest.TestCase):
    """Runs every test above under unittest (the cluster's SAFE environment has no pytest)."""


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(HkFreeTests, _name, lambda self, _fn=_fn: _fn())
del _name, _fn

if __name__ == "__main__":
    unittest.main()
