"""R14 of docs/xr1_review_analyses_plan.md: timers that do not use the time limit H_k. Exploratory.

H_k keeps two roles: a rollout fails when it reaches H_k, and every lead is a fraction of H_k. R14 removes
H_k from the monitors' timers. A timer with time unit D and per-task offset rho alarms at inference

    ceil(min(H_k, (rho + lam) D) - EPS),

with lam set by Proposition 1 on calibration successes (the cap only says that the rollout has ended).

    G           D = H_k, rho = 0       the current global timer (gate K1 reproduces it)
    G_abs       D = 1 inference        one elapsed-time cutoff for every task
    G_llm_zero  D = an LLM's duration estimate from the task's instruction alone
    G_llm_few   D = the same with the seen tasks' median success durations as examples (per fold;
                    a seen task's own estimate leaves that task out of the examples)
    P           D = H_k, rho = rho_k   the registered per-task timer (gate K5 reproduces it)
    P_mult      D = rho_k H_k, rho = 1 P with a margin proportional to its own duration, so no H_k term

With D = H_k every formula here is the one in confirm_lead (fit_fast, alarm_vec, V_vec): the EPS margin is
scaled by H_k / D, which is 1 there.

Subcommands:
  instructions  (CPU, RoboCasa installed)  one instruction per task, from the first scene at a fixed seed
  prompts       (CPU)  zero-shot prompt per task, few-shot prompt per fold and task
  llm           (GPU)  greedy answers of a pinned open-weights model, parsed to seconds
The timers are used by seen_unseen_tables.py (XR-1) and rldx_unseen.py (RLDX-1) with --r14-durations.

Usage:
  python -u hk_free.py instructions --folds cross_task_folds.json rldx_folds.json --seed 0 --output instructions.json
  python3 -I hk_free.py prompts --cohort xr1 --instructions instructions.json --folds cross_task_folds.json \\
      --runtime-bundle runtime_bundle.json --output prompts_xr1.json
  python3 -I hk_free.py prompts --cohort rldx --instructions instructions.json --folds rldx_folds.json \\
      --safe-dir <SAFE20>/calibration/lstm --output prompts_rldx.json
  python -u hk_free.py llm --prompts prompts_xr1.json --model <HF snapshot dir> --output durations_xr1.json
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import confirm_lead as cl  # noqa: E402

PLAN = "docs/xr1_review_analyses_plan.md (R14)"
EPS = cl.EPS
PROMPT_DIR = here / "hk_free_prompts"
PROMPT_FILES = ("system.txt", "zero_shot.txt", "few_shot.txt", "retry.txt")
SECONDS_PER_INFERENCE = {"xr1": 0.8, "rldx": 0.4}      # 16 and 8 control steps per inference at 20 Hz
TIMERS = ("G", "G_abs", "G_llm_zero", "G_llm_few")
SEEN_TIMERS = TIMERS + ("P", "P_mult")
MAX_PARSE_FAILURE_SHARE = 0.05
ALL = "all"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompts_sha256():
    return {n: sha256(PROMPT_DIR / n) for n in PROMPT_FILES}


# ---- timers with a time unit -----------------------------------------------------------------

def with_unit(c, D):
    """A copy of confirm_lead.Cohort c with time unit D (one value per row)."""
    out = c.take(np.arange(len(c.L)))
    out.D = np.asarray(D, float)
    if out.D.shape != c.L.shape or not np.all(out.D > 0):
        raise ValueError("Time units must be positive, one per row")
    return out


def sub(c, idx):
    return with_unit(c.take(idx), c.D[idx])


def timer_idx(c, rho, lam):
    return np.ceil(np.minimum(c.H, (rho + lam) * c.D) - EPS)


def alarm(c, rho, lam, u, s):
    a = timer_idx(c, rho, lam)
    if u != math.inf:
        a = np.minimum(a, cl.first_vec(c, u + s * lam))
    return a


def V(c, rho, u, s):
    """Smallest lam at which each success is silent (inf if never)."""
    v_t = c.L / c.D - rho
    if u == math.inf:
        v_s = np.full(len(c.L), -np.inf)
    elif s == 0:
        v_s = np.where(u > c.M, -np.inf, np.inf)
    else:
        v_s = (c.M - u) / s
    v = np.maximum(np.maximum(v_t, v_s), 0.0) + EPS * (c.H / c.D)
    return np.where(c.L >= c.H, np.inf, v)


def fail_lead(c, rho, lam, u, s):
    a = alarm(c, rho, lam, u, s)
    return float(np.mean(np.where(a <= c.L, (c.L - a) / c.H, 0.0)))


def fit(cal, rho):
    """confirm_lead.fit_fast with a time unit: same grid, order, tie-breaking and epsilon."""
    Sm, Fm = ~cal.failed, cal.failed
    S, rS = sub(cal, Sm), rho[Sm]
    t_lam = cl.conformal(V(S, rS, math.inf, 0.0))
    timer = dict(u=math.inf, s=0.0, lam=t_lam)
    if not Fm.any():
        return timer, dict(timer)
    F, rF = sub(cal, Fm), rho[Fm]
    us = sorted(set(np.round(np.quantile(F.M, np.linspace(0, 1, 21)), 4).tolist()))
    best = (fail_lead(F, rF, t_lam, math.inf, 0.0), math.inf, 0.0, t_lam)
    for u in us:
        for s in cl.S_GRID:
            lam = cl.conformal(V(S, rS, u, s))
            if not math.isfinite(lam):
                continue
            v = fail_lead(F, rF, lam, u, s)
            if v > best[0] + 1e-12:
                best = (v, u, s, lam)
    return timer, dict(u=best[1], s=best[2], lam=best[3])


def raw_eval(e, rho, det):
    """cross_task_lead.raw_eval with a time unit."""
    a = alarm(e, rho, det["lam"], det["u"], det["s"])
    hit, F = a <= e.L, e.failed
    return dict(fp=int(hit[~F].sum()), n=int((~F).sum()), lead=np.where(hit, (e.L - a) / e.H, 0.0)[F],
                hit=hit[F], frac=np.where(hit, a / e.H, 1.0)[F], final=int(np.sum(a[F] == e.L[F])))


def pool(parts):
    cat = lambda k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)
    return dict(fp=sum(p["fp"] for p in parts), n=sum(p["n"] for p in parts), lead=cat("lead"),
                hit=cat("hit").astype(bool), frac=cat("frac"), final=sum(p["final"] for p in parts))


def matched_timer(tests, target_fp):
    """The timer alone (u = inf) with one lam for all given test cohorts, the smallest with at most
    target_fp false alarms (test-tuned, as every matched comparison is). tests: [(cohort, rho)]."""
    v = np.concatenate([V(sub(e, ~e.failed), r[~e.failed], math.inf, 0.0) for e, r in tests])
    v = np.sort(v)
    n = len(v)
    if target_fp >= n:
        lam = 0.0
    else:
        lam = float(v[n - target_fp - 1])
        if not math.isfinite(lam):
            return None
    det = dict(u=math.inf, s=0.0, lam=lam)
    return pool([raw_eval(e, r, det) for e, r in tests]), lam


# ---- time units per timer ----------------------------------------------------------------------

def unit_and_rho(c, timer, F, durations, rk):
    """(D, rho) per row of cohort c for one timer and fold."""
    zero = np.zeros(len(c.L))
    if timer == "G":
        return c.H.copy(), zero
    if timer == "G_abs":
        return np.ones(len(c.L)), zero
    if timer == "G_llm_zero":
        return np.array([durations["zero"][t] for t in c.task], float), zero
    if timer == "G_llm_few":
        return np.array([durations["few"][F][t] for t in c.task], float), zero
    rho = np.array([rk[t] for t in c.task], float)
    if timer == "P":
        return c.H.copy(), rho
    if timer == "P_mult":
        return rho * c.H, np.ones(len(c.L))
    raise ValueError(f"unknown timer {timer}")


def prepared(c, timer, F, durations, rk):
    D, rho = unit_and_rho(c, timer, F, durations, rk)
    return with_unit(c, D), rho


def summarize(fused, timer, matched):
    fpr, lead = fused["fp"] / fused["n"], float(fused["lead"].mean())
    tfpr, tlead = timer["fp"] / timer["n"], float(timer["lead"].mean())
    out = dict(fpr=fpr, mean_lead=lead, gain=float(np.mean(fused["lead"] - timer["lead"])), twa=(lead + 1 - fpr) / 2,
               fp=fused["fp"], n_success=fused["n"], n_failure=int(len(fused["lead"])),
               timer=dict(fpr=tfpr, mean_lead=tlead, fp=timer["fp"], twa=(tlead + 1 - tfpr) / 2))
    if matched is not None:
        m, lam = matched
        out["matched"] = dict(gain=float(np.mean(fused["lead"] - m["lead"])), timer_fp=m["fp"],
                              timer_mean_lead=float(m["lead"].mean()), lam=lam)
    else:
        out["matched"] = None
    return out


def run_folds(pairs, timer, durations, rk):
    """pairs: {fold: (calibration Cohort, test Cohort)} for one score key. Fits per fold, pools the tests."""
    pt, pf, tests, paths = [], [], [], {}
    for F, (c, e) in sorted(pairs.items()):
        if len(e.L) == 0:
            continue
        cu, rc = prepared(c, timer, F, durations, rk)
        eu, re_ = prepared(e, timer, F, durations, rk)
        tm, fu = fit(cu, rc)
        pt.append(raw_eval(eu, re_, tm))
        pf.append(raw_eval(eu, re_, fu))
        tests.append((eu, re_))
        paths[F] = dict(timer=cl.det_json(tm), fused=cl.det_json(fu))
    fused, tmr = pool(pf), pool(pt)
    out = summarize(fused, tmr, matched_timer(tests, fused["fp"]))
    out["paths"] = paths
    return out


def mean_std(per_seed, keys=("fpr", "mean_lead", "gain", "twa")):
    out = {k: dict(mean=float(np.mean([p[k] for p in per_seed])), std=float(np.std([p[k] for p in per_seed], ddof=1)))
           for k in keys}
    mg = [p["matched"]["gain"] for p in per_seed if p.get("matched")]
    if len(mg) == len(per_seed):
        out["matched_gain"] = dict(mean=float(np.mean(mg)), std=float(np.std(mg, ddof=1)))
    return out


def block(pairs_of, labels, timers, durations, rk):
    """pairs_of(label) -> {fold: (cal, test)}; labels: 'seed_average', 'seed_0', ... Per timer and label."""
    out = {}
    for timer in timers:
        res = {lab: run_folds(pairs_of(lab), timer, durations, rk) for lab in labels}
        seeds = [lab for lab in labels if lab != "seed_average"]
        if len(seeds) > 1:
            res["mean_std_over_seeds"] = mean_std([res[s] for s in seeds])
        out[timer] = res
    return out


# ---- checks on the duration file ---------------------------------------------------------------

def check_durations(durations, tasks, held, folds_sha):
    """K4 and the fold/task contract. Returns (problems, few_shot_ok)."""
    problems = []
    if durations.get("prompt_sha256") != prompts_sha256():
        problems.append("K4: the prompts' SHA-256 differs from the committed templates")
    if durations.get("folds_sha256") != folds_sha:
        problems.append("the durations were made with another fold file")
    if set(durations.get("zero", {})) != set(tasks):
        problems.append("zero-shot durations do not cover exactly the tasks")
    for F in held:
        if set(durations.get("few", {}).get(F, {})) != set(tasks):
            problems.append(f"few-shot durations of fold {F} do not cover exactly the tasks")
    vals = list(durations.get("zero", {}).values()) + [v for d in durations.get("few", {}).values() for v in d.values()]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in vals):
        problems.append("a duration is not a positive number")
    n = durations.get("answers", 0)
    fails = durations.get("parse_failures", {})
    if not n or any(v.get("prompts") is None for v in fails.values()):
        problems.append("K4: the duration file does not record its answers")
    few_ok = {k: v["failed"] <= MAX_PARSE_FAILURE_SHARE * v["prompts"] for k, v in fails.items()}
    return problems, few_ok


def accuracy(durations, tasks, held, folds, ref_seconds):
    """Estimate against the observed rho_k H_k in seconds: zero-shot per task, few-shot per held-out task."""
    def stats(est):
        t = sorted(est)
        a = np.array([est[k] for k in t]); b = np.array([ref_seconds[k] for k in t])
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        lr = np.log(a / b)
        w = int(np.argmax(np.abs(lr)))
        return dict(tasks=len(t), spearman=float(np.corrcoef(ra, rb)[0, 1]), median_abs_log_ratio=float(np.median(np.abs(lr))),
                    median_ratio=float(np.exp(np.median(lr))), worst=dict(task=t[w], estimate=float(a[w]), observed=float(b[w])),
                    per_task={k: dict(estimate=float(est[k]), observed=float(ref_seconds[k])) for k in t})
    zero = {t: durations["zero"][t] for t in tasks}
    few = {t: durations["few"][F][t] for F in folds for t in held[F]}
    return dict(reference="95th percentile of the task's training-success durations (rho_k H_k), in seconds",
                zero=stats(zero), few_held_out=stats(few))


# ---- instructions (RoboCasa) --------------------------------------------------------------------

def instructions_cmd(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    from robocasa.utils.env_utils import create_env  # noqa: needs RoboCasa
    tasks = set(args.tasks or [])
    for f in args.folds or []:
        tasks |= set(json.loads(Path(f).read_text())["tasks"])
    if not tasks:
        raise SystemExit("error: give --tasks or --folds")
    out = {}
    for t in sorted(tasks):
        env = create_env(t, seed=args.seed, has_offscreen_renderer=False, use_camera_obs=False)
        env.reset()
        lang = env.get_ep_meta().get("lang", "")
        env.close()
        if not lang:
            raise SystemExit(f"error: {t} has no instruction")
        out[t] = lang
        print(f"{t}: {lang}")
    out_path.write_text(json.dumps(dict(plan=PLAN, seed=args.seed, rule="ep_meta lang of the first reset at this seed",
                                        instructions=out), indent=1) + "\n")
    print(f"wrote {out_path} ({len(out)} tasks)")
    return 0


# ---- prompts -------------------------------------------------------------------------------------

def median_seconds_xr1(bundle_path, spi):
    import quantile_timer as qt
    bundle = json.loads(Path(bundle_path).read_text())
    dev = qt.dev_fractions(bundle)
    H = bundle["task_horizons"]
    return {t: float(np.median(np.asarray(v) * H[t])) * spi for t, v in dev.items() if v}


def median_seconds_rldx(safe_dir, spi):
    import rldx_hide_and_seek as hs
    rows, _, tasks = hs.load_rows(safe_dir)
    out = {}
    for t in tasks:
        L = [r["L"] for r in rows if r["split"] == "train" and r["task_name"] == t and not r["failed"]]
        if L:
            out[t] = float(np.median(L)) * spi
    return out


def fmt_seconds(x):
    return f"{x:.0f} s"


def build_prompts(instructions, held, med):
    """Zero-shot per task; few-shot per fold and task (a task never among its own examples)."""
    tmpl = {n: (PROMPT_DIR / n).read_text().strip() for n in PROMPT_FILES}
    tasks = sorted(instructions)
    zero = {t: tmpl["zero_shot.txt"].format(instruction=instructions[t]) for t in tasks}
    few = {}
    for F in sorted(held):
        seen = [t for t in tasks if t not in held[F]]
        few[F] = {}
        for t in tasks:
            ex = [s for s in seen if s != t and s in med]
            lines = "\n".join(f'- "{instructions[s]}": {fmt_seconds(med[s])}' for s in ex)
            few[F][t] = tmpl["few_shot.txt"].format(examples=lines, instruction=instructions[t])
    return dict(system=tmpl["system.txt"], retry=tmpl["retry.txt"], zero=zero, few=few,
                examples={F: {t: [s for s in tasks if s not in held[F] and s != t and s in med] for t in tasks}
                          for F in sorted(held)})


def prompts_cmd(args):
    import cross_task_lead as ctl
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    spec = json.loads(Path(args.folds).read_text())
    tasks, held = ctl.fold_sets(spec)
    ins = json.loads(Path(args.instructions).read_text())["instructions"]
    missing = sorted(tasks - set(ins))
    if missing:
        raise SystemExit(f"error: no instruction for {missing}")
    ins = {t: ins[t] for t in sorted(tasks)}
    spi = SECONDS_PER_INFERENCE[args.cohort]
    if args.cohort == "xr1":
        if not args.runtime_bundle:
            raise SystemExit("error: --runtime-bundle is required for xr1")
        med = median_seconds_xr1(args.runtime_bundle, spi)
        src = dict(runtime_bundle=dict(path=str(Path(args.runtime_bundle).resolve()), sha256=sha256(args.runtime_bundle)))
    else:
        if not args.safe_dir:
            raise SystemExit("error: --safe-dir is required for rldx")
        med = median_seconds_rldx(args.safe_dir, spi)
        src = dict(safe_dir=str(Path(args.safe_dir).resolve()))
    p = build_prompts(ins, held, med)
    out = dict(plan=PLAN, cohort=args.cohort, seconds_per_inference=spi, folds_sha256=sha256(args.folds),
               instructions_sha256=sha256(args.instructions), prompt_sha256=prompts_sha256(), sources=src,
               example_median_seconds=med, instructions=ins, **p)
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    n = len(p["zero"]) + sum(len(v) for v in p["few"].values())
    print(f"wrote {out_path}: {len(ins)} tasks, folds {sorted(held)}, {n} prompts")
    return 0


# ---- LLM -----------------------------------------------------------------------------------------

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def parse_seconds(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = NUMBER.search(text)
    if not m:
        return None
    v = float(m.group(0))
    return v if math.isfinite(v) and v > 0 else None


def answer_all(prompts, generate):
    """generate(messages) -> text. Returns durations, raw records and parse-failure counts."""
    sys_msg = prompts["system"]
    raw, fails = [], {"zero": dict(prompts=0, failed=0), "few": dict(prompts=0, failed=0)}

    def ask(kind, fold, task, text, fallback):
        msgs = [dict(role="system", content=sys_msg), dict(role="user", content=text)]
        out1 = generate(msgs)
        v = parse_seconds(out1)
        out2 = None
        if v is None:
            msgs += [dict(role="assistant", content=out1), dict(role="user", content=prompts["retry"])]
            out2 = generate(msgs)
            v = parse_seconds(out2)
        fails[kind]["prompts"] += 1
        failed = v is None
        if failed:
            fails[kind]["failed"] += 1
        raw.append(dict(kind=kind, fold=fold, task=task, answer=out1, retry_answer=out2, parsed=v, fallback=failed))
        return v

    zero = {t: ask("zero", None, t, p, None) for t, p in sorted(prompts["zero"].items())}
    good = [v for v in zero.values() if v is not None]
    for t, v in zero.items():
        if v is None:
            zero[t] = float(np.median(good)) if good else 1.0          # rule: median of the other zero-shot answers
    few = {}
    med = prompts["example_median_seconds"]
    for F, by_task in sorted(prompts["few"].items()):
        few[F] = {}
        for t, p in sorted(by_task.items()):
            v = ask("few", F, t, p, None)
            if v is None:                                              # rule: median of the prompt's examples
                ex = prompts["examples"][F][t]
                v = float(np.median([med[s] for s in ex])) if ex else zero[t]
            few[F][t] = v
    return zero, few, raw, fails


def llm_cmd(args):
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"error: output exists: {out_path}")
    prompts = json.loads(Path(args.prompts).read_text())
    if prompts["prompt_sha256"] != prompts_sha256():
        raise SystemExit("error: prompts were built from other templates (K4)")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    model_dir = Path(args.model)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    def generate(msgs):
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            enc = tok.apply_chat_template(msgs, **kw)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    zero, few, raw, fails = answer_all(prompts, generate)
    files = {n: sha256(model_dir / n) for n in ("config.json", "generation_config.json") if (model_dir / n).exists()}
    out = dict(plan=PLAN, cohort=prompts["cohort"], model=dict(path=str(model_dir.resolve()), files_sha256=files,
                                                               snapshot=model_dir.resolve().name),
               generation=dict(do_sample=False, max_new_tokens=args.max_new_tokens, thinking=False),
               prompts_file=dict(path=str(Path(args.prompts).resolve()), sha256=sha256(args.prompts)),
               prompt_sha256=prompts["prompt_sha256"], folds_sha256=prompts["folds_sha256"],
               seconds_per_inference=prompts["seconds_per_inference"], answers=len(raw), parse_failures=fails,
               zero=zero, few=few, raw=raw)
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    for k, v in fails.items():
        print(f"{k}: {v['prompts']} prompts, {v['failed']} not parsed")
    for t in sorted(zero):
        print(f"  {t:32s} zero {zero[t]:7.1f} s | example median {prompts['example_median_seconds'].get(t, float('nan')):7.1f} s")
    print(f"wrote {out_path}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub_ = p.add_subparsers(dest="cmd", required=True)
    a = sub_.add_parser("instructions")
    a.add_argument("--tasks", nargs="+")
    a.add_argument("--folds", nargs="+", help="fold files whose tasks are all needed")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--output", required=True)
    b = sub_.add_parser("prompts")
    b.add_argument("--cohort", choices=sorted(SECONDS_PER_INFERENCE), required=True)
    b.add_argument("--instructions", required=True)
    b.add_argument("--folds", required=True)
    b.add_argument("--runtime-bundle")
    b.add_argument("--safe-dir")
    b.add_argument("--output", required=True)
    c = sub_.add_parser("llm")
    c.add_argument("--prompts", required=True)
    c.add_argument("--model", required=True, help="local HF snapshot directory (pinned revision)")
    c.add_argument("--max-new-tokens", type=int, default=16)
    c.add_argument("--output", required=True)
    args = p.parse_args(argv)
    return dict(instructions=instructions_cmd, prompts=prompts_cmd, llm=llm_cmd)[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
