"""Offline entry-time stall gate for trial 2's ArrangeTea mug stage.

Trial 2 (docs/stage_gated_reprompt_trial2_results_20261105.md) showed that
re-prompting at mug-stage entry helps the seeds whose stage later overruns 617
steps and harms the others. This module asks whether the state at stage entry
predicts the overrun. For a decision at the stage's first inference, the
counterfactual of re-prompting is observed exactly: it is the entry-gate arm
of the same seed. Plan: docs/stage_gate_entry_predictor.md.

Two steps:

- ``extract`` (cluster): reduce the stall arm's recorded XR-1 SAFE tokens to
  the frozen SAFE selector (last flow step, last action token), optionally
  score them with the frozen SAFE ensemble, and write one small ``.npz``.
- ``analyze`` (local): repeated cross-validation of pre-specified gates;
  policy value is the entry-gate outcome where the gate fires and the
  baseline outcome elsewhere, over all seeds of the task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TASK = "ArrangeTea"
STAGE = "mug_on_tray"
BASELINE, ENTRY, STALL = "baseline", "gated_reprompt", "stall_gated_reprompt"
SIGNALS = ("safe", "entry_step", "probe", "probe_plus_time")
DECISIONS = (0, 4)  # inference index in stage; 0 is exact, 4 is an estimate
C_GRID = (0.001, 0.01, 0.1, 1.0)
PCA_COMPONENTS = 16
PROCEED_MIN_GAIN = 0.04


def load_trial(root):
    """Return {(task, seed): {arm: rollout}} for the pairs listed in pairs.jsonl."""
    root = Path(root)
    uids = set()
    rows = {}
    shards = sorted(p for p in root.iterdir() if p.is_dir() and (p / "pairs.jsonl").exists())
    if not shards:
        raise ValueError(f"No shard with pairs.jsonl under {root}")
    for shard in shards:
        for line in (shard / "pairs.jsonl").read_text().splitlines():
            if line.strip():
                uids.add(json.loads(line)["pair_uid"])
    for shard in shards:
        for line in (shard / "rollouts.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["pair_uid"] in uids:
                row["_shard_dir"] = str(shard)
                rows.setdefault((row["task_name"], int(row["environment_seed"])), {})[row["arm"]] = row
    return rows


# ---- extract (cluster) --------------------------------------------------------------------

def extract(root, output, runtime_bundle=None, safe_repo=None, device="cpu"):
    output = Path(output)
    if output.exists():
        raise SystemExit(f"error: output exists: {output}")
    trial = load_trial(root)
    scorer = None
    if runtime_bundle:
        from robocasa.recovery.safe.xr1_best_of_k import FrozenXr1SafeEnsemble

        scorer = FrozenXr1SafeEnsemble(runtime_bundle=runtime_bundle, safe_repo=safe_repo, device=device)
    cols = {k: [] for k in ("seed", "inference_in_stage", "steps_in_stage", "step", "x_last", "safe", "safe_prob")}
    provenance = None
    for (task, seed), arms in sorted(trial.items()):
        row = arms.get(STALL)
        if task != TASK or row is None or not row.get("features_path"):
            continue
        with np.load(Path(row["_shard_dir"]) / row["features_path"]) as data:
            feats = np.asarray(data["features"])
        meta = row["feature_meta"]
        if len(meta) != len(feats):
            raise ValueError(f"{task} {seed}: {len(feats)} features but {len(meta)} metadata rows")
        keep = [i for i, m in enumerate(meta) if m["stage"] == STAGE and not m["reprompted"]]
        if not keep:
            continue
        sel = feats[keep]
        if sel.ndim != 4:
            raise ValueError(f"{task} {seed}: expected (n, flow, horizon, dim) features, got {sel.shape}")
        if scorer is not None:
            res = scorer.score_candidates(sel.astype(np.float32), task_name=TASK)
            safe = np.asarray(res["scores"], dtype=np.float64)
            safe_prob = np.mean(np.asarray(res["probability_normalized_scores_by_seed"], dtype=np.float64), axis=0)
            provenance = res["provenance"]
        else:
            safe = safe_prob = np.full(len(keep), np.nan)
        for j, i in enumerate(keep):
            m = meta[i]
            cols["seed"].append(seed)
            cols["inference_in_stage"].append(m["inference_in_stage"])
            cols["steps_in_stage"].append(m["steps_in_stage"])
            cols["step"].append(m["step"])
            cols["x_last"].append(sel[j, -1, -1, :].astype(np.float16))
            cols["safe"].append(safe[j])
            cols["safe_prob"].append(safe_prob[j])
    if not cols["seed"]:
        raise ValueError(f"No {TASK} {STALL} features under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        seed=np.array(cols["seed"], dtype=np.int64),
        inference_in_stage=np.array(cols["inference_in_stage"], dtype=np.int64),
        steps_in_stage=np.array(cols["steps_in_stage"], dtype=np.int64),
        step=np.array(cols["step"], dtype=np.int64),
        x_last=np.stack(cols["x_last"]),
        safe=np.array(cols["safe"], dtype=np.float64),
        safe_prob=np.array(cols["safe_prob"], dtype=np.float64),
        provenance=np.array(json.dumps(provenance or {})),
    )
    n_seeds = len(set(cols["seed"]))
    print(json.dumps({"output": str(output), "rows": len(cols["seed"]), "seeds": n_seeds,
                      "safe_scored": scorer is not None}))


# ---- cohort ---------------------------------------------------------------------------------

def build_cohort(trial, feats, decision):
    """Per-seed table for one decision point (inference index in stage)."""
    by_seed = {}
    for i, seed in enumerate(feats["seed"]):
        by_seed.setdefault(int(seed), []).append(i)
    seeds, entered, label, y_base, y_entry, entry_step, safe, x = [], [], [], [], [], [], [], []
    dim = feats["x_last"].shape[1]
    for (task, seed), arms in sorted(trial.items()):
        if task != TASK:
            continue
        stall, base, entry = arms[STALL], arms[BASELINE], arms[ENTRY]
        gates = [g for g in stall["gate_stages"] if g["stage"] == STAGE]
        idx = [i for i in by_seed.get(seed, []) if feats["inference_in_stage"][i] <= decision]
        is_in = bool(gates) and any(feats["inference_in_stage"][i] == 0 for i in idx)
        seeds.append(seed)
        entered.append(is_in)
        label.append(bool(gates and gates[0].get("switched")))
        y_base.append(int(bool(base["success"])))
        y_entry.append(int(bool(entry["success"])))
        entry_step.append(float(gates[0]["entry_step"]) if gates else np.nan)
        if is_in:
            safe.append(float(np.mean(feats["safe"][idx])))
            x.append(np.mean(feats["x_last"][idx].astype(np.float64), axis=0))
        else:
            safe.append(np.nan)
            x.append(np.full(dim, np.nan))
    return {
        "seed": np.array(seeds), "entered": np.array(entered), "label": np.array(label),
        "y_base": np.array(y_base), "y_entry": np.array(y_entry), "entry_step": np.array(entry_step),
        "safe": np.array(safe), "x": np.stack(x),
    }


# ---- gates ----------------------------------------------------------------------------------

def fit_threshold(scores, gains):
    """Threshold maximizing sum(gains[scores >= t]); ties go to fewer re-prompts."""
    order = np.argsort(-scores, kind="stable")
    s, g = scores[order], gains[order]
    best_value, best_t = 0.0, np.inf
    running = 0.0
    for k in range(len(s)):
        running += g[k]
        if k + 1 < len(s) and s[k + 1] == s[k]:
            continue
        if running > best_value + 1e-12:
            best_value, best_t = running, s[k]
    return best_t


def _probe(train_x, train_y, test_x, rng_seed, train_extra=None, test_extra=None):
    """Standardize -> PCA on the SAFE token, append extra columns, L2 logistic (inner-CV C)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_x)
    n_comp = min(PCA_COMPONENTS, train_x.shape[1], len(train_x) - 1)
    pca = PCA(n_components=n_comp, random_state=rng_seed).fit(scaler.transform(train_x))
    z_tr, z_te = pca.transform(scaler.transform(train_x)), pca.transform(scaler.transform(test_x))
    if train_extra is not None:
        z_tr, z_te = np.column_stack([z_tr, train_extra]), np.column_stack([z_te, test_extra])
    post = StandardScaler().fit(z_tr)
    model = LogisticRegressionCV(Cs=list(C_GRID), cv=5, scoring="neg_log_loss", max_iter=2000)
    model.fit(post.transform(z_tr), train_y)
    return model.predict_proba(post.transform(z_tr))[:, 1], model.predict_proba(post.transform(z_te))[:, 1]


def gate_scores(signal, cohort, train, test, rng_seed):
    """Return (train scores, test scores); higher means re-prompt."""
    if signal == "safe":
        s = cohort["safe"]
        if np.isnan(s[train]).any():
            raise ValueError("SAFE scores are missing; run extract with --runtime-bundle")
        return s[train], s[test]
    if signal == "entry_step":
        return cohort["entry_step"][train], cohort["entry_step"][test]
    if signal not in ("probe", "probe_plus_time"):
        raise ValueError(f"Unknown signal {signal!r}")
    x, y = cohort["x"], cohort["label"][train].astype(int)
    if signal == "probe":
        return _probe(x[train], y, x[test], rng_seed)
    t = cohort["entry_step"]
    return _probe(x[train], y, x[test], rng_seed, t[train, None], t[test, None])


def _stratified_folds(labels, folds, rng):
    assign = np.empty(len(labels), dtype=int)
    for value in (False, True):
        idx = np.flatnonzero(labels == value)
        rng.shuffle(idx)
        assign[idx] = np.arange(len(idx)) % folds
    return assign


def cross_validate(signal, cohort, repeats, folds, seed, x_perm=None):
    """Out-of-fold decisions; returns per-seed mean gain and diagnostics."""
    entered = np.flatnonzero(cohort["entered"])
    gains = (cohort["y_entry"] - cohort["y_base"]).astype(float)
    work = cohort
    if x_perm is not None:  # permutation null: break signal <-> outcome link
        work = dict(cohort)
        for key in ("x", "safe", "entry_step"):
            arr = cohort[key].copy()
            arr[entered] = cohort[key][entered][x_perm]
            work[key] = arr  # labels and outcomes stay with their seeds
    n = len(gains)
    fire_sum = np.zeros(n)
    aucs = []
    rng = np.random.default_rng(seed)
    for r in range(repeats):
        assign = _stratified_folds(work["label"][entered], folds, rng)
        oof = np.full(len(entered), np.nan)
        fire = np.zeros(n, dtype=bool)
        for f in range(folds):
            tr, te = entered[assign != f], entered[assign == f]
            s_tr, s_te = gate_scores(signal, work, tr, te, rng_seed=seed + r)
            t = fit_threshold(np.asarray(s_tr, float), gains[tr])
            fire[te] = np.asarray(s_te) >= t
            oof[assign == f] = s_te
        fire_sum += fire
        lab = work["label"][entered]
        if 0 < lab.sum() < len(lab):
            from sklearn.metrics import roc_auc_score

            aucs.append(float(roc_auc_score(lab, oof)))
    fire_rate = fire_sum / repeats
    per_seed_gain = fire_rate * gains
    return {"per_seed_gain": per_seed_gain, "fire_rate": fire_rate, "auroc_stall": float(np.mean(aucs)) if aucs else None}


def _bootstrap(values, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def evaluate_signal(signal, cohort, repeats, folds, permutations, seed):
    cv = cross_validate(signal, cohort, repeats, folds, seed)
    gain = cv["per_seed_gain"]
    obs = float(gain.mean())
    entered = np.flatnonzero(cohort["entered"])
    rng = np.random.default_rng(seed + 1)
    null = []
    for p in range(permutations):
        perm = rng.permutation(len(entered))
        null.append(float(cross_validate(signal, cohort, 1, folds, seed + 1000 + p, x_perm=perm)["per_seed_gain"].mean()))
    null = np.array(null)
    gains = (cohort["y_entry"] - cohort["y_base"])
    fr = cv["fire_rate"]
    return {
        "signal": signal,
        "delta_vs_baseline": round(obs, 4),
        "ci95": [round(v, 4) for v in _bootstrap(gain, 5000, seed + 2)],
        "permutation_p": float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null))) if permutations else None,
        "null_mean": round(float(null.mean()), 4) if permutations else None,
        "fire_fraction_of_entered": round(float(fr[entered].mean()), 4),
        "expected_helped": round(float(np.sum(fr * (gains > 0))), 2),
        "expected_harmed": round(float(np.sum(fr * (gains < 0))), 2),
        "auroc_stall": None if cv["auroc_stall"] is None else round(cv["auroc_stall"], 4),
    }


def references(cohort):
    gains = cohort["y_entry"] - cohort["y_base"]
    n = len(gains)
    return {
        "n_seeds": int(n),
        "n_entered": int(cohort["entered"].sum()),
        "n_stall": int(cohort["label"].sum()),
        "baseline_success": round(float(cohort["y_base"].mean()), 4),
        "always_entry_gate": round(float(gains.mean()), 4),
        "oracle_stall_label": round(float(np.sum(gains[cohort["label"]]) / n), 4),
        "oracle_entry_where_it_helps": round(float(np.sum(np.maximum(gains, 0)) / n), 4),
    }


def analyze(trial_root, features, out_dir, repeats=10, folds=5, permutations=200, seed=0, signals=SIGNALS):
    trial = load_trial(trial_root)
    with np.load(features) as data:
        feats = {k: data[k] for k in data.files}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have_safe = not np.isnan(feats["safe"]).all()
    signals = [s for s in signals if s != "safe" or have_safe]
    result = {"plan": "docs/stage_gate_entry_predictor.md", "repeats": repeats, "folds": folds,
              "permutations": permutations, "seed": seed, "safe_scored": bool(have_safe), "decisions": {}}
    for decision in DECISIONS:
        cohort = build_cohort(trial, feats, decision)
        rows = [evaluate_signal(s, cohort, repeats, folds, permutations, seed) for s in signals]
        if permutations:
            adj = holm(np.array([r["permutation_p"] for r in rows]))
            for r, a in zip(rows, adj):
                r["holm_p"] = round(float(a), 4)
        result["decisions"][str(decision)] = {
            "exact_counterfactual": decision == 0,
            "references": references(cohort),
            "signals": rows,
        }
        print(f"decision at inference {decision}:", flush=True)
        for r in rows:
            print(f"  {r['signal']:16s} delta {r['delta_vs_baseline']:+.3f} {r['ci95']} p {r['permutation_p']} "
                  f"holm {r.get('holm_p')} fire {r['fire_fraction_of_entered']:.2f} auroc {r['auroc_stall']}", flush=True)
    primary = result["decisions"]["0"]["signals"]
    passing = [r["signal"] for r in primary
               if r["delta_vs_baseline"] >= PROCEED_MIN_GAIN and (r.get("holm_p") or 1.0) < 0.05]
    result["proceed_to_trial3"] = {"criterion": f"decision 0, delta >= {PROCEED_MIN_GAIN} and Holm p < 0.05",
                                   "passing_signals": passing}
    (out_dir / "entry_predictor.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["proceed_to_trial3"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--run-root", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--runtime-bundle")
    e.add_argument("--safe-repo")
    e.add_argument("--device", default="cpu")
    a = sub.add_parser("analyze")
    a.add_argument("--run-root", required=True)
    a.add_argument("--features", required=True)
    a.add_argument("--output-dir", required=True)
    a.add_argument("--repeats", type=int, default=10)
    a.add_argument("--folds", type=int, default=5)
    a.add_argument("--permutations", type=int, default=200)
    a.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "extract":
        if bool(args.runtime_bundle) != bool(args.safe_repo):
            parser.error("--runtime-bundle and --safe-repo go together")
        extract(args.run_root, args.output, args.runtime_bundle, args.safe_repo, args.device)
    else:
        analyze(args.run_root, args.features, args.output_dir, args.repeats, args.folds, args.permutations, args.seed)


if __name__ == "__main__":
    main()
