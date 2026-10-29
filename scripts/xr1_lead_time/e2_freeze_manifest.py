"""R10 part B: E2's freeze manifest (plan: docs/xr1_review_analyses_plan.md, R10).

E2 freezes, from the existing RLDX-1 data: SAFE's LSTM head, Hide-and-Seek (three seeds), H_k, rho_k
and each detector's fused path (u, s). This script writes one JSON that records each of them with a
hash, after checking that:
- the 15.15 / 15.22 fits reproduce SAFE + P +0.0342 and Hide-and-Seek + P +0.0834 (the paths come
  from these fits);
- the freeze run of score_cohorts (--save-models) retrained Hide-and-Seek on the 15.22 inputs and
  reproduced its calibration and test trajectories exactly (reference checks), its normalizations
  exactly, and saved one model per seed that reloads to identical scores;
- each saved model's file SHA-256 and content hash still match what the freeze run recorded.
It refuses to write the manifest if any check fails. Its own SHA-256 is printed at the end; the paper
cites it.

Usage:
  python3 -I e2_freeze_manifest.py --calibration-dir <SAFE20>/calibration/lstm --hs-scores-dir <15.22>/scores \\
      --replication-json rldx_lead_lstm.json --hs-result rldx_hide_and_seek.json --freeze-scores-dir <freeze>/scores \\
      --tree safe_lstm_seed0=<SAFE20>/final/lstm_seed0 ... --tree policy_snapshot=<HF snapshot dir> \\
      --file split_manifest=<...> ... --git rldx_repo=<RLDX-1 checkout> --output e2_freeze_manifest.json
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parents[1]))
import stage2_frozen_sizing as fs
import stage2_sizing as s2
from robocasa.recovery.safe.baselines.detectors import content_hash

PLAN = "docs/xr1_review_analyses_plan.md (R10 part B, 2026-10-29)"
DETECTOR = "hide_and_seek"
SEEDS = ("0", "1", "2")
COHORTS = ("calibration", "test")
TREE_LISTING_MAX = 200


def tree_hash(root):
    """Sorted relative paths with per-file SHA-256 (symlinks followed); digest over 'path<TAB>sha' lines."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"STOP: not a directory: {root}")
    entries, total = [], 0
    for d, _, names in os.walk(root, followlinks=True):
        for name in names:
            p = Path(d) / name
            entries.append((p.relative_to(root).as_posix(), s2.sha256(p)))
            total += p.stat().st_size
    entries.sort()
    if not entries:
        raise SystemExit(f"STOP: empty directory: {root}")
    import hashlib
    digest = hashlib.sha256("".join(f"{rel}\t{sha}\n" for rel, sha in entries).encode()).hexdigest()
    return dict(path=str(root.resolve()), files=len(entries), bytes=total, digest=digest,
                entries=[dict(path=r, sha256=h) for r, h in entries] if len(entries) <= TREE_LISTING_MAX else None)


def git_state(repo):
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()
    return dict(path=str(Path(repo).resolve()), commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain")))


def labelled(pairs, flag):
    out = {}
    for item in pairs or []:
        label, sep, path = item.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"error: {flag} expects LABEL=PATH, got {item!r}")
        if label in out:
            raise SystemExit(f"error: {flag} label repeats: {label}")
        out[label] = path
    return out


def check_freeze_run(freeze_dir, hs_dir):
    """The freeze run must reproduce 15.22 exactly and hold one reloadable model per seed."""
    freeze_dir, hs_dir = Path(freeze_dir), Path(hs_dir)
    meta = json.loads((freeze_dir / "scoring.json").read_text())
    old = json.loads((hs_dir / "scoring.json").read_text())
    fail = []
    for c in COHORTS:
        chk = meta.get("reference_checks", {}).get(f"{c}/{DETECTOR}")
        if not chk or not chk["ok"] or chk["max_abs_diff"] != 0.0 or chk["rollouts_compared"] != chk["rollouts_scored"]:
            fail.append(f"reference check {c}/{DETECTOR}: {chk}")
    for k in ("feature_aggregation", "seeds", "epochs"):
        if meta.get(k) != old.get(k):
            fail.append(f"{k}: freeze {meta.get(k)} != 15.22 {old.get(k)}")
    for k, v in old["inputs"].items():
        if meta["inputs"].get(k, {}).get("sha256") != v["sha256"]:
            fail.append(f"input {k} differs from 15.22")
    if meta["detectors"].get(DETECTOR, {}).get("configs") != old["detectors"].get(DETECTOR, {}).get("configs"):
        fail.append("detector configs differ from 15.22")
    norm_new = json.loads((freeze_dir / "normalizations.json").read_text()).get(DETECTOR)
    norm_old = json.loads((hs_dir / "normalizations.json").read_text()).get(DETECTOR)
    if norm_new is None or norm_new != norm_old:
        fail.append("normalizations differ from 15.22")
    models = meta["detectors"].get(DETECTOR, {}).get("models", {})
    out = {}
    for s in SEEDS:
        m = models.get(s)
        if not m or not m.get("reload_identical"):
            fail.append(f"seed {s}: no reloadable model")
            continue
        p = Path(m["path"])
        try:
            with np.load(p, allow_pickle=False) as data:
                content = content_hash({k: data[k] for k in data.files if k != "meta_json"})
                config = json.loads(str(data["meta_json"]))["config"]
        except Exception as exc:                      # unreadable or not a saved model
            fail.append(f"seed {s}: cannot read {p} ({type(exc).__name__}: {exc})")
            continue
        sha = s2.sha256(p)
        if sha != m["sha256"] or content != m["content_sha256"]:
            fail.append(f"seed {s}: model file changed since the freeze run")
        out[s] = dict(path=str(p.resolve()), sha256=sha, content_sha256=content, config=config)
    return dict(ok=not fail, failures=fail, code_commit=meta.get("code_commit"),
                scoring_json_sha256=s2.sha256(freeze_dir / "scoring.json"),
                normalizations_sha256=s2.sha256(freeze_dir / "normalizations.json")), out, norm_new


def run(args):
    out_path = s2.check_out(args.output)
    trees, files, gits = labelled(args.tree, "--tree"), labelled(args.file, "--file"), labelled(args.git, "--git")
    rows, train, cal_s, test, tasks, rk = fs.load(args)
    checks, paths = fs.replication(args, cal_s, train, test, rk)
    freeze, models, norms = check_freeze_run(args.freeze_scores_dir, args.hs_scores_dir)
    checks["freeze_run"] = freeze
    print("check: freeze run reproduces 15.22 and holds three reloadable models -> "
          + ("OK" if freeze["ok"] else f"FAILED {freeze['failures']}"), flush=True)
    if not freeze["ok"]:
        sys.exit("STOP: the freeze run does not reproduce 15.22; no manifest written")
    H = {r["task_name"]: int(r["horizon"]) for r in rows}
    types = {r["task_name"]: r["task_type"] for r in rows}
    manifest = dict(
        purpose="E2 freeze manifest: frozen detectors, timer and fused paths for the RLDX-1 Stage 2 collection",
        plan=PLAN, alpha=0.05, q=0.95, tasks={t: dict(type=types[t], H_k=H[t], rho_k=float(rk[t])) for t in tasks},
        detectors={
            "safe_plus_P": dict(score="SAFE-LSTM, seeds 0-2, seed-averaged normalized running maximum (report 15.15)",
                                path=paths["safe"], replication_gain=checks["replication"]["safe"],
                                artifacts=sorted(k for k in trees if k.startswith("safe"))),
            "hide_and_seek_plus_P": dict(score="Hide-and-Seek, seeds 0-2, seed-averaged normalized running maximum (report 15.22)",
                                         path=paths["hide_and_seek"], replication_gain=checks["replication"]["hide_and_seek"],
                                         models=models, normalizations=norms)},
        trees={k: tree_hash(v) for k, v in trees.items()},
        files={k: dict(path=str(Path(v).resolve()), sha256=s2.sha256(v)) for k, v in files.items()},
        git={k: git_state(v) for k, v in gits.items()} | {"this_repository": git_state(here.parents[1])},
        inputs=fs.input_hashes(args), checks=checks)
    s2.write(out_path, manifest)
    print(f"tasks {len(tasks)}; paths SAFE u={paths['safe']['u']} s={paths['safe']['s']}, Hide-and-Seek "
          f"u={paths['hide_and_seek']['u']} s={paths['hide_and_seek']['s']}")
    for s, m in models.items():
        print(f"hide_and_seek seed {s}: content {m['content_sha256'][:16]}  file {m['sha256'][:16]}")
    for k, t in manifest["trees"].items():
        print(f"tree {k}: {t['files']} files, {t['bytes']} bytes, digest {t['digest'][:16]}")
    for k, g in manifest["git"].items():
        print(f"git {k}: {g['commit'][:12]}{' (dirty)' if g['dirty'] else ''}")
    print(f"MANIFEST {out_path} sha256 {s2.sha256(out_path)}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--calibration-dir", "--hs-scores-dir", "--replication-json", "--hs-result", "--freeze-scores-dir", "--output"):
        p.add_argument(name, required=True)
    p.add_argument("--tree", action="append", help="LABEL=DIR, hashed file by file")
    p.add_argument("--file", action="append", help="LABEL=FILE")
    p.add_argument("--git", action="append", help="LABEL=REPO, commit and dirty flag")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
