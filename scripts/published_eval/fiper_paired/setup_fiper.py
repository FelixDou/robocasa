"""Apply the paired-comparison additions to a pinned FIPER clone (plan: docs/fiper_paired_plan.md).

Adds an elapsed-time method (`time`), restricts the method list to rnd_oe, entropy and time
(FIPER's AND/OR combinations stay as configured), adds `time` to the summary filter, and adds
an opt-in per-seed dump of first-alarm indices. Every edit asserts its anchor, so a different
FIPER version fails loudly instead of being patched silently. Also links FIPER's data/
directory to a run folder whose task rollouts point at the released data.

Usage:
  python3 -I setup_fiper.py --repo <fiper clone> --rollouts-root <extracted/data_all> --run-dir <run folder>
"""
import argparse, subprocess, sys
from pathlib import Path

PINNED = "13d79c5c3069def843e454787ff128defc249838"
TASKS = ("sorting", "stacking", "push_t", "pretzel", "push_chair")

TIME_EVAL = '''from .base_eval_class import BaseEvalClass


class TIMEEval(BaseEvalClass):
    """Elapsed-time baseline added for a paired comparison: the score at policy step t (0-based)
    is t + 1. It reads no features and has no model; thresholds, windows and metrics are FIPER's."""

    def calculate_uncertainty_score(self, rollout_tensor_dict, **kwargs):
        raise NotImplementedError("TIMEEval scores whole rollouts in _process_one_rollout")

    def _process_one_rollout(self, rollout_dict):
        rollout_length = rollout_dict[self.required_tensors[0]].shape[0]
        num_robots = self.dataset.data["metadata"].get("num_robots", 1)
        small = 1e-6 if self.cfg.get("handle_zero_thresholds", {"style": ""})["style"] == "add_small_score" else 0.0
        scores = [float(t + 1) + small for t in range(rollout_length // num_robots)]
        return scores, 0.0
'''

TIME_CFG = '''defaults:
  - base

# Elapsed-time baseline (added for a paired comparison). The tensor is read only for the
# rollout length; the score is the policy step index + 1.
required_tensors:
  - "obs_embeddings"

normalize_tensors:
  obs_embeddings: False
'''

DUMP_FN = '''

def _dump_first_alarms(results, task, seed):
    """Added for a paired comparison: per-rollout first-alarm index (-1 = none) for every method,
    threshold style, quantile and window, with success labels and lengths. Written only when the
    FIPER_FIRST_ALARM_DIR environment variable is set; changes no computation."""
    dump_dir = os.environ.get("FIPER_FIRST_ALARM_DIR")
    if not dump_dir:
        return
    import pickle
    import numpy as np
    out = {}
    for method, res in results.items():
        sbt = res.get("test_scores_by_threshold")
        if not sbt:
            continue
        alarms, lengths = {}, None
        for style, by_q in sbt.items():
            for q, by_w in by_q.items():
                for w, episodes in by_w.items():
                    first = []
                    for s in episodes:
                        above = np.asarray(s) > 1
                        first.append(int(np.argmax(above)) if above.any() else -1)
                    alarms[(str(style), float(q) if not isinstance(q, str) else q, str(w))] = np.asarray(first, dtype=np.int64)
                    if lengths is None:
                        lengths = np.asarray([len(s) for s in episodes], dtype=np.int64)
        out[str(method)] = {
            "first_alarm": alarms,
            "lengths": lengths,
            "successful": np.asarray(res["successful_test_rollouts"], dtype=bool),
            "max_episode_length": int(res["max_episode_length"]),
        }
    os.makedirs(dump_dir, exist_ok=True)
    with open(os.path.join(dump_dir, f"seed{seed}_{task}.pkl"), "wb") as f:
        pickle.dump(out, f)
'''


def edit(path, old, new, count=1):
    text = path.read_text()
    if text.count(old) != count:
        sys.exit(f"error: anchor found {text.count(old)} times (expected {count}) in {path}: {old[:60]!r}")
    path.write_text(text.replace(old, new))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True); p.add_argument("--rollouts-root", required=True); p.add_argument("--run-dir", required=True)
    a = p.parse_args()
    repo, roots, run = Path(a.repo).resolve(), Path(a.rollouts_root).resolve(), Path(a.run_dir).resolve()
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != PINNED:
        sys.exit(f"error: FIPER HEAD is {head}, expected {PINNED}")
    if subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout.strip():
        sys.exit("error: the FIPER clone has local changes; start from a clean clone")
    (repo / "evaluation/method_eval_classes/time_eval.py").write_text(TIME_EVAL)
    (repo / "configs/eval/time.yaml").write_text(TIME_CFG)
    edit(repo / "evaluation/method_eval_classes/__init__.py", "from .logpzo_eval import *\n",
         "from .logpzo_eval import *\nfrom .time_eval import *\n")
    d = repo / "configs/default.yaml"
    # Whole-line anchors: 'rnd_models: [...]' also occurs inside 'available_rnd_models: [...]'.
    edit(d, '\nrnd_models: ["rnd_oe", "rnd_a"]\n', '\nrnd_models: ["rnd_oe"]\n')
    edit(d, '\nmethods: ["entropy", "similarity", "tc", "logpzo"]\n', '\nmethods: ["entropy", "time"]\n')
    edit(d, '\nimplemented_methods: ["rnd_oe", "rnd_a", "entropy", "tc", "similarity", "logpzo"]\n',
         '\nimplemented_methods: ["rnd_oe", "rnd_a", "entropy", "tc", "similarity", "logpzo", "time"]\n')
    edit(repo / "configs/results/base.yaml", "    - rnd_oe_and_entropy\n", "    - rnd_oe_and_entropy\n    - time\n")
    r = repo / "scripts/run_fiper.py"
    edit(r, "            total_results[task] = results\n",
         "            total_results[task] = results\n            _dump_first_alarms(results, task, seed)\n")
    edit(r, "\ndef check_inputs(", DUMP_FN + "\n\ndef check_inputs(")
    # Data: FIPER reads and writes under <repo>/data; point it at a /gs/bs run folder.
    data = run / "data"
    for task in TASKS:
        src = roots / task / "rollouts"
        if not src.is_dir():
            sys.exit(f"error: missing released rollouts {src}")
        (data / task).mkdir(parents=True, exist_ok=True)
        link = data / task / "rollouts"
        if not link.exists():
            link.symlink_to(src)
    target = repo / "data"
    if target.is_symlink() or target.exists():
        if not (target.is_symlink() and target.resolve() == data):
            sys.exit(f"error: {target} already exists and does not point to {data}")
    else:
        target.symlink_to(data)
    print(f"FIPER {PINNED[:8]} prepared: time method added, methods = rnd_oe, entropy, time; data -> {data}")
    print(subprocess.run(["git", "-C", str(repo), "status", "--short"], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
