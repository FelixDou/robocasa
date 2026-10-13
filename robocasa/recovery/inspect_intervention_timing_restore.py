"""Read trusted saved states and observable source; no simulation or model loading."""

import argparse
import ast
import gzip
import hashlib
import os
from pathlib import Path
import pickle

import numpy as np


def inspect_pair(pair_dir, step, robosuite_repo):
    states = []
    for index in range(2):
        path = pair_dir/f"attempt_{index}/states/state_{step:06d}.pkl.gz"
        print("SAVED STATE:", path, "sha256=", hashlib.sha256(path.read_bytes()).hexdigest())
        with gzip.open(path, "rb") as stream:
            states.append(pickle.load(stream))
    left, right = states
    print("CONTROL PATHS AT STATE", step)
    a, b = left["environment_control_state"], right["environment_control_state"]
    identities = lambda records: [(r["object_path"], r["attribute"]) for r in records]
    if identities(a) != identities(b):
        raise ValueError("Control-state identity support differs")
    for x, y in zip(a, b):
        print(x["object_path"], x["attribute"], "left=", x["value"], "right=", y["value"])
    for i, state in enumerate(states):
        print("CONTROLLER IDENTITIES", i, [
            (j, r["object_path"], r["class"]) for j, r in enumerate(state["controller_state"])])
    print("ALL OBSERVATION KEYS AT STATE", step)
    if set(left["observation"]) != set(right["observation"]):
        raise ValueError("Observation-key support differs")
    for key in sorted(left["observation"]):
        x, y = (np.asarray(s["observation"][key]) for s in states)
        result = {"exact": bool(np.array_equal(x, y)), "shape": list(x.shape)}
        if x.shape == y.shape and x.size and x.dtype.kind in "biuf" and y.dtype.kind in "biuf":
            difference = np.abs(x.astype(float)-y.astype(float))
            result.update(max_abs_error=float(difference.max()), mean_abs_error=float(difference.mean()))
        print(key, result)
    path = robosuite_repo/"robosuite/utils/observables.py"
    if path.is_file():
        source = path.read_text()
        print("OBSERVABLE SOURCE:", path, "sha256=", hashlib.sha256(path.read_bytes()).hexdigest())
        for cls in ast.parse(source).body:
            if isinstance(cls, ast.ClassDef) and cls.name == "Observable":
                for method in cls.body:
                    if isinstance(method, ast.FunctionDef) and method.name in ("update", "reset"):
                        print(ast.get_source_segment(source, method))
    else:
        print("Observable source not found at:", path)
    print("Scope: read-only inspection; hashes identify current files, not independently registered checksums.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    repo = Path(os.environ.get("ROBOCASA_REPO", Path(__file__).resolve().parents[2]))
    parser.add_argument("--robosuite-repo", type=Path,
                        default=repo.parent/"robocasa_benchmark_repos/robosuite")
    args = parser.parse_args()
    if args.step < 0:
        parser.error("--step must be nonnegative")
    inspect_pair(args.pair_dir, args.step, args.robosuite_repo)


if __name__ == "__main__":
    main()
