"""Inspect externally released rollout files without executing untrusted code.

Plan: docs/published_eval_reanalysis_plan.md. Pickles are loaded with a restricted unpickler
that admits only numpy arrays/scalars and built-in containers; any other class is refused and
reported (a stop-and-report condition). Prints the file-name patterns and, for a sample of
files, the nested structure (keys, types, array shapes and dtypes).

Usage (run from outside the data directory, with python -I):
  python3 -I inspect_external.py <data_dir> [--sample 6] [--max-depth 4]
"""
import argparse, collections, io, json, pickle, random, re, sys
from pathlib import Path

import numpy as np

ALLOWED = {
    ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"), ("numpy", "core"), ("collections", "OrderedDict"),
    ("collections", "defaultdict"), ("builtins", "list"), ("builtins", "dict"), ("builtins", "tuple"),
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "slice"), ("builtins", "complex"),
}


class Refused(Exception):
    pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED or (module in ("numpy", "numpy.core.numeric", "numpy._core.numeric") and name in ("dtype", "ndarray", "_frombuffer")):
            return super().find_class(module, name)
        raise Refused(f"{module}.{name}")


def safe_load(path):
    with open(path, "rb") as stream:
        return SafeUnpickler(io.BytesIO(stream.read())).load()


def describe(x, depth, max_depth, max_items=12):
    pad = "  " * depth
    if isinstance(x, np.ndarray):
        return f"ndarray shape={x.shape} dtype={x.dtype}"
    if isinstance(x, (np.generic, int, float, bool, str, type(None))):
        s = repr(x)
        return f"{type(x).__name__} {s[:60]}"
    if depth >= max_depth:
        return f"{type(x).__name__} (len {len(x)})" if hasattr(x, "__len__") else type(x).__name__
    if isinstance(x, dict):
        lines = [f"dict (len {len(x)})"]
        for k in list(x)[:max_items]:
            lines.append(f"{pad}  {k!r}: {describe(x[k], depth + 1, max_depth)}")
        if len(x) > max_items:
            lines.append(f"{pad}  ... {len(x) - max_items} more keys")
        return "\n".join(lines)
    if isinstance(x, (list, tuple)):
        head = f"{type(x).__name__} (len {len(x)})"
        if not x:
            return head
        return head + f"\n{pad}  [0]: {describe(x[0], depth + 1, max_depth)}" + (
            f"\n{pad}  [-1]: {describe(x[-1], depth + 1, max_depth)}" if len(x) > 1 else "")
    return type(x).__name__


def pattern(name):
    return re.sub(r"\d+", "<n>", name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir"); p.add_argument("--sample", type=int, default=6); p.add_argument("--max-depth", type=int, default=4)
    a = p.parse_args()
    root = Path(a.data_dir).resolve()
    if Path.cwd().resolve() == root or root in Path.cwd().resolve().parents:
        sys.exit("error: run from outside the data directory")
    files = sorted(f for f in root.rglob("*") if f.is_file())
    by_ext = collections.Counter(f.suffix for f in files)
    print("files:", len(files), dict(by_ext))
    dirs = collections.Counter(str(f.parent.relative_to(root)) for f in files)
    print("directories (top 20 by file count):")
    for d, n in dirs.most_common(20):
        print(f"  {n:6d}  {d}")
    pats = collections.Counter(pattern(f.name) for f in files)
    print("file-name patterns (top 20):")
    for pt, n in pats.most_common(20):
        print(f"  {n:6d}  {pt}")
    pkls = [f for f in files if f.suffix in (".pkl", ".pickle")]
    rng = random.Random(0)
    for f in rng.sample(pkls, min(a.sample, len(pkls))):
        print(f"\n== {f.relative_to(root)} ({f.stat().st_size / 1e6:.1f} MB)")
        try:
            print(describe(safe_load(f), 0, a.max_depth))
        except Refused as e:
            print(f"REFUSED class {e}: stop and report; do not load this file another way")
    for f in [f for f in files if f.suffix == ".json"][:3]:
        print(f"\n== {f.relative_to(root)} (json)")
        print(describe(json.loads(f.read_text()), 0, a.max_depth))


if __name__ == "__main__":
    main()
