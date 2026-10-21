import importlib.util
import os
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "published_eval" / "inspect_external.py"


def load():
    spec = importlib.util.spec_from_file_location("inspect_external", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Evil:
    def __reduce__(self):
        return (os.system, ("echo should-not-run",))


class TestSafeUnpickler(unittest.TestCase):
    def test_numpy_and_containers_load_but_code_is_refused(self):
        module = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = {"metadata": {"success": False}, "rollout": [{"emb": np.arange(6.0).reshape(2, 3), "t": np.int64(3)}]}
            (root / "good.pkl").write_bytes(pickle.dumps(good))
            loaded = module.safe_load(root / "good.pkl")
            np.testing.assert_array_equal(loaded["rollout"][0]["emb"], good["rollout"][0]["emb"])
            (root / "evil.pkl").write_bytes(pickle.dumps(Evil()))
            with self.assertRaises(module.Refused):
                module.safe_load(root / "evil.pkl")


if __name__ == "__main__":
    unittest.main()
