import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa"
    / "recovery"
    / "prepare_lingbot_human_pretrain.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_lingbot_human_pretrain", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareLingBotHumanPretrainTest(unittest.TestCase):
    def test_writes_lingbot_manifest_for_complete_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "task" / "lerobot"
            (dataset / "meta").mkdir(parents=True)
            (dataset / "meta" / "info.json").write_text("{}\n")
            (dataset / "data" / "chunk-000").mkdir(parents=True)
            (dataset / "data" / "chunk-000" / "episode_000000.parquet").touch()
            inventory = MODULE.inspect_dataset("Task", dataset)
            manifest = root / "manifest.txt"
            report = root / "report.json"

            MODULE.write_outputs(
                [inventory], manifest, report, "robocasa_lerobot", "pretrain300", root, False
            )

            self.assertEqual(
                manifest.read_text(), f"robocasa_lerobot {dataset.resolve()}\n"
            )
            self.assertIn('"complete": true', report.read_text())

    def test_refuses_incomplete_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = MODULE.inspect_dataset("MissingTask", root / "missing")
            manifest = root / "manifest.txt"
            report = root / "report.json"

            with self.assertRaisesRegex(RuntimeError, "datasets are incomplete"):
                MODULE.write_outputs(
                    [inventory], manifest, report, "robocasa_lerobot", "pretrain300", root, False
                )

            self.assertFalse(manifest.exists())
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
