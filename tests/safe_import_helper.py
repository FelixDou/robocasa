"""Load simulator-independent SAFE modules without importing RoboSuite registries."""

from pathlib import Path
import sys
import types


def install_lightweight_robocasa_packages():
    root = Path(__file__).resolve().parents[1] / "robocasa"
    packages = {
        "robocasa": root,
        "robocasa.recovery": root / "recovery",
        "robocasa.recovery.safe": root / "recovery" / "safe",
    }
    for name, path in packages.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__package__ = name
            sys.modules[name] = module
