"""RoboCasa package with optional lazy simulator registration.

Simulator-independent tooling, including SAFE dataset validation and export,
must remain importable without RoboSuite. When RoboSuite is installed, retain
the historical eager environment-registration behavior.
"""

from __future__ import annotations

import importlib.util


__version__ = "1.0.1"

if importlib.util.find_spec("robosuite") is not None:
    from ._simulator_init import *  # noqa: F401,F403
