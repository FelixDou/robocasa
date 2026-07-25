"""Install ABot subtask recording when this directory is on ``PYTHONPATH``.

MuJoCo's GLFW loader starts short Python subprocesses to probe shared
libraries. Those children inherit ``PYTHONPATH`` and would recursively import
RoboCasa through this hook. Mark the real client before importing RoboCasa so
the probe children skip the hook.
"""

import os


TRACKING_ENV = "ROBOCASA_TRACK_SUBTASK_PROGRESS"
BOOTSTRAP_GUARD = "ROBOCASA_SUBTASK_PROGRESS_BOOTSTRAPPED"


def _enabled():
    return os.environ.get(TRACKING_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


if _enabled() and os.environ.get(BOOTSTRAP_GUARD) != "1":
    os.environ[BOOTSTRAP_GUARD] = "1"
    try:
        from robocasa.scripts.abot_m05.subtask_progress_recorder import (
            install_gym_make_hook,
        )

        install_gym_make_hook()
    except Exception:
        os.environ.pop(BOOTSTRAP_GUARD, None)
        raise
