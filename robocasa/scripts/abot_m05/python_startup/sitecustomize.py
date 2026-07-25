"""Install ABot subtask recording when this directory is on ``PYTHONPATH``."""

from robocasa.scripts.abot_m05.subtask_progress_recorder import (
    install_gym_make_hook,
)


install_gym_make_hook()
