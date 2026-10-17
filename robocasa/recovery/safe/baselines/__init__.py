"""Published failure-detection baselines reimplemented on RoboCasa SAFE rollouts.

Every detector consumes the same raw rollouts as SAFE (per-inference VLA
features and recorded action chunks) and emits one causal score per policy
inference, so all methods can share the frozen matched-FPR evaluation.
"""

from .detectors import DETECTOR_REGISTRY, build_detector

__all__ = ["DETECTOR_REGISTRY", "build_detector"]
