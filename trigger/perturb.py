"""Mid-rollout xy perturbation sampler for data collection.

Direction uniform on the unit circle; magnitude uniform in [1, 3] cm.
Returned as meters so it composes with `LiberoEnv.perturb_object_xy`.
"""
import numpy as np

MAG_MIN_M = 0.01
MAG_MAX_M = 0.03


def sample_perturbation(rng: np.random.Generator) -> tuple[float, float]:
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    r = float(rng.uniform(MAG_MIN_M, MAG_MAX_M))
    return r * float(np.cos(theta)), r * float(np.sin(theta))
