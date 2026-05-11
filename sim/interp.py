"""Linear interpolation of waypoints to N dense OSC targets."""
import numpy as np


def linear_interp(waypoints: np.ndarray, N: int) -> np.ndarray:
    """Linearly interpolate [T, D] waypoints into [N, D].

    Endpoints are preserved exactly.
    """
    T, D = waypoints.shape
    t_src = np.linspace(0.0, 1.0, T, dtype=np.float32)
    t_dst = np.linspace(0.0, 1.0, N, dtype=np.float32)
    dense = np.empty((N, D), dtype=np.float32)
    for d in range(D):
        dense[:, d] = np.interp(t_dst, t_src, waypoints[:, d])
    return dense
