"""Trajectory smoothing and simple quadratic safety optimization."""

import numpy as np


def smooth_path(path: np.ndarray, iterations: int = 10, weight: float = 0.25) -> np.ndarray:
    """Iteratively smooth interior waypoints while preserving endpoints."""
    path = np.asarray(path, dtype=float).copy()
    if path.ndim != 2 or len(path) < 3:
        raise ValueError("path must contain at least three waypoints")
    if iterations < 0 or not 0 < weight <= 0.5:
        raise ValueError("invalid smoothing parameters")
    for _ in range(iterations):
        for i in range(1, len(path) - 1):
            path[i] += weight * (0.5 * (path[i - 1] + path[i + 1]) - path[i])
    return path


def project_box_qp(u_nominal: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Exact solution of a box-constrained QP: min ||u-u_nominal||²."""
    u = np.asarray(u_nominal, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if u.shape != lo.shape or u.shape != hi.shape or np.any(lo > hi):
        raise ValueError("incompatible or invalid QP bounds")
    return np.clip(u, lo, hi)
