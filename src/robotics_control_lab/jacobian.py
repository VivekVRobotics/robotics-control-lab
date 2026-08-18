"""Geometric Jacobian utilities for a planar 2R manipulator."""

from math import cos, sin

import numpy as np

from .planar_arm import Planar2R


def jacobian(arm: Planar2R, q1: float, q2: float) -> np.ndarray:
    """Return the 2x2 position Jacobian mapping joint velocity to XY velocity."""
    s1 = sin(q1)
    c1 = cos(q1)
    s12 = sin(q1 + q2)
    c12 = cos(q1 + q2)
    return np.array(
        [
            [-arm.l1 * s1 - arm.l2 * s12, -arm.l2 * s12],
            [arm.l1 * c1 + arm.l2 * c12, arm.l2 * c12],
        ],
        dtype=float,
    )


def manipulability(arm: Planar2R, q1: float, q2: float) -> float:
    """Return Yoshikawa's planar position manipulability measure."""
    return float(abs(np.linalg.det(jacobian(arm, q1, q2))))


def is_singular(arm: Planar2R, q1: float, q2: float, tolerance: float = 1e-9) -> bool:
    """Return whether the position Jacobian is numerically singular."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    return manipulability(arm, q1, q2) <= tolerance
