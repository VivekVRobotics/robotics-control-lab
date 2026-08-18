"""Energy and passivity diagnostics for the planar 2R model."""

import numpy as np

from .dynamics import TwoRRobot


def _joint_vector(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain exactly two finite values")
    return value


def kinetic_energy(robot: TwoRRobot, q: np.ndarray, qd: np.ndarray) -> float:
    """Return joint-space kinetic energy 1/2 qd^T M(q) qd."""
    q = _joint_vector(q, "q")
    qd = _joint_vector(qd, "qd")
    return float(0.5 * qd @ robot.mass_matrix(q) @ qd)


def potential_energy(robot: TwoRRobot, q: np.ndarray) -> float:
    """Return gravitational potential energy with zero potential at horizontal links."""
    q1, q2 = _joint_vector(q, "q")
    p = robot.parameters
    return float(
        (p.m1 * p.lc1 + p.m2 * robot.arm.l1) * p.gravity * np.sin(q1)
        + p.m2 * p.lc2 * p.gravity * np.sin(q1 + q2)
    )


def total_energy(robot: TwoRRobot, q: np.ndarray, qd: np.ndarray) -> float:
    """Return total mechanical energy."""
    return kinetic_energy(robot, q, qd) + potential_energy(robot, q)
