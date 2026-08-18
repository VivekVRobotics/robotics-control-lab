"""Operational-space control utilities for task-space regulation."""

import numpy as np

from .dynamics import TwoRRobot
from .jacobian import jacobian
from .planar_arm import Planar2R


def operational_space_inertia(robot: TwoRRobot, q: np.ndarray) -> np.ndarray:
    """Return Lambda=(J M^-1 J^T)^-1 when nonsingular."""
    J = jacobian(robot.arm, *np.asarray(q, dtype=float))
    M = robot.mass_matrix(q)
    A = J @ np.linalg.solve(M, J.T)
    return np.linalg.inv(A)


def task_space_pd(
    arm: Planar2R,
    robot: TwoRRobot,
    q: np.ndarray,
    qd: np.ndarray,
    x_des: np.ndarray,
    xd_des: np.ndarray | None = None,
    kp: float | np.ndarray = 20.0,
    kd: float | np.ndarray = 6.0,
) -> np.ndarray:
    """Compute a planar task-space PD torque with gravity compensation."""
    q = np.asarray(q, dtype=float); qd = np.asarray(qd, dtype=float)
    x_des = np.asarray(x_des, dtype=float)
    xd_des = np.zeros(2) if xd_des is None else np.asarray(xd_des, dtype=float)
    x = np.asarray(arm.forward(*q))
    J = jacobian(arm, *q)
    xdot = J @ qd
    F = np.asarray(kp) * (x_des - x) + np.asarray(kd) * (xd_des - xdot)
    return J.T @ F + robot.gravity(q)
