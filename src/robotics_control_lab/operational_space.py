"""Operational-space control utilities for planar task-space regulation."""

import numpy as np

from .dynamics import TwoRRobot
from .jacobian import jacobian
from .planar_arm import Planar2R


def _as_joint_vector(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain two finite values")
    return value


def operational_space_inertia(robot: TwoRRobot, q: np.ndarray, damping: float = 0.0) -> np.ndarray:
    """Return Lambda = (J M^-1 J^T)^-1, optionally regularized near singularities."""
    q = _as_joint_vector(q, "q")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    J = jacobian(robot.arm, *q)
    M = robot.mass_matrix(q)
    A = J @ np.linalg.solve(M, J.T)
    if damping > 0:
        A = A + damping**2 * np.eye(2)
    try:
        return np.linalg.solve(A, np.eye(2))
    except np.linalg.LinAlgError as exc:
        raise ValueError("task Jacobian is singular") from exc


def jdot_qdot(arm: Planar2R, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Return Jdot(q, qd) * qd for the planar 2R position Jacobian."""
    q = _as_joint_vector(q, "q")
    qd = _as_joint_vector(qd, "qd")
    q1, q2 = q
    dq1, dq2 = qd
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    total = dq1 + dq2
    return np.array(
        [
            -arm.l1 * c1 * dq1**2 - arm.l2 * c12 * total**2,
            -arm.l1 * s1 * dq1**2 - arm.l2 * s12 * total**2,
        ],
        dtype=float,
    )


def task_space_pd(
    arm: Planar2R,
    robot: TwoRRobot,
    q: np.ndarray,
    qd: np.ndarray,
    x_des: np.ndarray,
    xd_des: np.ndarray | None = None,
    kp: float | np.ndarray = 20.0,
    kd: float | np.ndarray = 6.0,
    *,
    damping: float = 1e-6,
) -> np.ndarray:
    """Compute dynamically consistent Cartesian PD torque with model compensation.

    The controller uses Lambda, the task-space bias term induced by joint-space
    Coriolis/gravity forces, and Jdot*qdot. A small damping term regularizes
    operation close to a kinematic singularity.
    """
    q = _as_joint_vector(q, "q")
    qd = _as_joint_vector(qd, "qd")
    x_des = np.asarray(x_des, dtype=float)
    xd_des = np.zeros(2) if xd_des is None else np.asarray(xd_des, dtype=float)
    if x_des.shape != (2,) or xd_des.shape != (2,):
        raise ValueError("x_des and xd_des must contain two values")
    kp = np.asarray(kp, dtype=float)
    kd = np.asarray(kd, dtype=float)
    if np.any(kp < 0) or np.any(kd < 0):
        raise ValueError("task-space gains must be non-negative")

    x = np.asarray(arm.forward(*q))
    J = jacobian(arm, *q)
    xdot = J @ qd
    acceleration_command = kp * (x_des - x) + kd * (xd_des - xdot)

    Lambda = operational_space_inertia(robot, q, damping=damping)
    bias_joint = robot.coriolis(q, qd) + robot.gravity(q)
    bias_task = Lambda @ J @ np.linalg.solve(robot.mass_matrix(q), bias_joint) - Lambda @ jdot_qdot(arm, q, qd)
    wrench = Lambda @ acceleration_command + bias_task
    return J.T @ wrench
