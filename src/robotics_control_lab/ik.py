"""Numerical inverse kinematics and task-space velocity control."""

from dataclasses import dataclass

import numpy as np

from .jacobian import analytic_jacobian
from .planar_arm import Planar2R


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    iterations: int
    converged: bool
    residual: float


def damped_least_squares_ik(
    arm: Planar2R,
    target: tuple[float, float] | np.ndarray,
    q0: tuple[float, float] | np.ndarray,
    *,
    damping: float = 1e-3,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    joint_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> IKResult:
    """Solve planar position IK using damped least squares.

    The damping term stabilizes iterations near singular configurations. Joint
    limits are enforced by clipping each iterate, making this useful as a
    deterministic teaching/reference implementation.
    """
    if damping <= 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("damping, max_iterations, and tolerance must be positive")
    q = np.asarray(q0, dtype=float).copy()
    target_vec = np.asarray(target, dtype=float)
    if q.shape != (2,) or target_vec.shape != (2,):
        raise ValueError("target and q0 must each contain two values")

    for iteration in range(1, max_iterations + 1):
        error = target_vec - np.asarray(arm.forward(*q))
        residual = float(np.linalg.norm(error))
        if residual <= tolerance:
            return IKResult(q=q.copy(), iterations=iteration - 1, converged=True, residual=residual)

        j = analytic_jacobian(arm, q)
        step = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(2), error)
        q += step
        if joint_limits is not None:
            lower = np.array([joint_limits[0][0], joint_limits[1][0]], dtype=float)
            upper = np.array([joint_limits[0][1], joint_limits[1][1]], dtype=float)
            if np.any(lower > upper):
                raise ValueError("joint lower limits must not exceed upper limits")
            q = np.clip(q, lower, upper)

    residual = float(np.linalg.norm(target_vec - np.asarray(arm.forward(*q))))
    return IKResult(q=q, iterations=max_iterations, converged=residual <= tolerance, residual=residual)


def cartesian_velocity_control(
    arm: Planar2R,
    q: np.ndarray,
    target_velocity: np.ndarray,
    *,
    damping: float = 1e-3,
) -> np.ndarray:
    """Map Cartesian velocity to joint velocity via damped least squares."""
    if damping <= 0:
        raise ValueError("damping must be positive")
    target_velocity = np.asarray(target_velocity, dtype=float)
    if target_velocity.shape != (2,):
        raise ValueError("target_velocity must contain two values")
    j = analytic_jacobian(arm, q)
    return j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(2), target_velocity)
