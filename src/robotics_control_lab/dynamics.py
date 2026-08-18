"""Rigid-body dynamics for a planar 2R manipulator.

The model uses the standard form

    M(q) qdd + C(q, qd) + G(q) = tau.
"""

from dataclasses import dataclass

import numpy as np

from .planar_arm import Planar2R


def _joint_vector(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain exactly two finite values")
    return value


@dataclass(frozen=True)
class TwoRParameters:
    """Physical parameters for a planar two-link serial manipulator."""

    m1: float
    m2: float
    lc1: float
    lc2: float
    i1: float
    i2: float
    gravity: float = 9.81

    def validate(self, arm: Planar2R) -> None:
        values = (self.m1, self.m2, self.lc1, self.lc2, self.i1, self.i2, self.gravity)
        if not np.all(np.isfinite(values)):
            raise ValueError("physical parameters must be finite")
        if min(self.m1, self.m2, self.lc1, self.lc2, self.i1, self.i2) <= 0:
            raise ValueError("masses, COM distances, and inertias must be positive")
        if self.lc1 > arm.l1 or self.lc2 > arm.l2:
            raise ValueError("center-of-mass distances cannot exceed link lengths")
        if self.gravity < 0:
            raise ValueError("gravity must be non-negative")


class TwoRRobot:
    """Analytical dynamics model for a planar 2R robot."""

    def __init__(self, arm: Planar2R, parameters: TwoRParameters) -> None:
        parameters.validate(arm)
        self.arm = arm
        self.parameters = parameters

    def mass_matrix(self, q) -> np.ndarray:
        """Return the symmetric 2x2 joint-space inertia matrix M(q)."""
        q1, q2 = _joint_vector(q, "q")
        p = self.parameters
        c2 = np.cos(q2)
        a = p.i1 + p.i2 + p.m1 * p.lc1**2
        m11 = a + p.m2 * (self.arm.l1**2 + p.lc2**2 + 2 * self.arm.l1 * p.lc2 * c2)
        m12 = p.i2 + p.m2 * (p.lc2**2 + self.arm.l1 * p.lc2 * c2)
        m22 = p.i2 + p.m2 * p.lc2**2
        return np.array([[m11, m12], [m12, m22]], dtype=float)

    def coriolis(self, q, qd) -> np.ndarray:
        """Return the Coriolis/centrifugal vector C(q, qd)."""
        q = _joint_vector(q, "q")
        dq1, dq2 = _joint_vector(qd, "qd")
        h = -self.parameters.m2 * self.arm.l1 * self.parameters.lc2 * np.sin(q[1])
        return np.array([h * (2 * dq1 * dq2 + dq2**2), -h * dq1**2], dtype=float)

    def gravity(self, q) -> np.ndarray:
        """Return the gravity load vector G(q)."""
        q1, q2 = _joint_vector(q, "q")
        p = self.parameters
        g = p.gravity
        return np.array(
            [
                (p.m1 * p.lc1 + p.m2 * self.arm.l1) * g * np.cos(q1)
                + p.m2 * p.lc2 * g * np.cos(q1 + q2),
                p.m2 * p.lc2 * g * np.cos(q1 + q2),
            ],
            dtype=float,
        )

    def acceleration(self, q, qd, tau) -> np.ndarray:
        """Solve ``M(q) qdd = tau - C(q,qd) - G(q)``."""
        q = _joint_vector(q, "q")
        qd = _joint_vector(qd, "qd")
        tau_vec = _joint_vector(tau, "tau")
        rhs = tau_vec - self.coriolis(q, qd) - self.gravity(q)
        try:
            return np.linalg.solve(self.mass_matrix(q), rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("inertia matrix is singular") from exc
