"""Safety primitives for simulated actuator and joint limits."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointLimits:
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray | None = None
    effort: np.ndarray | None = None

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.shape != upper.shape or lower.ndim != 1:
            raise ValueError("lower and upper must be matching 1-D arrays")
        if np.any(lower >= upper):
            raise ValueError("every lower limit must be less than its upper limit")
        if self.velocity is not None and np.any(np.asarray(self.velocity) <= 0):
            raise ValueError("velocity limits must be positive")
        if self.effort is not None and np.any(np.asarray(self.effort) <= 0):
            raise ValueError("effort limits must be positive")

    def clamp_position(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    def clamp_velocity(self, qd: np.ndarray) -> np.ndarray:
        qd = np.asarray(qd, dtype=float)
        if self.velocity is None:
            return qd
        return np.clip(qd, -np.asarray(self.velocity), np.asarray(self.velocity))

    def clamp_effort(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        if self.effort is None:
            return tau
        return np.clip(tau, -np.asarray(self.effort), np.asarray(self.effort))


def rate_limit(previous: np.ndarray, command: np.ndarray, max_delta: np.ndarray, dt: float) -> np.ndarray:
    """Limit command slew rate element-wise."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    previous = np.asarray(previous, dtype=float)
    command = np.asarray(command, dtype=float)
    max_delta = np.asarray(max_delta, dtype=float) * dt
    if np.any(max_delta < 0):
        raise ValueError("max_delta must be non-negative")
    return previous + np.clip(command - previous, -max_delta, max_delta)
