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
        if lower.shape != upper.shape or lower.ndim != 1 or lower.size == 0:
            raise ValueError("lower and upper must be matching non-empty 1-D arrays")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("position limits must be finite")
        if np.any(lower >= upper):
            raise ValueError("every lower limit must be less than its upper limit")
        for name, limit in (("velocity", self.velocity), ("effort", self.effort)):
            if limit is not None:
                value = np.asarray(limit, dtype=float)
                if value.shape != lower.shape or not np.all(np.isfinite(value)) or np.any(value <= 0):
                    raise ValueError(f"{name} limits must match joint dimension and be positive finite")

    @property
    def joints(self) -> int:
        return int(np.asarray(self.lower).size)

    def _vector(self, value: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if value.shape != np.asarray(self.lower).shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} has incorrect dimension or non-finite values")
        return value

    def clamp_position(self, q: np.ndarray) -> np.ndarray:
        return np.clip(self._vector(q, "position"), self.lower, self.upper)

    def clamp_velocity(self, qd: np.ndarray) -> np.ndarray:
        qd = self._vector(qd, "velocity")
        if self.velocity is None:
            return qd
        return np.clip(qd, -np.asarray(self.velocity), np.asarray(self.velocity))

    def clamp_effort(self, tau: np.ndarray) -> np.ndarray:
        tau = self._vector(tau, "effort")
        if self.effort is None:
            return tau
        return np.clip(tau, -np.asarray(self.effort), np.asarray(self.effort))


def rate_limit(previous: np.ndarray, command: np.ndarray, max_delta: np.ndarray, dt: float) -> np.ndarray:
    """Limit command slew rate element-wise."""
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be positive and finite")
    previous = np.asarray(previous, dtype=float)
    command = np.asarray(command, dtype=float)
    max_rate = np.asarray(max_delta, dtype=float)
    if previous.shape != command.shape or previous.shape != max_rate.shape or previous.ndim == 0:
        raise ValueError("previous, command, and max_delta must have matching vector shapes")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(command)):
        raise ValueError("previous and command must be finite")
    if not np.all(np.isfinite(max_rate)) or np.any(max_rate < 0):
        raise ValueError("max_delta must be finite and non-negative")
    delta = max_rate * dt
    return previous + np.clip(command - previous, -delta, delta)
