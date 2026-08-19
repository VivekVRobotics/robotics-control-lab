"""Smooth time-parameterized joint trajectories."""

import math


def _samples(duration: float, dt: float) -> int:
    duration = float(duration)
    dt = float(dt)
    if not math.isfinite(duration) or not math.isfinite(dt) or duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive and finite")
    return max(1, int(math.ceil(duration / dt)))


def _validate_scalar(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def cubic_interpolation(
    q0: float,
    qf: float,
    duration: float,
    dt: float = 0.01,
) -> list[tuple[float, float, float]]:
    """Return ``(time, position, velocity)`` samples with zero endpoint velocity."""
    q0 = _validate_scalar(q0, "q0")
    qf = _validate_scalar(qf, "qf")
    duration = _validate_scalar(duration, "duration")
    dt = _validate_scalar(dt, "dt")
    steps = _samples(duration, dt)
    delta = qf - q0
    result = []
    for i in range(steps + 1):
        t = duration * i / steps
        tau = t / duration
        position = q0 + delta * (3 * tau**2 - 2 * tau**3)
        velocity = delta * (6 * tau - 6 * tau**2) / duration
        result.append((t, position, velocity))
    result[-1] = (duration, qf, 0.0)
    return result


def quintic_interpolation(
    q0: float,
    qf: float,
    duration: float,
    dt: float = 0.01,
) -> list[tuple[float, float, float, float]]:
    """Return ``(time, position, velocity, acceleration)`` with zero endpoint v/a."""
    q0 = _validate_scalar(q0, "q0")
    qf = _validate_scalar(qf, "qf")
    duration = _validate_scalar(duration, "duration")
    dt = _validate_scalar(dt, "dt")
    steps = _samples(duration, dt)
    delta = qf - q0
    result = []
    for i in range(steps + 1):
        t = duration * i / steps
        tau = t / duration
        position = q0 + delta * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
        velocity = delta * (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
        acceleration = delta * (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
        result.append((t, position, velocity, acceleration))
    result[-1] = (duration, qf, 0.0, 0.0)
    return result
