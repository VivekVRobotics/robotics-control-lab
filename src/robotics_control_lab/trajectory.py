"""Smooth time-parameterized joint trajectories."""


def _samples(duration: float, dt: float) -> int:
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    return max(1, int(round(duration / dt)))


def cubic_interpolation(q0: float, qf: float, duration: float, dt: float = 0.01) -> list[tuple[float, float, float]]:
    """Return ``(time, position, velocity)`` samples with zero endpoint velocity."""
    steps = _samples(duration, dt)
    delta = qf - q0
    result = []
    for i in range(steps + 1):
        t = duration * i / steps
        tau = t / duration
        position = q0 + delta * (3 * tau**2 - 2 * tau**3)
        velocity = delta * (6 * tau - 6 * tau**2) / duration
        result.append((t, position, velocity))
    return result


def quintic_interpolation(q0: float, qf: float, duration: float, dt: float = 0.01) -> list[tuple[float, float, float, float]]:
    """Return ``(time, position, velocity, acceleration)`` with zero endpoint v/a."""
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
    return result
