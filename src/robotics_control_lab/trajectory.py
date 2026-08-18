"""Time-parameterized joint trajectories."""


def cubic_interpolation(q0: float, qf: float, duration: float, dt: float = 0.01) -> list[tuple[float, float, float]]:
    """Return ``(time, position, velocity)`` samples with zero endpoint velocity."""
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")

    steps = max(1, int(round(duration / dt)))
    result = []
    for i in range(steps + 1):
        t = duration * i / steps
        tau = t / duration
        delta = qf - q0
        position = q0 + delta * (3 * tau**2 - 2 * tau**3)
        velocity = delta * (6 * tau - 6 * tau**2) / duration
        result.append((t, position, velocity))
    return result
