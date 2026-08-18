"""Forward and inverse kinematics for a planar two-link arm."""

from dataclasses import dataclass
from math import atan2, cos, hypot, sin


@dataclass(frozen=True)
class Planar2R:
    l1: float
    l2: float

    def __post_init__(self) -> None:
        if self.l1 <= 0 or self.l2 <= 0:
            raise ValueError("link lengths must be positive")

    def forward(self, q1: float, q2: float) -> tuple[float, float]:
        if not (float(q1) == float(q1) and float(q2) == float(q2)):
            raise ValueError("joint angles must be finite")
        return (
            self.l1 * cos(q1) + self.l2 * cos(q1 + q2),
            self.l1 * sin(q1) + self.l2 * sin(q1 + q2),
        )

    def inverse(self, x: float, y: float, elbow: str = "up") -> tuple[float, float]:
        x = float(x)
        y = float(y)
        if not (x == x and y == y):
            raise ValueError("target coordinates must be finite")
        if elbow not in {"up", "down"}:
            raise ValueError("elbow must be 'up' or 'down'")

        radius = hypot(x, y)
        maximum = self.l1 + self.l2
        minimum = abs(self.l1 - self.l2)
        tolerance = 1e-12 * max(1.0, maximum)
        if radius > maximum + tolerance or radius < minimum - tolerance:
            raise ValueError("target is outside the reachable workspace")
        radius = min(max(radius, minimum), maximum)

        c2 = (radius**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        c2 = min(1.0, max(-1.0, c2))
        s2 = (1 if elbow == "up" else -1) * (max(0.0, 1.0 - c2 * c2) ** 0.5)
        q2 = atan2(s2, c2)
        q1 = atan2(y, x) - atan2(self.l2 * s2, self.l1 + self.l2 * c2)
        return q1, q2
