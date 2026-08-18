"""Discrete PID control with saturation and conditional anti-windup."""

from dataclasses import dataclass
import math


@dataclass
class PIDController:
    """Deterministic discrete PID controller with output/integral limits.

    When the provisional command saturates, the integral state is only allowed
    to continue changing when that change would move the command back toward
    the admissible output range.
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = float("-inf")
    output_max: float = float("inf")
    integral_min: float = float("-inf")
    integral_max: float = float("inf")

    def __post_init__(self) -> None:
        values = (self.kp, self.ki, self.kd, self.output_min, self.output_max, self.integral_min, self.integral_max)
        if not all(math.isfinite(v) or math.isinf(v) for v in values):
            raise ValueError("PID parameters must be numeric")
        if self.kp < 0 or self.ki < 0 or self.kd < 0:
            raise ValueError("PID gains must be non-negative")
        if self.output_min > self.output_max:
            raise ValueError("output_min must not exceed output_max")
        if self.integral_min > self.integral_max:
            raise ValueError("integral_min must not exceed integral_max")
        self._integral = 0.0
        self._previous_error: float | None = None

    @property
    def integral(self) -> float:
        return self._integral

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = None

    def step(self, setpoint: float, measurement: float, dt: float) -> float:
        setpoint = float(setpoint)
        measurement = float(measurement)
        dt = float(dt)
        if not math.isfinite(setpoint) or not math.isfinite(measurement):
            raise ValueError("setpoint and measurement must be finite")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")

        error = setpoint - measurement
        integral_candidate = min(max(self._integral + error * dt, self.integral_min), self.integral_max)
        derivative = 0.0 if self._previous_error is None else (error - self._previous_error) / dt
        base = self.kp * error + self.kd * derivative
        candidate = base + self.ki * integral_candidate

        if candidate > self.output_max and error > 0:
            integral_candidate = self._integral
        elif candidate < self.output_min and error < 0:
            integral_candidate = self._integral

        self._integral = integral_candidate
        self._previous_error = error
        command = base + self.ki * self._integral
        return min(max(command, self.output_min), self.output_max)
