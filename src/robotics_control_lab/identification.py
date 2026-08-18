"""System-identification helpers for first-order step-response fitting."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FirstOrderFit:
    gain: float
    time_constant: float
    bias: float
    residual_rms: float


def fit_first_order_step(time: np.ndarray, output: np.ndarray, step_amplitude: float) -> FirstOrderFit:
    """Estimate ``y=b+K*u*(1-exp(-(t-t0)/T))`` from a monotonic step response."""
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(output, dtype=float).reshape(-1)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 4:
        raise ValueError("time/output must be matching 1D arrays with at least four samples")
    if step_amplitude == 0 or not np.isfinite(step_amplitude):
        raise ValueError("step_amplitude must be finite and non-zero")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("time and output must contain finite values")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing")

    bias = float(y[0])
    tail = y[-max(3, len(y) // 10):]
    final = float(np.mean(tail))
    delta = final - bias
    gain = delta / step_amplitude
    if abs(delta) < np.finfo(float).eps:
        return FirstOrderFit(gain=0.0, time_constant=float(t[-1] - t[0]), bias=bias, residual_rms=float(np.sqrt(np.mean((y - bias) ** 2))))

    target = bias + 0.6321205588 * delta
    response = y - target
    crossing = np.flatnonzero(response[:-1] * response[1:] <= 0)
    if len(crossing):
        i = int(crossing[0])
        y0, y1 = y[i], y[i + 1]
        t0, t1 = t[i], t[i + 1]
        fraction = 0.0 if y1 == y0 else float(np.clip((target - y0) / (y1 - y0), 0.0, 1.0))
        T = max((t0 + fraction * (t1 - t0)) - t[0], np.finfo(float).eps)
    else:
        idx = int(np.argmin(np.abs(response)))
        T = max(float(t[idx] - t[0]), np.finfo(float).eps)

    model = bias + gain * step_amplitude * (1.0 - np.exp(-(t - t[0]) / T))
    rms = float(np.sqrt(np.mean((y - model) ** 2)))
    return FirstOrderFit(gain=gain, time_constant=T, bias=bias, residual_rms=rms)
