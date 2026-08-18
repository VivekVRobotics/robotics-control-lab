"""System-identification helpers for first-order and second-order fits."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FirstOrderFit:
    gain: float
    time_constant: float
    bias: float
    residual_rms: float


def fit_first_order_step(time: np.ndarray, output: np.ndarray, step_amplitude: float) -> FirstOrderFit:
    """Estimate y=b+K*u*(1-exp(-t/T)) from a single step response."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(output, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 4 or step_amplitude == 0:
        raise ValueError("time/output must be matching 1D arrays with at least four samples")
    bias = float(y[0])
    final = float(np.mean(y[-max(3, len(y)//10):]))
    gain = (final - bias) / step_amplitude
    target = bias + 0.6321205588 * (final - bias)
    idx = int(np.argmin(np.abs(y - target)))
    T = max(float(t[idx] - t[0]), np.finfo(float).eps)
    model = bias + gain * step_amplitude * (1.0 - np.exp(-(t - t[0]) / T))
    rms = float(np.sqrt(np.mean((y - model) ** 2)))
    return FirstOrderFit(gain=gain, time_constant=T, bias=bias, residual_rms=rms)
