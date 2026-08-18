"""Disturbance-observer and robustness utilities."""

import numpy as np


def momentum_disturbance_estimate(
    measured_momentum: np.ndarray,
    model_momentum_derivative: np.ndarray,
    commanded_torque: np.ndarray,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate external generalized torque from momentum balance.

    For a simple momentum observer, tau_ext ~= d(p)/dt - tau + model terms.
    The caller supplies the modeled momentum derivative so the boundary is
    explicit rather than hiding a robot-specific dynamics assumption.
    """
    p_dot = np.asarray(measured_momentum, dtype=float)
    model = np.asarray(model_momentum_derivative, dtype=float)
    tau = np.asarray(commanded_torque, dtype=float)
    g = np.zeros_like(tau) if gravity is None else np.asarray(gravity, dtype=float)
    return p_dot - model - tau + g


def residual_gate(residual: np.ndarray, limits: np.ndarray) -> bool:
    """Return True when all residual channels remain inside absolute limits."""
    return bool(np.all(np.abs(np.asarray(residual, dtype=float)) <= np.asarray(limits, dtype=float)))
