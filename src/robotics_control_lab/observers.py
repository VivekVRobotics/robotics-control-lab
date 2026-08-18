"""Disturbance-observer and robustness utilities."""

import numpy as np


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 1-D vector")
    return value


def momentum_disturbance_estimate(
    measured_momentum_derivative: np.ndarray,
    model_momentum_derivative: np.ndarray,
    commanded_torque: np.ndarray,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate external generalized torque from a momentum-balance residual.

    The sign convention is explicit:

        tau_ext = measured_pdot - modeled_pdot - tau_cmd + gravity.

    This helper is a residual computation, not a complete filtered momentum
    observer; the caller remains responsible for differentiation and filtering.
    """
    measured = _vector(measured_momentum_derivative, "measured_momentum_derivative")
    model = _vector(model_momentum_derivative, "model_momentum_derivative")
    tau = _vector(commanded_torque, "commanded_torque")
    if not (measured.shape == model.shape == tau.shape):
        raise ValueError("momentum derivatives and torque must have matching shapes")
    g = np.zeros_like(tau) if gravity is None else _vector(gravity, "gravity")
    if g.shape != tau.shape:
        raise ValueError("gravity must match torque dimension")
    return measured - model - tau + g


def residual_gate(residual: np.ndarray, limits: np.ndarray) -> bool:
    """Return True when all residual channels remain inside absolute limits."""
    residual = _vector(residual, "residual")
    limits = _vector(limits, "limits")
    if residual.shape != limits.shape or np.any(limits < 0):
        raise ValueError("limits must match residual shape and be non-negative")
    return bool(np.all(np.abs(residual) <= limits))
