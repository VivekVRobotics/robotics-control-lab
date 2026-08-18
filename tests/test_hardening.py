import numpy as np
import pytest

from robotics_control_lab import PIDController, cubic_interpolation, quintic_interpolation
from robotics_control_lab.observers import momentum_disturbance_estimate, residual_gate


def test_pid_rejects_nonfinite_runtime_inputs():
    pid = PIDController(kp=1.0)
    with pytest.raises(ValueError):
        pid.step(float("inf"), 0.0, 0.01)
    with pytest.raises(ValueError):
        pid.step(0.0, 0.0, float("nan"))


def test_trajectory_sampling_uses_ceil_and_exact_endpoint():
    cubic = cubic_interpolation(0.0, 1.0, 1.0, 0.3)
    quintic = quintic_interpolation(0.0, 1.0, 1.0, 0.3)
    assert [sample[0] for sample in cubic] == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert cubic[-1] == pytest.approx((1.0, 1.0, 0.0))
    assert quintic[-1] == pytest.approx((1.0, 1.0, 0.0, 0.0))


def test_disturbance_residual_sign_and_gate_contract():
    measured = np.array([3.0, 1.0])
    model = np.array([1.0, 0.5])
    torque = np.array([1.0, 0.25])
    gravity = np.array([0.5, 0.25])
    residual = momentum_disturbance_estimate(measured, model, torque, gravity)
    assert residual == pytest.approx([1.5, 0.5])
    assert residual_gate(residual, np.array([2.0, 1.0]))
    assert not residual_gate(residual, np.array([1.0, 0.4]))
    with pytest.raises(ValueError):
        residual_gate(residual, np.array([1.0]))
