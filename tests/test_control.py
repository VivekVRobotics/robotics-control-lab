import pytest

from robotics_control_lab import PIDController, Planar2R, cubic_interpolation


def test_pid_moves_toward_setpoint_and_respects_output_limit():
    pid = PIDController(kp=4.0, ki=1.0, output_min=-1.0, output_max=1.0)
    command = pid.step(1.0, 0.0, 0.1)
    assert command == pytest.approx(1.0)
    assert pid.integral == pytest.approx(0.1)


def test_pid_rejects_invalid_dt():
    pid = PIDController(kp=1.0)
    with pytest.raises(ValueError):
        pid.step(1.0, 0.0, 0.0)


def test_planar_arm_round_trip_kinematics():
    arm = Planar2R(1.0, 0.75)
    target = arm.forward(0.4, -0.7)
    q1, q2 = arm.inverse(*target, elbow="down")
    assert arm.forward(q1, q2) == pytest.approx(target, abs=1e-9)


def test_planar_arm_rejects_unreachable_target():
    with pytest.raises(ValueError):
        Planar2R(1.0, 1.0).inverse(3.0, 0.0)


def test_cubic_trajectory_has_zero_endpoint_velocity():
    samples = cubic_interpolation(0.0, 1.0, 2.0, 0.2)
    assert samples[0][1:] == pytest.approx((0.0, 0.0))
    assert samples[-1][1:] == pytest.approx((1.0, 0.0))
