import numpy as np
import pytest

from robotics_control_lab import (
    ComputedTorqueController,
    JointPD,
    Planar2R,
    TwoRParameters,
    TwoRRobot,
    is_singular,
    jacobian,
    manipulability,
    simulate,
)


def make_robot() -> TwoRRobot:
    arm = Planar2R(0.6, 0.4)
    return TwoRRobot(
        arm,
        TwoRParameters(m1=2.0, m2=1.0, lc1=0.3, lc2=0.2, i1=0.06, i2=0.02),
    )


def test_jacobian_matches_finite_difference():
    arm = Planar2R(0.6, 0.4)
    q = np.array([0.35, -0.8])
    eps = 1e-7
    numeric = np.column_stack(
        [
            (np.asarray(arm.forward(*(q + np.eye(2)[i] * eps))) - np.asarray(arm.forward(*(q - np.eye(2)[i] * eps)))) / (2 * eps)
            for i in range(2)
        ]
    )
    assert jacobian(arm, *q) == pytest.approx(numeric, abs=1e-7)


def test_jacobian_identifies_straight_arm_singularity():
    arm = Planar2R(0.6, 0.4)
    assert is_singular(arm, 0.2, 0.0)
    assert manipulability(arm, 0.2, np.pi / 2) > 0


def test_mass_matrix_is_symmetric_positive_definite():
    robot = make_robot()
    m = robot.mass_matrix((0.4, -0.7))
    assert m == pytest.approx(m.T)
    assert np.all(np.linalg.eigvalsh(m) > 0)


def test_gravity_compensation_controller_matches_static_load():
    robot = make_robot()
    controller = ComputedTorqueController(robot, kp=20.0, kd=5.0)
    q = np.array([0.2, -0.4])
    gravity = robot.gravity(q)
    tau = controller(q, np.zeros(2), q, np.zeros(2))
    assert tau == pytest.approx(gravity)


def test_joint_pd_respects_torque_limits():
    controller = JointPD(kp=100.0, kd=10.0, torque_limit=2.0)
    tau = controller(np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2))
    assert tau == pytest.approx([2.0, 2.0])


def test_simulation_moves_toward_static_target():
    robot = make_robot()
    controller = ComputedTorqueController(robot, kp=45.0, kd=14.0, torque_limit=40.0)
    target = np.array([0.45, -0.35])
    reference = lambda _t: (target, np.zeros(2), np.zeros(2))
    result = simulate(robot, controller, q0=(0.0, 0.0), qd0=(0.0, 0.0), q_des_fn=reference, duration=1.5, dt=0.002)
    assert result.q.shape == (751, 2)
    assert np.linalg.norm(result.q[-1] - target) < 0.03
    assert np.max(np.abs(result.tau)) <= 40.0 + 1e-12
