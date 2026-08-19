import numpy as np
import pytest

from robotics_control_lab import (
    ComputedTorqueController,
    JointLimits,
    JointPD,
    Planar2R,
    TwoRParameters,
    TwoRRobot,
    cartesian_velocity_control,
    damped_least_squares_ik,
    is_singular,
    jacobian,
    kinetic_energy,
    manipulability,
    potential_energy,
    quintic_interpolation,
    rate_limit,
    simulate,
    total_energy,
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
    numeric_columns = []
    for i in range(2):
        offset = np.eye(2)[i] * eps
        forward_plus = np.asarray(arm.forward(*(q + offset)))
        forward_minus = np.asarray(arm.forward(*(q - offset)))
        numeric_columns.append((forward_plus - forward_minus) / (2 * eps))
    numeric = np.column_stack(numeric_columns)
    assert jacobian(arm, *q) == pytest.approx(numeric, abs=1e-7)


def test_jacobian_identifies_straight_arm_singularity():
    arm = Planar2R(0.6, 0.4)
    assert is_singular(arm, 0.2, 0.0)
    assert manipulability(arm, 0.2, np.pi / 2) > 0


def test_mass_matrix_is_symmetric_positive_definite():
    robot = make_robot()
    mass = robot.mass_matrix((0.4, -0.7))
    assert mass == pytest.approx(mass.T)
    assert np.all(np.linalg.eigvalsh(mass) > 0)


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


def test_damped_ik_converges_to_reachable_target():
    arm = Planar2R(0.6, 0.4)
    target = np.asarray(arm.forward(0.5, -0.8))
    result = damped_least_squares_ik(arm, target, q0=(0.0, 0.0))
    assert result.converged
    assert result.residual < 1e-7
    assert np.asarray(arm.forward(*result.q)) == pytest.approx(target, abs=1e-7)


def test_cartesian_velocity_control_matches_jacobian_mapping():
    arm = Planar2R(0.6, 0.4)
    q = np.array([0.3, -0.7])
    qd = cartesian_velocity_control(arm, q, np.array([0.02, -0.01]))
    assert jacobian(arm, *q) @ qd == pytest.approx([0.02, -0.01], abs=1e-5)


def test_quintic_trajectory_has_zero_endpoint_velocity_and_acceleration():
    samples = quintic_interpolation(0.0, 1.0, 2.0, 0.2)
    assert samples[0][1:] == pytest.approx((0.0, 0.0, 0.0))
    assert samples[-1][1:] == pytest.approx((1.0, 0.0, 0.0))


def test_joint_limits_and_rate_limit_are_safe():
    limits = JointLimits(
        np.array([-1.0, -2.0]),
        np.array([1.0, 2.0]),
        velocity=np.array([2.0, 3.0]),
        effort=np.array([5.0, 6.0]),
    )
    assert limits.clamp_position(np.array([4.0, -5.0])) == pytest.approx([1.0, -2.0])
    assert limits.clamp_velocity(np.array([4.0, -5.0])) == pytest.approx([2.0, -3.0])
    assert limits.clamp_effort(np.array([8.0, -7.0])) == pytest.approx([5.0, -6.0])
    limited = rate_limit(np.zeros(2), np.array([5.0, -3.0]), np.array([2.0, 4.0]), 0.1)
    assert limited == pytest.approx([0.2, -0.3])


def test_energy_components_are_nonnegative_when_kinetic():
    robot = make_robot()
    q = np.array([0.4, -0.2])
    qd = np.array([0.7, -0.3])
    assert kinetic_energy(robot, q, qd) >= 0.0
    assert total_energy(robot, q, qd) == pytest.approx(
        kinetic_energy(robot, q, qd) + potential_energy(robot, q)
    )


def test_simulation_moves_toward_static_target():
    robot = make_robot()
    controller = ComputedTorqueController(robot, kp=45.0, kd=14.0, torque_limit=40.0)
    target = np.array([0.45, -0.35])

    def reference(_t):
        return target, np.zeros(2), np.zeros(2)

    result = simulate(
        robot,
        controller,
        q0=(0.0, 0.0),
        qd0=(0.0, 0.0),
        q_des_fn=reference,
        duration=1.5,
        dt=0.002,
    )
    assert result.q.shape == (751, 2)
    assert np.linalg.norm(result.q[-1] - target) < 0.03
    assert np.max(np.abs(result.tau)) <= 40.0 + 1e-12
