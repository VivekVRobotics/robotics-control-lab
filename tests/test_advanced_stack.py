import numpy as np
import pytest

from robotics_control_lab import (
    CircleObstacle,
    LinearKalmanFilter,
    Planar2R,
    Serial3R,
    adjoint,
    astar,
    damped_least_squares_ik,
    linear_mpc,
    project_box_qp,
    se3,
    se3_exp,
    simulate,
    smooth_path,
    so3_exp,
    so3_log,
)
from robotics_control_lab.controllers import JointPD
from robotics_control_lab.dynamics import TwoRParameters, TwoRRobot
from robotics_control_lab.hardware import SimulatedHardware
from robotics_control_lab.identification import fit_first_order_step
from robotics_control_lab.operational_space import operational_space_inertia, task_space_pd
from robotics_control_lab.planning import edge_collision_free, rrt_star
from robotics_control_lab.safety import JointLimits


def make_robot() -> TwoRRobot:
    return TwoRRobot(
        Planar2R(0.6, 0.4),
        TwoRParameters(m1=2.0, m2=1.0, lc1=0.3, lc2=0.2, i1=0.06, i2=0.02),
    )


def test_se3_adjoint_composition_shape_and_rotation():
    R = so3_exp(np.array([0.2, -0.1, 0.3]))
    T = se3(R, np.array([0.3, 0.1, -0.2]))
    A = adjoint(T)
    assert A.shape == (6, 6)
    assert np.allclose(A[:3, :3], R)


def test_so3_exp_log_roundtrip_near_pi():
    phi = np.array([np.pi - 1e-8, 0.0, 0.0])
    recovered = so3_log(so3_exp(phi))
    assert np.linalg.norm(recovered) == pytest.approx(np.linalg.norm(phi), abs=1e-6)
    assert np.allclose(so3_exp(recovered), so3_exp(phi), atol=1e-7)


def test_so3_log_rejects_non_rotation():
    with pytest.raises(ValueError):
        so3_log(np.eye(3) * 2.0)


def test_se3_exp_pure_translation():
    T = se3_exp(np.array([0, 0, 0, 1, 2, 3]), 2.0)
    assert np.allclose(T[:3, 3], [2, 4, 6])


def test_damped_ik_converges_for_reachable_target():
    arm = Planar2R(1.0, 0.75)
    result = damped_least_squares_ik(arm, (1.0, 0.25), (0.3, -0.5))
    assert result.converged
    assert np.linalg.norm(np.asarray(arm.forward(*result.q)) - np.array([1.0, 0.25])) < 1e-6


def test_astar_finds_grid_path():
    grid = np.zeros((10, 10), dtype=int)
    grid[4:8, 5] = 1
    path = astar(grid, (0, 0), (9, 9))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (9, 9)


def test_smooth_path_preserves_endpoints():
    path = np.array([[0, 0], [1, 1], [2, 0]], dtype=float)
    out = smooth_path(path, iterations=3)
    assert np.allclose(out[0], path[0])
    assert np.allclose(out[-1], path[-1])


def test_box_qp_projection():
    out = project_box_qp(np.array([3.0, -3.0]), np.array([-1.0, -2.0]), np.array([1.0, 2.0]))
    assert np.allclose(out, [1.0, -2.0])


def test_kalman_filter_reduces_measurement_error_and_preserves_covariance():
    filt = LinearKalmanFilter(
        A=[[1]], B=[[0]], H=[[1]], Q=[[0.01]], R=[[0.25]], x0=[0.0], P0=[[1.0]]
    )
    filt.predict()
    state = filt.update([1.0])
    assert 0.0 < state.x[0] < 1.0
    assert state.covariance[0, 0] >= 0.0
    assert state.covariance == pytest.approx(state.covariance.T)


def test_kalman_rejects_bad_measurement_shape():
    filt = LinearKalmanFilter(
        A=np.eye(2), B=np.zeros((2, 1)), H=np.eye(2), Q=np.eye(2) * 0.01,
        R=np.eye(2) * 0.1, x0=np.zeros(2), P0=np.eye(2),
    )
    with pytest.raises(ValueError):
        filt.update([1.0])


def test_linear_mpc_returns_bounded_sequence_in_correct_joint_order():
    result = linear_mpc(
        np.eye(2), np.eye(2), np.zeros(2), np.ones(2),
        np.eye(2), np.eye(2) * 0.1, horizon=4,
        u_lower=np.array([-0.2, -0.5]), u_upper=np.array([0.2, 0.5]),
    )
    assert result.converged
    assert result.sequence.shape == (4, 2)
    assert np.all(result.sequence[:, 0] <= 0.2 + 1e-10)
    assert np.all(result.sequence[:, 0] >= -0.2 - 1e-10)
    assert np.all(result.sequence[:, 1] <= 0.5 + 1e-10)
    assert np.all(result.sequence[:, 1] >= -0.5 - 1e-10)


def test_3r_space_jacobian_has_expected_shape():
    robot = Serial3R(np.eye(4), np.eye(6)[:, :3])
    assert robot.forward(np.zeros(3)).shape == (4, 4)
    assert robot.space_jacobian(np.zeros(3)).shape == (6, 3)


def test_obstacle_contains_and_edge_collision():
    obstacle = CircleObstacle(np.array([0.0, 0.0]), 1.0)
    assert obstacle.contains(np.array([0.5, 0.0]))
    assert not obstacle.contains(np.array([2.0, 0.0]))
    assert not edge_collision_free(np.array([-2.0, 0.0]), np.array([2.0, 0.0]), [obstacle])


def test_rrt_star_returns_collision_free_path_when_available():
    obstacle = CircleObstacle(np.array([0.0, 0.0]), 0.4)
    path = rrt_star(
        np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
        ((-1.5, 1.5), (-1.5, 1.5)), [obstacle],
        iterations=1200, step_size=0.12, neighbor_radius=0.3, seed=7,
    )
    assert path is not None
    assert np.allclose(path[0], [-1.0, -1.0])
    assert np.allclose(path[-1], [1.0, 1.0])
    assert all(edge_collision_free(a, b, [obstacle]) for a, b in zip(path[:-1], path[1:]))


def test_operational_space_inertia_is_symmetric_positive_definite_away_from_singularity():
    robot = make_robot()
    lam = operational_space_inertia(robot, np.array([0.4, -0.7]))
    assert lam == pytest.approx(lam.T)
    assert np.all(np.linalg.eigvalsh(lam) > 0)


def test_task_space_gravity_compensation_at_zero_task_error():
    robot = make_robot()
    arm = robot.arm
    q = np.array([0.4, -0.7])
    x = np.asarray(arm.forward(*q))
    tau = task_space_pd(arm, robot, q, np.zeros(2), x, kp=20.0, kd=6.0)
    assert tau == pytest.approx(robot.gravity(q), abs=1e-8)


def test_simulation_time_grid_matches_actual_step_sizes():
    robot = make_robot()
    controller = JointPD(kp=0.0, kd=0.0)
    result = simulate(
        robot, controller, (0.0, 0.0), (0.0, 0.0),
        lambda _t: (np.zeros(2), np.zeros(2)), duration=0.01, dt=0.003,
    )
    assert result.time[-1] == pytest.approx(0.01)
    assert np.all(np.diff(result.time) > 0)
    assert np.allclose(np.diff(result.time), [0.003, 0.003, 0.003, 0.001])


def test_simulator_does_not_swallow_controller_type_errors():
    robot = make_robot()

    def broken_controller(*_args):
        raise TypeError("controller bug")

    with pytest.raises(TypeError, match="controller bug"):
        simulate(
            robot, broken_controller, (0.0, 0.0), (0.0, 0.0),
            lambda _t: (np.zeros(2), np.zeros(2)), duration=0.01, dt=0.005,
        )


def test_hardware_enforces_monotonic_time_and_finite_commands():
    hardware = SimulatedHardware(2)
    hardware.write(np.array([1.0, -1.0]), 1.0)
    with pytest.raises(ValueError):
        hardware.write(np.array([0.0, 0.0]), 0.5)
    with pytest.raises(ValueError):
        hardware.write(np.array([np.inf, 0.0]), 1.0)


def test_joint_limits_reject_wrong_dimension():
    limits = JointLimits(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        limits.clamp_position(np.array([0.0]))


def test_first_order_identification_rejects_nonmonotonic_time():
    with pytest.raises(ValueError):
        fit_first_order_step(np.array([0.0, 0.1, 0.05, 0.2]), np.ones(4), 1.0)
