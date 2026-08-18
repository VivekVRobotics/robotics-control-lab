import numpy as np
import pytest

from robotics_control_lab import (
    CircleObstacle, LinearKalmanFilter, Planar2R, Serial3R,
    adjoint, astar, damped_least_squares_ik, linear_mpc,
    project_box_qp, se3, se3_exp, smooth_path, so3_exp,
)


def test_se3_adjoint_composition_shape_and_rotation():
    R = so3_exp(np.array([0.2, -0.1, 0.3]))
    T = se3(R, np.array([0.3, 0.1, -0.2]))
    A = adjoint(T)
    assert A.shape == (6, 6)
    assert np.allclose(A[:3, :3], R)


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


def test_kalman_filter_reduces_measurement_error():
    filt = LinearKalmanFilter(
        A=[[1]], B=[[0]], H=[[1]], Q=[[0.01]], R=[[0.25]], x0=[0.0], P0=[[1.0]]
    )
    filt.predict(); state = filt.update([1.0])
    assert 0.0 < state.x[0] < 1.0


def test_linear_mpc_returns_bounded_first_control():
    result = linear_mpc(
        np.array([[1.0]]), np.array([[1.0]]), np.array([0.0]), np.array([1.0]),
        np.array([[1.0]]), np.array([[0.1]]), horizon=5,
        u_lower=np.array([-0.2]), u_upper=np.array([0.2]),
    )
    assert -0.2 <= result.control[0] <= 0.2


def test_3r_space_jacobian_has_expected_shape():
    robot = Serial3R(np.eye(4), np.eye(6)[:, :3])
    assert robot.forward(np.zeros(3)).shape == (4, 4)
    assert robot.space_jacobian(np.zeros(3)).shape == (6, 3)


def test_obstacle_contains():
    obstacle = CircleObstacle(np.array([0.0, 0.0]), 1.0)
    assert obstacle.contains(np.array([0.5, 0.0]))
    assert not obstacle.contains(np.array([2.0, 0.0]))
