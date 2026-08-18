"""Joint-space and model-based controllers for the 2R robot."""

from dataclasses import dataclass

import numpy as np

from .dynamics import TwoRRobot


def _joint_vector(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain exactly two finite values")
    return value


def _gains(value: float | np.ndarray, name: str) -> np.ndarray:
    gain = np.asarray(value, dtype=float)
    if gain.ndim == 0:
        gain = np.full(2, float(gain))
    if gain.shape != (2,) or not np.all(np.isfinite(gain)) or np.any(gain < 0):
        raise ValueError(f"{name} must be a non-negative scalar or length-2 vector")
    return gain


@dataclass(frozen=True)
class JointPD:
    """Joint-space proportional-derivative controller."""

    kp: float | np.ndarray
    kd: float | np.ndarray
    torque_limit: float | None = None

    def __post_init__(self) -> None:
        _gains(self.kp, "kp")
        _gains(self.kd, "kd")
        if self.torque_limit is not None and (self.torque_limit <= 0 or not np.isfinite(self.torque_limit)):
            raise ValueError("torque_limit must be positive and finite")

    def __call__(self, q, qd, q_des, qd_des) -> np.ndarray:
        q = _joint_vector(q, "q")
        qd = _joint_vector(qd, "qd")
        q_des = _joint_vector(q_des, "q_des")
        qd_des = _joint_vector(qd_des, "qd_des")
        tau = _gains(self.kp, "kp") * (q_des - q) + _gains(self.kd, "kd") * (qd_des - qd)
        if self.torque_limit is not None:
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        return tau.astype(float, copy=False)


@dataclass(frozen=True)
class ComputedTorqueController:
    """Inverse-dynamics controller using desired joint acceleration."""

    robot: TwoRRobot
    kp: float | np.ndarray
    kd: float | np.ndarray
    torque_limit: float | None = None

    def __post_init__(self) -> None:
        _gains(self.kp, "kp")
        _gains(self.kd, "kd")
        if self.torque_limit is not None and (self.torque_limit <= 0 or not np.isfinite(self.torque_limit)):
            raise ValueError("torque_limit must be positive and finite")

    def __call__(self, q, qd, q_des, qd_des, qdd_des=None) -> np.ndarray:
        q = _joint_vector(q, "q")
        qd = _joint_vector(qd, "qd")
        q_des = _joint_vector(q_des, "q_des")
        qd_des = _joint_vector(qd_des, "qd_des")
        qdd_reference = np.zeros(2) if qdd_des is None else _joint_vector(qdd_des, "qdd_des")
        v = qdd_reference + _gains(self.kp, "kp") * (q_des - q) + _gains(self.kd, "kd") * (qd_des - qd)
        tau = self.robot.mass_matrix(q) @ v + self.robot.coriolis(q, qd) + self.robot.gravity(q)
        if self.torque_limit is not None:
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        return tau.astype(float, copy=False)
