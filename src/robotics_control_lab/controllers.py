"""Joint-space and model-based controllers for the 2R robot."""

from dataclasses import dataclass

import numpy as np

from .dynamics import TwoRRobot


@dataclass(frozen=True)
class JointPD:
    """Joint-space proportional-derivative controller."""

    kp: float | np.ndarray
    kd: float | np.ndarray
    torque_limit: float | None = None

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.kp) < 0) or np.any(np.asarray(self.kd) < 0):
            raise ValueError("controller gains must be non-negative")
        if self.torque_limit is not None and self.torque_limit <= 0:
            raise ValueError("torque_limit must be positive")

    def __call__(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray, qd_des: np.ndarray) -> np.ndarray:
        tau = np.asarray(self.kp) * (q_des - q) + np.asarray(self.kd) * (qd_des - qd)
        if self.torque_limit is not None:
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        return np.asarray(tau, dtype=float)


@dataclass(frozen=True)
class ComputedTorqueController:
    """Inverse-dynamics controller using desired joint acceleration."""

    robot: TwoRRobot
    kp: float | np.ndarray
    kd: float | np.ndarray
    torque_limit: float | None = None

    def __call__(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_des: np.ndarray,
        qd_des: np.ndarray,
        qdd_des: np.ndarray | None = None,
    ) -> np.ndarray:
        qdd_reference = np.zeros(2) if qdd_des is None else np.asarray(qdd_des, dtype=float)
        v = qdd_reference + np.asarray(self.kp) * (q_des - q) + np.asarray(self.kd) * (qd_des - qd)
        tau = self.robot.mass_matrix(q) @ v + self.robot.coriolis(q, qd) + self.robot.gravity(q)
        if self.torque_limit is not None:
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        return np.asarray(tau, dtype=float)
