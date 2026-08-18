"""Minimal 3D serial-chain manipulator model using screw axes."""

from dataclasses import dataclass

import numpy as np

from .se3 import adjoint, se3_exp


@dataclass(frozen=True)
class Serial3R:
    """Three-revolute-joint serial manipulator described by space-frame screws."""

    home: np.ndarray
    screw_axes: np.ndarray  # 6 x 3, space-frame twists

    def __post_init__(self) -> None:
        H = np.asarray(self.home, dtype=float)
        S = np.asarray(self.screw_axes, dtype=float)
        if H.shape != (4, 4) or S.shape != (6, 3):
            raise ValueError("home must be 4x4 and screw_axes must be 6x3")
        if not np.allclose(H[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("home must be a homogeneous transform")
        if not np.allclose(H[:3, :3].T @ H[:3, :3], np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(H[:3, :3]), 1.0, atol=1e-7):
            raise ValueError("home rotation must be valid")
        if not np.all(np.isfinite(H)) or not np.all(np.isfinite(S)):
            raise ValueError("home and screw axes must contain finite values")
        object.__setattr__(self, "home", H.copy())
        object.__setattr__(self, "screw_axes", S.copy())

    @property
    def joints(self) -> int:
        return 3

    def _q(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            raise ValueError("q must contain three finite joint values")
        return q

    def forward(self, q: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        q = self._q(q)
        T = np.eye(4)
        for i in range(3):
            T = T @ se3_exp(self.screw_axes[:, i], float(q[i]))
        return T @ self.home

    def space_jacobian(self, q: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        q = self._q(q)
        J = np.zeros((6, 3))
        T = np.eye(4)
        for i in range(3):
            J[:, i] = adjoint(T) @ self.screw_axes[:, i]
            T = T @ se3_exp(self.screw_axes[:, i], float(q[i]))
        return J
