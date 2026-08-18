"""Minimal 3D serial-chain manipulator model using screw axes."""

from dataclasses import dataclass
import numpy as np

from .se3 import adjoint, se3, se3_exp


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
        object.__setattr__(self, "home", H)
        object.__setattr__(self, "screw_axes", S)

    def forward(self, q: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(3)
        T = np.eye(4)
        for i in range(3):
            T = T @ se3_exp(self.screw_axes[:, i], float(q[i]))
        return T @ self.home

    def space_jacobian(self, q: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(3)
        J = np.zeros((6, 3))
        T = np.eye(4)
        for i in range(3):
            J[:, i] = adjoint(T) @ self.screw_axes[:, i]
            T = T @ se3_exp(self.screw_axes[:, i], float(q[i]))
        return J
