"""State-estimation utilities: complementary fusion and linear Kalman filtering."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class KalmanState:
    x: np.ndarray
    covariance: np.ndarray


class LinearKalmanFilter:
    """Discrete linear Kalman filter for x[k+1]=Ax[k]+Bu[k]+w."""

    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        n = self.x.shape[0]
        if self.A.shape != (n, n) or self.Q.shape != (n, n) or self.P.shape != (n, n):
            raise ValueError("state matrices have incompatible dimensions")

    def predict(self, u=None) -> KalmanState:
        u_vec = np.zeros(self.B.shape[1]) if u is None else np.asarray(u, dtype=float)
        self.x = self.A @ self.x + self.B @ u_vec
        self.P = self.A @ self.P @ self.A.T + self.Q
        return KalmanState(self.x.copy(), self.P.copy())

    def update(self, z) -> KalmanState:
        z = np.asarray(z, dtype=float)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return KalmanState(self.x.copy(), self.P.copy())


def complementary_fusion(angle_prediction: float, angle_measurement: float, alpha: float) -> float:
    """Fuse two scalar angle estimates with wrap-aware interpolation."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    error = (angle_measurement - angle_prediction + np.pi) % (2 * np.pi) - np.pi
    return angle_prediction + alpha * error
