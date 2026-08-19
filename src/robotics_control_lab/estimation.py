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
        self.x = np.asarray(x0, dtype=float).reshape(-1).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        n = self.x.size
        if self.A.shape != (n, n) or self.Q.shape != (n, n) or self.P.shape != (n, n):
            raise ValueError("state matrices have incompatible dimensions")
        if self.B.ndim != 2 or self.B.shape[0] != n:
            raise ValueError("B must have shape (state_dimension, input_dimension)")
        if self.H.ndim != 2 or self.H.shape[1] != n:
            raise ValueError("H must have shape (measurement_dimension, state_dimension)")
        m = self.H.shape[0]
        if self.R.shape != (m, m):
            raise ValueError("R must match the measurement dimension")
        for name, matrix in (("P0", self.P), ("Q", self.Q), ("R", self.R)):
            if not np.allclose(matrix, matrix.T, atol=1e-10):
                raise ValueError(f"{name} must be symmetric")
            if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
                raise ValueError(f"{name} must be positive semidefinite")
        try:
            np.linalg.cholesky(self.R + 1e-12 * np.eye(m))
        except np.linalg.LinAlgError as exc:
            raise ValueError("R must be positive definite enough for update") from exc

    def predict(self, u=None) -> KalmanState:
        u_vec = np.zeros(self.B.shape[1]) if u is None else np.asarray(u, dtype=float).reshape(-1)
        if u_vec.shape != (self.B.shape[1],):
            raise ValueError("input has incorrect dimension")
        if not np.all(np.isfinite(u_vec)):
            raise ValueError("input must contain finite values")
        self.x = self.A @ self.x + self.B @ u_vec
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        return KalmanState(self.x.copy(), self.P.copy())

    def update(self, z) -> KalmanState:
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.shape != (self.H.shape[0],):
            raise ValueError("measurement has incorrect dimension")
        if not np.all(np.isfinite(z)):
            raise ValueError("measurement must contain finite values")
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = np.linalg.solve(S, (self.P @ self.H.T).T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance is singular") from exc
        self.x = self.x + K @ innovation
        identity = np.eye(self.P.shape[0])
        # Joseph form preserves covariance symmetry/positive semidefiniteness better.
        self.P = (identity - K @ self.H) @ self.P @ (identity - K @ self.H).T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return KalmanState(self.x.copy(), self.P.copy())


def complementary_fusion(angle_prediction: float, angle_measurement: float, alpha: float) -> float:
    """Fuse two scalar angle estimates with wrap-aware interpolation."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not np.isfinite(angle_prediction) or not np.isfinite(angle_measurement):
        raise ValueError("angle inputs must be finite")
    error = (angle_measurement - angle_prediction + np.pi) % (2 * np.pi) - np.pi
    fused = angle_prediction + alpha * error
    return float((fused + np.pi) % (2 * np.pi) - np.pi)
