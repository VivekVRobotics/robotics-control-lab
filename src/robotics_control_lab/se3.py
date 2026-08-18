"""SE(3), twists, exponential coordinates, and adjoint operators."""

from dataclasses import dataclass

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    if not np.all(np.isfinite(v)):
        raise ValueError("vector must contain finite values")
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-8:
        theta2 = theta * theta
        return np.eye(3) + (1.0 - theta2 / 6.0) * K + (0.5 - theta2 / 24.0) * (K @ K)
    return np.eye(3) + np.sin(theta) / theta * K + (1.0 - np.cos(theta)) / theta**2 * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Return a principal rotation vector, including the numerically difficult pi case."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(R)) or not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-7):
        raise ValueError("R must be a valid rotation matrix")

    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-10:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=float,
        )
    if np.pi - theta < 1e-7:
        diag = np.maximum(np.diag(R) + 1.0, 0.0)
        axis = np.sqrt(diag / 2.0)
        idx = int(np.argmax(axis))
        if axis[idx] < 1e-8:
            raise ValueError("could not recover rotation axis near pi")
        if idx == 0:
            axis[1] = np.copysign(axis[1], R[0, 1] + R[1, 0])
            axis[2] = np.copysign(axis[2], R[0, 2] + R[2, 0])
        elif idx == 1:
            axis[0] = np.copysign(axis[0], R[0, 1] + R[1, 0])
            axis[2] = np.copysign(axis[2], R[1, 2] + R[2, 1])
        else:
            axis[0] = np.copysign(axis[0], R[0, 2] + R[2, 0])
            axis[1] = np.copysign(axis[1], R[1, 2] + R[2, 1])
        axis /= np.linalg.norm(axis)
        return theta * axis

    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )


def se3(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    p = np.asarray(p, dtype=float).reshape(3)
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-7):
        raise ValueError("R must be a valid rotation matrix")
    if not np.all(np.isfinite(p)):
        raise ValueError("translation must contain finite values")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def adjoint(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("T must be a homogeneous transform")
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, 3:] = R
    A[3:, :3] = skew(p) @ R
    return A


def twist_hat(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float).reshape(6)
    out = np.zeros((4, 4))
    out[:3, :3] = skew(V[:3])
    out[:3, 3] = V[3:]
    return out


def twist_vee(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(4, 4)
    if not np.allclose(X[:3, :3] + X[:3, :3].T, 0.0, atol=1e-9) or not np.allclose(X[3], 0.0, atol=1e-12):
        raise ValueError("X is not a valid twist matrix")
    w = np.array([X[2, 1], X[0, 2], X[1, 0]])
    return np.concatenate((w, X[:3, 3]))


def se3_exp(V: np.ndarray, theta: float = 1.0) -> np.ndarray:
    V = np.asarray(V, dtype=float).reshape(6)
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    R = so3_exp(V[:3] * theta)
    w = V[:3]
    v = V[3:]
    wn = float(np.linalg.norm(w))
    if wn < 1e-12:
        p = v * theta
    else:
        W = skew(w)
        angle = wn * theta
        A = np.eye(3) * theta + (1.0 - np.cos(angle)) / wn**2 * W + (angle - np.sin(angle)) / wn**3 * (W @ W)
        p = A @ v
    return se3(R, p)


@dataclass(frozen=True)
class PoseError:
    rotation: np.ndarray
    translation: np.ndarray
    twist: np.ndarray


def pose_error(T_current: np.ndarray, T_desired: np.ndarray) -> PoseError:
    Tc = np.asarray(T_current, dtype=float).reshape(4, 4)
    Td = np.asarray(T_desired, dtype=float).reshape(4, 4)
    T_err = np.linalg.inv(Tc) @ Td
    phi = so3_log(T_err[:3, :3])
    p = T_err[:3, 3]
    return PoseError(rotation=phi, translation=p, twist=np.concatenate((phi, p)))
