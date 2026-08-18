"""SE(3), twists, exponential coordinates, and adjoint operators."""

from dataclasses import dataclass
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(theta) / theta * K + (1.0 - np.cos(theta)) / theta**2 * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace_arg = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(trace_arg))
    if theta < 1e-10:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2.0 * np.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def se3(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    p = np.asarray(p, dtype=float).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def adjoint(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
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
    w = np.array([X[2, 1], X[0, 2], X[1, 0]])
    return np.concatenate((w, X[:3, 3]))


def se3_exp(V: np.ndarray, theta: float = 1.0) -> np.ndarray:
    V = np.asarray(V, dtype=float).reshape(6)
    R = so3_exp(V[:3] * theta)
    w = V[:3]
    v = V[3:]
    wn = float(np.linalg.norm(w))
    if wn < 1e-12:
        p = v * theta
    else:
        W = skew(w)
        A = np.eye(3) * theta + (1.0 - np.cos(wn * theta)) / wn**2 * W + (wn * theta - np.sin(wn * theta)) / wn**3 * (W @ W)
        p = A @ v
    return se3(R, p)


@dataclass(frozen=True)
class PoseError:
    rotation: np.ndarray
    translation: np.ndarray
    twist: np.ndarray


def pose_error(T_current: np.ndarray, T_desired: np.ndarray) -> PoseError:
    Tc = np.asarray(T_current, dtype=float)
    Td = np.asarray(T_desired, dtype=float)
    T_err = np.linalg.inv(Tc) @ Td
    phi = so3_log(T_err[:3, :3])
    p = T_err[:3, 3]
    return PoseError(rotation=phi, translation=p, twist=np.concatenate((phi, p)))
