"""Small linear MPC implementation based on finite-horizon quadratic cost."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MPCResult:
    control: np.ndarray
    sequence: np.ndarray
    cost: float


def linear_mpc(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    x_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    horizon: int = 10,
    u_lower: np.ndarray | None = None,
    u_upper: np.ndarray | None = None,
) -> MPCResult:
    """Solve a finite-horizon linear quadratic MPC by direct shooting.

    For portability this reference implementation enumerates the unconstrained
    quadratic solution through the lifted least-squares system and then clips
    each control move to box limits. It is intended as a benchmark baseline,
    not as a replacement for a production QP solver.
    """
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    x = np.asarray(x0, dtype=float).reshape(-1); ref = np.asarray(x_ref, dtype=float)
    Q = np.asarray(Q, dtype=float); R = np.asarray(R, dtype=float)
    n, m = B.shape
    if ref.shape == (n,):
        ref = np.tile(ref, (horizon, 1))
    if ref.shape != (horizon, n) or horizon <= 0:
        raise ValueError("x_ref must have shape (horizon, n)")
    powers = [np.eye(n)]
    for _ in range(horizon):
        powers.append(A @ powers[-1])
    S = np.zeros((horizon * n, horizon * m))
    for i in range(horizon):
        for j in range(i + 1):
            S[i*n:(i+1)*n, j*m:(j+1)*m] = powers[i-j] @ B
    x_free = np.concatenate([powers[i+1] @ x for i in range(horizon)])
    Qbar = np.kron(np.eye(horizon), Q)
    Rbar = np.kron(np.eye(horizon), R)
    H = S.T @ Qbar @ S + Rbar + 1e-9 * np.eye(horizon * m)
    g = S.T @ Qbar @ (x_free - ref.reshape(-1))
    u = -np.linalg.solve(H, g).reshape(horizon, m)
    if u_lower is not None or u_upper is not None:
        lo = np.full(m, -np.inf) if u_lower is None else np.asarray(u_lower, dtype=float)
        hi = np.full(m, np.inf) if u_upper is None else np.asarray(u_upper, dtype=float)
        u = np.clip(u, lo, hi)
    states = []
    cost = 0.0
    state = x.copy()
    for k in range(horizon):
        e = state - ref[k]
        cost += float(e @ Q @ e + u[k] @ R @ u[k])
        states.append(state.copy())
        state = A @ state + B @ u[k]
    return MPCResult(control=u[0].copy(), sequence=u, cost=cost)
