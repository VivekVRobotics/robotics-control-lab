"""Small linear MPC implementation based on a lifted convex quadratic program."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MPCResult:
    control: np.ndarray
    sequence: np.ndarray
    cost: float
    iterations: int
    converged: bool


def _symmetric_positive_semidefinite(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")


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
    max_iterations: int = 500,
    tolerance: float = 1e-7,
) -> MPCResult:
    """Solve a finite-horizon linear quadratic MPC.

    With no bounds, the lifted QP is solved directly. With box bounds, a
    projected-gradient method solves the actual constrained convex QP rather
    than clipping an unconstrained solution after optimization.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x = np.asarray(x0, dtype=float).reshape(-1)
    ref = np.asarray(x_ref, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be matrices")
    n, m = B.shape
    if A.shape != (n, n) or x.shape != (n,):
        raise ValueError("A, B, and x0 dimensions are inconsistent")
    if horizon <= 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("horizon, max_iterations, and tolerance must be positive")
    _symmetric_positive_semidefinite(Q, "Q")
    _symmetric_positive_semidefinite(R, "R")
    if Q.shape != (n, n) or R.shape != (m, m):
        raise ValueError("Q and R dimensions are inconsistent with A/B")
    if ref.shape == (n,):
        ref = np.tile(ref, (horizon, 1))
    if ref.shape != (horizon, n):
        raise ValueError("x_ref must have shape (n,) or (horizon, n)")

    lo = np.full(m, -np.inf) if u_lower is None else np.asarray(u_lower, dtype=float)
    hi = np.full(m, np.inf) if u_upper is None else np.asarray(u_upper, dtype=float)
    if lo.shape != (m,) or hi.shape != (m,) or np.any(lo > hi) or not np.all(np.isfinite(np.concatenate((lo[np.isfinite(lo)], hi[np.isfinite(hi)])))):
        raise ValueError("u_lower/u_upper must match input dimension and contain valid finite bounds")
    bounded = u_lower is not None or u_upper is not None
    lower = np.tile(lo, horizon)
    upper = np.tile(hi, horizon)

    powers = [np.eye(n)]
    for _ in range(horizon):
        powers.append(A @ powers[-1])
    S = np.zeros((horizon * n, horizon * m))
    for i in range(horizon):
        for j in range(i + 1):
            S[i * n:(i + 1) * n, j * m:(j + 1) * m] = powers[i - j] @ B
    x_free = np.concatenate([powers[i + 1] @ x for i in range(horizon)])
    Qbar = np.kron(np.eye(horizon), Q)
    Rbar = np.kron(np.eye(horizon), R)
    H = S.T @ Qbar @ S + Rbar
    H = 0.5 * (H + H.T) + 1e-10 * np.eye(horizon * m)
    g = S.T @ Qbar @ (x_free - ref.reshape(-1))

    if not bounded:
        u_vec = -np.linalg.solve(H, g)
        iterations = 1
        converged = True
    else:
        lipschitz = float(np.max(np.linalg.eigvalsh(H)))
        step = 1.0 / max(lipschitz, 1e-12)
        u_vec = np.clip(np.zeros(horizon * m), lower, upper)
        converged = False
        for iterations in range(1, max_iterations + 1):
            gradient = H @ u_vec + g
            updated = np.clip(u_vec - step * gradient, lower, upper)
            if np.linalg.norm(updated - u_vec, ord=np.inf) <= tolerance:
                u_vec = updated
                converged = True
                break
            u_vec = updated

    u = u_vec.reshape(horizon, m)
    cost = 0.0
    state = x.copy()
    for k in range(horizon):
        error = state - ref[k]
        cost += float(error @ Q @ error + u[k] @ R @ u[k])
        state = A @ state + B @ u[k]

    return MPCResult(control=u[0].copy(), sequence=u, cost=cost, iterations=iterations, converged=converged)
