"""Deterministic time-domain simulation of the 2R dynamics."""

from dataclasses import dataclass

import numpy as np

from .dynamics import TwoRRobot


@dataclass(frozen=True)
class SimulationResult:
    """Time history produced by a fixed-step simulator."""

    time: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    tau: np.ndarray


def simulate(
    robot: TwoRRobot,
    controller,
    q0: np.ndarray | tuple[float, float],
    qd0: np.ndarray | tuple[float, float],
    q_des_fn,
    duration: float = 5.0,
    dt: float = 0.002,
) -> SimulationResult:
    """Simulate a robot under a callable controller using semi-implicit Euler.

    ``q_des_fn(t)`` must return ``(q_des, qd_des[, qdd_des])``. Controllers
    accepting a fifth argument receive the desired acceleration as well.
    """
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")

    q = np.asarray(q0, dtype=float).copy()
    qd = np.asarray(qd0, dtype=float).copy()
    if q.shape != (2,) or qd.shape != (2,):
        raise ValueError("q0 and qd0 must each contain two joints")

    steps = int(np.ceil(duration / dt))
    time = np.linspace(0.0, duration, steps + 1)
    q_hist = np.zeros((steps + 1, 2))
    qd_hist = np.zeros((steps + 1, 2))
    qdd_hist = np.zeros((steps + 1, 2))
    tau_hist = np.zeros((steps + 1, 2))
    q_hist[0] = q
    qd_hist[0] = qd

    for i, t in enumerate(time[:-1]):
        reference = tuple(q_des_fn(float(t)))
        if len(reference) == 2:
            q_des, qd_des = reference
            tau = controller(q, qd, np.asarray(q_des), np.asarray(qd_des))
        elif len(reference) == 3:
            q_des, qd_des, qdd_des = reference
            try:
                tau = controller(q, qd, np.asarray(q_des), np.asarray(qd_des), np.asarray(qdd_des))
            except TypeError:
                tau = controller(q, qd, np.asarray(q_des), np.asarray(qd_des))
        else:
            raise ValueError("q_des_fn must return two or three arrays")

        qdd = robot.acceleration(q, qd, tau)
        h = min(dt, duration - time[i])
        qd = qd + qdd * h
        q = q + qd * h

        q_hist[i + 1] = q
        qd_hist[i + 1] = qd
        qdd_hist[i + 1] = qdd
        tau_hist[i + 1] = tau

    return SimulationResult(time=time, q=q_hist, qd=qd_hist, qdd=qdd_hist, tau=tau_hist)
