"""Deterministic time-domain simulation of the 2R dynamics."""

from dataclasses import dataclass
from typing import Callable

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


def _validate_vector(name: str, value: np.ndarray | tuple[float, float]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly two finite values")
    return array.copy()


def simulate(
    robot: TwoRRobot,
    controller: Callable,
    q0: np.ndarray | tuple[float, float],
    qd0: np.ndarray | tuple[float, float],
    q_des_fn: Callable,
    duration: float = 5.0,
    dt: float = 0.002,
) -> SimulationResult:
    """Simulate a robot under a callable controller using semi-implicit Euler.

    ``q_des_fn(t)`` must return ``(q_des, qd_des)`` or
    ``(q_des, qd_des, qdd_des)``. A controller must accept the corresponding
    four- or five-argument call; controller exceptions are never swallowed.
    """
    if duration <= 0 or dt <= 0 or not np.isfinite(duration + dt):
        raise ValueError("duration and dt must be positive and finite")

    q = _validate_vector("q0", q0)
    qd = _validate_vector("qd0", qd0)
    steps = int(np.ceil(duration / dt))
    time = np.minimum(np.arange(steps + 1, dtype=float) * dt, duration)
    time[-1] = duration

    q_hist = np.zeros((steps + 1, 2))
    qd_hist = np.zeros((steps + 1, 2))
    qdd_hist = np.zeros((steps + 1, 2))
    tau_hist = np.zeros((steps + 1, 2))
    q_hist[0] = q
    qd_hist[0] = qd

    for i, t in enumerate(time[:-1]):
        h = float(time[i + 1] - time[i])
        if h <= 0:
            raise RuntimeError("simulation time grid is not strictly increasing")

        reference = tuple(q_des_fn(float(t)))
        if len(reference) == 2:
            q_des, qd_des = map(lambda v: _validate_vector("reference", v), reference)
            tau = controller(q, qd, q_des, qd_des)
        elif len(reference) == 3:
            q_des, qd_des, qdd_des = map(lambda v: _validate_vector("reference", v), reference)
            tau = controller(q, qd, q_des, qd_des, qdd_des)
        else:
            raise ValueError("q_des_fn must return two or three arrays")

        tau = _validate_vector("controller output", tau)
        qdd = robot.acceleration(q, qd, tau)
        qd = qd + qdd * h
        q = q + qd * h

        q_hist[i + 1] = q
        qd_hist[i + 1] = qd
        qdd_hist[i + 1] = qdd
        tau_hist[i + 1] = tau

    return SimulationResult(time=time, q=q_hist, qd=qd_hist, qdd=qdd_hist, tau=tau_hist)
