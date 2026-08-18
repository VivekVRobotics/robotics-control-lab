"""Optional Matplotlib visualizations for simulation results."""

from pathlib import Path

from .planar_arm import Planar2R
from .simulation import SimulationResult


def plot_joint_tracking(result: SimulationResult, q_des=None, save_path: str | Path | None = None):
    """Plot joint position/velocity histories; return the Matplotlib figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    axes[0].plot(result.time, result.q[:, 0], label="q1")
    axes[0].plot(result.time, result.q[:, 1], label="q2")
    if q_des is not None:
        axes[0].plot(result.time, q_des[:, 0], "--", label="q1 ref")
        axes[0].plot(result.time, q_des[:, 1], "--", label="q2 ref")
    axes[0].set_ylabel("Joint position [rad]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(result.time, result.qd[:, 0], label="dq1")
    axes[1].plot(result.time, result.qd[:, 1], label="dq2")
    axes[1].set_ylabel("Joint velocity [rad/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_workspace(arm: Planar2R, samples: int = 180, save_path: str | Path | None = None):
    """Plot the reachable annulus of a planar 2R arm."""
    import numpy as np
    import matplotlib.pyplot as plt

    q = np.linspace(-np.pi, np.pi, samples)
    outer = arm.l1 + arm.l2
    inner = abs(arm.l1 - arm.l2)
    theta = np.linspace(0, 2 * np.pi, samples)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(outer * np.cos(theta), outer * np.sin(theta), label="outer workspace")
    if inner > 0:
        ax.plot(inner * np.cos(theta), inner * np.sin(theta), label="inner boundary")
    ax.axhline(0, linewidth=0.6)
    ax.axvline(0, linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Planar 2R Reachable Workspace")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig
