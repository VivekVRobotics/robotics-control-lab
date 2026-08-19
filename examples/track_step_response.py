"""Run a model-based 2R tracking simulation and save plots.

Usage:
    python examples/track_step_response.py
"""

from pathlib import Path

import numpy as np

from robotics_control_lab import (
    ComputedTorqueController,
    Planar2R,
    TwoRParameters,
    TwoRRobot,
    cubic_interpolation,
    simulate,
)
from robotics_control_lab.visualization import plot_joint_tracking, plot_workspace


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts"
OUTPUT.mkdir(exist_ok=True)


def reference(t: float):
    duration = 2.0
    samples = cubic_interpolation(0.0, 1.0, duration, dt=0.002)
    idx = min(int(round(t / 0.002)), len(samples) - 1)
    q1, dq1 = samples[idx][1], samples[idx][2]
    q2, dq2 = -0.65 * q1, -0.65 * dq1
    qdd1 = 0.0
    qdd2 = 0.0
    if 0.0 <= t <= duration and idx not in {0, len(samples) - 1}:
        tau = t / duration
        qdd1 = 6.0 * (1.0 - 2.0 * tau) / duration**2
        qdd2 = -0.65 * qdd1
    return np.array([q1, q2]), np.array([dq1, dq2]), np.array([qdd1, qdd2])


def main() -> None:
    arm = Planar2R(0.55, 0.45)
    robot = TwoRRobot(
        arm,
        TwoRParameters(m1=2.0, m2=1.2, lc1=0.275, lc2=0.225, i1=0.05, i2=0.02),
    )
    controller = ComputedTorqueController(robot, kp=36.0, kd=12.0, torque_limit=25.0)
    result = simulate(
        robot,
        controller,
        q0=(-0.4, 0.5),
        qd0=(0.0, 0.0),
        q_des_fn=reference,
    )

    references = np.array([reference(float(t))[0] for t in result.time])
    plot_joint_tracking(result, references, OUTPUT / "joint_tracking.png")
    plot_workspace(arm, save_path=OUTPUT / "workspace.png")
    error = np.linalg.norm(result.q[-1] - references[-1])
    print(f"final joint error: {error:.6f} rad")
    print(f"plots written to: {OUTPUT}")


if __name__ == "__main__":
    main()
