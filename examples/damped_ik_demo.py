"""Solve a Cartesian target with damped least-squares IK."""

import numpy as np

from robotics_control_lab import Planar2R, damped_least_squares_ik

arm = Planar2R(0.6, 0.4)
target = np.array([0.55, 0.15])
result = damped_least_squares_ik(
    arm,
    target,
    q0=(0.2, -0.4),
    damping=1e-2,
    joint_limits=((-np.pi, np.pi), (-np.pi, np.pi)),
)

print(f"converged={result.converged}")
print(f"iterations={result.iterations}")
print(f"residual={result.residual:.3e}")
print(f"joint_solution={result.q}")
print(f"end_effector={arm.forward(*result.q)}")
