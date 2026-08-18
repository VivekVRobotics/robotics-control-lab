# Research benchmark and design rationale

This repository uses established robotics-software and robotics-education stacks as reference points for what a serious laboratory should cover.

## Reference 1 — Modern Robotics

Modern Robotics organizes the subject around rigid-body motion, forward/inverse kinematics, Jacobians, dynamics, trajectory generation, and control. This lab follows the same progression but keeps the first implementation small and inspectable.

Reference: https://modernrobotics.northwestern.edu/

## Reference 2 — Drake

Drake's multibody/planning stack couples multibody dynamics with inverse kinematics, trajectory computation, collision checking, and mathematical optimization. The lab therefore treats planning and optimization as first-class layers instead of only adding controllers.

Reference: https://drake.mit.edu/doxygen_cxx/group__planning.html

## Reference 3 — ros2_control

ros2_control separates controllers from hardware through a controller-manager/resource-manager architecture and explicit state/command interfaces. The lab mirrors this separation with `HardwareInterface`, `SimulatedHardware`, a URDF reference, and a documented ROS 2 boundary.

Reference: https://control.ros.org/rolling/doc/ros2_control/doc/index.html

## Current implementation map

```text
SE(3) / twists / adjoints
        ↓
3D serial-chain kinematics
        ↓
2R + numerical IK
        ↓
Jacobian / manipulability / singularities
        ↓
trajectory generation
        ↓
rigid-body dynamics
        ↓
PD / PID / computed torque
        ↓
planning: A* + RRT
        ↓
path smoothing + box-QP safety projection
        ↓
state estimation + disturbance residuals
        ↓
linear MPC baseline
        ↓
hardware abstraction + HIL
        ↓
URDF + ROS 2 integration boundary
        ↓
benchmarks + sim-to-real protocol
```

## Design rule

A capability only counts as complete when it has an implementation boundary, an example or documented integration path, and a verification strategy. Optional production dependencies such as ROS 2 and dedicated QP solvers are kept outside the numerical core so the lab remains reproducible on a normal Python installation.
