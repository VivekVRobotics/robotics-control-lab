# Research Roadmap

`robotics-control-lab` is intentionally organized as a ladder from transparent mathematics to systems engineering.

## Tier 1 — Mathematical robotics

- configuration and task space
- rigid-body transforms
- forward and inverse kinematics
- Jacobians, singularities, manipulability
- statics and wrench mapping

## Tier 2 — Dynamics

- inertia matrix `M(q)`
- Coriolis/centrifugal terms
- gravity
- inverse/forward dynamics
- kinetic and potential energy
- physical consistency checks

## Tier 3 — Motion generation

- cubic trajectories
- quintic trajectories
- velocity/acceleration constraints
- time scaling
- joint-space and task-space references

## Tier 4 — Control

- PID
- joint PD
- computed torque
- task-space velocity control
- torque/velocity saturation
- safety and slew-rate limiting

## Tier 5 — Numerical robustness

- damped least-squares IK
- finite-difference cross-checks
- singularity diagnostics
- deterministic simulation
- numerical tolerances documented in tests

## Tier 6 — Planning and optimization

Planned extensions:

- configuration-space obstacle models
- graph search / sampling-based planning
- collision checking
- trajectory optimization
- quadratic-programming safety filters
- null-space objectives

## Tier 7 — Real robotics

Planned extensions:

- URDF robot descriptions
- ROS 2 interfaces
- ros2_control controller boundaries
- hardware abstraction
- sensor timing and state estimation
- hardware-in-the-loop tests

## Design philosophy

The repository should avoid becoming a black box. Every major algorithm should have:

1. a small implementation that can be read;
2. a mathematical explanation;
3. an independently motivated test;
4. an example showing how it behaves;
5. an explicit statement of numerical and physical assumptions.

This mirrors the educational structure of Modern Robotics while borrowing the separation between modeling, control, and optimization found in larger robotics toolboxes such as Drake and ros2_control.
