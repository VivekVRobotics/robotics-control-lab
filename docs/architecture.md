# Architecture

The lab is organized as a small robotics stack rather than a collection of independent scripts.

```text
                 ┌─────────────────────────┐
                 │      Experiment API     │
                 │      examples/          │
                 └────────────┬────────────┘
                              │
                 ┌────────────▼────────────┐
                 │     Simulation layer     │
                 │  state integration + I/O │
                 └────────────┬────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
│   Controllers   │   │    Dynamics     │   │   Kinematics    │
│ PID / PD / CT   │   │ M C G / qdd     │   │ FK / IK / J     │
└────────────────┘   └─────────────────┘   └─────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Robot models    │
                    │    Planar2R       │
                    └───────────────────┘
```

## Module boundaries

### `planar_arm.py`
Geometric model and forward/inverse position kinematics. It has no dependency on the simulator, controller, or plotting layer.

### `jacobian.py`
Differential kinematics derived from the planar arm geometry. It exposes the Jacobian, manipulability, and singularity checks.

### `dynamics.py`
Physical model in standard manipulator form:

```text
M(q) qdd + C(q, qd) + G(q) = tau
```

The implementation keeps the matrices/vectors explicit for inspectability and educational value.

### `controllers.py`
Controllers consume the robot state and desired state. `JointPD` is model-free; `ComputedTorqueController` uses the dynamics model for inverse-dynamics compensation.

### `simulation.py`
Owns time integration and records state history. It does not contain robot equations or controller gains, which keeps experiments composable.

### `visualization.py`
Optional Matplotlib helpers. Visualization is intentionally outside the numerical core, so the core package remains usable in headless or embedded environments.

## Design principles

1. **Equations remain visible.** Core robotics mathematics is expressed directly in Python rather than hidden behind a framework.
2. **Public seams are small.** Each module has a narrow API that can be tested independently.
3. **Simulation is deterministic.** Fixed-step integration makes regression experiments reproducible.
4. **Validation happens at boundaries.** Invalid physical parameters and numerical inputs fail early.
5. **Examples exercise real APIs.** Example scripts use the same package interfaces as tests and downstream users.

## Extension path

The intended progression is:

```text
2R baseline
   → trajectory planning
   → Jacobian / differential control
   → dynamics + computed torque
   → numerical simulation
   → visualization
   → 3D manipulator models
   → ROS 2 interfaces
   → hardware-in-the-loop
   → experimental identification
```
