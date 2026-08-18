# Robotics Control Lab

A from-scratch robotics engineering laboratory covering **SE(3), screw theory, 3D serial-chain kinematics, inverse kinematics, Jacobians, rigid-body dynamics, feedback control, trajectory generation, motion planning, safety filtering, state estimation, MPC, hardware abstraction, HIL, and reproducible benchmarking**.

The goal is not to hide robotics behind a framework. Core equations are implemented explicitly, tested numerically, and connected into reproducible experiments.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/VivekVRobotics/robotics-control-lab/ci.yml?branch=main&label=CI)

## Architecture

```text
SE(3) / twists / adjoints
        ↓
3D serial-chain kinematics
        ↓
2R analytical + numerical IK
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
path smoothing + QP safety projection
        ↓
state estimation + disturbance residuals
        ↓
linear MPC baseline
        ↓
hardware abstraction + HIL
        ↓
URDF + ROS 2 integration boundary
        ↓
benchmarks + sim-to-real methodology
```

## Capability map

| Area | Implementation |
|---|---|
| SE(3) | homogeneous transforms, SO(3) exp/log |
| Screw theory | twists, hat/vee operators, adjoint |
| 3D manipulator | 3R screw-axis forward kinematics + space Jacobian |
| 2R kinematics | analytic FK / IK + workspace checks |
| Numerical IK | damped least squares + joint limits |
| Differential kinematics | analytic Jacobian + damped pseudoinverse |
| Singularities | manipulability + tolerance checks |
| Trajectories | cubic + quintic profiles |
| Dynamics | `M(q)`, `C(q,qd)`, `G(q)` |
| Energy | kinetic + potential + total energy |
| Controllers | PID, PD, computed torque |
| Safety | limits, slew rate, boxed QP projection |
| Planning | A* grid search + deterministic RRT |
| Optimization | waypoint smoothing |
| Estimation | linear Kalman filter + complementary fusion |
| Disturbances | momentum residual + residual gate |
| MPC | finite-horizon linear quadratic baseline |
| Hardware | abstract read/write interface + simulated backend |
| HIL | deterministic hardware cycle adapter |
| Robot description | reference URDF |
| ROS 2 | explicit integration boundary and interface model |
| Identification | first-order step-response fit |
| Verification | numerical tests + CI matrix + compile gate |

## Repository structure

```text
robotics-control-lab/
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   ├── benchmarks.md
│   ├── research-benchmarks.md
│   ├── research-roadmap.md
│   └── robotics-foundations.md
├── examples/
│   ├── damped_ik_demo.py
│   ├── jacobian_singularity.py
│   └── track_step_response.py
├── robots/
│   └── planar_2r.urdf
├── ros2/
│   └── README.md
├── src/robotics_control_lab/
│   ├── controllers.py
│   ├── dynamics.py
│   ├── energy.py
│   ├── estimation.py
│   ├── hardware.py
│   ├── identification.py
│   ├── ik.py
│   ├── jacobian.py
│   ├── manipulator3d.py
│   ├── mpc.py
│   ├── observers.py
│   ├── optimization.py
│   ├── pid.py
│   ├── planning.py
│   ├── planar_arm.py
│   ├── safety.py
│   ├── se3.py
│   ├── simulation.py
│   ├── trajectory.py
│   └── visualization.py
├── tests/
│   ├── test_advanced_stack.py
│   ├── test_control.py
│   └── test_robotics_math.py
├── .gitignore
└── pyproject.toml
```

## Quick start

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

python -m pip install -e ".[dev]"
pytest -q
ruff check src tests examples
```

## Flagship experiments

```bash
python examples/track_step_response.py
python examples/damped_ik_demo.py
python examples/jacobian_singularity.py
```

## Research-informed scope

The stack is intentionally benchmarked against established robotics ecosystems. Modern Robotics provides a broad progression from rigid-body motion and screw theory through kinematics, Jacobians, dynamics, trajectories, and control. Drake couples multibody models with inverse kinematics, trajectory optimization, collision checking, and mathematical programming. ros2_control defines a controller/hardware architecture with explicit state and command interfaces and a controller manager separating control logic from hardware access.

References:

- Modern Robotics: https://modernrobotics.northwestern.edu/
- Drake planning: https://drake.mit.edu/doxygen_cxx/group__planning.html
- ros2_control: https://control.ros.org/

## Engineering quality bar

- Equations remain visible and auditable.
- Numerical behavior is verified independently where practical.
- Invalid physical parameters fail early.
- Optional production dependencies stay outside the numerical core.
- Planning/control interfaces are deterministic and benchmarkable.
- Safety boundaries are explicit.
- Examples are reproducible.
- CI spans supported Python versions and compiles the source tree.
- Simulation results are not treated as hardware validation.

## Roadmap status

### Advanced stack

- [x] SE(3) / twists / adjoints
- [x] 3D serial manipulator
- [x] Numerical IK and Cartesian velocity control
- [x] Rigid-body dynamics
- [x] Motion planning: A* + RRT
- [x] Trajectory smoothing
- [x] Box-QP safety projection
- [x] State estimation
- [x] Disturbance residuals
- [x] Linear MPC baseline
- [x] URDF reference model
- [x] ROS 2 integration boundary
- [x] Hardware abstraction
- [x] HIL adapter
- [x] System-identification baseline
- [x] Benchmark methodology
- [x] Sim-to-real methodology

### Next research depth

- [ ] Full operational-space dynamics
- [ ] Hierarchical null-space control
- [ ] RRT* / informed sampling
- [ ] Continuous collision geometry
- [ ] General QP solver integration
- [ ] Nonlinear MPC
- [ ] Extended/unscented Kalman filtering
- [ ] Online parameter estimation
- [ ] Disturbance-observer closed-loop experiments
- [ ] 6+ DOF benchmark manipulator
- [ ] Real ROS 2 controller package
- [ ] Real-time hardware driver
- [ ] HIL timing/jitter metrics
- [ ] Sim-to-real parameter randomization study
- [ ] Comparative benchmark suite against external robotics libraries

## Scope

This is an educational and research-oriented laboratory. The simulator and safety utilities are not safety-certified and must not be used as a substitute for validated industrial robot control, real-time guarantees, or a certified safety system.
