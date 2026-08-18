# Robotics Control Lab

A from-scratch robotics engineering laboratory focused on **kinematics, differential kinematics, inverse kinematics, rigid-body dynamics, feedback control, trajectories, numerical simulation, safety constraints, energy diagnostics, and reproducible verification**.

The goal is not to hide robotics behind a framework. The equations are implemented explicitly, tested numerically, and connected into reproducible experiments.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/VivekVRobotics/robotics-control-lab/ci.yml?branch=main&label=CI)

## Why this repository exists

A strong robotics portfolio should demonstrate more than a finished robot video. It should show that the engineer understands the stack underneath the robot:

```text
rigid-body geometry
        ↓
forward / inverse kinematics
        ↓
Jacobian / singularity analysis
        ↓
trajectory generation
        ↓
rigid-body dynamics
        ↓
feedback + inverse-dynamics control
        ↓
simulation + energy analysis
        ↓
safety constraints + verification
```

This repository is built around that progression.

## Current capabilities

| Area | Implementation |
|---|---|
| Forward kinematics | Planar 2R arm |
| Analytic inverse kinematics | Elbow-up / elbow-down |
| Numerical inverse kinematics | Damped least squares + joint limits |
| Workspace validation | Reachability checks |
| Jacobian | Analytical 2×2 position Jacobian |
| Cartesian velocity control | Damped pseudoinverse |
| Manipulability | Yoshikawa planar measure |
| Singularity detection | Numerical tolerance check |
| Trajectory generation | Cubic + quintic profiles |
| PID control | Saturation + integral bounds |
| Joint PD | Position/velocity feedback |
| Computed torque | Inverse-dynamics compensation |
| Dynamics | `M(q)`, `C(q,qd)`, `G(q)` |
| Energy | Kinetic + potential + total mechanical energy |
| Simulation | Fixed-step semi-implicit Euler |
| Safety | Position / velocity / effort / slew-rate limiting |
| Visualization | Optional Matplotlib plots |
| Verification | Mathematical unit tests + lint + coverage CI |

## Repository structure

```text
robotics-control-lab/
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   ├── research-roadmap.md
│   └── robotics-foundations.md
├── examples/
│   ├── damped_ik_demo.py
│   ├── jacobian_singularity.py
│   └── track_step_response.py
├── src/robotics_control_lab/
│   ├── __init__.py
│   ├── controllers.py
│   ├── dynamics.py
│   ├── energy.py
│   ├── ik.py
│   ├── jacobian.py
│   ├── pid.py
│   ├── planar_arm.py
│   ├── safety.py
│   ├── simulation.py
│   ├── trajectory.py
│   └── visualization.py
├── tests/
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

## Run the flagship experiment

The repository includes an end-to-end computed-torque simulation of a planar 2R robot:

```bash
python examples/track_step_response.py
```

The experiment:

1. builds a physical 2R robot model,
2. generates a joint reference,
3. closes the loop with computed-torque control,
4. simulates the rigid-body dynamics,
5. reports tracking error,
6. writes joint-tracking and workspace plots.

## Explore damped inverse kinematics

```bash
python examples/damped_ik_demo.py
```

This demonstrates a numerically robust alternative to closed-form IK. The solver regularizes the inverse Jacobian near singular configurations and can enforce joint limits.

## Explore singularities

```bash
python examples/jacobian_singularity.py
```

This experiment shows how the Jacobian determinant / manipulability measure approaches zero as the arm approaches a straight configuration.

## Core API

```python
import numpy as np

from robotics_control_lab import (
    Planar2R,
    TwoRParameters,
    TwoRRobot,
    ComputedTorqueController,
    damped_least_squares_ik,
    simulate,
)

arm = Planar2R(0.6, 0.4)
robot = TwoRRobot(
    arm,
    TwoRParameters(m1=2.0, m2=1.0, lc1=0.3, lc2=0.2, i1=0.06, i2=0.02),
)

ik = damped_least_squares_ik(arm, target=(0.55, 0.15), q0=(0.0, 0.0))
print(ik.q, ik.residual)

controller = ComputedTorqueController(robot, kp=45.0, kd=14.0)
target = np.array([0.4, -0.3])
result = simulate(
    robot,
    controller,
    q0=(0.0, 0.0),
    qd0=(0.0, 0.0),
    q_des_fn=lambda _t: (target, np.zeros(2), np.zeros(2)),
    duration=1.5,
    dt=0.002,
)
print(result.q[-1])
```

## Engineering quality bar

The project is deliberately built around a few strict rules:

- **Equations before abstractions:** robotics mathematics remains visible.
- **Small public APIs:** modules can be tested independently.
- **Independent numerical checks:** analytical Jacobians are compared against finite differences; dynamic matrices are checked for symmetry and positive definiteness; simulation behavior is tested end-to-end.
- **Numerical robustness:** singularities and inverse problems use explicit tolerances and regularization.
- **Physical safety boundaries:** joint, velocity, effort, and command slew limits are explicit rather than implicit.
- **Fail early:** invalid physical parameters and invalid time steps raise explicit errors.
- **No hidden visualization dependency:** numerical code remains usable without Matplotlib.
- **Reproducible experiments:** fixed-step simulation and checked-in examples make experiments repeatable.
- **Automated gates:** CI runs linting, tests, coverage reporting, and compilation across Python 3.10–3.12.

## Research-informed scope

The capability map deliberately follows the major topics found in modern robotics curricula—rigid-body motion, Jacobians, singularities, inverse kinematics, dynamics, trajectories, planning, and control—while taking architectural cues from established tools such as Drake and ros2_control.

Further reading:

- Modern Robotics: https://modernrobotics.northwestern.edu/nu-gm-book-resource/
- Drake: https://drake.mit.edu/
- ros2_control: https://control.ros.org/

See [research-roadmap.md](docs/research-roadmap.md) for the planned expansion into planning, optimization, state estimation, ROS 2, hardware-in-the-loop, and advanced control.

## Roadmap

### Phase 1 — classical robotics core

- [x] Forward/inverse kinematics
- [x] Jacobian and singularities
- [x] Damped-least-squares IK
- [x] Cubic / quintic trajectory generation
- [x] PID / PD control
- [x] 2R rigid-body dynamics
- [x] Computed-torque control
- [x] Deterministic simulation
- [x] Energy diagnostics
- [x] Safety limits
- [x] Visualization

### Phase 2 — deeper control and estimation

- [ ] Cartesian PD / operational-space control
- [ ] Feedforward trajectory acceleration from the planner
- [ ] Runge-Kutta integrator comparison
- [ ] Disturbance injection and robustness metrics
- [ ] Noise models and state estimation
- [ ] Control-effort / overshoot / settling-time benchmarking

### Phase 3 — planning and optimization

- [ ] Configuration-space obstacle models
- [ ] Graph search and sampling-based planning
- [ ] Collision checking
- [ ] Trajectory optimization
- [ ] Quadratic-programming safety filters
- [ ] Null-space objectives

### Phase 4 — robotics systems engineering

- [ ] 3D serial manipulator model
- [ ] URDF generation
- [ ] ROS 2 interface layer
- [ ] ros2_control integration boundary
- [ ] Hardware-in-the-loop adapters
- [ ] Encoder / actuator abstractions
- [ ] System identification experiments

### Phase 5 — advanced research track

- [ ] Operational-space dynamics
- [ ] Model predictive control experiments
- [ ] Adaptive / robust control
- [ ] Reinforcement-learning benchmark environment
- [ ] Sim-to-real validation methodology
- [ ] Comparative benchmarks against established robotics libraries

## Scope

This is an educational and research-oriented laboratory. The simulator is intentionally transparent and lightweight; it is **not** a safety-certified controller and is not a replacement for a validated industrial robotics stack.
