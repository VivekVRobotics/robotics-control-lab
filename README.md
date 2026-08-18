# Robotics Control Lab

A from-scratch robotics engineering laboratory focused on **kinematics, differential kinematics, dynamics, feedback control, trajectory generation, simulation, and visualization**.

The goal is not to hide robotics behind a framework. The equations are implemented explicitly, tested numerically, and connected into reproducible experiments.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/VivekVRobotics/robotics-control-lab/ci.yml?branch=main&label=CI)
![Tests](https://img.shields.io/badge/tests-15%2B-success)

## Why this repository exists

A strong robotics portfolio should demonstrate more than a finished robot video. It should show that the engineer understands the stack underneath the robot:

```text
geometry
   ↓
kinematics
   ↓
differential kinematics
   ↓
dynamics
   ↓
control
   ↓
simulation
   ↓
measurement + visualization
```

This repository is built around that progression.

## Current capabilities

| Area | Implementation |
|---|---|
| Forward kinematics | Planar 2R arm |
| Inverse kinematics | Elbow-up / elbow-down |
| Workspace validation | Reachability checks |
| Jacobian | Analytical 2×2 position Jacobian |
| Manipulability | Yoshikawa planar measure |
| Singularity detection | Numerical tolerance check |
| Trajectory planning | Cubic interpolation |
| PID control | Saturation + integral bounds |
| Joint PD | Position/velocity feedback |
| Computed torque | Inverse-dynamics compensation |
| Dynamics | `M(q)`, `C(q,qd)`, `G(q)` |
| Simulation | Fixed-step semi-implicit Euler |
| Visualization | Optional Matplotlib plots |
| Verification | Numerical unit tests + CI |

## Repository structure

```text
robotics-control-lab/
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   └── robotics-foundations.md
├── examples/
│   └── track_step_response.py
├── src/robotics_control_lab/
│   ├── __init__.py
│   ├── controllers.py
│   ├── dynamics.py
│   ├── jacobian.py
│   ├── pid.py
│   ├── planar_arm.py
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
```

## Run the flagship experiment

The repository includes an end-to-end computed-torque simulation of a planar 2R robot:

```bash
python examples/track_step_response.py
```

The experiment:

1. builds a physical 2R robot model,
2. generates a smooth joint reference,
3. closes the loop with computed-torque control,
4. simulates the rigid-body dynamics,
5. reports final tracking error,
6. writes joint-tracking and workspace plots.

Generated plots are intentionally ignored by Git.

## Core API

```python
import numpy as np

from robotics_control_lab import (
    Planar2R,
    TwoRParameters,
    TwoRRobot,
    ComputedTorqueController,
    simulate,
)

arm = Planar2R(0.6, 0.4)
robot = TwoRRobot(
    arm,
    TwoRParameters(
        m1=2.0,
        m2=1.0,
        lc1=0.3,
        lc2=0.2,
        i1=0.06,
        i2=0.02,
    ),
)

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
- **Numerical verification:** analytical Jacobians are compared against finite differences; dynamic matrices are checked for symmetry/positive definiteness; simulation behavior is tested end-to-end.
- **Fail early:** invalid physical parameters and invalid time steps raise explicit errors.
- **No hidden runtime dependency on visualization:** numerical code remains usable without Matplotlib.
- **Reproducible experiments:** fixed-step simulation and checked-in examples make experiments repeatable.
- **Automated gates:** CI runs across Python 3.10–3.12 and compiles source/tests/examples.

## Documentation

- [Architecture](docs/architecture.md)
- [Robotics foundations](docs/robotics-foundations.md)

## Roadmap

### Phase 1 — classical robotics core

- [x] Forward/inverse kinematics
- [x] Jacobian and singularities
- [x] Trajectory generation
- [x] PID / PD control
- [x] 2R rigid-body dynamics
- [x] Computed-torque control
- [x] Deterministic simulation
- [x] Visualization

### Phase 2 — deeper control and estimation

- [ ] Jacobian inverse / damped-least-squares velocity control
- [ ] Cartesian PD control
- [ ] Feedforward trajectory acceleration from the planner
- [ ] Runge-Kutta integrator comparison
- [ ] Disturbance injection and robustness metrics
- [ ] Noise models and state estimation
- [ ] Control-effort / overshoot / settling-time benchmarking

### Phase 3 — robotics systems engineering

- [ ] 3D serial manipulator model
- [ ] URDF generation
- [ ] ROS 2 interface layer
- [ ] Hardware-in-the-loop adapters
- [ ] Encoder / actuator abstractions
- [ ] System identification experiments
- [ ] Safety limits and fault handling

### Phase 4 — advanced research track

- [ ] Operational-space dynamics
- [ ] Model predictive control experiments
- [ ] Adaptive / robust control
- [ ] Reinforcement-learning benchmark environment
- [ ] Sim-to-real validation methodology
- [ ] Comparative study against established robotics libraries

## Scope

This is an educational and research-oriented laboratory. The simulator is intentionally transparent and lightweight; it is **not** a safety-certified controller or a replacement for a validated industrial robotics stack.
