# Robotics Control Lab

A small, testable Python lab for learning and demonstrating core robotics-control primitives.

## What is included

- **PID control** — discrete PID with integral clamping and output saturation.
- **2R planar-arm kinematics** — forward kinematics plus elbow-up/down inverse kinematics with workspace validation.
- **Trajectory generation** — cubic joint interpolation with zero endpoint velocity.
- **Automated tests** — kinematics round trips, controller limits, invalid inputs, and trajectory boundary conditions.
- **GitHub Actions CI** — tests run automatically on pushes and pull requests.

## Project structure

```text
robotics-control-lab/
├── src/robotics_control_lab/
│   ├── pid.py
│   ├── planar_arm.py
│   └── trajectory.py
├── tests/
│   └── test_control.py
├── .github/workflows/ci.yml
└── pyproject.toml
```

## Run locally

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip pytest
pytest -q
```

The package intentionally keeps the first layer dependency-light so the mathematics can be inspected directly. A natural next layer is adding dynamics, Jacobians, plotting, and hardware interfaces around these tested primitives.

## Engineering focus

This repository is designed as a portfolio project rather than a collection of disconnected scripts: each control primitive has an explicit interface, input validation, and regression tests so future experiments can build on a stable foundation.
