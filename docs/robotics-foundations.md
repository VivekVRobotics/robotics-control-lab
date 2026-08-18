# Robotics Foundations

## Coordinate conventions

The planar arm lives in the XY plane. Joint angles are measured counter-clockwise from the positive X axis.

## Forward kinematics

For link lengths `l1`, `l2` and joint angles `q1`, `q2`:

```text
x = l1 cos(q1) + l2 cos(q1 + q2)
y = l1 sin(q1) + l2 sin(q1 + q2)
```

## Inverse kinematics

The second joint satisfies:

```text
cos(q2) = (x^2 + y^2 - l1^2 - l2^2) / (2 l1 l2)
```

The implementation exposes both elbow configurations and rejects targets outside the annular workspace.

## Differential kinematics

The position Jacobian maps joint velocity into end-effector velocity:

```text
v = J(q) qdot
```

For the 2R arm:

```text
J = [ -l1 sin(q1) - l2 sin(q1+q2),  -l2 sin(q1+q2) ]
    [  l1 cos(q1) + l2 cos(q1+q2),   l2 cos(q1+q2) ]
```

The determinant becomes zero at the fully extended/folded configurations, which is why the lab exposes explicit singularity checks.

## Dynamics

The rigid-body model follows the standard manipulator equation:

```text
M(q) qdd + C(q, qd) + G(q) = tau
```

where:

- `M(q)` is the joint-space inertia matrix.
- `C(q, qd)` contains Coriolis and centrifugal terms.
- `G(q)` is the gravity load.
- `tau` is the applied joint torque.

## Computed torque control

The model-based controller forms a reference acceleration:

```text
v = qdd_des + Kp(q_des - q) + Kd(qdot_des - qdot)
```

and applies:

```text
tau = M(q) v + C(q, qdot) + G(q)
```

This is useful because it demonstrates the bridge from robot dynamics to feedback linearization instead of treating the robot as a generic black-box plant.

## Simulation assumptions

The default simulator uses fixed-step semi-implicit Euler integration. It is intentionally simple and transparent; it is not presented as a high-fidelity physics engine.

For higher-fidelity work, the intended future direction is comparing this integrator against fourth-order Runge-Kutta and external physics engines, with numerical-error tests and energy diagnostics.
