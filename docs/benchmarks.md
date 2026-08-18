# Benchmark suite

The lab should compare controllers and planners using repeatable metrics rather than visual inspection alone.

## Control metrics

- RMS joint-position error
- maximum absolute tracking error
- overshoot
- settling time
- steady-state error
- integrated squared error
- integrated absolute error
- peak torque
- RMS torque
- control-energy proxy `sum(tau^2 * dt)`

## Planning metrics

- success rate
- path length
- planning time
- number of collision checks
- minimum obstacle clearance
- smoothed-path length

## Estimation metrics

- state RMSE
- covariance consistency
- innovation RMS
- rejection/false-alarm rate

## Simulation-to-real protocol

Use the same reference trajectory and metric definitions across simulation and hardware. Report parameter uncertainty, sensor sampling, actuator saturation, latency, noise, and contact/impact assumptions explicitly. Never interpret simulation performance as hardware validation without a separate experiment.
