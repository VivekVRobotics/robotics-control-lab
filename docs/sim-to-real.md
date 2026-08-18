# Simulation-to-real methodology

Simulation is a development tool, not evidence of physical validity. A credible sim-to-real experiment keeps the controller and benchmark definition fixed while changing only the execution environment and measured parameters.

## 1. Calibrate the model

Record link masses, center-of-mass locations, inertias, actuator limits, joint offsets, encoder resolution, and communication/update rates.

## 2. Randomize the uncertain parameters

Run batches over plausible mass, inertia, friction, sensor noise, latency, torque saturation, and timing jitter ranges. Store the sampled parameter set with each result.

## 3. Freeze the benchmark

Use identical reference trajectories, initial conditions where meaningful, controller gains, sampling assumptions, and success criteria.

## 4. Measure the gap

Report simulation and hardware separately for tracking RMSE, maximum error, settling time, peak/RMS torque, failure rate, and minimum safety margin.

## 5. Diagnose rather than retune blindly

Separate model error, sensing error, actuator error, timing/transport delay, unmodeled friction, and contact/environment mismatch.

## 6. HIL before hardware

Use the `HardwareInterface` boundary and `SimulatedHardware` backend to exercise controller I/O contracts before connecting a physical actuator. HIL runs should report cycle time, worst-case jitter, command saturation, dropped cycles, and state timestamp age.

## Safety boundary

A successful simulation, HIL test, or numerical benchmark does not establish a safety case for physical robotics hardware. Hardware validation requires appropriate limits, emergency-stop behavior, independent safety mechanisms, and system-specific verification.
