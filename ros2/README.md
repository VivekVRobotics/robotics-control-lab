# ROS 2 / ros2_control integration boundary

This directory documents the integration boundary rather than forcing ROS 2 as a core Python dependency.

The reference `robots/planar_2r.urdf` defines the robot description. A future ROS 2 package should expose:

1. state interfaces: joint position / velocity / effort;
2. command interfaces: effort or position;
3. a controller node/plugin that owns the real-time update path;
4. lifecycle-aware startup/shutdown;
5. hardware adapters implementing `read()` and `write()` against `HardwareInterface`.

This mirrors the conceptual separation used by ros2_control: Controller Manager mediates controllers and hardware interfaces, while the hardware layer provides state and command interfaces. See the official ros2_control documentation for the runtime contract.
