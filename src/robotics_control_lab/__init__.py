"""Core primitives for robotics control experiments."""

from .pid import PIDController
from .planar_arm import Planar2R
from .trajectory import cubic_interpolation

__all__ = ["PIDController", "Planar2R", "cubic_interpolation"]
