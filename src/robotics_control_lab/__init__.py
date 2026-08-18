"""Robotics Control Lab public API."""

from .controllers import ComputedTorqueController, JointPD
from .dynamics import TwoRParameters, TwoRRobot
from .jacobian import is_singular, jacobian, manipulability
from .pid import PIDController
from .planar_arm import Planar2R
from .simulation import SimulationResult, simulate
from .trajectory import cubic_interpolation

__all__ = [
    "ComputedTorqueController",
    "JointPD",
    "TwoRParameters",
    "TwoRRobot",
    "is_singular",
    "jacobian",
    "manipulability",
    "PIDController",
    "Planar2R",
    "SimulationResult",
    "simulate",
    "cubic_interpolation",
]
