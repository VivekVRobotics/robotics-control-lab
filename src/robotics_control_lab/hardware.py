"""Hardware abstraction and HIL interfaces independent of transport libraries."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class JointState:
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    timestamp: float


class HardwareInterface:
    """Minimal read/write contract compatible with simulated or physical backends."""

    def read(self, timestamp: float) -> JointState:
        raise NotImplementedError

    def write(self, command: np.ndarray, timestamp: float) -> None:
        raise NotImplementedError


class SimulatedHardware(HardwareInterface):
    """Stateful HIL adapter for testing control code without real actuators."""

    def __init__(self, joints: int = 2) -> None:
        if joints <= 0:
            raise ValueError("joints must be positive")
        self.position = np.zeros(joints)
        self.velocity = np.zeros(joints)
        self.effort = np.zeros(joints)
        self.last_command = np.zeros(joints)

    def read(self, timestamp: float) -> JointState:
        return JointState(self.position.copy(), self.velocity.copy(), self.effort.copy(), float(timestamp))

    def write(self, command: np.ndarray, timestamp: float) -> None:
        command = np.asarray(command, dtype=float)
        if command.shape != self.position.shape:
            raise ValueError("command dimension mismatch")
        self.last_command = command.copy()
        self.effort = command.copy()


def run_hil_cycle(hardware: HardwareInterface, command: np.ndarray, timestamp: float) -> JointState:
    """Write a command and immediately return the post-write measured state."""
    hardware.write(command, timestamp)
    return hardware.read(timestamp)
