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
    """Minimal monotonic read/write contract for simulated or physical backends."""

    def read(self, timestamp: float) -> JointState:
        raise NotImplementedError

    def write(self, command: np.ndarray, timestamp: float) -> None:
        raise NotImplementedError


class SimulatedHardware(HardwareInterface):
    """Stateful backend for deterministic controller/HIL tests.

    This backend stores the commanded effort but intentionally does not model
    plant dynamics; that remains the simulator's responsibility.
    """

    def __init__(self, joints: int = 2) -> None:
        if joints <= 0:
            raise ValueError("joints must be positive")
        self.position = np.zeros(joints)
        self.velocity = np.zeros(joints)
        self.effort = np.zeros(joints)
        self.last_command = np.zeros(joints)
        self._last_timestamp: float | None = None

    def _validate_timestamp(self, timestamp: float) -> float:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("hardware timestamps must be monotonic")
        self._last_timestamp = timestamp
        return timestamp

    def read(self, timestamp: float) -> JointState:
        timestamp = self._validate_timestamp(timestamp)
        return JointState(self.position.copy(), self.velocity.copy(), self.effort.copy(), timestamp)

    def write(self, command: np.ndarray, timestamp: float) -> None:
        timestamp = self._validate_timestamp(timestamp)
        command = np.asarray(command, dtype=float)
        if command.shape != self.position.shape:
            raise ValueError("command dimension mismatch")
        if not np.all(np.isfinite(command)):
            raise ValueError("command must contain finite values")
        self.last_command = command.copy()
        self.effort = command.copy()


def run_hil_cycle(hardware: HardwareInterface, command: np.ndarray, timestamp: float) -> JointState:
    """Write a command and immediately return the measured state at that cycle timestamp."""
    hardware.write(command, timestamp)
    return hardware.read(timestamp)
