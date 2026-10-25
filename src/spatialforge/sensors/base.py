"""Sensor frontend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from spatialforge.types import RGBDFrame
from spatialforge.utils.config import SensorConfig


class SensorFrontend(ABC):
    """Produces timestamped RGB-D frames in a camera-optical frame."""

    def __init__(self, config: SensorConfig) -> None:
        self.config = config

    @abstractmethod
    def start(self) -> None:
        """Open the device or allocate simulation state."""

    @abstractmethod
    def capture(self) -> RGBDFrame | None:
        """Return the next frame, or None when the stream is exhausted."""

    def stop(self) -> None:
        """Release hardware. Default is a no-op."""

    def set_pose(self, pose_world_camera) -> None:  # noqa: ANN001
        """Optional hook used by the exploration controller in simulation."""

    def __enter__(self) -> "SensorFrontend":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.stop()
