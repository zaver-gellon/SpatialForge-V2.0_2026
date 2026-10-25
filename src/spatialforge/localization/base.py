"""Camera tracking interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from spatialforge.mapping.tsdf import TSDFVolume
from spatialforge.types import Pose, RGBDFrame
from spatialforge.utils.config import LocalizationConfig
from spatialforge.utils.geometry import identity_pose


class Localizer(ABC):
    def __init__(self, config: LocalizationConfig) -> None:
        self.config = config
        self.pose: Pose = identity_pose()
        self.initialized = False
        self.trajectory: list[Pose] = []

    @abstractmethod
    def track(self, frame: RGBDFrame, volume: TSDFVolume) -> Pose:
        """Estimate T_world_camera for the given frame."""

    def reset(self) -> None:
        self.pose = identity_pose()
        self.initialized = False
        self.trajectory = []


def downsample_points(points: np.ndarray, voxel: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int32)
    _, unique = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique)]
