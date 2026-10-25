"""Use sensor-provided poses (simulation, motion-capture, or bag metadata)."""

from __future__ import annotations

from spatialforge.localization.base import Localizer
from spatialforge.mapping.tsdf import TSDFVolume
from spatialforge.types import Pose, RGBDFrame
class GroundTruthLocalizer(Localizer):
    def track(self, frame: RGBDFrame, volume: TSDFVolume) -> Pose:
        if frame.pose_world_camera is None:
            raise RuntimeError("Ground-truth localization requires pose_world_camera on the frame")
        self.pose = frame.pose_world_camera.astype(float)
        self.initialized = True
        self.trajectory.append(self.pose.copy())
        return self.pose
