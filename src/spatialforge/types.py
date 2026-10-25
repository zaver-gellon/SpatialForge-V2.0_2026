"""Shared geometric and sensor types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

Float32 = NDArray[np.float32]
UInt8 = NDArray[np.uint8]
Pose = NDArray[np.float64]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera model in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def matrix(self) -> Float32:
        k = np.eye(3, dtype=np.float32)
        k[0, 0] = self.fx
        k[1, 1] = self.fy
        k[0, 2] = self.cx
        k[1, 2] = self.cy
        return k

    def backproject(self, u: NDArray, v: NDArray, z: NDArray) -> Float32:
        """Unproject pixel coordinates with depth Z into the camera frame."""
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=-1).astype(np.float32)


@dataclass
class RGBDFrame:
    """One synchronized RGB-D observation."""

    timestamp: float
    color: UInt8
    depth: Float32
    intrinsics: CameraIntrinsics
    pose_world_camera: Optional[Pose] = None
    frame_id: int = 0


@dataclass
class VolumeBounds:
    """Axis-aligned world volume in meters (z-up)."""

    min_xyz: tuple[float, float, float] = (-2.5, -2.5, 0.0)
    max_xyz: tuple[float, float, float] = (2.5, 2.5, 2.4)

    def size(self) -> np.ndarray:
        return np.array(self.max_xyz, dtype=np.float64) - np.array(self.min_xyz, dtype=np.float64)


@dataclass
class SurfaceHit:
    point: np.ndarray
    normal: np.ndarray
    distance: float
    occupied: bool


@dataclass
class FreeSpaceReport:
    """Free space immediately in front of the camera."""

    min_clearance_m: float
    mean_clearance_m: float
    blocked_fraction: float
    lookahead_m: float
    ray_hits: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class PathQuery:
    waypoints: list[np.ndarray]
    length_m: float
    reachable: bool
    reason: str = ""
