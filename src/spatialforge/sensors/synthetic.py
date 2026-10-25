"""Procedural RGB-D room for hardware-free development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatialforge.sensors.base import SensorFrontend
from spatialforge.types import CameraIntrinsics, RGBDFrame
from spatialforge.utils.config import SensorConfig
from spatialforge.utils.geometry import look_at


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in world meters with a solid RGB color."""

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    color: tuple[int, int, int]


def default_room() -> list[Box]:
    """A compact indoor scene: floor, walls, a table, and clutter."""
    return [
        Box((-2.5, -2.5, -0.05), (2.5, 2.5, 0.0), (90, 90, 95)),  # floor
        Box((-2.5, -2.5, 0.0), (-2.42, 2.5, 2.2), (150, 140, 130)),  # walls
        Box((2.42, -2.5, 0.0), (2.5, 2.5, 2.2), (150, 140, 130)),
        Box((-2.5, -2.5, 0.0), (2.5, -2.42, 2.2), (140, 150, 145)),
        Box((-2.5, 2.42, 0.0), (2.5, 2.5, 2.2), (140, 150, 145)),
        Box((-1.2, -0.4, 0.0), (-0.4, 0.4, 0.75), (120, 80, 40)),  # table
        Box((0.8, 0.8, 0.0), (1.15, 1.15, 1.4), (60, 90, 160)),  # pillar
        Box((0.2, -1.4, 0.0), (0.7, -0.9, 0.45), (180, 70, 60)),  # crate
        Box((-0.2, 1.1, 0.0), (0.3, 1.5, 0.35), (70, 160, 90)),  # crate
    ]


def _ray_aabb(
    origin: NDArray,
    direction: NDArray,
    bmin: NDArray,
    bmax: NDArray,
) -> NDArray:
    """Vectorized ray/AABB intersection. Returns t_hit or inf."""
    inv = 1.0 / np.where(np.abs(direction) < 1e-9, 1e-9, direction)
    t0 = (bmin - origin) * inv
    t1 = (bmax - origin) * inv
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    t_enter = np.max(tmin, axis=1)
    t_exit = np.min(tmax, axis=1)
    hit = (t_exit >= t_enter) & (t_exit > 0)
    t = np.where(t_enter > 0, t_enter, t_exit)
    return np.where(hit, t, np.inf)


class SyntheticSensor(SensorFrontend):
    """Renders depth and color by raycasting a box world along an orbit path."""

    def __init__(self, config: SensorConfig, boxes: list[Box] | None = None) -> None:
        super().__init__(config)
        self.boxes = boxes or default_room()
        self._frame_id = 0
        self._forced_pose: np.ndarray | None = None
        self.intrinsics = CameraIntrinsics(
            fx=config.fx,
            fy=config.fy,
            cx=config.cx if config.cx else config.width / 2,
            cy=config.cy if config.cy else config.height / 2,
            width=config.width,
            height=config.height,
        )
        self._bmin = np.array([b.min_xyz for b in self.boxes], dtype=np.float64)
        self._bmax = np.array([b.max_xyz for b in self.boxes], dtype=np.float64)
        self._colors = np.array([b.color for b in self.boxes], dtype=np.uint8)

    def start(self) -> None:
        self._frame_id = 0

    def set_pose(self, pose_world_camera) -> None:  # noqa: ANN001
        self._forced_pose = np.asarray(pose_world_camera, dtype=np.float64)

    def default_pose(self, frame_id: int) -> np.ndarray:
        """Slow orbit looking toward the room center, with a gentle bob."""
        t = frame_id / max(self.config.fps, 1.0)
        angle = 0.35 + 0.18 * t
        radius = 1.85
        eye = np.array(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
                1.05 + 0.08 * np.sin(0.7 * t),
            ]
        )
        return look_at(eye, np.array([0.0, 0.0, 0.55]))

    def capture(self) -> RGBDFrame | None:
        pose = self._forced_pose if self._forced_pose is not None else self.default_pose(self._frame_id)
        color, depth = self.render(pose)
        frame = RGBDFrame(
            timestamp=self._frame_id / max(self.config.fps, 1.0),
            color=color,
            depth=depth,
            intrinsics=self.intrinsics,
            pose_world_camera=pose,
            frame_id=self._frame_id,
        )
        self._frame_id += 1
        return frame

    def render(self, pose_world_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, h = self.intrinsics.width, self.intrinsics.height
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        x = (us - self.intrinsics.cx) / self.intrinsics.fx
        y = (vs - self.intrinsics.cy) / self.intrinsics.fy
        dirs_cam = np.stack([x, y, np.ones_like(x)], axis=-1)
        dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

        r = pose_world_camera[:3, :3]
        origin = pose_world_camera[:3, 3]
        dirs_world = dirs_cam.reshape(-1, 3) @ r.T

        best_t = np.full(dirs_world.shape[0], np.inf)
        best_idx = np.full(dirs_world.shape[0], -1, dtype=np.int32)
        for i in range(len(self.boxes)):
            t_hit = _ray_aabb(origin, dirs_world, self._bmin[i], self._bmax[i])
            closer = t_hit < best_t
            best_t[closer] = t_hit[closer]
            best_idx[closer] = i

        depth_cam = np.zeros(dirs_world.shape[0], dtype=np.float32)
        hit = best_idx >= 0
        # Convert Euclidean ray length to camera-Z (optical axis).
        optical_z = dirs_cam.reshape(-1, 3)[:, 2]
        depth_cam[hit] = (best_t[hit] * optical_z[hit]).astype(np.float32)

        color = np.zeros((dirs_world.shape[0], 3), dtype=np.uint8)
        color[hit] = self._colors[best_idx[hit]]
        # Mild distance falloff so the room is not a flat cartoon.
        atten = np.clip(1.4 / (1.0 + depth_cam * 0.25), 0.35, 1.0)
        color = (color.astype(np.float32) * atten[:, None]).astype(np.uint8)

        return color.reshape(h, w, 3), depth_cam.reshape(h, w)
