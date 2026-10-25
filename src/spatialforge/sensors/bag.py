"""Replay recorded RGB-D streams (.npz or RealSense .bag)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spatialforge.sensors.base import SensorFrontend
from spatialforge.types import CameraIntrinsics, RGBDFrame
from spatialforge.utils.config import SensorConfig
from spatialforge.utils.logging import get_logger

log = get_logger("sensors.bag")


class BagSensor(SensorFrontend):
    """Replays SpatialForge `.npz` recordings, with optional RealSense `.bag` support."""

    def __init__(self, config: SensorConfig) -> None:
        super().__init__(config)
        if not config.bag_path:
            raise ValueError("sensor.bag_path is required for bag/replay mode")
        self.path = Path(config.bag_path)
        self._index = 0
        self._colors: np.ndarray | None = None
        self._depths: np.ndarray | None = None
        self._poses: np.ndarray | None = None
        self._timestamps: np.ndarray | None = None
        self.intrinsics = CameraIntrinsics(
            fx=config.fx,
            fy=config.fy,
            cx=config.cx,
            cy=config.cy,
            width=config.width,
            height=config.height,
        )

    def start(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Recording not found: {self.path}")
        if self.path.suffix == ".npz":
            self._load_npz()
        elif self.path.suffix == ".bag":
            self._load_realsense_bag()
        else:
            raise ValueError(f"Unsupported recording format: {self.path.suffix}")
        self._index = 0
        log.info("Loaded %s frames from %s", len(self._depths), self.path)

    def _load_npz(self) -> None:
        payload = np.load(self.path, allow_pickle=False)
        self._colors = payload["color"]
        self._depths = payload["depth"]
        self._timestamps = payload["timestamp"] if "timestamp" in payload else None
        self._poses = payload["pose"] if "pose" in payload else None
        if "fx" in payload:
            self.intrinsics = CameraIntrinsics(
                fx=float(payload["fx"]),
                fy=float(payload["fy"]),
                cx=float(payload["cx"]),
                cy=float(payload["cy"]),
                width=int(self._colors.shape[2]),
                height=int(self._colors.shape[1]),
            )

    def _load_realsense_bag(self) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is required to replay Intel .bag files. "
                "Install with: pip install 'spatialforge[realsense]'"
            ) from exc

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device_from_file(str(self.path), repeat_playback=False)
        cfg.enable_stream(rs.stream.depth)
        cfg.enable_stream(rs.stream.color)
        profile = pipeline.start(cfg)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        align = rs.align(rs.stream.color)

        colors, depths, timestamps = [], [], []
        first_intrinsics = None
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                break
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue
            if first_intrinsics is None:
                intr = color.profile.as_video_stream_profile().intrinsics
                first_intrinsics = CameraIntrinsics(
                    fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                    width=intr.width, height=intr.height,
                )
            colors.append(np.asanyarray(color.get_data()))
            depths.append(np.asanyarray(depth.get_data()).astype(np.float32) * depth.get_units())
            timestamps.append(frames.get_timestamp() / 1000.0)

        pipeline.stop()
        if not depths:
            raise RuntimeError(f"No frames decoded from {self.path}")
        self._colors = np.stack(colors)
        self._depths = np.stack(depths)
        self._timestamps = np.array(timestamps, dtype=np.float64)
        if first_intrinsics:
            self.intrinsics = first_intrinsics

    def capture(self) -> RGBDFrame | None:
        if self._depths is None or self._index >= len(self._depths):
            return None
        pose = None
        if self._poses is not None:
            pose = self._poses[self._index]
        ts = float(self._timestamps[self._index]) if self._timestamps is not None else self._index / 15.0
        frame = RGBDFrame(
            timestamp=ts,
            color=self._colors[self._index],
            depth=self._depths[self._index],
            intrinsics=self.intrinsics,
            pose_world_camera=pose,
            frame_id=self._index,
        )
        self._index += 1
        return frame


def write_npz(
    path: str | Path,
    colors: list[np.ndarray],
    depths: list[np.ndarray],
    poses: list[np.ndarray],
    timestamps: list[float],
    intrinsics: CameraIntrinsics,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        color=np.stack(colors),
        depth=np.stack(depths),
        pose=np.stack(poses),
        timestamp=np.array(timestamps, dtype=np.float64),
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
