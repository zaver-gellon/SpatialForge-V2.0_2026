"""Intel RealSense live frontend (D435i / D455)."""

from __future__ import annotations

import numpy as np

from spatialforge.sensors.base import SensorFrontend
from spatialforge.types import CameraIntrinsics, RGBDFrame
from spatialforge.utils.config import SensorConfig
from spatialforge.utils.logging import get_logger

log = get_logger("sensors.realsense")


class RealSenseSensor(SensorFrontend):
    """Live aligned RGB-D stream via pyrealsense2."""

    def __init__(self, config: SensorConfig) -> None:
        super().__init__(config)
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001
        self.intrinsics = CameraIntrinsics(
            fx=config.fx,
            fy=config.fy,
            cx=config.cx,
            cy=config.cy,
            width=config.width,
            height=config.height,
        )
        self._frame_id = 0

    def start(self) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is not installed. On Jetson/Pi: pip install 'spatialforge[realsense]'"
            ) from exc

        pipeline = rs.pipeline()
        cfg = rs.config()
        if self.config.serial:
            cfg.enable_device(self.config.serial)
        cfg.enable_stream(
            rs.stream.depth, self.config.width, self.config.height, rs.format.z16, int(self.config.fps)
        )
        cfg.enable_stream(
            rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, int(self.config.fps)
        )
        profile = pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.intrinsics = CameraIntrinsics(
            fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy, width=intr.width, height=intr.height
        )
        self._align = rs.align(rs.stream.color) if self.config.align_to_color else None
        self._pipeline = pipeline
        self._rs = rs
        self._frame_id = 0
        log.info("RealSense started (%sx%s @ %s fps)", intr.width, intr.height, self.config.fps)

    def capture(self) -> RGBDFrame | None:
        if self._pipeline is None:
            raise RuntimeError("RealSenseSensor.start() was not called")
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return None
        color_np = np.asanyarray(color.get_data())[:, :, ::-1].copy()  # BGR -> RGB
        depth_np = np.asanyarray(depth.get_data()).astype(np.float32) * self._depth_scale
        frame = RGBDFrame(
            timestamp=frames.get_timestamp() / 1000.0,
            color=color_np,
            depth=depth_np,
            intrinsics=self.intrinsics,
            pose_world_camera=None,
            frame_id=self._frame_id,
        )
        self._frame_id += 1
        return frame

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            log.info("RealSense stopped")
