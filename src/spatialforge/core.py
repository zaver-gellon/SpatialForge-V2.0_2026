"""ForgeCore — the runtime that binds sensing, mapping, tracking, and SpatialMind."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from spatialforge.exploration.controller import ExplorationController
from spatialforge.localization.factory import create_localizer
from spatialforge.mapping.tsdf import TSDFVolume
from spatialforge.sensors.factory import create_sensor
from spatialforge.spatial.mind import SpatialMind
from spatialforge.types import Pose, RGBDFrame
from spatialforge.utils.config import ForgeConfig
from spatialforge.utils.geometry import identity_pose
from spatialforge.utils.logging import get_logger
from spatialforge.visualization.dashboard import OccupancyDashboard
from spatialforge.visualization.export import export_point_cloud

log = get_logger("core")


@dataclass
class StepResult:
    frame: RGBDFrame
    pose: Pose
    integrated_voxels: int
    dt_ms: float


class ForgeCore:
    """One object an application (or an AI brain host) holds onto.

    Typical loop::

        core = ForgeCore(cfg)
        core.start()
        while (step := core.step()):
            report = core.mind.free_space_ahead(step.pose)
        core.stop()
    """

    def __init__(self, config: ForgeConfig) -> None:
        self.config = config
        self.sensor = create_sensor(config.sensor)
        self.volume = TSDFVolume(config.volume)
        self.localizer = create_localizer(config.localization)
        self.mind = SpatialMind(self.volume, config.spatial)
        self.explorer = ExplorationController(config.exploration) if config.exploration.enabled else None
        self.dashboard: OccupancyDashboard | None = None
        self.last_pose: Pose = identity_pose()
        self.frames_processed = 0
        self.output_dir = Path(config.output_dir)

    def start(self) -> None:
        self.sensor.start()
        if self.config.visualization.enabled and self.config.visualization.backend == "dashboard":
            self.dashboard = OccupancyDashboard(
                self.config.visualization.dashboard_host,
                self.config.visualization.dashboard_port,
            )
            self.dashboard.start()
        log.info(
            "ForgeCore online  device=%s  sensor=%s  localizer=%s",
            self.config.device,
            self.config.sensor.type,
            self.config.localization.method,
        )

    def step(self) -> StepResult | None:
        t0 = perf_counter()
        frame = self.sensor.capture()
        if frame is None:
            return None

        pose = self.localizer.track(frame, self.volume)
        updated = self.volume.integrate(frame, pose)
        self.last_pose = pose
        self.frames_processed += 1

        if self.explorer is not None:
            commanded = self.explorer.next_pose(pose, self.mind)
            self.sensor.set_pose(commanded)

        if (
            self.dashboard is not None
            and self.frames_processed % max(self.config.visualization.update_every, 1) == 0
        ):
            points, _, _ = self.volume.extract_surface()
            stats = self.volume.stats()
            stats["points"] = points.shape[0]
            self.dashboard.update(self.mind.occupancy_xy(), pose, self.volume.origin, self.volume.voxel_size, stats)

        dt_ms = (perf_counter() - t0) * 1000.0
        if self.frames_processed == 1 or self.frames_processed % 10 == 0:
            stats = self.volume.stats()
            log.info(
                "frame %03d  integrate=%5d voxels  dt=%.1fms  observed=%.1f%%",
                frame.frame_id,
                updated,
                dt_ms,
                100.0 * stats["observed_fraction"],
            )
        return StepResult(frame=frame, pose=pose, integrated_voxels=updated, dt_ms=dt_ms)

    def run(self, max_frames: int | None = None) -> None:
        limit = self.config.max_frames if max_frames is None else max_frames
        processed = 0
        try:
            while processed < limit:
                result = self.step()
                if result is None:
                    log.info("Sensor stream ended after %s frames", processed)
                    break
                processed += 1
        finally:
            self.stop()

    def export_map(self, directory: str | Path | None = None) -> Path:
        out = Path(directory) if directory else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        points, colors, _ = self.volume.extract_surface()
        ply = out / "map.ply"
        export_point_cloud(ply, points, colors)
        np.savez_compressed(
            out / "volume.npz",
            sdf=self.volume.sdf,
            weight=self.volume.weight,
            color=self.volume.color,
            origin=self.volume.origin,
            voxel_size=np.array([self.volume.voxel_size]),
        )
        if self.localizer.trajectory:
            np.save(out / "trajectory.npy", np.stack(self.localizer.trajectory))
        log.info("Wrote map (%s points) to %s", points.shape[0], ply)
        return ply

    def stop(self) -> None:
        self.sensor.stop()
        if self.dashboard is not None:
            self.dashboard.stop()
            self.dashboard = None
        log.info("ForgeCore stopped after %s frames", self.frames_processed)
