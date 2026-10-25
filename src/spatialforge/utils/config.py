"""YAML configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from spatialforge.types import VolumeBounds


@dataclass
class SensorConfig:
    type: str = "synthetic"
    width: int = 320
    height: int = 240
    fps: float = 15.0
    fx: float = 240.0
    fy: float = 240.0
    cx: float = 160.0
    cy: float = 120.0
    bag_path: str | None = None
    serial: str | None = None
    align_to_color: bool = True


@dataclass
class VolumeConfig:
    voxel_size: float = 0.05
    truncation_voxels: float = 5.0
    max_weight: float = 16.0
    min_xyz: tuple[float, float, float] = (-2.5, -2.5, 0.0)
    max_xyz: tuple[float, float, float] = (2.5, 2.5, 2.4)

    def __post_init__(self) -> None:
        self.min_xyz = tuple(float(v) for v in self.min_xyz)  # type: ignore[assignment]
        self.max_xyz = tuple(float(v) for v in self.max_xyz)  # type: ignore[assignment]

    @property
    def bounds(self) -> VolumeBounds:
        return VolumeBounds(self.min_xyz, self.max_xyz)

    @property
    def truncation(self) -> float:
        return self.voxel_size * self.truncation_voxels


@dataclass
class LocalizationConfig:
    method: str = "ground_truth"
    icp_iterations: int = 12
    icp_max_correspondence: float = 0.12
    vo_ransac_iters: int = 64
    voxel_downsample: float = 0.06


@dataclass
class SpatialConfig:
    occupied_sdf: float = 0.04
    robot_height_min: float = 0.05
    robot_height_max: float = 1.2
    unknown_is_occupied: bool = True
    inflations_m: float = 0.08


@dataclass
class VisualizationConfig:
    enabled: bool = True
    backend: str = "dashboard"
    update_every: int = 5
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8765


@dataclass
class ExplorationConfig:
    enabled: bool = False
    max_steps: int = 80
    goal_radius: float = 0.25


@dataclass
class ForgeConfig:
    device: str = "simulation"
    log_level: str = "INFO"
    max_frames: int = 60
    output_dir: str = "outputs"
    sensor: SensorConfig = field(default_factory=SensorConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def _merge(dc_kwargs: dict[str, Any], data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return dc_kwargs
    for key, value in data.items():
        if key in dc_kwargs:
            dc_kwargs[key] = value
    return dc_kwargs


def load_config(path: str | Path) -> ForgeConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    sensor = SensorConfig(**_merge(SensorConfig().__dict__, data.get("sensor")))
    volume = VolumeConfig(**_merge(VolumeConfig().__dict__, data.get("volume")))
    localization = LocalizationConfig(
        **_merge(LocalizationConfig().__dict__, data.get("localization"))
    )
    spatial = SpatialConfig(**_merge(SpatialConfig().__dict__, data.get("spatial")))
    visualization = VisualizationConfig(
        **_merge(VisualizationConfig().__dict__, data.get("visualization"))
    )
    exploration = ExplorationConfig(
        **_merge(ExplorationConfig().__dict__, data.get("exploration"))
    )
    system = data.get("system", {})
    return ForgeConfig(
        device=system.get("device", "simulation"),
        log_level=system.get("log_level", "INFO"),
        max_frames=int(system.get("max_frames", 60)),
        output_dir=system.get("output_dir", "outputs"),
        sensor=sensor,
        volume=volume,
        localization=localization,
        spatial=spatial,
        visualization=visualization,
        exploration=exploration,
        raw=data,
    )
