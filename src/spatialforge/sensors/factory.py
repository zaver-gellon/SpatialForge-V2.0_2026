"""Construct a sensor frontend from configuration."""

from __future__ import annotations

from spatialforge.sensors.base import SensorFrontend
from spatialforge.sensors.bag import BagSensor
from spatialforge.sensors.realsense import RealSenseSensor
from spatialforge.sensors.synthetic import SyntheticSensor
from spatialforge.utils.config import SensorConfig


def create_sensor(config: SensorConfig) -> SensorFrontend:
    kind = config.type.lower()
    if kind in {"synthetic", "simulation", "sim"}:
        return SyntheticSensor(config)
    if kind in {"bag", "replay", "npz"}:
        return BagSensor(config)
    if kind in {"realsense", "rs", "d435", "d435i", "d455"}:
        return RealSenseSensor(config)
    raise ValueError(f"Unknown sensor type: {config.type}")
