import numpy as np

from spatialforge.mapping.tsdf import TSDFVolume
from spatialforge.sensors.synthetic import SyntheticSensor
from spatialforge.utils.config import SensorConfig, VolumeConfig


def test_tsdf_integrates_synthetic_frame():
    volume = TSDFVolume(
        VolumeConfig(
            voxel_size=0.08,
            min_xyz=(-2.0, -2.0, 0.0),
            max_xyz=(2.0, 2.0, 2.0),
        )
    )
    sensor = SyntheticSensor(SensorConfig(width=80, height=60, fx=60, fy=60, cx=40, cy=30))
    sensor.start()
    frame = sensor.capture()
    assert frame is not None
    updated = volume.integrate(frame, frame.pose_world_camera)
    assert updated > 100
    points, colors, normals = volume.extract_surface(min_weight=0.5)
    assert points.shape[0] > 20
    assert colors.shape == points.shape
    assert normals.shape == points.shape
