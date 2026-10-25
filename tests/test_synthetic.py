from spatialforge.sensors.synthetic import SyntheticSensor
from spatialforge.utils.config import SensorConfig
from spatialforge.utils.geometry import look_at
import numpy as np


def test_synthetic_renders_depth_and_color():
    sensor = SyntheticSensor(SensorConfig(width=80, height=60, fx=60, fy=60, cx=40, cy=30))
    sensor.start()
    frame = sensor.capture()
    assert frame is not None
    assert frame.color.shape == (60, 80, 3)
    assert frame.depth.shape == (60, 80)
    assert np.nanmax(frame.depth) > 0.5
    assert frame.pose_world_camera is not None


def test_look_at_optical_z_points_at_target():
    eye = np.array([2.0, 0.0, 1.0])
    target = np.array([0.0, 0.0, 0.5])
    pose = look_at(eye, target)
    forward = pose[:3, 2]
    expected = target - eye
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(forward, expected, atol=1e-6)
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)
