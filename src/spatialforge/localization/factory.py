from spatialforge.localization.base import Localizer
from spatialforge.localization.ground_truth import GroundTruthLocalizer
from spatialforge.localization.icp import ICPLocalizer
from spatialforge.localization.rgbd_odometry import RGBDOdometryLocalizer
from spatialforge.utils.config import LocalizationConfig


def create_localizer(config: LocalizationConfig) -> Localizer:
    method = config.method.lower()
    if method in {"ground_truth", "gt", "pose"}:
        return GroundTruthLocalizer(config)
    if method in {"icp", "frame_to_model"}:
        return ICPLocalizer(config)
    if method in {"vo", "rgbd", "odometry"}:
        return RGBDOdometryLocalizer(config)
    raise ValueError(f"Unknown localization method: {config.method}")
