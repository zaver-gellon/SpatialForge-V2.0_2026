"""Frame-to-model ICP against the TSDF surface."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from spatialforge.localization.base import Localizer, downsample_points
from spatialforge.mapping.tsdf import TSDFVolume
from spatialforge.types import Pose, RGBDFrame
from spatialforge.utils.geometry import pose_from_rt, transform_points, unproject_depth
from spatialforge.utils.logging import get_logger

log = get_logger("localization.icp")


def _best_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    h = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = dst_mean - r @ src_mean
    return r, t


class ICPLocalizer(Localizer):
    """Point-to-point ICP of the live cloud against the current TSDF surface."""

    def track(self, frame: RGBDFrame, volume: TSDFVolume) -> Pose:
        live, _ = unproject_depth(frame.depth, frame.color, frame.intrinsics, stride=4)
        live = downsample_points(live, self.config.voxel_downsample)
        if not self.initialized:
            if frame.pose_world_camera is not None:
                self.pose = frame.pose_world_camera.astype(float)
            self.initialized = True
            self.trajectory.append(self.pose.copy())
            return self.pose

        model, _, _ = volume.extract_surface(min_weight=1.0)
        model = downsample_points(model, self.config.voxel_downsample)
        if live.shape[0] < 30 or model.shape[0] < 30:
            log.debug("ICP skipped (insufficient points live=%s model=%s)", live.shape[0], model.shape[0])
            self.trajectory.append(self.pose.copy())
            return self.pose

        pose = self.pose.copy()
        max_corr = self.config.icp_max_correspondence
        tree = cKDTree(model)
        for _ in range(self.config.icp_iterations):
            src_world = transform_points(live, pose)
            dist, nn = tree.query(src_world, k=1)
            mask = dist < max_corr
            if int(mask.sum()) < 20:
                break
            r, t = _best_rigid(src_world[mask], model[nn[mask]])
            delta = pose_from_rt(r, t)
            pose = delta @ pose
            max_corr *= 0.92

        self.pose = pose
        self.trajectory.append(self.pose.copy())
        return self.pose
