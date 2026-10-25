"""Lightweight RGB-D visual odometry via ORB matches and RANSAC rigid fit."""

from __future__ import annotations

import cv2
import numpy as np

from spatialforge.localization.base import Localizer
from spatialforge.mapping.tsdf import TSDFVolume
from spatialforge.types import Pose, RGBDFrame
from spatialforge.utils.geometry import pose_from_rt
from spatialforge.utils.logging import get_logger

log = get_logger("localization.vo")


def _rigid_ransac(src: np.ndarray, dst: np.ndarray, iters: int, thresh: float) -> np.ndarray | None:
    n = src.shape[0]
    if n < 6:
        return None
    best_inliers = 0
    best_pose = None
    rng = np.random.default_rng(0)
    for _ in range(iters):
        sample = rng.choice(n, size=min(6, n), replace=False)
        r, t = _svd_rigid(src[sample], dst[sample])
        pred = (r @ src.T).T + t
        err = np.linalg.norm(pred - dst, axis=1)
        inliers = err < thresh
        count = int(inliers.sum())
        if count > best_inliers:
            best_inliers = count
            r, t = _svd_rigid(src[inliers], dst[inliers])
            best_pose = pose_from_rt(r, t)
    if best_pose is None or best_inliers < 12:
        return None
    return best_pose


def _svd_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    h = (src - src_mean).T @ (dst - dst_mean)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = dst_mean - r @ src_mean
    return r, t


class RGBDOdometryLocalizer(Localizer):
    """Frame-to-frame VO. Drift is expected; use ICP for map-relative tracking."""

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        self._orb = cv2.ORB_create(nfeatures=800)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_gray: np.ndarray | None = None
        self._prev_kps = None
        self._prev_desc = None
        self._prev_depth = None
        self._prev_intr = None

    def track(self, frame: RGBDFrame, volume: TSDFVolume) -> Pose:
        gray = cv2.cvtColor(frame.color, cv2.COLOR_RGB2GRAY)
        kps, desc = self._orb.detectAndCompute(gray, None)
        if not self.initialized:
            if frame.pose_world_camera is not None:
                self.pose = frame.pose_world_camera.astype(float)
            self.initialized = True
            self._store_prev(gray, kps, desc, frame)
            self.trajectory.append(self.pose.copy())
            return self.pose

        if desc is None or self._prev_desc is None or kps is None or self._prev_kps is None:
            self._store_prev(gray, kps, desc, frame)
            self.trajectory.append(self.pose.copy())
            return self.pose

        matches = self._matcher.match(self._prev_desc, desc)
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        src_pts, dst_pts = [], []
        for m in matches:
            u0, v0 = self._prev_kps[m.queryIdx].pt
            u1, v1 = kps[m.trainIdx].pt
            z0 = self._sample_depth(self._prev_depth, u0, v0)
            z1 = self._sample_depth(frame.depth, u1, v1)
            if z0 is None or z1 is None:
                continue
            src_pts.append(self._prev_intr.backproject(np.array([u0]), np.array([v0]), np.array([z0]))[0])
            dst_pts.append(frame.intrinsics.backproject(np.array([u1]), np.array([v1]), np.array([z1]))[0])

        if len(src_pts) >= 12:
            src = np.asarray(src_pts, dtype=np.float64)
            dst = np.asarray(dst_pts, dtype=np.float64)
            t_prev_curr = _rigid_ransac(dst, src, self.config.vo_ransac_iters, thresh=0.05)
            # t_prev_curr maps current-camera points into previous camera.
            if t_prev_curr is not None:
                self.pose = self.pose @ t_prev_curr
        else:
            log.debug("VO had too few 3D matches (%s)", len(src_pts))

        self._store_prev(gray, kps, desc, frame)
        self.trajectory.append(self.pose.copy())
        return self.pose

    def _store_prev(self, gray, kps, desc, frame: RGBDFrame) -> None:  # noqa: ANN001
        self._prev_gray = gray
        self._prev_kps = kps
        self._prev_desc = desc
        self._prev_depth = frame.depth
        self._prev_intr = frame.intrinsics

    @staticmethod
    def _sample_depth(depth: np.ndarray, u: float, v: float) -> float | None:
        x, y = int(round(u)), int(round(v))
        if x < 0 or y < 0 or y >= depth.shape[0] or x >= depth.shape[1]:
            return None
        z = float(depth[y, x])
        if not np.isfinite(z) or z < 0.15 or z > 8.0:
            return None
        return z
