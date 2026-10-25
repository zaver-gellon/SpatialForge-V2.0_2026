"""Rigid transforms, camera look-at, and depth unprojection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from spatialforge.types import CameraIntrinsics, Float32, Pose


def identity_pose() -> Pose:
    return np.eye(4, dtype=np.float64)


def invert_pose(t_ab: Pose) -> Pose:
    """Invert a 4x4 rigid transform."""
    r = t_ab[:3, :3]
    t = t_ab[:3, 3]
    t_ba = identity_pose()
    t_ba[:3, :3] = r.T
    t_ba[:3, 3] = -r.T @ t
    return t_ba


def pose_from_rt(rotation: NDArray, translation: NDArray) -> Pose:
    pose = identity_pose()
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return pose


def transform_points(points: NDArray, pose: Pose) -> Float32:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.astype(np.float32)
    out = pts @ pose[:3, :3].T + pose[:3, 3]
    return out.astype(np.float32)


def look_at(eye: NDArray, target: NDArray, world_up: NDArray | None = None) -> Pose:
    """Camera pose in a z-up world, with OpenCV optical axes (x right, y down, z forward)."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if world_up is None else np.asarray(world_up, dtype=np.float64)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("eye and target are coincident")
    z_axis = forward / norm
    down = -up
    x_axis = np.cross(down, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        down = np.array([0.0, -1.0, 0.0])
        x_axis = np.cross(down, z_axis)
        x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)

    pose = identity_pose()
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def unproject_depth(
    depth: NDArray,
    color: NDArray | None,
    intrinsics: CameraIntrinsics,
    stride: int = 2,
    min_depth: float = 0.15,
    max_depth: float = 8.0,
) -> tuple[Float32, Float32]:
    """Unproject a depth map into a camera-frame point cloud."""
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    z = depth[vs, us]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    us = us[valid].astype(np.float32)
    vs = vs[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    points = intrinsics.backproject(us, vs, z)
    if color is None:
        colors = np.full((points.shape[0], 3), 200, dtype=np.float32)
    else:
        colors = color[vs.astype(int), us.astype(int)].astype(np.float32)
    return points, colors


def sample_bilinear(image: NDArray, u: NDArray, v: NDArray) -> NDArray:
    """Bilinear sample a 2D image at floating pixel coordinates."""
    h, w = image.shape[:2]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, w - 1)
    v1 = np.clip(v0 + 1, 0, h - 1)
    u0 = np.clip(u0, 0, w - 1)
    v0 = np.clip(v0, 0, h - 1)
    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)
    if image.ndim == 2:
        i00 = image[v0, u0]
        i01 = image[v0, u1]
        i10 = image[v1, u0]
        i11 = image[v1, u1]
        du3 = du
        dv3 = dv
    else:
        i00 = image[v0, u0]
        i01 = image[v0, u1]
        i10 = image[v1, u0]
        i11 = image[v1, u1]
        du3 = du[..., None]
        dv3 = dv[..., None]
    return (1 - dv3) * ((1 - du3) * i00 + du3 * i01) + dv3 * ((1 - du3) * i10 + du3 * i11)
