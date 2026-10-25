"""Dense truncated signed-distance volume with color.

This is a KinectFusion-style integrator designed to stay in NumPy so it can run
on a Jetson Orin Nano or Raspberry Pi 5 without a CUDA TSDF backend. Volumes are
intentionally bounded; the default 5x5x2.4 m room at 5 cm is ~500k voxels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from spatialforge.types import RGBDFrame
from spatialforge.utils.config import VolumeConfig
from spatialforge.utils.geometry import invert_pose, sample_bilinear
from spatialforge.utils.logging import get_logger

log = get_logger("mapping.tsdf")


class TSDFVolume:
    """World-frame TSDF with per-voxel SDF, weight, and RGB."""

    def __init__(self, config: VolumeConfig) -> None:
        self.config = config
        self.bounds = config.bounds
        self.voxel_size = float(config.voxel_size)
        self.truncation = float(config.truncation)
        origin = np.array(self.bounds.min_xyz, dtype=np.float64)
        size = self.bounds.size()
        shape = np.ceil(size / self.voxel_size).astype(np.int32)
        self.origin = origin
        self.shape = tuple(int(v) for v in shape)
        self.sdf = np.ones(self.shape, dtype=np.float32)
        self.weight = np.zeros(self.shape, dtype=np.float32)
        self.color = np.zeros(self.shape + (3,), dtype=np.float32)
        self.integrate_count = 0
        log.info(
            "TSDF volume %s voxels (%.1f MB) voxel=%.3fm trunc=%.3fm",
            self.shape,
            self._nbytes() / 1e6,
            self.voxel_size,
            self.truncation,
        )

    def _nbytes(self) -> int:
        return int(self.sdf.nbytes + self.weight.nbytes + self.color.nbytes)

    def voxel_centers(self) -> NDArray[np.float32]:
        nx, ny, nz = self.shape
        xs = self.origin[0] + (np.arange(nx) + 0.5) * self.voxel_size
        ys = self.origin[1] + (np.arange(ny) + 0.5) * self.voxel_size
        zs = self.origin[2] + (np.arange(nz) + 0.5) * self.voxel_size
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
        return grid.astype(np.float32)

    def world_to_index(self, points: NDArray) -> NDArray[np.int32]:
        idx = np.floor((np.asarray(points, dtype=np.float64) - self.origin) / self.voxel_size)
        return idx.astype(np.int32)

    def in_bounds(self, idx: NDArray) -> NDArray[np.bool_]:
        idx = np.asarray(idx)
        return (
            (idx[..., 0] >= 0)
            & (idx[..., 0] < self.shape[0])
            & (idx[..., 1] >= 0)
            & (idx[..., 1] < self.shape[1])
            & (idx[..., 2] >= 0)
            & (idx[..., 2] < self.shape[2])
        )

    def integrate(self, frame: RGBDFrame, pose_world_camera: NDArray) -> int:
        """Fuse one RGB-D frame. Returns the number of updated voxels."""
        t_cw = invert_pose(np.asarray(pose_world_camera, dtype=np.float64))
        r = t_cw[:3, :3]
        t = t_cw[:3, 3]
        depth = frame.depth.astype(np.float32)
        color = frame.color.astype(np.float32)
        fx, fy, cx, cy = frame.intrinsics.fx, frame.intrinsics.fy, frame.intrinsics.cx, frame.intrinsics.cy
        h, w = depth.shape
        mu = self.truncation

        centers = self.voxel_centers().reshape(-1, 3).astype(np.float64)
        cam = (centers @ r.T) + t
        z = cam[:, 2]
        valid = z > 0.12
        u = fx * (cam[:, 0] / np.where(z == 0, 1e-6, z)) + cx
        v = fy * (cam[:, 1] / np.where(z == 0, 1e-6, z)) + cy
        valid &= (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)

        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return 0

        u_v = u[idx]
        v_v = v[idx]
        z_v = z[idx].astype(np.float32)
        d_sample = sample_bilinear(depth, u_v, v_v).astype(np.float32)
        good = np.isfinite(d_sample) & (d_sample > 0.12) & (d_sample < 8.0)
        sdf = d_sample - z_v
        # KinectFusion skips sdf > μ. We still write +μ so SpatialMind can see
        # free space between the camera and the surface, not only a thin shell.
        good &= sdf > -mu
        if not np.any(good):
            return 0

        idx = idx[good]
        sdf = np.clip(sdf[good], -mu, mu)
        rgb = sample_bilinear(color, u_v[good], v_v[good])

        nx, ny, nz = self.shape
        ix = idx // (ny * nz)
        rem = idx % (ny * nz)
        iy = rem // nz
        iz = rem % nz

        w_old = self.weight[ix, iy, iz]
        w_new = np.minimum(w_old + 1.0, self.config.max_weight)
        sdf_old = self.sdf[ix, iy, iz]
        self.sdf[ix, iy, iz] = (sdf_old * w_old + sdf) / w_new
        col_old = self.color[ix, iy, iz]
        self.color[ix, iy, iz] = (col_old * w_old[:, None] + rgb) / w_new[:, None]
        self.weight[ix, iy, iz] = w_new
        self.integrate_count += 1
        return int(idx.size)

    def extract_surface(
        self,
        min_weight: float = 1.5,
        iso: float | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Surface voxels (|sdf| < iso) as a colored point cloud plus normals."""
        iso = self.voxel_size if iso is None else iso
        mask = (self.weight >= min_weight) & (np.abs(self.sdf) <= iso)
        if not np.any(mask):
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty, empty
        points = self.voxel_centers()[mask]
        colors = np.clip(self.color[mask], 0, 255)
        normals = self._normals_at_mask(mask)
        return points, colors, normals

    def _normals_at_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.float32]:
        gx = np.zeros_like(self.sdf)
        gy = np.zeros_like(self.sdf)
        gz = np.zeros_like(self.sdf)
        gx[1:-1] = (self.sdf[2:] - self.sdf[:-2]) / (2 * self.voxel_size)
        gy[:, 1:-1] = (self.sdf[:, 2:] - self.sdf[:, :-2]) / (2 * self.voxel_size)
        gz[:, :, 1:-1] = (self.sdf[:, :, 2:] - self.sdf[:, :, :-2]) / (2 * self.voxel_size)
        normals = np.stack([gx[mask], gy[mask], gz[mask]], axis=-1)
        n = np.linalg.norm(normals, axis=1, keepdims=True)
        n = np.clip(n, 1e-6, None)
        return (normals / n).astype(np.float32)

    def occupancy_slice(self, z_min: float, z_max: float, occupied_sdf: float) -> tuple[NDArray, NDArray]:
        """Project a height band into a 2D occupancy grid: 0 free, 1 occupied, -1 unknown."""
        iz0 = int(np.clip((z_min - self.origin[2]) / self.voxel_size, 0, self.shape[2] - 1))
        iz1 = int(np.clip((z_max - self.origin[2]) / self.voxel_size, 0, self.shape[2] - 1)) + 1
        sdf = self.sdf[:, :, iz0:iz1]
        w = self.weight[:, :, iz0:iz1]
        occupied = np.any((w > 0.5) & (sdf <= occupied_sdf), axis=2)
        observed = np.any(w > 0.5, axis=2)
        grid = np.full(occupied.shape, -1, dtype=np.int8)
        grid[observed] = 0
        grid[occupied] = 1
        return grid, observed

    def sdf_at(self, points: NDArray) -> tuple[NDArray, NDArray]:
        idx = self.world_to_index(points)
        valid = self.in_bounds(idx)
        sdf = np.full(len(idx), np.nan, dtype=np.float32)
        weight = np.zeros(len(idx), dtype=np.float32)
        if np.any(valid):
            ii = idx[valid]
            sdf[valid] = self.sdf[ii[:, 0], ii[:, 1], ii[:, 2]]
            weight[valid] = self.weight[ii[:, 0], ii[:, 1], ii[:, 2]]
        return sdf, weight

    def stats(self) -> dict[str, float]:
        observed = float(np.count_nonzero(self.weight > 0.5))
        return {
            "voxels": float(np.prod(self.shape)),
            "observed": observed,
            "observed_fraction": observed / max(np.prod(self.shape), 1),
            "integrations": float(self.integrate_count),
        }
