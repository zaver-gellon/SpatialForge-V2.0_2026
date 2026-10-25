"""SpatialMind — the query surface an AI brain talks to."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from spatialforge.mapping.tsdf import TSDFVolume
from spatialforge.spatial.planner import astar, inflate, path_query_from_indices
from spatialforge.types import FreeSpaceReport, PathQuery, Pose, SurfaceHit
from spatialforge.utils.config import SpatialConfig
from spatialforge.utils.logging import get_logger

log = get_logger("spatial.mind")

OCCUPIED = 1
FREE = 0
UNKNOWN = -1


class SpatialMind:
    """Read-only spatial intelligence over the live TSDF world model.

    This is intentionally not a SLAM toolbox. It answers questions a planner,
    manipulator, or language-conditioned policy would ask of the physical world.
    """

    def __init__(self, volume: TSDFVolume, config: SpatialConfig) -> None:
        self.volume = volume
        self.config = config

    def occupancy_xy(self) -> NDArray[np.int8]:
        grid, _ = self.volume.occupancy_slice(
            self.config.robot_height_min,
            self.config.robot_height_max,
            self.config.occupied_sdf,
        )
        occupied = grid == OCCUPIED
        radius = int(round(self.config.inflations_m / self.volume.voxel_size))
        if radius > 0:
            occupied = inflate(occupied, radius)
            grid = grid.copy()
            grid[occupied] = OCCUPIED
        return grid

    def classify_point(self, point: NDArray) -> int:
        """Return 1 occupied, 0 free, -1 unknown."""
        sdf, weight = self.volume.sdf_at(np.asarray(point, dtype=np.float64).reshape(1, 3))
        if weight[0] < 0.5 or not np.isfinite(sdf[0]):
            return UNKNOWN
        if sdf[0] <= self.config.occupied_sdf:
            return OCCUPIED
        return FREE

    def is_occupied(self, point: NDArray) -> bool:
        return self.classify_point(point) == OCCUPIED

    def is_free(self, point: NDArray) -> bool:
        return self.classify_point(point) == FREE

    def nearest_surface(self, point: NDArray, search_radius: float = 1.5) -> SurfaceHit | None:
        """Closest observed surface voxel to a world point."""
        pts, _, normals = self.volume.extract_surface(min_weight=1.0, iso=self.volume.voxel_size * 1.5)
        if pts.shape[0] == 0:
            return None
        p = np.asarray(point, dtype=np.float32).reshape(1, 3)
        dist = np.linalg.norm(pts - p, axis=1)
        i = int(np.argmin(dist))
        if dist[i] > search_radius:
            return None
        return SurfaceHit(point=pts[i], normal=normals[i], distance=float(dist[i]), occupied=True)

    def surface_normal(self, point: NDArray) -> NDArray | None:
        hit = self.nearest_surface(point)
        return None if hit is None else hit.normal

    def nearest_flat_surface(
        self,
        point: NDArray,
        up: NDArray | None = None,
        max_tilt_deg: float = 25.0,
        search_radius: float = 2.0,
    ) -> SurfaceHit | None:
        """Nearest surface whose normal is close to world-up — a table, floor, or shelf."""
        pts, _, normals = self.volume.extract_surface(min_weight=1.5, iso=self.volume.voxel_size * 1.5)
        if pts.shape[0] == 0:
            return None
        up_v = np.array([0.0, 0.0, 1.0], dtype=np.float32) if up is None else np.asarray(up, dtype=np.float32)
        up_v = up_v / max(float(np.linalg.norm(up_v)), 1e-6)
        cos_thresh = float(np.cos(np.deg2rad(max_tilt_deg)))
        alignment = np.abs(normals @ up_v)
        p = np.asarray(point, dtype=np.float32).reshape(1, 3)
        dist = np.linalg.norm(pts - p, axis=1)
        mask = (alignment >= cos_thresh) & (dist <= search_radius) & (pts[:, 2] < 1.05)
        if not np.any(mask):
            return None
        candidates = np.flatnonzero(mask)
        score = dist[mask] / (alignment[mask] + 1e-3)
        i = int(candidates[int(np.argmin(score))])
        return SurfaceHit(point=pts[i], normal=normals[i], distance=float(dist[i]), occupied=True)

    def free_space_ahead(self, pose: Pose, lookahead_m: float = 2.0, n_rays: int = 21) -> FreeSpaceReport:
        """Fan of rays in the camera forward (optical z) / world-xy plane."""
        origin = pose[:3, 3]
        forward = pose[:3, 2]
        right = pose[:3, 0]
        forward[2] = 0.0
        right[2] = 0.0
        fn = np.linalg.norm(forward)
        rn = np.linalg.norm(right)
        if fn < 1e-6 or rn < 1e-6:
            forward = np.array([1.0, 0.0, 0.0])
            right = np.array([0.0, 1.0, 0.0])
        else:
            forward = forward / fn
            right = right / rn

        angles = np.linspace(-0.6, 0.6, n_rays)
        hits = []
        clearances = []
        step = self.volume.voxel_size * 0.8
        for ang in angles:
            direction = np.cos(ang) * forward + np.sin(ang) * right
            direction = direction / np.linalg.norm(direction)
            clearance = lookahead_m
            hit_pt = origin + direction * lookahead_m
            t = step
            while t <= lookahead_m:
                p = origin + direction * t
                p[2] = origin[2]
                label = self.classify_point(p)
                if label == OCCUPIED:
                    clearance = t
                    hit_pt = p
                    break
                t += step
            clearances.append(clearance)
            hits.append(hit_pt)

        blocked = float(np.mean(np.array(clearances) < lookahead_m * 0.95))
        return FreeSpaceReport(
            min_clearance_m=float(np.min(clearances)),
            mean_clearance_m=float(np.mean(clearances)),
            blocked_fraction=blocked,
            lookahead_m=lookahead_m,
            ray_hits=np.asarray(hits, dtype=np.float32),
        )

    def plan_path(self, start: NDArray, goal: NDArray) -> PathQuery:
        grid = self.occupancy_xy()
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        s_idx = self.volume.world_to_index(start.reshape(1, 3))[0]
        g_idx = self.volume.world_to_index(goal.reshape(1, 3))[0]
        start_ij = (int(s_idx[0]), int(s_idx[1]))
        goal_ij = (int(g_idx[0]), int(g_idx[1]))
        cells = astar(grid, start_ij, goal_ij, unknown_blocked=self.config.unknown_is_occupied)
        z = float(0.5 * (self.config.robot_height_min + self.config.robot_height_max))
        return path_query_from_indices(cells, self.volume.origin, self.volume.voxel_size, z)

    def frontiers(self) -> NDArray[np.float32]:
        """World-xy centers of free cells that touch unknown — exploration seeds."""
        grid = self.occupancy_xy()
        nx, ny = grid.shape
        pts = []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if grid[i, j] != FREE:
                    continue
                neighborhood = grid[i - 1 : i + 2, j - 1 : j + 2]
                if np.any(neighborhood == UNKNOWN):
                    x = self.volume.origin[0] + (i + 0.5) * self.volume.voxel_size
                    y = self.volume.origin[1] + (j + 0.5) * self.volume.voxel_size
                    pts.append((x, y, 0.9))
        if not pts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)
