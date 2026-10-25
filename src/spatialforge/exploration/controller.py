"""Frontier-based exploration that drives a simulated (or real) camera pose."""

from __future__ import annotations

import numpy as np

from spatialforge.spatial.mind import SpatialMind
from spatialforge.types import Pose
from spatialforge.utils.config import ExplorationConfig
from spatialforge.utils.geometry import look_at
from spatialforge.utils.logging import get_logger

log = get_logger("exploration")


class ExplorationController:
    """Picks the nearest frontier and walks the camera toward it.

    On hardware this is the seam where you plug in a mobile base or pan-tilt
    command interface. In simulation it writes a pose back into the sensor.
    """

    def __init__(self, config: ExplorationConfig) -> None:
        self.config = config
        self.goal: np.ndarray | None = None

    def next_pose(self, current: Pose, mind: SpatialMind) -> Pose:
        origin = current[:3, 3]
        if self.goal is None or np.linalg.norm(self.goal[:2] - origin[:2]) < self.config.goal_radius:
            self.goal = self._select_frontier(origin, mind)
            if self.goal is not None:
                log.info("New exploration goal: [%.2f, %.2f, %.2f]", *self.goal)

        if self.goal is None:
            return current

        path = mind.plan_path(origin, self.goal)
        if not path.reachable or len(path.waypoints) < 2:
            self.goal = None
            return current

        step = path.waypoints[min(3, len(path.waypoints) - 1)]
        eye = np.array([step[0], step[1], max(float(origin[2]), 0.95)])
        target = np.array([self.goal[0], self.goal[1], 0.5])
        return look_at(eye, target)

    def _select_frontier(self, origin: np.ndarray, mind: SpatialMind) -> np.ndarray | None:
        pts = mind.frontiers()
        if pts.shape[0] == 0:
            log.info("No remaining frontiers")
            return None
        dist = np.linalg.norm(pts[:, :2] - origin[:2], axis=1)
        return pts[int(np.argmin(dist))]
