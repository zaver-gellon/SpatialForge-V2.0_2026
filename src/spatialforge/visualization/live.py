"""Optional Open3D live window. The default path is the occupancy dashboard."""

from __future__ import annotations

import numpy as np

from spatialforge.utils.logging import get_logger

log = get_logger("viz.live")


class LiveCloudView:
    def __init__(self) -> None:
        try:
            import open3d as o3d  # type: ignore
        except ImportError as exc:
            raise ImportError("Open3D is optional. pip install 'spatialforge[viz]'") from exc
        self._o3d = o3d
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="SpatialForge", width=960, height=720)
        self._cloud = o3d.geometry.PointCloud()
        self._added = False

    def update(self, points: np.ndarray, colors: np.ndarray) -> None:
        if points.shape[0] == 0:
            return
        self._cloud.points = self._o3d.utility.Vector3dVector(points.astype(np.float64))
        self._cloud.colors = self._o3d.utility.Vector3dVector(np.clip(colors / 255.0, 0, 1))
        if not self._added:
            self._vis.add_geometry(self._cloud)
            self._added = True
        else:
            self._vis.update_geometry(self._cloud)
        self._vis.poll_events()
        self._vis.update_renderer()

    def close(self) -> None:
        self._vis.destroy_window()
