"""Occupancy view of a TSDF. Kept separate so a future native occupancy mapper can swap in."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from spatialforge.mapping.tsdf import TSDFVolume


class OccupancyGrid:
    def __init__(self, volume: TSDFVolume, occupied_sdf: float) -> None:
        self.volume = volume
        self.occupied_sdf = occupied_sdf

    def slice(self, z_min: float, z_max: float) -> NDArray[np.int8]:
        grid, _ = self.volume.occupancy_slice(z_min, z_max, self.occupied_sdf)
        return grid
