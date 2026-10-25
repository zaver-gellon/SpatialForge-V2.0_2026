"""Conservative 2D A* on a height-sliced occupancy map."""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray

from spatialforge.types import PathQuery


NEIGHBORS = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, 1.414),
    (1, -1, 1.414),
    (-1, 1, 1.414),
    (-1, -1, 1.414),
]


def inflate(occupied: NDArray[np.bool_], radius_cells: int) -> NDArray[np.bool_]:
    if radius_cells <= 0:
        return occupied
    from scipy.ndimage import binary_dilation

    structure = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    return binary_dilation(occupied, structure=structure)


def astar(
    occupancy: NDArray[np.int8],
    start_ij: tuple[int, int],
    goal_ij: tuple[int, int],
    unknown_blocked: bool = True,
) -> list[tuple[int, int]]:
    nx, ny = occupancy.shape

    def blocked(i: int, j: int) -> bool:
        if i < 0 or j < 0 or i >= nx or j >= ny:
            return True
        val = occupancy[i, j]
        if val == 1:
            return True
        if val < 0 and unknown_blocked:
            return True
        return False

    if blocked(*start_ij) or blocked(*goal_ij):
        return []

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    open_set: list[tuple[float, tuple[int, int]]] = [(0.0, start_ij)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    gscore = {start_ij: 0.0}
    seen: set[tuple[int, int]] = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in seen:
            continue
        seen.add(current)
        if current == goal_ij:
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            path.reverse()
            return path
        for di, dj, cost in NEIGHBORS:
            nb = (current[0] + di, current[1] + dj)
            if blocked(*nb):
                continue
            tentative = gscore[current] + cost
            if tentative < gscore.get(nb, 1e18):
                came[nb] = current
                gscore[nb] = tentative
                heapq.heappush(open_set, (tentative + heuristic(nb, goal_ij), nb))
    return []


def path_query_from_indices(
    cells: list[tuple[int, int]],
    origin: np.ndarray,
    voxel_size: float,
    z: float,
) -> PathQuery:
    if not cells:
        return PathQuery(waypoints=[], length_m=0.0, reachable=False, reason="no collision-free path")
    waypoints = [
        np.array(
            [
                origin[0] + (i + 0.5) * voxel_size,
                origin[1] + (j + 0.5) * voxel_size,
                z,
            ],
            dtype=np.float32,
        )
        for i, j in cells
    ]
    length = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        length += float(np.linalg.norm(b - a))
    return PathQuery(waypoints=waypoints, length_m=length, reachable=True, reason="ok")
