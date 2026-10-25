import numpy as np

from spatialforge.core import ForgeCore
from spatialforge.spatial.planner import astar
from spatialforge.utils.config import load_config


def test_astar_finds_path_around_wall():
    grid = np.zeros((12, 12), dtype=np.int8)
    grid[5, 1:11] = 1
    path = astar(grid, (2, 2), (8, 2), unknown_blocked=False)
    assert path
    assert path[0] == (2, 2)
    assert path[-1] == (8, 2)


def test_forgecore_simulation_and_queries(tmp_path):
    cfg = load_config("configs/simulation.yaml")
    cfg.max_frames = 6
    cfg.visualization.enabled = False
    cfg.output_dir = str(tmp_path)
    cfg.sensor.width = 96
    cfg.sensor.height = 72
    cfg.sensor.fx = 70
    cfg.sensor.fy = 70
    cfg.sensor.cx = 48
    cfg.sensor.cy = 36
    cfg.volume.voxel_size = 0.08

    core = ForgeCore(cfg)
    core.start()
    try:
        for _ in range(cfg.max_frames):
            result = core.step()
            assert result is not None
            assert result.integrated_voxels >= 0
    finally:
        ply = core.export_map()
        core.stop()

    assert ply.exists()
    assert core.volume.stats()["observed"] > 50
    free = core.mind.free_space_ahead(core.last_pose)
    assert free.min_clearance_m >= 0.0
    origin = core.last_pose[:3, 3]
    # A point far above the room should be unknown or free, not a crash.
    core.mind.classify_point(origin + np.array([0.0, 0.0, 3.0]))
