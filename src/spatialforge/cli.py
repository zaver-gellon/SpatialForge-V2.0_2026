"""Command-line entry points used by scripts/ and pyproject console_scripts."""

from __future__ import annotations

import argparse

from spatialforge.core import ForgeCore
from spatialforge.exploration.controller import ExplorationController
from spatialforge.sensors.bag import write_npz
from spatialforge.sensors.factory import create_sensor
from spatialforge.sensors.synthetic import SyntheticSensor
from spatialforge.utils.config import load_config
from spatialforge.utils.logging import setup_logging


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
        default="configs/simulation.yaml",
        help="Path to a YAML config (default: configs/simulation.yaml)",
    )
    parser.add_argument("--frames", type=int, default=None, help="Override system.max_frames")
    parser.add_argument("--no-viz", action="store_true", help="Disable the occupancy dashboard")
    return parser


def _core_from_args(args: argparse.Namespace) -> ForgeCore:
    cfg = load_config(args.config)
    setup_logging(cfg.log_level)
    if args.frames is not None:
        cfg.max_frames = args.frames
    if args.no_viz:
        cfg.visualization.enabled = False
    return ForgeCore(cfg)


def run_mapping(argv: list[str] | None = None) -> None:
    args = _parser("Build a live TSDF map and track the camera.").parse_args(argv)
    core = _core_from_args(args)
    core.start()
    try:
        processed = 0
        while processed < core.config.max_frames:
            if core.step() is None:
                break
            processed += 1
        core.export_map()
    finally:
        core.stop()


def run_exploration(argv: list[str] | None = None) -> None:
    args = _parser("Actively explore and expand the map.").parse_args(argv)
    core = _core_from_args(args)
    core.config.exploration.enabled = True
    core.explorer = ExplorationController(core.config.exploration)
    core.start()
    try:
        processed = 0
        while processed < core.config.max_frames:
            if core.step() is None:
                break
            processed += 1
        core.export_map()
    finally:
        core.stop()


def query_demo(argv: list[str] | None = None) -> None:
    args = _parser("Map briefly, then exercise SpatialMind queries.").parse_args(argv)
    core = _core_from_args(args)
    core.start()
    try:
        processed = 0
        while processed < core.config.max_frames:
            if core.step() is None:
                break
            processed += 1
        _print_queries(core)
        core.export_map()
    finally:
        core.stop()


def replay_bag(argv: list[str] | None = None) -> None:
    parser = _parser("Replay a recorded .npz or RealSense .bag through ForgeCore.")
    parser.add_argument("--bag", required=True, help="Path to recording")
    args = parser.parse_args(argv)
    core = _core_from_args(args)
    core.config.sensor.type = "bag"
    core.config.sensor.bag_path = args.bag
    core.sensor = create_sensor(core.config.sensor)
    core.start()
    try:
        while core.step() is not None:
            pass
        core.export_map()
    finally:
        core.stop()


def record_synthetic(path: str = "data/synthetic_room.npz", frames: int = 40) -> None:
    """Write a tiny NPZ recording so replay works without hardware."""
    from spatialforge.utils.config import SensorConfig

    sensor = SyntheticSensor(SensorConfig())
    sensor.start()
    colors, depths, poses, timestamps = [], [], [], []
    for _ in range(frames):
        frame = sensor.capture()
        assert frame is not None
        colors.append(frame.color)
        depths.append(frame.depth)
        poses.append(frame.pose_world_camera)
        timestamps.append(frame.timestamp)
    write_npz(path, colors, depths, poses, timestamps, sensor.intrinsics)


def _print_queries(core: ForgeCore) -> None:
    import numpy as np

    from spatialforge.utils.geometry import look_at

    pose = core.last_pose
    origin = pose[:3, 3]
    print("\n=== SpatialMind query demo ===")
    print(f"camera at {np.round(origin, 3)}")

    air = np.array([1.8, -1.6, 0.9])
    pillar = np.array([0.95, 0.95, 0.7])
    print(f"is_occupied(air {np.round(air, 2)}) -> {core.mind.is_occupied(air)}")
    print(f"is_occupied(pillar {np.round(pillar, 2)}) -> {core.mind.is_occupied(pillar)}")

    open_pose = look_at(np.array([1.9, -1.7, 1.0]), np.array([0.0, 0.0, 0.5]))
    free = core.mind.free_space_ahead(open_pose, lookahead_m=2.0)
    print(
        f"free space from corner: min={free.min_clearance_m:.2f}m  "
        f"mean={free.mean_clearance_m:.2f}m  blocked={free.blocked_fraction:.0%}"
    )

    floor = core.mind.nearest_flat_surface(np.array([1.6, -1.4, 1.0]))
    if floor is not None:
        print(f"nearest flat surface: {np.round(floor.point, 3)}  dist={floor.distance:.2f}m")
    else:
        print("nearest flat surface: none yet")

    start, goal = _demo_free_endpoints(core)
    path = core.mind.plan_path(start, goal)
    print(
        f"path {np.round(start[:2], 2)} -> {np.round(goal[:2], 2)}: "
        f"reachable={path.reachable}  waypoints={len(path.waypoints)}  "
        f"length={path.length_m:.2f}m  ({path.reason})"
    )
    print("=== end queries ===\n")


def _demo_free_endpoints(core: ForgeCore) -> tuple:
    import numpy as np

    grid = core.mind.occupancy_xy()
    free = np.argwhere(grid == 0)
    z = 0.9
    if free.shape[0] < 2:
        origin = core.last_pose[:3, 3]
        return origin, origin + np.array([0.8, 0.0, 0.0])
    pts = np.column_stack(
        [
            core.volume.origin[0] + (free[:, 0] + 0.5) * core.volume.voxel_size,
            core.volume.origin[1] + (free[:, 1] + 0.5) * core.volume.voxel_size,
            np.full(len(free), z),
        ]
    )
    i = int(np.argmin(pts[:, 0] + pts[:, 1]))
    j = int(np.argmax(np.linalg.norm(pts[:, :2] - pts[i, :2], axis=1)))
    return pts[i], pts[j]
