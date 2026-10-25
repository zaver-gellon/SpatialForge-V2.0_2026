# Architecture

SpatialForge is a **spatial foundation**: a live metric world model plus a query API that higher-level software can treat as memory of the physical world.

The runtime object is `ForgeCore`. The query object an AI brain holds is `SpatialMind`.

```
RGB-D frame
    │
    ▼
SensorFrontend  ──►  Localizer  ──►  T_world_camera
    │                      │
    │                      ▼
    └──────────►  TSDFVolume.integrate()
                         │
                         ▼
                   SpatialMind  ──► occupancy, free space, surfaces, paths
                         │
                         ├── OccupancyDashboard
                         └── ExplorationController (optional)
```

## Modules

| Module | Responsibility | Non-responsibility |
| --- | --- | --- |
| `sensors` | Produce aligned RGB-D + optional pose | Calibration UI, multi-camera fusion |
| `mapping` | Fuse frames into a bounded TSDF | Loop closures, global bundle adjustment |
| `localization` | Track `T_world_camera` | Long-term SLAM / pose-graph |
| `spatial` | Answer questions about the volume | Task planning, language, grasping |
| `exploration` | Choose poses that grow the map | Motor control |
| `visualization` | Occupancy dashboard + PLY export | Photoreal rendering |

## Data structures

**`RGBDFrame`** — color (`uint8` HxWx3 RGB), depth (`float32` meters), pinhole intrinsics, optional ground-truth pose.

**`TSDFVolume`** — dense 3D arrays `sdf`, `weight`, `color`. World origin is the AABB minimum, z-up. Voxel size is the resolution knob.

**`Pose`** — 4×4 `T_world_camera`. Camera axes are OpenCV optical: x right, y down, z forward.

## Mapping

Integration is KinectFusion-style:

1. Voxel centers → camera frame via `T_cw = inv(T_world_camera)`.
2. Project with the pinhole model; bilinear-sample depth and color.
3. `sdf = measured_z − voxel_z`. Positive in free space (in front of the surface).
4. Truncate to `± μ` (`μ = truncation_voxels * voxel_size`) and ignore voxels too far in front of the hit.
5. Weighted running average of SDF and RGB.

This is vectorized NumPy, not CUDA. That is a deliberate edge-device trade: a 5×5×2.4 m volume at 5 cm is ~500k voxels and integrates in tens of milliseconds on an Orin Nano class CPU. Grow the AABB or shrink the voxel size only as far as RAM and frame time allow.

## Localization

Three interchangeable trackers, selected in YAML:

- **`ground_truth`** — simulation, motion capture, or bag metadata.
- **`icp`** — frame-to-model point-to-point ICP against TSDF surface voxels. This is the map-relative tracker for RealSense.
- **`vo`** — ORB + RANSAC rigid fit on unprojected matches. Drift-prone; intended as a Pi-friendly bootstrap.

There is no pose graph. If you need globally consistent multi-room maps, run a SLAM system *alongside* SpatialForge and feed it poses, or replace the localizer. The TSDF does not care who authored `T_world_camera`.

## SpatialMind

Queries never mutate the volume.

- **Occupancy** — SDF ≤ `occupied_sdf` and weight > 0.
- **Unknown** — weight ≈ 0. Conservative planning treats unknown as occupied.
- **Free space ahead** — a fan of rays in the camera-forward / world-xy plane.
- **Nearest (flat) surface** — surface voxels whose SDF gradient is the normal; flatness is `|n · up|`.
- **Path** — 8-connected A* on a height-sliced, inflated occupancy image.

This is the layer a policy, LLM tool-caller, or motion planner should import. Do not reach into `TSDFVolume` from application code.

## Threading and time

The mapping loop is single-threaded and blocking. The dashboard HTTP server is a daemon thread that reads a PNG snapshot under a lock. Keep `ForgeCore.step()` on one thread; RealSense and TSDF are not internally synchronized for concurrent `integrate` calls.

## Failure modes (by design)

- **Drift** — VO and ICP without loop closure will smear the TSDF. Bound the session, or inject a better pose.
- **AABB overflow** — geometry outside `volume.min_xyz/max_xyz` is dropped. Shift the origin with the robot if you roam.
- **Glass / IR black** — RealSense will report missing depth; those voxels stay unknown.
- **Moving people** — they carve tunnels of free space and ghost surfaces. Filter dynamically later; v0.1 integrates everything.

## Why not “just use RTAB-Map / Open3D SLAM / nvblox”?

Those are excellent at reconstructing and localizing. SpatialForge is opinionated the other way: a **small, readable, queryable world model** that is allowed to be locally consistent, bounded, and boring — so an AI brain can call `is_occupied` without taking a ROS dependency. See the README for the product-level distinction.
