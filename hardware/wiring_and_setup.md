# Hardware wiring and setup

## Jetson Orin Nano + RealSense

1. Flash JetPack (L4T) using NVIDIA SDK Manager. Confirm `nvidia-smi` / `tegra` stats look healthy.
2. Plug the RealSense into a **USB 3** port on the carrier. Do not use USB 2.
3. Power the Jetson from a dedicated 5V/4A+ supply. Shared laptop ports are not enough.
4. Install userspace:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libusb-1.0-0
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[realsense]"
```

5. Confirm the camera:

```bash
python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"
```

6. Run with the Jetson profile:

```bash
python scripts/run_mapping.py -c configs/jetson_realsense.yaml
```

7. Open `http://<jetson-ip>:8765` from a laptop on the same network. The dashboard binds `0.0.0.0` in that config.

### Mechanical

- Mount the camera **rigidly**. Tape and gooseneck mounts inject fake motion that ICP will try to explain as geometry.
- Optical axis should be roughly level. A 10–15° downward tilt is fine for floor robots.
- Keep the IR projector unobstructed. Fingers, 3D-printed hoods, and glossy paint all punch holes in depth.

### Power / thermal

- Orin Nano 15W mode: drop to 424×240 and 0.05 m voxels if integration exceeds ~80 ms.
- Disable GUI if you are headless. The occupancy dashboard is cheaper than an Open3D window.

## Raspberry Pi 5

1. Raspberry Pi OS 64-bit, latest EEPROM.
2. Official cooler + 27W PSU.
3. RealSense on the USB 3 (blue) port.
4. Use `configs/rpi.yaml` — coarser voxels, lower fps, visual odometry instead of dense ICP.

Pi 5 can maintain a 4×4×2 m volume at 6 cm voxels. Do not expect KinectFusion desktop rates.

## Simulation / bags (no camera)

No wiring. Record once from RealSense:

```bash
# on a machine with librealsense
rs-record-rosbag  # or use the RealSense Viewer "Record" button
python scripts/replay_bag.py --bag path/to/session.bag -c configs/simulation.yaml
```

SpatialForge `.npz` recordings (from the synthetic camera) replay the same path:

```bash
python -c "from spatialforge.cli import record_synthetic; record_synthetic('data/synthetic_room.npz')"
python scripts/replay_bag.py --bag data/synthetic_room.npz
```

## Pan-tilt / mobile base hookup

`ExplorationController.next_pose()` returns a 4×4 camera pose in the world frame. That is the contract.

- **Pan-tilt:** convert the pose’s optical axis into pan/tilt setpoints relative to the mast. Keep the TSDF origin fixed to the robot base, not the camera, or you will smear the map.
- **Mobile base:** project the next waypoint onto the floor, run a low-level heading controller, and only integrate frames when IMU/wheel odometry says the platform is quasi-static if your camera is rolling-shutter and the base is fast.

Do not close a high-rate wheel loop inside Python. ForgeCore is the spatial layer; a microcontroller owns motor current.
