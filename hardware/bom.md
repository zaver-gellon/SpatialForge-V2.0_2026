# Bill of materials

Two supported stacks. Start with the simulation profile if you do not have hardware yet.

## Primary — Jetson spatial computer

| Item | Qty | Notes |
| --- | --- | --- |
| NVIDIA Jetson Orin Nano 8GB developer kit | 1 | Primary compute. 15W–25W modes both work; 25W is more comfortable at 640×480. |
| Intel RealSense D435i or D455 | 1 | D455 for outdoor / longer range; D435i for indoor and IMU. USB 3 required. |
| USB 3 cable (USB-C to USB-A, SuperSpeed) | 1 | Short, high-quality cable. Avoid unpowered hubs. |
| 5V / 4A+ power for Jetson | 1 | Undersized PSUs cause USB brownouts and dropped RealSense frames. |
| Optional: USB 3 powered hub | 1 | Only if you must share the bus with other devices. |
| Optional: aluminum camera mount | 1 | Rigid mount. Mapping quality is dominated by extrinsic stability. |
| microSD / NVMe | 1 | NVMe strongly preferred over SD for map dumps. |

## Fallback — Raspberry Pi 5

| Item | Qty | Notes |
| --- | --- | --- |
| Raspberry Pi 5 (8GB) | 1 | Use the official 27W PSU. |
| Active cooler | 1 | TSDF integration is CPU-bound; thermal throttle kills frame rate. |
| Intel RealSense D435i **or** stereo module | 1 | RealSense still preferred. Pi Camera v3 stereo is a future frontend. |
| USB 3 port (blue) | 1 | RealSense on USB 2 will starve depth. |

## Optional physical exploration

| Item | Qty | Notes |
| --- | --- | --- |
| Differential-drive base (e.g. iRobot Create / custom) | 1 | `ExplorationController` currently emits poses; you translate them to wheel commands. |
| Pan-tilt unit (2× hobby servos + driver) | 1 | Enough to grow a map from a fixed base. |
| Emergency stop | 1 | Required if the base can move under software control. |

## Consumables / lab

- Printed 60 mm calibration checkerboard
- Matte tape to kill RealSense IR glare on glossy floors
- Zip ties / 2020 extrusion for a rigid sensor mast
