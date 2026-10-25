"""Map export helpers that do not require Open3D."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_point_cloud(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    rgb = np.clip(colors, 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {n}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for p, c in zip(points, rgb):
            handle.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
