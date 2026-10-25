"""Zero-dependency occupancy dashboard for live map viewing."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np

from spatialforge.utils.logging import get_logger

log = get_logger("viz.dashboard")

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>SpatialForge — live occupancy</title>
  <style>
    :root { color-scheme: dark; }
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #0b0d12; color: #e8edf7; }
    header { padding: 1rem 1.5rem; border-bottom: 1px solid #1f2533; display: flex; justify-content: space-between; }
    h1 { font-size: 1.05rem; margin: 0; letter-spacing: 0.04em; }
    main { padding: 1.25rem; display: grid; gap: 1rem; grid-template-columns: 1fr 280px; }
    img { width: 100%; background: #11151f; border-radius: 12px; image-rendering: pixelated; }
    .card { background: #121722; border: 1px solid #1f2533; border-radius: 12px; padding: 1rem; }
    .muted { color: #8b95a8; font-size: 0.85rem; }
    code { color: #9ad0ff; }
  </style>
</head>
<body>
  <header>
    <h1>SPATIALFORGE / occupancy</h1>
    <div class="muted" id="meta">connecting…</div>
  </header>
  <main>
    <img id="map" alt="occupancy slice" src="/occupancy.png"/>
    <section class="card">
      <div class="muted">pose (world ← camera)</div>
      <pre id="pose" style="white-space:pre-wrap; font-size:12px;"></pre>
      <p class="muted">Legend: <span style="color:#5ee0a0">free</span> ·
      <span style="color:#f07178">occupied</span> ·
      <span style="color:#4b5568">unknown</span> ·
      <span style="color:#ffd166">camera</span></p>
      <p class="muted">This is the query layer an AI brain sits on top of — not a SLAM GUI.</p>
    </section>
  </main>
  <script>
    async function tick() {
      const st = await fetch('/status.json').then(r => r.json());
      document.getElementById('meta').textContent =
        `frame ${st.frame_id} · observed ${st.observed_fraction.toFixed(3)} · ${st.points} pts`;
      document.getElementById('pose').textContent = JSON.stringify(st.pose, null, 2);
      document.getElementById('map').src = '/occupancy.png?t=' + Date.now();
    }
    setInterval(tick, 400);
    tick();
  </script>
</body>
</html>
"""


class OccupancyDashboard:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._png = _blank_png()
        self._status: dict = {
            "frame_id": 0,
            "observed_fraction": 0.0,
            "points": 0,
            "pose": np.eye(4).tolist(),
        }
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/occupancy.png"):
                    with dashboard._lock:
                        payload = dashboard._png
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.path.startswith("/status.json"):
                    with dashboard._lock:
                        body = json.dumps(dashboard._status).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        log.info("Dashboard at http://%s:%s", self.host, self.port)

    def update(self, occupancy: np.ndarray, pose: np.ndarray, origin: np.ndarray, voxel: float, stats: dict) -> None:
        canvas, scale = _colorize(occupancy)
        _draw_camera(canvas, occupancy.shape, pose, origin, voxel, scale)
        ok, buf = cv2.imencode(".png", canvas)
        if not ok:
            return
        with self._lock:
            self._png = buf.tobytes()
            self._status = {
                "frame_id": int(stats.get("integrations", 0)),
                "observed_fraction": float(stats.get("observed_fraction", 0.0)),
                "points": int(stats.get("points", 0)),
                "pose": [[round(float(v), 3) for v in row] for row in pose[:3].tolist()],
            }

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None


def _colorize(occupancy: np.ndarray) -> tuple[np.ndarray, int]:
    # occupancy[i=x, j=y] → image with +x right and +y up.
    grid = occupancy.T[::-1]
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (28, 32, 42)
    img[grid == 0] = (90, 210, 150)
    img[grid == 1] = (80, 90, 230)
    scale = max(4, 480 // max(h, w))
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST), scale


def _draw_camera(
    canvas: np.ndarray,
    shape_xy: tuple[int, int],
    pose: np.ndarray,
    origin: np.ndarray,
    voxel: float,
    scale: int,
) -> None:
    nx, ny = shape_xy
    i = int((pose[0, 3] - origin[0]) / voxel)
    j = int((pose[1, 3] - origin[1]) / voxel)
    col = int((i + 0.5) * scale)
    row = int((ny - 1 - j + 0.5) * scale)
    if 0 <= col < canvas.shape[1] and 0 <= row < canvas.shape[0]:
        cv2.circle(canvas, (col, row), max(4, scale), (102, 209, 255), -1)
        forward = pose[:3, 2]
        tip_i = i + int(np.round(forward[0] * 4))
        tip_j = j + int(np.round(forward[1] * 4))
        tip_col = int((tip_i + 0.5) * scale)
        tip_row = int((ny - 1 - tip_j + 0.5) * scale)
        cv2.line(canvas, (col, row), (tip_col, tip_row), (102, 209, 255), 2)


def _blank_png() -> bytes:
    img = np.zeros((240, 240, 3), dtype=np.uint8)
    img[:] = (18, 20, 28)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def try_open3d_cloud(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors / 255.0, 0, 1))
    o3d.io.write_point_cloud(str(path), cloud)
