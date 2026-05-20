#!/usr/bin/env python3
"""
cloud_to_splat.py — convert webui /cloud.bin + /cloud_rgb.bin into an
Inria-format Gaussian Splat binary (.splat, 32 bytes/splat) and serve via
state/splats/latest.splat. Runs as a daemon polling every 5 s.

Per-splat layout (32 B, little-endian):
    [ 0:12]  position xyz        float32 × 3
    [12:24]  scale xyz (raw, m)  float32 × 3
    [24:28]  RGBA                uint8  × 4   (A = opacity, 0..255)
    [28:32]  rotation quaternion uint8  × 4   (encoding -1..1 -> 0..255,
                                                identity = (255,128,128,128))
"""
from __future__ import annotations

import os
import sys
import time
import shutil
import struct
import urllib.request
from pathlib import Path
from datetime import datetime

WEBUI = os.environ.get("SLAM_WEBUI", "http://localhost:8080")
STATE_DIR = Path(os.path.expanduser("~/PX4-Autopilot/slam_rover/state/splats"))
LATEST = STATE_DIR / "latest.splat"
POLL_SEC = 5.0
SCALE_M = 0.06          # voxel half-extent in meters (voxel size 0.10 → 0.06 gives slight overlap)
OPACITY = 242           # ~0.95
IDENT_QUAT = bytes((255, 128, 128, 128))   # (w=1, x=0, y=0, z=0) in 0..255
ARCHIVE_EVERY_SEC = 60  # write a timestamped copy at most once per minute


def fetch(path: str) -> bytes:
    with urllib.request.urlopen(WEBUI + path, timeout=5.0) as r:
        return r.read()


def build_splat(xyz: bytes, rgb: bytes) -> bytes:
    n_pts = len(xyz) // 12
    n_rgb = len(rgb) // 3
    n = min(n_pts, n_rgb)
    if n == 0:
        return b""

    scale_bytes = struct.pack("<fff", SCALE_M, SCALE_M, SCALE_M)
    out = bytearray(32 * n)
    mv = memoryview(out)
    for i in range(n):
        # position
        mv[i * 32:i * 32 + 12] = xyz[i * 12:i * 12 + 12]
        # scale
        mv[i * 32 + 12:i * 32 + 24] = scale_bytes
        # color RGBA
        r = rgb[i * 3 + 0]
        g = rgb[i * 3 + 1]
        b = rgb[i * 3 + 2]
        mv[i * 32 + 24] = r
        mv[i * 32 + 25] = g
        mv[i * 32 + 26] = b
        mv[i * 32 + 27] = OPACITY
        # rotation (identity)
        mv[i * 32 + 28:i * 32 + 32] = IDENT_QUAT
    return bytes(out)


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[splat] daemon → {WEBUI}  out={LATEST}", flush=True)
    last_archive = 0.0

    while True:
        t0 = time.time()
        try:
            xyz = fetch("/cloud.bin")
            rgb = fetch("/cloud_rgb.bin")
            blob = build_splat(xyz, rgb)
            if not blob:
                print(f"[splat] empty cloud (xyz={len(xyz)}B rgb={len(rgb)}B)", flush=True)
            else:
                atomic_write(LATEST, blob)
                n = len(blob) // 32
                print(f"[splat] wrote {len(blob)}B ({n} splats) in {time.time()-t0:.2f}s",
                      flush=True)
                # Archive periodically
                now = time.time()
                if now - last_archive >= ARCHIVE_EVERY_SEC:
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    shutil.copyfile(LATEST, STATE_DIR / f"{ts}.splat")
                    last_archive = now
        except Exception as e:
            print(f"[splat] err: {e}", flush=True)

        sleep_for = max(0.5, POLL_SEC - (time.time() - t0))
        time.sleep(sleep_for)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
