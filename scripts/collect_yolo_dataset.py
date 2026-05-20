#!/usr/bin/env python3
"""Auto-label YOLO dataset from gz GT for fine-tune.

Captures front+rear fisheye frames + computes YOLO-format bbox labels by
projecting GT object AABBs through the equidistant fisheye lens model.

Output structure:
  ~/PX4-Autopilot/slam_rover/datasets/gz_v1/
    images/train/{front,rear}_<seq>.jpg
    labels/train/{front,rear}_<seq>.txt   # YOLO format: cls cx cy w h (normalized)
    data.yaml

Run while rover explores. Target ~2000 frames.

YOLO class indices (canonical map):
  0=person, 1=cone, 2=box, 3=pillar, 4=tree, 5=bench, 6=shelf, 7=forklift,
  8=platform, 9=sign
"""
import json
import math
import os
import pathlib
import time
import urllib.request

import cv2  # uses /tmp/yoloenv python which has opencv
import numpy as np


WEBUI = "http://localhost:8080"
OUT = pathlib.Path.home() / "PX4-Autopilot/slam_rover/datasets/gz_v1"
IMG_DIR = OUT / "images/train"
LBL_DIR = OUT / "labels/train"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

# Fisheye lens model (equidistant) — matches webui resolver
FISH_W = 800
F_FISH = 400.0 / (math.pi * 0.5)  # px per radian

CLASS_MAP = {
    'person': 0, 'cone': 1, 'box': 2, 'pillar': 3, 'tree': 4,
    'bench': 5, 'shelf': 6, 'forklift': 7, 'platform': 8, 'sign': 9
}


def get_pose():
    try:
        return json.loads(urllib.request.urlopen(f"{WEBUI}/pose.json", timeout=1).read())
    except Exception:
        return None


def get_gt():
    try:
        d = json.loads(urllib.request.urlopen(f"{WEBUI}/ground_truth.json", timeout=2).read())
        objs = []
        for o in d.get('obstacles', []) + d.get('spawned', []):
            cls = o.get('cls', '?')
            if cls not in CLASS_MAP:
                continue
            p = o['pose']
            size = o.get('size', [0.4, 0.4, 1.0])
            objs.append((CLASS_MAP[cls], p[0], p[1], p[2],
                         size[0], size[1], size[2]))
        return objs
    except Exception:
        return []


def fetch_jpeg(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.read()
    except Exception:
        return None


def world_to_fisheye_bbox(cls, ox, oy, oz, sx, sy, sz,
                          rx, ry, rz, qx, qy, qz, qw, is_rear=False):
    """Project a world-frame AABB to fisheye image bbox. Returns
    (cls, cx, cy, w, h) normalized [0..1] or None if not in view."""
    # 8 corners
    corners_w = []
    for dx in (-sx/2, sx/2):
        for dy in (-sy/2, sy/2):
            for dz in (-sz/2, sz/2):
                corners_w.append((ox + dx, oy + dy, oz + dz))
    # World -> body frame
    # body_x_world = R * x + t  (rover pose)
    # Need inverse: body = R^T * (world - t)
    # quaternion to rotation matrix transposed
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    Rt = np.array([
        [1 - 2*(yy + zz),     2*(xy + wz),     2*(xz - wy)],
        [    2*(xy - wz), 1 - 2*(xx + zz),     2*(yz + wx)],
        [    2*(xz + wy),     2*(yz - wx), 1 - 2*(xx + yy)],
    ]).T

    # Camera offset on body
    cam_tx, cam_ty, cam_tz = 0.0, 0.0, 0.3
    yaw_cam = math.pi if is_rear else 0.0
    cy_a, sy_a = math.cos(yaw_cam), math.sin(yaw_cam)

    pts_img = []
    for (wx_, wy_, wz_) in corners_w:
        # body
        bx, by, bz = Rt @ np.array([wx_ - rx, wy_ - ry, wz_ - rz])
        bx -= cam_tx; by -= cam_ty; bz -= cam_tz
        # rotate by cam yaw
        bx2 = cy_a * bx - sy_a * by
        by2 = sy_a * bx + cy_a * by
        bz2 = bz
        # body (X fwd, Y left, Z up) -> optical (Z fwd, X right, Y down)
        zc = bx2
        xc = -by2
        yc = -bz2
        if zc <= 0.1:  # behind cam
            return None
        # Equidistant fisheye projection: theta = acos(zc / |p|), r = F_FISH * theta
        norm = math.sqrt(xc*xc + yc*yc + zc*zc)
        if norm < 1e-6:
            return None
        theta = math.acos(max(-1.0, min(1.0, zc / norm)))
        if theta > 1.5:  # outside 86° cone (180° fisheye)
            return None
        r_xy = math.sqrt(xc*xc + yc*yc) + 1e-9
        r_px = F_FISH * theta
        u = FISH_W * 0.5 + r_px * xc / r_xy
        v = FISH_W * 0.5 + r_px * yc / r_xy
        pts_img.append((u, v))

    if len(pts_img) < 2:
        return None
    us = [p[0] for p in pts_img]
    vs = [p[1] for p in pts_img]
    u_min, u_max = max(0, min(us)), min(FISH_W, max(us))
    v_min, v_max = max(0, min(vs)), min(FISH_W, max(vs))
    if u_max - u_min < 8 or v_max - v_min < 8:
        return None
    cx = (u_min + u_max) / 2 / FISH_W
    cy = (v_min + v_max) / 2 / FISH_W
    w = (u_max - u_min) / FISH_W
    h = (v_max - v_min) / FISH_W
    return (cls, cx, cy, w, h)


def main():
    print(f"[collect] output: {OUT}")
    print(f"[collect] writing 2000 frame-target dataset...")
    seq = 0
    while seq < 2000:
        time.sleep(0.5)
        pose = get_pose()
        if not pose or 'x' not in pose:
            continue
        gt = get_gt()
        if not gt:
            continue
        for cam_name, is_rear in (('front', False), ('rear', True)):
            jpg = fetch_jpeg(f"{WEBUI}/{cam_name}.jpg")
            if not jpg:
                continue
            # MJPEG returns single frame; save it
            fn_img = IMG_DIR / f"{cam_name}_{seq:05d}.jpg"
            fn_lbl = LBL_DIR / f"{cam_name}_{seq:05d}.txt"
            fn_img.write_bytes(jpg)
            # Project GT objects to bboxes
            labels = []
            for (cls_id, ox, oy, oz, sx, sy, sz) in gt:
                b = world_to_fisheye_bbox(cls_id, ox, oy, oz, sx, sy, sz,
                                           pose['x'], pose['y'], pose['z'],
                                           pose['qx'], pose['qy'], pose['qz'], pose['qw'],
                                           is_rear=is_rear)
                if b is not None:
                    labels.append(b)
            if not labels:
                # discard frame with no labeled objects (skip empty)
                fn_img.unlink()
                continue
            with fn_lbl.open('w') as f:
                for (c, cx, cy, w, h) in labels:
                    f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            seq += 1
            if seq % 50 == 0:
                print(f"[collect] {seq}/2000 frames", flush=True)
            if seq >= 2000:
                break
    # Write data.yaml
    yaml = f"""path: {OUT}
train: images/train
val: images/train  # use same for now; later split 80/20
names:
"""
    for name, idx in sorted(CLASS_MAP.items(), key=lambda kv: kv[1]):
        yaml += f"  {idx}: {name}\n"
    (OUT / "data.yaml").write_text(yaml)
    print(f"[collect] DONE. {seq} frames, data.yaml at {OUT}/data.yaml")


if __name__ == "__main__":
    main()
