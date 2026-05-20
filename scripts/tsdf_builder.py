#!/usr/bin/env python3.11
"""
Open3D ScalableTSDFVolume fed from the slam_rover webui HTTP endpoints.

This avoids any direct gz_transport import (incompatible with cv2 and open3d
on macOS arm64 in the same process). All input comes through the C++ webui:

  GET /depth_front.bin   float32 row-major HxW  (X-Width / X-Height headers)
  GET /rgb_front.bin     uint8   row-major HxWx3
  GET /pose.json         {x,y,z,qx,qy,qz,qw}

At ~5 Hz:
  - fetch depth + rgb + pose
  - build o3d.geometry.RGBDImage
  - compute extrinsic (world->camera) from pose
  - volume.integrate(...)

Every 60 s (or on SIGUSR1):
  - mesh = volume.extract_triangle_mesh()
  - write /tmp/slam_rover_mesh.ply  (served at webui /mesh.ply)

Run:
    /tmp/tsdf_env/bin/python3.11 tsdf_builder.py
"""
import argparse
import io
import json
import math
import sys
import time
import urllib.request

import numpy as np
import open3d as o3d


WEBUI = 'http://localhost:8080'
OUT_PLY = '/tmp/slam_rover_mesh.ply'


def fetch_bytes(path):
    with urllib.request.urlopen(f'{WEBUI}{path}', timeout=2.0) as r:
        return r.read(), dict(r.headers)


def fetch_json(path):
    with urllib.request.urlopen(f'{WEBUI}{path}', timeout=2.0) as r:
        return json.loads(r.read().decode())


def quat_to_R(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) -> 3x3 rotation."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),      1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ])


def build_intrinsic(w, h, hfov_rad):
    fx = (0.5 * w) / math.tan(0.5 * hfov_rad)
    fy = fx
    cx = 0.5 * w
    cy = 0.5 * h
    return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voxel', type=float, default=0.03,
                    help='TSDF voxel size in metres (default 3 cm)')
    ap.add_argument('--sdf-trunc', type=float, default=0.10)
    ap.add_argument('--integrate-hz', type=float, default=5.0)
    ap.add_argument('--mesh-interval', type=float, default=15.0,
                    help='seconds between mesh exports')
    ap.add_argument('--max-depth', type=float, default=10.0)
    args = ap.parse_args()

    print(f'[tsdf] voxel={args.voxel} sdf_trunc={args.sdf_trunc} '
          f'rate={args.integrate_hz} Hz', flush=True)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    intrinsic = None
    period = 1.0 / args.integrate_hz
    last_mesh_t = time.time()
    n_frames = 0

    # Camera pose in base_link (matches rover_360cam SDF: <pose>0 0 0.30 0 0 0</pose>
    # for depth_front; optical frame is +Z fwd / +X right / +Y down)
    CAM_X, CAM_Y, CAM_Z = 0.0, 0.0, 0.30

    while True:
        t0 = time.time()
        try:
            depth_buf, dh = fetch_bytes('/depth_front.bin')
            w = int(dh.get('X-Width', '0'))
            h = int(dh.get('X-Height', '0'))
            hfov = float(dh.get('X-Hfov', dh.get('X-HFov', '1.5708')))
            if w == 0 or len(depth_buf) != 4 * w * h:
                time.sleep(period); continue
            depth = np.frombuffer(depth_buf, dtype=np.float32).reshape(h, w)
            depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
            depth = np.where(depth > args.max_depth, 0.0, depth)

            rgb_buf, rh = fetch_bytes('/rgb_front.bin')
            rw = int(rh.get('X-Width', '0'))
            rhh = int(rh.get('X-Height', '0'))
            if rw == 0 or len(rgb_buf) != 3 * rw * rhh:
                time.sleep(period); continue
            rgb = np.frombuffer(rgb_buf, dtype=np.uint8).reshape(rhh, rw, 3)

            # Resize RGB to match depth if different sizes (fisheye 800 vs
            # depth 480). Cheap nearest sampling.
            if rw != w or rhh != h:
                from PIL import Image as PI
                img = PI.fromarray(rgb).resize((w, h), PI.NEAREST)
                rgb = np.asarray(img)

            pose = fetch_json('/pose.json')

            if intrinsic is None:
                intrinsic = build_intrinsic(w, h, hfov)
                print(f'[tsdf] intrinsic w={w} h={h} hfov={math.degrees(hfov):.0f}deg fx={(0.5*w)/math.tan(0.5*hfov):.1f}',
                      flush=True)

            # Build RGBDImage. Open3D's convert_rgb_to_intensity=False keeps colour.
            color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb))
            depth_o3d = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0,
                depth_trunc=args.max_depth,
                convert_rgb_to_intensity=False,
            )

            # Extrinsic = world->camera transform.
            #   T_world_cam = T_world_base * T_base_cam_optical
            # base_link frame: x fwd, y left, z up. Camera optical: x right, y down, z fwd.
            # Rotation base->optical:
            #   x_opt =  -y_base
            #   y_opt =  -z_base
            #   z_opt =   x_base
            R_b2o = np.array([[0, -1, 0],
                              [0, 0, -1],
                              [1, 0, 0]], dtype=np.float64)
            T_base_cam = np.eye(4)
            T_base_cam[:3, :3] = R_b2o
            T_base_cam[:3,  3] = [CAM_X, CAM_Y, CAM_Z]

            R_w2b = quat_to_R(pose['qx'], pose['qy'], pose['qz'], pose['qw'])
            T_world_base = np.eye(4)
            T_world_base[:3, :3] = R_w2b
            T_world_base[:3,  3] = [pose['x'], pose['y'], pose['z']]

            T_world_cam = T_world_base @ T_base_cam
            extrinsic = np.linalg.inv(T_world_cam)  # Open3D wants world->cam

            volume.integrate(rgbd, intrinsic, extrinsic)
            n_frames += 1

        except Exception as e:
            print(f'[tsdf] integrate err: {e}', flush=True)

        # Export mesh periodically
        if time.time() - last_mesh_t > args.mesh_interval:
            try:
                mesh = volume.extract_triangle_mesh()
                mesh.compute_vertex_normals()
                o3d.io.write_triangle_mesh(OUT_PLY, mesh,
                                            write_ascii=False,
                                            compressed=False)
                print(f'[tsdf] wrote {OUT_PLY}  verts={len(mesh.vertices)} '
                      f'tris={len(mesh.triangles)} frames={n_frames}',
                      flush=True)
            except Exception as e:
                print(f'[tsdf] mesh err: {e}', flush=True)
            last_mesh_t = time.time()

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


if __name__ == '__main__':
    sys.exit(main() or 0)
