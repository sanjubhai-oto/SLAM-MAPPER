#!/usr/bin/env python3.14
"""
Reactive navigator: drives rover toward a target set via the web UI.

  loop:
    - GET /target.json  -> if active, take (tx, ty)
    - subscribe to /world/slam_obstacles/dynamic_pose/info -> rover pose
    - subscribe to /depth_front (R_FLOAT32) -> obstacle distance
    - if obstacle in front (< OBSTACLE_DIST), turn in place toward clearer side
    - else drive toward target with proportional yaw correction
    - on target reached (< ARRIVE_RADIUS), stop and POST /target?clear=1

Publishes wheel commands at 30 Hz to /model/rover_360cam_0/command/motor_speed.

Run while PX4 is DISARMED.
"""
import math
import sys
import threading
import time
import urllib.request

import numpy as np
from gz.transport13 import Node
from gz.msgs10.actuators_pb2 import Actuators
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V


WHEEL_TOPIC = '/model/rover_360cam_0/command/motor_speed'
POSE_TOPIC = '/world/slam_obstacles/dynamic_pose/info'
DEPTH_TOPIC = '/depth_front'
WEBUI_BASE = 'http://localhost:8080'

ARRIVE_RADIUS = 0.6      # m
OBSTACLE_DIST = 1.2      # m, trigger avoidance
FWD_SPEED = 7.0          # wheel rad/s
TURN_SPEED = 4.0         # wheel rad/s

state = {
    'pose': None,        # (x, y, yaw)
    'front_min': None,   # m, min depth in central FOV
}
state_lock = threading.Lock()


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2 * (qw*qz + qx*qy),
                      1 - 2 * (qy*qy + qz*qz))


def on_pose(msg: Pose_V):
    for p in msg.pose:
        if p.name == 'rover_360cam_0':
            yaw = yaw_from_quat(
                p.orientation.x, p.orientation.y,
                p.orientation.z, p.orientation.w)
            with state_lock:
                state['pose'] = (p.position.x, p.position.y, yaw)
            return


def on_depth(msg: Image):
    w, h = msg.width, msg.height
    arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
    # Depth cam at world z~0.4 with 90 deg VFOV: row=h/2 is the horizon,
    # rows > h/2 look down and see the ground. Use rows just above the
    # horizon (h*0.35 .. h*0.50) so we see walls / vertical obstacles only.
    cy0 = int(h * 0.35)
    cy1 = int(h * 0.50)
    cx0 = w // 2 - w // 6
    cx1 = w // 2 + w // 6
    band = arr[cy0:cy1, cx0:cx1]
    valid = band[np.isfinite(band) & (band > 0.1) & (band < 30.0)]
    if valid.size == 0:
        return
    with state_lock:
        state['front_min'] = float(np.percentile(valid, 5))


def fetch_target():
    try:
        with urllib.request.urlopen(f'{WEBUI_BASE}/target.json', timeout=0.5) as r:
            import json
            j = json.loads(r.read().decode())
            if j.get('active'):
                return float(j['x']), float(j['y'])
    except Exception:
        pass
    return None


def clear_target():
    try:
        req = urllib.request.Request(f'{WEBUI_BASE}/target?clear=1', method='POST')
        urllib.request.urlopen(req, timeout=0.5).read()
    except Exception:
        pass


def actuators(left, right):
    msg = Actuators()
    msg.velocity.extend([right, left])
    return msg


def main():
    node = Node()
    pub = node.advertise(WHEEL_TOPIC, Actuators)
    if not pub.valid():
        print(f'[nav] cannot advertise {WHEEL_TOPIC}', file=sys.stderr)
        return 1
    node.subscribe(Pose_V, POSE_TOPIC, on_pose)
    node.subscribe(Image, DEPTH_TOPIC, on_depth)
    print('[nav] subscribed; waiting for first pose...', flush=True)

    # Wait for first pose
    t0 = time.time()
    while True:
        with state_lock:
            if state['pose'] is not None:
                break
        if time.time() - t0 > 10:
            print('[nav] no pose received, abort', file=sys.stderr)
            return 2
        time.sleep(0.1)
    print('[nav] ready', flush=True)

    period = 1.0 / 30.0
    log_t = time.time()
    iters = 0
    try:
        while True:
            iters += 1
            now = time.time()
            if now - log_t > 1.0:
                with state_lock:
                    p = state['pose']; fm = state['front_min']
                print(f'[nav] iters/s={iters/(now-log_t):.0f} pose={p} front_min={fm}',
                      flush=True)
                iters = 0; log_t = now
            tgt = fetch_target()
            if tgt is None:
                pub.publish(actuators(0, 0))
                time.sleep(0.1)
                continue
            tx, ty = tgt
            with state_lock:
                px, py, yaw = state['pose']
                fmin = state['front_min']

            dx = tx - px
            dy = ty - py
            dist = math.hypot(dx, dy)
            if dist < ARRIVE_RADIUS:
                pub.publish(actuators(0, 0))
                print(f'[nav] arrived at ({tx:.2f},{ty:.2f})', flush=True)
                clear_target()
                continue

            heading_to_target = math.atan2(dy, dx)
            yaw_err = math.atan2(math.sin(heading_to_target - yaw),
                                 math.cos(heading_to_target - yaw))

            # Obstacle avoidance: if front blocked, turn in place toward
            # whichever side has more clearance (sign of yaw_err is a fine
            # heuristic here since target side is usually the open side).
            blocked = fmin is not None and fmin < OBSTACLE_DIST
            print(f'[nav] tgt=({tx:.2f},{ty:.2f}) d={dist:.2f} '
                  f'yaw_err={math.degrees(yaw_err):.1f} blocked={blocked} '
                  f'fmin={fmin}', flush=True) if iters % 30 == 0 else None
            if blocked:
                # Spin away from target heading sign (turn left if target is
                # right & blocked, otherwise turn right) — really we just
                # need to clear the path.
                sign = -1.0 if yaw_err > 0 else 1.0
                pub.publish(actuators(sign * TURN_SPEED, -sign * TURN_SPEED))
            else:
                if abs(yaw_err) > math.radians(25):
                    # Rotate in place to align
                    sign = 1.0 if yaw_err > 0 else -1.0
                    pub.publish(actuators(-sign * TURN_SPEED, sign * TURN_SPEED))
                else:
                    # Drive forward, gentle bias on yaw error
                    bias = 0.4 * yaw_err / math.radians(45)  # +- ~0.4 wheel
                    left = FWD_SPEED * (1.0 + bias)
                    right = FWD_SPEED * (1.0 - bias)
                    pub.publish(actuators(left, right))
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(10):
            pub.publish(actuators(0, 0))
            time.sleep(0.03)
        print('[nav] stopped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
