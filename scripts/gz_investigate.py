#!/usr/bin/env python3.14
"""
Autonomous investigation behavior.

Picks a target class from the webui (POST /investigate?cls=car), then runs a
state machine:

  IDLE     : no class set, wheels stopped
  SEARCH   : spin in place; periodically nudge forward to escape dead spots
  APPROACH : center YOLO bbox horizontally with proportional yaw, drive
             forward at speed scaled by remaining distance (bbox size).
             Transition to CIRCLE once bbox covers > APPROACH_STOP_RATIO of
             frame width.
  CIRCLE   : orbit target with constant arc; bearing tweak keeps target near
             image edge. If target lost too long, fall back to SEARCH.

Run while PX4 is DISARMED. Publishes
  /model/rover_360cam_0/command/motor_speed (gz.msgs.Actuators)
"""
import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.request

from gz.transport13 import Node
from gz.msgs10.actuators_pb2 import Actuators


WHEEL_TOPIC = '/model/rover_360cam_0/command/motor_speed'
WEBUI = 'http://localhost:8080'
DETECT = 'http://localhost:8081'

FWD_SPEED = 6.0
TURN_SPEED = 4.0
APPROACH_STOP_RATIO = 0.30   # start follow when bbox ~30 % of frame
FOLLOW_GOAL_LOW = 0.25       # below -> drive forward
FOLLOW_GOAL_HIGH = 0.40      # above -> reverse
LOST_TIMEOUT = 4.0
RATE_HZ = 50.0


def http_get_json(url, timeout=0.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def push_state(s):
    try:
        url = f'{WEBUI}/investigate?state={urllib.parse.quote(s)}'
        urllib.request.urlopen(urllib.request.Request(url, method='POST'),
                               timeout=0.5).read()
    except Exception:
        pass


def best_detection_for(cls, dets, locked_id=None):
    """Pick best detection matching `cls`. If `locked_id` is set, prefer the
    detection with that track ID (object permanence); fall back to highest
    confidence match if the locked ID isn't present this frame."""
    if not dets:
        return None
    matches = [d for d in dets if d.get('cls') == cls]
    if not matches:
        return None
    if locked_id is not None:
        same = [d for d in matches if d.get('id') == locked_id]
        if same:
            return max(same, key=lambda d: d['conf'])
    return max(matches, key=lambda d: d['conf'])


def actuators(left, right):
    msg = Actuators()
    msg.velocity.extend([right, left])  # idx 0=right, 1=left per SDF plugin
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame-w', type=int, default=800)
    args = ap.parse_args()

    node = Node()
    pub = node.advertise(WHEEL_TOPIC, Actuators)
    if not pub.valid():
        print(f'[invest] cannot advertise {WHEEL_TOPIC}', file=sys.stderr)
        return 1
    print('[invest] ready', flush=True)

    period = 1.0 / RATE_HZ
    state = 'idle'
    last_seen = 0.0
    search_started = 0.0
    locked_id = None  # ByteTrack ID we're following
    explore_phase = 'fwd'
    explore_end = time.time() + 4.0
    EXPLORE_FWD_SPEED = 7.0
    EXPLORE_TURN_SPEED = 4.5
    OBSTACLE_DIST = 1.5

    try:
        while True:
            # ===== PATH FOLLOWING (top priority) =====
            path = http_get_json(f'{WEBUI}/path.json') or {}
            if path.get('active') and path.get('waypoints'):
                # Obstacle gate: stop forward + clear path if front_clear < 0.7m
                obst = http_get_json(f'{WEBUI}/obstacles.json') or {}
                front_d = float(obst.get('front', 99.0))
                if front_d < 0.7:
                    print(f'[invest] OBSTACLE front={front_d:.2f}m, clear path + back up',
                          flush=True)
                    pub.publish(actuators(-4.0, -4.0))  # reverse
                    time.sleep(0.4)
                    try:
                        urllib.request.urlopen(urllib.request.Request(
                            f'{WEBUI}/path/clear', method='POST'), timeout=0.3).read()
                    except Exception:
                        pass
                    time.sleep(0.2)
                    continue
                wpts = path['waypoints']
                cursor = int(path.get('cursor', 0))
                # Fetch pose from webui
                pj = http_get_json(f'{WEBUI}/pose.json') or {}
                if 'x' not in pj:
                    time.sleep(0.1); continue
                px, py = pj['x'], pj['y']
                qx,qy,qz,qw = pj['qx'], pj['qy'], pj['qz'], pj['qw']
                yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
                # Advance cursor: snap to NEAREST waypoint along remaining
                # path. Pure pursuit then aims for lookahead beyond it.
                best_d = 1e9
                best_k = cursor
                for k in range(cursor, min(cursor + 30, len(wpts))):
                    wx, wy = wpts[k]
                    d = math.hypot(wx - px, wy - py)
                    if d < best_d:
                        best_d = d; best_k = k
                cursor = best_k
                # Pure-pursuit: lookahead scales with speed, longer = smoother
                LOOKAHEAD = 2.5
                look_x, look_y = wpts[cursor]
                for k in range(cursor, len(wpts)):
                    xk, yk = wpts[k]
                    if math.hypot(xk - px, yk - py) >= LOOKAHEAD:
                        look_x, look_y = xk, yk; break
                bearing = math.atan2(look_y - py, look_x - px) - yaw
                bearing = math.atan2(math.sin(bearing), math.cos(bearing))
                FWD = 11.0                       # wheel rad/s -> ~1.1 m/s
                # Only cut speed when actually mis-aligned (>30 deg)
                speed_scale = 1.0
                if abs(bearing) > math.radians(30):
                    speed_scale = max(0.35, math.cos(bearing))
                fwd = FWD * speed_scale
                # spread<0 = turn left, spread>0 = turn right.
                # bearing>0 means target left -> spread negative.
                spread = -1.8 * bearing
                pub.publish(actuators(fwd + spread, fwd - spread))
                # tell webui new cursor
                try:
                    import urllib.request
                    urllib.request.urlopen(urllib.request.Request(
                        f'{WEBUI}/path/advance?cursor={cursor}',
                        method='POST'), timeout=0.3).read()
                except Exception:
                    pass
                gx, gy = wpts[-1]
                if math.hypot(gx - px, gy - py) < 0.5:
                    print(f'[invest] reached path goal ({gx:.2f},{gy:.2f})',
                          flush=True)
                    try:
                        import urllib.request
                        urllib.request.urlopen(urllib.request.Request(
                            f'{WEBUI}/path/clear', method='POST'), timeout=0.3).read()
                    except Exception:
                        pass
                if state != 'goto':
                    state = 'goto'
                    push_state('goto')
                time.sleep(period); continue

            inv = http_get_json(f'{WEBUI}/investigate.json') or {}
            if not inv.get('active'):
                # Autonomous EXPLORE pattern (lawnmower + obstacle avoidance).
                obst = http_get_json(f'{WEBUI}/obstacles.json') or {}
                front_clear = float(obst.get('front', 99.0))
                now = time.time()
                if explore_phase == 'fwd' and front_clear < OBSTACLE_DIST:
                    explore_phase = 'turn'; explore_end = now + 2.0
                elif now > explore_end:
                    if explore_phase == 'fwd':
                        explore_phase = 'turn'; explore_end = now + 2.0
                    else:
                        explore_phase = 'fwd'; explore_end = now + 4.0
                if explore_phase == 'fwd':
                    speed = EXPLORE_FWD_SPEED
                    if front_clear < 3.0:
                        speed *= max(0.3,
                            (front_clear-OBSTACLE_DIST)/(3.0-OBSTACLE_DIST))
                    pub.publish(actuators(speed, speed))
                else:
                    pub.publish(actuators(-EXPLORE_TURN_SPEED, EXPLORE_TURN_SPEED))
                if state != 'explore':
                    state = 'explore'
                    push_state('explore')
                    print('[invest] -> explore', flush=True)
                time.sleep(period)
                continue

            cls = inv.get('cls', '')
            df = http_get_json(f'{DETECT}/front/detections.json') or {}
            dr = http_get_json(f'{DETECT}/rear/detections.json') or {}
            d_front = best_detection_for(cls, df.get('detections', []), locked_id)
            d_rear = best_detection_for(cls, dr.get('detections', []), locked_id)
            d = d_front
            now = time.time()
            if d is not None:
                last_seen = now

            if state == 'idle':
                state = 'search'
                search_started = now
                push_state('search')
                print(f'[invest] hunting {cls!r} -> search', flush=True)

            if state == 'search' and d is not None:
                state = 'approach'
                locked_id = d.get('id') if d.get('id', -1) >= 0 else None
                push_state('approach')
                print(f'[invest] {cls}#{locked_id} acquired -> approach',
                      flush=True)

            if state == 'search':
                spin_phase = (now - search_started) % 8.0
                if d_rear is not None and d_front is None:
                    pub.publish(actuators(-TURN_SPEED * 1.3, TURN_SPEED * 1.3))
                elif spin_phase < 6.0:
                    pub.publish(actuators(-TURN_SPEED, TURN_SPEED))
                else:
                    pub.publish(actuators(FWD_SPEED * 0.6, FWD_SPEED * 0.6))

            elif state == 'approach':
                if now - last_seen > LOST_TIMEOUT:
                    state = 'search'
                    search_started = now
                    push_state('search')
                    print('[invest] lost -> search', flush=True)
                elif d is None:
                    pub.publish(actuators(FWD_SPEED * 0.4, FWD_SPEED * 0.4))
                else:
                    x, _y, w, _h = d['box']
                    cx = x + w / 2.0
                    bearing = (cx - args.frame_w / 2.0) / (args.frame_w / 2.0)
                    yaw_cmd = -1.5 * bearing
                    size_ratio = w / float(args.frame_w)
                    if size_ratio > APPROACH_STOP_RATIO:
                        state = 'follow'
                        push_state('follow')
                        print(f'[invest] close (size={size_ratio:.2f}) -> follow',
                              flush=True)
                    else:
                        fwd = FWD_SPEED * max(0.2, 1.0 - 1.5 * size_ratio)
                        spread = yaw_cmd * 1.5
                        pub.publish(actuators(fwd - spread, fwd + spread))

            elif state == 'follow':
                if now - last_seen > LOST_TIMEOUT * 1.5:
                    state = 'search'
                    search_started = now
                    locked_id = None
                    push_state('search')
                    print('[invest] follow lost -> search', flush=True)
                elif d is None:
                    pub.publish(actuators(0, 0))
                else:
                    x, _y, w, _h = d['box']
                    cx = x + w / 2.0
                    bearing = (cx - args.frame_w / 2.0) / (args.frame_w / 2.0)
                    size_ratio = w / float(args.frame_w)
                    yaw_cmd = -2.0 * bearing
                    spread = yaw_cmd * 1.5
                    # Distance band -- choose forward / hold / reverse
                    if size_ratio < FOLLOW_GOAL_LOW:
                        # Too far -> drive forward (slow down as we get closer)
                        gap = FOLLOW_GOAL_LOW - size_ratio
                        fwd = FWD_SPEED * min(1.0, 2.5 * gap + 0.3)
                    elif size_ratio > FOLLOW_GOAL_HIGH:
                        # Too close -> reverse
                        gap = size_ratio - FOLLOW_GOAL_HIGH
                        fwd = -FWD_SPEED * min(0.8, 2.5 * gap + 0.2)
                    else:
                        # In sweet spot -> hold, just yaw correct
                        fwd = 0.0
                    pub.publish(actuators(fwd - spread, fwd + spread))

            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(10):
            pub.publish(actuators(0, 0))
            time.sleep(0.03)
        print('[invest] stopped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
