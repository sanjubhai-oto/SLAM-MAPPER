#!/usr/bin/env python3
"""Deterministic boustrophedon (lawnmower) explorer.

Drives the rover through an S-pattern across the 18x18m operating zone so
the SLAM stack gets uniform coverage. Each waypoint is sent via webui /plan;
script waits until reached or 25s timeout, then advances. No frontier
detection — pure systematic coverage. Use when frontier exploration stalls
or when you need reproducible map runs.

Pattern: 8 horizontal stripes 2m apart, alternating east/west sweep.
Endpoints stay clear of walls (max |x| = 7.5, max |y| = 7.5).
"""
import json
import math
import time
import urllib.request


WEBUI = 'http://localhost:8080'


def get_json(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def post(url, timeout=2.0):
    try:
        urllib.request.urlopen(urllib.request.Request(url, method='POST'),
                               timeout=timeout).read()
        return True
    except Exception:
        return False


def waypoints():
    """S-pattern across yard."""
    pts = []
    y_lines = [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
    direction = 1
    for y in y_lines:
        if direction > 0:
            pts.append((-7.5, y))
            pts.append((7.5, y))
        else:
            pts.append((7.5, y))
            pts.append((-7.5, y))
        direction *= -1
    return pts


def drive_to(tx, ty, timeout_s=30):
    print(f'[plan-exp] driving to ({tx:.1f}, {ty:.1f})', flush=True)
    post(f'{WEBUI}/path/clear')
    time.sleep(0.4)
    r = get_json(f'{WEBUI}/plan?x={tx}&y={ty}')  # also fires GET-style for safety
    post(f'{WEBUI}/plan?x={tx}&y={ty}')
    t0 = time.time()
    last_xy = None
    stuck_ticks = 0
    while time.time() - t0 < timeout_s:
        pj = get_json(f'{WEBUI}/pose.json') or {}
        x, y = pj.get('x', 0), pj.get('y', 0)
        d = math.hypot(tx - x, ty - y)
        if d < 0.8:
            print(f'[plan-exp]   reached ({x:.1f},{y:.1f}) d={d:.2f}', flush=True)
            return True
        # Stuck detection
        if last_xy is not None:
            mv = math.hypot(x - last_xy[0], y - last_xy[1])
            if mv < 0.05:
                stuck_ticks += 1
            else:
                stuck_ticks = 0
        last_xy = (x, y)
        if stuck_ticks > 8:  # ~8s no motion
            print(f'[plan-exp]   STUCK at ({x:.1f},{y:.1f}), aborting wp', flush=True)
            return False
        time.sleep(1.0)
    print(f'[plan-exp]   timeout at d={d:.2f}', flush=True)
    return False


def main():
    pts = waypoints()
    print(f'[plan-exp] starting lawnmower, {len(pts)} waypoints', flush=True)
    for i, (tx, ty) in enumerate(pts):
        print(f'[plan-exp] [{i+1}/{len(pts)}]', flush=True)
        drive_to(tx, ty, timeout_s=35)
    print('[plan-exp] lawnmower complete — saving map', flush=True)
    post(f'{WEBUI}/save')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
