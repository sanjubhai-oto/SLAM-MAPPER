#!/usr/bin/env python3
"""Drive rover to within 2m of each GT object, dwell 8s, advance.

Lets YOLO-World accumulate enough close-range detections to promote each
object to a locked landmark.
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
    except Exception:
        pass


def main():
    gt = get_json(f'{WEBUI}/ground_truth.json') or {}
    targets = []
    for obj in gt.get('obstacles', []) + gt.get('spawned', []):
        pose = obj.get('pose', [0, 0, 0])
        targets.append((obj['name'], obj.get('cls', '?'), pose[0], pose[1]))
    print(f'[visit] {len(targets)} targets', flush=True)

    for name, cls, tx, ty in targets:
        # Stand-off point: 2.0 m short of object, line from origin
        norm = math.hypot(tx, ty) or 1.0
        sx = tx - 2.0 * tx / norm if abs(tx) > 0.1 or abs(ty) > 0.1 else 0.0
        sy = ty - 2.0 * ty / norm if abs(tx) > 0.1 or abs(ty) > 0.1 else 0.0
        # Keep within yard
        sx = max(-8.5, min(8.5, sx))
        sy = max(-8.5, min(8.5, sy))
        print(f'[visit] -> {name} ({cls}) at GT ({tx:.1f},{ty:.1f}), '
              f'standoff ({sx:.1f},{sy:.1f})', flush=True)
        post(f'{WEBUI}/path/clear')
        post(f'{WEBUI}/map/clear')   # avoid stale planner ghosts
        time.sleep(0.3)
        post(f'{WEBUI}/plan?x={sx:.2f}&y={sy:.2f}')
        t0 = time.time()
        stuck_t0 = None
        last_xy = None
        reached = False
        while time.time() - t0 < 25:
            pj = get_json(f'{WEBUI}/pose.json') or {}
            rx = pj.get('x', 0)
            ry = pj.get('y', 0)
            d = math.hypot(sx - rx, sy - ry)
            if d < 1.2:
                reached = True
                break
            if last_xy is not None:
                if math.hypot(rx - last_xy[0], ry - last_xy[1]) < 0.05:
                    if stuck_t0 is None:
                        stuck_t0 = time.time()
                    elif time.time() - stuck_t0 > 6:
                        print(f'[visit]   stuck, skipping', flush=True)
                        break
                else:
                    stuck_t0 = None
            last_xy = (rx, ry)
            time.sleep(1.0)
        if reached:
            # Dwell 8s to let YOLO accumulate detections at this distance
            print(f'[visit]   reached, dwelling 8s', flush=True)
            time.sleep(8)
    print('[visit] done — saving map', flush=True)
    post(f'{WEBUI}/save')


if __name__ == '__main__':
    main()
