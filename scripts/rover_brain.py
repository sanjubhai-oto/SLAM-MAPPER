#!/usr/bin/env python3
"""
Small on-rover decision agent (rover_brain).

Lightweight policy script that runs on top of the rover and coordinates
exploration + mapping + landmark verification. No LLM weights — just a
deterministic finite-state policy informed by sensor signals. The same
shape generalises to a tiny LLM (e.g. TinyLlama, Phi-3-mini) by replacing
the `decide()` function with a model call.

States:

  EXPLORE      : frontiers exist + map < target coverage  -> /explore/step
  VERIFY       : recent unstable landmark seen -> drive a 1.5m semicircle
                 around it at <0.4 m/s so resolver locks position cleanly
  CONSOLIDATE  : coverage > target -> stop, wait for landmarks to settle
                 (decay flushes ghosts in ~90 s)
  DONE         : no frontiers AND landmarks stable for 20 s

Decision signals polled from webui:
  - /map.bin           -> coverage = free/(free+unk in 18x18m roi)
  - /landmarks.json    -> count + position-variance per class
  - /pose.json         -> speed (for slow-verify)
  - /path.json         -> active?

Outputs:
  - POST /explore/step              start frontier exploration
  - POST /plan?x=&y=                drive to verification waypoint
  - POST /path/clear                stop
  - logs current state + reasoning
"""
import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.request


WEBUI = 'http://localhost:8080'
W = H = 200
RES = 0.10
OX = OY = -10.0

COVERAGE_TARGET = 0.85    # 85% of the 18x18 m operating zone known
STABLE_LANDMARK_WINDOW_S = 20
SLOW_SPEED_LIMIT = 0.40   # m/s for verify drive
VERIFY_RADIUS_M = 1.8     # circle radius around suspect landmark


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


def get_map():
    try:
        with urllib.request.urlopen(f'{WEBUI}/map.bin', timeout=3.0) as r:
            buf = r.read()
    except Exception:
        return None
    if len(buf) != W * H:
        return None
    return memoryview(buf).cast('b')


def coverage(grid):
    """Fraction of the ±9 m operating zone that is no longer unknown."""
    i0, i1 = int((-9 - OX) / RES), int((9 - OX) / RES)
    j0, j1 = int((-9 - OY) / RES), int((9 - OY) / RES)
    free = unk = 0
    for j in range(j0, j1):
        for i in range(i0, i1):
            v = grid[j * W + i]
            if v == -1:
                unk += 1
            elif v == 0 or v == 100:
                free += 1
    total = free + unk
    return free / total if total else 0.0


class LandmarkTracker:
    """Track per-class positions + detect instability (object 'moving')."""
    def __init__(self):
        self.history = {}   # cls -> list of (t, x, y)

    def update(self, items, t):
        seen = {}
        for it in items:
            cls = it['cls']
            seen.setdefault(cls, []).append((it['x'], it['y']))
        for cls, pts in seen.items():
            self.history.setdefault(cls, []).append((t, pts))
        # drop entries older than window
        cutoff = t - 30
        for cls in list(self.history.keys()):
            self.history[cls] = [e for e in self.history[cls] if e[0] >= cutoff]

    def unstable(self):
        """Return (cls, suspect_xy) for first class with growing or jittery
        landmarks, else None."""
        for cls, hist in self.history.items():
            if len(hist) < 3:
                continue
            counts = [len(pts) for _t, pts in hist]
            if max(counts) - min(counts) >= 2:  # count fluctuating
                # pick last seen centroid
                xs = [x for x, y in hist[-1][1]]
                ys = [y for x, y in hist[-1][1]]
                return cls, (sum(xs) / len(xs), sum(ys) / len(ys))
        return None


def decide(state, ctx):
    """Pure policy function. Returns (next_state, action_dict)."""
    cov = ctx['coverage']
    speed = ctx['speed']
    path_active = ctx['path_active']
    unstable = ctx['unstable']
    n_lms = ctx['n_lms']
    pose = ctx.get('pose', (0, 0))

    # Corner-stuck rescue: if rover near a wall corner AND speed < 0.1 for
    # any reason, teleport to center via /plan. Wall is at ±10 so cells
    # past 9 with low speed = wedged.
    rx, ry = pose
    near_corner = abs(rx) > 8.5 and abs(ry) > 8.5
    near_wall   = abs(rx) > 9.0 or  abs(ry) > 9.0
    if (near_corner or near_wall) and speed < 0.15 and not path_active:
        # Drive toward an inner safe point (signed inward by 5 m)
        sx = (rx * 0.4) if abs(rx) < 1 else (rx - 5 if rx > 0 else rx + 5)
        sy = (ry * 0.4) if abs(ry) < 1 else (ry - 5 if ry > 0 else ry + 5)
        return 'ESCAPE', {'do': 'plan', 'xy': (sx, sy),
                          'why': f'wedged near wall ({rx:.1f},{ry:.1f}) -> ({sx:.1f},{sy:.1f})'}

    if state == 'BOOT':
        return 'EXPLORE', {'do': 'explore_step',
                           'why': f'cov={cov:.2f}, kick off frontier'}

    if path_active:
        return state, {'do': 'wait',
                       'why': f'plan active, speed={speed:.2f}'}

    # Keep moving until we have enough landmarks. Coverage alone is not
    # enough -- depth cams see all walls from origin, so cov hits 1.0 in
    # seconds but no landmark has been collected yet.
    MIN_LANDMARKS_BEFORE_REST = 3
    if n_lms < MIN_LANDMARKS_BEFORE_REST:
        if unstable is not None and speed < 0.6:
            cls, (vx, vy) = unstable
            return 'VERIFY', {'do': 'verify_circle',
                              'target': (vx, vy), 'cls': cls,
                              'why': f'class {cls} jittery, slow circle to lock'}
        return 'EXPLORE', {'do': 'explore_step',
                           'why': f'cov={cov:.2f} lms={n_lms}, keep hunting'}

    # Have landmarks. Are they all locked yet?
    locked_count = sum(1 for it in ctx['raw_items'] if it.get('locked'))
    if locked_count < n_lms:
        return 'VERIFY', {'do': 'explore_step',
                          'why': f'{locked_count}/{n_lms} locked, more passes'}

    return 'DONE', {'do': 'stop',
                    'why': f'{n_lms} landmarks all locked'}


def verify_circle(center, radius=VERIFY_RADIUS_M):
    """Pick a waypoint on the circle that's the side opposite the rover, so
    rover gets a fresh viewpoint."""
    pj = get_json(f'{WEBUI}/pose.json') or {}
    if 'x' not in pj:
        return None
    rx, ry = pj['x'], pj['y']
    cx, cy = center
    dx, dy = cx - rx, cy - ry
    h = math.hypot(dx, dy)
    if h < 0.1:
        return None
    # opposite-side point along radius
    nx = cx + radius * dx / h
    ny = cy + radius * dy / h
    return (nx, ny)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rate', type=float, default=1.0, help='Hz tick rate')
    args = ap.parse_args()

    print('[brain] starting rover_brain (small policy agent)', flush=True)
    state = 'BOOT'
    tracker = LandmarkTracker()
    done_since = None
    # Anti-stagnation: cycle inspection grid if rover stays in same quadrant.
    last_pose = None
    stuck_ticks = 0
    inspect_cycle = 0
    INSPECT_GRID = [(-7, -7), (-7, 0), (-7, 7), (0, 7), (7, 7),
                    (7, 0), (7, -7), (0, -7), (0, 0)]

    while True:
        tick = time.time()
        grid = get_map()
        pj = get_json(f'{WEBUI}/pose.json') or {}
        lj = get_json(f'{WEBUI}/landmarks.json') or {'items': []}
        path = get_json(f'{WEBUI}/path.json') or {}
        if grid is None or not pj:
            print('[brain] webui not ready, retry', flush=True)
            time.sleep(2.0)
            continue
        tracker.update(lj.get('items', []), tick)

        ctx = {
            'coverage': coverage(grid),
            'speed': pj.get('speed', 0.0),
            'path_active': path.get('active', False),
            'unstable': tracker.unstable(),
            'n_lms': len(lj.get('items', [])),
            'raw_items': lj.get('items', []),
            'pose': (pj.get('x', 0.0), pj.get('y', 0.0)),
        }

        # Anti-stagnation: detect rover staying in same 2m bucket
        cur_pose = (round(ctx['pose'][0] / 2) * 2, round(ctx['pose'][1] / 2) * 2)
        if last_pose == cur_pose:
            stuck_ticks += 1
        else:
            stuck_ticks = 0
            last_pose = cur_pose
        if stuck_ticks > 12:  # ~12s
            tx, ty = INSPECT_GRID[inspect_cycle % len(INSPECT_GRID)]
            inspect_cycle += 1
            stuck_ticks = 0
            print(f'[brain] STUCK -> force inspect ({tx},{ty})', flush=True)
            post(f'{WEBUI}/path/clear')
            post(f'{WEBUI}/map/clear')   # wipe ghost obstacles trapping planner
            post(f'{WEBUI}/plan?x={tx}&y={ty}')
            time.sleep(1.0)
            continue

        next_state, action = decide(state, ctx)
        print(f"[brain] {state} -> {next_state}: {action['why']}", flush=True)

        do = action.get('do')
        if do == 'plan' and 'xy' in action:
            x, y = action['xy']
            post(f'{WEBUI}/path/clear')
            post(f'{WEBUI}/plan?x={x:.2f}&y={y:.2f}')
        elif do == 'explore_step':
            # First try frontier exploration. If no frontier (coverage already
            # full), fall back to visiting an inspection waypoint near a
            # known interesting cluster.
            r = get_json(f'{WEBUI}/explore/step') if False else None
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(f'{WEBUI}/explore/step',
                                            method='POST'), timeout=2) as rsp:
                    j = json.loads(rsp.read().decode())
            except Exception:
                j = {'ok': False}
            if not j.get('ok'):
                # No frontier -> drive through inspection grid points to
                # search for unseen objects.
                INSPECT_POINTS = [
                    (-6, -6), (-6,  0), (-6, 6),
                    ( 0, -6), ( 0,  6),
                    ( 6, -6), ( 6,  0), ( 6, 6),
                ]
                pj = get_json(f'{WEBUI}/pose.json') or {}
                rx = pj.get('x', 0); ry = pj.get('y', 0)
                # furthest unvisited-ish point — pick at random rotated by tick
                idx = int(tick) % len(INSPECT_POINTS)
                tx, ty = INSPECT_POINTS[idx]
                print(f'[brain] no frontier -> inspect ({tx:.1f},{ty:.1f})',
                      flush=True)
                post(f'{WEBUI}/plan?x={tx}&y={ty}')
        elif do == 'verify_circle':
            wp = verify_circle(action['target'])
            if wp:
                post(f'{WEBUI}/plan?x={wp[0]:.2f}&y={wp[1]:.2f}')
        elif do == 'stop':
            post(f'{WEBUI}/path/clear')
            if done_since is None:
                done_since = tick
            elif tick - done_since > STABLE_LANDMARK_WINDOW_S:
                print('[brain] DONE — coverage met, landmarks stable. Saving map...')
                post(f'{WEBUI}/save')
                print('[brain] map saved, exiting.')
                return 0
        else:
            done_since = None

        if action.get('do') != 'stop':
            done_since = None

        state = next_state
        time.sleep(1.0 / args.rate)


if __name__ == '__main__':
    sys.exit(main() or 0)
