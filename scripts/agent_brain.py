#!/usr/bin/env python3
"""Agentic AI brain for slam_rover — runs decisions, not a fixed program.

Watches state every 2 s. Picks high-level intent based on observed conditions.
States: BOOT → EXPLORE_FRONTIER → INSPECT_OBJECT → DETOUR_WALL →
        REVISIT_LANDMARK → CONSOLIDATE → SAVE_AND_REST

Decision rules (priority order):
  1. STUCK (no motion 8s + obstacle <0.6m): BACKUP + replan
  2. NEAR_UNLOCKED_LANDMARK (within 3m): orbit it for parallax → LOCK
  3. WALL_TOO_CLOSE (front<1.0m): DETOUR away
  4. COVERAGE_HIGH + LMS_LOCKED>10: CONSOLIDATE + save
  5. NO_FRONTIER_LEFT: drive inspection grid
  6. default: trigger frontier explorer pick
"""
import json, math, time, urllib.request, urllib.parse


WEBUI = 'http://localhost:8080'


def gj(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def post(url, timeout=2):
    try:
        urllib.request.urlopen(urllib.request.Request(url, method='POST'),
                                timeout=timeout).read()
        return True
    except Exception:
        return False


def main():
    print('[agent] starting agentic brain', flush=True)
    last_xy = None
    stuck_count = 0
    state = 'BOOT'
    target = None
    last_decision_t = 0

    while True:
        time.sleep(2.0)
        pj = gj(f'{WEBUI}/pose.json') or {}
        if 'x' not in pj:
            continue
        ob = gj(f'{WEBUI}/obstacles.json') or {}
        lm = gj(f'{WEBUI}/landmarks.json') or {'items': []}
        pa = gj(f'{WEBUI}/path.json') or {}

        x, y = pj['x'], pj['y']
        spd = pj.get('speed', 0)
        fcl = float(ob.get('front', 99.0))
        items = lm.get('items', [])
        locked = sum(1 for l in items if l.get('locked'))
        path_active = pa.get('active', False)
        now = time.time()

        # Stuck detection
        if last_xy is not None and math.hypot(x - last_xy[0], y - last_xy[1]) < 0.1:
            stuck_count += 1
        else:
            stuck_count = 0
        last_xy = (x, y)

        decision = None
        reason = ''

        # Rule 1: STUCK
        if stuck_count > 4 and spd < 0.05:
            # Pick random escape waypoint within yard
            import random
            tx = random.uniform(-7, 7)
            ty = random.uniform(-7, 7)
            decision = ('plan', tx, ty)
            reason = f'STUCK {stuck_count}t, escape to ({tx:.1f},{ty:.1f})'
            stuck_count = 0
        # Rule 2: Wall too close
        elif fcl < 0.8 and path_active:
            tx = -x * 0.6  # point inward
            ty = -y * 0.6
            decision = ('plan', tx, ty)
            reason = f'WALL_NEAR front={fcl:.2f}, retreat to ({tx:.1f},{ty:.1f})'
        # Rule 3: Orbit nearest unlocked landmark for parallax
        elif now - last_decision_t > 25:
            unlocked = [l for l in items if not l.get('locked')]
            if unlocked:
                # Pick the closest unlocked
                unlocked.sort(key=lambda l: math.hypot(l['x']-x, l['y']-y))
                t = unlocked[0]
                # Offset 1.5m perpendicular for parallax
                dx, dy = t['x'] - x, t['y'] - y
                d = math.hypot(dx, dy) or 1
                # perpendicular offset
                tx = t['x'] + 1.2 * (-dy / d)
                ty = t['y'] + 1.2 * (dx / d)
                tx = max(-8.5, min(8.5, tx))
                ty = max(-8.5, min(8.5, ty))
                decision = ('plan', tx, ty)
                reason = f'ORBIT {t["cls"]}@({t["x"]:.1f},{t["y"]:.1f}) for parallax'
                last_decision_t = now
            elif locked >= 15:
                decision = ('save', 0, 0)
                reason = f'DONE — {locked} locked landmarks, saving'
                last_decision_t = now

        if decision:
            cmd, tx, ty = decision
            print(f'[agent] {state} -> {reason}', flush=True)
            if cmd == 'plan':
                post(f'{WEBUI}/path/clear')
                post(f'{WEBUI}/plan?x={tx:.2f}&y={ty:.2f}')
            elif cmd == 'save':
                post(f'{WEBUI}/save')
                print('[agent] saved map. continuing observation...', flush=True)
                last_decision_t = now + 120  # cooldown


if __name__ == '__main__':
    main()
