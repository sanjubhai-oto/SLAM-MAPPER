#!/usr/bin/env python3
"""
Frontier-exploration agent for the slam_rover.

Reads the occupancy grid from the webui (/map.bin), finds the boundary
between known-free space and unknown space ("frontiers"), clusters them,
picks the most informative frontier, and POSTs /plan to drive the rover
there. Keeps doing this until coverage stops growing.

This replaces the dumb preset laps (square/oval) with a goal-directed
explorer aimed at fully mapping the area for photogrammetry-quality output.

Algorithm (Yamauchi 1997, simplified):

  loop:
    grid = fetch /map.bin                          ( -1 unk, 0 free, 100 occ )
    rover = fetch /pose.json
    if active plan -> wait
    frontiers = cells where (cell==free) AND (any 4-neighbour == unknown)
    clusters = connected_components(frontiers, 8-conn)
    score each cluster:
        gain = unknown cells within 1m of centroid    -- info-gain proxy
        dist = euclidean(rover, centroid)             -- travel cost
        utility = gain - 1.5 * dist
    target = centroid of max-utility cluster
    POST /plan?x=tx&y=ty
    wait until plan inactive or 30s timeout
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
MIN_CLUSTER = 8          # cells, ignore noise frontiers below this size
COVERAGE_DONE = 4        # stop after N straight ticks with no good frontier
WAIT_PLAN_S = 30
SLEEP_BETWEEN_S = 1.0


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


def get_map():
    try:
        with urllib.request.urlopen(f'{WEBUI}/map.bin', timeout=3.0) as r:
            buf = r.read()
    except Exception:
        return None
    if len(buf) != W * H:
        return None
    # Convert to signed grid
    return memoryview(buf).cast('b')


def to_xy(i, j):
    return (OX + (i + 0.5) * RES, OY + (j + 0.5) * RES)


def find_frontiers(grid):
    """Return list of (i, j) cells that are free and adjacent to unknown."""
    fronts = []
    for j in range(1, H - 1):
        for i in range(1, W - 1):
            if grid[j * W + i] != 0:
                continue
            # any 4-neighbour unknown
            if (grid[(j - 1) * W + i] == -1 or grid[(j + 1) * W + i] == -1 or
                grid[j * W + (i - 1)] == -1 or grid[j * W + (i + 1)] == -1):
                fronts.append((i, j))
    return fronts


def cluster(fronts):
    """8-connected components over a list of (i, j) cells. Returns list of
    clusters, each a list of (i, j)."""
    s = set(fronts)
    out = []
    visited = set()
    for p in fronts:
        if p in visited:
            continue
        stack = [p]
        comp = []
        while stack:
            q = stack.pop()
            if q in visited:
                continue
            visited.add(q)
            comp.append(q)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    nb = (q[0] + di, q[1] + dj)
                    if nb in s and nb not in visited:
                        stack.append(nb)
        if len(comp) >= MIN_CLUSTER:
            out.append(comp)
    return out


def cluster_score(comp, rover, grid):
    # centroid (cell), gain = number of unknown cells within 1m (10 cells)
    cx = sum(p[0] for p in comp) / len(comp)
    cy = sum(p[1] for p in comp) / len(comp)
    wx, wy = OX + cx * RES, OY + cy * RES
    rx, ry = rover
    dist = math.hypot(wx - rx, wy - ry)
    # gain: scan 10-cell box around centroid
    i0 = max(0, int(cx) - 10)
    i1 = min(W, int(cx) + 10)
    j0 = max(0, int(cy) - 10)
    j1 = min(H, int(cy) + 10)
    gain = 0
    for j in range(j0, j1):
        for i in range(i0, i1):
            if grid[j * W + i] == -1:
                gain += 1
    utility = gain - 1.5 * (dist / RES)
    return utility, (wx, wy), gain, dist


def fetch_pose():
    j = get_json(f'{WEBUI}/pose.json')
    if j is None:
        return None
    return (j['x'], j['y'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true',
                    help='pick a single frontier then exit')
    args = ap.parse_args()

    print('[explore-agent] starting frontier explorer', flush=True)
    no_good_in_a_row = 0
    while True:
        # If a plan is already active, just wait for it
        pj = get_json(f'{WEBUI}/path.json') or {}
        if pj.get('active'):
            time.sleep(SLEEP_BETWEEN_S)
            continue

        grid = get_map()
        rover = fetch_pose()
        if grid is None or rover is None:
            print('[explore-agent] no map / pose, retrying',
                  flush=True)
            time.sleep(2.0)
            continue

        fronts = find_frontiers(grid)
        if not fronts:
            print('[explore-agent] no frontiers — coverage complete')
            return 0
        clusters = cluster(fronts)
        if not clusters:
            no_good_in_a_row += 1
            if no_good_in_a_row >= COVERAGE_DONE:
                print('[explore-agent] no significant frontiers anywhere — done')
                return 0
            print(f'[explore-agent] no sizable cluster yet ({no_good_in_a_row})',
                  flush=True)
            time.sleep(2.0)
            continue
        no_good_in_a_row = 0

        scored = [(*cluster_score(c, rover, grid), len(c)) for c in clusters]
        scored.sort(key=lambda s: -s[0])  # highest utility first
        utility, (tx, ty), gain, dist, sz = scored[0]
        print(f'[explore-agent] target ({tx:5.2f}, {ty:5.2f})  '
              f'gain={gain} dist={dist:.1f}m clust_sz={sz} util={utility:.0f}',
              flush=True)

        # Dispatch plan
        try:
            post(f'{WEBUI}/plan?x={tx:.3f}&y={ty:.3f}')
        except Exception as e:
            print(f'[explore-agent] plan failed: {e}', flush=True)
            time.sleep(2)
            continue

        if args.once:
            return 0

        # Wait for plan completion (or timeout)
        t0 = time.time()
        while time.time() - t0 < WAIT_PLAN_S:
            pj = get_json(f'{WEBUI}/path.json') or {}
            if not pj.get('active'):
                break
            time.sleep(0.5)
        else:
            # Timed out — clear path and try a new frontier
            post(f'{WEBUI}/path/clear')
        time.sleep(SLEEP_BETWEEN_S)


if __name__ == '__main__':
    sys.exit(main() or 0)
