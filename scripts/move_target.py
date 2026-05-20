#!/usr/bin/env python3.14
"""
Animate a target entity in the gz world so the investigator has something
moving to follow. Calls /world/<world>/set_pose continuously via gz-transport
service requests.

Default: sine-wave path along Y axis around (x=3, y=0).
"""
import argparse
import math
import sys
import time

from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default='slam_obstacles')
    ap.add_argument('--name', default='test_car')
    ap.add_argument('--cx', type=float, default=3.0)
    ap.add_argument('--cy', type=float, default=0.0)
    ap.add_argument('--amplitude', type=float, default=3.0,
                    help='peak deviation in meters')
    ap.add_argument('--period', type=float, default=20.0,
                    help='full oscillation period in seconds')
    ap.add_argument('--rate', type=float, default=15.0)
    args = ap.parse_args()

    node = Node()
    svc = f'/world/{args.world}/set_pose'
    period = 1.0 / args.rate
    t0 = time.time()
    print(f'[move] animating {args.name} on /world/{args.world}', flush=True)
    while True:
        t = time.time() - t0
        phase = 2.0 * math.pi * t / args.period
        x = args.cx
        y = args.cy + args.amplitude * math.sin(phase)
        # face direction of motion
        yaw = math.pi / 2 if math.cos(phase) > 0 else -math.pi / 2
        msg = Pose()
        msg.name = args.name
        msg.position.x = x
        msg.position.y = y
        msg.position.z = 0.0
        msg.orientation.z = math.sin(yaw / 2)
        msg.orientation.w = math.cos(yaw / 2)
        rep = Boolean()
        ok, _ = node.request(svc, msg, Pose, Boolean, 200)
        if not ok:
            time.sleep(0.5)
        time.sleep(period)


if __name__ == '__main__':
    sys.exit(main() or 0)
