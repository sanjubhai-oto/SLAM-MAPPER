#!/usr/bin/env python3.14
"""
Direct gz wheel publisher using Python gz-transport13 bindings (continuous,
in-process — no per-message subprocess overhead). Publishes
/model/rover_360cam_0/command/motor_speed (gz.msgs.Actuators) at 30 Hz.

PX4 must be DISARMED so its mixer doesn't co-publish.

Forward + alternating turn pattern. Ctrl-C to stop.
"""
import argparse
import sys
import time

from gz.transport13 import Node
from gz.msgs10.actuators_pb2 import Actuators


TOPIC = '/model/rover_360cam_0/command/motor_speed'


def make_msg(left: float, right: float) -> Actuators:
    msg = Actuators()
    # SDF JointController plugins: actuator_number 0 = wheel_rear_right,
    #                              actuator_number 1 = wheel_rear_left
    msg.velocity.extend([right, left])
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=6.0,
                    help='forward wheel angular vel rad/s (default 6)')
    ap.add_argument('--turn', type=float, default=4.0,
                    help='turn wheel vel rad/s (default 4)')
    ap.add_argument('--fwd-sec', type=float, default=6.0)
    ap.add_argument('--turn-sec', type=float, default=3.0)
    ap.add_argument('--rate', type=float, default=30.0)
    args = ap.parse_args()

    node = Node()
    pub = node.advertise(TOPIC, Actuators)
    if not pub.valid():
        print(f'[gz_drive] failed to advertise {TOPIC}', file=sys.stderr)
        return 1
    # Wait briefly so subscribers connect before we start streaming
    time.sleep(0.5)

    period = 1.0 / args.rate
    phase = 'forward'
    phase_end = time.time() + args.fwd_sec
    print(f'[gz_drive] forward={args.speed} rad/s, turn={args.turn} rad/s, '
          f'rate={args.rate} Hz', flush=True)
    try:
        while True:
            now = time.time()
            if now > phase_end:
                if phase == 'forward':
                    phase = 'turn'
                    phase_end = now + args.turn_sec
                else:
                    phase = 'forward'
                    phase_end = now + args.fwd_sec
                print(f'[gz_drive] -> {phase}', flush=True)
            if phase == 'forward':
                pub.publish(make_msg(args.speed, args.speed))
            else:
                pub.publish(make_msg(-args.turn, args.turn))
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(15):
            pub.publish(make_msg(0, 0))
            time.sleep(0.03)
        print('[gz_drive] stopped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
