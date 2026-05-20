#!/usr/bin/env python3
"""
Body-frame exploration: arm PX4 rover, switch to OFFBOARD, then run a simple
forward+turn loop that doesn't require any frame-of-reference math. Designed
to make the rover wander the slam_obstacles world while the depth-camera
pointcloud accumulates in the web UI.

  - 8 s forward at `speed` m/s
  - 4 s in-place turn at +60 deg/s
  - repeat

Stop with Ctrl-C (sends disarm).
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault('MAVLINK20', '1')
from pymavlink import mavutil  # noqa: E402

PX4_OFFBOARD = 6
TM_IGNORE_POS = 0b0000_0000_0111
TM_IGNORE_ACC = 0b0001_1100_0000
TM_IGNORE_YAW = 0b0010_0000_0000
TYPE_MASK = TM_IGNORE_POS | TM_IGNORE_ACC | TM_IGNORE_YAW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=1.2)
    ap.add_argument('--turn-deg', type=float, default=60.0)
    ap.add_argument('--fwd-sec', type=float, default=6.0)
    ap.add_argument('--turn-sec', type=float, default=4.0)
    ap.add_argument('--rate', type=float, default=20.0)
    args = ap.parse_args()

    m = mavutil.mavlink_connection('udpin:0.0.0.0:14540', dialect='common')
    print('[explore] waiting heartbeat...', flush=True)
    m.wait_heartbeat(timeout=15)
    print(f'[explore] connected sys={m.target_system}', flush=True)

    period = 1.0 / args.rate
    yaw_rate_cmd = math.radians(args.turn_deg)

    def send(vx_body, yr):
        # MAV_FRAME_BODY_NED: vx_body is forward in body, vy=0
        m.mav.set_position_target_local_ned_send(
            0, m.target_system, m.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            TYPE_MASK,
            0, 0, 0,
            vx_body, 0, 0,
            0, 0, 0,
            0, yr)

    # Prime OFFBOARD with zero setpoints
    for _ in range(40):
        send(0, 0)
        time.sleep(0.05)

    # Switch to OFFBOARD
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        PX4_OFFBOARD, 0, 0, 0, 0, 0)
    time.sleep(0.5)
    # Force arm (magic 21196 bypasses pre-arm checks; rover SITL may otherwise
    # refuse to arm even with GPS lock if no RC / safety switch).
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 21196, 0, 0, 0, 0, 0)
    print('[explore] force-arm + OFFBOARD', flush=True)
    time.sleep(1.5)

    # Confirm armed; retry once if not
    hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=2.0)
    if hb and not (hb.base_mode & 128):
        print('[explore] not armed, retrying force-arm...', flush=True)
        m.mav.command_long_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 21196, 0, 0, 0, 0, 0)
        time.sleep(1.0)

    phase = 'forward'
    phase_end = time.time() + args.fwd_sec
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
                print(f'[explore] phase -> {phase}', flush=True)
            if phase == 'forward':
                send(args.speed, 0.0)
            else:
                send(0.0, yaw_rate_cmd)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        print('[explore] stop + disarm', flush=True)
        for _ in range(20):
            send(0, 0)
            time.sleep(period)
        m.mav.command_long_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            0, 21196, 0, 0, 0, 0, 0)


if __name__ == '__main__':
    sys.exit(main() or 0)
