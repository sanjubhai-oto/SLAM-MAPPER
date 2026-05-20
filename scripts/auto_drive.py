#!/usr/bin/env python3
"""
Drive PX4 rover (gz SITL) in a closed loop:

  1. Connect to PX4 onboard MAVLink (UDP 14540)
  2. Arm
  3. Switch to OFFBOARD
  4. Stream SET_POSITION_TARGET_LOCAL_NED with velocity+yaw_rate at 20 Hz
     following a square waypoint pattern (or one-shot 'go' command)
  5. On Ctrl-C: disarm

Usage:
    python3 auto_drive.py                    # default 8x8 m square loop
    python3 auto_drive.py --side 6 --speed 1 # custom
    python3 auto_drive.py --goto 5 5         # one-shot waypoint to local NED (5,5)
"""

import argparse
import math
import sys
import time

from pymavlink import mavutil

# PX4 OFFBOARD mode bits (PX4 custom mode encoding)
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6
PX4_CUSTOM_MAIN_MODE_AUTO = 4

# SET_POSITION_TARGET_LOCAL_NED type_mask bits
# Bit 0..2 = ignore position xyz
# Bit 3..5 = ignore velocity xyz
# Bit 6..8 = ignore accel xyz
# Bit 9    = force
# Bit 10   = ignore yaw
# Bit 11   = ignore yaw_rate
TM_IGNORE_POS = 0b0000_0000_0111
TM_IGNORE_VEL = 0b0000_0011_1000
TM_IGNORE_ACC = 0b0001_1100_0000
TM_IGNORE_YAW = 0b0010_0000_0000
TM_IGNORE_YAW_RATE = 0b0100_0000_0000

# Use velocity + yaw_rate (ignore position, accel, yaw)
TYPE_MASK_VEL_YAWRATE = (TM_IGNORE_POS | TM_IGNORE_ACC | TM_IGNORE_YAW)
# Use position + yaw (ignore velocity, accel, yaw_rate)
TYPE_MASK_POS_YAW = (TM_IGNORE_VEL | TM_IGNORE_ACC | TM_IGNORE_YAW_RATE)


def connect():
    print('[auto_drive] connecting to udpin:0.0.0.0:14540 ...', flush=True)
    m = mavutil.mavlink_connection('udpin:0.0.0.0:14540')
    m.wait_heartbeat(timeout=15)
    print(f'[auto_drive] heartbeat from system {m.target_system} '
          f'component {m.target_component}', flush=True)
    return m


def set_mode_offboard(m):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        PX4_CUSTOM_MAIN_MODE_OFFBOARD, 0, 0, 0, 0, 0)


def arm(m, arm_value=1):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        arm_value, 0, 0, 0, 0, 0, 0)


def send_vel_yawrate(m, vx, vy, yaw_rate):
    m.mav.set_position_target_local_ned_send(
        0,
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0,                # x y z (ignored)
        vx, vy, 0,              # vx vy vz (m/s, NED)
        0, 0, 0,                # ax ay az
        0,                      # yaw (ignored)
        yaw_rate)               # yaw_rate (rad/s)


def send_pos_yaw(m, x, y, yaw):
    m.mav.set_position_target_local_ned_send(
        0,
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x, y, 0,                # x y z (m, NED)
        0, 0, 0,                # vx vy vz
        0, 0, 0,                # ax ay az
        yaw,                    # yaw (rad)
        0)                      # yaw_rate


def square_corners(side):
    s = side
    return [(s, 0), (s, s), (0, s), (0, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--side', type=float, default=6.0,
                    help='square side length in meters (default 6)')
    ap.add_argument('--speed', type=float, default=1.0,
                    help='forward speed m/s (default 1.0)')
    ap.add_argument('--goto', nargs=2, type=float, metavar=('X', 'Y'),
                    help='one-shot NED waypoint (skip square pattern)')
    ap.add_argument('--rate', type=float, default=20.0,
                    help='setpoint stream rate Hz (default 20)')
    args = ap.parse_args()

    m = connect()

    # Stream initial setpoints (must arrive before OFFBOARD is accepted)
    for _ in range(40):
        send_vel_yawrate(m, 0, 0, 0)
        time.sleep(0.05)

    set_mode_offboard(m)
    time.sleep(0.5)
    arm(m, 1)
    print('[auto_drive] arm + offboard requested', flush=True)
    time.sleep(1.5)

    waypoints = [tuple(args.goto)] if args.goto else square_corners(args.side)
    print(f'[auto_drive] waypoints (NED x,y): {waypoints}', flush=True)

    period = 1.0 / args.rate
    radius_ok = 0.6
    try:
        for wx, wy in waypoints:
            print(f'[auto_drive] -> ({wx:.2f}, {wy:.2f})', flush=True)
            while True:
                # Grab latest local position from PX4
                msg = m.recv_match(type='LOCAL_POSITION_NED', blocking=False)
                if msg is not None:
                    px, py = msg.x, msg.y
                else:
                    px = py = None

                if px is None:
                    # No fix yet, keep streaming zero so OFFBOARD stays alive
                    send_vel_yawrate(m, 0, 0, 0)
                    time.sleep(period)
                    continue

                dx = wx - px
                dy = wy - py
                dist = math.hypot(dx, dy)
                if dist < radius_ok:
                    break

                # Desired yaw from current pos to waypoint
                target_yaw = math.atan2(dy, dx)
                # Current yaw from ATTITUDE
                att = m.recv_match(type='ATTITUDE', blocking=False)
                yaw = att.yaw if att else 0.0
                yaw_err = math.atan2(math.sin(target_yaw - yaw),
                                     math.cos(target_yaw - yaw))

                # Simple controller: forward speed gated by heading alignment
                fwd = args.speed * max(0.0, math.cos(yaw_err))
                yr = 1.5 * yaw_err  # rad/s

                # NED frame velocity: forward in body -> rotate to NED via yaw
                vx_ned = fwd * math.cos(yaw)
                vy_ned = fwd * math.sin(yaw)

                send_vel_yawrate(m, vx_ned, vy_ned, yr)
                time.sleep(period)

        # Stop
        for _ in range(30):
            send_vel_yawrate(m, 0, 0, 0)
            time.sleep(period)
        print('[auto_drive] route complete', flush=True)

    except KeyboardInterrupt:
        print('[auto_drive] interrupted, stopping rover', flush=True)
        for _ in range(20):
            send_vel_yawrate(m, 0, 0, 0)
            time.sleep(period)
    finally:
        arm(m, 0)
        print('[auto_drive] disarmed', flush=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
