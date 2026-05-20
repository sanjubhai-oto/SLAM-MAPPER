#!/usr/bin/env python3
"""MAVLink telemetry collector — polls PX4, writes /tmp/mavlink_telem.json"""
import json, time, threading
from pymavlink import mavutil

OUT = '/tmp/mavlink_telem.json'

state = {
    'connected': False,
    'armed': False,
    'mode': 'UNKNOWN',
    'main_mode': 0,
    'sub_mode': 0,
    'battery_v': 0.0,
    'battery_pct': 0,
    'gcs_link': False,
    'home_dist': 0.0,
    'last_status': '',
    'last_warning': '',
    'updated_ms': 0,
}

# Connect via 14550 (PX4 broadcasts here once peer registers). Kill QGC first.
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('0.0.0.0', 14550)); s.close()
    mav = mavutil.mavlink_connection('udpin:0.0.0.0:14550', source_system=255)
except OSError:
    # Fallback: udpout to PX4 18570 (GCS port)
    mav = mavutil.mavlink_connection('udpout:127.0.0.1:18570', source_system=255)

# Send heartbeats so PX4 streams data back to us
def beat():
    while True:
        try:
            mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                   mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        except Exception:
            pass
        time.sleep(0.5)

threading.Thread(target=beat, daemon=True).start()

# Map PX4 custom_mode (main_mode byte) -> string
PX4_MODES = {
    1: 'MANUAL', 2: 'ALTCTL', 3: 'POSCTL', 4: 'AUTO', 5: 'ACRO',
    6: 'OFFBOARD', 7: 'STAB', 8: 'RATTITUDE'
}


def pump():
    home_x, home_y = None, None
    last_x, last_y = 0.0, 0.0
    last_rx_ms = 0
    while True:
        msg = mav.recv_match(blocking=True, timeout=2)
        now_ms = int(time.time() * 1000)
        if msg is None:
            if now_ms - last_rx_ms > 3000:
                state['connected'] = False
                state['gcs_link'] = False
        else:
            last_rx_ms = now_ms
            state['connected'] = True
            state['gcs_link'] = True
            t = msg.get_type()
            if t == 'HEARTBEAT' and msg.type == mavutil.mavlink.MAV_TYPE_GROUND_ROVER:
                state['armed'] = bool(msg.base_mode & 128)
                state['main_mode'] = (msg.custom_mode >> 16) & 0xff
                state['sub_mode'] = (msg.custom_mode >> 24) & 0xff
                state['mode'] = PX4_MODES.get(state['main_mode'],
                                              f'MODE{state["main_mode"]}')
            elif t == 'BATTERY_STATUS':
                v = (msg.voltages[0] / 1000.0
                     if msg.voltages and msg.voltages[0] != -1 else 0.0)
                state['battery_v'] = v
                state['battery_pct'] = (max(0, min(100, msg.battery_remaining))
                                        if msg.battery_remaining >= 0 else 0)
            elif t == 'LOCAL_POSITION_NED':
                last_x, last_y = msg.x, msg.y
                if home_x is None:
                    home_x, home_y = msg.x, msg.y
                state['home_dist'] = ((last_x - home_x) ** 2 +
                                     (last_y - home_y) ** 2) ** 0.5
            elif t == 'STATUSTEXT':
                state['last_status'] = msg.text
                if msg.severity <= 4:
                    state['last_warning'] = msg.text
        state['updated_ms'] = now_ms
        try:
            tmp = OUT + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(state, f)
            import os
            os.replace(tmp, OUT)
        except Exception:
            pass


if __name__ == '__main__':
    pump()
