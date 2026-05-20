#!/usr/bin/env python3
"""
Autonomous YOLO-tracking investigator that drives the rover through PX4
OFFBOARD mode (not direct gz wheel publish). This is the deploy path that
also works on a real rover: tracking algorithm produces a velocity command,
PX4 differential-drive controller turns that into wheel torques.

Flow:
  pymavlink connect (udpin:14550)
  -> stream zero velocity setpoints for ~2 s
  -> request OFFBOARD mode
  -> force-arm (magic 21196 — bypasses RC / safety prearm checks in SITL)
  -> 20 Hz loop:
       * read /investigate.json     -> active target class
       * read /front/detections.json + /rear/detections.json (YOLO-World)
       * visual servo  -> body-frame (vx_body, yaw_rate)
       * rotate by current PX4 yaw -> NED velocity (vx, vy)
       * SET_POSITION_TARGET_LOCAL_NED  vx, vy, 0, yaw_rate

Run while PX4 is connected (SITL).  uses CPython 3.9 (system) so pymavlink
is available — gz transport NOT used here.
"""
import json
import math
import os
import sys
import threading
import time
import urllib.request

os.environ.setdefault('MAVLINK20', '1')
import threading  # noqa: E402
from pymavlink import mavutil  # noqa: E402


WEBUI = 'http://localhost:8080'
DETECT = 'http://localhost:8081'

# PX4 custom main modes
PX4_OFFBOARD = 6

# SET_POSITION_TARGET_LOCAL_NED type_mask bits (1 = ignore)
TM_IGNORE_POS = 0b0000_0000_0111
TM_IGNORE_VEL = 0b0000_0011_1000
TM_IGNORE_ACC = 0b0001_1100_0000
TM_IGNORE_YAW = 0b0010_0000_0000
TM_IGNORE_YAWRATE = 0b0100_0000_0000
# Use velocity (vx, vy, vz) + yaw_rate; ignore position, accel, yaw.
TYPE_MASK_VEL_YAWRATE = TM_IGNORE_POS | TM_IGNORE_ACC | TM_IGNORE_YAW
# Position + yaw: use x,y,yaw, ignore z,vel,acc,yaw_rate
TYPE_MASK_POS_YAW = TM_IGNORE_VEL | TM_IGNORE_ACC | TM_IGNORE_YAWRATE

# Behavior tuning (live overridable via webui GET /config.json)
FRAME_W = 800

# Defaults — get replaced by /config.json every CFG_REFRESH_S seconds.
CFG_REFRESH_S = 0.5
cfg = {
    'fwd_speed':            2.0,
    'max_yaw_rate_deg':     45.0,
    'search_yaw_rate_deg':  35.0,
    'approach_yaw_gain':    0.6,
    'follow_yaw_gain':      0.8,
    'approach_stop_ratio':  0.30,
    'follow_goal_low':      0.25,
    'follow_goal_high':     0.40,
    'lost_timeout':         4.0,
}
_cfg_lock = threading.Lock()
_cfg_last_pull = 0.0


def refresh_cfg():
    global _cfg_last_pull
    if time.time() - _cfg_last_pull < CFG_REFRESH_S:
        return
    _cfg_last_pull = time.time()
    j = http_get_json(f'{WEBUI}/config.json')
    if not j:
        return
    with _cfg_lock:
        for k in cfg.keys():
            if k in j:
                try:
                    cfg[k] = float(j[k])
                except (TypeError, ValueError):
                    pass


def http_get_json(url, timeout=0.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def push_state(s):
    import urllib.parse
    try:
        url = f'{WEBUI}/investigate?state={urllib.parse.quote(s)}'
        urllib.request.urlopen(urllib.request.Request(url, method='POST'),
                               timeout=0.5).read()
    except Exception:
        pass


def best_detection_for(cls, dets, locked_id=None):
    if not dets:
        return None
    matches = [d for d in dets if d.get('cls') == cls]
    if not matches:
        return None
    if locked_id is not None:
        same = [d for d in matches if d.get('id') == locked_id]
        if same:
            return max(same, key=lambda d: d['conf'])
    return max(matches, key=lambda d: d['conf'])


class Telem:
    """Background-thread MAVLink reader.  Keeps latest pose + arm/mode state."""
    def __init__(self, mav):
        self.mav = mav
        self.yaw = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.have_pos = False
        self.armed = False
        self.custom_mode = 0
        self.lock = threading.Lock()
        self.alive = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while self.alive:
            msg = self.mav.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            t = msg.get_type()
            if t == 'ATTITUDE':
                with self.lock:
                    self.yaw = msg.yaw
            elif t == 'LOCAL_POSITION_NED':
                with self.lock:
                    self.x = msg.x
                    self.y = msg.y
                    self.z = msg.z
                    self.have_pos = True
            elif t == 'HEARTBEAT' and msg.type == mavutil.mavlink.MAV_TYPE_GROUND_ROVER:
                with self.lock:
                    self.armed = bool(msg.base_mode & 128)
                    self.custom_mode = msg.custom_mode


def send_velocity_yawrate(mav, vx_ned, vy_ned, yaw_rate):
    mav.mav.set_position_target_local_ned_send(
        0,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0,
        vx_ned, vy_ned, 0,
        0, 0, 0,
        0,
        yaw_rate)


def send_position_yaw(mav, x_ned, y_ned, yaw):
    mav.mav.set_position_target_local_ned_send(
        0,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x_ned, y_ned, 0,
        0, 0, 0,
        0, 0, 0,
        yaw,
        0)


def force_arm(mav, value=1):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        value, 21196, 0, 0, 0, 0, 0)


def set_mode_offboard(mav):
    # PX4 custom_mode is a packed uint32: byte 3 = main_mode, byte 2 = sub_mode.
    # OFFBOARD has main_mode=6, sub_mode=0 -> 6 << 16 = 0x00060000.
    custom_mode = PX4_OFFBOARD << 16
    # 1) MAV_CMD_DO_SET_MODE  (preferred by PX4 v1.16)
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        PX4_OFFBOARD, 0, 0, 0, 0, 0)
    # 2) Belt-and-braces: also the legacy SET_MODE message with packed custom_mode
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        custom_mode)


def main():
    print('[off] connect udpin:14552 (via bridge) ...', flush=True)
    mav = mavutil.mavlink_connection('udpin:0.0.0.0:14550',
                                      source_system=255, source_component=0,
                                      dialect='common')
    hb = mav.wait_heartbeat(timeout=20)
    if mav.target_system == 0:
        mav.target_system = 1
    if mav.target_component == 0:
        mav.target_component = 1
    print(f'[off] sys={mav.target_system}  comp={mav.target_component}',
          flush=True)
    # Keep our outbound heartbeats alive so PX4 doesn't kill OFFBOARD link.
    last_hb = [time.time()]
    def beat():
        if time.time() - last_hb[0] > 0.8:
            mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                                    0, 0, 0)
            last_hb[0] = time.time()

    tel = Telem(mav)

    # Prime OFFBOARD with zero setpoints for ~3 s, then enter main loop
    # (which will re-assert arm + OFFBOARD continuously).
    print('[off] streaming setpoints to prime OFFBOARD ...', flush=True)
    for _ in range(75):
        send_velocity_yawrate(mav, 0, 0, 0)
        time.sleep(0.04)
    force_arm(mav, 1)
    time.sleep(0.3)
    set_mode_offboard(mav)
    time.sleep(0.5)
    with tel.lock:
        armed = tel.armed
        cm_main = (tel.custom_mode >> 16) & 0xff
    print(f'[off] init armed={armed} main_mode={cm_main} '
          f'(loop will keep retrying)', flush=True)

    rate_hz = 20.0
    period = 1.0 / rate_hz
    state = 'idle'
    last_seen = 0.0
    locked_id = None
    explore_t0 = time.time()
    explore_phase = 'fwd'
    explore_phase_end = explore_t0 + 5.0

    try:
        while True:
            beat()
            # ===== PATH FOLLOWING (highest priority) =====
            path = http_get_json(f'{WEBUI}/path.json') or {}
            if path.get('active') and path.get('waypoints'):
                wpts = path['waypoints']
                cursor = int(path.get('cursor', 0))
                with tel.lock:
                    px = tel.x
                    py = tel.y
                    yaw_now = tel.yaw
                    have_pos = tel.have_pos
                if not have_pos:
                    send_velocity_yawrate(mav, 0, 0, 0)
                    time.sleep(period); continue
                while cursor < len(wpts) - 1:
                    wx, wy = wpts[cursor]
                    if math.hypot(wx - px, wy - py) < 0.5:
                        cursor += 1
                    else:
                        break
                # Lookahead ~2 m
                look_x, look_y = wpts[cursor]
                for k in range(cursor, len(wpts)):
                    xk, yk = wpts[k]
                    if math.hypot(xk - px, yk - py) >= 2.0:
                        look_x, look_y = xk, yk; break
                # Pure pursuit velocity output (speed-limited).
                target_yaw = math.atan2(look_y - py, look_x - px)
                bearing = math.atan2(math.sin(target_yaw - yaw_now),
                                     math.cos(target_yaw - yaw_now))
                with _cfg_lock:
                    FWD = cfg['fwd_speed']
                    MAX_YAW = math.radians(cfg['max_yaw_rate_deg'])
                yaw_rate = max(-MAX_YAW, min(MAX_YAW, 1.2 * bearing))
                fwd = FWD * max(0.2, math.cos(bearing))
                vx_ned = fwd * math.cos(yaw_now)
                vy_ned = fwd * math.sin(yaw_now)
                send_velocity_yawrate(mav, vx_ned, vy_ned, yaw_rate)
                # Update cursor on webui
                try:
                    import urllib.request
                    urllib.request.urlopen(urllib.request.Request(
                        f'{WEBUI}/path/advance?cursor={cursor}', method='POST'),
                        timeout=0.3).read()
                except Exception:
                    pass
                gx, gy = wpts[-1]
                if math.hypot(gx - px, gy - py) < 0.6:
                    print(f'[off] reached path goal ({gx:.2f},{gy:.2f})',
                          flush=True)
                    try:
                        import urllib.request
                        urllib.request.urlopen(urllib.request.Request(
                            f'{WEBUI}/path/clear', method='POST'),
                            timeout=0.3).read()
                    except Exception:
                        pass
                if state != 'goto':
                    state = 'goto'
                    push_state('goto')
                # Maintain OFFBOARD + armed
                with tel.lock:
                    armed_now = tel.armed
                    mode_now = (tel.custom_mode >> 16) & 0xff
                if not armed_now or mode_now != PX4_OFFBOARD:
                    force_arm(mav, 1)
                    set_mode_offboard(mav)
                time.sleep(period); continue

            inv = http_get_json(f'{WEBUI}/investigate.json') or {}
            if not inv.get('active'):
                # No investigate target -> autonomous EXPLORE pattern (build
                # SLAM map). Lawnmower: forward 5s, turn 2.5s left, repeat.
                if state != 'explore':
                    state = 'explore'
                    push_state('explore')
                    print('[off] -> explore', flush=True)
                    explore_t0 = time.time()
                    explore_phase = 'fwd'
                    explore_phase_end = explore_t0 + 5.0

                now = time.time()
                # Obstacle awareness: poll /obstacles.json (cheap, 700ms cache
                # is fine).  If wall within OBSTACLE_DIST in front, abort fwd
                # phase and start turn phase immediately.
                obst = http_get_json(f'{WEBUI}/obstacles.json') or {}
                front_clear = float(obst.get('front', 99.0))
                OBSTACLE_DIST = 1.6  # m, force-turn threshold
                SLOW_DIST = 3.0      # m, slow forward

                if explore_phase == 'fwd' and front_clear < OBSTACLE_DIST:
                    explore_phase = 'turn'
                    explore_phase_end = now + 2.0
                    print(f'[off] obstacle {front_clear:.2f}m -> turn',
                          flush=True)
                elif now > explore_phase_end:
                    if explore_phase == 'fwd':
                        explore_phase = 'turn'
                        explore_phase_end = now + 2.5
                    else:
                        explore_phase = 'fwd'
                        explore_phase_end = now + 5.0
                    print(f'[off] explore phase -> {explore_phase}', flush=True)

                refresh_cfg()
                with _cfg_lock:
                    FWD_SPEED = cfg['fwd_speed']
                    SEARCH_YAW_RATE = math.radians(cfg['search_yaw_rate_deg'])

                if explore_phase == 'fwd':
                    # Scale forward speed by clearance
                    speed_scale = 0.7
                    if front_clear < SLOW_DIST:
                        speed_scale = 0.7 * max(0.25,
                            (front_clear - OBSTACLE_DIST) /
                            (SLOW_DIST - OBSTACLE_DIST))
                    fwd_body = FWD_SPEED * speed_scale
                    yaw_rate = 0.0
                else:
                    fwd_body = 0.0
                    yaw_rate = SEARCH_YAW_RATE

                # Use the same body-fwd -> NED position lookahead as below.
                with tel.lock:
                    yaw_now = tel.yaw
                    px = tel.x
                    py = tel.y
                    have_pos = tel.have_pos
                target_yaw = yaw_now + yaw_rate * 0.5
                lookahead = max(min(fwd_body, FWD_SPEED), -FWD_SPEED) * 1.5
                tx = px + lookahead * math.cos(target_yaw)
                ty = py + lookahead * math.sin(target_yaw)
                if have_pos:
                    send_position_yaw(mav, tx, ty, target_yaw)
                else:
                    send_velocity_yawrate(mav,
                                          fwd_body * math.cos(yaw_now),
                                          fwd_body * math.sin(yaw_now),
                                          yaw_rate)
                time.sleep(period)
                continue

            cls = inv.get('cls', '')
            df = http_get_json(f'{DETECT}/front/detections.json') or {}
            dr = http_get_json(f'{DETECT}/rear/detections.json') or {}
            d_front = best_detection_for(cls, df.get('detections', []),
                                         locked_id)
            d_rear = best_detection_for(cls, dr.get('detections', []),
                                        locked_id)
            d = d_front
            now = time.time()
            if d is not None:
                last_seen = now

            # State transitions
            if state == 'idle':
                state = 'search'
                push_state('search')
                print(f'[off] hunting {cls!r} -> search', flush=True)

            if state == 'search' and d is not None:
                state = 'approach'
                locked_id = d.get('id') if d.get('id', -1) >= 0 else None
                push_state('approach')
                print(f'[off] {cls}#{locked_id} acquired -> approach',
                      flush=True)

            # Compute body-frame command (fwd_body, yaw_rate)
            fwd_body = 0.0
            yaw_rate = 0.0

            refresh_cfg()
            with _cfg_lock:
                FWD_SPEED = cfg['fwd_speed']
                MAX_YAW_RATE = math.radians(cfg['max_yaw_rate_deg'])
                SEARCH_YAW_RATE = math.radians(cfg['search_yaw_rate_deg'])
                APPROACH_YAW_GAIN = cfg['approach_yaw_gain']
                FOLLOW_YAW_GAIN = cfg['follow_yaw_gain']
                APPROACH_STOP_RATIO = cfg['approach_stop_ratio']
                FOLLOW_GOAL_LOW = cfg['follow_goal_low']
                FOLLOW_GOAL_HIGH = cfg['follow_goal_high']
                LOST_TIMEOUT = cfg['lost_timeout']

            if state == 'search':
                if d_rear is not None and d_front is None:
                    yaw_rate = SEARCH_YAW_RATE * 1.5  # spin faster
                else:
                    yaw_rate = SEARCH_YAW_RATE

            elif state == 'approach':
                if now - last_seen > LOST_TIMEOUT:
                    state = 'search'
                    push_state('search')
                    print('[off] lost -> search', flush=True)
                elif d is None:
                    fwd_body = 0.3 * FWD_SPEED
                else:
                    x, _y, w, _h = d['box']
                    cx = x + w / 2.0
                    bearing = (cx - FRAME_W / 2.0) / (FRAME_W / 2.0)
                    size_ratio = w / float(FRAME_W)
                    yaw_rate = -APPROACH_YAW_GAIN * bearing
                    if size_ratio > APPROACH_STOP_RATIO:
                        state = 'follow'
                        push_state('follow')
                        print(f'[off] close (size={size_ratio:.2f}) -> follow',
                              flush=True)
                    else:
                        fwd_body = FWD_SPEED * max(0.3, 1.0 - 1.5 * size_ratio)

            elif state == 'follow':
                if now - last_seen > LOST_TIMEOUT * 1.5:
                    state = 'search'
                    locked_id = None
                    push_state('search')
                    print('[off] follow lost -> search', flush=True)
                elif d is None:
                    fwd_body = 0.0
                else:
                    x, _y, w, _h = d['box']
                    cx = x + w / 2.0
                    bearing = (cx - FRAME_W / 2.0) / (FRAME_W / 2.0)
                    size_ratio = w / float(FRAME_W)
                    yaw_rate = -FOLLOW_YAW_GAIN * bearing
                    if size_ratio < FOLLOW_GOAL_LOW:
                        gap = FOLLOW_GOAL_LOW - size_ratio
                        fwd_body = FWD_SPEED * min(1.0, 2.5 * gap + 0.3)
                    elif size_ratio > FOLLOW_GOAL_HIGH:
                        gap = size_ratio - FOLLOW_GOAL_HIGH
                        fwd_body = -FWD_SPEED * min(0.8, 2.5 * gap + 0.2)
                    else:
                        fwd_body = 0.0

            # Cap yaw_rate
            if yaw_rate > MAX_YAW_RATE:
                yaw_rate = MAX_YAW_RATE
            elif yaw_rate < -MAX_YAW_RATE:
                yaw_rate = -MAX_YAW_RATE

            # PX4 rover OFFBOARD reliably honors position+yaw setpoints
            # (pure-pursuit controller). Convert our body-frame (fwd, yaw_rate)
            # command into an absolute waypoint a short distance ahead.
            with tel.lock:
                yaw_now = tel.yaw
                px = tel.x
                py = tel.y
                have_pos = tel.have_pos

            # Predict desired yaw 0.5 s into the future.
            target_yaw = yaw_now + yaw_rate * 0.5
            # Lookahead in metres (positive forward, negative reverse).
            lookahead = max(min(fwd_body, FWD_SPEED), -FWD_SPEED) * 1.5
            tx = px + lookahead * math.cos(target_yaw)
            ty = py + lookahead * math.sin(target_yaw)

            if have_pos:
                send_position_yaw(mav, tx, ty, target_yaw)
            else:
                # No LOCAL_POSITION_NED yet — fall back to velocity setpoint
                vx_ned = fwd_body * math.cos(yaw_now)
                vy_ned = fwd_body * math.sin(yaw_now)
                send_velocity_yawrate(mav, vx_ned, vy_ned, yaw_rate)

            # Re-assert arm/OFFBOARD aggressively — PX4 SITL demotes the rover
            # to disarmed when it thinks setpoints are stale or pose drifts.
            with tel.lock:
                armed_now = tel.armed
                mode_now = (tel.custom_mode >> 16) & 0xff
            if not armed_now or mode_now != PX4_OFFBOARD:
                force_arm(mav, 1)
                set_mode_offboard(mav)

            time.sleep(period)

    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(20):
            send_velocity_yawrate(mav, 0, 0, 0)
            time.sleep(period)
        force_arm(mav, 0)
        tel.alive = False
        print('[off] disarmed, exit', flush=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
