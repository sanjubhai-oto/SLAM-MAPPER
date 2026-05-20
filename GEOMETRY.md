# slam_rover — Project Geometry Reference

All dimensions in metres / radians unless noted.  Coordinates use REP-103
robotics convention: **base_link = X forward, Y left, Z up**.

## World — `slam_obstacles.sdf`

| Element          | Position (x, y, z) | Size (L, W, H) | Notes |
|------------------|--------------------|-----------------|-------|
| Ground plane     | 0, 0, 0            | 200 × 200       | Friction μ = 2.0 |
| Wall north       | 0, +10, 1          | 20 × 0.2 × 2    | orange |
| Wall south       | 0, -10, 1          | 20 × 0.2 × 2    | cyan |
| Wall east        | +10, 0, 1          | 0.2 × 20 × 2    | yellow |
| Wall west        | -10, 0, 1          | 0.2 × 20 × 2    | purple |
| Pillar 1         | 3, 2, 0.75         | r = 0.3, h = 1.5 | white |
| Pillar 2         | -4, -3, 0.75       | r = 0.3, h = 1.5 | green |
| Box obstacle 1   | 5, -4, 0.5         | 1.2 × 0.8 × 1.0 yaw 0.3 rad | red |
| Box obstacle 2   | -3, 5, 0.4         | 0.8 × 1.5 × 0.8 yaw -0.5 rad | blue |
| Landmark board   | 0, 9.85, 1         | 2 × 0.05 × 1.2  | white reference plate |
| Sun              | directional        | intensity 1.2   | shadows ON |

Operating clear area: **-9 m ≤ x,y ≤ +9 m**.

## Rover — `rover_360cam` (differential drive)

### Chassis
- base_link pose:  `(0, 0, 0.1)` (relative to `__model__`)
- chassis visual mesh:  `rover_differential_base.dae`
- chassis collision box:  **1.0 × 0.5 × 0.1 m**
- mass: 5.0 kg, COM offset: -0.2 m in X (rear-biased)

### Wheels
- rear_left:   `(-0.2,  0.3, 0)` rotated -90° X → wheel axis Y
- rear_right:  `(-0.2, -0.3, 0)` rotated -90° X
- cast front:  `( 0.25, 0,  -0.025)` ball joint, sphere r=0.075
- cast rear:   `(-0.30, 0,  -0.025)` ball joint, sphere r=0.075
- powered wheel radius: **0.1 m**, length 0.1
- wheel track (left-to-right): **0.6 m**
- max wheel angular vel commanded: ~8 rad/s → linear 0.8 m/s

### Sensors (all attached to `base_link`)
| Sensor | Pose (x,y,z, roll,pitch,yaw) | Spec |
|--------|------------------------------|------|
| IMU             | base_link origin       | 250 Hz, IIM42653-noise |
| Magnetometer    | base_link origin       | 100 Hz, IIS2MDC |
| Air pressure    | base_link origin       | 50 Hz, BMP390 |
| NavSat (GPS)    | base_link origin       | 30 Hz |
| Fisheye front   | (0.05, 0, 0.25, 0, 0, 0)        | wide_angle_camera 800×800, hfov 180°, lens equidistant, cutoff 90° |
| Fisheye rear    | (-0.05, 0, 0.25, 0, 0, π)       | same as front |
| Depth front     | (0, 0, 0.30, 0, 0, 0)           | depth_camera 480×480 R_FLOAT32, hfov 90°, near 0.1, far 30 |
| Depth right     | (0, 0, 0.30, 0, 0, -π/2)        | same |
| Depth rear      | (0, 0, 0.30, 0, 0, +π)          | same |
| Depth left      | (0, 0, 0.30, 0, 0, +π/2)        | same |

### Camera intrinsics (derived)

**Fisheye (equidistant model)**
```
image: 800 × 800
hfov:   π   rad
r_max = 400 px at θ_max = π/2
f_fish = r_max / θ_max = 400 / (π/2) = 254.65 px/rad
pixel (u,v) -> bearing:
    du = u - 400,  dv = v - 400
    r  = sqrt(du² + dv²)            (px)
    θ  = r / f_fish                  (rad from optical axis)
    φ  = atan2(-dv, du)              (azimuth around +X)
    direction (optical, X right, Y down, Z fwd) =
        ( sin θ cos φ, -sin θ sin φ, cos θ )
```

**Depth perspective**
```
image: 480 × 480
hfov:  π/2 rad
fx = fy = 0.5 · 480 / tan(π/4) = 240
cx = cy = 240
pixel  (u, v, depth d) -> 3D in optical frame
    X = (u - cx) · d / fx
    Y = (v - cy) · d / fy
    Z = d
```

### Optical ↔ base_link transform
gz `depth_camera` and `wide_angle_camera` optical frame:
**X right, Y down, Z forward**.  base_link is X forward, Y left, Z up.
Conversion:
```
xb =  zc
yb = -xc
zb = -yc
```

## PX4 airframe — `4022_gz_rover_360cam`

- ROMFS file: `ROMFS/px4fmu_common/init.d-posix/airframes/4022_gz_rover_360cam`
- Sourced: `rc.rover_differential_defaults`
- Key params:
  - `RD_WHEEL_TRACK 0.6`
  - `RO_MAX_THR_SPEED 2.1`
  - `RO_SPEED_LIM 2.0`
  - `RO_YAW_RATE_LIM 180 deg/s`
  - `RD_TRANS_DRV_TRN  20°`
  - actuator 0 = right wheel, 1 = left wheel (matched in SDF JointController plugins)
- Spawned model name: **`rover_360cam_0`** (PX4 appends `_0` per instance).

## Topic map (gz)

| Direction | Topic | Type | Used by |
|-----------|-------|------|---------|
| pub | `/fisheye_front/image` | gz.msgs.Image (R8G8B8) | webui MJPEG, YOLO |
| pub | `/fisheye_rear/image`  | gz.msgs.Image | webui MJPEG, YOLO |
| pub | `/depth_front` `/depth_right` `/depth_rear` `/depth_left` | gz.msgs.Image (R_FLOAT32) | webui voxel cloud, obstacle wedge, landmark resolver |
| pub | `/world/slam_obstacles/dynamic_pose/info` | gz.msgs.Pose_V | webui rover pose |
| sub | `/model/rover_360cam_0/command/motor_speed` | gz.msgs.Actuators | direct wheel pubs (now unused — PX4 owns wheels via OFFBOARD) |

## HTTP map (webui + detector)

| URL | Method | Body |
|-----|--------|------|
| `localhost:8080/` | GET | HTML SPA |
| `localhost:8080/front.mjpeg` `/rear.mjpeg` | GET | multipart JPEG |
| `localhost:8080/pose.json` | GET | `{x,y,z,qx,qy,qz,qw,seq}` |
| `localhost:8080/cloud.bin` | GET | float32 XYZ stream, header X-Cloud-Seq |
| `localhost:8080/target` | POST `?x=&y=` or `?clear=1` | nav waypoint |
| `localhost:8080/target.json` | GET | `{x,y,active,seq}` |
| `localhost:8080/investigate` | POST `?cls=` or `?clear=1` or `?state=` | investigation control |
| `localhost:8080/investigate.json` | GET | `{cls,state,active,seq}` |
| `localhost:8080/config.json` | GET | full config dict |
| `localhost:8080/config` | POST `?key=val` | one-shot tune |
| `localhost:8080/landmarks.json` | GET | `{items:[{cls,x,y,count,last_seen_ms}]}` |
| `localhost:8080/landmarks/clear` | POST | wipe memory |
| `localhost:8080/obstacles.json` | GET | `{front,right,rear,left}` (min depth m) |
| `localhost:8081/front/detect.mjpeg` `/rear/detect.mjpeg` | GET | YOLO annotated |
| `localhost:8081/front/detections.json` `/rear/detections.json` | GET | `{detections:[{id,cls,conf,box=[x,y,w,h]}]}` |

## Landmark resolver pipeline (current)

```
YOLO bbox (cls, x, y, w, h, conf)        [fisheye 800×800]
        │
        ▼  pick sample pixel = (x + w/2, y + h*0.80)   (ground-anchored)
fisheye pixel  ──equidistant→ optical-frame direction (3-D unit vector)
        │
        ▼  pinhole-project into depth-front (480×480)
depth pixel  ──6×6 patch median, IQR < 1.5 m──> range d_m
        │
        ▼  optical dir × d_m  →  3-D point in optical frame
optical → base_link → world (rover pose quaternion)
        │
        ▼  reject if conf < 0.4, range > 12 m, outside ±11 m world bounds
add_landmark(cls, xw, yw)
        │
        ▼  candidate pool: needs ≥3 hits within 0.8 m radius in 8 s
        ▼  promoted landmark merged to existing within 0.6 m
        ▼  capped at 200, oldest dropped
```

## Behavior state machine

```
IDLE — invest cleared — only entered briefly during boot
 │
 ▼  invest.active=false       investigator default
EXPLORE — obstacle-aware lawnmower
   forward 5 s  ─if front_clear < 1.6 m─►  turn 2 s
                ─if 1.6 m < front < 3 m→  scale fwd by clearance
 │
 ▼  invest.active=true
SEARCH ──YOLO match──► APPROACH ──bbox > 30 % frame──► FOLLOW
                          ▲                              │
                          └───── lost > 4 s ──────────────┘
```

All wheel control routes through PX4 OFFBOARD `SET_POSITION_TARGET_LOCAL_NED`
(position + yaw setpoints), so the same code runs on a real Pixhawk-class
rover with no modification.
