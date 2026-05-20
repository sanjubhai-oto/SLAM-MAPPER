# slam_rover Master Plan — Accurate 3D + Object Mapping

Living doc. Read this first when resuming work. Updated 2026-05-20.

## North Star
Rover drives any environment (sim or real) → outputs:
- Accurate 3D geometry matching ground truth ≤ 0.10m wall RMSE
- Per-object: correct class, position (≤ 0.20m), oriented bbox (size+yaw), textured mesh
- Live on M-series CPU, deployable on Orin Nano at 10-15 Hz

## Status as of 2026-05-20 morning

### What works
- C++ webui w/ TSDF voxel + marching-cubes + InstancedMesh + Gaussian-splat shader + Inria-format .splat export
- YOLOv11n COCO classes + canonicalize_class remap (surfboard→person, vase→cone, chair→cone, etc.)
- gz factory world (42 GT objects: walls, pillars×5, boxes×5, cones×8, persons×4, shelves×2, pallets×2, drums×3, forklift, crates×2, tree×2, bench, sign, platform, worker)
- gz_invest.py direct wheel control + obstacle gate (front<0.7m → reverse+clear)
- gz_explore_agent.py Yamauchi frontier exploration
- agent_brain.py decision-driven agentic AI (STUCK/WALL/ORBIT/SAVE rules)
- Telemetry HUD: pos/att/vel/dist/uptime/detections/landmarks/classes/seq
- Splat daemon writes /tmp/.../latest.splat → webui serves /splat.bin → GaussianSplats3D viewer
- collect_yolo_dataset.py: 2000 auto-labeled frames via fisheye projection of GT AABBs → datasets/gz_v1/
- train_yolo_finetune.py: MPS fine-tune YOLOv11n on synthetic gz, freeze=10, lr=1e-3, epochs=20

### What's running RIGHT NOW
- PX4 SITL + gz Harmonic 8 (rover_360cam_slam_obstacles)
- webui :8080
- gz_invest direct wheel
- agent_brain (decision-driven AI)
- gz_detect (yolo11n.pt currently — will swap to slamrover_v1.pt after training)
- collect_yolo_dataset DONE (2000 frames written)
- **train_yolo_finetune RUNNING on MPS** — 20 epochs, ETA ~1.5h
- cloud_to_splat.py daemon (latest.splat every 5s)
- mavlink_telem.py daemon spawning (telemetry endpoint)

### Current accuracy
- 22 landmarks tracked, ~15 locked
- Position error: mean 1.3m, best ones < 0.5m (cone_SE 0.48m, person_c 0.36m, drum_3 0.71m)
- Class accuracy: ~30% (YOLO bias on synthetic textures, fix = fine-tune)
- Cloud: 1900 pts, walls present after gate loosen (xy 10.30 bound, min_hits=3)
- Mean class accuracy expected after fine-tune: 80%+

## File map

```
~/PX4-Autopilot/
├── Tools/simulation/gz/
│   ├── models/rover_360cam/model.sdf        # 4-wheel rover + 2 fisheye + 4 depth
│   └── worlds/slam_obstacles.sdf            # 42 GT obj factory world
├── ROMFS/.../airframes/4022_gz_rover_360cam # PX4 airframe + prearm bypass
└── slam_rover/
    ├── webui/src/main.cpp                   # ALL C++ logic (single TU)
    ├── webui/static/index.html              # Three.js UI
    ├── docs/MASTER_PLAN.md                  # this file
    ├── datasets/gz_v1/                      # auto-labeled YOLO training set
    ├── state/weights/                       # fine-tuned models
    ├── state/splats/                        # .splat history
    └── scripts/
        ├── gz_detect.py                     # YOLO detector
        ├── gz_investigate.py                # direct gz wheel motion
        ├── gz_invest_offboard.py            # PX4 OFFBOARD motion
        ├── gz_explore_agent.py              # frontier exploration
        ├── agent_brain.py                   # agentic AI decision brain
        ├── visit_objects.py                 # GT-cheating visit tour (debug only)
        ├── planned_explore.py               # S-pattern lawnmower
        ├── collect_yolo_dataset.py          # auto-labeler
        ├── train_yolo_finetune.py           # MPS fine-tune
        ├── cloud_to_splat.py                # Inria .splat exporter
        ├── mavlink_telem.py                 # MAVLink → /tmp/mavlink_telem.json
        └── tsdf_builder.py                  # legacy Open3D TSDF (DEPRECATED — /mesh.ply 404)
```

## Phases done

| Phase | Status | What |
|-------|--------|------|
| 0 | ✅ | Marching cubes mesh from TSDF |
| 1 | ✅ | YOLOv11n swap + COCO remap |
| 2a | ✅ | Auto-labeled dataset (2000 frames) |
| 2b | ⏳ | YOLO fine-tune running ~1.5h |
| 3a | partial | Gaussian splat export pipeline live |
| 4 | ✅ | Agentic brain + obstacle gate |
| 5 | ⏳ | MAVLink telemetry agent spawning |

## Phases pending

| Phase | Goal | Effort |
|-------|------|--------|
| 2c | Swap detector to slamrover_v1.pt after training | 5 min |
| 3b | Per-object Poisson mesh extraction from voxel clusters | 4h |
| 6 | MASt3R-SLAM for honest pose (kill gz GT cheat) | 8h |
| 7 | Loop closure DBoW3 + iSAM2 PGO | 12h |
| 8 | MCAP recorder + offline 3DGS beautify | 8h |
| 9 | ONNX/TensorRT export for Orin Nano deploy | 3h |

## Lessons learned this session

### Class bias on synthetic gz
YOLO-World fixates on whichever class prompts are listed (e.g. "box" → labels everything box). Use closed-vocab YOLOv11n + remap table instead. Real fix: fine-tune.

### Object positions vs classes
Position is mostly geometric (depth + bbox-centroid + pose). Class is mostly perceptual (CNN logits). Therefore: position accurate even when class wrong. Don't conflate.

### Bouncing 4-wheel chassis
PX4 differential controller hits 4-wheel rover too hard → z bounces 0-1m. Skip cloud integration when z>0.5 OR |qx,qy|>0.20. Direct gz wheel is smoother.

### Walls need approach
Walls at ±10. xy bound 10.05 was too tight. 10.30 lets wall surface through. Rover must drive within 5-8m to see wall depth at z=0.3-1.7.

### Mountain mesh = stale 542MB /mesh.ply
Old tsdf_builder.py output never cleared. Delete /tmp/slam_rover_mesh.ply.

### OFFBOARD prearm
Airframe 4022 patched with COM_PREARM_MODE=0 + CBRK_USB_CHK + COM_ARM_WO_GPS=1. Then magic 21196 force-arm works. main_mode=6 OFFBOARD reachable. Earlier failures were missing param reload.

### Real SLAM vs scripted
visit_objects.py = GT cheat. gz_explore_agent.py + agent_brain.py = real Yamauchi frontier + decision AI.

## Recovery checklist (when resuming after crash)

1. Read this file + ROBOTICS_LESSONS.md + slam_rover_session_may19_state.md
2. `pkill -9 -f "slam_rover_webui|gz_invest|brain|gz_detect|bin/px4|gz sim|make px4|cmake"`
3. `rm -f /tmp/px4_lock-0 /tmp/slam_rover_mesh.ply`
4. `cd ~/PX4-Autopilot && nohup make px4_sitl gz_rover_360cam_slam_obstacles > /tmp/px4_boot.log 2>&1 &`
5. Wait 25s, then:
   ```
   nohup ~/PX4-Autopilot/slam_rover/webui/build/slam_rover_webui --port 8080 > /tmp/webui.log 2>&1 &
   nohup /tmp/yoloenv/bin/python -u ~/PX4-Autopilot/slam_rover/scripts/gz_detect.py --model yolo11n.pt --device cpu --conf 0.20 > /tmp/yolo.log 2>&1 &
   nohup /opt/homebrew/bin/python3.14 -u ~/PX4-Autopilot/slam_rover/scripts/gz_investigate.py > /tmp/gz_invest.log 2>&1 &
   nohup python3 -u ~/PX4-Autopilot/slam_rover/scripts/agent_brain.py > /tmp/agent_brain.log 2>&1 &
   nohup python3 -u ~/PX4-Autopilot/slam_rover/scripts/cloud_to_splat.py > /tmp/splat.log 2>&1 &
   nohup python3 -u ~/PX4-Autopilot/slam_rover/scripts/mavlink_telem.py > /tmp/telem.log 2>&1 &
   ```
6. Browser: http://localhost:8080. Toggle `gz truth`, `splat-view`, `cloud`, `tri-mesh` as desired.

## Open issues

- 4-wheel rover physics bounces under PX4 OFFBOARD; chose direct gz wheel for now
- Tracking config sliders POST to /config but motion controllers don't always re-read live
- Splat positions occasionally drift when rover bounces (z>0.5 hits)
- Wall holes where rover hasn't been within 5-8m
- Class accuracy bottlenecked until fine-tune completes
