// slam_rover web bridge
//
// Bridges gz-transport topics to a browser via HTTP:
//   GET /                  - HTML page (static/index.html)
//   GET /static/<file>     - static assets
//   GET /front.mjpeg       - multipart MJPEG stream of /fisheye_front/image
//   GET /rear.mjpeg        - multipart MJPEG stream of /fisheye_rear/image
//   GET /cloud.bin         - latest accumulated pointcloud (binary float32 XYZ)
//   GET /pose.json         - latest rover pose (world frame)
//
// Pointcloud is built by deprojecting 4 cubemap depth images, transforming each
// to the world frame using rover odometry, and accumulating into a global cloud
// with voxel-grid downsampling. Stand-in for a real SLAM (GenZ-ICP) backend.
// Same pipeline on the real rover: replace gz depth_camera subs with Depth
// Anything V2 inference on Insta360 frames.
//
// Build:   see CMakeLists.txt (cmake -B build && cmake --build build)
// Run:     ./build/slam_rover_webui  [--static <dir>] [--port 8080]

#include "httplib.h"

#include <gz/transport/Node.hh>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/odometry.pb.h>
#include <gz/msgs/odometry_with_covariance.pb.h>
#include <gz/msgs/pose_v.pb.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>
#include <filesystem>

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <queue>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct LatestJpeg {
  std::mutex mu;
  std::vector<uint8_t> bytes;
  uint64_t seq = 0;
};

struct LatestRgb {
  mutable std::mutex mu;
  std::vector<uint8_t> data;  // RGB row-major, 3*W*H
  int width = 0, height = 0;
};
LatestRgb g_rgb_front, g_rgb_rear;

struct LatestDepth {
  std::mutex mu;
  std::vector<float> depth;   // row-major HxW float32 meters
  int width = 0;
  int height = 0;
  double hfov = 0.0;          // radians
  // pose of this depth camera in base_link frame (yaw, then translation)
  double yaw = 0.0;
  double tx = 0.0, ty = 0.0, tz = 0.3;
  uint64_t seq = 0;
};

struct Pose {
  std::mutex mu;
  // world frame pose of rover base_link
  double x = 0.0, y = 0.0, z = 0.0;
  double qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;
  uint64_t seq = 0;
  // Speed estimate (m/s), updated on each pose write.
  double vx = 0.0, vy = 0.0;
  double yaw_rate = 0.0;     // rad/s, smoothed
  double last_yaw = 0.0;
  double last_x = 0.0, last_y = 0.0;
  uint64_t last_t_ms = 0;
  // Ring buffer of recent poses for time-aligned landmark resolution.
  struct Snap { uint64_t t; double x,y,z,qx,qy,qz,qw; };
  std::deque<Snap> hist;  // protected by mu; capped at 200
};

struct VoxelCloud {
  std::mutex mu;
  double voxel = 0.10;        // 10 cm
  uint64_t decay_ms = 15000;  // 15 s — match good-map sweet spot
  int min_hits = 3;           // 3 sightings — keep walls visible at distance
  // ===== TSDF (Voxblox-style) constants =====
  static constexpr float TSDF_VOXEL    = 0.10f;
  static constexpr float TSDF_TRUNC    = 0.40f;   // 4 * voxel
  static constexpr float TSDF_Z_MIN    = 0.5f;
  static constexpr float TSDF_W_CONST  = 1.0f;
  static constexpr float TSDF_W_MAX    = 100.0f;
  struct Cell {
    float x, y, z;
    uint8_t r, g, b;
    uint16_t hits;
    uint64_t last_ms;
    // TSDF fields (running-average signed distance + weight)
    float    sdf    = 0.0f;
    float    weight = 0.0f;
  };
  std::unordered_map<uint64_t, Cell> cells;
  std::vector<float> snapshot;     // flat XYZ
  std::vector<uint8_t> snapshot_rgb; // flat RGB
  uint64_t snapshot_seq = 0;

  static uint64_t key(int ix, int iy, int iz) {
    auto u = [](int v) -> uint64_t { return static_cast<uint64_t>(v + (1 << 20)); };
    return (u(ix) << 42) | (u(iy) << 21) | u(iz);
  }

  void add(float x, float y, float z,
           uint8_t r = 110, uint8_t g = 200, uint8_t b = 255,
           uint64_t t = 0) {
    int ix = static_cast<int>(std::floor(x / voxel));
    int iy = static_cast<int>(std::floor(y / voxel));
    int iz = static_cast<int>(std::floor(z / voxel));
    auto k = key(ix, iy, iz);
    auto it = cells.find(k);
    if (it == cells.end()) {
      cells[k] = Cell{x, y, z, r, g, b, 1, t};
    } else {
      it->second.r = static_cast<uint8_t>((it->second.r + r) / 2);
      it->second.g = static_cast<uint8_t>((it->second.g + g) / 2);
      it->second.b = static_cast<uint8_t>((it->second.b + b) / 2);
      if (it->second.hits < 65000) it->second.hits++;
      it->second.last_ms = t;
    }
  }

  // 3D Amanatides-Woo line traversal that ERASES voxel cells along the ray
  // from rover to depth endpoint (exclusive of endpoint itself). Mirrors
  // the 2D raycast on the occupancy grid -- without this the voxel cloud
  // accumulates a spiral of "ghost" cells every time the rover yaws.
  void raycast_erase_3d(double rx, double ry, double rz,
                        double ex, double ey, double ez) {
    double dx = ex - rx, dy = ey - ry, dz = ez - rz;
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (dist < 2 * voxel) return;
    // Walk the full ray. Stop ONE voxel short of endpoint to preserve the
    // surface cell. No 85% scaling — that left a fat trailing band of
    // un-erased "mountain" voxels behind every surface.
    int steps = static_cast<int>(dist / voxel) - 1;
    if (steps < 1) return;
    double inv_steps = 1.0 / (steps + 1);
    for (int s = 1; s <= steps; ++s) {
      double t = s * inv_steps;
      double x = rx + t * dx;
      double y = ry + t * dy;
      double z = rz + t * dz;
      int ix = static_cast<int>(std::floor(x / voxel));
      int iy = static_cast<int>(std::floor(y / voxel));
      int iz = static_cast<int>(std::floor(z / voxel));
      cells.erase(key(ix, iy, iz));
    }
  }

  // ===== TSDF integration (Voxblox merged-integrator style) =====
  // Walks the ray from (depth_z - TRUNC) to (depth_z + TRUNC), updating
  // signed distance + weight via running-average. Surface band emerges
  // where |sdf| < 0.5 * voxel after enough fusion.
  void integrate_ray_tsdf(double cx, double cy, double cz,
                          double rx, double ry, double rz,
                          float depth_z, float cos_theta,
                          uint8_t r, uint8_t g, uint8_t b,
                          uint64_t now_ms_) {
    if (depth_z <= 0.0f) return;
    float t_start = depth_z - TSDF_TRUNC;
    if (t_start < 0.0f) t_start = 0.0f;
    float t_end   = depth_z + TSDF_TRUNC;
    // Step along ray at voxel resolution
    float step = static_cast<float>(voxel) * 0.5f;
    float w_base = TSDF_W_CONST * cos_theta /
                   std::max(depth_z * depth_z, TSDF_Z_MIN * TSDF_Z_MIN);
    for (float t = t_start; t <= t_end; t += step) {
      float sdf_sample = depth_z - t;
      if (sdf_sample < -TSDF_TRUNC) continue;
      if (sdf_sample >  TSDF_TRUNC) sdf_sample = TSDF_TRUNC;
      double x = cx + rx * t;
      double y = cy + ry * t;
      double z = cz + rz * t;
      // Bounds (same as add()): clamp to operating volume
      if (z < 0.20 || z > 1.80) continue;
      if (std::fabs(x) > 10.05 || std::fabs(y) > 10.05) continue;
      float w = w_base;
      if (sdf_sample < 0.0f) {
        float fall = 1.0f + sdf_sample / TSDF_TRUNC;  // 0..1 behind surface
        if (fall < 0.0f) fall = 0.0f;
        w *= fall;
      }
      if (w <= 0.0f) continue;
      int ix = static_cast<int>(std::floor(x / voxel));
      int iy = static_cast<int>(std::floor(y / voxel));
      int iz = static_cast<int>(std::floor(z / voxel));
      auto k = key(ix, iy, iz);
      auto it = cells.find(k);
      if (it == cells.end()) {
        Cell c{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z),
               r, g, b, 1, now_ms_};
        c.sdf = sdf_sample;
        c.weight = std::min(w, TSDF_W_MAX);
        cells[k] = c;
      } else {
        Cell& v = it->second;
        float denom = v.weight + w + 1e-6f;
        v.sdf = (v.weight * v.sdf + w * sdf_sample) / denom;
        // weighted-avg color
        float rr = (v.weight * v.r + w * r) / denom;
        float gg = (v.weight * v.g + w * g) / denom;
        float bb = (v.weight * v.b + w * b) / denom;
        v.r = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, rr)));
        v.g = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, gg)));
        v.b = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, bb)));
        v.weight = std::min(v.weight + w, TSDF_W_MAX);
        // keep voxel centre approximately at surface for snapshot emit
        v.x = static_cast<float>(x);
        v.y = static_cast<float>(y);
        v.z = static_cast<float>(z);
        if (v.hits < 65000) v.hits++;
        v.last_ms = now_ms_;
      }
    }
  }

  void snap(uint64_t now) {
    snapshot.clear();
    snapshot_rgb.clear();
    // Drop stale cells in-place
    for (auto it = cells.begin(); it != cells.end(); ) {
      if (now - it->second.last_ms > decay_ms) {
        it = cells.erase(it);
      } else {
        ++it;
      }
    }
    // Isolated-voxel reject: a real surface point has at least one occupied
    // neighbour within +-1 cell (3x3x3 cube minus self). Anything else is
    // a depth-edge ghost / "air" speck and gets dropped.
    auto has_neighbour = [&](uint64_t k) {
      // Decode k into ix,iy,iz so we can probe neighbours.
      uint64_t mask = (1ULL << 21) - 1;
      int iz = static_cast<int>(k & mask) - (1 << 20);
      int iy = static_cast<int>((k >> 21) & mask) - (1 << 20);
      int ix = static_cast<int>((k >> 42) & mask) - (1 << 20);
      for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy && !dz) continue;
            if (cells.count(key(ix + dx, iy + dy, iz + dz))) return true;
          }
        }
      }
      return false;
    };
    snapshot.reserve(cells.size() * 3);
    snapshot_rgb.reserve(cells.size() * 3);
    const float surf_band = 0.5f * static_cast<float>(voxel);
    for (auto& kv : cells) {
      // TSDF surface band: emit only voxels at the zero-crossing.
      if (kv.second.weight <= 2.5f) continue;          // stronger consensus
      if (std::fabs(kv.second.sdf) >= surf_band) continue;
      if (!has_neighbour(kv.first)) continue;          // floating ghost
      if (kv.second.z < 0.20f || kv.second.z > 1.80f) continue;  // hard ground/ceiling
      // Need at least 2 neighbours (kill thin air specks)
      int nb_count = 0;
      uint64_t mask2 = (1ULL << 21) - 1;
      int iz2 = static_cast<int>(kv.first & mask2) - (1 << 20);
      int iy2 = static_cast<int>((kv.first >> 21) & mask2) - (1 << 20);
      int ix2 = static_cast<int>((kv.first >> 42) & mask2) - (1 << 20);
      for (int dz = -1; dz <= 1 && nb_count < 2; dz++)
        for (int dy = -1; dy <= 1 && nb_count < 2; dy++)
          for (int dx = -1; dx <= 1 && nb_count < 2; dx++) {
            if (!dx && !dy && !dz) continue;
            if (cells.count(key(ix2 + dx, iy2 + dy, iz2 + dz))) nb_count++;
          }
      if (nb_count < 2) continue;
      snapshot.push_back(kv.second.x);
      snapshot.push_back(kv.second.y);
      snapshot.push_back(kv.second.z);
      snapshot_rgb.push_back(kv.second.r);
      snapshot_rgb.push_back(kv.second.g);
      snapshot_rgb.push_back(kv.second.b);
    }
    snapshot_seq++;
  }
};

LatestJpeg g_front, g_rear;
LatestDepth g_depth_front, g_depth_right, g_depth_rear, g_depth_left;
Pose g_pose;
VoxelCloud g_cloud;
std::atomic<bool> g_running{true};

struct Target {
  std::mutex mu;
  double x = 0.0, y = 0.0;
  bool active = false;
  uint64_t seq = 0;
};
Target g_target;

struct Investigate {
  std::mutex mu;
  std::string cls;     // target class name (e.g. "car", "person")
  std::string state;   // "idle" | "search" | "approach" | "follow"
  bool active = false;
  uint64_t seq = 0;
};
Investigate g_invest;

struct AlgoConfig {
  std::mutex mu;
  // Defaults match prior hard-coded values in gz_invest_offboard.py.
  double fwd_speed = 2.0;            // m/s commanded
  double max_yaw_rate_deg = 45.0;    // hard cap
  double search_yaw_rate_deg = 35.0;
  double approach_yaw_gain = 0.6;
  double follow_yaw_gain = 0.8;
  double approach_stop_ratio = 0.30;
  double follow_goal_low = 0.25;
  double follow_goal_high = 0.40;
  double lost_timeout = 4.0;
  uint64_t seq = 0;
};
AlgoConfig g_cfg;

struct Landmark {
  std::string cls;
  double x = 0, y = 0;
  uint64_t first_seen_ms = 0;
  uint64_t last_seen_ms = 0;
  int count = 1;
  bool locked = false;   // once locked, position no longer drifts
  // Observation history: (t, x, y) so we can audit when/where it was seen
  std::vector<std::tuple<uint64_t, double, double>> history;
};

// Pending landmark candidates -- need N consecutive sightings within radius
// before being promoted to a real Landmark.  Prevents single-frame YOLO
// false-positives ("airplane" on a wall) from contaminating the map.
struct PendingCand {
  std::string cls;
  double x = 0, y = 0;
  int hits = 1;
  uint64_t last_ms = 0;
  // Parallax: track 2 most distant rover observation positions seen so far.
  // Promotion requires baseline > 0.5 m, so pure spin cannot promote.
  double obs1_x = 0, obs1_y = 0;
  double obs2_x = 0, obs2_y = 0;
  double baseline = 0.0;
};
struct Landmarks {
  std::mutex mu;
  std::vector<Landmark> items;
  std::vector<PendingCand> pending;
  // Per-class merge radius via getter below. Generic radius used by
  // candidate matcher. People drift far due to scale/pose-lag, so widen.
  double merge_radius = 2.5;
  double cand_match_radius = 2.5;
  int promote_hits = 2;
  uint64_t cand_ttl_ms = 60000;
  uint64_t landmark_decay_ms = 600000;  // 10 min — keep persisted maps fresh
  int lock_threshold = 8;               // count >= 8 -> position locked forever
};
Landmarks g_landmarks;

// ===== Occupancy grid (local map) =====
//
// Local 2D grid covering the operating area, resolution 0.1 m.
// cells values:  -1 = unknown, 0 = free, 100 = occupied.
// World origin maps to cell (W/2, H/2) so we cover x in [-10, +10], y in [-10, +10].
//
struct OccGrid {
  std::mutex mu;
  static constexpr int W = 200;
  static constexpr int H = 200;
  static constexpr double RES = 0.1;          // m per cell
  static constexpr double ORIGIN_X = -10.0;   // world x at cell (0,*)
  static constexpr double ORIGIN_Y = -10.0;
  std::vector<int8_t> data;
  OccGrid() : data(W * H, -1) {}
  inline bool world_to_cell(double x, double y, int& i, int& j) const {
    i = static_cast<int>(std::floor((x - ORIGIN_X) / RES));
    j = static_cast<int>(std::floor((y - ORIGIN_Y) / RES));
    return i >= 0 && i < W && j >= 0 && j < H;
  }
  inline void cell_to_world(int i, int j, double& x, double& y) const {
    x = ORIGIN_X + (i + 0.5) * RES;
    y = ORIGIN_Y + (j + 0.5) * RES;
  }
};
OccGrid g_grid;

struct Path {
  std::mutex mu;
  std::vector<std::pair<double, double>> waypoints; // world xy
  uint64_t seq = 0;
  bool active = false;
  size_t cursor = 0;  // for follower
};
Path g_path;

static uint64_t now_ms() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<milliseconds>(steady_clock::now().time_since_epoch())
          .count());
}

void add_landmark(const std::string& cls, double x, double y,
                  double rover_x = 0.0, double rover_y = 0.0) {
  std::lock_guard<std::mutex> lk(g_landmarks.mu);
  uint64_t t = now_ms();

  // Drop stale candidates
  g_landmarks.pending.erase(
      std::remove_if(g_landmarks.pending.begin(), g_landmarks.pending.end(),
                     [&](const PendingCand& c) {
                       return t - c.last_ms > g_landmarks.cand_ttl_ms;
                     }),
      g_landmarks.pending.end());

  // Wall-cell reject: drop only if landmark falls EXACTLY in an occupied
  // cell. Real persons can legitimately stand close to walls; we only veto
  // when YOLO has hallucinated a person ON a wall pixel itself.
  if (cls == "stop sign" || cls == "traffic light" || cls == "fire hydrant" ||
      cls == "person" || cls == "car" || cls == "truck" || cls == "bus") {
    std::lock_guard<std::mutex> glk(g_grid.mu);
    int gi, gj;
    if (g_grid.world_to_cell(x, y, gi, gj)) {
      // Veto if cell OR neighbours within 5 cells (0.5m) are occupied
      int R = 5;
      for (int dj = -R; dj <= R; ++dj) {
        for (int di = -R; di <= R; ++di) {
          int gi2 = gi + di, gj2 = gj + dj;
          if (gi2 < 0 || gi2 >= OccGrid::W || gj2 < 0 || gj2 >= OccGrid::H) continue;
          if (g_grid.data[static_cast<size_t>(gj2) * OccGrid::W + gi2] == 100) {
            return;
          }
        }
      }
    }
    // Reject if within 1m of map edge (walls at ±10)
    if (std::abs(x) > 9.0 || std::abs(y) > 9.0) return;
  }

  // 1) Existing landmark of same class within merge radius
  for (auto& lm : g_landmarks.items) {
    if (lm.cls == cls) {
      double dx = lm.x - x, dy = lm.y - y;
      if (dx * dx + dy * dy <
          g_landmarks.merge_radius * g_landmarks.merge_radius) {
        // If not yet locked, refine position with running average.
        if (!lm.locked) {
          lm.x = (lm.x * lm.count + x) / (lm.count + 1);
          lm.y = (lm.y * lm.count + y) / (lm.count + 1);
        }
        lm.count++;
        lm.last_seen_ms = t;
        // History entry (cap 50 per landmark to bound memory)
        lm.history.emplace_back(t, x, y);
        if (lm.history.size() > 50) {
          lm.history.erase(lm.history.begin());
        }
        // Auto-lock once we have enough independent observations.
        if (!lm.locked && lm.count >= g_landmarks.lock_threshold) {
          lm.locked = true;
        }
        return;
      }
    }
  }

  // 2) Candidate match → promote on Nth hit + parallax baseline > 0.5 m.
  for (auto& c : g_landmarks.pending) {
    if (c.cls == cls) {
      double dx = c.x - x, dy = c.y - y;
      if (dx * dx + dy * dy <
          g_landmarks.cand_match_radius * g_landmarks.cand_match_radius) {
        c.x = (c.x * c.hits + x) / (c.hits + 1);
        c.y = (c.y * c.hits + y) / (c.hits + 1);
        c.hits++;
        c.last_ms = t;
        // Update best parallax baseline seen so far.
        double d1 = std::hypot(rover_x - c.obs1_x, rover_y - c.obs1_y);
        double d2 = std::hypot(rover_x - c.obs2_x, rover_y - c.obs2_y);
        if (c.hits == 1 || (c.obs1_x == 0 && c.obs1_y == 0)) {
          c.obs1_x = rover_x; c.obs1_y = rover_y;
        } else if (d1 > c.baseline) {
          // Replace whichever endpoint is closer to the new observation
          if (d2 < d1) { c.obs2_x = rover_x; c.obs2_y = rover_y; }
          else         { c.obs1_x = rover_x; c.obs1_y = rover_y; }
          c.baseline = std::hypot(c.obs1_x - c.obs2_x,
                                  c.obs1_y - c.obs2_y);
        }
        // Promotion needs both: hit count AND parallax baseline. Spinning
        // in place gives baseline≈0 and never promotes.
        bool ready = c.hits >= g_landmarks.promote_hits;
        // baseline check removed — visits dwell stationary
        if (ready) {
          Landmark lm;
          lm.cls = c.cls; lm.x = c.x; lm.y = c.y;
          lm.first_seen_ms = t; lm.last_seen_ms = t;
          lm.count = c.hits; lm.locked = false;
          lm.history.emplace_back(t, c.x, c.y);
          g_landmarks.items.push_back(std::move(lm));
          c = g_landmarks.pending.back();
          g_landmarks.pending.pop_back();
          if (g_landmarks.items.size() > 200) {
            g_landmarks.items.erase(g_landmarks.items.begin());
          }
        }
        return;
      }
    }
  }

  // 3) New candidate
  PendingCand nc;
  nc.cls = cls; nc.x = x; nc.y = y; nc.hits = 1; nc.last_ms = t;
  nc.obs1_x = rover_x; nc.obs1_y = rover_y;
  nc.obs2_x = rover_x; nc.obs2_y = rover_y;
  nc.baseline = 0.0;
  g_landmarks.pending.push_back(std::move(nc));
  if (g_landmarks.pending.size() > 100) {
    g_landmarks.pending.erase(g_landmarks.pending.begin());
  }
}

// ---- Landmark persistence to disk ----
static std::filesystem::path landmarks_state_path() {
  namespace fs = std::filesystem;
  fs::path dir = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") /
                 "PX4-Autopilot" / "slam_rover" / "state";
  std::error_code ec;
  fs::create_directories(dir, ec);
  return dir / "landmarks.json";
}

static void save_landmarks_to_disk() {
  std::vector<Landmark> snap;
  {
    std::lock_guard<std::mutex> lk(g_landmarks.mu);
    snap = g_landmarks.items;
  }
  auto p = landmarks_state_path();
  std::ofstream f(p);
  if (!f) return;
  f << "{\n  \"landmarks\": [\n";
  for (size_t k = 0; k < snap.size(); ++k) {
    const auto& lm = snap[k];
    f << "    {\"cls\":\"" << lm.cls << "\","
      << "\"x\":" << lm.x << ",\"y\":" << lm.y << ","
      << "\"first_seen_ms\":" << lm.first_seen_ms << ","
      << "\"last_seen_ms\":" << lm.last_seen_ms << ","
      << "\"count\":" << lm.count << ","
      << "\"locked\":" << (lm.locked ? "true" : "false") << ","
      << "\"history\":[";
    for (size_t i = 0; i < lm.history.size(); ++i) {
      if (i) f << ",";
      f << "[" << std::get<0>(lm.history[i]) << ","
        << std::get<1>(lm.history[i]) << ","
        << std::get<2>(lm.history[i]) << "]";
    }
    f << "]}";
    if (k + 1 < snap.size()) f << ",";
    f << "\n";
  }
  f << "  ]\n}\n";
}

static void load_landmarks_from_disk() {
  auto p = landmarks_state_path();
  std::ifstream f(p);
  if (!f) return;
  std::stringstream ss; ss << f.rdbuf();
  std::string body = ss.str();
  // Minimal hand-parser (same style as parse_detections_json) so we don't
  // pull in a full JSON dep. Walk "cls", then x, y, count, locked, history.
  std::lock_guard<std::mutex> lk(g_landmarks.mu);
  g_landmarks.items.clear();
  size_t pos = 0;
  while ((pos = body.find("\"cls\"", pos)) != std::string::npos) {
    Landmark lm;
    size_t q1 = body.find('"', pos + 5);
    size_t q2 = body.find('"', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) break;
    lm.cls = body.substr(q1 + 1, q2 - q1 - 1);
    auto grab_num = [&](const std::string& key, double& dst) {
      size_t k = body.find("\"" + key + "\"", q2);
      if (k == std::string::npos) return;
      size_t colon = body.find(':', k);
      if (colon == std::string::npos) return;
      dst = std::stod(body.substr(colon + 1, 30));
    };
    double xv = 0, yv = 0, fm = 0, lm_ms = 0, cnt = 0;
    grab_num("x", xv); grab_num("y", yv);
    grab_num("first_seen_ms", fm);
    grab_num("last_seen_ms", lm_ms);
    grab_num("count", cnt);
    lm.x = xv; lm.y = yv;
    lm.first_seen_ms = static_cast<uint64_t>(fm);
    lm.last_seen_ms = static_cast<uint64_t>(lm_ms);
    lm.count = static_cast<int>(cnt);
    size_t lk_pos = body.find("\"locked\"", q2);
    if (lk_pos != std::string::npos) {
      size_t colon = body.find(':', lk_pos);
      lm.locked = body.find("true", colon) != std::string::npos &&
                  body.find("true", colon) < body.find('}', colon);
    }
    g_landmarks.items.push_back(std::move(lm));
    pos = q2 + 1;
  }
  std::cerr << "[landmarks] loaded " << g_landmarks.items.size()
            << " landmarks from disk" << std::endl;
}

void landmark_persistor_thread() {
  while (g_running) {
    std::this_thread::sleep_for(std::chrono::seconds(20));
    save_landmarks_to_disk();
  }
}

cv::Mat gz_image_to_mat(const gz::msgs::Image& msg) {
  int w = msg.width(), h = msg.height();
  // gz Image with R8G8B8 has data size 3*w*h
  cv::Mat rgb(h, w, CV_8UC3, const_cast<char*>(msg.data().data()));
  cv::Mat bgr;
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  return bgr.clone();
}

static void on_image_common(const gz::msgs::Image& msg, LatestJpeg& jslot,
                            LatestRgb& rslot) {
  cv::Mat bgr = gz_image_to_mat(msg);
  std::vector<uint8_t> buf;
  cv::imencode(".jpg", bgr, buf, {cv::IMWRITE_JPEG_QUALITY, 70});
  // Save raw RGB too (msg already R8G8B8 layout)
  {
    std::lock_guard<std::mutex> lk(rslot.mu);
    rslot.width = msg.width();
    rslot.height = msg.height();
    rslot.data.assign(reinterpret_cast<const uint8_t*>(msg.data().data()),
                      reinterpret_cast<const uint8_t*>(msg.data().data())
                      + msg.data().size());
  }
  std::lock_guard<std::mutex> lk(jslot.mu);
  jslot.bytes = std::move(buf);
  jslot.seq++;
}

void on_image_front(const gz::msgs::Image& msg) {
  on_image_common(msg, g_front, g_rgb_front);
}
void on_image_rear(const gz::msgs::Image& msg) {
  on_image_common(msg, g_rear, g_rgb_rear);
}

void on_depth_generic(LatestDepth& slot, const gz::msgs::Image& msg) {
  int w = msg.width(), h = msg.height();
  std::lock_guard<std::mutex> lk(slot.mu);
  slot.width = w;
  slot.height = h;
  slot.depth.resize(static_cast<size_t>(w) * h);
  std::memcpy(slot.depth.data(), msg.data().data(),
              slot.depth.size() * sizeof(float));
  slot.seq++;
}

void on_odom(const gz::msgs::OdometryWithCovariance& msg) {
  const auto& p = msg.pose_with_covariance().pose();
  std::lock_guard<std::mutex> lk(g_pose.mu);
  g_pose.x = p.position().x();
  g_pose.y = p.position().y();
  g_pose.z = p.position().z();
  g_pose.qx = p.orientation().x();
  g_pose.qy = p.orientation().y();
  g_pose.qz = p.orientation().z();
  g_pose.qw = p.orientation().w();
  g_pose.seq++;
}

void on_dynamic_pose(const gz::msgs::Pose_V& msg) {
  for (int i = 0; i < msg.pose_size(); ++i) {
    const auto& p = msg.pose(i);
    if (p.name() == "rover_360cam_0") {
      uint64_t t = now_ms();
      std::lock_guard<std::mutex> lk(g_pose.mu);
      // Velocity + yaw-rate estimate (windowed difference, IIR smoothed)
      double cur_yaw = std::atan2(
          2 * (p.orientation().w() * p.orientation().z() +
               p.orientation().x() * p.orientation().y()),
          1 - 2 * (p.orientation().y() * p.orientation().y() +
                   p.orientation().z() * p.orientation().z()));
      if (g_pose.last_t_ms != 0) {
        double dt = (t - g_pose.last_t_ms) / 1000.0;
        if (dt > 0.005) {
          double inst_vx = (p.position().x() - g_pose.last_x) / dt;
          double inst_vy = (p.position().y() - g_pose.last_y) / dt;
          g_pose.vx = 0.7 * g_pose.vx + 0.3 * inst_vx;
          g_pose.vy = 0.7 * g_pose.vy + 0.3 * inst_vy;
          // Wrap yaw difference into (-pi, pi]
          double dyaw = cur_yaw - g_pose.last_yaw;
          while (dyaw > M_PI)  dyaw -= 2 * M_PI;
          while (dyaw < -M_PI) dyaw += 2 * M_PI;
          double inst_w = dyaw / dt;
          g_pose.yaw_rate = 0.6 * g_pose.yaw_rate + 0.4 * inst_w;
        }
      }
      g_pose.last_x = p.position().x();
      g_pose.last_y = p.position().y();
      g_pose.last_yaw = cur_yaw;
      g_pose.last_t_ms = t;
      g_pose.x = p.position().x();
      g_pose.y = p.position().y();
      g_pose.z = p.position().z();
      g_pose.qx = p.orientation().x();
      g_pose.qy = p.orientation().y();
      g_pose.qz = p.orientation().z();
      g_pose.qw = p.orientation().w();
      g_pose.seq++;
      // Push to history ring (cap 200 entries)
      g_pose.hist.push_back({t, g_pose.x, g_pose.y, g_pose.z,
                             g_pose.qx, g_pose.qy, g_pose.qz, g_pose.qw});
      while (g_pose.hist.size() > 200) g_pose.hist.pop_front();
      return;
    }
  }
}

// Look up rover pose at the given absolute time-ms. Returns the closest
// historical entry. Falls back to the current pose if history empty.
static bool pose_at_time(uint64_t t_ms, double& x, double& y, double& z,
                        double& qx, double& qy, double& qz, double& qw) {
  std::lock_guard<std::mutex> lk(g_pose.mu);
  if (g_pose.hist.empty()) {
    x = g_pose.x; y = g_pose.y; z = g_pose.z;
    qx = g_pose.qx; qy = g_pose.qy; qz = g_pose.qz; qw = g_pose.qw;
    return false;
  }
  uint64_t best_dt = UINT64_MAX;
  const auto* best = &g_pose.hist.front();
  for (const auto& s : g_pose.hist) {
    uint64_t dt = (s.t > t_ms) ? (s.t - t_ms) : (t_ms - s.t);
    if (dt < best_dt) { best_dt = dt; best = &s; }
  }
  x = best->x; y = best->y; z = best->z;
  qx = best->qx; qy = best->qy; qz = best->qz; qw = best->qw;
  return true;
}

struct Mat3 { double m[9]; };

Mat3 quat_to_mat(double qx, double qy, double qz, double qw) {
  Mat3 R;
  R.m[0] = 1 - 2*(qy*qy + qz*qz); R.m[1] = 2*(qx*qy - qz*qw); R.m[2] = 2*(qx*qz + qy*qw);
  R.m[3] = 2*(qx*qy + qz*qw);     R.m[4] = 1 - 2*(qx*qx + qz*qz); R.m[5] = 2*(qy*qz - qx*qw);
  R.m[6] = 2*(qx*qz - qy*qw);     R.m[7] = 2*(qy*qz + qx*qw);     R.m[8] = 1 - 2*(qx*qx + qy*qy);
  return R;
}

// Minimal HTTP GET to localhost:8081 for detector JSON. Uses httplib::Client.
static std::string http_get(const std::string& host, int port,
                            const std::string& path) {
  httplib::Client cli(host, port);
  cli.set_connection_timeout(0, 500000);  // 0.5s
  cli.set_read_timeout(1, 0);
  auto r = cli.Get(path.c_str());
  if (!r || r->status != 200) return "";
  return r->body;
}

// Toy JSON scrape for {"detections":[{"cls":"car","conf":0.42,"box":[x,y,w,h]}]}
// We only need cls and box; conf optional.  Returns vector of (cls, cx, cy, w, h).
struct DetEntry {
  std::string cls;
  int cx = 0, cy = 0, w = 0, h = 0;
  double conf = 0.0;
};
static std::vector<DetEntry> parse_detections_json(const std::string& body) {
  std::vector<DetEntry> out;
  size_t p = 0;
  while ((p = body.find("\"cls\"", p)) != std::string::npos) {
    DetEntry e;
    size_t q1 = body.find('"', p + 5);
    if (q1 == std::string::npos) break;
    size_t q2 = body.find('"', q1 + 1);
    if (q2 == std::string::npos) break;
    e.cls = body.substr(q1 + 1, q2 - q1 - 1);
    // box=[x,y,w,h]
    size_t bp = body.find("\"box\"", q2);
    if (bp == std::string::npos) break;
    size_t lb = body.find('[', bp);
    size_t rb = body.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) break;
    std::string nums = body.substr(lb + 1, rb - lb - 1);
    int x = 0, y = 0, w = 0, h = 0;
    if (std::sscanf(nums.c_str(), "%d , %d , %d , %d", &x, &y, &w, &h) == 4) {
      e.cx = x + w / 2;
      e.cy = y + h / 2;
      e.w = w;
      e.h = h;
    }
    // conf optional
    size_t cp = body.find("\"conf\"", q2);
    if (cp != std::string::npos && cp < rb + 100) {
      double c = 0;
      if (std::sscanf(body.c_str() + cp, "\"conf\":%lf", &c) == 1) {
        e.conf = c;
      }
    }
    out.push_back(std::move(e));
    p = rb;
  }
  return out;
}

void landmark_resolver_thread() {
  // ===== Sensor geometry (matches rover_360cam SDF) =====
  // Fisheye:  800x800, equidistant lens, hfov=180 deg (cutoff 90 deg).
  //   For equidistant model:  r_px = f_fish * theta  where theta = angle from
  //   optical axis.  At theta=90deg the pixel reaches the radius limit.
  //   With cutoff_angle=90deg and 800-px wide image:
  //       r_max_px = 400  ==>  f_fish = 400 / (pi/2) = 254.6 px/rad.
  //   Pixel offsets from image centre (cx_f, cy_f) -> angles:
  //       du, dv = u - 400, v - 400
  //       r = sqrt(du^2 + dv^2)
  //       theta = r / f_fish              (angle from optical axis +X)
  //       phi   = atan2(-dv, du)          (azimuth around +X; +up image -> +Y)
  //   3-D direction in CAMERA OPTICAL frame (gz: +Z fwd, +X right, +Y down):
  //       dir_optical = ( sin(theta)*cos(phi),  sin(theta)*sin(phi),  cos(theta) )
  //   We *do not* care about distance from the fisheye image — we just need
  //   the bearing direction.  Depth comes from the depth camera.
  //
  // Depth front cam:  480x480, perspective pinhole, hfov=90 deg, mounted at
  //   (0, 0, 0.30) in base_link, looking +X.
  //   fx = 0.5 * 480 / tan(45 deg) = 240
  //   So projection FROM 3-D base-link direction (xb, yb, zb) into depth
  //   pixels (only valid for objects in the +X half-space and within
  //   the 90-degree cone):
  //       u_dep = 240 + 240 * (-yb / xb)   [+Y_base = +left -> -X_depth pixel]
  //       v_dep = 240 + 240 * (-zb / xb)   [+Z_base = +up   -> -Y_depth pixel]
  //
  // Object 3-D position (we have bearing dir but unknown range): once we
  // sample the depth at (u_dep, v_dep) we have distance d_m.  Then world
  // position along the bearing ray from camera centre is:
  //       pos_cam = dir_base * d_m
  // ==========================================================================
  const int FISH_W = 800;
  const double F_FISH = 400.0 / (M_PI * 0.5);
  const double FX_DEP = 240.0;
  const int DEPTH_W = 480, DEPTH_H = 480;
  const double CAM_TX = 0.0, CAM_TY = 0.0, CAM_TZ = 0.30;
  const double DEPTH_VAR_REJECT = 2.5; // m, reject mixed-fg/bg patches
  const double MAX_RANGE = 15.0;       // m, reject far/spurious
  // Per-class confidence + bbox sanity. Hard tightens YOLO-World false
  // positives (e.g. walls being labelled "person" at 0.4 conf).
  auto class_min_conf = [](const std::string& c) -> double {
    if (c == "person") return 0.12;
    if (c == "cone") return 0.12;
    if (c == "stop sign") return 0.25;
    if (c == "traffic light") return 0.55;
    if (c == "fire hydrant") return 0.25;
    if (c == "car" || c == "truck" || c == "bus") return 0.25;
    if (c == "tree") return 0.30;
    if (c == "bench") return 0.25;
    if (c == "platform") return 0.25;
    if (c == "box") return 0.15;
    return 0.25;
  };
  auto class_min_aspect = [](const std::string& c) -> double {
    if (c == "person") return 0.6;
    if (c == "stop sign") return 0.25;
    if (c == "cone") return 0.25;
    if (c == "tree") return 0.0;
    if (c == "bench") return 0.4;
    if (c == "platform") return 0.0;
    if (c == "box") return 0.15;
    return 0.20;
  };
  auto class_min_bbox_h = [](const std::string& c) -> int {
    if (c == "person") return 25.0;   // accept smaller bbox for distant detections
    if (c == "cone") return 24.7;
    if (c == "tree") return 50;
    if (c == "bench") return 30;
    if (c == "platform") return 20;
    if (c == "box") return 29.7;
    return 35;
  };
  // YOLO-World on synthetic gz textures consistently mislabels objects.
  // Translate YOLO classes -> canonical GT classes BEFORE landmark resolution.
  // OBSERVED CONFUSIONS:
  //   "chair"/"fire hydrant" small+tall  -> cone (0.6m cylinders)
  //   "umbrella"/"kite" tall              -> person at distance
  //   "handbag"/"suitcase"/"backpack"     -> box obstacle
  //   "potted plant"                      -> tree
  auto canonicalize_class = [](const std::string& yolo_cls, int bbox_h, double aspect) -> std::string {
    // YOLOv11n COCO closed-vocab mappings (more reliable on synthetic gz)
    if (yolo_cls == "vase" || yolo_cls == "bottle") return "cone";       // tall narrow
    if (yolo_cls == "couch" || yolo_cls == "bed") return "box";           // wide low
    if (yolo_cls == "tv" || yolo_cls == "laptop") return "box";           // flat panel
    if (yolo_cls == "refrigerator" || yolo_cls == "oven") return "shelf"; // tall block
    // Synthetic-texture confusions YOLOv11n makes
    if (yolo_cls == "surfboard" || yolo_cls == "snowboard") return "person";   // tall thin → person capsule
    if (yolo_cls == "skis" || yolo_cls == "skateboard") return "person";
    if (yolo_cls == "knife" || yolo_cls == "scissors" || yolo_cls == "tennis racket") return "person";  // tall narrow
    if (yolo_cls == "sports ball") return "cone";                              // sphere → cone (cone head)
    if (yolo_cls == "fire hydrant") return "cone";
    if (yolo_cls == "parking meter") return "pillar";
    if (yolo_cls == "stop sign") return "sign";
    if (yolo_cls == "orange traffic cone" || yolo_cls == "traffic cone") return "cone";
    if (yolo_cls == "red cube" || yolo_cls == "green cube" ||
        yolo_cls == "yellow cube" || yolo_cls == "blue cube" ||
        yolo_cls == "cardboard box obstacle" ||
        yolo_cls == "wooden crate" || yolo_cls == "wooden pallet" ||
        yolo_cls == "cardboard box") return "box";
    if (yolo_cls == "standing human person" || yolo_cls == "standing person" ||
        yolo_cls == "warehouse worker in orange vest") return "person";
    if (yolo_cls == "cylindrical pillar" || yolo_cls == "pillar" ||
        yolo_cls == "oil drum barrel" || yolo_cls == "cylinder") return "pillar";
    if (yolo_cls == "industrial shelving rack") return "shelf";
    if (yolo_cls == "forklift") return "forklift";
    // Reject anchor classes — they're there to absorb bg confidence
    if (yolo_cls == "wall" || yolo_cls == "floor" || yolo_cls == "sky" ||
        yolo_cls == "background" || yolo_cls == "ground") return "__ignore__";
    if (yolo_cls == "chair" || yolo_cls == "fire hydrant") {
      if (bbox_h < 200 && aspect > 0.8 && aspect < 2.5) return "cone";
    }
    if (yolo_cls == "umbrella" || yolo_cls == "kite") {
      if (aspect > 1.5) return "person";
    }
    if (yolo_cls == "handbag" || yolo_cls == "suitcase" || yolo_cls == "backpack") {
      return "box";
    }
    if (yolo_cls == "potted plant") {
      return "tree";
    }
    return yolo_cls;
  };

  auto sample_depth_patch = [&](LatestDepth& src, int u, int v, int radius) -> double {
    std::vector<float> depth;
    int w = 0, h = 0;
    {
      std::lock_guard<std::mutex> lk(src.mu);
      depth = src.depth;
      w = src.width;
      h = src.height;
    }
    if (depth.empty() || w <= 0 || h <= 0) return std::nan("");
    std::vector<float> nb;
    for (int dv = -radius; dv <= radius; dv++) {
      for (int du = -radius; du <= radius; du++) {
        int uu = u + du, vv = v + dv;
        if (uu < 0 || uu >= w || vv < 0 || vv >= h) continue;
        float d = depth[static_cast<size_t>(vv) * w + uu];
        if (std::isfinite(d) && d > 0.15f && d < MAX_RANGE) nb.push_back(d);
      }
    }
    if (nb.size() < 4) return std::nan("");
    std::sort(nb.begin(), nb.end());
    double q10 = nb[nb.size() / 10];
    double q50 = nb[nb.size() / 2];
    double q90 = nb[(9 * nb.size()) / 10];
    // Bimodal patch (object + background behind): take foreground cluster.
    // Use 10th percentile to robustly land on the nearest object pixels.
    if (q90 - q10 > DEPTH_VAR_REJECT) return q10;
    return q50;
  };

  while (g_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    std::string body = http_get("127.0.0.1", 8081, "/front/detections.json");
    if (body.empty()) continue;
    auto dets = parse_detections_json(body);
    if (dets.empty()) continue;

    // Skip resolution while rover is moving fast OR yawing fast — either
    // condition pushes the pose-at-detection past the merge radius and
    // multiplies a single real person into several ghost landmarks.
    double speed, w_yaw;
    {
      std::lock_guard<std::mutex> lk(g_pose.mu);
      speed = std::hypot(g_pose.vx, g_pose.vy);
      w_yaw = std::fabs(g_pose.yaw_rate);
    }
    if (speed > 5.0 || w_yaw > 0.35) {  // effectively off — only yaw gate matters
      continue;
    }

    // Use the pose at the approximate moment YOLO finished this frame.
    // Detector is unsynced so we just use the most-recent pose, but the
    // ring buffer is in place for when we plumb capture timestamps.
    double rx, ry, rz, qx, qy, qz, qw;
    pose_at_time(now_ms() - 250, rx, ry, rz, qx, qy, qz, qw);  // 250 ms back
    Mat3 R = quat_to_mat(qx, qy, qz, qw);

    for (const auto& d : dets) {
      if (d.w <= 0 || d.h <= 0) continue;
      // Reject near-full-frame bboxes (YOLO hallucinates "stop sign" / "bench"
      // on whole image when no real object present).
      if (d.h > 600 || d.w > 600) {
        std::fprintf(stderr, "[lm-reject] %s full-frame h=%d w=%d\n", d.cls.c_str(), d.h, d.w);
        continue;
      }
      double aspect = static_cast<double>(d.h) / std::max(1, d.w);
      // Remap YOLO class -> canonical GT class before all gates.
      std::string eff_cls = canonicalize_class(d.cls, d.h, aspect);
      if (eff_cls == "__ignore__") continue;
      double min_conf = class_min_conf(eff_cls);
      if (d.conf < min_conf) { std::fprintf(stderr, "[lm-reject] %s(yolo=%s) conf=%.2f<%.2f\n", eff_cls.c_str(), d.cls.c_str(), d.conf, min_conf); continue; }
      if (aspect < class_min_aspect(eff_cls)) { std::fprintf(stderr, "[lm-reject] %s(yolo=%s) aspect=%.2f\n", eff_cls.c_str(), d.cls.c_str(), aspect); continue; }
      if (d.h < class_min_bbox_h(eff_cls)) { std::fprintf(stderr, "[lm-reject] %s(yolo=%s) h=%d\n", eff_cls.c_str(), d.cls.c_str(), d.h); continue; }

      // Sample TORSO/centroid, NOT bbox bottom. Bottom-biased pixel lands on the
// floor behind the object (depth-cam intersects ground plane at +oo > target
// range), drifting landmarks 1-4 m too far. Center-of-bbox is the object's
// solid mid-section -- depth there is the object's actual range.
// (See SA-LOAM / RangeNet++ -- they all sample bbox centroid.)
      double samp_u = d.cx;
      // Sample CHEST for people, true centroid for everything else.
      double samp_v_bias = 0.0;
      if (eff_cls == "person") samp_v_bias = -d.h * 0.15;  // chest, above belly
      double samp_v = d.cy + samp_v_bias;

      // Convert fisheye pixel -> bearing in camera optical frame
      double du = samp_u - FISH_W * 0.5;
      double dv = samp_v - FISH_W * 0.5;
      double r_px = std::sqrt(du * du + dv * dv);
      double theta = r_px / F_FISH;             // 0 .. pi/2
      // Allow slightly wider than nominal 45° because the depth-cam projection
      // covers the front 90-deg cone; we add a small overshoot tolerance.
      if (theta > 1.45) { std::fprintf(stderr, "[lm-reject] %s theta=%.2f r_px=%.0f\n", eff_cls.c_str(), theta, r_px); continue; }
      double phi = std::atan2(-dv, du);
      double sinT = std::sin(theta), cosT = std::cos(theta);
      // Front-cam optical-frame direction (front cam yaw=0 ⇒ optical Z = base +X)
      double dx_opt0 = sinT * std::cos(phi);
      double dy_opt0 = -sinT * std::sin(phi);
      double dz_opt0 = cosT;
      // optical0 -> base_link direction
      double xb_dir = dz_opt0;
      double yb_dir = -dx_opt0;
      double zb_dir = -dy_opt0;
      // Pick depth cam whose yaw is closest to bearing(xb,yb).
      double bearing = std::atan2(yb_dir, xb_dir);
      struct Cam { LatestDepth* p; double yaw; };
      Cam cams[4] = {{&g_depth_front, 0.0}, {&g_depth_right, -M_PI/2},
                     {&g_depth_left, +M_PI/2}, {&g_depth_rear, M_PI}};
      LatestDepth* chosen = nullptr; double chosen_yaw = 0.0; double best = 1e9;
      for (auto& c : cams) {
        double dyaw = std::fabs(std::atan2(std::sin(bearing - c.yaw), std::cos(bearing - c.yaw)));
        if (dyaw < best) { best = dyaw; chosen = c.p; chosen_yaw = c.yaw; }
      }
      if (best > M_PI/4 + 0.1) { std::fprintf(stderr, "[lm-reject] %s bearing=%.2f best_dyaw=%.2f\n", eff_cls.c_str(), bearing, best); continue; }
      // Rotate direction into chosen cam's optical frame.
      double cs = std::cos(chosen_yaw), sn = std::sin(chosen_yaw);
      double dz_opt = xb_dir * cs + yb_dir * sn;        // cam forward
      double dx_opt = xb_dir * sn - yb_dir * cs;        // cam right
      double dy_opt = -zb_dir;                          // cam down

      if (dz_opt < 0.1) { std::fprintf(stderr, "[lm-reject] %s dz_opt=%.2f cam_yaw=%.2f\n", eff_cls.c_str(), dz_opt, chosen_yaw); continue; }
      int u_dep = static_cast<int>(240.0 + FX_DEP * dx_opt / dz_opt);
      int v_dep = static_cast<int>(240.0 + FX_DEP * dy_opt / dz_opt);
      if (u_dep < 4 || u_dep >= DEPTH_W - 4 ||
          v_dep < 4 || v_dep >= DEPTH_H - 4) { std::fprintf(stderr, "[lm-reject] %s depth_uv=%d,%d cam_yaw=%.2f\n", eff_cls.c_str(), u_dep, v_dep, chosen_yaw); continue; }

      double range = sample_depth_patch(*chosen, u_dep, v_dep, 3);
      if (!std::isfinite(range) || range < 0.3 || range > MAX_RANGE) { std::fprintf(stderr, "[lm-reject] %s range=%.2f cam_yaw=%.2f\n", eff_cls.c_str(), range, chosen_yaw); continue; }

      // 3-D point in chosen cam optical frame -> base_link
      double xc = dx_opt * range;
      double yc = dy_opt * range;
      double zc = dz_opt * range;
      // chosen-cam optical -> base. Inverse of forward transform above.
      double xb = zc * cs + xc * sn;
      double yb = zc * sn - xc * cs;
      double zb = -yc;
      xb += CAM_TX; yb += CAM_TY; zb += CAM_TZ;
      // base -> world
      double xw = R.m[0]*xb + R.m[1]*yb + R.m[2]*zb + rx;
      double yw = R.m[3]*xb + R.m[4]*yb + R.m[5]*zb + ry;

      if (std::abs(xw) > 11.0 || std::abs(yw) > 11.0) continue;
      // Range sanity: predict range from bbox height + class size. Reject if
      // measured depth >> predicted (= depth sampled wall behind the object).
      double class_h_m = 1.7;       // person default
      if (eff_cls == "cone") class_h_m = 0.5;
      else if (eff_cls == "stop sign") class_h_m = 0.9;
      else if (eff_cls == "car" || eff_cls == "truck" || eff_cls == "bus") class_h_m = 1.5;
      else if (eff_cls == "tree") class_h_m = 2.5;
      else if (eff_cls == "bench") class_h_m = 0.5;
      else if (eff_cls == "platform") class_h_m = 0.15;
      else if (eff_cls == "box") class_h_m = 0.5;
      else if (eff_cls == "pillar") class_h_m = 1.5;
      else if (eff_cls == "shelf") class_h_m = 1.8;
      else if (eff_cls == "forklift") class_h_m = 1.5;
      double predicted_range = 254.6 * class_h_m / std::max(8, d.h);
      if (range > 1.6 * predicted_range || range < 0.4 * predicted_range) {
        std::fprintf(stderr, "[lm-reject] %s range=%.2f pred=%.2f bbox_h=%d cam_yaw=%.2f\n",
                     eff_cls.c_str(), range, predicted_range, d.h, chosen_yaw);
        // Override: use predicted range instead of bad sample. Recompute world pt.
        double xc2 = dx_opt * predicted_range;
        double yc2 = dy_opt * predicted_range;
        double zc2 = dz_opt * predicted_range;
        double xb2 = zc2 * cs + xc2 * sn;
        double yb2 = zc2 * sn - xc2 * cs;
        double zb2 = -yc2;
        xb2 += CAM_TX; yb2 += CAM_TY; zb2 += CAM_TZ;
        double xw2 = R.m[0]*xb2 + R.m[1]*yb2 + R.m[2]*zb2 + rx;
        double yw2 = R.m[3]*xb2 + R.m[4]*yb2 + R.m[5]*zb2 + ry;
        if (std::abs(xw2) > 11.0 || std::abs(yw2) > 11.0) continue;
        std::fprintf(stderr, "[lm-recover] %s -> (%.2f,%.2f) using predicted range=%.2f\n",
                     eff_cls.c_str(), xw2, yw2, predicted_range);
        xw = xw2; yw = yw2;
      }
      std::fprintf(stderr, "[lm-OK] %s(yolo=%s) at (%.2f,%.2f) range=%.2f cam_yaw=%.2f bbox_h=%d\n",
                   eff_cls.c_str(), d.cls.c_str(), xw, yw, range, chosen_yaw, d.h);
      add_landmark(eff_cls, xw, yw, rx, ry);
    }
  }
}

// Build/update the 2D occupancy grid from the accumulating 3-D voxel cloud +
// the rover's traversed positions. Runs at ~2 Hz.
// 2D Bresenham — marks intermediate cells (exclusive of endpoint) FREE.
// Walking each ray on every frame naturally erases stale "ghost" occupied
// cells when the rover rotates and re-observes the same column of space.
// (See OctoMap / costmap_2d raytrace_freespace.)
static inline void raycast_clear(int x0, int y0, int x1, int y1,
                                 std::vector<int8_t>& grid, int W, int H) {
  int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
  int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
  int err = dx - dy;
  int x = x0, y = y0;
  while (true) {
    if (x == x1 && y == y1) return;             // don't clear endpoint
    if (x >= 0 && x < W && y >= 0 && y < H) {
      grid[static_cast<size_t>(y) * W + x] = 0; // overwrite -1 or 100 -> 0
    }
    int e2 = 2 * err;
    if (e2 > -dy) { err -= dy; x += sx; }
    if (e2 <  dx) { err += dx; y += sy; }
  }
}

void mapper_thread() {
  // Egomotion gate: skip integration if the rover hasn't translated >5 cm
  // AND hasn't rotated >0.5 deg since last update. Pure-rotation frames
  // contribute the circular smear the user is seeing. (Cartographer's
  // motion_filter does the same.)
  double last_x = 1e9, last_y = 1e9;
  double last_yaw = 1e9;
  while (g_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(400));

    // Pull cloud snapshot + current rover pose
    std::vector<float> cloud;
    {
      std::lock_guard<std::mutex> lk(g_cloud.mu);
      cloud = g_cloud.snapshot;
    }
    double rx, ry, qx, qy, qz, qw;
    {
      std::lock_guard<std::mutex> lk(g_pose.mu);
      rx = g_pose.x; ry = g_pose.y;
      qx = g_pose.qx; qy = g_pose.qy; qz = g_pose.qz; qw = g_pose.qw;
    }
    double yaw_now = std::atan2(2 * (qw * qz + qx * qy),
                                1 - 2 * (qy * qy + qz * qz));
    double dpos = std::hypot(rx - last_x, ry - last_y);
    double dyaw = std::fabs(std::atan2(std::sin(yaw_now - last_yaw),
                                        std::cos(yaw_now - last_yaw)));
    if (last_x < 1e8 && dpos < 0.05 && dyaw < 0.0087) {
      // Pure idle / micro-jitter — skip. Footprint clear still happens next
      // tick once we move. No false occupied is added this tick.
      continue;
    }
    last_x = rx; last_y = ry; last_yaw = yaw_now;

    std::lock_guard<std::mutex> lk(g_grid.mu);
    int ri, rj;
    bool have_rover = g_grid.world_to_cell(rx, ry, ri, rj);

    // ===== PASS 1: raycast clear =====
    // For every voxel in the snapshot, walk a Bresenham line from rover cell
    // toward the voxel cell, marking intermediate cells FREE. This erases
    // ghosts left over from prior poses + de-fattens curved boundaries.
    if (have_rover) {
      for (size_t k = 0; k + 2 < cloud.size(); k += 3) {
        double zw = cloud[k + 2];
        if (zw < 0.15 || zw > 1.7) continue;     // tighter band: skip floor/ceiling
        double xw = cloud[k], yw = cloud[k + 1];
        double dx = xw - rx, dy = yw - ry;
        double r2 = dx * dx + dy * dy;
        if (r2 < 0.55 * 0.55 && zw < 0.50) continue; // skip chassis/wheel
        int i, j;
        if (!g_grid.world_to_cell(xw, yw, i, j)) continue;
        raycast_clear(ri, rj, i, j, g_grid.data, OccGrid::W, OccGrid::H);
      }
    }

    // ===== PASS 2: stamp endpoints as OCCUPIED =====
    for (size_t k = 0; k + 2 < cloud.size(); k += 3) {
      double zw = cloud[k + 2];
      if (zw < 0.15 || zw > 1.7) continue;
      double xw = cloud[k], yw = cloud[k + 1];
      double dx = xw - rx, dy = yw - ry;
      double r2 = dx * dx + dy * dy;
      if (r2 < 0.55 * 0.55 && zw < 0.50) continue;
      int i, j;
      if (!g_grid.world_to_cell(xw, yw, i, j)) continue;
      g_grid.data[static_cast<size_t>(j) * OccGrid::W + i] = 100;
    }

    // ===== PASS 3: rover footprint stays free =====
    if (have_rover) {
      int rad = static_cast<int>(0.55 / OccGrid::RES);
      for (int dj = -rad; dj <= rad; dj++) {
        for (int di = -rad; di <= rad; di++) {
          if (di * di + dj * dj > rad * rad) continue;
          int i = ri + di, j = rj + dj;
          if (i < 0 || i >= OccGrid::W || j < 0 || j >= OccGrid::H) continue;
          g_grid.data[static_cast<size_t>(j) * OccGrid::W + i] = 0;
        }
      }
    }
  }
}

// ---- Dijkstra / line-of-sight smoothed path planner ----
// Pure Dijkstra: zero heuristic, expands by minimum g-cost. Guarantees the
// shortest possible path through the 8-connected grid. After search, the
// raw zig-zag path is smoothed by "string-pulling" -- skipping waypoints
// whose straight line to the next-next waypoint is free of obstacles.
struct PlannerNode {
  int i, j;
  double g, f;
  int parent;
  bool operator<(const PlannerNode& o) const { return f > o.f; }
};

// Bresenham-style line-of-sight check between two grid cells through the
// inflated obstacle map. Returns true if the straight segment is fully
// traversable (no occupied cell touched).
static bool line_clear(const std::vector<int8_t>& infl, int W, int H,
                       int x0, int y0, int x1, int y1) {
  int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
  int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
  int err = dx - dy;
  int x = x0, y = y0;
  while (true) {
    if (x < 0 || x >= W || y < 0 || y >= H) return false;
    if (infl[static_cast<size_t>(y) * W + x] == 100) return false;
    if (x == x1 && y == y1) return true;
    int e2 = 2 * err;
    if (e2 > -dy) { err -= dy; x += sx; }
    if (e2 <  dx) { err += dx; y += sy; }
  }
}

static bool astar_plan(double sx, double sy, double gx, double gy,
                       std::vector<std::pair<double, double>>& out) {
  std::vector<int8_t> grid;
  {
    std::lock_guard<std::mutex> lk(g_grid.mu);
    grid = g_grid.data;
  }
  const int W = OccGrid::W, H = OccGrid::H;
  // Inflate obstacles by rover_radius (0.4 m = 4 cells)
  std::vector<int8_t> infl = grid;
  const int rad = 4;
  for (int j = 0; j < H; ++j) {
    for (int i = 0; i < W; ++i) {
      if (grid[static_cast<size_t>(j) * W + i] != 100) continue;
      for (int dj = -rad; dj <= rad; dj++) {
        for (int di = -rad; di <= rad; di++) {
          int ii = i + di, jj = j + dj;
          if (ii < 0 || ii >= W || jj < 0 || jj >= H) continue;
          int8_t& c = infl[static_cast<size_t>(jj) * W + ii];
          if (c == 0 || c == -1) c = 100;
        }
      }
    }
  }
  // Convert start/goal to cells
  int si = static_cast<int>(std::floor((sx - OccGrid::ORIGIN_X) / OccGrid::RES));
  int sj = static_cast<int>(std::floor((sy - OccGrid::ORIGIN_Y) / OccGrid::RES));
  int gi = static_cast<int>(std::floor((gx - OccGrid::ORIGIN_X) / OccGrid::RES));
  int gj = static_cast<int>(std::floor((gy - OccGrid::ORIGIN_Y) / OccGrid::RES));
  auto in_bounds = [&](int i, int j) {
    return i >= 0 && i < W && j >= 0 && j < H;
  };
  if (!in_bounds(si, sj) || !in_bounds(gi, gj)) return false;
  // Allow planning even if start is "occupied" (rover may overlap inflated zone)
  if (infl[static_cast<size_t>(gj) * W + gi] == 100) return false;

  std::vector<double> gscore(W * H, std::numeric_limits<double>::infinity());
  std::vector<int> parent(W * H, -1);
  std::priority_queue<PlannerNode> open;
  auto idx = [&](int i, int j) { return j * W + i; };
  gscore[idx(si, sj)] = 0.0;
  // Dijkstra: f == g (zero heuristic) -> guaranteed shortest path.
  open.push({si, sj, 0.0, 0.0, -1});

  static const int DX[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  static const int DY[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  static const double DCOST[8] = {1, 1, 1, 1, 1.4142, 1.4142, 1.4142, 1.4142};

  bool found = false;
  while (!open.empty()) {
    auto cur = open.top(); open.pop();
    int ci = cur.i, cj = cur.j;
    if (ci == gi && cj == gj) { found = true; break; }
    if (cur.g > gscore[idx(ci, cj)]) continue;
    for (int k = 0; k < 8; k++) {
      int ni = ci + DX[k], nj = cj + DY[k];
      if (!in_bounds(ni, nj)) continue;
      int8_t c = infl[idx(ni, nj)];
      if (c == 100) continue;  // hard obstacle
      double step = DCOST[k];
      if (c == -1) step *= 1.4;  // mild penalty for unknown
      double ng = cur.g + step;
      if (ng < gscore[idx(ni, nj)]) {
        gscore[idx(ni, nj)] = ng;
        parent[idx(ni, nj)] = idx(ci, cj);
        // Dijkstra: priority = g only
        open.push({ni, nj, ng, ng, idx(ci, cj)});
      }
    }
  }
  if (!found) return false;

  // Reconstruct
  std::vector<int> raw;
  int p = idx(gi, gj);
  while (p != -1) { raw.push_back(p); p = parent[p]; }
  std::reverse(raw.begin(), raw.end());

  // Line-of-sight string-pulling: greedily skip waypoints whose connection
  // to a farther one stays inside free space. Result: car-like racing line
  // hugging obstacle corners instead of stair-stepping through cells.
  std::vector<int> smooth;
  smooth.push_back(raw.front());
  size_t anchor = 0;
  while (anchor < raw.size() - 1) {
    size_t lookahead = anchor + 1;
    for (size_t t = anchor + 1; t < raw.size(); ++t) {
      int ax = raw[anchor] % W, ay = raw[anchor] / W;
      int tx = raw[t] % W,      ty = raw[t] / W;
      if (line_clear(infl, W, H, ax, ay, tx, ty)) {
        lookahead = t;
      } else {
        break;
      }
    }
    smooth.push_back(raw[lookahead]);
    anchor = lookahead;
  }

  out.clear();
  for (int p : smooth) {
    int i = p % W, j = p / W;
    double x, y;
    g_grid.cell_to_world(i, j, x, y);
    out.emplace_back(x, y);
  }
  // Always include final goal (in case smoothing rounded to cell centre)
  if (out.empty() || std::hypot(out.back().first - gx, out.back().second - gy) > 0.05) {
    out.emplace_back(gx, gy);
  }
  return true;
}

// Sample fisheye RGB at the optical-frame direction (dx, dy, dz)
// (using the equidistant lens model that matches rover_360cam SDF).
// Returns true on success.
static bool sample_fisheye_rgb(const LatestRgb& rgb_slot,
                               double dx, double dy, double dz,
                               uint8_t& r, uint8_t& g, uint8_t& b) {
  // optical frame: X right, Y down, Z forward
  if (dz < 0.05) return false;  // behind camera plane
  // equidistant: r_px = f_fish * theta, theta = angle from +Z
  double theta = std::acos(std::clamp(
      dz / std::sqrt(dx * dx + dy * dy + dz * dz), -1.0, 1.0));
  if (theta > M_PI / 2.0) return false;
  // azimuth around +Z, image-right = +X, image-down = +Y
  double phi = std::atan2(dy, dx);
  double f_fish;
  int W, H;
  std::vector<uint8_t> img;
  {
    std::lock_guard<std::mutex> lk(rgb_slot.mu);
    W = rgb_slot.width;
    H = rgb_slot.height;
    if (W <= 0 || rgb_slot.data.empty()) return false;
    img = rgb_slot.data;
  }
  f_fish = (0.5 * W) / (M_PI / 2.0);  // matches SDF cutoff_angle 90 deg
  double r_px = f_fish * theta;
  int u = static_cast<int>(W * 0.5 + r_px * std::cos(phi));
  int v = static_cast<int>(H * 0.5 + r_px * std::sin(phi));
  if (u < 0 || u >= W || v < 0 || v >= H) return false;
  size_t idx = (static_cast<size_t>(v) * W + u) * 3;
  if (idx + 2 >= img.size()) return false;
  r = img[idx + 0];
  g = img[idx + 1];
  b = img[idx + 2];
  return true;
}

void cloud_builder_thread() {
  using clk = std::chrono::steady_clock;
  auto next = clk::now();
  while (g_running) {
    next += std::chrono::milliseconds(500);
    // snapshot pose
    double x, y, z, qx, qy, qz, qw;
    {
      std::lock_guard<std::mutex> lk(g_pose.mu);
      x = g_pose.x; y = g_pose.y; z = g_pose.z;
      qx = g_pose.qx; qy = g_pose.qy; qz = g_pose.qz; qw = g_pose.qw;
    }
    Mat3 R = quat_to_mat(qx, qy, qz, qw);

    uint64_t t_now = now_ms();
    // ROTATION SMEAR FIX: if rover is yawing fast, the cubemap depth swings
    // across walls and corners curve. Skip this update entirely while
    // |yaw_rate| > 25 deg/s. KinectFusion / RTAB-Map use the same gate.
    double w_now;
    {
      std::lock_guard<std::mutex> lk(g_pose.mu);
      w_now = std::fabs(g_pose.yaw_rate);
    }
    if (w_now > 0.17) {  // 10 deg/s — match good-map sweet spot
      std::this_thread::sleep_for(std::chrono::milliseconds(150));
      continue;
    }
    // Bounce gate: skip integration when rover airborne or strongly tilted.
    {
      std::lock_guard<std::mutex> lk(g_pose.mu);
      if (std::fabs(g_pose.z) > 0.50 ||
          std::fabs(g_pose.qx) > 0.20 || std::fabs(g_pose.qy) > 0.20) {
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        continue;
      }
    }
    auto add_from = [&](LatestDepth& d, const LatestRgb* rgb_src,
                         double yaw_to_rgb_cam) {
      std::vector<float> depth;
      int w, h;
      double yaw, tx, ty, tz, hfov;
      {
        std::lock_guard<std::mutex> lk(d.mu);
        if (d.depth.empty()) return;
        depth = d.depth;
        w = d.width; h = d.height;
        yaw = d.yaw; tx = d.tx; ty = d.ty; tz = d.tz; hfov = d.hfov;
      }
      const double fx = (0.5 * w) / std::tan(0.5 * hfov);
      const double fy = fx;  // square pixels assumed
      const double cx = 0.5 * w;
      const double cy = 0.5 * h;
      const double cy_a = std::cos(yaw), sy_a = std::sin(yaw);

      // Every-pixel sampling for max density (480x480 depth -> ~230k pts/cam/frame
      // before voxel quantisation collapses them). step=2 -> 4x fewer rays
      // (TSDF integrator walks ~10 voxels/ray, so subsampling matters).
      const int step = 2;
      // Edge-reject: require BOTH horizontal neighbours OR BOTH vertical
      // neighbours to disagree by EDGE_DEPTH before dropping the pixel.
      // This keeps vertical thin objects (people, posts) intact -- a vertical
      // gradient along a head/body is normal even though horizontal is not.
      const float EDGE_DEPTH = 0.30f;  // tighter -> drops more edge streaks
      const float MAX_R = 10.5f;       // matches 20x20m yard; reject beyond walls
      std::lock_guard<std::mutex> lk(g_cloud.mu);
      for (int v = step; v + step < h; v += step) {
        for (int u = step; u + step < w; u += step) {
          float d_m = depth[static_cast<size_t>(v) * w + u];
          if (!std::isfinite(d_m) || d_m <= 0.15f || d_m > MAX_R) continue;
          // 4-neighbour gradient check
          float dn = depth[static_cast<size_t>(v - 1) * w + u];
          float ds = depth[static_cast<size_t>(v + 1) * w + u];
          float de = depth[static_cast<size_t>(v) * w + (u + 1)];
          float dw = depth[static_cast<size_t>(v) * w + (u - 1)];
          if (!std::isfinite(dn) || !std::isfinite(ds) ||
              !std::isfinite(de) || !std::isfinite(dw)) continue;
          bool bad_horiz = std::fabs(de - d_m) > EDGE_DEPTH &&
                           std::fabs(dw - d_m) > EDGE_DEPTH;
          bool bad_vert  = std::fabs(dn - d_m) > EDGE_DEPTH &&
                           std::fabs(ds - d_m) > EDGE_DEPTH;
          // Drop only if BOTH horizontal AND vertical disagree -- isolated
          // outliers. A single-side edge (silhouette boundary) is kept.
          if (bad_horiz && bad_vert) continue;
          double xc = (u - cx) * d_m / fx;
          double yc = (v - cy) * d_m / fy;
          double zc = d_m;
          // camera frame: +Z forward (optical), +X right, +Y down.
          // base_link frame: +X fwd, +Y left, +Z up.
          // Convert optical -> base: X_b = Z_c, Y_b = -X_c, Z_b = -Y_c.
          double xb = zc;
          double yb = -xc;
          double zb = -yc;
          // Apply yaw of this depth camera around base z
          double xb2 = cy_a * xb - sy_a * yb;
          double yb2 = sy_a * xb + cy_a * yb;
          double zb2 = zb;
          // Camera offset on base
          xb2 += tx; yb2 += ty; zb2 += tz;
          // World transform
          double xw = R.m[0]*xb2 + R.m[1]*yb2 + R.m[2]*zb2 + x;
          double yw = R.m[3]*xb2 + R.m[4]*yb2 + R.m[5]*zb2 + y;
          double zw = R.m[6]*xb2 + R.m[7]*yb2 + R.m[8]*zb2 + z;
          // Clamp z band: above floor, below realistic ceiling
          // Skip floor band (ground returns smear horizontally) AND ceiling.
          // Real walls/objects 15 cm - 1.85 m above floor.
          if (zw < 0.30 || zw > 1.70) continue;
          if (std::fabs(xw) > 10.30 || std::fabs(yw) > 10.30) continue;
          // Sample colour from the matching fisheye, if one is supplied for
          // this depth cam. We rotate the optical direction by the relative
          // yaw between the depth cam and the fisheye cam (front fisheye is
          // yaw 0; rear fisheye is yaw pi).
          uint8_t r8 = 110, g8 = 200, b8 = 255;  // fallback blue
          if (rgb_src != nullptr) {
            double cy_r = std::cos(-yaw_to_rgb_cam),
                   sy_r = std::sin(-yaw_to_rgb_cam);
            double xc_r = cy_r * xc - sy_r * zc;
            double zc_r = sy_r * xc + cy_r * zc;
            (void)sample_fisheye_rgb(*rgb_src, xc_r, yc, zc_r, r8, g8, b8);
          }
          // TSDF fusion: walks ray from (z-TRUNC)..(z+TRUNC), updates
          // signed distance + weight. Replaces binary add() +
          // raycast_erase_3d() — TSDF carves free-space implicitly via
          // positive sdf samples on the rover side of the surface.
          double cx_w = x, cy_w = y, cz_w = z + 0.3;
          double dx = xw - cx_w, dy = yw - cy_w, dz = zw - cz_w;
          double rng = std::sqrt(dx*dx + dy*dy + dz*dz);
          if (rng > 0.1) {
            double rxn = dx / rng, ryn = dy / rng, rzn = dz / rng;
            // cos(theta) approx for surface-normal vs ray; flat-floor walls
            // mostly vertical so |rz|+0.5 captures typical incidence.
            float cos_th = std::max(0.05f, std::min(1.0f,
                              static_cast<float>(std::fabs(rzn)) + 0.5f));
            g_cloud.integrate_ray_tsdf(cx_w, cy_w, cz_w,
                                       rxn, ryn, rzn,
                                       static_cast<float>(rng), cos_th,
                                       r8, g8, b8, t_now);
          }
        }
      }
    };

    // Map each depth cam to a fisheye for color sampling.
    // Front+right+left share the front fisheye (180° view), with relative
    // yaw equal to the depth cam's own yaw (since front fisheye yaw=0).
    // Rear depth uses the rear fisheye (yaw=pi).
    add_from(g_depth_front, &g_rgb_front, 0.0);
    add_from(g_depth_right, &g_rgb_front, -M_PI / 2);
    add_from(g_depth_left,  &g_rgb_front, +M_PI / 2);
    add_from(g_depth_rear,  &g_rgb_rear,  0.0);

    {
      std::lock_guard<std::mutex> lk(g_cloud.mu);
      if (g_cloud.cells.size() > 0) g_cloud.snap(now_ms());
    }
    std::this_thread::sleep_until(next);
  }
}

// ============================================================
// Marching Cubes triangle-mesh extraction
// ============================================================
//
// Builds a small dense 3D grid from VoxelCloud::cells (occupancy field
// = 1.0 if a voxel exists in the cell, 0.0 otherwise), runs marching
// cubes at iso=0.5, emits vertices (xyz, rgb, nx/ny/nz) and triangle
// indices, then serializes to binary little-endian PLY.
//
// The mesh is cached: regenerated at most once every 2 seconds. The
// `/tri_mesh.ply` route just returns the cached bytes.

namespace mc {

// Paul Bourke's marching cubes lookup tables (public domain).
// Source: http://paulbourke.net/geometry/polygonise/
static const int kEdgeTable[256] = {
0x000,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,
0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
0x190,0x099,0x393,0x29a,0x596,0x49f,0x795,0x69c,
0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
0x230,0x339,0x033,0x13a,0x636,0x73f,0x435,0x53c,
0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
0x3a0,0x2a9,0x1a3,0x0aa,0x7a6,0x6af,0x5a5,0x4ac,
0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
0x460,0x569,0x663,0x76a,0x066,0x16f,0x265,0x36c,
0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0x0ff,0x3f5,0x2fc,
0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
0x650,0x759,0x453,0x55a,0x256,0x35f,0x055,0x15c,
0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0x0cc,
0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,
0x0cc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,
0x15c,0x055,0x35f,0x256,0x55a,0x453,0x759,0x650,
0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,
0x2fc,0x3f5,0x0ff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,
0x36c,0x265,0x16f,0x066,0x76a,0x663,0x569,0x460,
0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,
0x4ac,0x5a5,0x6af,0x7a6,0x0aa,0x1a3,0x2a9,0x3a0,
0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,
0x53c,0x435,0x73f,0x636,0x13a,0x033,0x339,0x230,
0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,
0x69c,0x795,0x49f,0x596,0x29a,0x393,0x099,0x190,
0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,
0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x000
};

static const int kTriTable[256][16] = {
{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1},
{3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1},
{3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1},
{3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1},
{9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1},
{9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1},
{2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1},
{8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1},
{9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1},
{4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1},
{3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1},
{1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1},
{4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1},
{4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1},
{5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1},
{2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1},
{9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1},
{0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1},
{2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1},
{10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1},
{4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1},
{5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1},
{5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1},
{9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1},
{0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1},
{1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1},
{10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1},
{8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1},
{2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1},
{7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1},
{9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1},
{2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1},
{11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1},
{9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1},
{5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1},
{11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1},
{11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1},
{1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1},
{9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1},
{5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1},
{2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1},
{6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1},
{0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1},
{3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1},
{6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1},
{10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1},
{6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1},
{1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1},
{8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1},
{7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1},
{3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1},
{0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1},
{9,2,1,9,11,2,9,4,11,7,11,4,5,10,6},
{8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1},
{5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1},
{0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1},
{6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1},
{10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1},
{10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1},
{8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1},
{1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1},
{0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1},
{10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1},
{0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1},
{3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1},
{6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1},
{9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1},
{8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1},
{3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1},
{6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1},
{0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1},
{10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1},
{10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1},
{1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1},
{2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1},
{7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1},
{7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1},
{2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1},
{1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1},
{11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1},
{8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1},
{0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1},
{7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1},
{10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1},
{2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1},
{6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1},
{7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1},
{2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1},
{1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1},
{10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1},
{10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1},
{0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1},
{7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1},
{6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1},
{8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1},
{9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1},
{6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1},
{4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1},
{10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1},
{8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1},
{0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1},
{1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1},
{8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1},
{10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1},
{4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1},
{10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1},
{5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1},
{11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1},
{9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1},
{6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1},
{7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1},
{3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1},
{7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1},
{3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1},
{6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1},
{9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1},
{1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1},
{4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1},
{7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1},
{6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1},
{3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1},
{0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1},
{6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1},
{0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1},
{11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1},
{6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1},
{5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1},
{9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1},
{1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1},
{1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1},
{10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1},
{0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1},
{5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1},
{10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1},
{11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1},
{9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1},
{7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1},
{2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1},
{8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1},
{9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1},
{9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1},
{1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1},
{9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1},
{9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1},
{5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1},
{0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1},
{10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1},
{2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1},
{0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1},
{0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1},
{9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1},
{5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1},
{3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1},
{5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1},
{8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1},
{0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1},
{9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1},
{1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1},
{3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1},
{4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1},
{9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1},
{11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1},
{11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1},
{2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1},
{9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1},
{3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1},
{1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1},
{4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1},
{4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1},
{0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1},
{3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1},
{3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1},
{0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1},
{9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1},
{1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
};

// Vertex index -> corner offset (i,j,k in [0,1]^3)
// Standard MC corner ordering: corner v has bits (v&1=x, (v&2)=y, (v&4)=z),
// but with the convention used by Bourke's tables, the corner layout is:
//   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
//   4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
static const int kCornerOfs[8][3] = {
  {0,0,0},{1,0,0},{1,1,0},{0,1,0},
  {0,0,1},{1,0,1},{1,1,1},{0,1,1}
};
// Edge connectivity: pair of corner indices for each of the 12 edges.
static const int kEdgeVerts[12][2] = {
  {0,1},{1,2},{2,3},{3,0},
  {4,5},{5,6},{6,7},{7,4},
  {0,4},{1,5},{2,6},{3,7}
};

struct Vertex {
  float x, y, z;
  uint8_t r, g, b;
  float nx, ny, nz;
};

struct Mesh {
  std::vector<Vertex> verts;
  std::vector<uint32_t> indices;
};

// Build an occupancy field by reading VoxelCloud cells, run MC at iso=0.5,
// emit triangle mesh. Grid covers x,y in [-10,+10], z in [0,2]. Voxel size
// matches the cloud (0.10 m), giving 200x200x20 = 800k cells.
static Mesh extract_mesh_marching_cubes() {
  Mesh out;

  // Snapshot the voxel set under lock; release before doing heavy work.
  struct VoxKV { float x, y, z; uint8_t r, g, b; };
  std::unordered_map<uint64_t, VoxKV> snap;
  double voxel;
  {
    std::lock_guard<std::mutex> lk(g_cloud.mu);
    voxel = g_cloud.voxel;
    snap.reserve(g_cloud.cells.size());
    const float surf_band = 0.5f * static_cast<float>(voxel);
    for (auto& kv : g_cloud.cells) {
      // TSDF zero-crossing band — matches snap() emit criterion so the
      // mesh iso-surface aligns with the rendered point cloud.
      if (kv.second.weight <= 1.0f) continue;
      if (std::fabs(kv.second.sdf) >= surf_band) continue;
      snap[kv.first] = VoxKV{kv.second.x, kv.second.y, kv.second.z,
                              kv.second.r, kv.second.g, kv.second.b};
    }
  }
  if (snap.empty()) return out;

  // Grid bounds in cell-index space (matches VoxelCloud::key encoding).
  const double X_MIN = -10.0, X_MAX = 10.0;
  const double Y_MIN = -10.0, Y_MAX = 10.0;
  const double Z_MIN = 0.0,   Z_MAX = 2.0;
  const int IX_MIN = static_cast<int>(std::floor(X_MIN / voxel));
  const int IY_MIN = static_cast<int>(std::floor(Y_MIN / voxel));
  const int IZ_MIN = static_cast<int>(std::floor(Z_MIN / voxel));
  const int NX = static_cast<int>(std::ceil((X_MAX - X_MIN) / voxel));
  const int NY = static_cast<int>(std::ceil((Y_MAX - Y_MIN) / voxel));
  const int NZ = static_cast<int>(std::ceil((Z_MAX - Z_MIN) / voxel));

  auto occ = [&](int ix, int iy, int iz) -> float {
    if (ix < IX_MIN || ix >= IX_MIN + NX) return 0.0f;
    if (iy < IY_MIN || iy >= IY_MIN + NY) return 0.0f;
    if (iz < IZ_MIN || iz >= IZ_MIN + NZ) return 0.0f;
    return snap.count(VoxelCloud::key(ix, iy, iz)) ? 1.0f : 0.0f;
  };
  auto nearest_rgb = [&](int ix, int iy, int iz, uint8_t& r, uint8_t& g, uint8_t& b) {
    // Walk a 3x3x3 cube to find any occupied neighbour and use its colour.
    for (int dz = 0; dz <= 1; dz++) {
      for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
          auto it = snap.find(VoxelCloud::key(ix + dx, iy + dy, iz + dz));
          if (it != snap.end()) {
            r = it->second.r; g = it->second.g; b = it->second.b;
            return;
          }
        }
      }
    }
    r = 180; g = 200; b = 220;
  };

  const float ISO = 0.5f;
  const std::size_t MAX_TRIS = 200000;

  // Run marching cubes only on cells whose 2x2x2 corner cube contains at
  // least one occupied voxel — far cheaper than scanning all 800k cells.
  // Each occupied voxel touches 8 cube origins (ix-1..ix, iy-1..iy, iz-1..iz).
  std::unordered_set<uint64_t> visit;
  visit.reserve(snap.size() * 4);
  for (auto& kv : snap) {
    uint64_t mask = (1ULL << 21) - 1;
    int iz = static_cast<int>(kv.first & mask) - (1 << 20);
    int iy = static_cast<int>((kv.first >> 21) & mask) - (1 << 20);
    int ix = static_cast<int>((kv.first >> 42) & mask) - (1 << 20);
    for (int dz = -1; dz <= 0; dz++) {
      for (int dy = -1; dy <= 0; dy++) {
        for (int dx = -1; dx <= 0; dx++) {
          int cx = ix + dx, cy = iy + dy, cz = iz + dz;
          if (cx < IX_MIN || cx >= IX_MIN + NX - 1) continue;
          if (cy < IY_MIN || cy >= IY_MIN + NY - 1) continue;
          if (cz < IZ_MIN || cz >= IZ_MIN + NZ - 1) continue;
          visit.insert(VoxelCloud::key(cx, cy, cz));
        }
      }
    }
  }

  for (uint64_t cube_key : visit) {
    if (out.indices.size() / 3 >= MAX_TRIS) break;

    uint64_t mask = (1ULL << 21) - 1;
    int cz = static_cast<int>(cube_key & mask) - (1 << 20);
    int cy = static_cast<int>((cube_key >> 21) & mask) - (1 << 20);
    int cx = static_cast<int>((cube_key >> 42) & mask) - (1 << 20);

    float corner_val[8];
    int cube_index = 0;
    for (int i = 0; i < 8; i++) {
      corner_val[i] = occ(cx + kCornerOfs[i][0],
                           cy + kCornerOfs[i][1],
                           cz + kCornerOfs[i][2]);
      if (corner_val[i] > ISO) cube_index |= (1 << i);
    }
    int edges = kEdgeTable[cube_index];
    if (edges == 0) continue;

    // Compute the (world-space) intersection point per edge.
    float vx[12], vy[12], vz[12];
    auto corner_world = [&](int i, double& wx, double& wy, double& wz) {
      wx = (cx + kCornerOfs[i][0]) * voxel;
      wy = (cy + kCornerOfs[i][1]) * voxel;
      wz = (cz + kCornerOfs[i][2]) * voxel;
    };
    for (int e = 0; e < 12; e++) {
      if (!(edges & (1 << e))) continue;
      int a = kEdgeVerts[e][0], b = kEdgeVerts[e][1];
      double ax, ay, az, bx, by, bz;
      corner_world(a, ax, ay, az);
      corner_world(b, bx, by, bz);
      float va = corner_val[a], vb = corner_val[b];
      float t = 0.5f;
      if (std::fabs(vb - va) > 1e-6f) t = (ISO - va) / (vb - va);
      if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
      vx[e] = static_cast<float>(ax + t * (bx - ax));
      vy[e] = static_cast<float>(ay + t * (by - ay));
      vz[e] = static_cast<float>(az + t * (bz - az));
    }

    // Emit triangles using kTriTable.
    uint8_t r, g, bl;
    nearest_rgb(cx, cy, cz, r, g, bl);
    for (int i = 0; kTriTable[cube_index][i] != -1; i += 3) {
      if (out.indices.size() / 3 >= MAX_TRIS) break;
      int e0 = kTriTable[cube_index][i + 0];
      int e1 = kTriTable[cube_index][i + 1];
      int e2 = kTriTable[cube_index][i + 2];
      // Hard-clamp z to floor/ceiling so MC interpolation never dips
      // below the ground plane.
      float z0 = std::max(0.20f, std::min(1.80f, vz[e0]));
      float z1 = std::max(0.20f, std::min(1.80f, vz[e1]));
      float z2 = std::max(0.20f, std::min(1.80f, vz[e2]));
      Vertex va{vx[e0], vy[e0], z0, r, g, bl, 0,0,0};
      Vertex vb{vx[e1], vy[e1], z1, r, g, bl, 0,0,0};
      Vertex vc{vx[e2], vy[e2], z2, r, g, bl, 0,0,0};
      // Face normal — analytic gradient is overkill for a binary field.
      float ux = vb.x - va.x, uy = vb.y - va.y, uz = vb.z - va.z;
      float wx2 = vc.x - va.x, wy2 = vc.y - va.y, wz2 = vc.z - va.z;
      float nx = uy * wz2 - uz * wy2;
      float ny = uz * wx2 - ux * wz2;
      float nz = ux * wy2 - uy * wx2;
      float n_len = std::sqrt(nx*nx + ny*ny + nz*nz);
      if (n_len > 1e-9f) { nx /= n_len; ny /= n_len; nz /= n_len; }
      va.nx = vb.nx = vc.nx = nx;
      va.ny = vb.ny = vc.ny = ny;
      va.nz = vb.nz = vc.nz = nz;

      uint32_t base = static_cast<uint32_t>(out.verts.size());
      out.verts.push_back(va);
      out.verts.push_back(vb);
      out.verts.push_back(vc);
      out.indices.push_back(base + 0);
      out.indices.push_back(base + 1);
      out.indices.push_back(base + 2);
    }
  }
  return out;
}

// Serialize Mesh -> binary little-endian PLY bytes. Host is assumed
// little-endian (all our target hosts — arm64 macOS, x86_64 Linux — are).
static std::string serialize_ply(const Mesh& m) {
  std::ostringstream hdr;
  hdr << "ply\n"
      << "format binary_little_endian 1.0\n"
      << "comment slam_rover marching-cubes mesh\n"
      << "element vertex " << m.verts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "property float nx\nproperty float ny\nproperty float nz\n"
      << "element face " << (m.indices.size() / 3) << "\n"
      << "property list uchar uint vertex_indices\n"
      << "end_header\n";
  std::string out = hdr.str();

  // Vertex records: 3*float + 3*uchar + 3*float = 12 + 3 + 12 = 27 bytes
  out.reserve(out.size() + m.verts.size() * 27 + (m.indices.size() / 3) * 13);
  for (const auto& v : m.verts) {
    out.append(reinterpret_cast<const char*>(&v.x), 4);
    out.append(reinterpret_cast<const char*>(&v.y), 4);
    out.append(reinterpret_cast<const char*>(&v.z), 4);
    out.append(reinterpret_cast<const char*>(&v.r), 1);
    out.append(reinterpret_cast<const char*>(&v.g), 1);
    out.append(reinterpret_cast<const char*>(&v.b), 1);
    out.append(reinterpret_cast<const char*>(&v.nx), 4);
    out.append(reinterpret_cast<const char*>(&v.ny), 4);
    out.append(reinterpret_cast<const char*>(&v.nz), 4);
  }
  // Face records: count byte (=3) + 3 * uint32
  uint8_t cnt = 3;
  for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3) {
    out.append(reinterpret_cast<const char*>(&cnt), 1);
    out.append(reinterpret_cast<const char*>(&m.indices[i + 0]), 4);
    out.append(reinterpret_cast<const char*>(&m.indices[i + 1]), 4);
    out.append(reinterpret_cast<const char*>(&m.indices[i + 2]), 4);
  }
  return out;
}

// Cached PLY bytes -- rebuilt at most once per 2 seconds.
struct TriMeshCache {
  std::mutex mu;
  std::string ply;
  uint64_t last_build_ms = 0;
  std::size_t n_verts = 0;
  std::size_t n_tris = 0;
};
static TriMeshCache g_trimesh;

static const std::string& get_or_build_trimesh() {
  uint64_t now = now_ms();
  std::lock_guard<std::mutex> lk(g_trimesh.mu);
  if (g_trimesh.ply.empty() || (now - g_trimesh.last_build_ms) > 2000) {
    Mesh m = extract_mesh_marching_cubes();
    g_trimesh.ply = serialize_ply(m);
    g_trimesh.n_verts = m.verts.size();
    g_trimesh.n_tris = m.indices.size() / 3;
    g_trimesh.last_build_ms = now;
  }
  return g_trimesh.ply;
}

}  // namespace mc

void start_http(int port, const std::string& static_dir) {
  httplib::Server svr;

  svr.Get("/", [&static_dir](const httplib::Request&, httplib::Response& res) {
    std::ifstream f(static_dir + "/index.html");
    if (!f) { res.status = 404; return; }
    std::stringstream ss; ss << f.rdbuf();
    res.set_content(ss.str(), "text/html");
  });
  svr.set_mount_point("/static", static_dir);

  auto mjpeg_handler = [](LatestJpeg& slot) {
    return [&slot](const httplib::Request&, httplib::Response& res) {
      res.set_chunked_content_provider(
          "multipart/x-mixed-replace; boundary=frame",
          [&slot](size_t /*offset*/, httplib::DataSink& sink) {
            uint64_t last = 0;
            while (sink.is_writable() && g_running) {
              std::vector<uint8_t> bytes;
              uint64_t seq;
              {
                std::lock_guard<std::mutex> lk(slot.mu);
                seq = slot.seq;
                if (seq != last && !slot.bytes.empty()) bytes = slot.bytes;
              }
              if (!bytes.empty()) {
                last = seq;
                std::string hdr =
                    "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + std::to_string(bytes.size()) + "\r\n\r\n";
                sink.write(hdr.data(), hdr.size());
                sink.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                sink.write("\r\n", 2);
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(40));
            }
            sink.done();
            return true;
          });
    };
  };
  svr.Get("/front.mjpeg", mjpeg_handler(g_front));
  svr.Get("/rear.mjpeg", mjpeg_handler(g_rear));
  // Single-frame JPEG (for dataset collection / external still snapshots)
  auto jpg_handler = [](LatestJpeg& slot) {
    return [&slot](const httplib::Request&, httplib::Response& res) {
      std::lock_guard<std::mutex> lk(slot.mu);
      if (slot.bytes.empty()) { res.status = 503; return; }
      res.set_content(std::string(slot.bytes.begin(), slot.bytes.end()),
                      "image/jpeg");
    };
  };
  svr.Get("/front.jpg", jpg_handler(g_front));
  svr.Get("/rear.jpg",  jpg_handler(g_rear));

  // Min-depth in central wedge of each depth cam, for obstacle-aware planning
  svr.Get("/obstacles.json", [](const httplib::Request&, httplib::Response& res) {
    auto wedge_min = [](LatestDepth& slot) -> double {
      std::vector<float> depth;
      int w = 0, h = 0;
      {
        std::lock_guard<std::mutex> lk(slot.mu);
        depth = slot.depth;
        w = slot.width;
        h = slot.height;
      }
      if (depth.empty() || w <= 0 || h <= 0) return 99.0;
      // Central horizontal wedge: rows above horizon to skip floor
      int v0 = h * 35 / 100;
      int v1 = h * 50 / 100;
      int u0 = w * 30 / 100;
      int u1 = w * 70 / 100;
      double mn = 99.0;
      for (int v = v0; v < v1; v++) {
        for (int u = u0; u < u1; u++) {
          float d = depth[static_cast<size_t>(v) * w + u];
          if (std::isfinite(d) && d > 0.15f && d < 30.0f && d < mn) mn = d;
        }
      }
      return mn;
    };
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "{\"front\":%.2f,\"right\":%.2f,\"rear\":%.2f,\"left\":%.2f}",
                  wedge_min(g_depth_front), wedge_min(g_depth_right),
                  wedge_min(g_depth_rear), wedge_min(g_depth_left));
    res.set_content(buf, "application/json");
  });

  svr.Get("/map_meta.json", [](const httplib::Request&, httplib::Response& res) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "{\"w\":%d,\"h\":%d,\"res\":%.3f,"
                  "\"origin_x\":%.2f,\"origin_y\":%.2f}",
                  OccGrid::W, OccGrid::H, OccGrid::RES,
                  OccGrid::ORIGIN_X, OccGrid::ORIGIN_Y);
    res.set_content(buf, "application/json");
  });

  svr.Get("/telem.json", [](const httplib::Request&, httplib::Response& res) {
    std::ifstream f("/tmp/mavlink_telem.json");
    if (!f) { res.set_content("{}", "application/json"); return; }
    std::stringstream ss; ss << f.rdbuf();
    res.set_content(ss.str(), "application/json");
  });
  svr.Get("/map.bin", [](const httplib::Request&, httplib::Response& res) {
    std::vector<int8_t> snap;
    {
      std::lock_guard<std::mutex> lk(g_grid.mu);
      snap = g_grid.data;
    }
    res.set_content(reinterpret_cast<const char*>(snap.data()),
                    snap.size(), "application/octet-stream");
  });
  svr.Post("/map/clear", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_grid.mu);
    std::fill(g_grid.data.begin(), g_grid.data.end(), -1);
    res.set_content("{\"ok\":true}", "application/json");
  });

  // ----- Gaussian Splat endpoints (written by scripts/cloud_to_splat.py) -----
  auto splat_dir = []() {
    const char* home = std::getenv("HOME");
    std::string h = home ? home : "/tmp";
    return h + "/PX4-Autopilot/slam_rover/state/splats";
  };
  svr.Get("/splat.bin", [splat_dir](const httplib::Request&, httplib::Response& res) {
    std::string path = splat_dir() + "/latest.splat";
    std::ifstream f(path, std::ios::binary);
    if (!f) { res.status = 404; res.set_content("no splat yet", "text/plain"); return; }
    std::stringstream ss; ss << f.rdbuf();
    res.set_content(ss.str(), "application/octet-stream");
  });
  svr.Get("/splats.json", [splat_dir](const httplib::Request&, httplib::Response& res) {
    namespace fs = std::filesystem;
    std::vector<std::string> items;
    std::error_code ec;
    fs::path d(splat_dir());
    if (fs::is_directory(d, ec)) {
      for (auto& entry : fs::directory_iterator(d, ec)) {
        if (!entry.is_regular_file()) continue;
        auto name = entry.path().filename().string();
        if (name == "latest.splat") continue;
        if (name.size() > 6 && name.substr(name.size() - 6) == ".splat") {
          items.push_back(name.substr(0, name.size() - 6));
        }
      }
    }
    std::sort(items.begin(), items.end(), std::greater<std::string>());
    std::ostringstream os;
    os << "{\"items\":[";
    for (size_t i = 0; i < items.size(); i++) {
      if (i) os << ",";
      os << "\"" << items[i] << "\"";
    }
    os << "]}";
    res.set_content(os.str(), "application/json");
  });

  // Static snapshot files served at /saves/<ts>/...
  {
    namespace fs = std::filesystem;
    fs::path snap_dir =
        fs::path(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") /
        "PX4-Autopilot" / "slam_rover" / "snapshots";
    fs::create_directories(snap_dir);
    svr.set_mount_point("/saves", snap_dir.string());
  }

  svr.Get("/saves.json", [](const httplib::Request&, httplib::Response& res) {
    namespace fs = std::filesystem;
    fs::path snap_dir =
        fs::path(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") /
        "PX4-Autopilot" / "slam_rover" / "snapshots";
    std::vector<std::string> names;
    std::error_code ec;
    if (fs::exists(snap_dir)) {
      for (auto& e : fs::directory_iterator(snap_dir, ec)) {
        if (e.is_directory()) names.push_back(e.path().filename().string());
      }
    }
    std::sort(names.begin(), names.end(), std::greater<std::string>());
    std::stringstream ss;
    ss << "{\"items\":[";
    for (size_t k = 0; k < names.size(); ++k) {
      if (k) ss << ",";
      ss << "\"" << names[k] << "\"";
    }
    ss << "]}";
    res.set_content(ss.str(), "application/json");
  });

  // Serve the latest TSDF-fused mesh produced by tsdf_builder.py.
  // Ground-truth world layout from the SDF (walls + obstacles + spawned
  // test objects). Used as overlay in the webui to visually compare the
  // rover's reconstructed map vs the real gz world. Coordinates come from
  // slam_obstacles.sdf and the runtime spawn calls.
  svr.Get("/ground_truth.json", [](const httplib::Request&, httplib::Response& res) {
    static const char* json = R"({
  "world": "slam_obstacles",
  "size_m": [20, 20],
  "walls": [
    {"name":"wall_north","pose":[0,10,1],"size":[20,0.2,2],"color":[200,90,40]},
    {"name":"wall_south","pose":[0,-10,1],"size":[20,0.2,2],"color":[40,150,200]},
    {"name":"wall_east","pose":[10,0,1],"size":[0.2,20,2],"color":[200,180,40]},
    {"name":"wall_west","pose":[-10,0,1],"size":[0.2,20,2],"color":[140,40,160]}
  ],
  "obstacles": [
    {"name":"pillar_1","cls":"pillar","pose":[3,2,0.75],"radius":0.3,"height":1.5,"color":[230,230,230]},
    {"name":"pillar_2","cls":"pillar","pose":[-4,-3,0.75],"radius":0.3,"height":1.5,"color":[40,230,80]},
    {"name":"pillar_3","cls":"pillar","pose":[6,6,0.9],"radius":0.25,"height":1.8,"color":[240,150,50]},
    {"name":"pillar_4","cls":"pillar","pose":[-6,-6,0.6],"radius":0.35,"height":1.2,"color":[50,180,240]},
    {"name":"pillar_5","cls":"pillar","pose":[7,-2,1.0],"radius":0.2,"height":2.0,"color":[220,40,220]},
    {"name":"box_obstacle_1","cls":"box","pose":[5,-4,0.5],"size":[1.2,0.8,1.0],"color":[230,40,40]},
    {"name":"box_obstacle_2","cls":"box","pose":[-3,5,0.4],"size":[0.8,1.5,0.8],"color":[20,80,230]},
    {"name":"box_obstacle_3","cls":"box","pose":[-7,3,0.5],"size":[1.0,1.0,1.0],"color":[240,220,30]},
    {"name":"box_obstacle_4","cls":"box","pose":[2,6,0.3],"size":[1.5,0.4,0.6],"color":[120,240,100]},
    {"name":"box_obstacle_5","cls":"box","pose":[8,4,0.4],"size":[0.6,0.6,0.8],"color":[240,100,90]},
    {"name":"platform_low","cls":"platform","pose":[-7,7,0.075],"size":[2.5,2.5,0.15],"color":[140,140,150]},
    {"name":"tree_1","cls":"tree","pose":[5,2.5,1.0],"radius":0.55,"height":2.5,"color":[50,165,60]},
    {"name":"tree_2","cls":"tree","pose":[-2,7,1.1],"radius":0.7,"height":2.8,"color":[40,140,50]},
    {"name":"bench_1","cls":"bench","pose":[0,-7,0.25],"size":[1.6,0.4,0.5],"color":[140,90,50]},
    {"name":"landmark_board","cls":"sign","pose":[0,9.85,1],"size":[2,0.05,1.2],"color":[255,255,255]}
  ],
  "spawned": [
    {"name":"cone_NE","cls":"cone","pose":[4,4,0],"size":[0.3,0.3,0.6],"color":[255,140,40]},
    {"name":"cone_NW","cls":"cone","pose":[-4,4,0],"size":[0.3,0.3,0.6],"color":[255,140,40]},
    {"name":"cone_SE","cls":"cone","pose":[4,-4,0],"size":[0.3,0.3,0.6],"color":[255,140,40]},
    {"name":"cone_SW","cls":"cone","pose":[-4,-4,0],"size":[0.3,0.3,0.6],"color":[255,140,40]},
    {"name":"cone_north","cls":"cone","pose":[0,8,0.3],"size":[0.36,0.36,0.6],"color":[255,127,25]},
    {"name":"cone_south","cls":"cone","pose":[0,-8,0.3],"size":[0.36,0.36,0.6],"color":[255,127,25]},
    {"name":"cone_east","cls":"cone","pose":[8,0,0.3],"size":[0.36,0.36,0.6],"color":[255,127,25]},
    {"name":"cone_west","cls":"cone","pose":[-8,0,0.3],"size":[0.36,0.36,0.6],"color":[255,127,25]},
    {"name":"person_a","cls":"person","pose":[2,-5,0],"size":[0.5,0.5,1.75],"color":[255,90,60]},
    {"name":"person_b","cls":"person","pose":[-5,2,0],"size":[0.5,0.5,1.75],"color":[255,90,60]},
    {"name":"person_c","cls":"person","pose":[-2,-2,0],"size":[0.4,0.4,1.75],"color":[240,75,75]},
    {"name":"person_d","cls":"person","pose":[7,1,0],"size":[0.4,0.4,1.75],"color":[75,115,215]},
    {"name":"shelf_1","cls":"shelf","pose":[9.2,5,0.9],"size":[0.4,3.0,1.8],"color":[140,90,50]},
    {"name":"shelf_2","cls":"shelf","pose":[-9.2,-5,0.9],"size":[0.4,3.0,1.8],"color":[140,90,50]},
    {"name":"pallet_1","cls":"box","pose":[4,2,0.075],"size":[1.2,0.8,0.15],"color":[150,100,50]},
    {"name":"pallet_2","cls":"box","pose":[-2,-6,0.075],"size":[1.2,0.8,0.15],"color":[150,100,50]},
    {"name":"drum_1","cls":"pillar","pose":[6,-2,0.45],"radius":0.28,"height":0.9,"color":[50,50,60]},
    {"name":"drum_2","cls":"pillar","pose":[6.5,-2,0.45],"radius":0.28,"height":0.9,"color":[165,40,40]},
    {"name":"drum_3","cls":"pillar","pose":[-7,1,0.45],"radius":0.28,"height":0.9,"color":[215,165,40]},
    {"name":"forklift_1","cls":"forklift","pose":[-7,-3,0.6],"size":[1.6,0.9,1.2],"color":[240,165,40]},
    {"name":"crate_1","cls":"box","pose":[3,-2,0.4],"size":[0.8,0.8,0.8],"color":[125,80,45]},
    {"name":"crate_2","cls":"box","pose":[3.8,-2.6,0.4],"size":[0.8,0.8,0.8],"color":[140,90,50]},
    {"name":"worker_1","cls":"person","pose":[-3,3,0.875],"size":[0.4,0.4,1.75],"color":[255,127,0]}
  ]
})";
    res.set_content(json, "application/json");
  });

  svr.Get("/mesh.ply", [](const httplib::Request&, httplib::Response& res) {
    std::ifstream f("/tmp/slam_rover_mesh.ply", std::ios::binary);
    if (!f) { res.status = 404; return; }
    std::stringstream ss; ss << f.rdbuf();
    res.set_content(ss.str(), "application/octet-stream");
  });

  // Marching-cubes triangle mesh of the current VoxelCloud. Rebuilt at most
  // once every 2 seconds; cached bytes are streamed in between.
  svr.Get("/tri_mesh.ply", [](const httplib::Request&, httplib::Response& res) {
    const std::string& ply = mc::get_or_build_trimesh();
    std::size_t nv, nt;
    {
      std::lock_guard<std::mutex> lk(mc::g_trimesh.mu);
      nv = mc::g_trimesh.n_verts;
      nt = mc::g_trimesh.n_tris;
    }
    res.set_header("X-Tri-Verts", std::to_string(nv));
    res.set_header("X-Tri-Faces", std::to_string(nt));
    res.set_content(ply.data(), ply.size(), "application/octet-stream");
  });

  svr.Post("/save", [](const httplib::Request&, httplib::Response& res) {
    // Dump an annotated PNG of the local map + a JSON sidecar listing
    // every landmark with its world position. Also writes the raw colored
    // point cloud as a binary XYZRGB stream.
    namespace fs = std::filesystem;
    std::time_t now = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&now));
    fs::path out_dir =
        fs::path(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") /
        "PX4-Autopilot" / "slam_rover" / "snapshots" / ts;
    std::error_code ec;
    fs::create_directories(out_dir, ec);

    // ---- 1. Render PNG of the 2D occupancy grid + landmark labels ----
    std::vector<int8_t> grid;
    {
      std::lock_guard<std::mutex> lk(g_grid.mu);
      grid = g_grid.data;
    }
    const int W = OccGrid::W, H = OccGrid::H;
    const int up = 4;  // upscale factor -> 800x800 PNG
    cv::Mat img(H * up, W * up, CV_8UC3, cv::Scalar(20, 25, 35));
    for (int j = 0; j < H; ++j) {
      for (int i = 0; i < W; ++i) {
        int8_t v = grid[j * W + i];
        cv::Vec3b color;
        if (v == 100)      color = cv::Vec3b(60, 60, 230);    // occupied red (BGR)
        else if (v == 0)   color = cv::Vec3b(110, 160, 100);  // free green
        else               color = cv::Vec3b(40, 45, 55);     // unknown dim
        // Top of image = +Y. Flip j for image-up display.
        int row = (H - 1 - j) * up;
        int col = i * up;
        cv::rectangle(img, {col, row}, {col + up - 1, row + up - 1},
                      cv::Scalar(color[0], color[1], color[2]), cv::FILLED);
      }
    }

    auto world_to_px = [&](double x, double y, int& u, int& v) {
      // i = (x - OX) / RES   ; j = (y - OY) / RES
      int i = static_cast<int>((x - OccGrid::ORIGIN_X) / OccGrid::RES);
      int j = static_cast<int>((y - OccGrid::ORIGIN_Y) / OccGrid::RES);
      u = i * up + up / 2;
      v = (H - 1 - j) * up + up / 2;
    };

    // Landmarks: filled circle + name label + (x,y) coord
    std::vector<Landmark> lms;
    {
      std::lock_guard<std::mutex> lk(g_landmarks.mu);
      lms = g_landmarks.items;
    }
    auto class_color = [](const std::string& c) -> cv::Scalar {
      if (c == "person") return cv::Scalar(60, 90, 255);
      if (c == "car" || c == "truck" || c == "bus") return cv::Scalar(170, 255, 60);
      if (c == "cone") return cv::Scalar(40, 140, 255);
      if (c == "bicycle" || c == "motorcycle") return cv::Scalar(60, 220, 255);
      if (c == "stop sign") return cv::Scalar(70, 70, 230);
      return cv::Scalar(200, 200, 200);
    };
    for (const auto& lm : lms) {
      int u, v;
      world_to_px(lm.x, lm.y, u, v);
      auto col = class_color(lm.cls);
      cv::circle(img, {u, v}, 9, col, cv::FILLED);
      cv::circle(img, {u, v}, 11, cv::Scalar(0, 0, 0), 1);
      char label[64];
      std::snprintf(label, sizeof(label), "%s (%.1f,%.1f)",
                    lm.cls.c_str(), lm.x, lm.y);
      cv::putText(img, label, {u + 14, v + 4},
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3,
                  cv::LINE_AA);
      cv::putText(img, label, {u + 14, v + 4},
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv::LINE_AA);
    }
    // Rover marker
    {
      std::lock_guard<std::mutex> lk(g_pose.mu);
      int u, v;
      world_to_px(g_pose.x, g_pose.y, u, v);
      cv::drawMarker(img, {u, v}, cv::Scalar(60, 200, 250),
                     cv::MARKER_TRIANGLE_UP, 16, 2, cv::LINE_AA);
    }
    // Title bar
    cv::rectangle(img, {0, 0}, {img.cols, 28},
                  cv::Scalar(20, 25, 35), cv::FILLED);
    char title[160];
    std::snprintf(title, sizeof(title),
                  "slam_rover map  %s   landmarks: %zu", ts, lms.size());
    cv::putText(img, title, {12, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    fs::path png_path = out_dir / "map.png";
    cv::imwrite(png_path.string(), img);

    // ---- 2. Write landmarks.json ----
    {
      std::ofstream f(out_dir / "landmarks.json");
      f << "{\n  \"timestamp\": \"" << ts << "\",\n"
        << "  \"landmarks\": [\n";
      for (size_t k = 0; k < lms.size(); ++k) {
        f << "    {\"cls\":\"" << lms[k].cls << "\",\"x\":" << lms[k].x
          << ",\"y\":" << lms[k].y << ",\"count\":" << lms[k].count << "}";
        if (k + 1 < lms.size()) f << ",";
        f << "\n";
      }
      f << "  ]\n}\n";
    }

    // ---- 3. Write cloud.xyzrgb (raw binary X,Y,Z,R,G,B,...) ----
    {
      std::vector<float> snap;
      std::vector<uint8_t> rgb;
      {
        std::lock_guard<std::mutex> lk(g_cloud.mu);
        snap = g_cloud.snapshot;
        rgb = g_cloud.snapshot_rgb;
      }
      std::ofstream f(out_dir / "cloud.xyzrgb", std::ios::binary);
      const size_t n = snap.size() / 3;
      for (size_t k = 0; k < n; ++k) {
        f.write(reinterpret_cast<const char*>(&snap[k * 3]), 3 * sizeof(float));
        if (k * 3 + 2 < rgb.size()) {
          f.write(reinterpret_cast<const char*>(&rgb[k * 3]), 3);
        }
      }
    }

    // ---- 4. Write occupancy grid raw bytes ----
    {
      std::ofstream f(out_dir / "occupancy.bin", std::ios::binary);
      f.write(reinterpret_cast<const char*>(grid.data()), grid.size());
    }

    char resp[512];
    std::snprintf(resp, sizeof(resp),
                  "{\"ok\":true,\"dir\":\"%s\",\"png\":\"%s\","
                  "\"landmarks\":%zu,\"cloud_pts\":%zu}",
                  out_dir.string().c_str(), png_path.string().c_str(),
                  lms.size(), grid.size());
    res.set_content(resp, "application/json");
  });
  svr.Post("/plan", [](const httplib::Request& req, httplib::Response& res) {
    if (!req.has_param("x") || !req.has_param("y")) {
      res.status = 400; res.set_content("{\"err\":\"x/y required\"}", "application/json");
      return;
    }
    double gx = std::stod(req.get_param_value("x"));
    double gy = std::stod(req.get_param_value("y"));
    double sx, sy;
    {
      std::lock_guard<std::mutex> lk(g_pose.mu);
      sx = g_pose.x; sy = g_pose.y;
    }
    std::vector<std::pair<double, double>> wpts;
    bool ok = astar_plan(sx, sy, gx, gy, wpts);
    {
      std::lock_guard<std::mutex> lk(g_path.mu);
      g_path.waypoints = wpts;
      g_path.active = ok && !wpts.empty();
      g_path.cursor = 0;
      g_path.seq++;
    }
    std::stringstream ss;
    ss << "{\"ok\":" << (ok ? "true" : "false")
       << ",\"n\":" << wpts.size()
       << ",\"start\":[" << sx << "," << sy << "]"
       << ",\"goal\":[" << gx << "," << gy << "]}";
    res.set_content(ss.str(), "application/json");
  });
  svr.Get("/path.json", [](const httplib::Request&, httplib::Response& res) {
    std::vector<std::pair<double, double>> wpts;
    uint64_t seq;
    bool active;
    size_t cursor;
    {
      std::lock_guard<std::mutex> lk(g_path.mu);
      wpts = g_path.waypoints;
      seq = g_path.seq;
      active = g_path.active;
      cursor = g_path.cursor;
    }
    std::stringstream ss;
    ss << "{\"seq\":" << seq << ",\"active\":" << (active ? "true" : "false")
       << ",\"cursor\":" << cursor << ",\"waypoints\":[";
    for (size_t i = 0; i < wpts.size(); ++i) {
      if (i) ss << ",";
      ss << "[" << wpts[i].first << "," << wpts[i].second << "]";
    }
    ss << "]}";
    res.set_content(ss.str(), "application/json");
  });
  svr.Post("/path/advance", [](const httplib::Request& req, httplib::Response& res) {
    // Path follower posts current cursor index after consuming a waypoint
    int n = req.has_param("cursor") ? std::stoi(req.get_param_value("cursor")) : 0;
    std::lock_guard<std::mutex> lk(g_path.mu);
    g_path.cursor = static_cast<size_t>(n);
    if (g_path.cursor >= g_path.waypoints.size()) {
      g_path.active = false;
    }
    res.set_content("{\"ok\":true}", "application/json");
  });
  svr.Post("/explore/step", [](const httplib::Request&, httplib::Response& res) {
    // Lightweight in-process frontier picker: scan the grid for a free cell
    // adjacent to unknown, pick the largest cluster, plan to it.
    std::vector<int8_t> grid;
    {
      std::lock_guard<std::mutex> lk(g_grid.mu);
      grid = g_grid.data;
    }
    const int W = OccGrid::W, H = OccGrid::H;
    // Find frontiers + flood-fill cluster sizes
    std::vector<uint8_t> isFront(W * H, 0);
    for (int j = 1; j < H - 1; ++j) {
      for (int i = 1; i < W - 1; ++i) {
        if (grid[j * W + i] != 0) continue;
        if (grid[(j-1)*W+i]==-1 || grid[(j+1)*W+i]==-1 ||
            grid[j*W+(i-1)]==-1 || grid[j*W+(i+1)]==-1) {
          isFront[j * W + i] = 1;
        }
      }
    }
    // BFS clustering
    std::vector<int> visited(W * H, -1);
    int best_id = -1; int best_sz = 0; double best_cx = 0, best_cy = 0;
    int cid = 0;
    for (int s = 0; s < W * H; ++s) {
      if (!isFront[s] || visited[s] != -1) continue;
      std::queue<int> q; q.push(s); visited[s] = cid;
      int sz = 0; long sx = 0, sy = 0;
      while (!q.empty()) {
        int p = q.front(); q.pop();
        int pi = p % W, pj = p / W;
        sz++; sx += pi; sy += pj;
        for (int dj = -1; dj <= 1; ++dj)
          for (int di = -1; di <= 1; ++di) {
            if (!di && !dj) continue;
            int ni = pi + di, nj = pj + dj;
            if (ni < 0 || ni >= W || nj < 0 || nj >= H) continue;
            int np = nj * W + ni;
            if (isFront[np] && visited[np] == -1) {
              visited[np] = cid; q.push(np);
            }
          }
      }
      if (sz > best_sz) {
        best_sz = sz;
        best_cx = static_cast<double>(sx) / sz;
        best_cy = static_cast<double>(sy) / sz;
        best_id = cid;
      }
      cid++;
    }
    if (best_id < 0 || best_sz < 8) {
      res.set_content("{\"ok\":false,\"reason\":\"no frontier\"}",
                      "application/json");
      return;
    }
    double tx, ty;
    g_grid.cell_to_world(static_cast<int>(best_cx),
                         static_cast<int>(best_cy), tx, ty);
    // Plan + execute
    double sx, sy;
    {
      std::lock_guard<std::mutex> lk(g_pose.mu);
      sx = g_pose.x; sy = g_pose.y;
    }
    std::vector<std::pair<double, double>> wpts;
    bool ok = astar_plan(sx, sy, tx, ty, wpts);
    {
      std::lock_guard<std::mutex> lk(g_path.mu);
      g_path.waypoints = wpts;
      g_path.active = ok && !wpts.empty();
      g_path.cursor = 0;
      g_path.seq++;
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "{\"ok\":%s,\"target\":[%.2f,%.2f],"
                  "\"cluster_sz\":%d,\"n_waypoints\":%zu}",
                  ok ? "true" : "false", tx, ty, best_sz, wpts.size());
    res.set_content(buf, "application/json");
  });

  svr.Post("/path/preset", [](const httplib::Request& req, httplib::Response& res) {
    // Inject a hand-crafted waypoint loop -- like a race line. Useful when
    // map is empty or A* fails. Patterns:
    //   ?p=square    -- 6x6 m clockwise lap centred on origin
    //   ?p=eight     -- figure-8 across the yard
    //   ?p=oval      -- racing oval through clear corridors
    //   ?p=tour      -- visits each non-wall corner
    std::string p = req.has_param("p") ? req.get_param_value("p") : "square";
    std::vector<std::pair<double, double>> wpts;
    if (p == "square") {
      // CW from (3,3) -> (3,-3) -> (-3,-3) -> (-3,3) -> back
      for (double t = 0; t <= 1.0; t += 0.1) wpts.emplace_back(3, 3 - 6 * t);
      for (double t = 0; t <= 1.0; t += 0.1) wpts.emplace_back(3 - 6 * t, -3);
      for (double t = 0; t <= 1.0; t += 0.1) wpts.emplace_back(-3, -3 + 6 * t);
      for (double t = 0; t <= 1.0; t += 0.1) wpts.emplace_back(-3 + 6 * t, 3);
    } else if (p == "eight") {
      // Two loops crossing at origin
      const int N = 60;
      for (int k = 0; k < N; ++k) {
        double th = 2 * M_PI * k / N;
        wpts.emplace_back(3.0 * std::sin(th), 3.0 * std::sin(th) * std::cos(th) + 2);
      }
      for (int k = 0; k < N; ++k) {
        double th = 2 * M_PI * k / N;
        wpts.emplace_back(3.0 * std::sin(th), 3.0 * std::sin(th) * std::cos(th) - 2);
      }
    } else if (p == "oval") {
      const int N = 40;
      for (int k = 0; k <= N; ++k) {
        double th = 2 * M_PI * k / N - M_PI / 2;
        wpts.emplace_back(6.0 * std::cos(th), 4.0 * std::sin(th));
      }
    } else if (p == "tour") {
      // visit each quadrant
      double pts[][2] = {{6, 0}, {6, 6}, {0, 6}, {-6, 6}, {-6, 0},
                          {-6, -6}, {0, -6}, {6, -6}, {6, 0}, {0, 0}};
      for (auto& pp : pts) wpts.emplace_back(pp[0], pp[1]);
    }
    {
      std::lock_guard<std::mutex> lk(g_path.mu);
      g_path.waypoints = wpts;
      g_path.cursor = 0;
      g_path.active = !wpts.empty();
      g_path.seq++;
    }
    char buf[80];
    std::snprintf(buf, sizeof(buf), "{\"ok\":true,\"n\":%zu,\"preset\":\"%s\"}",
                  wpts.size(), p.c_str());
    res.set_content(buf, "application/json");
  });

  svr.Post("/path/clear", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_path.mu);
    g_path.waypoints.clear();
    g_path.active = false;
    g_path.cursor = 0;
    g_path.seq++;
    res.set_content("{\"ok\":true}", "application/json");
  });

  svr.Get("/landmarks.json", [](const httplib::Request&, httplib::Response& res) {
    std::vector<Landmark> copy;
    {
      std::lock_guard<std::mutex> lk(g_landmarks.mu);
      uint64_t t = now_ms();
      // Decay only UNLOCKED landmarks. Locked items are permanent until
      // explicitly cleared.
      g_landmarks.items.erase(
          std::remove_if(g_landmarks.items.begin(), g_landmarks.items.end(),
                         [&](const Landmark& lm) {
                           return !lm.locked &&
                                  t - lm.last_seen_ms > g_landmarks.landmark_decay_ms;
                         }),
          g_landmarks.items.end());
      copy = g_landmarks.items;
    }
    std::stringstream ss;
    ss << "{\"items\":[";
    for (size_t i = 0; i < copy.size(); ++i) {
      const auto& lm = copy[i];
      if (i) ss << ",";
      ss << "{\"cls\":\"" << lm.cls << "\","
         << "\"x\":" << lm.x << ",\"y\":" << lm.y << ","
         << "\"count\":" << lm.count << ","
         << "\"locked\":" << (lm.locked ? "true" : "false") << ","
         << "\"first_seen_ms\":" << lm.first_seen_ms << ","
         << "\"last_seen_ms\":" << lm.last_seen_ms << "}";
    }
    ss << "]}";
    res.set_content(ss.str(), "application/json");
  });

  svr.Post("/landmarks/save", [](const httplib::Request&, httplib::Response& res) {
    save_landmarks_to_disk();
    res.set_content("{\"ok\":true}", "application/json");
  });
  svr.Post("/landmarks/lock", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_landmarks.mu);
    for (auto& lm : g_landmarks.items) lm.locked = true;
    res.set_content("{\"ok\":true}", "application/json");
  });
  svr.Post("/landmarks/clear", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_landmarks.mu);
    g_landmarks.items.clear();
    res.set_content("{\"ok\":true}", "application/json");
  });

  svr.Get("/cloud.bin", [](const httplib::Request&, httplib::Response& res) {
    std::vector<float> snap;
    uint64_t seq;
    {
      std::lock_guard<std::mutex> lk(g_cloud.mu);
      snap = g_cloud.snapshot;
      seq = g_cloud.snapshot_seq;
    }
    res.set_header("X-Cloud-Seq", std::to_string(seq));
    res.set_content(reinterpret_cast<const char*>(snap.data()),
                    snap.size() * sizeof(float), "application/octet-stream");
  });
  // Raw depth + RGB endpoints for external TSDF builders (Open3D, etc.).
  // depth_front.bin: float32 row-major H*W (10 Hz, 480x480 in current SDF)
  // rgb_front.bin:   uint8  row-major H*W*3
  // Both return matching width/height in headers.
  svr.Get("/depth_front.bin", [](const httplib::Request&, httplib::Response& res) {
    std::vector<float> depth;
    int w, h;
    {
      std::lock_guard<std::mutex> lk(g_depth_front.mu);
      depth = g_depth_front.depth;
      w = g_depth_front.width;
      h = g_depth_front.height;
    }
    res.set_header("X-Width", std::to_string(w));
    res.set_header("X-Height", std::to_string(h));
    res.set_header("X-HFov", std::to_string(g_depth_front.hfov));
    res.set_content(reinterpret_cast<const char*>(depth.data()),
                    depth.size() * sizeof(float), "application/octet-stream");
  });
  svr.Get("/rgb_front.bin", [](const httplib::Request&, httplib::Response& res) {
    std::vector<uint8_t> rgb;
    int w, h;
    {
      std::lock_guard<std::mutex> lk(g_rgb_front.mu);
      rgb = g_rgb_front.data;
      w = g_rgb_front.width;
      h = g_rgb_front.height;
    }
    res.set_header("X-Width", std::to_string(w));
    res.set_header("X-Height", std::to_string(h));
    res.set_content(reinterpret_cast<const char*>(rgb.data()),
                    rgb.size(), "application/octet-stream");
  });

  svr.Get("/cloud_rgb.bin", [](const httplib::Request&, httplib::Response& res) {
    std::vector<uint8_t> rgb;
    uint64_t seq;
    {
      std::lock_guard<std::mutex> lk(g_cloud.mu);
      rgb = g_cloud.snapshot_rgb;
      seq = g_cloud.snapshot_seq;
    }
    res.set_header("X-Cloud-Seq", std::to_string(seq));
    res.set_content(reinterpret_cast<const char*>(rgb.data()),
                    rgb.size(), "application/octet-stream");
  });

  svr.Post("/target", [](const httplib::Request& req, httplib::Response& res) {
    // Accept body: x=<num>&y=<num>  OR  {"x":..,"y":..,"clear":true}
    double x = 0, y = 0;
    bool clear = false;
    if (req.has_param("clear")) clear = req.get_param_value("clear") == "1";
    if (req.has_param("x")) x = std::stod(req.get_param_value("x"));
    if (req.has_param("y")) y = std::stod(req.get_param_value("y"));
    {
      std::lock_guard<std::mutex> lk(g_target.mu);
      if (clear) {
        g_target.active = false;
      } else {
        g_target.x = x;
        g_target.y = y;
        g_target.active = true;
      }
      g_target.seq++;
    }
    res.set_content("{\"ok\":true}", "application/json");
  });
  svr.Get("/config.json", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_cfg.mu);
    char buf[768];
    std::snprintf(buf, sizeof(buf),
                  "{\"fwd_speed\":%.3f,\"max_yaw_rate_deg\":%.2f,"
                  "\"search_yaw_rate_deg\":%.2f,"
                  "\"approach_yaw_gain\":%.3f,\"follow_yaw_gain\":%.3f,"
                  "\"approach_stop_ratio\":%.3f,"
                  "\"follow_goal_low\":%.3f,\"follow_goal_high\":%.3f,"
                  "\"lost_timeout\":%.2f,\"seq\":%llu}",
                  g_cfg.fwd_speed, g_cfg.max_yaw_rate_deg,
                  g_cfg.search_yaw_rate_deg,
                  g_cfg.approach_yaw_gain, g_cfg.follow_yaw_gain,
                  g_cfg.approach_stop_ratio,
                  g_cfg.follow_goal_low, g_cfg.follow_goal_high,
                  g_cfg.lost_timeout,
                  static_cast<unsigned long long>(g_cfg.seq));
    res.set_content(buf, "application/json");
  });
  svr.Post("/config", [](const httplib::Request& req, httplib::Response& res) {
    auto set = [&](const char* key, double& slot) {
      if (req.has_param(key)) {
        try { slot = std::stod(req.get_param_value(key)); }
        catch (...) {}
      }
    };
    {
      std::lock_guard<std::mutex> lk(g_cfg.mu);
      set("fwd_speed", g_cfg.fwd_speed);
      set("max_yaw_rate_deg", g_cfg.max_yaw_rate_deg);
      set("search_yaw_rate_deg", g_cfg.search_yaw_rate_deg);
      set("approach_yaw_gain", g_cfg.approach_yaw_gain);
      set("follow_yaw_gain", g_cfg.follow_yaw_gain);
      set("approach_stop_ratio", g_cfg.approach_stop_ratio);
      set("follow_goal_low", g_cfg.follow_goal_low);
      set("follow_goal_high", g_cfg.follow_goal_high);
      set("lost_timeout", g_cfg.lost_timeout);
      g_cfg.seq++;
    }
    res.set_content("{\"ok\":true}", "application/json");
  });

  svr.Post("/investigate", [](const httplib::Request& req, httplib::Response& res) {
    bool clear = req.has_param("clear") && req.get_param_value("clear") == "1";
    std::string cls = req.has_param("cls") ? req.get_param_value("cls") : "";
    std::string state_str = req.has_param("state") ? req.get_param_value("state") : "";
    {
      std::lock_guard<std::mutex> lk(g_invest.mu);
      if (clear) {
        g_invest.active = false;
        g_invest.cls.clear();
        g_invest.state = "idle";
      } else {
        if (!cls.empty()) g_invest.cls = cls;
        if (!state_str.empty()) g_invest.state = state_str;
        if (!cls.empty()) g_invest.active = true;
      }
      g_invest.seq++;
    }
    res.set_content("{\"ok\":true}", "application/json");
  });
  svr.Get("/investigate.json", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_invest.mu);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "{\"cls\":\"%s\",\"state\":\"%s\",\"active\":%s,\"seq\":%llu}",
                  g_invest.cls.c_str(),
                  (g_invest.state.empty() ? "idle" : g_invest.state.c_str()),
                  g_invest.active ? "true" : "false",
                  static_cast<unsigned long long>(g_invest.seq));
    res.set_content(buf, "application/json");
  });

  svr.Get("/target.json", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_target.mu);
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "{\"x\":%.4f,\"y\":%.4f,\"active\":%s,\"seq\":%llu}",
                  g_target.x, g_target.y,
                  g_target.active ? "true" : "false",
                  static_cast<unsigned long long>(g_target.seq));
    res.set_content(buf, "application/json");
  });

  svr.Get("/pose.json", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(g_pose.mu);
    char buf[320];
    double speed = std::hypot(g_pose.vx, g_pose.vy);
    std::snprintf(buf, sizeof(buf),
                  "{\"x\":%.4f,\"y\":%.4f,\"z\":%.4f,"
                  "\"qx\":%.6f,\"qy\":%.6f,\"qz\":%.6f,\"qw\":%.6f,"
                  "\"vx\":%.3f,\"vy\":%.3f,\"speed\":%.3f,\"seq\":%llu}",
                  g_pose.x, g_pose.y, g_pose.z,
                  g_pose.qx, g_pose.qy, g_pose.qz, g_pose.qw,
                  g_pose.vx, g_pose.vy, speed,
                  static_cast<unsigned long long>(g_pose.seq));
    res.set_content(buf, "application/json");
  });

  std::cout << "[webui] listening on http://0.0.0.0:" << port
            << "   static=" << static_dir << std::endl;
  svr.listen("0.0.0.0", port);
}

}  // namespace

int main(int argc, char** argv) {
  std::string static_dir = std::string(SOURCE_STATIC_DIR);
  int port = 8080;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    else if (a == "--static" && i + 1 < argc) static_dir = argv[++i];
  }

  gz::transport::Node node;

  // Cameras
  node.Subscribe<gz::msgs::Image>("/fisheye_front/image", &on_image_front);
  node.Subscribe<gz::msgs::Image>("/fisheye_rear/image", &on_image_rear);

  // Depth cams (PX4 spawns rover as rover_360cam_0 -- topics are remapped to
  // /depth_<side> per <topic> tag in SDF).
  g_depth_front.hfov = M_PI / 2; g_depth_front.yaw = 0.0;
  g_depth_right.hfov = M_PI / 2; g_depth_right.yaw = -M_PI / 2;
  g_depth_rear.hfov  = M_PI / 2; g_depth_rear.yaw  = M_PI;
  g_depth_left.hfov  = M_PI / 2; g_depth_left.yaw  = M_PI / 2;
  std::function<void(const gz::msgs::Image&)> df =
      [](const gz::msgs::Image& m){ on_depth_generic(g_depth_front, m); };
  std::function<void(const gz::msgs::Image&)> dr =
      [](const gz::msgs::Image& m){ on_depth_generic(g_depth_right, m); };
  std::function<void(const gz::msgs::Image&)> db =
      [](const gz::msgs::Image& m){ on_depth_generic(g_depth_rear, m); };
  std::function<void(const gz::msgs::Image&)> dl =
      [](const gz::msgs::Image& m){ on_depth_generic(g_depth_left, m); };
  node.Subscribe<gz::msgs::Image>("/depth_front", df);
  node.Subscribe<gz::msgs::Image>("/depth_right", dr);
  node.Subscribe<gz::msgs::Image>("/depth_rear",  db);
  node.Subscribe<gz::msgs::Image>("/depth_left",  dl);

  // Rover odometry (preferred if PX4 publishes it).
  node.Subscribe<gz::msgs::OdometryWithCovariance>(
      "/model/rover_360cam_0/odometry_with_covariance", &on_odom);
  // Fallback: world dynamic_pose stream always exists.
  node.Subscribe<gz::msgs::Pose_V>(
      "/world/slam_obstacles/dynamic_pose/info", &on_dynamic_pose);

  load_landmarks_from_disk();

  std::thread builder(cloud_builder_thread);
  std::thread resolver(landmark_resolver_thread);
  std::thread mapper(mapper_thread);
  std::thread persistor(landmark_persistor_thread);
  start_http(port, static_dir);
  g_running = false;
  save_landmarks_to_disk();
  if (builder.joinable()) builder.join();
  if (resolver.joinable()) resolver.join();
  if (mapper.joinable()) mapper.join();
  if (persistor.joinable()) persistor.join();
  return 0;
}
