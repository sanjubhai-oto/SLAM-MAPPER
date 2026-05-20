#!/usr/bin/env bash
# Build stella_vslam + ROS2 wrapper inside the active RoboStack env.
# Run after: mamba activate slam_rover
set -euo pipefail

ROOT="${HOME}/PX4-Autopilot/slam_rover"
SRC="${ROOT}/ros2_ws/src"
PREFIX="${CONDA_PREFIX:?activate slam_rover env first}"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "ROS_DISTRO not set. Activate slam_rover (mamba activate slam_rover) first."
  exit 1
fi

mkdir -p "${SRC}"
cd "${SRC}"

# Core SLAM lib
[[ -d stella_vslam ]] || git clone --recursive https://github.com/stella-cv/stella_vslam.git
[[ -d FBoW ]]         || git clone https://github.com/stella-cv/FBoW.git
[[ -d sioclient ]]    || git clone https://github.com/shinsumicco/socket.io-client-cpp.git sioclient

# ROS2 wrapper
[[ -d stella_vslam_ros ]] || git clone --branch ros2 https://github.com/stella-cv/stella_vslam_ros.git

# Build FBoW + stella_vslam outside ament so we can install into CONDA_PREFIX
cd "${SRC}/FBoW"
cmake -B build -G Ninja -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install

cd "${SRC}/sioclient"
cmake -B build -G Ninja -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build --target install

cd "${SRC}/stella_vslam"
cmake -B build -G Ninja -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_PANGOLIN_VIEWER=OFF -DUSE_IRIDESCENCE_VIEWER=OFF \
  -DUSE_SOCKET_PUBLISHER=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build build --target install

# Vocab file
VOCAB_DIR="${ROOT}/ros2_ws/src/rover_bringup/config"
if [[ ! -f "${VOCAB_DIR}/orb_vocab.fbow" ]]; then
  curl -L https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow \
    -o "${VOCAB_DIR}/orb_vocab.fbow"
fi

# Build ROS2 workspace
cd "${ROOT}/ros2_ws"
colcon build --symlink-install --event-handlers console_direct+
echo "Build OK. Source: source ${ROOT}/ros2_ws/install/setup.bash"
