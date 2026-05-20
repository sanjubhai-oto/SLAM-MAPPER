cmake_minimum_required(VERSION 3.16)
project(slam_rover_webui CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# brew gz Harmonic
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

# Threads + pthread for httplib
find_package(Threads REQUIRED)
# httplib autodetects OpenSSL via __has_include and calls OPENSSL_thread_stop;
# link OpenSSL to satisfy the symbol on macOS where brew openssl headers are visible.
find_package(OpenSSL REQUIRED)

add_executable(slam_rover_webui src/main.cpp)

target_compile_definitions(slam_rover_webui PRIVATE
  SOURCE_STATIC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/static"
  CPPHTTPLIB_OPENSSL_SUPPORT=0)

target_link_libraries(slam_rover_webui PRIVATE
  gz-transport13::core
  gz-msgs10::core
  ${OpenCV_LIBS}
  OpenSSL::SSL OpenSSL::Crypto
  Threads::Threads)

if(APPLE)
  target_link_libraries(slam_rover_webui PRIVATE
    "-framework Security" "-framework CoreFoundation")
endif()

target_include_directories(slam_rover_webui PRIVATE ${OpenCV_INCLUDE_DIRS})
