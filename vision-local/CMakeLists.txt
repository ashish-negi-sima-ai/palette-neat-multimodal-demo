# drone-seminar-usb-vision - the dual USB camera vision binary (devkit edition).
#
#   ./build.sh          (cross-compiles in the SDK container, aarch64)
#
# One graph per USB camera that branches the decoded NV12 buffer before the
# application touches it:
#
#     v4l2src MJPEG -> neatdecoder NV12 -> Branch -+-> direct NV12 -> neatencoder -> RTP/UDP
#                                                  +-> Output("frames", Latest) -> YOLO -> metadata
#
# ALL SOURCES ARE LOCAL (src/). The project builds and runs without any other
# /workspace project. encoder_low_watermark.h is included by stream_worker.cpp
# (optional encoder input-queue bound).

cmake_minimum_required(VERSION 3.16)
project(drone-seminar-usb-vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)
find_package(Threads REQUIRED)

set(VISION_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")

foreach(required_file main.cpp usb_camera.cpp mla_gate.cpp yolo_render.cpp stream_worker.cpp
                     stream_worker.h mla_gate.h yolo_render.h usb_camera.h)
  if(NOT EXISTS "${VISION_SRC}/${required_file}")
    message(FATAL_ERROR "missing ${VISION_SRC}/${required_file}")
  endif()
endforeach()

add_executable(drone-seminar-usb-vision
  "${VISION_SRC}/main.cpp"
  "${VISION_SRC}/usb_camera.cpp"
  "${VISION_SRC}/mla_gate.cpp"
  "${VISION_SRC}/yolo_render.cpp"
  "${VISION_SRC}/stream_worker.cpp"
)

target_include_directories(drone-seminar-usb-vision PRIVATE "${VISION_SRC}")

target_link_libraries(drone-seminar-usb-vision PRIVATE
  SimaNeat::sima_neat
  Threads::Threads
)
