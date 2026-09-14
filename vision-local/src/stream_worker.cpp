// stream_worker.cpp - PARALLEL NV12 version (drone_seminar_parallel).
//
// Derived from drone_seminar/vision-local/src/stream_worker.cpp, itself
// derived from whisper_yolo/src/stream_worker.cpp (read-only, not modified).
//
// WHAT CHANGED FROM drone_seminar
//
// drone_seminar built TWO graphs per stream:
//
//     camera graph : libcamerasrc NV12 -> appsink            (pulled by the app)
//     video graph  : appsrc RGB -> VideoSenderRawIngress(videoconvert RGB->NV12)
//                    -> neatencoder -> h264parse -> rtph264pay -> udpsink
//
// and the app converted every frame NV12 -> BGR -> RGB and pushed it back into the
// video graph.  The CPU RGB->NV12 videoconvert drained at ~19 fps, so frames stood
// in the chain for ~0.9 s.
//
// This file builds ONE graph per stream that branches the camera's own NV12 buffer
// before the application ever sees it - the pattern proven in
// yolov11s-1280-detection-pipeline:
//
//     CameraInputWithCaptureBuffers (NV12)
//          -> Branch --+-> VideoSender (direct NV12 -> neatencoder -> RTP/UDP)
//                      +-> Output("frames", Latest)  -> app -> YOLO -> metadata
//
// The video branch never waits for the application: the frames output keeps only
// the newest buffer (drop=true), so a slow YOLO iteration drops inference frames
// instead of stalling the tee feeding the encoder.
//
// Video and metadata carry the same source timestamp: the encoder receives the
// camera buffer's own PTS, and the metadata is stamped from `sample.pts_ns` of the
// very buffer the inference ran on.
//
// The old appsrc/handoff/encoder-wrapper path is not compiled here; the original
// binary is kept for A/B as vision-local/build/drone-seminar-vision
// (scripts/start_pipeline.sh --legacy-video).
//
// Project-local tuning knobs (environment, read once at graph build):
//   DRONE_PRESET          realtime (default) | balanced
//   DRONE_QUEUE_DEPTH     RunOptions::queue_depth (default: SDK/preset default)
//   DRONE_FRAMES_OUTPUT   latest (default) | every
//   DRONE_OUTPUT_MEMORY   auto (default) | owned | zerocopy
//   DRONE_STATS_S         seconds between per-stream health/timing lines (default 5, 0 off)
//   DRONE_RUN_EXPORT_DIR  write the SDK's build-time Run/graph JSON export per stream

// drone_seminar_USB: the camera source is a USB (UVC) camera instead of a MIPI sensor.
//
//     v4l2src MJPEG -> neatdecoder NV12 (SiMaAI memory)          <- replaces CameraInput
//          -> Branch --+-> direct NV12 -> neatencoder -> RTP/UDP  (unchanged)
//                      +-> Output("frames", Latest) -> app -> YOLO -> metadata (unchanged)
//
// The camera is kept by its STABLE USB identity and re-resolved to its current
// /dev/videoN at every (re)start, and a camera fault is repaired on that camera alone
// (see run_loop): two USB cameras share no capture pipeline, so the MIPI edition's
// "stop every camera to rebuild one" maintenance is not used for camera faults.

#include "encoder_low_watermark.h"
#include "stream_worker.h"

#include "mla_gate.h"
#include "usb_camera.h"
#include "yolo_render.h"

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <nlohmann/json.hpp>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace nn = simaai::neat;

namespace pipeline {
namespace {

// Serialises the interleaved logging of the concurrent worker threads.
std::mutex g_log_mutex;

/// Serialises Model/Runner construction across ALL streams (see upstream: building two
/// Models at once makes the MLA refuse a stage). Construction only - never held per frame.
std::mutex g_model_build_mutex;

/// Serialises stream Graph construction and teardown PER CAMERA. The MIPI edition held one
/// lock for all cameras (libcamera's shared CameraManager). Two v4l2src cameras share
/// nothing, and with one lock a graph build hung on a vanished USB camera blocked the other
/// camera's recovery (measured). Timed, so a hung build is reported instead of waited on.
/// Construction only - never held per frame.
std::array<std::timed_mutex, 16> g_camera_build_mutex;
std::timed_mutex& camera_build_mutex(int channel) {
  return g_camera_build_mutex[static_cast<std::size_t>(channel) % g_camera_build_mutex.size()];
}

/// Graph builds abandoned by build_stream_run_bounded() because they missed their deadline.
std::atomic<int> g_abandoned_builds{0};

/// CLOCK_REALTIME in nanoseconds (the board clock is NTP-disciplined by chrony).
std::int64_t wall_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void log_line(const std::string& s) {
  std::lock_guard<std::mutex> lock(g_log_mutex);
  std::cout << s << std::endl;
}

std::string env_str(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v != nullptr && *v != '\0') ? std::string(v) : std::string(fallback);
}

int env_int(const char* key, int fallback) {
  const char* v = std::getenv(key);
  if (v == nullptr || *v == '\0')
    return fallback;
  try {
    return std::stoi(v);
  } catch (...) {
    return fallback;
  }
}

/// The device box decoder emits `uint32 count` followed by 24-byte records of
/// (x, y, w, h : int32) (score : float) (class_id : int32), in MODEL (letterboxed)
/// pixel coordinates.
constexpr std::size_t kBoxRecordBytes = 24;

/// Read one 24-byte BBOX record.
bool read_box(const std::vector<std::uint8_t>& p, std::size_t off, float min_score,
              render::Box& out) {
  std::int32_t x = 0, y = 0, w = 0, h = 0, cls = 0;
  float score = 0.0f;
  std::memcpy(&x, &p[off], 4);
  std::memcpy(&y, &p[off + 4], 4);
  std::memcpy(&w, &p[off + 8], 4);
  std::memcpy(&h, &p[off + 12], 4);
  std::memcpy(&score, &p[off + 16], 4);
  std::memcpy(&cls, &p[off + 20], 4);
  if (w <= 0 || h <= 0 || score < min_score)
    return false;
  out = {static_cast<float>(x), static_cast<float>(y), static_cast<float>(x + w),
         static_cast<float>(y + h), score, cls};
  return true;
}

/// Detection payload: `uint32 count | count x BBOX`.
std::vector<render::Box> parse_bbox_payload(const std::vector<std::uint8_t>& p, float min_score) {
  std::vector<render::Box> out;
  if (p.size() < sizeof(std::uint32_t))
    return out;
  std::uint32_t count = 0;
  std::memcpy(&count, p.data(), sizeof(count));
  const std::size_t capacity = (p.size() - sizeof(count)) / kBoxRecordBytes;
  count = std::min<std::uint32_t>(count, static_cast<std::uint32_t>(capacity));

  render::Box b;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (read_box(p, sizeof(count) + i * kBoxRecordBytes, min_score, b))
      out.push_back(b);
  }
  return out;
}

/// Segmentation payload, parsed directly rather than via `neat::decode_segmentation()`.
///
///     uint32 count | top_k x BBOX(24B) | top_k x (edge x edge) uint8 masks
///
/// Read through `copy_dense_bytes_tight()` (see upstream: `copy_payload_bytes()` fails on
/// this board for the ~400 KB segmentation output while a camera Run exists).
struct SegPayload {
  std::vector<render::Box> boxes; ///< in letterboxed model coordinates
  std::vector<std::uint8_t> masks;
  int mask_edge = 0;
  std::size_t mask_stride = 0; ///< bytes per instance mask
};

SegPayload parse_seg_payload(const std::vector<std::uint8_t>& p, int top_k, float min_score) {
  SegPayload out;
  if (top_k <= 0 || p.size() < sizeof(std::uint32_t))
    return out;

  std::uint32_t count = 0;
  std::memcpy(&count, p.data(), sizeof(count));

  const std::size_t box_block = static_cast<std::size_t>(top_k) * kBoxRecordBytes;
  if (p.size() < sizeof(count) + box_block)
    return out;

  const std::size_t mask_block = p.size() - sizeof(count) - box_block;
  out.mask_stride = mask_block / static_cast<std::size_t>(top_k);
  const auto edge = static_cast<int>(std::lround(std::sqrt(static_cast<double>(out.mask_stride))));
  if (edge > 0 && static_cast<std::size_t>(edge) * edge == out.mask_stride)
    out.mask_edge = edge;

  const std::size_t mask_base = sizeof(count) + box_block;
  const auto n = std::min<std::size_t>(count, static_cast<std::size_t>(top_k));
  render::Box b;
  for (std::size_t i = 0; i < n; ++i) {
    if (!read_box(p, sizeof(count) + i * kBoxRecordBytes, min_score, b))
      continue;
    out.boxes.push_back(b);
    const std::size_t src = mask_base + i * out.mask_stride;
    if (out.mask_edge > 0 && src + out.mask_stride <= p.size())
      out.masks.insert(out.masks.end(), p.begin() + static_cast<long>(src),
                       p.begin() + static_cast<long>(src + out.mask_stride));
    else
      out.masks.insert(out.masks.end(), out.mask_stride, 0);
  }
  return out;
}

nn::Model::Options model_options(const StreamConfig& cfg) {
  nn::Model::Options mo;
  mo.preprocess.kind = nn::InputKind::Image;
  mo.preprocess.enable = nn::AutoFlag::On;
  // Frames reach the model as BGR cv::Mat (OpenCV native) after the letterbox.
  mo.preprocess.color_convert.input_format = nn::PreprocessColorFormat::BGR;
  // COCO_YOLO == x/255 with mean 0 / std 1 -- exactly how both archives were compiled.
  mo.preprocess.preset = nn::NormalizePreset::COCO_YOLO;
  mo.decode_type =
      cfg.task == Task::Segment ? nn::BoxDecodeType::YoloV26Seg : nn::BoxDecodeType::YoloV26;
  mo.score_threshold = cfg.min_score;
  mo.nms_iou_threshold = cfg.nms_iou;
  mo.top_k = cfg.max_detections;
  return mo;
}

/// RunOptions for the YOLO Model::Runner.
///
/// `input_timeout_ms` is raised so an idle Runner is not torn down by the framework's
/// input-stream timeout: measured (runtime/cp/diag-pause.log), a Runner left without
/// input for ~5 s answers `runtime.output_timeout` on its next run(), and every recovery
/// that parks a stream for longer than that cascaded into another Runner restart.
/// Each run() still passes its own `infer_timeout_ms`, so a hung inference is still caught.
nn::RunOptions runner_run_options() {
  nn::RunOptions run_opt;
  run_opt.preset = nn::RunPreset::Balanced;
  run_opt.queue_depth = 4;
  run_opt.overflow_policy = nn::OverflowPolicy::Block;
  run_opt.output_memory = nn::OutputMemory::Owned;
  run_opt.input_timeout_ms = env_int("DRONE_RUNNER_INPUT_TIMEOUT_MS", 3600000);
  return run_opt;
}

/// Thrown by build_stream_run() when the USB camera's stable identity is not on the bus.
struct CameraAbsent : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// THE USB CAMERA SOURCE: v4l2src on the node the camera has NOW, decoded to NV12.
///
///   MJPG: v4l2src ! image/jpeg caps ! queue(leaky, 2) ! neatdecoder dec-type=mjpeg NV12
///         -> NV12 in SiMaAI memory, the same kind of buffer libcamerasrc delivered in the
///            MIPI edition, so the direct encoder branch and the app branch are unchanged.
///   YUYV: v4l2src ! YUY2 caps ! queue(leaky, 2) ! videoconvert ! NV12 (CPU, system memory)
///
/// The leaky queue plays the role of CameraInput's live-source queue: a slow consumer
/// drops a frame instead of back-pressuring the USB capture. jpegparse is left out by
/// default (measured: neatdecoder takes the camera's JPEG frames directly, and jpegparse
/// warns "Failed to parse app0 segment" on every C922 frame); DRONE_USB_JPEGPARSE=1 adds it.
std::string add_usb_source(nn::Graph& g, const StreamConfig& cfg, const std::string& node) {
  const std::string ch = std::to_string(cfg.channel);
  const std::string size = "width=" + std::to_string(cfg.cam_width) +
                           ",height=" + std::to_string(cfg.cam_height) +
                           ",framerate=" + std::to_string(cfg.cam_fps) + "/1";
  const std::string queue = " ! queue name=usb" + ch +
                            "_queue leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0";
  std::string src = "v4l2src name=usb" + ch + "_src device=" + node + " io-mode=mmap";
  if (cfg.pixel_format == "YUYV") {
    src += " ! video/x-raw,format=YUY2," + size + queue + " ! videoconvert name=usb" + ch +
           "_convert ! video/x-raw,format=NV12," + size;
    g.add(nn::nodes::Custom(src, nn::InputRole::Source));
    return src;
  }
  src += " ! image/jpeg," + size;
  if (env_int("DRONE_USB_JPEGPARSE", 0) != 0)
    src += " ! jpegparse name=usb" + ch + "_jpegparse";
  src += queue;
  g.add(nn::nodes::Custom(src, nn::InputRole::Source));

  nn::SimaDecodeOptions dopt;
  dopt.type = nn::SimaDecodeType::MJPEG;
  dopt.out_format = nn::FormatTag::NV12;
  dopt.next_element = env_str("DRONE_USB_DECODER_NEXT", "CVU");
  dopt.dec_width = static_cast<int>(cfg.cam_width);
  dopt.dec_height = static_cast<int>(cfg.cam_height);
  dopt.dec_fps = static_cast<int>(cfg.cam_fps);
  auto decoder = nn::nodes::SimaDecode(dopt);
  g.add(decoder);
  return src + " ! " + decoder->backend_fragment(1);
}

/// VideoSender for the RAW NV12 branch, sized and paced exactly like the camera caps so
/// the SDK's raw ingress can specialise to direct NV12 and the encoder runs at the
/// negotiated camera rate.
nn::nodes::groups::VideoSenderOptions video_sender_options(const StreamConfig& cfg) {
  auto vopt = nn::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      static_cast<int>(cfg.cam_width), static_cast<int>(cfg.cam_height),
      static_cast<int>(cfg.cam_fps));
  vopt.host = cfg.insight_host;
  vopt.channel = cfg.channel; // -> port = video_port_base + channel
  vopt.video_port_base = cfg.video_port_base;
  vopt.encoder.bitrate_kbps = cfg.bitrate_kbps;
  return vopt;
}

nn::OutputOptions frames_output_options() {
  const std::string mode = env_str("DRONE_FRAMES_OUTPUT", "latest");
  if (mode == "every")
    return nn::OutputOptions::EveryFrame(4);
  return nn::OutputOptions::Latest(); // drop=true, max_buffers=1: the app can never stall the tee
}

nn::RunOptions stream_run_options(const StreamConfig& cfg, std::string& summary) {
  nn::RunOptions ro;
  // Build-time Run/graph JSON export: the proof of what each materialized pipeline
  // segment (including the VideoSender branch) actually contains.
  const std::string export_dir = env_str("DRONE_RUN_EXPORT_DIR", "");
  if (!export_dir.empty()) {
    ro.run_export.path = export_dir + "/stream_ch" + std::to_string(cfg.channel) + ".json";
    ro.run_export.label = "stream_ch" + std::to_string(cfg.channel);
  }
  const std::string preset = env_str("DRONE_PRESET", "realtime");
  if (preset == "balanced") {
    ro.preset = nn::RunPreset::Balanced;
  } else {
    ro.preset = nn::RunPreset::Realtime;
    ro.overflow_policy = nn::OverflowPolicy::KeepLatest;
  }
  const int depth = env_int("DRONE_QUEUE_DEPTH", -1);
  if (depth > 0)
    ro.queue_depth = depth;
  const std::string mem = env_str("DRONE_OUTPUT_MEMORY", "auto");
  if (mem == "owned")
    ro.output_memory = nn::OutputMemory::Owned;
  else if (mem == "zerocopy")
    ro.output_memory = nn::OutputMemory::ZeroCopy;

  summary = "preset=" + std::string(preset == "balanced" ? "Balanced" : "Realtime") +
            " overflow=" + std::string(preset == "balanced" ? "default" : "KeepLatest") +
            " queue_depth=" + (depth > 0 ? std::to_string(depth) : std::string("default")) +
            " output_memory=" + mem +
            " frames_output=" + env_str("DRONE_FRAMES_OUTPUT", "latest");
  return ro;
}

/// THE STREAM GRAPH: one USB camera, decoded to NV12, branched to the encoder and the app.
///
/// Also used by the stall/pause/reconnect recovery paths, which must rebuild it
/// identically. `current_node` is the /dev/videoN the camera had last time; the stable
/// identity is resolved again here, so a re-enumerated camera is found under its new
/// number. Throws CameraAbsent when the camera is not on the bus.
/// With video disabled (--no-video) it degrades to the plain camera -> Output graph.
std::unique_ptr<nn::Run> build_stream_run(const StreamConfig& cfg, const std::string& tag,
                                          std::string& current_node) {
  std::unique_lock<std::timed_mutex> build_lock(camera_build_mutex(cfg.channel), std::defer_lock);
  if (!build_lock.try_lock_for(std::chrono::seconds(2)))
    throw std::runtime_error("an earlier stream graph build of this camera is still hung "
                             "(abandoned); it is rebuilt once that build returns - restart the "
                             "demo if it never does");
  const std::string ch = std::to_string(cfg.channel);
  std::string run_summary;
  const nn::RunOptions ro = stream_run_options(cfg, run_summary);

  const auto cam = usbcam::resolve(cfg.usb_identity);
  if (!cam)
    throw CameraAbsent("USB camera " + cfg.usb_identity + " (\"" + cfg.camera_card +
                       "\", last seen as " + current_node + ") is not connected");
  if (!current_node.empty() && cam->video_node != current_node)
    log_line(tag + " [usb] CAMERA RE-ENUMERATED: " + cfg.usb_identity + " is now " +
             cam->video_node + " (was " + current_node + ")");
  current_node = cam->video_node;
  usbcam::Mode want;
  want.fourcc = cfg.pixel_format;
  want.width = static_cast<int>(cfg.cam_width);
  want.height = static_cast<int>(cfg.cam_height);
  want.fps = static_cast<int>(cfg.cam_fps);
  if (!cam->supports(want))
    throw std::runtime_error("USB camera " + cfg.usb_identity + " on " + current_node +
                             " no longer offers " + usbcam::format_mode(want));
  const auto holders = usbcam::node_holders(current_node);
  if (!holders.empty()) {
    std::string pids;
    for (int p : holders)
      pids += " " + std::to_string(p);
    log_line(tag + " [usb] WARNING: " + current_node + " is already open in pid(s)" + pids);
  }
  std::string ctrl_note;
  usbcam::disable_dynamic_framerate(current_node, ctrl_note);
  log_line(tag + " [usb] " + current_node + " (" + cfg.usb_identity + "): " + ctrl_note);

  if (!cfg.video_enabled) {
    nn::Graph cam_graph("usb_camera_" + ch);
    const std::string frag = add_usb_source(cam_graph, cfg, current_node);
    cam_graph.add(nn::nodes::Output("frames", frames_output_options()));
    log_line(tag + " building camera-only graph (video disabled): " + frag + ", " + run_summary);
    return std::make_unique<nn::Run>(cam_graph.build(ro));
  }

  nn::Graph source("source");
  const std::string source_fragment = add_usb_source(source, cfg, current_node);
  log_line(tag + " [usb] source: " + source_fragment);

  auto branch = nn::graphs::Branch("source", {"video", "frames"});

  // THE VIDEO BRANCH.
  //
  // "direct" (default): Input -> neatencoder -> h264parse -> rtph264pay -> udpsink,
  //   built from the same public nodes the VideoSender group uses, minus its raw
  //   ingress. Measured (runtime/cp/cp2b/gst-elements.log): behind a Branch the SDK
  //   ingress specialises to `VideoSenderRawIngress[convert_to_nv12]` and emits
  //   `capsfilter ! videoconvert ! capsfilter NV12` even though the branch carries
  //   camera NV12 already. The camera buffer is NV12 in SimaAI memory at exactly the
  //   encoder's configured size, so the encoder takes it as is.
  // "sdk": the stock VideoSender group, for A/B comparison.
  const auto vopt = video_sender_options(cfg);
  nn::Graph video("video");
  // YUYV frames are converted on the CPU into system memory, which the direct encoder
  // branch cannot take; they go through the SDK's VideoSender ingress instead.
  const std::string ingress_mode =
      cfg.pixel_format == "YUYV" ? std::string("sdk") : env_str("DRONE_VIDEO_INGRESS", "direct");
  if (ingress_mode == "sdk") {
    video.connect(nn::nodes::Input("video"), nn::nodes::groups::VideoSender(vopt));
    log_line(tag + " video branch: SDK VideoSender group (raw ingress chosen by the SDK)");
  } else {
    // Optional bound on neatencoder's input queue (element default: block above 30),
    // i.e. on how many CAMERA buffers the encoder may hold at once.
    const int ipq_max = env_int("DRONE_ENC_IPQ_MAX", 0);
    const int ipq_min = env_int("DRONE_ENC_IPQ_MIN", 1);
    std::shared_ptr<nn::Node> encoder;
    if (ipq_max > 0) {
      encoder = std::make_shared<drone_seminar::EncoderLowWatermark>(
          vopt.width(), vopt.height(), vopt.fps(), vopt.encoder.bitrate_kbps, vopt.encoder.profile,
          vopt.encoder.level, ipq_min, ipq_max);
    } else {
      encoder = nn::nodes::H264EncodeSima(vopt.width(), vopt.height(), vopt.fps(),
                                          vopt.encoder.bitrate_kbps, vopt.encoder.profile,
                                          vopt.encoder.level);
    }
    nn::UdpOutputOptions udp_opt;
    udp_opt.host = vopt.host;
    udp_opt.port = vopt.video_port();
    udp_opt.sync = vopt.sync;
    udp_opt.async = vopt.async;
    video.add(nn::nodes::Input("video"));
    video.add(encoder);
    video.add(nn::nodes::H264Parse(vopt.rtp.config_interval));
    video.add(nn::nodes::H264Packetize(nn::H264Packetize::PayloadType(vopt.rtp.payload_type),
                                       nn::H264Packetize::ConfigInterval(vopt.rtp.config_interval)));
    video.add(nn::nodes::UdpOutput(udp_opt));
    log_line(tag + " video branch: DIRECT NV12 -> " + encoder->backend_fragment(1) +
             " ! h264parse ! rtph264pay pt=" + std::to_string(vopt.rtp.payload_type) +
             " ! udpsink " + vopt.host + ":" + std::to_string(vopt.video_port()));
  }

  nn::Graph frames("frames");
  frames.add(nn::nodes::Output("frames", frames_output_options()));

  nn::Graph graph("stream_" + ch);
  graph.connect(source, branch);
  graph.connect(branch, video);
  graph.connect(branch, frames);

  log_line(tag + " building parallel graph: Camera NV12 -> Branch{ video -> VideoSender, "
                 "frames -> app }, " + run_summary);
  return std::make_unique<nn::Run>(graph.build(ro));
}

/// build_stream_run() with a deadline. Measured on this board: a USB camera that re-enumerated
/// and then dropped off the bus again while its graph was being built left that build
/// spinning inside the framework for good. The build therefore runs on a helper thread; if it
/// misses `timeout_ms` it is abandoned - that thread closes the Run itself should it ever
/// return - and the caller carries on, so this camera can retry and nothing else waits on it.
std::unique_ptr<nn::Run> build_stream_run_bounded(const StreamConfig& cfg, const std::string& tag,
                                                  std::string& current_node, int timeout_ms) {
  struct Shared {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    bool abandoned = false;
    std::unique_ptr<nn::Run> run;
    std::exception_ptr error;
    std::string node;
  };
  auto sh = std::make_shared<Shared>();
  sh->node = current_node;
  std::thread([sh, cfg, tag]() {
    std::string node = sh->node;
    std::unique_ptr<nn::Run> run;
    std::exception_ptr error;
    try {
      run = build_stream_run(cfg, tag, node);
    } catch (...) {
      error = std::current_exception();
    }
    std::unique_lock<std::mutex> lk(sh->mu);
    if (sh->abandoned) {
      lk.unlock();
      if (run) {
        try {
          run->close();
        } catch (...) {
        }
      }
      log_line(tag + " [usb] an abandoned stream graph build returned late" +
               std::string(run ? "; its graph was closed" : ""));
      return;
    }
    sh->run = std::move(run);
    sh->error = error;
    sh->node = node;
    sh->done = true;
    sh->cv.notify_all();
  }).detach();

  std::unique_lock<std::mutex> lk(sh->mu);
  if (!sh->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return sh->done; })) {
    sh->abandoned = true;
    const int n = ++g_abandoned_builds;
    throw std::runtime_error("stream graph build HUNG: not finished after " +
                             std::to_string(timeout_ms) + " ms; abandoned (hung build #" +
                             std::to_string(n) + " in this process)");
  }
  current_node = sh->node;
  if (sh->error)
    std::rethrow_exception(sh->error);
  return std::move(sh->run);
}

/// What one decoded frame yields: boxes in frame space, plus the per-instance masks
/// (segmentation only) the metadata polygons are built from.
struct Decoded {
  std::vector<render::Box> boxes;
  SegPayload seg; ///< empty for detection
};

Decoded decode_output(const StreamConfig& cfg, const nn::TensorList& out,
                      const render::LetterboxInfo& lb, const cv::Size& frame_size) {
  Decoded d;
  if (out.empty())
    return d;

  const auto payload = out.front().copy_dense_bytes_tight();

  if (cfg.task != Task::Segment) {
    d.boxes = parse_bbox_payload(payload, cfg.min_score);
    render::remap_boxes(d.boxes, lb, frame_size.width, frame_size.height);
    return d;
  }

  d.seg = parse_seg_payload(payload, cfg.max_detections, cfg.min_score);
  d.boxes = d.seg.boxes;
  render::remap_boxes(d.boxes, lb, frame_size.width, frame_size.height);
  return d;
}

/// One instance mask as a cv::Mat view into the decoded payload, or empty.
cv::Mat mask_at(const SegPayload& seg, std::size_t i) {
  if (seg.mask_edge <= 0 || seg.mask_stride == 0 ||
      (i + 1) * seg.mask_stride > seg.masks.size())
    return {};
  return cv::Mat(seg.mask_edge, seg.mask_edge, CV_8UC1,
                 const_cast<std::uint8_t*>(seg.masks.data() + i * seg.mask_stride));
}

/// Burn boxes (and instance masks) into a frame. Still-image test only.
void draw_overlay(const StreamConfig& cfg, const Decoded& d,
                  const std::vector<std::string>& labels, const render::LetterboxInfo& lb,
                  cv::Mat& frame) {
  if (cfg.task == Task::Segment) {
    for (std::size_t i = 0; i < d.boxes.size(); ++i) {
      const cv::Mat m = mask_at(d.seg, i);
      if (!m.empty())
        render::draw_instance_mask(frame, m, d.boxes[i], lb, cfg.mask_alpha, cfg.mask_threshold);
    }
  }
  render::draw_boxes(frame, d.boxes, labels);
}

std::string label_for(const std::vector<std::string>& labels, int class_id) {
  return (class_id >= 0 && class_id < static_cast<int>(labels.size()))
             ? labels[static_cast<std::size_t>(class_id)]
             : ("id" + std::to_string(class_id));
}

/// Build the Insight metadata `data` object for this frame.
///
///   detection    {"objects":  [{id,label,confidence,bbox:[x,y,w,h]}]}
///   segmentation {"segments": [{id,label,confidence,bbox:[x,y,w,h],
///                               mask_format:"polygon", mask:[[x,y],...]}]}
std::string metadata_data_json(const StreamConfig& cfg, const Decoded& d,
                               const std::vector<std::string>& labels,
                               const render::LetterboxInfo& lb, const cv::Size& frame_size,
                               int& dropped, const nlohmann::json* timing = nullptr) {
  const bool seg = cfg.task == Task::Segment;
  const char* key = seg ? "segments" : "objects";

  std::vector<std::size_t> order(d.boxes.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return d.boxes[a].score > d.boxes[b].score;
  });

  nlohmann::json entries = nlohmann::json::array();
  std::size_t bytes = std::strlen("{\"segments\":[]}");
  std::size_t emitted = 0;

  for (const std::size_t i : order) {
    const render::Box& b = d.boxes[i];
    const cv::Rect r = render::frame_rect_for(b, frame_size);
    nlohmann::json entry = {
        {"id", std::to_string(i)},
        {"label", label_for(labels, b.class_id)},
        {"confidence", b.score},
        {"bbox", {r.x, r.y, r.width, r.height}},
    };
    if (seg) {
      const auto poly =
          render::mask_polygon(mask_at(d.seg, i), b, lb, frame_size, cfg.mask_threshold);
      if (poly.size() < 3)
        continue;
      nlohmann::json points = nlohmann::json::array();
      for (const auto& pt : poly)
        points.push_back({pt.x, pt.y});
      entry["mask_format"] = "polygon";
      entry["mask"] = std::move(points);
    }

    const std::size_t entry_bytes = entry.dump().size() + 1;
    if (bytes + entry_bytes > static_cast<std::size_t>(cfg.metadata_budget_bytes))
      break;
    bytes += entry_bytes;
    entries.push_back(std::move(entry));
    ++emitted;
  }

  dropped = static_cast<int>(d.boxes.size() - emitted);
  nlohmann::json out{{key, std::move(entries)}};
  if (timing != nullptr)
    out["timing"] = *timing;
  return out.dump();
}

/// `render::letterbox()` with the same geometry, interpolation and pad value, writing into
/// buffers owned by the stream instead of allocating two new Mats per frame.
void letterbox_into(const cv::Mat& src, int size, render::LetterboxInfo& info, cv::Mat& resized,
                    cv::Mat& out) {
  constexpr int kPadValue = 114; // same as yolo_render.cpp
  info.size = size;
  info.scale = std::min(static_cast<float>(size) / static_cast<float>(src.cols),
                        static_cast<float>(size) / static_cast<float>(src.rows));
  const int new_w = std::max(1, static_cast<int>(std::round(src.cols * info.scale)));
  const int new_h = std::max(1, static_cast<int>(std::round(src.rows * info.scale)));
  info.off_x = (size - new_w) / 2;
  info.off_y = (size - new_h) / 2;

  cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  out.create(size, size, src.type());
  out.setTo(cv::Scalar::all(kPadValue));
  resized.copyTo(out(cv::Rect(info.off_x, info.off_y, new_w, new_h)));
}

/// NV12 camera tensor -> BGR into a reused Mat, reading the planes in place.
/// Returns false when the tensor cannot be mapped as NV12 (caller falls back to the SDK).
bool nv12_to_bgr_into(const nn::Tensor& t, cv::Mat& bgr) {
  const auto mapped = t.map_nv12_read();
  if (!mapped || mapped->view.y == nullptr || mapped->view.uv == nullptr)
    return false;
  const auto& v = mapped->view;
  const cv::Mat y(v.height, v.width, CV_8UC1, const_cast<std::uint8_t*>(v.y),
                  static_cast<std::size_t>(v.y_stride));
  const cv::Mat uv(v.height / 2, v.width / 2, CV_8UC2, const_cast<std::uint8_t*>(v.uv),
                   static_cast<std::size_t>(v.uv_stride));
  cv::cvtColorTwoPlane(y, uv, bgr, cv::COLOR_YUV2BGR_NV12);
  return !bgr.empty();
}

/// Per-stage timing accumulator for the low-frequency health line.
struct StageAcc {
  double sum = 0.0;
  double max = 0.0;
  long n = 0;
  void add(double v) {
    sum += v;
    max = std::max(max, v);
    ++n;
  }
  double mean() const { return n > 0 ? sum / static_cast<double>(n) : 0.0; }
};

std::string fmt_acc(const char* name, const StageAcc& a) {
  char buf[96];
  std::snprintf(buf, sizeof(buf), " %s=%.1f/%.1f", name, a.mean(), a.max);
  return buf;
}

} // namespace

std::string task_name(Task task) {
  return task == Task::Segment ? "segmentation" : "detection";
}

int video_port_of(const StreamConfig& cfg) {
  return cfg.video_port_base + cfg.channel;
}

int metadata_port_of(const StreamConfig& cfg) {
  return cfg.metadata_port_base + cfg.channel;
}

int abandoned_builds() {
  return g_abandoned_builds.load();
}

// ---------------------------------------------------------------------------

struct StreamRunner::Impl {
  StreamConfig cfg;
  std::string tag;
  std::vector<std::string> labels;

  /// ONE Run per stream: camera + Branch + VideoSender + frames output.
  std::unique_ptr<nn::Run> cam_run;
  std::unique_ptr<nn::Model> model;
  std::unique_ptr<nn::Model::Runner> runner;
  std::unique_ptr<nn::MetadataSender> meta;

  cv::Mat pending_frame;            ///< the probe frame from open(), processed first
  std::int64_t pending_pts_ns = -1; ///< its source PTS
  std::int64_t pending_wall_ns = -1; ///< board wall clock when it was pulled
  std::int64_t pending_mono_ns = -1; ///< CLOCK_MONOTONIC when it was pulled
  std::chrono::steady_clock::time_point video_t0{};
  render::LetterboxInfo lb;
  int frame_w = 0;
  int frame_h = 0;
  int result = 0;
  bool closed = false;

  /// The /dev/videoN this stream's USB camera had at its last (re)start.
  std::string camera_node;
  /// True while the USB camera is known to be missing from the bus.
  bool camera_absent = false;
  /// When the camera was first seen on the bus again (settle timer), or epoch.
  std::chrono::steady_clock::time_point camera_present_since{};

  mla::Gate* gate = nullptr;
  std::atomic<long> frames_done{0};
  std::atomic<double> fps{0.0};
  int camera_restarts = 0;
  int runner_restarts = 0;

  explicit Impl(StreamConfig c) : cfg(std::move(c)) {
    tag = "[ch" + std::to_string(cfg.channel) + " " + task_name(cfg.task) + "]";
    camera_node = cfg.video_node;
  }
};

StreamRunner::StreamRunner(StreamConfig cfg) : impl_(std::make_unique<Impl>(std::move(cfg))) {}
StreamRunner::~StreamRunner() {
  close();
}

const StreamConfig& StreamRunner::config() const {
  return impl_->cfg;
}
int StreamRunner::result() const {
  return impl_->result;
}
long StreamRunner::frames_done() const {
  return impl_->frames_done.load(std::memory_order_relaxed);
}
double StreamRunner::fps() const {
  return impl_->fps.load(std::memory_order_relaxed);
}
int StreamRunner::camera_restarts() const {
  return impl_->camera_restarts;
}
int StreamRunner::runner_restarts() const {
  return impl_->runner_restarts;
}

void StreamRunner::open() {
  Impl& s = *impl_;
  const StreamConfig& cfg = s.cfg;
  s.labels = render::load_labels(cfg.labels_path);

  // Checkpoint 6: OpenCV's pool is process-wide and shared by BOTH streams' conversions.
  const int cv_threads = env_int("DRONE_CV_THREADS", -1);
  if (cv_threads >= 0)
    cv::setNumThreads(cv_threads);
  log_line(s.tag + " OpenCV threads: " + std::to_string(cv::getNumThreads()) +
           (cv_threads >= 0 ? " (DRONE_CV_THREADS)" : " (OpenCV default)"));

  if (cfg.burn_in_overlay || cfg.burn_in_hud)
    log_line(s.tag + " WARNING: --burn-in/--hud are not supported by the parallel NV12 "
                     "video path (the encoder never sees application pixels); ignored. "
                     "Use scripts/start_pipeline.sh --legacy-video for burn-in.");

  // ---- the stream graph: USB camera -> NV12, branched to VideoSender and to the app ----
  s.cam_run = build_stream_run_bounded(cfg, s.tag, s.camera_node, cfg.camera_build_timeout_ms);
  s.video_t0 = std::chrono::steady_clock::now();
  log_line(s.tag + " camera \"" + cfg.camera_card + "\" (" + s.camera_node + ", " +
           cfg.usb_identity + ") " + std::to_string(cfg.cam_width) + "x" +
           std::to_string(cfg.cam_height) + "@" + std::to_string(cfg.cam_fps) + " " +
           cfg.pixel_format + " -> NV12");

  // Pull one frame: proves the sensor streams and seeds the model with real geometry.
  nn::Sample first;
  nn::PullError perr;
  if (s.cam_run->pull("frames", cfg.first_frame_timeout_ms, first, &perr) != nn::PullStatus::Ok)
    throw std::runtime_error("no first frame from camera \"" + cfg.camera_card +
                             "\": " + perr.message);
  s.pending_wall_ns = wall_now_ns();
  s.pending_mono_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch()).count();
  s.pending_frame = nn::tensors_from_sample(first, true).front().to_cv_mat_copy(nn::ImageType::BGR);
  s.pending_pts_ns = first.pts_ns;
  if (s.pending_frame.empty())
    throw std::runtime_error("first camera frame decoded empty");
  s.frame_w = s.pending_frame.cols;
  s.frame_h = s.pending_frame.rows;
  if (cfg.video_enabled &&
      (s.frame_w != static_cast<int>(cfg.cam_width) || s.frame_h != static_cast<int>(cfg.cam_height)))
    throw std::runtime_error("camera delivered " + std::to_string(s.frame_w) + "x" +
                             std::to_string(s.frame_h) + " but the VideoSender was sized " +
                             std::to_string(cfg.cam_width) + "x" + std::to_string(cfg.cam_height));

  // ---- model ----------------------------------------------------------------
  if (cfg.inference_enabled) {
    std::lock_guard<std::mutex> build_lock(g_model_build_mutex);
    s.model = std::make_unique<nn::Model>(cfg.model_path, model_options(cfg));
    cv::Mat seed = render::letterbox(s.pending_frame, cfg.model_size, s.lb);

    const nn::RunOptions run_opt = runner_run_options();
    s.runner = std::make_unique<nn::Model::Runner>(
        s.model->build(std::vector<cv::Mat>{seed}, nn::Model::RouteOptions{}, run_opt));
    s.runner->run(std::vector<cv::Mat>{seed}, cfg.infer_timeout_ms); // warm up
    log_line(s.tag + " model " + cfg.model_path + " (" + task_name(cfg.task) + ", warmed up)");
  } else {
    render::letterbox(s.pending_frame, cfg.model_size, s.lb);
    log_line(s.tag + " DIAGNOSTIC --no-inference: no Model and no Runner built");
  }

  // ---- UDP metadata output --------------------------------------------------
  if (cfg.metadata_enabled) {
    nn::MetadataSenderOptions mopt;
    mopt.host = cfg.insight_host;
    mopt.channel = cfg.channel;
    mopt.metadata_port_base = cfg.metadata_port_base;
    std::string merr;
    s.meta = std::make_unique<nn::MetadataSender>(mopt, &merr);
    if (!s.meta->ok())
      throw std::runtime_error("metadata sender failed: " + merr);
    log_line(s.tag + " sending " + (cfg.task == Task::Segment ? "segmentation" : "object-detection") +
             " JSON -> udp://" + cfg.insight_host + ":" + std::to_string(s.meta->metadata_port()));
  }

  if (cfg.video_enabled) {
    const auto vopt = video_sender_options(cfg);
    log_line(s.tag + " [parallel] video: camera NV12 -> Branch -> VideoSender(" +
             std::to_string(vopt.width()) + "x" + std::to_string(vopt.height()) + "@" +
             std::to_string(vopt.fps()) + ") -> RTP/H.264 udp://" + cfg.insight_host + ":" +
             std::to_string(vopt.video_port()) + " (pt " + std::to_string(vopt.rtp.payload_type) +
             ", " + std::to_string(cfg.bitrate_kbps) + " kbps, " + vopt.encoder.profile + " " +
             vopt.encoder.level + ") | RGB appsrc: none | app pixels in video: none");
  } else {
    log_line(s.tag + " UDP video output DISABLED");
  }
  log_line(s.tag + " [parallel] metadata timestamp = source buffer PTS (ms), clock option \"" +
           cfg.metadata_clock + "\"");
}

void StreamRunner::set_mla_gate(mla::Gate* gate) {
  impl_->gate = gate;
}

void StreamRunner::run_loop(std::atomic<bool>& stop) {
  Impl& s = *impl_;
  const StreamConfig& cfg = s.cfg;
  try {
    if (!s.cam_run || (cfg.inference_enabled && !s.runner))
      throw std::runtime_error("run_loop() called before open()");

    using clock = std::chrono::steady_clock;
    using ms_d = std::chrono::duration<double, std::milli>;

    const int stats_s = env_int("DRONE_STATS_S", 5);
    // DIAGNOSTIC (off unless DRONE_SNAPSHOT_DIR is set): see the snapshot block below.
    const std::string snapshot_dir = env_str("DRONE_SNAPSHOT_DIR", "");
    const int snapshot_every_s = std::max(1, env_int("DRONE_SNAPSHOT_EVERY_S", 10));
    clock::time_point last_snapshot{};
    const bool timing_meta = env_int("DRONE_TIMING_META", 1) != 0;
    std::int64_t pull_wall_ns = s.pending_wall_ns;
    std::int64_t pull_mono_ns = s.pending_mono_ns;
    double mono_minus_pts_min = 1e18;
    const int debug_pause_s = env_int("DRONE_DEBUG_INFER_PAUSE_S", 0);
    const int debug_pause_ch = env_int("DRONE_DEBUG_INFER_PAUSE_CH", 1);
    const long debug_pause_at = env_int("DRONE_DEBUG_INFER_PAUSE_AT", 450);
    bool debug_paused = false;
    auto last = clock::now();
    double fps_ema = 0.0;
    long processed = 0;
    cv::Mat frame = s.pending_frame; // the probe frame is the first one processed
    std::int64_t pts_ns = s.pending_pts_ns;
    bool have_frame = !frame.empty();
    // Checkpoint 5: inference-branch buffers reused across frames.
    bool use_fast_convert = env_str("DRONE_NV12_CONVERT", "opencv") != "sdk";
    bool convert_checked = false;
    const bool use_prealloc_letterbox = env_str("DRONE_LETTERBOX", "prealloc") != "upstream";
    cv::Mat frame_buf, lb_resized, lb_buf;
    log_line(s.tag + " [parallel] inference conversion: " +
             (use_fast_convert ? "NV12 planes -> BGR into a reused buffer (checked against SDK)"
                               : "SDK to_cv_mat_copy") +
             ", letterbox: " + (use_prealloc_letterbox ? "reused buffers" : "upstream allocating"));
    log_line(s.tag + " [recovery] fixes active: usb-local-camera-restart usb-rediscovery "
                     "silent-timer-lease-fix pull-error-recovery concurrent-restart-yield");
    long meta_sent = 0, meta_over_budget = 0, meta_failed = 0;
    long pts_gaps = 0;           ///< pulled frames more than 3.5 frame intervals apart
    std::int64_t prev_pts_ns = -1;
    clock::time_point stalled_since{};
    clock::time_point last_pull_error_log{};
    int restarts = 0;

    // Low-frequency health window.
    StageAcc acc_wait, acc_convert, acc_letterbox, acc_infer, acc_decode, acc_meta, acc_loop;
    long win_frames = 0;
    std::int64_t win_first_pts = -1, win_last_pts = -1;
    auto win_start = clock::now();

    // The metadata timestamp for one frame. In the parallel graph the encoder receives the
    // camera buffer itself, so its RTP timestamp is derived from that buffer's PTS and the
    // metadata must be stamped from the same PTS.
    auto stamp_ms = [&](std::int64_t p, clock::time_point now) -> std::int64_t {
      std::int64_t ts = -1;
      if (cfg.metadata_clock == "pipeline")
        ts = std::chrono::duration_cast<std::chrono::milliseconds>(now - s.video_t0).count();
      else
        ts = p >= 0 ? p / 1000000 : -1;
      return ts >= 0 ? ts + cfg.metadata_offset_ms : ts;
    };

    int runner_restarts = 0;
    auto rebuild_runner = [&](const cv::Mat& seed) {
      std::lock_guard<std::mutex> build_lock(g_model_build_mutex);
      if (s.runner)
        s.runner->close();
      s.runner.reset();
      s.model.reset();
      // Give the accelerator time to release the old model's stages. Rebuilding
      // immediately after teardown was measured failing with
      // infra.accelerator_execution_failed (runtime/cp/cp2.log).
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      s.model = std::make_unique<nn::Model>(cfg.model_path, model_options(cfg));
      const nn::RunOptions run_opt = runner_run_options();
      s.runner = std::make_unique<nn::Model::Runner>(
          s.model->build(std::vector<cv::Mat>{seed}, nn::Model::RouteOptions{}, run_opt));
      s.runner->run(std::vector<cv::Mat>{seed}, cfg.infer_timeout_ms);
    };

    // ---- the controlled vision pause (MLA arbiter; only used by a BOARD voice server) ----
    // With the default off-board voice server nothing ever requests it. DrainToVideo and
    // DrainAndDrop are equivalent here: video is encoded inside the graph regardless of what
    // the app does, so the parked worker just consumes (and drops) inference frames.
    // StopCamera stops the whole stream graph, video included, for the window.
    auto enter_controlled_pause = [&]() -> bool {
      long dropped = 0;
      bool camera_rebuilt = false; // true only if THIS worker's camera Graph was rebuilt
      bool restore_failed = false; // USB: the camera could not be rebuilt in this window
      const auto t_enter = clock::now();
      // drone seminar: a CAMERA-FAULT maintenance window (another
      // stream is rebuilding its camera) stops this camera too, exactly like the
      // fallback's stop-camera; a voice window keeps the configured behaviour (drop:
      // the camera and the video keep running).
      const bool camera_maintenance = s.gate->camera_maintenance();
      // A worker without a camera graph (a restore failed) is treated as camera-stopped:
      // it must never pull from a missing graph while parked.
      bool stop_camera = s.gate->behaviour() == mla::PauseBehaviour::StopCamera ||
                         camera_maintenance || !s.cam_run;
      // USB edition: a stream parked only because it has no graph (its camera is unplugged
      // or not streaming yet) does not rebuild it here; run_loop's local recovery does, with
      // its settle time, probe and bounded build.
      const bool graph_missing_only = !s.cam_run && !camera_maintenance &&
                                      s.gate->behaviour() != mla::PauseBehaviour::StopCamera;
      if (graph_missing_only)
        restore_failed = true;

      if (stop_camera) {
        log_line(s.tag + " [mla] stream graph STOP (camera + video) for the " +
                 (camera_maintenance ? "camera-fault maintenance window" : "voice window"));
        try {
          if (s.cam_run) {
            std::lock_guard<std::timed_mutex> build_lock(camera_build_mutex(cfg.channel));
            s.cam_run->close();
          }
        } catch (const std::exception& e) {
          log_line(s.tag + " [mla] stream stop reported: " + std::string(e.what()));
        }
        s.cam_run.reset();
      }

      s.gate->worker_parked(s.tag);

      for (bool left = false; !left;) {
        while (s.gate->pause_requested() && !stop.load()) {
          if (stop_camera && !restore_failed &&
              s.gate->behaviour() != mla::PauseBehaviour::StopCamera &&
              !s.gate->camera_maintenance()) {
            // The camera-fault maintenance ended and a VOICE window began before this
            // worker left the parked state: bring the camera back now, so a voice
            // window never runs with this camera stopped.
            const auto t0 = clock::now();
            try {
              s.cam_run = build_stream_run_bounded(cfg, s.tag, s.camera_node,
                                                   cfg.camera_build_timeout_ms);
              stop_camera = false;
              camera_rebuilt = true;
              log_line(s.tag + " [mla] stream graph RESTORED in " +
                       std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          clock::now() - t0).count()) +
                       " ms (maintenance over; voice window keeps the camera running)");
            } catch (const std::exception& e) {
              // USB edition: a camera that cannot come back (unplugged) must not end this
              // stream; it stays stopped and run_loop keeps looking for it.
              s.cam_run.reset();
              restore_failed = true;
              log_line(s.tag + " [usb] stream graph restore failed: " + std::string(e.what()));
            }
          }
          if (stop_camera) {
            s.gate->wait_while_paused(cfg.pause_poll_ms);
            continue;
          }
          nn::Sample sample;
          nn::PullError err;
          const auto st = s.cam_run->pull("frames", cfg.pause_poll_ms, sample, &err);
          if (st != nn::PullStatus::Ok) {
            s.gate->wait_while_paused(cfg.pause_poll_ms);
            continue;
          }
          ++dropped;
        }
        if (stop.load()) {
          s.gate->force_unpark(s.tag);
          left = true;
        } else if (s.gate->try_unpark(s.tag, dropped, 0)) {
          if (s.gate->pause_requested())
            s.gate->worker_parked(s.tag);
          else
            left = true;
        }
      }

      if (stop_camera && !stop.load() && !graph_missing_only) {
        const auto t0 = clock::now();
        try {
          s.cam_run = build_stream_run_bounded(cfg, s.tag, s.camera_node,
                                               cfg.camera_build_timeout_ms);
          camera_rebuilt = true;
          const auto ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
          log_line(s.tag + " [mla] stream graph RESTORED in " + std::to_string(ms) + " ms");
        } catch (const std::exception& e) {
          s.cam_run.reset();
          log_line(s.tag + " [usb] stream graph restore after the pause failed: " +
                   std::string(e.what()) + "; the camera is looked for again");
        }
      }

      const auto window_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t_enter).count();
      log_line(s.tag + " [mla] controlled vision pause finished after " +
               std::to_string(window_ms) + " ms");
      return camera_rebuilt;
    };

    // ---- worker-initiated maintenance (secondary recovery only) ----------
    auto maintenance_window = [&](const std::string& reason, const std::function<void()>& work,
                                  bool yield_if_busy) -> bool {
      const bool gated = s.gate != nullptr;
      if (gated) {
        bool busy = false;
        s.gate->begin_maintenance(s.tag, reason, cfg.quiesce_timeout_ms,
                                  yield_if_busy ? &busy : nullptr);
        if (busy)
          return false; // FIX 4: park with the current owner at the loop top instead
      }
      try {
        if (s.cam_run) {
          std::lock_guard<std::timed_mutex> build_lock(camera_build_mutex(cfg.channel));
          s.cam_run->close();
        }
        s.cam_run.reset();
        log_line(s.tag + " [recovery] stream graph STOPPED for maintenance (" + reason + ")");
      } catch (const std::exception& e) {
        log_line(s.tag + " [recovery] stream stop reported: " + std::string(e.what()));
        s.cam_run.reset();
      }

      work();

      try {
        s.cam_run = build_stream_run_bounded(cfg, s.tag, s.camera_node,
                                             cfg.camera_build_timeout_ms);
        log_line(s.tag + " [recovery] stream graph restored after maintenance");
      } catch (const std::exception& e) {
        log_line(s.tag + " [recovery] stream restore after maintenance failed: " +
                 std::string(e.what()));
      }
      if (gated)
        s.gate->end_maintenance(s.tag);
      return true;
    };

    // ---- USB camera recovery: THIS camera only -------------------------------------
    // The MIPI edition had to park every other camera before rebuilding one (a libcamera
    // rebuild next to a streaming camera aborted the process). Two USB cameras share no
    // capture pipeline, so a USB camera fault is repaired here, on this stream alone: the
    // other camera, its YOLO stream and the voice runtime keep running. A camera that is
    // not on the bus is looked for again every camera_retry_ms, under whatever
    // /dev/videoN Linux gives it when it comes back.
    clock::time_point next_camera_attempt{};
    clock::time_point last_retry_log{};
    auto rebuild_camera_locally = [&](const std::string& reason) -> bool {
      if (s.cam_run) {
        try {
          std::lock_guard<std::timed_mutex> build_lock(camera_build_mutex(cfg.channel));
          s.cam_run->close();
        } catch (const std::exception& e) {
          log_line(s.tag + " [recovery] stream stop reported: " + std::string(e.what()));
        }
        s.cam_run.reset();
        log_line(s.tag + " [recovery] stream graph STOPPED (" + reason +
                 "); this camera only - the other stream keeps running");
      }
      prev_pts_ns = -1;
      // A camera that has just (re)appeared is left alone for camera_settle_ms, and every
      // rebuild is preceded by a short raw-V4L2 streaming probe, which cannot hang (poll with
      // a timeout). Measured: a camera that re-enumerated and dropped off the bus again
      // during the framework's graph build left that build spinning for good.
      const auto now_pre = clock::now();
      const auto present = usbcam::resolve(cfg.usb_identity);
      if (!present) {
        s.camera_present_since = {};
      } else {
        if (s.camera_present_since == clock::time_point{})
          s.camera_present_since = now_pre;
        const auto settle = std::chrono::milliseconds(cfg.camera_settle_ms);
        if (s.camera_absent && now_pre - s.camera_present_since < settle) {
          next_camera_attempt = s.camera_present_since + settle;
          return false;
        }
        usbcam::Mode want;
        want.fourcc = cfg.pixel_format;
        want.width = static_cast<int>(cfg.cam_width);
        want.height = static_cast<int>(cfg.cam_height);
        want.fps = static_cast<int>(cfg.cam_fps);
        std::vector<usbcam::ProbeStats> probe;
        if (!usbcam::probe_streaming(std::vector<usbcam::Camera>{*present}, want, 1200, probe)) {
          if (last_retry_log == clock::time_point{} ||
              clock::now() - last_retry_log >= std::chrono::seconds(10)) {
            last_retry_log = clock::now();
            log_line(s.tag + " [usb] camera " + cfg.usb_identity + " is on " +
                     present->video_node + " but does not stream " + usbcam::format_mode(want) +
                     " (" + (probe.empty() ? std::string("probe failed") : probe.front().error) +
                     "); retrying every " + std::to_string(cfg.camera_retry_ms) + " ms");
          }
          next_camera_attempt = clock::now() + std::chrono::milliseconds(cfg.camera_retry_ms);
          return false;
        }
      }
      try {
        const auto t0 = clock::now();
        s.cam_run = build_stream_run_bounded(cfg, s.tag, s.camera_node,
                                             cfg.camera_build_timeout_ms);
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
        if (s.camera_absent) {
          s.camera_absent = false;
          log_line(s.tag + " [usb] CAMERA RECONNECTED: " + cfg.usb_identity + " on " +
                   s.camera_node);
        }
        log_line(s.tag + " [recovery] stream graph restored (" + reason + ") on " +
                 s.camera_node + " in " + std::to_string(ms) + " ms");
        stalled_since = clock::now(); // the new graph gets its own silence budget
        return true;
      } catch (const CameraAbsent& e) {
        if (!s.camera_absent) {
          s.camera_absent = true;
          log_line(s.tag + " [usb] CAMERA DISCONNECTED: " + std::string(e.what()) +
                   "; looking for it every " + std::to_string(cfg.camera_retry_ms) + " ms");
        }
      } catch (const std::exception& e) {
        if (last_retry_log == clock::time_point{} ||
            clock::now() - last_retry_log >= std::chrono::seconds(10)) {
          last_retry_log = clock::now();
          log_line(s.tag + " [recovery] stream graph restore failed (" + reason +
                   "): " + std::string(e.what()) + "; retrying every " +
                   std::to_string(cfg.camera_retry_ms) + " ms");
        }
      }
      s.cam_run.reset();
      next_camera_attempt = clock::now() + std::chrono::milliseconds(cfg.camera_retry_ms);
      return false;
    };

    while (!stop.load() && (cfg.frames <= 0 || processed < cfg.frames)) {
      if (s.gate && s.gate->pause_requested()) {
        // drone seminar: only a pause that REBUILT this camera starts
        // its silence clock again. A voice window (drop: the camera is left as it is)
        // must not reset it, or commands arriving more often than camera_restart_ms
        // would postpone the restart of a camera that is already silent.
        if (enter_controlled_pause())
          stalled_since = {};
        last = clock::now();
        have_frame = false;
        prev_pts_ns = -1;
        continue;
      }

      const auto t_iter = clock::now();
      if (!have_frame) {
        if (!s.cam_run) {
          // No camera graph: a restore failed, normally because the USB camera is
          // unplugged. Try again every camera_retry_ms, on this camera only; a voice
          // pause is still honoured at the loop top in the meantime.
          if (clock::now() < next_camera_attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
          }
          rebuild_camera_locally("camera-missing");
          continue;
        }
        nn::Sample sample;
        nn::PullError err;
        const auto st = s.cam_run->pull("frames", cfg.pull_timeout_ms, sample, &err);
        if (st == nn::PullStatus::Timeout) {
          if (stop.load())
            break;
          if (s.gate && s.gate->pause_requested())
            continue;
          const auto now_stall = clock::now();
          if (stalled_since == clock::time_point{})
            stalled_since = now_stall;
          const auto stalled_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(now_stall - stalled_since)
                  .count();
          if (cfg.stall_grace_ms > 0 && stalled_ms >= cfg.stall_grace_ms)
            throw std::runtime_error("camera delivered no frame for " +
                                     std::to_string(stalled_ms) + " ms (grace " +
                                     std::to_string(cfg.stall_grace_ms) + " ms): " + err.message);
          if (cfg.camera_restart_ms > 0 && stalled_ms >= cfg.camera_restart_ms) {
            ++restarts;
            s.camera_restarts = restarts;
            log_line(s.tag + " [recovery] CAMERA RESTART: silent " + std::to_string(stalled_ms) +
                     " ms; rebuilding the stream graph (restart " + std::to_string(restarts) +
                     ")");
            rebuild_camera_locally("camera-restart");
            continue;
          }
          if (!cfg.quiet && stalled_ms >= cfg.pull_timeout_ms && stalled_ms % 1000 < 300)
            log_line(s.tag + " [recovery] CAMERA TIMEOUT: stalled " + std::to_string(stalled_ms) +
                     " ms, retrying; rebuilding at " + std::to_string(cfg.camera_restart_ms) +
                     " ms, giving up at " + std::to_string(cfg.stall_grace_ms) + " ms");
          continue;
        }
        if (st != nn::PullStatus::Ok) {
          if (stop.load())
            break;
          // FIX 3 (camera pull-error recovery): a camera pull/start ERROR - measured:
          // V4L2 "Failed to start streaming: Operation not permitted" right after a CSI
          // watchdog restart - is treated like a silent camera. It goes through the
          // existing camera-restart maintenance path instead of ending the process.
          const auto now_err = clock::now();
          if (stalled_since == clock::time_point{})
            stalled_since = now_err;
          const auto err_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(now_err - stalled_since).count();
          if (last_pull_error_log == clock::time_point{} ||
              now_err - last_pull_error_log >= std::chrono::seconds(1)) {
            last_pull_error_log = now_err;
            log_line(s.tag + " [recovery] CAMERA PULL ERROR (treated as a silent camera, " +
                     std::to_string(err_ms) + " ms): " + err.message.substr(0, 160) +
                     "; rebuilding at " + std::to_string(cfg.camera_restart_ms) + " ms");
          }
          if (!usbcam::resolve(cfg.usb_identity)) {
            // The camera has left the USB bus: waiting camera_restart_ms cannot help.
            ++restarts;
            s.camera_restarts = restarts;
            s.camera_absent = true;
            log_line(s.tag + " [usb] CAMERA DISCONNECTED: " + cfg.usb_identity + " (" +
                     s.camera_node + ") is no longer on the USB bus; looking for it every " +
                     std::to_string(cfg.camera_retry_ms) + " ms");
            rebuild_camera_locally("usb-disconnect");
            continue;
          }
          if (cfg.stall_grace_ms > 0 && err_ms >= cfg.stall_grace_ms)
            throw std::runtime_error("camera pull kept failing for " + std::to_string(err_ms) +
                                     " ms (grace " + std::to_string(cfg.stall_grace_ms) +
                                     " ms): " + err.message);
          if (cfg.camera_restart_ms > 0 && err_ms >= cfg.camera_restart_ms) {
            ++restarts;
            s.camera_restarts = restarts;
            log_line(s.tag + " [recovery] CAMERA RESTART: pull errors for " +
                     std::to_string(err_ms) + " ms; rebuilding the stream graph (restart " +
                     std::to_string(restarts) + ")");
            rebuild_camera_locally("camera-restart");
            continue;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(cfg.pull_timeout_ms));
          continue;
        }
        if (stalled_since != clock::time_point{}) {
          log_line(s.tag + " [recovery] camera recovered after " +
                   std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                      clock::now() - stalled_since)
                                      .count()) +
                   " ms");
          stalled_since = {};
        }
        const auto t_pulled = clock::now();
        pull_wall_ns = wall_now_ns();
        pull_mono_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t_pulled.time_since_epoch()).count();
        acc_wait.add(ms_d(t_pulled - t_iter).count());
        const auto tensors = nn::tensors_from_sample(sample, false);
        if (tensors.empty())
          continue;
        // Inference-branch-only conversion. The video branch already has this buffer.
        bool converted = false;
        if (use_fast_convert) {
          converted = nv12_to_bgr_into(tensors.front(), frame_buf);
          if (!convert_checked) {
            // One-time equivalence check: the model must see the same pixels the SDK
            // conversion would have given it. Any real difference -> keep the SDK path.
            convert_checked = true;
            const cv::Mat ref = tensors.front().to_cv_mat_copy(nn::ImageType::BGR);
            double max_diff = -1.0, mean_diff = -1.0;
            if (converted && ref.size() == frame_buf.size() && ref.type() == frame_buf.type()) {
              cv::Mat diff;
              cv::absdiff(ref, frame_buf, diff);
              cv::minMaxLoc(diff.reshape(1), nullptr, &max_diff);
              const cv::Scalar m = cv::mean(diff);
              mean_diff = (m[0] + m[1] + m[2]) / 3.0;
            }
            const bool ok = converted && max_diff >= 0.0 && mean_diff <= 0.5 && max_diff <= 4.0;
            log_line(s.tag + " [parallel] NV12->BGR check vs SDK: mapped=" +
                     (converted ? "yes" : "no") + " mean_abs_diff=" + std::to_string(mean_diff) +
                     " max_abs_diff=" + std::to_string(max_diff) +
                     (ok ? " -> using reused-buffer conversion"
                         : " -> DIFFERENT, falling back to the SDK conversion"));
            if (!ok) {
              use_fast_convert = false;
              converted = false;
            }
          }
        }
        if (converted)
          frame = frame_buf;
        else
          frame = tensors.front().to_cv_mat_copy(nn::ImageType::BGR);
        if (frame.empty())
          continue;
        acc_convert.add(ms_d(clock::now() - t_pulled).count());
        pts_ns = sample.pts_ns;
        if (prev_pts_ns >= 0 && pts_ns > prev_pts_ns && cfg.cam_fps > 0 &&
            (pts_ns - prev_pts_ns) > static_cast<std::int64_t>(3.5e9 / cfg.cam_fps))
          ++pts_gaps;
        prev_pts_ns = pts_ns;
      }
      have_frame = false;

      if (s.gate && s.gate->pause_requested())
        continue;

      if (win_first_pts < 0)
        win_first_pts = pts_ns;
      win_last_pts = pts_ns;
      if (pull_mono_ns >= 0 && pts_ns >= 0)
        mono_minus_pts_min = std::min(mono_minus_pts_min, (pull_mono_ns - pts_ns) / 1e6);

      // DIAGNOSTIC (off by default): stall THIS stream's YOLO loop once, to prove the video
      // branch keeps streaming while inference is blocked, and whether an idle Runner
      // survives a long gap between inferences.
      if (debug_pause_s > 0 && !debug_paused && cfg.channel == debug_pause_ch &&
          processed == debug_pause_at) {
        debug_paused = true;
        log_line(s.tag + " [diag] YOLO loop sleeping " + std::to_string(debug_pause_s) +
                 " s at frame " + std::to_string(processed) + " (video branch must keep flowing)");
        std::this_thread::sleep_for(std::chrono::seconds(debug_pause_s));
        log_line(s.tag + " [diag] YOLO loop resumed");
        continue;
      }

      if (!cfg.inference_enabled) {
        ++processed;
        ++win_frames;
        s.frames_done.store(processed, std::memory_order_relaxed);
      } else {
        const auto t_lb = clock::now();
        cv::Mat lb_frame;
        if (use_prealloc_letterbox) {
          letterbox_into(frame, cfg.model_size, s.lb, lb_resized, lb_buf);
          lb_frame = lb_buf;
        } else {
          lb_frame = render::letterbox(frame, cfg.model_size, s.lb);
        }
        const auto t_inf = clock::now();
        acc_letterbox.add(ms_d(t_inf - t_lb).count());
        nn::TensorList out;
        try {
          if (s.gate) {
            mla::Gate::InFlight marker(*s.gate);
            out = s.runner->run(std::vector<cv::Mat>{lb_frame}, cfg.infer_timeout_ms);
          } else {
            out = s.runner->run(std::vector<cv::Mat>{lb_frame}, cfg.infer_timeout_ms);
          }
        } catch (const std::exception& e) {
          s.runner_restarts = runner_restarts + 1;
          if (++runner_restarts > cfg.max_runner_restarts) {
            throw std::runtime_error("model Runner failed " + std::to_string(runner_restarts) +
                                     " times (limit " + std::to_string(cfg.max_runner_restarts) +
                                     "); last error: " + e.what());
          }
          log_line(s.tag + " [recovery] RUNNER TIMEOUT: " + std::string(e.what()));
          log_line(s.tag + " [recovery] RUNNER RESTART " + std::to_string(runner_restarts));
          std::string rebuild_error;
          maintenance_window("runner-restart", [&]() {
            // Bounded retries: one failed rebuild used to end the whole application.
            constexpr int kRebuildAttempts = 3;
            for (int attempt = 1; attempt <= kRebuildAttempts; ++attempt) {
              try {
                rebuild_runner(lb_frame);
                rebuild_error.clear();
                log_line(s.tag + " [recovery] Runner rebuilt and warmed up (attempt " +
                         std::to_string(attempt) + ")");
                return;
              } catch (const std::exception& e2) {
                rebuild_error = e2.what();
                log_line(s.tag + " [recovery] Runner rebuild attempt " + std::to_string(attempt) +
                         " of " + std::to_string(kRebuildAttempts) + " failed: " + rebuild_error);
                std::this_thread::sleep_for(std::chrono::seconds(2 * attempt));
              }
            }
          }, false);
          if (!rebuild_error.empty())
            throw std::runtime_error("model Runner rebuild failed: " + rebuild_error);
          continue;
        }
        const auto t_dec = clock::now();
        acc_infer.add(ms_d(t_dec - t_inf).count());
        const Decoded decoded = decode_output(cfg, out, s.lb, frame.size());
        const auto now = clock::now();
        acc_decode.add(ms_d(now - t_dec).count());

        // DIAGNOSTIC (off unless DRONE_SNAPSHOT_DIR is set): every DRONE_SNAPSHOT_EVERY_S
        // seconds, write the camera frame YOLO just ran on with its boxes/masks drawn,
        // plus the metadata JSON built for it - evidence of real frames and real results.
        if (!snapshot_dir.empty() && now - last_snapshot >= std::chrono::seconds(snapshot_every_s)) {
          last_snapshot = now;
          cv::Mat shot = frame.clone();
          draw_overlay(cfg, decoded, s.labels, s.lb, shot);
          int snap_dropped = 0;
          const std::string snap_json =
              metadata_data_json(cfg, decoded, s.labels, s.lb, frame.size(), snap_dropped);
          const std::string base = snapshot_dir + "/ch" + std::to_string(cfg.channel) + "_" +
                                   task_name(cfg.task) + "_" + std::to_string(processed);
          const bool wrote = cv::imwrite(base + ".jpg", shot);
          std::ofstream(base + ".json") << "{\"pts_ms\":" << (pts_ns / 1000000) << ",\"device\":\""
                                        << s.camera_node << "\",\"data\":" << snap_json << "}\n";
          log_line(s.tag + " [diag] snapshot " + base + ".jpg " + (wrote ? "" : "(WRITE FAILED) ") +
                   std::to_string(decoded.boxes.size()) + " object(s)");
        }

        const double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        if (dt > 0.0)
          fps_ema = fps_ema > 0.0 ? (0.9 * fps_ema + 0.1 / dt) : (1.0 / dt);

        if (s.meta) {
          int over_budget = 0;
          // LATENCY INSTRUMENTATION (DRONE_TIMING_META, default on): the board wall
          // clock (NTP-disciplined) when this frame reached the application, its PTS,
          // and the send instant. Downstream (probe, browser) can subtract these from
          // their own clocks for the SAME frame - vf pairs this message with it.
          nlohmann::json timing;
          const nlohmann::json* timing_ptr = nullptr;
          if (timing_meta) {
            timing = {{"pull_wall_ms", pull_wall_ns >= 0 ? pull_wall_ns / 1e6 : -1.0},
                      {"pts_ms", pts_ns >= 0 ? pts_ns / 1e6 : -1.0},
                      {"send_wall_ms", wall_now_ns() / 1e6},
                      {"mono_minus_pts_ms",
                       (pull_mono_ns >= 0 && pts_ns >= 0) ? (pull_mono_ns - pts_ns) / 1e6 : -1.0}};
            timing_ptr = &timing;
          }
          const std::string data_json = metadata_data_json(cfg, decoded, s.labels, s.lb,
                                                           frame.size(), over_budget, timing_ptr);
          const std::int64_t ts_ms = stamp_ms(pts_ns, now);
          std::string merr;
          if (s.meta->send_metadata(cfg.task == Task::Segment ? "segmentation" : "object-detection",
                                    data_json, ts_ms, std::to_string(processed), &merr)) {
            ++meta_sent;
          } else {
            if (meta_failed == 0)
              log_line(s.tag + " metadata send failed: " + merr);
            ++meta_failed;
          }
          meta_over_budget += over_budget;
          acc_meta.add(ms_d(clock::now() - now).count());
        }

        ++processed;
        ++win_frames;
        s.frames_done.store(processed, std::memory_order_relaxed);
        s.fps.store(fps_ema, std::memory_order_relaxed);
      }
      acc_loop.add(ms_d(clock::now() - t_iter).count());

      // ---- low-frequency health line (never per frame) ----------------------
      if (stats_s > 0) {
        const auto now = clock::now();
        const double win = std::chrono::duration<double>(now - win_start).count();
        if (win >= stats_s) {
          std::ostringstream os;
          os.setf(std::ios::fixed);
          os.precision(1);
          const double pts_span_s =
              (win_first_pts >= 0 && win_last_pts > win_first_pts)
                  ? static_cast<double>(win_last_pts - win_first_pts) / 1e9
                  : 0.0;
          os << s.tag << " [stats] yolo_fps=" << (static_cast<double>(win_frames) / win)
             << " frames=" << processed << " dev=" << s.camera_node << " pts_gaps=" << pts_gaps
             << " meta_sent=" << meta_sent
             << (meta_failed ? " meta_failed=" + std::to_string(meta_failed) : "")
             << (meta_over_budget ? " over_budget=" + std::to_string(meta_over_budget) : "")
             << " pts_span_s=" << pts_span_s << " last_ts_ms=" << (pts_ns / 1000000)
             << " mono_minus_pts_min_ms=" << mono_minus_pts_min
             << " | ms mean/max:" << fmt_acc("wait", acc_wait) << fmt_acc("nv12_bgr", acc_convert)
             << fmt_acc("letterbox", acc_letterbox) << fmt_acc("infer", acc_infer)
             << fmt_acc("decode", acc_decode) << fmt_acc("meta", acc_meta)
             << fmt_acc("loop", acc_loop);
          log_line(os.str());
          acc_wait = acc_convert = acc_letterbox = acc_infer = acc_decode = acc_meta = acc_loop =
              StageAcc{};
          win_frames = 0;
          win_first_pts = win_last_pts = -1;
          win_start = now;
        }
      }
    }

    log_line(s.tag + " stopping after " + std::to_string(processed) +
             " frames (camera restarts=" + std::to_string(restarts) +
             ", runner restarts=" + std::to_string(runner_restarts) + ")");
    s.runner_restarts = runner_restarts;
    s.camera_restarts = restarts;
    s.result = 0;
  } catch (const std::exception& e) {
    log_line(s.tag + " ERROR: " + std::string(e.what()));
    s.result = 1;
  }
  if (s.gate)
    s.gate->worker_gone();
}

void StreamRunner::close() {
  Impl& s = *impl_;
  if (s.closed)
    return;
  s.closed = true;
  if (s.runner)
    s.runner->close();
  if (s.cam_run)
    s.cam_run->close();
  s.runner.reset();
  s.model.reset();
  s.cam_run.reset();
}

// ---------------------------------------------------------------------------

int run_test_image(const StreamConfig& cfg, const std::string& in_path,
                   const std::string& out_path, int loops) {
  const std::string tag = "[test " + task_name(cfg.task) + "]";
  try {
    cv::Mat frame = cv::imread(in_path, cv::IMREAD_COLOR);
    if (frame.empty())
      throw std::runtime_error("cannot read image: " + in_path);
    const auto labels = render::load_labels(cfg.labels_path);

    nn::Model model(cfg.model_path, model_options(cfg));
    render::LetterboxInfo lb;
    cv::Mat lb_frame = render::letterbox(frame, cfg.model_size, lb);

    nn::RunOptions run_opt;
    run_opt.preset = nn::RunPreset::Balanced;
    run_opt.output_memory = nn::OutputMemory::Owned;
    auto runner = model.build(std::vector<cv::Mat>{lb_frame}, nn::Model::RouteOptions{}, run_opt);
    nn::TensorList out = runner.run(std::vector<cv::Mat>{lb_frame}, cfg.infer_timeout_ms);
    if (loops > 1) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 1; i < loops; ++i)
        out = runner.run(std::vector<cv::Mat>{lb_frame}, cfg.infer_timeout_ms);
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();
      log_line(tag + " " + std::to_string(loops) + " back-to-back inferences in " +
               std::to_string(ms) + " ms (" +
               std::to_string(loops * 1000.0 / (ms > 0 ? ms : 1)) + " /s)");
    }

    const Decoded decoded = decode_output(cfg, out, lb, frame.size());
    const std::vector<render::Box>& boxes = decoded.boxes;
    draw_overlay(cfg, decoded, labels, lb, frame);

    for (const auto& b : boxes) {
      const std::string name = label_for(labels, b.class_id);
      std::ostringstream os;
      os << tag << "   " << name << " score=" << b.score << " box=[" << int(b.x1) << "," << int(b.y1)
         << "," << int(b.x2) << "," << int(b.y2) << "]";
      log_line(os.str());
    }
    log_line(tag + " " + in_path + " -> " + std::to_string(boxes.size()) + " object(s)");
    if (!out_path.empty() && cv::imwrite(out_path, frame))
      log_line(tag + " annotated image written to " + out_path);

    runner.close();
    return boxes.empty() ? 2 : 0;
  } catch (const std::exception& e) {
    log_line(tag + " ERROR: " + std::string(e.what()));
    return 1;
  }
}

} // namespace pipeline
