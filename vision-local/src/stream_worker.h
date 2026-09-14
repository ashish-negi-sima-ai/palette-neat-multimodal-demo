// One USB (UVC) camera -> one YOLO26 model -> one CLEAN RTP/H.264 UDP stream plus one
// UDP metadata stream.
//
// Two of these run concurrently (one per camera, one per model), on adjacent UDP
// ports, so Neat Insight can show both results side by side.
//
// NO-OVERLAY VARIANT: inference is unchanged, but nothing derived from it is drawn
// into the video. The encoded H.264 carries camera pixels only; the boxes, labels,
// scores and masks travel as JSON metadata and Insight renders them itself.
//
// IMPORTANT lifecycle rule: `open()` must be called for EVERY stream, serially, on
// one thread, BEFORE any `run_loop()` is started. Building two Models and their
// Runners concurrently makes the MLA reject a stage
// ("infra.accelerator_execution_failed") and leaves the other stream unable to map
// its output. Setup is serial; only the per-frame loops are concurrent.
#pragma once

#include "mla_gate.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace pipeline {

enum class Task {
  Detect,  ///< yolo26m      -> BoxDecodeType::YoloV26
  Segment, ///< yolo26m-seg  -> BoxDecodeType::YoloV26Seg
};

std::string task_name(Task task);

struct StreamConfig {
  Task task = Task::Detect;

  // Camera: a USB (UVC) camera known by its STABLE identity (usb_camera.h). The
  // /dev/videoN it has right now is resolved again at every camera (re)start, so a
  // camera that Linux re-enumerates under another number after a reconnect is found.
  std::string usb_identity;          ///< e.g. "usb-serial:046d:085c:D91EAE8F"
  std::string video_node;            ///< node at selection time, e.g. "/dev/video4"
  std::string camera_card;           ///< V4L2 card name, e.g. "C922 Pro Stream Webcam"
  std::string pixel_format = "MJPG"; ///< MJPG (hardware decode) or YUYV (CPU convert)
  std::uint32_t cam_width = 1920;
  std::uint32_t cam_height = 1080;
  std::uint32_t cam_fps = 30;
  /// While the camera is absent from the USB bus, how often to look for it again.
  int camera_retry_ms = 2000;
  /// How long a camera that has just reappeared on the bus is left to settle before its
  /// graph is rebuilt.
  int camera_settle_ms = 2000;
  /// Deadline for one camera graph build; a build that misses it is abandoned.
  int camera_build_timeout_ms = 20000;

  // Model.
  std::string model_path;
  std::string labels_path;
  int model_size = 640;
  float min_score = 0.35f;
  float nms_iou = 0.50f;
  int max_detections = 50;

  // Instance-mask overlay (segmentation only).
  float mask_alpha = 0.50f;
  float mask_threshold = 0.50f;

  // UDP output. Video and metadata travel on SEPARATE ports, both indexed by
  // `channel`, which is what Neat Insight expects:
  //     video    -> video_port_base    + channel
  //     metadata -> metadata_port_base + channel
  // Final configuration is the whisper_demo host proxy (7000/7100); the Stage A
  // direct-to-Insight test uses 9000/9100. See README.
  std::string insight_host = "10.42.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  int channel = 0;
  int bitrate_kbps = 6000;
  bool video_enabled = true;
  bool metadata_enabled = true;
  /// Burn boxes/masks into the video itself. OFF in this project -- that is its whole
  /// point: Insight pairs each metadata message with its video frame and draws the
  /// overlays from the metadata. `--burn-in` turns it back on for an A/B comparison
  /// against yolo26m_det_seg.
  bool burn_in_overlay = false;
  /// Burn the model/fps/object-count HUD into the video. Also OFF: the object count and
  /// the model name are inference output, so the HUD is AI visualization too. `--hud`
  /// restores it for debugging a stream nobody is correlating.
  bool burn_in_hud = false;
  /// One UDP datagram must hold the whole metadata message; excess segments are
  /// dropped lowest-confidence-first.
  int metadata_budget_bytes = 32768;
  /// Clock the metadata timestamp is drawn from. Insight matches `timestamp * 90`
  /// against the frame's incoming RTP timestamp within +/-90 units (1 ms), so this is
  /// what decides whether pairing can work at all:
  ///   "frame"    - the shared per-source-frame millisecond stamp (see below). The video
  ///                PTS is set to exactly the same value, so the RTP timestamp Insight
  ///                compares against is `timestamp * 90` by construction. THE DEFAULT.
  ///   "pipeline" - ms since the video pipeline started (what yolo26m_det_seg used)
  ///   "source"   - the camera PTS in ms, unanchored
  /// The latter two are kept only so the failure they produce can be re-measured.
  std::string metadata_clock = "frame";
  /// Constant correction added to the metadata timestamp, in ms. Lets a fixed offset
  /// between the two clocks be dialled out without a rebuild.
  int metadata_offset_ms = 0;
  /// Memory the UDP video Input node allocates from: "ev74", "system", "dms0" or "auto".
  std::string video_memory = "ev74";
  /// Carry the source-frame timestamp into the encoder instead of letting `appsrc`
  /// stamp its own running time (`do-timestamp=false` on the video Input node).
  /// This is the fix for the pairing failure: with appsrc stamping, the RTP timestamp
  /// is on a clock the application never sees, so no metadata timestamp can match it.
  /// Setting it false reproduces the original behaviour.
  bool video_pts_from_source = true;

  // Runtime.
  int frames = 0; ///< 0 = run until stopped
  /// How long one `Run::pull()` waits for a camera frame. With `stall_grace_ms` this is a
  /// poll interval, not a deadline: a timeout means "nothing yet", and the loop asks again.
  int pull_timeout_ms = 15000;
  /// How long `open()` waits for the stream's FIRST camera frame.  Deliberately
  /// separate from `pull_timeout_ms`: that one is a poll interval inside the loop and
  /// is short so a pause request is noticed quickly, while a cold sensor legitimately
  /// takes seconds to produce its first frame.
  int first_frame_timeout_ms = 15000;
  /// How long the camera may produce nothing before the stream is declared dead.
  ///
  /// The MLA is shared with whatever else runs on the board -- in the voice demo, Whisper
  /// and Qwen inference -- and while it is busy the camera graph can stop delivering for
  /// longer than one `pull_timeout_ms`. That is back-pressure, not a broken sensor, so a
  /// timeout is retried rather than thrown. This bounds the retrying, so a camera that
  /// really has died is still reported instead of being waited on forever.
  /// 0 disables the bound (retry indefinitely).
  int stall_grace_ms = 120000;
  /// How long the camera may be silent before its Graph is torn down and rebuilt.
  ///
  /// Measured on this board, a stall caused by another MLA consumer does NOT resolve on its
  /// own: the camera Graph stops delivering silently -- no GStreamer error, `pull()` just
  /// returns Timeout forever -- so waiting is not a recovery strategy. A fresh camera Graph
  /// does work, so the stall path rebuilds one. Must be < stall_grace_ms to ever run.
  /// 0 disables rebuilding (wait only).
  int camera_restart_ms = 20000;
  int infer_timeout_ms = 20000;
  /// How many times one stream may rebuild its Model/Runner before the stream is
  /// declared dead.  SECONDARY recovery only: a Runner that has lost its MLA
  /// residency answers `runtime.output_timeout` for ever, so a rebuild is the only
  /// thing that can work - but if it keeps happening, something is wrong that
  /// rebuilding will not fix, and saying so beats an endless loop.
  int max_runner_restarts = 5;
  /// How long a worker-initiated maintenance pause (camera or Runner rebuild) waits
  /// for the other streams to park before going ahead regardless.
  int quiesce_timeout_ms = 10000;
  bool quiet = false;

  /// How long one camera `pull()` waits while the stream is inside a controlled
  /// vision pause.  Short, because this is also how often the parked worker gets a
  /// chance to notice that the pause has been released.  Timeouts here are EXPECTED
  /// and are neither logged as stalls nor counted toward `stall_grace_ms`.
  int pause_poll_ms = 200;

  /// DIAGNOSTIC ONLY (`--no-inference`): build the camera Graph and the UDP video
  /// Graph but no Model and no Runner, so the camera path can be observed in
  /// isolation from the MLA.  Never used by the demo.
  bool inference_enabled = true;
};

/// UDP ports this config will publish on.
int video_port_of(const StreamConfig& cfg);
int metadata_port_of(const StreamConfig& cfg);

/// Camera graph builds abandoned because they hung (see build_stream_run_bounded).
int abandoned_builds();

/// Owns the camera graph, the model runner and the UDP video graph for one stream.
class StreamRunner {
public:
  explicit StreamRunner(StreamConfig cfg);
  ~StreamRunner();

  StreamRunner(const StreamRunner&) = delete;
  StreamRunner& operator=(const StreamRunner&) = delete;

  /// Serial setup: camera graph, first frame, model + runner, UDP video graph.
  /// Throws on failure. See the lifecycle rule at the top of this file.
  void open();

  /// Attach the shared MLA gate.  Must be called before `run_loop()`.  Without it the
  /// stream behaves exactly as the standalone project does.
  void set_mla_gate(mla::Gate* gate);

  /// Per-frame loop. Run on its own thread once every stream has been `open()`ed.
  void run_loop(std::atomic<bool>& stop);

  /// Release every Run/Runner handle. Idempotent.
  void close();

  const StreamConfig& config() const;
  int result() const;

  /// Frames this stream has inferred (or, during a pause, passed through).
  long frames_done() const;
  /// Smoothed frames per second of the inference loop.
  double fps() const;
  /// How many times the camera Graph had to be rebuilt. A normal voice command must
  /// never increase this - the controlled pause exists precisely so it does not.
  int camera_restarts() const;
  /// How many times the Model/Runner had to be rebuilt (secondary recovery).
  int runner_restarts() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Run ONE still image through the exact same model + on-device decode + overlay
/// path the live stream uses, and write the annotated result to `out_path`.
/// Exists so the YOLO26 inference path can be verified against a known scene when
/// the cameras are pointed at something the model has nothing to say about.
int run_test_image(const StreamConfig& cfg, const std::string& in_path,
                   const std::string& out_path, int loops = 1);

} // namespace pipeline
