// encoder_low_watermark.h - the one thing this local build changes.
//
// WHAT AND WHY
//
// MEASURED (README sections 0b-0d): for one frame, the video leaves the board
// ~890-945 ms after the metadata for that same frame - 17.3-17.8 frames
// standing in the encoder path - and neither capping the SDK's async
// inter-node queues (SIMA_ASYNC_QUEUE2_DEPTH=1, -190 ms) nor making every
// queue drop-oldest (SIMA_QUEUE_LEAKY_DOWNSTREAM=1, no effect, zero drops)
// moved the remainder.
//
// `gst-inspect-1.0 neatencoder` on the board says why:
//
//     ip-queue-max-buffers : High water mark for input queue, if > then it blocks
//                            Integer. Range: 0 - 2147483647   Default: 30
//     ip-queue-min-buffers : Low water mark for input queue
//                            Integer. Range: 0 - 2147483647   Default: 15
//
// A LOW-water mark of 15 buffers is ~750 ms at a 50 ms loop period, and a
// queue held at its low-water mark by design is never "full", which is exactly
// why a leaky policy had nothing to leak.  Neither watermark is settable
// through any environment variable (`ip-queue` appears nowhere in neat-core),
// any Run/Graph option (`VideoSenderEncoderOptions` is bitrate/profile/level
// only), or any public element-property API.
//
// So this experiment sets it where it CAN be set: in the node that emits the
// element, in this project.
//
// HOW, WITHOUT TOUCHING ANYTHING OUTSIDE THIS DIRECTORY
//
// `simaai::neat::H264EncodeSima` is `final`, so it cannot be subclassed - but
// it is publicly constructible and its `backend_fragment()` is public.  This
// node therefore COMPOSES one and delegates to it, appending a single
// property to the fragment the SDK itself produced:
//
//     <the SDK's own neatencoder fragment, verbatim> + " ip-queue-min-buffers=1"
//
// Nothing about the fragment is duplicated or re-authored, so if the SDK
// changes the encoder fragment this node follows it automatically.  Every
// other Node method is delegated too, so the graph is identical in kind,
// element names, caps behaviour and output spec.
//
// EXPERIMENT 3 RESULT: min 15 -> 1 changed NOTHING (ch0 -6 ms, ch1 -12 ms,
// both inside run-to-run variation).  The low mark is the *unblock* threshold
// of a hysteresis pair, not a drain target, so it does not define normal
// standing occupancy.
//
// EXPERIMENT 4, the one variable now: ip-queue-max-buffers 30 -> 4.  The high
// mark is where the encoder BLOCKS its producer, so it is the only property
// that bounds occupancy rather than policy.  A startup capture showed the
// queue reaching ~12 frames within one second of streaming and then never
// draining, because input and output rates are equal (~19.5/s both ways, zero
// drops) - so only a hard cap can stop that backlog forming.
//
// The risk this must be measured against, not assumed away: a cap creates
// BACKPRESSURE.  The frames may simply wait in appsrc instead, which is
// block=true, so the wait would show up as time spent inside
// `Run::push()` - instrumented in stream_worker.cpp.
//
// DELIBERATELY NOT CHANGED: the profile, the level, the bitrate, enc-ip-mode
// (still async), ip-rate-ctrl, the appsrc limits, the camera buffering, the
// YOLO settings, and the async-queue enable flag.

#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "nodes/io/Input.h"
#include "nodes/sima/H264EncodeSima.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace drone_seminar {

/// The SDK's private raw ingress for the H.264 VideoSender.
///
/// Declared rather than included: the header lives in the SDK's source tree
/// (`src/nodes/groups/internal/VideoSenderRawIngress.h`) and is not installed,
/// but the symbol IS exported by libsima_neat -
///     _ZN6simaai4neat5nodes6groups8internal21VideoSenderRawIngressEiii
/// - so declaring it here links against the same adaptive ingress the SDK's
/// own group uses.  That matters: it decides at build time whether upstream
/// NV12 can feed neatencoder directly or whether a conversion fallback is
/// needed, and hand-rolling a replacement would have changed a second
/// variable.
std::shared_ptr<simaai::neat::Node> upstream_raw_ingress(int width, int height,
                                                         int fps);

} // namespace drone_seminar

// The SDK's own declaration, repeated here so the call links.  The return type
// is not part of the C++ mangled name, and the parameter list matches the
// exported symbol exactly, so this binds to the SDK's definition rather than
// creating a second one.
namespace simaai::neat::nodes::groups::internal {
std::shared_ptr<simaai::neat::Node> VideoSenderRawIngress(int width, int height,
                                                          int fps);
} // namespace simaai::neat::nodes::groups::internal

namespace drone_seminar {

inline std::shared_ptr<simaai::neat::Node>
upstream_raw_ingress(int width, int height, int fps) {
  return simaai::neat::nodes::groups::internal::VideoSenderRawIngress(
      width, height, fps);
}

// APPSRC: WHY THERE IS NO WRAPPER HERE.
//
// `gst-inspect-1.0 appsrc` on the board confirms the element supports both
//
//     max-buffers : The maximum number of buffers to queue internally
//                   (0 = unlimited)   Unsigned Integer64, Default: 0
//     leaky-type  : Whether to drop buffers once the internal queue is full
//                   Enum "GstAppLeakyType", Default: 0, "none"
//
// and the SDK's `Input` node emits neither.  Wrapping `Input` the way the
// encoder is wrapped was TRIED and rejected by the graph builder:
//
//     [misconfig.pipeline_shape] Graph::build(input): missing Input() node
//
// The builder identifies the public input endpoint by its concrete type, so a
// delegating node cannot stand in for it however faithfully it answers kind()
// and input_role().  The ingress is therefore bounded through the supported
// option instead - `InputOptions::max_bytes`, set to exactly two frames in
// stream_worker.cpp, which for a fixed-size frame is the same bound that
// max-buffers=2 would give.  Nothing in the installed SDK is modified to get
// around this.

/// `neatencoder` with a shallow input-queue low-water mark.
///
/// The wrapped `H264EncodeSima` is constructed exactly as
/// `nodes::groups::VideoSender()` constructs it, so this differs from the
/// stock graph by one property and nothing else.
class EncoderLowWatermark final : public simaai::neat::Node,
                                  public simaai::neat::OutputSpecProvider {
public:
  EncoderLowWatermark(int width, int height, int fps, int bitrate_kbps,
                      std::string profile, std::string level,
                      int ip_queue_min_buffers, int ip_queue_max_buffers)
      : inner_(width, height, fps, bitrate_kbps, std::move(profile),
               std::move(level)),
        ip_queue_min_buffers_(ip_queue_min_buffers),
        ip_queue_max_buffers_(ip_queue_max_buffers) {}

  /// The SDK's own fragment, plus the two input-queue watermarks.
  ///
  /// `ip-queue-max-buffers` is emitted only when it is positive, so leaving it
  /// at the element default stays a one-token difference from upstream rather
  /// than a hard-coded 30.
  std::string backend_fragment(int node_index) const override {
    std::string fragment = inner_.backend_fragment(node_index) +
                           " ip-queue-min-buffers=" +
                           std::to_string(ip_queue_min_buffers_);
    if (ip_queue_max_buffers_ > 0) {
      fragment += " ip-queue-max-buffers=" +
                  std::to_string(ip_queue_max_buffers_);
    }
    return fragment;
  }

  // Everything else is the wrapped node's answer, unchanged.
  std::string kind() const override {
    return inner_.kind();
  }
  std::vector<std::string> element_names(int node_index) const override {
    return inner_.element_names(node_index);
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return inner_.caps_behavior();
  }
  simaai::neat::OutputSpec
  output_spec(const simaai::neat::OutputSpec& input) const override {
    return inner_.output_spec(input);
  }

  int ip_queue_min_buffers() const {
    return ip_queue_min_buffers_;
  }
  int ip_queue_max_buffers() const {
    return ip_queue_max_buffers_;
  }

private:
  simaai::neat::H264EncodeSima inner_;
  int ip_queue_min_buffers_;
  int ip_queue_max_buffers_;   ///< <= 0 leaves the element default (30)
};

} // namespace drone_seminar
