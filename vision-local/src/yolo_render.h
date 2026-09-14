// Overlay rendering for the YOLO26 detection and segmentation streams.
//
// Both models are decoded ON DEVICE (BoxDecodeType::YoloV26 / YoloV26Seg), so this
// file only has to map results back onto the full-resolution camera frame and draw.
#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace render {

/// One detection in ORIGINAL camera-frame pixel coordinates.
struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

/// Letterbox geometry produced by `letterbox()`, needed to map results back.
struct LetterboxInfo {
  float scale = 1.0f; ///< frame_px * scale = letterboxed_px (before padding)
  int off_x = 0;      ///< left padding, in letterboxed pixels
  int off_y = 0;      ///< top padding, in letterboxed pixels
  int size = 640;     ///< square model input edge
};

/// Resize `src` into a `size`x`size` letterboxed square, padding with 114 (the
/// Ultralytics convention the models were trained and compiled with).
cv::Mat letterbox(const cv::Mat& src, int size, LetterboxInfo& info);

/// Map boxes from letterboxed model space back to original frame pixels, dropping
/// anything that collapses to zero area after clamping.
void remap_boxes(std::vector<Box>& boxes, const LetterboxInfo& lb, int frame_w, int frame_h);

/// Load newline-separated class names; returns empty on failure (caller falls back
/// to printing raw class ids).
std::vector<std::string> load_labels(const std::string& path);

/// Stable per-class colour.
cv::Scalar class_color(int class_id);

/// Draw boxes + "label score" tags onto `frame` (BGR, modified in place).
void draw_boxes(cv::Mat& frame, const std::vector<Box>& boxes,
                const std::vector<std::string>& labels);

/// Blend one instance mask over `frame` inside `box`.
///
/// `mask` is the raw 160x160 (stride-4 over the 640x640 model input) uint8 mask the
/// device decoder emits. It is cropped to the part of the mask that `box` covers in
/// LETTERBOXED space, then stretched onto the box in FRAME space -- so the letterbox
/// padding never bleeds into the overlay and the resize cost scales with the box
/// rather than with the whole frame.
void draw_instance_mask(cv::Mat& frame, const cv::Mat& mask, const Box& box,
                        const LetterboxInfo& lb, float alpha, float threshold);

/// Integer frame-space ROI a box paints into, clamped to the frame.
cv::Rect frame_rect_for(const Box& box, const cv::Size& frame_size);

/// Frame-absolute silhouette of `mask` inside `box`, for Insight's `polygon`
/// mask_format. Empty when the thresholded mask holds nothing drawable.
/// Upscaling before thresholding is what makes the outline match the rendered
/// overlay in `draw_instance_mask`.
std::vector<cv::Point> mask_polygon(const cv::Mat& mask, const Box& box, const LetterboxInfo& lb,
                                    const cv::Size& frame_size, float threshold);

/// Frames-per-second / count HUD in the top-left corner.
void draw_hud(cv::Mat& frame, const std::string& tag, double fps, std::size_t count,
              const std::string& extra = "");

} // namespace render
