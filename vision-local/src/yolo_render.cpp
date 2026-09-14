#include "yolo_render.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace render {
namespace {

// Ultralytics letterbox pad value; the models were exported and calibrated with it.
constexpr int kPadValue = 114;
// The device decoder emits masks at one quarter of the model input per dimension
// (640 -> 160), i.e. stride 4.
constexpr int kMaskStride = 4;

const cv::Scalar kPalette[10] = {
    {56, 56, 255}, {151, 157, 255}, {31, 112, 255}, {29, 178, 255}, {49, 210, 207},
    {10, 249, 72}, {23, 204, 146},  {134, 219, 61}, {52, 147, 26},  {187, 212, 0},
};

} // namespace

cv::Mat letterbox(const cv::Mat& src, int size, LetterboxInfo& info) {
  info.size = size;
  info.scale = std::min(static_cast<float>(size) / static_cast<float>(src.cols),
                        static_cast<float>(size) / static_cast<float>(src.rows));
  const int new_w = std::max(1, static_cast<int>(std::round(src.cols * info.scale)));
  const int new_h = std::max(1, static_cast<int>(std::round(src.rows * info.scale)));
  info.off_x = (size - new_w) / 2;
  info.off_y = (size - new_h) / 2;

  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  cv::Mat out(size, size, src.type(), cv::Scalar::all(kPadValue));
  resized.copyTo(out(cv::Rect(info.off_x, info.off_y, new_w, new_h)));
  return out;
}

void remap_boxes(std::vector<Box>& boxes, const LetterboxInfo& lb, int frame_w, int frame_h) {
  std::vector<Box> kept;
  kept.reserve(boxes.size());
  const float fw = static_cast<float>(frame_w);
  const float fh = static_cast<float>(frame_h);
  for (const Box& b : boxes) {
    Box o = b;
    o.x1 = std::clamp((b.x1 - lb.off_x) / lb.scale, 0.0f, fw);
    o.y1 = std::clamp((b.y1 - lb.off_y) / lb.scale, 0.0f, fh);
    o.x2 = std::clamp((b.x2 - lb.off_x) / lb.scale, 0.0f, fw);
    o.y2 = std::clamp((b.y2 - lb.off_y) / lb.scale, 0.0f, fh);
    if (o.x2 > o.x1 && o.y2 > o.y1)
      kept.push_back(o);
  }
  boxes.swap(kept);
}

std::vector<std::string> load_labels(const std::string& path) {
  std::vector<std::string> labels;
  std::ifstream in(path);
  if (!in)
    return labels;
  std::string line;
  while (std::getline(in, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
      line.pop_back();
    if (!line.empty())
      labels.push_back(line);
  }
  return labels;
}

cv::Scalar class_color(int class_id) {
  if (class_id < 0)
    return kPalette[0];
  return kPalette[static_cast<std::size_t>(class_id) % 10];
}

void draw_boxes(cv::Mat& frame, const std::vector<Box>& boxes,
                const std::vector<std::string>& labels) {
  for (const Box& b : boxes) {
    const cv::Scalar color = class_color(b.class_id);
    const cv::Point p1(static_cast<int>(b.x1), static_cast<int>(b.y1));
    const cv::Point p2(static_cast<int>(b.x2), static_cast<int>(b.y2));
    cv::rectangle(frame, p1, p2, color, 2);

    const std::string name = (b.class_id >= 0 && b.class_id < static_cast<int>(labels.size()))
                                 ? labels[static_cast<std::size_t>(b.class_id)]
                                 : ("id" + std::to_string(b.class_id));
    std::ostringstream tag;
    tag << name << ' ' << std::fixed << std::setprecision(2) << b.score;
    const std::string text = tag.str();

    int baseline = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int ty = std::max(p1.y, ts.height + 4);
    cv::rectangle(frame, cv::Point(p1.x, ty - ts.height - 4),
                  cv::Point(p1.x + ts.width + 4, ty + baseline - 2), color, cv::FILLED);
    cv::putText(frame, text, cv::Point(p1.x + 2, ty - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

namespace {

/// Crop `mask` to the slice `box` covers in LETTERBOXED space, then stretch that onto
/// `frame_rect`. Cropping first keeps the letterbox padding and the neighbouring
/// instances out of this instance's overlay, and keeps the resize proportional to the
/// box rather than to the whole frame.
cv::Mat project_mask(const cv::Mat& mask, const Box& box, const LetterboxInfo& lb,
                     const cv::Rect& frame_rect) {
  // Frame pixel -> letterboxed model pixel -> mask pixel (the mask is stride-4 over
  // the model input, so dividing the letterboxed coordinate by kMaskStride lands on it).
  const auto to_mask_x = [&](float fx) { return (fx * lb.scale + lb.off_x) / kMaskStride; };
  const auto to_mask_y = [&](float fy) { return (fy * lb.scale + lb.off_y) / kMaskStride; };

  const int mx1 = std::clamp(static_cast<int>(std::floor(to_mask_x(box.x1))), 0, mask.cols - 1);
  const int my1 = std::clamp(static_cast<int>(std::floor(to_mask_y(box.y1))), 0, mask.rows - 1);
  const int mx2 = std::clamp(static_cast<int>(std::ceil(to_mask_x(box.x2))), mx1 + 1, mask.cols);
  const int my2 = std::clamp(static_cast<int>(std::ceil(to_mask_y(box.y2))), my1 + 1, mask.rows);

  cv::Mat projected;
  cv::resize(mask(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1)), projected, frame_rect.size(), 0, 0,
             cv::INTER_LINEAR);
  return projected;
}

} // namespace

cv::Rect frame_rect_for(const Box& box, const cv::Size& frame_size) {
  const int bx1 = std::clamp(static_cast<int>(std::floor(box.x1)), 0, frame_size.width - 1);
  const int by1 = std::clamp(static_cast<int>(std::floor(box.y1)), 0, frame_size.height - 1);
  const int bx2 = std::clamp(static_cast<int>(std::ceil(box.x2)), bx1 + 1, frame_size.width);
  const int by2 = std::clamp(static_cast<int>(std::ceil(box.y2)), by1 + 1, frame_size.height);
  return {bx1, by1, bx2 - bx1, by2 - by1};
}

std::vector<cv::Point> mask_polygon(const cv::Mat& mask, const Box& box, const LetterboxInfo& lb,
                                    const cv::Size& frame_size, float threshold) {
  if (mask.empty())
    return {};
  const cv::Rect frame_rect = frame_rect_for(box, frame_size);

  cv::Mat binary;
  cv::threshold(project_mask(mask, box, lb, frame_rect), binary, threshold * 255.0, 255,
                cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty())
    return {};

  const auto& largest =
      *std::max_element(contours.begin(), contours.end(),
                        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                          return cv::contourArea(a) < cv::contourArea(b);
                        });

  std::vector<cv::Point> polygon;
  cv::approxPolyDP(largest, polygon, 0.004 * cv::arcLength(largest, true), true);
  if (polygon.size() < 3)
    return {};

  // Contour points lie inside frame_rect, which is already clamped to the frame, so
  // shifting them into frame space cannot leave the image.
  for (auto& point : polygon)
    point += frame_rect.tl();
  return polygon;
}

void draw_instance_mask(cv::Mat& frame, const cv::Mat& mask, const Box& box,
                        const LetterboxInfo& lb, float alpha, float threshold) {
  if (mask.empty())
    return;
  const cv::Rect frame_rect = frame_rect_for(box, frame.size());

  cv::Mat binary;
  cv::threshold(project_mask(mask, box, lb, frame_rect), binary, threshold * 255.0, 255,
                cv::THRESH_BINARY);

  cv::Mat region = frame(frame_rect);
  cv::Mat tint(region.size(), region.type(), class_color(box.class_id));
  cv::Mat blended;
  cv::addWeighted(region, 1.0 - alpha, tint, alpha, 0.0, blended);
  blended.copyTo(region, binary);
}

void draw_hud(cv::Mat& frame, const std::string& tag, double fps, std::size_t count,
              const std::string& extra) {
  std::ostringstream line;
  line << tag << "  " << std::fixed << std::setprecision(1) << fps << " fps  n=" << count;
  if (!extra.empty())
    line << "  " << extra;
  const std::string text = line.str();

  int baseline = 0;
  const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
  cv::rectangle(frame, cv::Point(8, 8), cv::Point(16 + ts.width, 20 + ts.height),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(frame, text, cv::Point(12, 14 + ts.height), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

} // namespace render
